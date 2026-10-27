"""V3 Reader data/augmentation, bone-local dense fields, temporal/TTA inference.

Spatial transforms are represented as pixel boxes. RGB, all labels, context and dense
fields use the same box; inverse maps are applied before observations leave this module.
"""

import json
import math
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
from studio_camera import cameras_from_calibration

READER_CONTRACT = "monster-reader-v3"
CONTEXT_SHARE = 0.5
CONTEXT_GEOMETRY_CONTRACT = "monster-context-pose-v1"
TEMPORAL_CONTEXT_CONTRACT = "monster-reader-bidirectional-context-v2"
LEGACY_TEMPORAL_CONTEXT_CONTRACT = "monster-reader-causal-context-v1"


def resolve_temporal_context_contract(checkpoint):
    """Old weights retain their causal channels; only new weights use neighbours."""
    contract = checkpoint.get("temporal_context_contract", LEGACY_TEMPORAL_CONTEXT_CONTRACT)
    if contract not in (TEMPORAL_CONTEXT_CONTRACT, LEGACY_TEMPORAL_CONTEXT_CONTRACT):
        raise ValueError(f"Unknown Reader temporal context contract: {contract!r}")
    return contract


def temporal_context_offsets(contract):
    if contract == TEMPORAL_CONTEXT_CONTRACT:
        return (-1, 1)
    if contract == LEGACY_TEMPORAL_CONTEXT_CONTRACT:
        return (-1, -2)
    raise ValueError(f"Unknown Reader temporal context contract: {contract!r}")
WINDOW_CONTRACT = "monster-reader-training-p95-window-v1"


def training_window(samples, views, width, height):
    """1.12 times p95 of training-picture extents, not p95 of one frame's points."""
    extents = []
    for sample in samples:
        for view in views:
            uv = np.asarray(sample["reader_points"][view]["uv"], dtype=float)
            inside = np.isfinite(uv).all(-1) & (uv >= 0).all(-1) & (uv <= [width, height]).all(-1)
            if inside.sum() >= 3:
                extents.append(float(np.ptp(uv[inside], axis=0).max()))
    if not extents:
        raise ValueError("Reader p95 window needs at least three in-panel training points")
    side = max(1, int(round(min(1.12 * np.percentile(extents, 95), width, height))))
    return {"contract": WINDOW_CONTRACT, "side_px": side, "panel_width": int(width),
            "panel_height": int(height), "margin": 1.12, "percentile": 95}


def scaled_window_side(window, width, height):
    if not isinstance(window, dict) or window.get("contract") != WINDOW_CONTRACT:
        raise ValueError("Reader checkpoint lacks the training p95 window; retrain with current Reader labels")
    values = np.asarray([window.get("side_px", np.nan), window.get("panel_width", np.nan), window.get("panel_height", np.nan)], float)
    if not np.isfinite(values).all() or (values <= 0).any() or values[0] > min(values[1:]):
        raise ValueError("Invalid Reader training p95 window dimensions")
    return float(min(values[0] * height / values[2], width, height))


def fixed_window(points, valid, width, height, side):
    points = np.asarray(points)
    good = np.asarray(valid, bool) & np.isfinite(points).all(-1)
    center = np.median(points[good], axis=0) if good.sum() >= 3 else np.array([width / 2, height / 2])
    return np.r_[center - side / 2, center + side / 2]


def temporal_pose_points(rig, joints, rng, contract=TEMPORAL_CONTEXT_CONTRACT):
    """Coherent neighbouring FK poses from a shared articulated velocity."""
    from scipy.spatial.transform import Rotation

    params = rig.zero()
    velocity = rng.normal(0, 0.025, rig.size)
    rotations = {}
    for name in rig.order:
        local = Rotation.from_euler("xyz", joints[name]["euler_xyz"]).as_matrix()
        parent = rig.parents[name]
        rotations[name] = (
            rotations[parent] @ rig.R[parent].T @ rig.R[name] @ local
            if parent
            else rig.R[name] @ local
        )
    for name, entry in rig.slices.items():
        euler = np.asarray(joints[name]["euler_xyz"], float)
        if isinstance(entry, slice):
            R = rotations[name] @ rig.R[name].T
            params[entry.start : entry.start + 3] = Rotation.from_matrix(R).as_rotvec()
            params[entry.start + 3 : entry.stop] = (
                np.asarray(joints[name]["head_3d"]) - rig.h[name]
            )
            velocity[entry.start + 3 : entry.stop] *= rig.height * 0.2
        else:
            indices, channels = entry
            params[indices] = euler[channels]
    return np.stack(
        [
            rig.fk(np.clip(params + offset * velocity, rig.lower, rig.upper))[0]
            for offset in temporal_context_offsets(contract)
        ]
    )


def preceding_pose_points(rig, joints, rng):
    """Compatibility helper for the original causal t-1/t-2 geometry sampler."""
    return temporal_pose_points(rig, joints, rng, LEGACY_TEMPORAL_CONTEXT_CONTRACT)


def projected_body_height(camera, height, center):
    """World reference H in panel pixels, including the current perspective depth."""
    scale = (
        camera.height / camera.scale
        if camera.type == "ortho"
        else camera.focal_px / max(float(camera.depth(center)), 1e-5)
    )
    return float(height * scale)


def moving_window(points, confidence, width, height, margin=1.12):
    pts = np.asarray(points)
    good = (np.asarray(confidence) > 0.3) & np.isfinite(pts).all(-1)
    if good.sum() < 3:
        size = float(max(width, height))
        center = np.array([width / 2, height / 2])
        return np.r_[center - size / 2, center + size / 2]
    pts = pts[good]
    center = np.median(pts, axis=0)
    extent = np.quantile(np.abs(pts - center), 0.95, axis=0) * 2
    size = max(float(extent.max() * margin), min(width, height) * 0.15)
    return np.r_[center - size / 2, center + size / 2]


def creature_crop_box(projected_vertices, margin=1.12):
    """Square full-silhouette bounds, including tails/props and occluded surfaces.

    Keep coordinates outside the panel: the reader pads them and shares the exact
    affine transform with keypoints, temporal context and dense fields.
    """
    points = np.asarray(projected_vertices, dtype=float)
    points = points[np.isfinite(points).all(axis=-1)]
    if not len(points):
        raise ValueError("Cannot crop an empty creature surface")
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    side = max(1.0, float((high - low).max()) * margin)
    return np.r_[center - side / 2, center + side / 2]


def map_points(points, box, size, inverse=False):
    box = np.asarray(box)
    span = box[2:] - box[:2]
    return (
        np.asarray(points) * span / size + box[:2]
        if inverse
        else (np.asarray(points) - box[:2]) * size / span
    )


def decode_heatmap_parabola_numpy(logits):
    """video2rig 3-point parabolic subpixel heatmap peak decode (NumPy)."""
    arr = np.asarray(logits, dtype=float)
    squeeze_batch = False
    if arr.ndim == 3:
        arr = arr[None]
        squeeze_batch = True
    hm = 1.0 / (1.0 + np.exp(-np.clip(arr, -40.0, 40.0)))
    B, K, H, W = hm.shape
    flat = hm.reshape(B, K, H * W)
    idx = np.argmax(flat, axis=-1)
    conf = np.take_along_axis(flat, idx[..., None], axis=-1)[..., 0]
    iy = idx // W
    ix = idx % W
    coords = np.zeros((B, K, 2), dtype=float)
    denom_x = max(W - 1, 1)
    denom_y = max(H - 1, 1)
    for b in range(B):
        for k in range(K):
            y0, x0 = int(iy[b, k]), int(ix[b, k])
            dx = 0.0
            dy = 0.0
            c = hm[b, k, y0, x0]
            if 0 < x0 < W - 1:
                l_val = hm[b, k, y0, x0 - 1]
                r_val = hm[b, k, y0, x0 + 1]
                curv_x = 2.0 * c - l_val - r_val
                if curv_x > 1e-6:
                    dx = float(np.clip(0.5 * (r_val - l_val) / curv_x, -0.5, 0.5))
            if 0 < y0 < H - 1:
                u_val = hm[b, k, y0 - 1, x0]
                d_val = hm[b, k, y0 + 1, x0]
                curv_y = 2.0 * c - u_val - d_val
                if curv_y > 1e-6:
                    dy = float(np.clip(0.5 * (d_val - u_val) / curv_y, -0.5, 0.5))
            coords[b, k, 0] = np.clip((x0 + dx) / denom_x, 0.0, 1.0)
            coords[b, k, 1] = np.clip((y0 + dy) / denom_y, 0.0, 1.0)
    if squeeze_batch:
        return coords[0], conf[0]
    return coords, conf


def bone_neighbours(bones_meta, bone):
    """Return immediate hierarchy neighbours (and hand/foot root for digits) for Z-buffer visibility."""
    keep = {bone}
    info = bones_meta.get(bone, {})
    parent = info.get("parent")
    if parent:
        keep.add(parent)
    for other, other_info in bones_meta.items():
        if other_info.get("parent") == bone:
            keep.add(other)
    if bone.startswith(("Finger", "Toe", "Thumb")):
        cur = parent
        while cur:
            keep.add(cur)
            if cur.startswith(("Hand", "Foot", "FrontPaw", "HindPaw", "WingHand")):
                break
            cur = bones_meta.get(cur, {}).get("parent")
    return keep


_RASTER_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 1 << 14)


def triangle_bone_owners(rig_skinning, num_verts=None):
    """Per-triangle bone index carrying the largest summed skin weight across its 3 vertices."""
    tris = np.asarray(rig_skinning.get("triangles", np.zeros((0, 3), dtype=int)), dtype=int)
    if not len(tris):
        return np.zeros(0, dtype=int)
    b_idx = np.asarray(
        rig_skinning.get("skin_indices", rig_skinning.get("bone_idx")), dtype=int
    )
    b_w = np.asarray(
        rig_skinning.get("skin_weights", rig_skinning.get("bone_w")), dtype=float
    )
    n_v = int(num_verts if num_verts is not None else len(b_idx))
    n_b = max(
        len(rig_skinning.get("bone_names", ())),
        int(b_idx.max()) + 1 if b_idx.size else 1,
    )
    W = np.zeros((n_v, n_b), dtype=float)
    np.add.at(W, (np.arange(len(b_idx))[:, None], b_idx), b_w)
    tw = W[tris].sum(axis=1)
    return np.argmax(tw, axis=1).astype(int)


def rasterize_triangles_numpy(pix, depth, tris, size):
    """Barycentric pixel-center Z-buffer rasterizer (video2rig/raster.py:rasterize_numpy).
    Returns `(tri_id, bary, zbuf)` of shape `(H, W)`, `(H, W, 3)`, `(H, W)`.
    """
    width, height = int(size[0]), int(size[1])
    tris = np.asarray(tris, dtype=np.int64)
    if not len(tris):
        return (
            np.full((height, width), -1, dtype=np.int64),
            np.zeros((height, width, 3), dtype=float),
            np.full((height, width), np.inf, dtype=float),
        )
    pix = np.asarray(pix, dtype=float)
    depth = np.asarray(depth, dtype=float)
    p = pix[tris]
    z = depth[tris]
    x0 = np.maximum(np.ceil(p[:, :, 0].min(1) - 0.5).astype(np.int64), 0)
    x1 = np.minimum(np.floor(p[:, :, 0].max(1) - 0.5).astype(np.int64), width - 1)
    y0 = np.maximum(np.ceil(p[:, :, 1].min(1) - 0.5).astype(np.int64), 0)
    y1 = np.minimum(np.floor(p[:, :, 1].max(1) - 0.5).astype(np.int64), height - 1)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    a, bb, c = p[:, 0], p[:, 1], p[:, 2]
    area = (bb[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (bb[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    ok = (bw > 0) & (bh > 0) & (np.abs(area) > 1e-12)
    size_px = np.maximum(bw, bh)
    pix_all, z_all, t_all, w_all = [], [], [], []
    lo = 0
    for S in _RASTER_BUCKETS:
        sel = np.where(ok & (size_px > lo) & (size_px <= S))[0]
        lo = S
        if not len(sel):
            continue
        if S > 256:
            for t in sel:
                ys, xs = np.mgrid[y0[t] : y1[t] + 1, x0[t] : x1[t] + 1]
                cx, cy = xs.ravel() + 0.5, ys.ravel() + 0.5
                A_, B_, C_ = a[t], bb[t], c[t]
                w0 = ((B_[0] - cx) * (C_[1] - cy) - (B_[1] - cy) * (C_[0] - cx)) / area[t]
                w1 = ((C_[0] - cx) * (A_[1] - cy) - (C_[1] - cy) * (A_[0] - cx)) / area[t]
                w2 = 1.0 - w0 - w1
                inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
                if inside.any():
                    ww = np.stack([w0[inside], w1[inside], w2[inside]], 1)
                    pix_all.append(ys.ravel()[inside] * width + xs.ravel()[inside])
                    z_all.append(ww @ z[t])
                    t_all.append(np.full(int(inside.sum()), t, dtype=np.int64))
                    w_all.append(ww)
            continue
        oy, ox = np.divmod(np.arange(S * S), S)
        px, py = x0[sel, None] + ox[None], y0[sel, None] + oy[None]
        inb = (ox[None] < bw[sel, None]) & (oy[None] < bh[sel, None])
        cx, cy = px + 0.5, py + 0.5
        A, B, C, ar = a[sel], bb[sel], c[sel], area[sel][:, None]
        w0 = ((B[:, 0, None] - cx) * (C[:, 1, None] - cy) - (B[:, 1, None] - cy) * (C[:, 0, None] - cx)) / ar
        w1 = ((C[:, 0, None] - cx) * (A[:, 1, None] - cy) - (C[:, 1, None] - cy) * (A[:, 0, None] - cx)) / ar
        w2 = 1.0 - w0 - w1
        inside = inb & (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
        r, q = np.nonzero(inside)
        if not len(r):
            continue
        t = sel[r]
        ww = np.stack([w0[r, q], w1[r, q], w2[r, q]], 1)
        pix_all.append(py[r, q] * width + px[r, q])
        z_all.append((ww * z[t]).sum(1))
        t_all.append(t)
        w_all.append(ww)
    tri = np.full(width * height, -1, dtype=np.int64)
    bary = np.zeros((width * height, 3), dtype=float)
    zbuf = np.full(width * height, np.inf, dtype=float)
    if pix_all:
        P = np.concatenate(pix_all)
        Z = np.concatenate(z_all)
        T = np.concatenate(t_all)
        WW = np.concatenate(w_all)
        order = np.lexsort((Z, P))
        P = P[order]
        first = np.r_[True, P[1:] != P[:-1]]
        idx = order[first]
        tri[P[first]] = T[idx]
        bary[P[first]] = WW[idx]
        zbuf[P[first]] = Z[idx]
    return tri.reshape(height, width), bary.reshape(height, width, 3), zbuf.reshape(height, width)


def compute_zbuffer_visibility(
    posed_verts,
    rig_skinning,
    camera,
    keypoints_3d,
    keypoint_bones,
    bones_meta,
    bone_names=None,
    window=2,
    scale=0.5,
):
    """CPU Z-buffer bone-owner visibility test (video2rig/reader/visible.py)."""
    verts = np.asarray(posed_verts, dtype=float)
    pts3d = np.asarray(keypoints_3d, dtype=float)
    K = len(pts3d)
    if K == 0 or len(verts) == 0:
        return np.ones(K, dtype=float)
    names = list(bone_names if bone_names is not None else rig_skinning.get("bone_names", ()))
    name_to_idx = {str(n): i for i, n in enumerate(names)}

    W = max(int(round(camera.width * scale)), 1)
    H = max(int(round(camera.height * scale)), 1)
    uv_v = camera.project(verts) * scale
    z_v = camera.depth(verts)
    tris = np.asarray(rig_skinning.get("triangles", np.zeros((0, 3), dtype=int)), dtype=int)
    if len(tris):
        tri_owner = triangle_bone_owners(rig_skinning, num_verts=len(verts))
        tri_buf, _, _ = rasterize_triangles_numpy(uv_v, z_v, tris, (W, H))
        owner_buf = np.where(tri_buf >= 0, tri_owner[np.maximum(tri_buf, 0)], -1)
    else:
        b_idx = np.asarray(
            rig_skinning.get("skin_indices", rig_skinning.get("bone_idx")), dtype=int
        )
        b_w = np.asarray(
            rig_skinning.get("skin_weights", rig_skinning.get("bone_w")), dtype=float
        )
        dom_slot = np.argmax(b_w, axis=1)
        vert_owner = b_idx[np.arange(len(verts)), dom_slot]
        zbuf = np.full((H, W), np.inf, dtype=float)
        owner_buf = np.full((H, W), -1, dtype=int)
        ix = np.clip(np.floor(uv_v[:, 0]).astype(int), 0, W - 1)
        iy = np.clip(np.floor(uv_v[:, 1]).astype(int), 0, H - 1)
        for v_i in np.argsort(z_v)[::-1]:
            y0, x0 = iy[v_i], ix[v_i]
            if z_v[v_i] < zbuf[y0, x0]:
                zbuf[y0, x0] = z_v[v_i]
                owner_buf[y0, x0] = int(vert_owner[v_i])

    uv_k = camera.project(pts3d) * scale
    vis = np.zeros(K, dtype=float)
    for k in range(K):
        if not np.isfinite(uv_k[k]).all():
            continue
        x, y = int(uv_k[k, 0]), int(uv_k[k, 1])
        if not (0 <= x < W and 0 <= y < H):
            continue
        allowed = {
            name_to_idx[n]
            for n in bone_neighbours(bones_meta, str(keypoint_bones[k]))
            if n in name_to_idx
        }
        win = owner_buf[
            max(0, y - window) : min(H, y + window + 1),
            max(0, x - window) : min(W, x + window + 1),
        ]
        win = win[win >= 0]
        if len(win) and any(int(o) in allowed for o in win):
            vis[k] = 1.0
    return vis


def export_video2rig_npz_labels(
    out_dir,
    records,
    bones_meta,
    sorted_bones,
    extra_points,
    calib,
    keypoint_names=None,
):
    """Export video2rig-compatible labels.npz, visible.npz, and poses.npz archives."""
    from scipy.spatial.transform import Rotation

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    view_order = list(calib.get("view_order", ("front", "side")))
    bone_names = list(sorted_bones)
    N = len(records)
    B = len(bone_names)
    heads = np.zeros((N, B, 3), dtype=np.float32)
    tails = np.zeros((N, B, 3), dtype=np.float32)
    quats = np.zeros((N, B, 4), dtype=np.float32)
    quats[..., 0] = 1.0
    eulers_all = np.zeros((N, B, 3), dtype=np.float32)
    root_offsets = np.zeros((N, 3), dtype=np.float32)
    K = len(keypoint_names) if keypoint_names is not None else (2 * B + len(extra_points))
    V = len(view_order)
    vis_bool = np.zeros((N, V, K), dtype=bool)
    vis_ratio = np.zeros((N, V, K), dtype=np.float32)
    root_bone = next(
        (
            b
            for b in bone_names
            if isinstance(bones_meta, dict)
            and b in bones_meta
            and bones_meta[b].get("parent") is None
            and not str(b).startswith("Prop_")
        ),
        bone_names[0] if bone_names else None,
    )

    for i, rec in enumerate(records):
        joints = rec.get("joints", {})
        for b_i, b_name in enumerate(bone_names):
            j = joints.get(b_name, {})
            if "head_3d" in j:
                heads[i, b_i] = np.asarray(j["head_3d"], dtype=np.float32)
            if "tail_3d" in j:
                tails[i, b_i] = np.asarray(j["tail_3d"], dtype=np.float32)
            if "quat_wxyz" in j:
                q_wxyz = np.asarray(j["quat_wxyz"], dtype=np.float64)
                quats[i, b_i] = q_wxyz.astype(np.float32)
                q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float64)
                if np.linalg.norm(q_xyzw) > 1e-8:
                    eulers_all[i, b_i] = Rotation.from_quat(q_xyzw).as_euler("xyz").astype(np.float32)
            elif "euler_xyz" in j:
                eul = np.asarray(j["euler_xyz"], dtype=np.float64)
                eulers_all[i, b_i] = eul.astype(np.float32)
                q_xyzw = Rotation.from_euler("xyz", eul).as_quat()
                quats[i, b_i] = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)
        if "root_offset" in rec:
            root_offsets[i] = np.asarray(rec["root_offset"], dtype=np.float32)
        elif isinstance(rec.get("v3_sampling"), dict) and "translation" in rec["v3_sampling"]:
            root_offsets[i] = np.asarray(rec["v3_sampling"]["translation"], dtype=np.float32)
        elif (
            root_bone is not None
            and root_bone in joints
            and "head_3d" in joints[root_bone]
            and isinstance(bones_meta, dict)
            and root_bone in bones_meta
            and "rest_head_world" in bones_meta[root_bone]
        ):
            root_offsets[i] = (
                np.asarray(joints[root_bone]["head_3d"], dtype=np.float32)
                - np.asarray(bones_meta[root_bone]["rest_head_world"], dtype=np.float32)
            )
        rp = rec.get("reader_points", {})
        for v_i, view in enumerate(view_order):
            vp = rp.get(view, {})
            v_arr = np.asarray(vp.get("visibility", np.zeros(K)), dtype=np.float32)
            L = min(len(v_arr), K)
            vis_ratio[i, v_i, :L] = v_arr[:L]
            if "zbuffer_visible" in vp:
                zb = np.asarray(vp["zbuffer_visible"], dtype=bool)
                vis_bool[i, v_i, : min(len(zb), K)] = zb[: min(len(zb), K)]
            else:
                vis_bool[i, v_i, :L] = v_arr[:L] > 0.3

    try:
        from reprojection_solver import RigModel

        rig = RigModel(bones_meta, names=bone_names, extra_points=extra_points)
        params = np.zeros((N, rig.size), dtype=np.float32)
        for i, rec in enumerate(records):
            joints = rec.get("joints", {})
            p_vec = rig.zero()
            for b_i, b_name in enumerate(bone_names):
                entry = rig.slices.get(b_name)
                if entry is None:
                    continue
                j = joints.get(b_name, {})
                if isinstance(entry, slice):
                    if "rot_mat_world" in j:
                        R_world = np.asarray(j["rot_mat_world"], dtype=float)
                    else:
                        R_loc = Rotation.from_euler(
                            "xyz", eulers_all[i, b_i].astype(float)
                        ).as_matrix()
                        R_world = rig.R[b_name] @ R_loc
                    R_delta_world = R_world @ rig.R[b_name].T
                    p_vec[entry.start : entry.start + 3] = Rotation.from_matrix(
                        R_delta_world
                    ).as_rotvec()
                    if "head_3d" in j:
                        p_vec[entry.start + 3 : entry.stop] = (
                            np.asarray(j["head_3d"], dtype=float) - rig.h[b_name]
                        )
                    elif b_name == root_bone:
                        p_vec[entry.start + 3 : entry.stop] = root_offsets[i].astype(float)
                else:
                    ix, ch = entry
                    p_vec[ix] = eulers_all[i, b_i, ch].astype(float)
            params[i] = p_vec.astype(np.float32)
    except Exception:
        params = np.concatenate([root_offsets, eulers_all.reshape(N, -1)], axis=1).astype(np.float32)

    idx_arr = np.arange(N, dtype=np.int32)
    np.savez_compressed(
        out_path / "labels.npz",
        index=idx_arr,
        names=np.asarray(bone_names),
        heads=heads,
        tails=tails,
        views=np.asarray(view_order),
        extra_points_json=np.asarray(json.dumps(extra_points)),
    )
    np.savez_compressed(
        out_path / "visible.npz",
        index=idx_arr,
        views=np.asarray(view_order),
        vis=vis_bool,
        vis_ratio=vis_ratio,
        keypoint_names=np.asarray(list(keypoint_names or [])),
    )
    np.savez_compressed(
        out_path / "poses.npz",
        index=idx_arr,
        names=np.asarray(bone_names),
        bones=np.asarray(json.dumps(bones_meta if isinstance(bones_meta, dict) else {})),
        params=params,
        quats=quats,
        root_offsets=root_offsets,
    )
    return {
        "labels_npz": str(out_path / "labels.npz"),
        "visible_npz": str(out_path / "visible.npz"),
        "poses_npz": str(out_path / "poses.npz"),
    }


def crop_rgb(image, box, size=448):
    return np.asarray(
        Image.fromarray(np.asarray(image).astype(np.uint8)).transform(
            (size, size), Image.Transform.EXTENT, tuple(box), Image.Resampling.BILINEAR
        )
    )


def sight_depth_targets(points_3d, camera, height=None, non_prop_mask=None):
    """Per-keypoint sight depth `(X - mid_body) @ cam_fwd / H` in creature heights (video2rig/reader/train.py)."""
    if hasattr(points_3d, "height") and hasattr(points_3d, "point_bones"):
        rig_obj = points_3d
        points_3d = height
        height = rig_obj.height
        if non_prop_mask is None:
            non_prop_mask = [not str(b).startswith("Prop_") for b in rig_obj.point_bones]
    pts = np.asarray(points_3d, dtype=np.float64)
    H = max(float(height if height is not None else 1.0), 1e-6)
    if pts.ndim == 2:
        if non_prop_mask is not None and np.any(non_prop_mask):
            mid = np.nanmean(pts[np.asarray(non_prop_mask, dtype=bool)], axis=0, keepdims=True)
        else:
            mid = np.nanmean(pts, axis=0, keepdims=True)
        return ((camera.depth(pts) - camera.depth(mid)[0]) / H).astype(np.float32)
    if non_prop_mask is not None and np.any(non_prop_mask):
        mid = np.nanmean(pts[..., np.asarray(non_prop_mask, dtype=bool), :], axis=-2, keepdims=True)
    else:
        mid = np.nanmean(pts, axis=-2, keepdims=True)
    z_pts = np.stack([camera.depth(p) for p in pts], axis=0)
    z_mid = np.stack([camera.depth(m) for m in mid], axis=0)
    return ((z_pts - z_mid) / H).astype(np.float32)


CROP_K = 1.6
CROP_SIDE_MIN = 72.0
CROP_SIDE_MAX = 320.0
CROP_JITTER = 4.0


def end_groups(rig):
    """Hand/foot/claw limb-end groups for `reader-crops.pt` (video2rig/reader/crops.py:end_groups).

    For each lateral non-loose limb chain (`len(chain) >= 2`), collects the last bone in the chain
    plus all its descendants (hands, feet, fingers, toes, claws), their keypoint indices `ks`,
    and `reach` (sum of the last 2 bones' rest lengths).
    """
    kp_Idx = {n: i for i, n in enumerate(rig.point_names)}
    children = {n: [] for n in rig.names}
    for n in rig.names:
        p = rig.parents.get(n)
        if p in children:
            children[p].append(n)

    def _sub_bones(bone_name):
        out = [bone_name]
        stack = list(children.get(bone_name, ()))
        while stack:
            cur = stack.pop(0)
            out.append(cur)
            stack.extend(children.get(cur, ()))
        return out

    # Identify limb roots (bones with .L/.R whose parent does not have the same side suffix)
    groups = []
    seen_roots = set()
    for n in rig.names:
        if not (n.endswith(".L") or n.endswith(".R")):
            continue
        side_suf = n[-2:]
        p = rig.parents.get(n)
        if p is not None and p.endswith(side_suf):
            continue
        if n in seen_roots:
            continue
        seen_roots.add(n)
        kind = rig.meta.get(n, {}).get("kind", "")
        if kind in ("tail", "ear", "loose") or n in getattr(rig, "loose", ()):
            continue
        # Follow primary limb chain down to the distal limb/paw/hand/foot bone before digits split
        chain = [n]
        cur = n
        while True:
            ch = [
                c
                for c in children.get(cur, ())
                if c.endswith(side_suf)
                and not c.startswith(("Finger", "Toe", "FrontToes", "HindToes"))
                and rig.meta.get(c, {}).get("kind") != "rigid"
            ]
            if len(ch) != 1:
                break
            cur = ch[0]
            chain.append(cur)
        if len(chain) < 2:
            # Also allow 1-bone hand/foot + digit chains if descendants exist
            all_desc = _sub_bones(n)
            if len(all_desc) < 2:
                continue
            end_bone = n
            reach_bones = all_desc[:2]
        else:
            end_bone = chain[-1]
            reach_bones = chain[-2:]
        bones = _sub_bones(end_bone)
        bone_set = set(bones)
        ks = [
            i
            for i, owner in enumerate(rig.point_bones)
            if owner in bone_set
        ]
        # Include the parent joint head/tail of end_bone
        end_head = f"{end_bone}.head"
        if end_head in kp_Idx and kp_Idx[end_head] not in ks:
            ks.insert(0, kp_Idx[end_head])
        ks = sorted(set(ks))
        if len(ks) < 2:
            continue
        reach = float(
            sum(np.linalg.norm(rig.t[b] - rig.h[b]) for b in reach_bones)
        )
        groups.append({"name": end_bone, "bones": bones, "ks": ks, "reach": reach})
    return groups


def crop_sides(rig, cameras, window_side=None, groups=None):
    """Per-end-group crop square side in pixels (video2rig/reader/crops.py:crop_sides)."""
    if isinstance(window_side, (list, tuple)):
        groups = window_side
        window_side = None
    if groups is None:
        groups = end_groups(rig)
    if not groups:
        return []
    cam0 = (
        cameras[0]
        if isinstance(cameras, (list, tuple))
        else (next(iter(cameras.values())) if isinstance(cameras, dict) else cameras)
    )
    center = np.mean(np.array(list(rig.h.values()), dtype=float), axis=0)
    px_per_unit = projected_body_height(cam0, 1.0, center)
    if window_side is not None and float(window_side) > 0:
        px_per_unit = px_per_unit * (448.0 / float(window_side))
    return [
        float(np.clip(CROP_K * g["reach"] * px_per_unit, CROP_SIDE_MIN, CROP_SIDE_MAX))
        for g in groups
    ]


def rasterize_dense_fields(posed, skin, camera, grid=224, local_norm="height"):
    """Z-buffered triangle rasterization with perspective-correct rest/weight interpolation.

    By default (`local_norm="height"`), bone-local coordinates `local` are normalized by creature
    total height `H` (`video2rig/reader/prepare.py:242`), rather than individual bone length.
    Pass `local_norm="bone_length"` for legacy per-bone length normalization.
    """
    from dataclasses import replace

    W, H = camera.width, camera.height
    c = replace(
        camera,
        width=grid,
        height=grid,
        scale=camera.scale,
        focal_px=camera.focal_px * grid / W,
        shift_px=(camera.shift_px[0] * grid / W, camera.shift_px[1] * grid / H),
    )
    # Non-square native panels preserve their independent x/y pixel scale.
    uv = camera.project(posed) * [grid / W, grid / H]
    depth = camera.depth(posed)
    zbuffer = np.full((grid, grid), np.inf)
    part = np.zeros((grid, grid), np.int64)
    local = np.zeros((3, grid, grid), np.float32)
    relative = np.zeros((1, grid, grid), np.float32)
    rest = np.asarray(skin["rest_verts"])
    ids = np.asarray(skin["skin_indices"])
    weights = np.asarray(skin["skin_weights"])
    heads = np.asarray(skin["rest_heads"])
    rotations = np.asarray(skin["rest_R"])
    lengths = np.asarray(skin["rest_lengths"])
    height = max(float(skin["height"]), 1e-6)
    center_depth = float(camera.depth(np.mean(posed, axis=0)))
    for tri in np.asarray(skin["triangles"], int):
        if camera.type == "persp" and np.any(depth[tri] <= 0):
            continue
        p = uv[tri]
        xmin, ymin = np.maximum(np.floor(p.min(0)).astype(int), 0)
        xmax, ymax = np.minimum(np.ceil(p.max(0)).astype(int), grid - 1)
        if xmin > xmax or ymin > ymax:
            continue
        A = np.column_stack([p[1] - p[0], p[2] - p[0]])
        if abs(np.linalg.det(A)) < 1e-8:
            continue
        yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
        pixels = np.c_[xx.ravel() + 0.5, yy.ravel() + 0.5]
        ab = (pixels - p[0]) @ np.linalg.inv(A).T
        bary = np.c_[1 - ab.sum(-1), ab]
        valid = (bary >= -1e-6).all(-1)
        if camera.type == "persp":
            bary = bary / depth[tri]
            bary /= bary.sum(-1, keepdims=True)
        z = bary @ depth[tri]
        x = xx.ravel()
        y = yy.ravel()
        valid &= z < zbuffer[y, x]
        if not valid.any():
            continue
        bary = bary[valid]
        x = x[valid]
        y = y[valid]
        z = z[valid]
        bone_ids = np.unique(ids[tri])
        mass = np.zeros((len(bary), len(bone_ids)))
        for corner in range(3):
            for slot in range(ids.shape[1]):
                j = int(np.searchsorted(bone_ids, ids[tri[corner], slot]))
                mass[:, j] += bary[:, corner] * weights[tri[corner], slot]
        owner = bone_ids[mass.argmax(-1)]
        xyz = bary @ rest[tri]
        raw_local = np.einsum(
            "nij,nj->ni", rotations[owner].transpose(0, 2, 1), xyz - heads[owner]
        )
        if local_norm == "bone_length":
            coords = raw_local / np.maximum(lengths[owner], 0.05 * height)[:, None]
        else:
            coords = raw_local / height
        zbuffer[y, x] = z
        part[y, x] = owner + 1
        local[:, y, x] = coords.T
        relative[0, y, x] = (z - center_depth) / height
    return {"part": part, "local": local, "depth": relative, "local_norm": local_norm}


def joints_from_dense_fields(
    part_logits,
    local_fields,
    depth_field,
    camera,
    rig,
    center_depth,
    local_norm="auto",
):
    """Recover bone rigid transforms by weighted SVD, then vote for every attached keypoint.

    Supports creature-height normalization (`local_norm="height"`, `video2rig/reader/dense.py`),
    bone-length normalization (`local_norm="bone_length"`), and automatic scale detection (`"auto"`).
    """
    logits = np.asarray(part_logits, float)
    logits -= logits.max(0, keepdims=True)
    prob = np.exp(logits)
    prob /= np.maximum(prob.sum(0, keepdims=True), 1e-12)
    owner = prob.argmax(0)
    hh, ww = owner.shape
    local_fields = np.asarray(local_fields).reshape(len(rig.names), 3, hh, ww)
    yy, xx = np.mgrid[:hh, :ww]
    pixels = np.stack(
        [(xx + 0.5) * camera.width / ww, (yy + 0.5) * camera.height / hh], axis=-1
    )
    depth_arr = np.asarray(depth_field, dtype=float)
    if depth_arr.ndim == 3 and depth_arr.shape[0] > 1:
        depth_2d = depth_arr.mean(axis=0)
    else:
        depth_2d = depth_arr.reshape(hh, ww)
    X = camera.ray_at_depth(
        pixels, center_depth + depth_2d * rig.height
    )
    result = np.full((len(rig.point_names), 3), np.nan)
    confidence = np.zeros(len(result))
    for i, n in enumerate(rig.names):
        good = (owner == i + 1) & (prob[i + 1] > 0.5) & np.isfinite(X).all(-1)
        if good.sum() < 6:
            continue
        raw_a = local_fields[i][:, good].T
        b = X[good]
        w = prob[i + 1, good]
        w /= w.sum()
        bone_scale = max(np.linalg.norm(rig.t[n] - rig.h[n]), 0.05 * rig.height)
        height_scale = max(float(rig.height), 1e-6)
        if local_norm == "height":
            scale = height_scale
        elif local_norm == "bone_length":
            scale = bone_scale
        else:
            raw_ac = (raw_a * w[:, None]).sum(0)
            bc_tmp = (b * w[:, None]).sum(0)
            rms_a = float(np.sqrt(np.sum(((raw_a - raw_ac) ** 2) * w[:, None])))
            rms_b = float(np.sqrt(np.sum(((b - bc_tmp) ** 2) * w[:, None])))
            if rms_a > 1e-8:
                est_scale = rms_b / rms_a
                scale = (
                    height_scale
                    if abs(est_scale - height_scale) <= abs(est_scale - bone_scale)
                    else bone_scale
                )
            else:
                scale = height_scale
        a = raw_a * scale
        ac = (a * w[:, None]).sum(0)
        bc = (b * w[:, None]).sum(0)
        A = a - ac
        B = b - bc
        U, S, V = np.linalg.svd((A * w[:, None]).T @ B)
        if S[1] < 1e-7:
            continue
        R = V.T @ np.diag([1, 1, np.linalg.det(V.T @ U.T)]) @ U.T
        for k, (bone, point) in enumerate(zip(rig.point_bones, rig.local_points)):
            if bone != n:
                continue
            votes = b + (np.asarray(point) - a) @ R.T
            result[k] = (votes * w[:, None]).sum(0)
            confidence[k] = float(prob[i + 1, good].mean())
    return result, confidence


# Torch is optional for geometry-only workers and archive validation.
try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset
    import torch.nn.functional as F
except ImportError:
    torch = None


if torch is not None:

    def sample_maps(dmap, uv, size=448, normalized=True):
        """Sample each keypoint's own channel `dmap[:, k]` at `uv[:, k]` via bilinear grid_sample (video2rig/reader/net.py).

        Parameters
        ----------
        dmap : (B, K, H, W) tensor (or (B, 1, H, W) which broadcasts across K)
        uv : (B, K, 2) tensor of coordinates
        size : input resolution when `normalized=False`
        normalized : True if `uv` is in [0, 1], False if `uv` is in [0, size]
        """
        B, K, _ = uv.shape
        if dmap.shape[1] == 1 and K > 1:
            dmap = dmap.expand(-1, K, -1, -1)
        g = (uv * 2.0 - 1.0) if normalized else (uv * (2.0 / float(size)) - 1.0)
        g = g.to(dmap.dtype).unsqueeze(2)  # (B, K, 1, 2)
        s = F.grid_sample(
            dmap, g, mode="bilinear", padding_mode="border", align_corners=False
        )  # (B, K, K, 1)
        return s.diagonal(dim1=1, dim2=2).squeeze(1)  # (B, K)

    def _v2r_block(ci, co, stride=1):
        return nn.Sequential(
            nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
        )

    class Video2RigReaderNet(nn.Module):
        """video2rig `facebook/dinov2-small` (`d=384`, `FROZEN=4`, `CH=96`, `STRIDE=4`, `HEAT=112`) Reader architecture.

        Supports modular base (`reader.pt`), hand/claw crop (`reader-crops.pt`), and dense field (`reader-dense.pt`)
        configurations, while remaining drop-in compatible with `MonsterDINOv2PoseProbe` call sites.
        """

        SIZE = 448
        PATCH = 14
        STRIDE = 4
        HEAT = SIZE // STRIDE
        CH = 96
        FROZEN = 4

        def __init__(
            self,
            num_keypoints: int,
            *,
            context_channels: int | None = None,
            depth: bool = True,
            dense_bones: int = 0,
            backbone_name: str = "dinov2-small",
            custom_backbone: nn.Module | None = None,
        ):
            super().__init__()
            self.num_keypoints = int(num_keypoints)
            self.k = int(num_keypoints)
            self.context = (
                2 * self.num_keypoints
                if context_channels is None
                else int(context_channels)
            )
            self.has_depth = bool(depth)
            self.dense_bones = int(dense_bones)
            self.backbone_name = str(backbone_name)

            if custom_backbone is not None:
                self.backbone = custom_backbone
            else:
                self.backbone = self._load_dinov2_small(backbone_name)

            d = int(
                getattr(
                    self.backbone,
                    "embed_dim",
                    getattr(getattr(self.backbone, "config", None), "hidden_size", 384),
                )
            )
            self.embed_dim = d
            self.fine = nn.Sequential(
                _v2r_block(3, 32, stride=2),
                _v2r_block(32, 48, stride=2),
            )
            self.proj = nn.Sequential(
                nn.Conv2d(d, self.CH, 1, bias=False),
                nn.BatchNorm2d(self.CH),
                nn.ReLU(inplace=True),
            )
            self.up1 = _v2r_block(self.CH, self.CH)
            self.mix = nn.Sequential(
                _v2r_block(self.CH + 48 + self.context, self.CH),
                _v2r_block(self.CH, self.CH),
            )
            self.head = nn.Conv2d(self.CH, self.k, 1)
            self.dhead = (
                nn.Sequential(
                    _v2r_block(self.CH, self.CH),
                    nn.Conv2d(self.CH, self.k, 1),
                )
                if self.has_depth
                else None
            )
            if self.dense_bones > 0:
                self.phead = nn.Conv2d(self.CH, self.dense_bones + 1, 1)
                self.lhead = nn.Sequential(
                    _v2r_block(self.CH, self.CH),
                    nn.Conv2d(self.CH, 3, 1),
                )
                self.shead = nn.Conv2d(self.CH, 3, 1)
            else:
                self.phead = None
                self.lhead = None
                self.shead = None
            self.freeze_early()

        @staticmethod
        def _load_dinov2_small(backbone_name: str) -> nn.Module:
            try:
                from transformers import AutoModel

                hf_name = (
                    backbone_name
                    if "/" in backbone_name
                    else "facebook/dinov2-small"
                )
                return AutoModel.from_pretrained(hf_name)
            except Exception:
                hub_name = (
                    "dinov2_vits14_reg"
                    if "reg" in backbone_name
                    else "dinov2_vits14"
                )
                return torch.hub.load("facebookresearch/dinov2", hub_name)

        def freeze_early(self) -> None:
            """Freeze patch/position embeddings and first FROZEN=4 transformer blocks (video2rig/reader/net.py)."""
            if hasattr(self.backbone, "embeddings"):
                for p in self.backbone.embeddings.parameters():
                    p.requires_grad_(False)
            for attr in ("patch_embed", "cls_token", "pos_embed", "register_tokens"):
                obj = getattr(self.backbone, attr, None)
                if isinstance(obj, nn.Parameter):
                    obj.requires_grad_(False)
                elif isinstance(obj, nn.Module):
                    for p in obj.parameters():
                        p.requires_grad_(False)
            layers = None
            if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
                layers = self.backbone.encoder.layer
            elif hasattr(self.backbone, "blocks"):
                layers = self.backbone.blocks
            if layers is not None:
                for blk in list(layers)[: self.FROZEN]:
                    for p in blk.parameters():
                        p.requires_grad_(False)

        def _extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
            B, _, H, W = x.shape
            gh, gw = H // self.PATCH, W // self.PATCH
            n_patch = gh * gw
            if hasattr(self.backbone, "embeddings") and hasattr(self.backbone, "encoder"):
                with torch.no_grad():
                    h = self.backbone.embeddings(x)
                    for blk in self.backbone.encoder.layer[: self.FROZEN]:
                        h = blk(h)[0]
                for blk in self.backbone.encoder.layer[self.FROZEN :]:
                    h = blk(h)[0]
                h = self.backbone.layernorm(h)
                tok = h[:, -n_patch:, :]
                return tok.transpose(1, 2).reshape(B, -1, gh, gw)
            if hasattr(self.backbone, "forward_features"):
                feats = self.backbone.forward_features(x)["x_norm_patchtokens"]
                return feats[:, -n_patch:, :].transpose(1, 2).reshape(B, -1, gh, gw)
            out = self.backbone(x)
            if hasattr(out, "last_hidden_state"):
                tok = out.last_hidden_state[:, -n_patch:, :]
                return tok.transpose(1, 2).reshape(B, -1, gh, gw)
            raise TypeError("Unsupported DINOv2-small backbone interface")

        def forward_features_map(
            self, x: torch.Tensor, context: torch.Tensor | None = None
        ) -> torch.Tensor:
            B, _, H, W = x.shape
            target_h, target_w = max(1, H // self.STRIDE), max(1, W // self.STRIDE)
            tok = self._extract_tokens(x)
            f = self.proj(tok)
            f = F.interpolate(
                f, scale_factor=2, mode="bilinear", align_corners=False
            )
            f = self.up1(f)
            f = F.interpolate(
                f, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            fine = self.fine(x)
            if fine.shape[-2:] != (target_h, target_w):
                fine = F.interpolate(
                    fine, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            parts = [f, fine]
            if self.context > 0:
                if context is None:
                    ctx = f.new_zeros((B, self.context, target_h, target_w))
                elif context.shape[-2:] != (target_h, target_w):
                    ctx = F.interpolate(
                        context,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    ctx = context
                parts.append(ctx)
            return self.mix(torch.cat(parts, dim=1))

        def forward(
            self,
            x: torch.Tensor,
            view_ids: Any = None,
            return_aux: bool = False,
            context: torch.Tensor | None = None,
        ):
            from dinov2_pose_trainer import MonsterDINOv2PoseProbe

            feat = self.forward_features_map(x, context=context)
            logits = self.head(feat)
            coords, conf = MonsterDINOv2PoseProbe._decode_logits_dsnt(
                logits, peak_height_confidence=True
            )
            if not return_aux:
                return coords, conf, logits
            aux: dict[str, torch.Tensor] = {}
            if self.dhead is not None:
                dmap = self.dhead(feat)
                aux["relative_depth"] = dmap
                aux["point_depth"] = sample_maps(dmap, coords, normalized=True)
            if self.phead is not None and self.lhead is not None:
                aux["part_logits"] = self.phead(feat)
                aux["local_fields"] = self.lhead(feat)
                if self.shead is not None:
                    aux["skin_logits"] = self.shead(feat)
            return coords, conf, logits, aux

    class ReaderHeads(nn.Module):
        def __init__(self, num_bones, num_points):
            super().__init__()
            self.num_bones = num_bones
            self.num_points = num_points
            self.context = nn.Sequential(nn.Conv2d(2 * num_points, 32, 1), nn.GELU())
            self.fuse = nn.Sequential(
                nn.Conv2d(192 + 32, 128, 3, padding=1), nn.GroupNorm(16, 128), nn.GELU()
            )
            self.keypoints = nn.Conv2d(128, num_points, 1)
            self.phead = nn.Conv2d(128, num_bones + 1, 1)
            self.lhead = nn.Conv2d(128, 3 * num_bones, 1)
            # K-channel per-keypoint depth head (video2rig/reader/net.py: self.dhead -> k channels)
            self.dhead = nn.Conv2d(128, num_points, 1)

        def forward(self, fused, context=None, coords_norm=None):
            B, _, H, W = fused.shape
            if context is None:
                context = fused.new_zeros((B, 2 * self.num_points, H, W))
            context = F.interpolate(
                context, size=(H, W), mode="bilinear", align_corners=False
            )
            feature = self.fuse(torch.cat([fused, self.context(context)], dim=1))
            kp_logits = self.keypoints(feature)
            dmap = self.dhead(feature)
            out = {
                "part_logits": self.phead(feature),
                "local_fields": self.lhead(feature),
                "relative_depth": dmap,
            }
            if coords_norm is not None:
                out["point_depth"] = sample_maps(dmap, coords_norm, normalized=True)
            return kp_logits, out

    def heatmaps(coords, size=128, sigma=1.8):
        coords = torch.as_tensor(coords)
        yy, xx = torch.meshgrid(
            torch.arange(size, device=coords.device),
            torch.arange(size, device=coords.device),
            indexing="ij",
        )
        d = (xx - coords[..., 0, None, None] * (size - 1)) ** 2 + (
            yy - coords[..., 1, None, None] * (size - 1)
        ) ** 2
        return torch.exp(-d / (2 * sigma * sigma))

    def corrupt_context(
        uv,
        vis,
        twins=(),
        size=448,
        heat_size=112,
        sigma=2.0,
        normalized=True,
        none_prob=0.2,
        generator=None,
        return_coords=False,
    ):
        """video2rig anti-copy temporal context corruption (video2rig/reader/train.py:corrupt_context).

        Applies:
          - 1.5 px Gaussian coordinate jitter (`N(0, 1.5 / SIZE)`)
          - 5% far outliers (`U(-0.2, 0.2) * SIZE`)
          - 16% per-keypoint random dropout
          - 25% whole twin-limb left/right swap (`twins`)
          - random confidence scaling `U(0.3, 1.0)`
          - 20% full-frame context dropout (`none < 0.2`)
        """
        uv_t = torch.as_tensor(uv, dtype=torch.float32).clone()
        vis_b = torch.as_tensor(vis) > 0.3
        B, K, _ = uv_t.shape
        dev = uv_t.device
        scale = 1.0 if normalized else float(size)
        jitter = (
            torch.randn((B, K, 2), device=dev, generator=generator)
            * (1.5 / float(size))
            * scale
        )
        far = (
            torch.rand((B, K, 1), device=dev, generator=generator) < 0.05
        ).float()
        far_shift = (
            (torch.rand((B, K, 2), device=dev, generator=generator) - 0.5)
            * 0.4
            * scale
        )
        uv_c = uv_t + jitter + far * far_shift
        keep = (
            torch.rand((B, K), device=dev, generator=generator) > 0.16
        ) & vis_b
        for li, ri in twins:
            li_t = torch.as_tensor(li, dtype=torch.long, device=dev).reshape(-1)
            ri_t = torch.as_tensor(ri, dtype=torch.long, device=dev).reshape(-1)
            if li_t.numel() == 0 or ri_t.numel() != li_t.numel():
                continue
            sw = torch.rand((B,), device=dev, generator=generator) < 0.25
            sw_idx = torch.nonzero(sw, as_tuple=False).flatten()
            if sw_idx.numel() > 0:
                u_l = uv_c[sw_idx[:, None], li_t[None, :]].clone()
                u_r = uv_c[sw_idx[:, None], ri_t[None, :]].clone()
                uv_c[sw_idx[:, None], li_t[None, :]] = u_r
                uv_c[sw_idx[:, None], ri_t[None, :]] = u_l
                k_l = keep[sw_idx[:, None], li_t[None, :]].clone()
                k_r = keep[sw_idx[:, None], ri_t[None, :]].clone()
                keep[sw_idx[:, None], li_t[None, :]] = k_r
                keep[sw_idx[:, None], ri_t[None, :]] = k_l
        conf = 0.3 + 0.7 * torch.rand((B, K), device=dev, generator=generator)
        if none_prob > 0:
            none = torch.rand((B,), device=dev, generator=generator) < float(none_prob)
            keep[none] = False
        weights = keep.float() * conf
        if return_coords:
            return uv_c, weights
        uv_norm = uv_c if normalized else (uv_c / float(size))
        hm = heatmaps(uv_norm, size=heat_size, sigma=sigma)
        return hm * weights[..., None, None]

    def temporal_training_context(
        coords,
        valid,
        enabled=True,
        previous_coords=None,
        previous_valid=None,
        *,
        twins=(),
        anti_copy=False,
        heat_size=128,
        sigma=1.8,
    ):
        B, K, _ = coords.shape
        if not enabled:
            return coords.new_zeros(B, 2 * K, heat_size, heat_size)
        if anti_copy:
            contexts = []
            use = torch.rand(B, 1, 1, 1, device=coords.device) < CONTEXT_SHARE
            for lag in range(2):
                source = coords if previous_coords is None else previous_coords[:, lag]
                visibility = valid if previous_valid is None else previous_valid[:, lag]
                h = corrupt_context(
                    source,
                    visibility,
                    twins=twins,
                    heat_size=heat_size,
                    sigma=sigma,
                    normalized=True,
                    none_prob=0.0,
                )
                contexts.append(h * use)
            return torch.cat(contexts, dim=1)
        contexts = []
        use = torch.rand(B, 1, 1, 1, device=coords.device) < CONTEXT_SHARE
        for lag in range(2):
            source = coords if previous_coords is None else previous_coords[:, lag]
            visibility = valid if previous_valid is None else previous_valid[:, lag]
            noisy = (
                source
                + torch.randn_like(source) * 0.004
                + torch.randn(B, 1, 2, device=coords.device) * 0.015
            )
            h = heatmaps(noisy) * visibility[..., None, None]
            h *= use
            contexts.append(h)
        return torch.cat(contexts, dim=1)

    def smear_limbs_capsules_gpu(
        images,
        uv,
        vis,
        probability=0.35,
        generator=None,
        p=None,
    ):
        """Render a strike streak: sweep a soft capsule around 1-2 limb segments across the frame
        using 13 `torch.roll` copies and premultiplied alpha blending (video2rig/reader/train.py:smear_limbs).

        `uv` can be normalized in [0, 1] or pixel coordinates in [0, S].
        """
        if p is not None:
            probability = float(p)
        out = images.clone()
        B, _, H, W = out.shape
        S = float(max(H, W))
        dev = out.device
        uv_t = torch.as_tensor(uv, dtype=torch.float32, device=dev)
        if uv_t.numel() > 0 and float(uv_t.abs().max()) <= 2.0:
            uv_px = uv_t * torch.tensor([float(W), float(H)], device=dev)
        else:
            uv_px = uv_t.clone()
        vis_b = torch.as_tensor(vis, device=dev) > 0.3
        py, px = torch.meshgrid(
            torch.arange(H, device=dev, dtype=torch.float32),
            torch.arange(W, device=dev, dtype=torch.float32),
            indexing="ij",
        )
        for b in range(B):
            if float(torch.rand((), device=dev, generator=generator)) >= probability:
                continue
            u = uv_px[b]
            ok = (
                vis_b[b]
                & torch.isfinite(u).all(dim=-1)
                & (u[:, 0] > 0)
                & (u[:, 0] < W)
                & (u[:, 1] > 0)
                & (u[:, 1] < H)
            )
            idx = torch.nonzero(ok, as_tuple=False).squeeze(1)
            if idx.numel() < 4:
                continue
            u_ok = u[idx]
            tall = float((u_ok[:, 1].max() - u_ok[:, 1].min()).clamp(min=0.2 * S))
            n_seg = int(
                torch.randint(1, 3, (1,), device=dev, generator=generator).item()
            )
            for _ in range(n_seg):
                pick = int(
                    torch.randint(
                        0, idx.numel(), (1,), device=dev, generator=generator
                    ).item()
                )
                a = u_ok[pick]
                d_all = torch.norm(u_ok - a[None], dim=1)
                d_all[d_all == 0] = 1e9
                near = int(d_all.argmin().item())
                if float(d_all[near]) > 0.3 * S:
                    continue
                b_pt = u_ok[near]
                ab = b_pt - a
                ab2 = float((ab @ ab).clamp(min=1.0))
                t = (((px - a[0]) * ab[0] + (py - a[1]) * ab[1]) / ab2).clamp(
                    0.0, 1.0
                )
                d = torch.hypot(
                    px - (a[0] + t * ab[0]), py - (a[1] + t * ab[1])
                )
                r = (
                    0.025
                    + 0.020
                    * float(torch.rand((), device=dev, generator=generator))
                ) * S
                m = ((r - d) / (0.5 * r)).clamp(0.0, 1.0)
                ang = (
                    float(torch.atan2(ab[1], ab[0]))
                    + 0.5 * math.pi
                    + (float(torch.rand((), device=dev, generator=generator)) - 0.5)
                    * 1.2
                )
                length = (
                    0.15
                    + 0.55 * float(torch.rand((), device=dev, generator=generator))
                ) * tall
                dx = math.cos(ang) * length
                dy = math.sin(ang) * length
                img = out[b]
                src = img * m[None]
                acc = torch.zeros_like(img)
                w = torch.zeros_like(m)
                n_copies = 13
                for s in range(1, n_copies + 1):
                    f = s / n_copies
                    sx = int(round(dx * f))
                    sy = int(round(dy * f))
                    ms = torch.roll(m, shifts=(sy, sx), dims=(0, 1))
                    acc = acc + torch.roll(src, shifts=(sy, sx), dims=(1, 2))
                    w = w + ms
                smear = acc / w.clamp(min=1.0)[None]
                alpha_scale = 0.55 + 0.30 * float(
                    torch.rand((), device=dev, generator=generator)
                )
                alpha = (w / n_copies * alpha_scale).clamp(0.0, 0.85)[None]
                out[b] = img * (1.0 - alpha) + smear * alpha
        return out

    def smear_limbs_gpu(
        images,
        masks,
        directions=None,
        probability=0.35,
        generator=None,
        body_height=None,
    ):
        """13 limb-only copies displaced along each limb's image-space normal."""
        out = images.clone()
        B = images.shape[0]
        for b in range(B):
            if torch.rand((), device=images.device, generator=generator) >= probability:
                continue
            distance = (
                torch.rand((), device=images.device, generator=generator) * 0.55 + 0.15
            ) * (1.0 if body_height is None else body_height[b])
            for m in range(masks.shape[1]):
                mask = masks[b : b + 1, m : m + 1].float()
                if not bool(mask.any()):
                    continue
                normal = (
                    images.new_tensor([1.0, 0.0])
                    if directions is None
                    else torch.stack([-directions[b, m, 1], directions[b, m, 0]])
                )
                normal = normal / normal.norm().clamp_min(1e-6)
                rgb = images[b : b + 1] * mask
                for amount in (
                    torch.linspace(-1, 1, 13, device=images.device) * distance
                ):
                    dx, dy = 2 * amount * normal
                    A = torch.stack(
                        [dx * 0 + 1, dx * 0, dx, dy * 0, dy * 0 + 1, dy]
                    ).reshape(1, 2, 3)
                    grid = F.affine_grid(A, rgb.shape, align_corners=False)
                    ghost = F.grid_sample(rgb, grid, align_corners=False)
                    gm = F.grid_sample(mask, grid, align_corners=False)
                    out[b : b + 1] = out[b : b + 1] * (1 - 0.08 * gm) + 0.08 * ghost
        return out

    def inject_contact_shadow_gpu(
        images,
        feet=None,
        valid=None,
        body_height=None,
        probability=0.45,
        generator=None,
    ):
        B, _, H, W = images.shape
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H, device=images.device),
            torch.linspace(0, 1, W, device=images.device),
            indexing="ij",
        )
        if feet is None or feet.shape[1] == 0:
            return images
        height = images.new_ones(B) if body_height is None else body_height
        dx = (xx[None, None] - feet[..., 0, None, None]) / (
            0.12 * height[:, None, None, None]
        ).clamp_min(1 / W)
        dy = (yy[None, None] - feet[..., 1, None, None]) / (
            0.015 * height[:, None, None, None]
        ).clamp_min(1 / H)
        shadow = torch.exp(-(dx**2) - dy**2)
        if valid is not None:
            shadow *= valid[..., None, None]
        shadow = shadow.amax(1, keepdim=True)
        use = (
            torch.rand(B, 1, 1, 1, device=images.device, generator=generator)
            < probability
        )
        mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        return ((images * std + mean) * (1 - 0.3 * shadow * use) - mean) / std

    def decode_heatmap_parabola_torch(logits):
        """video2rig 3-point parabolic subpixel heatmap peak decode (pure PyTorch, vectorized)."""
        squeeze_batch = logits.ndim == 3
        t = logits[None].float() if squeeze_batch else logits.float()
        hm = torch.sigmoid(t.clamp(-40.0, 40.0))
        B, K, H, W = hm.shape
        flat = hm.reshape(B, K, H * W)
        conf, idx = flat.max(dim=-1)
        iy = idx // W
        ix = idx % W
        b_idx = torch.arange(B, device=logits.device)[:, None]
        k_idx = torch.arange(K, device=logits.device)[None, :]
        c = hm[b_idx, k_idx, iy, ix]
        ix_l = (ix - 1).clamp(0, max(W - 1, 0))
        ix_r = (ix + 1).clamp(0, max(W - 1, 0))
        iy_u = (iy - 1).clamp(0, max(H - 1, 0))
        iy_d = (iy + 1).clamp(0, max(H - 1, 0))
        l_val = hm[b_idx, k_idx, iy, ix_l]
        r_val = hm[b_idx, k_idx, iy, ix_r]
        u_val = hm[b_idx, k_idx, iy_u, ix]
        d_val = hm[b_idx, k_idx, iy_d, ix]
        curv_x = 2.0 * c - l_val - r_val
        curv_y = 2.0 * c - u_val - d_val
        valid_x = (ix > 0) & (ix < W - 1) & (curv_x > 1e-6)
        valid_y = (iy > 0) & (iy < H - 1) & (curv_y > 1e-6)
        dx = torch.where(
            valid_x,
            (0.5 * (r_val - l_val) / curv_x.clamp(min=1e-6)).clamp(-0.5, 0.5),
            torch.zeros_like(c),
        )
        dy = torch.where(
            valid_y,
            (0.5 * (d_val - u_val) / curv_y.clamp(min=1e-6)).clamp(-0.5, 0.5),
            torch.zeros_like(c),
        )
        denom_x = float(max(W - 1, 1))
        denom_y = float(max(H - 1, 1))
        u = ((ix.float() + dx) / denom_x).clamp(0.0, 1.0)
        v = ((iy.float() + dy) / denom_y).clamp(0.0, 1.0)
        coords = torch.stack([u, v], dim=-1)
        if squeeze_batch:
            return coords[0], conf[0]
        return coords, conf

    def dense_reader_losses(
        part_logits,
        local_fields,
        gt_part,
        gt_local,
        skin_logits=None,
        gt_skin_w=None,
        part_weight=5.0,
        local_weight=20.0,
        skin_weight=5.0,
    ):
        """Dedicated losses for the modular dense surface detector `reader-dense.pt` (video2rig/reader/dense.py)."""
        H, W = part_logits.shape[-2:]
        part = F.interpolate(
            gt_part[:, None].float(), size=(H, W), mode="nearest"
        )[:, 0].long()
        local = F.interpolate(
            gt_local.float(), size=(H, W), mode="bilinear", align_corners=False
        )
        num_classes = part_logits.shape[1]
        cls_w = part_logits.new_full((num_classes,), 3.0, dtype=torch.float32)
        cls_w[0] = 1.0
        part_loss = F.cross_entropy(part_logits.float(), part, weight=cls_w)
        B = local.shape[0]
        owner = (part - 1).clamp(min=0)
        predicted = (
            local_fields.float()
            .reshape(B, -1, 3, H, W)
            .gather(1, owner[:, None, None].expand(-1, 1, 3, -1, -1))[:, 0]
        )
        fg = (part > 0).float()
        denom = (fg.sum() * 3.0).clamp(min=1.0)
        local_loss = ((predicted - local).abs() * fg[:, None]).sum() / denom
        total = part_weight * part_loss + local_weight * local_loss
        out = {
            "part": part_loss,
            "local": local_loss,
            "total": total,
        }
        if skin_logits is not None and gt_skin_w is not None:
            sw_gt = F.interpolate(
                gt_skin_w.float(), size=(H, W), mode="bilinear", align_corners=False
            )
            sw_pred = F.softmax(skin_logits.float(), dim=1)
            skin_loss = ((sw_pred - sw_gt).abs() * fg[:, None]).sum() / denom
            out["skin"] = skin_loss
            out["total"] = out["total"] + skin_weight * skin_loss
        return out

    def reader_losses(logits, aux, gt_uv, labels):
        visible = (
            labels["visibility_ratio"] * labels["in_frame"] * labels["label_valid"]
        )
        target = heatmaps(gt_uv, logits.shape[-1]) * visible[..., None, None]
        loss = (
            (torch.sigmoid(logits.float()) - target) ** 2 * (1 + 19 * target)
        ).mean()
        dmap = aux["relative_depth"].float()
        ok = (labels["in_frame"] * labels["label_valid"]).float()
        sampled_depth = sample_maps(dmap, gt_uv.float(), normalized=True)
        if "point_depth" in labels:
            dt = labels["point_depth"].float()
            point_depth_loss = ((sampled_depth - dt).abs() * ok).sum() / ok.sum().clamp(
                min=1.0
            )
        else:
            point_depth_loss = sampled_depth.new_zeros(())

        if "part_logits" in aux and "part" in labels and "local" in labels:
            part = F.interpolate(
                labels["part"][:, None].float(), size=logits.shape[-2:], mode="nearest"
            )[:, 0].long()
            part_loss = F.cross_entropy(aux["part_logits"].float(), part)
            local = F.interpolate(
                labels["local"].float(),
                size=logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            B, _, H, W = local.shape
            owner = (part - 1).clamp(min=0)
            predicted = (
                aux["local_fields"]
                .reshape(B, -1, 3, H, W)
                .gather(1, owner[:, None, None].expand(-1, 1, 3, -1, -1))[:, 0]
            )
            fg = (part > 0).float()
            denom = fg.sum().clamp(min=1)
            local_loss = (
                F.smooth_l1_loss(predicted.float(), local, reduction="none").mean(1) * fg
            ).sum() / denom
            if "relative_depth" in labels:
                depth = F.interpolate(
                    labels["relative_depth"].float(),
                    size=logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                # Supervise all K depth channels on foreground + sampled keypoint depth
                map_depth_loss = (
                    F.smooth_l1_loss(
                        dmap, depth.expand_as(dmap), reduction="none"
                    ).mean(1)
                    * fg
                ).sum() / denom
                depth_loss = (
                    map_depth_loss + point_depth_loss
                    if "point_depth" in labels
                    else map_depth_loss
                )
            else:
                depth_loss = point_depth_loss
            return {
                "heatmap": loss,
                "part": part_loss,
                "local": local_loss,
                "dense_depth": depth_loss,
                "point_depth": point_depth_loss,
                "total": loss + part_loss + 5 * local_loss + depth_loss,
            }
        return {
            "heatmap": loss,
            "point_depth": point_depth_loss,
            "dense_depth": point_depth_loss,
            "total": loss + 20.0 * point_depth_loss,
        }

    class ReaderDataset(Dataset):
        def __init__(self, synth_dir, bone_names, augment=True, **kwargs):
            self.root = Path(synth_dir)
            self.data = json.loads((self.root / "keypoints_gt.json").read_text())
            if self.data.get("reader_contract") != READER_CONTRACT:
                raise ValueError("V3 Reader requires regenerated v3 labels")
            from rig_pose import START_POSE_CONTRACT

            if self.data.get("start_pose_contract") != START_POSE_CONTRACT:
                raise ValueError(
                    "V3 labels do not preserve the starting pose; regenerate the dataset before training"
                )
            self.samples = self.data["samples"]
            self.bone_names = bone_names
            self.augment = augment
            self.temporal_context_contract = TEMPORAL_CONTEXT_CONTRACT
            self.calib = self.data["calibration"]
            self.render_calib = self.data.get("training_calibration", self.calib)
            self.view_order = tuple(self.render_calib["view_order"])
            self.num_views = len(self.view_order)
            self.keypoint_names = self.data["keypoint_names"]
            self.window = training_window(self.samples, self.view_order,
                self.render_calib["panel_width"], self.render_calib["panel_height"])
            self.dense_surface = True
            with np.load(self.root / "rig_skinning.npz") as archive:
                self.rig_skinning = {k: archive[k] for k in archive.files}
            self.rig_skinning["bone_names"] = bone_names
            if self.data.get("context_geometry_contract") != CONTEXT_GEOMETRY_CONTRACT:
                raise ValueError(
                    "V3 labels lack geometric temporal poses; regenerate the dataset"
                )
            from pose_sampling_v3 import SAMPLING_CONTRACT

            if self.data.get("sampling_contract") != SAMPLING_CONTRACT:
                raise ValueError(
                    "V3 labels predate calibrated anatomical sampling; regenerate the dataset"
                )
            from reprojection_solver import RigModel

            self.rig = RigModel(
                self.data["bones"],
                bone_names,
                self.rig_skinning,
                self.data["extra_points"],
            )
            self.cameras = cameras_from_calibration(self.render_calib)
            from reading_pipeline import _twin_groups

            self.twins = _twin_groups(self.rig)
            self.non_prop_mask = np.array(
                [not str(b).startswith("Prop_") for b in self.rig.point_bones],
                dtype=bool,
            )
            self.foot_slots = [
                2 * i + 1
                for i, n in enumerate(bone_names)
                if n.startswith(
                    (
                        "Foot",
                        "FrontPaw",
                        "HindPaw",
                        "Tarsus",
                        "Toes",
                        "HindToes",
                        "FrontToes",
                    )
                )
            ]

        def __len__(self):
            return len(self.samples) * self.num_views

        def __getitem__(self, index):
            item = self.samples[index // self.num_views]
            view = self.view_order[index % self.num_views]
            uv = np.asarray(item["reader_points"][view]["uv"], float)
            visibility = np.asarray(item["reader_points"][view]["visibility"], float)
            W, H = self.render_calib["panel_width"], self.render_calib["panel_height"]
            p = index % self.num_views
            with Image.open(self.root / "images" / item["image"]) as im:
                image = np.asarray(
                    im.crop(
                        (0, p * H, W, (p + 1) * H)
                        if self.render_calib.get("layout") == "stacked"
                        else (p * W, 0, (p + 1) * W, H)
                    )
                )
            inside_panel = np.isfinite(uv).all(-1) & (uv >= 0).all(-1) & (uv <= [W, H]).all(-1)
            box = fixed_window(uv, inside_panel, W, H, self.window["side_px"])
            camera = self.cameras[view]
            world_center = np.mean(
                [p["head_3d"] for p in item["joints"].values()], axis=0
            )
            body_pixels = projected_body_height(camera, self.rig.height, world_center)
            # The same network sees hand/paw closeups: 0.30 H -> 224 pixels.
            size = 448
            if self.augment and np.random.rand() < 0.25:
                candidates = [
                    k
                    for k, n in enumerate(self.keypoint_names)
                    if n.endswith(".head")
                    and any(
                        n.startswith(s)
                        for s in ("Hand", "Foot", "FrontPaw", "HindPaw", "WingHand")
                    )
                    and visibility[k] > 0.3
                ]
                if candidates:
                    center = uv[np.random.choice(candidates)]
                    extent = 0.30 * body_pixels
                    box = np.r_[center - extent / 2, center + extent / 2]
            if self.augment:
                scale = np.random.uniform(0.95, 1.05)
                center = (box[:2] + box[2:]) / 2 + np.random.uniform(-6, 6, 2)
                extent = (box[2:] - box[:2]) * scale
                box = np.r_[center - extent / 2, center + extent / 2]
            image = crop_rgb(image, box, size)
            coords = map_points(uv, box, size) / size
            if self.augment:
                previous_world = temporal_pose_points(
                    self.rig, item["joints"], np.random, self.temporal_context_contract
                )
                previous = map_points(camera.project(previous_world), box, 1)
                previous_valid = ((previous >= 0) & (previous <= 1)).all(
                    -1
                ) * visibility[None]
            else:
                previous = np.zeros((2, len(coords), 2))
                previous_valid = np.zeros((2, len(coords)))
            if "points_3d" in item["reader_points"][view]:
                pts_3d = np.asarray(item["reader_points"][view]["points_3d"], dtype=float)
            else:
                pts_3d = camera.ray_at_depth(
                    uv, np.full(len(uv), float(camera.depth(world_center)))
                )
                for b_i, b_name in enumerate(self.bone_names):
                    j_info = item["joints"].get(b_name, {})
                    if "head_3d" in j_info and 2 * b_i < len(pts_3d):
                        pts_3d[2 * b_i] = np.asarray(j_info["head_3d"], dtype=float)
                    if "tail_3d" in j_info and 2 * b_i + 1 < len(pts_3d):
                        pts_3d[2 * b_i + 1] = np.asarray(j_info["tail_3d"], dtype=float)
            point_depth = sight_depth_targets(
                pts_3d, camera, self.rig.height, self.non_prop_mask
            )
            for i1, i2 in getattr(self.rig, "end_pairs", ()):
                if (
                    0 <= i1 < len(coords)
                    and 0 <= i2 < len(coords)
                    and coords[i1, 0] > coords[i2, 0]
                ):
                    coords[[i1, i2]] = coords[[i2, i1]]
                    visibility[[i1, i2]] = visibility[[i2, i1]]
                    previous[:, [i1, i2]] = previous[:, [i2, i1]]
                    previous_valid[:, [i1, i2]] = previous_valid[:, [i2, i1]]
                    point_depth[[i1, i2]] = point_depth[[i2, i1]]
            feet = coords[self.foot_slots]
            foot_valid = visibility[self.foot_slots] * ((feet >= 0) & (feet <= 1)).all(
                -1
            )
            for j, k in enumerate(self.foot_slots):
                endpoint = item["joints"][self.bone_names[k // 2]]["tail_3d"]
                if endpoint[2] - self.rig.floor > 0.05 * self.rig.height:
                    foot_valid[j] = 0
            inp = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
            inp = (
                inp - torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            ) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            inside = ((coords >= 0) & (coords <= 1)).all(-1).astype(np.float32)
            with np.load(self.root / "dense" / item["dense_map"]) as fields:
                part = torch.from_numpy(fields[view + "_part"].astype(np.int64))
                local = torch.from_numpy(fields[view + "_local"].astype(np.float32))
                depth = torch.from_numpy(fields[view + "_depth"].astype(np.float32))
            # Dense fields were rasterized in panel coordinates; one inverse grid moves every target.
            gh, gw = part.shape
            yy, xx = torch.meshgrid(
                torch.linspace(0, 1, 128), torch.linspace(0, 1, 128), indexing="ij"
            )
            x = (box[0] + xx * (box[2] - box[0])) / W * 2 - 1
            y = (box[1] + yy * (box[3] - box[1])) / H * 2 - 1
            grid = torch.stack([x, y], -1)[None].float()
            part = F.grid_sample(
                part[None, None].float(), grid, mode="nearest", align_corners=False
            )[0, 0].long()
            local = F.grid_sample(local[None], grid, align_corners=False)[0]
            depth = F.grid_sample(depth[None], grid, align_corners=False)[0]
            labels = {
                "view_id": torch.tensor(index % self.num_views),
                "in_frame": torch.from_numpy(inside),
                "label_valid": torch.ones(len(coords)),
                "visibility_ratio": torch.tensor(visibility, dtype=torch.float32),
                "part": part,
                "local": local,
                "relative_depth": depth,
                "point_depth": torch.tensor(point_depth, dtype=torch.float32),
                "previous_coords": torch.tensor(previous, dtype=torch.float32),
                "previous_valid": torch.tensor(previous_valid, dtype=torch.float32),
                "body_height": torch.tensor(
                    body_pixels / (box[3] - box[1]), dtype=torch.float32
                ),
                "feet": torch.tensor(feet, dtype=torch.float32),
                "foot_valid": torch.tensor(foot_valid, dtype=torch.float32),
            }
            return inp, torch.tensor(coords, dtype=torch.float32), labels

    def _tensor_rgb(image, device):
        a = (
            torch.from_numpy(np.asarray(image).copy())
            .permute(2, 0, 1)
            .float()
            .to(device)
            / 255
        )
        return (a - a.new_tensor([0.485, 0.456, 0.406])[:, None, None]) / a.new_tensor(
            [0.229, 0.224, 0.225]
        )[:, None, None]

    def read_crops(
        crop_model,
        panels,
        view_id,
        uv,
        conf,
        rig,
        camera,
        device,
        window_side=None,
        temporal_context_contract=TEMPORAL_CONTEXT_CONTRACT,
    ):
        """Run the dedicated hand/claw crop amplifier (`reader-crops.pt`) and replace limb-end
        keypoints where crop confidence exceeds full-body confidence (`cq > c`, video2rig/reader/crops.py).
        """
        uv_out = np.asarray(uv, dtype=float).copy()
        conf_out = np.asarray(conf, dtype=float).copy()
        T = len(panels)
        if not T or rig is None or crop_model is None:
            return uv_out, conf_out, {"groups_refined": 0, "replacements": 0}
        groups = end_groups(rig)
        if not groups:
            return uv_out, conf_out, {"groups_refined": 0, "replacements": 0}
        sides = crop_sides(rig, [camera], window_side=window_side)
        context_offsets = temporal_context_offsets(temporal_context_contract)
        ids = torch.tensor([view_id], device=device)
        replacements = 0

        def _predict_crop(image, context=None):
            x = _tensor_rgb(image, device)[None]
            return crop_model(x, ids, return_aux=True, context=context)

        with torch.inference_mode():
            for g, side in zip(groups, sides):
                ks = np.asarray(g["ks"], dtype=int)
                boxes = np.zeros((T, 4), dtype=float)
                active = np.zeros(T, dtype=bool)
                pass1_hm = []
                for t in range(T):
                    w_k = conf_out[t, ks]
                    good = (w_k > 0.2) & np.isfinite(uv_out[t, ks]).all(-1)
                    if not np.any(good):
                        pass1_hm.append(None)
                        continue
                    active[t] = True
                    w_sum = np.maximum(w_k[good].sum(), 1e-6)
                    center = (uv_out[t, ks[good]] * w_k[good, None]).sum(0) / w_sum
                    box = np.r_[center - side / 2.0, center + side / 2.0]
                    boxes[t] = box
                    _, _, logits, _ = _predict_crop(crop_rgb(panels[t], box, 448))
                    pass1_hm.append(torch.sigmoid(logits.detach().float()))

                for t in range(T):
                    if not active[t]:
                        continue
                    box = boxes[t]
                    hm_ref = pass1_hm[t]
                    assert hm_ref is not None
                    hh, ww = hm_ref.shape[-2:]
                    ctx_list = []
                    for offset in context_offsets:
                        nb = min(max(t + offset, 0), T - 1)
                        if not active[nb] or pass1_hm[nb] is None:
                            ctx_list.append(hm_ref.new_zeros(hm_ref.shape))
                            continue
                        nb_box = boxes[nb]
                        yy, xx = torch.meshgrid(
                            torch.linspace(0, 1, hh, device=device),
                            torch.linspace(0, 1, ww, device=device),
                            indexing="ij",
                        )
                        gx = (
                            (box[0] + xx * (box[2] - box[0]) - nb_box[0])
                            / max(float(nb_box[2] - nb_box[0]), 1e-6)
                            * 2.0
                            - 1.0
                        )
                        gy = (
                            (box[1] + yy * (box[3] - box[1]) - nb_box[1])
                            / max(float(nb_box[3] - nb_box[1]), 1e-6)
                            * 2.0
                            - 1.0
                        )
                        ctx_list.append(
                            F.grid_sample(
                                pass1_hm[nb],
                                torch.stack([gx, gy], -1)[None],
                                align_corners=False,
                            )
                        )
                    coords_c, conf_c, logits_c, _ = _predict_crop(
                        crop_rgb(panels[t], box, 448), torch.cat(ctx_list, dim=1)
                    )
                    hm_c = torch.sigmoid(logits_c.float())
                    cq = hm_c.amax(dim=(-2, -1))[0].cpu().numpy()
                    cu = map_points(coords_c[0].cpu().numpy() * 448, box, 448, True)
                    better = cq[ks] > conf_out[t, ks]
                    if np.any(better):
                        win_ks = ks[better]
                        uv_out[t, win_ks] = cu[win_ks]
                        conf_out[t, win_ks] = cq[win_ks]
                        replacements += int(better.sum())
        return (
            uv_out,
            conf_out,
            {"groups_refined": len(groups), "replacements": replacements},
        )

    def infer_reader_sequence(
        model,
        panels,
        view_id,
        device,
        rig=None,
        camera=None,
        first_pass=None,
        coarse_only=False,
        center_depths=None,
        window=None,
        temporal_context_contract=TEMPORAL_CONTEXT_CONTRACT,
        crop_model=None,
        dense_model=None,
        crops_model=None,
    ):
        """Coarse pass, moving windows, contract-specific temporal context and inverse TTA."""
        if crop_model is None and crops_model is not None:
            crop_model = crops_model
        context_offsets = temporal_context_offsets(temporal_context_contract)
        T = len(panels)
        K = model.num_keypoints
        if not T:
            return np.zeros((0, K, 2)), np.zeros((0, K)), {}
        W, H = panels[0].shape[1], panels[0].shape[0]
        coarse = []
        confidence = []
        first_heatmaps = []
        ids = torch.tensor([view_id], device=device)

        def predict(image, context=None):
            x = _tensor_rgb(image, device)[None]
            return model(x, ids, return_aux=True, context=context)

        with torch.inference_mode():
            if first_pass is None:
                for image in panels:
                    coords, conf, logits, _ = predict(crop_rgb(image, [0, 0, W, H]))
                    coarse.append(coords[0].float().cpu().numpy() * [W, H])
                    confidence.append(conf[0].float().cpu().numpy())
                    first_heatmaps.append(torch.sigmoid(logits.detach().float()))
            else:
                coarse, confidence, first_heatmaps = first_pass
            if coarse_only:
                return np.asarray(coarse), np.asarray(confidence), first_heatmaps
            if window is not None:
                side = scaled_window_side(window, W, H)
                centers = np.full((T, 2), np.nan)
                for t, (points, trust) in enumerate(zip(coarse, confidence)):
                    points = np.asarray(points)
                    good = (np.asarray(trust) > 0.3) & np.isfinite(points).all(-1)
                    if good.sum() >= 3:
                        centers[t] = np.median(points[good], axis=0)
                have = np.isfinite(centers).all(-1)
                if have.any():
                    for axis in range(2):
                        centers[:, axis] = np.interp(np.arange(T), np.flatnonzero(have), centers[have, axis])
                else:
                    centers[:] = [W / 2, H / 2]
                boxes = np.c_[centers - side / 2, centers + side / 2]
            else:
                boxes = np.array([moving_window(p, c, W, H) for p, c in zip(coarse, confidence)])
            if T >= 3:
                from scipy.ndimage import median_filter, gaussian_filter1d

                boxes = gaussian_filter1d(
                    median_filter(boxes, size=(3, 1)), 1.0, axis=0
                )
            coordinates = []
            final_conf = []
            point_depths = []
            transforms = [
                (0, 0, 1),
                (-6, 0, 1),
                (6, 0, 1),
                (0, -6, 1),
                (0, 6, 1),
                (0, 0, 0.95),
                (0, 0, 1.05),
                (4, -4, 1),
            ]
            for t, image in enumerate(panels):
                base = boxes[t]
                acc = None
                aux_base = None
                for shiftx, shifty, scale in transforms:
                    center = (base[:2] + base[2:]) / 2 + np.array([shiftx, shifty]) * (
                        base[2:] - base[:2]
                    ) / 448
                    extent = (base[2:] - base[:2]) * scale
                    box = np.r_[center - extent / 2, center + extent / 2]
                    context = []
                    for offset in context_offsets:
                        neighbour = t + offset
                        if temporal_context_contract == LEGACY_TEMPORAL_CONTEXT_CONTRACT and neighbour < 0:
                            context.append(
                                first_heatmaps[0].new_zeros(first_heatmaps[0].shape)
                            )
                            continue
                        # Offline neighbours at the boundary use the current edge frame,
                        # matching upstream read.py, rather than dropping half the context.
                        neighbour = min(max(neighbour, 0), T - 1)
                        hm = first_heatmaps[neighbour]
                        hh, ww = hm.shape[-2:]
                        yy, xx = torch.meshgrid(
                            torch.linspace(0, 1, hh, device=device),
                            torch.linspace(0, 1, ww, device=device),
                            indexing="ij",
                        )
                        gx = (box[0] + xx * (box[2] - box[0])) / W * 2 - 1
                        gy = (box[1] + yy * (box[3] - box[1])) / H * 2 - 1
                        context.append(
                            F.grid_sample(
                                hm, torch.stack([gx, gy], -1)[None], align_corners=False
                            )
                        )
                    coords, conf, logits, aux = predict(
                        crop_rgb(image, box), torch.cat(context, dim=1)
                    )
                    hm = torch.sigmoid(logits.float())
                    hh, ww = hm.shape[-2:]
                    yy, xx = torch.meshgrid(
                        torch.linspace(0, 1, hh, device=device),
                        torch.linspace(0, 1, ww, device=device),
                        indexing="ij",
                    )
                    gx = (base[0] + xx * (base[2] - base[0]) - box[0]) / (
                        box[2] - box[0]
                    ) * 2 - 1
                    gy = (base[1] + yy * (base[3] - base[1]) - box[1]) / (
                        box[3] - box[1]
                    ) * 2 - 1
                    aligned = F.grid_sample(
                        hm, torch.stack([gx, gy], -1)[None], align_corners=False
                    )
                    acc = aligned if acc is None else acc + aligned
                    if shiftx == 0 and shifty == 0 and scale == 1:
                        aux_base = aux
                average = acc / len(transforms)
                decode_logits = torch.logit(average.clamp(1e-5, 1 - 1e-5))
                coords, _ = model._decode_logits_dsnt(decode_logits)
                # The averaged maps retain absolute visibility evidence. Decoding
                # coordinates must not promote their normalized spatial peak mass
                # into confidence, or a near-zero occluded map becomes certain.
                conf = average.amax(dim=(-2, -1))
                uv = map_points(coords[0].cpu().numpy() * 448, base, 448, True)
                cf = conf[0].cpu().numpy()
                if dense_model is not None:
                    x_base = _tensor_rgb(crop_rgb(image, base), device)[None]
                    _, _, _, dense_aux = dense_model(x_base, ids, return_aux=True)
                    if isinstance(dense_aux, dict):
                        aux_base = {**(aux_base or {}), **dense_aux}
                if isinstance(aux_base, dict) and "relative_depth" in aux_base:
                    d_sampled = sample_maps(
                        aux_base["relative_depth"].float(),
                        coords.float(),
                        normalized=True,
                    )[0].cpu().numpy()
                    point_depths.append(d_sampled)
                else:
                    point_depths.append(np.zeros(K, dtype=np.float32))
                if (
                    rig is not None
                    and camera is not None
                    and isinstance(aux_base, dict)
                    and "part_logits" in aux_base
                    and "local_fields" in aux_base
                    and "relative_depth" in aux_base
                ):
                    from dataclasses import replace

                    extent = float(base[2] - base[0])
                    factor = 448 / extent
                    principal = (
                        np.array([camera.width / 2, camera.height / 2])
                        + camera.shift_px
                    )
                    crop_camera = replace(
                        camera,
                        width=448,
                        height=448,
                        scale=camera.scale * extent / H,
                        focal_px=camera.focal_px * factor,
                        shift_px=tuple((principal - base[:2]) * factor - 224),
                    )
                    dense, dc = joints_from_dense_fields(
                        aux_base["part_logits"][0].float().cpu().numpy(),
                        aux_base["local_fields"][0].float().cpu().numpy(),
                        aux_base["relative_depth"][0].float().cpu().numpy(),
                        crop_camera,
                        rig,
                        float(center_depths[t])
                        if center_depths is not None
                        else float(
                            camera.depth(
                                np.mean(np.array(list(rig.h.values())), axis=0)
                            )
                        ),
                    )
                    good = np.isfinite(dense).all(-1) & (dc > 0.7)
                    if good.any():
                        dense_uv = camera.project(dense[good])
                        agreement = np.linalg.norm(
                            dense_uv - uv[good], axis=-1
                        ) < 0.12 * min(W, H)
                        indices = np.flatnonzero(good)[agreement]
                        uv[indices] = 0.5 * uv[indices] + 0.5 * camera.project(
                            dense[indices]
                        )
                        cf[indices] = np.maximum(cf[indices], dc[indices])
                if crop_model is None:
                    # Same weights refine dense fingers/toes from local 224-pixel crops when no dedicated reader-crops.pt is provided.
                    for bone in rig.names if rig else []:
                        if not bone.startswith(
                            ("Hand", "Foot", "FrontPaw", "HindPaw", "WingHand")
                        ):
                            continue
                        anchor = 2 * rig.index[bone]
                        if cf[anchor] < 0.4:
                            continue
                        reference = np.mean(np.array(list(rig.h.values())), axis=0)
                        if center_depths is not None:
                            reference = camera.ray_at_depth(uv[anchor], center_depths[t])
                        extent = 0.30 * projected_body_height(camera, rig.height, reference)
                        box = np.r_[uv[anchor] - extent / 2, uv[anchor] + extent / 2]
                        refined, rc, _, _ = predict(crop_rgb(image, box, 224))
                        ru = map_points(refined[0].cpu().numpy() * 224, box, 224, True)
                        rc = rc[0].cpu().numpy()
                        descendants = {bone}
                        changed = True
                        while changed:
                            before = len(descendants)
                            descendants.update(
                                n for n in rig.names if rig.parents[n] in descendants
                            )
                            changed = len(descendants) > before
                        for k, owner in enumerate(rig.point_bones):
                            if (
                                owner in descendants
                                and (
                                    owner.startswith(("Finger", "Toe"))
                                    or ".side" in rig.point_names[k]
                                )
                                and rc[k] > 0.5
                            ):
                                uv[k] = ru[k]
                                cf[k] = rc[k]
                coordinates.append(uv)
                final_conf.append(cf)
            coords_arr = np.asarray(coordinates)
            conf_arr = np.asarray(final_conf)
            crop_diag = None
            if crop_model is not None and rig is not None and camera is not None:
                win_side = (
                    scaled_window_side(window, W, H) if window is not None else None
                )
                coords_arr, conf_arr, crop_diag = read_crops(
                    crop_model,
                    panels,
                    view_id,
                    coords_arr,
                    conf_arr,
                    rig,
                    camera,
                    device,
                    window_side=win_side,
                    temporal_context_contract=temporal_context_contract,
                )
        return (
            coords_arr,
            conf_arr,
            {
                "passes": 2,
                "tta": 8,
                "context_lags": [-offset for offset in context_offsets],
                "context_offsets": list(context_offsets),
                "temporal_context_contract": temporal_context_contract,
                "windows": boxes.tolist(),
                "crop_reader_pixels": 448 if crop_model is not None else 224,
                "crop_reader_report": crop_diag,
                "point_depth": np.asarray(point_depths, dtype=np.float32),
                "training_window": window,
            },
        )
