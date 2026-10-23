#!/usr/bin/env python3
"""3D Pose FK Tweaking, Fast LBS / Blender Single-Frame Rendering, Bone Attribution & Diagnostic Cards."""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


# ==============================================================================
# 1. Multi-View Studio Layout Resolution
# ==============================================================================

def resolve_views_for_canvas(
    width: int,
    height: int,
    layout: str = "auto",
    explicit_views: list[str] | None = None,
    explicit_rois: list[str] | None = None,
    calib: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Resolves view rectangles `[left, top, right, bottom]` and optional ROI sub-rectangles."""
    views: list[dict[str, Any]] = []
    if explicit_views:
        for spec in explicit_views:
            name, rect = spec.split(":", 1)
            box = [int(n) for n in rect.split(",")]
            views.append({"name": name.strip(), "box": box, "regions": []})
    else:
        mode = (layout or "auto").strip().lower()
        if mode == "auto" and calib:
            v_order = calib.get("view_order")
            pw = calib.get("panel_width")
            ph = calib.get("panel_height")
            if isinstance(v_order, list) and len(v_order) in (2, 3) and pw and ph:
                if int(pw) * len(v_order) == width and int(ph) == height:
                    for i, v_name in enumerate(v_order):
                        views.append({
                            "name": str(v_name),
                            "box": [i * int(pw), 0, (i + 1) * int(pw), height],
                            "regions": [],
                        })
        if not views:
            if mode == "auto":
                aspect = width / max(1.0, float(height))
                if aspect >= 2.35 and width % 3 == 0:
                    mode = "3view"
                elif aspect >= 1.50 and width % 2 == 0:
                    mode = "2view"
                else:
                    mode = "single"

            if mode == "2view":
                if width % 2 != 0:
                    raise ValueError(f"2view layout requires even canvas width, got {width}x{height}")
                pw = width // 2
                views = [
                    {"name": "front", "box": [0, 0, pw, height], "regions": []},
                    {"name": "side", "box": [pw, 0, width, height], "regions": []},
                ]
            elif mode == "3view":
                if width % 3 != 0:
                    raise ValueError(f"3view layout requires canvas width divisible by 3, got {width}x{height}")
                pw = width // 3
                views = [
                    {"name": "left", "box": [0, 0, pw, height], "regions": []},
                    {"name": "front", "box": [pw, 0, 2 * pw, height], "regions": []},
                    {"name": "right", "box": [2 * pw, 0, width, height], "regions": []},
                ]
            elif mode == "single":
                views = [{"name": "front", "box": [0, 0, width, height], "regions": []}]
            else:
                raise ValueError(f"Unsupported layout mode: {layout!r} (expected auto, single, 2view, or 3view)")

    if explicit_rois:
        view_map = {v["name"]: v for v in views}
        for roi_spec in explicit_rois:
            parts = roi_spec.split(":", 2)
            if len(parts) != 3:
                raise ValueError(f"Invalid --roi spec {roi_spec!r}; expected VIEW:NAME:left,top,right,bottom")
            v_name, r_name, rect = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if v_name not in view_map:
                raise ValueError(f"ROI references unknown view {v_name!r}")
            r_box = [int(n) for n in rect.split(",")]
            view_map[v_name].setdefault("regions", []).append({"name": r_name, "box": r_box})

    return views


# ==============================================================================
# 2. SO(3) Rotation Math, Orthographic Projection & Forward Kinematics (FK)
# ==============================================================================

def euler_xyz_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Extrinsic XYZ Euler angles (radians) -> 3x3 rotation matrix `Rz(rz) @ Ry(ry) @ Rx(rx)`."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )


def matrix_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> extrinsic XYZ Euler angles `[rx, ry, rz]` in radians."""
    ry = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    return np.array(
        [
            math.atan2(float(R[2, 1]), float(R[2, 2])),
            ry,
            math.atan2(float(R[1, 0]), float(R[0, 0])),
        ],
        dtype=np.float64,
    )


def axis_angle_to_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation formula around unit 3D vector `axis` by `angle_rad`."""
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle_rad) < 1e-12:
        return np.eye(3, dtype=np.float64)
    u = np.asarray(axis, dtype=np.float64) / norm
    K = np.array(
        [[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(angle_rad) * K + (1.0 - math.cos(angle_rad)) * (K @ K)


def minimum_swing_matrix(curr_dir: np.ndarray, target_dir: np.ndarray) -> np.ndarray:
    """Minimum-angle rotation matrix in SO(3) mapping `curr_dir -> target_dir`."""
    u = curr_dir / max(1e-8, float(np.linalg.norm(curr_dir)))
    v = target_dir / max(1e-8, float(np.linalg.norm(target_dir)))
    cross = np.cross(u, v)
    sin_a = float(np.linalg.norm(cross))
    cos_a = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if sin_a < 1e-6:
        if cos_a >= 0.0:
            return np.eye(3, dtype=np.float64)
        ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(float(u[0])) < 0.9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        ax = np.cross(u, ortho)
        ax /= max(1e-8, float(np.linalg.norm(ax)))
        return 2.0 * np.outer(ax, ax) - np.eye(3, dtype=np.float64)
    axis = cross / sin_a
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + sin_a * K + (1.0 - cos_a) * (K @ K)


def _default_rest_rot_matrix(d0: np.ndarray) -> np.ndarray:
    x_proj = np.array([1.0, 0.0, 0.0], dtype=np.float64) - d0 * float(d0[0])
    if float(np.linalg.norm(x_proj)) < 1e-6:
        x_proj = np.array([0.0, 0.0, 1.0], dtype=np.float64) - d0 * float(d0[2])
    x_ax = x_proj / max(1e-8, float(np.linalg.norm(x_proj)))
    z_ax = np.cross(x_ax, d0)
    z_ax /= max(1e-8, float(np.linalg.norm(z_ax)))
    return np.column_stack([x_ax, d0, z_ax])


def view_rotation_world_axis(view_name: str) -> np.ndarray:
    """Returns the 3D world rotation axis corresponding to a clockwise 2D rotation (`+u -> +v`)
    on the image plane of `view_name`:
      - `front`: `+u = +X`, `+v = -Z` => `+u x +v = +Y = [0, 1, 0]`
      - `side` / `left`: `+u = +Y`, `+v = -Z` => `+u x +v = -X = [-1, 0, 0]`
      - `right`: `+u = -Y`, `+v = -Z` => `+u x +v = +X = [1, 0, 0]`
    """
    v_low = view_name.strip().lower()
    if v_low in ("side", "left"):
        return np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    if v_low == "right":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


def project_3d_points_to_ortho_view(
    pts_3d: np.ndarray,
    calib: dict[str, Any],
    view_name: str,
    panel_w: int | None = None,
    panel_h: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Projects `(..., 3)` 3D world coordinates into `(..., 2)` view-local pixel coordinates `(u, v)`
    and `(...,)` relative camera depth (smaller = closer to camera).
    """
    pts = np.asarray(pts_3d, dtype=np.float64)
    cx, cy, cz = (float(c) for c in calib.get("center", [0.0, 0.0, 0.5]))
    W = float(panel_w if panel_w is not None else calib.get("panel_width", 672.0))
    H = float(panel_h if panel_h is not None else calib.get("panel_height", 768.0))
    S = max(1e-4, float(calib.get("ortho_scale", 1.5)))
    D = max(1e-4, float(calib.get("depth_scale", S / 1.48)))
    span_x = S * (W / H)
    span_z = S

    px, py, pz = pts[..., 0], pts[..., 1], pts[..., 2]
    v_px = (0.5 - (pz - cz) / span_z) * H
    v_low = view_name.strip().lower()
    if v_low in ("side", "left"):
        u_px = ((py - cy) / span_x + 0.5) * W
        depth_rel = (cx - px) / D
    elif v_low == "right":
        u_px = ((cy - py) / span_x + 0.5) * W
        depth_rel = (px - cx) / D
    else:
        u_px = ((px - cx) / span_x + 0.5) * W
        depth_rel = (py - cy) / D
    return np.stack([u_px, v_px], axis=-1), depth_rel


def project_bones_to_views(
    frame_bones: dict[str, dict[str, Any]],
    bones_meta: dict[str, dict[str, Any]],
    calib: dict[str, Any],
    views: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Projects all bones' `head_3d` and `tail_3d` into each view's local pixel coordinates."""
    bone_names = sorted(frame_bones.keys())
    if not bone_names:
        return {}
    heads = np.array([frame_bones[b]["head_3d"] for b in bone_names], dtype=np.float64)
    tails = np.array([frame_bones[b]["tail_3d"] for b in bone_names], dtype=np.float64)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for view in views:
        v_name = view["name"]
        box = view["box"]
        pw, ph = box[2] - box[0], box[3] - box[1]
        uv_h, d_h = project_3d_points_to_ortho_view(heads, calib, v_name, panel_w=pw, panel_h=ph)
        uv_t, d_t = project_3d_points_to_ortho_view(tails, calib, v_name, panel_w=pw, panel_h=ph)
        v_dict: dict[str, dict[str, Any]] = {}
        for i, b in enumerate(bone_names):
            info = bones_meta.get(b, {})
            v_dict[b] = {
                "head_uv": [round(float(uv_h[i, 0]), 2), round(float(uv_h[i, 1]), 2)],
                "tail_uv": [round(float(uv_t[i, 0]), 2), round(float(uv_t[i, 1]), 2)],
                "depth_rel": round(float(0.5 * (d_h[i] + d_t[i])), 4),
                "kind": str(info.get("kind", info.get("role", "limb"))),
                "sub_role": str(info.get("sub_role", "")),
                "fixed_local": bool(info.get("fixed_local", False)),
                "parent": info.get("parent"),
            }
        out[v_name] = v_dict
    return out


def parse_and_apply_pose_tweaks(
    rig_meta: dict[str, Any],
    pose_data: dict[str, Any] | None,
    frame_idx: int = 0,
    tweaks: list[str] | None = None,
    set_eulers: list[str] | None = None,
    view_tweaks: list[str] | None = None,
    root_offset: str | list[float] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Extracts the bone pose at `frame_idx`, applies local Euler or view-plane rotation tweaks,
    and propagates all positions and rotations top-down via exact rest-offset Forward Kinematics (FK).
    """
    bones_meta = rig_meta.get("bones", {})
    bone_names = sorted(bones_meta.keys())
    if not bone_names:
        raise ValueError("rig_meta contains no bones")

    # Reconstruct depths & topological order
    def _depth(b: str) -> int:
        d = 0
        curr = bones_meta.get(b, {}).get("parent")
        seen = {b}
        while curr and curr in bones_meta and curr not in seen:
            seen.add(curr)
            d += 1
            curr = bones_meta[curr].get("parent")
        return d

    order = sorted(bone_names, key=lambda b: (int(bones_meta[b].get("depth", _depth(b))), b))
    root_name = order[0]

    rest_H: dict[str, np.ndarray] = {}
    rest_T: dict[str, np.ndarray] = {}
    rest_R: dict[str, np.ndarray] = {}
    for b in bone_names:
        info = bones_meta[b]
        h0 = np.asarray(info.get("rest_head_world", [0.0, 0.0, 0.0]), dtype=np.float64)
        t0 = np.asarray(info.get("rest_tail_world", h0 + np.array([0.0, 0.0, 0.05])), dtype=np.float64)
        rest_H[b] = h0
        rest_T[b] = t0
        if "rest_rot_mat_world" in info:
            rest_R[b] = np.asarray(info["rest_rot_mat_world"], dtype=np.float64)
        else:
            v0 = t0 - h0
            d0 = v0 / max(1e-8, float(np.linalg.norm(v0)))
            rest_R[b] = _default_rest_rot_matrix(d0)

    # Extract target frame from pose_data
    raw_frame_bones: dict[str, Any] = {}
    if pose_data is not None:
        if "frames" in pose_data and isinstance(pose_data["frames"], list):
            frames_list = pose_data["frames"]
            matched = next((f for f in frames_list if int(f.get("frame", -1)) == int(frame_idx)), None)
            if matched is None:
                if 0 <= int(frame_idx) < len(frames_list):
                    matched = frames_list[int(frame_idx)]
                else:
                    raise ValueError(f"Frame {frame_idx} not found in pose_json (len={len(frames_list)})")
            raw_frame_bones = copy.deepcopy(matched.get("bones", matched.get("joints", {})))
        elif "bones" in pose_data or "joints" in pose_data:
            raw_frame_bones = copy.deepcopy(pose_data.get("bones", pose_data.get("joints", {})))
        else:
            raw_frame_bones = copy.deepcopy(pose_data)

    # Extract initial world rotations and root head position
    orig_R: dict[str, np.ndarray] = {}
    for b in order:
        p = bones_meta[b].get("parent")
        R_init_orig = (orig_R[p] @ (rest_R[p].T @ rest_R[b])) if (p and p in orig_R) else rest_R[b]
        b_obs = raw_frame_bones.get(b, {})
        if "rot_mat_world" in b_obs:
            orig_R[b] = np.asarray(b_obs["rot_mat_world"], dtype=np.float64)
        elif "euler_xyz" in b_obs:
            eul = [float(x) for x in b_obs["euler_xyz"]]
            orig_R[b] = R_init_orig @ euler_xyz_to_matrix(eul[0], eul[1], eul[2])
        elif "head_3d" in b_obs and "tail_3d" in b_obs:
            v_obs = np.asarray(b_obs["tail_3d"], dtype=np.float64) - np.asarray(b_obs["head_3d"], dtype=np.float64)
            if float(np.linalg.norm(v_obs)) > 1e-6:
                orig_R[b] = minimum_swing_matrix(R_init_orig[:, 1], v_obs) @ R_init_orig
            else:
                orig_R[b] = R_init_orig.copy()
        else:
            orig_R[b] = R_init_orig.copy()

    # Compute each bone's current local rotation delta in rest-local frame
    local_deltas: dict[str, np.ndarray] = {}
    for b in order:
        p = bones_meta[b].get("parent")
        R_init_orig = (orig_R[p] @ (rest_R[p].T @ rest_R[b])) if (p and p in orig_R) else rest_R[b]
        local_deltas[b] = R_init_orig.T @ orig_R[b]

    root_h = np.asarray(
        raw_frame_bones.get(root_name, {}).get("head_3d", rest_H[root_name]),
        dtype=np.float64,
    ).copy()

    # Parse tweak specifications
    tweak_records: list[dict[str, Any]] = []
    if root_offset:
        vals = [float(x) for x in (root_offset.split(",") if isinstance(root_offset, str) else root_offset)]
        if len(vals) != 3:
            raise ValueError(f"--tweak-root expects dx,dy,dz, got {root_offset!r}")
        root_h += np.array(vals, dtype=np.float64)
        tweak_records.append({"kind": "root_offset", "delta_xyz": vals})

    for spec in (set_eulers or []):
        b_name, angles_str = spec.split(":", 1)
        b_name = b_name.strip()
        if b_name not in bones_meta:
            raise ValueError(f"--set-euler references unknown bone {b_name!r}")
        vals_deg = [float(x) for x in angles_str.split(",")]
        if len(vals_deg) != 3:
            raise ValueError(f"--set-euler expects BONE:rx_deg,ry_deg,rz_deg, got {spec!r}")
        local_deltas[b_name] = euler_xyz_to_matrix(
            math.radians(vals_deg[0]), math.radians(vals_deg[1]), math.radians(vals_deg[2])
        )
        tweak_records.append({"kind": "set_euler", "bone": b_name, "euler_deg": vals_deg})

    for spec in (tweaks or []):
        b_name, angles_str = spec.split(":", 1)
        b_name = b_name.strip()
        if b_name not in bones_meta:
            raise ValueError(f"--tweak references unknown bone {b_name!r}")
        vals_deg = [float(x) for x in angles_str.split(",")]
        if len(vals_deg) != 3:
            raise ValueError(f"--tweak expects BONE:drx_deg,dry_deg,drz_deg, got {spec!r}")
        cur_eul = matrix_to_euler_xyz(local_deltas[b_name])
        new_eul = cur_eul + np.radians(vals_deg)
        local_deltas[b_name] = euler_xyz_to_matrix(float(new_eul[0]), float(new_eul[1]), float(new_eul[2]))
        tweak_records.append({"kind": "tweak_local_euler", "bone": b_name, "delta_euler_deg": vals_deg})

    # View-plane tweaks are applied during the forward pass in world/view space
    view_tweak_map: dict[str, list[tuple[str, float]]] = {}
    for spec in (view_tweaks or []):
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"--tweak-view expects VIEW:BONE:deg, got {spec!r}")
        v_name, b_name, deg_val = parts[0].strip(), parts[1].strip(), float(parts[2].strip())
        if b_name not in bones_meta:
            raise ValueError(f"--tweak-view references unknown bone {b_name!r}")
        view_tweak_map.setdefault(b_name, []).append((v_name, deg_val))
        tweak_records.append({"kind": "tweak_view_rot", "view": v_name, "bone": b_name, "delta_deg": deg_val})

    # Forward Kinematics propagation in topological order
    posed_R: dict[str, np.ndarray] = {}
    posed_H: dict[str, np.ndarray] = {}
    posed_T: dict[str, np.ndarray] = {}
    out_bones: dict[str, dict[str, Any]] = {}

    for b in order:
        info = bones_meta[b]
        p = info.get("parent")
        if p and p in posed_R:
            R_init = posed_R[p] @ (rest_R[p].T @ rest_R[b])
            h_curr = posed_H[p] + posed_R[p] @ (rest_R[p].T @ (rest_H[b] - rest_H[p]))
        else:
            R_init = rest_R[b]
            h_curr = root_h.copy()

        R_w = R_init @ local_deltas[b]
        if b in view_tweak_map:
            for v_name, deg_val in view_tweak_map[b]:
                w_axis = view_rotation_world_axis(v_name)
                R_view_rot = axis_angle_to_matrix(w_axis, math.radians(deg_val))
                R_w = R_view_rot @ R_w

        rest_vec = rest_T[b] - rest_H[b]
        t_curr = h_curr + R_w @ (rest_R[b].T @ rest_vec)

        posed_R[b] = R_w
        posed_H[b] = h_curr
        posed_T[b] = t_curr

        base_entry = copy.deepcopy(raw_frame_bones.get(b, {}))
        base_entry["head_3d"] = [round(float(x), 5) for x in h_curr]
        base_entry["tail_3d"] = [round(float(x), 5) for x in t_curr]
        base_entry["rot_mat_world"] = [[round(float(x), 5) for x in row] for row in R_w]
        out_bones[b] = base_entry

    return out_bones, {"frame": int(frame_idx), "tweaks_applied": tweak_records}


def write_updated_pose_json(
    orig_pose_data: dict[str, Any] | None,
    frame_idx: int,
    updated_bones: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    """Writes updated bone transforms back to a trajectory or single-frame pose JSON."""
    if orig_pose_data is not None and "frames" in orig_pose_data and isinstance(orig_pose_data["frames"], list):
        payload = copy.deepcopy(orig_pose_data)
        updated = False
        for item in payload["frames"]:
            if int(item.get("frame", -1)) == int(frame_idx):
                item["bones"] = copy.deepcopy(updated_bones)
                updated = True
                break
        if not updated and 0 <= int(frame_idx) < len(payload["frames"]):
            payload["frames"][int(frame_idx)]["bones"] = copy.deepcopy(updated_bones)
    else:
        payload = copy.deepcopy(orig_pose_data) if isinstance(orig_pose_data, dict) else {}
        payload["frame"] = int(frame_idx)
        payload["bones"] = copy.deepcopy(updated_bones)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ==============================================================================
# 3. Fast LBS In-Memory Renderer & Headless Blender Single-Frame Renderer
# ==============================================================================

def lbs_deform_vertices(
    rest_verts: np.ndarray,
    skin_indices: np.ndarray,
    skin_weights: np.ndarray,
    rest_heads: np.ndarray,
    rest_rots: np.ndarray,
    posed_heads: np.ndarray,
    posed_rots: np.ndarray,
) -> np.ndarray:
    """Vectorized Linear Blend Skinning (LBS) transforming rest-pose vertices `(V, 3)` into posed 3D space."""
    v_rest = np.asarray(rest_verts, dtype=np.float64)
    idx = np.asarray(skin_indices, dtype=np.int64)
    w = np.asarray(skin_weights, dtype=np.float64)
    if idx.ndim == 1:
        idx = idx[:, None]
    if w.ndim == 1:
        w = w[:, None]
    h_rest = np.asarray(rest_heads, dtype=np.float64)
    r_rest = np.asarray(rest_rots, dtype=np.float64)
    h_posed = np.asarray(posed_heads, dtype=np.float64)
    r_posed = np.asarray(posed_rots, dtype=np.float64)

    B = h_rest.shape[0]
    idx_safe = np.clip(idx, 0, B - 1)
    g_mats = np.einsum("bij,bkj->bik", r_posed, r_rest)
    offsets = v_rest[:, None, :] - h_rest[idx_safe]
    transformed = np.einsum("vmij,vmj->vmi", g_mats[idx_safe], offsets) + h_posed[idx_safe]
    return np.sum(w[:, :, None] * transformed, axis=1)


def resolve_blender_binary() -> str:
    env_bin = os.environ.get("BLENDER_BIN")
    if env_bin and Path(env_bin).is_file():
        return env_bin
    which_bin = shutil.which("blender") or shutil.which("Blender")
    if which_bin:
        return which_bin
    mac_app = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    if mac_app.is_file():
        return str(mac_app)
    raise FileNotFoundError("Blender executable not found. Set BLENDER_BIN or install Blender.")


_BLENDER_EXTRACT_SKINNING_SCRIPT = r"""
import json, os, sys
from pathlib import Path
import bpy
import numpy as np

cfg = json.loads(Path(os.environ["FRAME_REVIEW_BLENDER_CFG"]).read_text(encoding="utf-8"))
bpy.ops.wm.open_mainfile(filepath=cfg["blend_path"])
bone_names = cfg["bone_names"]
b_idx_map = {n: i for i, n in enumerate(bone_names)}

meshes = [
    o for o in bpy.context.scene.objects
    if o.type == "MESH" and o.name != "Studio_Ground" and not o.hide_render
]
all_verts = []
all_tris = []
all_idx = []
all_w = []
v_offset = 0

for obj in meshes:
    mw = obj.matrix_world
    me = obj.data
    me.calc_loop_triangles()
    vg_to_bone = {}
    for vg in obj.vertex_groups:
        if vg.name in b_idx_map:
            vg_to_bone[vg.index] = b_idx_map[vg.name]

    for v in me.vertices:
        co_w = mw @ v.co
        all_verts.append([float(co_w.x), float(co_w.y), float(co_w.z)])
        pairs = []
        for g in v.groups:
            if g.group in vg_to_bone and g.weight > 1e-5:
                pairs.append((vg_to_bone[g.group], float(g.weight)))
        pairs.sort(key=lambda x: x[1], reverse=True)
        pairs = pairs[:4]
        while len(pairs) < 4:
            pairs.append((0, 0.0))
        w_sum = sum(p[1] for p in pairs)
        if w_sum > 1e-6:
            pairs = [(p[0], p[1] / w_sum) for p in pairs]
        else:
            pairs = [(0, 1.0), (0, 0.0), (0, 0.0), (0, 0.0)]
        all_idx.append([p[0] for p in pairs])
        all_w.append([p[1] for p in pairs])

    for lt in me.loop_triangles:
        all_tris.append([
            v_offset + int(lt.vertices[0]),
            v_offset + int(lt.vertices[1]),
            v_offset + int(lt.vertices[2]),
        ])
    v_offset += len(me.vertices)

np.savez_compressed(
    cfg["out_npz"],
    rest_verts=np.asarray(all_verts, dtype=np.float32),
    triangles=np.asarray(all_tris, dtype=np.int32),
    skin_indices=np.asarray(all_idx, dtype=np.int32),
    skin_weights=np.asarray(all_w, dtype=np.float32),
)
"""


def load_or_extract_rig_skinning(
    rig_meta: dict[str, Any],
    rig_skinning_path: Path | None = None,
    blend_path: Path | None = None,
    cache_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Loads `(rest_verts, triangles, skin_indices, skin_weights)` from `rig_skinning.npz`,
    `rig_meta["rig_skinning"]`, or extracts and caches it from `blend_path` via headless Blender.
    """
    if rig_skinning_path and rig_skinning_path.is_file():
        with np.load(rig_skinning_path, allow_pickle=False) as data:
            return {k: data[k] for k in ("rest_verts", "triangles", "skin_indices", "skin_weights") if k in data}

    if isinstance(rig_meta.get("rig_skinning"), dict):
        rs = rig_meta["rig_skinning"]
        if "rest_verts" in rs and "skin_indices" in rs and "skin_weights" in rs:
            return {
                "rest_verts": np.asarray(rs["rest_verts"], dtype=np.float32),
                "triangles": np.asarray(rs.get("triangles", []), dtype=np.int32),
                "skin_indices": np.asarray(rs["skin_indices"], dtype=np.int32),
                "skin_weights": np.asarray(rs["skin_weights"], dtype=np.float32),
            }

    if blend_path and blend_path.is_file():
        import hashlib
        st = blend_path.stat()
        key = hashlib.sha256(f"{blend_path.resolve()}:{st.st_size}:{st.st_mtime_ns}".encode("utf-8")).hexdigest()[:12]
        default_cache = Path(tempfile.gettempdir()) / "frame_overlay_review_skin_cache"
        default_cache.mkdir(parents=True, exist_ok=True)
        target_npz = (cache_dir or default_cache) / f"skinning_{blend_path.stem}_{key}.npz"
        if not target_npz.is_file():
            blender_bin = resolve_blender_binary()
            bone_names = sorted(rig_meta.get("bones", {}).keys())
            with tempfile.TemporaryDirectory(prefix="frame_review_skin_") as tmp_d:
                cfg_file = Path(tmp_d) / "cfg.json"
                script_file = Path(tmp_d) / "extract.py"
                cfg_file.write_text(
                    json.dumps({
                        "blend_path": str(blend_path.resolve()),
                        "bone_names": bone_names,
                        "out_npz": str(target_npz.resolve()),
                    }),
                    encoding="utf-8",
                )
                script_file.write_text(_BLENDER_EXTRACT_SKINNING_SCRIPT, encoding="utf-8")
                env = os.environ.copy()
                env["FRAME_REVIEW_BLENDER_CFG"] = str(cfg_file)
                subprocess.run(
                    [blender_bin, "--background", "--factory-startup", "--python-exit-code", "1", "--python", str(script_file)],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=True,
                )
        with np.load(target_npz, allow_pickle=False) as data:
            return {k: data[k] for k in ("rest_verts", "triangles", "skin_indices", "skin_weights") if k in data}

    raise ValueError("LBS rendering requires --rig-skinning <npz> or --blend <blend_file>")


def render_pose_lbs_canvas(
    rig_meta: dict[str, Any],
    frame_bones: dict[str, dict[str, Any]],
    views: list[dict[str, Any]],
    canvas_size: tuple[int, int],
    skinning: dict[str, np.ndarray],
    bg_rgb: tuple[int, int, int] = (9, 10, 14),
) -> Image.Image:
    """Renders deformed 3D mesh via LBS + Z-sorted triangle scanline rasterization across all `views`.
    Returns an RGBA `Image.Image` of size `canvas_size` where the alpha channel is the exact silhouette mask.
    """
    bones_meta = rig_meta.get("bones", {})
    calib = rig_meta.get("calibration", {})
    bone_names = sorted(bones_meta.keys())
    B = len(bone_names)

    rest_H = np.zeros((B, 3), dtype=np.float64)
    rest_R = np.zeros((B, 3, 3), dtype=np.float64)
    posed_H = np.zeros((B, 3), dtype=np.float64)
    posed_R = np.zeros((B, 3, 3), dtype=np.float64)

    for i, b in enumerate(bone_names):
        info = bones_meta[b]
        h0 = np.asarray(info.get("rest_head_world", [0.0, 0.0, 0.0]), dtype=np.float64)
        t0 = np.asarray(info.get("rest_tail_world", h0 + np.array([0.0, 0.0, 0.05])), dtype=np.float64)
        rest_H[i] = h0
        if "rest_rot_mat_world" in info:
            rest_R[i] = np.asarray(info["rest_rot_mat_world"], dtype=np.float64)
        else:
            v0 = t0 - h0
            d0 = v0 / max(1e-8, float(np.linalg.norm(v0)))
            rest_R[i] = _default_rest_rot_matrix(d0)
        b_cur = frame_bones.get(b, {})
        posed_H[i] = np.asarray(b_cur.get("head_3d", h0), dtype=np.float64)
        posed_R[i] = np.asarray(b_cur.get("rot_mat_world", rest_R[i]), dtype=np.float64)

    v_posed = lbs_deform_vertices(
        skinning["rest_verts"],
        skinning["skin_indices"],
        skinning["skin_weights"],
        rest_H,
        rest_R,
        posed_H,
        posed_R,
    )
    tris = np.asarray(skinning.get("triangles", []), dtype=np.int64)
    W_tot, H_tot = int(canvas_size[0]), int(canvas_size[1])
    canvas_rgba = np.zeros((H_tot, W_tot, 4), dtype=np.uint8)
    canvas_rgba[:, :, 0] = bg_rgb[0]
    canvas_rgba[:, :, 1] = bg_rgb[1]
    canvas_rgba[:, :, 2] = bg_rgb[2]
    canvas_rgba[:, :, 3] = 0

    # Precompute world triangle normals for studio clay shading
    has_tris = tris.ndim == 2 and tris.shape[0] > 0
    if has_tris:
        v0 = v_posed[tris[:, 0]]
        v1 = v_posed[tris[:, 1]]
        v2 = v_posed[tris[:, 2]]
        fnorm = np.cross(v1 - v0, v2 - v0)
        fnorm_len = np.linalg.norm(fnorm, axis=-1, keepdims=True)
        fnorm = np.where(fnorm_len > 1e-9, fnorm / np.maximum(fnorm_len, 1e-9), np.array([0.0, -1.0, 0.0]))
        key_dir = np.array([0.42, -0.58, 0.69], dtype=np.float64)
        fill_dir = np.array([-0.62, 0.52, 0.58], dtype=np.float64)
        dot_k = np.abs(fnorm @ key_dir)
        dot_f = np.maximum(0.0, fnorm @ fill_dir)
        intensity = np.clip(0.32 + 0.52 * dot_k + 0.16 * dot_f, 0.22, 0.98)
        tri_rgb = np.column_stack([
            np.rint(intensity * 202.0).astype(np.uint8),
            np.rint(intensity * 190.0).astype(np.uint8),
            np.rint(intensity * 174.0).astype(np.uint8),
        ])

    for view in views:
        v_name = view["name"]
        l, t, r, b_edge = view["box"]
        pw, ph = r - l, b_edge - t
        uv_px, depth_rel = project_3d_points_to_ortho_view(v_posed, calib, v_name, panel_w=pw, panel_h=ph)
        panel_rgb = np.empty((ph, pw, 3), dtype=np.uint8)
        panel_rgb[:, :, 0] = bg_rgb[0]
        panel_rgb[:, :, 1] = bg_rgb[1]
        panel_rgb[:, :, 2] = bg_rgb[2]
        panel_a = np.zeros((ph, pw), dtype=np.uint8)

        if has_tris:
            pts_2d = np.rint(uv_px).astype(np.int32)
            tri_pts = pts_2d[tris]  # (F, 3, 2)
            tri_d = depth_rel[tris].mean(axis=1)
            # Back-to-front Painter's sort (largest depth = farthest first, smallest depth = closest last)
            order = np.argsort(tri_d, kind="mergesort")[::-1]
            for f_idx in order:
                poly = tri_pts[f_idx]
                c_r, c_g, c_b = int(tri_rgb[f_idx, 0]), int(tri_rgb[f_idx, 1]), int(tri_rgb[f_idx, 2])
                cv2.fillConvexPoly(panel_rgb, poly, (c_r, c_g, c_b))
                cv2.fillConvexPoly(panel_a, poly, 255)
        else:
            pts_2d = np.rint(uv_px).astype(np.int32)
            order = np.argsort(depth_rel)[::-1]
            for v_idx in order:
                px_i, py_i = int(pts_2d[v_idx, 0]), int(pts_2d[v_idx, 1])
                if 0 <= px_i < pw and 0 <= py_i < ph:
                    cv2.circle(panel_rgb, (px_i, py_i), 4, (195, 185, 170), -1)
                    cv2.circle(panel_a, (px_i, py_i), 4, 255, -1)

        canvas_rgba[t:b_edge, l:r, :3] = panel_rgb
        canvas_rgba[t:b_edge, l:r, 3] = panel_a

    return Image.fromarray(canvas_rgba, mode="RGBA")


_BLENDER_SINGLE_FRAME_RENDER_SCRIPT = r"""
import json, math, os, sys
from pathlib import Path
import bpy
from mathutils import Euler, Matrix, Vector
from PIL import Image

cfg = json.loads(Path(os.environ["FRAME_REVIEW_BLENDER_CFG"]).read_text(encoding="utf-8"))
bpy.ops.wm.open_mainfile(filepath=cfg["blend_path"])
scene = bpy.context.scene
meta = cfg["rig_meta"]
frame_bones = cfg["frame_bones"]
views = cfg["views"]
engine_req = cfg.get("engine", "WORKBENCH").upper()

rig = next(o for o in scene.objects if o.type == "ARMATURE")
meshes = [o for o in scene.objects if o.type == "MESH" and o.name != "Studio_Ground" and not o.hide_render]
for obj in list(scene.objects):
    if obj != rig and obj not in meshes:
        bpy.data.objects.remove(obj, do_unlink=True)

rig.animation_data_clear()
topo = sorted(rig.pose.bones, key=lambda pb: int(meta.get("bones", {}).get(pb.name, {}).get("depth", 0)))
for pb in topo:
    pb.rotation_mode = "QUATERNION"
    pb.matrix_basis = Matrix.Identity(4)
bpy.context.view_layer.update()

for pb in topo:
    b_info = meta.get("bones", {}).get(pb.name, {})
    if pb.name == "Core_Rigid" or b_info.get("fixed_local", False):
        pb.matrix_basis = Matrix.Identity(4)
    elif pb.name in frame_bones:
        pred = frame_bones[pb.name]
        world = Matrix(pred["rot_mat_world"]).to_4x4()
        world.translation = Vector(pred["head_3d"]) if not pb.parent else (rig.matrix_world @ pb.head)
        pb.matrix = rig.matrix_world.inverted() @ world
        if pb.parent:
            pb.location = (0.0, 0.0, 0.0)
        pb.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()

calib = meta.get("calibration", {})
center = Vector(calib.get("center", [0.0, 0.0, 0.5]))
ortho_scale = float(calib.get("ortho_scale", 1.5))
dist = ortho_scale * 2.5

if engine_req == "CYCLES":
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16
else:
    for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        try:
            scene.render.engine = eng
            break
        except TypeError:
            continue

scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"

for l_name, loc, energy in [
    ("KeyLight", (2.5, -3.5, 4.0), 900.0),
    ("FillLight", (-3.0, 2.5, 3.0), 550.0),
    ("RimLight", (0.0, 3.5, 4.5), 700.0),
]:
    ldata = bpy.data.lights.new(name=l_name, type="AREA")
    ldata.energy = energy
    ldata.size = 3.0
    lobj = bpy.data.objects.new(name=l_name, object_data=ldata)
    scene.collection.objects.link(lobj)
    lobj.location = loc
    lobj.rotation_euler = (center - lobj.location).to_track_quat("-Z", "Y").to_euler()

def make_cam(name, loc, rot):
    cdata = bpy.data.cameras.new(name=name)
    cdata.type = "ORTHO"
    cdata.sensor_fit = "VERTICAL"
    cdata.ortho_scale = ortho_scale
    cdata.clip_start = 0.05
    cdata.clip_end = dist * 4.0
    cobj = bpy.data.objects.new(name=name, object_data=cdata)
    scene.collection.objects.link(cobj)
    cobj.location = loc
    cobj.rotation_euler = Euler(rot, "XYZ")
    return cobj

cams = {
    "front": make_cam("Cam_Front", (center.x, center.y - dist, center.z), (math.pi / 2.0, 0.0, 0.0)),
    "side": make_cam("Cam_Side", (center.x + dist, center.y, center.z), (math.pi / 2.0, 0.0, math.pi / 2.0)),
    "left": make_cam("Cam_Left", (center.x + dist, center.y, center.z), (math.pi / 2.0, 0.0, math.pi / 2.0)),
    "right": make_cam("Cam_Right", (center.x - dist, center.y, center.z), (math.pi / 2.0, 0.0, -math.pi / 2.0)),
}
bpy.context.view_layer.update()

W_tot, H_tot = cfg["canvas_size"]
comp = Image.new("RGBA", (W_tot, H_tot), (9, 10, 14, 0))
tmp_dir = Path(cfg["tmp_dir"])

for idx, v in enumerate(views):
    v_name = v["name"].lower()
    l, t, r, b_edge = v["box"]
    pw, ph = r - l, b_edge - t
    scene.render.resolution_x = pw
    scene.render.resolution_y = ph
    scene.render.resolution_percentage = 100
    scene.camera = cams.get(v_name, cams["front"])
    out_p = tmp_dir / f"view_{idx}.png"
    scene.render.filepath = str(out_p)
    bpy.ops.render.render(write_still=True)
    with Image.open(out_p) as vim:
        comp.paste(vim.convert("RGBA"), (l, t))

comp.save(cfg["out_image"])
"""


def render_pose_blender_canvas(
    blend_path: Path,
    rig_meta: dict[str, Any],
    frame_bones: dict[str, dict[str, Any]],
    views: list[dict[str, Any]],
    canvas_size: tuple[int, int],
    engine: str = "WORKBENCH",
) -> Image.Image:
    """Renders the single frame in headless Blender and returns the composite RGBA image."""
    blender_bin = resolve_blender_binary()
    with tempfile.TemporaryDirectory(prefix="frame_review_bpy_") as tmp_d:
        tmp_root = Path(tmp_d)
        cfg_file = tmp_root / "cfg.json"
        script_file = tmp_root / "render_one.py"
        out_img = tmp_root / "rendered_canvas.png"
        cfg_file.write_text(
            json.dumps({
                "blend_path": str(blend_path.resolve()),
                "rig_meta": rig_meta,
                "frame_bones": frame_bones,
                "views": views,
                "canvas_size": [int(canvas_size[0]), int(canvas_size[1])],
                "engine": engine,
                "tmp_dir": str(tmp_root),
                "out_image": str(out_img),
            }),
            encoding="utf-8",
        )
        script_file.write_text(_BLENDER_SINGLE_FRAME_RENDER_SCRIPT, encoding="utf-8")
        env = os.environ.copy()
        env["FRAME_REVIEW_BLENDER_CFG"] = str(cfg_file)
        proc = subprocess.run(
            [blender_bin, "--background", "--factory-startup", "--python-exit-code", "1", "--python", str(script_file)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not out_img.is_file():
            raise RuntimeError(f"Blender single-frame render failed:\n{proc.stdout}")
        with Image.open(out_img) as im:
            return im.copy()


# ==============================================================================
# 4. Bone Attribution & Actionable Rotation Suggestions
# ==============================================================================

def attribute_clusters_to_bones(
    error_clusters: list[dict[str, Any]],
    bones_2d_view: dict[str, dict[str, Any]] | None,
    frame_bones: dict[str, dict[str, Any]] | None,
    rig_meta: dict[str, Any] | None,
    view_name: str,
) -> list[dict[str, Any]]:
    """Matches each contour error cluster to the nearest active deforming bone in `bones_2d_view`
    and computes actionable 2D view-plane and 3D local-Euler rotation suggestions.
    """
    if not error_clusters or not bones_2d_view:
        return []

    bones_meta = (rig_meta or {}).get("bones", {})
    w_axis = view_rotation_world_axis(view_name)

    # Filter active deforming bones with non-degenerate 2D projection
    active_bones: list[tuple[str, np.ndarray, np.ndarray, float, str]] = []
    for b_name, b_2d in bones_2d_view.items():
        if b_2d.get("fixed_local") or b_2d.get("kind") in ("rigid", "root"):
            continue
        h_uv = np.asarray(b_2d["head_uv"], dtype=np.float64)
        t_uv = np.asarray(b_2d["tail_uv"], dtype=np.float64)
        seg_len = float(np.linalg.norm(t_uv - h_uv))
        if seg_len < 3.5:
            continue
        b_kind = str(b_2d.get("kind", "limb")).lower()
        active_bones.append((b_name, h_uv, t_uv, seg_len, b_kind))

    if not active_bones:
        return []

    has_non_digit = any(bk != "digit" for _, _, _, _, bk in active_bones)
    active_map = {b_name: (h_uv, t_uv, seg_len, b_kind) for b_name, h_uv, t_uv, seg_len, b_kind in active_bones}
    agg_by_bone: dict[str, dict[str, Any]] = {}

    for cluster in error_clusters:
        anchor = np.asarray(cluster["anchor_contour_px"], dtype=np.float64)
        target = np.asarray(cluster["target_contour_px"], dtype=np.float64)
        centroid = np.asarray(cluster.get("centroid_px", anchor), dtype=np.float64)
        query_pt = 0.5 * anchor + 0.5 * centroid
        corr_vec = np.asarray(cluster["correction_vector_px"], dtype=np.float64)
        corr_mag = float(np.linalg.norm(corr_vec))
        if corr_mag < 1.0:
            continue
        c_area = int(cluster.get("area_px", 0))
        cluster_span = math.sqrt(max(1.0, float(c_area)))

        best_bone = None
        best_score = 1e18
        best_h = None
        for b_name, h_uv, t_uv, seg_len, b_kind in active_bones:
            if has_non_digit and b_kind == "digit" and c_area > 400:
                continue
            ab = t_uv - h_uv
            s = float(np.clip(np.dot(query_pt - h_uv, ab) / max(1e-8, seg_len * seg_len), 0.0, 1.0))
            proj = h_uv + s * ab
            dist = float(np.linalg.norm(query_pt - proj))
            # Slightly penalize projections right at the pivot (s < 0.15) where lever arm is tiny
            lever_penalty = 12.0 if s < 0.15 else 0.0
            score = dist + lever_penalty
            if score < best_score:
                best_score = score
                best_bone = b_name
                best_h = h_uv

        if best_bone is None or best_h is None:
            continue

        # If the error cluster spans more than this distal bone segment, promote up the same-kind parent chain
        # so the rotation pivot swings the entire displaced limb/tentacle.
        for _ in range(2):
            cur_h, cur_t, cur_len, cur_kind = active_map[best_bone]
            parent_name = bones_2d_view.get(best_bone, {}).get("parent")
            if (
                parent_name
                and parent_name in active_map
                and active_map[parent_name][3] == cur_kind
                and cluster_span > 0.75 * cur_len
            ):
                best_bone = parent_name
                best_h = active_map[parent_name][0]
            else:
                break

        r0 = anchor - best_h
        r1 = target - best_h
        n0 = float(np.linalg.norm(r0))
        n1 = float(np.linalg.norm(r1))
        if n0 > 5.0 and n1 > 5.0:
            cross_2d = float(r0[0] * r1[1] - r0[1] * r1[0])
            dot_2d = float(r0[0] * r1[0] + r0[1] * r1[1])
            raw_deg = math.degrees(math.atan2(cross_2d, dot_2d))
            dtheta_deg = round(float(np.clip(0.75 * raw_deg, -22.0, 22.0)), 2)
        else:
            dtheta_deg = 0.0

        local_euler_deg = [0.0, 0.0, 0.0]
        if frame_bones and best_bone in frame_bones and "rot_mat_world" in frame_bones[best_bone]:
            R_w = np.asarray(frame_bones[best_bone]["rot_mat_world"], dtype=np.float64)
            w_loc = R_w.T @ w_axis
            local_euler_deg = [round(float(dtheta_deg * w_loc[i]), 2) for i in range(3)]

        cluster["nearest_bone"] = best_bone
        cluster["bone_distance_px"] = round(float(best_score), 2)
        cluster["suggested_view_rot_deg"] = dtheta_deg
        cluster["suggested_tweak_view_arg"] = f"{view_name}:{best_bone}:{dtheta_deg:+.2f}"
        cluster["suggested_local_euler_deg"] = local_euler_deg
        cluster["suggested_tweak_arg"] = (
            f"{best_bone}:{local_euler_deg[0]:+.2f},{local_euler_deg[1]:+.2f},{local_euler_deg[2]:+.2f}"
        )

        weight = float(cluster["area_px"]) * float(cluster["max_distance_px"])
        entry = agg_by_bone.setdefault(
            best_bone,
            {
                "bone": best_bone,
                "view": view_name,
                "kind": str(bones_meta.get(best_bone, {}).get("kind", "limb")),
                "weight_sum": 0.0,
                "weighted_deg": 0.0,
                "clusters_count": 0,
                "total_error_area_px": 0,
            },
        )
        entry["weight_sum"] += weight
        entry["weighted_deg"] += weight * dtheta_deg
        entry["clusters_count"] += 1
        entry["total_error_area_px"] += int(cluster["area_px"])

    suggestions: list[dict[str, Any]] = []
    for b_name, entry in sorted(agg_by_bone.items(), key=lambda kv: kv[1]["weight_sum"], reverse=True):
        w_sum = max(1e-8, float(entry["weight_sum"]))
        avg_deg = round(float(np.clip(entry["weighted_deg"] / w_sum, -22.0, 22.0)), 2)
        if abs(avg_deg) < 0.25:
            continue
        local_euler_deg = [0.0, 0.0, 0.0]
        if frame_bones and b_name in frame_bones and "rot_mat_world" in frame_bones[b_name]:
            R_w = np.asarray(frame_bones[b_name]["rot_mat_world"], dtype=np.float64)
            w_loc = R_w.T @ w_axis
            local_euler_deg = [round(float(avg_deg * w_loc[i]), 2) for i in range(3)]
        suggestions.append({
            "bone": b_name,
            "view": view_name,
            "kind": entry["kind"],
            "clusters_count": entry["clusters_count"],
            "total_error_area_px": entry["total_error_area_px"],
            "suggested_view_rot_deg": avg_deg,
            "suggested_tweak_view_arg": f"{view_name}:{b_name}:{avg_deg:+.2f}",
            "suggested_local_euler_deg": local_euler_deg,
            "suggested_tweak_arg": f"{b_name}:{local_euler_deg[0]:+.2f},{local_euler_deg[1]:+.2f},{local_euler_deg[2]:+.2f}",
        })

    return suggestions


# ==============================================================================
# 5. Multi-View 3-in-1 Diagnostic Card Composer (`panel_compare.png`)
# ==============================================================================

def draw_skeleton_overlay_on_image(
    pil_img: Image.Image,
    bones_2d_view: dict[str, dict[str, Any]] | None,
    highlighted_bones: set[str] | None = None,
) -> Image.Image:
    """Draws semi-transparent octahedral bone shafts and joint pivots onto `pil_img`
    (matching the `OUR MODEL · RENDERED` style with visible 3D bone shafts).
    """
    base = pil_img.convert("RGBA")
    if not bones_2d_view:
        return base.convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    hi_set = highlighted_bones or set()

    # Sort back-to-front by relative depth (largest depth = farthest first)
    sorted_bones = sorted(bones_2d_view.items(), key=lambda kv: float(kv[1].get("depth_rel", 0.0)), reverse=True)
    for b_name, b_info in sorted_bones:
        if b_name == "Core_Rigid":
            continue
        h = b_info["head_uv"]
        t = b_info["tail_uv"]
        dx, dy = float(t[0] - h[0]), float(t[1] - h[1])
        seg_len = math.hypot(dx, dy)
        if seg_len < 2.0:
            continue
        nx, ny = -dy / seg_len, dx / seg_len
        half_w = min(7.0, max(2.2, 0.085 * seg_len))
        mx, my = h[0] + 0.22 * dx, h[1] + 0.22 * dy
        poly = [
            (h[0], h[1]),
            (mx + nx * half_w, my + ny * half_w),
            (t[0], t[1]),
            (mx - nx * half_w, my - ny * half_w),
        ]
        if b_name in hi_set:
            fill_c = (255, 200, 70, 195)
            out_c = (255, 245, 190, 235)
        else:
            fill_c = (235, 242, 250, 155)
            out_c = (40, 55, 75, 210)
        draw.polygon(poly, fill=fill_c, outline=out_c)
        r_j = max(2.5, min(4.5, half_w * 0.7))
        draw.ellipse([h[0] - r_j, h[1] - r_j, h[0] + r_j, h[1] + r_j], fill=(255, 255, 255, 220), outline=(30, 40, 55, 220))

    return Image.alpha_composite(base, overlay).convert("RGB")


def draw_error_clusters_and_arrows(
    blend_rgb: np.ndarray,
    error_clusters: list[dict[str, Any]],
) -> np.ndarray:
    """Draws bounding boxes and directional pull arrows (`anchor_contour_px -> target_contour_px`)
    for the top contour error clusters onto the `COMPARE` overlay panel.
    """
    canvas = blend_rgb.copy()
    H, W = canvas.shape[:2]
    scale = max(1.0, min(W, H) / 640.0)
    thickness = max(2, int(round(2.0 * scale)))

    placed_labels: list[tuple[int, int, int, int]] = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.42 * scale
    font_thick = max(1, int(round(1.3 * scale)))

    for idx, cl in enumerate(error_clusters[:5]):
        l, t, r, b = cl["box"]
        is_prot = cl["kind"] == "protruding"
        box_color = (255, 110, 225) if is_prot else (60, 235, 255)
        cv2.rectangle(canvas, (int(l), int(t)), (int(r), int(b)), box_color, max(1, thickness - 1))

        p0 = (int(round(cl["anchor_contour_px"][0])), int(round(cl["anchor_contour_px"][1])))
        p1 = (int(round(cl["target_contour_px"][0])), int(round(cl["target_contour_px"][1])))
        if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) >= 2.0:
            cv2.arrowedLine(canvas, p0, p1, (255, 215, 40), thickness + 1, tipLength=0.32)

        label = f"#{idx + 1} {cl.get('nearest_bone') or cl['kind']} ({cl['correction_vector_px'][0]:+.0f},{cl['correction_vector_px'][1]:+.0f}px)"
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)
        lx = max(6, min(W - tw - 10, int(l)))
        ly = max(th + 10, int(t) - 6)
        for _ in range(8):
            bx0, by0, bx1, by1 = lx - 3, ly - th - 4, lx + tw + 3, ly + 4
            if any(not (bx1 < ox0 or bx0 > ox1 or by1 < oy0 or by0 > oy1) for ox0, oy0, ox1, oy1 in placed_labels):
                ly = min(H - 8, ly + th + int(round(10 * scale)))
            else:
                break
        bx0, by0, bx1, by1 = lx - 3, ly - th - 4, lx + tw + 3, ly + 4
        placed_labels.append((bx0, by0, bx1, by1))
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (18, 22, 30), -1)
        cv2.putText(
            canvas,
            label,
            (lx, ly),
            font,
            font_scale,
            (255, 235, 120),
            font_thick,
            cv2.LINE_AA,
        )
    return canvas


def _draw_pill_badge(
    img: Image.Image,
    xy: tuple[int, int],
    text: str,
    bg_color: tuple[int, int, int] = (24, 28, 36),
    border_color: tuple[int, int, int] = (75, 90, 110),
    text_color: tuple[int, int, int] = (235, 240, 248),
    scale: float = 1.0,
) -> None:
    arr = np.asarray(img).copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52 * scale
    thick = max(1, int(round(1.5 * scale)))
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thick)
    pad_x = int(round(12 * scale))
    pad_y = int(round(8 * scale))
    x0, y0 = xy
    x1, y1 = x0 + tw + 2 * pad_x, y0 + th + 2 * pad_y
    cv2.rectangle(arr, (x0, y0), (x1, y1), bg_color, -1)
    cv2.rectangle(arr, (x0, y0), (x1, y1), border_color, max(1, int(round(1.5 * scale))))
    cv2.putText(arr, text, (x0 + pad_x, y0 + pad_y + th), font, font_scale, text_color, thick, cv2.LINE_AA)
    img.paste(Image.fromarray(arr))


def build_view_diagnostic_card(
    frame_idx: int,
    view_name: str,
    ref_crop: Image.Image,
    cand_crop: Image.Image,
    compare_rgb: np.ndarray,
    view_metrics: dict[str, Any],
    bones_2d_view: dict[str, dict[str, Any]] | None = None,
    max_panel_w: int = 560,
) -> Image.Image:
    """Builds the 3-in-1 diagnostic card for a single view:
      [ VIDEO · FRAME N (VIEW) ] | [ OUR MODEL · RENDERED (with bones) ] | [ COMPARE + OUTLINE ✓/✗ ]
    """
    orig_w, orig_h = ref_crop.size
    scale_vis = min(1.0, float(max_panel_w) / max(1.0, float(orig_w)))
    pw = max(240, int(round(orig_w * scale_vis)))
    ph = max(200, int(round(orig_h * scale_vis)))
    badge_scale = max(0.75, min(1.25, pw / 480.0))

    clusters = view_metrics.get("error_clusters") or []
    hi_bones = {c["nearest_bone"] for c in clusters if c.get("nearest_bone")}
    cand_with_rig = draw_skeleton_overlay_on_image(cand_crop, bones_2d_view, highlighted_bones=hi_bones)
    comp_annotated = draw_error_clusters_and_arrows(compare_rgb, clusters)

    p_ref = ref_crop.convert("RGB").resize((pw, ph), Image.Resampling.BILINEAR)
    p_cand = cand_with_rig.resize((pw, ph), Image.Resampling.BILINEAR)
    p_comp = Image.fromarray(comp_annotated).resize((pw, ph), Image.Resampling.BILINEAR)

    _draw_pill_badge(p_ref, (14, 14), f"VIDEO - FRAME {frame_idx:03d} ({view_name.upper()})", scale=badge_scale)
    _draw_pill_badge(p_cand, (14, 14), f"OUR MODEL - RENDERED ({view_name.upper()})", scale=badge_scale)

    passed = bool(view_metrics.get("outline_passed"))
    iou_v = view_metrics.get("iou")
    bmean_v = view_metrics.get("boundary_mean_px")
    bp95_v = view_metrics.get("boundary_p95_px")
    iou_str = f"{iou_v:.3f}" if isinstance(iou_v, (int, float)) else "N/A"
    bm_str = f"{bmean_v:.2f}px" if isinstance(bmean_v, (int, float)) else "N/A"
    bp_str = f"{bp95_v:.1f}px" if isinstance(bp95_v, (int, float)) else "N/A"

    _draw_pill_badge(
        p_comp,
        (14, 14),
        f"COMPARE ({view_name.upper()}) | IoU {iou_str} | Mean {bm_str} | P95 {bp_str}",
        scale=badge_scale,
    )
    if passed:
        _draw_pill_badge(
            p_comp,
            (14, ph - int(round(52 * badge_scale))),
            "OUTLINE PASS [OK]",
            bg_color=(16, 58, 34),
            border_color=(65, 225, 130),
            text_color=(110, 255, 165),
            scale=badge_scale * 1.08,
        )
    else:
        _draw_pill_badge(
            p_comp,
            (14, ph - int(round(52 * badge_scale))),
            f"OUTLINE DIFF [TWEAK NEEDED]",
            bg_color=(68, 22, 28),
            border_color=(255, 90, 110),
            text_color=(255, 145, 160),
            scale=badge_scale * 1.08,
        )

    pad = 16
    card_w = pw * 3 + pad * 4
    card_h = ph + pad * 2
    card = Image.new("RGB", (card_w, card_h), (14, 18, 24))
    card.paste(p_ref, (pad, pad))
    card.paste(p_cand, (pad * 2 + pw, pad))
    card.paste(p_comp, (pad * 3 + pw * 2, pad))
    return card


def compose_multiview_diagnostic_board(
    cards: list[Image.Image],
    out_path: Path,
) -> Path:
    """Stacks per-view 3-in-1 cards vertically into a single `panel_compare.png` board."""
    if not cards:
        raise ValueError("No view cards to compose")
    max_w = max(c.width for c in cards)
    total_h = sum(c.height for c in cards)
    board = Image.new("RGB", (max_w, total_h), (14, 18, 24))
    y_off = 0
    for c in cards:
        board.paste(c, (0, y_off))
        y_off += c.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(out_path)
    return out_path
