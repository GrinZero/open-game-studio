"""CPU/GPU software triangle rasterizer, Lambertian textured mesh renderer, and fast 4-column review generator.

Absorbs `canitcode/video2rig/raster.py` and `canitcode/video2rig/review.py` so silhouette IoU,
visibility checks, dense field rasterization, and 4-column review sheets/videos can run in seconds
without launching a headless Blender subprocess.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAS_TORCH = False

BUCKETS = ((16, 4), (64, 16), (256, 64), (10**9, 256))
MAX_PIXELS = 40_000_000

# Three-point directional lighting in camera view space (x right, y down, z toward viewer):
# key from upper-front-left, fill from upper-front-right, rim from below-behind.
LIGHTS = (
    (0.35, -0.45, 0.82, 0.55),
    (-0.55, -0.25, 0.80, 0.25),
    (0.0, 0.70, 0.71, 0.15),
)


def rasterize_triangles_torch(
    uv: np.ndarray,
    z: np.ndarray,
    tris: np.ndarray,
    size: tuple[int, int] | list[int],
    device: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bucketed PyTorch CPU/CUDA software triangle rasterizer (video2rig/raster.py).

    Parameters
    ----------
    uv : (V, 2) float array of pixel coordinates
    z : (V,) float array of camera depths (smaller = closer)
    tris : (F, 3) int array of triangle vertex indices
    size : (width, height)

    Returns
    -------
    tri_id : (H, W) int64 array (-1 where background)
    bary : (H, W, 3) float32 array of barycentric weights
    zbuf : (H, W) float32 array of depth (inf where background)
    """
    from reader_v3 import rasterize_triangles_numpy

    if not HAS_TORCH:
        return rasterize_triangles_numpy(uv, z, tris, size)

    W, H = int(size[0]), int(size[1])
    tris_arr = np.asarray(tris, dtype=np.int64)
    if len(tris_arr) == 0 or W <= 0 or H <= 0:
        return (
            np.full((H, W), -1, dtype=np.int64),
            np.zeros((H, W, 3), dtype=np.float32),
            np.full((H, W), np.inf, dtype=np.float32),
        )

    dev = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    uv_t = torch.as_tensor(np.asarray(uv, dtype=np.float32), dtype=torch.float32, device=dev)
    z_t = torch.as_tensor(np.asarray(z, dtype=np.float32), dtype=torch.float32, device=dev)
    tris_t = torch.as_tensor(tris_arr, dtype=torch.long, device=dev)

    p = uv_t[tris_t]  # (F, 3, 2)
    z3 = z_t[tris_t]  # (F, 3)
    A, B, C = p[:, 0], p[:, 1], p[:, 2]
    area = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (
        C[:, 0] - A[:, 0]
    )

    x0 = torch.clamp(torch.floor(p[:, :, 0].amin(dim=1)).long(), min=0)
    x1 = torch.clamp(torch.ceil(p[:, :, 0].amax(dim=1)).long(), max=W - 1)
    y0 = torch.clamp(torch.floor(p[:, :, 1].amin(dim=1)).long(), min=0)
    y1 = torch.clamp(torch.ceil(p[:, :, 1].amax(dim=1)).long(), max=H - 1)
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    span = torch.maximum(bw, bh)
    ok = (area.abs() >= 1e-8) & (bw > 0) & (bh > 0)

    pix_all = []
    z_all = []
    t_all = []
    w_all = []
    lo = 0
    for hi, step in BUCKETS:
        sel = torch.nonzero(ok & (span > lo) & (span <= hi), as_tuple=False).squeeze(1)
        lo = hi
        if sel.numel() == 0:
            continue
        mw = int(bw[sel].max().item())
        mh = int(bh[sel].max().item())
        batch = max(1, min(int(sel.numel()), MAX_PIXELS // max(1, mw * mh)))
        for s in range(0, int(sel.numel()), batch):
            sub = sel[s : s + batch]
            sw = int(bw[sub].max().item())
            sh = int(bh[sub].max().item())
            gy, gx = torch.meshgrid(
                torch.arange(sh, device=dev),
                torch.arange(sw, device=dev),
                indexing="ij",
            )
            px = x0[sub, None] + gx.reshape(1, -1)
            py = y0[sub, None] + gy.reshape(1, -1)
            inb = (px <= x1[sub, None]) & (py <= y1[sub, None])
            cx = px.float() + 0.5
            cy = py.float() + 0.5
            a, b, c = A[sub], B[sub], C[sub]
            ar = area[sub, None]
            w0 = (
                (b[:, 0, None] - cx) * (c[:, 1, None] - cy)
                - (b[:, 1, None] - cy) * (c[:, 0, None] - cx)
            ) / ar
            w1 = (
                (c[:, 0, None] - cx) * (a[:, 1, None] - cy)
                - (c[:, 1, None] - cy) * (a[:, 0, None] - cx)
            ) / ar
            w2 = 1.0 - w0 - w1
            inside = inb & (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            r, q = torch.nonzero(inside, as_tuple=True)
            if r.numel() == 0:
                continue
            t = sub[r]
            ww = torch.stack([w0[r, q], w1[r, q], w2[r, q]], dim=1)
            pix_all.append(py[r, q] * W + px[r, q])
            z_all.append((ww * z3[t]).sum(dim=1))
            t_all.append(t)
            w_all.append(ww)

    tri = torch.full((W * H,), -1, dtype=torch.long, device=dev)
    bary = torch.zeros((W * H, 3), dtype=torch.float32, device=dev)
    zbuf = torch.full((W * H,), float("inf"), dtype=torch.float32, device=dev)
    if pix_all:
        P = torch.cat(pix_all)
        Z = torch.cat(z_all)
        T = torch.cat(t_all)
        WW = torch.cat(w_all)
        # Sort by depth first (stable), then pixel index so first occurrence per pixel is closest
        oz = torch.argsort(Z, stable=True)
        op = torch.argsort(P[oz], stable=True)
        order = oz[op]
        P_sorted = P[order]
        first = torch.ones_like(P_sorted, dtype=torch.bool)
        first[1:] = P_sorted[1:] != P_sorted[:-1]
        idx = order[first]
        p_win = P_sorted[first]
        tri[p_win] = T[idx]
        bary[p_win] = WW[idx]
        zbuf[p_win] = Z[idx]
    return (
        tri.reshape(H, W).cpu().numpy(),
        bary.reshape(H, W, 3).cpu().numpy(),
        zbuf.reshape(H, W).cpu().numpy(),
    )


def vertex_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex unit normals (video2rig/raster.py)."""
    v = np.asarray(verts, dtype=np.float64)
    t = np.asarray(tris, dtype=np.int64)
    vn = np.zeros_like(v, dtype=np.float64)
    if len(t) > 0 and len(v) > 0:
        v0, v1, v2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)
        np.add.at(vn, t[:, 0], fn)
        np.add.at(vn, t[:, 1], fn)
        np.add.at(vn, t[:, 2], fn)
    nlen = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.where(nlen > 1e-12, vn / np.maximum(nlen, 1e-12), np.array([0.0, 0.0, 1.0]))
    return vn


def _sample_uv_texture_bilinear(tex_img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear texture lookup for UV coordinates in [0, 1] with wrap/repeat semantics."""
    th, tw = tex_img.shape[:2]
    u = np.mod(uv[:, 0], 1.0) * (tw - 1)
    # Standard glTF/Blender UV has v=0 at bottom or top; match image row indexing (1 - v)
    v = np.mod(1.0 - uv[:, 1], 1.0) * (th - 1)
    x0 = np.clip(np.floor(u).astype(np.int64), 0, tw - 1)
    y0 = np.clip(np.floor(v).astype(np.int64), 0, th - 1)
    x1 = np.clip(x0 + 1, 0, tw - 1)
    y1 = np.clip(y0 + 1, 0, th - 1)
    fx = (u - x0)[:, None]
    fy = (v - y0)[:, None]
    c00 = tex_img[y0, x0].astype(np.float64)
    c10 = tex_img[y0, x1].astype(np.float64)
    c01 = tex_img[y1, x0].astype(np.float64)
    c11 = tex_img[y1, x1].astype(np.float64)
    return (
        c00 * (1.0 - fx) * (1.0 - fy)
        + c10 * fx * (1.0 - fy)
        + c01 * (1.0 - fx) * fy
        + c11 * fx * fy
    )[:, :3]


def render_mesh_software(
    verts: np.ndarray,
    tris: np.ndarray,
    cam: Any,
    colors: np.ndarray | None = None,
    uv_tex: np.ndarray | None = None,
    tex_img: np.ndarray | None = None,
    bg: int | tuple[int, int, int] = 238,
    size: tuple[int, int] | None = None,
    width: int | None = None,
    height: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render a posed triangle mesh with Lambertian 3-point shading and optional vertex/UV texture colors.

    Matches `video2rig/raster.py:render` and adds optional UV-map texture sampling.

    Returns
    -------
    img : (H, W, 3) uint8 RGB image
    fg : (H, W) bool foreground silhouette mask
    tri_id : (H, W) int64 triangle index buffer
    bary : (H, W, 3) float32 barycentric coordinates
    zbuf : (H, W) float32 depth buffer
    """
    verts_arr = np.asarray(verts, dtype=np.float64)
    tris_arr = np.asarray(tris, dtype=np.int64)
    if width is not None and height is not None:
        size = (int(width), int(height))
    W = int(size[0] if size is not None else cam.width)
    H = int(size[1] if size is not None else cam.height)
    sx = W / max(float(cam.width), 1.0)
    sy = H / max(float(cam.height), 1.0)

    uv = cam.project(verts_arr) * np.array([sx, sy], dtype=np.float64)
    z = cam.depth(verts_arr)
    tri_id, bary, zbuf = rasterize_triangles_torch(uv, z, tris_arr, (W, H))
    fg = tri_id >= 0

    if isinstance(bg, (tuple, list, np.ndarray)):
        img = np.broadcast_to(np.asarray(bg, dtype=np.uint8).reshape(1, 1, 3), (H, W, 3)).copy()
    else:
        img = np.full((H, W, 3), int(bg), dtype=np.uint8)
    if not np.any(fg):
        return img, fg, tri_id, bary, zbuf

    vn = vertex_normals(verts_arr, tris_arr) @ np.asarray(cam.R, dtype=np.float64).T
    vn[:, 2] = -vn[:, 2]  # flip from camera-forward to toward-viewer

    hit_t = tris_arr[tri_id[fg]]  # (N_fg, 3)
    hit_w = bary[fg].astype(np.float64)  # (N_fg, 3)

    n = (vn[hit_t] * hit_w[:, :, None]).sum(axis=1)
    nlen = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.where(nlen > 1e-8, n / np.maximum(nlen, 1e-8), 0.0)

    shade = np.full(len(n), 0.30, dtype=np.float64)
    for lx, ly, lz, w in LIGHTS:
        ldir = np.array([lx, ly, lz], dtype=np.float64)
        ldir /= np.linalg.norm(ldir)
        shade += w * np.maximum(0.0, n @ ldir)
    shade = np.clip(shade, 0.0, 1.15)

    if tex_img is not None and uv_tex is not None:
        uv_arr = np.asarray(uv_tex, dtype=np.float64)
        if uv_arr.shape == (len(tris_arr), 3, 2):
            pix_uv = (uv_arr[tri_id[fg]] * hit_w[:, :, None]).sum(axis=1)
        else:
            pix_uv = (uv_arr[hit_t] * hit_w[:, :, None]).sum(axis=1)
        base = _sample_uv_texture_bilinear(np.asarray(tex_img), pix_uv)
    elif colors is not None:
        col_arr = np.asarray(colors, dtype=np.float64)
        if col_arr.max() <= 1.0:
            col_arr = col_arr * 255.0
        base = (col_arr[hit_t] * hit_w[:, :, None]).sum(axis=1)
    else:
        base = np.array([[185.0, 190.0, 200.0]], dtype=np.float64)

    img[fg] = np.clip(base * shade[:, None], 0.0, 255.0).astype(np.uint8)
    return img, fg, tri_id, bary, zbuf


def orbit_camera_three_quarter(
    rig: Any,
    base_camera: Any,
    yaw_deg: float = 38.0,
    tilt_deg: float = 18.0,
    size: tuple[int, int] | None = None,
) -> Any:
    """Construct a 3/4 orbit camera around the creature rest center (video2rig/review.py:orbit_camera)."""
    from studio_camera import Camera

    W = int(size[0] if size is not None else base_camera.width)
    H = int(size[1] if size is not None else base_camera.height)
    center = np.mean(np.array(list(rig.h.values()), dtype=np.float64), axis=0)
    yaw = math.radians(yaw_deg)
    tilt = math.radians(tilt_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    ct, st = math.cos(tilt), math.sin(tilt)
    forward = np.array([-sy * ct, cy * ct, -st], dtype=np.float64)
    right = np.array([cy, sy, 0.0], dtype=np.float64)
    up = np.cross(right, forward)
    R = np.stack([right, -up, forward], axis=0)
    dist = float(np.linalg.norm(np.asarray(base_camera.t, dtype=np.float64)))
    if dist < 1e-3:
        dist = 2.5 * max(float(base_camera.scale), float(rig.height), 1.0)
    t_vec = -R @ center + np.array([0.0, 0.0, dist], dtype=np.float64)
    scale = float(base_camera.scale)
    focal = float(base_camera.focal_px) * (W / max(float(base_camera.width), 1.0))
    return Camera(
        R=R,
        t=t_vec,
        width=W,
        height=H,
        type=base_camera.type,
        scale=scale,
        focal_px=focal,
        shift_px=(0.0, 0.0),
    )


def _mask_outline(mask: np.ndarray) -> np.ndarray:
    """1-pixel morphological interior boundary of a boolean silhouette mask."""
    m = np.asarray(mask, dtype=bool)
    if m.shape[0] < 3 or m.shape[1] < 3 or not m.any():
        return np.zeros_like(m, dtype=bool)
    interior = np.zeros_like(m, dtype=bool)
    interior[1:-1, 1:-1] = (
        m[1:-1, 1:-1]
        & m[:-2, 1:-1]
        & m[2:, 1:-1]
        & m[1:-1, :-2]
        & m[1:-1, 2:]
    )
    return m & ~interior


def _estimate_video_foreground(frame_rgb: np.ndarray, thresh: float = 18.0) -> np.ndarray:
    """Estimate foreground silhouette from a studio video panel via border median background color."""
    im = np.asarray(frame_rgb, dtype=np.float32)
    border = np.concatenate(
        [
            im[:4].reshape(-1, 3),
            im[-4:].reshape(-1, 3),
            im[:, :4].reshape(-1, 3),
            im[:, -4:].reshape(-1, 3),
        ],
        axis=0,
    )
    bg = np.median(border, axis=0)
    return np.linalg.norm(im - bg[None, None, :], axis=-1) > thresh


def render_fast_review_frames(
    rig: Any,
    cameras: list[Any],
    params_or_frames: Any,
    skin: dict[str, Any],
    video_panels: list[list[np.ndarray]] | None = None,
    *,
    cell_size: tuple[int, int] | None = None,
    bg_thresh: float = 18.0,
    yaw_deg: float = 38.0,
    tilt_deg: float = 18.0,
    out_mp4: str | Path | None = None,
    fps: float = 24.0,
) -> dict[str, Any]:
    """Render fast 4-column comparison frames `[Video | Fit | 50/50 Overlay + Contours | 3/4 Orbit]`
    using the CPU/GPU software rasterizer (video2rig/review.py).

    Works with either a parameter matrix `(T, rig.size)` or a list of reconstructed frame dicts.
    """
    from pose_features import lbs_deform_vertices_numpy

    rest_verts = np.asarray(skin["rest_verts"], dtype=np.float64)
    skin_indices = np.asarray(skin["skin_indices"], dtype=np.int64)
    skin_weights = np.asarray(skin["skin_weights"], dtype=np.float64)
    rest_heads = np.asarray(skin["rest_heads"], dtype=np.float64)
    rest_R = np.asarray(skin["rest_R"], dtype=np.float64)
    tris = np.asarray(skin["triangles"], dtype=np.int64)
    colors = skin.get("vertex_colors", skin.get("colors"))
    uv_tex = skin.get("uv_coords", skin.get("uv"))
    tex_img = skin.get("texture_image")

    V = len(cameras)
    W = int(cell_size[0] if cell_size is not None else cameras[0].width)
    H = int(cell_size[1] if cell_size is not None else cameras[0].height)
    orbit_cam = orbit_camera_three_quarter(
        rig, cameras[0], yaw_deg=yaw_deg, tilt_deg=tilt_deg, size=(W, H)
    )

    if isinstance(params_or_frames, np.ndarray):
        T = len(params_or_frames)
        is_params = True
    else:
        T = len(params_or_frames)
        is_params = False

    composites: list[np.ndarray] = []
    iou_per_frame: list[list[float]] = []

    for t in range(T):
        if is_params:
            _, heads_map, rot_map = rig.fk(params_or_frames[t])
            posed_heads = np.array([heads_map[n] for n in rig.names], dtype=np.float64)
            posed_R = np.array([rot_map[n] for n in rig.names], dtype=np.float64)
        else:
            b_dict = params_or_frames[t]["bones"]
            posed_heads = np.array([b_dict[n]["head_3d"] for n in rig.names], dtype=np.float64)
            posed_R = np.array([b_dict[n]["rot_mat_world"] for n in rig.names], dtype=np.float64)

        posed_verts = lbs_deform_vertices_numpy(
            rest_verts,
            skin_indices,
            skin_weights,
            rest_heads,
            rest_R,
            posed_heads,
            posed_R,
        )
        orbit_img, _, _, _, _ = render_mesh_software(
            posed_verts,
            tris,
            orbit_cam,
            colors=colors,
            uv_tex=uv_tex,
            tex_img=tex_img,
            size=(W, H),
        )

        rows = []
        frame_ious = []
        for v, cam in enumerate(cameras):
            fit_img, fg_fit, _, _, _ = render_mesh_software(
                posed_verts,
                tris,
                cam,
                colors=colors,
                uv_tex=uv_tex,
                tex_img=tex_img,
                size=(W, H),
            )
            if video_panels is not None and v < len(video_panels) and t < len(video_panels[v]):
                vid_raw = np.asarray(video_panels[v][t], dtype=np.uint8)
                if (vid_raw.shape[1], vid_raw.shape[0]) != (W, H):
                    from PIL import Image

                    vid_img = np.asarray(
                        Image.fromarray(vid_raw).resize((W, H), Image.Resampling.BILINEAR),
                        dtype=np.uint8,
                    )
                else:
                    vid_img = vid_raw.copy()
                fg_vid = _estimate_video_foreground(vid_img, thresh=bg_thresh)
            else:
                vid_img = fit_img.copy()
                fg_vid = fg_fit.copy()

            union = int(np.logical_or(fg_fit, fg_vid).sum())
            inter = int(np.logical_and(fg_fit, fg_vid).sum())
            iou = float(inter / union) if union > 0 else 1.0
            frame_ious.append(iou)

            overlay = (
                0.5 * vid_img.astype(np.float32) + 0.5 * fit_img.astype(np.float32)
            ).astype(np.uint8)
            overlay[_mask_outline(fg_vid)] = (220, 60, 60)
            overlay[_mask_outline(fg_fit)] = (40, 210, 90)

            row = np.concatenate([vid_img, fit_img, overlay, orbit_img], axis=1)
            rows.append(row)

        iou_per_frame.append(frame_ious)
        composites.append(np.concatenate(rows, axis=0))

    encoded = False
    if out_mp4 is not None and composites and shutil.which("ffmpeg"):
        out_path = Path(out_mp4)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fh, fw = composites[0].shape[:2]
        # Ensure even dimensions for yuv420p
        pad_h = fh % 2
        pad_w = fw % 2
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{fw + pad_w}x{fh + pad_h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(out_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin is not None
        for comp in composites:
            if pad_h or pad_w:
                comp = np.pad(comp, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            proc.stdin.write(comp.tobytes())
        proc.stdin.close()
        encoded = proc.wait() == 0

    iou_arr = np.asarray(iou_per_frame, dtype=np.float64) if iou_per_frame else np.zeros((0, V))
    return {
        "frames": composites,
        "iou_per_frame": iou_per_frame,
        "mean_iou": float(iou_arr.mean()) if iou_arr.size else 0.0,
        "mean_iou_per_view": iou_arr.mean(axis=0).tolist() if iou_arr.size else [],
        "encoded_mp4": str(out_mp4) if encoded else None,
        "columns": ["video", "fit_render", "overlay_contours", "orbit_three_quarter"],
    }
