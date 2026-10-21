"""Shared v2.3 data contract. No torch/bpy imports: usable in both worker runtimes."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

FEATURE_ENV = "MONSTER_ENABLE_DEPTH_VISIBILITY"
RESOLUTION_2K_ENV = "MONSTER_ENABLE_2K"
DATASET_SCHEMA = "monster-pose-depth-visibility-v1"
FEATURE_VERSION = "v2.3-depth-visibility-1"
DEFAULT_SYNTH_SAMPLES = 5000
DEFAULT_STUDIO_RESOLUTION = (672, 768, 1344, 768)
STUDIO_2K_RESOLUTION = (1280, 1440, 2560, 1440)


def features_enabled() -> bool:
    value = os.environ.get(FEATURE_ENV, "1").strip().lower()
    if value not in {"0", "1", "true", "false", "on", "off"}:
        raise ValueError(f"Invalid {FEATURE_ENV}: {value!r}")
    return value in {"1", "true", "on"}


def resolution_2k_enabled() -> bool:
    value = os.environ.get(RESOLUTION_2K_ENV, "0").strip().lower()
    if value not in {"0", "1", "true", "false", "on", "off"}:
        raise ValueError(f"Invalid {RESOLUTION_2K_ENV}: {value!r}")
    return value in {"1", "true", "on"}


def studio_resolution(enable_2k: bool | None = None) -> tuple[int, int, int, int]:
    """Returns (panel_width, panel_height, composite_width, composite_height)."""
    use_2k = resolution_2k_enabled() if enable_2k is None else bool(enable_2k)
    return STUDIO_2K_RESOLUTION if use_2k else DEFAULT_STUDIO_RESOLUTION


def idle_split_filename(comp_w: int | None = None, comp_h: int | None = None) -> str:
    if comp_w is None or comp_h is None:
        _, _, comp_w, comp_h = studio_resolution()
    return f"idle_split_{int(comp_w)}x{int(comp_h)}.png"


def dinov2_panel_padding(panel_w: int, panel_h: int, patch_size: int = 14) -> tuple[int, int, int, int]:
    """Returns symmetric (pad_top, pad_bottom, pad_left, pad_right) so padded (H, W) are multiples of patch_size=14."""
    pad_w = (patch_size - (int(panel_w) % patch_size)) % patch_size
    pad_h = (patch_size - (int(panel_h) % patch_size)) % patch_size
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    return pad_top, pad_bottom, pad_left, pad_right


def dataset_signature(blend_path, meta_path, num_samples, seed=42):
    meta = json.loads(Path(meta_path).read_text())
    meta.pop("calibration", None)  # Stage 2 adds calibration; it is not a rig change.
    digest = hashlib.sha256()
    with open(blend_path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode())
    res_tag = ":2560x1440" if resolution_2k_enabled() else ""
    digest.update(f"{FEATURE_VERSION}:{features_enabled()}:{int(num_samples)}:{int(seed)}{res_tag}".encode())
    return digest.hexdigest()


def dataset_cache_matches(synth_dir, blend_path, meta_path, num_samples):
    try:
        path = Path(synth_dir)
        data = json.loads((path / "keypoints_gt.json").read_text())
        if data.get("signature") != dataset_signature(blend_path, meta_path, num_samples):
            return False
        if data.get("depth_visibility_enabled", False) != features_enabled():
            return False
        panel_w, panel_h, comp_w, comp_h = studio_resolution()
        calib = data.get("calibration", {})
        if calib.get("panel_width") != panel_w or calib.get("panel_height") != panel_h:
            return False
        if calib.get("composite_width") != comp_w or calib.get("composite_height") != comp_h:
            return False
        samples = data["samples"]
        return len(samples) == int(num_samples) and all(
            (path / "images" / sample["image"]).is_file() for sample in samples
        ) and (path / idle_split_filename(comp_w, comp_h)).is_file()
    except (OSError, ValueError, KeyError, TypeError):
        return False


def validate_enhanced_dataset(data, bone_names):
    if data.get("schema") != DATASET_SCHEMA or not data.get("depth_visibility_enabled"):
        raise ValueError("Depth/visibility enabled: regenerate Stage 2 (old dataset has no supervised depth/visibility).")
    if data.get("bone_names") != sorted(bone_names):
        raise ValueError("Dataset skeleton/keypoint order differs from studio_rig_meta.json.")
    if not data.get("samples"):
        raise ValueError("Empty synthetic dataset")
    if float(data["calibration"].get("depth_scale", 0)) <= 0:
        raise ValueError("Missing depth scale")
    cov_audit = data.get("coverage_audit")
    if isinstance(cov_audit, dict) and not cov_audit.get("passed", True):
        raise ValueError(f"Synthetic pose dataset failed coverage/decoupling audit: {cov_audit.get('failures')}")
    known_visibility = 0
    for sample in data["samples"]:
        for bone in bone_names:
            joint = sample["joints"][bone]
            for view in ("front", "side"):
                for endpoint in ("head", "tail"):
                    label = joint[f"{view}_{endpoint}_aux"]
                    for field in ("depth_camera", "depth_relative", "visibility_ratio"):
                        if not np.isfinite(label[field]):
                            raise ValueError(f"Non-finite {field}")
                    if not 0 <= label["visibility_ratio"] <= 1:
                        raise ValueError("Invalid visibility ratio")
                    known_visibility += bool(label["visibility_valid"])
    if not known_visibility:
        raise ValueError("No valid surface visibility labels. Inspect skin weights/part ownership before training.")


def video_domain_degrade(image):
    """Photometric only: preserve pixel-coordinate labels. Uses worker-seeded NumPy RNG."""
    image = image.convert("RGB")
    w, h = image.size
    if np.random.rand() < 0.5:
        scale = float(np.random.uniform(0.45, 0.9))
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
        image = image.resize((w, h), Image.Resampling.BILINEAR)
    if np.random.rand() < 0.45:
        arr = np.asarray(image, dtype=np.float32)
        arr *= np.random.uniform(0.85, 1.15, size=(1, 1, 3))
        ramp = np.linspace(0.85, 1.15, w, dtype=np.float32)[None, :, None]
        if np.random.rand() < 0.5:
            ramp = ramp[:, ::-1]
        image = Image.fromarray(np.clip(arr * ramp, 0, 255).astype(np.uint8))
    if np.random.rand() < 0.25:
        # Edge-pad rather than wrap pixels from the opposite panel edge.
        dx, dy = np.random.randint(-3, 4, size=2)
        arr = np.asarray(image)
        padded = np.pad(arr, ((3, 3), (3, 3), (0, 0)), mode="edge")
        ghost = Image.fromarray(padded[3 + dy:3 + dy + h, 3 + dx:3 + dx + w])
        image = Image.blend(image, ghost, float(np.random.uniform(0.08, 0.22)))
    if np.random.rand() < 0.5:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=int(np.random.randint(35, 86)))
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
    return image


def fuse_depth_visibility(uv_front, uv_side, conf_front, conf_side, calib,
                          depth_front, depth_side, visibility_front, visibility_side,
                          logvar_front, logvar_side):
    """Fixed dual-orthographic studio only; inferred depth remains an uncertain prior.

    Front camera looks +Y; side camera looks -X. Relative depth is referenced to
    the fixed studio center, divided by depth_scale (not the world vertical Z).
    """
    scale = float(calib["depth_scale"])
    cx, cy, cz = calib["center"]
    W, H, S = float(calib["panel_width"]), float(calib["panel_height"]), float(calib["ortho_scale"])
    vf, vs = np.clip(visibility_front, 0, 1), np.clip(visibility_side, 0, 1)
    cf, cs = np.clip(conf_front, 0.001, 1), np.clip(conf_side, 0.001, 1)
    x_direct = cx + (uv_front[..., 0] / W - 0.5) * S * W / H
    y_direct = cy + (uv_side[..., 0] / W - 0.5) * S * W / H
    x_depth = cx - depth_side * scale
    y_depth = cy + depth_front * scale
    zf, zs = cz + (0.5 - uv_front[..., 1] / H) * S, cz + (0.5 - uv_side[..., 1] / H) * S

    # Asymmetric epipolar V-consistency & cross-view depth consistency gating:
    # When |zf - zs| is large (one view is occluded or jumped to a wrong landmark),
    # compare each view's visibility * confidence and cross-check its direct coordinate
    # against the other view's depth prior so the occluded view is downweighted rather than averaged 50/50.
    s_safe = max(S, 1e-4)
    dz = np.abs(zf - zs)
    epi_ok = np.exp(-0.5 * (dz / (0.065 * s_safe)) ** 2)
    qf = cf * (0.10 + 0.90 * vf)
    qs = cs * (0.10 + 0.90 * vs)
    q_sum = np.maximum(1e-6, qf + qs)
    f_ratio = qf / q_sum
    s_ratio = qs / q_sum

    wf = (cf ** 2) * (0.05 + 0.95 * vf) * (epi_ok + (1.0 - epi_ok) * (2.0 * f_ratio) ** 2)
    ws = (cs ** 2) * (0.05 + 0.95 * vs) * (epi_ok + (1.0 - epi_ok) * (2.0 * s_ratio) ** 2)
    # Depth is capped below a clear direct observation; boost cross-view depth when the orthogonal view is heavily occluded.
    f_occ_boost = 1.0 + 1.8 * np.maximum(0.0, vf - vs - 0.35)
    s_occ_boost = 1.0 + 1.8 * np.maximum(0.0, vs - vf - 0.35)
    wdf = (
        0.35
        * (cf ** 2)
        * (0.25 + 0.75 * vf)
        * f_occ_boost
        / (1.0 + np.exp(np.clip(logvar_front, -6, 3)) / (0.05 ** 2))
    )
    wds = (
        0.35
        * (cs ** 2)
        * (0.25 + 0.75 * vs)
        * s_occ_boost
        / (1.0 + np.exp(np.clip(logvar_side, -6, 3)) / (0.05 ** 2))
    )
    points = np.stack(
        (
            (wf * x_direct + wds * x_depth) / np.maximum(1e-12, wf + wds),
            (ws * y_direct + wdf * y_depth) / np.maximum(1e-12, ws + wdf),
            (wf * zf + ws * zs) / np.maximum(1e-12, wf + ws),
        ),
        axis=-1,
    ).astype(np.float32)
    reliability = np.maximum(cf * (0.2 + 0.8 * vf), cs * (0.2 + 0.8 * vs))
    return points, reliability
