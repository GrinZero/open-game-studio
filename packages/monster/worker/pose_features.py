"""Shared v2.3 data contract. No torch/bpy imports: usable in both worker runtimes."""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from pathlib import Path

_WORKER_DIR = str(Path(__file__).resolve().parent)
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore

MOCAP_PRIORS_SCHEMA = "monster-mocap-priors-v1"
MOCAP_PRIORS_ENV = "MONSTER_ENABLE_MOCAP_PRIORS"


def mocap_priors_enabled() -> bool:
    """Legacy v2 mocap priors stub; deprecated and always returns False in v3."""
    return False


def extract_unique_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    """Extracts unique undirected mesh edges (M, 2) from triangle indices (F, 3)."""
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] < 3 or len(f) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    e01 = np.sort(f[:, [0, 1]], axis=1)
    e12 = np.sort(f[:, [1, 2]], axis=1)
    e20 = np.sort(f[:, [2, 0]], axis=1)
    all_edges = np.vstack([e01, e12, e20])
    return np.unique(all_edges, axis=0)


def compute_mesh_edge_stretch_metric(
    rest_vertices: np.ndarray,
    posed_vertices: np.ndarray,
    edges_or_faces: np.ndarray,
    max_stretch: float = 1.80,
    min_stretch: float = 0.45,
    max_bad_ratio: float = 0.02,
) -> dict[str, Any]:
    """Evaluates deformed mesh edge stretch ratios r = L_posed / L_rest to detect skinning tear/collapse."""
    v_rest = np.asarray(rest_vertices, dtype=np.float64)
    v_posed = np.asarray(posed_vertices, dtype=np.float64)
    ef = np.asarray(edges_or_faces, dtype=np.int64)
    if ef.ndim == 2 and ef.shape[1] == 3:
        edges = extract_unique_edges_from_faces(ef)
    elif ef.ndim == 2 and ef.shape[1] == 2:
        edges = ef
    else:
        edges = np.zeros((0, 2), dtype=np.int64)

    if len(edges) == 0 or len(v_rest) == 0 or v_rest.shape != v_posed.shape:
        return {
            "passed": True,
            "bad_edge_ratio": 0.0,
            "max_stretch_ratio": 1.0,
            "min_stretch_ratio": 1.0,
            "edge_count": 0,
        }

    d_rest = np.linalg.norm(v_rest[edges[:, 0]] - v_rest[edges[:, 1]], axis=-1)
    d_posed = np.linalg.norm(v_posed[edges[:, 0]] - v_posed[edges[:, 1]], axis=-1)
    valid = d_rest > 1e-5
    if not np.any(valid):
        return {
            "passed": True,
            "bad_edge_ratio": 0.0,
            "max_stretch_ratio": 1.0,
            "min_stretch_ratio": 1.0,
            "edge_count": int(len(edges)),
        }

    ratios = d_posed[valid] / d_rest[valid]
    bad_mask = (ratios > float(max_stretch)) | (ratios < float(min_stretch))
    bad_ratio = float(np.mean(bad_mask))
    return {
        "passed": bool(bad_ratio <= float(max_bad_ratio)),
        "bad_edge_ratio": round(bad_ratio, 5),
        "max_stretch_ratio": round(float(np.max(ratios)), 4),
        "min_stretch_ratio": round(float(np.min(ratios)), 4),
        "edge_count": int(np.sum(valid)),
    }


def verify_blender_lbs_skin_consistency(
    rest_vertices: np.ndarray,
    blender_posed_vertices: np.ndarray,
    bone_weights: np.ndarray,
    bone_indices: np.ndarray,
    rest_heads: np.ndarray,
    rest_rots: np.ndarray,
    posed_heads: np.ndarray,
    posed_rots: np.ndarray,
    lbs_deform_fn: Any,
    tol_ratio: float = 0.015,
) -> dict[str, Any]:
    """Verifies that Blender evaluated mesh deformation matches analytical LBS."""
    v_rest = np.asarray(rest_vertices, dtype=np.float64)
    v_bl = np.asarray(blender_posed_vertices, dtype=np.float64)
    if len(v_rest) == 0 or v_rest.shape != v_bl.shape:
        return {
            "consistent": True,
            "max_vertex_error": 0.0,
            "mean_vertex_error": 0.0,
            "relative_max_error": 0.0,
        }
    v_lbs = lbs_deform_fn(
        v_rest,
        bone_weights,
        bone_indices,
        rest_heads,
        rest_rots,
        posed_heads,
        posed_rots,
    )
    errs = np.linalg.norm(v_lbs - v_bl, axis=-1)
    max_err = float(np.max(errs))
    mean_err = float(np.mean(errs))
    bbox_diag = float(np.linalg.norm(np.max(v_rest, axis=0) - np.min(v_rest, axis=0)))
    rel_max = max_err / max(1e-6, bbox_diag)
    return {
        "consistent": bool(rel_max <= float(tol_ratio)),
        "max_vertex_error": round(max_err, 6),
        "mean_vertex_error": round(mean_err, 6),
        "relative_max_error": round(rel_max, 6),
    }


def dejitter_velocity_gated(
    x: np.ndarray,
    v_fast: float | None = None,
    passes: int = 2,
    strength: float = 0.55,
) -> np.ndarray:
    """Velocity-gated micro-jitter filter suppressing stationary tremor while passing motion peaks."""
    arr = np.asarray(x, dtype=np.float64)
    T = int(arr.shape[0])
    if T < 5 or strength <= 0.0:
        return arr.copy()
    lp = arr.copy()
    for _ in range(max(1, int(passes))):
        nxt = lp.copy()
        nxt[1:-1] = 0.25 * lp[:-2] + 0.50 * lp[1:-1] + 0.25 * lp[2:]
        lp = nxt
    step_diff = np.linalg.norm(arr[1:] - arr[:-1], axis=-1, keepdims=True)
    speed_prev = np.concatenate([step_diff[:1], step_diff], axis=0)
    speed_next = np.concatenate([step_diff, step_diff[-1:]], axis=0)
    speed = np.maximum(speed_prev, speed_next)
    if v_fast is None or v_fast <= 0.0:
        med_speed = np.median(speed, axis=0, keepdims=True)
        v_thresh = np.maximum(1e-5, 1.35 * med_speed)
    else:
        v_thresh = float(v_fast)
    gate = float(strength) / (1.0 + (speed / v_thresh) ** 2)
    return arr + gate * (lp - arr)


def bone_motion_blur_speed_factor(kind: str, sub_role: str = "", chain_index: int = 0) -> float:
    """Returns per-bone relative speed factor for synthetic motion blur."""
    k = kind.strip().lower()
    idx = max(0, int(chain_index))
    if k == "rigid":
        return 0.0
    if k in ("root", "spine"):
        return 0.15
    if k in ("head", "jaw"):
        return 0.35
    if k in ("limb", "digit", "wing"):
        base = 1.45 if sub_role in ("forearm_hinge", "lower_limb", "hand_hub") else 1.10
        return min(2.4, base + 0.28 * idx)
    if k in ("tail", "flex"):
        return min(2.2, 0.90 + 0.25 * idx)
    return 1.0


def calibrate_view_scale_drift(
    uv_front: np.ndarray,
    uv_side: np.ndarray,
    conf_front: np.ndarray | None = None,
    conf_side: np.ndarray | None = None,
    calib: dict[str, Any] | None = None,
    uv_right: np.ndarray | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Passthrough fallback for view drift compensation in v3."""
    diag = {
        "view_drift_calibrated": False,
        "max_abs_dy_px": 0.0,
        "mean_abs_dy_px": 0.0,
        "max_abs_scale_pct": 0.0,
    }
    return uv_front.copy(), (uv_right.copy() if uv_right is not None else None), diag

FEATURE_ENV = "MONSTER_ENABLE_DEPTH_VISIBILITY"
RESOLUTION_2K_ENV = "MONSTER_ENABLE_2K"
THREE_VIEW_ENV = "MONSTER_ENABLE_3VIEW"
DENSE_SURFACE_ENV = "MONSTER_ENABLE_DENSE_SURFACE"
PIPELINE_MODE_ENV = "MONSTER_PIPELINE_MODE"
RIG_CONTRACT_ENV = "MONSTER_RIG_CONTRACT"
FOLD_DIGITS_ENV = "MONSTER_FOLD_DIGITS"
KEYPOINT_MODE_ENV = "MONSTER_KEYPOINT_MODE"
VISIBILITY_MODE_ENV = "MONSTER_VISIBILITY_MODE"
DECODE_MODE_ENV = "MONSTER_DECODE_MODE"
SOLVER_MODE_ENV = "MONSTER_SOLVER_MODE"
CLEAN_MOTION_ENV = "MONSTER_CLEAN_MOTION"
SURFACE_CONTACT_ENV = "MONSTER_SURFACE_CONTACT"
EXPORT_NPZ_LABELS_ENV = "MONSTER_EXPORT_NPZ_LABELS"
NO_CUT_TRACKS_ENV = "MONSTER_NO_CUT_TRACKS"
TWOWAY_JUMP_ENV = "MONSTER_TWOWAY_JUMP"
PANEL_DRIFT_ENV = "MONSTER_PANEL_DRIFT"
BACKBONE_ENV = "MONSTER_BACKBONE"
READER_CROPS_ENV = "MONSTER_READER_CROPS"
READER_DENSE_ENV = "MONSTER_READER_DENSE"
DATASET_SCHEMA = "monster-pose-depth-visibility-v1"
FEATURE_VERSION = "v3-anatomical-reader-4"
DEFAULT_SYNTH_SAMPLES = 5000
DEFAULT_STUDIO_RESOLUTION = (672, 768, 1344, 768)
STUDIO_2K_RESOLUTION = (1280, 1440, 2560, 1440)
STUDIO_3VIEW_DEFAULT_RESOLUTION = (448, 768, 1344, 768)
STUDIO_3VIEW_2K_RESOLUTION = (896, 1440, 2688, 1440)
DUAL_VIEW_ORDER = ("front", "side")
THREE_VIEW_ORDER = ("left", "front", "right")


def _parse_bool_env(env_key: str, default: str = "0") -> bool:
    value = os.environ.get(env_key, default).strip().lower()
    if value not in {"0", "1", "true", "false", "on", "off"}:
        raise ValueError(f"Invalid {env_key}: {value!r}")
    return value in {"1", "true", "on"}


def pipeline_mode() -> str:
    """Master switch between the two supported end-to-end pipelines:
    - `'video2rig'` (Path 1, default): Full video2rig pipeline + MonsterStudio enhancements.
    - `'legacy'`    (Path 2, feature-flagged): Legacy MonsterStudio v2.3 pipeline.
    """
    raw = os.environ.get(PIPELINE_MODE_ENV, "video2rig").strip().lower()
    if raw in {"video2rig", "v3", "1", "true", "on", "default"}:
        return "video2rig"
    if raw in {"legacy", "v2", "v2.3", "0", "false", "off"}:
        return "legacy"
    raise ValueError(
        f"Invalid {PIPELINE_MODE_ENV}: {raw!r} (expected 'video2rig' or 'legacy')"
    )


def video2rig_mode_enabled() -> bool:
    return pipeline_mode() == "video2rig"


def rig_contract_v3_enabled() -> bool:
    raw = os.environ.get(RIG_CONTRACT_ENV, "").strip().lower()
    if not raw:
        return video2rig_mode_enabled()
    if raw in {"v3", "video2rig", "1", "true", "on"}:
        return True
    if raw in {"legacy", "v2", "0", "false", "off"}:
        return False
    raise ValueError(f"Invalid {RIG_CONTRACT_ENV}: {raw!r}")


def fold_digits_enabled() -> bool:
    """Our default preserves full finger/toe digit chains (`fold_digits=False`) unless explicitly enabled."""
    if FOLD_DIGITS_ENV in os.environ:
        return _parse_bool_env(FOLD_DIGITS_ENV, "0")
    if "MONSTER_PRUNE_DIGITS" in os.environ:
        return _parse_bool_env("MONSTER_PRUNE_DIGITS", "0")
    return False


def keypoint_mode() -> str:
    raw = os.environ.get(KEYPOINT_MODE_ENV, "").strip().lower()
    if not raw:
        return "video2rig" if video2rig_mode_enabled() else "legacy"
    if raw in {"video2rig", "v3", "surface"}:
        return "video2rig"
    if raw in {"legacy", "v2", "head_tail"}:
        return "legacy"
    raise ValueError(f"Invalid {KEYPOINT_MODE_ENV}: {raw!r}")


def visibility_mode() -> str:
    raw = os.environ.get(VISIBILITY_MODE_ENV, "").strip().lower()
    if not raw:
        return "hybrid" if video2rig_mode_enabled() else "bvh"
    if raw in {"zbuffer", "visible_npz", "video2rig"}:
        return "zbuffer"
    if raw in {"bvh", "raycast", "legacy"}:
        return "bvh"
    if raw in {"hybrid", "both"}:
        return "hybrid"
    raise ValueError(f"Invalid {VISIBILITY_MODE_ENV}: {raw!r}")


def decode_mode() -> str:
    raw = os.environ.get(DECODE_MODE_ENV, "").strip().lower()
    if not raw:
        return "parabola" if video2rig_mode_enabled() else "dsnt"
    if raw in {"parabola", "video2rig", "peak_parabola"}:
        return "parabola"
    if raw in {"dsnt", "soft_argmax", "windowed_dsnt"}:
        return "dsnt"
    raise ValueError(f"Invalid {DECODE_MODE_ENV}: {raw!r}")


def solver_mode() -> str:
    raw = os.environ.get(SOLVER_MODE_ENV, "").strip().lower()
    if not raw:
        return "reprojection_v3" if video2rig_mode_enabled() else "legacy_3d_kind"
    if raw in {"reprojection_v3", "video2rig", "v3", "2d_reprojection"}:
        return "reprojection_v3"
    if raw in {"legacy_3d_kind", "legacy", "v2", "3d_triangulate"}:
        return "legacy_3d_kind"
    raise ValueError(f"Invalid {SOLVER_MODE_ENV}: {raw!r}")


def clean_motion_enabled() -> bool:
    if CLEAN_MOTION_ENV in os.environ:
        return _parse_bool_env(CLEAN_MOTION_ENV, "1")
    return video2rig_mode_enabled()


def surface_contact_enabled() -> bool:
    if SURFACE_CONTACT_ENV in os.environ:
        return _parse_bool_env(SURFACE_CONTACT_ENV, "1")
    return video2rig_mode_enabled()


def export_npz_labels_enabled() -> bool:
    if EXPORT_NPZ_LABELS_ENV in os.environ:
        return _parse_bool_env(EXPORT_NPZ_LABELS_ENV, "1")
    return video2rig_mode_enabled()


def no_cut_tracks_enabled() -> bool:
    if NO_CUT_TRACKS_ENV in os.environ:
        return _parse_bool_env(NO_CUT_TRACKS_ENV, "1")
    return video2rig_mode_enabled()


def twoway_jump_enabled() -> bool:
    if TWOWAY_JUMP_ENV in os.environ:
        return _parse_bool_env(TWOWAY_JUMP_ENV, "1")
    return video2rig_mode_enabled()


def panel_drift_enabled() -> bool:
    if PANEL_DRIFT_ENV in os.environ:
        return _parse_bool_env(PANEL_DRIFT_ENV, "1")
    return video2rig_mode_enabled()


def backbone_mode() -> str:
    """Resolve the active DINOv2 backbone variant:
    - `'dinov2-small'`      (default in video2rig mode: ViT-S/14 mode, d=384, 12 blocks, FROZEN=4)
    - `'dinov2_vitb14_reg'` (default in legacy mode: ViT-B/14 with registers, d=768, 12 blocks)
    """
    raw = os.environ.get(BACKBONE_ENV, "").strip().lower()
    if not raw:
        return "dinov2-small" if video2rig_mode_enabled() else "dinov2_vitb14_reg"
    if raw in {"dinov2_vitb14_reg", "vitb14_reg", "vitb14", "base", "facebook/dinov2-base"}:
        return "dinov2_vitb14_reg"
    if raw in {
        "dinov2-small",
        "facebook/dinov2-small",
        "dinov2_vits14",
        "dinov2_vits14_reg",
        "vits14",
        "vits14_reg",
        "small",
    }:
        return "dinov2-small"
    raise ValueError(
        f"Invalid {BACKBONE_ENV}: {raw!r} (expected 'dinov2_vitb14_reg' or 'dinov2-small')"
    )


def reader_crops_enabled() -> bool:
    """Whether the independent hand/claw crop amplifier (`reader-crops.pt`) is enabled (default off)."""
    return _parse_bool_env(READER_CROPS_ENV, "0")


def reader_dense_enabled() -> bool:
    """Whether the independent dense surface reader (`reader-dense.pt`) is enabled (default off)."""
    return _parse_bool_env(READER_DENSE_ENV, "0")


def pipeline_feature_flags() -> dict[str, object]:
    """Returns a complete snapshot of the active two-path feature flag configuration."""
    mode = pipeline_mode()
    return {
        "pipeline_mode": mode,
        "video2rig_mode": mode == "video2rig",
        "backbone_mode": backbone_mode(),
        "reader_crops": reader_crops_enabled(),
        "reader_dense": reader_dense_enabled(),
        "rig_contract_v3": rig_contract_v3_enabled(),
        "fold_digits": fold_digits_enabled(),
        "keypoint_mode": keypoint_mode(),
        "visibility_mode": visibility_mode(),
        "decode_mode": decode_mode(),
        "solver_mode": solver_mode(),
        "clean_motion": clean_motion_enabled(),
        "surface_contact": surface_contact_enabled(),
        "export_npz_labels": export_npz_labels_enabled(),
        "no_cut_tracks": no_cut_tracks_enabled(),
        "twoway_jump": twoway_jump_enabled(),
        "panel_drift": panel_drift_enabled(),
        "depth_visibility": features_enabled(),
        "dense_surface": dense_surface_enabled(),
        "enable_2k": resolution_2k_enabled(),
        "enable_3view": three_view_enabled(),
    }


def features_enabled() -> bool:
    return _parse_bool_env(FEATURE_ENV, "1")


def dense_surface_enabled() -> bool:
    return _parse_bool_env(DENSE_SURFACE_ENV, "0")


def resolution_2k_enabled() -> bool:
    return _parse_bool_env(RESOLUTION_2K_ENV, "0")


def three_view_enabled() -> bool:
    return _parse_bool_env(THREE_VIEW_ENV, "0")


def studio_view_order(enable_3view: bool | None = None) -> tuple[str, ...]:
    """Returns panel order left-to-right: ('front', 'side') for 2-view or ('left', 'front', 'right') for 3-view."""
    use_3view = three_view_enabled() if enable_3view is None else bool(enable_3view)
    return THREE_VIEW_ORDER if use_3view else DUAL_VIEW_ORDER


def studio_resolution(
    enable_2k: bool | None = None, enable_3view: bool | None = None
) -> tuple[int, int, int, int]:
    """Returns (panel_width, panel_height, composite_width, composite_height)."""
    use_2k = resolution_2k_enabled() if enable_2k is None else bool(enable_2k)
    use_3view = three_view_enabled() if enable_3view is None else bool(enable_3view)
    if use_3view:
        return STUDIO_3VIEW_2K_RESOLUTION if use_2k else STUDIO_3VIEW_DEFAULT_RESOLUTION
    return STUDIO_2K_RESOLUTION if use_2k else DEFAULT_STUDIO_RESOLUTION


def idle_split_filename(
    comp_w: int | None = None, comp_h: int | None = None, enable_3view: bool | None = None
) -> str:
    use_3view = three_view_enabled() if enable_3view is None else bool(enable_3view)
    if comp_w is None or comp_h is None:
        _, _, comp_w, comp_h = studio_resolution(enable_3view=use_3view)
    prefix = "idle_3view" if use_3view else "idle_split"
    return f"{prefix}_{int(comp_w)}x{int(comp_h)}.png"


def dinov2_panel_padding(panel_w: int, panel_h: int, patch_size: int = 14) -> tuple[int, int, int, int]:
    """Returns symmetric (pad_top, pad_bottom, pad_left, pad_right) so padded (H, W) are multiples of patch_size=14."""
    pad_w = (patch_size - (int(panel_w) % patch_size)) % patch_size
    pad_h = (patch_size - (int(panel_h) % patch_size)) % patch_size
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    return pad_top, pad_bottom, pad_left, pad_right


def dense_grid_shape(panel_w: int, panel_h: int, patch_size: int = 14) -> tuple[int, int]:
    """Returns (H_hm, W_hm) spatial grid dimensions of the 4x deconv heatmap & dense surface head."""
    pad_top, pad_bottom, pad_left, pad_right = dinov2_panel_padding(panel_w, panel_h, patch_size=patch_size)
    h_pad = int(panel_h) + pad_top + pad_bottom
    w_pad = int(panel_w) + pad_left + pad_right
    return (h_pad // patch_size) * 4, (w_pad // patch_size) * 4


def dataset_signature(blend_path, meta_path, num_samples, seed=42):
    meta = json.loads(Path(meta_path).read_text())
    meta.pop("calibration", None)  # Stage 2 adds calibration; it is not a rig change.
    digest = hashlib.sha256()
    from pose_sampling_v3 import SAMPLING_CONTRACT, POSE_POLICY
    digest.update(f"{SAMPLING_CONTRACT}:{POSE_POLICY}".encode())
    with open(blend_path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    digest.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode())
    use_2k = resolution_2k_enabled()
    use_3view = three_view_enabled()
    use_dense = dense_surface_enabled()
    if use_3view:
        res_tag = ":2688x1440:3view" if use_2k else ":1344x768:3view"
    else:
        res_tag = ":2560x1440" if use_2k else ""
    if use_dense:
        res_tag += ":dense"
    if pipeline_mode() != "video2rig":
        res_tag += f":mode={pipeline_mode()}"
    if keypoint_mode() not in ("video2rig", "legacy" if pipeline_mode() == "legacy" else "video2rig"):
        res_tag += f":kp={keypoint_mode()}"
    if visibility_mode() not in ("hybrid" if pipeline_mode() == "video2rig" else "bvh",):
        res_tag += f":vis={visibility_mode()}"
    digest.update(json.dumps({key: os.environ.get(key) for key in ("MONSTER_CAMERA_LENS", "MONSTER_STUDIO_LAYOUT", "MONSTER_FOLD_DIGITS")},sort_keys=True).encode())
    from strip_renderer import strip_settings
    from blender_runtime import BPY_VERSION
    digest.update(json.dumps({
        "render_contract": "monster-eevee-crop-v1",
        "bpy_version": BPY_VERSION,
        "engine": os.environ.get("MONSTER_RENDER_ENGINE", "EEVEE").upper(),
        "eevee_samples": int(os.environ.get("MONSTER_EEVEE_SAMPLES", "8")),
        "cycles_samples": int(os.environ.get("MONSTER_CYCLES_SAMPLES", "16")),
        "strip": strip_settings(),
    }, sort_keys=True).encode())
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
        v3 = data.get("reader_contract") == "monster-reader-v3"
        if v3 and data.get("start_pose_contract") != "monster-start-pose-v1":
            return False
        if v3 and data.get("context_geometry_contract") != "monster-context-pose-v1":
            return False
        if v3 and data.get('sampling_contract') != 'monster-calibrated-sampling-v2':
            return False
        if bool(data.get("dense_surface_enabled", False)) != (v3 or dense_surface_enabled()):
            return False
        panel_w, panel_h, comp_w, comp_h = studio_resolution()
        calib = data.get("calibration", {})
        layout=os.environ.get('MONSTER_STUDIO_LAYOUT','horizontal')
        if v3 and calib.get('layout','horizontal') != layout:
            return False
        if v3 and layout=='stacked':
            comp_w,comp_h=panel_w,panel_h*2
        if calib.get("panel_width") != panel_w or calib.get("panel_height") != panel_h:
            return False
        if calib.get("composite_width") != comp_w or calib.get("composite_height") != comp_h:
            return False
        expected_views = 3 if three_view_enabled() else 2
        if int(calib.get("num_views", 2)) != expected_views:
            return False
        samples = data["samples"]
        if v3 or dense_surface_enabled():
            if not (path / "rig_skinning.npz").is_file():
                return False
            if not all(
                sample.get("dense_map") and (path / "dense" / sample["dense_map"]).is_file()
                for sample in samples
            ):
                return False
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
    num_views = int(data["calibration"].get("num_views", 2))
    aux_views = ("front", "side", "right") if num_views == 3 else ("front", "side")
    check_dense = bool(data.get("dense_surface_enabled", False))
    known_visibility = 0
    for sample in data["samples"]:
        if check_dense and not sample.get("dense_map"):
            raise ValueError("Dense surface dataset is missing sample dense_map references.")
        for bone in bone_names:
            joint = sample["joints"][bone]
            if num_views == 3 and ("right_uv_head" not in joint or "right_uv_tail" not in joint):
                raise ValueError("3-view dataset is missing right_uv_head / right_uv_tail labels.")
            for view in aux_views:
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
                          logvar_front, logvar_side,
                          uv_right=None, conf_right=None, depth_right=None,
                          visibility_right=None, logvar_right=None):
    """Fixed dual- or tri-orthographic studio triangulation with visibility/depth fusion.

    Front camera looks +Y; left/side camera looks -X; optional right camera looks +X.
    Relative depth is referenced to the fixed studio center, divided by depth_scale.
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
    s_safe = max(S, 1e-4)

    # Learned depth is a completion cue, not an independent ground-truth ray.
    # Gate inconsistent depth against a visible direct image coordinate even
    # when the auxiliary head reports an overconfident variance. Tolerances are
    # expressed in pixels at the reference resolution, hence resolution invariant.
    direct_sigma = 3.0 * (H / 768.0) * S / H

    def depth_agreement(predicted, direct, direct_quality):
        residual = (predicted - direct) / max(1e-8, direct_sigma)
        return 1.0 / (1.0 + np.clip(direct_quality, 0.0, 1.0) * residual ** 2)

    if uv_right is not None and conf_right is not None:
        cr = np.clip(conf_right, 0.001, 1)
        vr = np.clip(visibility_right if visibility_right is not None else np.ones_like(cr), 0, 1)
        dr = depth_right if depth_right is not None else np.zeros_like(cr)
        lvr = logvar_right if logvar_right is not None else np.zeros_like(cr)

        # Right camera at -X looking +X: image +X points toward -Y (character front)
        y_direct_r = cy - (uv_right[..., 0] / W - 0.5) * S * W / H
        zr = cz + (0.5 - uv_right[..., 1] / H) * S
        x_depth_r = cx + dr * scale

        qf = cf * (0.10 + 0.90 * vf)
        qs = cs * (0.10 + 0.90 * vs)
        qr = cr * (0.10 + 0.90 * vr)

        epi_sigma = 8.0 * (H / 768.0) * S / H
        epi_fs = np.exp(-0.5 * (np.abs(zf - zs) / epi_sigma) ** 2)
        epi_fr = np.exp(-0.5 * (np.abs(zf - zr) / epi_sigma) ** 2)
        epi_sr = np.exp(-0.5 * (np.abs(zs - zr) / epi_sigma) ** 2)

        q_all = np.maximum(1e-6, qf + qs + qr)
        f_dom = (3.0 * qf / q_all) ** 2
        s_dom = (3.0 * qs / q_all) ** 2
        r_dom = (3.0 * qr / q_all) ** 2

        epi_f = np.maximum(epi_fs, epi_fr)
        epi_s = np.maximum(epi_fs, epi_sr)
        epi_r = np.maximum(epi_fr, epi_sr)

        wf = (cf ** 2) * (0.05 + 0.95 * vf) * (epi_f + (1.0 - epi_f) * f_dom)
        ws_z = (cs ** 2) * (0.05 + 0.95 * vs) * (epi_s + (1.0 - epi_s) * s_dom)
        wr_z = (cr ** 2) * (0.05 + 0.95 * vr) * (epi_r + (1.0 - epi_r) * r_dom)

        # Opposing lateral Y fusion: when Left and Right views disagree on Y (one side is occluded),
        # exponentially favor the side with higher visibility * confidence and front-epipolar agreement.
        dy_lr = np.abs(y_direct - y_direct_r)
        y_agree = np.exp(-0.5 * (dy_lr / (0.075 * s_safe)) ** 2)
        qs_eff = qs * (0.20 + 0.80 * epi_fs)
        qr_eff = qr * (0.20 + 0.80 * epi_fr)
        q_lr = np.maximum(1e-6, qs_eff + qr_eff)
        s_lr_ratio = qs_eff / q_lr
        r_lr_ratio = qr_eff / q_lr

        ws_y = ws_z * (y_agree + (1.0 - y_agree) * (2.0 * s_lr_ratio) ** 3)
        wr_y = wr_z * (y_agree + (1.0 - y_agree) * (2.0 * r_lr_ratio) ** 3)

        v_side_max = np.maximum(vs, vr)
        f_occ_boost = 1.0 + 1.8 * np.maximum(0.0, vf - v_side_max - 0.35)
        s_occ_boost = 1.0 + 1.8 * np.maximum(0.0, vs - vf - 0.35)
        r_occ_boost = 1.0 + 1.8 * np.maximum(0.0, vr - vf - 0.35)

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
        wdr = (
            0.35
            * (cr ** 2)
            * (0.25 + 0.75 * vr)
            * r_occ_boost
            / (1.0 + np.exp(np.clip(lvr, -6, 3)) / (0.05 ** 2))
        )

        y_geometry = (ws_y * y_direct + wr_y * y_direct_r) / np.maximum(1e-12, ws_y + wr_y)
        wdf *= depth_agreement(y_depth, y_geometry, np.maximum(cs * vs * epi_fs, cr * vr * epi_fr))
        wds *= depth_agreement(x_depth, x_direct, cf * vf)
        wdr *= depth_agreement(x_depth_r, x_direct, cf * vf)

        points = np.stack(
            (
                (wf * x_direct + wds * x_depth + wdr * x_depth_r) / np.maximum(1e-12, wf + wds + wdr),
                (ws_y * y_direct + wr_y * y_direct_r + wdf * y_depth) / np.maximum(1e-12, ws_y + wr_y + wdf),
                (wf * zf + ws_z * zs + wr_z * zr) / np.maximum(1e-12, wf + ws_z + wr_z),
            ),
            axis=-1,
        ).astype(np.float32)
        reliability = np.maximum(
            np.maximum(cf * (0.2 + 0.8 * vf), cs * (0.2 + 0.8 * vs)),
            cr * (0.2 + 0.8 * vr),
        )
        return points, reliability

    # Asymmetric epipolar V-consistency & cross-view depth consistency gating:
    # When |zf - zs| is large (one view is occluded or jumped to a wrong landmark),
    # compare each view's visibility * confidence and cross-check its direct coordinate
    # against the other view's depth prior so the occluded view is downweighted rather than averaged 50/50.
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


# ==============================================================================
# Multi-View Dense Surface Correspondence & Linear Blend Skinning (LBS) Helpers
# ==============================================================================


def normalize_rest_vertices_numpy(rest_verts, calib: dict) -> np.ndarray:
    """Normalizes rest-pose 3D mesh vertices `(V, 3)` into `[0, 1]^3` relative to studio center and depth_scale."""
    pts = np.asarray(rest_verts, dtype=np.float32)
    center = np.asarray(calib["center"], dtype=np.float32)
    scale = max(1e-4, float(calib.get("depth_scale", float(calib.get("ortho_scale", 1.0)) / 1.48)))
    return np.clip((pts - center) / scale + 0.5, 0.0, 1.0).astype(np.float32)


def denormalize_rest_vertices_numpy(rest_xyz_norm, calib: dict) -> np.ndarray:
    """Maps normalized `[0, 1]^3` rest-pose surface coordinates back to 3D world rest coordinates."""
    norm_pts = np.asarray(rest_xyz_norm, dtype=np.float32)
    center = np.asarray(calib["center"], dtype=np.float32)
    scale = max(1e-4, float(calib.get("depth_scale", float(calib.get("ortho_scale", 1.0)) / 1.48)))
    return ((norm_pts - 0.5) * scale + center).astype(np.float32)


def project_3d_points_to_ortho_view_numpy(pts_3d, calib: dict, view_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized orthographic projection of `(..., 3)` 3D world points into `(..., 2)` unpadded
    panel pixel coordinates `(u_px, v_px)` and `(...,)` relative camera depth (smaller = closer to camera).
    Supports all studio views: `'front'`, `'side'`/`'left'`, and `'right'`.
    """
    pts = np.asarray(pts_3d, dtype=np.float64)
    cx, cy, cz = (float(c) for c in calib["center"])
    W = float(calib["panel_width"])
    H = float(calib["panel_height"])
    S = max(1e-4, float(calib["ortho_scale"]))
    D = max(1e-4, float(calib.get("depth_scale", S / 1.48)))
    span_x = S * (W / H)
    span_z = S

    px, py, pz = pts[..., 0], pts[..., 1], pts[..., 2]
    v_px = (0.5 - (pz - cz) / span_z) * H
    v_low = view_name.strip().lower()
    if v_low == "front":
        u_px = ((px - cx) / span_x + 0.5) * W
        depth_rel = (py - cy) / D
    elif v_low in ("side", "left"):
        u_px = ((py - cy) / span_x + 0.5) * W
        depth_rel = (cx - px) / D
    elif v_low == "right":
        u_px = ((cy - py) / span_x + 0.5) * W
        depth_rel = (px - cx) / D
    else:
        raise ValueError(f"Unsupported studio orthographic view: {view_name!r}")
    uv_px = np.stack([u_px, v_px], axis=-1)
    return uv_px, depth_rel


def lbs_deform_vertices_numpy(
    rest_verts,
    skin_indices,
    skin_weights,
    rest_heads,
    rest_rots,
    posed_heads,
    posed_rots,
) -> np.ndarray:
    """Vectorized Linear Blend Skinning (LBS) transforming rest-pose vertices `(V, 3)` into posed
    3D world space:
        `v_posed[i] = sum_m w[i, m] * (R_posed[b] @ R_rest[b].T @ (v_rest[i] - h_rest[b]) + h_posed[b])`
    Supports single-frame `(B, 3)` / `(B, 3, 3)` or multi-frame `(T, B, 3)` / `(T, B, 3, 3)` bone transforms.
    """
    v_rest = np.asarray(rest_verts, dtype=np.float64)
    idx = np.asarray(skin_indices, dtype=np.int64)
    w = np.asarray(skin_weights, dtype=np.float64)
    h_rest = np.asarray(rest_heads, dtype=np.float64)
    r_rest = np.asarray(rest_rots, dtype=np.float64)
    h_posed = np.asarray(posed_heads, dtype=np.float64)
    r_posed = np.asarray(posed_rots, dtype=np.float64)

    # G_b = R_posed @ R_rest^T : shape (..., B, 3, 3)
    g_mats = np.einsum("...bij,bkj->...bik", r_posed, r_rest)
    # Offset in rest space per vertex-influence pair (V, M, 3)
    offsets = v_rest[:, None, :] - h_rest[idx]
    if g_mats.ndim == 3:
        # Single frame: g_mats[idx] is (V, M, 3, 3), h_posed[idx] is (V, M, 3)
        transformed = np.einsum("vmij,vmj->vmi", g_mats[idx], offsets) + h_posed[idx]
        return np.sum(w[:, :, None] * transformed, axis=1)
    # Multi-frame (T, B, 3, 3):
    g_vm = g_mats[:, idx]  # (T, V, M, 3, 3)
    h_vm = h_posed[:, idx]  # (T, V, M, 3)
    transformed = np.einsum("tvmij,vmj->tvmi", g_vm, offsets) + h_vm
    return np.sum(w[None, :, :, None] * transformed, axis=2)


def erode_mask_3x3_numpy(mask: np.ndarray) -> np.ndarray:
    """Applies a 3x3 morphological erosion on a 2D `(H, W)` or 3D `(N, H, W)` boolean/float mask
    so boundary silhouette pixels are excluded prior to dense correspondence matching.
    """
    m = np.asarray(mask) > 0.5
    if m.ndim == 2:
        padded = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        H, W = m.shape
        out = np.ones((H, W), dtype=bool)
        for dy in range(3):
            for dx in range(3):
                out &= padded[dy : dy + H, dx : dx + W]
        return out
    if m.ndim == 3:
        padded = np.pad(m, ((0, 0), (1, 1), (1, 1)), mode="constant", constant_values=False)
        N, H, W = m.shape
        out = np.ones((N, H, W), dtype=bool)
        for dy in range(3):
            for dx in range(3):
                out &= padded[:, dy : dy + H, dx : dx + W]
        return out
    raise ValueError(f"Expected 2D or 3D mask for 3x3 erosion, got shape {m.shape}")


_BARY_LATTICE_WEIGHTS: np.ndarray | None = None


def _get_barycentric_lattice(subdiv: int = 6) -> np.ndarray:
    """Returns a cached symmetric barycentric coordinate lattice `(M, 3)` covering a triangle."""
    global _BARY_LATTICE_WEIGHTS
    if _BARY_LATTICE_WEIGHTS is not None:
        return _BARY_LATTICE_WEIGHTS
    pts = []
    for i in range(subdiv + 1):
        for j in range(subdiv + 1 - i):
            k = subdiv - i - j
            pts.append((i / subdiv, j / subdiv, k / subdiv))
    # Include centroid and interior third-points for dense fill
    pts.append((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    _BARY_LATTICE_WEIGHTS = np.array(pts, dtype=np.float32)
    return _BARY_LATTICE_WEIGHTS


def rasterize_ortho_dense_rest_map_numpy(
    posed_verts,
    rest_xyz_norm,
    triangles,
    calib: dict,
    view_name: str | None = None,
    grid_hw: tuple[int, int] | None = None,
    *,
    view: str | None = None,
    grid_h: int | None = None,
    grid_w: int | None = None,
    already_normalized: bool = False,
) -> dict[str, np.ndarray]:
    """Vectorized orthographic Z-buffer rasterizer generating ground-truth dense surface maps
    `{"mask": (H_hm, W_hm), "rest_xyz": (3, H_hm, W_hm)}` aligned with the padded DINOv2 heatmap
    grid `(H_hm, W_hm)` for any view (`'front'`, `'side'`/`'left'`, `'right'`).
    """
    eff_view = view_name or view or "front"
    W_panel = int(calib["panel_width"])
    H_panel = int(calib["panel_height"])
    pad_top, pad_bottom, pad_left, pad_right = dinov2_panel_padding(W_panel, H_panel)
    W_pad = float(W_panel + pad_left + pad_right)
    H_pad = float(H_panel + pad_top + pad_bottom)
    if grid_hw is not None:
        H_hm, W_hm = int(grid_hw[0]), int(grid_hw[1])
    elif grid_h is not None and grid_w is not None:
        H_hm, W_hm = int(grid_h), int(grid_w)
    else:
        H_hm, W_hm = dense_grid_shape(W_panel, H_panel)

    mask_out = np.zeros((H_hm, W_hm), dtype=np.float32)
    rest_out = np.zeros((3, H_hm, W_hm), dtype=np.float32)

    verts_p = np.asarray(posed_verts, dtype=np.float64)
    verts_r = (
        np.asarray(rest_xyz_norm, dtype=np.float32)
        if already_normalized
        else normalize_rest_vertices_numpy(rest_xyz_norm, calib)
    )
    if verts_p.shape[0] == 0:
        return {"mask": mask_out, "rest_xyz": rest_out}

    uv_px, depth_rel = project_3d_points_to_ortho_view_numpy(verts_p, calib, eff_view)
    u_grid = ((uv_px[:, 0] + pad_left) / W_pad) * float(W_hm - 1)
    v_grid = ((uv_px[:, 1] + pad_top) / H_pad) * float(H_hm - 1)

    tris = np.asarray(triangles, dtype=np.int64) if triangles is not None and len(triangles) > 0 else None
    if tris is not None and tris.ndim == 2 and tris.shape[0] > 0:
        bary = _get_barycentric_lattice(subdiv=6)  # (M, 3)
        t_u = u_grid[tris]  # (F, 3)
        t_v = v_grid[tris]  # (F, 3)
        t_d = depth_rel[tris]  # (F, 3)
        t_r = verts_r[tris]  # (F, 3, 3)

        samp_u = (t_u @ bary.T).reshape(-1)
        samp_v = (t_v @ bary.T).reshape(-1)
        samp_d = (t_d @ bary.T).reshape(-1)
        samp_r = np.einsum("fvc,mv->fmc", t_r, bary).reshape(-1, 3)
    else:
        samp_u = u_grid.reshape(-1)
        samp_v = v_grid.reshape(-1)
        samp_d = depth_rel.reshape(-1)
        samp_r = verts_r.reshape(-1, 3)

    ix = np.rint(samp_u).astype(np.int64)
    iy = np.rint(samp_v).astype(np.int64)
    in_bounds = (ix >= 0) & (ix < W_hm) & (iy >= 0) & (iy < H_hm)
    if not np.any(in_bounds):
        return {"mask": mask_out, "rest_xyz": rest_out}

    ix = ix[in_bounds]
    iy = iy[in_bounds]
    samp_d = samp_d[in_bounds]
    samp_r = samp_r[in_bounds]

    # Back-to-front Painter's sort: largest depth (farthest) first, smallest depth (closest) last
    order = np.argsort(samp_d, kind="mergesort")[::-1]
    iy_ord = iy[order]
    ix_ord = ix[order]
    r_ord = samp_r[order]

    mask_out[iy_ord, ix_ord] = 1.0
    rest_out[0, iy_ord, ix_ord] = r_ord[:, 0]
    rest_out[1, iy_ord, ix_ord] = r_ord[:, 1]
    rest_out[2, iy_ord, ix_ord] = r_ord[:, 2]
    return {"mask": mask_out, "rest_xyz": rest_out}


def match_dense_surface_correspondences(
    dense_views: dict[str, dict[str, np.ndarray]],
    rest_verts,
    calib: dict,
    max_points_per_view: int = 256,
    dist_thresh_rel: float = 0.04,
    mask_thresh: float = 0.60,
    max_logvar: float = 1.5,
) -> list[dict[str, dict[str, np.ndarray]]]:
    """Matches per-view predicted dense rest-pose coordinates (`rest_xyz`) to nearest rest-mesh
    vertices across all active studio views (`'front'`, `'side'`/`'left'`, `'right'`).

    Key Improvements over raw hard KDTree lookup:
      1. 3x3 morphological mask erosion (`erode_mask_3x3_numpy`) to reject silhouette bleeding;
      2. Predictive uncertainty gating via `dense_logvar` (`exp(-0.5 * logvar)`) to suppress
         symmetric/textureless or motion-blurred regions;
      3. Scale-relative rest-coordinate distance threshold (`dist < dist_thresh_rel * depth_scale`)
         instead of a hardcoded world-unit `0.04`.
    """
    raw_rest = rest_verts.get("rest_verts", []) if isinstance(rest_verts, dict) else rest_verts
    v_rest = np.asarray(raw_rest, dtype=np.float64)
    if v_rest.ndim != 2 or v_rest.shape[0] == 0 or not dense_views:
        return []

    W_panel = float(calib["panel_width"])
    H_panel = float(calib["panel_height"])
    pad_top, pad_bottom, pad_left, pad_right = dinov2_panel_padding(int(W_panel), int(H_panel))
    W_pad = W_panel + float(pad_left + pad_right)
    H_pad = H_panel + float(pad_top + pad_bottom)
    depth_scale = max(1e-4, float(calib.get("depth_scale", float(calib.get("ortho_scale", 1.0)) / 1.48)))
    max_dist_world = max(1e-5, float(dist_thresh_rel) * depth_scale)

    first_view = next(iter(dense_views.values()))
    first_mask_raw = first_view.get("mask") if "mask" in first_view else first_view.get("mask_prob")
    if first_mask_raw is None:
        return []
    first_mask = np.asarray(first_mask_raw)
    T = int(first_mask.shape[0]) if first_mask.ndim >= 3 else 1

    per_frame_matches: list[dict[str, dict[str, np.ndarray]]] = [{} for _ in range(T)]

    for raw_view_name, view_payload in dense_views.items():
        if view_payload is None or "rest_xyz" not in view_payload:
            continue
        m_raw = view_payload.get("mask") if "mask" in view_payload else view_payload.get("mask_prob")
        if m_raw is None:
            continue
        v_canon = "side" if raw_view_name == "left" else raw_view_name
        mask_arr = np.asarray(m_raw, dtype=np.float32)
        xyz_arr = np.asarray(view_payload["rest_xyz"], dtype=np.float32)
        logvar_arr = (
            np.asarray(view_payload["logvar"], dtype=np.float32)
            if "logvar" in view_payload and view_payload["logvar"] is not None
            else None
        )

        if mask_arr.ndim == 4 and mask_arr.shape[1] == 1:
            mask_arr = mask_arr[:, 0]
        elif mask_arr.ndim == 2:
            mask_arr = mask_arr[None]

        if xyz_arr.ndim == 3:
            xyz_arr = xyz_arr[None]
        if xyz_arr.shape[1] == 3 and xyz_arr.shape[-1] != 3:
            # (T, 3, H_hm, W_hm) -> (T, H_hm, W_hm, 3)
            xyz_arr = np.transpose(xyz_arr, (0, 2, 3, 1))

        if logvar_arr is not None:
            if logvar_arr.ndim == 4 and logvar_arr.shape[1] == 1:
                logvar_arr = logvar_arr[:, 0]
            elif logvar_arr.ndim == 2:
                logvar_arr = logvar_arr[None]

        _, H_hm, W_hm = mask_arr.shape
        eroded_fg = erode_mask_3x3_numpy(mask_arr >= float(mask_thresh))
        # Fall back to non-eroded foreground if thin structures (e.g. tendrils/digits) were completely eroded
        for t in range(T):
            fg_t = eroded_fg[t]
            if int( np.sum(fg_t) ) < 4:
                fg_t = mask_arr[t] >= float(mask_thresh)
            if logvar_arr is not None:
                fg_t = fg_t & (logvar_arr[t] <= float(max_logvar))

            ys, xs = np.nonzero(fg_t)
            if len(xs) == 0:
                continue

            m_probs = np.clip(mask_arr[t, ys, xs], 0.01, 1.0)
            if logvar_arr is not None:
                lv_vals = np.clip(logvar_arr[t, ys, xs], -6.0, 3.0)
                cert = m_probs * np.exp(-0.5 * lv_vals)
            else:
                cert = m_probs

            if len(xs) > max_points_per_view:
                # Deterministic top-certainty + spatial stride selection
                top_pool = min(len(xs), max_points_per_view * 3)
                top_idx = np.argsort(cert)[::-1][:top_pool]
                stride = np.linspace(0, len(top_idx) - 1, max_points_per_view, dtype=np.int64)
                sel = top_idx[stride]
                ys, xs, cert = ys[sel], xs[sel], cert[sel]

            # Convert heatmap grid coordinates (xs, ys) to unpadded panel pixel coordinates (u_px, v_px)
            u_norm = xs.astype(np.float64) / float(max(1, W_hm - 1))
            v_norm = ys.astype(np.float64) / float(max(1, H_hm - 1))
            u_px = u_norm * W_pad - float(pad_left)
            v_px = v_norm * H_pad - float(pad_top)

            pred_norm = xyz_arr[t, ys, xs]  # (P, 3)
            pred_rest_3d = denormalize_rest_vertices_numpy(pred_norm, calib).astype(np.float64)

            # Exact nearest-neighbor vertex lookup in rest-pose 3D space: (P, V) squared distances
            diff = pred_rest_3d[:, None, :] - v_rest[None, :, :]
            dist_sq = np.sum(diff * diff, axis=-1)
            nearest_idx = np.argmin(dist_sq, axis=1)
            nearest_dist = np.sqrt(np.maximum(0.0, dist_sq[np.arange(len(nearest_idx)), nearest_idx]))

            keep = nearest_dist <= max_dist_world
            if not np.any(keep):
                continue

            vert_indices = nearest_idx[keep].astype(np.int64)
            target_uv = np.stack([u_px[keep], v_px[keep]], axis=-1).astype(np.float64)
            d_keep = nearest_dist[keep]
            proximity_w = np.exp(-0.5 * (d_keep / (0.5 * max_dist_world)) ** 2)
            weights = np.clip(cert[keep].astype(np.float64) * proximity_w, 0.01, 1.0)

            per_frame_matches[t][v_canon] = {
                "vert_indices": vert_indices,
                "target_uv": target_uv,
                "weights": weights,
                "pred_rest_3d": pred_rest_3d[keep],
                "match_dist": d_keep,
            }

    return per_frame_matches
