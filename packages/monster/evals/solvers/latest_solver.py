#!/usr/bin/env python3
"""latest_solver.py — Adapter for the latest Monster Studio 3-layer kinematic solver (`taxonomy_kind_v2`).

Executes:
  - Stage 1: Raw 2D & Aux Observations unpacking (`observations.npz`)
  - Stage 2: Bounded short-gap (`<= 3` frames) Hampel 2D cleaning + dual-ortho epipolar V-consistency gating
  - Stage 3: Depth/visibility weighted 3D triangulation (`fuse_depth_visibility`) + 5-tap quadratic Savitzky-Golay smoothing
  - Stage 4: Layer 1-3 Hybrid Bone Taxonomy Classification + Multi-anchor IRLS Kabsch root fit + Kind-Specific FK Solvers
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MONSTER_ROOT = Path(__file__).resolve().parents[2]
WORKER_DIR = MONSTER_ROOT / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from bone_taxonomy import (  # noqa: E402
    BONE_TAXONOMY_SCHEMA,
    apply_joint_limits_profile,
    classify_rig_bones,
    clean_uv_short_gaps_hampel,
    reconstruct_frames_with_kind_solvers,
    savgol_smooth_array,
    whittaker_smooth_trajectory,
)
from pose_features import (  # noqa: E402
    calibrate_view_scale_drift,
    dense_surface_enabled,
    fuse_depth_visibility,
    match_dense_surface_correspondences,
    mocap_priors_enabled,
)

LATEST_SOLVER_NAME = "taxonomy_kind_v2"


@dataclass
class SolverExecutionResult:
    solver_name: str
    schema: str
    bone_names: list[str]
    classified_bones: dict[str, dict[str, Any]]
    calib: dict[str, Any]
    stage1_raw_obs: dict[str, np.ndarray]
    stage2_cleaned_2d: dict[str, np.ndarray]
    stage3_triangulation: dict[str, np.ndarray]
    stage4_frames: list[dict[str, Any]]
    stage4_diagnostics: dict[str, Any]
    elapsed_ms: float


def run_latest_solver(
    observations: dict[str, np.ndarray],
    studio_rig_meta: dict[str, Any],
    constraints_meta: dict[str, Any] | None = None,
) -> SolverExecutionResult:
    """Runs the latest 3-layer kinematic solver (`taxonomy_kind_v2`) and retains intermediate
    stage tensors for full 5-stage attribution and per-frame diagnosis.
    """
    t0 = time.perf_counter()

    bones_raw = studio_rig_meta.get("bones", {})
    bone_names = sorted(bones_raw.keys())
    calib = dict(studio_rig_meta.get("calibration", {}))
    panel_h = float(calib.get("panel_height", 1440.0))
    depth_scale = float(calib.get("depth_scale", 1.0))
    scale_2k = panel_h / 768.0

    # Layer 1: Classify bones into canonical kinds & sub-roles
    classified_bones = classify_rig_bones(
        bones_raw,
        bbox_min=studio_rig_meta.get("bbox_min"),
        bbox_max=studio_rig_meta.get("bbox_max"),
        constraints_meta=constraints_meta,
    )
    joint_limits_profile = studio_rig_meta.get("joint_limits_profile")
    if isinstance(joint_limits_profile, dict):
        classified_bones = apply_joint_limits_profile(
            classified_bones,
            joint_limits_profile=joint_limits_profile,
            bbox_min=studio_rig_meta.get("bbox_min"),
            bbox_max=studio_rig_meta.get("bbox_max"),
        )

    # Stage 1: Raw 2D & Aux Observations
    uv_f_raw = np.asarray(observations["uv_front_all"], dtype=np.float64)
    uv_s_raw = np.asarray(observations["uv_side_all"], dtype=np.float64)
    cf_raw = np.asarray(observations["conf_front_all"], dtype=np.float64)
    cs_raw = np.asarray(observations["conf_side_all"], dtype=np.float64)
    df_raw = np.asarray(observations["depth_front_all"], dtype=np.float64)
    ds_raw = np.asarray(observations["depth_side_all"], dtype=np.float64)
    vf_raw = np.asarray(observations["visibility_front_all"], dtype=np.float64)
    vs_raw = np.asarray(observations["visibility_side_all"], dtype=np.float64)
    lvf_raw = np.asarray(observations["logvar_front_all"], dtype=np.float64)
    lvs_raw = np.asarray(observations["logvar_side_all"], dtype=np.float64)
    has_right = "uv_right_all" in observations and "conf_right_all" in observations

    stage1_raw_obs = {
        "uv_front": uv_f_raw,
        "uv_side": uv_s_raw,
        "conf_front": cf_raw,
        "conf_side": cs_raw,
        "depth_front": df_raw,
        "depth_side": ds_raw,
        "visibility_front": vf_raw,
        "visibility_side": vs_raw,
        "logvar_front": lvf_raw,
        "logvar_side": lvs_raw,
    }
    if has_right:
        uv_r_raw = np.asarray(observations["uv_right_all"], dtype=np.float64)
        cr_raw = np.asarray(observations["conf_right_all"], dtype=np.float64)
        dr_raw = np.asarray(observations.get("depth_right_all", np.zeros_like(ds_raw)), dtype=np.float64)
        vr_raw = np.asarray(observations.get("visibility_right_all", np.ones_like(vs_raw)), dtype=np.float64)
        lvr_raw = np.asarray(observations.get("logvar_right_all", np.zeros_like(lvs_raw)), dtype=np.float64)
        stage1_raw_obs.update({
            "uv_right": uv_r_raw,
            "conf_right": cr_raw,
            "depth_right": dr_raw,
            "visibility_right": vr_raw,
            "logvar_right": lvr_raw,
        })
    else:
        uv_r_raw = cr_raw = dr_raw = vr_raw = lvr_raw = None

    # Stage 2: Bounded short-gap Hampel cleaning + View Scale/Drift Calibration + Dual/Tri-ortho epipolar gating
    uv_f_hampel = clean_uv_short_gaps_hampel(uv_f_raw, scale=scale_2k, max_gap=3)
    uv_s_clean = clean_uv_short_gaps_hampel(uv_s_raw, scale=scale_2k, max_gap=3)
    uv_r_hampel = (
        clean_uv_short_gaps_hampel(uv_r_raw, scale=scale_2k, max_gap=3)
        if (has_right and uv_r_raw is not None)
        else None
    )

    replaced_f_mask = np.linalg.norm(uv_f_hampel - uv_f_raw, axis=-1) > 1e-5
    replaced_s_mask = np.linalg.norm(uv_s_clean - uv_s_raw, axis=-1) > 1e-5

    if mocap_priors_enabled():
        uv_f_clean, uv_r_clean, drift_diag = calibrate_view_scale_drift(
            uv_front=uv_f_hampel,
            uv_side=uv_s_clean,
            conf_front=cf_raw,
            conf_side=cs_raw,
            calib=calib,
            uv_right=uv_r_hampel,
            conf_right=cr_raw,
            vis_front=vf_raw,
            vis_side=vs_raw,
            vis_right=vr_raw,
        )
    else:
        uv_f_clean = uv_f_hampel
        uv_r_clean = uv_r_hampel
        drift_diag = {
            "view_drift_calibrated": False,
            "max_abs_dy_px": 0.0,
            "mean_abs_dy_px": 0.0,
            "max_abs_scale_pct": 0.0,
        }

    epipolar_dv_px = np.abs(uv_f_clean[..., 1] - uv_s_clean[..., 1])
    epipolar_gate = np.exp(-0.5 * (epipolar_dv_px / (28.0 * scale_2k)) ** 2)
    cs_gated = np.clip(cs_raw * (0.35 + 0.65 * epipolar_gate), 0.01, 1.0)
    if has_right and uv_r_raw is not None and cr_raw is not None and uv_r_clean is not None and uv_r_hampel is not None:
        replaced_r_mask = np.linalg.norm(uv_r_hampel - uv_r_raw, axis=-1) > 1e-5
        epipolar_dv_fr_px = np.abs(uv_f_clean[..., 1] - uv_r_clean[..., 1])
        epipolar_gate_fr = np.exp(-0.5 * (epipolar_dv_fr_px / (28.0 * scale_2k)) ** 2)
        cr_gated = np.clip(cr_raw * (0.35 + 0.65 * epipolar_gate_fr), 0.01, 1.0)
        epipolar_gate_f = np.maximum(epipolar_gate, epipolar_gate_fr)
        cf_gated = np.clip(cf_raw * (0.35 + 0.65 * epipolar_gate_f), 0.01, 1.0)
    else:
        uv_r_clean = cr_gated = replaced_r_mask = epipolar_dv_fr_px = epipolar_gate_fr = None
        cf_gated = np.clip(cf_raw * (0.35 + 0.65 * epipolar_gate), 0.01, 1.0)

    stage2_cleaned_2d = {
        "uv_front_clean": uv_f_clean,
        "uv_side_clean": uv_s_clean,
        "conf_front_gated": cf_gated,
        "conf_side_gated": cs_gated,
        "epipolar_dv_px": epipolar_dv_px,
        "epipolar_gate": epipolar_gate,
        "replaced_front_mask": replaced_f_mask,
        "replaced_side_mask": replaced_s_mask,
        **drift_diag,
    }
    if has_right and uv_r_clean is not None and cr_gated is not None:
        stage2_cleaned_2d.update({
            "uv_right_clean": uv_r_clean,
            "conf_right_gated": cr_gated,
            "epipolar_dv_fr_px": epipolar_dv_fr_px,
            "epipolar_gate_fr": epipolar_gate_fr,
            "replaced_right_mask": replaced_r_mask,
        })

    # Stage 3: 3D Triangulation & Whittaker Smoothing
    pts_3d_raw, trust_raw = fuse_depth_visibility(
        uv_f_clean,
        uv_s_clean,
        cf_gated,
        cs_gated,
        calib,
        df_raw,
        ds_raw,
        vf_raw,
        vs_raw,
        lvf_raw,
        lvs_raw,
        uv_right=uv_r_clean,
        conf_right=cr_gated,
        depth_right=dr_raw,
        visibility_right=vr_raw,
        logvar_right=lvr_raw,
    )
    pts_3d_raw = np.asarray(pts_3d_raw, dtype=np.float64)
    trust_raw = np.asarray(trust_raw, dtype=np.float64)
    pts_3d_smoothed = whittaker_smooth_trajectory(
        pts_3d_raw, weights=trust_raw ** 2, lam_accel=4.0, lam_jerk=10.0
    )

    stage3_triangulation = {
        "pts_3d_raw": pts_3d_raw,
        "trust_raw": trust_raw,
        "pts_3d_smoothed": pts_3d_smoothed,
    }

    # Stage 4: Layer 2 + Layer 3 Kinematic Reconstruction (+ optional Dense Surface LBS Refinement)
    rig_skinning = studio_rig_meta.get("rig_skinning") or observations.get("rig_skinning")
    dense_corrs = observations.get("dense_correspondences")
    if dense_corrs is None and dense_surface_enabled() and rig_skinning is not None and "dense_views_all" in observations:
        dv_all = observations["dense_views_all"]
        if isinstance(dv_all, dict):
            dense_corrs = match_dense_surface_correspondences(dv_all, rig_skinning, calib)
        else:
            dense_corrs = [
                match_dense_surface_correspondences(dv, rig_skinning, calib)[0]
                for dv in dv_all
            ]
    frames_pred, stage4_diag = reconstruct_frames_with_kind_solvers(
        pts_3d=pts_3d_smoothed,
        trust=trust_raw,
        bone_names=bone_names,
        classified_bones=classified_bones,
        depth_scale=depth_scale,
        dense_correspondences=dense_corrs if dense_surface_enabled() else None,
        rig_skinning=rig_skinning if dense_surface_enabled() else None,
        calib=calib,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return SolverExecutionResult(
        solver_name=LATEST_SOLVER_NAME,
        schema=BONE_TAXONOMY_SCHEMA,
        bone_names=bone_names,
        classified_bones=classified_bones,
        calib=calib,
        stage1_raw_obs=stage1_raw_obs,
        stage2_cleaned_2d=stage2_cleaned_2d,
        stage3_triangulation=stage3_triangulation,
        stage4_frames=frames_pred,
        stage4_diagnostics=stage4_diag,
        elapsed_ms=round(elapsed_ms, 2),
    )
