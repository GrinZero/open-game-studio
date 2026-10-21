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
    classify_rig_bones,
    clean_uv_short_gaps_hampel,
    reconstruct_frames_with_kind_solvers,
    savgol_smooth_array,
    whittaker_smooth_trajectory,
)
from pose_features import fuse_depth_visibility  # noqa: E402

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

    # Stage 2: Bounded short-gap Hampel cleaning + Dual-ortho epipolar V-consistency gating
    uv_f_clean = clean_uv_short_gaps_hampel(uv_f_raw, scale=scale_2k, max_gap=3)
    uv_s_clean = clean_uv_short_gaps_hampel(uv_s_raw, scale=scale_2k, max_gap=3)

    replaced_f_mask = np.linalg.norm(uv_f_clean - uv_f_raw, axis=-1) > 1e-5
    replaced_s_mask = np.linalg.norm(uv_s_clean - uv_s_raw, axis=-1) > 1e-5

    epipolar_dv_px = np.abs(uv_f_clean[..., 1] - uv_s_clean[..., 1])
    epipolar_gate = np.exp(-0.5 * (epipolar_dv_px / (28.0 * scale_2k)) ** 2)
    cf_gated = np.clip(cf_raw * (0.35 + 0.65 * epipolar_gate), 0.01, 1.0)
    cs_gated = np.clip(cs_raw * (0.35 + 0.65 * epipolar_gate), 0.01, 1.0)

    stage2_cleaned_2d = {
        "uv_front_clean": uv_f_clean,
        "uv_side_clean": uv_s_clean,
        "conf_front_gated": cf_gated,
        "conf_side_gated": cs_gated,
        "epipolar_dv_px": epipolar_dv_px,
        "epipolar_gate": epipolar_gate,
        "replaced_front_mask": replaced_f_mask,
        "replaced_side_mask": replaced_s_mask,
    }

    # Stage 3: 3D Triangulation & Savitzky-Golay Smoothing
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

    # Stage 4: Layer 2 + Layer 3 Kinematic Reconstruction
    frames_pred, stage4_diag = reconstruct_frames_with_kind_solvers(
        pts_3d=pts_3d_smoothed,
        trust=trust_raw,
        bone_names=bone_names,
        classified_bones=classified_bones,
        depth_scale=depth_scale,
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
