#!/usr/bin/env python3
"""evaluator.py — 5-Stage Attribution, Stratified Kinematics, Temporal Quality, and Golden-Frame Evaluator."""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MONSTER_ROOT = Path(__file__).resolve().parents[2]
WORKER_DIR = MONSTER_ROOT / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from bone_taxonomy import compute_stratified_motion_metrics  # noqa: E402
from evals.solvers.latest_solver import SolverExecutionResult  # noqa: E402


def _project_3d_to_uv_numpy(pts_3d: np.ndarray, calib: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Projects `(..., 3)` world points to `(uv_front, uv_side)` pixel coordinates using studio `calib`."""
    cx, cy, cz = [float(x) for x in calib["center"]]
    W = float(calib["panel_width"])
    H = float(calib["panel_height"])
    S = float(calib["ortho_scale"])
    aspect = W / H

    u_f = ((pts_3d[..., 0] - cx) / (S * aspect) + 0.5) * W
    v_f = (0.5 - (pts_3d[..., 2] - cz) / S) * H
    u_s = ((pts_3d[..., 1] - cy) / (S * aspect) + 0.5) * W
    v_s = v_f.copy()

    uv_front = np.stack([u_f, v_f], axis=-1)
    uv_side = np.stack([u_s, v_s], axis=-1)
    return uv_front, uv_side


def extract_ground_truth_tensors(
    frames_gt: list[dict[str, Any]],
    bone_names: list[str],
    calib: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Extracts dense NumPy tensors from `exam_gt["frames"]` for fast vectorized stage evaluation."""
    T = len(frames_gt)
    N = len(bone_names)
    K = 2 * N

    gt_pts_3d = np.zeros((T, K, 3), dtype=np.float64)
    gt_uv_front = np.zeros((T, K, 2), dtype=np.float64)
    gt_uv_side = np.zeros((T, K, 2), dtype=np.float64)
    gt_rot_world = np.zeros((T, N, 3, 3), dtype=np.float64)
    has_explicit_uv = True

    for t, frame_item in enumerate(frames_gt):
        bmap = frame_item.get("bones", frame_item.get("joints", {}))
        for i, b in enumerate(bone_names):
            j = bmap[b]
            h3 = np.asarray(j["head_3d"], dtype=np.float64)
            t3 = np.asarray(j["tail_3d"], dtype=np.float64)
            gt_pts_3d[t, 2 * i] = h3
            gt_pts_3d[t, 2 * i + 1] = t3
            if "rot_mat_world" in j:
                gt_rot_world[t, i] = np.asarray(j["rot_mat_world"], dtype=np.float64)
            if "front_uv_head" in j and "side_uv_head" in j and "front_uv_tail" in j and "side_uv_tail" in j:
                gt_uv_front[t, 2 * i] = np.asarray(j["front_uv_head"], dtype=np.float64)
                gt_uv_front[t, 2 * i + 1] = np.asarray(j["front_uv_tail"], dtype=np.float64)
                gt_uv_side[t, 2 * i] = np.asarray(j["side_uv_head"], dtype=np.float64)
                gt_uv_side[t, 2 * i + 1] = np.asarray(j["side_uv_tail"], dtype=np.float64)
            else:
                has_explicit_uv = False

    if not has_explicit_uv:
        gt_uv_front, gt_uv_side = _project_3d_to_uv_numpy(gt_pts_3d, calib)

    cx, cy, _ = [float(x) for x in calib["center"]]
    scale = max(1e-6, float(calib.get("depth_scale", 1.0)))
    gt_depth_front = (gt_pts_3d[..., 1] - cy) / scale
    gt_depth_side = (cx - gt_pts_3d[..., 0]) / scale

    return {
        "pts_3d": gt_pts_3d,
        "uv_front": gt_uv_front,
        "uv_side": gt_uv_side,
        "depth_front": gt_depth_front,
        "depth_side": gt_depth_side,
        "rot_world": gt_rot_world,
    }


def extract_predicted_fk_tensors(
    frames_pred: list[dict[str, Any]],
    bone_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts `(pred_pts_3d, pred_rot_world)` of shapes `((T, 2*N, 3), (T, N, 3, 3))` from solved FK frames."""
    T = len(frames_pred)
    N = len(bone_names)
    pts = np.zeros((T, 2 * N, 3), dtype=np.float64)
    rots = np.zeros((T, N, 3, 3), dtype=np.float64)
    for t, f_item in enumerate(frames_pred):
        bmap = f_item.get("bones", f_item.get("joints", {}))
        for i, b in enumerate(bone_names):
            b_data = bmap[b]
            pts[t, 2 * i] = np.asarray(b_data["head_3d"], dtype=np.float64)
            pts[t, 2 * i + 1] = np.asarray(b_data["tail_3d"], dtype=np.float64)
            rots[t, i] = np.asarray(b_data["rot_mat_world"], dtype=np.float64)
    return pts, rots


def _bone_length_cv_pct(pts_3d: np.ndarray, bone_names: list[str]) -> float:
    """Computes mean coefficient of variation (`std / mean * 100`) of bone lengths across time."""
    cvs: list[float] = []
    for i in range(len(bone_names)):
        seg = pts_3d[:, 2 * i + 1] - pts_3d[:, 2 * i]
        lengths = np.linalg.norm(seg, axis=-1)
        mean_l = float(np.mean(lengths))
        if mean_l > 1e-5:
            cvs.append(float(np.std(lengths) / mean_l * 100.0))
    return round(float(np.mean(cvs)) if cvs else 0.0, 4)


def _accel_magnitude_mean(pts_3d: np.ndarray) -> float:
    """Computes mean second-order finite-difference acceleration `||p(t+1) - 2p(t) + p(t-1)||`."""
    if pts_3d.shape[0] < 3:
        return 0.0
    acc = pts_3d[2:] - 2.0 * pts_3d[1:-1] + pts_3d[:-2]
    return float(np.mean(np.linalg.norm(acc, axis=-1)))


def _estimate_primary_lag_frames(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    classified_bones: dict[str, dict[str, Any]],
    bone_names: list[str],
    max_lag: int = 8,
) -> float:
    """Estimates phase lag (in frames) between predicted and GT vertical Z trajectories on primary limb bones."""
    limb_indices = [
        2 * i + 1
        for i, b in enumerate(bone_names)
        if classified_bones[b].get("kind") in ("limb", "head", "spine")
    ]
    if not limb_indices or pred_pts.shape[0] < (2 * max_lag + 5):
        return 0.0

    lags: list[int] = []
    for k in limb_indices:
        s_pred = pred_pts[:, k, 2] - np.mean(pred_pts[:, k, 2])
        s_gt = gt_pts[:, k, 2] - np.mean(gt_pts[:, k, 2])
        std_p = float(np.std(s_pred))
        std_g = float(np.std(s_gt))
        if std_p < 1e-4 or std_g < 1e-4:
            continue
        best_lag = 0
        best_corr = -2.0
        T = len(s_pred)
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                a, b = s_pred, s_gt
            elif lag > 0:
                # Positive lag means pred trails GT by `lag` frames: pred[t + lag] ~ gt[t]
                a, b = s_pred[lag:], s_gt[: T - lag]
            else:
                a, b = s_pred[: T + lag], s_gt[-lag:]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom > 1e-8:
                corr = float(np.dot(a, b) / denom)
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag
        lags.append(best_lag)
    return round(float(np.mean(lags)) if lags else 0.0, 3)


def evaluate_golden_frames(
    solver_res: SolverExecutionResult,
    frames_gt: list[dict[str, Any]],
    golden_spec: dict[str, Any],
) -> dict[str, Any]:
    """Evaluates deterministic golden-frame regression probes (e.g., Frame 17 root flip check)."""
    probes = golden_spec.get("probes", [])
    bone_names = solver_res.bone_names
    classified = solver_res.classified_bones
    primary_bones = [b for b in bone_names if classified[b].get("is_primary", True)]
    root_bones = [b for b in bone_names if classified[b].get("kind") == "root"]
    root_name = root_bones[0] if root_bones else bone_names[0]

    results: list[dict[str, Any]] = []
    all_passed = True

    for probe in probes:
        f_idx = int(probe["frame"])
        name = str(probe.get("name", f"frame_{f_idx}"))
        desc = str(probe.get("description", ""))
        max_root_rot_deg = float(probe.get("max_root_rot_deg", 12.0))
        max_primary_dir_deg = float(probe.get("max_primary_dir_deg", 15.0))

        pr_bmap = solver_res.stage4_frames[f_idx]["bones"]
        gt_bmap = frames_gt[f_idx].get("bones", frames_gt[f_idx].get("joints", {}))

        # Root rotation error
        R_p_root = np.asarray(pr_bmap[root_name]["rot_mat_world"], dtype=np.float64)
        R_g_root = np.asarray(gt_bmap[root_name]["rot_mat_world"], dtype=np.float64)
        tr_root = float(np.clip((np.trace(R_g_root.T @ R_p_root) - 1.0) * 0.5, -1.0, 1.0))
        root_rot_err_deg = math.degrees(math.acos(tr_root))

        # Root step from previous frame (detect sudden 180-deg flip)
        root_step_deg = 0.0
        if f_idx > 0:
            R_prev = np.asarray(solver_res.stage4_frames[f_idx - 1]["bones"][root_name]["rot_mat_world"], dtype=np.float64)
            tr_step = float(np.clip((np.trace(R_prev.T @ R_p_root) - 1.0) * 0.5, -1.0, 1.0))
            root_step_deg = math.degrees(math.acos(tr_step))

        # Primary bones mean direction error at this frame
        dir_errs: list[float] = []
        for b in primary_bones:
            hp = np.asarray(pr_bmap[b]["head_3d"], dtype=np.float64)
            tp = np.asarray(pr_bmap[b]["tail_3d"], dtype=np.float64)
            hg = np.asarray(gt_bmap[b]["head_3d"], dtype=np.float64)
            tg = np.asarray(gt_bmap[b]["tail_3d"], dtype=np.float64)
            vp, vg = tp - hp, tg - hg
            np_l, ng_l = float(np.linalg.norm(vp)), float(np.linalg.norm(vg))
            if np_l > 1e-6 and ng_l > 1e-6:
                c = float(np.clip(np.dot(vp / np_l, vg / ng_l), -1.0, 1.0))
                dir_errs.append(math.degrees(math.acos(c)))

        primary_dir_mean_deg = float(np.mean(dir_errs)) if dir_errs else 0.0
        passed = (root_rot_err_deg <= max_root_rot_deg) and (primary_dir_mean_deg <= max_primary_dir_deg)
        if not passed:
            all_passed = False

        results.append(
            {
                "frame": f_idx,
                "name": name,
                "description": desc,
                "passed": passed,
                "root_rot_err_deg": round(root_rot_err_deg, 3),
                "root_single_frame_step_deg": round(root_step_deg, 3),
                "primary_dir_mean_deg": round(primary_dir_mean_deg, 3),
                "thresholds": {
                    "max_root_rot_deg": max_root_rot_deg,
                    "max_primary_dir_deg": max_primary_dir_deg,
                },
            }
        )

    return {
        "all_passed": all_passed,
        "probe_count": len(results),
        "probes": results,
    }


def evaluate_full_pipeline(
    benchmark_id: str,
    tier: str,
    solver_res: SolverExecutionResult,
    exam_gt: dict[str, Any],
    golden_spec: dict[str, Any],
    bake_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Computes the complete `monster-eval-report-v1` evaluation report across Stages 1-5."""
    frames_gt = exam_gt["frames"]
    bone_names = solver_res.bone_names
    classified = solver_res.classified_bones
    calib = solver_res.calib

    gt_tensors = extract_ground_truth_tensors(frames_gt, bone_names, calib)
    fk_pts_3d, fk_rot_world = extract_predicted_fk_tensors(solver_res.stage4_frames, bone_names)

    # ------------------------------------------------------------------
    # Stage 1: 2D Perception & Aux Head Accuracy
    # ------------------------------------------------------------------
    uv_f_raw = solver_res.stage1_raw_obs["uv_front"]
    uv_s_raw = solver_res.stage1_raw_obs["uv_side"]
    err_f_raw = np.linalg.norm(uv_f_raw - gt_tensors["uv_front"], axis=-1)
    err_s_raw = np.linalg.norm(uv_s_raw - gt_tensors["uv_side"], axis=-1)
    err_uv_all = np.concatenate([err_f_raw.ravel(), err_s_raw.ravel()])

    depth_f_err = np.abs(solver_res.stage1_raw_obs["depth_front"] - gt_tensors["depth_front"])
    depth_s_err = np.abs(solver_res.stage1_raw_obs["depth_side"] - gt_tensors["depth_side"])

    stage1_metrics = {
        "front_uv_mae_px": round(float(np.mean(err_f_raw)), 3),
        "side_uv_mae_px": round(float(np.mean(err_s_raw)), 3),
        "overall_uv_mae_px": round(float(np.mean(err_uv_all)), 3),
        "uv_p95_px": round(float(np.percentile(err_uv_all, 95)), 3),
        "pck_at_8px": round(float(np.mean(err_uv_all <= 8.0) * 100.0), 2),
        "pck_at_16px": round(float(np.mean(err_uv_all <= 16.0) * 100.0), 2),
        "pck_at_32px": round(float(np.mean(err_uv_all <= 32.0) * 100.0), 2),
        "depth_front_mae": round(float(np.mean(depth_f_err)), 4),
        "depth_side_mae": round(float(np.mean(depth_s_err)), 4),
        "depth_overall_mae": round(float(0.5 * (np.mean(depth_f_err) + np.mean(depth_s_err))), 4),
        "mean_confidence_front": round(float(np.mean(solver_res.stage1_raw_obs["conf_front"])), 4),
        "mean_confidence_side": round(float(np.mean(solver_res.stage1_raw_obs["conf_side"])), 4),
    }

    # ------------------------------------------------------------------
    # Stage 2: Short-Gap Hampel Cleaning & Epipolar Gating
    # ------------------------------------------------------------------
    uv_f_clean = solver_res.stage2_cleaned_2d["uv_front_clean"]
    uv_s_clean = solver_res.stage2_cleaned_2d["uv_side_clean"]
    err_f_clean = np.linalg.norm(uv_f_clean - gt_tensors["uv_front"], axis=-1)
    err_s_clean = np.linalg.norm(uv_s_clean - gt_tensors["uv_side"], axis=-1)
    err_uv_clean = np.concatenate([err_f_clean.ravel(), err_s_clean.ravel()])

    epipolar_raw_px = float(np.mean(np.abs(uv_f_raw[..., 1] - uv_s_raw[..., 1])))
    epipolar_clean_px = float(np.mean(solver_res.stage2_cleaned_2d["epipolar_dv_px"]))

    stage2_metrics = {
        "cleaned_overall_uv_mae_px": round(float(np.mean(err_uv_clean)), 3),
        "uv_mae_improvement_px": round(float(np.mean(err_uv_all) - np.mean(err_uv_clean)), 4),
        "epipolar_v_consistency_raw_px": round(epipolar_raw_px, 3),
        "epipolar_v_consistency_clean_px": round(epipolar_clean_px, 3),
        "hampel_replaced_ratio_front_pct": round(
            float(np.mean(solver_res.stage2_cleaned_2d["replaced_front_mask"]) * 100.0), 3
        ),
        "hampel_replaced_ratio_side_pct": round(
            float(np.mean(solver_res.stage2_cleaned_2d["replaced_side_mask"]) * 100.0), 3
        ),
        "short_gap_max_frames": int(solver_res.stage4_diagnostics.get("short_gap_max_frames", 3)),
        "cross_frame_confidence_replacement": bool(
            solver_res.stage4_diagnostics.get("cross_frame_confidence_replacement", False)
        ),
    }

    # ------------------------------------------------------------------
    # Stage 3: 3D Triangulation & Savitzky-Golay Filtering
    # ------------------------------------------------------------------
    pts_3d_raw = solver_res.stage3_triangulation["pts_3d_raw"]
    pts_3d_sg = solver_res.stage3_triangulation["pts_3d_smoothed"]
    mpjpe_raw = float(np.mean(np.linalg.norm(pts_3d_raw - gt_tensors["pts_3d"], axis=-1)))
    mpjpe_sg = float(np.mean(np.linalg.norm(pts_3d_sg - gt_tensors["pts_3d"], axis=-1)))
    acc_raw = _accel_magnitude_mean(pts_3d_raw)
    acc_sg = _accel_magnitude_mean(pts_3d_sg)

    stage3_metrics = {
        "raw_triangulated_mpjpe": round(mpjpe_raw, 5),
        "sg_smoothed_mpjpe": round(mpjpe_sg, 5),
        "raw_bone_length_cv_pct": _bone_length_cv_pct(pts_3d_raw, bone_names),
        "sg_bone_length_cv_pct": _bone_length_cv_pct(pts_3d_sg, bone_names),
        "raw_accel_jitter": round(acc_raw, 5),
        "sg_accel_jitter": round(acc_sg, 5),
        "sg_jitter_reduction_pct": round((1.0 - acc_sg / max(1e-9, acc_raw)) * 100.0, 2),
    }

    # ------------------------------------------------------------------
    # Stage 4: Layer 2 + Layer 3 Kinematic FK Reconstruction
    # ------------------------------------------------------------------
    mpjpe_fk = float(np.mean(np.linalg.norm(fk_pts_3d - gt_tensors["pts_3d"], axis=-1)))

    # Check exact rest bone length conservation
    max_len_drift = 0.0
    for i, b in enumerate(bone_names):
        rest_l = float(classified[b].get("rest_length", 0.0))
        if rest_l > 0.0:
            seg_l = np.linalg.norm(fk_pts_3d[:, 2 * i + 1] - fk_pts_3d[:, 2 * i], axis=-1)
            max_len_drift = max(max_len_drift, float(np.max(np.abs(seg_l - rest_l))))

    # Check rigid bone local drift (`fixed_local=True`)
    rigid_drift_max_deg = 0.0
    for i, b in enumerate(bone_names):
        if not classified[b].get("fixed_local", False):
            continue
        p = classified[b].get("parent")
        if p and p in bone_names:
            p_idx = bone_names.index(p)
            R_rest_b = np.asarray(classified[b]["rest_rot_mat_world"], dtype=np.float64)
            R_rest_p = np.asarray(classified[p]["rest_rot_mat_world"], dtype=np.float64)
            R_rel_rest = R_rest_p.T @ R_rest_b
            for t in range(fk_rot_world.shape[0]):
                R_init = fk_rot_world[t, p_idx] @ R_rel_rest
                R_delta = R_init.T @ fk_rot_world[t, i]
                # Use chordal-to-geodesic identity 2*arcsin(||R_delta - I||_F / sqrt(8)) to avoid arccos(1-eps) precision loss
                frob = float(np.linalg.norm(R_delta - np.eye(3, dtype=np.float64)))
                ang_deg = math.degrees(2.0 * math.asin(min(1.0, frob / math.sqrt(8.0))))
                rigid_drift_max_deg = max(rigid_drift_max_deg, ang_deg)

    stage4_metrics = {
        "fk_mpjpe": round(mpjpe_fk, 5),
        "fk_bone_length_cv_pct": _bone_length_cv_pct(fk_pts_3d, bone_names),
        "bone_length_drift_max": round(max_len_drift, 6),
        "joint_limit_max_violation_deg": round(
            math.degrees(float(solver_res.stage4_diagnostics.get("final_limit_max_violation_rad", 0.0))), 5
        ),
        "rigid_local_drift_max_deg": round(rigid_drift_max_deg, 5),
        "root_anchors_count": len(solver_res.stage4_diagnostics.get("root_anchors", [])),
        "solver_elapsed_ms": solver_res.elapsed_ms,
    }

    # ------------------------------------------------------------------
    # Stage 5: GLB Export & Skinning Roundtrip (Tier 3)
    # ------------------------------------------------------------------
    if bake_report is not None:
        stage5_metrics = {
            "executed": True,
            "valid": bool(bake_report.get("valid", False)),
            "skin_roundtrip_max": round(float(bake_report.get("skin_roundtrip_max", 0.0)), 6),
            "source_unchanged": bool(bake_report.get("source_unchanged", True)),
            "glb_sha256": bake_report.get("glb_sha256"),
        }
    else:
        stage5_metrics = {
            "executed": False,
            "valid": True,
            "skin_roundtrip_max": None,
            "source_unchanged": True,
            "glb_sha256": None,
        }

    # ------------------------------------------------------------------
    # Stratified Kinematics (All Bones, Primary Bones, By-Kind, Per-Bone, Worst Frames)
    # ------------------------------------------------------------------
    stratified = compute_stratified_motion_metrics(
        frames_pred=solver_res.stage4_frames,
        frames_gt=frames_gt,
        bones_meta=classified,
    )

    primary_mask = [i for i, b in enumerate(bone_names) if classified[b].get("is_primary", True)]
    per_frame_rows: list[dict[str, Any]] = []
    for t in range(len(frames_gt)):
        vec_p = fk_pts_3d[t, 1::2] - fk_pts_3d[t, 0::2]
        vec_g = gt_tensors["pts_3d"][t, 1::2] - gt_tensors["pts_3d"][t, 0::2]
        len_p = np.maximum(1e-8, np.linalg.norm(vec_p, axis=-1))
        len_g = np.maximum(1e-8, np.linalg.norm(vec_g, axis=-1))
        cos_all = np.clip(np.sum((vec_p / len_p[:, None]) * (vec_g / len_g[:, None]), axis=-1), -1.0, 1.0)
        deg_all = np.degrees(np.arccos(cos_all))
        per_frame_rows.append(
            {
                "frame": t,
                "all_direction_mean_deg": round(float(np.mean(deg_all)), 3),
                "primary_direction_mean_deg": round(float(np.mean(deg_all[primary_mask])), 3),
                "endpoint_mpjpe": round(
                    float(np.mean(np.linalg.norm(fk_pts_3d[t] - gt_tensors["pts_3d"][t], axis=-1))), 5
                ),
            }
        )

    stratified["worst_frames_primary"] = sorted(
        per_frame_rows, key=lambda r: r["primary_direction_mean_deg"], reverse=True
    )[:10]

    # ------------------------------------------------------------------
    # Temporal Quality (Jitter, Acceleration, Angular Smoothness, Phase Lag)
    # ------------------------------------------------------------------
    acc_fk = _accel_magnitude_mean(fk_pts_3d)
    acc_gt = _accel_magnitude_mean(gt_tensors["pts_3d"])

    # Frame-to-frame angular step on all bones and on root bone
    R_prev = fk_rot_world[:-1]
    R_curr = fk_rot_world[1:]
    # Trace of R_prev^T @ R_curr
    traces = np.einsum("tnij,tnij->tn", R_prev, R_curr)
    step_angles_deg = np.degrees(np.arccos(np.clip((traces - 1.0) * 0.5, -1.0, 1.0)))
    root_idx = next((i for i, b in enumerate(bone_names) if classified[b].get("kind") == "root"), 0)
    max_root_step_deg = float(np.max(step_angles_deg[:, root_idx])) if step_angles_deg.size else 0.0

    lag_frames = _estimate_primary_lag_frames(fk_pts_3d, gt_tensors["pts_3d"], classified, bone_names)

    temporal_quality = {
        "raw_3d_accel_mean": round(acc_raw, 5),
        "sg_3d_accel_mean": round(acc_sg, 5),
        "fk_3d_accel_mean": round(acc_fk, 5),
        "gt_3d_accel_mean": round(acc_gt, 5),
        "total_accel_jitter_reduction_pct": round((1.0 - acc_fk / max(1e-9, acc_raw)) * 100.0, 2),
        "fk_vs_gt_accel_ratio": round(acc_fk / max(1e-9, acc_gt), 3),
        "angular_step_mean_deg": round(float(np.mean(step_angles_deg)), 3),
        "angular_step_p95_deg": round(float(np.percentile(step_angles_deg, 95)), 3),
        "max_single_frame_root_step_deg": round(max_root_step_deg, 3),
        "estimated_primary_lag_frames": lag_frames,
        "estimated_primary_lag_ms": round(lag_frames * (1000.0 / 24.0), 2),
    }

    # ------------------------------------------------------------------
    # Golden Frame Regression Checks
    # ------------------------------------------------------------------
    golden_results = evaluate_golden_frames(solver_res, frames_gt, golden_spec)

    return {
        "schema": "monster-eval-report-v1",
        "benchmark_id": benchmark_id,
        "solver_name": solver_res.solver_name,
        "tier": tier,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stages": {
            "stage1_perception_2d": stage1_metrics,
            "stage2_cleaning_epipolar": stage2_metrics,
            "stage3_triangulation_3d": stage3_metrics,
            "stage4_kinematic_fk": stage4_metrics,
            "stage5_glb_roundtrip": stage5_metrics,
        },
        "stratified_kinematics": stratified,
        "temporal_quality": temporal_quality,
        "golden_frames": golden_results,
        "per_frame_summary": per_frame_rows,
    }


def diagnose_single_frame(
    solver_res: SolverExecutionResult,
    exam_gt: dict[str, Any],
    frame_idx: int,
    top_k: int = 12,
) -> dict[str, Any]:
    """Traces a single frame (e.g., Frame 17) across Stages 1 -> 2 -> 3 -> 4 to pinpoint
    whether errors originate from 2D perception, epipolar mismatch, 3D fusion, or FK solving.
    """
    frames_gt = exam_gt["frames"]
    if frame_idx < 0 or frame_idx >= len(frames_gt):
        raise ValueError(f"Frame index {frame_idx} out of range [0, {len(frames_gt) - 1}]")

    bone_names = solver_res.bone_names
    classified = solver_res.classified_bones
    calib = solver_res.calib
    gt_tensors = extract_ground_truth_tensors(frames_gt, bone_names, calib)

    pr_bmap = solver_res.stage4_frames[frame_idx]["bones"]
    gt_bmap = frames_gt[frame_idx].get("bones", frames_gt[frame_idx].get("joints", {}))

    bone_traces: list[dict[str, Any]] = []
    for i, b in enumerate(bone_names):
        k_head, k_tail = 2 * i, 2 * i + 1
        # Stage 1 raw UV error (mean of head & tail across front & side)
        uv_f_err_raw = 0.5 * (
            float(np.linalg.norm(solver_res.stage1_raw_obs["uv_front"][frame_idx, k_head] - gt_tensors["uv_front"][frame_idx, k_head]))
            + float(np.linalg.norm(solver_res.stage1_raw_obs["uv_front"][frame_idx, k_tail] - gt_tensors["uv_front"][frame_idx, k_tail]))
        )
        uv_s_err_raw = 0.5 * (
            float(np.linalg.norm(solver_res.stage1_raw_obs["uv_side"][frame_idx, k_head] - gt_tensors["uv_side"][frame_idx, k_head]))
            + float(np.linalg.norm(solver_res.stage1_raw_obs["uv_side"][frame_idx, k_tail] - gt_tensors["uv_side"][frame_idx, k_tail]))
        )
        # Stage 2 epipolar mismatch & Hampel replacement
        epi_dv = 0.5 * (
            float(solver_res.stage2_cleaned_2d["epipolar_dv_px"][frame_idx, k_head])
            + float(solver_res.stage2_cleaned_2d["epipolar_dv_px"][frame_idx, k_tail])
        )
        hampel_fixed = bool(
            solver_res.stage2_cleaned_2d["replaced_front_mask"][frame_idx, k_head]
            or solver_res.stage2_cleaned_2d["replaced_front_mask"][frame_idx, k_tail]
            or solver_res.stage2_cleaned_2d["replaced_side_mask"][frame_idx, k_head]
            or solver_res.stage2_cleaned_2d["replaced_side_mask"][frame_idx, k_tail]
        )
        # Stage 3 smoothed 3D direction error before FK
        v_sg = (
            solver_res.stage3_triangulation["pts_3d_smoothed"][frame_idx, k_tail]
            - solver_res.stage3_triangulation["pts_3d_smoothed"][frame_idx, k_head]
        )
        v_gt = gt_tensors["pts_3d"][frame_idx, k_tail] - gt_tensors["pts_3d"][frame_idx, k_head]
        stage3_dir_deg = math.degrees(
            math.acos(
                float(
                    np.clip(
                        np.dot(v_sg, v_gt) / max(1e-8, float(np.linalg.norm(v_sg) * np.linalg.norm(v_gt))),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        # Stage 4 FK direction & rotation error
        hp = np.asarray(pr_bmap[b]["head_3d"], dtype=np.float64)
        tp = np.asarray(pr_bmap[b]["tail_3d"], dtype=np.float64)
        v_fk = tp - hp
        stage4_dir_deg = math.degrees(
            math.acos(
                float(
                    np.clip(
                        np.dot(v_fk, v_gt) / max(1e-8, float(np.linalg.norm(v_fk) * np.linalg.norm(v_gt))),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        R_p = np.asarray(pr_bmap[b]["rot_mat_world"], dtype=np.float64)
        R_g = np.asarray(gt_bmap[b]["rot_mat_world"], dtype=np.float64)
        stage4_rot_deg = math.degrees(
            math.acos(float(np.clip((np.trace(R_g.T @ R_p) - 1.0) * 0.5, -1.0, 1.0)))
        )

        bone_traces.append(
            {
                "bone": b,
                "canonical_name": classified[b].get("canonical_name", b),
                "kind": classified[b].get("kind", "limb"),
                "sub_role": classified[b].get("sub_role", ""),
                "is_primary": bool(classified[b].get("is_primary", True)),
                "stage1_uv_front_err_px": round(uv_f_err_raw, 2),
                "stage1_uv_side_err_px": round(uv_s_err_raw, 2),
                "stage2_epipolar_dv_px": round(epi_dv, 2),
                "stage2_hampel_interpolated": hampel_fixed,
                "stage3_triangulated_dir_deg": round(stage3_dir_deg, 3),
                "stage4_fk_dir_deg": round(stage4_dir_deg, 3),
                "stage4_fk_rot_deg": round(stage4_rot_deg, 3),
                "fk_dir_gain_deg": round(stage3_dir_deg - stage4_dir_deg, 3),
            }
        )

    primary_traces = [bt for bt in bone_traces if bt["is_primary"]]
    worst_primary = sorted(primary_traces, key=lambda x: x["stage4_fk_dir_deg"], reverse=True)[:top_k]
    root_trace = next((bt for bt in bone_traces if bt["kind"] == "root"), bone_traces[0])

    return {
        "frame": frame_idx,
        "time_s": round(frame_idx / 24.0, 4),
        "root_trace": root_trace,
        "primary_mean_stage3_dir_deg": round(float(np.mean([x["stage3_triangulated_dir_deg"] for x in primary_traces])), 3),
        "primary_mean_stage4_dir_deg": round(float(np.mean([x["stage4_fk_dir_deg"] for x in primary_traces])), 3),
        "primary_mean_stage4_rot_deg": round(float(np.mean([x["stage4_fk_rot_deg"] for x in primary_traces])), 3),
        "worst_primary_bones": worst_primary,
        "all_bones": bone_traces,
    }
