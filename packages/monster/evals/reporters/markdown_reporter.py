#!/usr/bin/env python3
"""markdown_reporter.py — Formats Monster Studio evaluation reports and frame diagnostics as Markdown."""

from __future__ import annotations

from typing import Any


def _fmt_delta(curr: float, base: float | None, unit: str = "", lower_is_better: bool = True) -> str:
    if base is None:
        return "—"
    diff = curr - float(base)
    if abs(diff) < 1e-4:
        return f"`0.000{unit}`"
    sign = "+" if diff > 0 else ""
    improved = (diff < 0) if lower_is_better else (diff > 0)
    tag = "better" if improved else "worse"
    return f"`{sign}{diff:.3f}{unit}` ({tag})"


def render_eval_summary_markdown(
    report: dict[str, Any],
    baseline: dict[str, Any] | None = None,
) -> str:
    """Renders a comprehensive Markdown report from a `monster-eval-report-v1` dictionary."""
    strat = report["stratified_kinematics"]
    all_b = strat["all_bones"]
    pri_b = strat["primary_bones"]
    stages = report["stages"]
    s1 = stages["stage1_perception_2d"]
    s2 = stages["stage2_cleaning_epipolar"]
    s3 = stages["stage3_triangulation_3d"]
    s4 = stages["stage4_kinematic_fk"]
    s5 = stages["stage5_glb_roundtrip"]
    temp = report["temporal_quality"]
    golden = report["golden_frames"]

    base_strat = (baseline or {}).get("stratified_kinematics", {})
    base_all = base_strat.get("all_bones", {})
    base_pri = base_strat.get("primary_bones", {})

    lines: list[str] = [
        f"# Monster Studio Eval Report — `{report['benchmark_id']}`",
        "",
        f"- **Solver**: `{report['solver_name']}` (`{strat.get('schema', 'monster-bone-taxonomy-v2')}`)",
        f"- **Tier**: `{report['tier']}`",
        f"- **Timestamp**: `{report['created_at']}`",
        f"- **Golden Frames Gate**: **{'PASS' if golden['all_passed'] else 'FAIL'}** (`{golden['probe_count']}` probes)",
        "",
        "## 1. Executive Summary (Primary vs All Bones)",
        "",
        "| Metric | Primary Bones (Excl. Digits) | vs Baseline | All Bones (77 Bones) | vs Baseline |",
        "| :--- | :---: | :---: | :---: | :---: |",
        (
            f"| **Bone Count** | `{pri_b['count']}` | — | `{all_b['count']}` | — |"
        ),
        (
            f"| **Mean Direction Error** | **`{pri_b['direction_mean_deg']:.3f}°`** | "
            f"{_fmt_delta(pri_b['direction_mean_deg'], base_pri.get('direction_mean_deg'), '°')} | "
            f"`{all_b['direction_mean_deg']:.3f}°` | "
            f"{_fmt_delta(all_b['direction_mean_deg'], base_all.get('direction_mean_deg'), '°')} |"
        ),
        (
            f"| **Max Per-Bone Direction Error** | `{pri_b['direction_max_bone_deg']:.3f}°` | "
            f"{_fmt_delta(pri_b['direction_max_bone_deg'], base_pri.get('direction_max_bone_deg'), '°')} | "
            f"`{all_b['direction_max_bone_deg']:.3f}°` | "
            f"{_fmt_delta(all_b['direction_max_bone_deg'], base_all.get('direction_max_bone_deg'), '°')} |"
        ),
        (
            f"| **Non-Leaf SO(3) Geodesic Error** | **`{pri_b['nonleaf_rot_mean_deg']:.3f}°`** | "
            f"{_fmt_delta(pri_b['nonleaf_rot_mean_deg'], base_pri.get('nonleaf_rot_mean_deg'), '°')} | "
            f"`{all_b['nonleaf_rot_mean_deg']:.3f}°` | "
            f"{_fmt_delta(all_b['nonleaf_rot_mean_deg'], base_all.get('nonleaf_rot_mean_deg'), '°')} |"
        ),
        (
            f"| **Full-Bone SO(3) Geodesic Error** | `{pri_b['full_rot_mean_deg']:.3f}°` | "
            f"{_fmt_delta(pri_b['full_rot_mean_deg'], base_pri.get('full_rot_mean_deg'), '°')} | "
            f"`{all_b['full_rot_mean_deg']:.3f}°` | "
            f"{_fmt_delta(all_b['full_rot_mean_deg'], base_all.get('full_rot_mean_deg'), '°')} |"
        ),
        (
            f"| **3D Endpoint MPJPE (World Units)** | `{pri_b['endpoint_position_mean']:.5f}` | "
            f"{_fmt_delta(pri_b['endpoint_position_mean'], base_pri.get('endpoint_position_mean'))} | "
            f"`{all_b['endpoint_position_mean']:.5f}` | "
            f"{_fmt_delta(all_b['endpoint_position_mean'], base_all.get('endpoint_position_mean'))} |"
        ),
        "",
        "## 2. Five-Stage Error Attribution",
        "",
        "| Stage | Sub-Pipeline | Key Indicators |",
        "| :--- | :--- | :--- |",
        (
            f"| **Stage 1** | 2D Perception & Aux (`1280×1440`/panel) | "
            f"UV MAE: `{s1['overall_uv_mae_px']:.2f} px` (Front `{s1['front_uv_mae_px']:.2f}`, Side `{s1['side_uv_mae_px']:.2f}`), "
            f"PCK@8px: `{s1['pck_at_8px']:.1f}%`, PCK@16px: `{s1['pck_at_16px']:.1f}%`, Depth MAE: `{s1['depth_overall_mae']:.4f}` |"
        ),
        (
            f"| **Stage 2** | Short-Gap Cleaning & Epipolar Gate | "
            f"Cleaned UV MAE: `{s2['cleaned_overall_uv_mae_px']:.2f} px`, "
            f"Epipolar $|v_F - v_S|$: `{s2['epipolar_v_consistency_clean_px']:.2f} px`, "
            f"Hampel Replaced: Front `{s2['hampel_replaced_ratio_front_pct']:.2f}%` / Side `{s2['hampel_replaced_ratio_side_pct']:.2f}%` |"
        ),
        (
            f"| **Stage 3** | 3D Triangulation & Savitzky-Golay | "
            f"Raw MPJPE: `{s3['raw_triangulated_mpjpe']:.5f}` → SG MPJPE: `{s3['sg_smoothed_mpjpe']:.5f}`, "
            f"Pre-FK Length CV: `{s3['sg_bone_length_cv_pct']:.2f}%`, SG Jitter Reduction: `{s3['sg_jitter_reduction_pct']:.1f}%` |"
        ),
        (
            f"| **Stage 4** | Kind-Specific FK & Joint Limits | "
            f"FK MPJPE: `{s4['fk_mpjpe']:.5f}`, Bone Length Drift: `{s4['bone_length_drift_max']:.6f}`, "
            f"Limit Violation: `{s4['joint_limit_max_violation_deg']:.5f}°`, Rigid Local Drift: `{s4['rigid_local_drift_max_deg']:.5f}°` ({s4['solver_elapsed_ms']:.1f} ms) |"
        ),
        (
            f"| **Stage 5** | GLB Export & Skinning Roundtrip | "
            + (
                f"Executed: `True`, Max Skin Roundtrip Error: `{s5['skin_roundtrip_max']:.6f}`, Source Unchanged: `{s5['source_unchanged']}` |"
                if s5.get("executed")
                else "Skipped in Tier 1/2 (`--tier full` executes headless Blender GLB roundtrip & 2K render) |"
            )
        ),
        "",
        "## 3. Anatomical Kind Breakdown (`by_kind`)",
        "",
        "| Anatomical Kind | Count | Mean Direction (°) | Max Bone Direction (°) | Non-Leaf SO(3) (°) | Full SO(3) (°) | Endpoint MPJPE |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for kind, k_stats in strat.get("by_kind", {}).items():
        lines.append(
            f"| **`{kind}`** | `{k_stats['count']}` | `{k_stats['direction_mean_deg']:.3f}°` | "
            f"`{k_stats['direction_max_bone_deg']:.3f}°` | `{k_stats['nonleaf_rot_mean_deg']:.3f}°` | "
            f"`{k_stats['full_rot_mean_deg']:.3f}°` | `{k_stats['endpoint_position_mean']:.5f}` |"
        )

    lines.extend(
        [
            "",
            "## 4. Temporal Smoothness & Lag Analysis",
            "",
            f"- **3D Endpoint Acceleration (Jitter)**: Raw Triangulated `{temp['raw_3d_accel_mean']:.5f}` → "
            f"SG Smoothed `{temp['sg_3d_accel_mean']:.5f}` → Solved FK `{temp['fk_3d_accel_mean']:.5f}` "
            f"(GT Reference: `{temp['gt_3d_accel_mean']:.5f}`, Total Reduction: **`{temp['total_accel_jitter_reduction_pct']:.1f}%`**)",
            f"- **Frame-to-Frame Angular Velocity**: Mean `{temp['angular_step_mean_deg']:.3f}°/frame`, "
            f"P95 `{temp['angular_step_p95_deg']:.3f}°/frame`, Max Root Single-Frame Step `{temp['max_single_frame_root_step_deg']:.3f}°`",
            f"- **Estimated Primary Phase Lag**: `{temp['estimated_primary_lag_frames']:.2f} frames` (`{temp['estimated_primary_lag_ms']:.1f} ms` at 24 FPS)",
            "",
            "## 5. Golden Frame Regression Probes",
            "",
            "| Frame | Probe | Status | Root Rot Error (°) | Root 1-Frame Step (°) | Primary Mean Dir (°) | Thresholds (Root / Primary) |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for p in golden.get("probes", []):
        status_str = "PASS" if p["passed"] else "FAIL"
        th = p["thresholds"]
        lines.append(
            f"| `{p['frame']}` | **{p['name']}** ({p['description']}) | **{status_str}** | "
            f"`{p['root_rot_err_deg']:.3f}°` | `{p['root_single_frame_step_deg']:.3f}°` | "
            f"`{p['primary_dir_mean_deg']:.3f}°` | `≤ {th['max_root_rot_deg']:.1f}°` / `≤ {th['max_primary_dir_deg']:.1f}°` |"
        )

    lines.append("")
    return "\n".join(lines)


def render_frame_diagnosis_markdown(diag: dict[str, Any]) -> str:
    """Renders a per-frame stage-by-stage attribution trace in Markdown."""
    rt = diag["root_trace"]
    lines = [
        f"# Frame `{diag['frame']}` Attribution Trace (`t = {diag['time_s']:.3f}s`)",
        "",
        f"- **Root (`{rt['bone']}` / `{rt['canonical_name']}`)**: Stage 1 UV Err (`{rt['stage1_uv_front_err_px']:.1f}px` F / `{rt['stage1_uv_side_err_px']:.1f}px` S) → "
        f"Stage 3 Dir `{rt['stage3_triangulated_dir_deg']:.3f}°` → Stage 4 FK Dir `{rt['stage4_fk_dir_deg']:.3f}°`, FK Rot **`{rt['stage4_fk_rot_deg']:.3f}°`**",
        f"- **Primary Bones Mean**: Stage 3 Triangulated Dir `{diag['primary_mean_stage3_dir_deg']:.3f}°` → "
        f"Stage 4 FK Dir **`{diag['primary_mean_stage4_dir_deg']:.3f}°`**, Stage 4 FK Rot **`{diag['primary_mean_stage4_rot_deg']:.3f}°`**",
        "",
        "## Top Primary Bones by Stage 4 FK Direction Error",
        "",
        "| Bone | Canonical | Kind | Stage 1 UV (F/S px) | Stage 2 $\\Delta v$ (px) | Stage 3 3D Dir (°) | Stage 4 FK Dir (°) | Stage 4 FK Rot (°) | FK Gain (°) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for b in diag["worst_primary_bones"]:
        lines.append(
            f"| `{b['bone']}` | `{b['canonical_name']}` | `{b['kind']}` | "
            f"`{b['stage1_uv_front_err_px']:.1f}` / `{b['stage1_uv_side_err_px']:.1f}` | "
            f"`{b['stage2_epipolar_dv_px']:.1f}` | `{b['stage3_triangulated_dir_deg']:.2f}°` | "
            f"**`{b['stage4_fk_dir_deg']:.2f}°`** | `{b['stage4_fk_rot_deg']:.2f}°` | "
            f"`{b['fk_dir_gain_deg']:+.2f}°` |"
        )
    lines.append("")
    return "\n".join(lines)
