#!/usr/bin/env python3
"""tier1_fast_solver.py — Tier 1 Fast CPU Kinematic Solver & Attribution Runner (~1-3s)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evals.datasets.drive_bundle import BENCHMARKS_ROOT, VAR_EVALS_ROOT
from evals.datasets.loader import load_benchmark_fixture
from evals.metrics.evaluator import evaluate_full_pipeline
from evals.reporters.html_review_builder import build_html_review_page
from evals.reporters.markdown_reporter import render_eval_summary_markdown
from evals.solvers.latest_solver import run_latest_solver

RUNS_ROOT = VAR_EVALS_ROOT / "runs"


def execute_tier1_fast(
    benchmark_id: str = "fantasy_creature_2k_v2",
    fixture_dir: Path | None = None,
    observations_override: dict[str, np.ndarray] | None = None,
    tier_label: str = "fast",
    run_dir: Path | None = None,
    save_baseline: bool = False,
    bake_report: dict[str, Any] | None = None,
    comparison_video_rel: str | None = None,
) -> dict[str, Any]:
    """Executes Tier 1 (`fast`): runs the latest 3-layer kinematic solver on cached (or overridden)
    2K `observations.npz`, computes full 5-stage attribution + anatomical kind breakdown + golden-frame
    checks, and writes artifacts to `var/monster/evals/runs/<run_id>/`.
    """
    fixture = load_benchmark_fixture(benchmark_id=benchmark_id, fixture_dir=fixture_dir)
    obs = observations_override if observations_override is not None else fixture["observations"]

    solver_res = run_latest_solver(
        observations=obs,
        studio_rig_meta=fixture["rig_meta"],
        constraints_meta=fixture["constraints"],
    )

    report = evaluate_full_pipeline(
        benchmark_id=benchmark_id,
        tier=tier_label,
        solver_res=solver_res,
        exam_gt=fixture["exam_gt"],
        golden_spec=fixture["golden_frames"],
        bake_report=bake_report,
    )

    if run_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_ROOT / f"{ts}_{benchmark_id}_{tier_label}_{solver_res.solver_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save predicted trajectory for inspection or Tier 3 Blender GLB baking
    traj_payload = {
        "schema": solver_res.schema,
        "benchmark_id": benchmark_id,
        "solver_name": solver_res.solver_name,
        "frames": solver_res.stage4_frames,
        "diagnostics": solver_res.stage4_diagnostics,
    }
    traj_path = run_dir / "predicted_trajectory.json"
    traj_path.write_text(json.dumps(traj_payload, indent=2) + "\n", encoding="utf-8")

    # Save JSON report, Markdown summary, and HTML review page
    report_path = run_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary_md = render_eval_summary_markdown(report, baseline=fixture["baseline_latest"])
    summary_path = run_dir / "summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    html_path = build_html_review_page(
        report=report,
        output_path=run_dir / "review.html",
        comparison_video_rel=comparison_video_rel,
    )

    baseline_saved_path: str | None = None
    if save_baseline:
        b_path = BENCHMARKS_ROOT / benchmark_id / "baseline_latest.json"
        b_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep baseline_latest.json concise in Git (exclude per-frame array if desired, or keep structured summary)
        baseline_doc = {
            "schema": report["schema"],
            "benchmark_id": report["benchmark_id"],
            "solver_name": report["solver_name"],
            "tier": report["tier"],
            "created_at": report["created_at"],
            "stages": report["stages"],
            "stratified_kinematics": report["stratified_kinematics"],
            "temporal_quality": report["temporal_quality"],
            "golden_frames": report["golden_frames"],
        }
        b_path.write_text(json.dumps(baseline_doc, indent=2) + "\n", encoding="utf-8")
        baseline_saved_path = str(b_path)

    return {
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "html_path": str(html_path),
        "trajectory_path": str(traj_path),
        "baseline_saved_path": baseline_saved_path,
        "report": report,
        "summary_markdown": summary_md,
    }
