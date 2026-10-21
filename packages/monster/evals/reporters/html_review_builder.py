#!/usr/bin/env python3
"""html_review_builder.py — Builds a self-contained HTML review page for Monster Studio eval runs."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def build_html_review_page(
    report: dict[str, Any],
    output_path: Path,
    comparison_video_rel: str | None = None,
) -> Path:
    """Writes a standalone `review.html` dashboard with stage attribution cards, kind tables,
    golden frame probes, and optional embedded 2K comparison video player.
    """
    strat = report["stratified_kinematics"]
    pri = strat["primary_bones"]
    all_b = strat["all_bones"]
    s1 = report["stages"]["stage1_perception_2d"]
    s3 = report["stages"]["stage3_triangulation_3d"]
    s4 = report["stages"]["stage4_kinematic_fk"]
    temp = report["temporal_quality"]
    golden = report["golden_frames"]

    kind_rows_html = "\n".join(
        f"<tr><td><code>{html.escape(k)}</code></td>"
        f"<td>{v['count']}</td>"
        f"<td>{v['direction_mean_deg']:.3f}°</td>"
        f"<td>{v['direction_max_bone_deg']:.3f}°</td>"
        f"<td>{v['nonleaf_rot_mean_deg']:.3f}°</td>"
        f"<td>{v['full_rot_mean_deg']:.3f}°</td>"
        f"<td>{v['endpoint_position_mean']:.5f}</td></tr>"
        for k, v in strat.get("by_kind", {}).items()
    )

    golden_rows_html = "\n".join(
        f"<tr><td><code>{p['frame']}</code></td>"
        f"<td><strong>{html.escape(p['name'])}</strong> — {html.escape(p['description'])}</td>"
        f"<td class='{'pass' if p['passed'] else 'fail'}'>{'PASS' if p['passed'] else 'FAIL'}</td>"
        f"<td>{p['root_rot_err_deg']:.3f}°</td>"
        f"<td>{p['root_single_frame_step_deg']:.3f}°</td>"
        f"<td>{p['primary_dir_mean_deg']:.3f}°</td></tr>"
        for p in golden.get("probes", [])
    )

    video_section = ""
    if comparison_video_rel:
        video_section = f"""
        <section class="card">
          <h2>2K Split-Screen Reconstruction Comparison</h2>
          <video controls loop muted playsinline style="width:100%;border-radius:8px;background:#000;">
            <source src="{html.escape(comparison_video_rel)}" type="video/mp4">
          </video>
        </section>
        """

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Monster Studio Eval — {html.escape(report['benchmark_id'])}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f141c; color: #e6edf3; margin: 0; padding: 24px; }}
    .container {{ max-width: 1180px; margin: 0 auto; }}
    h1, h2 {{ color: #f0f6fc; margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 18px; margin-bottom: 20px; }}
    .metric-val {{ font-size: 28px; font-weight: 700; color: #58a6ff; margin: 6px 0; }}
    .metric-sub {{ font-size: 13px; color: #8b949e; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #21262d; padding: 10px 12px; text-align: left; }}
    th {{ color: #8b949e; font-weight: 600; }}
    .pass {{ color: #3fb950; font-weight: 700; }}
    .fail {{ color: #f85149; font-weight: 700; }}
    code {{ background: #21262d; padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Monster Studio Eval — <code>{html.escape(report['benchmark_id'])}</code></h1>
    <p class="metric-sub">Solver: <code>{html.escape(report['solver_name'])}</code> | Tier: <code>{html.escape(report['tier'])}</code> | Created: <code>{html.escape(report['created_at'])}</code></p>
    <div class="grid">
      <div class="card">
        <div>Primary Bones Direction Error</div>
        <div class="metric-val">{pri['direction_mean_deg']:.3f}°</div>
        <div class="metric-sub">{pri['count']} Primary Bones | All 77 Bones: {all_b['direction_mean_deg']:.3f}°</div>
      </div>
      <div class="card">
        <div>Primary Non-Leaf SO(3) Error</div>
        <div class="metric-val">{pri['nonleaf_rot_mean_deg']:.3f}°</div>
        <div class="metric-sub">Full Primary Rot: {pri['full_rot_mean_deg']:.3f}° | All Non-Leaf: {all_b['nonleaf_rot_mean_deg']:.3f}°</div>
      </div>
      <div class="card">
        <div>2D Perception UV MAE (2K)</div>
        <div class="metric-val">{s1['overall_uv_mae_px']:.2f} px</div>
        <div class="metric-sub">PCK@8px: {s1['pck_at_8px']:.1f}% | PCK@16px: {s1['pck_at_16px']:.1f}%</div>
      </div>
      <div class="card">
        <div>Temporal Jitter Reduction</div>
        <div class="metric-val">{temp['total_accel_jitter_reduction_pct']:.1f}%</div>
        <div class="metric-sub">Raw Accel {s3['raw_accel_jitter']:.4f} → FK {temp['fk_3d_accel_mean']:.4f} | Lag: {temp['estimated_primary_lag_frames']:.2f}f</div>
      </div>
    </div>
    {video_section}
    <section class="card">
      <h2>Anatomical Kind Breakdown</h2>
      <table>
        <thead>
          <tr><th>Kind</th><th>Count</th><th>Mean Dir</th><th>Max Bone Dir</th><th>Non-Leaf SO(3)</th><th>Full SO(3)</th><th>Endpoint MPJPE</th></tr>
        </thead>
        <tbody>{kind_rows_html}</tbody>
      </table>
    </section>
    <section class="card">
      <h2>Golden Frame Regression Probes</h2>
      <table>
        <thead>
          <tr><th>Frame</th><th>Probe</th><th>Status</th><th>Root Rot Err</th><th>Root 1-Frame Step</th><th>Primary Mean Dir</th></tr>
        </thead>
        <tbody>{golden_rows_html}</tbody>
      </table>
    </section>
  </div>
  <script id="eval-report-data" type="application/json">{json.dumps(report)}</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
    return output_path
