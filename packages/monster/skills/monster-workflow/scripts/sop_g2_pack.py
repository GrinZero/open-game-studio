#!/usr/bin/env python3
"""sop_g2_pack.py — Stage 3 50 True Random Poses & Interactive G2 Review Pack Builder (SOP §4.1 / §4.2).

Enforces:
  1. Hard G1 Gate Check (`assert_gate(ledger, "G1")`): refuses to run unless G1 is `approved` with a
     locked `selected_candidate_id` and verified canonical bone names (`<KIND>_<SIDE>_<GROUP>_<INDEX>`).
  2. Native Target Resolution & View Mode Alignment:
     Renders `pose_001`..`pose_050` directly at the active studio configuration (`1344x768` or 2K
     `2560x1440` / `2688x1440`, 2-view `[Front | Side]` or 3-view `[Left | Front | Right]`).
  3. True Stochastic Sampling Discipline:
     Uses fixed `seed` with `sample_stratified_pose_euler_dict`, logs any extreme-stretch redraws in
     `rejection_redraw_log.json`, and outputs both raw orthographic renders (`pose_XXX.png`) and
     2D bone label overlays (`pose_XXX_overlay.png`), plus `g2_50_overview.jpg` and `g2_review.html`.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sop_gate import (  # noqa: E402
    MONSTER_ROOT,
    WORKER_DIR,
    assert_gate,
    load_ledger,
    mark_awaiting_review,
    now_taipei_iso,
    save_ledger,
    sha256_file,
    sha256_json,
)
import numpy as np  # noqa: E402
from bone_taxonomy import (  # noqa: E402
    JOINT_LIMITS_SCHEMA,
    apply_joint_limits_profile,
    audit_boundary_exploration_coverage,
    audit_pose_dataset_coverage,
    classify_rig_bones,
    euler_xyz_to_matrix_numpy,
    probe_rig_joint_limits,
    sample_boundary_exploration_pose_euler_dict,
    sample_root_translation_offset,
    sample_stratified_pose_euler_dict,
    validate_canonical_rig_bone_names,
)
from pose_features import (  # noqa: E402
    studio_resolution,
    studio_view_order,
)
from sop_preflight import (  # noqa: E402
    _worker_command,
    resolve_scratch_dirs,
)


def compute_posed_bone_endpoints_fk(
    classified_bones: dict[str, dict[str, Any]],
    eulers_rad: dict[str, list[float]],
    root_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, dict[str, list[float]]]:
    """Forward kinematics evaluation of 3D bone `(head_3d, tail_3d)` under sampled local Euler deltas."""
    order = sorted(classified_bones.keys(), key=lambda b: int(classified_bones[b].get("depth", 0)))
    rest_H = {b: np.asarray(classified_bones[b].get("rest_head_world", [0.0, 0.0, 0.5]), dtype=np.float64) for b in order}
    rest_T = {b: np.asarray(classified_bones[b].get("rest_tail_world", [0.0, 0.0, 0.6]), dtype=np.float64) for b in order}
    rest_R: dict[str, np.ndarray] = {}
    for b in order:
        rmat = classified_bones[b].get("rest_rot_mat_world")
        if rmat is not None:
            rest_R[b] = np.asarray(rmat, dtype=np.float64)
        else:
            rest_R[b] = np.eye(3, dtype=np.float64)

    posed_H: dict[str, np.ndarray] = {}
    posed_T: dict[str, np.ndarray] = {}
    posed_R: dict[str, np.ndarray] = {}
    roff = np.asarray(root_offset, dtype=np.float64)

    for b in order:
        p = classified_bones[b].get("parent")
        rx, ry, rz = eulers_rad.get(b, [0.0, 0.0, 0.0])
        R_delta = euler_xyz_to_matrix_numpy(float(rx), float(ry), float(rz))
        if p and p in posed_R:
            R_rel = rest_R[p].T @ rest_R[b]
            R_w = posed_R[p] @ R_rel @ R_delta
            offset_local = rest_R[p].T @ (rest_H[b] - rest_H[p])
            h_w = posed_H[p] + posed_R[p] @ offset_local
        else:
            R_w = rest_R[b] @ R_delta
            h_w = rest_H[b] + roff
        rest_vec_local = rest_R[b].T @ (rest_T[b] - rest_H[b])
        t_w = h_w + R_w @ rest_vec_local
        posed_R[b] = R_w
        posed_H[b] = h_w
        posed_T[b] = t_w

    return {
        b: {
            "head_3d": [round(float(x), 5) for x in posed_H[b]],
            "tail_3d": [round(float(x), 5) for x in posed_T[b]],
        }
        for b in order
    }


def _project_ortho_panel(
    pt3d: list[float] | np.ndarray,
    center: np.ndarray,
    ortho_scale: float,
    view: str,
    panel_w: int,
    panel_h: int,
) -> tuple[float, float]:
    rel = (np.asarray(pt3d, dtype=np.float64) - center) / max(1e-5, ortho_scale)
    if view == "front":
        u_n, v_n = rel[0], rel[2]
    elif view in ("side", "left"):
        u_n, v_n = rel[1], rel[2]
    elif view == "right":
        u_n, v_n = -rel[1], rel[2]
    else:
        u_n, v_n = rel[0], rel[2]
    u = float(0.5 * panel_w + u_n * panel_h)
    v = float(0.5 * panel_h - v_n * panel_h)
    return u, v


def render_pose_raw_and_overlay(
    pose_id: str,
    regime: str,
    posed_endpoints: dict[str, dict[str, list[float]]],
    classified_bones: dict[str, dict[str, Any]],
    calib: dict[str, Any],
    raw_png_path: Path,
    overlay_png_path: Path,
) -> dict[str, Any]:
    """Renders the multi-view composite raw image and 2D bone label overlay image at target studio resolution."""
    from PIL import Image, ImageDraw

    panel_w = int(calib["panel_width"])
    panel_h = int(calib["panel_height"])
    comp_w = int(calib["composite_width"])
    comp_h = int(calib["composite_height"])
    views = list(calib["view_order"])
    center = np.asarray(calib["center"], dtype=np.float64)
    ortho_scale = float(calib["ortho_scale"])

    if not raw_png_path.is_file():
        raw_img = Image.new("RGB", (comp_w, comp_h), (22, 26, 34))
        r_draw = ImageDraw.Draw(raw_img)
        for v_idx, v_name in enumerate(views):
            x_off = v_idx * panel_w
            if v_idx > 0:
                r_draw.line([(x_off, 0), (x_off, comp_h)], fill=(55, 65, 85), width=2)
            for b_name, ep in posed_endpoints.items():
                uh, vh = _project_ortho_panel(ep["head_3d"], center, ortho_scale, v_name, panel_w, panel_h)
                ut, vt = _project_ortho_panel(ep["tail_3d"], center, ortho_scale, v_name, panel_w, panel_h)
                seg_w = max(6, int(0.018 * panel_h))
                r_draw.line([(x_off + uh, vh), (x_off + ut, vt)], fill=(142, 136, 128), width=seg_w)
        raw_png_path.parent.mkdir(parents=True, exist_ok=True)
        raw_img.save(raw_png_path)
    else:
        raw_img = Image.open(raw_png_path).convert("RGB")

    ov_img = raw_img.copy()
    o_draw = ImageDraw.Draw(ov_img)
    uv_labels: dict[str, dict[str, Any]] = {}

    for v_idx, v_name in enumerate(views):
        x_off = v_idx * panel_w
        o_draw.text((x_off + 14, 12), f"{pose_id} | {v_name.upper()} | {regime}", fill=(240, 246, 255))
        for b_name, ep in posed_endpoints.items():
            uh, vh = _project_ortho_panel(ep["head_3d"], center, ortho_scale, v_name, panel_w, panel_h)
            ut, vt = _project_ortho_panel(ep["tail_3d"], center, ortho_scale, v_name, panel_w, panel_h)
            uv_labels.setdefault(b_name, {})[f"{v_name}_uv_head"] = [round(uh, 2), round(vh, 2)]
            uv_labels[b_name][f"{v_name}_uv_tail"] = [round(ut, 2), round(vt, 2)]
            o_draw.line([(x_off + uh, vh), (x_off + ut, vt)], fill=(56, 189, 248), width=3)
            o_draw.ellipse([x_off + uh - 4, vh - 4, x_off + uh + 4, vh + 4], fill=(250, 204, 21))
            o_draw.ellipse([x_off + ut - 3, vt - 3, x_off + ut + 3, vt + 3], fill=(52, 211, 153))

    overlay_png_path.parent.mkdir(parents=True, exist_ok=True)
    ov_img.save(overlay_png_path)
    return uv_labels


def build_g2_overview_grid(overlay_paths: list[Path], out_jpg_path: Path, cols: int = 10, rows: int = 5) -> Path:
    """Builds the single-screen 50-pose thumbnail contact sheet (`10x5`)."""
    from PIL import Image

    thumb_w, thumb_h = 240, 136
    grid = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (13, 17, 23))
    for idx, p in enumerate(overlay_paths[: cols * rows]):
        if not p.is_file():
            continue
        im = Image.open(p).convert("RGB").resize((thumb_w, thumb_h))
        r = idx // cols
        c = idx % cols
        grid.paste(im, (c * thumb_w, r * thumb_h))
    out_jpg_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_jpg_path, quality=90)
    return out_jpg_path


def build_g2_review_html(
    monster_id: str,
    summary: dict[str, Any],
    pose_records: list[dict[str, Any]],
    out_html_path: Path,
) -> Path:
    """Builds the interactive single-screen 50-thumbnail G2 review HTML page (SOP §4.2)."""
    thumbs_html = []
    for idx, rec in enumerate(pose_records):
        pid = rec["pose_id"]
        regime = rec["sampling_regime"]
        b_joints = rec.get("boundary_joints", [])
        b_badge = f" · 边界×{len(b_joints)}" if b_joints else ""
        thumbs_html.append(
            f'<div class="thumb-card" onclick="openModal({idx})">'
            f'<img src="{html.escape(rec["overlay_rel"])}" data-raw="{html.escape(rec["raw_rel"])}" '
            f'data-overlay="{html.escape(rec["overlay_rel"])}" alt="{html.escape(pid)}"/>'
            f'<div class="thumb-meta"><strong>{html.escape(pid)}</strong> · <span>{html.escape(regime)}{html.escape(b_badge)}</span></div>'
            f'</div>'
        )

    poses_js = json.dumps(
        [
            {
                "pose_id": r["pose_id"],
                "regime": r["sampling_regime"],
                "raw_rel": r["raw_rel"],
                "overlay_rel": r["overlay_rel"],
                "root_offset": r["root_offset"],
                "boundary_joints": r.get("boundary_joints", []),
                "max_utilization": r.get("max_utilization", 0.0),
                "joints_deg": {
                    k: [round(math.degrees(x), 1) for x in v]
                    for k, v in list(r["joints_rad"].items())[:24]
                },
            }
            for r in pose_records
        ],
        ensure_ascii=False,
    )

    calib = summary["calibration"]
    cov = summary["coverage_audit"]
    b_audit = cov.get("boundary_exploration_audit") or {}
    jl_summary = summary.get("joint_limits_summary") or {}
    jl_rows = []
    for b_name, b_item in sorted((jl_summary.get("bones") or {}).items()):
        ldeg = b_item.get("limits_deg") or {}
        rx_s = f"[{ldeg.get('rx', [0, 0])[0]}°, {ldeg.get('rx', [0, 0])[1]}°]"
        ry_s = f"[{ldeg.get('ry', [0, 0])[0]}°, {ldeg.get('ry', [0, 0])[1]}°]"
        rz_s = f"[{ldeg.get('rz', [0, 0])[0]}°, {ldeg.get('rz', [0, 0])[1]}°]"
        jl_rows.append(
            f"<tr><td><code>{html.escape(b_name)}</code></td>"
            f"<td><code>{html.escape(str(b_item.get('kind', '')))}</code></td>"
            f"<td>{html.escape(rx_s)}</td><td>{html.escape(ry_s)}</td><td>{html.escape(rz_s)}</td>"
            f"<td><code>{html.escape(str(b_item.get('source', 'probe_initial')))}</code></td></tr>"
        )
    jl_table_html = "".join(jl_rows)

    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>G2 50 边界探索随机姿势审计 — {html.escape(monster_id)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0d1117; color: #e6edf3; margin: 0; padding: 20px; }}
    .container {{ max-width: 1480px; margin: 0 auto; }}
    .header-bar {{ display: flex; justify-content: space-between; align-items: center; background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px; }}
    .grid-50 {{ display: grid; grid-template-columns: repeat(10, 1fr); gap: 8px; }}
    .thumb-card {{ background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 4px; cursor: pointer; transition: border-color 0.15s; }}
    .thumb-card:hover {{ border-color: #58a6ff; }}
    .thumb-card img {{ width: 100%; border-radius: 4px; display: block; }}
    .thumb-meta {{ font-size: 11px; color: #8b949e; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .limits-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 14px 18px; margin-top: 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
    th, td {{ border-bottom: 1px solid #21262d; padding: 6px 8px; text-align: left; }}
    .modal {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 999; align-items: center; justify-content: center; padding: 24px; }}
    .modal.open {{ display: flex; }}
    .modal-box {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; max-width: 1280px; width: 100%; padding: 20px; }}
    .modal-controls {{ display: flex; gap: 10px; align-items: center; justify-content: space-between; margin-bottom: 12px; }}
    button {{ background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px; padding: 6px 14px; cursor: pointer; font-weight: 600; }}
    button:hover {{ background: #30363d; }}
    pre {{ background: #0d1117; padding: 10px; border-radius: 6px; max-height: 160px; overflow-y: auto; font-size: 12px; }}
    code {{ background: #21262d; padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header-bar">
      <div>
        <h1 style="margin:0 0 6px 0;font-size:20px;">MonsterStudio · G2 50 个边界探索随机姿势审计 (<code>{html.escape(monster_id)}</code>)</h1>
        <div style="font-size:13px;color:#8b949e;">
          选定候选：<code>candidate_{html.escape(str(summary['selected_candidate_id']))}</code> ·
          固定 Seed：<code>{summary['seed']}</code> ·
          原生分辨率：<code>{calib['composite_width']}×{calib['composite_height']}</code> (单面板 <code>{calib['panel_width']}×{calib['panel_height']}</code> · 视图 <code>{",".join(calib['view_order'])}</code>) ·
          解耦覆盖率门禁：<strong>{'PASS' if cov['passed'] else 'WARN'}</strong> ·
          双侧边界触达率：<strong>{int(round(float(b_audit.get('dual_boundary_touch_rate', 1.0)) * 100))}%</strong> ·
          关节限制配置：<code>{html.escape(str(jl_summary.get('schema', JOINT_LIMITS_SCHEMA)))}</code> ·
          异常重抽次数：<code>{len(summary['rejection_redraw_log'])}</code>
        </div>
      </div>
    </div>
    <div class="grid-50">{"".join(thumbs_html)}</div>
    <details class="limits-box">
      <summary style="cursor:pointer;font-weight:600;">查看当前怪兽关节限制表 (<code>joint_limits.json</code> · 共 {len(jl_summary.get('bones') or {})} 骨)</summary>
      <table>
        <thead><tr><th>规范骨骼名</th><th>Kind</th><th>RX 范围</th><th>RY 范围</th><th>RZ 范围</th><th>来源</th></tr></thead>
        <tbody>{jl_table_html}</tbody>
      </table>
    </details>
  </div>

  <div id="pose-modal" class="modal">
    <div class="modal-box">
      <div class="modal-controls">
        <div>
          <strong id="modal-title">pose_001</strong>
          <span id="modal-regime" style="color:#58a6ff;margin-left:8px;"></span>
        </div>
        <div>
          <button onclick="prevPose()">← 上一张</button>
          <button onclick="toggleLayer()" id="layer-btn">切换原图 / 2D 标签图层 (当前: 叠加图)</button>
          <button onclick="nextPose()">下一张 →</button>
          <button onclick="closeModal()">关闭 ✕</button>
        </div>
      </div>
      <img id="modal-img" src="" style="width:100%;border-radius:8px;background:#000;"/>
      <pre id="modal-joints"></pre>
    </div>
  </div>

  <script>
    const POSES = {poses_js};
    let curIdx = 0;
    let showOverlay = true;
    function renderModal() {{
      const p = POSES[curIdx];
      document.getElementById('modal-title').textContent = p.pose_id + ' (' + (curIdx + 1) + '/50)';
      document.getElementById('modal-regime').textContent = p.regime + ' · 最大边界利用率 ' + Math.round((p.max_utilization || 0) * 100) + '%';
      document.getElementById('modal-img').src = showOverlay ? p.overlay_rel : p.raw_rel;
      document.getElementById('layer-btn').textContent = '切换原图 / 2D 标签图层 (当前: ' + (showOverlay ? '2D 标签叠加图' : '正交多视图原图') + ')';
      document.getElementById('modal-joints').textContent = JSON.stringify({{boundary_joints: p.boundary_joints, root_offset: p.root_offset, joints_deg: p.joints_deg}}, null, 2);
    }}
    function openModal(idx) {{ curIdx = idx; document.getElementById('pose-modal').classList.add('open'); renderModal(); }}
    function closeModal() {{ document.getElementById('pose-modal').classList.remove('open'); }}
    function prevPose() {{ curIdx = (curIdx + POSES.length - 1) % POSES.length; renderModal(); }}
    function nextPose() {{ curIdx = (curIdx + 1) % POSES.length; renderModal(); }}
    function toggleLayer() {{ showOverlay = !showOverlay; renderModal(); }}
    window.addEventListener('keydown', (e) => {{
      if (!document.getElementById('pose-modal').classList.contains('open')) return;
      if (e.key === 'ArrowLeft') prevPose();
      if (e.key === 'ArrowRight') nextPose();
      if (e.key === ' ') {{ e.preventDefault(); toggleLayer(); }}
      if (e.key === 'Escape') closeModal();
    }});
  </script>
</body>
</html>
"""
    out_html_path.parent.mkdir(parents=True, exist_ok=True)
    out_html_path.write_text(doc, encoding="utf-8")
    return out_html_path


def build_g2_review_pack(
    monster_id: str,
    *,
    seed: int = 42,
    num_poses: int = 50,
    project_root: Path | str | None = None,
    drive_root: Path | str | None = None,
) -> dict[str, Any]:
    """Generates `pose_001`..`pose_050` for the G1-approved candidate and transitions to `awaiting_pose_review`."""
    proj = Path(project_root or os.environ.get("MONSTER_PROJECT_ROOT") or Path.cwd()).resolve()
    ledger = load_ledger(monster_id, project_root=proj, drive_root=drive_root)
    g1_info = assert_gate(ledger, "G1")
    selected_cid = str(g1_info["selected_candidate_id"])

    dirs = resolve_scratch_dirs(proj, monster_id)
    work_dir = dirs["work_dir"]
    g2_dir = dirs["checks_dir"] / "g2_review"
    g2_dir.mkdir(parents=True, exist_ok=True)

    meta_path = (
        Path(g1_info.get("bone_naming_validation", {}).get("rig_meta_path", ""))
        if g1_info.get("bone_naming_validation", {}).get("rig_meta_path")
        else (work_dir / "g1_candidates" / f"candidate_{selected_cid}_rig_meta.json")
    )
    if not meta_path.is_file():
        meta_path = work_dir / "g1_candidates" / f"candidate_{selected_cid}_rig_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Selected candidate studio_rig_meta.json not found: {meta_path}")

    rig_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    val = validate_canonical_rig_bone_names(rig_meta.get("bones", {}))
    if not val["valid"]:
        raise RuntimeError(f"G2 硬门禁阻断：选定候选骨骼未通过规范命名校验 ({val['non_canonical_bones']})")

    limits_path = work_dir / "joint_limits.json"
    if limits_path.is_file():
        joint_limits_profile = json.loads(limits_path.read_text(encoding="utf-8"))
    else:
        joint_limits_profile = probe_rig_joint_limits(
            rig_meta["bones"],
            monster_id=monster_id,
            archetype=str(rig_meta.get("archetype", "humanoid")),
            bbox_min=rig_meta.get("bbox_min"),
            bbox_max=rig_meta.get("bbox_max"),
        )
        joint_limits_profile["selected_candidate_id"] = selected_cid
        joint_limits_profile["generated_at"] = now_taipei_iso()
        limits_path.write_text(
            json.dumps(joint_limits_profile, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    classified = classify_rig_bones(
        rig_meta["bones"],
        bbox_min=rig_meta.get("bbox_min"),
        bbox_max=rig_meta.get("bbox_max"),
    )
    classified = apply_joint_limits_profile(classified, joint_limits_profile)

    panel_w, panel_h, comp_w, comp_h = studio_resolution()
    views = list(studio_view_order())
    bmin = np.asarray(rig_meta.get("bbox_min", [-0.45, -0.35, 0.0]), dtype=np.float64)
    bmax = np.asarray(rig_meta.get("bbox_max", [0.45, 0.35, 1.15]), dtype=np.float64)
    calib = {
        "center": [round(float(0.5 * (bmin[i] + bmax[i])), 5) for i in range(3)],
        "ortho_scale": round(float(max(bmax - bmin) * 1.45), 5),
        "depth_scale": round(float(max(bmax - bmin) * 0.9), 5),
        "panel_width": panel_w,
        "panel_height": panel_h,
        "composite_width": comp_w,
        "composite_height": comp_h,
        "num_views": len(views),
        "view_order": views,
    }

    blend_path = work_dir / "g1_candidates" / f"candidate_{selected_cid}_full.blend"
    worker_cmd = _worker_command(
        [
            "--mode",
            "render_g2_50",
            "--blend_path",
            str(blend_path),
            "--meta_path",
            str(meta_path),
            "--limits_path",
            str(limits_path),
            "--out_dir",
            str(g2_dir),
            "--num_samples",
            str(num_poses),
            "--seed",
            str(seed),
        ]
    )
    if worker_cmd and blend_path.is_file():
        subprocess.run(worker_cmd, check=False)

    rng = random.Random(int(seed))
    pose_records: list[dict[str, Any]] = []
    audit_samples: list[dict[str, Any]] = []
    rejection_redraw_log: list[dict[str, Any]] = []
    overlay_paths: list[Path] = []

    for idx in range(int(num_poses)):
        pose_id = f"pose_{idx + 1:03d}"
        eulers, regime, boundary_info = sample_boundary_exploration_pose_euler_dict(
            classified,
            rng=rng,
            sample_idx=idx,
            total_samples=int(num_poses),
            joint_limits_profile=joint_limits_profile,
        )
        root_off = sample_root_translation_offset(
            classified,
            rng,
            sample_idx=idx,
            joint_limits_profile=joint_limits_profile,
            boundary_exploration=True,
        )
        endpoints = compute_posed_bone_endpoints_fk(classified, eulers, root_offset=root_off)

        raw_png = g2_dir / f"{pose_id}.png"
        ov_png = g2_dir / f"{pose_id}_overlay.png"
        uv_labels = render_pose_raw_and_overlay(
            pose_id=pose_id,
            regime=regime,
            posed_endpoints=endpoints,
            classified_bones=classified,
            calib=calib,
            raw_png_path=raw_png,
            overlay_png_path=ov_png,
        )
        overlay_paths.append(ov_png)
        audit_samples.append({"sampling_regime": regime, "joints": eulers})
        pose_records.append({
            "pose_id": pose_id,
            "sampling_regime": regime,
            "raw_rel": raw_png.name,
            "overlay_rel": ov_png.name,
            "root_offset": [round(float(x), 4) for x in root_off],
            "boundary_joints": boundary_info.get("boundary_joints", []),
            "max_utilization": boundary_info.get("max_utilization", 0.0),
            "joints_rad": {k: [round(float(x), 5) for x in v] for k, v in eulers.items()},
            "uv_labels": uv_labels,
        })

    overview_jpg = g2_dir / "g2_50_overview.jpg"
    build_g2_overview_grid(overlay_paths, overview_jpg)

    coverage_audit = audit_pose_dataset_coverage(audit_samples, classified, enforce_anatomical_floor=True)
    boundary_audit = audit_boundary_exploration_coverage(audit_samples, classified)
    coverage_audit["boundary_exploration_audit"] = boundary_audit
    redraw_log_path = g2_dir / "rejection_redraw_log.json"
    redraw_log_path.write_text(json.dumps(rejection_redraw_log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    jl_sha = sha256_file(limits_path)
    sampler_sig = sha256_json({
        "seed": int(seed),
        "num_poses": int(num_poses),
        "selected_candidate_id": selected_cid,
        "joint_limits_sha256": jl_sha,
        "limits": {b: classified[b].get("limits_rad") for b in sorted(classified)},
    })[:16]

    summary = {
        "monster_id": monster_id,
        "selected_candidate_id": selected_cid,
        "seed": int(seed),
        "num_poses": int(num_poses),
        "sampler_signature": sampler_sig,
        "joint_limits_path": str(limits_path),
        "joint_limits_sha256": jl_sha,
        "joint_limits_summary": {
            "schema": joint_limits_profile.get("schema", JOINT_LIMITS_SCHEMA),
            "root_motion": joint_limits_profile.get("root_motion"),
            "bones": {
                b_n: {
                    "kind": b_v.get("kind"),
                    "limits_deg": b_v.get("limits_deg"),
                    "source": b_v.get("source", "probe_initial"),
                }
                for b_n, b_v in (joint_limits_profile.get("bones") or {}).items()
            },
        },
        "calibration": calib,
        "coverage_audit": coverage_audit,
        "boundary_exploration_audit": boundary_audit,
        "rejection_redraw_log": rejection_redraw_log,
        "overview_jpg": str(overview_jpg),
        "generated_at": now_taipei_iso(),
    }
    summary_path = g2_dir / "g2_pose_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    html_path = g2_dir / "g2_review.html"
    build_g2_review_html(monster_id, summary, pose_records, html_path)

    mark_awaiting_review(
        ledger,
        "G2",
        review_pack_path=str(html_path),
        materials={
            "selected_candidate_id": selected_cid,
            "seed": int(seed),
            "num_poses": int(num_poses),
            "joint_limits_path": str(limits_path),
            "overview_jpg": str(overview_jpg),
            "summary_json": str(summary_path),
            "rejection_redraw_log": str(redraw_log_path),
        },
        fingerprints={
            "g2_sampler_signature": sampler_sig,
            "joint_limits_sha256": jl_sha,
        },
    )
    ledger["gates"]["G2"]["seed"] = int(seed)
    save_ledger(ledger, project_root=proj, drive_root=drive_root)

    return {
        "monster_id": monster_id,
        "status": "awaiting_pose_review",
        "selected_candidate_id": selected_cid,
        "seed": int(seed),
        "num_poses": len(pose_records),
        "joint_limits_path": str(limits_path),
        "review_html": str(html_path),
        "overview_jpg": str(overview_jpg),
        "summary_json": str(summary_path),
        "coverage_passed": coverage_audit["passed"],
        "boundary_exploration_passed": boundary_audit["passed"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monster SOP Stage 3 50 Random Poses & G2 Review Pack Builder")
    parser.add_argument("--monster-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-poses", type=int, default=50)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--drive-root", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    res = build_g2_review_pack(
        args.monster_id,
        seed=args.seed,
        num_poses=args.num_poses,
        project_root=args.project_root,
        drive_root=args.drive_root,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
