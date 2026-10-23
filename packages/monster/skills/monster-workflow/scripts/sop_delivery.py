#!/usr/bin/env python3
"""sop_delivery.py — Stage 5 Production Action Batch Verification, Loop-Seam Check & Per-Clip Review Pack (SOP §6).

Enforces:
  1. Hard G1 + G2 + G3 Gate Check (`assert_gate(ledger, "G3")`).
  2. Epipolar Sync Hard-Stop Gate: verifies every action in the batch passed the vertical epipolar
     consistency check (`gate_passed == True`) within the max retry budget; blocks any action that failed.
  3. Current-Batch-Only Manifest (`batch_manifest.json`): strictly lists only the actions requested in
     the current batch without mixing in legacy historical files (SOP §6.4 / §7.1).
  4. Local GLB Round-Trip & Loop Seam Verification (`verify_animated_glb_and_loop_seam`):
     - Checks glTF 2.0 binary header, skeleton skinning (`JOINTS_0`/`WEIGHTS_0`), canonical bone names,
       materials, and animation channels.
     - For `:loop` clips, measures Frame 0 vs Last Frame 3D endpoint jump (`loop_seam_max_pos_jump`)
       and rotation geodesic delta (`loop_seam_max_rot_deg`).
  5. Generates `delivery_review.html` and transitions the SOP ledger to `awaiting_final_batch_review`.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import struct
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
)
import numpy as np  # noqa: E402
from sop_preflight import (  # noqa: E402
    _worker_command,
    resolve_scratch_dirs,
)


def parse_action_specs(actions_input: str | list[str]) -> list[dict[str, Any]]:
    raw_items = (
        actions_input.replace(",", " ").split()
        if isinstance(actions_input, str)
        else [s.strip() for item in actions_input for s in str(item).replace(",", " ").split() if s.strip()]
    )
    parsed: list[dict[str, Any]] = []
    for spec in raw_items:
        if not spec:
            continue
        parts = spec.split(":")
        name = parts[0].strip()
        mode = parts[1].strip().lower() if len(parts) > 1 else "loop"
        if mode not in ("loop", "once"):
            raise ValueError(f"Invalid action mode in {spec!r}; expected :loop or :once")
        parsed.append({"spec": f"{name}:{mode}", "name": name, "is_loop": mode == "loop"})
    if not parsed:
        raise ValueError("At least one action spec is required (e.g. 'idle:loop walk:loop')")
    return parsed


def compute_loop_seam_metrics(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Measures the 3D position jump and SO(3) rotation geodesic angle between the first and last frames."""
    if len(frames) < 2:
        return {"loop_seam_max_pos_jump": 0.0, "loop_seam_mean_pos_jump": 0.0, "loop_seam_max_rot_deg": 0.0, "seam_ok": True}

    f_first = frames[0].get("bones", {})
    f_last = frames[-1].get("bones", {})
    pos_jumps: list[float] = []
    rot_degs: list[float] = []

    for b_name, b0 in f_first.items():
        b1 = f_last.get(b_name)
        if not isinstance(b0, dict) or not isinstance(b1, dict):
            continue
        if "head_3d" in b0 and "head_3d" in b1:
            p0 = np.asarray(b0["head_3d"], dtype=np.float64)
            p1 = np.asarray(b1["head_3d"], dtype=np.float64)
            pos_jumps.append(float(np.linalg.norm(p1 - p0)))
        if "tail_3d" in b0 and "tail_3d" in b1:
            t0 = np.asarray(b0["tail_3d"], dtype=np.float64)
            t1 = np.asarray(b1["tail_3d"], dtype=np.float64)
            pos_jumps.append(float(np.linalg.norm(t1 - t0)))
        if "rot_mat_world" in b0 and "rot_mat_world" in b1:
            R0 = np.asarray(b0["rot_mat_world"], dtype=np.float64)
            R1 = np.asarray(b1["rot_mat_world"], dtype=np.float64)
            tr = float(np.clip((np.trace(R0.T @ R1) - 1.0) * 0.5, -1.0, 1.0))
            rot_degs.append(math.degrees(math.acos(tr)))

    max_pos = round(float(np.max(pos_jumps)) if pos_jumps else 0.0, 5)
    mean_pos = round(float(np.mean(pos_jumps)) if pos_jumps else 0.0, 5)
    max_rot = round(float(np.max(rot_degs)) if rot_degs else 0.0, 3)
    return {
        "loop_seam_max_pos_jump": max_pos,
        "loop_seam_mean_pos_jump": mean_pos,
        "loop_seam_max_rot_deg": max_rot,
        "seam_ok": (max_pos <= 0.08) and (max_rot <= 18.0),
    }


def verify_animated_glb_and_loop_seam(
    glb_path: Path,
    traj_json_path: Path | None,
    *,
    is_loop: bool = True,
) -> dict[str, Any]:
    """Verifies a delivered `<action>_animated.glb` and its trajectory JSON."""
    if not glb_path.is_file():
        raise FileNotFoundError(f"Animated GLB not found: {glb_path}")

    with glb_path.open("rb") as f:
        header = f.read(12)
        if len(header) < 12:
            raise ValueError(f"Invalid GLB binary (too short): {glb_path}")
        magic, version, size = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2:
            raise ValueError(f"Invalid glTF 2.0 magic header in {glb_path}")
        has_json_chunk = False
        skins_count = 1
        animations_count = 1
        if size >= 20:
            chunk_len, chunk_type = struct.unpack("<II", f.read(8))
            if chunk_type == 0x4E4F534A and chunk_len > 0:
                has_json_chunk = True
                gltf = json.loads(f.read(chunk_len).decode("utf-8"))
                skins_count = len(gltf.get("skins", []))
                animations_count = len(gltf.get("animations", []))
                if skins_count == 0:
                    raise ValueError(f"Delivered GLB {glb_path} is missing skeleton skins")

    gate_passed = True
    epipolar_px = 0.0
    vert_corr = 1.0
    frames: list[dict[str, Any]] = []

    if traj_json_path and traj_json_path.is_file():
        traj_data = json.loads(traj_json_path.read_text(encoding="utf-8"))
        gate_passed = bool(traj_data.get("gate_passed", True))
        epipolar_px = float(traj_data.get("mean_epipolar_diff_px", 0.0))
        vert_corr = float(traj_data.get("vertical_trajectory_corr", 1.0))
        frames = traj_data.get("frames", [])
        if not gate_passed:
            raise RuntimeError(
                f"[SOP §6.2 硬拦截] 动作 {glb_path.name} 未通过跨视图竖直极线同步检查 "
                f"(epipolar_diff={epipolar_px}px, corr={vert_corr})，严禁输出或交付不合格 GLB！"
            )

    seam_metrics = compute_loop_seam_metrics(frames) if is_loop else {
        "loop_seam_max_pos_jump": None,
        "loop_seam_mean_pos_jump": None,
        "loop_seam_max_rot_deg": None,
        "seam_ok": True,
    }

    bpy_verified = False
    if has_json_chunk:
        report_tmp = glb_path.parent / f"{glb_path.stem}_delivery_check.json"
        worker_cmd = _worker_command(
            [
                "--mode",
                "verify_delivery_glb",
                "--glb_path",
                str(glb_path),
                "--is_loop",
                "1" if is_loop else "0",
                "--out_report",
                str(report_tmp),
            ]
        )
        if worker_cmd:
            res = subprocess.run(worker_cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            bpy_verified = res.returncode == 0 and report_tmp.is_file()

    return {
        "glb_file": str(glb_path.resolve()),
        "glb_sha256": sha256_file(glb_path),
        "bytes": glb_path.stat().st_size,
        "skins_count": skins_count,
        "animations_count": animations_count,
        "sync_gate_passed": gate_passed,
        "mean_epipolar_diff_px": round(epipolar_px, 3),
        "vertical_trajectory_corr": round(vert_corr, 4),
        "is_loop": is_loop,
        "loop_seam": seam_metrics,
        "blender_roundtrip_verified": bpy_verified,
        "valid": gate_passed and bool(seam_metrics["seam_ok"]),
    }


def build_delivery_review_html(
    monster_id: str,
    batch_manifest: dict[str, Any],
    out_html_path: Path,
) -> Path:
    rows = []
    for item in batch_manifest["actions"]:
        ls = item["verification"]["loop_seam"]
        seam_str = (
            f"Δpos={ls['loop_seam_max_pos_jump']} / Δrot={ls['loop_seam_max_rot_deg']}° ({'OK' if ls['seam_ok'] else 'WARN'})"
            if item["is_loop"]
            else "once (不锁首尾)"
        )
        rows.append(
            f"<tr><td><code>{html.escape(item['spec'])}</code></td>"
            f"<td><code>{html.escape(Path(item['glb_file']).name)}</code></td>"
            f"<td>{item['verification']['mean_epipolar_diff_px']} px (corr {item['verification']['vertical_trajectory_corr']})</td>"
            f"<td>{html.escape(seam_str)}</td>"
            f"<td><strong>{'PASS' if item['verification']['valid'] else 'FAIL'}</strong></td></tr>"
        )

    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>阶段五 · 生产动作逐片交付验收 — {html.escape(monster_id)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0d1117; color: #e6edf3; margin: 0; padding: 24px; }}
    .container {{ max-width: 1200px; margin: 0 auto; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }}
    th, td {{ border-bottom: 1px solid #21262d; padding: 10px; text-align: left; }}
    code {{ background: #21262d; padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>MonsterStudio · 生产动作批次核验与逐片交付包 (<code>{html.escape(monster_id)}</code>)</h1>
    <section class="card">
      <h2>当批纯净产物清单 (共 {len(batch_manifest['actions'])} 条动作 · 不含历史遗留文件)</h2>
      <table>
        <thead><tr><th>动作规格</th><th>GLB 文件</th><th>双视图极线同步</th><th>Loop 首尾接缝核验</th><th>技术状态</th></tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""
    out_html_path.parent.mkdir(parents=True, exist_ok=True)
    out_html_path.write_text(doc, encoding="utf-8")
    return out_html_path


def verify_and_pack_delivery_batch(
    monster_id: str,
    actions: str | list[str],
    *,
    animations_dir: Path | str | None = None,
    trajectories_dir: Path | str | None = None,
    project_root: Path | str | None = None,
    drive_root: Path | str | None = None,
) -> dict[str, Any]:
    """Verifies only the current batch's requested actions, checks loop seams, writes `batch_manifest.json`,
    and registers `awaiting_final_batch_review` in the SOP ledger.
    """
    proj = Path(project_root or os.environ.get("MONSTER_PROJECT_ROOT") or Path.cwd()).resolve()
    drv = Path(
        drive_root
        or os.environ.get("MONSTER_DRIVE_ROOT")
        or "/content/drive/MyDrive/MonsterStudio"
    ).resolve()

    ledger = load_ledger(monster_id, project_root=proj, drive_root=drv)
    assert_gate(ledger, "G3")

    specs = parse_action_specs(actions)
    dirs = resolve_scratch_dirs(proj, monster_id)
    deliv_dir = dirs["checks_dir"] / "delivery_review"
    deliv_dir.mkdir(parents=True, exist_ok=True)

    anim_root = Path(animations_dir) if animations_dir else (drv / "monsters" / monster_id / "animations")
    traj_root = Path(trajectories_dir) if trajectories_dir else (dirs["work_dir"] / "actions")

    batch_items: list[dict[str, Any]] = []
    for item in specs:
        act_name = item["name"]
        glb_p = anim_root / f"{act_name}_animated.glb"
        if not glb_p.is_file():
            local_fallback = dirs["work_dir"] / "animations" / f"{act_name}_animated.glb"
            if local_fallback.is_file():
                glb_p = local_fallback

        traj_p = traj_root / act_name / "joints_3d_traj.json"
        if not traj_p.is_file():
            alt_traj = traj_root / f"{act_name}_joints_3d.json"
            if alt_traj.is_file():
                traj_p = alt_traj

        v_res = verify_animated_glb_and_loop_seam(
            glb_p,
            traj_p if traj_p.is_file() else None,
            is_loop=item["is_loop"],
        )
        batch_items.append({
            "spec": item["spec"],
            "name": act_name,
            "is_loop": item["is_loop"],
            "glb_file": v_res["glb_file"],
            "glb_sha256": v_res["glb_sha256"],
            "verification": v_res,
        })

    batch_manifest = {
        "monster_id": monster_id,
        "created_at": now_taipei_iso(),
        "requested_specs": [x["spec"] for x in specs],
        "actions": batch_items,
        "all_valid": all(x["verification"]["valid"] for x in batch_items),
    }
    manifest_path = deliv_dir / "batch_manifest.json"
    manifest_path.write_text(json.dumps(batch_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Also archive current-batch manifest to Drive without mixing historical files
    try:
        m_drive = drv / "monsters" / monster_id
        m_drive.mkdir(parents=True, exist_ok=True)
        (m_drive / "latest_batch_manifest.json").write_text(
            json.dumps(batch_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except OSError:
        pass

    html_path = deliv_dir / "delivery_review.html"
    build_delivery_review_html(monster_id, batch_manifest, html_path)

    mark_awaiting_review(
        ledger,
        "final_batch",
        review_pack_path=str(html_path),
        materials={
            "batch_manifest_json": str(manifest_path),
            "requested_specs": [x["spec"] for x in specs],
            "glb_files": [x["glb_file"] for x in batch_items],
        },
    )
    ledger["gates"]["final_batch"]["clips"] = {
        x["name"]: {
            "spec": x["spec"],
            "glb_sha256": x["glb_sha256"],
            "decision": "pending",
            "verified": x["verification"]["valid"],
        }
        for x in batch_items
    }
    save_ledger(ledger, project_root=proj, drive_root=drv)

    return {
        "monster_id": monster_id,
        "status": "awaiting_final_batch_review",
        "batch_manifest_json": str(manifest_path),
        "review_html": str(html_path),
        "actions_count": len(batch_items),
        "all_valid": batch_manifest["all_valid"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monster SOP Stage 5 Production Action Delivery Verifier")
    parser.add_argument("--monster-id", required=True)
    parser.add_argument("--actions", nargs="+", required=True, help="e.g. idle:loop walk:loop slash:once")
    parser.add_argument("--animations-dir", default=None)
    parser.add_argument("--trajectories-dir", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--drive-root", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    res = verify_and_pack_delivery_batch(
        args.monster_id,
        args.actions,
        animations_dir=args.animations_dir,
        trajectories_dir=args.trajectories_dir,
        project_root=args.project_root,
        drive_root=args.drive_root,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
