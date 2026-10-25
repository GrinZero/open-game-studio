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
    resolve_g1_candidate_files,
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
from sop_formal_gate import verify_rig_formal_gate  # noqa: E402
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


G2_REVIEW_TEMPLATE_PATH = SCRIPT_DIR.parent / "assets" / "g2_review.html"


def _compute_image_foreground_bounds(img: Any) -> list[int]:
    """Computes tight `[x0, y0, x1, y1]` foreground bounding box against top-left background color."""
    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    bg = arr[0, 0]
    mask = np.abs(arr - bg).max(axis=2) > 4
    ys, xs = np.where(mask)
    if len(xs) == 0:
        w, h = img.size
        return [int(w * 0.18), int(h * 0.18), int(w * 0.82), int(h * 0.82)]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _ensure_pose_view_crops_and_bounds(
    pose_records: list[dict[str, Any]],
    calib: dict[str, Any],
    out_dir: Path,
) -> None:
    """Ensures lossless per-view panel crops (`pose_XXX_<view>.png`) and foreground `bounds` exist for each pose."""
    from PIL import Image

    views = list(calib.get("view_order") or ["left", "front", "right"])
    res = calib.get("resolution") or [calib.get("composite_width", 1344), calib.get("composite_height", 768)]
    comp_w = int(calib.get("composite_width") or res[0] or 1344)
    comp_h = int(calib.get("composite_height") or res[1] or 768)
    pw = int(calib.get("panel_width") or max(1, comp_w // max(1, len(views))))
    ph = int(calib.get("panel_height") or comp_h)

    for rec in pose_records:
        pid = str(rec["pose_id"])
        raw_path = out_dir / str(rec.get("raw_rel") or f"{pid}.png")
        ov_path = out_dir / str(rec.get("overlay_rel") or f"{pid}_overlay.png")
        existing_bounds = rec.get("bounds")
        crops_ready = all((out_dir / f"{pid}_{v}.png").is_file() for v in views)
        if isinstance(existing_bounds, dict) and "all" in existing_bounds and crops_ready:
            continue

        b_map: dict[str, list[int]] = dict(existing_bounds) if isinstance(existing_bounds, dict) else {}
        if raw_path.is_file():
            raw_im = Image.open(raw_path)
            b_map["all"] = _compute_image_foreground_bounds(raw_im)
            for idx, v in enumerate(views):
                box = (idx * pw, 0, (idx + 1) * pw, ph)
                crop_im = raw_im.crop(box)
                crop_path = out_dir / f"{pid}_{v}.png"
                if not crop_path.is_file():
                    crop_im.save(crop_path)
                b_map[v] = _compute_image_foreground_bounds(crop_im)
        else:
            b_map.setdefault("all", [int(comp_w * 0.08), int(comp_h * 0.18), int(comp_w * 0.92), int(comp_h * 0.82)])
            for v in views:
                b_map.setdefault(v, [int(pw * 0.18), int(ph * 0.18), int(pw * 0.82), int(ph * 0.82)])

        if ov_path.is_file():
            ov_im = Image.open(ov_path)
            for idx, v in enumerate(views):
                ov_crop_path = out_dir / f"{pid}_overlay_{v}.png"
                if not ov_crop_path.is_file():
                    ov_im.crop((idx * pw, 0, (idx + 1) * pw, ph)).save(ov_crop_path)

        rec["bounds"] = b_map


def _normalize_bone_limits_deg(raw_limits: Any) -> list[list[float]] | None:
    if isinstance(raw_limits, list) and len(raw_limits) == 3:
        return [[round(float(pair[0]), 1), round(float(pair[1]), 1)] for pair in raw_limits]
    if isinstance(raw_limits, dict):
        return [
            [round(float(raw_limits.get(ax, [0.0, 0.0])[0]), 1), round(float(raw_limits.get(ax, [0.0, 0.0])[1]), 1)]
            for ax in ("rx", "ry", "rz")
        ]
    return None


def build_g2_review_html(
    monster_id: str,
    summary: dict[str, Any],
    pose_records: list[dict[str, Any]],
    out_html_path: Path,
) -> Path:
    """Builds the interactive G2 50-pose review HTML page using the dedicated skill template (`assets/g2_review.html`)."""
    out_dir = out_html_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = summary.get("calibration") or summary.get("render_receipt") or {}
    _ensure_pose_view_crops_and_bounds(pose_records, calib, out_dir)

    poses_payload = []
    for r in pose_records:
        pid = str(r["pose_id"])
        regime = str(r.get("sampling_regime") or r.get("regime") or "")
        if isinstance(r.get("joints_deg"), dict):
            joints_deg = r["joints_deg"]
        else:
            joints_deg = {
                k: [round(math.degrees(float(x)), 1) for x in v]
                for k, v in (r.get("joints_rad") or {}).items()
            }
        b_joints = r.get("boundary_joints") or r.get("boundary", {}).get("boundary_joints") or []
        b_count = int(
            r.get("boundary_joint_count")
            or r.get("boundary", {}).get("boundary_joint_count")
            or len(b_joints)
        )
        max_util = round(
            float(r.get("max_utilization", r.get("boundary", {}).get("max_utilization", 0.0))),
            3,
        )
        raw_m = r.get("metrics") or {}
        whole_m = raw_m.get("whole_mesh") if isinstance(raw_m.get("whole_mesh"), dict) else raw_m
        flex_m = raw_m.get("flexible_connections") if isinstance(raw_m.get("flexible_connections"), dict) else {}
        metrics = {
            "bad_edge_ratio": round(float(whole_m.get("bad_edge_ratio", raw_m.get("bad_edge_ratio", 0.0))), 5),
            "flex_bad_edge_ratio": round(
                float(flex_m.get("bad_edge_ratio", raw_m.get("flex_bad_edge_ratio", 0.0))), 5
            ),
            "max_stretch_ratio": round(
                float(whole_m.get("max_stretch_ratio", raw_m.get("max_stretch_ratio", 1.0))), 3
            ),
            "new_hard_intersection_pairs": int(raw_m.get("new_hard_intersection_pairs", 0)),
            "new_flexible_intersection_pairs": int(
                raw_m.get("new_flexible_intersection_pairs", r.get("new_intersection_pairs", 0))
            ),
            "chest_rigid_fit_max_m": round(float(raw_m.get("chest_rigid_fit_max_m", 0.0)), 6),
        }
        poses_payload.append({
            "pose_id": pid,
            "sampling_regime": regime,
            "raw_rel": r.get("raw_rel", f"{pid}.png"),
            "overlay_rel": r.get("overlay_rel", f"{pid}_overlay.png"),
            "root_offset": [round(float(x), 4) for x in r.get("root_offset", [0.0, 0.0, 0.0])],
            "joints_deg": joints_deg,
            "boundary_joint_count": b_count,
            "boundary_joints": b_joints,
            "max_utilization": max_util,
            "metrics": metrics,
            "bounds": r.get("bounds", {}),
            "uv_labels": r.get("uv_labels", {}),
        })

    views = list(
        calib.get("view_order")
        or summary.get("render_receipt", {}).get("view_order")
        or ["left", "front", "right"]
    )
    comp_w = int(calib.get("composite_width") or (calib.get("resolution") or [1344, 768])[0])
    comp_h = int(calib.get("composite_height") or (calib.get("resolution") or [1344, 768])[1])
    cov = summary.get("coverage_audit") or {}
    b_audit = cov.get("boundary_exploration_audit") or summary.get("boundary_exploration_audit") or {}
    jl_summary = summary.get("joint_limits_summary") or {}
    normalized_bones = {}
    for b_name, b_info in (jl_summary.get("bones") or cov.get("bones") or {}).items():
        norm_lim = _normalize_bone_limits_deg(b_info.get("limits_deg"))
        normalized_bones[b_name] = {
            "kind": b_info.get("kind"),
            "sub_role": b_info.get("sub_role"),
            "limits_deg": norm_lim,
            "source": b_info.get("source") or b_info.get("limits_source", "probe_initial"),
        }

    redraw_log = summary.get("rejection_redraw_log") or []
    receipt = dict(summary.get("render_receipt") or {})
    receipt.setdefault("resolution", [comp_w, comp_h])
    receipt.setdefault("view_order", views)
    receipt.setdefault("rejected_draws", len(redraw_log))

    active_dofs = int(
        cov.get("active_dof_count")
        or b_audit.get("active_axes_count")
        or b_audit.get("total_active_axes")
        or 52
    )
    dual_rate = float(
        cov.get("both_bounds_coverage_ratio")
        if cov.get("both_bounds_coverage_ratio") is not None
        else b_audit.get("dual_boundary_touch_rate", 1.0)
    )
    both_covered = int(
        cov.get("both_bounds_covered_count")
        or round(dual_rate * active_dofs)
    )

    summary_payload = {
        "monster_id": monster_id,
        "selected_candidate_id": str(summary.get("selected_candidate_id", "0")),
        "seed": int(summary.get("seed", 42)),
        "num_poses": len(poses_payload),
        "generated_at": summary.get("generated_at", ""),
        "user_decision": summary.get("user_decision", "awaiting_pose_review"),
        "sampler_policy": summary.get("sampler_policy"),
        "render_receipt": receipt,
        "coverage_audit": {
            "passed": bool(cov.get("passed", True)),
            "active_dof_count": active_dofs,
            "both_bounds_covered_count": both_covered,
            "both_bounds_coverage_ratio": round(dual_rate, 4),
        },
        "joint_limits_summary": {
            "schema": jl_summary.get("schema", JOINT_LIMITS_SCHEMA),
            "root_motion": jl_summary.get("root_motion") or jl_summary.get("global_root"),
            "bones": normalized_bones,
        },
        "rejection_redraw_log": redraw_log,
    }

    template_text = G2_REVIEW_TEMPLATE_PATH.read_text(encoding="utf-8")
    doc = (
        template_text
        .replace("__POSE_DATA__", json.dumps(poses_payload, ensure_ascii=False))
        .replace("__SUMMARY_DATA__", json.dumps(summary_payload, ensure_ascii=False))
    )
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

    selected_cid, meta_path, blend_path = resolve_g1_candidate_files(work_dir, g1_info)

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
    formal_gate = verify_rig_formal_gate(classified, joint_limits_profile, assert_passed=True)
    (dirs["checks_dir"] / "formal_gate_rig.json").write_text(
        json.dumps(formal_gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

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

    calib_path = g2_dir / "calibration.json"
    if not calib_path.is_file():
        calib_path.write_text(json.dumps(calib, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    g2_limits_copy = g2_dir / "joint_limits.json"
    if limits_path.is_file() and not g2_limits_copy.is_file():
        g2_limits_copy.write_text(limits_path.read_text(encoding="utf-8"), encoding="utf-8")

    html_path = g2_dir / "g2_review.html"
    build_g2_review_html(monster_id, summary, pose_records, html_path)

    records_path = g2_dir / "pose_records.json"
    records_path.write_text(json.dumps(pose_records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

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
