#!/usr/bin/env python3
"""sop_gate.py — Deterministic Audit Ledger, Version Fingerprinting & Gate State Machine (SOP §1.2 / reviews.md).

Maintains synchronized audit records across:
  - Project ledger: `<project_root>/.game-studio/monster/<monster_id>/reviews.json`
  - Drive state:    `<drive_root>/monsters/<monster_id>/stage_state.json`

Enforces:
  1. Hard blocking before downstream stages (`assert_gate`) unless the required gate (`G1`, `G2`, `G3`,
     `final_batch`) has `user_decision == "approved"` bound to the current upstream fingerprints.
  2. Mandatory canonical bone naming verification (`bone_naming_validation.valid == True` and zero
     anonymous `bone_0..N` bones) before `G1` can be approved or passed.
  3. Automatic graded invalidation (`derive_invalidations`) implementing the 10 invalidation rules
     in `references/reviews.md` without deleting historical records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_monster_root(script_dir: Path = SCRIPT_DIR) -> Path:
    """Locates `packages/monster` whether invoked from `packages/monster/skills/...`,
    `<repo>/skills/...`, `plugins/game-studio/skills/...`, or `~/.agents/skills/...`.
    """
    candidates = [
        script_dir.parents[2],  # packages/monster/skills/monster-workflow/scripts -> packages/monster
        script_dir.parents[2] / "packages" / "monster",  # <repo>/skills/monster-workflow/scripts -> <repo>/packages/monster
        script_dir.parents[4] / "runtime" / "packages" / "monster",  # plugins/game-studio/skills/... -> runtime/packages/monster
        Path.home() / ".local/share/codex-integrations/game-studio-local/runtime/packages/monster",
        Path("/Users/bugyaluwang/project/codex-game-studio/packages/monster"),
    ]
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for v_cand in candidates:
        for sp in sorted((v_cand / ".venv" / "lib").glob(f"{py_ver}/site-packages")):
            if sp.is_dir() and str(sp) not in sys.path:
                sys.path.insert(0, str(sp))
    for cand in candidates:
        if (cand / "worker" / "bone_taxonomy.py").is_file():
            return cand.resolve()
    return script_dir.parents[2]


MONSTER_ROOT = resolve_monster_root(SCRIPT_DIR)
WORKER_DIR = MONSTER_ROOT / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from bone_taxonomy import validate_canonical_rig_bone_names  # noqa: E402
from pose_features import (  # noqa: E402
    clean_motion_enabled,
    decode_mode,
    dense_surface_enabled,
    export_npz_labels_enabled,
    features_enabled,
    fold_digits_enabled,
    keypoint_mode,
    mocap_priors_enabled,
    no_cut_tracks_enabled,
    panel_drift_enabled,
    pipeline_mode,
    resolution_2k_enabled,
    rig_contract_v3_enabled,
    solver_mode,
    studio_resolution,
    studio_view_order,
    surface_contact_enabled,
    three_view_enabled,
    twoway_jump_enabled,
    visibility_mode,
)

VALID_GATES = ("G1", "G2", "G3", "final_batch")
VALID_DECISIONS = ("pending", "approved", "changes_requested", "rejected")
WAIT_STAGE_BY_GATE = {
    "G1": "awaiting_rig_review",
    "G2": "awaiting_pose_review",
    "G3": "awaiting_exam_review",
    "final_batch": "awaiting_final_batch_review",
}


def now_taipei_iso() -> str:
    return datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path: Path | str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def current_studio_config_snapshot(env: dict[str, str] | None = None) -> dict[str, Any]:
    """Captures the active resolution, view layout, and solver/surface feature flags."""
    old_env = os.environ.copy()
    if env is not None:
        os.environ.update(env)
    try:
        panel_w, panel_h, comp_w, comp_h = studio_resolution()
        views = list(studio_view_order())
        return {
            "camera_lens": os.environ.get("MONSTER_CAMERA_LENS", "0"),
            "camera_layout": os.environ.get("MONSTER_STUDIO_LAYOUT", "horizontal"),
            "panel_width": panel_w,
            "panel_height": panel_h,
            "composite_width": comp_w,
            "composite_height": comp_h,
            "num_views": len(views),
            "view_order": views,
            "enable_2k": resolution_2k_enabled(),
            "enable_3view": three_view_enabled(),
            "depth_visibility": features_enabled(),
            "dense_surface": dense_surface_enabled(),
            "mocap_priors": mocap_priors_enabled(),
            "pipeline_mode": pipeline_mode(),
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
            "h3_profile": os.environ.get("H3_PROFILE", "full"),
        }
    finally:
        if env is not None:
            os.environ.clear()
            os.environ.update(old_env)


def dataset_config_signature(cfg: dict[str, Any]) -> str:
    """Signature of studio parameters that require re-synthesizing dataset & re-training DINOv2.
    Note: `mocap_priors` is an inference/kinematic solver prior and does NOT alter dataset_signature!
    """
    core = {
        "camera_lens": cfg.get("camera_lens", "0"),
        "camera_layout": cfg.get("camera_layout", "horizontal"),
        "panel_width": cfg.get("panel_width"),
        "panel_height": cfg.get("panel_height"),
        "composite_width": cfg.get("composite_width"),
        "composite_height": cfg.get("composite_height"),
        "num_views": cfg.get("num_views"),
        "view_order": cfg.get("view_order"),
        "depth_visibility": bool(cfg.get("depth_visibility", True)),
        "dense_surface": bool(cfg.get("dense_surface", False)),
        "pipeline_mode": cfg.get("pipeline_mode", "video2rig"),
        "rig_contract_v3": bool(cfg.get("rig_contract_v3", True)),
        "fold_digits": bool(cfg.get("fold_digits", False)),
        "keypoint_mode": cfg.get("keypoint_mode", "video2rig"),
        "visibility_mode": cfg.get("visibility_mode", "hybrid"),
        "export_npz_labels": bool(cfg.get("export_npz_labels", True)),
    }
    return sha256_json(core)[:16]


def resolve_ledger_paths(
    monster_id: str,
    project_root: Path | str | None = None,
    drive_root: Path | str | None = None,
) -> tuple[Path, Path]:
    proj = Path(project_root or os.environ.get("MONSTER_PROJECT_ROOT") or Path.cwd()).resolve()
    drv = Path(
        drive_root
        or os.environ.get("MONSTER_DRIVE_ROOT")
        or "/content/drive/MyDrive/MonsterStudio"
    ).resolve()
    reviews_path = proj / ".game-studio" / "monster" / monster_id / "reviews.json"
    stage_state_path = drv / "monsters" / monster_id / "stage_state.json"
    return reviews_path, stage_state_path


def load_ledger(
    monster_id: str,
    project_root: Path | str | None = None,
    drive_root: Path | str | None = None,
) -> dict[str, Any]:
    reviews_path, stage_state_path = resolve_ledger_paths(monster_id, project_root, drive_root)
    ledger: dict[str, Any] = {
        "schema_version": 2,
        "monster_id": monster_id,
        "updated_at": now_taipei_iso(),
        "studio_config": current_studio_config_snapshot(),
        "fingerprints": {},
        "transfers": {},
        "gates": {
            "G1": {"status": "not_ready", "user_decision": "pending", "history": []},
            "G2": {"status": "not_ready", "user_decision": "pending", "history": []},
            "G3": {"status": "not_ready", "user_decision": "pending", "history": []},
            "final_batch": {"status": "not_ready", "user_decision": "pending", "clips": {}, "history": []},
        },
        "invalidation_log": [],
    }
    if reviews_path.is_file():
        try:
            loaded = json.loads(reviews_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for k, v in loaded.items():
                    if k == "gates" and isinstance(v, dict):
                        for gk, gv in v.items():
                            ledger["gates"][gk] = {**ledger["gates"].get(gk, {}), **gv}
                    else:
                        ledger[k] = v
        except Exception:
            pass
    elif stage_state_path.is_file():
        try:
            st = json.loads(stage_state_path.read_text(encoding="utf-8"))
            if isinstance(st, dict) and isinstance(st.get("sop_ledger"), dict):
                for k, v in st["sop_ledger"].items():
                    ledger[k] = v
        except Exception:
            pass
    return ledger


def save_ledger(
    ledger: dict[str, Any],
    project_root: Path | str | None = None,
    drive_root: Path | str | None = None,
) -> dict[str, str]:
    monster_id = ledger["monster_id"]
    reviews_path, stage_state_path = resolve_ledger_paths(monster_id, project_root, drive_root)
    ledger["updated_at"] = now_taipei_iso()

    reviews_path.parent.mkdir(parents=True, exist_ok=True)
    reviews_path.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    written = {"reviews_json": str(reviews_path)}
    try:
        stage_state_path.parent.mkdir(parents=True, exist_ok=True)
        stage_data: dict[str, Any] = {}
        if stage_state_path.is_file():
            try:
                stage_data = json.loads(stage_state_path.read_text(encoding="utf-8"))
            except Exception:
                stage_data = {}
        stage_data["sop_ledger"] = {
            "updated_at": ledger["updated_at"],
            "studio_config": ledger.get("studio_config"),
            "fingerprints": ledger.get("fingerprints"),
            "transfers": ledger.get("transfers"),
            "gates": {
                g: {k: v for k, v in ginfo.items() if k != "history"}
                for g, ginfo in ledger.get("gates", {}).items()
            },
        }
        stage_state_path.write_text(json.dumps(stage_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written["stage_state_json"] = str(stage_state_path)
    except OSError:
        pass
    return written


def _invalidate_gate(ledger: dict[str, Any], gate: str, reason: str, timestamp: str) -> bool:
    ginfo = ledger["gates"].setdefault(gate, {"status": "not_ready", "user_decision": "pending", "history": []})
    if ginfo.get("user_decision") == "approved" or ginfo.get("status") in ("approved", WAIT_STAGE_BY_GATE.get(gate)):
        old_snapshot = {k: v for k, v in ginfo.items() if k != "history"}
        ginfo.setdefault("history", []).append({**old_snapshot, "invalidated_at": timestamp, "invalidation_reason": reason})
        ginfo["status"] = "invalidated"
        ginfo["user_decision"] = "pending"
        ginfo["invalidated_reason"] = reason
        ginfo["invalidated_at"] = timestamp
        ginfo.pop("approved_at", None)
        ledger.setdefault("invalidation_log", []).append(
            {"gate": gate, "reason": reason, "timestamp": timestamp}
        )
        return True
    return False


def derive_invalidations(
    ledger: dict[str, Any],
    *,
    new_fingerprints: dict[str, Any] | None = None,
    new_studio_config: dict[str, Any] | None = None,
    change_type: str | None = None,
) -> list[dict[str, str]]:
    """Implements the graded invalidation table from `references/reviews.md`.

    Supported explicit `change_type` values or automatic fingerprint diffing:
      - `u1_input_defect`: Invalidates U1, G1, G2, G3, final_batch.
      - `skeleton_topology_defect`: Invalidates G1, G2, G3, final_batch (preserves U1).
      - `local_model_or_bone_naming_change`: Marks G1 for supplemental review and invalidates G2, G3, final_batch.
      - `sampler_script_or_seed_change`: Preserves G1; invalidates G2, G3, final_batch.
      - `resolution_or_view_mode_change`: Preserves G1 and G2; invalidates dataset, checkpoint, G3, final_batch.
      - `dense_surface_enabled_0_to_1`: Preserves G1 and G2; invalidates dataset, checkpoint, G3, final_batch.
      - `dense_surface_disabled_1_to_0`: Preserves G1, G2, dataset, and checkpoint; only invalidates G3 for re-exam.
      - `mocap_priors_change`: Preserves G1, G2, dataset, and checkpoint; only invalidates G3 for re-exam.
      - `exam_video_or_ik_fix`: Preserves G1, G2, dataset, and checkpoint; only invalidates G3 for re-exam.
      - `action_prompt_or_seed_change`: Preserves G1, G2, G3; only invalidates final_batch.
    """
    stamp = now_taipei_iso()
    old_fp = ledger.get("fingerprints", {})
    old_cfg = ledger.get("studio_config", {})
    cur_fp = {**old_fp, **(new_fingerprints or {})}
    cur_cfg = {**old_cfg, **(new_studio_config or {})}
    triggered: list[dict[str, str]] = []

    def mark(gates: list[str], reason: str) -> None:
        for g in gates:
            if _invalidate_gate(ledger, g, reason, stamp):
                triggered.append({"gate": g, "reason": reason})

    # 1. Explicit change_type rules
    if change_type == "u1_input_defect":
        mark(["G1", "G2", "G3", "final_batch"], "输入模型基础缺陷（姿态/拓扑粘连/遮挡）变更，原 U1 与全部下游关卡失效")
    elif change_type == "skeleton_topology_defect":
        mark(["G1", "G2", "G3", "final_batch"], "自动骨架结构或层级缺陷，选定候选与全部依赖关卡失效")
    elif change_type in ("local_model_or_bone_naming_change", "anatomical_reader_v3_migration"):
        mark(["G1", "G2", "G3", "final_batch"], "网格、骨骼规范命名/KIND分类、骨骼轴、蒙皮权重或装饰绑定发生实质改动")
    elif change_type == "sampler_script_or_seed_change":
        mark(["G2", "G3", "final_batch"], "随机采样脚本、关节角度范围或 seed 变更（保留 G1，失效 G2 及下游）")
    elif change_type == "pose_generalization_insufficient":
        mark(["G2", "G3", "final_batch"], "姿态泛化不足，退回 G2 重新采样、训练与考试（保留 G1）")
    elif change_type == "resolution_or_view_mode_change":
        mark(["G3", "final_batch"], "目标最高分辨率档位（MONSTER_ENABLE_2K）或视角模式（MONSTER_ENABLE_3VIEW）变更，旧训练集/checkpoint/G3 失效（保留 G1/G2）")
    elif change_type in ("dense_surface_enabled_0_to_1", "reader_label_contract_change"):
        mark(["G3", "final_batch"], "稠密表面坐标与可微 LBS 开关从 0 开启为 1，dataset_signature 变更，需重合成标签、训练 dense_head 并重考 G3（保留 G1/G2）")
    elif change_type in ("dense_surface_disabled_1_to_0", "mocap_priors_change", "exam_video_or_ik_fix"):
        mark(["G3", "final_batch"], f"仅运动学求解器/考试链路变更 ({change_type})，兼容现有数据集与 dinov2_head.pt，仅需重考 G3（保留 G1/G2/checkpoint）")
    elif change_type == "action_prompt_or_seed_change":
        mark(["final_batch"], "仅修改生产动作规格、提示词或动作 seed（保留 G1/G2/G3，仅重审对应动作批次）")

    # 2. Automatic fingerprint & studio_config diffing
    if old_fp.get('g2_review_sha256') and cur_fp.get('g2_review_sha256') and old_fp['g2_review_sha256'] != cur_fp['g2_review_sha256']:
        mark(['G2','G3','final_batch'], 'G2 实际姿态、标签、相机或审查图像已变更，需重新审查当前采样结果')
    if (
        old_fp.get("g3_review_sha256")
        and cur_fp.get("g3_review_sha256")
        and old_fp["g3_review_sha256"] != cur_fp["g3_review_sha256"]
    ):
        mark(["G3", "final_batch"], "G3 轨迹、诊断报告或审查材料已变更，需重新审查当前考试结果")
    if old_fp.get("u1_glb_sha256") and cur_fp.get("u1_glb_sha256") and old_fp["u1_glb_sha256"] != cur_fp["u1_glb_sha256"]:
        mark(["G1", "G2", "G3", "final_batch"], "U1 输入模型 SHA-256 变更，下游全部关卡自动失效")

    if (
        old_fp.get("selected_candidate_blend_sha256")
        and cur_fp.get("selected_candidate_blend_sha256")
        and old_fp["selected_candidate_blend_sha256"] != cur_fp["selected_candidate_blend_sha256"]
    ):
        mark(["G1", "G2", "G3", "final_batch"], "选定完整绑定模型 (.blend) SHA-256 变更，需补审 G1 并重做 G2/G3")

    if (
        old_fp.get("rig_meta_sha256")
        and cur_fp.get("rig_meta_sha256")
        and old_fp["rig_meta_sha256"] != cur_fp["rig_meta_sha256"]
    ):
        mark(["G1", "G2", "G3", "final_batch"], "studio_rig_meta.json 骨骼命名或分类元数据变更，需补审 G1 并重做 G2/G3")

    if (
        old_fp.get("g2_sampler_signature")
        and cur_fp.get("g2_sampler_signature")
        and old_fp["g2_sampler_signature"] != cur_fp["g2_sampler_signature"]
    ):
        mark(["G2", "G3", "final_batch"], "G2 采样策略签名或固定 seed 变更（保留 G1，自动失效 G2/G3）")

    if (
        old_fp.get("joint_limits_sha256")
        and cur_fp.get("joint_limits_sha256")
        and old_fp["joint_limits_sha256"] != cur_fp["joint_limits_sha256"]
    ):
        mark(["G2", "G3", "final_batch"], "joint_limits.json 关节角度限制变更（保留 G1，自动失效 G2/G3 及下游）")

    if old_cfg and new_studio_config:
        if old_cfg.get('camera_lens','0') != cur_cfg.get('camera_lens','0') or old_cfg.get('camera_layout','horizontal') != cur_cfg.get('camera_layout','horizontal'):
            mark(['G2','G3','final_batch'],'相机投影/排布变更，必须重新检查采样、生成标签并训练 Reader')
        old_res_view = (old_cfg.get("composite_width"), old_cfg.get("composite_height"), old_cfg.get("num_views"))
        cur_res_view = (cur_cfg.get("composite_width"), cur_cfg.get("composite_height"), cur_cfg.get("num_views"))
        if old_res_view[0] is not None and cur_res_view[0] is not None and old_res_view != cur_res_view:
            mark(
                ["G3", "final_batch"],
                f"摄影棚分辨率/视角数从 {old_res_view} 变更为 {cur_res_view}，旧数据集、checkpoint 与 G3 自动失效",
            )
        old_dense = bool(old_cfg.get("dense_surface", False))
        cur_dense = bool(cur_cfg.get("dense_surface", False))
        if not old_dense and cur_dense:
            mark(
                ["G3", "final_batch"],
                "MONSTER_ENABLE_DENSE_SURFACE 从 0 开启为 1，需重新合成稠密标签并训练 dense_head、重考 G3",
            )
        elif old_dense and not cur_dense:
            mark(
                ["G3", "final_batch"],
                "MONSTER_ENABLE_DENSE_SURFACE 从 1 回退为 0，无需重训权重，但需重跑 G3 考试确认纯关键点解析解表现",
            )
        if old_cfg.get("mocap_priors") is not None and cur_cfg.get("mocap_priors") is not None:
            if bool(old_cfg.get("mocap_priors")) != bool(cur_cfg.get("mocap_priors")):
                mark(
                    ["G3", "final_batch"],
                    "MONSTER_ENABLE_MOCAP_PRIORS 开关变更，不改变数据集签名且兼容已有 dinov2_head.pt，仅失效 G3 需重考",
                )
        if old_cfg.get("pipeline_mode") is not None and cur_cfg.get("pipeline_mode") is not None:
            if old_cfg.get("pipeline_mode") != cur_cfg.get("pipeline_mode"):
                mark(
                    ["G1", "G2", "G3", "final_batch"],
                    f"主链路模式 MONSTER_PIPELINE_MODE 从 {old_cfg.get('pipeline_mode')} 切换为 {cur_cfg.get('pipeline_mode')}，骨骼/标签/求解器契约变更，全部关卡需重新验收",
                )
        if old_cfg.get("solver_mode") is not None and cur_cfg.get("solver_mode") is not None:
            if old_cfg.get("solver_mode") != cur_cfg.get("solver_mode"):
                mark(
                    ["G3", "final_batch"],
                    f"求解器模式 MONSTER_SOLVER_MODE 从 {old_cfg.get('solver_mode')} 切换为 {cur_cfg.get('solver_mode')}，需重考 G3",
                )
        for g1_flag in ("rig_contract_v3", "fold_digits"):
            if old_cfg.get(g1_flag) is not None and cur_cfg.get(g1_flag) is not None:
                if old_cfg.get(g1_flag) != cur_cfg.get(g1_flag):
                    mark(
                        ["G1", "G2", "G3", "final_batch"],
                        f"骨架契约开关 {g1_flag} 从 {old_cfg.get(g1_flag)} 切换为 {cur_cfg.get(g1_flag)}，需重新执行 G1–G3",
                    )
        for g2_flag in ("keypoint_mode", "visibility_mode", "export_npz_labels"):
            if old_cfg.get(g2_flag) is not None and cur_cfg.get(g2_flag) is not None:
                if old_cfg.get(g2_flag) != cur_cfg.get(g2_flag):
                    mark(
                        ["G2", "G3", "final_batch"],
                        f"标签/可见性契约开关 {g2_flag} 从 {old_cfg.get(g2_flag)} 切换为 {cur_cfg.get(g2_flag)}，需重新生成标签并验收 G2/G3",
                    )
        for g3_flag in (
            "decode_mode",
            "clean_motion",
            "surface_contact",
            "no_cut_tracks",
            "twoway_jump",
            "panel_drift",
        ):
            if old_cfg.get(g3_flag) is not None and cur_cfg.get(g3_flag) is not None:
                if old_cfg.get(g3_flag) != cur_cfg.get(g3_flag):
                    mark(
                        ["G3", "final_batch"],
                        f"推理/后处理开关 {g3_flag} 从 {old_cfg.get(g3_flag)} 切换为 {cur_cfg.get(g3_flag)}，需重考 G3",
                    )

    ledger["fingerprints"] = cur_fp
    ledger["studio_config"] = cur_cfg
    return triggered


def verify_rig_meta_canonical_bones(rig_meta_path: Path | str) -> dict[str, Any]:
    """Hard gate verifying that `studio_rig_meta.json` has 100% canonical bone names (`<KIND>_<SIDE>_<GROUP>_<INDEX>`)."""
    p = Path(rig_meta_path)
    if not p.is_file():
        return {"valid": False, "error": f"studio_rig_meta.json not found: {p}", "non_canonical_bones": []}
    meta = json.loads(p.read_text(encoding="utf-8"))
    bones = meta.get("bones", meta if isinstance(meta, dict) else {})
    validation = validate_canonical_rig_bone_names(bones)
    return {
        **validation,
        "rig_meta_path": str(p.resolve()),
        "rig_meta_sha256": sha256_file(p),
    }


def resolve_g1_candidate_files(work_dir: Path, g1_info: dict[str, Any]) -> tuple[str, Path, Path]:
    """Resolve the reviewed revision; explicit missing paths must never fall back to old snapshots."""
    cid = str(g1_info["selected_candidate_id"])
    mats = g1_info.get("materials") or {}
    bound_cid = mats.get("selected_candidate_id")
    if bound_cid is not None and str(bound_cid) != cid:
        raise ValueError(f"G1 selected revision belongs to candidate {bound_cid}, not {cid}")
    meta_value = (g1_info.get("bone_naming_validation") or {}).get("rig_meta_path") or mats.get("rig_meta_path")
    meta = Path(meta_value) if meta_value else work_dir / "g1_candidates" / f"candidate_{cid}_rig_meta.json"
    blend_value = mats.get("selected_blend_path")
    candidates = mats.get("candidate_blends") or []
    if not blend_value and candidates:
        if not cid.isdigit() or not 0 <= int(cid) < len(candidates):
            raise ValueError(f"G1 candidate_blends has no entry for selected candidate {cid}")
        blend_value = candidates[int(cid)]
    blend = Path(blend_value) if blend_value else work_dir / "g1_candidates" / f"candidate_{cid}_full.blend"
    for label, path in (("rig metadata", meta), ("full rig", blend)):
        if not path.is_file():
            raise FileNotFoundError(f"G1 selected {label} not found: {path}")
    return cid, meta, blend


def snapshot_g3_materials(materials: dict[str, Any], review_pack_path: str) -> dict[str, Any]:
    """Bind the report and the actual files the user will review, not their filenames."""
    paths = {"review_html": review_pack_path}
    for field in (
        "exam_report_json", "prediction_trajectory_json", "source_exam_video",
        "comparison_video_mp4", "baked_glb", "three_quarter_video", "silhouette_video",
    ):
        if materials.get(field):
            paths[field] = materials[field]
    if materials.get("reader_contract") == "monster-reader-v3":
        missing = {"exam_report_json", "prediction_trajectory_json", "source_exam_video"} - paths.keys()
        if missing:
            raise ValueError(f"V3 G3 review lacks required materials: {sorted(missing)}")
    artifacts = {}
    for field, path in paths.items():
        digest = sha256_file(path)
        if digest is None:
            raise ValueError(f"G3 review material missing: {field}={path}")
        artifacts[field] = {"path": str(Path(path).resolve()), "sha256": digest}
    return artifacts


def snapshot_g2_materials(materials: dict[str, Any], review_pack_path: str) -> dict[str, Any]:
    paths = {'review_html': review_pack_path}
    for field in ('native_labels_json', 'pose_records_json', 'calibration_json', 'summary_json', 'joint_limits_path', 'overview_jpg'):
        if not materials.get(field):
            raise ValueError(f'V3 G2 review lacks {field}')
        paths[field] = materials[field]
    for i, image in enumerate(materials.get('pose_images') or []):
        paths[f'pose_image_{i}'] = image
    if not materials.get('pose_images'):
        raise ValueError('V3 G2 review lacks actual pose images')
    artifacts = {}
    for field, path in paths.items():
        digest = sha256_file(path)
        if digest is None:
            raise ValueError(f'V3 G2 material missing: {field}={path}')
        artifacts[field] = {'path': str(Path(path).resolve()), 'sha256': digest}
    return artifacts


def verify_review_materials(ledger: dict[str, Any], ginfo: dict[str, Any], gate: str) -> None:
    """Reject edits/deletions after submission or approval, keeping upstream decisions."""
    materials = ginfo.get("materials") or {}
    artifacts = materials.get(f"{gate.lower()}_review_artifacts")
    if not artifacts:
        if materials.get("reader_contract") == "monster-reader-v3":
            raise RuntimeError(f"{gate} v3 review has no bound artifact fingerprints; rebuild the review pack")
        return  # Historical ledgers remain readable; new submissions always bind files.
    changed = [field for field, item in artifacts.items() if sha256_file(item["path"]) != item["sha256"]]
    expected = sha256_json(artifacts)
    if changed or (ginfo.get("bound_fingerprints") or {}).get(f"{gate.lower()}_review_sha256") != expected:
        reason = f"{gate} 审查材料已变更或缺失，请重建并审查当前结果: " + ", ".join(changed or ["fingerprint"])
        stamp = now_taipei_iso()
        for downstream in (("G2", "G3", "final_batch") if gate == "G2" else ("G3", "final_batch")):
            _invalidate_gate(ledger, downstream, reason, stamp)
        raise RuntimeError(reason)


def mark_awaiting_review(
    ledger: dict[str, Any],
    gate: str,
    *,
    review_pack_path: str,
    materials: dict[str, Any] | None = None,
    fingerprints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if gate not in VALID_GATES:
        raise ValueError(f"Unsupported gate: {gate!r}; expected one of {VALID_GATES}")
    if not review_pack_path or not Path(review_pack_path).exists():
        raise ValueError(f"Review pack material must exist before entering {WAIT_STAGE_BY_GATE[gate]}: {review_pack_path}")

    materials = dict(materials or {})
    fingerprints = dict(fingerprints or {})
    if gate == "G3":
        artifacts = snapshot_g3_materials(materials, review_pack_path)
        materials["g3_review_artifacts"] = artifacts
        fingerprints["g3_review_sha256"] = sha256_json(artifacts)
    if gate == "G2" and materials.get('reader_contract') == 'monster-reader-v3':
        artifacts = snapshot_g2_materials(materials, review_pack_path)
        materials['g2_review_artifacts'] = artifacts
        fingerprints['g2_review_sha256'] = sha256_json(artifacts)
    if fingerprints:
        derive_invalidations(ledger, new_fingerprints=fingerprints)

    if gate == "G1":
        # G1 requires canonical bone naming verification on all candidate rigs or the target rig_meta
        mats = materials or {}
        meta_p = mats.get("rig_meta_path")
        if meta_p:
            bone_check = verify_rig_meta_canonical_bones(meta_p)
            if not bone_check.get("valid"):
                raise ValueError(
                    f"G1 硬门禁阻断：存在未规范命名的匿名骨骼 {bone_check.get('non_canonical_bones')}，"
                    "必须先完成 <KIND>_<SIDE>_<GROUP>_<INDEX> 强制重命名并通过 validate_canonical_rig_bone_names！"
                )
        for blend_p in mats.get("candidate_blends") or []:
            bp = Path(blend_p)
            if not bp.is_file() or bp.stat().st_size < 1024:
                raise ValueError(f"G1 硬门禁阻断：候选 .blend 不存在或为占位文件: {bp}")
            hdr = bp.read_bytes()[:32]
            if not hdr.startswith(b"BLENDER") or b"G1-CANDIDATE-" in hdr:
                raise ValueError(f"G1 硬门禁阻断：候选 .blend 非真实未压缩 Blender 文件: {bp}")
        for vid_p in (mats.get("sync_videos") or {}).values():
            vp = Path(vid_p)
            if not vp.is_file() or vp.stat().st_size < 256:
                raise ValueError(f"G1 硬门禁阻断：同步对比视频不存在或为 24 字节占位文件: {vp}")
    if gate in ("G2", "G3", "final_batch"):
        assert_gate(ledger, "G1")
    if gate in ("G3", "final_batch"):
        assert_gate(ledger, "G2")
    if gate == "final_batch":
        assert_gate(ledger, "G3")

    ginfo = ledger["gates"].setdefault(gate, {})
    ginfo.update({
        "status": WAIT_STAGE_BY_GATE[gate],
        "user_decision": "pending",
        "review_pack_path": str(Path(review_pack_path).resolve()),
        "materials": materials or {},
        "bound_fingerprints": dict(ledger.get("fingerprints", {})),
        "bound_studio_config": dict(ledger.get("studio_config", {})),
        "submitted_at": now_taipei_iso(),
    })
    ginfo.pop("approved_at", None)
    return ginfo


def record_user_decision(
    ledger: dict[str, Any],
    gate: str,
    *,
    decision: str,
    user_quote: str,
    selected_candidate_id: str | int | None = None,
    rig_meta_path: str | None = None,
    seed: int | None = None,
    checkpoint_path: str | None = None,
    clip_decisions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Records an explicit user decision (`approved`, `changes_requested`, `rejected`) bound to version hashes."""
    if gate not in VALID_GATES:
        raise ValueError(f"Unsupported gate: {gate!r}")
    if decision not in VALID_DECISIONS or decision == "pending":
        raise ValueError(f"User decision must be approved, changes_requested, or rejected (got {decision!r})")
    if not user_quote or not user_quote.strip():
        raise ValueError("必须提供用户的明确原始反馈引用 (user_quote)；禁止无用户证据写入决定。")

    ginfo = ledger["gates"].setdefault(gate, {"status": "not_ready", "user_decision": "pending", "history": []})

    if decision == "approved":
        if gate == "G1":
            cand_id = selected_candidate_id if selected_candidate_id is not None else ginfo.get("selected_candidate_id")
            if cand_id is None or str(cand_id).strip() == "":
                raise ValueError("G1 通过时必须指定唯一的 selected_candidate_id")
            meta_file = rig_meta_path or (ginfo.get("materials") or {}).get("rig_meta_path")
            if not meta_file:
                raise ValueError("G1 通过时必须提供选定候选的 studio_rig_meta.json 路径以锁定骨骼规范命名校验")
            bone_check = verify_rig_meta_canonical_bones(meta_file)
            if not bone_check.get("valid"):
                raise ValueError(
                    f"G1 通过被阻断：选定候选骨骼未通过规范命名校验 ({bone_check.get('non_canonical_bones')})"
                )
            ginfo["selected_candidate_id"] = str(cand_id)
            ginfo["bone_naming_validation"] = bone_check
            ledger.setdefault("fingerprints", {})["rig_meta_sha256"] = bone_check["rig_meta_sha256"]
            mats = ginfo.get("materials") or {}
            cand_blends = mats.get("candidate_blends") or []
            selected_blend = mats.get("selected_blend_path")
            if selected_blend or (str(cand_id).isdigit() and 0 <= int(str(cand_id)) < len(cand_blends)):
                if mats.get("selected_candidate_id") is not None and str(mats["selected_candidate_id"]) != str(cand_id):
                    raise ValueError("G1 selected_blend_path belongs to another candidate")
                sel_blend = Path(selected_blend or cand_blends[int(str(cand_id))])
                if not sel_blend.is_file() or sel_blend.stat().st_size < 1024:
                    raise ValueError(f"G1 通过被阻断：选定候选 .blend 为空或占位文件: {sel_blend}")
                ledger["fingerprints"]["selected_candidate_blend_sha256"] = sha256_file(sel_blend)
            for vid_p in (mats.get("sync_videos") or {}).values():
                vp = Path(vid_p)
                if not vp.is_file() or vp.stat().st_size < 256:
                    raise ValueError(f"G1 通过被阻断：同步对比视频不存在或为 24 字节占位文件: {vp}")

        elif gate == "G2":
            assert_gate(ledger, "G1")
            verify_review_materials(ledger, ginfo, "G2")
            eff_seed = seed if seed is not None else ginfo.get("seed")
            if eff_seed is None:
                raise ValueError("G2 通过时必须锁定固定采样 seed")
            ginfo["seed"] = int(eff_seed)

        elif gate == "G3":
            assert_gate(ledger, "G1")
            assert_gate(ledger, "G2")
            verify_review_materials(ledger, ginfo, "G3")
            ckpt = checkpoint_path or (ginfo.get("materials") or {}).get("checkpoint_path")
            if ckpt and Path(ckpt).is_file():
                ginfo["checkpoint_path"] = str(Path(ckpt).resolve())
                ledger.setdefault("fingerprints", {})["ckpt_sha256"] = sha256_file(ckpt)

        elif gate == "final_batch":
            assert_gate(ledger, "G3")
            if clip_decisions:
                clips = ginfo.setdefault("clips", {})
                for cname, cdec in clip_decisions.items():
                    clips[cname] = {"decision": cdec, "updated_at": now_taipei_iso()}
            clips = ginfo.get("clips", {})
            if clips and any(v.get("decision") != "approved" for v in clips.values()):
                raise ValueError("批次中尚有未通过验收的动作视频，不能将 final_batch 标记为全部 approved")

        ginfo["status"] = "approved"
        ginfo["user_decision"] = "approved"
        ginfo["user_quote"] = user_quote.strip()
        ginfo["approved_at"] = now_taipei_iso()
        ginfo["bound_fingerprints"] = dict(ledger.get("fingerprints", {}))
        ginfo["bound_studio_config"] = dict(ledger.get("studio_config", {}))
    else:
        old_snap = {k: v for k, v in ginfo.items() if k != "history"}
        ginfo.setdefault("history", []).append({**old_snap, "recorded_at": now_taipei_iso()})
        ginfo["status"] = decision
        ginfo["user_decision"] = decision
        ginfo["user_quote"] = user_quote.strip()
        ginfo["decision_at"] = now_taipei_iso()
        ginfo.pop("approved_at", None)

    return ginfo


def assert_gate(ledger: dict[str, Any], required_gate: str) -> dict[str, Any]:
    """Hard gate assertion: raises RuntimeError if `required_gate` (or any upstream gate) is not approved."""
    order = ["G1", "G2", "G3", "final_batch"]
    if required_gate not in order:
        raise ValueError(f"Unknown gate: {required_gate!r}")
    idx = order.index(required_gate)
    for g in order[: idx + 1]:
        ginfo = ledger.get("gates", {}).get(g, {})
        if ginfo.get("user_decision") != "approved" or ginfo.get("status") != "approved":
            reason = ginfo.get("invalidated_reason") or f"当前状态为 {ginfo.get('status', 'not_ready')} (decision={ginfo.get('user_decision', 'pending')})"
            raise RuntimeError(
                f"[SOP 门禁拦截] 尚未通过 {g} 用户审计关卡（{reason}），严禁越级执行下游任务！"
            )
        if g == "G1":
            bval = ginfo.get("bone_naming_validation", {})
            if not bval.get("valid"):
                raise RuntimeError("[SOP 门禁拦截] G1 缺少有效的骨骼规范命名校验记录 (bone_naming_validation.valid != True)")
        if g in ("G2", "G3"):
            verify_review_materials(ledger, ginfo, g)
    return ledger["gates"][required_gate]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monster SOP Audit Ledger & Hard Gate State Machine")
    parser.add_argument("--monster-id", required=True, help="Monster identifier")
    parser.add_argument("--project-root", default=None, help="Project root directory")
    parser.add_argument("--drive-root", default=None, help="Google Drive MonsterStudio root directory")
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("status", help="Show current stage state, studio config, fingerprints, and gate statuses")

    p_wait = sub.add_parser("await-review", help="Register a completed review pack and enter awaiting_*_review")
    p_wait.add_argument("--gate", required=True, choices=VALID_GATES)
    p_wait.add_argument("--review-pack", required=True, help="Path to generated review HTML or pack directory")
    p_wait.add_argument("--rig-meta", default=None, help="Path to studio_rig_meta.json (required for G1)")

    p_app = sub.add_parser("approve", help="Record explicit user approval for a gate")
    p_app.add_argument("--gate", required=True, choices=VALID_GATES)
    p_app.add_argument("--quote", required=True, help="Exact user feedback quote")
    p_app.add_argument("--candidate-id", default=None, help="Selected candidate ID (required for G1)")
    p_app.add_argument("--rig-meta", default=None, help="Path to selected candidate studio_rig_meta.json")
    p_app.add_argument("--seed", type=int, default=None, help="Locked random seed (required for G2)")
    p_app.add_argument("--checkpoint", default=None, help="Path to dinov2_head.pt (for G3)")

    p_rev = sub.add_parser("request-changes", help="Record user revision request and trigger graded invalidation")
    p_rev.add_argument("--gate", required=True, choices=VALID_GATES)
    p_rev.add_argument("--quote", required=True, help="Exact user feedback quote")
    p_rev.add_argument(
        "--change-type",
        default=None,
        choices=(
            "u1_input_defect",
            "skeleton_topology_defect",
            "local_model_or_bone_naming_change",
            "anatomical_reader_v3_migration",
            "sampler_script_or_seed_change",
            "pose_generalization_insufficient",
            "resolution_or_view_mode_change",
            "dense_surface_enabled_0_to_1",
            "reader_label_contract_change",
            "dense_surface_disabled_1_to_0",
            "mocap_priors_change",
            "exam_video_or_ik_fix",
            "action_prompt_or_seed_change",
        ),
    )

    p_assert = sub.add_parser("assert-gate", help="Fail with non-zero exit code if required gate is not approved")
    p_assert.add_argument("--require", required=True, choices=VALID_GATES)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    ledger = load_ledger(args.monster_id, args.project_root, args.drive_root)
    # Always check if studio environment flags drifted
    derive_invalidations(ledger, new_studio_config=current_studio_config_snapshot())

    if args.action == "status":
        paths = save_ledger(ledger, args.project_root, args.drive_root)
        print(json.dumps({"ledger": ledger, "saved_to": paths}, indent=2, ensure_ascii=False))
        return 0

    if args.action == "await-review":
        materials = {}
        if args.rig_meta:
            materials["rig_meta_path"] = args.rig_meta
        mark_awaiting_review(ledger, args.gate, review_pack_path=args.review_pack, materials=materials)
        paths = save_ledger(ledger, args.project_root, args.drive_root)
        print(json.dumps({"gate": args.gate, "state": ledger["gates"][args.gate], "saved_to": paths}, indent=2, ensure_ascii=False))
        return 0

    if args.action == "approve":
        record_user_decision(
            ledger,
            args.gate,
            decision="approved",
            user_quote=args.quote,
            selected_candidate_id=args.candidate_id,
            rig_meta_path=args.rig_meta,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
        )
        paths = save_ledger(ledger, args.project_root, args.drive_root)
        print(json.dumps({"gate": args.gate, "state": ledger["gates"][args.gate], "saved_to": paths}, indent=2, ensure_ascii=False))
        return 0

    if args.action == "request-changes":
        record_user_decision(
            ledger,
            args.gate,
            decision="changes_requested",
            user_quote=args.quote,
        )
        inv = derive_invalidations(ledger, change_type=args.change_type) if args.change_type else []
        paths = save_ledger(ledger, args.project_root, args.drive_root)
        print(json.dumps({"gate": args.gate, "invalidated": inv, "saved_to": paths}, indent=2, ensure_ascii=False))
        return 0

    if args.action == "assert-gate":
        try:
            ginfo = assert_gate(ledger, args.require)
            print(json.dumps({"passed": True, "gate": args.require, "state": ginfo}, indent=2, ensure_ascii=False))
            return 0
        except RuntimeError as exc:
            print(json.dumps({"passed": False, "gate": args.require, "error": str(exc)}, indent=2, ensure_ascii=False))
            return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
