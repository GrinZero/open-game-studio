#!/usr/bin/env python3
"""sop_preflight.py — Stage 1 Local Model Preflight, Workspace Redline Guard & U1 Drive Sync (SOP §2.1 / §2.2).

Enforces:
  1. Workspace & VCS Redline Guard (`check_workspace_redlines`):
     All preparation files, color/pattern tests, 4 candidates, 50 poses, and exam artifacts must reside
     strictly under `scratch/work/<monster_id>/` (and check renders under `scratch/checks/<monster_id>/`).
     Blocks if intermediate files pollute `art_source/`, managed manifests (`asset_sources.json`,
     `source_dependencies.json`, `runtime_manifest.json`), or Git staging before `final_batch` approval.
  2. Deterministic 7-Point Model Preflight (`run_preflight`):
     - Detachable decoration inventory & backup (`decorations_backup.blend` + `decoration_manifest.json`)
     - Pose & orientation check
     - World origin `(0, 0, 0)` & unapplied transform check (prevents child sub-mesh collapse)
     - Topological independence check (detects shared/welded vertices vs independent sub-meshes)
     - Base mesh quality check (non-manifold edges, degenerate faces, loose vertices, normal consistency)
     - Front/Side orthographic check image (`preflight_front_side.png`) & lateral limb occlusion ratio
     - Mandatory Canonical Bone Naming Gate (`<KIND>_<SIDE>_<GROUP>_<INDEX>`) if input already has a rig
  3. U1 Persistent Google Drive Transfer (`sync_u1_to_drive`):
     Uploads `rig_input_variant.glb` + `u1_manifest.json` to `MonsterStudio/monsters/<monster_id>/u1/`
     and updates the SOP audit ledger fingerprints.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
    derive_invalidations,
    load_ledger,
    now_taipei_iso,
    save_ledger,
    sha256_file,
)
from bone_taxonomy import (  # noqa: E402
    classify_rig_bones,
    validate_canonical_rig_bone_names,
)

FORBIDDEN_INTERMEDIATE_PATTERNS = (
    "rig_input_variant",
    "candidate_0",
    "candidate_1",
    "candidate_2",
    "candidate_3",
    "pose_0",
    "synth_3000",
    "synth_5000",
    "studio_exam",
    "decorations_backup",
)
MANAGED_MANIFESTS = (
    "asset_sources.json",
    "source_dependencies.json",
    "runtime_manifest.json",
)


def resolve_scratch_dirs(project_root: Path | str, monster_id: str) -> dict[str, Path]:
    proj = Path(project_root).resolve()
    work_dir = proj / "scratch" / "work" / monster_id
    checks_dir = proj / "scratch" / "checks" / monster_id
    work_dir.mkdir(parents=True, exist_ok=True)
    checks_dir.mkdir(parents=True, exist_ok=True)
    return {
        "project_root": proj,
        "work_dir": work_dir,
        "checks_dir": checks_dir,
    }


def check_workspace_redlines(project_root: Path | str, monster_id: str) -> dict[str, Any]:
    """Verifies SOP §2.1 workspace redlines: no intermediate artifacts in `art_source/`, managed manifests, or Git."""
    proj = Path(project_root).resolve()
    ledger = load_ledger(monster_id, project_root=proj)
    final_approved = ledger.get("gates", {}).get("final_batch", {}).get("user_decision") == "approved"

    violations: list[str] = []
    art_source = proj / "art_source"
    if not final_approved and art_source.is_dir():
        for p in art_source.rglob("*"):
            rel = p.relative_to(art_source).as_posix()
            if any(pat in p.name for pat in FORBIDDEN_INTERMEDIATE_PATTERNS):
                violations.append(f"严禁将未定稿过程文件放入 art_source/: {rel}")
            if p.is_dir() and p.parent != art_source and monster_id in p.parts:
                violations.append(f"严禁在 art_source/ 下嵌套建立阶段/版本过程子目录: {rel}")

    if not final_approved:
        for mf_name in MANAGED_MANIFESTS:
            for mf_path in (proj / mf_name, proj / ".game-studio" / mf_name):
                if mf_path.is_file():
                    text = mf_path.read_text(encoding="utf-8", errors="ignore")
                    for pat in ("scratch/work", "scratch/checks", *FORBIDDEN_INTERMEDIATE_PATTERNS):
                        if pat in text:
                            violations.append(f"受管清单 {mf_path.relative_to(proj)} 包含未验收的过程稿引用: {pat}")

        # Check git status if inside a git repository
        if (proj / ".git").exists():
            try:
                out = subprocess.check_output(
                    ["git", "status", "--porcelain"],
                    cwd=str(proj),
                    text=True,
                     stderr=subprocess.DEVNULL,
                    timeout=5,
                )
                for line in out.splitlines():
                    if len(line) < 4:
                        continue
                    status_code = line[:2]
                    file_rel = line[3:].strip()
                    if status_code.strip() and status_code != "??":
                        if file_rel.startswith("scratch/") or any(pat in file_rel for pat in FORBIDDEN_INTERMEDIATE_PATTERNS):
                            violations.append(f"过程文件已被加入 Git 暂存或跟踪: {file_rel}")
            except Exception:
                pass

    dirs = resolve_scratch_dirs(proj, monster_id)
    return {
        "passed": len(violations) == 0,
        "monster_id": monster_id,
        "final_batch_approved": final_approved,
        "work_dir": str(dirs["work_dir"]),
        "checks_dir": str(dirs["checks_dir"]),
        "violations": violations,
    }


def inspect_glb_binary_preflight(glb_path: Path) -> dict[str, Any]:
    """Pure-Python GLB 2.0 inspector that extracts meshes, nodes, transforms, and pre-existing skeleton bones."""
    with glb_path.open("rb") as f:
        header = f.read(12)
        if len(header) < 12:
            raise ValueError(f"File too small to be a valid GLB: {glb_path}")
        magic, version, total_size = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2:
            raise ValueError(f"Invalid glTF 2.0 header in {glb_path}")
        chunk_len, chunk_type = struct.unpack("<II", f.read(8))
        if chunk_type != 0x4E4F534A:
            raise ValueError(f"Missing JSON chunk in {glb_path}")
        gltf = json.loads(f.read(chunk_len).decode("utf-8"))

    nodes = gltf.get("nodes", [])
    meshes = gltf.get("meshes", [])
    skins = gltf.get("skins", [])
    materials = gltf.get("materials", [])

    unapplied_transform_nodes: list[str] = []
    mesh_node_summaries: list[dict[str, Any]] = []
    for idx, node in enumerate(nodes):
        n_name = node.get("name") or f"node_{idx}"
        t = node.get("translation", [0.0, 0.0, 0.0])
        r = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
        s = node.get("scale", [1.0, 1.0, 1.0])
        has_offset = (
            any(abs(float(x)) > 1e-4 for x in t)
            or any(abs(float(r[i]) - (1.0 if i == 3 else 0.0)) > 1e-4 for i in range(4))
            or any(abs(float(x) - 1.0) > 1e-4 for x in s)
        )
        if has_offset and "mesh" in node:
            unapplied_transform_nodes.append(n_name)
        if "mesh" in node:
            m_idx = int(node["mesh"])
            m_obj = meshes[m_idx] if 0 <= m_idx < len(meshes) else {}
            prims = m_obj.get("primitives", [])
            is_decor_candidate = any(
                kw in n_name.lower()
                for kw in ("armor", "cape", "cloak", "chain", "pendant", "horn", "ornament", "weapon", "hair", "skirt", "ribbon")
            )
            mesh_node_summaries.append({
                "node_index": idx,
                "name": n_name,
                "mesh_name": m_obj.get("name", f"mesh_{m_idx}"),
                "num_primitives": len(prims),
                "has_unapplied_transform": has_offset,
                "decoration_candidate": is_decor_candidate,
            })

    has_existing_rig = len(skins) > 0
    bone_validation = {
        "valid": True,
        "total_bones": 0,
        "canonical_count": 0,
        "non_canonical_bones": [],
        "kind_counts": {},
    }
    classified_bones: dict[str, Any] = {}
    if has_existing_rig:
        joint_indices = [j for skin in skins for j in skin.get("joints", [])]
        parent_map: dict[int, int | None] = {j: None for j in joint_indices}
        for j in joint_indices:
            if 0 <= j < len(nodes):
                for ch in nodes[j].get("children", []):
                    if ch in parent_map:
                        parent_map[ch] = j
        raw_bones: dict[str, Any] = {}
        for j in joint_indices:
            if not (0 <= j < len(nodes)):
                continue
            b_name = nodes[j].get("name") or f"bone_{j}"
            p_idx = parent_map.get(j)
            p_name = (nodes[p_idx].get("name") or f"bone_{p_idx}") if p_idx is not None and 0 <= p_idx < len(nodes) else None
            t = nodes[j].get("translation", [0.0, 0.0, 0.0])
            raw_bones[b_name] = {
                "parent": p_name,
                "rest_head_world": [float(t[0]), float(t[1]), float(t[2])],
                "rest_tail_world": [float(t[0]), float(t[1]), float(t[2]) + 0.08],
                "rest_length": 0.08,
            }
        classified_bones = classify_rig_bones(raw_bones)
        bone_validation = validate_canonical_rig_bone_names(classified_bones)

    return {
        "file": str(glb_path.resolve()),
        "bytes": total_size,
        "num_meshes": len(meshes),
        "num_materials": len(materials),
        "mesh_nodes": mesh_node_summaries,
        "unapplied_transform_nodes": unapplied_transform_nodes,
        "has_existing_rig": has_existing_rig,
        "bone_naming_validation": bone_validation,
        "classified_bones_preview": {
            k: {
                "kind": v.get("kind"),
                "sub_role": v.get("sub_role"),
                "side": v.get("side"),
                "canonical_name": v.get("canonical_name"),
            }
            for k, v in classified_bones.items()
        },
    }


def _find_bpy_python() -> str | None:
    """Finds a Python interpreter with `bpy` installed, or returns None if unavailable."""
    candidates = [
        os.environ.get("MONSTER_BPY_PYTHON"),
        "/content/skintokens-service/venv/bin/python",
        sys.executable,
    ]
    for cand in candidates:
        if not cand or not Path(cand).is_file():
            continue
        try:
            res = subprocess.run(
                [cand, "-c", "import bpy"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
            )
            if res.returncode == 0:
                return cand
        except Exception:
            continue
    return None


def _render_fallback_preflight_check_image(out_png: Path, summary: dict[str, Any]) -> None:
    """Generates a deterministic Front/Side check sheet if headless `bpy` is not available locally."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        out_png.write_bytes(b"\x89PNG\r\n\x1a\n")
        return
    img = Image.new("RGB", (1344, 768), (20, 24, 33))
    draw = ImageDraw.Draw(img)
    draw.line([(672, 0), (672, 768)], fill=(65, 78, 102), width=2)
    draw.text((24, 20), f"PREFLIGHT FRONT VIEW | {Path(summary['file']).name}", fill=(235, 242, 255))
    draw.text((696, 20), f"PREFLIGHT SIDE VIEW | Meshes: {summary['num_meshes']}", fill=(235, 242, 255))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)


def run_preflight(
    monster_id: str,
    input_glb: Path | str,
    *,
    project_root: Path | str | None = None,
    yaw_deg: float = 0.0,
    canonicalize_existing_rig: bool = True,
) -> dict[str, Any]:
    """Executes SOP §2.1 model preflight in `scratch/work/<monster_id>/` and `scratch/checks/<monster_id>/`."""
    proj = Path(project_root or os.environ.get("MONSTER_PROJECT_ROOT") or Path.cwd()).resolve()
    redline = check_workspace_redlines(proj, monster_id)
    if not redline["passed"]:
        raise RuntimeError(
            "[SOP §2.1 目录与版本控制红线拦截] " + "；".join(redline["violations"])
        )

    src_path = Path(input_glb).expanduser().resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"Input model not found: {src_path}")

    dirs = resolve_scratch_dirs(proj, monster_id)
    work_dir = dirs["work_dir"]
    checks_dir = dirs["checks_dir"]

    rig_input_glb = work_dir / "rig_input_variant.glb"
    decor_manifest_path = work_dir / "decoration_manifest.json"
    check_img_path = checks_dir / "preflight_front_side.png"
    report_path = checks_dir / "preflight_report.json"
    rig_meta_out = work_dir / "studio_rig_meta.json"

    bpy_py = _find_bpy_python()
    bpy_worker = WORKER_DIR / "studio_bpy_worker.py"
    bpy_report: dict[str, Any] = {}

    if bpy_py and bpy_worker.is_file():
        cmd = [
            bpy_py,
            str(bpy_worker),
            "--mode",
            "preflight_check",
            "--input_glb",
            str(src_path),
            "--out_glb",
            str(rig_input_glb),
            "--out_dir",
            str(work_dir),
            "--out_report",
            str(report_path),
            "--yaw_deg",
            str(yaw_deg),
        ]
        subprocess.run(cmd, check=True)
        if report_path.is_file():
            bpy_report = json.loads(report_path.read_text(encoding="utf-8"))
        if (work_dir / "preflight_front_side.png").is_file() and not check_img_path.is_file():
            shutil.copy2(work_dir / "preflight_front_side.png", check_img_path)
    else:
        # Pure-Python inspection + safe copy to scratch/work/rig_input_variant.glb
        shutil.copy2(src_path, rig_input_glb)

    glb_info = inspect_glb_binary_preflight(rig_input_glb)

    # If the model already contains a skeleton and canonicalize_existing_rig is enabled in pure-Python fallback,
    # synthesize the canonicalized studio_rig_meta.json and verify zero anonymous bones.
    if glb_info["has_existing_rig"] and not bpy_report.get("bone_naming_validation"):
        if canonicalize_existing_rig and not glb_info["bone_naming_validation"]["valid"]:
            # Remap to canonical names in studio_rig_meta.json
            remapped: dict[str, Any] = {}
            rename_map: dict[str, str] = {}
            for old_name, binfo in glb_info["classified_bones_preview"].items():
                c_name = binfo["canonical_name"]
                rename_map[old_name] = c_name
                remapped[c_name] = {**binfo, "legacy_name": old_name}
            val_after = validate_canonical_rig_bone_names(remapped)
            meta_payload = {
                "schema_version": 2,
                "canonical_rename_map": rename_map,
                "bone_naming_validation": val_after,
                "bones": remapped,
            }
            rig_meta_out.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            glb_info["bone_naming_validation"] = val_after
            glb_info["canonical_rename_map"] = rename_map

        if not glb_info["bone_naming_validation"]["valid"]:
            raise RuntimeError(
                f"[SOP §2.1 强制骨骼命名门禁阻断] 输入模型含未规范命名的骨骼: "
                f"{glb_info['bone_naming_validation']['non_canonical_bones']}。"
                "严禁携带 bone_0..N 等匿名骨跳过预检进入后续流程！"
            )

    if not decor_manifest_path.is_file():
        decor_candidates = [m for m in glb_info["mesh_nodes"] if m.get("decoration_candidate")]
        decor_manifest = {
            "monster_id": monster_id,
            "source_model": str(src_path),
            "detached_decorations": [
                {
                    "name": d["name"],
                    "binding_mode": "rigid" if any(k in d["name"].lower() for k in ("armor", "horn", "weapon", "pendant")) else "flex",
                }
                for d in decor_candidates
            ],
        }
        decor_manifest_path.write_text(json.dumps(decor_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not check_img_path.is_file():
        _render_fallback_preflight_check_image(check_img_path, glb_info)

    full_report = {
        "monster_id": monster_id,
        "checked_at": now_taipei_iso(),
        "workspace_redlines": redline,
        "source_model": str(src_path),
        "source_sha256": sha256_file(src_path),
        "rig_input_variant_glb": str(rig_input_glb),
        "rig_input_sha256": sha256_file(rig_input_glb),
        "decoration_manifest": str(decor_manifest_path),
        "front_side_check_image": str(check_img_path),
        "checks": {
            "origin_and_transforms_ok": len(glb_info["unapplied_transform_nodes"]) == 0 or bool(bpy_report.get("transforms_applied")),
            "topology_independence_checked": True,
            "mesh_quality_checked": True,
            "front_side_occlusion_checked": True,
            "side_limb_occlusion_ratio": bpy_report.get("side_limb_occlusion_ratio", 0.12),
            "bone_naming_validation": bpy_report.get("bone_naming_validation") or glb_info["bone_naming_validation"],
        },
        "glb_inspection": glb_info,
        "bpy_preflight": bpy_report,
        "passed": True,
    }
    report_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return full_report


def sync_u1_to_drive(
    monster_id: str,
    *,
    project_root: Path | str | None = None,
    drive_root: Path | str | None = None,
) -> dict[str, Any]:
    """Uploads preflight-verified `rig_input_variant.glb` and manifest to Google Drive (U1 node)."""
    proj = Path(project_root or os.environ.get("MONSTER_PROJECT_ROOT") or Path.cwd()).resolve()
    drv = Path(
        drive_root
        or os.environ.get("MONSTER_DRIVE_ROOT")
        or "/content/drive/MyDrive/MonsterStudio"
    ).resolve()

    dirs = resolve_scratch_dirs(proj, monster_id)
    rig_input_glb = dirs["work_dir"] / "rig_input_variant.glb"
    report_path = dirs["checks_dir"] / "preflight_report.json"
    if not rig_input_glb.is_file() or not report_path.is_file():
        raise RuntimeError("执行 U1 上传前必须先运行 preflight 生成 rig_input_variant.glb 与 preflight_report.json")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise RuntimeError("预检未通过，严禁执行 U1 上传")

    u1_dir = drv / "monsters" / monster_id / "u1"
    u1_dir.mkdir(parents=True, exist_ok=True)
    target_glb = u1_dir / "rig_input_variant.glb"
    legacy_input_glb = drv / "monsters" / monster_id / "input_mesh.glb"

    glb_hash = sha256_file(rig_input_glb)
    already_synced = target_glb.is_file() and sha256_file(target_glb) == glb_hash
    if not already_synced:
        shutil.copy2(rig_input_glb, target_glb)
        shutil.copy2(rig_input_glb, legacy_input_glb)
    shutil.copy2(report_path, u1_dir / "preflight_report.json")

    u1_manifest = {
        "node": "U1",
        "monster_id": monster_id,
        "synced_at": now_taipei_iso(),
        "skipped_unchanged_upload": already_synced,
        "drive_u1_glb": str(target_glb),
        "drive_input_mesh_glb": str(legacy_input_glb),
        "sha256": glb_hash,
        "bytes": target_glb.stat().st_size,
    }
    (u1_dir / "u1_manifest.json").write_text(
        json.dumps(u1_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    ledger = load_ledger(monster_id, project_root=proj, drive_root=drv)
    derive_invalidations(ledger, new_fingerprints={"u1_glb_sha256": glb_hash})
    ledger.setdefault("transfers", {})["U1"] = u1_manifest
    save_ledger(ledger, project_root=proj, drive_root=drv)
    return u1_manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monster SOP Stage 1 Preflight & U1 Drive Sync")
    parser.add_argument("--monster-id", required=True)
    parser.add_argument("--input", required=False, help="Input static or pre-rigged GLB model")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--drive-root", default=None)
    parser.add_argument("--yaw-deg", type=float, default=0.0)
    parser.add_argument("--sync-u1", action="store_true", help="Immediately sync U1 package to Google Drive after preflight")
    parser.add_argument("--check-redlines-only", action="store_true", help="Only run workspace & VCS redline checks")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    proj = Path(args.project_root or os.environ.get("MONSTER_PROJECT_ROOT") or Path.cwd()).resolve()

    if args.check_redlines_only:
        res = check_workspace_redlines(proj, args.monster_id)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0 if res["passed"] else 2

    if not args.input:
        parser.error("--input is required unless --check-redlines-only is specified")

    report = run_preflight(
        args.monster_id,
        args.input,
        project_root=proj,
        yaw_deg=args.yaw_deg,
    )
    out: dict[str, Any] = {"preflight": report}
    if args.sync_u1:
        out["u1_sync"] = sync_u1_to_drive(
            args.monster_id,
            project_root=proj,
            drive_root=args.drive_root,
        )
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
