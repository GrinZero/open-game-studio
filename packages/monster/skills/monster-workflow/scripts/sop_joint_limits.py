#!/usr/bin/env python3
"""sop_joint_limits.py — Post-G1 Geometric & LBS Axis Probe and Agent-Authored `joint_limits.json` Manager.

Bridges Stage 2 (`G1` skeleton approval) and Stage 3 (`G2` 50-pose boundary-exploration sampling):
  1. `probe`: Runs `probe_rig_joint_limits` on the G1-approved candidate rig (`candidate_<id>_rig_meta.json`
     and `candidate_<id>_full.blend`), measuring world-space motion semantics for each bone's local
     `+X/+Y/+Z` rotation axes and LBS mesh edge-stretch safe limits, and writes
     `scratch/work/<monster_id>/joint_limits.json` (`monster-joint-limits-v1`).
  2. `apply`: Applies Agent-authored custom joint angle limits (`--set-bone`, `--set-root-*`, or
     `--from-json`) to `joint_limits.json` and synchronizes `limits_rad` into `candidate_<id>_rig_meta.json`
     without invalidating `G1` (only invalidating `G2`/`G3` if limits changed).
  3. `show`: Displays the active `joint_limits.json` profile and axis probe table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sop_gate import (  # noqa: E402
    assert_gate,
    resolve_g1_candidate_files,
    derive_invalidations,
    load_ledger,
    now_taipei_iso,
    save_ledger,
    sha256_file,
)
from bone_taxonomy import (  # noqa: E402
    JOINT_LIMITS_SCHEMA,
    apply_joint_limits_profile,
    probe_rig_joint_limits,
    validate_canonical_rig_bone_names,
)
from sop_preflight import (  # noqa: E402
    _worker_command,
    resolve_scratch_dirs,
)
from sop_formal_gate import verify_rig_formal_gate  # noqa: E402
from rig_skinning_artifact import load_probe_skinning  # noqa: E402


def resolve_joint_limits_path(project_root: Path, monster_id: str) -> Path:
    dirs = resolve_scratch_dirs(project_root, monster_id)
    return dirs["work_dir"] / "joint_limits.json"


def _resolve_g1_candidate_files(
    project_root: Path,
    monster_id: str,
    g1_info: dict[str, Any],
) -> tuple[str, Path, Path]:
    work_dir = resolve_scratch_dirs(project_root, monster_id)["work_dir"]
    return resolve_g1_candidate_files(work_dir, g1_info)


def probe_joint_limits(
    monster_id: str,
    *,
    archetype: str = "humanoid",
    project_root: Path | str | None = None,
    drive_root: Path | str | None = None,
) -> dict[str, Any]:
    """Runs the Geometric & LBS Axis Probe on the G1-approved candidate rig and writes `joint_limits.json`."""
    proj = Path(project_root or os.environ.get("MONSTER_PROJECT_ROOT") or Path.cwd()).resolve()
    ledger = load_ledger(monster_id, project_root=proj, drive_root=drive_root)
    g1_info = assert_gate(ledger, "G1")
    selected_cid, meta_path, blend_path = _resolve_g1_candidate_files(proj, monster_id, g1_info)

    rig_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    val = validate_canonical_rig_bone_names(rig_meta.get("bones", {}))
    if not val["valid"]:
        raise RuntimeError(f"关节限制探测阻断：选定候选骨骼未通过规范命名校验 ({val['non_canonical_bones']})")

    limits_path = resolve_joint_limits_path(proj, monster_id)
    limits_path.parent.mkdir(parents=True, exist_ok=True)

    worker_cmd = _worker_command(
        [
            "--mode",
            "probe_joint_limits",
            "--monster_id",
            monster_id,
            "--archetype",
            archetype,
            "--blend_path",
            str(blend_path),
            "--meta_path",
            str(meta_path),
            "--limits_path",
            str(limits_path),
        ]
    )
    ran_blender_probe = False
    if worker_cmd and blend_path.is_file():
        hdr = blend_path.read_bytes()[:16]
        if hdr.startswith(b"BLENDER") and b"G1-CANDIDATE-" not in hdr:
            proc = subprocess.run(worker_cmd, check=False)
            ran_blender_probe = proc.returncode == 0 and limits_path.is_file()

    if ran_blender_probe:
        profile = json.loads(limits_path.read_text(encoding="utf-8"))
    else:
        profile = probe_rig_joint_limits(
            rig_meta.get("bones", {}),
            monster_id=monster_id,
            archetype=archetype,
            bbox_min=rig_meta.get("bbox_min"),
            bbox_max=rig_meta.get("bbox_max"),
        )
        profile["selected_candidate_id"] = selected_cid
        profile["generated_at"] = now_taipei_iso()
        limits_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    jl_sha = sha256_file(limits_path)
    derive_invalidations(ledger, new_fingerprints={"joint_limits_sha256": jl_sha})
    save_ledger(ledger, project_root=proj, drive_root=drive_root)

    formal_gate = verify_rig_formal_gate(
        rig_meta.get("bones", {}), profile,
        rig_skinning=load_probe_skinning(profile, blend_path, rig_meta.get("bones", {})),
        assert_passed=True,
    )
    checks_dir = resolve_scratch_dirs(proj, monster_id)["checks_dir"]
    checks_dir.mkdir(parents=True, exist_ok=True)
    (checks_dir / "formal_gate_rig.json").write_text(
        json.dumps(formal_gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return {
        "monster_id": monster_id,
        "selected_candidate_id": selected_cid,
        "schema": profile.get("schema", JOINT_LIMITS_SCHEMA),
        "joint_limits_path": str(limits_path),
        "joint_limits_sha256": jl_sha,
        "num_bones": len(profile.get("bones", {})),
        "root_motion": profile.get("root_motion"),
        "coordination_present": bool(profile.get("coordination")),
        "inter_bone_couplings_count": len((profile.get("coordination") or {}).get("inter_bone_couplings", [])),
        "formal_gate_passed": bool(formal_gate.get("passed", False)),
    }


def _parse_range_pair(spec: str) -> list[float]:
    parts = [p.strip() for p in spec.split(":")]
    if len(parts) != 2:
        raise ValueError(f"Expected range 'lo:hi' in degrees, got {spec!r}")
    lo, hi = float(parts[0]), float(parts[1])
    if lo > hi:
        lo, hi = hi, lo
    return [round(lo, 2), round(hi, 2)]


def _parse_bone_limit_spec(spec: str) -> tuple[str, dict[str, list[float]]]:
    """Parses `BONE_NAME=rx_lo:rx_hi,ry_lo:ry_hi,rz_lo:rz_hi` in degrees."""
    if "=" not in spec:
        raise ValueError(f"Expected BONE_NAME=rx_lo:rx_hi,ry_lo:ry_hi,rz_lo:rz_hi, got {spec!r}")
    b_name, rhs = spec.split("=", 1)
    axes = [a.strip() for a in rhs.split(",")]
    if len(axes) != 3:
        raise ValueError(f"Expected 3 comma-separated axis ranges (rx,ry,rz) for {b_name!r}, got {rhs!r}")
    return b_name.strip(), {
        "rx": _parse_range_pair(axes[0]),
        "ry": _parse_range_pair(axes[1]),
        "rz": _parse_range_pair(axes[2]),
    }


def apply_custom_joint_limits(
    monster_id: str,
    *,
    set_bone_specs: list[str] | None = None,
    set_root_pitch: str | None = None,
    set_root_roll: str | None = None,
    set_root_yaw: str | None = None,
    set_root_trans: str | None = None,
    from_json: Path | str | None = None,
    notes: str | None = None,
    allow_exceed_ceiling: bool = False,
    project_root: Path | str | None = None,
    drive_root: Path | str | None = None,
) -> dict[str, Any]:
    """Applies Agent-customized joint limits to `joint_limits.json` and `candidate_<id>_rig_meta.json`,
    preserving G1 approval while invalidating G2/G3 if `joint_limits_sha256` changed.
    """
    proj = Path(project_root or os.environ.get("MONSTER_PROJECT_ROOT") or Path.cwd()).resolve()
    ledger = load_ledger(monster_id, project_root=proj, drive_root=drive_root)
    g1_info = assert_gate(ledger, "G1")
    selected_cid, meta_path, _blend_path = _resolve_g1_candidate_files(proj, monster_id, g1_info)

    limits_path = resolve_joint_limits_path(proj, monster_id)
    if not limits_path.is_file():
        probe_joint_limits(monster_id, project_root=proj, drive_root=drive_root)

    if from_json:
        src_path = Path(from_json).expanduser().resolve()
        profile = json.loads(src_path.read_text(encoding="utf-8"))
    else:
        profile = json.loads(limits_path.read_text(encoding="utf-8"))

    bones_prof = profile.setdefault("bones", {})
    updated_bones: list[str] = []

    for spec in set_bone_specs or []:
        b_name, new_lims_deg = _parse_bone_limit_spec(spec)
        if b_name not in bones_prof:
            raise KeyError(f"Bone {b_name!r} not found in joint_limits.json ({sorted(bones_prof.keys())})")
        b_entry = bones_prof[b_name]
        ceil_deg = b_entry.get("safe_lbs_ceiling_deg", {})
        if not allow_exceed_ceiling and isinstance(ceil_deg, dict):
            for ax in ("rx", "ry", "rz"):
                if ax in ceil_deg and isinstance(ceil_deg[ax], (list, tuple)) and len(ceil_deg[ax]) == 2:
                    c_lo, c_hi = float(ceil_deg[ax][0]), float(ceil_deg[ax][1])
                    new_lims_deg[ax] = [
                        round(max(c_lo, float(new_lims_deg[ax][0])), 2),
                        round(min(c_hi, float(new_lims_deg[ax][1])), 2),
                    ]
        if b_entry.get("is_leaf") or b_entry.get("kind") == "rigid":
            new_lims_deg["ry"] = [0.0, 0.0]
        if b_entry.get("kind") == "rigid":
            new_lims_deg = {"rx": [0.0, 0.0], "ry": [0.0, 0.0], "rz": [0.0, 0.0]}
        b_entry["limits_deg"] = new_lims_deg
        b_entry["limits_deg_list"] = [
            list(new_lims_deg["rx"]),
            list(new_lims_deg["ry"]),
            list(new_lims_deg["rz"]),
        ]
        b_entry["limits_rad"] = [
            [round(math.radians(new_lims_deg["rx"][0]), 4), round(math.radians(new_lims_deg["rx"][1]), 4)],
            [round(math.radians(new_lims_deg["ry"][0]), 4), round(math.radians(new_lims_deg["ry"][1]), 4)],
            [round(math.radians(new_lims_deg["rz"][0]), 4), round(math.radians(new_lims_deg["rz"][1]), 4)],
        ]
        b_entry["source"] = "agent_custom"
        if notes:
            b_entry["agent_rationale"] = notes
        updated_bones.append(b_name)

    root_motion = profile.setdefault(
        "root_motion",
        {
            "global_pitch_deg": [-28.0, 32.0],
            "global_roll_deg": [-25.0, 25.0],
            "global_yaw_deg": [-22.0, 22.0],
            "translation_fraction_xyz": [0.08, 0.08, 0.10],
        },
    )
    if set_root_pitch:
        root_motion["global_pitch_deg"] = _parse_range_pair(set_root_pitch)
    if set_root_roll:
        root_motion["global_roll_deg"] = _parse_range_pair(set_root_roll)
    if set_root_yaw:
        root_motion["global_yaw_deg"] = _parse_range_pair(set_root_yaw)
    if set_root_trans:
        t_parts = [float(x.strip()) for x in set_root_trans.split(",")]
        if len(t_parts) != 3:
            raise ValueError(f"Expected 3 comma-separated fractions for --set-root-trans, got {set_root_trans!r}")
        root_motion["translation_fraction_xyz"] = [round(max(0.0, x), 4) for x in t_parts]

    if isinstance(profile.get("global_root"), dict):
        # V3 SO(3) profiles intentionally omit the legacy Euler root box.
        if all(key in root_motion for key in ("global_pitch_deg", "global_yaw_deg", "global_roll_deg")):
            profile["global_root"]["rotation_limits_deg"] = [
                list(root_motion["global_pitch_deg"]),
                list(root_motion["global_yaw_deg"]),
                list(root_motion["global_roll_deg"]),
            ]
        profile["global_root"]["translation_fraction_xyz"] = list(root_motion["translation_fraction_xyz"])
    # Fractions and absolute bounds are alternative consumers of the same
    # envelope. Never leave an old absolute clamp after changing its fraction.
    if set_root_trans:
        bmin, bmax = profile.get("bbox_min"), profile.get("bbox_max")
        if not (isinstance(bmin, list) and isinstance(bmax, list) and len(bmin) == len(bmax) == 3):
            raise ValueError("Updating root translation requires the profile's three-dimensional bounding box")
        absolute = [
            round(max(0.0, float(bmax[i]) - float(bmin[i])) * root_motion["translation_fraction_xyz"][i], 10)
            for i in range(3)
        ]
        root_motion["max_translation_world_xyz"] = absolute
        profile.setdefault("global_root", {})["max_translation_world_xyz"] = list(absolute)

    # Sync ROOT bone limits with root_motion pitch/yaw/roll if ROOT bone exists and wasn't explicitly overridden
    for b_name, b_entry in bones_prof.items():
        if b_entry.get("kind") == "root" and b_name not in updated_bones and any(
            x is not None for x in (set_root_pitch, set_root_roll, set_root_yaw)
        ):
            p_deg = root_motion["global_pitch_deg"]
            r_deg = root_motion["global_roll_deg"]
            y_deg = root_motion["global_yaw_deg"]
            b_entry["limits_deg"] = {"rx": list(p_deg), "ry": list(y_deg), "rz": list(r_deg)}
            b_entry["limits_deg_list"] = [list(p_deg), list(y_deg), list(r_deg)]
            b_entry["limits_rad"] = [
                [round(math.radians(p_deg[0]), 4), round(math.radians(p_deg[1]), 4)],
                [round(math.radians(y_deg[0]), 4), round(math.radians(y_deg[1]), 4)],
                [round(math.radians(r_deg[0]), 4), round(math.radians(r_deg[1]), 4)],
            ]
            b_entry["source"] = "agent_custom"
            updated_bones.append(b_name)

    profile["schema"] = JOINT_LIMITS_SCHEMA
    profile["updated_at"] = now_taipei_iso()
    if notes:
        profile["agent_notes"] = notes

    limits_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    jl_sha = sha256_file(limits_path)
    invalidated = derive_invalidations(ledger, new_fingerprints={"joint_limits_sha256": jl_sha})
    save_ledger(ledger, project_root=proj, drive_root=drive_root)

    formal_gate_passed = True
    if meta_path.is_file():
        rig_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        formal_gate = verify_rig_formal_gate(
            rig_meta.get("bones", {}), profile,
            rig_skinning=load_probe_skinning(profile, _blend_path, rig_meta.get("bones", {})),
            assert_passed=True,
        )
        formal_gate_passed = bool(formal_gate.get("passed", False))
        checks_dir = resolve_scratch_dirs(proj, monster_id)["checks_dir"]
        checks_dir.mkdir(parents=True, exist_ok=True)
        (checks_dir / "formal_gate_rig.json").write_text(
            json.dumps(formal_gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    return {
        "monster_id": monster_id,
        "selected_candidate_id": selected_cid,
        "joint_limits_path": str(limits_path),
        "joint_limits_sha256": jl_sha,
        "updated_bones": updated_bones,
        "root_motion": profile.get("root_motion"),
        "coordination_present": bool(profile.get("coordination")),
        "inter_bone_couplings_count": len((profile.get("coordination") or {}).get("inter_bone_couplings", [])),
        "invalidated_gates": invalidated,
        "formal_gate_passed": formal_gate_passed,
    }


def show_joint_limits(
    monster_id: str,
    *,
    project_root: Path | str | None = None,
) -> dict[str, Any]:
    proj = Path(project_root or os.environ.get("MONSTER_PROJECT_ROOT") or Path.cwd()).resolve()
    limits_path = resolve_joint_limits_path(proj, monster_id)
    if not limits_path.is_file():
        raise FileNotFoundError(f"joint_limits.json not found for {monster_id!r}: {limits_path}")
    return json.loads(limits_path.read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monster SOP Post-G1 Joint Limits Probe & Customizer")
    parser.add_argument("--monster-id", required=True)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--drive-root", default=None)
    sub = parser.add_subparsers(dest="action", required=True)

    p_probe = sub.add_parser("probe", help="Run geometric & LBS axis probe and generate initial joint_limits.json")
    p_probe.add_argument("--archetype", default="humanoid")

    p_apply = sub.add_parser("apply", help="Apply Agent-authored custom bone/root limits to joint_limits.json")
    p_apply.add_argument(
        "--set-bone",
        action="append",
        default=[],
        help="Bone limit override: BONE_NAME=rx_lo:rx_hi,ry_lo:ry_hi,rz_lo:rz_hi (in degrees)",
    )
    p_apply.add_argument("--set-root-pitch", default=None, help="Global root pitch range lo:hi in degrees")
    p_apply.add_argument("--set-root-roll", default=None, help="Global root roll range lo:hi in degrees")
    p_apply.add_argument("--set-root-yaw", default=None, help="Global root yaw range lo:hi in degrees")
    p_apply.add_argument("--set-root-trans", default=None, help="Root translation fractions dx,dy,dz of rig extent")
    p_apply.add_argument("--from-json", default=None, help="Load complete joint_limits.json from path")
    p_apply.add_argument("--notes", default=None, help="Agent rationale notes")
    p_apply.add_argument("--allow-exceed-ceiling", action="store_true")

    sub.add_parser("show", help="Print current joint_limits.json")
    return parser


def _normalize_negative_range_argv(argv: list[str] | None) -> list[str] | None:
    """Allows `--set-root-pitch "-24:24"` (where `-24:24` starts with `-` and contains `:`)
    without Python `argparse` mistaking `-24:24` for a short option flag.
    """
    if argv is None:
        return None
    range_flags = {"--set-root-pitch", "--set-root-roll", "--set-root-yaw", "--set-root-trans", "--set-bone"}
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in range_flags and (i + 1) < len(argv) and argv[i + 1].startswith("-"):
            out.append(f"{tok}={argv[i + 1]}")
            i += 2
        else:
            out.append(tok)
            i += 1
    return out


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(_normalize_negative_range_argv(list(sys.argv[1:] if argv is None else argv)))
    if args.action == "probe":
        res = probe_joint_limits(
            args.monster_id,
            archetype=args.archetype,
            project_root=args.project_root,
            drive_root=args.drive_root,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    if args.action == "apply":
        res = apply_custom_joint_limits(
            args.monster_id,
            set_bone_specs=args.set_bone,
            set_root_pitch=args.set_root_pitch,
            set_root_roll=args.set_root_roll,
            set_root_yaw=args.set_root_yaw,
            set_root_trans=args.set_root_trans,
            from_json=args.from_json,
            notes=args.notes,
            allow_exceed_ceiling=bool(args.allow_exceed_ceiling),
            project_root=args.project_root,
            drive_root=args.drive_root,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    if args.action == "show":
        res = show_joint_limits(args.monster_id, project_root=args.project_root)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
