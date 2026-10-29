#!/usr/bin/env python3
"""sop_formal_gate.py — Deterministic Formal Verification Gate for Kinematic Solver & Surface Contact Invariants.

Enforces four mathematical invariants as a hard gate across CI (`npm run monster:check`)
and the Monster SOP workflow (`joint-limits`, `g2-pack`, `delivery`, and standalone `formal-gate`):

  - P1 (Manifold Soundness, Joint-Limit Box Compliance & FK Conservation):
      Every output bone rotation lies on SO(3), respects local Euler box limits [l_b, u_b]
      (and 0 for fixed_local/rigid), and conserves exact rest bone length and parent-child head offset.
  - P2 (Oracle Zero-Error Fixed-Point & Boundary Reachability):
      Any noiseless pose synthesized inside the rig's joint limits (neutral, boundary-active
      corners, and random feasible poses) is recovered with sub-millimeter endpoint accuracy
      and sub-degree twist accuracy on twist-identifiable bones.
  - P3 (Non-Destructive Lateral Limb Invariance under Trunk Macro-Collapse):
      Valid horizontal/kneeling lateral forearm_hinge limb observations at rest length are not
      overwritten downward when trunk macro_collapse == 1.0.
  - P4 (Surface Contact Fixed-Point Identity, Idempotence & Local Basin Preservation):
      Non-penetrating poses above floor_z have zero root/joint perturbation, repeated surface
      projection is idempotent within tolerance, and minor distal penetration resolves locally
      without resetting upper-limb joints to mid-bounds.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sop_gate import (  # noqa: E402
    assert_gate,
    load_ledger,
    now_taipei_iso,
    resolve_g1_candidate_files,
)
import numpy as np  # noqa: E402
from sop_preflight import resolve_scratch_dirs  # noqa: E402
import bone_taxonomy as tax  # noqa: E402
from surface_contact import (  # noqa: E402
    apply_surface_floor_constraints,
    assert_surface_contact_quality,
)

FORMAL_GATE_SCHEMA = "monster-formal-verification-gate-v1"


def _v3_bones(bones):
    from bone_taxonomy import parse_anatomical_name
    return bool(bones) and all(parse_anatomical_name(n) for n in bones)


def build_canonical_verification_rig() -> dict[str, dict[str, Any]]:
    """Constructs a canonical 12-bone multi-branch skeleton spanning all 9 solver roles:
    root, spine (axial_flex), clavicle_socket, ball_socket (limb_ball), forearm_hinge,
    hand_hub, distal_limb, fixed_local, and rigid.
    """
    deg = np.deg2rad
    raw_bones = {
        "ROOT": {
            "head": [0.0, 0.0, 0.90],
            "tail": [0.0, 0.0, 1.05],
            "parent": None,
            "kind": "root",
            "sub_role": "root",
            "side": "C",
            "limits_rad": [[-float(deg(26)), float(deg(26))], [-float(deg(22)), float(deg(22))], [-float(deg(20)), float(deg(20))]],
            "limits_source": "agent_custom",
        },
        "SPINE_01": {
            "head": [0.0, 0.0, 1.05],
            "tail": [0.0, 0.02, 1.30],
            "parent": "ROOT",
            "kind": "spine",
            "sub_role": "spine",
            "side": "C",
            "limits_rad": [[-float(deg(20)), float(deg(24))], [-float(deg(15)), float(deg(15))], [-float(deg(16)), float(deg(16))]],
            "limits_source": "agent_custom",
        },
        "HEAD": {
            "head": [0.0, 0.02, 1.30],
            "tail": [0.0, 0.04, 1.50],
            "parent": "SPINE_01",
            "kind": "head",
            "sub_role": "fixed_local",
            "fixed_local": True,
            "side": "C",
            "limits_rad": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "limits_source": "agent_custom",
        },
        "JAW_RIGID": {
            "head": [0.0, 0.08, 1.36],
            "tail": [0.0, 0.14, 1.33],
            "parent": "HEAD",
            "kind": "rigid",
            "sub_role": "rigid",
            "fixed_local": True,
            "side": "C",
            "limits_rad": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            "limits_source": "agent_custom",
        },
        "SHOULDER_L": {
            "head": [0.04, 0.02, 1.26],
            "tail": [0.18, 0.02, 1.26],
            "parent": "SPINE_01",
            "kind": "limb",
            "sub_role": "clavicle_socket",
            "side": "L",
            "limits_rad": [[-float(deg(18)), float(deg(18))], [-float(deg(12)), float(deg(12))], [-float(deg(20)), float(deg(20))]],
            "limits_source": "agent_custom",
        },
        "UPPERARM_L": {
            "head": [0.18, 0.02, 1.26],
            "tail": [0.38, 0.05, 1.06],
            "parent": "SHOULDER_L",
            "kind": "limb",
            "sub_role": "ball_socket",
            "side": "L",
            "limits_rad": [[-float(deg(48)), float(deg(48))], [-float(deg(30)), float(deg(30))], [-float(deg(42)), float(deg(42))]],
            "limits_source": "agent_custom",
        },
        "FOREARM_L": {
            "head": [0.38, 0.05, 1.06],
            "tail": [0.54, 0.16, 0.90],
            "parent": "UPPERARM_L",
            "kind": "limb",
            "sub_role": "forearm_hinge",
            "side": "L",
            "limits_rad": [[-float(deg(25)), float(deg(38))], [-float(deg(12)), float(deg(12))], [-float(deg(20)), float(deg(25))]],
            "limits_source": "agent_custom",
        },
        "HAND_L": {
            "head": [0.54, 0.16, 0.90],
            "tail": [0.61, 0.22, 0.84],
            "parent": "FOREARM_L",
            "kind": "limb",
            "sub_role": "hand_hub",
            "side": "L",
            "limits_rad": [[-float(deg(20)), float(deg(20))], [-float(deg(12)), float(deg(12))], [-float(deg(18)), float(deg(18))]],
            "limits_source": "agent_custom",
        },
        "FINGER_01_L": {
            "head": [0.61, 0.22, 0.84],
            "tail": [0.66, 0.26, 0.80],
            "parent": "HAND_L",
            "kind": "digit",
            "sub_role": "distal_limb",
            "side": "L",
            "limits_rad": [[-float(deg(22)), float(deg(22))], [0.0, 0.0], [-float(deg(15)), float(deg(15))]],
            "limits_source": "agent_custom",
        },
        "THIGH_R": {
            "head": [-0.12, 0.0, 0.90],
            "tail": [-0.14, 0.03, 0.50],
            "parent": "ROOT",
            "kind": "limb",
            "sub_role": "ball_socket",
            "side": "R",
            "limits_rad": [[-float(deg(27)), float(deg(62))], [-float(deg(31)), float(deg(31))], [-float(deg(18)), float(deg(9))]],
            "limits_source": "agent_custom",
        },
        "SHIN_R": {
            "head": [-0.14, 0.03, 0.50],
            "tail": [-0.14, -0.02, 0.12],
            "parent": "THIGH_R",
            "kind": "limb",
            "sub_role": "forearm_hinge",
            "side": "R",
            "limits_rad": [[0.0, float(deg(95))], [-float(deg(10)), float(deg(10))], [-float(deg(7)), float(deg(12))]],
            "limits_source": "agent_custom",
        },
        "FOOT_R": {
            "head": [-0.14, -0.02, 0.12],
            "tail": [-0.14, 0.14, 0.04],
            "parent": "SHIN_R",
            "kind": "limb",
            "sub_role": "distal_limb",
            "side": "R",
            "limits_rad": [[-float(deg(75)), float(deg(36))], [-float(deg(5)), float(deg(12))], [-float(deg(28)), float(deg(16))]],
            "limits_source": "agent_custom",
        },
    }
    for b_name, src in raw_bones.items():
        h = np.asarray(src["head"], dtype=np.float64)
        t = np.asarray(src["tail"], dtype=np.float64)
        src["rest_head_world"] = h.tolist()
        src["rest_tail_world"] = t.tolist()
        src["rest_length"] = float(np.linalg.norm(t - h))
    classified = tax.classify_rig_bones(raw_bones)
    for b_name, src in raw_bones.items():
        classified[b_name]["rest_head_world"] = src["rest_head_world"]
        classified[b_name]["rest_tail_world"] = src["rest_tail_world"]
        classified[b_name]["rest_length"] = src["rest_length"]
        classified[b_name]["kind"] = src["kind"]
        classified[b_name]["sub_role"] = src["sub_role"]
        classified[b_name]["fixed_local"] = bool(src.get("fixed_local", False))
        classified[b_name]["limits_rad"] = src["limits_rad"]
        classified[b_name]["limits_source"] = src.get("limits_source", "agent_custom")
    return classified


def _topo_order(bones: dict[str, dict[str, Any]]) -> list[str]:
    order: list[str] = []
    visited: set[str] = set()

    def visit(b: str) -> None:
        if b in visited or b not in bones:
            return
        p = bones[b].get("parent")
        if p and p in bones:
            visit(p)
        visited.add(b)
        order.append(b)

    for name in bones:
        visit(name)
    return order


def _geodesic_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Numerically well-conditioned geodesic angle on SO(3) in degrees.
    Uses 2 * arcsin(||R1 - R2||_F / sqrt(8)) to avoid the infinite derivative of arccos(1 - eps) at 0.
    """
    fro = float(np.linalg.norm(np.asarray(R1, dtype=np.float64) - np.asarray(R2, dtype=np.float64), ord="fro"))
    return math.degrees(2.0 * math.asin(min(1.0, fro / math.sqrt(8.0))))


def _rest_head_tail(info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    h = info.get("rest_head_world") if "rest_head_world" in info else info.get("head")
    t = info.get("rest_tail_world") if "rest_tail_world" in info else info.get("tail")
    return np.asarray(h, dtype=np.float64), np.asarray(t, dtype=np.float64)


def _compute_rest_rot_matrices(
    classified: dict[str, dict[str, Any]],
    order: list[str],
) -> dict[str, np.ndarray]:
    rest_R: dict[str, np.ndarray] = {}
    for b in order:
        rh, rt = _rest_head_tail(classified[b])
        if "rest_rot_mat_world" in classified[b] and classified[b]["rest_rot_mat_world"] is not None:
            rest_R[b] = np.asarray(classified[b]["rest_rot_mat_world"], dtype=np.float64)
        else:
            v0 = rt - rh
            norm = float(np.linalg.norm(v0))
            rest_R[b] = tax._build_default_rest_rot_matrix(v0 / max(norm, 1e-8))
    return rest_R


def _prepare_classified_bones(
    rig_bones: dict[str, Any] | None,
    joint_limits_profile: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    if rig_bones is None:
        return build_canonical_verification_rig()
    norm_bones = copy.deepcopy(rig_bones)
    for _b_name, info in norm_bones.items():
        if isinstance(info, dict):
            if "rest_head_world" not in info and "head" in info:
                info["rest_head_world"] = list(info["head"])
            if "rest_tail_world" not in info and "tail" in info:
                info["rest_tail_world"] = list(info["tail"])
            if "rest_head_world" in info and "rest_tail_world" in info:
                info["rest_length"] = float(
                    np.linalg.norm(
                        np.asarray(info["rest_tail_world"], dtype=np.float64)
                        - np.asarray(info["rest_head_world"], dtype=np.float64)
                    )
                )
    already_classified = all(
        isinstance(v, dict) and "sub_role" in v
        for v in norm_bones.values()
    ) and len(norm_bones) > 0
    classified = norm_bones if already_classified else tax.classify_rig_bones(norm_bones)
    if isinstance(joint_limits_profile, dict) and isinstance(joint_limits_profile.get("bones"), dict):
        classified = tax.apply_joint_limits_profile(classified, joint_limits_profile)
    else:
        for b_name, src in norm_bones.items():
            if b_name in classified and isinstance(src, dict):
                for key in ("rest_head_world", "rest_tail_world", "rest_length", "kind", "sub_role", "fixed_local", "limits_rad", "limits_source"):
                    if key in src:
                        classified[b_name][key] = src[key]
                classified[b_name].setdefault("limits_source", "probe_initial")
    return classified


def _forward_kinematics_from_local_eulers(
    classified: dict[str, dict[str, Any]],
    local_eulers_rad: dict[str, tuple[float, float, float]],
    root_trans: np.ndarray | None = None,
) -> dict[str, Any]:
    """Synthesizes an exact forward-kinematic frame from local XYZ Euler angles."""
    order = _topo_order(classified)
    rest_R = _compute_rest_rot_matrices(classified, order)
    rest_H = {}
    rest_T = {}
    for b in order:
        rest_H[b], rest_T[b] = _rest_head_tail(classified[b])

    world_R: dict[str, np.ndarray] = {}
    pose_H: dict[str, np.ndarray] = {}
    pose_T: dict[str, np.ndarray] = {}
    t_root = np.zeros(3, dtype=np.float64) if root_trans is None else np.asarray(root_trans, dtype=np.float64)

    bones_dict: dict[str, dict[str, Any]] = {}
    for b in order:
        p = classified[b].get("parent")
        is_fixed = bool(classified[b].get("fixed_local")) or classified[b].get("kind") == "rigid" or classified[b].get("sub_role") in ("fixed_local", "rigid")
        rx, ry, rz = (0.0, 0.0, 0.0) if is_fixed else local_eulers_rad.get(b, (0.0, 0.0, 0.0))
        R_init = rest_R[b] if (p is None or p not in world_R) else world_R[p] @ (rest_R[p].T @ rest_R[b])
        R_delta = tax.euler_xyz_to_matrix_numpy(float(rx), float(ry), float(rz))
        R_b = R_init @ R_delta
        world_R[b] = R_b
        if p and p in pose_H:
            H_b = pose_H[p] + world_R[p] @ (rest_R[p].T @ (rest_H[b] - rest_H[p]))
        else:
            H_b = rest_H[b] + t_root
        T_b = H_b + R_b @ (rest_R[b].T @ (rest_T[b] - rest_H[b]))
        pose_H[b] = H_b
        pose_T[b] = T_b
        bones_dict[b] = {
            "head_3d": H_b.tolist(),
            "tail_3d": T_b.tolist(),
            "rot_mat_world": R_b.tolist(),
        }
    return {"bones": bones_dict}


def verify_trajectory_formal_gate(
    frames: list[dict[str, Any]],
    rig_bones: dict[str, Any],
    joint_limits_profile: dict[str, Any] | None = None,
    *,
    trajectory_meta: dict[str, Any] | None = None,
    action: str = "trajectory",
    so3_tol: float = 1e-3,
    limit_tol_deg: float = 0.20,
    length_tol_m: float = 1e-4,
    offset_tol_m: float = 1e-4,
    assert_passed: bool = True,
) -> dict[str, Any]:
    """Verifies P1 (SO(3) manifold, joint-limit box compliance, exact FK length/offset conservation)
    and P4 (surface contact quality if metadata present) across all frames of a trajectory.
    """
    classified = _prepare_classified_bones(rig_bones, joint_limits_profile)
    v3 = (trajectory_meta or {}).get('reader_contract') == 'monster-reader-v3' or _v3_bones(classified)
    structure = (trajectory_meta or {}).get('motion_structure') or {}
    fitted = set(structure.get('fitted_bones', classified))
    motion_roots = set(structure.get('motion_roots', [n for n,b in classified.items() if b.get('kind') == 'root' or (not b.get('parent') and not b.get('fixed_local'))]))
    if not motion_roots <= fitted <= set(classified) or any(classified[n].get('kind') != 'root' and not n.startswith('Prop_') and classified[n].get('parent') for n in motion_roots):
        raise ValueError('Invalid v3 fitted-channel/motion-root structure')
    order = _topo_order(classified)
    rest_R = _compute_rest_rot_matrices(classified, order)
    rest_H = {}
    rest_T = {}
    rest_len = {}
    for b in order:
        rh, rt = _rest_head_tail(classified[b])
        rest_H[b] = rh
        rest_T[b] = rt
        rest_len[b] = float(np.linalg.norm(rt - rh))

    max_so3_ortho_err = 0.0
    max_so3_det_err = 0.0
    max_limit_violation_deg = 0.0
    max_length_err_m = 0.0
    max_offset_err_m = 0.0
    violations: list[str] = []
    if v3 and not frames:
        violations.append('P1_EMPTY_TRAJECTORY')

    for f_idx, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        bones_map = frame.get("bones") if isinstance(frame.get("bones"), dict) else frame
        world_R: dict[str, np.ndarray] = {}
        pose_H: dict[str, np.ndarray] = {}
        for b in order:
            b_state = bones_map.get(b)
            if not isinstance(b_state, dict):
                if v3: violations.append(f'P1_MISSING_POSE frame={f_idx} bone={b}')
                continue
            h_raw = b_state.get("head_3d") if "head_3d" in b_state else b_state.get("head")
            t_raw = b_state.get("tail_3d") if "tail_3d" in b_state else b_state.get("tail")
            r_raw = b_state.get("rot_mat_world") if "rot_mat_world" in b_state else b_state.get("matrix_3x3")
            if h_raw is None or t_raw is None or r_raw is None:
                if v3: violations.append(f'P1_MISSING_POSE frame={f_idx} bone={b}')
                continue
            H_b = np.asarray(h_raw, dtype=np.float64)
            T_b = np.asarray(t_raw, dtype=np.float64)
            R_b = np.asarray(r_raw, dtype=np.float64)
            if v3 and (H_b.shape != (3,) or T_b.shape != (3,) or R_b.shape != (3,3) or not np.isfinite(H_b).all() or not np.isfinite(T_b).all() or not np.isfinite(R_b).all()):
                violations.append(f'P1_INVALID_POSE frame={f_idx} bone={b}')
                continue
            world_R[b] = R_b
            pose_H[b] = H_b

            # 1. SO(3) manifold check
            ortho_err = float(np.linalg.norm(R_b.T @ R_b - np.eye(3), ord="fro"))
            det_err = abs(float(np.linalg.det(R_b)) - 1.0)
            max_so3_ortho_err = max(max_so3_ortho_err, ortho_err)
            max_so3_det_err = max(max_so3_det_err, det_err)
            if ortho_err > so3_tol or det_err > so3_tol:
                violations.append(
                    f"P1_SO3 frame={f_idx} bone={b}: ortho_err={ortho_err:.2e}, det_err={det_err:.2e}"
                )

            # 2. Exact FK bone-length conservation
            cur_len = float(np.linalg.norm(T_b - H_b))
            len_err = abs(cur_len - rest_len[b])
            max_length_err_m = max(max_length_err_m, len_err)
            if len_err > length_tol_m:
                violations.append(
                    f"P1_BONE_LENGTH frame={f_idx} bone={b}: len_err={len_err * 1000.0:.3f}mm > {length_tol_m * 1000.0:.3f}mm"
                )
            if v3:
                tail_err = float(np.linalg.norm(T_b - H_b - R_b @ (rest_R[b].T @ (rest_T[b] - rest_H[b]))))
                if tail_err > length_tol_m:
                    violations.append(f'P1_TAIL_FRAME frame={f_idx} bone={b}: error={tail_err}')

            # 3. Parent-child head offset conservation & local Euler box compliance
            p = classified[b].get("parent")
            basis = np.asarray(classified[b].get('pose_basis', np.eye(4)))
            fixed = b not in fitted or bool(classified[b].get('fixed_local')) or classified[b].get('kind') == 'rigid'
            if p and p in pose_H and p in world_R and not (v3 and b in motion_roots):
                expected_H = pose_H[p] + world_R[p] @ (rest_R[p].T @ (rest_H[b] - rest_H[p]))
                if v3 and fixed:
                    expected_H += world_R[p] @ rest_R[p].T @ rest_R[b] @ basis[:3,3]
                off_err = float(np.linalg.norm(H_b - expected_H))
                max_offset_err_m = max(max_offset_err_m, off_err)
                if off_err > offset_tol_m:
                    violations.append(
                        f"P1_HEAD_OFFSET frame={f_idx} bone={b}: offset_err={off_err * 1000.0:.3f}mm > {offset_tol_m * 1000.0:.3f}mm"
                    )
            elif v3 and not p and b not in motion_roots:
                off_err = float(np.linalg.norm(H_b - rest_H[b] - rest_R[b] @ basis[:3,3]))
                max_offset_err_m = max(max_offset_err_m, off_err)
                if off_err > offset_tol_m:
                    violations.append(f'P1_HEAD_OFFSET frame={f_idx} bone={b}: fixed root translated')

            R_init = rest_R[b] if (p is None or p not in world_R) else world_R[p] @ (rest_R[p].T @ rest_R[b])
            R_delta = R_init.T @ R_b
            is_fixed = (
                bool(classified[b].get("fixed_local"))
                or classified[b].get("kind") == "rigid"
                or classified[b].get("sub_role") in ("fixed_local", "rigid")
            )
            if v3:
                if b in motion_roots:
                    continue
                is_fixed = fixed
            if is_fixed:
                fixed_err_deg = _geodesic_angle_deg(R_delta, basis[:3,:3] if v3 else np.eye(3, dtype=np.float64))
                max_limit_violation_deg = max(max_limit_violation_deg, fixed_err_deg)
                if fixed_err_deg > limit_tol_deg:
                    violations.append(
                        f"P1_FIXED_LOCAL frame={f_idx} bone={b}: delta_deg={fixed_err_deg:.3f}deg > {limit_tol_deg:.2f}deg"
                    )
            else:
                lims = classified[b].get("limits_rad")
                if isinstance(lims, list) and len(lims) == 3:
                    R_clamped = tax.clamp_local_delta_euler(R_delta, lims)
                    over_deg = _geodesic_angle_deg(R_delta, R_clamped)
                    max_limit_violation_deg = max(max_limit_violation_deg, over_deg)
                    if over_deg > limit_tol_deg:
                        violations.append(
                            f"P1_LIMIT_BOX frame={f_idx} bone={b}: exceeds joint_limits box by {over_deg:.3f}deg > {limit_tol_deg:.2f}deg"
                        )

    surface_contact_checked = False
    if isinstance(trajectory_meta, dict) and isinstance(trajectory_meta.get("taxonomy_diagnostics"), dict):
        if "contact_quality_passed" in trajectory_meta["taxonomy_diagnostics"]:
            surface_contact_checked = True
            try:
                assert_surface_contact_quality(trajectory_meta, action=action)
            except Exception as exc:
                violations.append(f"P4_SURFACE_CONTACT action={action}: {exc}")

    passed = len(violations) == 0
    report = {
        "schema": FORMAL_GATE_SCHEMA,
        "mode": "trajectory",
        "action": action,
        "passed": passed,
        "num_frames": len(frames),
        "num_bones": len(order),
        "p1_soundness": {
            "passed": len([v for v in violations if v.startswith("P1_")]) == 0,
            "max_so3_ortho_err": round(max_so3_ortho_err, 8),
            "max_so3_det_err": round(max_so3_det_err, 8),
            "max_limit_violation_deg": round(max_limit_violation_deg, 5),
            "max_length_err_mm": round(max_length_err_m * 1000.0, 5),
            "max_offset_err_mm": round(max_offset_err_m * 1000.0, 5),
        },
        "p4_surface_contact": {
            "checked": surface_contact_checked,
            "passed": len([v for v in violations if v.startswith("P4_")]) == 0,
        },
        "violations": violations[:25],
        "total_violations": len(violations),
        "verified_at": now_taipei_iso(),
    }
    if assert_passed and not passed:
        raise RuntimeError(
            f"Trajectory Formal Verification Gate failed ({len(violations)} violations): {violations[:3]}"
        )
    return report


def _v3_oracle_probes(rig, seed, num_random_poses):
    """Deterministic feasible targets; never hand their parameters to the solver.

    Boundary checks sample up to twelve fitted scalar channels, spread over the
    channel list. They are an audit sample, not an exhaustive proof of the box.
    Guard-infeasible targets remain in the receipt as excluded, never as passes.
    """
    rng = np.random.default_rng(seed)
    start = rig.start_params()
    probes = [("neutral", start)]
    excluded = []
    finite = np.flatnonzero(np.isfinite(rig.lower) & np.isfinite(rig.upper))
    selected = finite[np.unique(np.linspace(0, len(finite) - 1, min(12, len(finite))).astype(int))] if len(finite) else []

    def add(label, params):
        points, jac, _, _ = rig.fk(params, jac=True)
        guard, _ = rig.guard(points, jac)
        worst = float(np.max(guard, initial=0.0))
        if worst > 1e-5:
            excluded.append({"pose": label, "reason": "guard_infeasible", "max_guard_residual": worst})
            return False
        probes.append((label, params))
        return True

    for channel in selected:
        for side, value in (("lower", rig.lower[channel]), ("upper", rig.upper[channel])):
            params = start.copy()
            params[channel] = value
            add(f"boundary_{int(channel)}_{side}", params)
    # World roots have no finite angular/translation bounds. Exercise their full
    # motion interface with a finite displacement, while staying above the floor.
    for name, entry in rig.slices.items():
        if isinstance(entry, slice):
            params = start.copy()
            params[entry.start:entry.start + 3] += [0.12, -0.08, 0.2]
            params[entry.start + 3:entry.stop] += rig.height * np.array([0.04, -0.03, 0.08])
            add(f"root_motion_{name}", params)
    accepted = 0
    for attempt in range(max(0, num_random_poses) * 12):
        if accepted >= num_random_poses:
            break
        params = start.copy()
        for entry in rig.slices.values():
            if isinstance(entry, slice):
                params[entry.start:entry.start + 3] += rng.uniform(-0.35, 0.35, 3)
                params[entry.start + 3:entry.stop] += rig.height * rng.uniform([-0.05, -0.05, 0.0], [0.05, 0.05, 0.08])
            else:
                indices, _ = entry
                params[indices] = rng.uniform(rig.lower[indices], rig.upper[indices])
        if add(f"random_{attempt:03d}", params):
            accepted += 1
    return probes, {
        "scope": "sampled guard-feasible poses and locally identifiable world rotations",
        "exhaustive": False,
        "finite_channels": len(finite),
        "boundary_channels_sampled": [int(i) for i in selected],
        "boundary_channels_unsampled": [int(i) for i in finite if i not in selected],
        "random_requested": num_random_poses,
        "random_checked": accepted,
        "excluded_poses": excluded,
    }


def _v3_identifiable_rotations(rig, params, jac, rotations):
    """A world rotation is identifiable only if point-Jacobian null modes cannot change it."""
    if not rig.size:
        return list(rig.names)
    matrix = jac.reshape(-1, rig.size)
    _, singular, right = np.linalg.svd(matrix, full_matrices=matrix.shape[0] < matrix.shape[1])
    cutoff = max(float(singular.max(initial=0.0)) * 1e-8, 1e-10)
    rank = int(np.count_nonzero(singular > cutoff))
    null = right[rank:].T
    if not null.shape[1]:
        return list(rig.names)
    identifiable = []
    for name in rig.names:
        rotation_jac = np.concatenate([
            rig.vector_jacobian(params, rotations, name, axis)[1]
            for axis in np.eye(3)
        ], axis=0)
        if np.linalg.norm(rotation_jac @ null, ord="fro") <= 1e-7:
            identifiable.append(name)
    return identifiable


def _verify_v3_rig(classified, skin, seed, num_random_poses, assert_passed):
    from reprojection_solver import RigModel, fit_clip_2d_reprojection
    from studio_camera import cameras_from_calibration
    from v3_verification import verify_v3_numerics
    from threadpoolctl import threadpool_limits

    rig = RigModel(classified, skin=skin)
    structure = {"motion_roots": sorted(rig.motion_roots), "fitted_bones": sorted(
        n for n, entry in rig.slices.items() if isinstance(entry, slice) or len(entry[0]))}
    probes, coverage = _v3_oracle_probes(rig, seed, num_random_poses)
    params = np.asarray([p for _, p in probes])
    confidence = np.ones((len(params), 2, len(rig.point_names)))
    frames = rig.frames(params, confidence)
    p1 = verify_trajectory_formal_gate(frames, classified, trajectory_meta={
        "reader_contract": "monster-reader-v3", "motion_structure": structure}, limit_tol_deg=3.0, assert_passed=False)
    numerical = verify_v3_numerics(classified, seed, skin=skin)
    points = np.array([rig.fk(p)[0] for p in params])
    cameras = list(cameras_from_calibration({"center": points[0].mean(0).tolist(),
        "ortho_scale": max(rig.height * 3, 1.0), "panel_width": 448, "panel_height": 448}).values())
    observed = np.array([[c.project(x) for c in cameras] for x in points])
    # One real clip starts from its saved p0 and estimates subsequent frames from
    # readings/previous solutions. No ground-truth initial channels enter fitting.
    fit, _ = fit_clip_2d_reprojection(rig, cameras, observed, confidence)
    fit = np.asarray(fit, dtype=float)
    if fit.shape != params.shape or not np.isfinite(fit).all():
        raise ValueError("P2_V3_INVALID_SOLUTION: solver must return one finite parameter vector per oracle frame")
    violations = list(p1["violations"])
    records = []
    with threadpool_limits(limits=1):
        for (label, target), recovered, truth in zip(probes, fit, points):
            predicted, _, solved_rotations = rig.fk(recovered)
            _, jac, _, rotations = rig.fk(target, jac=True)
            identifiable = _v3_identifiable_rotations(rig, target, jac, rotations)
            point_mm = float(np.linalg.norm(predicted - truth, axis=-1).max() * 1000)
            rotation_deg = max((_geodesic_angle_deg(rotations[n], solved_rotations[n]) for n in identifiable), default=0.0)
            records.append({"pose": label, "max_point_error_mm": point_mm,
                "max_identifiable_rotation_error_deg": rotation_deg,
                "identifiable_rotation_bones": identifiable,
                "unobservable_rotation_bones": [n for n in rig.names if n not in identifiable]})
            if point_mm > 1.5:
                violations.append(f"P2_V3_PIXEL_REACHABILITY pose={label}: max={point_mm:.6f}mm > 1.5mm")
            if rotation_deg > 0.35:
                violations.append(f"P2_V3_ROTATION_REACHABILITY pose={label}: max={rotation_deg:.6f}deg > 0.35deg")
    recovered_frames = rig.frames(fit, confidence)
    recovered_p1 = verify_trajectory_formal_gate(recovered_frames, classified, trajectory_meta={
        "reader_contract": "monster-reader-v3", "motion_structure": structure}, limit_tol_deg=3.0, assert_passed=False)
    violations.extend(recovered_p1["violations"])
    hidden, _ = fit_clip_2d_reprojection(rig, cameras, np.zeros_like(observed[:1]), np.zeros_like(confidence[:1]))
    hidden = np.asarray(hidden, dtype=float)
    if hidden.shape != params[:1].shape or not np.isfinite(hidden).all():
        raise ValueError("P3_V3_INVALID_SOLUTION: solver must return one finite parameter vector for the hidden frame")
    hidden_mm = float(np.linalg.norm(rig.fk(hidden[0])[0] - points[0], axis=-1).max() * 1000)
    if hidden_mm > 1e-5:
        violations.append(f"P3_V3_OCCLUSION_INVARIANCE: max={hidden_mm}mm")
    coverage["poses_checked"] = len(records)
    coverage["nonneutral_observable_poses"] = sum(
        bool(np.linalg.norm(truth - points[0], axis=-1).max() > 1e-6 * rig.height)
        for truth in points[1:])
    coverage["guard_feasible_boundary_poses"] = sum(label.startswith("boundary_") for label, _ in probes)
    start_jac = rig.fk(params[0], jac=True)[1]
    observable_motion = bool(np.max(np.abs(start_jac), initial=0.0) > 1e-8)
    coverage["observable_motion_at_start"] = observable_motion
    if observable_motion and not coverage["nonneutral_observable_poses"]:
        violations.append("P2_V3_INSUFFICIENT_COVERAGE: no guard-feasible nonneutral observable pose was checked")
    report = {"schema": FORMAL_GATE_SCHEMA, "reader_contract": "monster-reader-v3", "mode": "rig",
        "passed": not violations, "num_bones": len(rig.names), "motion_structure": structure,
        "p1_soundness": p1["p1_soundness"], "p1_recovered_soundness": recovered_p1["p1_soundness"], "v3_fk_camera": numerical,
        "p2_oracle_completeness": {"passed": not any(v.startswith("P2_") for v in violations),
            "neutral_hierarchy_max_mm": records[0]["max_point_error_mm"],
            "max_pose_point_error_mm": max(r["max_point_error_mm"] for r in records),
            "max_identifiable_rotation_error_deg": max(r["max_identifiable_rotation_error_deg"] for r in records),
            "coverage": coverage, "poses": records},
        "p3_occlusion_invariance": {"passed": hidden_mm <= 1e-5, "max_error_mm": hidden_mm},
        "p4_surface_contact": {"checked": False, "reason": "Rig-only input; trajectory surface-contact receipts are checked on delivery."},
        "violations": violations[:25], "total_violations": len(violations), "verified_at": now_taipei_iso()}
    if assert_passed and violations:
        raise RuntimeError(f"V3 Rig Formal Verification Gate failed: {violations[:3]}")
    return report


def verify_rig_formal_gate(
    rig_bones: dict[str, Any] | None = None,
    joint_limits_profile: dict[str, Any] | None = None,
    *,
    seed: int = 20261025,
    num_random_poses: int = 16,
    assert_passed: bool = True,
    rig_skinning=None,
) -> dict[str, Any]:
    """Runs formal verification of P1, P2, P3, and P4 on a rig + joint_limits_profile
    (or on the canonical 12-bone all-roles fixture when `rig_bones is None`).
    """
    classified = _prepare_classified_bones(rig_bones, joint_limits_profile)
    if rig_skinning is not None or _v3_bones(classified):
        return _verify_v3_rig(classified,rig_skinning,seed,num_random_poses,assert_passed)
    order = _topo_order(classified)
    rest_R = _compute_rest_rot_matrices(classified, order)
    rest_H = {}
    rest_T = {}
    for b in order:
        rest_H[b], rest_T[b] = _rest_head_tail(classified[b])

    children_map: dict[str, list[str]] = {b: [] for b in order}
    for b in order:
        p = classified[b].get("parent")
        if p in children_map:
            children_map[p].append(b)

    rng = np.random.default_rng(seed)
    violations: list[str] = []

    # Build deterministic test suite of feasible poses inside L = prod_b [l_b, u_b]:
    #   0: neutral pose (all 0)
    #   1: lower boundary-active pose (0.95 * l_b)
    #   2: upper boundary-active pose (0.95 * u_b)
    #   3..N: random feasible poses across [l_b, u_b]
    pose_eulers_list: list[tuple[str, dict[str, tuple[float, float, float]]]] = []
    pose_eulers_list.append(("neutral", {b: (0.0, 0.0, 0.0) for b in order}))

    lo_pose: dict[str, tuple[float, float, float]] = {}
    hi_pose: dict[str, tuple[float, float, float]] = {}
    for b in order:
        is_fixed = (
            bool(classified[b].get("fixed_local"))
            or classified[b].get("kind") == "rigid"
            or classified[b].get("sub_role") in ("fixed_local", "rigid")
        )
        if is_fixed:
            lo_pose[b] = (0.0, 0.0, 0.0)
            hi_pose[b] = (0.0, 0.0, 0.0)
        else:
            lims = classified[b].get("limits_rad", [[-0.3, 0.3], [-0.2, 0.2], [-0.3, 0.3]])
            lo_pose[b] = tuple(float(lims[ax][0]) * 0.95 for ax in range(3))  # type: ignore[assignment]
            hi_pose[b] = tuple(float(lims[ax][1]) * 0.95 for ax in range(3))  # type: ignore[assignment]
    pose_eulers_list.append(("boundary_lower", lo_pose))
    pose_eulers_list.append(("boundary_upper", hi_pose))

    for idx in range(num_random_poses):
        eulers: dict[str, tuple[float, float, float]] = {}
        for b in order:
            is_fixed = (
                bool(classified[b].get("fixed_local"))
                or classified[b].get("kind") == "rigid"
                or classified[b].get("sub_role") in ("fixed_local", "rigid")
            )
            if is_fixed:
                eulers[b] = (0.0, 0.0, 0.0)
            else:
                lims = classified[b].get("limits_rad", [[-0.3, 0.3], [-0.2, 0.2], [-0.3, 0.3]])
                vals = []
                for ax in range(3):
                    lo, hi = float(lims[ax][0]), float(lims[ax][1])
                    vals.append(float(rng.uniform(lo, hi)) if (hi - lo) > 1e-6 else 0.0)
                eulers[b] = (vals[0], vals[1], vals[2])
        pose_eulers_list.append((f"random_{idx:02d}", eulers))

    # --- P2 Part A: Chain & Single-Bone Exact Reachability + Twist Identifiability ---
    max_single_bone_tail_err_mm = 0.0
    max_twist_identifiable_rot_err_deg = 0.0
    twist_identifiable_bones: set[str] = set()

    for pose_name, eulers in pose_eulers_list:
        gt_frame = _forward_kinematics_from_local_eulers(classified, eulers)["bones"]
        obs = {
            b: {
                "head_3d": list(gt_frame[b]["head_3d"]),
                "tail_3d": list(gt_frame[b]["tail_3d"]),
                "confidence": 1.0,
                "tail_conf": 1.0,
            }
            for b in order
        }
        for b in order:
            p = classified[b].get("parent")
            gt_R_b = np.asarray(gt_frame[b]["rot_mat_world"], dtype=np.float64)
            R_init = (
                rest_R[b]
                if (p is None or p not in gt_frame)
                else np.asarray(gt_frame[p]["rot_mat_world"], dtype=np.float64) @ (rest_R[p].T @ rest_R[b])
            )
            R_sol, _ = tax.solve_bone_rotation_by_kind(
                b,
                R_init,
                obs,
                classified,
                rest_R,
                children_map,
                {},
                {},
                raw_obs_bones=obs,
            )
            lims = classified[b].get("limits_rad")
            is_fixed = (
                bool(classified[b].get("fixed_local"))
                or classified[b].get("kind") == "rigid"
                or classified[b].get("sub_role") in ("fixed_local", "rigid")
            )
            if is_fixed:
                R_clamped = R_init
            elif isinstance(lims, list) and len(lims) == 3:
                R_clamped = R_init @ tax.clamp_local_delta_euler(R_init.T @ R_sol, lims)
            else:
                R_clamped = R_sol

            H_b = np.asarray(gt_frame[b]["head_3d"], dtype=np.float64)
            T_gt = np.asarray(gt_frame[b]["tail_3d"], dtype=np.float64)
            T_sol = H_b + R_clamped @ (rest_R[b].T @ (rest_T[b] - rest_H[b]))
            tail_err_mm = float(np.linalg.norm(T_sol - T_gt)) * 1000.0
            max_single_bone_tail_err_mm = max(max_single_bone_tail_err_mm, tail_err_mm)
            if tail_err_mm > 0.15:
                violations.append(
                    f"P2_SINGLE_BONE_REACHABILITY pose={pose_name} bone={b}: tail_err={tail_err_mm:.4f}mm > 0.15mm"
                )

            ev = tax._fixed_child_rotation_evidence(
                b,
                children_map.get(b, []),
                classified,
                rest_R,
                obs,
                children_map=children_map,
            )
            if ev is not None:
                twist_identifiable_bones.add(b)
                rot_err_deg = _geodesic_angle_deg(gt_R_b, R_clamped)
                max_twist_identifiable_rot_err_deg = max(max_twist_identifiable_rot_err_deg, rot_err_deg)
                if rot_err_deg > 0.35:
                    violations.append(
                        f"P2_TWIST_IDENTIFIABILITY pose={pose_name} bone={b}: rot_err={rot_err_deg:.4f}deg > 0.35deg"
                    )

    # --- P1 & P2 Part B: Full Hierarchy Reconstruction Fixed-Point & Manifold Soundness ---
    oracle_poses = pose_eulers_list[:5]  # neutral, boundary_lower, boundary_upper, random_00, random_01
    reconstructed_frames: list[dict[str, Any]] = []
    neutral_rms_mm = 0.0
    neutral_max_mm = 0.0

    for pose_name, eulers in oracle_poses:
        gt_frame = _forward_kinematics_from_local_eulers(classified, eulers)["bones"]
        pts_3d = np.zeros((1, len(order) * 2, 3), dtype=np.float64)
        for i, b in enumerate(order):
            pts_3d[0, 2 * i] = gt_frame[b]["head_3d"]
            pts_3d[0, 2 * i + 1] = gt_frame[b]["tail_3d"]
        trust = np.ones((1, len(order) * 2), dtype=np.float64)
        frames_out, _ = tax.reconstruct_frames_with_kind_solvers(
            pts_3d=pts_3d,
            trust=trust,
            bone_names=order,
            classified_bones=classified,
            depth_scale=1.0,
        )
        recon = frames_out[0]
        reconstructed_frames.append(recon)

        if pose_name == "neutral":
            sq_errs: list[float] = []
            for b in order:
                for key in ("head_3d", "tail_3d"):
                    diff_mm = (
                        float(
                            np.linalg.norm(
                                np.asarray(recon["bones"][b][key], dtype=np.float64)
                                - np.asarray(gt_frame[b][key], dtype=np.float64)
                            )
                        )
                        * 1000.0
                    )
                    sq_errs.append(diff_mm * diff_mm)
            neutral_rms_mm = math.sqrt(sum(sq_errs) / max(1, len(sq_errs)))
            neutral_max_mm = math.sqrt(max(sq_errs)) if sq_errs else 0.0
            if neutral_rms_mm > 0.5 or neutral_max_mm > 1.5:
                violations.append(
                    f"P2_NEUTRAL_FIXED_POINT: rms={neutral_rms_mm:.4f}mm, max={neutral_max_mm:.4f}mm"
                )

    p1_report = verify_trajectory_formal_gate(
        reconstructed_frames,
        classified,
        joint_limits_profile=None,
        action="rig_formal_gate",
        assert_passed=False,
    )
    if not p1_report["passed"]:
        violations.extend(p1_report["violations"])

    # --- P3: Non-Destructive Lateral Limb Invariance under Trunk Macro-Collapse ---
    deg = np.deg2rad
    p3_bone_names = ["ROOT", "SPINE", "THIGH_L", "SHIN_L", "FOOT_L", "THIGH_R", "SHIN_R", "FOOT_R"]
    p3_bones = {
        "ROOT": {
            "parent": None, "kind": "root", "sub_role": "root", "side": "C", "group_id": "axial", "chain_idx": 0,
            "rest_head_world": [0.0, 0.0, 0.30], "rest_tail_world": [0.0, 0.0, 0.45], "rest_length": 0.15,
            "limits_rad": [[-float(deg(80)), float(deg(80))], [-float(deg(45)), float(deg(45))], [-float(deg(45)), float(deg(45))]],
            "limits_source": "agent_custom",
        },
        "SPINE": {
            "parent": "ROOT", "kind": "spine", "sub_role": "spine", "side": "C", "group_id": "axial", "chain_idx": 1,
            "rest_head_world": [0.0, 0.0, 0.45], "rest_tail_world": [0.0, 0.0, 0.75], "rest_length": 0.30,
            "limits_rad": [[-float(deg(60)), float(deg(60))], [-float(deg(30)), float(deg(30))], [-float(deg(30)), float(deg(30))]],
            "limits_source": "agent_custom",
        },
        "THIGH_L": {
            "parent": "ROOT", "kind": "limb", "sub_role": "upper_limb", "side": "L", "group_id": "leg", "chain_idx": 0,
            "rest_head_world": [-0.10, 0.0, 0.30], "rest_tail_world": [-0.10, 0.0, 0.16], "rest_length": 0.14,
            "limits_rad": [[-float(deg(90)), float(deg(90))], [-float(deg(45)), float(deg(45))], [-float(deg(45)), float(deg(45))]],
            "limits_source": "agent_custom",
        },
        "SHIN_L": {
            "parent": "THIGH_L", "kind": "limb", "sub_role": "forearm_hinge", "side": "L", "group_id": "leg", "chain_idx": 1,
            "rest_head_world": [-0.10, 0.0, 0.16], "rest_tail_world": [-0.10, 0.0, 0.04], "rest_length": 0.12,
            "limits_rad": [[-float(deg(90)), float(deg(90))], [-float(deg(30)), float(deg(30))], [-float(deg(30)), float(deg(30))]],
            "limits_source": "agent_custom",
        },
        "FOOT_L": {
            "parent": "SHIN_L", "kind": "limb", "sub_role": "distal_limb", "side": "L", "group_id": "leg", "chain_idx": 2,
            "rest_head_world": [-0.10, -0.02, 0.04], "rest_tail_world": [-0.10, -0.08, 0.0], "rest_length": 0.072,
            "limits_rad": [[-float(deg(60)), float(deg(60))], [-float(deg(30)), float(deg(30))], [-float(deg(30)), float(deg(30))]],
            "limits_source": "agent_custom",
        },
        "THIGH_R": {
            "parent": "ROOT", "kind": "limb", "sub_role": "upper_limb", "side": "R", "group_id": "leg", "chain_idx": 0,
            "rest_head_world": [0.10, 0.0, 0.30], "rest_tail_world": [0.10, 0.0, 0.16], "rest_length": 0.14,
            "limits_rad": [[-float(deg(90)), float(deg(90))], [-float(deg(45)), float(deg(45))], [-float(deg(45)), float(deg(45))]],
            "limits_source": "agent_custom",
        },
        "SHIN_R": {
            "parent": "THIGH_R", "kind": "limb", "sub_role": "forearm_hinge", "side": "R", "group_id": "leg", "chain_idx": 1,
            "rest_head_world": [0.10, 0.0, 0.16], "rest_tail_world": [0.10, 0.0, 0.04], "rest_length": 0.12,
            "limits_rad": [[-float(deg(90)), float(deg(90))], [-float(deg(30)), float(deg(30))], [-float(deg(30)), float(deg(30))]],
            "limits_source": "agent_custom",
        },
        "FOOT_R": {
            "parent": "SHIN_R", "kind": "limb", "sub_role": "distal_limb", "side": "R", "group_id": "leg", "chain_idx": 2,
            "rest_head_world": [0.10, -0.02, 0.04], "rest_tail_world": [0.10, -0.08, 0.0], "rest_length": 0.072,
            "limits_rad": [[-float(deg(60)), float(deg(60))], [-float(deg(30)), float(deg(30))], [-float(deg(30)), float(deg(30))]],
            "limits_source": "agent_custom",
        },
    }
    p3_pts_frame = {
        "ROOT": ([0.0, 0.0, 0.18], [0.0, -0.14, 0.22]),
        "SPINE": ([0.0, -0.14, 0.22], [0.0, -0.42, 0.26]),
        "THIGH_L": ([-0.10, 0.0, 0.18], [-0.10, -0.08, 0.07]),
        "SHIN_L": ([-0.10, -0.08, 0.07], [-0.10, 0.03, 0.02]),
        "FOOT_L": ([-0.10, 0.03, 0.02], [-0.10, 0.09, 0.0]),
        "THIGH_R": ([0.10, 0.0, 0.18], [0.10, -0.02, 0.042]),
        "SHIN_R": ([0.10, -0.02, 0.042], [0.10, 0.10, 0.034]),
        "FOOT_R": ([0.10, 0.10, 0.034], [0.10, 0.16, 0.00]),
    }
    p3_pts = np.zeros((1, len(p3_bone_names) * 2, 3), dtype=np.float64)
    for i, n in enumerate(p3_bone_names):
        p3_pts[0, 2 * i] = p3_pts_frame[n][0]
        p3_pts[0, 2 * i + 1] = p3_pts_frame[n][1]
    p3_frames_out, _ = tax.reconstruct_frames_with_kind_solvers(
        pts_3d=p3_pts,
        trust=np.ones((1, len(p3_bone_names) * 2), dtype=np.float64),
        bone_names=p3_bone_names,
        classified_bones=p3_bones,
        depth_scale=1.0,
    )
    shin_r_state = p3_frames_out[0]["bones"]["SHIN_R"]
    dz_shin_r = float(shin_r_state["tail_3d"][2]) - float(shin_r_state["head_3d"][2])
    dy_shin_r = float(shin_r_state["tail_3d"][1]) - float(shin_r_state["head_3d"][1])
    if dy_shin_r <= 0.08 or dz_shin_r <= -0.03:
        violations.append(
            f"P3_LATERAL_LIMB_INVARIANCE: SHIN_R overwritten downward (dy={dy_shin_r:.4f}m, dz={dz_shin_r:.4f}m)"
        )

    # --- P4: Surface Contact Fixed-Point Identity, Idempotence & Local Basin Preservation ---
    sc_order = ["root", "thigh", "shin", "foot"]
    sc_rest_H = {
        "root": np.array([0.0, 0.0, 0.50]),
        "thigh": np.array([0.0, 0.0, 0.40]),
        "shin": np.array([0.0, 0.0, 0.22]),
        "foot": np.array([0.0, 0.0, 0.06]),
    }
    sc_rest_T = {
        "root": np.array([0.0, 0.0, 0.70]),
        "thigh": np.array([0.0, 0.0, 0.22]),
        "shin": np.array([0.0, 0.0, 0.06]),
        "foot": np.array([0.0, -0.08, 0.00]),
    }
    sc_rest_R = {n: np.eye(3) for n in sc_order}
    sc_meta = {
        "root": {"parent": None, "kind": "root"},
        "thigh": {"parent": "root", "kind": "limb"},
        "shin": {"parent": "thigh", "kind": "limb"},
        "foot": {"parent": "shin", "kind": "limb"},
    }
    sc_limits = {
        "root": [[0.0, 0.0]] * 3,
        "thigh": [[float(-deg(85)), float(deg(85))], [0.0, 0.0], [0.0, 0.0]],
        "shin": [[float(-deg(85)), float(deg(85))], [0.0, 0.0], [0.0, 0.0]],
        "foot": [[float(-deg(65)), float(deg(65))], [0.0, 0.0], [0.0, 0.0]],
    }
    R_root = np.eye(3)
    R_thigh = tax.euler_xyz_to_matrix_numpy(float(-deg(60)), 0.0, 0.0)
    R_shin = R_thigh @ tax.euler_xyz_to_matrix_numpy(float(deg(75)), 0.0, 0.0)
    R_foot = R_shin @ tax.euler_xyz_to_matrix_numpy(float(-deg(25)), 0.0, 0.0)
    sc_rots = {"root": R_root, "thigh": R_thigh, "shin": R_shin, "foot": R_foot}
    sc_heads = {"root": np.array([0.0, 0.0, 0.34])}
    sc_tails = {}
    for b in sc_order:
        p = sc_meta[b]["parent"]
        if p:
            sc_heads[b] = sc_heads[p] + sc_rots[p] @ (sc_rest_H[b] - sc_rest_H[p])
        sc_tails[b] = sc_heads[b] + sc_rots[b] @ (sc_rest_T[b] - sc_rest_H[b])

    base_sc_frames = [
        {
            "bones": {
                b: {
                    "head_3d": sc_heads[b].tolist(),
                    "tail_3d": sc_tails[b].tolist(),
                    "rot_mat_world": sc_rots[b].tolist(),
                }
                for b in sc_order
            }
        }
    ]
    # P4a: Non-penetrating sample (+0.05m above floor_z=0.0) -> exact fixed point
    clear_sample_local = (R_foot.T @ (np.array([0.0, -0.05, 0.05]) - sc_heads["foot"])).tolist()
    clear_frames = copy.deepcopy(base_sc_frames)
    res_clear = apply_surface_floor_constraints(
        clear_frames,
        sc_order,
        "root",
        sc_meta,
        sc_rest_H,
        sc_rest_T,
        sc_rest_R,
        sc_limits,
        {"floor_z": 0.0, "max_iterations": 24, "support_sets": [{"bone": "foot", "lock_xy": False, "samples_local": [clear_sample_local]}]},
    )
    clear_corr_mm = float(res_clear["root_contact_max_correction"]) * 1000.0
    if clear_corr_mm > 1e-6:
        violations.append(f"P4_FIXED_POINT_IDENTITY: non-penetrating pose modified by {clear_corr_mm:.6f}mm")

    # P4b & P4c: Minor penetration (-0.012m) -> local basin preservation + idempotence
    pen_sample_local = (R_foot.T @ (np.array([0.0, -0.05, -0.012]) - sc_heads["foot"])).tolist()
    pen_cfg = {
        "floor_z": 0.0,
        "max_iterations": 24,
        "support_sets": [{"bone": "foot", "lock_xy": False, "samples_local": [pen_sample_local]}],
    }
    pen_frames = copy.deepcopy(base_sc_frames)
    res_pass1 = apply_surface_floor_constraints(
        pen_frames, sc_order, "root", sc_meta, sc_rest_H, sc_rest_T, sc_rest_R, sc_limits, pen_cfg
    )
    knee_shift_mm = float(
        np.linalg.norm(np.asarray(pen_frames[0]["bones"]["shin"]["head_3d"], dtype=np.float64) - sc_heads["shin"])
    ) * 1000.0
    res_pass2 = apply_surface_floor_constraints(
        pen_frames, sc_order, "root", sc_meta, sc_rest_H, sc_rest_T, sc_rest_R, sc_limits, pen_cfg
    )
    idemp_corr_mm = float(res_pass2["root_contact_max_correction"]) * 1000.0
    if not res_pass1.get("contact_quality_passed", False):
        violations.append("P4_CONVERGENCE: kneeling minor penetration failed to resolve")
    if knee_shift_mm > 25.0:
        violations.append(f"P4_LOCAL_BASIN: knee shifted {knee_shift_mm:.2f}mm > 25.0mm")
    if idemp_corr_mm > 0.05:
        violations.append(f"P4_IDEMPOTENCE: second-pass correction={idemp_corr_mm:.4f}mm > 0.05mm")

    passed = len(violations) == 0
    report = {
        "schema": FORMAL_GATE_SCHEMA,
        "mode": "rig",
        "passed": passed,
        "num_bones": len(order),
        "twist_identifiable_bones": sorted(twist_identifiable_bones),
        "p1_soundness": p1_report["p1_soundness"],
        "p2_oracle_completeness": {
            "passed": len([v for v in violations if v.startswith("P2_")]) == 0,
            "max_single_bone_tail_err_mm": round(max_single_bone_tail_err_mm, 5),
            "max_twist_identifiable_rot_err_deg": round(max_twist_identifiable_rot_err_deg, 5),
            "neutral_hierarchy_rms_mm": round(neutral_rms_mm, 5),
            "neutral_hierarchy_max_mm": round(neutral_max_mm, 5),
        },
        "p3_collapse_invariance": {
            "passed": len([v for v in violations if v.startswith("P3_")]) == 0,
            "kneeling_shin_dy_m": round(dy_shin_r, 5),
            "kneeling_shin_dz_m": round(dz_shin_r, 5),
        },
        "p4_surface_contact": {
            "passed": len([v for v in violations if v.startswith("P4_")]) == 0,
            "non_penetrating_correction_mm": round(clear_corr_mm, 6),
            "second_pass_idempotence_mm": round(idemp_corr_mm, 5),
            "local_basin_knee_shift_mm": round(knee_shift_mm, 4),
        },
        "violations": violations[:25],
        "total_violations": len(violations),
        "verified_at": now_taipei_iso(),
    }
    if assert_passed and not passed:
        raise RuntimeError(
            f"Rig Formal Verification Gate failed ({len(violations)} violations): {violations[:3]}"
        )
    return report


def run_monster_rig_formal_gate(
    monster_id: str,
    *,
    project_root: Path | str | None = None,
    drive_root: Path | str | None = None,
    assert_passed: bool = True,
) -> dict[str, Any]:
    """Loads a monster's G1-approved `candidate_<id>_rig_meta.json` and `joint_limits.json`,
    runs `verify_rig_formal_gate`, writes `scratch/checks/<monster_id>/formal_gate_rig.json`,
    and returns the report.
    """
    proj = Path(project_root or os.environ.get("MONSTER_PROJECT_ROOT") or Path.cwd()).resolve()
    ledger = load_ledger(monster_id, project_root=proj, drive_root=drive_root)
    g1_info = assert_gate(ledger, "G1")
    dirs = resolve_scratch_dirs(proj, monster_id)
    selected_cid, meta_path, _blend_path = resolve_g1_candidate_files(dirs["work_dir"], g1_info)

    rig_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    limits_path = dirs["work_dir"] / "joint_limits.json"
    jl_profile = (
        json.loads(limits_path.read_text(encoding="utf-8"))
        if limits_path.is_file()
        else None
    )

    from rig_skinning_artifact import load_probe_skinning
    report = verify_rig_formal_gate(
        rig_meta.get("bones", {}),
        jl_profile,
        rig_skinning=load_probe_skinning(jl_profile, _blend_path, rig_meta.get("bones", {})),
        assert_passed=False,
    )
    report["monster_id"] = monster_id
    report["selected_candidate_id"] = selected_cid
    report["rig_meta_path"] = str(meta_path)
    report["joint_limits_path"] = str(limits_path) if limits_path.is_file() else None

    out_path = dirs["checks_dir"] / "formal_gate_rig.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report_path"] = str(out_path)

    if assert_passed and not report["passed"]:
        raise RuntimeError(
            f"形式化验证门禁阻断 ({monster_id}): {report['violations'][:3]} (详见 {out_path})"
        )
    return report


def run_monster_trajectory_formal_gate(
    monster_id: str,
    traj_json_path: Path | str,
    *,
    action: str = "trajectory",
    project_root: Path | str | None = None,
    drive_root: Path | str | None = None,
    assert_passed: bool = True,
) -> dict[str, Any]:
    """Verifies a reconstructed trajectory JSON file against the monster's rig and `joint_limits.json`."""
    proj = Path(project_root or os.environ.get("MONSTER_PROJECT_ROOT") or Path.cwd()).resolve()
    ledger = load_ledger(monster_id, project_root=proj, drive_root=drive_root)
    g1_info = assert_gate(ledger, "G1")
    dirs = resolve_scratch_dirs(proj, monster_id)
    selected_cid, meta_path, _blend_path = resolve_g1_candidate_files(dirs["work_dir"], g1_info)

    rig_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    limits_path = dirs["work_dir"] / "joint_limits.json"
    jl_profile = (
        json.loads(limits_path.read_text(encoding="utf-8"))
        if limits_path.is_file()
        else None
    )

    t_path = Path(traj_json_path).expanduser().resolve()
    traj_payload = json.loads(t_path.read_text(encoding="utf-8"))
    if isinstance(traj_payload, list):
        frames = traj_payload
        traj_meta = None
    elif isinstance(traj_payload, dict):
        frames = traj_payload.get("frames") or traj_payload.get("smooth_frames") or []
        traj_meta = traj_payload
    else:
        raise ValueError(f"Unsupported trajectory JSON structure in {t_path}")

    report = verify_trajectory_formal_gate(
        frames,
        rig_meta.get("bones", {}),
        jl_profile,
        trajectory_meta=traj_meta,
        action=action,
        assert_passed=False,
    )
    report["monster_id"] = monster_id
    report["selected_candidate_id"] = selected_cid
    report["trajectory_json_path"] = str(t_path)

    out_path = dirs["checks_dir"] / f"formal_gate_trajectory_{action}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report["report_path"] = str(out_path)

    if assert_passed and not report["passed"]:
        raise RuntimeError(
            f"轨迹形式化验证门禁阻断 ({monster_id}/{action}): {report['violations'][:3]} (详见 {out_path})"
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monster SOP Formal Verification Gate (P1 Soundness, P2 Oracle Completeness, P3 Collapse Invariance, P4 Surface Contact)"
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_self = sub.add_parser("self-test", help="Run formal verification gate on canonical all-roles rig & surface contact fixtures")
    p_self.add_argument("--seed", type=int, default=20261025)
    p_self.add_argument("--num-random-poses", type=int, default=16)

    p_rig = sub.add_parser("rig", help="Run formal verification gate on a monster's G1-approved rig + joint_limits.json")
    p_rig.add_argument("--monster-id", required=True)
    p_rig.add_argument("--project-root", default=None)
    p_rig.add_argument("--drive-root", default=None)

    p_traj = sub.add_parser("trajectory", help="Run formal verification gate on a reconstructed trajectory JSON")
    p_traj.add_argument("--monster-id", required=True)
    p_traj.add_argument("--traj-json", required=True)
    p_traj.add_argument("--action", default="trajectory")
    p_traj.add_argument("--project-root", default=None)
    p_traj.add_argument("--drive-root", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.subcommand == "self-test":
        report = verify_rig_formal_gate(
            None,
            None,
            seed=args.seed,
            num_random_poses=args.num_random_poses,
            assert_passed=True,
        )
        from v3_verification import verify_v3_numerics
        report['v3_fk_camera']=verify_v3_numerics(build_canonical_verification_rig(),args.seed)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.subcommand == "rig":
        report = run_monster_rig_formal_gate(
            args.monster_id,
            project_root=args.project_root,
            drive_root=args.drive_root,
            assert_passed=True,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.subcommand == "trajectory":
        report = run_monster_trajectory_formal_gate(
            args.monster_id,
            args.traj_json,
            action=args.action,
            project_root=args.project_root,
            drive_root=args.drive_root,
            assert_passed=True,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
