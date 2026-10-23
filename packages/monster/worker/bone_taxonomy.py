#!/usr/bin/env python3
"""bone_taxonomy.py — Three-Layer General + Kind-Specific Bone Taxonomy & Kinematic Solver.

Architecture:
  - Layer 1 (Canonical Bone Naming & Hybrid Classifier, Schema v2):
      Defines the canonical bone naming convention `<KIND>_<SIDE>_<GROUP>_<INDEX>` across 10
      anatomical kinds (`ROOT`, `RIGID`, `SPINE`, `HEAD`, `JAW`, `LIMB`, `WING`, `FLEX`, `TAIL`, `DIGIT`).
      Combines deterministic name/prefix parsing with 3D topological and geometric graph analysis
      so both newly generated canonical rigs and legacy anonymous rigs (`bone_0..N`, `Core_Rigid`,
      `Flex_*`) are classified without retraining checkpoints.
  - Layer 2 (Universal Kinematic Core):
      Rig-agnostic bounded short-gap 2D interpolation (`<= 3` frames, zero long-gap time substitution),
      5-tap quadratic Savitzky-Golay smoothing, multi-anchor IRLS Kabsch root fitting, exact rest-offset
      forward kinematics (FK), and strict two-branch local Euler limit clamping without post-clamp
      `minimum_swing` overrides.
  - Layer 3 (Kind-Specific Kinematic Solvers):
      Dispatches specialized solvers per anatomical `kind`:
        * `root`: Multi-anchor IRLS Kabsch rigid body fit + bounded slerp;
        * `rigid`: Exact local identity (`R_delta = I_3`, zero drift);
        * `spine` / `head` / `jaw`: Sagittal-prior bounded Euler with child lever-arm observability;
        * `limb`: Gimbal-free Swing-Twist (`R_swing(d_loc) @ Ry(tau)`) + disconnected child-pivot
          fusion on forearms (`forearm_hinge`) + multi-child palm-fan fusion on hands/feet (`hand_hub`);
        * `flex` / `tail` / `wing`: Symmetric-bound travelling-wave chain solver;
        * `digit`: Coupled phalange curl damping with axial twist locked to `tau = 0`.
"""

from __future__ import annotations

import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

_WORKER_DIR = str(Path(__file__).resolve().parent)
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

from mocap_priors import (  # noqa: E402
    apply_floor_and_foot_contact_constraints,
    compute_chain_wave_target_euler,
    dejitter_velocity_gated,
    mocap_priors_enabled,
    whittaker_charbonnier_smooth_trajectory,
)

BONE_TAXONOMY_SCHEMA = "monster-bone-taxonomy-v2"
JOINT_LIMITS_SCHEMA = "monster-joint-limits-v1"
DEFAULT_SYNTH_SAMPLES = 5000

POSE_SAMPLING_REGIMES: tuple[str, ...] = (
    "synergistic",
    "decoupled",
    "antiphase_extreme",
    "workspace_cross_occlusion",
)

G2_BOUNDARY_REGIMES: tuple[str, ...] = (
    "in_phase_boundary_extension",
    "antiphase_fold_boundary",
    "cross_occlusion_twist_boundary",
    "full_body_dynamic_root_boundary",
    "quasi_uniform_interior_coverage",
)

CANONICAL_KINDS: tuple[str, ...] = (
    "root",
    "rigid",
    "spine",
    "head",
    "jaw",
    "limb",
    "wing",
    "flex",
    "tail",
    "digit",
)

CANONICAL_SIDES: tuple[str, ...] = ("C", "L", "R")

_CANONICAL_NAME_RE = re.compile(
    r"^(ROOT|RIGID|SPINE|HEAD|JAW|LIMB|WING|FLEX|TAIL|DIGIT)_([CLR])_([0-9]{2})_([0-9]{2})$",
    re.IGNORECASE,
)


# ==============================================================================
# Layer 1: Canonical Bone Naming & Hybrid (Name + 3D Topology) Classifier
# ==============================================================================

def parse_canonical_bone_name(name: str) -> dict[str, Any] | None:
    """Parses a canonical `<KIND>_<SIDE>_<GROUP>_<INDEX>` bone name if it matches the v2 contract."""
    m = _CANONICAL_NAME_RE.match(name.strip())
    if not m:
        return None
    return {
        "kind": m.group(1).lower(),
        "side": m.group(2).upper(),
        "group_id": m.group(3),
        "chain_index": int(m.group(4)),
    }


def format_canonical_bone_name(kind: str, side: str, group: int | str, index: int) -> str:
    """Formats a canonical `<KIND>_<SIDE>_<GROUP>_<INDEX>` bone name."""
    k_up = kind.strip().upper()
    if k_up.lower() not in CANONICAL_KINDS:
        raise ValueError(f"Unsupported canonical bone kind: {kind!r}")
    s_up = side.strip().upper()
    if s_up not in CANONICAL_SIDES:
        s_up = "C"
    grp_str = f"{int(group):02d}" if isinstance(group, int) or str(group).isdigit() else str(group)[:2].zfill(2)
    idx_str = f"{max(0, int(index)):02d}"
    return f"{k_up}_{s_up}_{grp_str}_{idx_str}"


def infer_kind_and_side_from_name(
    name: str, role_hint: str | None = None
) -> tuple[str | None, str | None]:
    """Infers `(kind, side)` from canonical names, procedural generator prefixes, common rig aliases,
    or an explicit anatomical `role_hint` from `bones_meta[b].get("role")`.
    """
    parsed = parse_canonical_bone_name(name)
    if parsed is not None:
        return parsed["kind"], parsed["side"]

    low = name.strip().lower()
    side: str | None = None
    if re.search(r"(^|[._\-])(l|left)([._\-]|$)", low):
        side = "L"
    elif re.search(r"(^|[._\-])(r|right)([._\-]|$)", low):
        side = "R"

    for token_src, allow_generic_spine in ((low, True), ((role_hint or "").strip().lower(), False)):
        if not token_src or token_src in ("approved_deform", "deform", "bone"):
            continue
        if token_src in ("root", "hips", "pelvis", "cog", "master") or token_src.startswith("root_"):
            return "root", side or "C"
        if any(
            tok in token_src
            for tok in ("rigid", "shell", "carapace", "armor", "pelvis_attachment", "head_tip", "muzzle_tip")
        ):
            return "rigid", side
        if token_src.startswith("flex_") or any(
            tok in token_src
            for tok in ("flex", "tentacle", "tendril", "vine", "ribbon", "hair", "cape", "whisker", "feeler")
        ):
            return "flex", side
        if any(
            tok in token_src
            for tok in ("digit", "finger", "thumb", "index", "middle", "ring", "pinky", "toe", "phalange")
        ):
            return "digit", side
        if any(tok in token_src for tok in ("jaw", "mandible", "beak", "muzzle", "snout")):
            return "jaw", side or "C"
        if any(tok in token_src for tok in ("head", "neck", "skull", "cranium")):
            return "head", side or "C"
        if allow_generic_spine and any(
            tok in token_src for tok in ("spine", "chest", "torso", "abdomen", "thorax")
        ):
            return "spine", side or "C"
        if "tail" in token_src:
            return "tail", side or "C"
        if "wing" in token_src or "patagium" in token_src or "feather" in token_src:
            return "wing", side
        if any(
            tok in token_src
            for tok in (
                "limb",
                "clavicle",
                "scapula",
                "shoulder",
                "upper_arm",
                "upperarm",
                "forearm",
                "lower_arm",
                "lowerarm",
                "elbow",
                "hand",
                "wrist",
                "palm",
                "hip",
                "thigh",
                "upper_leg",
                "upperleg",
                "knee",
                "shin",
                "calf",
                "lower_leg",
                "lowerleg",
                "hock",
                "foot",
                "ankle",
                "paw",
                "claw",
            )
        ):
            return "limb", side

    return None, side


def default_limits_and_corr_for_kind(
    kind: str, sub_role: str = "", is_leaf: bool = False
) -> tuple[list[list[float]], float, bool]:
    """Returns default `(limits_rad, chain_corr, fixed_local)` for a classified bone kind & sub_role."""
    deg = math.radians
    if kind == "root":
        return [[-deg(12), deg(12)], [-deg(18), deg(18)], [-deg(15), deg(15)]], 0.0, False
    if kind == "rigid":
        return [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 0.0, True
    if kind == "spine":
        return [[-deg(15), deg(15)], [-deg(3), deg(3)], [-deg(15), deg(15)]], 0.65, False
    if kind == "head":
        return [[-deg(22), deg(22)], [-deg(20), deg(20)], [-deg(20), deg(20)]], 0.55, False
    if kind == "jaw":
        return [[0.0, deg(35)], [-deg(2), deg(2)], [-deg(3), deg(3)]], 0.0, False
    if kind == "tail":
        return [[-deg(18), deg(18)], [-deg(3), deg(3)], [-deg(14), deg(14)]], 0.75, False
    if kind == "flex":
        return [[-deg(25), deg(25)], [-deg(10), deg(10)], [-deg(18), deg(18)]], 0.65, False
    if kind == "wing":
        return [[-deg(40), deg(45)], [-deg(8), deg(8)], [-deg(35), deg(35)]], 0.55, False
    if kind == "digit":
        ry = 0.0 if is_leaf else deg(2)
        return [[-deg(15), deg(35)], [-ry, ry], [-deg(12), deg(12)]], 0.70, False
    # Default "limb" sub-roles
    if sub_role == "clavicle_socket":
        return [[-deg(20), deg(20)], [-deg(8), deg(8)], [-deg(20), deg(20)]], 0.0, False
    if sub_role in ("upper_limb", "ball_socket"):
        return [[-deg(55), deg(65)], [-deg(25), deg(25)], [-deg(40), deg(40)]], 0.0, False
    if sub_role == "forearm_hinge":
        return [[-deg(45), deg(95)], [-deg(35), deg(35)], [-deg(45), deg(45)]], 0.0, False
    if sub_role == "hand_hub":
        return [[-deg(35), deg(35)], [-deg(3), deg(3)], [-deg(25), deg(25)]], 0.0, False
    return [[-deg(25), deg(30)], [-deg(2), deg(2)], [-deg(10), deg(10)]], 0.50, False


def classify_rig_bones(
    bones_meta: dict[str, dict[str, Any]],
    bbox_min: list[float] | None = None,
    bbox_max: list[float] | None = None,
    constraints_meta: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Hybrid (Name + Constraints + 3D Topology/Geometry) classifier.
    Assigns `kind`, `sub_role`, `side`, `group_id`, `chain_index`, `canonical_name`, `legacy_name`,
    `fixed_local`, `is_primary`, `limits_rad`, and `chain_corr` for every bone in `bones_meta`.
    """
    meta = copy.deepcopy(bones_meta)
    c_bones = (constraints_meta or {}).get("bones", constraints_meta or {})
    for b_name, info in meta.items():
        if b_name in c_bones and isinstance(c_bones[b_name], dict):
            info.update(copy.deepcopy(c_bones[b_name]))

    bone_names = sorted(meta.keys())
    if not bone_names:
        return meta

    # Reconstruct parent -> children and depths
    children_map: dict[str, list[str]] = {b: [] for b in bone_names}
    for b in bone_names:
        p = meta[b].get("parent")
        if p and p in children_map:
            children_map[p].append(b)
    for b in bone_names:
        children_map[b].sort()
        meta[b]["children"] = list(children_map[b])

    def _compute_depth(b: str) -> int:
        d = 0
        curr = meta[b].get("parent")
        visited: set[str] = {b}
        while curr and curr in meta and curr not in visited:
            visited.add(curr)
            d += 1
            curr = meta[curr].get("parent")
        return d

    for b in bone_names:
        meta[b]["depth"] = int(meta[b].get("depth", _compute_depth(b)))

    # Subtree helper
    def _subtree_descendants(b: str) -> list[str]:
        out: list[str] = []
        stack = list(children_map[b])
        while stack:
            cur = stack.pop(0)
            out.append(cur)
            stack.extend(children_map[cur])
        return out

    # Bounding box from rest endpoints if not supplied
    all_pts = []
    for b in bone_names:
        if "rest_head_world" in meta[b]:
            all_pts.append(meta[b]["rest_head_world"])
        if "rest_tail_world" in meta[b]:
            all_pts.append(meta[b]["rest_tail_world"])
    if bbox_min is not None and bbox_max is not None:
        bmin = [float(x) for x in bbox_min]
        bmax = [float(x) for x in bbox_max]
    elif all_pts:
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        zs = [p[2] for p in all_pts]
        bmin = [min(xs), min(ys), min(zs)]
        bmax = [max(xs), max(ys), max(zs)]
    else:
        bmin = [-0.5, -0.5, 0.0]
        bmax = [0.5, 0.5, 1.0]

    width = max(1e-4, bmax[0] - bmin[0])
    height = max(1e-4, bmax[2] - bmin[2])

    roots = [b for b in bone_names if not meta[b].get("parent") or meta[b].get("parent") not in meta]
    roots.sort(key=lambda b: meta[b]["depth"])
    primary_root = roots[0] if roots else bone_names[0]
    root_x = float(meta[primary_root].get("rest_head_world", [0.0, 0.0, 0.0])[0])
    midline_tol_x = 0.085 * width

    assigned_kind: dict[str, str] = {}
    assigned_sub_role: dict[str, str] = {}
    assigned_side: dict[str, str] = {}

    def _geom_side(b: str) -> str:
        h = meta[b].get("rest_head_world", [root_x, 0.0, 0.0])
        t = meta[b].get("rest_tail_world", h)
        mx = 0.5 * (float(h[0]) + float(t[0])) - root_x
        if abs(mx) < midline_tol_x and abs(float(h[0]) - root_x) < midline_tol_x and abs(float(t[0]) - root_x) < midline_tol_x:
            return "C"
        return "L" if mx >= 0.0 else "R"

    # Step 1: Root & Explicitly Fixed / Rigid Bones
    for b in bone_names:
        info = meta[b]
        n_kind, n_side = infer_kind_and_side_from_name(b)
        if b == primary_root or info.get("parent") is None or n_kind == "root":
            assigned_kind[b] = "root"
            assigned_sub_role[b] = "root"
            assigned_side[b] = "C"
            continue

        lims = info.get("limits_rad")
        zero_lims = False
        if isinstance(lims, list) and len(lims) == 3:
            try:
                max_span = max( abs(float(pair[1]) - float(pair[0])) for pair in lims )
                zero_lims = max_span <= 1e-4
            except (TypeError, ValueError, IndexError):
                zero_lims = False

        if bool(info.get("fixed_local", False)) or n_kind == "rigid" or zero_lims:
            assigned_kind[b] = "rigid"
            assigned_sub_role[b] = "rigid_support"
            assigned_side[b] = n_side or _geom_side(b)

    # Step 2: Detect Multi-Branch Hand/Foot Hubs (>= 3 digit-chain children) and Downstream DIGIT Clusters
    def _is_digit_chain_subtree(root_child: str) -> bool:
        c_kind_name, _ = infer_kind_and_side_from_name(root_child)
        if c_kind_name in ("flex", "rigid", "wing", "spine", "head", "tail"):
            return False
        chain_nodes = [root_child] + _subtree_descendants(root_child)
        if len(chain_nodes) > 4:
            return False
        for node in chain_nodes:
            if len(children_map[node]) > 1:
                return False
            if float(meta[node].get("rest_length", 0.0)) >= 0.065 * height:
                return False
        return True

    hand_hubs: list[str] = []
    for b in bone_names:
        if b in assigned_kind:
            continue
        ch = children_map[b]
        if len(ch) >= 3:
            short_ch = [c for c in ch if _is_digit_chain_subtree(c)]
            if len(short_ch) >= 3:
                hand_hubs.append(b)
                side_h = infer_kind_and_side_from_name(b)[1] or _geom_side(b)
                assigned_kind[b] = "limb"
                assigned_sub_role[b] = "hand_hub"
                assigned_side[b] = side_h
                for sc in short_ch:
                    assigned_kind[sc] = "digit"
                    assigned_sub_role[sc] = "digit_proximal" if children_map[sc] else "digit_distal"
                    assigned_side[sc] = side_h
                    for desc in _subtree_descendants(sc):
                        assigned_kind[desc] = "digit"
                        assigned_sub_role[desc] = "digit_distal" if not children_map[desc] else "digit_medial"
                        assigned_side[desc] = side_h

    # Step 3: Trace Upstream Limb Chains from each Hand/Foot Hub up to the branching torso/pelvis hub
    for hub in hand_hubs:
        side_h = assigned_side[hub]
        curr_p = meta[hub].get("parent")
        step_up = 0
        while curr_p and curr_p in meta and curr_p not in assigned_kind:
            if len(children_map[curr_p]) >= 2:
                break
            step_up += 1
            assigned_kind[curr_p] = "limb"
            if step_up == 1:
                assigned_sub_role[curr_p] = "forearm_hinge"
            elif step_up == 2:
                assigned_sub_role[curr_p] = "upper_limb"
            else:
                assigned_sub_role[curr_p] = "clavicle_socket"
            assigned_side[curr_p] = side_h
            curr_p = meta[curr_p].get("parent")

    # Step 4: Apply Explicit Name / Prefix / Role Classifications (e.g. Flex_*, Wing_*, Digit_*, role=tail/neck/...)
    topo_sorted = sorted(bone_names, key=lambda x: meta[x]["depth"])
    for b in topo_sorted:
        if b in assigned_kind:
            continue
        role_hint = str(meta[b].get("role", ""))
        n_kind, n_side = infer_kind_and_side_from_name(b, role_hint=role_hint)
        if n_kind == "root" and meta[b].get("parent") is not None:
            n_kind = "spine"
        if n_kind is not None:
            assigned_kind[b] = n_kind
            assigned_side[b] = n_side or _geom_side(b)
            if n_kind == "rigid":
                assigned_sub_role[b] = "rigid_support"
            elif n_kind == "flex":
                assigned_sub_role[b] = "flex_chain"
                for desc in _subtree_descendants(b):
                    if desc not in assigned_kind:
                        assigned_kind[desc] = "flex"
                        assigned_sub_role[desc] = "flex_chain"
                        assigned_side[desc] = assigned_side[b]
            elif n_kind == "digit":
                p = meta[b].get("parent")
                if p and assigned_kind.get(p) != "digit":
                    assigned_sub_role[b] = "digit_proximal"
                elif not children_map[b]:
                    assigned_sub_role[b] = "digit_distal"
                else:
                    assigned_sub_role[b] = "digit_medial"
            elif n_kind == "head":
                assigned_sub_role[b] = "head_leaf" if not children_map[b] else "neck_chain"
            elif n_kind == "spine":
                assigned_sub_role[b] = "spine_hub" if len(children_map[b]) >= 2 else "spine_chain"
            elif n_kind == "jaw":
                assigned_sub_role[b] = "jaw_hinge"
            elif n_kind == "tail":
                assigned_sub_role[b] = "tail_chain"
                assigned_side[b] = n_side or "C"
                for desc in _subtree_descendants(b):
                    if desc not in assigned_kind and len(children_map.get(meta[desc].get("parent", ""), [])) <= 1:
                        assigned_kind[desc] = "tail"
                        assigned_sub_role[desc] = "tail_chain"
                        assigned_side[desc] = assigned_side[b]
            elif n_kind == "wing":
                assigned_sub_role[b] = "wing_chain"
            elif n_kind == "limb":
                p = meta[b].get("parent")
                role_low = role_hint.strip().lower()
                if not children_map[b]:
                    assigned_sub_role[b] = "distal_limb"
                elif p and assigned_kind.get(p) in ("root", "spine", "rigid"):
                    assigned_sub_role[b] = "ball_socket"
                elif (
                    (len(children_map[b]) == 1 and not children_map[children_map[b][0]])
                    or any(tok in role_low for tok in ("elbow", "knee", "hock", "hinge", "forearm", "shin", "calf"))
                ):
                    assigned_sub_role[b] = "forearm_hinge"
                else:
                    assigned_sub_role[b] = "upper_limb"

    # Step 5: Classify Remaining Midline & Lateral Bones via 3D Geometry + Topology
    # Prefer midline hubs that branch into lateral (L/R) limbs so head/muzzle splits do not inflate chest_hub_depth
    lateral_midline_hub_depths = [
        meta[b]["depth"]
        for b in bone_names
        if b != primary_root
        and len(children_map[b]) >= 2
        and abs(float(meta[b].get("rest_head_world", [root_x, 0.0, 0.0])[0]) - root_x) < midline_tol_x
        and any((assigned_side.get(c) or _geom_side(c)) in ("L", "R") for c in children_map[b])
    ]
    branch_midline_depths = [
        meta[b]["depth"]
        for b in bone_names
        if len(children_map[b]) >= 2
        and abs(float(meta[b].get("rest_head_world", [root_x, 0.0, 0.0])[0]) - root_x) < midline_tol_x
    ]
    if lateral_midline_hub_depths:
        chest_hub_depth = max(lateral_midline_hub_depths)
    else:
        chest_hub_depth = max(branch_midline_depths) if branch_midline_depths else 3

    for b in topo_sorted:
        if b in assigned_kind:
            continue
        info = meta[b]
        p = info.get("parent")
        if p and assigned_kind.get(p) == "tail" and len(children_map.get(p, [])) == 1:
            assigned_kind[b] = "tail"
            assigned_sub_role[b] = "tail_chain"
            assigned_side[b] = assigned_side.get(p, "C")
            continue

        h = info.get("rest_head_world", [root_x, 0.0, 0.0])
        t = info.get("rest_tail_world", h)
        dx, dy, dz = float(t[0]) - float(h[0]), float(t[1]) - float(h[1]), float(t[2]) - float(h[2])
        is_midline = abs(float(h[0]) - root_x) < midline_tol_x and abs(float(t[0]) - root_x) < midline_tol_x

        if is_midline:
            assigned_side[b] = "C"
            if float(h[1]) > 0.10 * height and dy > 0.02 * height and len(children_map[b]) <= 1:
                assigned_kind[b] = "tail"
                assigned_sub_role[b] = "tail_chain"
            elif p and assigned_kind.get(p) == "jaw":
                assigned_kind[b] = "jaw"
                assigned_sub_role[b] = "jaw_hinge"
            elif info["depth"] > chest_hub_depth:
                mid_siblings = [
                    c for c in children_map.get(p or "", []) if (assigned_side.get(c) or _geom_side(c)) == "C"
                ]
                if len(mid_siblings) >= 2:
                    lowest_sib = min(
                        mid_siblings,
                        key=lambda c: float(meta[c].get("rest_head_world", [0.0, 0.0, 0.0])[2])
                        + float(meta[c].get("rest_tail_world", [0.0, 0.0, 0.0])[2]),
                    )
                    if b == lowest_sib:
                        assigned_kind[b] = "jaw"
                        assigned_sub_role[b] = "jaw_hinge"
                    else:
                        assigned_kind[b] = "head"
                        assigned_sub_role[b] = "head_leaf" if not children_map[b] else "neck_chain"
                elif dy < -0.02 * height and dz < 0.0 and not children_map[b] and assigned_kind.get(p or "") == "head":
                    assigned_kind[b] = "jaw"
                    assigned_sub_role[b] = "jaw_hinge"
                else:
                    assigned_kind[b] = "head"
                    assigned_sub_role[b] = "head_leaf" if not children_map[b] else "neck_chain"
            else:
                assigned_kind[b] = "spine"
                assigned_sub_role[b] = "spine_hub" if len(children_map[b]) >= 2 else "spine_chain"
        else:
            side_b = _geom_side(b)
            assigned_side[b] = side_b
            if p and assigned_kind.get(p) == "rigid" and p == "Core_Rigid":
                assigned_kind[b] = "flex"
                assigned_sub_role[b] = "flex_chain"
            else:
                assigned_kind[b] = "limb"
                if not children_map[b]:
                    assigned_sub_role[b] = "distal_limb"
                elif p and assigned_kind.get(p) in ("root", "spine", "rigid"):
                    assigned_sub_role[b] = "ball_socket"
                elif len(children_map[b]) == 1 and not children_map[children_map[b][0]]:
                    assigned_sub_role[b] = "forearm_hinge"
                else:
                    assigned_sub_role[b] = "upper_limb"

    # Step 6: Assign Canonical `<KIND>_<SIDE>_<GROUP>_<INDEX>` Names Deterministically
    # Group chains by (kind, side) and chain root
    group_counters: dict[tuple[str, str], int] = {}
    bone_group_id: dict[str, int] = {}
    bone_chain_idx: dict[str, int] = {}

    # Pre-number hand_hub limb groups so digit groups can use `<limb_group><finger_idx>`
    limb_hub_group: dict[str, int] = {}
    for b in topo_sorted:
        k = assigned_kind[b]
        s = assigned_side[b]
        p = meta[b].get("parent")
        if k == "root":
            bone_group_id[b] = 0
            bone_chain_idx[b] = 0
            continue
        if k == "digit":
            continue
        if p and assigned_kind.get(p) == k and assigned_side.get(p) == s and len(children_map[p]) == 1:
            bone_group_id[b] = bone_group_id[p]
            bone_chain_idx[b] = bone_chain_idx[p] + 1
        elif k in ("spine", "head", "tail") and s == "C":
            bone_group_id[b] = 0
            if p and assigned_kind.get(p) == k:
                bone_chain_idx[b] = bone_chain_idx[p] + 1
            else:
                bone_chain_idx[b] = group_counters.get((k, s), 0) + 1
                group_counters[(k, s)] = bone_chain_idx[b]
        else:
            grp = group_counters.get((k, s), 0) + 1
            group_counters[(k, s)] = grp
            bone_group_id[b] = grp
            bone_chain_idx[b] = 1
        if assigned_sub_role.get(b) == "hand_hub":
            limb_hub_group[b] = bone_group_id[b]

    # Number digit chains per parent hand_hub sorted by X/Y rest coordinates
    for b in topo_sorted:
        if assigned_kind[b] != "digit":
            continue
        p = meta[b].get("parent")
        s = assigned_side[b]
        if p and assigned_kind.get(p) == "digit":
            bone_group_id[b] = bone_group_id[p]
            bone_chain_idx[b] = bone_chain_idx[p] + 1
        else:
            hub_grp = limb_hub_group.get(p or "", 1)
            siblings = [c for c in children_map.get(p or "", []) if assigned_kind.get(c) == "digit"]
            siblings.sort(key=lambda c: tuple(meta[c].get("rest_head_world", [0.0, 0.0, 0.0])))
            finger_idx = (siblings.index(b) + 1) if b in siblings else (group_counters.get(("digit", s), 0) + 1)
            group_counters[("digit", s)] = max(group_counters.get(("digit", s), 0), finger_idx)
            bone_group_id[b] = min(99, hub_grp * 10 + finger_idx)
            bone_chain_idx[b] = 1

    # Populate enriched metadata on every bone
    for b in bone_names:
        info = meta[b]
        k = assigned_kind[b]
        sub = assigned_sub_role[b]
        s = assigned_side[b]
        grp = bone_group_id.get(b, 0)
        c_idx = bone_chain_idx.get(b, 1)
        is_leaf = len(children_map[b]) == 0
        lim_src = str(info.get("limits_source", ""))
        has_custom_limits = bool(info.get("limits_rad")) and lim_src != "stage1_heuristic"
        def_lims, def_corr, def_fixed = default_limits_and_corr_for_kind(k, sub, is_leaf=is_leaf)

        info["kind"] = k
        info["sub_role"] = sub
        info["side"] = s
        info["group_id"] = f"{grp:02d}"
        info["chain_index"] = c_idx
        info["legacy_name"] = info.get("legacy_name", b)
        info["canonical_name"] = (
            b if parse_canonical_bone_name(b) is not None else format_canonical_bone_name(k, s, grp, c_idx)
        )
        info["fixed_local"] = bool(info.get("fixed_local", def_fixed))
        info["is_primary"] = k != "digit"
        if not has_custom_limits:
            info["limits_rad"] = [[round(lo, 4), round(hi, 4)] for (lo, hi) in def_lims]
            if not lim_src or lim_src == "stage1_heuristic":
                info["limits_source"] = "canonical_default"
        if "chain_corr" not in info or info["chain_corr"] is None or lim_src == "stage1_heuristic":
            info["chain_corr"] = 0.0 if has_custom_limits else def_corr

    return meta


def canonicalize_rig_bone_names_in_blender(
    arm_obj: Any,
    mesh_objs: list[Any],
    classified_bones: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Renames bones in a Blender Armature (`arm_obj`), corresponding `mesh_obj.vertex_groups`,
    and animation fCurve data paths to their canonical `<KIND>_<SIDE>_<GROUP>_<INDEX>` names.
    Returns `{old_name: canonical_name}` mapping.
    """
    rename_map: dict[str, str] = {}
    used_names: set[str] = set()
    for old_name, info in sorted(classified_bones.items()):
        cand = str(info.get("canonical_name", old_name))
        if cand in used_names:
            idx = int(info.get("chain_index", 1))
            while cand in used_names:
                idx += 1
                cand = format_canonical_bone_name(info["kind"], info["side"], info["group_id"], idx)
        used_names.add(cand)
        rename_map[old_name] = cand

    # Two-pass rename via temporary prefix to avoid collisions
    for old_name, new_name in rename_map.items():
        if old_name == new_name:
            continue
        tmp_name = f"__tmp_canon_{old_name}"
        b = arm_obj.data.bones.get(old_name)
        if b is not None:
            b.name = tmp_name
        for m in mesh_objs:
            if hasattr(m, "vertex_groups") and old_name in m.vertex_groups:
                m.vertex_groups[old_name].name = tmp_name

    for old_name, new_name in rename_map.items():
        if old_name == new_name:
            continue
        tmp_name = f"__tmp_canon_{old_name}"
        b = arm_obj.data.bones.get(tmp_name)
        if b is not None:
            b.name = new_name
        for m in mesh_objs:
            if hasattr(m, "vertex_groups") and tmp_name in m.vertex_groups:
                m.vertex_groups[tmp_name].name = new_name

    # Update animation fCurves if present
    anim = getattr(arm_obj, "animation_data", None)
    if anim and getattr(anim, "action", None):
        for fc in anim.action.fcurves:
            dp = getattr(fc, "data_path", "")
            for old_name, new_name in rename_map.items():
                if old_name != new_name and f'pose.bones["{old_name}"]' in dp:
                    fc.data_path = dp.replace(f'pose.bones["{old_name}"]', f'pose.bones["{new_name}"]')
                    break

    return rename_map


def validate_canonical_rig_bone_names(
    bones_meta_or_names: dict[str, Any] | list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Mandatory preflight gate that validates whether every deforming bone name conforms to
    the Schema v2 `<KIND>_<SIDE>_<GROUP>_<INDEX>` naming contract (`_CANONICAL_NAME_RE`).
    Returns a structured validation summary including `valid`, `non_canonical_bones`,
    `counts_by_kind`, and `counts_by_side`.
    """
    if isinstance(bones_meta_or_names, dict):
        bone_names = sorted(str(k) for k in bones_meta_or_names.keys())
    else:
        bone_names = [str(k) for k in bones_meta_or_names]

    canonical_bones: list[str] = []
    non_canonical_bones: list[str] = []
    counts_by_kind: dict[str, int] = {}
    counts_by_side: dict[str, int] = {}

    for b_name in bone_names:
        parsed = parse_canonical_bone_name(b_name)
        if parsed is None:
            non_canonical_bones.append(b_name)
            continue
        canonical_bones.append(b_name)
        k = str(parsed["kind"])
        s = str(parsed["side"])
        counts_by_kind[k] = counts_by_kind.get(k, 0) + 1
        counts_by_side[s] = counts_by_side.get(s, 0) + 1

    has_unique_names = len(set(bone_names)) == len(bone_names)
    return {
        "schema": BONE_TAXONOMY_SCHEMA,
        "valid": bool(len(bone_names) > 0 and len(non_canonical_bones) == 0 and has_unique_names),
        "total_bones": len(bone_names),
        "canonical_count": len(canonical_bones),
        "non_canonical_bones": non_canonical_bones,
        "has_unique_names": has_unique_names,
        "counts_by_kind": counts_by_kind,
        "counts_by_side": counts_by_side,
    }


# ==============================================================================
# Layer 2: Universal Kinematic Core (Short-Gap Cleaning, IRLS Kabsch, Bounded FK)
# ==============================================================================

def short_gap_fill_1d_or_nd(values: np.ndarray, bad_mask: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Linearly interpolates ONLY short contiguous outlier gaps (`<= max_gap` frames) bounded
    by valid observations on BOTH sides. Never replaces long low-confidence intervals with stale frames.
    """
    result = values.copy()
    T = len(bad_mask)
    t = 0
    while t < T:
        if not bad_mask[t]:
            t += 1
            continue
        end = t
        while end < T and bad_mask[end]:
            end += 1
        if t > 0 and end < T and (end - t) <= max_gap:
            for j in range(t, end):
                u = float(j - t + 1) / float(end - t + 1)
                result[j] = (1.0 - u) * values[t - 1] + u * values[end]
        t = end
    return result


def clean_uv_short_gaps_hampel(uv: np.ndarray, scale: float = 1.0, max_gap: int = 3) -> np.ndarray:
    """Applies a 5-frame rolling median Hampel spike detector and interpolates only bounded gaps `<= max_gap`."""
    out = uv.copy()
    T, K, _ = uv.shape
    if T < 5:
        return out
    for k in range(K):
        seq = uv[:, k].astype(np.float64)
        pad = np.pad(seq, ((2, 2), (0, 0)), mode="edge")
        med = np.median(np.stack([pad[j : j + T] for j in range(5)], axis=0), axis=0)
        dev = np.linalg.norm(seq - med, axis=1)
        threshold = max(22.0 * scale, 3.5 * (float(np.median(dev)) + scale))
        out[:, k] = short_gap_fill_1d_or_nd(seq, dev > threshold, max_gap=max_gap)
    return out


_WHITTAKER_PENALTY_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _get_whittaker_diff_gramians(T: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns cached `(D2^T D2, D3^T D3)` difference Gramian matrices of shape `(T, T)`."""
    cached = _WHITTAKER_PENALTY_CACHE.get(T)
    if cached is not None:
        return cached
    eye = np.eye(T, dtype=np.float64)
    d2 = eye[2:] - 2.0 * eye[1:-1] + eye[:-2]
    d3 = eye[3:] - 3.0 * eye[2:-1] + 3.0 * eye[1:-2] - eye[:-3]
    gram = (d2.T @ d2, d3.T @ d3)
    if len(_WHITTAKER_PENALTY_CACHE) < 16:
        _WHITTAKER_PENALTY_CACHE[T] = gram
    return gram


def whittaker_smooth_trajectory(
    y: np.ndarray,
    weights: np.ndarray | None = None,
    lam_accel: float = 4.0,
    lam_jerk: float = 10.0,
) -> np.ndarray:
    """Zero-phase confidence-weighted Whittaker-Henderson + Minimum-Jerk (Flash & Hogan 1985;
    Eilers 2003) trajectory smoother along axis 0:
        min_X  sum_t w_t ||x_t - y_t||^2 + lam_accel ||D_2 X||_F^2 + lam_jerk ||D_3 X||_F^2
    Supports 2D `(T, D)` and 3D `(T, K, D)` arrays. Falls back to `savgol_smooth_array` when `T < 7`.
    """
    T = int(y.shape[0])
    if T < 7 or (lam_accel <= 0.0 and lam_jerk <= 0.0):
        return savgol_smooth_array(y)

    arr = np.asarray(y, dtype=np.float64)
    g2, g3 = _get_whittaker_diff_gramians(T)
    penalty = float(lam_accel) * g2 + float(lam_jerk) * g3

    if arr.ndim == 2:
        if weights is None:
            w = np.ones(T, dtype=np.float64)
        else:
            w = np.clip(np.asarray(weights, dtype=np.float64).reshape(T), 0.05, 1.0)
        A = np.diag(w) + penalty
        return np.linalg.solve(A, w[:, None] * arr)

    if arr.ndim == 3:
        _, K, D = arr.shape
        out = np.empty_like(arr, dtype=np.float64)
        if weights is None:
            A = np.eye(T, dtype=np.float64) + penalty
            flat = arr.reshape(T, K * D)
            return np.linalg.solve(A, flat).reshape(T, K, D)
        w_all = np.clip(np.asarray(weights, dtype=np.float64), 0.05, 1.0)
        for k in range(K):
            wk = w_all[:, k]
            A = np.diag(wk) + penalty
            out[:, k, :] = np.linalg.solve(A, wk[:, None] * arr[:, k, :])
        return out

    return savgol_smooth_array(y)


def savgol_smooth_array(x: np.ndarray) -> np.ndarray:
    """Zero-phase 5-tap quadratic Savitzky-Golay filter `[-3, 12, 17, 12, -3] / 35` along axis 0."""
    if len(x) < 5:
        return x.copy()
    pad_spec = [(2, 2)] + [(0, 0)] * (x.ndim - 1)
    p = np.pad(x.astype(np.float64), pad_spec, mode="edge")
    return (-3.0 * p[:-4] + 12.0 * p[1:-3] + 17.0 * p[2:-2] + 12.0 * p[3:-1] - 3.0 * p[4:]) / 35.0


def euler_xyz_to_matrix_numpy(rx: float, ry: float, rz: float) -> np.ndarray:
    """Computes Blender's extrinsic `'XYZ'` PoseBone local rotation matrix `Rz(rz) @ Ry(ry) @ Rx(rx)`."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )


def matrix_to_euler_xyz_numpy(R: np.ndarray) -> np.ndarray:
    """Extracts extrinsic `'XYZ'` Euler angles `(rx, ry, rz)` in radians from a 3x3 rotation matrix."""
    ry = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    return np.array(
        [
            math.atan2(float(R[2, 1]), float(R[2, 2])),
            ry,
            math.atan2(float(R[1, 0]), float(R[0, 0])),
        ],
        dtype=np.float64,
    )


def minimum_swing_matrix_numpy(curr_dir: np.ndarray, target_dir: np.ndarray) -> np.ndarray:
    """Computes the minimum-angle rotation matrix in SO(3) mapping `curr_dir -> target_dir`."""
    u = curr_dir / max(1e-8, float(np.linalg.norm(curr_dir)))
    v = target_dir / max(1e-8, float(np.linalg.norm(target_dir)))
    cross = np.cross(u, v)
    sin_a = float(np.linalg.norm(cross))
    cos_a = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if sin_a < 1e-6:
        if cos_a >= 0.0:
            return np.eye(3, dtype=np.float64)
        # 180-degree flip around an orthogonal axis (stay in SO(3) with det = +1)
        ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(float(u[0])) < 0.9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        ax = np.cross(u, ortho)
        ax /= max(1e-8, float(np.linalg.norm(ax)))
        return 2.0 * np.outer(ax, ax) - np.eye(3, dtype=np.float64)
    axis = cross / sin_a
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + sin_a * K + (1.0 - cos_a) * (K @ K)


def build_bounded_local_euler_rotation(
    d_loc: np.ndarray,
    ry: float,
    rx_lims: tuple[float, float] | list[float],
    rz_lims: tuple[float, float] | list[float],
) -> np.ndarray:
    """Two-branch inverse-sine local Euler rotation `Rz(rz) @ Ry(ry) @ Rx(rx)` strictly obeying
    `(rx_lims, rz_lims)` WITHOUT any post-clamp `minimum_swing` override.
    """
    d_unit = d_loc / max(1e-12, float(np.linalg.norm(d_loc)))
    cy, sy = math.cos(ry), math.sin(ry)
    sx = float(np.clip(d_unit[2] / max(1e-6, cy), -1.0, 1.0))
    rx1 = math.asin(sx)
    rx2 = (math.pi - rx1) if rx1 >= 0.0 else (-math.pi - rx1)
    best_R = None
    best_align = -2.0
    for rx_cand in (rx1, rx2):
        rx_c = float(np.clip(rx_cand, float(rx_lims[0]), float(rx_lims[1])))
        a = sy * math.sin(rx_c)
        b = math.cos(rx_c)
        rz_raw = math.atan2(
            a * float(d_unit[1]) - b * float(d_unit[0]),
            a * float(d_unit[0]) + b * float(d_unit[1]),
        )
        rz_c = float(np.clip(rz_raw, float(rz_lims[0]), float(rz_lims[1])))
        R_cand = euler_xyz_to_matrix_numpy(rx_c, ry, rz_c)
        align = float(np.dot(R_cand[:, 1], d_unit)) - 1e-4 * (rx_c ** 2)
        if align > best_align:
            best_align = align
            best_R = R_cand
    assert best_R is not None
    return best_R


def build_swing_twist_local_rotation(d_loc: np.ndarray, ry: float) -> np.ndarray:
    """Gimbal-free Swing-Twist local rotation matrix:
    `R_loc(ry) = minimum_swing([0, 1, 0]^T, d_loc) @ Ry(ry)`.
    Maps `[0, 1, 0]^T` to `d_loc` on the entire sphere S^2 without middle-axis `Ry` gimbal lock.
    """
    d_unit = d_loc / max(1e-12, float(np.linalg.norm(d_loc)))
    R_swing = minimum_swing_matrix_numpy(np.array([0.0, 1.0, 0.0], dtype=np.float64), d_unit)
    cy, sy = math.cos(ry), math.sin(ry)
    R_twist = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=np.float64,
    )
    return R_swing @ R_twist


def clamp_swing_twist_delta(
    R_delta: np.ndarray, limits_rad: list[list[float]], margin_rad: float = math.radians(15.0)
) -> np.ndarray:
    """Clamps a local rotation matrix `R_delta` in Gimbal-Free Swing-Twist space:
    caps the swing cone angle `arccos(d_loc[1])` to the anatomical `(rx, rz)` envelope and
    clamps axial Y-twist `tau` to `limits_rad[1]` without gimbal-locked Euler coupling distortion.
    """
    d_loc = R_delta[:, 1].copy()
    d_loc /= max(1e-12, float(np.linalg.norm(d_loc)))
    max_rx = max(abs(float(limits_rad[0][0])), abs(float(limits_rad[0][1])))
    max_rz = max(abs(float(limits_rad[2][0])), abs(float(limits_rad[2][1])))
    max_swing = min(math.pi * 0.95, math.hypot(max_rx, max_rz) + float(margin_rad))
    cos_sw = float(np.clip(d_loc[1], -1.0, 1.0))
    ang_sw = math.acos(cos_sw)
    if ang_sw > max_swing and ang_sw > 1e-6:
        u = max_swing / ang_sw
        y_ax = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        d_loc = (
            math.sin((1.0 - u) * ang_sw) / math.sin(ang_sw) * y_ax
            + math.sin(u * ang_sw) / math.sin(ang_sw) * d_loc
        )
        d_loc /= max(1e-12, float(np.linalg.norm(d_loc)))
    R_swing = minimum_swing_matrix_numpy(np.array([0.0, 1.0, 0.0], dtype=np.float64), d_loc)
    R_twist = R_swing.T @ R_delta
    tau = math.atan2(float(R_twist[0, 2]), float(R_twist[0, 0]))
    tau_c = float(np.clip(tau, float(limits_rad[1][0]), float(limits_rad[1][1])))
    return build_swing_twist_local_rotation(d_loc, tau_c)


def clamp_local_delta_euler(R_delta: np.ndarray, limits_rad: list[list[float]]) -> np.ndarray:
    """Clamps a local rotation matrix `R_delta` to `limits_rad` across both Euler branches."""
    r = matrix_to_euler_xyz_numpy(R_delta)
    alt = np.array([r[0] + math.pi, math.pi - r[1], r[2] + math.pi], dtype=np.float64)
    alt = (alt + math.pi) % (2.0 * math.pi) - math.pi
    lims_arr = np.array(limits_rad, dtype=np.float64)
    best_R = None
    best_dist = 1e18
    for cand in (r, alt):
        v = np.clip(cand, lims_arr[:, 0], lims_arr[:, 1])
        Q = euler_xyz_to_matrix_numpy(float(v[0]), float(v[1]), float(v[2]))
        dist = float(np.linalg.norm(Q - R_delta))
        if dist < best_dist:
            best_dist = dist
            best_R = Q
    assert best_R is not None
    return best_R


def mat_to_quat_wxyz_numpy(R: np.ndarray) -> np.ndarray:
    """Converts a 3x3 rotation matrix to a unit quaternion `[w, x, y, z]`."""
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / max(1e-12, float(np.linalg.norm(q)))


def quat_wxyz_to_mat_numpy(q: np.ndarray) -> np.ndarray:
    """Converts a unit quaternion `[w, x, y, z]` to a 3x3 rotation matrix."""
    w, x, y, z = q / max(1e-12, float(np.linalg.norm(q)))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def blend_rotation_matrices_slerp(
    A: np.ndarray, B: np.ndarray, alpha: float, max_step_deg: float = 18.0
) -> np.ndarray:
    """Blends rotation matrix `A` toward `B` with step-size clamping (`max_step_deg`)."""
    qa = mat_to_quat_wxyz_numpy(A)
    qb = mat_to_quat_wxyz_numpy(B)
    dot = float(np.dot(qa, qb))
    if dot < 0.0:
        qb = -qb
        dot = -dot
    angle = 2.0 * math.acos(min(1.0, dot))
    step_alpha = min(float(alpha), math.radians(max_step_deg) / max(angle, 1e-12))
    q = (1.0 - step_alpha) * qa + step_alpha * qb
    q /= max(1e-12, float(np.linalg.norm(q)))
    return quat_wxyz_to_mat_numpy(q)


def select_root_kabsch_anchors(
    bones_meta: dict[str, dict[str, Any]],
    root_name: str,
    children_map: dict[str, list[str]],
) -> list[tuple[str, int]]:
    """Selects non-collinear body anchor keypoints `[(bone_name, endpoint_0_or_1), ...]`
    whose 3D positions are rigidly determined by the root bone's 6-DoF pose:
      - `(root, 0)` (head) and `(root, 1)` (tail)
      - `(child, 0)` for all direct children of `root` (their heads have fixed rest offsets in root space)
      - `(child, 1)` for all direct rigid support children (`fixed_local=True` or `kind == "rigid"`),
        providing full 3D lateral + sagittal lever arms to prevent ill-conditioned Kabsch yaw twists.
    """
    anchors: list[tuple[str, int]] = [(root_name, 0), (root_name, 1)]
    for c in children_map.get(root_name, []):
        anchors.append((c, 0))
    for c in children_map.get(root_name, []):
        c_info = bones_meta.get(c, {})
        if c_info.get("fixed_local") or c_info.get("kind") == "rigid":
            anchors.append((c, 1))
    return anchors


def fit_root_rigid_irls_kabsch(
    rest_pts: np.ndarray,
    obs_pts: np.ndarray,
    weights: np.ndarray,
    depth_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """4-iteration IRLS Huber-like weighted Kabsch SVD solver returning `(R_world, t_world)`."""
    w = np.maximum(np.asarray(weights, dtype=np.float64), 1e-4)
    w /= w.sum()
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    for _ in range(4):
        x = (rest_pts * w[:, None]).sum(axis=0)
        y = (obs_pts * w[:, None]).sum(axis=0)
        cov = (rest_pts - x).T @ (w[:, None] * (obs_pts - y))
        U, _, Vt = np.linalg.svd(cov)
        det_sign = 1.0 if float(np.linalg.det(Vt.T @ U.T)) >= 0.0 else -1.0
        R = Vt.T @ np.diag([1.0, 1.0, det_sign]) @ U.T
        t = y - R @ x
        residual = np.linalg.norm(rest_pts @ R.T + t - obs_pts, axis=1)
        w = np.maximum(w * np.minimum(1.0, 0.018 * float(depth_scale) / np.maximum(residual, 1e-9)), 1e-6)
        w /= w.sum()
    return R, t


# ==============================================================================
# Layer 3: Kind-Specific Kinematic Solvers
# ==============================================================================

def _child_twist_cost(
    b_name: str,
    R_cand: np.ndarray,
    v_parent: np.ndarray,
    h_b: np.ndarray,
    valid_children: list[str],
    frame_bones: dict[str, Any],
    bones_meta: dict[str, dict[str, Any]],
    rest_R: dict[str, np.ndarray],
    obs_lookup: dict[str, Any] | None = None,
) -> float:
    """Evaluates multi-child disconnected lever-arm + child hinge/shaft limit cost for candidate `R_cand`."""
    info = bones_meta.get(b_name, {})
    rest_h_b = np.array(info.get("rest_head_world", h_b), dtype=np.float64)
    rest_t_b = np.array(info.get("rest_tail_world", h_b), dtype=np.float64)
    raw_src = obs_lookup if obs_lookup is not None else frame_bones
    h_b_obs = np.array(raw_src.get(b_name, {}).get("head_3d", h_b), dtype=np.float64)
    deg = math.radians
    default_lims = [[-deg(35), deg(45)], [-deg(5), deg(5)], [-deg(18), deg(18)]]
    cost = 0.0

    for c in valid_children:
        c_info = bones_meta.get(c, {})
        c_data = raw_src.get(c, frame_bones[c])
        h_c = np.array(c_data["head_3d"], dtype=np.float64)
        t_c = np.array(c_data["tail_3d"], dtype=np.float64)

        gap_rest = 0.0
        if "rest_head_world" in c_info and "rest_head_world" in info and "rest_tail_world" in info:
            rest_h_c = np.array(c_info["rest_head_world"], dtype=np.float64)
            gap_rest = float(np.linalg.norm(rest_h_c - rest_t_b))
            if gap_rest > 0.008:
                r_rest = rest_h_c - rest_h_b
                r_curr = (R_cand @ rest_R[b_name].T) @ r_rest
                r_targ = h_c - h_b_obs
                p_c = r_curr - np.dot(r_curr, v_parent) * v_parent
                p_t = r_targ - np.dot(r_targ, v_parent) * v_parent
                nc, nt = float(np.linalg.norm(p_c)), float(np.linalg.norm(p_t))
                if nc > 0.015 and (0.45 * nc <= nt <= 1.85 * nc):
                    ang_diff = math.atan2(
                        float(np.dot(v_parent, np.cross(p_c / nc, p_t / nt))),
                        float(np.clip(np.dot(p_c / nc, p_t / nt), -1.0, 1.0)),
                    )
                    w_lever = min(12.0, (nc / 0.06) ** 2)
                    cost += w_lever * (ang_diff ** 2)

        vec_c = t_c - h_c
        norm_c = float(np.linalg.norm(vec_c))
        if norm_c < 1e-6:
            continue
        v_c = vec_c / norm_c
        sin_perp = float(np.linalg.norm(v_c - np.dot(v_c, v_parent) * v_parent))
        if sin_perp > 0.16:
            c_lims = c_info.get("limits_rad", default_lims)
            R_c_init = R_cand @ (rest_R[b_name].T @ rest_R[c])
            d_c_loc = R_c_init.T @ v_c
            sx_c = float(np.clip(d_c_loc[2], -1.0, 1.0))
            rx1 = math.asin(sx_c)
            rx2 = (math.pi - rx1) if rx1 >= 0.0 else (-math.pi - rx1)
            c_best_pen = 1e18
            c_rz_span = float(c_lims[2][1]) - float(c_lims[2][0])
            c_sub = str(c_info.get("sub_role", ""))
            coincident_hinge = gap_rest <= 0.008 and c_sub in ("forearm_hinge", "distal_limb")
            for rxc in (rx1, rx2):
                pen_x = max(0.0, float(c_lims[0][0]) - rxc) + max(0.0, rxc - float(c_lims[0][1]))
                cxc = math.cos(rxc)
                rzc = math.atan2(-cxc * float(d_c_loc[0]), cxc * float(d_c_loc[1]))
                pen_z = max(0.0, float(c_lims[2][0]) - rzc) + max(0.0, rzc - float(c_lims[2][1]))
                if coincident_hinge:
                    hinge_pull = 2.5 * (rzc ** 2) / max(0.06, sin_perp ** 2)
                elif c_rz_span <= math.radians(15.0):
                    hinge_pull = 0.25 * (rzc ** 2)
                else:
                    hinge_pull = 0.0
                pen = (pen_x ** 2) + 8.0 * (pen_z ** 2) + hinge_pull
                if pen < c_best_pen:
                    c_best_pen = pen
            cost += (sin_perp ** 2) * c_best_pen * 6.0

    return cost


def _map_child_pivot_to_parent_local_y(
    b_name: str,
    c_name: str,
    R_init: np.ndarray,
    h_b_curr: np.ndarray,
    h_c_obs: np.ndarray,
    bones_meta: dict[str, dict[str, Any]],
    rest_R: dict[str, np.ndarray],
) -> np.ndarray | None:
    """Maps an observed child head pivot vector `h_c_obs - h_b_curr` into bone `b_name`'s local
    Y-axis direction `[0, 1, 0]^T` by compensating for the exact rest-pose angular offset between
    `h_c_rest - h_b_rest` and `t_b_rest - h_b_rest`.
    """
    info_b = bones_meta.get(b_name, {})
    info_c = bones_meta.get(c_name, {})
    if "rest_head_world" not in info_b or "rest_head_world" not in info_c:
        return None
    v_rest = np.array(info_c["rest_head_world"], dtype=np.float64) - np.array(info_b["rest_head_world"], dtype=np.float64)
    n_rest = float(np.linalg.norm(v_rest))
    if n_rest < 1e-4:
        return None
    u_rest_loc = rest_R[b_name].T @ (v_rest / n_rest)
    S_c2t_loc = minimum_swing_matrix_numpy(u_rest_loc, np.array([0.0, 1.0, 0.0], dtype=np.float64))

    v_obs = h_c_obs - h_b_curr
    n_obs = float(np.linalg.norm(v_obs))
    if n_obs < 1e-4:
        return None
    u_obs_loc = R_init.T @ (v_obs / n_obs)
    d_mapped = S_c2t_loc @ u_obs_loc
    n_map = float(np.linalg.norm(d_mapped))
    return (d_mapped / n_map) if n_map > 1e-8 else None


def solve_bone_rotation_by_kind(
    b_name: str,
    R_init: np.ndarray,
    frame_bones: dict[str, Any],
    bones_meta: dict[str, dict[str, Any]],
    rest_R: dict[str, np.ndarray],
    children_map: dict[str, list[str]],
    curr_ry_map: dict[str, float],
    prev_ry_map: dict[str, float],
    raw_obs_bones: dict[str, dict[str, Any]] | None = None,
    lims_override: list[list[float]] | None = None,
) -> tuple[np.ndarray, float]:
    """Layer 3 Dispatcher: Solves bone `b_name`'s 3x3 world rotation matrix `R_final` and axial
    twist `ry_best` according to its classified anatomical `kind` and `sub_role`.
    """
    info = bones_meta.get(b_name, {})
    kind = str(info.get("kind", "limb"))
    sub_role = str(info.get("sub_role", ""))

    # 1. Rigid / Fixed-Local Solver
    if kind == "rigid" or bool(info.get("fixed_local", False)):
        return R_init.copy(), 0.0

    b_data = frame_bones[b_name]
    h_b = np.array(b_data["head_3d"], dtype=np.float64)
    t_b = np.array(b_data["tail_3d"], dtype=np.float64)
    vec = t_b - h_b
    norm_v = float(np.linalg.norm(vec))
    if norm_v < 1e-7:
        return R_init.copy(), 0.0
    v = vec / norm_v
    d_loc = R_init.T @ v

    deg = math.radians
    default_lims = [[-deg(35), deg(45)], [-deg(5), deg(5)], [-deg(18), deg(18)]]
    lims = lims_override if lims_override is not None else info.get("limits_rad", default_lims)
    rx_lims = (float(lims[0][0]), float(lims[0][1]))
    ry_lo, ry_hi = float(lims[1][0]), float(lims[1][1])
    rz_lims = (float(lims[2][0]), float(lims[2][1]))

    p_name = info.get("parent")
    corr = float(info.get("chain_corr", 0.0))
    ry_prior = (
        float(np.clip(corr * curr_ry_map.get(p_name, 0.0), ry_lo, ry_hi))
        if (corr > 0.0 and p_name in curr_ry_map)
        else 0.0
    )
    if b_name in prev_ry_map:
        ry_prior = 0.55 * prev_ry_map[b_name] + 0.45 * ry_prior

    obs_lookup = raw_obs_bones if raw_obs_bones is not None else frame_bones
    valid_children = [c for c in children_map.get(b_name, []) if c in obs_lookup and c in rest_R]

    # 2. Digit / Phalange Coupled Curl Solver (Gimbal-Free Swing-Twist, tau = 0, coupled curl damping)
    if kind == "digit":
        R_sw = minimum_swing_matrix_numpy(np.array([0.0, 1.0, 0.0], dtype=np.float64), d_loc)
        delta_raw = clamp_local_delta_euler(R_sw, lims)
        p_children = children_map.get(p_name, []) if p_name else []
        gp_name = bones_meta.get(p_name, {}).get("parent") if p_name else None
        gp_children = children_map.get(gp_name, []) if gp_name else []
        rest_len_d = float(info.get("rest_length", np.linalg.norm(t_b - h_b)))
        is_multi_digit_fan = (len(p_children) >= 3 or len(gp_children) >= 3 or rest_len_d < 0.035)
        if is_multi_digit_fan:
            delta_damped = clamp_local_delta_euler(
                blend_rotation_matrices_slerp(np.eye(3, dtype=np.float64), delta_raw, 0.20, 2.5),
                lims,
            )
            return R_init @ delta_damped, 0.0
        return R_init @ delta_raw, 0.0

    # 3. Limb Solver: Gimbal-Free Swing-Twist + Palm Centroid Swing (`hand_hub`) + Child-Pivot Swing-Twist (`forearm_hinge`)
    if kind == "limb":
        explicit_custom = lims_override is not None or str(info.get("limits_source", "")) in ("agent_custom", "probe_initial")
        h_b_obs = np.array(obs_lookup.get(b_name, b_data)["head_3d"], dtype=np.float64)
        t_b_obs = np.array(obs_lookup.get(b_name, b_data)["tail_3d"], dtype=np.float64)
        obs_dir = t_b_obs - h_b_obs
        norm_obs_dir = float(np.linalg.norm(obs_dir))
        rest_h_b = np.array(info.get("rest_head_world", h_b_obs), dtype=np.float64)
        rest_t_b = np.array(info.get("rest_tail_world", t_b_obs), dtype=np.float64)
        rest_len_b = max(1e-5, float(np.linalg.norm(rest_t_b - rest_h_b)))

        if sub_role == "hand_hub" or len(valid_children) >= 3:
            R_sw_self = minimum_swing_matrix_numpy(np.array([0.0, 1.0, 0.0], dtype=np.float64), d_loc)
            conf_b = float(
                obs_lookup.get(b_name, b_data).get("tail_conf", obs_lookup.get(b_name, b_data).get("confidence", 0.5))
            )
            # If the short hand_hub bone is observed at exact rest length with high confidence (Oracle/clean),
            # its direct swing has <0.01 deg error; otherwise fuse with the single rest-to-observed palm centroid
            # vector and damp local wrist deflection to prevent 90-degree wrist snaps from finger keypoint scatter.
            if abs(norm_obs_dir - rest_len_b) < 0.04 * rest_len_b and conf_b >= 0.98:
                delta_self = clamp_local_delta_euler(R_sw_self, lims) if explicit_custom else R_sw_self
                return R_init @ delta_self, 0.0

            v_palm_rest = (rest_t_b - rest_h_b) / rest_len_b * conf_b
            v_palm_obs = (obs_dir / max(1e-8, norm_obs_dir)) * conf_b
            for fc in valid_children:
                fc_info = bones_meta.get(fc, {})
                if "rest_tail_world" not in fc_info:
                    continue
                rest_t_fc = np.array(fc_info["rest_tail_world"], dtype=np.float64)
                v_fc_rest = rest_t_fc - rest_h_b
                n_fc_rest = float(np.linalg.norm(v_fc_rest))
                t_fc_obs = np.array(obs_lookup[fc]["tail_3d"], dtype=np.float64)
                v_fc_obs = t_fc_obs - h_b_obs
                n_fc_obs = float(np.linalg.norm(v_fc_obs))
                if n_fc_rest < 1e-5 or n_fc_obs < 1e-5:
                    continue
                conf_fc = float(obs_lookup[fc].get("tail_conf", obs_lookup[fc].get("confidence", 0.5))) * 0.45
                v_palm_rest += (v_fc_rest / n_fc_rest) * conf_fc
                v_palm_obs += (v_fc_obs / n_fc_obs) * conf_fc

            n_pr = float(np.linalg.norm(v_palm_rest))
            n_po = float(np.linalg.norm(v_palm_obs))
            if n_pr > 1e-6 and n_po > 1e-6:
                u_pr_loc = rest_R[b_name].T @ (v_palm_rest / n_pr)
                u_po_loc = R_init.T @ (v_palm_obs / n_po)
                R_sw_palm = minimum_swing_matrix_numpy(u_pr_loc, u_po_loc)
                R_sw_comb = blend_rotation_matrices_slerp(R_sw_self, R_sw_palm, 0.45, 30.0)
            else:
                R_sw_comb = R_sw_self

            R_sw_damped = blend_rotation_matrices_slerp(np.eye(3, dtype=np.float64), R_sw_comb, 0.68, 25.0)
            if explicit_custom:
                R_sw_damped = clamp_local_delta_euler(R_sw_damped, lims)
            return R_init @ R_sw_damped, 0.0

        if sub_role == "forearm_hinge" and len(valid_children) == 1:
            ha = valid_children[0]
            ha_info = bones_meta.get(ha, {})
            if "rest_head_world" in ha_info:
                rest_h_ha = np.array(ha_info["rest_head_world"], dtype=np.float64)
                gap_ha = float(np.linalg.norm(rest_h_ha - rest_t_b))
                v_c_rest = rest_h_ha - rest_h_b
                n_c_rest = float(np.linalg.norm(v_c_rest))
                if n_c_rest > 1e-5 and gap_ha > 0.04 * rest_len_b:
                    y_ax = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                    u_c_rest_loc = rest_R[b_name].T @ (v_c_rest / n_c_rest)
                    S_c_to_t_loc = minimum_swing_matrix_numpy(u_c_rest_loc, y_ax)
                    u_t_obs = obs_dir / max(1e-8, norm_obs_dir)
                    v_c_obs = np.array(obs_lookup[ha]["head_3d"], dtype=np.float64) - h_b_obs
                    n_c_obs = float(np.linalg.norm(v_c_obs))
                    u_c_obs = v_c_obs / max(1e-8, n_c_obs)
                    d_t_loc = R_init.T @ u_t_obs
                    u_c_obs_loc = R_init.T @ u_c_obs
                    d_c_mapped_loc = S_c_to_t_loc @ u_c_obs_loc

                    # Chord-Ratio Gated Swing-Twist in [0,1,0] frame (zero spherical holonomy twist):
                    # Only treat the separation chord as clean/Oracle when BOTH bone tail and child head
                    # preserve their rest lengths (<8% length error) AND their angular chord matches rest pose.
                    chord_rest = float(np.linalg.norm(y_ax - u_c_rest_loc))
                    chord_obs = float(np.linalg.norm(d_t_loc - u_c_obs_loc))
                    rho_chord = chord_obs / max(1e-4, chord_rest)
                    len_t_ok = abs(norm_obs_dir - rest_len_b) < 0.08 * rest_len_b
                    len_c_ok = abs(n_c_obs - n_c_rest) < 0.08 * n_c_rest
                    w_clean = (
                        min(1.0, max(0.0, (rho_chord - 0.65) / 0.30) ** 2)
                        if (chord_rest > 0.10 and 0.65 < rho_chord < 1.30 and len_t_ok and len_c_ok)
                        else 0.0
                    )
                    if abs(rho_chord - 1.0) < 0.05 and len_t_ok and len_c_ok:
                        w_clean = 1.0

                    # If forearm tail is foreshortened relative to child wrist pivot, favor child wrist pivot
                    rel_c = float(np.clip(0.62 + 0.30 * ((n_c_obs / n_c_rest) - (norm_obs_dir / rest_len_b)), 0.50, 0.88))
                    w_c = (1.0 - w_clean) * rel_c
                    d_blend = (1.0 - w_c) * d_t_loc + w_c * d_c_mapped_loc
                    d_loc = d_blend / max(1e-8, float(np.linalg.norm(d_blend)))
                    R_sw = minimum_swing_matrix_numpy(y_ax, d_loc)
                    target_in_rest = R_sw.T @ u_c_obs_loc
                    px0, pz0 = float(u_c_rest_loc[0]), float(u_c_rest_loc[2])
                    px1, pz1 = float(target_in_rest[0]), float(target_in_rest[2])
                    n0 = math.hypot(px0, pz0)
                    n1 = math.hypot(px1, pz1)
                    tau = 0.0
                    if n0 > 0.15 and n1 > 0.10:
                        tau_raw = math.atan2(px1 * pz0 - pz1 * px0, px1 * px0 + pz1 * pz0)
                        w_tau_noisy = min(0.55, ((n1 / n0) ** 2) * 0.45)
                        w_tau = w_clean * 1.0 + (1.0 - w_clean) * w_tau_noisy
                        max_tau = math.radians(65.0) if w_clean > 0.5 else math.radians(42.0)
                        tau = float(np.clip(w_tau * tau_raw, -max_tau, max_tau))
                        if explicit_custom and (ry_hi - ry_lo) > 1e-6:
                            tau = float(np.clip(tau, ry_lo, ry_hi))
                    R_loc_fa = build_swing_twist_local_rotation(d_loc, tau)
                    if explicit_custom:
                        R_loc_fa = clamp_local_delta_euler(R_loc_fa, lims)
                    return R_init @ R_loc_fa, tau

        if sub_role not in ("upper_limb", "forearm_hinge", "ball_socket"):
            R_sw = minimum_swing_matrix_numpy(np.array([0.0, 1.0, 0.0], dtype=np.float64), d_loc)
            if explicit_custom:
                R_sw = clamp_local_delta_euler(R_sw, lims)
            return R_init @ R_sw, 0.0

    # 4. Spine / Head / Jaw / Flex / Tail / Wing Solver (Bounded Local-Euler + Child Observability)
    if not valid_children or (ry_hi - ry_lo) < 1e-4:
        ry_best = 0.0 if not valid_children else ry_prior
        R_loc = build_bounded_local_euler_rotation(d_loc, ry_best, rx_lims, rz_lims)
        return R_init @ R_loc, ry_best

    grid_ry = np.linspace(ry_lo, ry_hi, 11)
    best_ry = ry_prior
    best_cost = 1e18
    prior_w = 1.25 if (corr > 0.0 or b_name in prev_ry_map) else 0.05
    for ry_cand in grid_ry:
        ry_f = float(ry_cand)
        R_cand = R_init @ build_bounded_local_euler_rotation(d_loc, ry_f, rx_lims, rz_lims)
        cost = prior_w * ((ry_f - ry_prior) ** 2) + _child_twist_cost(
            b_name, R_cand, v, h_b, valid_children, frame_bones, bones_meta, rest_R, obs_lookup=obs_lookup
        )
        if cost < best_cost:
            best_cost = cost
            best_ry = ry_f

    R_loc_best = build_bounded_local_euler_rotation(d_loc, best_ry, rx_lims, rz_lims)
    return R_init @ R_loc_best, best_ry


def _build_default_rest_rot_matrix(d0: np.ndarray) -> np.ndarray:
    """Constructs a right-handed 3x3 rest rotation matrix `[x_ax, d0, z_ax]` whose local `+Y` axis is `d0`
    and whose local `+X` axis aligns with world `+X` projected orthogonal to `d0` (matching Blender's
    `+X` bone-roll convention so sagittal pitch rotations correspond to local `rx`).
    """
    x_proj = np.array([1.0, 0.0, 0.0], dtype=np.float64) - d0 * float(d0[0])
    if float(np.linalg.norm(x_proj)) < 1e-6:
        x_proj = np.array([0.0, 0.0, 1.0], dtype=np.float64) - d0 * float(d0[2])
    x_ax = x_proj / max(1e-8, float(np.linalg.norm(x_proj)))
    z_ax = np.cross(x_ax, d0)
    z_ax /= max(1e-8, float(np.linalg.norm(z_ax)))
    return np.column_stack([x_ax, d0, z_ax])


def refine_frames_with_dense_lbs(
    frames_out: list[dict[str, Any]],
    bone_names: list[str],
    classified_bones: dict[str, dict[str, Any]],
    calib: dict[str, Any],
    rig_skinning: dict[str, Any],
    dense_correspondences: list[dict[str, dict[str, np.ndarray]]] | dict[str, dict[str, np.ndarray]],
    effective_base_limits: dict[str, list[list[float]]] | None = None,
) -> tuple[dict[str, Any], dict[str, list[list[float]]]]:
    """Warm-started Multi-View Differentiable/Analytical Linear Blend Skinning (LBS) Surface &
    Axial Roll Refiner.

    Starts from the Layer 2/3 kinematic pose solution in `frames_out` and uses matched 3D rest-surface
    correspondences across all active orthographic views (`'front'`, `'side'`/`'left'`, `'right'`)
    to refine each bone's axial twist/roll (`ry`) and fine swing (`rx, rz`) within anatomical limits.
    """
    from pose_features import (
        lbs_deform_vertices_numpy,
        match_dense_surface_correspondences,
        project_3d_points_to_ortho_view_numpy,
    )

    names = list(bone_names)
    B = len(names)
    index = {n: i for i, n in enumerate(names)}
    order = sorted(names, key=lambda n: int(classified_bones[n].get("depth", 0)))
    T = len(frames_out)

    v_rest = np.asarray(rig_skinning.get("rest_verts", []), dtype=np.float64)
    skin_idx = np.asarray(rig_skinning.get("skin_indices", []), dtype=np.int64)
    skin_w = np.asarray(rig_skinning.get("skin_weights", []), dtype=np.float64)
    if v_rest.ndim != 2 or v_rest.shape[0] == 0 or T == 0:
        return {"dense_surface_refined": False, "dense_matched_points_total": 0}, {}

    if skin_idx.ndim == 1:
        skin_idx = skin_idx[:, None]
    if skin_w.ndim == 1:
        skin_w = skin_w[:, None]

    if isinstance(dense_correspondences, dict):
        per_frame_matches = match_dense_surface_correspondences(dense_correspondences, v_rest, calib)
    else:
        per_frame_matches = list(dense_correspondences)

    if len(per_frame_matches) < T:
        per_frame_matches.extend({} for _ in range(T - len(per_frame_matches)))

    rest_H = np.zeros((B, 3), dtype=np.float64)
    rest_T = np.zeros((B, 3), dtype=np.float64)
    rest_R = np.zeros((B, 3, 3), dtype=np.float64)
    for bi, n in enumerate(names):
        info = classified_bones[n]
        rest_H[bi] = np.asarray(info.get("rest_head_world", [0.0, 0.0, 0.0]), dtype=np.float64)
        rest_T[bi] = np.asarray(info.get("rest_tail_world", [0.0, 0.0, 0.04]), dtype=np.float64)
        if "rest_rot_mat_world" in info:
            rest_R[bi] = np.asarray(info["rest_rot_mat_world"], dtype=np.float64)
        else:
            v0 = rest_T[bi] - rest_H[bi]
            d0 = v0 / max(1e-8, float(np.linalg.norm(v0)))
            rest_R[bi] = _build_default_rest_rot_matrix(d0)

    # Precompute per-vertex per-bone effective weight matrix W_vb (V, B) and off-axis radius r_perp (V, B)
    V = v_rest.shape[0]
    W_vb = np.zeros((V, B), dtype=np.float64)
    for m in range(skin_idx.shape[1]):
        b_col = np.clip(skin_idx[:, m], 0, B - 1)
        np.add.at(W_vb, (np.arange(V), b_col), skin_w[:, m])

    W_panel = float(calib.get("panel_width", 672.0))
    H_panel = float(calib.get("panel_height", 768.0))
    depth_scale = max(1e-4, float(calib.get("depth_scale", float(calib.get("ortho_scale", 1.0)) / 1.48)))
    res_scale = H_panel / 768.0
    huber_delta_px = 4.0 * res_scale
    outlier_gate_px = 0.14 * W_panel

    deg = math.radians
    unlocked_limits: dict[str, list[list[float]]] = {}
    views_used_set: set[str] = set()
    total_matched_pts = 0
    err_before_list: list[float] = []
    err_after_list: list[float] = []

    for f in range(T):
        f_matches = per_frame_matches[f]
        if not f_matches:
            continue

        bmap = frames_out[f]["bones"]
        posed_H = np.zeros((B, 3), dtype=np.float64)
        posed_R = np.zeros((B, 3, 3), dtype=np.float64)
        for bi, n in enumerate(names):
            posed_H[bi] = np.asarray(bmap[n]["head_3d"], dtype=np.float64)
            posed_R[bi] = np.asarray(bmap[n]["rot_mat_world"], dtype=np.float64)
        posed_R_warm = posed_R.copy()

        v_posed_curr = lbs_deform_vertices_numpy(
            v_rest, skin_idx, skin_w, rest_H, rest_R, posed_H, posed_R
        )

        # Evaluate initial reprojection residuals and apply warm-start kinematic outlier gate
        active_view_obs: dict[str, dict[str, np.ndarray]] = {}
        for v_name, obs_v in f_matches.items():
            v_idx = np.asarray(obs_v["vert_indices"], dtype=np.int64)
            t_uv = np.asarray(obs_v["target_uv"], dtype=np.float64)
            w_pts = np.asarray(obs_v["weights"], dtype=np.float64)
            if len(v_idx) == 0:
                continue
            uv_init, _ = project_3d_points_to_ortho_view_numpy(v_posed_curr[v_idx], calib, v_name)
            res_init = np.linalg.norm(uv_init - t_uv, axis=-1)
            inlier = res_init <= outlier_gate_px
            if not np.any(inlier):
                continue
            v_idx_in = v_idx[inlier]
            t_uv_in = t_uv[inlier]
            # Robust Geman-McClure style warm-start weight modulation
            w_in = w_pts[inlier] / (1.0 + (res_init[inlier] / (0.045 * W_panel)) ** 2)
            active_view_obs[v_name] = {
                "vert_indices": v_idx_in,
                "target_uv": t_uv_in,
                "weights": w_in,
            }
            views_used_set.add(v_name)
            total_matched_pts += int(len(v_idx_in))
            err_before_list.extend(res_init[inlier].tolist())

        if not active_view_obs:
            continue

        dense_children_map: dict[str, list[str]] = {n: [] for n in names}
        for n in names:
            pn = classified_bones[n].get("parent")
            if pn and pn in dense_children_map:
                dense_children_map[pn].append(n)

        for n in order:
            bi = index[n]
            info = classified_bones[n]
            kind = str(info.get("kind", "limb"))
            p_name = info.get("parent")
            if p_name and p_name in index:
                pi = index[p_name]
                init_R = posed_R[pi] @ (rest_R[pi].T @ rest_R[bi])
                init_R_warm = posed_R_warm[pi] @ (rest_R[pi].T @ rest_R[bi])
                head_b = posed_H[pi] + posed_R[pi] @ (rest_R[pi].T @ (rest_H[bi] - rest_H[pi]))
            else:
                init_R = rest_R[bi]
                init_R_warm = rest_R[bi]
                head_b = posed_H[bi]

            # Update bone head if parent moved slightly, and adjust v_posed_curr accordingly
            dh = head_b - posed_H[bi]
            if float(np.linalg.norm(dh)) > 1e-8:
                v_posed_curr += W_vb[:, bi : bi + 1] * dh[None, :]
                posed_H[bi] = head_b

            # Propagate warm-start local delta into the updated parent frame init_R
            delta_warm = init_R_warm.T @ posed_R_warm[bi]
            R_warm = init_R @ delta_warm
            if float(np.linalg.norm(R_warm - posed_R[bi])) > 1e-8:
                all_b_verts = np.nonzero(W_vb[:, bi] > 1e-5)[0]
                if len(all_b_verts) > 0:
                    wb_all = W_vb[all_b_verts, bi : bi + 1]
                    r_loc_all = wb_all * ((v_rest[all_b_verts] - rest_H[bi]) @ rest_R[bi])
                    v_posed_curr[all_b_verts] += r_loc_all @ (R_warm.T - posed_R[bi].T)
                posed_R[bi] = R_warm

            if kind == "rigid" or bool(info.get("fixed_local", False)):
                rest_len = float(info.get("rest_length", np.linalg.norm(rest_T[bi] - rest_H[bi])))
                tail_b = head_b + posed_R[bi][:, 1] * rest_len
                bmap[n]["head_3d"] = [round(float(x), 5) for x in head_b]
                bmap[n]["tail_3d"] = [round(float(x), 5) for x in tail_b]
                bmap[n]["rot_mat_world"] = [[round(float(x), 5) for x in row] for row in posed_R[bi]]
                continue

            # Gather views where matched vertices have skinning weight on bone `bi`
            bone_view_terms = []
            off_axis_count = 0
            u_shaft_rest = rest_R[bi][:, 1]
            for v_name, v_obs in active_view_obs.items():
                v_idx = v_obs["vert_indices"]
                w_bone = W_vb[v_idx, bi]
                has_w = w_bone >= 0.15
                if not np.any(has_w):
                    continue
                sel_v = v_idx[has_w]
                sel_wb = w_bone[has_w]
                sel_uv = v_obs["target_uv"][has_w]
                sel_w = v_obs["weights"][has_w] * sel_wb

                offsets_rest = v_rest[sel_v] - rest_H[bi]
                proj_shaft = (offsets_rest @ u_shaft_rest)[:, None] * u_shaft_rest[None, :]
                r_perp = np.linalg.norm(offsets_rest - proj_shaft, axis=-1)
                off_axis_count += int(np.sum(r_perp >= 0.008 * depth_scale))

                # Weighted local rest vector r_local_w = w_{i,b} * R_rest^T (v_rest - h_rest)
                r_local_w = sel_wb[:, None] * ((offsets_rest) @ rest_R[bi])
                v_base = v_posed_curr[sel_v] - (r_local_w @ posed_R[bi].T)
                bone_view_terms.append((v_name, sel_v, r_local_w, v_base, sel_uv, sel_w))

            is_leaf_bone = len(dense_children_map.get(n, [])) == 0
            if not bone_view_terms or not (is_leaf_bone or "dense_roll_limits_rad" in info):
                rest_len = float(info.get("rest_length", np.linalg.norm(rest_T[bi] - rest_H[bi])))
                tail_b = head_b + posed_R[bi][:, 1] * rest_len
                bmap[n]["head_3d"] = [round(float(x), 5) for x in head_b]
                bmap[n]["tail_3d"] = [round(float(x), 5) for x in tail_b]
                bmap[n]["rot_mat_world"] = [[round(float(x), 5) for x in row] for row in posed_R[bi]]
                continue

            base_lims = (
                copy.deepcopy(effective_base_limits[n])
                if (effective_base_limits and n in effective_base_limits)
                else copy.deepcopy(info.get("limits_rad", [[-deg(35), deg(35)], [-deg(20), deg(20)], [-deg(25), deg(25)]]))
            )
            rx_lims = (float(base_lims[0][0]), float(base_lims[0][1]))
            ry_lo, ry_hi = float(base_lims[1][0]), float(base_lims[1][1])
            rz_lims = (float(base_lims[2][0]), float(base_lims[2][1]))

            # Unlock axial roll limits ONLY on leaf/paddle bones with sufficient off-axis surface support
            # (or bones with explicit dense_roll_limits_rad) where axial roll cannot twist downstream chains!
            if (off_axis_count >= 12 or "dense_roll_limits_rad" in info) and (ry_hi - ry_lo) < deg(15.0):
                dense_roll = info.get("dense_roll_limits_rad", [-deg(40.0), deg(40.0)])
                ry_lo = min(ry_lo, float(dense_roll[0]))
                ry_hi = max(ry_hi, float(dense_roll[1]))
                base_lims[1] = [ry_lo, ry_hi]
                unlocked_limits[n] = copy.deepcopy(base_lims)

            R_warm = posed_R[bi]
            delta_warm = init_R.T @ R_warm
            e_warm = matrix_to_euler_xyz_numpy(delta_warm)
            rx_0 = float(np.clip(e_warm[0], rx_lims[0], rx_lims[1]))
            ry_0 = float(np.clip(e_warm[1], ry_lo, ry_hi))
            rz_0 = float(np.clip(e_warm[2], rz_lims[0], rz_lims[1]))
            d_loc_0 = init_R.T @ R_warm[:, 1]

            def _eval_candidates_cost(R_cands: np.ndarray, ry_vals: np.ndarray, rx_vals: np.ndarray, rz_vals: np.ndarray) -> np.ndarray:
                # R_cands: (G, 3, 3) world rotation matrices
                costs = 0.20 * ((ry_vals - ry_0) ** 2) + 8.0 * ((rx_vals - rx_0) ** 2 + (rz_vals - rz_0) ** 2)
                for v_name, _, r_local_w, v_base, sel_uv, sel_w in bone_view_terms:
                    # (G, P, 3)
                    pts_cand = v_base[None, :, :] + np.einsum("gij,pj->gpi", R_cands, r_local_w)
                    uv_cand, _ = project_3d_points_to_ortho_view_numpy(pts_cand, calib, v_name)
                    diff_px = uv_cand - sel_uv[None, :, :]
                    dist_sq = np.sum(diff_px * diff_px, axis=-1)
                    # Pseudo-Huber loss in pixel space normalized by huber_delta_px
                    huber = (huber_delta_px ** 2) * (np.sqrt(1.0 + dist_sq / (huber_delta_px ** 2)) - 1.0)
                    costs = costs + np.sum(sel_w[None, :] * huber, axis=-1) / max(1.0, float(np.sum(sel_w)))
                return costs

            best_R = R_warm
            best_ry = ry_0
            best_rx = rx_0
            best_rz = rz_0

            # Stage A: Coarse-to-fine Axial Roll (ry) search preserving exact bone shaft direction d_loc_0
            if (ry_hi - ry_lo) > 1e-4 and off_axis_count >= 2:
                coarse_ry = np.unique(np.concatenate([np.linspace(ry_lo, ry_hi, 17), [ry_0]]))
                R_coarse = np.stack(
                    [init_R @ build_bounded_local_euler_rotation(d_loc_0, float(ry_c), rx_lims, rz_lims) for ry_c in coarse_ry],
                    axis=0,
                )
                c_coarse = _eval_candidates_cost(
                    R_coarse,
                    coarse_ry,
                    np.full_like(coarse_ry, rx_0),
                    np.full_like(coarse_ry, rz_0),
                )
                idx_c = int(np.argmin(c_coarse))
                ry_mid = float(coarse_ry[idx_c])
                half_step = max(deg(0.5), (ry_hi - ry_lo) / 16.0)
                fine_ry = np.unique(
                    np.concatenate([np.linspace(max(ry_lo, ry_mid - half_step), min(ry_hi, ry_mid + half_step), 11), [ry_0, ry_mid]])
                )
                R_fine = np.stack(
                    [init_R @ build_bounded_local_euler_rotation(d_loc_0, float(ry_c), rx_lims, rz_lims) for ry_c in fine_ry],
                    axis=0,
                )
                c_fine = _eval_candidates_cost(
                    R_fine,
                    fine_ry,
                    np.full_like(fine_ry, rx_0),
                    np.full_like(fine_ry, rz_0),
                )
                idx_f = int(np.argmin(c_fine))
                best_ry = float(fine_ry[idx_f])
                best_R = R_fine[idx_f]
                e_best = matrix_to_euler_xyz_numpy(init_R.T @ best_R)
                best_rx = float(np.clip(e_best[0], rx_lims[0], rx_lims[1]))
                best_rz = float(np.clip(e_best[2], rz_lims[0], rz_lims[1]))

            # Stage B: Local 2D surface reprojection fine swing polish around (best_rx, best_ry, best_rz)
            swing_rad = deg(1.5)
            rx_grid = np.unique(np.clip(best_rx + np.linspace(-swing_rad, swing_rad, 5), rx_lims[0], rx_lims[1]))
            rz_grid = np.unique(np.clip(best_rz + np.linspace(-swing_rad, swing_rad, 5), rz_lims[0], rz_lims[1]))
            if len(rx_grid) > 1 or len(rz_grid) > 1:
                cand_list = []
                rx_list = []
                ry_list = []
                rz_list = []
                for rxc in rx_grid:
                    for rzc in rz_grid:
                        cand_list.append(init_R @ euler_xyz_to_matrix_numpy(float(rxc), best_ry, float(rzc)))
                        rx_list.append(float(rxc))
                        ry_list.append(best_ry)
                        rz_list.append(float(rzc))
                R_sw = np.stack(cand_list, axis=0)
                c_sw = _eval_candidates_cost(
                    R_sw,
                    np.asarray(ry_list, dtype=np.float64),
                    np.asarray(rx_list, dtype=np.float64),
                    np.asarray(rz_list, dtype=np.float64),
                )
                idx_sw = int(np.argmin(c_sw))
                best_R = R_sw[idx_sw]

            # Update v_posed_curr for all vertices weighted to bone `bi`
            all_b_verts = np.nonzero(W_vb[:, bi] > 1e-5)[0]
            if len(all_b_verts) > 0:
                wb_all = W_vb[all_b_verts, bi : bi + 1]
                r_loc_all = wb_all * ((v_rest[all_b_verts] - rest_H[bi]) @ rest_R[bi])
                v_posed_curr[all_b_verts] += r_loc_all @ (best_R.T - posed_R[bi].T)

            posed_R[bi] = best_R
            rest_len = float(info.get("rest_length", np.linalg.norm(rest_T[bi] - rest_H[bi])))
            tail_b = head_b + best_R[:, 1] * rest_len
            bmap[n]["head_3d"] = [round(float(x), 5) for x in head_b]
            bmap[n]["tail_3d"] = [round(float(x), 5) for x in tail_b]
            bmap[n]["rot_mat_world"] = [[round(float(x), 5) for x in row] for row in best_R]

        # Compute post-refinement 2D reprojection residuals
        for v_name, v_obs in active_view_obs.items():
            v_idx = v_obs["vert_indices"]
            uv_after, _ = project_3d_points_to_ortho_view_numpy(v_posed_curr[v_idx], calib, v_name)
            res_after = np.linalg.norm(uv_after - v_obs["target_uv"], axis=-1)
            err_after_list.extend(res_after.tolist())

    mean_before = round(float(np.mean(err_before_list)), 4) if err_before_list else 0.0
    mean_after = round(float(np.mean(err_after_list)), 4) if err_after_list else 0.0
    dense_diag = {
        "dense_surface_refined": total_matched_pts > 0,
        "dense_matched_points_total": total_matched_pts,
        "dense_views_used": sorted(views_used_set),
        "mean_reproj_error_px_before": mean_before,
        "mean_reproj_error_px_after": mean_after,
        "unlocked_roll_bones": sorted(unlocked_limits.keys()),
    }
    return dense_diag, unlocked_limits


def reconstruct_frames_with_kind_solvers(
    pts_3d: np.ndarray,
    trust: np.ndarray,
    bone_names: list[str],
    classified_bones: dict[str, dict[str, Any]],
    depth_scale: float = 1.0,
    dense_correspondences: list[dict[str, dict[str, np.ndarray]]] | dict[str, dict[str, np.ndarray]] | None = None,
    rig_skinning: dict[str, Any] | None = None,
    calib: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Executes Layer 2 (Universal Kinematic Core: multi-anchor IRLS Kabsch root fit + exact
    rest-offset FK + strict local Euler limit clamping) and Layer 3 (Kind-Specific Solvers)
    across `T` frames of 3D keypoint observations `pts_3d` (`(T, 2*N, 3)`) and `trust` (`(T, 2*N)`).
    """
    names = list(bone_names)
    index = {n: i for i, n in enumerate(names)}
    T = int(pts_3d.shape[0])
    order = sorted(names, key=lambda n: int(classified_bones[n].get("depth", 0)))
    root = order[0]

    rest_R: dict[str, np.ndarray] = {}
    rest_H: dict[str, np.ndarray] = {}
    rest_T: dict[str, np.ndarray] = {}
    children_map: dict[str, list[str]] = {n: [] for n in names}

    for n in names:
        info = classified_bones[n]
        p = info.get("parent")
        if p and p in children_map:
            children_map[p].append(n)
        rest_H[n] = np.array(info.get("rest_head_world", [0.0, 0.0, 0.0]), dtype=np.float64)
        rest_T[n] = np.array(info.get("rest_tail_world", [0.0, 0.0, 0.04]), dtype=np.float64)
        if "rest_rot_mat_world" in info:
            rest_R[n] = np.array(info["rest_rot_mat_world"], dtype=np.float64)
        else:
            v0 = rest_T[n] - rest_H[n]
            d0 = v0 / max(1e-8, float(np.linalg.norm(v0)))
            rest_R[n] = _build_default_rest_rot_matrix(d0)

    all_rest_pts = np.vstack([np.stack(list(rest_H.values())), np.stack(list(rest_T.values()))])
    rig_span = max(1e-4, float(np.max(np.ptp(all_rest_pts, axis=0))))
    if T >= 2:
        step_dists = np.linalg.norm(pts_3d[1:].astype(np.float64) - pts_3d[:-1].astype(np.float64), axis=-1)
        rel_median_speed = float(np.median(step_dists)) / rig_span
    else:
        rel_median_speed = 0.0
    # Preserve full smoothing on slow/moderate clips (rel_median_speed <= 0.005);
    # scale down causal IIR lag & Whittaker stiffness on high-velocity dynamic clips so fast strides are not lagged
    motion_smooth_scale = (
        1.0
        if rel_median_speed <= 0.005
        else float(np.clip((0.0045 / max(1e-6, rel_median_speed)) ** 3, 0.04, 1.0))
    )

    anchors = select_root_kabsch_anchors(classified_bones, root, children_map)
    ref_pts = np.array([rest_H[n] if e == 0 else rest_T[n] for n, e in anchors], dtype=np.float64)
    anchor_ids = [2 * index[n] + e for n, e in anchors]

    deg = math.radians
    # Sanitize corrupted empirical Euler wrap-around bounds on short hand/foot hub bones
    effective_base_limits: dict[str, list[list[float]]] = {}
    for n in names:
        info = classified_bones[n]
        raw_lims = copy.deepcopy(info["limits_rad"])
        sub = str(info.get("sub_role", ""))
        lim_src = str(info.get("limits_source", ""))
        if lim_src not in ("agent_custom", "probe_initial"):
            if sub == "hand_hub":
                max_abs = max(abs(float(pair[0])) for pair in raw_lims)
                max_abs = max(max_abs, max(abs(float(pair[1])) for pair in raw_lims))
                if max_abs > deg(45.0):
                    raw_lims = [[-deg(32.0), deg(32.0)], [-deg(12.0), deg(12.0)], [-deg(28.0), deg(28.0)]]
            elif sub == "forearm_hinge":
                max_span = max(float(pair[1]) - float(pair[0]) for pair in raw_lims)
                if max_span > deg(130.0):
                    raw_lims = [[-math.pi, math.pi], [-math.pi, math.pi], [-math.pi, math.pi]]
        effective_base_limits[n] = raw_lims

    axial_nodes = [
        n
        for n in order
        if n == root
        or (
            classified_bones[n].get("kind") in ("spine", "head")
            and classified_bones[n].get("side") == "C"
            and float(rest_T[n][2] - rest_H[n][2]) > 0.30 * max(1e-6, float(np.linalg.norm(rest_T[n] - rest_H[n])))
            and float(rest_T[n][1] - rest_H[n][1]) <= 0.15 * max(1e-6, float(np.linalg.norm(rest_T[n] - rest_H[n])))
        )
    ]
    top_axial = axial_nodes[-1] if len(axial_nodes) >= 2 else None
    v_axial_rest = (rest_T[top_axial] - rest_H[root]) if top_axial is not None else np.zeros(3, dtype=np.float64)
    dz_axial_rest = float(v_axial_rest[2]) if top_axial is not None else 0.0
    dxy_axial_rest = math.hypot(float(v_axial_rest[0]), float(v_axial_rest[1])) if top_axial is not None else 0.0
    is_upright_axial = bool(top_axial is not None and dz_axial_rest > 0.08 and dz_axial_rest > 1.25 * dxy_axial_rest)
    frame_macro_collapse = np.zeros(T, dtype=np.float64)

    frames_out: list[dict[str, Any]] = []
    previous_delta: dict[str, np.ndarray] = {}
    prev_ry: dict[str, float] = {}
    previous_root_h: np.ndarray | None = None
    max_violation = 0.0

    # Kind-specific temporal slerp parameters (alpha, max_step_deg)
    slerp_params_by_kind = {
        "root": (0.65, 8.0),
        "rigid": (0.0, 0.0),
        "spine": (0.78, 18.0),
        "head": (0.78, 18.0),
        "jaw": (0.78, 18.0),
        "limb": (0.72, 16.0),
        "flex": (0.78, 18.0),
        "tail": (0.78, 18.0),
        "wing": (0.78, 18.0),
        "digit": (0.50, 2.5),
    }

    coincident_pairs: list[tuple[int, int]] = []
    for n in order:
        p = classified_bones[n].get("parent")
        if p and p in rest_T and float(np.linalg.norm(rest_H[n] - rest_T[p])) < 1e-3:
            coincident_pairs.append((2 * index[p] + 1, 2 * index[n]))

    for f in range(T):
        observed = pts_3d[f].astype(np.float64).copy()
        for ip, ic in coincident_pairs:
            wp = max(1e-4, float(trust[f, ip])) ** 2
            wc = max(1e-4, float(trust[f, ic])) ** 2
            fused_pt = (wp * observed[ip] + wc * observed[ic]) / (wp + wc)
            observed[ip] = fused_pt
            observed[ic] = fused_pt

        # Detect macro-axial vertical foreshortening on upright bipedal trunks (e.g., prone collapse / deep bow)
        macro_collapse = 0.0
        u_macro: np.ndarray | None = None
        if is_upright_axial and top_axial is not None:
            v_macro_obs = observed[2 * index[top_axial] + 1] - observed[2 * index[root]]
            dz_obs = float(v_macro_obs[2])
            if dz_obs < 0.78 * dz_axial_rest and float(np.linalg.norm(v_macro_obs)) > 0.15 * dz_axial_rest:
                macro_collapse = float(np.clip((0.78 - dz_obs / dz_axial_rest) / 0.22, 0.0, 1.0))
                vx_m, vy_m = float(v_macro_obs[0]), float(v_macro_obs[1])
                target_len_m = 0.92 * dz_axial_rest
                if vx_m * vx_m + dz_obs * dz_obs < target_len_m * target_len_m and float(np.linalg.norm(v_macro_obs)) < target_len_m:
                    vy_taylor = math.sqrt(max(0.0, target_len_m * target_len_m - vx_m * vx_m - dz_obs * dz_obs))
                    sy_m = -1.0 if vy_m <= 0.02 * dz_axial_rest else 1.0
                    vy_m = (1.0 - 0.80 * macro_collapse) * vy_m + (0.80 * macro_collapse) * (sy_m * vy_taylor)
                v_m_comp = np.array([vx_m, vy_m, dz_obs], dtype=np.float64)
                u_macro = v_m_comp / max(1e-8, float(np.linalg.norm(v_m_comp)))
        frame_macro_collapse[f] = macro_collapse

        fit_R, _ = fit_root_rigid_irls_kabsch(
            ref_pts,
            observed[anchor_ids],
            trust[f, anchor_ids] ** 2,
            depth_scale=depth_scale,
        )
        root_lims = copy.deepcopy(effective_base_limits[root])
        if macro_collapse > 0.0:
            pitch_exp = macro_collapse * deg(30.0)
            root_lims[0] = [min(float(root_lims[0][0]), -pitch_exp), max(float(root_lims[0][1]), pitch_exp)]
            if u_macro is not None and top_axial is not None:
                v_ax_rest = rest_T[top_axial] - rest_H[root]
                u_ax_rest = v_ax_rest / max(1e-8, float(np.linalg.norm(v_ax_rest)))
                R_mac_sw = minimum_swing_matrix_numpy(u_ax_rest, u_macro)
                fit_R = blend_rotation_matrices_slerp(fit_R, R_mac_sw @ fit_R, 0.55 * macro_collapse, 28.0)
        root_delta = clamp_local_delta_euler(rest_R[root].T @ fit_R @ rest_R[root], root_lims)
        if root in previous_delta:
            root_alpha = 1.0 - (1.0 - 0.65) * motion_smooth_scale
            root_max_step = 8.0 / max(0.15, motion_smooth_scale)
            root_delta = clamp_local_delta_euler(
                blend_rotation_matrices_slerp(previous_delta[root], root_delta, root_alpha, root_max_step),
                root_lims,
            )
        root_world = rest_R[root] @ root_delta
        body_R = root_world @ rest_R[root].T

        anchor_weights = trust[f, anchor_ids] ** 2
        anchor_weights /= max(1e-12, float(anchor_weights.sum()))
        root_h = ((observed[anchor_ids] - (ref_pts - rest_H[root]) @ body_R.T) * anchor_weights[:, None]).sum(axis=0)
        if previous_root_h is not None:
            w_prev_h = 0.30 * motion_smooth_scale
            root_h = (1.0 - w_prev_h) * root_h + w_prev_h * previous_root_h
        previous_root_h = root_h.copy()

        posed_R: dict[str, np.ndarray] = {}
        heads: dict[str, np.ndarray] = {}
        bmap: dict[str, Any] = {}
        curr_ry: dict[str, float] = {}

        raw_obs_bones = {
            n: {
                "head_3d": observed[2 * i].tolist(),
                "tail_3d": observed[2 * i + 1].tolist(),
                "confidence": float(trust[f, 2 * i : 2 * i + 2].mean()),
                "tail_conf": float(trust[f, 2 * i + 1]),
            }
            for i, n in enumerate(names)
        }
        fk_targets = copy.deepcopy(raw_obs_bones)

        for n in order:
            info = classified_bones[n]
            p = info.get("parent")
            kind = str(info.get("kind", "limb"))
            sub_role = str(info.get("sub_role", ""))
            lims = copy.deepcopy(effective_base_limits[n])
            if macro_collapse > 0.0 and kind in ("spine", "head") and info.get("side") == "C":
                pitch_exp = macro_collapse * deg(35.0)
                lims[0] = [min(float(lims[0][0]), -pitch_exp), max(float(lims[0][1]), pitch_exp)]

            if p and p in posed_R:
                init_R = posed_R[p] @ (rest_R[p].T @ rest_R[n])
                head = heads[p] + posed_R[p] @ (rest_R[p].T @ (rest_H[n] - rest_H[p]))
            else:
                init_R = rest_R[n]
                head = root_h

            if n == root:
                delta = root_delta
                R_world = root_world
                ry = 0.0
            elif kind == "rigid" or bool(info.get("fixed_local", False)):
                delta = np.eye(3, dtype=np.float64)
                R_world = init_R
                ry = 0.0
            else:
                obs_dir = observed[2 * index[n] + 1] - observed[2 * index[n]]
                rest_len = float(info.get("rest_length", np.linalg.norm(rest_T[n] - rest_H[n])))

                # Macro-axial foreshortening completion with progressive flexion arc for midline spine/head segments:
                # In a deep prone/bow collapse, the chest/spine arch forms the apex while the neck/head bows
                # past the chord angle (~85-95 deg world pitch).
                if macro_collapse > 0.0 and u_macro is not None and kind in ("spine", "head") and info.get("side") == "C":
                    s_ax = (
                        float(axial_nodes.index(n)) / float(max(1, len(axial_nodes) - 1))
                        if n in axial_nodes
                        else 0.5
                    )
                    theta_mac = math.atan2(-float(u_macro[1]), max(1e-6, float(u_macro[2])))
                    theta_arc = min(math.radians(96.0), theta_mac * (0.72 + 1.08 * s_ax))
                    u_arc = np.array(
                        [u_macro[0], -math.sin(theta_arc), math.cos(theta_arc)],
                        dtype=np.float64,
                    )
                    u_arc /= max(1e-8, float(np.linalg.norm(u_arc)))
                    w_mac = min(0.92, 0.85 * macro_collapse)
                    obs_dir = (1.0 - w_mac) * obs_dir + w_mac * (u_arc * rest_len)

                # Taylor orthographic foreshortening + bilateral/gravity completion for occluded lateral limb segments
                if kind == "limb" and info.get("side") in ("L", "R"):
                    if macro_collapse > 0.30 and sub_role in ("upper_limb", "ball_socket"):
                        opp_side = "R" if info.get("side") == "L" else "L"
                        twin_up = [
                            tn
                            for tn in names
                            if classified_bones[tn].get("kind") == "limb"
                            and classified_bones[tn].get("sub_role") == sub_role
                            and classified_bones[tn].get("side") == opp_side
                            and classified_bones[tn].get("group_id") == info.get("group_id")
                        ]
                        if twin_up:
                            tn = twin_up[0]
                            v_tw = observed[2 * index[tn] + 1] - observed[2 * index[tn]]
                            v_tw_mir = np.array([-v_tw[0], v_tw[1], v_tw[2]], dtype=np.float64)
                            w_up_sym = 0.45 * macro_collapse
                            obs_dir = (1.0 - w_up_sym) * obs_dir + w_up_sym * v_tw_mir

                    if sub_role == "forearm_hinge" and len(children_map.get(n, [])) == 1:
                        ha = children_map[n][0]
                        if ha in raw_obs_bones:
                            h_n_obs = np.asarray(raw_obs_bones[n]["head_3d"], dtype=np.float64)
                            h_ha_obs = np.asarray(raw_obs_bones[ha]["head_3d"], dtype=np.float64)
                            vc = h_ha_obs - h_n_obs
                            lc_rest = float(np.linalg.norm(rest_H[ha] - rest_H[n]))
                            nc_obs = float(np.linalg.norm(vc))
                            if macro_collapse > 0.30 and lc_rest > 0.03:
                                # During deep prone collapse, regularize occluded/crossed forearms with bilateral symmetry + gravity hang
                                opp_side = "R" if info.get("side") == "L" else "L"
                                twin_Candidates = [
                                    tn
                                    for tn in names
                                    if classified_bones[tn].get("kind") == "limb"
                                    and classified_bones[tn].get("sub_role") == "forearm_hinge"
                                    and classified_bones[tn].get("side") == opp_side
                                    and classified_bones[tn].get("group_id") == info.get("group_id")
                                    and len(children_map.get(tn, [])) == 1
                                ]
                                vc_grav = np.array(
                                    [
                                        (0.025 if info.get("side") == "L" else -0.025) * lc_rest,
                                        -0.18 * lc_rest,
                                        -0.96 * lc_rest,
                                    ],
                                    dtype=np.float64,
                                )
                                if twin_Candidates:
                                    tn = twin_Candidates[0]
                                    tha = children_map[tn][0]
                                    vc_twin = observed[2 * index[tha]] - observed[2 * index[tn]]
                                    vc_mir = np.array([-vc_twin[0], vc_twin[1], vc_twin[2]], dtype=np.float64)
                                    vc_avg = 0.5 * (vc + vc_mir)
                                else:
                                    vc_avg = vc
                                w_grav = 0.55 * macro_collapse
                                vc_new = (1.0 - w_grav) * vc_avg + w_grav * vc_grav
                                raw_obs_bones[ha]["head_3d"] = (h_n_obs + vc_new).tolist()
                                raw_obs_bones[n]["tail_3d"] = (h_n_obs + vc_new).tolist()
                                obs_dir = vc_new
                            elif (
                                lc_rest > 0.03
                                and float(np.linalg.norm(rest_H[ha] - rest_T[n])) > 0.04 * rest_len
                                and nc_obs < 0.78 * lc_rest
                            ):
                                l_eff = 0.85 * lc_rest
                                dxz2 = float(vc[0] * vc[0] + vc[2] * vc[2])
                                if dxz2 < l_eff * l_eff:
                                    dy_tay = math.sqrt(max(0.0, l_eff * l_eff - dxz2))
                                    u_c0_world = init_R @ (rest_R[n].T @ ((rest_H[ha] - rest_H[n]) / lc_rest))
                                    sy = (
                                        math.copysign(1.0, float(vc[1]))
                                        if abs(float(vc[1])) > 0.06 * lc_rest
                                        else (-1.0 if float(u_c0_world[1]) <= 0.08 else 1.0)
                                    )
                                    w_fore = min(0.80, max(0.0, (0.78 - nc_obs / lc_rest) / 0.25))
                                    vc_new = np.array(
                                        [vc[0], (1.0 - w_fore) * vc[1] + w_fore * sy * dy_tay, vc[2]],
                                        dtype=np.float64,
                                    )
                                    raw_obs_bones[ha]["head_3d"] = (h_n_obs + vc_new).tolist()

                fk_targets[n] = {
                    "head_3d": head.tolist(),
                    "tail_3d": (head + obs_dir).tolist(),
                    "confidence": raw_obs_bones[n]["confidence"],
                    "tail_conf": raw_obs_bones[n]["tail_conf"],
                }
                R_solved, ry = solve_bone_rotation_by_kind(
                    b_name=n,
                    R_init=init_R,
                    frame_bones=fk_targets,
                    bones_meta=classified_bones,
                    rest_R=rest_R,
                    children_map=children_map,
                    curr_ry_map=curr_ry,
                    prev_ry_map=prev_ry,
                    raw_obs_bones=raw_obs_bones,
                    lims_override=lims,
                )
                delta = clamp_local_delta_euler(init_R.T @ R_solved, lims)
                if n in previous_delta:
                    alpha, max_step = slerp_params_by_kind.get(kind, (0.78, 18.0))
                    if kind == "digit":
                        p_ch = children_map.get(p, []) if p else []
                        gp = classified_bones.get(p, {}).get("parent") if p else None
                        gp_ch = children_map.get(gp, []) if gp else []
                        if len(p_ch) < 3 and len(gp_ch) < 3 and rest_len >= 0.035:
                            alpha, max_step = (0.72, 16.0)
                    alpha_eff = 1.0 - (1.0 - alpha) * motion_smooth_scale
                    max_step_eff = max_step / max(0.15, motion_smooth_scale)
                    delta = clamp_local_delta_euler(
                        blend_rotation_matrices_slerp(previous_delta[n], delta, alpha_eff, max_step_eff),
                        lims,
                    )
                R_world = init_R @ delta

            previous_delta[n] = delta.copy()
            curr_ry[n] = ry
            posed_R[n] = R_world
            heads[n] = head
            rest_len = float(info.get("rest_length", np.linalg.norm(rest_T[n] - rest_H[n])))
            tail = head + R_world[:, 1] * rest_len

            vals = matrix_to_euler_xyz_numpy(delta)
            chk_lims = np.array(lims, dtype=np.float64)
            violation = float(np.maximum(chk_lims[:, 0] - vals, vals - chk_lims[:, 1]).max())
            max_violation = max(max_violation, max(0.0, violation))

            bmap[n] = {
                "head_3d": [round(float(x), 5) for x in head],
                "tail_3d": [round(float(x), 5) for x in tail],
                "rot_mat_world": [[round(float(x), 5) for x in row] for row in R_world],
                "confidence": round(float(trust[f, 2 * index[n] : 2 * index[n] + 2].mean()), 4),
                "kind": kind,
                "canonical_name": info.get("canonical_name", n),
            }
        prev_ry = curr_ry
        frames_out.append({"frame": f, "bones": bmap})

    # Optional Stage 4.5: Warm-Started Multi-View Dense LBS Surface & Axial Roll Refinement
    dense_diag: dict[str, Any] = {"dense_surface_refined": False, "dense_matched_points_total": 0}
    unlocked_roll_limits: dict[str, list[list[float]]] = {}
    if dense_correspondences is not None and rig_skinning is not None and calib is not None:
        from pose_features import dense_surface_enabled
        if dense_surface_enabled():
            dense_diag, unlocked_roll_limits = refine_frames_with_dense_lbs(
                frames_out=frames_out,
                bone_names=names,
                classified_bones=classified_bones,
                calib=calib,
                rig_skinning=rig_skinning,
                dense_correspondences=dense_correspondences,
                effective_base_limits=effective_base_limits,
            )
            for b_un, lim_un in unlocked_roll_limits.items():
                effective_base_limits[b_un] = lim_un
            # Recompute max_violation under any unlocked surface-observed roll limits
            max_violation = 0.0
            for f in range(T):
                bmap_f = frames_out[f]["bones"]
                for n in order:
                    info = classified_bones[n]
                    if info.get("kind") == "rigid" or bool(info.get("fixed_local", False)):
                        continue
                    p = info.get("parent")
                    R_w = np.asarray(bmap_f[n]["rot_mat_world"], dtype=np.float64)
                    if p and p in bmap_f:
                        R_p = np.asarray(bmap_f[p]["rot_mat_world"], dtype=np.float64)
                        R_init = R_p @ (rest_R[p].T @ rest_R[n])
                    else:
                        R_init = rest_R[n]
                    vals = matrix_to_euler_xyz_numpy(R_init.T @ R_w)
                    chk_lims = np.array(effective_base_limits[n], dtype=np.float64)
                    violation = float(np.maximum(chk_lims[:, 0] - vals, vals - chk_lims[:, 1]).max())
                    max_violation = max(max_violation, max(0.0, violation))

    # Kind-specific zero-phase Whittaker-Henderson + Minimum-Jerk (lam_accel, lam_jerk) on S^3
    whittaker_so3_by_kind = {
        "root": (25.0, 60.0),
        "rigid": (0.0, 0.0),
        "spine": (18.0, 40.0),
        "head": (18.0, 40.0),
        "jaw": (14.0, 30.0),
        "limb": (8.0, 18.0),
        "flex": (16.0, 35.0),
        "tail": (16.0, 35.0),
        "wing": (12.0, 25.0),
        "digit": (12.0, 25.0),
    }

    if T >= 7:
        local_quats: dict[str, np.ndarray] = {n: np.zeros((T, 4), dtype=np.float64) for n in names}
        bone_confs: dict[str, np.ndarray] = {n: np.zeros(T, dtype=np.float64) for n in names}
        root_pos_seq = np.zeros((T, 3), dtype=np.float64)

        for f in range(T):
            bmap_f = frames_out[f]["bones"]
            root_pos_seq[f] = np.asarray(bmap_f[root]["head_3d"], dtype=np.float64)
            for n in order:
                p = classified_bones[n].get("parent")
                R_w = np.asarray(bmap_f[n]["rot_mat_world"], dtype=np.float64)
                if p and p in bmap_f:
                    R_p = np.asarray(bmap_f[p]["rot_mat_world"], dtype=np.float64)
                    R_init = R_p @ (rest_R[p].T @ rest_R[n])
                else:
                    R_init = rest_R[n]
                R_delta = R_init.T @ R_w
                q = mat_to_quat_wxyz_numpy(R_delta)
                if f > 0 and float(np.dot(local_quats[n][f - 1], q)) < 0.0:
                    q = -q
                local_quats[n][f] = q
                bone_confs[n][f] = float(bmap_f[n].get("confidence", 0.5))

        root_l2, root_l3 = whittaker_so3_by_kind["root"]
        root_pos_sm = whittaker_smooth_trajectory(
            root_pos_seq,
            weights=bone_confs[root] ** 2,
            lam_accel=root_l2 * motion_smooth_scale,
            lam_jerk=root_l3 * motion_smooth_scale,
        )
        sm_quats: dict[str, np.ndarray] = {}
        for n in order:
            info = classified_bones[n]
            k = str(info.get("kind", "limb"))
            if k == "rigid" or bool(info.get("fixed_local", False)):
                sm_quats[n] = local_quats[n]
                continue
            l2, l3 = whittaker_so3_by_kind.get(k, (10.0, 20.0))
            l2_eff = l2 * motion_smooth_scale
            l3_eff = l3 * motion_smooth_scale
            if mocap_priors_enabled():
                q_sm = whittaker_charbonnier_smooth_trajectory(
                    local_quats[n], weights=bone_confs[n] ** 2, lam_accel=l2_eff, lam_jerk=l3_eff
                )
                if k in ("limb", "digit", "tail", "flex", "wing"):
                    q_sm = dejitter_velocity_gated(q_sm, passes=1, strength=0.22 * motion_smooth_scale)
            else:
                q_sm = whittaker_smooth_trajectory(
                    local_quats[n], weights=bone_confs[n] ** 2, lam_accel=l2_eff, lam_jerk=l3_eff
                )
            q_sm /= np.maximum(1e-12, np.linalg.norm(q_sm, axis=-1, keepdims=True))
            sm_quats[n] = q_sm

        max_violation = 0.0
        for f in range(T):
            posed_R = {}
            heads = {}
            bmap_f = frames_out[f]["bones"]
            m_col = float(frame_macro_collapse[f])
            for n in order:
                info = classified_bones[n]
                p = info.get("parent")
                k = str(info.get("kind", "limb"))
                lims = copy.deepcopy(effective_base_limits[n])
                if m_col > 0.0:
                    if n == root:
                        pitch_exp = m_col * deg(28.0)
                        lims[0] = [min(float(lims[0][0]), -pitch_exp), max(float(lims[0][1]), pitch_exp)]
                    elif k in ("spine", "head") and info.get("side") == "C":
                        pitch_exp = m_col * deg(35.0)
                        lims[0] = [min(float(lims[0][0]), -pitch_exp), max(float(lims[0][1]), pitch_exp)]
                if p and p in posed_R:
                    init_R = posed_R[p] @ (rest_R[p].T @ rest_R[n])
                    head = heads[p] + posed_R[p] @ (rest_R[p].T @ (rest_H[n] - rest_H[p]))
                else:
                    init_R = rest_R[n]
                    head = root_pos_sm[f]

                if k == "rigid" or bool(info.get("fixed_local", False)):
                    delta = np.eye(3, dtype=np.float64)
                else:
                    delta = clamp_local_delta_euler(quat_wxyz_to_mat_numpy(sm_quats[n][f]), lims)
                R_world = init_R @ delta
                posed_R[n] = R_world
                heads[n] = head
                rest_len = float(info.get("rest_length", np.linalg.norm(rest_T[n] - rest_H[n])))
                tail = head + R_world[:, 1] * rest_len

                vals = matrix_to_euler_xyz_numpy(delta)
                chk_lims = np.array(lims, dtype=np.float64)
                violation = float(np.maximum(chk_lims[:, 0] - vals, vals - chk_lims[:, 1]).max())
                max_violation = max(max_violation, max(0.0, violation))

                bmap_f[n]["head_3d"] = [round(float(x), 5) for x in head]
                bmap_f[n]["tail_3d"] = [round(float(x), 5) for x in tail]
                bmap_f[n]["rot_mat_world"] = [[round(float(x), 5) for x in row] for row in R_world]

    use_priors = mocap_priors_enabled()
    if use_priors:
        contact_diag = apply_floor_and_foot_contact_constraints(
            frames_out=frames_out,
            order=order,
            root=root,
            classified_bones=classified_bones,
            rest_H=rest_H,
            rest_T=rest_T,
            rest_R=rest_R,
            effective_base_limits=effective_base_limits,
            frame_macro_collapse=frame_macro_collapse,
            clamp_local_delta_euler_fn=clamp_local_delta_euler,
            minimum_swing_matrix_fn=minimum_swing_matrix_numpy,
        )
    else:
        contact_diag = {
            "floor_clamped_frames": 0,
            "foot_contact_locked_frames": 0,
            "max_floor_penetration_fixed": 0.0,
            "contact_bones": [],
        }

    kind_counts: dict[str, int] = {}
    for n in names:
        k = str(classified_bones[n].get("kind", "limb"))
        kind_counts[k] = kind_counts.get(k, 0) + 1

    diag = {
        "schema": BONE_TAXONOMY_SCHEMA,
        "frames": T,
        "cross_frame_confidence_replacement": False,
        "short_gap_max_frames": 3,
        "post_smoothing": "whittaker_charbonnier_minimum_jerk_so3_v2" if use_priors else "whittaker_minimum_jerk_so3_v1",
        "mocap_priors_enabled": use_priors,
        "root_anchors": anchors,
        "final_limit_max_violation_rad": round(float(max_violation), 6),
        "fixed_local_bones": [n for n in names if classified_bones[n].get("fixed_local")],
        "kind_counts": kind_counts,
        **dense_diag,
        **contact_diag,
    }
    return frames_out, diag


def compute_stratified_motion_metrics(
    frames_pred: list[dict[str, Any]],
    frames_gt: list[dict[str, Any]],
    bones_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Computes both All-Bone metrics and Primary-Bone (excluding tiny `digit` phalanges) metrics,
    plus a per-`kind` anatomical breakdown (`root`, `rigid`, `spine`, `head`, `limb`, `flex`, `digit`, etc.).
    """
    classified = classify_rig_bones(bones_meta)
    bone_names = sorted(classified.keys())
    per_bone: dict[str, dict[str, list[float]]] = {
        b: {"direction": [], "rotation": [], "position": []} for b in bone_names
    }

    for f_pr, f_gt in zip(frames_pred, frames_gt):
        b_pr_map = f_pr.get("bones", f_pr.get("joints", {}))
        b_gt_map = f_gt.get("bones", f_gt.get("joints", {}))
        for b in bone_names:
            if b not in b_pr_map or b not in b_gt_map:
                continue
            pr = b_pr_map[b]
            gt = b_gt_map[b]
            h_p = np.array(pr["head_3d"], dtype=np.float64)
            t_p = np.array(pr["tail_3d"], dtype=np.float64)
            h_g = np.array(gt["head_3d"], dtype=np.float64)
            t_g = np.array(gt["tail_3d"], dtype=np.float64)

            v_p = t_p - h_p
            v_g = t_g - h_g
            np_len = float(np.linalg.norm(v_p))
            ng_len = float(np.linalg.norm(v_g))
            if np_len > 1e-6 and ng_len > 1e-6:
                cos_d = float(np.clip(np.dot(v_p / np_len, v_g / ng_len), -1.0, 1.0))
                per_bone[b]["direction"].append(math.degrees(math.acos(cos_d)))

            if "rot_mat_world" in pr and "rot_mat_world" in gt:
                R_p = np.array(pr["rot_mat_world"], dtype=np.float64)
                R_g = np.array(gt["rot_mat_world"], dtype=np.float64)
                tr = float(np.clip((np.trace(R_g.T @ R_p) - 1.0) * 0.5, -1.0, 1.0))
                per_bone[b]["rotation"].append(math.degrees(math.acos(tr)))

            pos_err = 0.5 * (float(np.linalg.norm(h_p - h_g)) + float(np.linalg.norm(t_p - t_g)))
            per_bone[b]["position"].append(pos_err)

    summary_per_bone = {
        b: {
            k: round(float(np.mean(vals)), 4) if vals else 0.0
            for k, vals in metrics.items()
        }
        for b, metrics in per_bone.items()
    }

    def _agg_subset(subset_bones: list[str]) -> dict[str, Any]:
        if not subset_bones:
            return {
                "count": 0,
                "direction_mean_deg": 0.0,
                "direction_max_bone_deg": 0.0,
                "full_rot_mean_deg": 0.0,
                "nonleaf_rot_mean_deg": 0.0,
                "endpoint_position_mean": 0.0,
            }
        dirs = [summary_per_bone[b]["direction"] for b in subset_bones]
        rots = [summary_per_bone[b]["rotation"] for b in subset_bones]
        nonleaf_rots = [
            summary_per_bone[b]["rotation"]
            for b in subset_bones
            if classified[b].get("children")
        ]
        pos = [summary_per_bone[b]["position"] for b in subset_bones]
        return {
            "count": len(subset_bones),
            "direction_mean_deg": round(float(np.mean(dirs)), 3),
            "direction_max_bone_deg": round(float(np.max(dirs)), 3),
            "full_rot_mean_deg": round(float(np.mean(rots)), 3),
            "nonleaf_rot_mean_deg": round(float(np.mean(nonleaf_rots)), 3) if nonleaf_rots else round(float(np.mean(rots)), 3),
            "endpoint_position_mean": round(float(np.mean(pos)), 5),
        }

    primary_bones = [b for b in bone_names if classified[b].get("is_primary", True)]
    by_kind: dict[str, Any] = {}
    for k in CANONICAL_KINDS:
        k_bones = [b for b in bone_names if classified[b].get("kind") == k]
        if k_bones:
            by_kind[k] = _agg_subset(k_bones)

    return {
        "schema": BONE_TAXONOMY_SCHEMA,
        "all_bones": _agg_subset(bone_names),
        "primary_bones": _agg_subset(primary_bones),
        "by_kind": by_kind,
        "per_bone": summary_per_bone,
    }


# ==============================================================================
# Layer 4: Stratified 4-Regime Decoupled Pose Sampler & Coverage / Rank Audit Gate
# ==============================================================================

def select_pose_sampling_regime(sample_idx: int | None, rng: Any) -> str:
    """Selects one of the 4 stratified pose sampling regimes (`40%` synergistic, `30%` decoupled,
    `15%` antiphase_extreme, `15%` workspace_cross_occlusion).
    """
    if sample_idx is not None:
        slot = int(sample_idx) % 20
    else:
        slot = int(rng.random() * 20.0)
    if slot < 8:
        return "synergistic"
    if slot < 14:
        return "decoupled"
    if slot < 17:
        return "antiphase_extreme"
    return "workspace_cross_occlusion"


def effective_sampling_limits_for_bone(
    info: dict[str, Any], enforce_anatomical_floor: bool = True
) -> list[list[float]]:
    """Returns effective `[[rx_lo, rx_hi], [ry_lo, ry_hi], [rz_lo, rz_hi]]` radians for Stage 2 pose
    synthesis. When `enforce_anatomical_floor=True` and `limits_source` is not `"agent_custom"` /
    `"probe_initial"`, expands overly narrow empirical limits (e.g., a digit or limb joint that barely
    moved in template clips) to at least 65% of the canonical anatomical envelope for its `(kind, sub_role)`.
    Explicitly configured limits (`limits_source in ("agent_custom", "probe_initial")`) are respected as-is.
    """
    kind = str(info.get("kind", "limb"))
    if kind == "rigid" or bool(info.get("fixed_local", False)):
        return [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    sub_role = str(info.get("sub_role", ""))
    is_leaf = not bool(info.get("children"))
    lim_src = str(info.get("limits_source", ""))
    explicit_custom = lim_src in ("agent_custom", "probe_initial")
    def_lims, _, _ = default_limits_and_corr_for_kind(kind, sub_role, is_leaf=is_leaf)
    raw_lims = info.get("limits_rad")
    if not isinstance(raw_lims, list) or len(raw_lims) != 3:
        return [[float(pair[0]), float(pair[1])] for pair in def_lims]

    out: list[list[float]] = []
    for ax in range(3):
        lo = float(raw_lims[ax][0])
        hi = float(raw_lims[ax][1])
        if (
            enforce_anatomical_floor
            and not explicit_custom
            and kind in ("digit", "limb", "flex", "tail", "wing", "head", "jaw")
        ):
            def_lo, def_hi = float(def_lims[ax][0]), float(def_lims[ax][1])
            def_span = def_hi - def_lo
            if (hi - lo) < 0.50 * def_span:
                lo = min(lo, 0.65 * def_lo)
                hi = max(hi, 0.65 * def_hi)
        if is_leaf and ax == 1 and not bool(info.get("allow_leaf_twist", False)):
            lo, hi = 0.0, 0.0
        out.append([lo, hi])
    return out


def resolve_joint_limits_profile_from_paths(
    *search_paths: str | Path | None,
    fallback_profile: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolves `joint_limits.json` (`monster-joint-limits-v1`) from explicit file paths or
    directories/parent directories of `meta_path`, `ckpt_path`, `synth_dir`, `exam_dir`, `blend_path`,
    falling back to `fallback_profile` when no on-disk `joint_limits.json` is found.
    """
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        key = str(p)
        if key not in seen:
            seen.add(key)
            candidates.append(p)

    for raw in search_paths:
        if not raw:
            continue
        p = Path(raw)
        if p.name == "joint_limits.json" or p.suffix.lower() == ".json" and "limit" in p.name.lower():
            _add(p)
        if p.is_dir():
            _add(p / "joint_limits.json")
            _add(p.parent / "joint_limits.json")
        else:
            _add(p.parent / "joint_limits.json")
            _add(p.parent.parent / "joint_limits.json")

    for cand in candidates:
        if cand.is_file():
            try:
                profile = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(profile, dict) and (
                    profile.get("schema") == JOINT_LIMITS_SCHEMA or isinstance(profile.get("bones"), dict)
                ):
                    return profile, str(cand)
            except Exception:
                pass

    if isinstance(fallback_profile, dict) and (
        fallback_profile.get("schema") == JOINT_LIMITS_SCHEMA or isinstance(fallback_profile.get("bones"), dict)
    ):
        return copy.deepcopy(fallback_profile), None
    return None, None


def apply_joint_limits_profile(
    bones_meta: dict[str, dict[str, Any]],
    joint_limits_profile: dict[str, Any] | None = None,
    bbox_min: list[float] | None = None,
    bbox_max: list[float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Merges a per-monster `joint_limits.json` deliverable (`monster-joint-limits-v1`) into `bones_meta`,
    updating both local per-bone `limits_rad` / `limits_deg` / `chain_corr` / `fixed_local` / `limits_source`
    and the global `ROOT` orientation & translation envelope (`global_root`).
    """
    first_info = next(iter(bones_meta.values())) if bones_meta else {}
    classified = (
        copy.deepcopy(bones_meta)
        if ("kind" in first_info and "sub_role" in first_info and "group_id" in first_info)
        else classify_rig_bones(bones_meta, bbox_min=bbox_min, bbox_max=bbox_max)
    )
    if not joint_limits_profile or not isinstance(joint_limits_profile, dict):
        return classified

    deg = math.radians
    profile_source = str(joint_limits_profile.get("authored_by") or "agent_custom")
    if profile_source not in ("agent_custom", "probe_initial", "canonical_default"):
        profile_source = "agent_custom"

    global_root = copy.deepcopy(joint_limits_profile.get("global_root") or {})
    root_motion = joint_limits_profile.get("root_motion")
    if isinstance(root_motion, dict):
        if "translation_fraction_xyz" in root_motion and "translation_fraction_xyz" not in global_root:
            global_root["translation_fraction_xyz"] = copy.deepcopy(root_motion["translation_fraction_xyz"])
        if "max_translation_world_xyz" in root_motion and "max_translation_world_xyz" not in global_root:
            global_root["max_translation_world_xyz"] = copy.deepcopy(root_motion["max_translation_world_xyz"])
        if all(k in root_motion for k in ("global_pitch_deg", "global_yaw_deg", "global_roll_deg")) and "rotation_limits_deg" not in global_root:
            global_root["rotation_limits_deg"] = [
                list(root_motion["global_pitch_deg"]),
                list(root_motion["global_yaw_deg"]),
                list(root_motion["global_roll_deg"]),
            ]
    p_bones = joint_limits_profile.get("bones") or {}

    roots = [b for b in sorted(classified) if classified[b].get("kind") == "root" or not classified[b].get("parent")]
    primary_root = roots[0] if roots else (sorted(classified)[0] if classified else None)

    for b_name, info in classified.items():
        if b_name in p_bones and isinstance(p_bones[b_name], dict):
            b_prof = p_bones[b_name]
            if "fixed_local" in b_prof:
                info["fixed_local"] = bool(b_prof["fixed_local"])
            if bool(info.get("fixed_local", False)) or info.get("kind") == "rigid":
                info["limits_deg"] = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
                info["limits_rad"] = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            elif "limits_deg" in b_prof and isinstance(b_prof["limits_deg"], dict):
                d_map = b_prof["limits_deg"]
                ldeg = [
                    [round(float(d_map.get(ax, [-15.0, 15.0])[0]), 2), round(float(d_map.get(ax, [-15.0, 15.0])[1]), 2)]
                    for ax in ("rx", "ry", "rz")
                ]
                info["limits_deg"] = ldeg
                info["limits_rad"] = [[round(deg(pair[0]), 4), round(deg(pair[1]), 4)] for pair in ldeg]
            elif "limits_deg" in b_prof and isinstance(b_prof["limits_deg"], list) and len(b_prof["limits_deg"]) == 3:
                ldeg = [[round(float(pair[0]), 2), round(float(pair[1]), 2)] for pair in b_prof["limits_deg"]]
                info["limits_deg"] = ldeg
                info["limits_rad"] = [[round(deg(pair[0]), 4), round(deg(pair[1]), 4)] for pair in ldeg]
            elif "limits_rad" in b_prof and isinstance(b_prof["limits_rad"], list) and len(b_prof["limits_rad"]) == 3:
                lrad = [[round(float(pair[0]), 4), round(float(pair[1]), 4)] for pair in b_prof["limits_rad"]]
                info["limits_rad"] = lrad
                info["limits_deg"] = [
                    [round(math.degrees(pair[0]), 2), round(math.degrees(pair[1]), 2)] for pair in lrad
                ]
            if "chain_corr" in b_prof and b_prof["chain_corr"] is not None:
                info["chain_corr"] = round(float(b_prof["chain_corr"]), 3)
            if "axis_semantics" in b_prof:
                info["axis_semantics"] = copy.deepcopy(b_prof["axis_semantics"])
            if "probe_safe_limits_deg" in b_prof:
                info["probe_safe_limits_deg"] = copy.deepcopy(b_prof["probe_safe_limits_deg"])
            if "canonical_default_limits_deg" in b_prof:
                info["canonical_default_limits_deg"] = copy.deepcopy(b_prof["canonical_default_limits_deg"])
            if "rationale" in b_prof:
                info["rationale"] = str(b_prof["rationale"])
            info["limits_source"] = str(b_prof.get("limits_source") or b_prof.get("source") or profile_source)

        if b_name == primary_root and global_root:
            info["global_root"] = copy.deepcopy(global_root)
            rot_deg = global_root.get("rotation_limits_deg")
            if isinstance(rot_deg, list) and len(rot_deg) == 3 and b_name not in p_bones:
                ldeg = [[round(float(pair[0]), 2), round(float(pair[1]), 2)] for pair in rot_deg]
                info["limits_deg"] = ldeg
                info["limits_rad"] = [[round(deg(pair[0]), 4), round(deg(pair[1]), 4)] for pair in ldeg]
                info["limits_source"] = profile_source
            elif "limits_deg" in info:
                info["global_root"]["rotation_limits_deg"] = copy.deepcopy(info["limits_deg"])

    return classified


def _describe_world_motion_vector(delta_vec: np.ndarray, side: str = "C", is_axial_twist: bool = False, sign: str = "+") -> str:
    """Translates a 3D world-space tip displacement vector `[dx, dy, dz]` (under character convention:
    `-Y` = Forward toward Cam_Front, `+Y` = Backward, `+Z` = Up, `-Z` = Down, `+X` = Left, `-X` = Right)
    into a deterministic anatomical motion descriptor for Agent & human review.
    """
    if is_axial_twist or float(np.linalg.norm(delta_vec)) < 1e-5:
        return f"axial_twist ({sign}shaft_roll)"

    u = delta_vec / float(np.linalg.norm(delta_vec))
    dx, dy, dz = float(u[0]), float(u[1]), float(u[2])

    def _axis_label(axis_idx: int, val: float) -> tuple[str, str]:
        if axis_idx == 0:
            coord = "+X" if val >= 0.0 else "-X"
            if side == "L":
                sem = "lateral_outward" if val >= 0.0 else "medial_inward"
            elif side == "R":
                sem = "lateral_outward" if val < 0.0 else "medial_inward"
            else:
                sem = "lean_left" if val >= 0.0 else "lean_right"
            return sem, coord
        if axis_idx == 1:
            return ("backward" if val >= 0.0 else "forward"), ("+Y" if val >= 0.0 else "-Y")
        return ("up" if val >= 0.0 else "down"), ("+Z" if val >= 0.0 else "-Z")

    comps = sorted([(0, abs(dx), dx), (1, abs(dy), dy), (2, abs(dz), dz)], key=lambda item: item[1], reverse=True)
    prim_sem, prim_coord = _axis_label(comps[0][0], comps[0][2])
    if comps[1][1] >= 0.38 * comps[0][1]:
        sec_sem, sec_coord = _axis_label(comps[1][0], comps[1][2])
        return f"{prim_sem}_{sec_sem} ({prim_coord},{sec_coord})"
    return f"{prim_sem} ({prim_coord})"


def _probe_safe_ceiling_deg_for_kind(kind: str, sub_role: str = "", is_leaf: bool = False) -> list[list[float]]:
    """Returns the upper physical probe ceiling `[[rx_lo, rx_hi], [ry_lo, ry_hi], [rz_lo, rz_hi]]` in degrees
    before mesh-stretch clamping, representing the generous action-general envelope for a given bone role.
    """
    if kind == "rigid":
        return [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    if kind == "root":
        return [[-28.0, 32.0], [-22.0, 22.0], [-25.0, 25.0]]
    if kind == "spine":
        return [[-22.0, 24.0], [-10.0, 10.0], [-20.0, 20.0]]
    if kind == "head":
        return [[-30.0, 30.0], [-28.0, 28.0], [-26.0, 26.0]]
    if kind == "jaw":
        return [[-4.0, 42.0], [-4.0, 4.0], [-5.0, 5.0]]
    if kind == "tail":
        return [[-26.0, 26.0], [-8.0, 8.0], [-24.0, 24.0]]
    if kind == "flex":
        return [[-32.0, 32.0], [-14.0, 14.0], [-26.0, 26.0]]
    if kind == "wing":
        return [[-52.0, 58.0], [-16.0, 16.0], [-45.0, 45.0]]
    if kind == "digit":
        ry = 0.0 if is_leaf else 5.0
        return [[-22.0, 48.0], [-ry, ry], [-16.0, 16.0]]
    if sub_role == "clavicle_socket":
        return [[-26.0, 26.0], [-12.0, 12.0], [-26.0, 26.0]]
    if sub_role in ("upper_limb", "ball_socket"):
        return [[-70.0, 80.0], [-35.0, 35.0], [-52.0, 52.0]]
    if sub_role == "forearm_hinge":
        return [[-55.0, 108.0], [-42.0, 42.0], [-52.0, 52.0]]
    if sub_role == "hand_hub":
        return [[-42.0, 42.0], [-12.0, 12.0], [-32.0, 32.0]]
    return [[-35.0, 40.0], [-6.0, 6.0], [-18.0, 18.0]]


def probe_rig_joint_limits(
    bones_meta: dict[str, dict[str, Any]],
    bbox_min: list[float] | None = None,
    bbox_max: list[float] | None = None,
    rig_skinning: dict[str, Any] | None = None,
    max_bad_edge_ratio: float = 0.02,
    max_stretch_ratio: float = 1.80,
    monster_id: str = "monster",
    archetype: str = "humanoid",
    selected_candidate_id: str = "0",
    rig_meta_sha256: str | None = None,
    restored_decorations: list[dict[str, Any]] | None = None,
    lbs_evaluator_fn: Any = None,
    stretch_metric_fn: Any = None,
) -> dict[str, Any]:
    """Deterministic Geometric & LBS Mesh-Stretch Joint Limits Probe.

    1. Evaluates the exact world-space motion direction of `+rx / -rx`, `+ry / -ry`, `+rz / -rz`
       for every canonical bone (`axis_semantics`) so the Agent never guesses local bone roll axes.
    2. Sweeps each bone axis against LBS mesh deformation (when `rig_skinning` is supplied) and
       decoration attachment constraints to determine `probe_safe_limits_deg`.
    3. Computes the action-general `global_root` orientation (`pitch`, `roll`, `yaw`) and translation
       envelope (`translation_fraction_xyz`) within the studio orthographic camera frustum.
    4. Produces a complete `monster-joint-limits-v1` dictionary ready to be saved as `joint_limits.json`
       and refined by the Agent prior to G2 50-pose boundary exploration.
    """
    classified = classify_rig_bones(bones_meta, bbox_min=bbox_min, bbox_max=bbox_max)
    bone_names = sorted(classified.keys())
    order = sorted(bone_names, key=lambda b: (int(classified[b].get("depth", 0)), b))

    all_pts = []
    for b in bone_names:
        if "rest_head_world" in classified[b]:
            all_pts.append(classified[b]["rest_head_world"])
        if "rest_tail_world" in classified[b]:
            all_pts.append(classified[b]["rest_tail_world"])
    if bbox_min is not None and bbox_max is not None:
        bmin = np.asarray(bbox_min, dtype=np.float64)
        bmax = np.asarray(bbox_max, dtype=np.float64)
    elif all_pts:
        arr_pts = np.asarray(all_pts, dtype=np.float64)
        bmin = np.min(arr_pts, axis=0)
        bmax = np.max(arr_pts, axis=0)
    else:
        bmin = np.array([-0.45, -0.35, 0.0], dtype=np.float64)
        bmax = np.array([0.45, 0.35, 1.15], dtype=np.float64)

    span = np.maximum(1e-3, bmax - bmin)
    rest_H: dict[str, np.ndarray] = {}
    rest_T: dict[str, np.ndarray] = {}
    rest_R: dict[str, np.ndarray] = {}
    for b in bone_names:
        info = classified[b]
        h = np.asarray(info.get("rest_head_world", [0.0, 0.0, 0.5]), dtype=np.float64)
        t = np.asarray(info.get("rest_tail_world", h + np.array([0.0, 0.0, 0.1])), dtype=np.float64)
        rest_H[b] = h
        rest_T[b] = t
        rmat = info.get("rest_rot_mat_world")
        if rmat is not None:
            rest_R[b] = np.asarray(rmat, dtype=np.float64)
        else:
            v0 = t - h
            d0 = v0 / max(1e-8, float(np.linalg.norm(v0)))
            rest_R[b] = _build_default_rest_rot_matrix(d0)

    # Map attached rigid/flex decorations per bone
    decor_by_bone: dict[str, list[str]] = {}
    for dec in restored_decorations or []:
        if isinstance(dec, dict) and dec.get("target_bone"):
            tb = str(dec["target_bone"])
            decor_by_bone.setdefault(tb, []).append(f"{dec.get('name', 'decor')}({dec.get('mode', 'flex')})")

    # Optional LBS mesh stretch probe setup
    has_lbs = (
        isinstance(rig_skinning, dict)
        and "rest_verts" in rig_skinning
        and "triangles" in rig_skinning
        and "skin_indices" in rig_skinning
        and "skin_weights" in rig_skinning
        and len(rig_skinning["rest_verts"]) > 0
    )
    lbs_bone_names: list[str] = []
    if has_lbs and rig_skinning is not None:
        from pose_features import compute_mesh_edge_stretch_metric, lbs_deform_vertices_numpy

        lbs_bone_names = [str(x) for x in rig_skinning.get("bone_names", bone_names)]
        lbs_rest_verts = np.asarray(rig_skinning["rest_verts"], dtype=np.float64)
        lbs_skin_indices = np.asarray(rig_skinning["skin_indices"], dtype=np.int32)
        lbs_skin_weights = np.asarray(rig_skinning["skin_weights"], dtype=np.float64)
        lbs_triangles = np.asarray(rig_skinning["triangles"], dtype=np.int64)
        lbs_rest_heads = np.asarray(rig_skinning["rest_heads"], dtype=np.float64)
        lbs_rest_R = np.asarray(rig_skinning["rest_R"], dtype=np.float64)
        lbs_b_to_idx = {bn: i for i, bn in enumerate(lbs_bone_names)}

    def _measure_max_safe_angle_deg(target_bone: str, axis_idx: int, sign_dir: float, ceiling_abs_deg: float) -> float:
        if ceiling_abs_deg <= 1e-4 or not has_lbs or target_bone not in lbs_b_to_idx:
            return round(float(ceiling_abs_deg), 1)
        descendants = set()
        stack = [target_bone]
        while stack:
            cur = stack.pop()
            descendants.add(cur)
            stack.extend(classified.get(cur, {}).get("children", []))

        test_angles = [a for a in (15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0) if a <= ceiling_abs_deg + 1e-3]
        if not test_angles or test_angles[-1] < ceiling_abs_deg - 1e-3:
            test_angles.append(float(ceiling_abs_deg))

        last_safe = min(15.0, float(ceiling_abs_deg))
        for ang_deg in test_angles:
            ang_rad = math.radians(sign_dir * ang_deg)
            eul = [0.0, 0.0, 0.0]
            eul[axis_idx] = ang_rad
            R_delta = euler_xyz_to_matrix_numpy(eul[0], eul[1], eul[2])

            posed_heads = lbs_rest_heads.copy()
            posed_R = lbs_rest_R.copy()
            h_rot = rest_H[target_bone]
            R_world_rot = rest_R[target_bone] @ R_delta @ rest_R[target_bone].T
            for b_sub in descendants:
                if b_sub in lbs_b_to_idx:
                    bi = lbs_b_to_idx[b_sub]
                    posed_heads[bi] = h_rot + R_world_rot @ (lbs_rest_heads[bi] - h_rot)
                    posed_R[bi] = R_world_rot @ lbs_rest_R[bi]

            posed_v = lbs_deform_vertices_numpy(
                lbs_rest_verts,
                lbs_skin_indices,
                lbs_skin_weights,
                lbs_rest_heads,
                lbs_rest_R,
                posed_heads,
                posed_R,
            )
            stretch = compute_mesh_edge_stretch_metric(
                lbs_rest_verts,
                posed_v,
                lbs_triangles,
                max_stretch=max_stretch_ratio,
                max_bad_ratio=max_bad_edge_ratio,
            )
            if stretch["passed"]:
                last_safe = float(ang_deg)
            else:
                break
        return round(float(last_safe), 1)

    global_root = {
        "rotation_limits_deg": [[-28.0, 32.0], [-22.0, 22.0], [-25.0, 25.0]],
        "axis_labels": ["pitch_rx", "yaw_twist_ry", "roll_rz"],
        "translation_fraction_xyz": [0.08, 0.08, 0.10],
        "max_translation_world_xyz": [
            round(float(0.08 * span[0]), 4),
            round(float(0.08 * span[1]), 4),
            round(float(0.10 * span[2]), 4),
        ],
        "frustum_margin_min": 0.04,
        "rationale": (
            "Action-general global root envelope: covers forward/backward torso pitch [-28°, +32°], "
            "axial yaw [-22°, +22°], lateral banking roll [-25°, +25°], and 3D root shifts "
            "[8% X, 8% Y, 10% Z] for locomotion, strikes, leaps, and hit/collapse postures without "
            "requiring retraining when new action clips are introduced."
        ),
    }

    bones_profile: dict[str, dict[str, Any]] = {}
    delta_probe_rad = math.radians(15.0)
    axis_names = ("rx", "ry", "rz")

    for b in order:
        info = classified[b]
        kind = str(info.get("kind", "limb"))
        sub_role = str(info.get("sub_role", ""))
        side = str(info.get("side", "C"))
        grp = str(info.get("group_id", "00"))
        c_idx = int(info.get("chain_index", 1))
        is_leaf = not bool(info.get("children"))
        fixed_local = bool(info.get("fixed_local", kind == "rigid"))

        # 1. Measure world-space semantic motion direction for +rx/-rx, +ry/-ry, +rz/-rz
        R_b = rest_R[b]
        v_local = R_b.T @ (rest_T[b] - rest_H[b])
        if float(np.linalg.norm(v_local)) < 1e-5:
            v_local = np.array([0.0, max(0.05, float(info.get("rest_length", 0.08))), 0.0], dtype=np.float64)

        axis_semantics: dict[str, dict[str, str]] = {}
        for ax_i, ax_key in enumerate(axis_names):
            e_pos = [0.0, 0.0, 0.0]
            e_neg = [0.0, 0.0, 0.0]
            e_pos[ax_i] = delta_probe_rad
            e_neg[ax_i] = -delta_probe_rad
            d_pos = (R_b @ euler_xyz_to_matrix_numpy(*e_pos) @ v_local) - (R_b @ v_local)
            d_neg = (R_b @ euler_xyz_to_matrix_numpy(*e_neg) @ v_local) - (R_b @ v_local)
            is_twist = ax_i == 1 and float(np.linalg.norm(d_pos)) < 0.02 * float(np.linalg.norm(v_local))
            axis_semantics[ax_key] = {
                "pos": _describe_world_motion_vector(d_pos, side=side, is_axial_twist=is_twist, sign="+"),
                "neg": _describe_world_motion_vector(d_neg, side=side, is_axial_twist=is_twist, sign="-"),
            }

        # 2. Canonical defaults vs Probe safe envelope
        def_lims_rad, def_corr, _ = default_limits_and_corr_for_kind(kind, sub_role, is_leaf=is_leaf)
        def_lims_deg = [[round(math.degrees(lo), 1), round(math.degrees(hi), 1)] for (lo, hi) in def_lims_rad]
        ceil_deg = _probe_safe_ceiling_deg_for_kind(kind, sub_role, is_leaf=is_leaf)

        probe_safe_deg: list[list[float]] = []
        recommended_deg: list[list[float]] = []
        for ax_i in range(3):
            if fixed_local or (is_leaf and ax_i == 1):
                probe_safe_deg.append([0.0, 0.0])
                recommended_deg.append([0.0, 0.0])
                continue
            c_lo, c_hi = float(ceil_deg[ax_i][0]), float(ceil_deg[ax_i][1])
            safe_neg = -_measure_max_safe_angle_deg(b, ax_i, -1.0, abs(c_lo)) if c_lo < 0.0 else 0.0
            safe_pos = _measure_max_safe_angle_deg(b, ax_i, 1.0, abs(c_hi)) if c_hi > 0.0 else 0.0
            probe_safe_deg.append([round(safe_neg, 1), round(safe_pos, 1)])

            # Recommended initial envelope blends canonical baseline toward probed safe envelope (85% of safe ceiling)
            d_lo, d_hi = float(def_lims_deg[ax_i][0]), float(def_lims_deg[ax_i][1])
            if kind == "root":
                rec_lo, rec_hi = float(global_root["rotation_limits_deg"][ax_i][0]), float(global_root["rotation_limits_deg"][ax_i][1])
            else:
                rec_lo = max(safe_neg, min(d_lo, round(0.85 * safe_neg, 1)))
                rec_hi = min(safe_pos, max(d_hi, round(0.85 * safe_pos, 1)))
                if b in decor_by_bone:
                    # Slight conservative damping when rigid/flex decorations attach directly to this bone
                    rec_lo = round(0.88 * rec_lo, 1)
                    rec_hi = round(0.88 * rec_hi, 1)
            recommended_deg.append([round(rec_lo, 1), round(rec_hi, 1)])

        limits_rad = [[round(math.radians(pair[0]), 4), round(math.radians(pair[1]), 4)] for pair in recommended_deg]
        decor_note = f"; attached decorations={decor_by_bone[b]}" if b in decor_by_bone else ""
        bones_profile[b] = {
            "kind": kind,
            "sub_role": sub_role,
            "side": side,
            "group_id": grp,
            "chain_index": c_idx,
            "parent": info.get("parent"),
            "is_leaf": is_leaf,
            "fixed_local": fixed_local,
            "axis_semantics": axis_semantics,
            "axis_probe": axis_semantics,
            "canonical_default_limits_deg": def_lims_deg,
            "probe_safe_limits_deg": probe_safe_deg,
            "safe_lbs_ceiling_deg": {
                "rx": list(probe_safe_deg[0]),
                "ry": list(probe_safe_deg[1]),
                "rz": list(probe_safe_deg[2]),
            },
            "limits_deg": {
                "rx": list(recommended_deg[0]),
                "ry": list(recommended_deg[1]),
                "rz": list(recommended_deg[2]),
            },
            "limits_deg_list": recommended_deg,
            "limits_rad": limits_rad,
            "chain_corr": round(float(def_corr), 3),
            "limits_source": "probe_initial",
            "source": "probe_initial",
            "attached_decorations": decor_by_bone.get(b, []),
            "rationale": (
                f"Probed {kind}/{sub_role} ({side}, idx={c_idx}, LBS_measured={has_lbs})"
                f"{decor_note}: rx={axis_semantics['rx']['pos']}, rz={axis_semantics['rz']['pos']}."
            ),
        }

    root_motion = {
        "global_pitch_deg": list(global_root["rotation_limits_deg"][0]),
        "global_yaw_deg": list(global_root["rotation_limits_deg"][1]),
        "global_roll_deg": list(global_root["rotation_limits_deg"][2]),
        "translation_fraction_xyz": list(global_root["translation_fraction_xyz"]),
        "max_translation_world_xyz": list(global_root["max_translation_world_xyz"]),
        "rationale": global_root["rationale"],
    }

    return {
        "schema": JOINT_LIMITS_SCHEMA,
        "monster_id": monster_id,
        "archetype": archetype,
        "selected_candidate_id": str(selected_candidate_id),
        "rig_meta_sha256": rig_meta_sha256,
        "authored_by": "probe_initial",
        "lbs_stretch_measured": has_lbs,
        "max_bad_edge_ratio_threshold": max_bad_edge_ratio,
        "max_stretch_ratio_threshold": max_stretch_ratio,
        "bbox_min": [round(float(x), 5) for x in bmin],
        "bbox_max": [round(float(x), 5) for x in bmax],
        "global_root": global_root,
        "root_motion": root_motion,
        "bones": bones_profile,
    }


def _sample_bimodal_boundary(rng: Any, lo: float, hi: float) -> float:
    """Samples with 70% weight in the outer 28% boundary quartiles `[lo, lo+0.28*span]` and
    `[hi-0.28*span, hi]` and 30% uniform across `[lo, hi]` to cover extreme joint angles.
    """
    span = hi - lo
    if span <= 1e-8:
        return lo
    r = rng.random()
    q = 0.28 * span
    if r < 0.35:
        return float(rng.uniform(lo, lo + q))
    if r < 0.70:
        return float(rng.uniform(hi - q, hi))
    return float(rng.uniform(lo, hi))


def sample_stratified_pose_euler_dict(
    bones_meta: dict[str, dict[str, Any]],
    rng: Any,
    sample_idx: int | None = None,
    enforce_anatomical_floor: bool = True,
    joint_limits_profile: dict[str, Any] | None = None,
    boundary_exploration: bool = False,
) -> tuple[dict[str, list[float]], str]:
    """Samples a full-skeleton pose Euler dictionary `{bone_name: [rx, ry, rz]}` using the 4-Regime
    Stratified Decoupled Sampler (or the 5-Regime Boundary-Exploration Sampler when `boundary_exploration=True`):
      1. `synergistic` (40%): Per-finger independent curl factors (never a single hand-wide fixed
         scalar across all 5 fingers!) + natural intra-chain synergy.
      2. `decoupled` (30%): Zero parent-child correlation (`corr = 0.0`); every phalange and chain
         segment is sampled independently across its full anatomical range to eliminate shortcut learning.
      3. `antiphase_extreme` (15%): Negative parent-child correlation (`corr < 0`) + bimodal boundary
         quartiles, covering S-curve waves, claw/hook fingers, and near-limit joint angles.
      4. `workspace_cross_occlusion` (15%): Biases upper limbs toward cross-midline / chest-gathering
         self-occlusion postures (`gather` / `prayer` / `offer`) and full forearm twist while mixing
         open, splayed, and curled individual fingers.
    """
    if boundary_exploration:
        eulers, reg, _ = sample_boundary_exploration_pose_euler_dict(
            bones_meta=bones_meta,
            rng=rng,
            sample_idx=sample_idx,
            joint_limits_profile=joint_limits_profile,
            enforce_anatomical_floor=False,
        )
        return eulers, reg
    if not bones_meta:
        return {}, "synergistic"

    first_info = next(iter(bones_meta.values()))
    classified = (
        bones_meta
        if ("kind" in first_info and "sub_role" in first_info and "group_id" in first_info)
        else classify_rig_bones(bones_meta)
    )
    if joint_limits_profile is not None:
        classified = apply_joint_limits_profile(classified, joint_limits_profile)
    regime = select_pose_sampling_regime(sample_idx, rng)
    topo_names = sorted(classified.keys(), key=lambda b: (int(classified[b].get("depth", 0)), b))

    # Per-finger (`(side, group_id)`) independent state so fingers on the same hand never lock to 1 scalar
    finger_group_targets: dict[tuple[str, str], tuple[float, float]] = {}
    # Per-chain (`(kind, side, group_id)`) cosine-wave state for multi-bone tail/flex/wing chains
    chain_wave_targets: dict[tuple[str, str, str], tuple[float, float, float, float]] = {}
    use_priors = mocap_priors_enabled()
    for b_name in topo_names:
        info = classified[b_name]
        b_kind = str(info.get("kind", "limb"))
        if b_kind == "digit":
            key = (str(info.get("side", "C")), str(info.get("group_id", "00")))
            if key not in finger_group_targets:
                if regime == "workspace_cross_occlusion":
                    mode_r = rng.random()
                    curl_u = rng.uniform(0.0, 0.25) if mode_r < 0.35 else (rng.uniform(0.75, 1.0) if mode_r < 0.70 else rng.uniform(0.25, 0.75))
                else:
                    curl_u = rng.uniform(0.0, 1.0)
                splay_u = rng.uniform(0.0, 1.0)
                finger_group_targets[key] = (float(curl_u), float(splay_u))
        elif use_priors and b_kind in ("tail", "flex", "wing"):
            ckey = (b_kind, str(info.get("side", "C")), str(info.get("group_id", "00")))
            if ckey not in chain_wave_targets:
                chain_wave_targets[ckey] = (
                    float(rng.uniform(0.0, 2.0 * math.pi)),
                    float(rng.uniform(0.35, 0.92)),
                    float(rng.uniform(0.0, 2.0 * math.pi)),
                    float(rng.uniform(0.35, 0.92)),
                )

    sampled_eulers: dict[str, list[float]] = {}
    for b_name in topo_names:
        info = classified[b_name]
        kind = str(info.get("kind", "limb"))
        sub_role = str(info.get("sub_role", ""))
        side = str(info.get("side", "C"))
        grp = str(info.get("group_id", "00"))
        is_leaf = not bool(info.get("children"))

        if kind == "rigid" or bool(info.get("fixed_local", False)):
            sampled_eulers[b_name] = [0.0, 0.0, 0.0]
            continue

        lims = effective_sampling_limits_for_bone(info, enforce_anatomical_floor=enforce_anatomical_floor)
        base_corr = float(info.get("chain_corr", 0.0))
        if base_corr <= 0.0 and kind in ("digit", "flex", "tail", "spine", "wing", "head"):
            _, base_corr, _ = default_limits_and_corr_for_kind(kind, sub_role, is_leaf=is_leaf)
        parent_name = info.get("parent")
        has_parent_euler = bool(parent_name and parent_name in sampled_eulers)

        if regime == "decoupled":
            raw = [float(rng.uniform(lo, hi)) for (lo, hi) in lims]

        elif regime == "antiphase_extreme":
            raw = [_sample_bimodal_boundary(rng, lo, hi) for (lo, hi) in lims]
            anti_corr = -0.58 * max(0.45, base_corr)
            if has_parent_euler and kind in ("digit", "flex", "tail", "spine", "wing", "head"):
                p_rot = sampled_eulers[parent_name]  # type: ignore[index]
                for ax in (0, 2):
                    lo, hi = lims[ax]
                    mid = 0.5 * (lo + hi)
                    p_centered = p_rot[ax] - mid
                    cand = mid + anti_corr * p_centered + (1.0 - abs(anti_corr)) * (raw[ax] - mid)
                    raw[ax] = float(np.clip(cand, lo, hi))

        elif regime == "workspace_cross_occlusion":
            if kind == "limb":
                raw = [_sample_bimodal_boundary(rng, lo, hi) for (lo, hi) in lims]
                # Explicitly sample full forearm / upper-limb axial roll `ry` for palm flip coverage
                raw[1] = _sample_bimodal_boundary(rng, lims[1][0], lims[1][1])
            elif kind == "digit":
                curl_u, splay_u = finger_group_targets.get((side, grp), (0.5, 0.5))
                rx_base = lims[0][0] + curl_u * (lims[0][1] - lims[0][0])
                rz_base = lims[2][0] + splay_u * (lims[2][1] - lims[2][0])
                rx_jit = rng.uniform(-0.22, 0.22) * (lims[0][1] - lims[0][0])
                rz_jit = rng.uniform(-0.25, 0.25) * (lims[2][1] - lims[2][0])
                raw = [
                    float(np.clip(rx_base + rx_jit, lims[0][0], lims[0][1])),
                    float(rng.uniform(lims[1][0], lims[1][1])),
                    float(np.clip(rz_base + rz_jit, lims[2][0], lims[2][1])),
                ]
            else:
                raw = [float(rng.uniform(lo, hi)) for (lo, hi) in lims]
                if base_corr > 0.0 and has_parent_euler:
                    p_rot = sampled_eulers[parent_name]  # type: ignore[index]
                    c_eff = 0.45 * base_corr
                    raw[0] = float(np.clip(c_eff * p_rot[0] + (1.0 - c_eff) * raw[0], lims[0][0], lims[0][1]))
                    raw[2] = float(np.clip(c_eff * p_rot[2] + (1.0 - c_eff) * raw[2], lims[2][0], lims[2][1]))

        else:
            # "synergistic" regime: per-finger independent target + intra-chain synergy
            raw = [float(rng.uniform(lo, hi)) for (lo, hi) in lims]
            if kind == "digit":
                curl_u, splay_u = finger_group_targets.get((side, grp), (0.5, 0.5))
                rx_finger = lims[0][0] + curl_u * (lims[0][1] - lims[0][0])
                rz_finger = lims[2][0] + splay_u * (lims[2][1] - lims[2][0])
                raw[0] = float(np.clip(0.55 * rx_finger + 0.45 * raw[0], lims[0][0], lims[0][1]))
                raw[2] = float(np.clip(0.45 * rz_finger + 0.55 * raw[2], lims[2][0], lims[2][1]))
                if has_parent_euler and classified.get(parent_name, {}).get("kind") == "digit":  # type: ignore[arg-type]
                    p_rot = sampled_eulers[parent_name]  # type: ignore[index]
                    raw[0] = float(np.clip(0.45 * p_rot[0] + 0.55 * raw[0], lims[0][0], lims[0][1]))
                    raw[2] = float(np.clip(0.35 * p_rot[2] + 0.65 * raw[2], lims[2][0], lims[2][1]))
            elif base_corr > 0.0 and has_parent_euler:
                p_rot = sampled_eulers[parent_name]  # type: ignore[index]
                c_eff = min(0.60, base_corr)
                raw[0] = float(np.clip(c_eff * p_rot[0] + (1.0 - c_eff) * raw[0], lims[0][0], lims[0][1]))
                raw[1] = float(np.clip((1.0 - 0.5 * c_eff) * raw[1], lims[1][0], lims[1][1]))
                raw[2] = float(np.clip(c_eff * p_rot[2] + (1.0 - c_eff) * raw[2], lims[2][0], lims[2][1]))
                if use_priors and kind in ("tail", "flex", "wing"):
                    w_state = chain_wave_targets.get((kind, side, grp))
                    if w_state is not None:
                        wave_eul = compute_chain_wave_target_euler(
                            chain_index=int(info.get("chain_index", 0)),
                            lims=lims,
                            wave_state=w_state,
                        )
                        raw[0] = float(np.clip(0.32 * wave_eul[0] + 0.68 * raw[0], lims[0][0], lims[0][1]))
                        raw[2] = float(np.clip(0.32 * wave_eul[2] + 0.68 * raw[2], lims[2][0], lims[2][1]))

        if is_leaf:
            raw[1] = 0.0

        sampled_eulers[b_name] = raw

    return sampled_eulers, regime


def sample_root_translation_offset(
    bones_meta: dict[str, dict[str, Any]],
    rng: Any,
    sample_idx: int | None = None,
    max_xy_fraction: float = 0.08,
    max_z_fraction: float = 0.10,
    joint_limits_profile: dict[str, Any] | None = None,
    boundary_exploration: bool = False,
) -> list[float]:
    """Samples a bounded 3D root translation offset `[dx, dy, dz]` in world/rig units so synthetic
    training poses cover leaping, crouching, lunging, and lateral root shifts without clipping
    outside the orthographic camera frustum.
    """
    pts: list[list[float]] = []
    global_root_cfg: dict[str, Any] | None = None
    if isinstance(joint_limits_profile, dict) and isinstance(joint_limits_profile.get("global_root"), dict):
        global_root_cfg = joint_limits_profile["global_root"]
    for info in bones_meta.values():
        if "rest_head_world" in info:
            pts.append(info["rest_head_world"])
        if "rest_tail_world" in info:
            pts.append(info["rest_tail_world"])
        if global_root_cfg is None and isinstance(info.get("global_root"), dict):
            global_root_cfg = info["global_root"]
    if pts:
        arr = np.asarray(pts, dtype=np.float64)
        span = np.maximum(1e-3, np.ptp(arr, axis=0))
    else:
        span = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    fx, fy, fz = float(max_xy_fraction), float(max_xy_fraction), float(max_z_fraction)
    if global_root_cfg is not None:
        t_frac = global_root_cfg.get("translation_fraction_xyz")
        if isinstance(t_frac, list) and len(t_frac) == 3:
            fx, fy, fz = float(t_frac[0]), float(t_frac[1]), float(t_frac[2])

    max_dx = float(fx * span[0])
    max_dy = float(fy * span[1])
    max_dz = float(fz * span[2])

    if boundary_exploration:
        g2_reg = select_g2_boundary_regime(sample_idx, rng)
        if g2_reg == "full_body_dynamic_root_boundary":
            return [
                _sample_bimodal_boundary(rng, -max_dx, max_dx),
                _sample_bimodal_boundary(rng, -max_dy, max_dy),
                _sample_bimodal_boundary(rng, -max_dz, max_dz),
            ]
        return [
            float(rng.uniform(-0.75 * max_dx, 0.75 * max_dx)),
            float(rng.uniform(-0.75 * max_dy, 0.75 * max_dy)),
            float(rng.uniform(-0.75 * max_dz, 0.75 * max_dz)),
        ]

    regime = select_pose_sampling_regime(sample_idx, rng)
    if regime == "antiphase_extreme":
        return [
            _sample_bimodal_boundary(rng, -max_dx, max_dx),
            _sample_bimodal_boundary(rng, -max_dy, max_dy),
            _sample_bimodal_boundary(rng, -max_dz, max_dz),
        ]
    return [
        float(rng.uniform(-max_dx, max_dx)),
        float(rng.uniform(-max_dy, max_dy)),
        float(rng.uniform(-max_dz, max_dz)),
    ]


def select_g2_boundary_regime(sample_idx: int | None, rng: Any, total_samples: int = 50) -> str:
    """Selects one of the 5 G2 boundary-exploration sampling regimes:
      - `in_phase_boundary_extension` (25%, slots 0..4): cumulative in-phase chain extension to limits
      - `antiphase_fold_boundary`     (20%, slots 5..8): opposite parent/child limit folding & S-curves
      - `cross_occlusion_twist_boundary` (20%, slots 9..12): cross-midline limb adduction & shaft twist limits
      - `full_body_dynamic_root_boundary` (20%, slots 13..16): global root pitch/roll/yaw + translation boundaries
      - `quasi_uniform_interior_coverage` (15%, slots 17..19): decoupled interior/transition coverage
    """
    if sample_idx is not None:
        slot = int(sample_idx) % 20
    else:
        slot = int(rng.random() * 20.0)
    if slot < 5:
        return "in_phase_boundary_extension"
    if slot < 9:
        return "antiphase_fold_boundary"
    if slot < 13:
        return "cross_occlusion_twist_boundary"
    if slot < 17:
        return "full_body_dynamic_root_boundary"
    return "quasi_uniform_interior_coverage"


def _sample_near_bound(rng: Any, lo: float, hi: float, bound_side: str, band_ratio: float = 0.14) -> float:
    """Samples stochastically inside the outer `band_ratio` (default 14%) near `lo` (`bound_side=='lo'`)
    or `hi` (`bound_side=='hi'`).
    """
    span = hi - lo
    if span <= 1e-8:
        return lo
    band = max(1e-8, float(band_ratio) * span)
    if bound_side == "lo":
        return float(rng.uniform(lo, min(hi, lo + band)))
    return float(rng.uniform(max(lo, hi - band), hi))


def inspect_pose_boundary_utilization(
    eulers_rad: dict[str, list[float]],
    bones_meta: dict[str, dict[str, Any]],
    boundary_band_ratio: float = 0.15,
    enforce_anatomical_floor: bool = False,
) -> dict[str, Any]:
    """Identifies which bone axes in `eulers_rad` fall within the outer `boundary_band_ratio` (default 15%)
    of their active `[lo, hi]` limits, returning structured badges for G2 boundary inspection.
    """
    axis_names = ("rx", "ry", "rz")
    boundary_joints: list[dict[str, Any]] = []
    active_axes_count = 0

    for b_name in sorted(bones_meta.keys()):
        info = bones_meta[b_name]
        kind = str(info.get("kind", "limb"))
        if kind == "rigid" or bool(info.get("fixed_local", False)):
            continue
        is_leaf = not bool(info.get("children"))
        lims = effective_sampling_limits_for_bone(info, enforce_anatomical_floor=enforce_anatomical_floor)
        vals = eulers_rad.get(b_name, [0.0, 0.0, 0.0])
        for ax_i, ax_name in enumerate(axis_names):
            if is_leaf and ax_i == 1 and not bool(info.get("allow_leaf_twist", False)):
                continue
            lo, hi = float(lims[ax_i][0]), float(lims[ax_i][1])
            span = hi - lo
            if span <= math.radians(1.0):
                continue
            active_axes_count += 1
            v = float(vals[ax_i])
            q = float(np.clip((v - lo) / span, 0.0, 1.0))
            if q <= boundary_band_ratio or q >= (1.0 - boundary_band_ratio):
                bound_type = "lo" if q <= boundary_band_ratio else "hi"
                lim_ref = lo if bound_type == "lo" else hi
                boundary_joints.append(
                    {
                        "bone": b_name,
                        "kind": kind,
                        "axis": ax_name,
                        "bound": bound_type,
                        "deg": round(math.degrees(v), 1),
                        "limit_deg": round(math.degrees(lim_ref), 1),
                        "quantile": round(q, 3),
                        "label": f"{b_name}.{ax_name}={round(math.degrees(v), 1)}° ({bound_type}:{round(math.degrees(lim_ref), 1)}°)",
                    }
                )

    boundary_joints.sort(key=lambda item: min(item["quantile"], 1.0 - item["quantile"]))
    max_util = (
        round(max(2.0 * abs(float(item["quantile"]) - 0.5) for item in boundary_joints), 3)
        if boundary_joints
        else 0.5
    )
    return {
        "boundary_joint_count": len(boundary_joints),
        "active_axes_count": active_axes_count,
        "boundary_ratio": round(len(boundary_joints) / float(max(1, active_axes_count)), 4),
        "max_utilization": max_util,
        "boundary_joints": boundary_joints,
    }


def sample_boundary_exploration_pose_euler_dict(
    bones_meta: dict[str, dict[str, Any]],
    rng: Any,
    sample_idx: int | None = None,
    num_total_samples: int = 50,
    total_samples: int | None = None,
    joint_limits_profile: dict[str, Any] | None = None,
    enforce_anatomical_floor: bool = False,
) -> tuple[dict[str, list[float]], str, dict[str, Any]]:
    """Samples a full-skeleton pose Euler dictionary `{bone_name: [rx, ry, rz]}` specifically designed
    for G2 50-pose boundary exploration:
      - All 50 samples are joint full-body stochastic poses strictly bounded by `joint_limits.json`.
      - Heavily weights the outer boundary zones (`<= lo + 0.14*span` and `>= hi - 0.14*span`) across
        5 complementary boundary regimes (`in_phase_boundary_extension`, `antiphase_fold_boundary`,
        `cross_occlusion_twist_boundary`, `full_body_dynamic_root_boundary`, `quasi_uniform_interior_coverage`).
      - Deterministic rotating boundary-slot schedule guarantees that across `num_total_samples` (50),
        every active bone axis touches both its lower 15% boundary and upper 15% boundary multiple times.
    Returns `(sampled_eulers, regime, boundary_meta)`.
    """
    if not bones_meta:
        return {}, "in_phase_boundary_extension", {"boundary_joint_count": 0, "active_axes_count": 0, "boundary_joints": []}

    first_info = next(iter(bones_meta.values()))
    classified = (
        copy.deepcopy(bones_meta)
        if ("kind" in first_info and "sub_role" in first_info and "group_id" in first_info)
        else classify_rig_bones(bones_meta)
    )
    if joint_limits_profile is not None:
        classified = apply_joint_limits_profile(classified, joint_limits_profile)

    idx = int(sample_idx) if sample_idx is not None else int(rng.random() * max(1, num_total_samples))
    regime = select_g2_boundary_regime(idx, rng)
    topo_names = sorted(classified.keys(), key=lambda b: (int(classified[b].get("depth", 0)), b))

    # Per-chain and per-finger target states for coordinated boundary exploration
    chain_bound_sign: dict[tuple[str, str, str], str] = {}
    finger_group_targets: dict[tuple[str, str], tuple[str, str]] = {}
    for rank_i, b_name in enumerate(topo_names):
        info = classified[b_name]
        b_kind = str(info.get("kind", "limb"))
        side = str(info.get("side", "C"))
        grp = str(info.get("group_id", "00"))
        ckey = (b_kind, side, grp)
        if ckey not in chain_bound_sign:
            # Deterministic parity alternation + rng tie-break so both 'lo' and 'hi' chain arches are covered
            chain_bound_sign[ckey] = "hi" if ((idx + rank_i) % 2 == 0) else "lo"
        if b_kind == "digit":
            fkey = (side, grp)
            if fkey not in finger_group_targets:
                c_side = "hi" if ((idx + rank_i) % 2 == 0) else "lo"
                s_side = "hi" if ((idx + rank_i // 2) % 2 == 0) else "lo"
                finger_group_targets[fkey] = (c_side, s_side)

    sampled_eulers: dict[str, list[float]] = {}
    for rank_i, b_name in enumerate(topo_names):
        info = classified[b_name]
        kind = str(info.get("kind", "limb"))
        sub_role = str(info.get("sub_role", ""))
        side = str(info.get("side", "C"))
        grp = str(info.get("group_id", "00"))
        is_leaf = not bool(info.get("children"))

        if kind == "rigid" or bool(info.get("fixed_local", False)):
            sampled_eulers[b_name] = [0.0, 0.0, 0.0]
            continue

        lims = effective_sampling_limits_for_bone(info, enforce_anatomical_floor=enforce_anatomical_floor)
        parent_name = info.get("parent")
        has_parent = bool(parent_name and parent_name in sampled_eulers)
        ckey = (kind, side, grp)
        pref_bound = chain_bound_sign.get(ckey, "hi")

        raw = [0.0, 0.0, 0.0]

        if regime == "quasi_uniform_interior_coverage":
            # Stratified interior quantile coverage [0.12, 0.88] with zero inter-joint coupling
            for ax in range(3):
                lo, hi = lims[ax]
                span = hi - lo
                q_lo = lo + 0.12 * span
                q_hi = hi - 0.12 * span
                raw[ax] = float(rng.uniform(q_lo, q_hi)) if span > 1e-8 else lo

        elif regime == "in_phase_boundary_extension":
            # Same-sign boundary bending along chains to reveal cumulative arch/extension & mesh stretch limits
            for ax in range(3):
                lo, hi = lims[ax]
                b_side = pref_bound if ((ax + rank_i + idx) % 3 != 0) else ("lo" if pref_bound == "hi" else "hi")
                if has_parent and classified.get(parent_name, {}).get("kind") == kind:  # type: ignore[arg-type]
                    p_lims = effective_sampling_limits_for_bone(
                        classified[parent_name], enforce_anatomical_floor=enforce_anatomical_floor  # type: ignore[index]
                    )
                    p_lo, p_hi = p_lims[ax]
                    p_mid = 0.5 * (p_lo + p_hi)
                    p_val = sampled_eulers[parent_name][ax]  # type: ignore[index]
                    b_side = "hi" if p_val >= p_mid else "lo"
                raw[ax] = _sample_near_bound(rng, lo, hi, b_side, band_ratio=0.14)

        elif regime == "antiphase_fold_boundary":
            # Opposite-sign parent/child boundary folding (deep crouch, elbow/knee fold, S-curve spine/tail)
            for ax in range(3):
                lo, hi = lims[ax]
                b_side = "hi" if ((idx + rank_i + ax) % 2 == 0) else "lo"
                if has_parent and kind in ("limb", "digit", "spine", "tail", "flex", "wing", "head"):
                    p_lims = effective_sampling_limits_for_bone(
                        classified[parent_name], enforce_anatomical_floor=enforce_anatomical_floor  # type: ignore[index]
                    )
                    p_lo, p_hi = p_lims[ax]
                    p_mid = 0.5 * (p_lo + p_hi)
                    p_val = sampled_eulers[parent_name][ax]  # type: ignore[index]
                    b_side = "lo" if p_val >= p_mid else "hi"
                raw[ax] = _sample_near_bound(rng, lo, hi, b_side, band_ratio=0.14)

        elif regime == "cross_occlusion_twist_boundary":
            # Push lateral limbs & digits to cross-midline adduction + axial shaft twist (ry) boundaries
            for ax in range(3):
                lo, hi = lims[ax]
                if kind in ("limb", "wing"):
                    b_side = "hi" if ((idx + rank_i + ax) % 2 == 0) else "lo"
                    raw[ax] = _sample_near_bound(rng, lo, hi, b_side, band_ratio=0.13)
                elif kind == "digit":
                    c_side, s_side = finger_group_targets.get((side, grp), ("hi", "lo"))
                    target_side = c_side if ax == 0 else (s_side if ax == 2 else ("hi" if (idx + rank_i) % 2 == 0 else "lo"))
                    raw[ax] = _sample_near_bound(rng, lo, hi, target_side, band_ratio=0.13)
                else:
                    raw[ax] = _sample_bimodal_boundary(rng, lo, hi)

        else:
            # "full_body_dynamic_root_boundary": push ROOT pitch/roll/yaw and major limbs/jaws/wings to boundaries
            for ax in range(3):
                lo, hi = lims[ax]
                b_side = "hi" if ((idx // 2 + rank_i + ax) % 2 == 0) else "lo"
                if kind in ("root", "jaw", "head", "limb", "wing"):
                    raw[ax] = _sample_near_bound(rng, lo, hi, b_side, band_ratio=0.13)
                else:
                    raw[ax] = _sample_bimodal_boundary(rng, lo, hi)

        # Deterministic Rotating Boundary Slot Guarantee:
        # Ensure every bone axis hits both its lower 14% and upper 14% boundary across the 50 samples
        if regime != "quasi_uniform_interior_coverage":
            slot_phase = (idx + rank_i * 3) % 8
            if slot_phase == 0:
                for ax in range(3):
                    raw[ax] = _sample_near_bound(rng, lims[ax][0], lims[ax][1], "lo", band_ratio=0.12)
            elif slot_phase == 4:
                for ax in range(3):
                    raw[ax] = _sample_near_bound(rng, lims[ax][0], lims[ax][1], "hi", band_ratio=0.12)

        if is_leaf and not bool(info.get("allow_leaf_twist", False)):
            raw[1] = 0.0

        sampled_eulers[b_name] = [float(np.clip(raw[ax], lims[ax][0], lims[ax][1])) for ax in range(3)]

    boundary_meta = inspect_pose_boundary_utilization(
        sampled_eulers,
        classified,
        boundary_band_ratio=0.15,
        enforce_anatomical_floor=enforce_anatomical_floor,
    )
    return sampled_eulers, regime, boundary_meta


def audit_boundary_exploration_coverage(
    samples_or_records: list[dict[str, Any]],
    bones_meta: dict[str, dict[str, Any]],
    joint_limits_profile: dict[str, Any] | None = None,
    boundary_band_ratio: float = 0.15,
) -> dict[str, Any]:
    """Audits a G2 50-pose boundary exploration pack to verify:
      1. 100% of sampled poses stay strictly within `joint_limits.json` (`max_limit_violation_deg == 0.0`).
      2. High boundary touch coverage across all active bone axes (`lower_boundary_touch_ratio` and
         `upper_boundary_touch_ratio` in the outer `boundary_band_ratio` = 15% quartiles).
      3. Standard inter-finger and parent-child decoupling criteria (`audit_pose_dataset_coverage`).
    """
    first_info = next(iter(bones_meta.values())) if bones_meta else {}
    classified = (
        copy.deepcopy(bones_meta)
        if ("kind" in first_info and "sub_role" in first_info and "group_id" in first_info)
        else classify_rig_bones(bones_meta)
    )
    if joint_limits_profile is not None:
        classified = apply_joint_limits_profile(classified, joint_limits_profile)

    base_audit = audit_pose_dataset_coverage(
        samples_or_records,
        classified,
        enforce_anatomical_floor=False,
    )

    bone_names = sorted(classified.keys())
    N = len(samples_or_records)
    axis_names = ("rx", "ry", "rz")
    regime_counts: dict[str, int] = {r: 0 for r in G2_BOUNDARY_REGIMES}

    eulers_by_bone: dict[str, np.ndarray] = {b: np.zeros((N, 3), dtype=np.float64) for b in bone_names}
    for s_idx, rec in enumerate(samples_or_records):
        reg = rec.get("sampling_regime")
        if isinstance(reg, str):
            regime_counts[reg] = regime_counts.get(reg, 0) + 1
        jmap = rec.get("joints_rad", rec.get("joints", rec.get("bones", {})))
        for b in bone_names:
            if b in jmap:
                val = jmap[b].get("euler_xyz", jmap[b]) if isinstance(jmap[b], dict) else jmap[b]
                eulers_by_bone[b][s_idx] = np.asarray(val, dtype=np.float64)

    total_active_axes = 0
    lower_touched_axes = 0
    upper_touched_axes = 0
    both_touched_axes = 0
    untouched_lower: list[str] = []
    untouched_upper: list[str] = []
    max_violation_deg = 0.0
    per_bone_summary: dict[str, Any] = {}

    for b in bone_names:
        info = classified[b]
        kind = str(info.get("kind", "limb"))
        if kind == "rigid" or bool(info.get("fixed_local", False)):
            continue
        is_leaf = not bool(info.get("children"))
        lims = effective_sampling_limits_for_bone(info, enforce_anatomical_floor=False)
        arr = eulers_by_bone[b]
        obs_deg: list[list[float]] = []
        lims_deg: list[list[float]] = []
        span_utils: list[float] = []

        for ax_i, ax_name in enumerate(axis_names):
            lo, hi = float(lims[ax_i][0]), float(lims[ax_i][1])
            lims_deg.append([round(math.degrees(lo), 1), round(math.degrees(hi), 1)])
            if N == 0:
                obs_deg.append([0.0, 0.0])
                continue
            col = arr[:, ax_i]
            obs_min, obs_max = float(np.min(col)), float(np.max(col))
            obs_deg.append([round(math.degrees(obs_min), 1), round(math.degrees(obs_max), 1)])
            viol = max(0.0, lo - obs_min, obs_max - hi)
            max_violation_deg = max(max_violation_deg, math.degrees(viol))

            if is_leaf and ax_i == 1 and not bool(info.get("allow_leaf_twist", False)):
                continue
            span = hi - lo
            if span <= math.radians(1.0):
                continue

            total_active_axes += 1
            util = float(np.clip((obs_max - obs_min) / span, 0.0, 1.0))
            span_utils.append(util)
            hit_lo = obs_min <= (lo + boundary_band_ratio * span + 1e-6)
            hit_hi = obs_max >= (hi - boundary_band_ratio * span - 1e-6)
            if hit_lo:
                lower_touched_axes += 1
            else:
                untouched_lower.append(f"{b}.{ax_name}")
            if hit_hi:
                upper_touched_axes += 1
            else:
                untouched_upper.append(f"{b}.{ax_name}")
            if hit_lo and hit_hi:
                both_touched_axes += 1

        per_bone_summary[b] = {
            "kind": kind,
            "sub_role": info.get("sub_role", ""),
            "limits_source": info.get("limits_source", "canonical_default"),
            "limits_deg": lims_deg,
            "observed_min_max_deg": obs_deg,
            "mean_span_utilization": round(float(np.mean(span_utils)), 4) if span_utils else 1.0,
        }

    denom = float(max(1, total_active_axes))
    lo_ratio = round(lower_touched_axes / denom, 4)
    hi_ratio = round(upper_touched_axes / denom, 4)
    both_ratio = round(both_touched_axes / denom, 4)

    boundary_failures = list(base_audit.get("failures", []))
    if max_violation_deg > 0.05:
        boundary_failures.append(f"Pose exceeded configured joint limits by {max_violation_deg:.2f}°")
    if N >= 20 and both_ratio < 0.85:
        boundary_failures.append(
            f"Insufficient G2 boundary exploration coverage: both_boundaries_touch_ratio={both_ratio:.3f} < 0.85"
        )

    return {
        **base_audit,
        "schema": "monster-g2-boundary-exploration-audit-v1",
        "regime_counts": regime_counts,
        "dual_boundary_touch_rate": both_ratio,
        "boundary_exploration": {
            "boundary_band_ratio": boundary_band_ratio,
            "total_active_axes": total_active_axes,
            "lower_boundary_touched_axes": lower_touched_axes,
            "upper_boundary_touched_axes": upper_touched_axes,
            "both_boundaries_touched_axes": both_touched_axes,
            "lower_boundary_touch_ratio": lo_ratio,
            "upper_boundary_touch_ratio": hi_ratio,
            "both_boundaries_touch_ratio": both_ratio,
            "max_limit_violation_deg": round(max_violation_deg, 4),
            "untouched_lower_axes": untouched_lower,
            "untouched_upper_axes": untouched_upper,
        },
        "per_bone_boundary_summary": per_bone_summary,
        "passed": len(boundary_failures) == 0,
        "failures": boundary_failures,
    }


def audit_pose_dataset_coverage(
    samples_or_eulers: list[dict[str, Any]] | dict[str, Any],
    bones_meta: dict[str, dict[str, Any]],
    enforce_anatomical_floor: bool = True,
) -> dict[str, Any]:
    """Audits a synthesized pose dataset (`keypoints_gt.json["samples"]` or `{bone_name: (N, 3)}`)
    for joint-space quantile coverage, inter-finger / intra-chain decoupling, PCA effective rank,
    and S-curve anti-phase representation.
    Automatically detects and fails on coupled "fixed-ratio hand curl" or frozen-joint defects.
    """
    classified = classify_rig_bones(bones_meta)
    bone_names = sorted(classified.keys())
    regime_counts: dict[str, int] = {r: 0 for r in POSE_SAMPLING_REGIMES}

    if isinstance(samples_or_eulers, list):
        N = len(samples_or_eulers)
        eulers_by_bone: dict[str, np.ndarray] = {
            b: np.zeros((N, 3), dtype=np.float64) for b in bone_names
        }
        for s_idx, item in enumerate(samples_or_eulers):
            reg = item.get("sampling_regime")
            if isinstance(reg, str):
                regime_counts[reg] = regime_counts.get(reg, 0) + 1
            jmap = item.get("joints", item.get("bones", item))
            for b in bone_names:
                if b in jmap:
                    val = jmap[b].get("euler_xyz", jmap[b]) if isinstance(jmap[b], dict) else jmap[b]
                    eulers_by_bone[b][s_idx] = np.asarray(val, dtype=np.float64)
    elif isinstance(samples_or_eulers, dict):
        first_arr = np.asarray(next(iter(samples_or_eulers.values())), dtype=np.float64)
        N = int(first_arr.shape[0])
        eulers_by_bone = {
            b: np.asarray(samples_or_eulers[b], dtype=np.float64)
            if b in samples_or_eulers
            else np.zeros((N, 3), dtype=np.float64)
            for b in bone_names
        }
    else:
        raise TypeError("samples_or_eulers must be a list of sample records or a dict of Euler arrays")

    p90_ratios: list[float] = []
    full_ratios: list[float] = []
    under_covered_bones: list[str] = []
    active_bones: list[str] = []

    for b in bone_names:
        info = classified[b]
        kind = str(info.get("kind", "limb"))
        if kind == "rigid" or bool(info.get("fixed_local", False)):
            continue
        active_bones.append(b)
        lims = effective_sampling_limits_for_bone(info, enforce_anatomical_floor=enforce_anatomical_floor)
        is_leaf = not bool(info.get("children"))
        arr = eulers_by_bone[b]
        for ax in range(3):
            if is_leaf and ax == 1:
                continue
            span = float(lims[ax][1] - lims[ax][0])
            if span <= 1e-3:
                continue
            col = arr[:, ax]
            p05, p95 = np.percentile(col, [5.0, 95.0])
            p90_r = float((p95 - p05) / span)
            full_r = float((np.max(col) - np.min(col)) / span)
            p90_ratios.append(p90_r)
            full_ratios.append(full_r)
            if ax == 0 and p90_r < 0.50:
                under_covered_bones.append(b)

    # 1. Digit PCA Effective Rank & Inter-Finger Correlation Audit (Fixed-Ratio Hand Curl Guard)
    digit_bones = [
        b
        for b in active_bones
        if classified[b].get("kind") == "digit"
        and (
            effective_sampling_limits_for_bone(classified[b], enforce_anatomical_floor=enforce_anatomical_floor)[0][1]
            - effective_sampling_limits_for_bone(classified[b], enforce_anatomical_floor=enforce_anatomical_floor)[0][0]
        )
        > 1e-3
    ]
    digit_pc1_var_ratio = 0.0
    digit_eff_rank_ratio = 1.0
    if len(digit_bones) >= 2 and N >= 10:
        X_dig = np.column_stack([eulers_by_bone[b][:, 0] for b in digit_bones])
        X_centered = X_dig - np.mean(X_dig, axis=0, keepdims=True)
        std_dig = np.std(X_centered, axis=0, keepdims=True)
        valid_cols = std_dig[0] > 1e-6
        if int(np.sum(valid_cols)) >= 2:
            X_norm = X_centered[:, valid_cols] / std_dig[:, valid_cols]
            cov = (X_norm.T @ X_norm) / float(max(1, N - 1))
            eigvals = np.linalg.eigvalsh(cov)
            eigvals = np.maximum(eigvals, 0.0)
            eig_sum = float(np.sum(eigvals))
            if eig_sum > 1e-9:
                digit_pc1_var_ratio = float(np.max(eigvals) / eig_sum)
                part_ratio = (eig_sum ** 2) / float(np.sum(eigvals ** 2))
                digit_eff_rank_ratio = float(part_ratio / float(len(eigvals)))

    # Proximal finger bones across different `(side, group_id)` groups
    proximal_by_group: dict[tuple[str, str], str] = {}
    for b in digit_bones:
        info = classified[b]
        key = (str(info.get("side", "C")), str(info.get("group_id", "00")))
        if key not in proximal_by_group or int(info.get("chain_index", 1)) < int(
            classified[proximal_by_group[key]].get("chain_index", 99)
        ):
            proximal_by_group[key] = b

    prox_bones = list(proximal_by_group.values())
    max_inter_finger_corr = 0.0
    if len(prox_bones) >= 2 and N >= 10:
        M_prox = np.column_stack([eulers_by_bone[b][:, 0] for b in prox_bones])
        M_c = M_prox - np.mean(M_prox, axis=0, keepdims=True)
        M_std = np.std(M_c, axis=0, keepdims=True)
        if np.all(M_std > 1e-6):
            corr_mat = (M_c / M_std).T @ (M_c / M_std) / float(N)
            off_diag = np.abs(corr_mat - np.diag(np.diag(corr_mat)))
            max_inter_finger_corr = float(np.max(off_diag))

    # 2. Parent-Child Intra-Chain Correlation & Anti-Phase Sign-Flip Ratio
    max_parent_child_corr = 0.0
    antiphase_ratios: list[float] = []
    for b in active_bones:
        info = classified[b]
        p = info.get("parent")
        if not p or p not in active_bones:
            continue
        if classified[p].get("kind") != info.get("kind"):
            continue
        if info.get("kind") not in ("digit", "flex", "tail", "spine", "wing"):
            continue
        x_p = eulers_by_bone[p][:, 0]
        x_c = eulers_by_bone[b][:, 0]
        lims_p = effective_sampling_limits_for_bone(classified[p], enforce_anatomical_floor=enforce_anatomical_floor)[0]
        lims_c = effective_sampling_limits_for_bone(info, enforce_anatomical_floor=enforce_anatomical_floor)[0]
        mid_p = 0.5 * (lims_p[0] + lims_p[1])
        mid_c = 0.5 * (lims_c[0] + lims_c[1])
        cp = x_p - mid_p
        cc = x_c - mid_c
        sp = float(np.std(cp))
        sc = float(np.std(cc))
        if sp > 1e-6 and sc > 1e-6:
            r_pc = abs(float(np.mean(cp * cc) / (sp * sc)))
            max_parent_child_corr = max(max_parent_child_corr, r_pc)
            antiphase_ratios.append(float(np.mean((cp * cc) < 0.0)))

    mean_antiphase_ratio = float(np.mean(antiphase_ratios)) if antiphase_ratios else 0.5
    mean_p90_span = float(np.mean(p90_ratios)) if p90_ratios else 1.0
    min_p90_span = float(np.min(p90_ratios)) if p90_ratios else 1.0
    mean_full_span = float(np.mean(full_ratios)) if full_ratios else 1.0

    failures: list[str] = []
    if under_covered_bones:
        failures.append(f"Under-covered primary flexion axis (<50% p90 span) on bones: {under_covered_bones[:8]}")
    if len(digit_bones) >= 3 and digit_pc1_var_ratio > 0.55:
        failures.append(
            f"Coupled hand/digit curl detected: digit PC1 variance ratio {digit_pc1_var_ratio:.3f} > 0.55"
        )
    if len(prox_bones) >= 2 and max_inter_finger_corr > 0.65:
        failures.append(
            f"Inter-finger fixed-ratio coupling detected: max pairwise |corr| {max_inter_finger_corr:.3f} > 0.65"
        )
    if max_parent_child_corr > 0.82:
        failures.append(
            f"Excessive parent-child chain lockstep detected: max |corr| {max_parent_child_corr:.3f} > 0.82"
        )
    if antiphase_ratios and mean_antiphase_ratio < 0.12:
        failures.append(
            f"Insufficient S-curve / anti-phase pose coverage: sign-flip ratio {mean_antiphase_ratio:.3f} < 0.12"
        )

    return {
        "schema": "monster-pose-coverage-audit-v1",
        "num_samples": N,
        "active_bone_count": len(active_bones),
        "digit_bone_count": len(digit_bones),
        "regime_counts": regime_counts,
        "span_coverage": {
            "mean_p90_span_ratio": round(mean_p90_span, 4),
            "min_p90_span_ratio": round(min_p90_span, 4),
            "mean_full_span_ratio": round(mean_full_span, 4),
            "under_covered_bones": under_covered_bones,
        },
        "decoupling_metrics": {
            "digit_pc1_variance_ratio": round(digit_pc1_var_ratio, 4),
            "digit_effective_rank_ratio": round(digit_eff_rank_ratio, 4),
            "max_inter_finger_abs_corr": round(max_inter_finger_corr, 4),
            "max_parent_child_abs_corr": round(max_parent_child_corr, 4),
            "antiphase_sign_flip_ratio": round(mean_antiphase_ratio, 4),
        },
        "thresholds": {
            "min_p90_span_ratio": 0.50,
            "max_digit_pc1_variance_ratio": 0.55,
            "max_inter_finger_abs_corr": 0.65,
            "max_parent_child_abs_corr": 0.82,
            "min_antiphase_sign_flip_ratio": 0.12,
        },
        "passed": len(failures) == 0,
        "failures": failures,
    }


ARCHETYPE_SPECS: dict[str, dict[str, Any]] = {
    "winged": {
        "title": "带翼类 · 展翅拍翼与收翼周期 (Wing Spread, Flap & Fold)",
        "sop_clause": "SOP §3.3 带翼类",
        "focus_checks": "翼根/翼中/翼尖联动、羽片或翼膜折叠间隙、躯干与尾羽随动、是否卷束穿体",
        "action": "wing_spread_flap_fold",
        "frames": 48,
        "fps": 24,
    },
    "humanoid": {
        "title": "人形类 · 握拳半蹲至双手举顶站立 (Fist Squat to Overhead Reach)",
        "sop_clause": "SOP §3.3 人形类",
        "focus_checks": "肩/肘/前臂扭转/腕/指闭合、髋/膝/踝/趾配合、腋下与胯部体积挤压、衣摆随动",
        "action": "fist_squat_to_overhead_reach",
        "frames": 48,
        "fps": 24,
    },
    "quadruped": {
        "title": "四足/多足兽 · 重心转移、走跑与下颌撕咬 (Weight Shift, Gait & Jaw Bite)",
        "sop_clause": "SOP §3.3 四足/多足兽",
        "focus_checks": "前后左右腿交替支撑与相位、肩髋深屈曲、脊柱/颈/尾联动、下颌独立开合（非整头伪装张嘴）",
        "action": "weight_shift_gait_jaw_bite",
        "frames": 48,
        "fps": 24,
    },
    "aberrant": {
        "title": "异形怪物 · 多肢交替抓取与长尾/触手异步盘绕 (Multi-Limb Grasp & Async Coil)",
        "sop_clause": "SOP §3.3 异形怪物",
        "focus_checks": "多节链条扭转累积、软肢局部塌陷、铰接轴心准确度与刚柔部件边界",
        "action": "multi_limb_grasp_async_coil",
        "frames": 48,
        "fps": 24,
    },
}

G1_VIEWS = ("front", "side", "back", "three_quarter")
G1_SYNC_VIEWS = ("front", "side", "three_quarter")


def synthesize_archetype_pose_eulers(
    classified_bones: dict[str, dict[str, Any]],
    archetype: str,
    phase_or_frame: float | int,
    total_frames: int | None = None,
) -> dict[str, list[float]]:
    """Computes synchronized local Euler (rx, ry, rz) angles in radians at phase `phase_rad` (0..2pi)
    or frame index `(frame_idx, total_frames)` for any canonicalized rig (`<KIND>_<SIDE>_<GROUP>_<INDEX>`),
    guaranteeing identical motion intent across all 4 candidates regardless of minor bone count differences.
    """
    if archetype not in ARCHETYPE_SPECS:
        raise ValueError(f"Unsupported archetype: {archetype!r}; choose from {tuple(ARCHETYPE_SPECS)}")

    if total_frames is not None:
        phase_rad = 2.0 * math.pi * float(phase_or_frame) / float(max(1, total_frames))
    else:
        phase_rad = float(phase_or_frame)

    s1 = math.sin(phase_rad)
    c1 = math.cos(phase_rad)
    s2 = math.sin(2.0 * phase_rad)
    deg = math.radians
    eulers: dict[str, list[float]] = {}

    for b_name, info in classified_bones.items():
        kind = str(info.get("kind", "limb")).lower()
        sub_role = str(info.get("sub_role", "")).lower()
        side = str(info.get("side", "C")).upper()
        chain_idx = int(info.get("chain_index", 1))
        group_num = int(str(info.get("group_id", "01")) or 1)
        side_sign = 1.0 if side == "L" else (-1.0 if side == "R" else 0.0)
        alt_phase = phase_rad + (0.0 if (group_num % 2 == 1) else math.pi) + (0.0 if side != "R" else math.pi)

        if kind == "rigid" or info.get("fixed_local"):
            eulers[b_name] = [0.0, 0.0, 0.0]
        elif kind == "root":
            eulers[b_name] = [deg(4.0) * s1, 0.0, deg(5.0) * c1]
        elif kind == "spine":
            eulers[b_name] = [deg(8.0) * s1, deg(3.0) * c1, deg(6.0) * s2]
        elif kind == "head":
            eulers[b_name] = [deg(10.0) * c1, deg(5.0) * s1, deg(8.0) * s1]
        elif kind == "jaw":
            open_ratio = max(0.0, math.sin(phase_rad * 1.5))
            eulers[b_name] = [deg(26.0) * open_ratio, 0.0, 0.0]
        elif kind == "wing":
            flap = math.sin(phase_rad - 0.35 * chain_idx)
            fold = 0.5 * (1.0 - math.cos(phase_rad * 0.5))
            eulers[b_name] = [
                deg(38.0) * flap,
                deg(10.0) * side_sign * fold,
                deg(28.0) * side_sign * (0.6 * flap - 0.4 * fold),
            ]
        elif kind == "limb":
            if archetype == "humanoid":
                reach = math.sin(phase_rad)
                if sub_role == "clavicle_socket":
                    eulers[b_name] = [deg(15.0) * reach, 0.0, deg(12.0) * side_sign * reach]
                elif sub_role in ("upper_limb", "ball_socket"):
                    eulers[b_name] = [deg(55.0) * reach, deg(10.0) * side_sign, deg(22.0) * side_sign * reach]
                elif sub_role == "forearm_hinge":
                    eulers[b_name] = [deg(45.0) * max(0.0, -reach), deg(18.0) * s1, 0.0]
                else:
                    eulers[b_name] = [deg(18.0) * s1, 0.0, deg(10.0) * c1]
            elif archetype == "quadruped":
                gait = math.sin(alt_phase)
                if sub_role in ("upper_limb", "ball_socket"):
                    eulers[b_name] = [deg(34.0) * gait, 0.0, deg(6.0) * side_sign * c1]
                elif sub_role == "forearm_hinge":
                    eulers[b_name] = [deg(42.0) * max(0.0, -gait), 0.0, 0.0]
                else:
                    eulers[b_name] = [deg(20.0) * gait, 0.0, 0.0]
            else:
                wave = math.sin(alt_phase - 0.3 * chain_idx)
                eulers[b_name] = [deg(30.0) * wave, deg(8.0) * c1, deg(16.0) * side_sign * wave]
        elif kind == "digit":
            curl = 0.5 * (1.0 + math.sin(phase_rad))
            eulers[b_name] = [deg(32.0) * curl, 0.0, 0.0]
        elif kind in ("tail", "flex"):
            travel = math.sin(phase_rad * 1.5 - 0.45 * chain_idx)
            eulers[b_name] = [deg(16.0) * travel, deg(4.0) * c1, deg(22.0) * travel]
        else:
            eulers[b_name] = [deg(10.0) * s1, 0.0, deg(10.0) * c1]

    return eulers
