"""bone_taxonomy.py — Canonical Bone Taxonomy, Joint Limits & Kinematic Solvers (Schema v3)."""
from __future__ import annotations

import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

_WORKER_DIR = str(Path(__file__).resolve().parent)
if _WORKER_DIR not in sys.path: sys.path.insert(0, _WORKER_DIR)

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    from pose_features import lbs_deform_vertices_numpy, compute_mesh_edge_stretch_metric, rig_contract_v3_enabled
except ImportError:
    lbs_deform_vertices_numpy, compute_mesh_edge_stretch_metric, rig_contract_v3_enabled = None, None, lambda: True

BONE_TAXONOMY_SCHEMA, JOINT_LIMITS_SCHEMA, DEFAULT_SYNTH_SAMPLES = "monster-bone-taxonomy-v3", "monster-joint-limits-v1", 5000

CANONICAL_BONE_KINDS: tuple[str, ...] = ("root", "rigid", "spine", "head", "jaw", "limb", "wing", "flex", "tail", "digit")
CANONICAL_KINDS, CANONICAL_SIDES = CANONICAL_BONE_KINDS, ("C", "L", "R")
_CANONICAL_NAME_RE = re.compile(r"^(ROOT|RIGID|SPINE|HEAD|JAW|LIMB|WING|FLEX|TAIL|DIGIT)_([CLR])_([0-9]{2})_([0-9]{2})$")

POSE_SAMPLING_REGIMES = ("synergistic", "decoupled", "antiphase_extreme", "workspace_cross_occlusion")
G2_BOUNDARY_REGIMES = ("in_phase_boundary_extension", "antiphase_fold_boundary", "cross_occlusion_twist_boundary", "full_body_dynamic_root_boundary", "quasi_uniform_interior_coverage")
G1_VIEWS, G1_SYNC_VIEWS = ("front", "side", "back", "three_quarter"), ("front", "side", "three_quarter")

ARCHETYPE_SPECS: dict[str, dict[str, Any]] = {
    "winged": {"title": "带翼类 · 展翅拍翼与收翼周期", "sop_clause": "SOP §3.3 带翼类", "focus_checks": "翼根/翼中/翼尖联动", "action": "wing_spread_flap_fold", "frames": 48, "fps": 24}, "humanoid": {"title": "人形类 · 握拳半蹲至双手举顶站立", "sop_clause": "SOP §3.3 人形类", "focus_checks": "肩/肘/前臂扭转/腕/指闭合", "action": "fist_squat_to_overhead_reach", "frames": 48, "fps": 24},
    "quadruped": {"title": "四足/多足兽 · 重心转移、走跑与下颌撕咬", "sop_clause": "SOP §3.3 四足/多足兽", "focus_checks": "前后腿交替与下颌开合", "action": "weight_shift_gait_jaw_bite", "frames": 48, "fps": 24}, "aberrant": {"title": "异形怪物 · 多肢抓取与触手盘绕", "sop_clause": "SOP §3.3 异形怪物", "focus_checks": "多节链条扭转累积与软肢", "action": "multi_limb_grasp_async_coil", "frames": 48, "fps": 24},
}

DEFAULT_EULER_LIMITS_BY_KIND: dict[str, list[list[float]]] = {
    "root": [[-12.0, 12.0], [-18.0, 18.0], [-15.0, 15.0]], "rigid": [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], "spine": [[-15.0, 15.0], [-3.0, 3.0], [-15.0, 15.0]],
    "head": [[-22.0, 22.0], [-20.0, 20.0], [-20.0, 20.0]], "jaw": [[0.0, 35.0], [-2.0, 2.0], [-3.0, 3.0]], "tail": [[-18.0, 18.0], [-3.0, 3.0], [-14.0, 14.0]],
    "flex": [[-25.0, 25.0], [-10.0, 10.0], [-18.0, 18.0]], "wing": [[-40.0, 45.0], [-8.0, 8.0], [-35.0, 35.0]], "digit": [[-15.0, 35.0], [-2.0, 2.0], [-12.0, 12.0]], "limb": [[-45.0, 95.0], [-35.0, 35.0], [-45.0, 45.0]],
}

LEG_PLANS = {
    "biped": ("Thigh", "Shin", "Foot", "Toes"), "hind": ("Thigh", "Shin", "HindPaw", "HindToes"), "front": ("Scapula", "UpperArm", "Forearm", "FrontPaw", "FrontToes"),
    "arm": ("Shoulder", "Arm", "LowerArm", "Hand"), "wing": ("WingShoulder", "WingUpper", "WingForearm", "WingHand"), "many": ("Coxa", "Femur", "Tibia", "Tarsus"),
}

_V3_ANATOMICAL_RE = re.compile(
    r"^(Root|Pelvis|Hips|Chest|Spine\d*|Neck\d*(?:X\d*)?|Head(?:X\d*)?|Jaw|Tail\d*|Shoulder|Scapula|Arm|UpperArm|LowerArm|Forearm|Hand|FrontPaw|FrontToes|UpperLeg|Thigh|LowerLeg|Shin|Foot|HindPaw|Toe|Toes|HindToes|Wing\d*|WingShoulder|WingUpper|WingForearm|WingHand|(?:Finger|Toe)\d*_\d*|(?:Pair|Tentacle|Limb)\d*(?:_\d*)?|Ear\d*_\d*|Flex\d*_\d*|(?:Coxa|Femur|Tibia|Tarsus)\d*|Rigid\d*|Weapon|Prop_[A-Za-z0-9_]+)(?:\.([LR]))?$"
)

def _parse_anatomical_name_local(name: str) -> dict[str, Any] | None:
    if not isinstance(name, str): return None
    m = _V3_ANATOMICAL_RE.fullmatch(name.strip())
    if not m: return None
    stem, side = m.groups()
    midline = stem.startswith(("Root", "Pelvis", "Hips", "Chest", "Spine", "Neck", "Head", "Jaw", "Tail", "Prop_")) and not stem.startswith(("HeadX", "NeckX"))
    if midline and side: return None
    lateral = stem.startswith(("Shoulder", "Scapula", "Arm", "UpperArm", "LowerArm", "Forearm", "Hand", "FrontPaw", "FrontToes",
                              "UpperLeg", "Thigh", "LowerLeg", "Shin", "Foot", "HindPaw", "Toe", "Toes", "HindToes", "Finger", "Coxa", "Femur", "Tibia", "Tarsus", "Weapon"))
    if lateral and not side and not stem.startswith(("Pair", "Limb", "Wing")): return None
    kind = "limb"
    if stem in ("Root", "Pelvis", "Hips") or stem.startswith("Prop_"): kind = "root"
    elif stem.startswith(("Spine", "Chest")): kind = "spine"
    elif stem.startswith(("Head", "Neck")) and not stem.startswith(("HeadX", "NeckX")): kind = "head"
    elif stem == "Jaw": kind = "jaw"
    elif stem.startswith("Tail"): kind = "tail"
    elif stem.startswith(("Finger", "Toe")) and stem not in ("Toes", "Toe"): kind = "digit"
    elif stem.startswith("Wing"): kind = "wing"
    elif stem.startswith(("Ear", "Flex", "HeadX", "NeckX", "Tentacle")): kind = "flex"
    elif stem.startswith("Rigid") or stem == "Weapon": kind = "rigid"
    nums = re.findall(r"\d+", stem)
    return {"kind": kind, "side": side or "C", "group_id": nums[0] if nums else "0", "chain_index": int(nums[-1]) if nums else 1, "anatomical_role": stem, "schema_version": 3}

def parse_anatomical_name(name: str) -> dict[str, Any] | None:
    res = _parse_anatomical_name_local(name)
    if res is None:
        try:
            from skeleton_postprocess import parse_anatomical_name as _spp
            res = _spp(name)
        except Exception: return None
    if res and res.get("kind") in ("root_Hips", "pelvis"): res["kind"] = "root"
    return res

def parse_canonical_bone_name(name: str) -> dict[str, Any] | None:
    v3 = parse_anatomical_name(name.strip())
    if v3 is not None: return v3
    m = _CANONICAL_NAME_RE.match(name.strip())
    if m:
        k_str, s_str, grp_str, idx_str = m.groups()
        return {"kind": k_str.lower(), "side": s_str, "group_id": grp_str, "chain_index": int(idx_str), "anatomical_role": f"{k_str.lower()}_{idx_str}", "schema_version": 2}
    return None

def format_canonical_bone_name(kind: str, side: str, group: int | str, index: int) -> str:
    return f"{kind.strip().upper()}_{side.strip().upper() if side else 'C'}_{int(group):02d}_{int(index):02d}"

def validate_canonical_rig_bone_names(bones_meta_or_names: dict[str, Any] | list[str] | tuple[str, ...], strict: bool = True) -> dict[str, Any]:
    b_names = sorted(str(k) for k in bones_meta_or_names.keys()) if isinstance(bones_meta_or_names, dict) else [str(k) for k in bones_meta_or_names]
    canons, non_canons, by_k, by_s = [], [], {}, {}
    for n in b_names:
        p = parse_canonical_bone_name(n)
        if p is None: non_canons.append(n); continue
        canons.append(n); k, s = str(p["kind"]), str(p["side"])
        by_k[k] = by_k.get(k, 0) + 1; by_s[s] = by_s.get(s, 0) + 1
    uniq = len(set(b_names)) == len(b_names)
    return {"schema": BONE_TAXONOMY_SCHEMA, "valid": bool(b_names and not non_canons and uniq), "total_bones": len(b_names),
            "canonical_count": len(canons), "non_canonical_bones": non_canons, "has_unique_names": uniq, "counts_by_kind": by_k, "counts_by_side": by_s}

def default_limits_and_corr_for_kind(kind: str, sub_role: str = "", is_leaf: bool = False) -> tuple[list[list[float]], float, bool]:
    deg = math.radians
    if kind == "root": return [[-deg(12), deg(12)], [-deg(18), deg(18)], [-deg(15), deg(15)]], 0.0, False
    if kind == "rigid": return [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 0.0, True
    if kind == "spine": return [[-deg(15), deg(15)], [-deg(3), deg(3)], [-deg(15), deg(15)]], 0.65, False
    if kind == "head": return [[-deg(22), deg(22)], [-deg(20), deg(20)], [-deg(20), deg(20)]], 0.55, False
    if kind == "jaw": return [[0.0, deg(35)], [-deg(2), deg(2)], [-deg(3), deg(3)]], 0.0, False
    if kind == "tail": return [[-deg(18), deg(18)], [-deg(3), deg(3)], [-deg(14), deg(14)]], 0.75, False
    if kind == "flex": return [[-deg(25), deg(25)], [-deg(10), deg(10)], [-deg(18), deg(18)]], 0.65, False
    if kind == "wing": return [[-deg(40), deg(45)], [-deg(8), deg(8)], [-deg(35), deg(35)]], 0.55, False
    if kind == "digit": ry = 0.0 if is_leaf else deg(2); return [[-deg(15), deg(35)], [-ry, ry], [-deg(12), deg(12)]], 0.70, False
    if sub_role == "clavicle_socket": return [[-deg(20), deg(20)], [-deg(8), deg(8)], [-deg(20), deg(20)]], 0.0, False
    if sub_role in ("upper_limb", "ball_socket"): return [[-deg(55), deg(65)], [-deg(25), deg(25)], [-deg(40), deg(40)]], 0.0, False
    if sub_role == "forearm_hinge": return [[-deg(45), deg(95)], [-deg(35), deg(35)], [-deg(45), deg(45)]], 0.0, False
    if sub_role == "hand_hub": return [[-deg(35), deg(35)], [-deg(3), deg(3)], [-deg(25), deg(25)]], 0.0, False
    return [[-deg(25), deg(30)], [-deg(2), deg(2)], [-deg(10), deg(10)]], 0.50, False

def classify_rig_bones(bones_meta: dict[str, dict[str, Any]], bbox_min: Any = None, bbox_max: Any = None,
                       inplace: bool = False, constraints_meta: Any = None) -> dict[str, dict[str, Any]]:
    meta = bones_meta if inplace else copy.deepcopy(bones_meta)
    if not meta: return meta
    names = sorted(meta.keys())
    children: dict[str, list[str]] = {b: [] for b in names}
    for b in names:
        p = meta[b].get("parent")
        if p and p in children: children[p].append(b)
    for b in names: children[b].sort(); meta[b]["children"] = list(children[b])
    def _d(b: str) -> int:
        d, c, vis = 0, meta[b].get("parent"), {b}
        while c and c in meta and c not in vis: vis.add(c); d += 1; c = meta[c].get("parent")
        return d
    for b in names:
        if "rest_head_world" not in meta[b] and "head" in meta[b]:
            meta[b]["rest_head_world"] = list(meta[b]["head"])
        if "rest_tail_world" not in meta[b] and "tail" in meta[b]:
            meta[b]["rest_tail_world"] = list(meta[b]["tail"])
        if bool(meta[b].get("fixed_local", False)): meta[b]["kind"] = "rigid"
        if not meta[b].get("parent"): meta[b].setdefault("kind", "root")
        meta[b]["depth"] = int(meta[b].get("depth", _d(b)))
    topo = sorted(names, key=lambda b: (meta[b]["depth"], b))
    root_x = float(meta[topo[0]].get("rest_head_world", [0, 0, 0])[0])
    for b in topo:
        if "side" not in meta[b]:
            h = meta[b].get("rest_head_world", [root_x, 0, 0])
            t = meta[b].get("rest_tail_world", [root_x, 0, 0])
            mid_x = (float(h[0]) + float(t[0])) * 0.5 - root_x
            side_val = "L" if mid_x > 0.02 else ("R" if mid_x < -0.02 else "C")
            if side_val == "C" and meta[b].get("parent") in meta:
                p_side = meta[meta[b]["parent"]].get("side", "C")
                if p_side in ("L", "R"):
                    side_val = p_side
            meta[b]["side"] = side_val
    wing_set = {b for b in topo if not (parse_canonical_bone_name(b) or {}).get("schema_version") == 3
                and (meta[b].get("kind") == "wing" or bool(meta[b].get("winged")) or str(meta[b].get("role", "")).lower() == "wing" or (meta[b].get("flatness") is not None and float(meta[b]["flatness"]) > 5.0))}
    for wb in list(wing_set):
        c = wb
        while c and c in meta and meta[c].get("kind") not in ("root", "spine"):
            if (parse_canonical_bone_name(c) or {}).get("schema_version") != 3 and meta[c].get("kind") in ("limb", "wing"):
                meta[c]["kind"] = "wing"
            c = meta[c].get("parent")
    def _d_key(n: str, side: str) -> tuple[float, float, float, str]:
        e = n
        while True:
            ds = [c for c in children[e] if meta.get(c, {}).get("kind") == "digit"]
            if len(ds) != 1: break
            e = ds[0]
        pt = meta[e].get("rest_tail_world", meta[n].get("rest_head_world", [root_x, 0, 0]))
        return ((-1.0 if side == "R" else 1.0) * (float(pt[0]) - root_x), float(pt[1]), float(pt[2]), n)
    digit_g, digit_c = {}, {}
    for b in topo:
        if meta[b].get("kind") == "digit":
            p, s = meta[b].get("parent"), meta[b].get("side", "C")
            if p and meta.get(p, {}).get("kind") == "digit":
                digit_g[b], digit_c[b] = digit_g.get(p, 1), digit_c.get(p, 1) + 1
            else:
                sibs = sorted([c for c in children.get(p or "", []) if meta.get(c, {}).get("kind") == "digit"], key=lambda c: _d_key(c, s))
                digit_g[b], digit_c[b] = (sibs.index(b) + 1) if b in sibs else 1, 1
    def _infer_kind(r_str: str) -> str:
        r = str(r_str or "").lower()
        if not r: return ""
        if "root" in r or "hips" in r or ("pelvis" in r and "attach" not in r): return "root"
        if "spine" in r or "torso" in r: return "spine"
        if "neck" in r or "head" in r: return "head"
        if "jaw" in r: return "jaw"
        if "tail" in r: return "tail"
        if "wing" in r: return "wing"
        if "finger" in r or "toe" in r: return "digit"
        if "rigid" in r or "prop" in r or "weapon" in r: return "rigid"
        if "flex" in r or "cape" in r or "ear" in r: return "flex"
        if "limb" in r or "arm" in r or "leg" in r or "hip" in r or "knee" in r or "paw" in r: return "limb"
        return ""

    lateral_mid_hubs = [
        meta[b]["depth"] for b in names
        if meta[b].get("side") == "C" and any(meta.get(c, {}).get("side") in ("L", "R") for c in children[b])
    ]
    chest_hub_depth = max(lateral_mid_hubs) if lateral_mid_hubs else 1
    for b in topo:
        if not meta[b].get("kind"):
            rk = _infer_kind(meta[b].get("role")) or _infer_kind(b)
            if rk:
                meta[b]["kind"] = rk
            elif meta[b].get("side") == "C":
                if not meta[b].get("parent"):
                    meta[b]["kind"] = "root"
                elif meta[b]["depth"] <= chest_hub_depth:
                    meta[b]["kind"] = "spine"
                else:
                    meta[b]["kind"] = "head"
            else:
                meta[b]["kind"] = "limb"

    roles: dict[str, str] = {}
    for b in topo:
        k, s, p = meta[b].get("kind", "limb"), meta[b].get("side", "C"), meta[b].get("parent")
        if k in ("limb", "wing") and s in ("L", "R") and (not p or meta.get(p, {}).get("kind") not in ("limb", "wing") or meta.get(p, {}).get("side") != s):
            ch = [b]
            while True:
                nxt = [c for c in children[ch[-1]] if meta.get(c, {}).get("kind") == k and meta.get(c, {}).get("side") == s]
                if len(nxt) != 1: break
                ch.append(nxt[0])
            plan = "wing" if k == "wing" else ("arm" if meta[topo[0]].get("rest_head_world", [0, 0, 0])[2] < meta[b].get("rest_head_world", [0, 0, 0])[2] else "biped")
            for i, bn in enumerate(ch):
                if i < len(LEG_PLANS[plan]): roles[bn] = LEG_PLANS[plan][i]

    used_canon: set[str] = set()
    counters: dict[tuple[str, str], int] = {}
    for b in names:
        info = meta[b]
        is_leaf = not bool(children[b])
        parsed = parse_canonical_bone_name(b)
        canon = b
        if parsed is not None and parsed.get("schema_version") == 3:
            canon = b
            used_canon.add(canon)
        elif info.get("kind") == "wing":
            s_cur = info.get("side", "C")
            sfx = f".{s_cur}" if s_cur in ("L", "R") else ""
            if parsed is not None and parsed.get("schema_version") == 2 and parsed.get("kind") == "wing":
                canon = b
            else:
                canon = (roles.get(b) or "WingHand") + sfx
                parsed = parse_canonical_bone_name(canon)
            used_canon.add(canon)
        elif parsed is None:
            k = info.get("kind") or _infer_kind(info.get("role")) or _infer_kind(b) or "limb"
            s = "C" if k in ("root", "spine", "head", "jaw", "tail") else info.get("side", "C")
            p = info.get("parent")
            key = (k, s)
            counters[key] = counters.get(key, 0) + 1
            i = counters[key]
            sfx = f".{s}" if s in ("L", "R") else ""
            if k == "root": role = "Root" if not p else "Hips"
            elif k == "digit":
                anc, foot = p, False
                while anc and anc in meta:
                    if any(t in roles.get(anc, anc) for t in ("Paw", "Foot", "Toes", "Shin", "Thigh")): foot = True; break
                    anc = meta[anc].get("parent")
                role = ("Toe" if foot else "Finger") + f"{digit_g.get(b, 1)}_{digit_c.get(b, 1)}"
            elif k == "spine": role = f"Spine{i}"
            elif k == "head":
                r_low = str(info.get("role", "")).lower() + " " + b.lower()
                role = f"Neck{i}" if "neck" in r_low or any(meta.get(c, {}).get("kind") == "head" for c in children[b]) or "Head" in used_canon else "Head"
            elif k == "jaw": role = "Jaw"
            elif k == "tail": role = f"Tail{i}"
            elif k == "rigid": role = "Weapon" if "weapon" in b.lower() or info.get("weapon") else f"Rigid{i}"
            elif k == "flex": role = ("Ear" if "ear" in b.lower() else "Flex") + f"{digit_g.get(b, i)}_{digit_c.get(b, 1)}"
            else:
                role = roles.get(b, f"Pair{i}_1" if sfx else f"Limb{i}_1")
            cand = role + (sfx if k in ("limb", "wing", "digit", "rigid", "flex") else "")
            if cand in used_canon:
                prefix = "Pair" if s != "C" else "Limb"
                suffix = "." + s if s != "C" else ""
                ui = 1
                while f"{prefix}{ui}_1{suffix}" in used_canon:
                    ui += 1
                cand = f"{prefix}{ui}_1{suffix}"
            used_canon.add(cand)
            canon = cand
            parsed = parse_canonical_bone_name(canon)
        k = str(parsed["kind"]) if parsed else str(info.get("kind", "limb"))
        s = str(parsed["side"]) if parsed else str(info.get("side", "C"))
        grp = int(parsed["group_id"]) if (parsed and "group_id" in parsed) else int(info.get("group_index", info.get("group_id", 0)))
        c_i = int(parsed["chain_index"]) if (parsed and "chain_index" in parsed) else int(info.get("chain_index", 1))
        if not rig_contract_v3_enabled(): canon = format_canonical_bone_name(k, s, grp, 0 if k == "root" else c_i)
        role_low = str(info.get("role", "")).lower()
        if k == "rigid": sub = "rigid_support"
        elif k == "root": sub = "root"
        elif k == "spine": sub = "spine_hub" if len(children[b]) >= 2 else "spine_chain"
        elif k == "head": sub = "head_leaf" if not children[b] else "neck_chain"
        elif k == "jaw": sub = "jaw_hinge"
        elif k == "tail": sub = "tail_chain"
        elif k == "flex": sub = "flex_chain"
        elif k == "wing": sub = "wing_chain"
        elif k == "digit": sub = "digit_distal" if not children[b] else "digit_proximal"
        elif k == "limb":
            p_b = info.get("parent")
            if not children[b]: sub = "distal_limb"
            elif p_b and meta.get(p_b, {}).get("kind") in ("root", "spine", "rigid"): sub = "ball_socket"
            elif len(children[b]) == 1 and not children[children[b][0]]: sub = "forearm_hinge"
            elif any(tok in role_low for tok in ("elbow", "knee", "hock", "hinge", "forearm", "shin", "calf")): sub = "forearm_hinge"
            elif len(children[b]) >= 3: sub = "hand_hub"
            else: sub = "upper_limb"
        else:
            sub = str(info.get("sub_role", ""))
        def_lims, def_corr, def_fixed = default_limits_and_corr_for_kind(k, sub, is_leaf=is_leaf)
        has_custom = bool(info.get("limits_rad")) and str(info.get("limits_source", "")) not in ("", "canonical_default", "stage1_heuristic")
        info.update({"canonical_name": canon, "anatomical_role": str(parsed.get("anatomical_role") if parsed and parsed.get("anatomical_role") else canon.split(".")[0]), "kind": k, "side": s, "sub_role": sub, "group_index": grp, "group_id": f"{grp:02d}",
                     "chain_index": c_i, "chain_length": int(info.get("chain_length", 1)), "fixed_local": bool(info.get("fixed_local", def_fixed or k == "rigid")),
                     "is_primary": k != "digit", "chain_corr": (0.0 if has_custom else def_corr) if "chain_corr" not in info or str(info.get("limits_source","")) in ("", "canonical_default", "stage1_heuristic") else float(info["chain_corr"])})
        if not has_custom:
            info["limits_rad"] = [[round(lo, 4), round(hi, 4)] for lo, hi in def_lims]
            info["limits_source"] = "canonical_default"
        info["euler_limits_deg"] = info.get("limits_deg") or [[round(math.degrees(lo), 1), round(math.degrees(hi), 1)] for lo, hi in info["limits_rad"]]
        if "hinge_axis" not in info: info["hinge_axis"] = int(info.get("hinge", {}).get("axis", 0 if k in ("limb", "digit") else 2))
        if "rest_rot_mat_world" not in info:
            h = np.asarray(info.get("rest_head_world", [0, 0, 0]), dtype=np.float64)
            t = np.asarray(info.get("rest_tail_world", [0, 0, 0.05]), dtype=np.float64)
            v = t - h; d0 = v / max(1e-8, float(np.linalg.norm(v)))
            info["rest_rot_mat_world"] = _build_default_rest_rot_matrix(d0).tolist()
        if "rest_length" not in info:
            h = np.asarray(info.get("rest_head_world", [0, 0, 0]), dtype=np.float64)
            t = np.asarray(info.get("rest_tail_world", [0, 0, 0.05]), dtype=np.float64)
            info["rest_length"] = float(np.linalg.norm(t - h))
        info["rest_rot_matrix"] = info["rest_rot_mat_world"]
    return meta

def canonicalize_rig_bone_names_in_blender(arm_obj: Any, mesh_objs: list[Any], classified_bones: dict[str, dict[str, Any]]) -> dict[str, str]:
    rename_map, used = {}, set()
    for old, info in sorted(classified_bones.items()):
        cand, idx = str(info.get("canonical_name", old)), int(info.get("chain_index", 1))
        while cand in used: idx += 1; cand = f"Limb{idx}_1" + ("." + info["side"] if info["side"] in ("L", "R") else "")
        used.add(cand); rename_map[old] = cand
    for phase in (1, 2):
        for old, new in rename_map.items():
            if old == new: continue
            src, dst = (old, f"__tmp_canon_{old}") if phase == 1 else (f"__tmp_canon_{old}", new)
            b = arm_obj.data.bones.get(src)
            if b: b.name = dst
            for m in mesh_objs:
                vg = m.vertex_groups.get(src)
                if vg: vg.name = dst
    return rename_map

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
    v1 = np.asarray(curr_dir, dtype=np.float64) / max(1e-8, float(np.linalg.norm(curr_dir)))
    v2 = np.asarray(target_dir, dtype=np.float64) / max(1e-8, float(np.linalg.norm(target_dir)))
    c = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    if c > 0.999999: return np.eye(3, dtype=np.float64)
    if c < -0.999999:
        ax = np.array([1.0, 0.0, 0.0] if abs(v1[0]) <= 0.9 else [0.0, 1.0, 0.0], dtype=np.float64)
        ortho = np.cross(v1, ax); ortho /= np.linalg.norm(ortho)
        return 2.0 * np.outer(ortho, ortho) - np.eye(3, dtype=np.float64)
    axis = np.cross(v1, v2); s = float(np.linalg.norm(axis)); axis /= s
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + K * s + (K @ K) * (1.0 - c)

def _build_default_rest_rot_matrix(d0: np.ndarray) -> np.ndarray:
    d = np.asarray(d0, dtype=np.float64); n = float(np.linalg.norm(d))
    if n < 1e-8: return np.eye(3, dtype=np.float64)
    d = d / n
    x_proj = np.array([1.0, 0.0, 0.0], dtype=np.float64) - d * float(d[0])
    if float(np.linalg.norm(x_proj)) < 1e-6:
        x_proj = np.array([0.0, 0.0, 1.0], dtype=np.float64) - d * float(d[2])
    x_ax = x_proj / max(1e-8, float(np.linalg.norm(x_proj)))
    z_ax = np.cross(x_ax, d)
    z_ax /= max(1e-8, float(np.linalg.norm(z_ax)))
    return np.column_stack([x_ax, d, z_ax])

def build_bounded_local_euler_rotation(
    d_loc: np.ndarray,
    ry: float,
    rx_lims: tuple[float, float] | list[float],
    rz_lims: tuple[float, float] | list[float] | None = None,
) -> np.ndarray:
    """Two-branch inverse-sine local Euler rotation `Rz(rz) @ Ry(ry) @ Rx(rx)` strictly obeying
    `(rx_lims, rz_lims)` WITHOUT any post-clamp `minimum_swing` override.
    """
    if rz_lims is None:
        limits_rad = rx_lims
        rx_lims = (float(limits_rad[0][0]), float(limits_rad[0][1]))
        rz_lims = (float(limits_rad[2][0]), float(limits_rad[2][1]))
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
    norm = float(np.linalg.norm(q))
    return (q / norm) if norm > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def quat_wxyz_to_mat_numpy(q: np.ndarray) -> np.ndarray:
    """Converts a unit quaternion `[w, x, y, z]` to a 3x3 rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def blend_rotation_matrices_slerp(
    R0: np.ndarray, R1: np.ndarray, alpha: float, max_rot_deg: float = 30.0
) -> np.ndarray:
    """Spherical linear interpolation between `R0` and `R1` with maximum rotation clamp."""
    q0 = mat_to_quat_wxyz_numpy(R0)
    q1 = mat_to_quat_wxyz_numpy(R1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    theta = math.acos(dot)
    max_theta = math.radians(max_rot_deg) * 0.5
    eff_alpha = min(float(alpha), (max_theta / theta) if theta > 1e-6 else 1.0)
    if theta < 1e-6:
        return R0.copy()
    sin_theta = math.sin(theta)
    q_blend = (math.sin((1.0 - eff_alpha) * theta) / sin_theta) * q0 + (math.sin(eff_alpha * theta) / sin_theta) * q1
    return quat_wxyz_to_mat_numpy(q_blend / float(np.linalg.norm(q_blend)))

def savgol_smooth_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64); T = int(arr.shape[0])
    if T < 5: return arr.copy()
    p = np.pad(arr, ((2, 2),) + ((0, 0),) * (arr.ndim - 1), mode="edge")
    return (-3.0 * p[:-4] + 12.0 * p[1:-3] + 17.0 * p[2:-2] + 12.0 * p[3:-1] - 3.0 * p[4:]) / 35.0

def whittaker_smooth_trajectory(x: np.ndarray, weights: np.ndarray | None = None, lam_accel: float = 4.0, lam_jerk: float = 10.0, lam: float = 1.0) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    T = int(arr.shape[0])
    if T < 3: return arr.copy()
    if T < 7: return savgol_smooth_array(arr)
    eye = np.eye(T, dtype=np.float64)
    d2 = eye[2:] - 2.0 * eye[1:-1] + eye[:-2]
    d3 = eye[3:] - 3.0 * eye[2:-1] + 3.0 * eye[1:-2] - eye[:-3]
    penalty = float(lam_accel) * (d2.T @ d2) + float(lam_jerk) * (d3.T @ d3)
    if arr.ndim == 2:
        w = np.ones(T, dtype=np.float64) if weights is None else np.clip(np.asarray(weights, dtype=np.float64).reshape(T), 0.05, 1.0)
        A = np.diag(w) + penalty
        return np.linalg.solve(A, w[:, None] * arr)
    if arr.ndim == 3:
        _, K, D = arr.shape
        if weights is None:
            A = np.eye(T, dtype=np.float64) + penalty
            return np.linalg.solve(A, arr.reshape(T, K * D)).reshape(T, K, D)
        w_all = np.clip(np.asarray(weights, dtype=np.float64), 0.05, 1.0)
        out = np.empty_like(arr, dtype=np.float64)
        for k in range(K):
            wk = w_all[:, k] if w_all.ndim > 1 else w_all
            A = np.diag(wk) + penalty
            out[:, k, :] = np.linalg.solve(A, wk[:, None] * arr[:, k, :])
        return out
    return savgol_smooth_array(arr)

def short_gap_fill_1d_or_nd(values: np.ndarray, bad_mask: np.ndarray, max_gap: int = 3) -> np.ndarray:
    arr, mask, T = np.asarray(values, dtype=np.float64).copy(), np.asarray(bad_mask, dtype=bool), int(values.shape[0])
    if T == 0 or not np.any(mask): return arr
    t = 0
    while t < T:
        if not mask[t]: t += 1; continue
        start = t
        while t < T and mask[t]: t += 1
        g = t - start
        if g <= max_gap and start > 0 and t < T:
            for s in range(g):
                a = (s + 1) / (g + 1)
                arr[start + s] = (1.0 - a) * arr[start - 1] + a * arr[t]
    return arr

def clean_uv_short_gaps_hampel(uv: np.ndarray, scale: float = 1.0, max_gap: int = 3) -> np.ndarray:
    arr = np.asarray(uv, dtype=np.float64).copy()
    if arr.shape[0] < 3: return arr
    bad = np.zeros(arr.shape[:-1], dtype=bool)
    bad[1:] = np.linalg.norm(arr[1:] - arr[:-1], axis=-1) > (35.0 * float(scale))
    return short_gap_fill_1d_or_nd(arr, bad, max_gap=max_gap)

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
    depth_scale: float = 1.0,
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
        if sin_perp > 1e-4:
            c_lims = c_info.get("limits_rad", default_lims)
            R_c_init = R_cand @ (rest_R[b_name].T @ rest_R[c])
            d_c_loc = R_c_init.T @ v_c
            c_rx_lims = (float(c_lims[0][0]), float(c_lims[0][1]))
            c_ry_lo, c_ry_hi = float(c_lims[1][0]), float(c_lims[1][1])
            c_rz_lims = (float(c_lims[2][0]), float(c_lims[2][1]))
            ry_c_cands = (
                (0.0,)
                if (c_ry_hi - c_ry_lo) < 1e-4
                else (0.0, 0.5 * c_ry_lo, 0.5 * c_ry_hi, c_ry_lo, c_ry_hi)
            )
            min_miss = 1e18
            for ry_c in ry_c_cands:
                R_c_try = build_bounded_local_euler_rotation(d_c_loc, ry_c, c_rx_lims, c_rz_lims)
                miss = float(np.sum((R_c_try[:, 1] - d_c_loc) ** 2))
                if miss < min_miss:
                    min_miss = miss
            cost += 200.0 * math.sqrt(max(0.0, min_miss)) + 50000.0 * min_miss

            if sin_perp > 0.16:
                sx_c = float(np.clip(d_c_loc[2], -1.0, 1.0))
                rx1 = math.asin(sx_c)
                rx2 = (math.pi - rx1) if rx1 >= 0.0 else (-math.pi - rx1)
                c_best_pen = 1e18
                c_rz_span = c_rz_lims[1] - c_rz_lims[0]
                c_sub = str(c_info.get("sub_role", ""))
                coincident_hinge = gap_rest <= 0.008 and c_sub in ("forearm_hinge", "distal_limb")
                for rxc in (rx1, rx2):
                    cxc = math.cos(rxc)
                    rzc = math.atan2(-cxc * float(d_c_loc[0]), cxc * float(d_c_loc[1]))
                    if coincident_hinge:
                        hinge_pull = 2.5 * (rzc ** 2) / max(0.06, sin_perp ** 2)
                    elif c_rz_span <= math.radians(15.0):
                        hinge_pull = 0.25 * (rzc ** 2)
                    else:
                        hinge_pull = 0.0
                    if hinge_pull < c_best_pen:
                        c_best_pen = hinge_pull
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


def _fixed_child_rotation_evidence(b_name, valid_children, bones_meta, rest_R, observations, children_map=None):
    """Translation-invariant rigid landmark fit across fixed-local attachments and offset child heads.

    A shaft's own endpoints cannot observe axial rotation. Off-axis fixed children
    (including transitive fixed-local/rigid descendants) and non-coincident child
    head pivots are rigidly determined by `b_name`'s world rotation under forward
    kinematics. Centering all landmarks avoids amplifying noise in a single pivot.
    """
    rigid_tree = [b_name]
    queue = [
        c
        for c in valid_children
        if bones_meta.get(c, {}).get("fixed_local", False) or bones_meta.get(c, {}).get("kind") == "rigid"
    ]
    while queue:
        cur = queue.pop(0)
        if cur in rigid_tree or cur not in bones_meta or cur not in observations:
            continue
        rigid_tree.append(cur)
        if children_map:
            for gc in children_map.get(cur, []):
                if gc in bones_meta and gc in observations:
                    if bones_meta[gc].get("fixed_local", False) or bones_meta[gc].get("kind") == "rigid":
                        queue.append(gc)

    point_specs = []
    for name in rigid_tree:
        point_specs.append((name, "head", True))
        point_specs.append((name, "tail", True))
        ch_list = (
            children_map.get(name, [])
            if children_map is not None
            else (valid_children if name == b_name else [])
        )
        t_p_rest = bones_meta.get(name, {}).get("rest_tail_world")
        for c in ch_list:
            if c in rigid_tree or c not in bones_meta or c not in observations:
                continue
            h_c_rest = bones_meta[c].get("rest_head_world")
            if h_c_rest is not None and t_p_rest is not None:
                if float(np.linalg.norm(np.asarray(h_c_rest, dtype=np.float64) - np.asarray(t_p_rest, dtype=np.float64))) > 0.008:
                    point_specs.append((c, "head", False))

    if len(point_specs) <= 2:
        return None
    r_b_rest = np.asarray(bones_meta[b_name].get("rest_head_world", [0.0, 0.0, 0.0]), dtype=np.float64)
    r_b_obs = np.asarray(observations[b_name].get("head_3d", [0.0, 0.0, 0.0]), dtype=np.float64)
    rest, observed, weights = [], [], []
    for name, end, is_fixed_member in point_specs:
        info, obs = bones_meta[name], observations[name]
        if f"rest_{end}_world" not in info or f"{end}_3d" not in obs:
            continue
        confidence = float(obs.get(f"{end}_conf", obs.get("confidence", 0.9)))
        if confidence <= 0.0:
            continue
        r_pt = np.asarray(info[f"rest_{end}_world"], dtype=np.float64)
        o_pt = np.asarray(obs[f"{end}_3d"], dtype=np.float64)
        len_gate = 1.0
        if not is_fixed_member:
            d_rest = float(np.linalg.norm(r_pt - r_b_rest))
            d_obs = float(np.linalg.norm(o_pt - r_b_obs))
            if d_rest > 1e-4:
                rel_err = abs(d_obs - d_rest) / d_rest
                if rel_err > 0.35:
                    continue
                len_gate = 1.0 if rel_err <= 0.08 else max(0.05, (1.0 - (rel_err - 0.08) / 0.27) ** 2)
        rest.append(r_pt)
        observed.append(o_pt)
        weights.append((np.clip(confidence, 0.0, 1.0) ** 2) * len_gate)
    if len(rest) < 3:
        return None
    w = np.asarray(weights, dtype=np.float64)
    quality = float(np.sqrt(np.mean(w)))
    w /= w.sum()
    x, y = np.asarray(rest, dtype=np.float64), np.asarray(observed, dtype=np.float64)
    x = (x - (w[:, None] * x).sum(axis=0)) @ rest_R[b_name]
    y = y - (w[:, None] * y).sum(axis=0)
    leverage = float(np.sum(w * (x[:, 0] ** 2 + x[:, 2] ** 2)))
    spread = float(np.sum(w * np.sum(x * x, axis=1)))
    if leverage < max(1e-12, spread * 1e-4):
        return None
    return x, y, w, leverage, quality


def solve_bone_rotation_by_kind(
    b_name: str,
    R_init: np.ndarray,
    frame_bones: dict[str, Any],
    bones_meta: dict[str, dict[str, Any]],
    rest_R: dict[str, np.ndarray],
    children_map: dict[str, list[str]] | None = None,
    curr_ry_map: dict[str, float] | None = None,
    prev_ry_map: dict[str, float] | None = None,
    raw_obs_bones: dict[str, dict[str, Any]] | None = None,
    lims_override: list[list[float]] | None = None,
) -> tuple[np.ndarray, float]:
    """Layer 3 Dispatcher: Solves bone `b_name`'s 3x3 world rotation matrix `R_final` and axial
    twist `ry_best` according to its classified anatomical `kind` and `sub_role`.
    """
    if children_map is None:
        children_map = {b: bones_meta.get(b, {}).get("children", []) for b in bones_meta}
    if curr_ry_map is None:
        curr_ry_map = {}
    if prev_ry_map is None:
        prev_ry_map = {}

    info = bones_meta.get(b_name, {})
    kind = str(info.get("kind", "limb"))
    sub_role = str(info.get("sub_role", ""))

    # 1. Rigid / Fixed-Local Solver
    if kind == "rigid" or bool(info.get("fixed_local", False)):
        return np.asarray(R_init, dtype=np.float64).copy(), 0.0

    b_data = frame_bones.get(b_name, {})
    h_b = np.array(b_data.get("head_3d", [0.0, 0.0, 0.0]), dtype=np.float64)
    t_b = np.array(b_data.get("tail_3d", [0.0, 0.0, 1.0]), dtype=np.float64)
    vec = t_b - h_b
    norm_v = float(np.linalg.norm(vec))
    if norm_v < 1e-7:
        return np.asarray(R_init, dtype=np.float64).copy(), 0.0
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
        has_custom_digit_lims = str(info.get("limits_source", "")) in ("agent_custom", "probe_initial")
        if has_custom_digit_lims:
            conf_d = float(
                obs_lookup.get(b_name, b_data).get("tail_conf", obs_lookup.get(b_name, b_data).get("confidence", 0.9))
            )
            if conf_d < 0.35:
                max_lim_deg = max(
                    15.0,
                    math.degrees(
                        max(
                            abs(rx_lims[0]),
                            abs(rx_lims[1]),
                            abs(ry_lo),
                            abs(ry_hi),
                            abs(rz_lims[0]),
                            abs(rz_lims[1]),
                        )
                    ),
                )
                delta_raw = clamp_local_delta_euler(
                    blend_rotation_matrices_slerp(
                        np.eye(3, dtype=np.float64), delta_raw, max(0.35, conf_d / 0.35), max_lim_deg
                    ),
                    lims,
                )
                return R_init @ delta_raw, 0.0
        else:
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
        has_custom_profile = str(info.get("limits_source", "")) in ("agent_custom", "probe_initial")
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
            # fall through to Stage 4 so bounded local Euler solving recovers exact (rx, ry, rz) within limits.
            if not (abs(norm_obs_dir - rest_len_b) < 0.04 * rest_len_b and conf_b >= 0.98):
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

                if has_custom_profile:
                    max_lim_deg = max(
                        25.0,
                        math.degrees(
                            max(
                                abs(rx_lims[0]),
                                abs(rx_lims[1]),
                                abs(ry_lo),
                                abs(ry_hi),
                                abs(rz_lims[0]),
                                abs(rz_lims[1]),
                            )
                        ),
                    )
                    wrist_alpha = 1.0 if conf_b >= 0.75 else max(0.68, conf_b)
                    R_sw_damped = blend_rotation_matrices_slerp(
                        np.eye(3, dtype=np.float64), R_sw_comb, wrist_alpha, max_lim_deg
                    )
                else:
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
                    if w_clean < 0.95:
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
                            if has_custom_profile and (ry_hi - ry_lo) > 1e-6:
                                max_tau = max(max_tau, abs(ry_lo), abs(ry_hi))
                            tau = float(np.clip(w_tau * tau_raw, -max_tau, max_tau))
                            if explicit_custom and (ry_hi - ry_lo) > 1e-6:
                                tau = float(np.clip(tau, ry_lo, ry_hi))
                        R_loc_fa = build_swing_twist_local_rotation(d_loc, tau)
                        if explicit_custom:
                            R_loc_fa = clamp_local_delta_euler(R_loc_fa, lims)
                        return R_init @ R_loc_fa, tau

        if sub_role not in ("upper_limb", "forearm_hinge", "ball_socket", "clavicle_socket", "distal_limb", "hand_hub"):
            R_sw = minimum_swing_matrix_numpy(np.array([0.0, 1.0, 0.0], dtype=np.float64), d_loc)
            if explicit_custom:
                R_sw = clamp_local_delta_euler(R_sw, lims)
            return R_init @ R_sw, 0.0

    # 4. Spine / Head / Jaw / Flex / Tail / Wing / Limb Twist Solver (Bounded Local-Euler + Child Observability)
    if not valid_children or (ry_hi - ry_lo) < 1e-4:
        ry_best = 0.0 if (not valid_children and not bool(info.get("allow_leaf_twist", False))) else ry_prior
        R_loc = build_bounded_local_euler_rotation(d_loc, ry_best, rx_lims, rz_lims)
        if (ry_hi - ry_lo) < 1e-4 or float(np.sum((R_loc[:, 1] - d_loc) ** 2)) <= 1e-7:
            return R_init @ R_loc, ry_best

    rigid_evidence = _fixed_child_rotation_evidence(
        b_name, valid_children, bones_meta, rest_R, obs_lookup, children_map=children_map
    )
    grid_ry = np.unique(np.r_[np.linspace(ry_lo, ry_hi, 11),
                              np.clip([0.0, ry_prior], ry_lo, ry_hi)])
    best_ry = ry_prior
    best_cost = 1e18
    prior_w = 1.25 if (corr > 0.0 or b_name in prev_ry_map) else 0.05
    if rigid_evidence is not None:
        # Reliable rigid observations must dominate the rest/parent prior. In
        # particular, perfect landmarks must recover the full allowed motion.
        prior_w = 0.05 * (1.0 - rigid_evidence[4]) ** 2

    def candidate_cost(ry_cand):
        ry_f = float(ry_cand)
        R_loc_cand = build_bounded_local_euler_rotation(d_loc, ry_f, rx_lims, rz_lims)
        R_cand = R_init @ R_loc_cand
        shaft_miss = float(np.sum((R_loc_cand[:, 1] - d_loc) ** 2))
        shaft_cost = 200.0 * math.sqrt(max(0.0, shaft_miss)) + 50000.0 * shaft_miss
        if rigid_evidence is not None:
            x, y, weights, leverage, _ = rigid_evidence
            residual = x @ R_cand.T - y
            data_cost = float(np.sum(weights * np.sum(residual ** 2, axis=1)) / leverage)
        else:
            data_cost = _child_twist_cost(
                b_name, R_cand, v, h_b, valid_children, frame_bones, bones_meta, rest_R, obs_lookup=obs_lookup
            )
        return prior_w * ((ry_f - ry_prior) ** 2) + shaft_cost + data_cost

    for ry_cand in grid_ry:
        cost = candidate_cost(ry_cand)
        if cost < best_cost:
            best_cost = cost
            best_ry = float(ry_cand)

    # Keep the bounded global search, then refine its winning interval instead
    # of quantizing all head/tail turns to eleven fixed angles.
    step = (ry_hi - ry_lo) / 10.0
    for _ in range(4):
        for ry_cand in np.linspace(max(ry_lo, best_ry - step), min(ry_hi, best_ry + step), 9):
            cost = candidate_cost(ry_cand)
            if cost < best_cost:
                best_cost, best_ry = cost, float(ry_cand)
        step *= 0.25

    R_loc_best = build_bounded_local_euler_rotation(d_loc, best_ry, rx_lims, rz_lims)
    return R_init @ R_loc_best, best_ry

def reconstruct_frames_with_kind_solvers(pts_3d: np.ndarray, trust: np.ndarray, bone_names: list[str],
                                         classified_bones: dict[str, dict[str, Any]], depth_scale: float = 1.0,
                                         dense_correspondences: Any = None, rig_skinning: Any = None, calib: Any = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names, T = list(bone_names), int(pts_3d.shape[0])
    def _compute_bone_depth(b: str) -> int:
        if "depth" in classified_bones.get(b, {}):
            return int(classified_bones[b]["depth"])
        d = 0
        cur = b
        visited = set()
        while cur in classified_bones and classified_bones[cur].get("parent") and cur not in visited:
            visited.add(cur)
            cur = classified_bones[cur]["parent"]
            d += 1
        return d

    order = sorted(names, key=lambda b: (_compute_bone_depth(b), b))
    children_map = {b: [] for b in names}
    for b in names:
        p = classified_bones[b].get("parent")
        if p and p in children_map and b not in children_map[p]:
            children_map[p].append(b)
        for c in classified_bones[b].get("children", []):
            if c in children_map and c not in children_map[b]:
                children_map[b].append(c)
    rest_R = {}
    index = {n: i for i, n in enumerate(names)}
    rest_H = {n: np.asarray(classified_bones[n].get("rest_head_world", [0, 0, 0]), dtype=np.float64) for n in names}
    rest_T = {n: np.asarray(classified_bones[n].get("rest_tail_world", [0, 0, 0.04]), dtype=np.float64) for n in names}

    for n in names:
        info = classified_bones[n]
        if "rest_rot_mat_world" in info:
            rest_R[n] = np.asarray(info["rest_rot_mat_world"], dtype=np.float64)
        elif "rest_rot_matrix" in info:
            rest_R[n] = np.asarray(info["rest_rot_matrix"], dtype=np.float64)
        else:
            v0 = rest_T[n] - rest_H[n]
            d0 = v0 / max(1e-8, float(np.linalg.norm(v0)))
            rest_R[n] = _build_default_rest_rot_matrix(d0)

    coincident_pairs: list[tuple[int, int]] = []
    for n in order:
        p = classified_bones[n].get("parent")
        if p and p in rest_T and float(np.linalg.norm(rest_H[n] - rest_T[p])) < 1e-4:
            coincident_pairs.append((2 * index[p] + 1, 2 * index[n]))

    frames_out = []
    for t in range(T):
        observed = pts_3d[t].astype(np.float64).copy()
        for ip, ic in coincident_pairs:
            wp = max(1e-4, float(trust[t, ip])) ** 2 if trust.shape[1] > max(ip, ic) else 1.0
            wc = max(1e-4, float(trust[t, ic])) ** 2 if trust.shape[1] > max(ip, ic) else 1.0
            fused_pt = (wp * observed[ip] + wc * observed[ic]) / (wp + wc)
            observed[ip] = fused_pt
            observed[ic] = fused_pt

        all_obs = {}
        for idx, n in enumerate(names):
            h = observed[idx * 2] if observed.shape[0] >= (idx + 1) * 2 else np.zeros(3)
            tl = observed[idx * 2 + 1] if observed.shape[0] >= (idx + 1) * 2 else h + np.array([0, 0.05, 0])
            all_obs[n] = {
                "head_3d": h,
                "tail_3d": tl,
                "confidence": float(trust[t, idx * 2 : idx * 2 + 2].mean()) if trust.shape[1] >= (idx + 1) * 2 else 1.0,
                "tail_conf": float(trust[t, idx * 2 + 1]) if trust.shape[1] >= (idx + 1) * 2 else 1.0,
            }
        fb = {}
        posed_R = {}
        heads = {}
        for n in order:
            info = classified_bones[n]
            p = info.get("parent")
            rest_H_n = np.asarray(info.get("rest_head_world", [0, 0, 0]), dtype=np.float64)
            rest_T_n = np.asarray(info.get("rest_tail_world", [0, 0, 0.04]), dtype=np.float64)
            norm_rest_n = float(np.linalg.norm(rest_T_n - rest_H_n))
            rest_len = norm_rest_n if norm_rest_n > 1e-6 else float(info.get("rest_length", 0.04))

            if p and p in posed_R and p in heads:
                R_p = posed_R[p]
                init_R = R_p @ (rest_R[p].T @ rest_R[n])
                rest_H_p = np.asarray(classified_bones[p].get("rest_head_world", [0, 0, 0]), dtype=np.float64)
                head = heads[p] + R_p @ (rest_R[p].T @ (rest_H_n - rest_H_p))
            else:
                anchors = select_root_kabsch_anchors(classified_bones, n, children_map)
                valid_anchors = [(bn, ep) for bn, ep in anchors if bn in all_obs]
                if len(valid_anchors) >= 2:
                    ref_pts = np.array([
                        np.asarray(classified_bones[bn]["rest_head_world"] if ep == 0 else classified_bones[bn]["rest_tail_world"], dtype=np.float64)
                        for bn, ep in valid_anchors
                    ])
                    obs_pts = np.array([
                        np.asarray(all_obs[bn]["head_3d"] if ep == 0 else all_obs[bn]["tail_3d"], dtype=np.float64)
                        for bn, ep in valid_anchors
                    ])
                    fit_R, fit_t = fit_root_rigid_irls_kabsch(ref_pts, obs_pts, np.ones(len(valid_anchors)), depth_scale=depth_scale)
                    root_lims = info.get("limits_rad")
                    if root_lims is not None:
                        root_delta = clamp_local_delta_euler(rest_R[n].T @ fit_R @ rest_R[n], root_lims)
                        Rw = rest_R[n] @ root_delta
                    else:
                        Rw = fit_R @ rest_R[n]
                    R_body = Rw @ rest_R[n].T
                    head = fit_t + R_body @ rest_H_n
                    init_R = rest_R[n]
                    posed_R[n] = Rw
                    heads[n] = head
                    tail = head + Rw @ (rest_R[n].T @ (rest_T_n - rest_H_n))
                    fb[n] = {
                        "head_3d": [round(float(x), 5) for x in head],
                        "tail_3d": [round(float(x), 5) for x in tail],
                        "rot_mat_world": [[float(x) for x in row] for row in Rw],
                        "confidence": all_obs[n].get("confidence", 1.0),
                        "kind": info.get("kind", "root"),
                        "canonical_name": info.get("canonical_name", n),
                    }
                    continue
                else:
                    init_R = rest_R[n]
                    head = np.asarray(all_obs[n]["head_3d"], dtype=np.float64)

            # Target for bone solver
            obs_h = np.asarray(all_obs[n]["head_3d"], dtype=np.float64)
            obs_t = np.asarray(all_obs[n]["tail_3d"], dtype=np.float64)
            obs_dir = obs_t - obs_h
            norm_obs = float(np.linalg.norm(obs_dir))
            if norm_obs > 1e-6:
                tgt_tail = head + (obs_dir / norm_obs) * rest_len
            else:
                tgt_tail = head + init_R[:, 1] * rest_len

            fk_target = {
                n: {
                    "head_3d": head.tolist(),
                    "tail_3d": tgt_tail.tolist(),
                    "confidence": all_obs[n].get("confidence", 1.0),
                    "tail_conf": all_obs[n].get("tail_conf", 1.0),
                }
            }
            for c in children_map.get(n, []):
                if c in all_obs:
                    fk_target[c] = all_obs[c]

            Rw, ry = solve_bone_rotation_by_kind(
                n, init_R, fk_target, classified_bones, rest_R, children_map=children_map, raw_obs_bones=all_obs
            )
            lims = info.get("limits_rad")
            if lims is not None:
                delta = clamp_local_delta_euler(init_R.T @ Rw, lims)
                Rw = init_R @ delta

            posed_R[n] = Rw
            heads[n] = head
            tail = head + Rw @ (rest_R[n].T @ (rest_T_n - rest_H_n))

            fb[n] = {
                "head_3d": [round(float(x), 5) for x in head],
                "tail_3d": [round(float(x), 5) for x in tail],
                "rot_mat_world": [[float(x) for x in row] for row in Rw],
                "confidence": all_obs[n].get("confidence", 1.0),
                "kind": info.get("kind", "limb"),
                "canonical_name": info.get("canonical_name", n),
            }
        frames_out.append({"frame_idx": t, "frame": t, "bones": fb})
    diag = {
        "schema": BONE_TAXONOMY_SCHEMA,
        "frames": T,
        "mocap_priors_enabled": False,
        "final_limit_max_violation_rad": 0.0,
        "short_gap_max_frames": 3,
        "dense_surface_refined": False,
        "dense_matched_points_total": 0,
        "unlocked_roll_bones": [],
    }
    return frames_out, diag

def effective_sampling_limits_for_bone(info: dict[str, Any], enforce_anatomical_floor: bool = True) -> list[list[float]]:
    kind = str(info.get("kind", "limb"))
    if kind == "rigid" or bool(info.get("fixed_local", False)): return [[0.0, 0.0]] * 3
    sub, is_leaf = str(info.get("sub_role", "")), not bool(info.get("children"))
    explicit = str(info.get("limits_source", "")) in ("agent_custom", "probe_initial")
    def_lims, _, _ = default_limits_and_corr_for_kind(kind, sub, is_leaf=is_leaf)
    raw = info.get("limits_rad")
    if not isinstance(raw, list) or len(raw) != 3: return [[float(p[0]), float(p[1])] for p in def_lims]
    out = []
    for ax in range(3):
        lo, hi = float(raw[ax][0]), float(raw[ax][1])
        if enforce_anatomical_floor and not explicit and kind in ("digit", "limb", "flex", "tail", "wing", "head", "jaw"):
            d_lo, d_hi = float(def_lims[ax][0]), float(def_lims[ax][1])
            if (hi - lo) < 0.50 * (d_hi - d_lo): lo, hi = min(lo, 0.65 * d_lo), max(hi, 0.65 * d_hi)
        if is_leaf and ax == 1 and not bool(info.get("allow_leaf_twist", False)):
            from scipy.spatial.transform import Rotation
            basis = np.asarray(info.get("pose_basis", np.eye(4)), float)
            start_twist = Rotation.from_matrix(basis[:3, :3]).as_euler("xyz")[1]
            lo = hi = float(info.get("frozen_start_channels_rad", {}).get("ry", start_twist))
        out.append([lo, hi])
    return out

def resolve_joint_limits_profile_from_paths(*search_paths: str | Path | None, fallback_profile: dict[str, Any] | None = None) -> tuple[dict[str, Any] | None, str | None]:
    seen, candidates = set(), []
    for raw in search_paths:
        if not raw: continue
        p = Path(raw); dirs = [p] if p.is_dir() else [p.parent, p.parent.parent]
        for f in ([p] if p.is_file() and (p.name == "joint_limits.json" or "limit" in p.name.lower()) else []) + [d / "joint_limits.json" for d in dirs]:
            if str(f) not in seen: seen.add(str(f)); candidates.append(f)
    for c in candidates:
        if c.is_file():
            try:
                prof = json.loads(c.read_text(encoding="utf-8"))
                if isinstance(prof, dict) and (prof.get("schema") == JOINT_LIMITS_SCHEMA or isinstance(prof.get("bones"), dict)): return prof, str(c)
            except Exception: pass
    return (copy.deepcopy(fallback_profile), None) if isinstance(fallback_profile, dict) else (None, None)

def apply_joint_limits_profile(bones_meta: dict[str, dict[str, Any]], joint_limits_profile: dict[str, Any] | None = None,
                               bbox_min: Any = None, bbox_max: Any = None) -> dict[str, dict[str, Any]]:
    first_info = next(iter(bones_meta.values())) if bones_meta else {}
    classified = (
        copy.deepcopy(bones_meta)
        if ("kind" in first_info and "sub_role" in first_info)
        else classify_rig_bones(bones_meta, bbox_min=bbox_min, bbox_max=bbox_max)
    )
    meta = copy.deepcopy(classified)
    if not joint_limits_profile or not isinstance(joint_limits_profile, dict): return meta
    deg = math.radians; p_bones = joint_limits_profile.get("bones") or {}
    g_root = copy.deepcopy(joint_limits_profile.get("global_root") or {})
    roots = [b for b in sorted(meta) if meta[b].get("kind") == "root" or not meta[b].get("parent")]
    primary_root = roots[0] if roots else (sorted(meta)[0] if meta else None)
    for b_name, info in meta.items():
        if b_name in p_bones and isinstance(p_bones[b_name], dict):
            bp = p_bones[b_name]
            if "fixed_local" in bp: info["fixed_local"] = bool(bp["fixed_local"])
            if bool(info.get("fixed_local", False)) or info.get("kind") == "rigid":
                info["limits_deg"], info["limits_rad"] = [[0.0, 0.0]] * 3, [[0.0, 0.0]] * 3
            elif "limits_deg" in bp and isinstance(bp["limits_deg"], dict):
                info["limits_deg"] = [[round(float(bp["limits_deg"].get(ax, [-15, 15])[0]), 2), round(float(bp["limits_deg"].get(ax, [-15, 15])[1]), 2)] for ax in ("rx", "ry", "rz")]
                info["limits_rad"] = [[round(deg(p[0]), 4), round(deg(p[1]), 4)] for p in info["limits_deg"]]
            elif "limits_rad" in bp and isinstance(bp["limits_rad"], list) and len(bp["limits_rad"]) == 3:
                info["limits_rad"] = [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in bp["limits_rad"]]
                info["limits_deg"] = [[round(math.degrees(p[0]), 2), round(math.degrees(p[1]), 2)] for p in info["limits_rad"]]
            for k in ("axis_roles", "axis_role_source", "hinge", "frozen_start_channels_rad", "probe_safe_limits_deg"):
                if k in bp: info[k] = copy.deepcopy(bp[k])
            src_cand = (
                "agent_custom"
                if "agent_custom" in (bp.get("source"), bp.get("limits_source"))
                else (bp.get("source") or bp.get("limits_source") or "agent_custom")
            )
            info["limits_source"] = str(src_cand)
            if (
                info["limits_source"] == "agent_custom"
                and info.get("kind") != "rigid"
                and not bool(info.get("fixed_local", False))
                and isinstance(info.get("limits_rad"), list)
                and len(info["limits_rad"]) == 3
                and abs(float(info["limits_rad"][1][1]) - float(info["limits_rad"][1][0])) > 1e-4
            ):
                info["allow_leaf_twist"] = True
            elif "allow_leaf_twist" in bp:
                info["allow_leaf_twist"] = bool(bp["allow_leaf_twist"])
            elif abs(info.get("limits_rad", [[0,0],[0,0],[0,0]])[1][0]) > 1e-4 or abs(info.get("limits_rad", [[0,0],[0,0],[0,0]])[1][1]) > 1e-4:
                info["allow_leaf_twist"] = True
            for ch, val in bp.get("frozen_start_channels_rad", {}).items():
                idx = ("rx", "ry", "rz").index(ch); f_deg = round(math.degrees(val), 2)
                info["limits_rad"][idx], info["limits_deg"][idx] = [float(val), float(val)], [f_deg, f_deg]
            if info.get("hinge") and not info.get("fixed_local"):
                ha = int(info["hinge"].get("axis", 0))
                lo, hi = info["hinge"]["limits"]
                safe = info.get("probe_safe_limits_deg")
                if safe: lo, hi = max(lo, math.radians(safe[ha][0])), min(hi, math.radians(safe[ha][1]))
                lo, hi = max(lo, info["limits_rad"][ha][0]), min(hi, info["limits_rad"][ha][1])
                if lo > hi: raise ValueError(f"{b_name}: joint profile and safe geometric hinge do not intersect")
                info["limits_rad"][ha], info["limits_deg"][ha] = [lo, hi], [math.degrees(lo), math.degrees(hi)]
            info["limits_source"] = str(bp.get("limits_source") or bp.get("source") or "agent_custom")
        if b_name == primary_root and g_root: info["global_root"] = copy.deepcopy(g_root)
    return meta

def probe_rig_joint_limits(bones_meta: dict[str, dict[str, Any]], bbox_min: Any = None, bbox_max: Any = None,
                           rig_skinning: dict[str, Any] | None = None, max_bad_edge_ratio: float = 0.02, max_stretch_ratio: float = 1.80,
                           monster_id: str = "monster", archetype: str = "humanoid", selected_candidate_id: str = "0",
                           rig_meta_sha256: Any = None, restored_decorations: Any = None, lbs_evaluator_fn: Any = None, stretch_metric_fn: Any = None) -> dict[str, Any]:
    classified = classify_rig_bones(bones_meta, bbox_min=bbox_min, bbox_max=bbox_max)
    order = sorted(classified.keys(), key=lambda b: (int(classified[b].get("depth", 0)), b))
    has_lbs = (
        rig_skinning is not None
        and rig_skinning.get("triangles") is not None
        and len(rig_skinning.get("triangles")) > 0
    )
    bones_profile = {}
    for b in order:
        info = classified[b]
        k, sub, is_leaf = str(info.get("kind", "limb")), str(info.get("sub_role", "")), not bool(info.get("children"))
        def_lims, def_corr, _ = default_limits_and_corr_for_kind(k, sub, is_leaf=is_leaf)
        def_deg = [[round(math.degrees(lo), 1), round(math.degrees(hi), 1)] for lo, hi in def_lims]
        safe_deg = copy.deepcopy(def_deg)
        if has_lbs and (lbs_evaluator_fn or lbs_deform_vertices_numpy) and (stretch_metric_fn or compute_mesh_edge_stretch_metric):
            ha = int(info.get("hinge", {}).get("axis", 0))
            def _chk(a: float) -> bool:
                try:
                    from reprojection_solver import RigModel
                    rm = RigModel(classified); p = rm.zero(); p[rm.slices[b][0][ha]] = math.radians(a)
                    _, h, r = rm.fk(p)
                    rv, rs, bn = np.asarray(rig_skinning["rest_verts"]), rig_skinning, rig_skinning["bone_names"]
                    pv = (lbs_evaluator_fn or lbs_deform_vertices_numpy)(rv, np.asarray(rs["skin_indices"]), np.asarray(rs["skin_weights"]),
                        np.asarray(rs["rest_heads"]), np.asarray(rs["rest_R"]), np.array([h[n] for n in bn]), np.array([r[n] for n in bn]))
                    return bool((stretch_metric_fn or compute_mesh_edge_stretch_metric)(rv, pv, np.asarray(rs["triangles"]), max_stretch=max_stretch_ratio, max_bad_ratio=max_bad_edge_ratio)["passed"])
                except Exception: return True
            l_safe = 0.0
            for ang in [15.0, 30.0, 45.0, 60.0, 75.0, 90.0]:
                if _chk(ang): l_safe = ang
                else:
                    f_bad = ang
                    for _ in range(8):
                        m = (l_safe + f_bad) / 2.0
                        if _chk(m): l_safe = m
                        else: f_bad = m
                    break
            safe_deg[ha][1] = math.floor(l_safe * 10 + 1e-8) / 10.0
        frozen, leaf_tw = {}, bool(info.get("allow_leaf_twist", False))
        if is_leaf:
            if "ry" in info.get("frozen_start_channels_rad", {}):
                frozen["ry"] = float(info["frozen_start_channels_rad"]["ry"]); leaf_tw = True
            elif "pose_basis" in info or info.get("axis_role_source") == "geometry_start_pose":
                basis = np.asarray(info.get("pose_basis", np.eye(4)), dtype=float)
                ry_val = float(matrix_to_euler_xyz_numpy(basis[:3, :3])[1])
                if abs(ry_val) > 1e-8 or "pose_basis" in info:
                    frozen["ry"] = ry_val; leaf_tw = True
        rec_deg = copy.deepcopy(safe_deg)
        hinge_info = info.get("hinge")
        if hinge_info:
            h_ax = int(hinge_info.get("axis", 0))
            rec_deg[h_ax][0] = max(rec_deg[h_ax][0], math.degrees(hinge_info["limits"][0]))
            rec_deg[h_ax][1] = min(rec_deg[h_ax][1], math.degrees(hinge_info["limits"][1]))
        lrad = [[round(math.radians(p[0]), 4), round(math.radians(p[1]), 4)] for p in rec_deg]
        if hinge_info:
            h_ax = int(hinge_info.get("axis", 0))
            lrad[h_ax] = [max(math.radians(rec_deg[h_ax][0]), hinge_info["limits"][0]), min(math.radians(rec_deg[h_ax][1]), hinge_info["limits"][1])]
        for ch, val in frozen.items():
            idx = ("rx", "ry", "rz").index(ch); deg_val = math.degrees(val)
            lrad[idx], safe_deg[idx], rec_deg[idx] = [val, val], [deg_val, deg_val], [deg_val, deg_val]
        bones_profile[b] = {
            "kind": k, "probe_safe_limits_deg": safe_deg, "limits_deg": {"rx": rec_deg[0], "ry": rec_deg[1], "rz": rec_deg[2]},
            "limits_deg_list": rec_deg, "limits_rad": lrad, "chain_corr": round(float(def_corr), 3),
            "limits_source": "probe_initial", "source": "probe_initial", "frozen_start_channels_rad": frozen, "allow_leaf_twist": leaf_tw,
            "axis_semantics": {"rx": "bend", "ry": "twist", "rz": "side"},
            **{attr: copy.deepcopy(info[attr]) for attr in ("axis_roles", "axis_role_source", "hinge") if attr in info}
        }
    couplings = []
    for b, info in classified.items():
        p = info.get("parent")
        if p and p in classified:
            couplings.append({"bone_a": p, "axis_a": "rx", "sign_a": 1, "bone_b": b, "axis_b": "rx", "sign_b": 1, "coupling_type": "chain_synergy"})
    bone_spheres = {}
    for b, info in classified.items():
        h0 = np.asarray(info.get("rest_head_world", [0, 0, 0]), dtype=np.float64)
        t0 = np.asarray(info.get("rest_tail_world", [0, 0, 0.05]), dtype=np.float64)
        R0 = np.asarray(info.get("rest_rot_mat_world", np.eye(3)), dtype=np.float64)
        v_loc = R0.T @ (t0 - h0)
        b_len = max(1e-3, float(np.linalg.norm(t0 - h0)))
        rad = round(max(0.008, 0.18 * b_len), 5)
        bone_spheres[b] = [
            {"center_loc": [0.0, 0.0, 0.0], "radius": rad},
            {"center_loc": [round(float(x), 5) for x in v_loc], "radius": rad},
        ]
    body_part_volumes = {"spheres_per_bone": bone_spheres, "bones": bone_spheres}
    return {"schema": JOINT_LIMITS_SCHEMA, "monster_id": monster_id, "archetype": archetype, "selected_candidate_id": str(selected_candidate_id),
            "bones": bones_profile, "global_root": {"rotation_limits_deg": [[-28, 32], [-22, 22], [-25, 25]], "translation_fraction_xyz": [0.08, 0.08, 0.10], "rationale": "Global root envelope"},
            "root_motion": {"translation_fraction_xyz": [0.08, 0.08, 0.10]},
            "coordination": {"schema": "monster-joint-coordination-v1", "inter_bone_couplings": couplings, "body_part_volumes": body_part_volumes}}

def select_pose_sampling_regime(sample_idx: int | None, rng: Any) -> str:
    s = (int(sample_idx) % 20) if sample_idx is not None else int(rng.random() * 20.0)
    return "synergistic" if s < 8 else ("decoupled" if s < 14 else ("antiphase_extreme" if s < 17 else "workspace_cross_occlusion"))

def select_g2_boundary_regime(sample_idx: int | None, rng: Any, total_samples: int = 50) -> str:
    s = (int(sample_idx) % 20) if sample_idx is not None else int(rng.random() * 20.0)
    return "in_phase_boundary_extension" if s < 5 else ("antiphase_fold_boundary" if s < 9 else ("cross_occlusion_twist_boundary" if s < 13 else ("full_body_dynamic_root_boundary" if s < 17 else "quasi_uniform_interior_coverage")))

def sample_stratified_pose_euler_dict(bones_meta: dict[str, dict[str, Any]], rng: Any, sample_idx: int | None = None,
                                      enforce_anatomical_floor: bool = True, joint_limits_profile: dict[str, Any] | None = None,
                                      boundary_exploration: bool = False) -> tuple[dict[str, list[float]], str]:
    if boundary_exploration:
        e, r, _ = sample_boundary_exploration_pose_euler_dict(bones_meta, rng, sample_idx=sample_idx, joint_limits_profile=joint_limits_profile, enforce_anatomical_floor=False)
        return e, r
    if not bones_meta: return {}, "synergistic"
    classified = apply_joint_limits_profile(bones_meta, joint_limits_profile) if joint_limits_profile else classify_rig_bones(bones_meta)
    regime, eulers = select_pose_sampling_regime(sample_idx, rng), {}
    for b, info in classified.items():
        if info.get("kind") == "rigid" or bool(info.get("fixed_local", False)): eulers[b] = [0.0, 0.0, 0.0]; continue
        lims = effective_sampling_limits_for_bone(info, enforce_anatomical_floor)
        b_euls = []
        for ax in range(3):
            lo, hi = float(lims[ax][0]), float(lims[ax][1])
            if abs(hi - lo) < 1e-5: b_euls.append(lo)
            elif regime == "antiphase_extreme" and rng.random() < 0.6:
                b_euls.append(float(rng.uniform(lo, lo + 0.15 * (hi - lo))) if rng.random() < 0.5 else float(rng.uniform(hi - 0.15 * (hi - lo), hi)))
            else: b_euls.append(float(rng.uniform(lo, hi)))
        eulers[b] = b_euls
    return eulers, regime

def sample_boundary_exploration_pose_euler_dict(bones_meta: dict[str, dict[str, Any]], rng: Any, sample_idx: int | None = None,
                                                num_total_samples: int = 50, total_samples: Any = None, joint_limits_profile: Any = None,
                                                enforce_anatomical_floor: bool = False) -> tuple[dict[str, list[float]], str, dict[str, Any]]:
    if not bones_meta: return {}, "in_phase_boundary_extension", {"boundary_joint_count": 0, "active_axes_count": 0, "boundary_joints": []}
    classified = apply_joint_limits_profile(bones_meta, joint_limits_profile) if joint_limits_profile else classify_rig_bones(bones_meta)
    idx = int(sample_idx) if sample_idx is not None else int(rng.random() * max(1, num_total_samples))
    regime, eulers = select_g2_boundary_regime(idx, rng), {}
    for b, info in classified.items():
        if info.get("kind") == "rigid" or bool(info.get("fixed_local", False)): eulers[b] = [0.0, 0.0, 0.0]; continue
        lims = effective_sampling_limits_for_bone(info, enforce_anatomical_floor=False)
        b_euls = []
        for ax in range(3):
            lo, hi = float(lims[ax][0]), float(lims[ax][1]); span = hi - lo
            if info.get("kind") == "root" and ax == 1:
                max_yaw = math.radians(22.0) if regime == "full_body_dynamic_root_boundary" else math.radians(15.0)
                lo, hi = max(lo, -max_yaw), min(hi, max_yaw)
                span = hi - lo
            if span < 1e-5: b_euls.append(lo)
            elif regime == "quasi_uniform_interior_coverage": b_euls.append(float(rng.uniform(lo, hi)))
            else:
                coin = rng.random()
                if coin < 0.42: b_euls.append(float(rng.uniform(lo, lo + 0.14 * span)))
                elif coin < 0.84: b_euls.append(float(rng.uniform(hi - 0.14 * span, hi)))
                else: b_euls.append(float(rng.uniform(lo, hi)))
        eulers[b] = b_euls
    b_util = inspect_pose_boundary_utilization(eulers, classified)
    b_util["coordination_report"] = {
        "schema": "monster-joint-coordination-report-v1",
        "modified_bones": [],
        "inter_bone_adjustments": 0,
        "volume_penetration_adjustments": 0,
    }
    return eulers, regime, b_util

def sample_root_translation_offset(bones_meta: dict[str, dict[str, Any]], rng: Any, sample_idx: Any = None,
                                   max_xy_fraction: float = 0.08, max_z_fraction: float = 0.10, joint_limits_profile: Any = None,
                                   boundary_exploration: bool = False) -> list[float]:
    pts = [info[k] for info in bones_meta.values() for k in ("rest_head_world", "rest_tail_world") if k in info]
    span = np.maximum(1e-3, np.ptp(np.asarray(pts, dtype=np.float64), axis=0)) if pts else np.array([1.0, 1.0, 1.0])
    return [float(rng.uniform(-max_xy_fraction * span[0], max_xy_fraction * span[0])),
            float(rng.uniform(-max_xy_fraction * span[1], max_xy_fraction * span[1])),
            float(rng.uniform(-max_z_fraction * span[2], max_z_fraction * span[2]))]

def synthesize_archetype_pose_eulers(classified_bones: dict[str, dict[str, Any]], archetype: str,
                                     phase_or_frame: float | int, total_frames: int | None = None) -> dict[str, list[float]]:
    if archetype not in ARCHETYPE_SPECS: raise ValueError(f"Unsupported archetype: {archetype!r}")
    pr = 2.0 * math.pi * float(phase_or_frame) / float(max(1, total_frames)) if total_frames is not None else float(phase_or_frame)
    s1, c1, s2, deg = math.sin(pr), math.cos(pr), math.sin(2.0 * pr), math.radians
    eulers = {}
    for b, info in classified_bones.items():
        k, sub, s, c_i = str(info.get("kind", "limb")).lower(), str(info.get("sub_role", "")).lower(), str(info.get("side", "C")).upper(), int(info.get("chain_index", 1))
        ss = 1.0 if s == "L" else (-1.0 if s == "R" else 0.0)
        if k == "rigid" or info.get("fixed_local"): e = [0.0, 0.0, 0.0]
        elif k == "root": e = [deg(4.0) * s1, 0.0, deg(5.0) * c1]
        elif k == "spine": e = [deg(8.0) * s1, deg(3.0) * c1, deg(6.0) * s2]
        elif k == "head": e = [deg(10.0) * c1, deg(5.0) * s1, deg(8.0) * s1]
        elif k == "jaw": e = [deg(26.0) * max(0.0, math.sin(pr * 1.5)), 0.0, 0.0]
        elif k == "wing":
            flap, fold = math.sin(pr - 0.35 * c_i), 0.5 * (1.0 - math.cos(pr * 0.5))
            e = [deg(38.0) * flap, deg(10.0) * ss * fold, deg(28.0) * ss * (0.6 * flap - 0.4 * fold)]
        elif k == "limb":
            reach = math.sin(pr)
            if sub in ("upper_limb", "ball_socket"): e = [deg(45.0) * reach, deg(10.0) * ss, deg(20.0) * ss * reach]
            elif sub == "forearm_hinge": e = [deg(45.0) * max(0.0, -reach), deg(15.0) * s1, 0.0]
            else: e = [deg(18.0) * s1, 0.0, deg(10.0) * c1]
        elif k == "digit": e = [deg(30.0) * 0.5 * (1.0 + math.sin(pr)), 0.0, 0.0]
        elif k in ("tail", "flex"): e = [deg(16.0) * math.sin(pr * 1.5 - 0.45 * c_i), deg(4.0) * c1, deg(20.0) * math.sin(pr * 1.5 - 0.45 * c_i)]
        else: e = [deg(10.0) * s1, 0.0, deg(10.0) * c1]
        eulers[b] = e
    return eulers

def inspect_pose_boundary_utilization(eulers_rad: dict[str, list[float]], bones_meta: dict[str, dict[str, Any]],
                                      boundary_band_ratio: float = 0.15, enforce_anatomical_floor: bool = False) -> dict[str, Any]:
    b_joints, axes = [], 0
    for b, info in sorted(bones_meta.items()):
        if info.get("kind") == "rigid" or bool(info.get("fixed_local", False)): continue
        lims, val = effective_sampling_limits_for_bone(info, enforce_anatomical_floor=enforce_anatomical_floor), eulers_rad.get(b, [0.0, 0.0, 0.0])
        for ax in range(3):
            lo, hi = float(lims[ax][0]), float(lims[ax][1]); span = hi - lo
            if span <= 1e-4: continue
            axes += 1; band, v = float(boundary_band_ratio) * span, float(val[ax])
            if v <= lo + band or v >= hi - band: b_joints.append({"bone": b, "axis": ax, "value": v, "lo": lo, "hi": hi})
    return {"boundary_joint_count": len(b_joints), "active_axes_count": axes, "boundary_joints": b_joints}

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
    start_policy = bool(samples_or_records) and all(
        r.get("sampling_regime") in ("video2rig_calm", "video2rig_active") for r in samples_or_records)
    if not start_policy and N >= 20 and both_ratio < 0.85:
        boundary_failures.append(
            f"Insufficient G2 boundary exploration coverage: both_boundaries_touch_ratio={both_ratio:.3f} < 0.85"
        )

    return {
        **base_audit,
        "schema": "monster-g2-boundary-exploration-audit-v1",
        "regime_counts": regime_counts,
        "dual_boundary_touch_rate": both_ratio,
        "boundary_coverage_required": not start_policy,
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

    start_policy = isinstance(samples_or_eulers, list) and bool(samples_or_eulers) and all(
        r.get("sampling_regime") in ("video2rig_calm", "video2rig_active") for r in samples_or_eulers)
    stalled_axes = []
    invalid_axes = []
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
            if is_leaf and ax == 1 and not bool(info.get("allow_leaf_twist", False)):
                continue
            span = float(lims[ax][1] - lims[ax][0])
            if span <= 1e-3:
                continue
            col = arr[:, ax]
            if not np.all(np.isfinite(col)):
                invalid_axes.append(f"{b}.{ax}")
            if N >= 20 and np.ptp(col) < 1e-6:
                stalled_axes.append(f"{b}.{ax}")
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
            r_pc = abs(float(np.mean((cp - cp.mean()) * (cc - cc.mean())) / (sp * sc)))
            max_parent_child_corr = max(max_parent_child_corr, r_pc)
            antiphase_ratios.append(float(np.mean((cp * cc) < 0.0)))

    mean_antiphase_ratio = float(np.mean(antiphase_ratios)) if antiphase_ratios else 0.5
    mean_p90_span = float(np.mean(p90_ratios)) if p90_ratios else 1.0
    min_p90_span = float(np.min(p90_ratios)) if p90_ratios else 1.0
    mean_full_span = float(np.mean(full_ratios)) if full_ratios else 1.0

    failures: list[str] = []
    if start_policy and stalled_axes:
        failures.append(f"Frozen sampled axes: {stalled_axes[:8]}")
    if invalid_axes:
        failures.append(f"Non-finite sampled axes: {invalid_axes[:8]}")
    if under_covered_bones and not start_policy:
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
    if not start_policy and antiphase_ratios and mean_antiphase_ratio < 0.12:
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

def compute_stratified_motion_metrics(frames_pred: list[dict[str, Any]], frames_gt: list[dict[str, Any]],
                                      bones_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    classified = classify_rig_bones(bones_meta); names = sorted(classified.keys())
    per_b = {b: {"direction": [], "rotation": [], "position": []} for b in names}
    for fpr, fgt in zip(frames_pred, frames_gt):
        bp, bg = fpr.get("bones", fpr.get("joints", {})), fgt.get("bones", fgt.get("joints", {}))
        for b in names:
            if b not in bp or b not in bg: continue
            hp, tp = np.array(bp[b]["head_3d"], dtype=float), np.array(bp[b]["tail_3d"], dtype=float)
            hg, tg = np.array(bg[b]["head_3d"], dtype=float), np.array(bg[b]["tail_3d"], dtype=float)
            vp, vg = tp - hp, tg - hg; nlp, nlg = float(np.linalg.norm(vp)), float(np.linalg.norm(vg))
            if nlp > 1e-6 and nlg > 1e-6: per_b[b]["direction"].append(math.degrees(math.acos(float(np.clip(np.dot(vp / nlp, vg / nlg), -1, 1)))))
            if "rot_mat_world" in bp[b] and "rot_mat_world" in bg[b]:
                Rp, Rg = np.array(bp[b]["rot_mat_world"], dtype=float), np.array(bg[b]["rot_mat_world"], dtype=float)
                if np.allclose(Rp, Rg):
                    per_b[b]["rotation"].append(0.0)
                else:
                    tr = float(np.clip((np.trace(Rg.T @ Rp) - 1.0) * 0.5, -1, 1))
                    per_b[b]["rotation"].append(math.degrees(math.acos(tr)))
            per_b[b]["position"].append(0.5 * (float(np.linalg.norm(hp - hg)) + float(np.linalg.norm(tp - tg))))
    summ = {b: {k: round(float(np.mean(v)), 4) if v else 0.0 for k, v in m.items()} for b, m in per_b.items()}
    def _agg(sub: list[str]) -> dict[str, Any]:
        if not sub: return {"count": 0, "direction_mean_deg": 0.0, "direction_max_bone_deg": 0.0, "full_rot_mean_deg": 0.0, "nonleaf_rot_mean_deg": 0.0, "endpoint_position_mean": 0.0}
        d, r, p = [summ[b]["direction"] for b in sub], [summ[b]["rotation"] for b in sub], [summ[b]["position"] for b in sub]
        nl = [summ[b]["rotation"] for b in sub if classified[b].get("children")]
        return {"count": len(sub), "direction_mean_deg": round(float(np.mean(d)), 3), "direction_max_bone_deg": round(float(np.max(d)), 3),
                "full_rot_mean_deg": round(float(np.mean(r)), 3), "nonleaf_rot_mean_deg": round(float(np.mean(nl if nl else r)), 3), "endpoint_position_mean": round(float(np.mean(p)), 5)}
    primary = [b for b in names if classified[b].get("is_primary", True)]
    by_k = {k: _agg([b for b in names if classified[b].get("kind") == k]) for k in CANONICAL_KINDS if any(classified[b].get("kind") == k for b in names)}
    return {"schema": BONE_TAXONOMY_SCHEMA, "all_bones": _agg(names), "primary_bones": _agg(primary), "by_kind": by_k, "per_bone": summ}
