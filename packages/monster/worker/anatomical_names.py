"""V3 anatomical names. V2 names remain readable; new Blender exports use v3 only."""

import re
import numpy as np

LEG_PLANS = {
    "biped": ("Thigh", "Shin", "Foot", "Toes"),
    "hind": ("Thigh", "Shin", "HindPaw", "HindToes"),
    "front": ("Scapula", "UpperArm", "Forearm", "FrontPaw", "FrontToes"),
    "arm": ("Shoulder", "Arm", "LowerArm", "Hand"),
    "wing": ("WingShoulder", "WingUpper", "WingForearm", "WingHand"),
    "many": ("Coxa", "Femur", "Tibia", "Tarsus"),
}
_V3 = re.compile(
    r"^(Root|Hips|Spine[1-9][0-9]*|Neck[1-9][0-9]*|Head|Jaw|Tail[1-9][0-9]*|"
    r"Thigh|Shin|Foot|Toes|HindPaw|HindToes|Scapula|UpperArm|Forearm|FrontPaw|FrontToes|"
    r"Shoulder|Arm|LowerArm|Hand|WingShoulder|WingUpper|WingForearm|WingHand|"
    r"(?:Coxa|Femur|Tibia|Tarsus)[1-9][0-9]*|(?:Finger|Toe|Pair|Ear|Flex)[1-9][0-9]*_[1-9][0-9]*|"
    r"Limb[1-9][0-9]*_[1-9][0-9]*|Rigid[1-9][0-9]*|Weapon|Prop_[A-Za-z0-9_]+)(?:\.([LR]))?$"
)


def parse_anatomical_name(name):
    m = _V3.fullmatch(name)
    if not m:
        return None
    stem, side = m.groups()
    lateral = (
        stem in sum((tuple(v) for v in LEG_PLANS.values()), ())
        or stem == "Weapon"
        or stem.startswith(
            ("Finger", "Toe", "Coxa", "Femur", "Tibia", "Tarsus", "Pair")
        )
    )
    if lateral and not side:
        return None
    if (
        stem.startswith(
            ("Root", "Hips", "Spine", "Neck", "Head", "Jaw", "Tail", "Prop_")
        )
        and side
    ):
        return None
    kind = "limb"
    if stem in ("Root", "Hips") or stem.startswith("Prop_"):
        kind = "root"
    elif stem.startswith("Spine"):
        kind = "spine"
    elif stem == "Head" or stem.startswith("Neck"):
        kind = "head"
    elif stem == "Jaw":
        kind = "jaw"
    elif stem.startswith("Tail"):
        kind = "tail"
    elif stem.startswith(("Finger", "Toe")) and stem != "Toes":
        kind = "digit"
    elif stem.startswith("Wing"):
        kind = "wing"
    elif stem.startswith(("Ear", "Flex")):
        kind = "flex"
    elif stem.startswith("Rigid") or stem == "Weapon":
        kind = "rigid"
    nums = re.findall(r"\d+", stem)
    return {
        "kind": kind,
        "side": side or "C",
        "group_id": nums[0] if nums else "0",
        "chain_index": int(nums[-1]) if nums else 1,
        "anatomical_role": stem,
        "schema_version": 3,
    }


def anatomical_names(meta, height):
    """Name complete chains by topology, contact height and anterior/posterior position.

    Recognized v3 names survive unchanged. Extra chains get explicit Pair/Limb names,
    so a six-legged or asymmetric animal is never squeezed into a biped template.
    """
    out, used = {}, set()
    children = {n: [] for n in meta}
    for n, b in meta.items():
        if b.get("parent") in children:
            children[b["parent"]].append(n)
    order = sorted(meta, key=lambda n: (meta[n].get("depth", 0), n))
    floor = min(float(b.get("rest_tail_world", [0, 0, 0])[2]) for b in meta.values())
    chains = []
    for n in order:
        b = meta[n]
        # Paired anatomical roles require a side; midline appendages keep Limb names.
        if b.get("side", "C") not in ("L", "R"):
            continue
        p = b.get("parent")
        if b["kind"] not in ("limb", "wing") or (
            p in meta
            and meta[p]["kind"] == b["kind"]
            and meta[p].get("side") == b.get("side")
        ):
            continue
        chain = [n]
        while True:
            ch = [c for c in children[chain[-1]] if meta[c]["kind"] == b["kind"]]
            if len(ch) != 1:
                break
            chain.append(ch[0])
        end = meta[chain[-1]].get("rest_tail_world", [0, 0, 1])
        chains.append(
            (
                chain,
                float(end[2]) < floor + 0.22 * height,
                float(meta[n].get("rest_head_world", [0, 0, 0])[1]),
            )
        )
    contacts = [c for c in chains if c[1]]
    y_mid = float(np.median([c[2] for c in contacts])) if contacts else 0
    chain_roles = {}
    many_number = {"L": 0, "R": 0, "C": 0}
    for chain, contact, y in sorted(chains, key=lambda c: (c[2], c[0][0])):
        n = chain[0]
        side = meta[n]["side"]
        if meta[n]["kind"] == "wing":
            plan = "wing"
        elif contact and len(contacts) >= 6:
            plan = "many"
        elif contact and len(contacts) >= 4:
            plan = "front" if y < y_mid else "hind"
        elif contact:
            plan = "biped"
        else:
            plan = "arm"
        names = LEG_PLANS[plan]
        if plan == "many":
            many_number[side] += 1
        # Short arm/front chains omit the proximal shoulder socket; distal identity stays stable.
        offset = (
            max(0, len(names) - len(chain)) if plan in ("arm", "front", "wing") else 0
        )
        for i, bone in enumerate(chain):
            if i + offset < len(names):
                role = names[i + offset]
                if plan == "many":
                    role += str(many_number[side])
                chain_roles[bone] = role
    counters = {}
    for n in order:
        b = meta[n]
        kind = b["kind"]
        side = b.get("side", "C")
        parsed = parse_anatomical_name(n)
        if parsed:
            candidate = n
        else:
            key = (kind, side)
            counters[key] = counters.get(key, 0) + 1
            i = counters[key]
            suffix = "." + side if side in ("L", "R") else ""
            if kind == "root":
                role = "Root" if not b.get("parent") else "Hips"
            elif kind == "spine":
                role = f"Spine{i}"
            elif kind == "head":
                role = (
                    "Head"
                    if not any(meta[c]["kind"] == "head" for c in children[n])
                    else f"Neck{i}"
                )
            elif kind == "jaw":
                role = "Jaw"
            elif kind == "tail":
                role = f"Tail{i}"
            elif kind == "digit":
                p = b.get("parent")
                ancestors = []
                while p in meta:
                    ancestors.append(p)
                    p = meta[p].get("parent")
                foot = any(
                    any(
                        t in (chain_roles.get(a, a))
                        for t in ("Paw", "Foot", "Toes", "Tarsus")
                    )
                    for a in ancestors
                )
                role = (
                    ("Toe" if foot else "Finger")
                    + f"{max(1, int(b.get('group_id', 1)))}_{max(1, int(b.get('chain_index', 1)))}"
                )
            elif kind == "rigid":
                role = (
                    "Weapon"
                    if "weapon" in n.lower() or b.get("weapon")
                    else f"Rigid{i}"
                )
            elif kind == "flex":
                role = (
                    ("Ear" if "ear" in n.lower() else "Flex")
                    + f"{max(1, int(b.get('group_id', i)))}_{max(1, int(b.get('chain_index', 1)))}"
                )
            else:
                role = chain_roles.get(n, f"Pair{i}_1" if suffix else f"Limb{i}_1")
            candidate = role + (
                suffix if kind in ("limb", "wing", "digit", "rigid", "flex") else ""
            )
        if candidate in used:
            i = 1
            while (
                f"{'Pair' if side != 'C' else 'Limb'}{i}_1"
                + ("." + side if side != "C" else "")
                in used
            ):
                i += 1
            candidate = f"{'Pair' if side != 'C' else 'Limb'}{i}_1" + (
                "." + side if side != "C" else ""
            )
        used.add(candidate)
        out[n] = candidate
    return out
