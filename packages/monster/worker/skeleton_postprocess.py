"""Skeleton post-processing and analysis for MonsterStudio.

Self-contained module based on video2rig autorig (skeleton.py and unirig.py).
Constructs a clean joint-tree from input skeletons (SkinTokens or mesh centerlines),
analyzes body topology (quadruped, biped, centaur, winged), detects spines, limbs,
and tails (including curled tails), and assigns canonical v3 anatomical names.
"""

from __future__ import annotations

from copy import deepcopy
import heapq
import math
from pathlib import Path
import re
import sys
from typing import Any

import networkx as nx
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

# Auto-add local blender site-packages if available
_EXTRA_SITE = "/Users/bugyaluwang/.local/share/codex-integrations/game-studio-local/blender-python-5.2.3/lib/python3.13/site-packages"
if _EXTRA_SITE not in sys.path and Path(_EXTRA_SITE).is_dir():
    sys.path.append(_EXTRA_SITE)

AXES = {
    "-Y": np.array([0.0, -1.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "+X": np.array([1.0, 0.0, 0.0]),
}
SIDES = {1: ".L", -1: ".R"}

LEG_PLANS = {
    "biped": ("Thigh", "Shin", "Foot", "Toes"),
    "hind": ("Thigh", "Shin", "HindPaw", "HindToes"),
    "front": ("Scapula", "UpperArm", "Forearm", "FrontPaw", "FrontToes"),
    "arm": ("Shoulder", "Arm", "LowerArm", "Hand"),
    "wing": ("WingShoulder", "WingUpper", "WingForearm", "WingHand"),
    "many": ("Coxa", "Femur", "Tibia", "Tarsus"),
}

PLANS = {
    "hind": (None, ["UpperLeg", "LowerLeg", "Foot", "Toe"], [0.40, 0.75, 0.90]),
    "front": ("Shoulder", ["Arm", "LowerArm", "Hand"], [0.42, 0.80]),
    "biped": (None, ["UpperLeg", "LowerLeg", "Foot", "Toe"], [0.45, 0.82, 0.92]),
    "arm": ("Shoulder", ["Arm", "LowerArm", "Hand"], [0.42, 0.80]),
    "wing": ("WingShoulder", ["WingUpper", "WingForearm", "WingHand"], [0.30, 0.62]),
}

_V3 = re.compile(
    r"^(Root|Hips|Pelvis|Spine[1-9][0-9]*|Neck[1-9][0-9]*|Head|Jaw|Tail[0-9]*|"
    r"UpperLeg|LowerLeg|Thigh|Shin|Foot|Toes|Toe|HindPaw|HindToes|"
    r"Scapula|UpperArm|Forearm|FrontPaw|FrontToes|"
    r"Shoulder|Arm|LowerArm|Hand|WingShoulder|WingUpper|WingForearm|WingHand|Wing[0-9]*|"
    r"(?:Coxa|Femur|Tibia|Tarsus)[1-9][0-9]*|(?:Finger|Toe|Pair|Ear|Flex|HeadX|NeckX)[0-9]*_[0-9]*|"
    r"Limb[0-9]*_[0-9]*|Pair[0-9]*|Limb[0-9]*|HeadX[0-9]*|NeckX[0-9]*|Rigid[0-9]*|Weapon|Prop_[A-Za-z0-9_]+)(?:\.([LR]))?$"
)


def parse_anatomical_name(name: str) -> dict[str, Any] | None:
    """Parse a bone name into structured v3 anatomical metadata."""
    if not isinstance(name, str):
        return None
    m = _V3.fullmatch(name)
    if not m:
        return None
    stem, side = m.groups()
    lateral_stems = sum((tuple(v) for v in LEG_PLANS.values()), ()) + (
        "UpperLeg", "LowerLeg", "Toe", "Weapon"
    )
    lateral = (
        stem in lateral_stems
        or stem.startswith(("Finger", "Toe", "Coxa", "Femur", "Tibia", "Tarsus", "Pair"))
    )
    if lateral and not side and not stem.startswith(("Pair", "Limb", "Wing")):
        return None
    if (
        stem.startswith(
            ("Root", "Hips", "Pelvis", "Spine", "Neck", "Head", "Jaw", "Tail", "Prop_")
        )
        and not stem.startswith(("HeadX", "NeckX"))
        and side
    ):
        return None

    kind = "limb"
    if stem in ("Root", "Hips", "Pelvis") or stem.startswith("Prop_"):
        kind = "root_Hips" if stem == "Root" else "pelvis"
    elif stem.startswith("Spine"):
        kind = "spine"
    elif stem == "Head" or stem.startswith("Neck"):
        kind = "head"
    elif stem == "Jaw":
        kind = "jaw"
    elif stem.startswith("Tail"):
        kind = "tail"
    elif stem.startswith(("Finger", "Toe")) and stem not in ("Toes", "Toe"):
        kind = "digit"
    elif stem.startswith("Wing"):
        kind = "wing"
    elif stem.startswith(("Ear", "Flex", "HeadX", "NeckX")):
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


def anatomical_names(meta: dict[str, Any], height: float) -> dict[str, str]:
    """Name complete chains by topology, contact height and anterior/posterior position."""
    out = {}
    used = set()
    children = {n: [] for n in meta}
    for n, b in meta.items():
        if b.get("parent") in children:
            children[b["parent"]].append(n)
    order = sorted(meta, key=lambda n: (meta[n].get("depth", 0), n))
    floor = min(float(b.get("rest_tail_world", [0, 0, 0])[2]) for b in meta.values())
    chains = []
    for n in order:
        b = meta[n]
        if b.get("side", "C") not in ("L", "R"):
            continue
        p = b.get("parent")
        if b.get("kind") not in ("limb", "wing") or (
            p in meta
            and meta[p].get("kind") == b.get("kind")
            and meta[p].get("side") == b.get("side")
        ):
            continue
        chain = [n]
        while True:
            ch = [c for c in children[chain[-1]] if meta[c].get("kind") == b.get("kind")]
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
    y_mid = float(np.median([c[2] for c in contacts])) if contacts else 0.0
    chain_roles = {}
    many_number = {"L": 0, "R": 0, "C": 0}
    for chain, contact, y in sorted(chains, key=lambda c: (c[2], c[0][0])):
        n = chain[0]
        side = meta[n].get("side", "C")
        if meta[n].get("kind") == "wing":
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
        offset = max(0, len(names) - len(chain)) if plan in ("arm", "front", "wing") else 0
        for i, bone in enumerate(chain):
            if i + offset < len(names):
                role = names[i + offset]
                if plan == "many":
                    role += str(many_number[side])
                chain_roles[bone] = role
    counters: dict[tuple[str, str], int] = {}
    for n in order:
        b = meta[n]
        kind = b.get("kind", "limb")
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
                    if not any(meta[c].get("kind") == "head" for c in children[n])
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
                    any(t in (chain_roles.get(a, a)) for t in ("Paw", "Foot", "Toes", "Tarsus"))
                    for a in ancestors
                )
                role = (
                    ("Toe" if foot else "Finger")
                    + f"{max(1, int(b.get('group_id', 1)))}_{max(1, int(b.get('chain_index', 1)))}"
                )
            elif kind == "rigid":
                role = "Weapon" if "weapon" in n.lower() or b.get("weapon") else f"Rigid{i}"
            elif kind == "flex":
                role = (
                    ("Ear" if "ear" in n.lower() else "Flex")
                    + f"{max(1, int(b.get('group_id', i)))}_{max(1, int(b.get('chain_index', 1)))}"
                )
            else:
                role = chain_roles.get(n, f"Pair{i}_1" if suffix else f"Limb{i}_1")
            candidate = role + (suffix if kind in ("limb", "wing", "digit", "rigid", "flex") else "")
        if candidate in used:
            i = 1
            while (
                f"{'Pair' if side != 'C' else 'Limb'}{i}_1" + ("." + side if side != "C" else "")
                in used
            ):
                i += 1
            candidate = f"{'Pair' if side != 'C' else 'Limb'}{i}_1" + (
                "." + side if side != "C" else ""
            )
        used.add(candidate)
        out[n] = candidate
    return out


# Make anatomical_names importable via `import anatomical_names`
sys.modules.setdefault("anatomical_names", sys.modules[__name__])


def rest_coords(
    posed_vertices: np.ndarray,
    skin_indices: np.ndarray,
    skin_weights: np.ndarray,
    rest_matrices: np.ndarray,
    posed_matrices: np.ndarray,
) -> np.ndarray:
    """Invert the blended affine transform per vertex (inverse LBS, not blended inverses)."""
    transforms = np.asarray(posed_matrices) @ np.linalg.inv(rest_matrices)
    blend = (
        np.asarray(skin_weights)[..., None, None]
        * transforms[np.asarray(skin_indices)]
    ).sum(axis=1)
    return (
        np.linalg.inv(blend)
        @ np.pad(posed_vertices, ((0, 0), (0, 1)), constant_values=1)[..., None]
    )[..., :3, 0]


class Shell:
    """Watertight voxel representation of creature geometry."""

    def __init__(
        self,
        surface: np.ndarray | None = None,
        solid: np.ndarray | None = None,
        origin: np.ndarray | None = None,
        step: float = 0.01,
        distance: np.ndarray | None = None,
        verts: np.ndarray | None = None,
        tris: np.ndarray | None = None,
        remesh_verts: np.ndarray | None = None,
        remesh_faces: np.ndarray | None = None,
    ):
        # Support Shell(solid, origin, step)
        if (
            surface is not None
            and solid is not None
            and isinstance(origin, (int, float))
            and isinstance(surface, np.ndarray)
            and surface.ndim == 3
        ):
            solid, origin, step = surface, solid, float(origin)
            surface = None

        self.surface = surface
        self.solid = solid if solid is not None else np.zeros((1, 1, 1), bool)
        self.origin = np.asarray(origin if origin is not None else [0.0, 0.0, 0.0], float)
        self.step = float(step)
        self.distance = distance
        self.verts = verts
        self.tris = tris
        self.remesh_verts = remesh_verts
        self.remesh_faces = remesh_faces
        self._kd: cKDTree | None = None

    @property
    def vox(self) -> np.ndarray:
        ijk = np.argwhere(self.solid)
        return self.origin + (ijk + 0.5) * self.step

    def indices(self, points: np.ndarray) -> np.ndarray:
        return np.rint((np.asarray(points) - self.origin) / self.step).astype(int)

    def contains(self, points: np.ndarray) -> np.ndarray:
        i = self.indices(points)
        good = ((i >= 0) & (i < np.array(self.solid.shape))).all(axis=-1)
        answer = np.zeros(i.shape[:-1], bool)
        answer[good] = self.solid[tuple(i[good].T)]
        return answer

    def radius(self, points: np.ndarray) -> np.ndarray:
        if self.distance is None:
            self.distance = ndimage.distance_transform_edt(self.solid)
        i = np.clip(self.indices(points), 0, np.array(self.solid.shape) - 1)
        return self.distance[tuple(i.T)] * self.step

    def distance_inside(self, points: np.ndarray) -> np.ndarray:
        if self.distance is None:
            self.distance = ndimage.distance_transform_edt(self.solid)
        points = np.atleast_2d(points)
        ijk = np.floor((points - self.origin) / self.step).astype(int)
        ok = np.all((ijk >= 0) & (ijk < np.array(self.solid.shape)), axis=1)
        d = np.zeros(len(points), float)
        d[ok] = self.distance[tuple(ijk[ok].T)] * self.step
        return d

    def off_skin(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(points)
        inside = self.contains(points)
        if self._kd is None:
            ijk = np.argwhere(self.solid)
            if len(ijk) > 0:
                vox_pts = self.origin + (ijk + 0.5) * self.step
                if len(vox_pts) > 60000:
                    vox_pts = vox_pts[:: len(vox_pts) // 30000]
                self._kd = cKDTree(vox_pts)
        if self._kd is not None:
            d, _ = self._kd.query(points)
            return np.where(inside, 0.0, d)
        return np.where(inside, 0.0, 99.0)

    def snap_inside(self, head: np.ndarray, tail: np.ndarray, max_ratio: float = 0.7) -> np.ndarray:
        a, b = np.asarray(head, float), np.asarray(tail, float)
        for s in np.linspace(1.0, 1.0 - max_ratio, 15):
            p = a + (b - a) * s
            if bool(np.asarray(self.contains(p)).flat[0]):
                return p
        return a + (b - a) * (1.0 - max_ratio)

    def mirror_overlap(self, axis: int, plane: float | None = None) -> float:
        occupied = np.argwhere(self.solid)
        if not len(occupied):
            return 0.0
        coordinate = (
            (occupied[:, axis].min() + occupied[:, axis].max()) / 2
            if plane is None
            else (plane - self.origin[axis]) / self.step
        )
        reflected = np.rint(2 * coordinate - np.arange(self.solid.shape[axis])).astype(int)
        valid = (reflected >= 0) & (reflected < self.solid.shape[axis])
        shape = [1] * 3
        shape[axis] = len(valid)
        mirror = np.take(
            self.solid, np.clip(reflected, 0, self.solid.shape[axis] - 1), axis=axis
        ) & valid.reshape(shape)
        intersection = np.count_nonzero(mirror & self.solid)
        count = np.count_nonzero(self.solid)
        return float(intersection / max(1, 2 * count - intersection))

    def save(self, path: str | Path, vertices: np.ndarray | None = None, triangles: np.ndarray | None = None, **extra):
        v = vertices if vertices is not None else self.verts
        f = triangles if triangles is not None else self.tris
        payload = dict(
            vertices=v,
            triangles=f,
            verts=(
                self.remesh_verts
                if self.remesh_verts is not None
                else (np.asarray(v, dtype=np.float32) if v is not None else None)
            ),
            faces=(
                self.remesh_faces
                if self.remesh_faces is not None
                else (np.asarray(f, dtype=np.int32) if f is not None else None)
            ),
            skin_verts=np.asarray(v, dtype=np.float32) if v is not None else None,
            skin_faces=np.asarray(f, dtype=np.int32) if f is not None else None,
            thin=self.surface if self.surface is not None else self.solid,
            solid=self.solid,
            origin=self.origin,
            voxel=self.step,
        )
        payload = {k: val for k, val in payload.items() if val is not None}
        payload.update(extra)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> Shell:
        d = np.load(path)
        solid = d["solid"]
        origin = d["origin"]
        step = float(d["voxel"] if "voxel" in d else d.get("step", 0.01))
        verts = d.get("verts", d.get("vertices", None))
        tris = d.get("faces", d.get("triangles", None))
        surface = d.get("thin", None)
        dist = ndimage.distance_transform_edt(solid)
        return cls(
            surface=surface,
            solid=solid,
            origin=origin,
            step=step,
            distance=dist,
            verts=verts,
            tris=tris,
        )


def voxelize(V: np.ndarray, F: np.ndarray | None = None, step: float | None = None, voxel: float | None = None) -> Shell:
    """Conservative triangle sampling plus closing/flood fill, returning Shell."""
    V = np.asarray(V, dtype=np.float64)
    height = max(float(np.ptp(V[:, 2])), 1e-4)
    size = float(step or voxel or (0.01 * height))
    lo = V.min(0) - 2 * size
    dims = np.ceil((V.max(0) + 2 * size - lo) / size).astype(int) + 1
    grid = np.zeros(dims, bool)

    if F is None or len(F) == 0:
        ijk = np.floor((V - lo) / size).astype(int)
        valid = (ijk >= 0).all(1) & (ijk < np.array(dims)).all(1)
        grid[ijk[valid, 0], ijk[valid, 1], ijk[valid, 2]] = True
        solid = ndimage.binary_fill_holes(grid)
        return Shell(surface=grid, solid=solid, origin=lo, step=size, verts=V, tris=np.empty((0, 3), int))

    F = np.asarray(F, dtype=int)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    longest = np.maximum(
        np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - a, axis=1)),
        np.linalg.norm(c - b, axis=1),
    )
    n = np.clip(np.ceil(longest / (size * 0.5)).astype(int), 1, 64)
    for k in np.unique(n):
        sel = n == k
        u, v = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = (u + v) <= k
        u, v = u[keep] / k, v[keep] / k
        P = (
            a[sel, None] * (1 - u - v)[None, :, None]
            + b[sel, None] * u[None, :, None]
            + c[sel, None] * v[None, :, None]
        ).reshape(-1, 3)
        ijk = np.floor((P - lo) / size).astype(int)
        valid = (ijk >= 0).all(1) & (ijk < np.array(dims)).all(1)
        grid[ijk[valid, 0], ijk[valid, 1], ijk[valid, 2]] = True

    closed = ndimage.binary_closing(grid, iterations=1)
    solid = ndimage.binary_fill_holes(closed) | grid
    dist = ndimage.distance_transform_edt(solid)
    return Shell(surface=grid, solid=solid, origin=lo, step=size, distance=dist, verts=V, tris=F)


class FacingResult(dict):
    """Result object from find_facing that acts as a dictionary and compares/prints as forward string."""

    def __str__(self) -> str:
        return str(self.get("forward", "-Y"))

    def __repr__(self) -> str:
        return str(self.get("forward", "-Y"))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.get("forward") == other
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.get("forward", "-Y"))


def find_facing(
    vertices: np.ndarray,
    triangles: np.ndarray | None = None,
    shell: Shell | None = None,
    bones: Any = None,
    log: Any = None,
) -> FacingResult:
    """Find creature forward orientation from mirror symmetry, toe directions and protrusion."""
    v = np.asarray(vertices, float)
    H = max(float(np.ptp(v[:, 2])), 1e-4)
    mid = np.median(v, axis=0)
    tree = cKDTree(v)
    scores = []
    for axis in (0, 1):
        mirror = v.copy()
        mirror[:, axis] = 2 * mid[axis] - mirror[:, axis]
        scores.append(float(np.mean(tree.query(mirror)[0]) / H))
    if shell is None:
        shell = voxelize(v, np.empty((0, 3), int) if triangles is None else triangles, 0.01 * H)
    overlap = [shell.mirror_overlap(axis) for axis in (0, 1)]
    lateral = int(np.argmax(overlap))
    forward = 1 - lateral

    # If bones are available, toe direction and head orientation are decisive
    if bones is not None:
        bone_list = [{"name": k, **val} if "name" not in val else val for k, val in bones.items()] if isinstance(bones, dict) else bones
        floor = float(v[:, 2].min())
        distal_fwd = []
        for b in bone_list:
            h = np.asarray(b.get("head", b.get("rest_head_world", [0, 0, 0])), float)
            t = np.asarray(b.get("tail", b.get("rest_tail_world", [0, 0, 0])), float)
            children = b.get("children", [])
            is_leaf = (len(children) == 0) if isinstance(children, (list, tuple)) else True
            if (min(h[2], t[2]) - floor < 0.15 * H) or (is_leaf and h[2] - floor < 0.25 * H):
                distal_fwd.append(t[forward] - h[forward])
        if distal_fwd and abs(np.mean(distal_fwd)) > 1e-4:
            sign = 1 if np.mean(distal_fwd) > 0 else -1
            fwd_str = ("+" if sign > 0 else "-") + "XY"[forward]
            return FacingResult({
                "forward": fwd_str,
                "yaw_to_minus_y_deg": float((-90 - math.degrees(math.atan2(sign if forward == 1 else 0, sign if forward == 0 else 0)) + 180) % 360 - 180),
                "mirror_error": float(min(scores)),
                "symmetry": float(overlap[lateral]),
                "mirror_overlap_by_axis": {"X": overlap[0], "Y": overlap[1]},
                "lateral_axis": "XY"[lateral],
            })

    z = (v[:, 2] - v[:, 2].min()) / H
    top = v[z >= 0.95, forward]
    feet = v[z <= 0.04, forward]
    ankle = v[(z >= 0.06) & (z <= 0.15), forward]
    head = np.median(top) - mid[forward] if len(top) else 0.0
    toe = (np.mean(feet) - np.mean(ankle)) if len(feet) and len(ankle) else 0.0

    bias = (head + 2.0 * toe) if (len(feet) and len(ankle)) else (head + toe)
    sign = 1 if bias >= 0 else -1
    fwd_str = ("+" if sign > 0 else "-") + "XY"[forward]
    return FacingResult({
        "forward": fwd_str,
        "yaw_to_minus_y_deg": float((-90 - math.degrees(math.atan2(sign if forward == 1 else 0, sign if forward == 0 else 0)) + 180) % 360 - 180),
        "mirror_error": float(min(scores)),
        "symmetry": float(overlap[lateral]),
        "mirror_overlap_by_axis": {"X": overlap[0], "Y": overlap[1]},
        "lateral_axis": "XY"[lateral],
    })


def _footprints(shell: Shell, *, thin_legs: bool = False) -> list[np.ndarray]:
    occupied = np.argwhere(shell.solid)
    if not len(occupied):
        return []
    floor = occupied[:, 2].min()
    slab = shell.solid[:, :, floor : floor + 3].any(2)
    band_depth = 8
    if thin_legs:
        band_depth = max(3, int(np.ceil(0.08 * (np.ptp(occupied[:, 2]) + 1))))
        labels, count = ndimage.label(
            shell.solid[:, :, floor : floor + band_depth],
            ndimage.generate_binary_structure(3, 3),
        )
        labels = labels[:, :, :3].max(2)
    else:
        if shell.distance is None:
            shell.distance = ndimage.distance_transform_edt(shell.solid)
        slab &= (shell.distance[:, :, floor : floor + 8] >= 1.8).any(2)
        labels, count = ndimage.label(slab)
    feet = []
    for k in range(1, count + 1):
        xy = np.argwhere(labels == k)
        if len(xy) < (1 if thin_legs else 3):
            continue
        center = np.rint(np.median(xy, axis=0)).astype(int)
        if shell.distance is None:
            shell.distance = ndimage.distance_transform_edt(shell.solid)
        column = shell.distance[
            max(0, center[0] - 1) : center[0] + 2,
            max(0, center[1] - 1) : center[1] + 2,
            floor : floor + max(8, band_depth),
        ]
        if column.size and column.max() >= (0.9 if thin_legs else 1.8):
            feet.append(shell.origin + np.r_[center, floor + 1] * shell.step)
    return feet


def _resample(path: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    return np.array(
        [np.interp(np.asarray(fractions) * d[-1], d, path[:, i]) for i in range(3)]
    ).T


def _interior_paths(shell: Shell, root: np.ndarray, ends: list[np.ndarray]) -> list[np.ndarray]:
    occupied = np.argwhere(shell.solid)
    if not len(occupied) or not len(ends):
        return []
    ids = np.full(shell.solid.shape, -1, dtype=np.int32)
    ids[tuple(occupied.T)] = np.arange(len(occupied))
    if shell.distance is None:
        shell.distance = ndimage.distance_transform_edt(shell.solid)
    depth = shell.distance[tuple(occupied.T)]
    rows, columns, costs = [], [], []
    for offset in np.ndindex(3, 3, 3):
        delta = np.asarray(offset) - 1
        if tuple(delta) <= (0, 0, 0):
            continue
        neighbors = occupied + delta
        valid = ((neighbors >= 0) & (neighbors < shell.solid.shape)).all(1)
        a = np.flatnonzero(valid)
        b = ids[tuple(neighbors[valid].T)]
        connected = b >= 0
        a, b = a[connected], b[connected]
        rows.append(a)
        columns.append(b)
        costs.append(np.linalg.norm(delta) * (1 + 3.0 / (0.5 * (depth[a] + depth[b]) + 0.5)))
    rows, columns, costs = map(np.concatenate, (rows, columns, costs))
    graph = coo_matrix(
        (np.r_[costs, costs], (np.r_[rows, columns], np.r_[columns, rows])),
        shape=(len(occupied), len(occupied)),
    ).tocsr()
    nearest = cKDTree(occupied)
    start = int(nearest.query(shell.indices(root))[1])
    targets = nearest.query(shell.indices(ends))[1]
    distances, previous = dijkstra(graph, indices=start, return_predecessors=True)
    paths = []
    for target in targets:
        if not np.isfinite(distances[target]):
            continue
        route = [int(target)]
        while route[-1] != start:
            route.append(int(previous[route[-1]]))
        paths.append(shell.origin + occupied[route[::-1]] * shell.step)
    return paths


def _leg_joint_fractions(path: np.ndarray, step: float, height: float) -> np.ndarray:
    length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    planned = np.array([0.22, 0.50, 0.78])
    if length <= step:
        return planned
    samples = max(3, int(np.ceil(length / step)) + 1)
    points = ndimage.gaussian_filter1d(
        _resample(path, np.linspace(0, 1, samples)), 1.5, axis=0, mode="nearest"
    )
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    fractions = arc / max(arc[-1], 1e-10)
    window = max(2, int(round(0.035 * height / step)))
    indices = np.arange(samples)
    incoming = points - points[np.maximum(0, indices - window)]
    outgoing = points[np.minimum(samples - 1, indices + window)] - points
    norms = np.linalg.norm(incoming, axis=1) * np.linalg.norm(outgoing, axis=1)
    cosine = np.divide(
        np.sum(incoming * outgoing, axis=1),
        norms,
        out=np.ones(samples),
        where=norms > 1e-12,
    )
    bends = np.arccos(np.clip(cosine, -1, 1))
    result = []
    for desired in planned:
        candidates = np.flatnonzero(abs(fractions - desired) <= 0.12)
        best = candidates[np.argmax(bends[candidates])] if len(candidates) else 0
        result.append(float(fractions[best]) if bends[best] > math.radians(12) else float(desired))
    return np.asarray(result)


def _paired_leg_paths(left: list[np.ndarray], right: list[np.ndarray], plane: float, height: float) -> list[list[tuple[str, np.ndarray]]]:
    gap = 0.15 * height
    costs = np.zeros((len(left) + 1, len(right) + 1))
    costs[:, 0] = np.arange(len(left) + 1) * gap
    costs[0, :] = np.arange(len(right) + 1) * gap
    choices = {}
    for i, a in enumerate(left, 1):
        mirrored = a[0].copy()
        mirrored[0] = 2 * plane - mirrored[0]
        for j, b in enumerate(right, 1):
            options = (
                costs[i - 1, j - 1] + np.linalg.norm(mirrored - b[0]),
                costs[i - 1, j] + gap,
                costs[i, j - 1] + gap,
            )
            choice = int(np.argmin(options))
            costs[i, j], choices[i, j] = options[choice], choice
    pairs = []
    i, j = len(left), len(right)
    while i or j:
        choice = choices[i, j] if i and j else 1 if i else 2
        if choice == 0:
            pairs.append([("L", left[i - 1]), ("R", right[j - 1])])
            i, j = i - 1, j - 1
        elif choice == 1:
            pairs.append([("L", left[i - 1])])
            i -= 1
        else:
            pairs.append([("R", right[j - 1])])
            j -= 1
    return pairs[::-1]


def free_limb_paths(shell: Shell, paths: list[np.ndarray]) -> list[np.ndarray]:
    counts: dict[tuple[int, int, int], int] = {}
    for path in paths:
        for point in np.unique(shell.indices(path), axis=0):
            key = (int(point[0]), int(point[1]), int(point[2]))
            counts[key] = counts.get(key, 0) + 1
    output = []
    for path in paths:
        common = [i for i, p in enumerate(shell.indices(path)) if counts.get((int(p[0]), int(p[1]), int(p[2])), 0) > 1]
        radius = shell.radius(path)
        core = max(float(np.quantile(radius[: max(1, len(path) // 3)], 0.8)), 2 * shell.step)
        distal = float(np.median(radius[max(0, len(path) * 2 // 3) :]))
        thick = (
            np.flatnonzero((radius >= 0.65 * core) & (np.arange(len(path)) < 0.8 * len(path)))
            if core > 1.5 * max(distal, shell.step)
            else []
        )
        start = max(common[-1] if common else 0, int(thick[-1]) if len(thick) else 0)
        start = min(start, max(0, len(path) - 5))
        output.append(path[start:])
    return output


def manylegs_build(shell: Shell, root: np.ndarray) -> dict[str, Any]:
    feet = _footprints(shell, thin_legs=True)
    if len(feet) < 6:
        return {}
    paths = _interior_paths(shell, root, feet)
    if len(paths) < 6:
        return {}
    paths = free_limb_paths(shell, paths)
    height = max(np.ptp(np.argwhere(shell.solid)[:, 2]) * shell.step, shell.step)
    if shell.distance is None:
        shell.distance = ndimage.distance_transform_edt(shell.solid)
    core = shell.distance >= 0.5 * shell.distance.max()
    body = (ndimage.distance_transform_edt(~core) <= 1.2 * shell.distance.max()) & shell.solid
    for i, path in enumerate(paths):
        inside = np.flatnonzero(body[tuple(shell.indices(path).T)])
        if not len(inside):
            continue
        start = int(inside[-1])
        remainder = np.linalg.norm(np.diff(path[start:], axis=0), axis=1).sum()
        if remainder >= 0.3 * height and len(path) - start >= 5:
            paths[i] = path[start:]
    left = sorted([p for p in paths if p[-1, 0] > root[0]], key=lambda p: p[0, 1])
    right = sorted([p for p in paths if p[-1, 0] <= root[0]], key=lambda p: p[0, 1])
    result = {}
    for k, pair in enumerate(_paired_leg_paths(left, right, root[0], height), 1):
        fractions = [_leg_joint_fractions(path, shell.step, height) for _, path in pair]
        common = np.r_[0, np.mean(fractions, axis=0), 1]
        for side, path in pair:
            points = _resample(path, common)
            parent = None
            for j, role in enumerate(("Coxa", "Femur", "Tibia", "Tarsus")):
                name = f"{role}{k}.{side}"
                result[name] = {
                    "parent": parent,
                    "rest_head_world": points[j].tolist(),
                    "rest_tail_world": points[j + 1].tolist(),
                    "kind": "limb",
                    "side": side,
                    "paired_arc_fractions": common.tolist(),
                }
                parent = name
    return result


# -----------------------------------------------------------------------------
# Core Joint-Tree Post-Processing Engine
# -----------------------------------------------------------------------------

class PostprocessResult(tuple):
    """Dual-access result: unpackable as (bones, report) or accessible as dict."""

    def __new__(cls, bones: dict[str, Any], report: dict[str, Any]):
        return super().__new__(cls, (bones, report))

    @property
    def bones(self) -> dict[str, Any]:
        return self[0]

    @property
    def report(self) -> dict[str, Any]:
        return self[1]

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key == "bones":
                return self[0]
            if key in self[1]:
                return self[1][key]
            raise KeyError(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "bones":
            return self[0]
        return self[1].get(key, default)

    def __contains__(self, key: Any) -> bool:
        return key == "bones" or key in self[1]


def _build_joint_tree(
    bone_list: list[dict[str, Any]],
    height: float,
    floor: float,
) -> tuple[nx.Graph, str, dict[str, str]]:
    G = nx.Graph()
    raw_by_name = {b["name"]: b for b in bone_list}
    children_of: dict[str, list[str]] = {b["name"]: [] for b in bone_list}
    for b in bone_list:
        p = b.get("parent")
        if p and p in raw_by_name:
            children_of[p].append(b["name"])

    for b in bone_list:
        head = np.asarray(b.get("head", b.get("rest_head_world")), float)
        G.add_node(b["name"], pos=head, raw_name=b["name"], is_tip=False)

    for b in bone_list:
        p = b.get("parent")
        if p and p in raw_by_name:
            dist = float(np.linalg.norm(G.nodes[p]["pos"] - G.nodes[b["name"]]["pos"]))
            G.add_edge(p, b["name"], w=dist + 1e-6)

    tip_to_raw = {}
    for b in bone_list:
        if len(children_of[b["name"]]) == 0:
            head = np.asarray(b.get("head", b.get("rest_head_world")), float)
            raw_tail = np.asarray(b.get("tail", b.get("rest_tail_world")), float)
            d = raw_tail - head
            if d[2] < -0.05 * height and head[2] < 0.35 * height:
                t = (floor - head[2]) / d[2] if d[2] < -1e-6 else 0.0
                tip = head + t * d if 0 < t <= 2.5 else raw_tail.copy()
            else:
                tip = raw_tail.copy()
            tip[2] = max(floor, tip[2])
            tip_node = f"{b['name']}__tip"
            G.add_node(tip_node, pos=tip, raw_name=b["name"], is_tip=True)
            dist = float(np.linalg.norm(head - tip))
            G.add_edge(b["name"], tip_node, w=dist + 1e-6)
            tip_to_raw[tip_node] = b["name"]

    root = next(
        (b["name"] for b in bone_list if not b.get("parent") or b["parent"] not in raw_by_name),
        bone_list[0]["name"],
    )
    if root in G:
        comp = nx.node_connected_component(G, root)
        G = G.subgraph(comp).copy()
    return G, root, tip_to_raw


def _clean_joint_tree(
    G: nx.Graph,
    root: str,
    shell: Shell,
    height: float,
    floor: float,
    fwd: np.ndarray,
    left: np.ndarray,
    mid: float,
) -> tuple[nx.Graph, int, int]:
    dropped_outside = 0
    tol = 2.5 * shell.step

    for tip in [n for n in list(G.nodes) if G.nodes[n].get("is_tip", False)]:
        if tip not in G:
            continue
        path = [tip]
        curr = tip
        while True:
            nb = [m for m in G.neighbors(curr) if m not in path]
            if not nb:
                break
            curr = nb[0]
            path.append(curr)
            if G.degree(curr) != 2 or curr == root:
                break
        if root in path[:-1] or len(path) < 2:
            continue
        pts = np.array(
            [
                G.nodes[a]["pos"] + (G.nodes[b]["pos"] - G.nodes[a]["pos"]) * s
                for a, b in zip(path, path[1:])
                for s in np.linspace(0, 1, 6)
            ]
        )
        if (shell.off_skin(pts) > tol).mean() > 0.5:
            G.remove_nodes_from(path[:-1])
            dropped_outside += 1

    for tip in [n for n in list(G.nodes) if G.nodes[n].get("is_tip", False)]:
        if tip not in G:
            continue
        nb = list(G.neighbors(tip))
        if not nb:
            continue
        j = nb[0]
        a = G.nodes[j]["pos"]
        b = G.nodes[tip]["pos"].copy()
        for s in np.linspace(1.0, 0.3, 15):
            pt = a + (b - a) * s
            if shell.off_skin(pt)[0] <= 1.5 * shell.step:
                b = pt
                break
        else:
            b = a + (b - a) * 0.3
        b[2] = max(b[2], floor)
        G.nodes[tip]["pos"] = b

    pruned_spurs = 0
    all_fwd = [G.nodes[n]["pos"] @ fwd for n in G.nodes]
    max_fwd = max(all_fwd) if all_fwd else 0.0

    for leaf in [n for n in list(G.nodes) if G.degree(n) == 1]:
        if leaf not in G or G.nodes[leaf].get("is_tip", False) or leaf == root:
            continue
        par = list(G.neighbors(leaf))[0]
        if G.degree(par) >= 3:
            length = float(np.linalg.norm(G.nodes[leaf]["pos"] - G.nodes[par]["pos"]))
            p = G.nodes[leaf]["pos"]
            lat_val = abs((p @ left) - mid)
            is_snout = (lat_val < 0.06 * height) and (p @ fwd >= max_fwd - 0.05 * height)
            if length < 0.10 * height and not is_snout:
                G.remove_node(leaf)
                pruned_spurs += 1

    if root not in G and len(G):
        root = max(G.nodes, key=G.degree)
    if len(G):
        G = G.subgraph(nx.node_connected_component(G, root)).copy()

    for n in list(G.nodes):
        if not G.nodes[n].get("is_tip", False) and G.degree(n) == 1 and n != root:
            p = list(G.neighbors(n))[0]
            a, b = G.nodes[p]["pos"], G.nodes[n]["pos"]
            tip_id = f"{n}__tip"
            tip_pos = b + (b - a) * 0.3
            tip_pos[2] = max(floor, tip_pos[2])
            G.add_node(tip_id, pos=tip_pos, raw_name=G.nodes[n].get("raw_name", n), is_tip=True)
            G.add_edge(n, tip_id, w=float(np.linalg.norm(b - tip_pos)) + 1e-6)

    return G, dropped_outside, pruned_spurs


def _analyse_tree(
    G: nx.Graph,
    height: float,
    floor: float,
    facing: str | dict,
    mid: float = 0.0,
    mid_tol: float = 0.06,
) -> dict[str, Any]:
    facing_key = facing.get("forward", "-Y") if isinstance(facing, dict) else str(facing)
    fwd = AXES.get(facing_key, AXES["-Y"])
    up = np.array([0.0, 0.0, 1.0])
    left = np.cross(up, fwd)
    lat = lambda p: p @ left - mid

    ends = [n for n in G.nodes if G.degree(n) == 1]
    P = {n: G.nodes[n]["pos"] for n in ends}

    feet = [n for n in ends if P[n][2] - floor < 0.08 * height]
    if len(feet) > 2:
        back = np.mean([P[n] @ fwd for n in feet])
        feet = [n for n in feet if not (abs(lat(P[n])) < 0.1 * height and P[n] @ fwd < back - 0.25 * height)]
    rest = [n for n in ends if n not in feet]

    middle = [n for n in rest if abs(lat(P[n])) < mid_tol * height]
    if middle:
        head = max(middle, key=lambda n: P[n] @ fwd + 0.25 * P[n] @ up)
    else:
        mids = [n for n in G.nodes if abs(lat(G.nodes[n]["pos"])) < 0.06 * height]
        if mids:
            head = max(mids, key=lambda n: G.nodes[n]["pos"] @ up + 0.25 * G.nodes[n]["pos"] @ fwd)
        else:
            head = max(ends, key=lambda n: P[n] @ up)
        P[head] = G.nodes[head]["pos"]

    mean_ends_fwd = np.mean([P[m] @ fwd for m in ends]) if ends else 0.0
    tails = [
        n
        for n in rest
        if n != head
        and abs(lat(P[n])) < 0.2 * height
        and P[n] @ fwd < mean_ends_fwd - 0.12 * height
    ]
    tail = min(tails, key=lambda n: P[n] @ fwd) if tails else None

    if tail is not None:
        spine = nx.shortest_path(G, tail, head, weight="w")
    else:
        rear = sorted(feet, key=lambda n: P[n] @ fwd)[:2] or [head]
        paths = [nx.shortest_path(G, f, head, weight="w") for f in rear]
        common = [n for n in paths[0] if all(n in p for p in paths[1:])]
        spine = common if len(common) > 2 else paths[0]

    # Crucial Fix 2: Curled tail recovery
    if tail is None and len(spine) > 0:
        hips_node = spine[0]
        step_threshold = -0.03 * height
        best_curled_tip = None
        best_curled_dist = -1.0
        for nb in G.neighbors(hips_node):
            if nb in spine:
                continue
            step_vec = G.nodes[nb]["pos"] - G.nodes[hips_node]["pos"]
            lat_nb = abs(lat(G.nodes[nb]["pos"]))
            if step_vec @ fwd < step_threshold and lat_nb < 0.15 * height:
                H = G.copy()
                H.remove_edge(hips_node, nb)
                comp = nx.node_connected_component(H, nb)
                dists = nx.single_source_dijkstra_path_length(G.subgraph(comp | {hips_node}), hips_node, weight="w")
                furthest = max(comp, key=lambda x: dists.get(x, 0.0))
                if dists.get(furthest, 0.0) > best_curled_dist:
                    best_curled_dist = dists[furthest]
                    best_curled_tip = furthest

        if best_curled_tip is not None and best_curled_dist >= 0.15 * height:
            tail = best_curled_tip
            tail_path = nx.shortest_path(G, tail, hips_node, weight="w")
            spine = tail_path[:-1] + spine

    on_spine = {n: i for i, n in enumerate(spine)}

    def attach(end: str) -> tuple[list[str], int]:
        path = nx.shortest_path(G, end, spine[0], weight="w")
        for k, n in enumerate(path):
            if n in on_spine:
                return path[: k + 1][::-1], on_spine[n]
        return path[::-1], 0

    limbs = []
    candidates = [n for n in ends if n not in (head, tail)]
    for n in candidates:
        path, at = attach(n)
        if len(path) < 2:
            continue
        side = 1 if lat(P[n]) > 0 else -1
        pts = np.array([G.nodes[x]["pos"] for x in path])
        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        limbs.append(dict(end=n, path=path, at=at, side=side, foot=n in feet, length=length))

    def limb_key(l: dict[str, Any]) -> str:
        acc, prev = 0.0, l["path"][0]
        for n in l["path"][1:]:
            acc += G.edges[prev, n]["w"] if G.has_edge(prev, n) else 0.0
            prev = n
            if acc >= 0.10 * height:
                return n
        return l["path"][-1]

    best_limbs: dict[str, dict[str, Any]] = {}
    for l in limbs:
        k = limb_key(l)
        if k not in best_limbs or l["length"] > best_limbs[k]["length"]:
            best_limbs[k] = l
        if l["foot"]:
            best_limbs[k]["foot"] = True
    limbs = list(best_limbs.values())

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used: set[int] = set()
    for i, a in enumerate(limbs):
        if i in used:
            continue
        best_j, bd = None, 1e9
        for j, b in enumerate(limbs):
            if j == i or j in used or a["side"] == b["side"] or a["foot"] != b["foot"]:
                continue
            pa, pb = P[a["end"]], P[b["end"]]
            d = float(np.linalg.norm((pa - (pa @ left) * left) - (pb - (pb @ left) * left)))
            if d < bd:
                best_j, bd = j, d
        if best_j is not None and bd < 0.35 * height:
            used |= {i, best_j}
            pairs.append((a, limbs[best_j]))

    singles = [l for k, l in enumerate(limbs) if k not in used]
    leg_pairs = sorted([p for p in pairs if p[0]["foot"]], key=lambda p: p[0]["at"])
    up_pairs = sorted([p for p in pairs if not p[0]["foot"]], key=lambda p: -p[0]["length"])

    # Crucial Fix 1: Filter horns/ears ahead of front legs in quadrupeds
    if len(leg_pairs) >= 2:
        front_at = leg_pairs[1][0]["at"]
        kept_ups = []
        for p in up_pairs:
            if p[0]["at"] > front_at and p[0]["length"] < 0.35 * height:
                singles.extend(p)
            else:
                kept_ups.append(p)
        up_pairs = kept_ups

    body = (
        "centaur"
        if len(leg_pairs) >= 2 and up_pairs
        else "quadruped"
        if len(leg_pairs) >= 2
        else "winged"
        if up_pairs and abs(P[up_pairs[0][0]["end"]] @ left) > 0.35 * height
        else "biped"
    )

    return dict(
        fwd=fwd,
        up=up,
        left=left,
        spine=spine,
        head=head,
        tail=tail,
        leg_pairs=leg_pairs,
        up_pairs=up_pairs,
        body=body,
        floor=floor,
        singles=singles,
    )


def _label_skeleton(
    G: nx.Graph,
    A: dict[str, Any],
    height: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    spine = A["spine"]
    floor = A["floor"]
    leg_pairs = A["leg_pairs"]
    up_pairs = A["up_pairs"]
    body = A["body"]

    if leg_pairs:
        hips_idx = leg_pairs[0][0]["at"]
    elif A["tail"] is not None:
        hips_idx = min(3, len(spine) // 2)
    else:
        hips_idx = 0
    hips_idx = int(np.clip(hips_idx, 0, len(spine) - 1))
    hips_node = spine[hips_idx]

    back_nodes = [n for n in spine[:hips_idx][::-1] if not G.nodes[n].get("is_tip", False)]
    front_nodes = [n for n in spine[hips_idx + 1 :] if not G.nodes[n].get("is_tip", False)]
    attach_indices = [p[0]["at"] for p in leg_pairs + up_pairs[:1]]
    last_attach = max([a for a in attach_indices if a > hips_idx], default=hips_idx)

    node_to_name: dict[str, str] = {}
    node_to_role: dict[str, str] = {}
    node_to_kind: dict[str, str] = {}
    node_to_tail: dict[str, np.ndarray] = {}

    node_to_name[hips_node] = "Hips"
    node_to_role[hips_node] = "trunk"
    node_to_kind[hips_node] = "pelvis"

    # Preserve already canonical names on joints if present
    for n in G.nodes:
        if not G.nodes[n].get("is_tip", False):
            raw = G.nodes[n].get("raw_name", n)
            if parse_anatomical_name(raw) and raw != "Root":
                node_to_name[n] = raw

    for k, n in enumerate(back_nodes):
        if n not in node_to_name:
            node_to_name[n] = f"Tail{k + 1}"
        node_to_role[n] = "tail"
        node_to_kind[n] = "tail"
        idx = spine.index(n)
        node_to_tail[n] = G.nodes[spine[idx - 1]]["pos"] if idx > 0 else G.nodes[A["tail"]]["pos"]

    head_joint = front_nodes[-1] if front_nodes else None
    ns = nn = 0
    for n in front_nodes:
        idx = spine.index(n)
        node_to_tail[n] = (
            G.nodes[spine[idx + 1]]["pos"]
            if idx + 1 < len(spine)
            else G.nodes[A["head"]]["pos"]
        )
        if n == head_joint:
            if n not in node_to_name:
                node_to_name[n] = "Head"
            node_to_role[n] = "head"
            node_to_kind[n] = "head"
        elif idx <= last_attach:
            ns += 1
            if n not in node_to_name:
                node_to_name[n] = f"Spine{ns}"
            node_to_role[n] = "trunk"
            node_to_kind[n] = "spine"
        else:
            nn += 1
            if n not in node_to_name:
                node_to_name[n] = f"Neck{nn}"
            node_to_role[n] = "neck"
            node_to_kind[n] = "head"

    if hips_node not in node_to_tail:
        if back_nodes:
            node_to_tail[hips_node] = G.nodes[back_nodes[0]]["pos"]
        elif front_nodes:
            node_to_tail[hips_node] = G.nodes[front_nodes[0]]["pos"]
        else:
            node_to_tail[hips_node] = G.nodes[hips_node]["pos"] + np.array([0, 0, 0.1 * height])

    raw_to_canonical: dict[str, str] = {}
    kinds: list[tuple[str, list[dict[str, Any]]]] = []
    if body in ("quadruped", "centaur"):
        if leg_pairs:
            kinds.append(("hind", leg_pairs[0]))
        if len(leg_pairs) > 1:
            kinds.append(("front", leg_pairs[1]))
        if up_pairs:
            kinds.append(("arm", up_pairs[0]))
    elif body == "winged":
        if leg_pairs:
            kinds.append(("biped", leg_pairs[0]))
        if up_pairs:
            kinds.append(("wing", up_pairs[0]))
    else:
        if leg_pairs:
            kinds.append(("biped", leg_pairs[0]))
        if up_pairs:
            kinds.append(("arm", up_pairs[0]))

    for kind, pair in kinds:
        top_name, names, fracs = PLANS[kind]
        for limb in pair:
            sfx = SIDES[limb["side"]]
            path_joints = [n for n in limb["path"][1:] if not G.nodes[n].get("is_tip", False)]
            if not path_joints:
                continue
            tip_pos = G.nodes[limb["path"][-1]]["pos"]
            all_pts = [G.nodes[n]["pos"] for n in path_joints] + [tip_pos]
            dists = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(all_pts, axis=0), axis=1))])
            total_len = max(dists[-1], 1e-6)

            use_top = (top_name is not None) and (len(path_joints) >= len(names) + 1)
            chain_joints = path_joints[1:] if use_top else path_joints
            if use_top:
                if path_joints[0] not in node_to_name:
                    node_to_name[path_joints[0]] = top_name + sfx
                node_to_role[path_joints[0]] = kind
                node_to_kind[path_joints[0]] = "limb"
                node_to_tail[path_joints[0]] = G.nodes[chain_joints[0]]["pos"]

            if len(chain_joints) <= len(names):
                for m, n in enumerate(chain_joints):
                    if n not in node_to_name:
                        node_to_name[n] = names[min(m, len(names) - 1)] + sfx
                    node_to_role[n] = kind
                    node_to_kind[n] = "limb"
                    node_to_tail[n] = (
                        G.nodes[chain_joints[m + 1]]["pos"]
                        if m + 1 < len(chain_joints)
                        else tip_pos
                    )
            else:
                # Select exactly len(names) joints: first is names[0], last is names[-1], intermediates pick closest to fracs
                f_norm = dists[1:-1] / total_len
                chosen_indices = [0]
                K = len(names)
                # Pick K - 2 middle joints from chain_joints[1:-1]
                mid_candidates = list(range(1, len(chain_joints) - 1))
                target_mid_fracs = fracs[:-1] if len(fracs) >= K - 1 else fracs
                picked_mid = []
                last_c = 0
                for target_f in target_mid_fracs[: K - 2]:
                    valid_c = [c for c in mid_candidates if c > last_c]
                    if valid_c:
                        best_c = min(valid_c, key=lambda c: abs(f_norm[c - 1] - target_f))
                        picked_mid.append(best_c)
                        last_c = best_c
                chosen_indices.extend(picked_mid)
                chosen_indices.append(len(chain_joints) - 1)

                chosen_joints = [chain_joints[idx] for idx in chosen_indices]
                for k_idx, n in enumerate(chosen_joints):
                    node_to_name[n] = names[k_idx] + sfx
                    node_to_role[n] = kind
                    node_to_kind[n] = "limb"
                    node_to_tail[n] = (
                        G.nodes[chosen_joints[k_idx + 1]]["pos"]
                        if k_idx + 1 < len(chosen_joints)
                        else tip_pos
                    )
                # Map unchosen joints to preceding chosen bone
                for idx, n in enumerate(chain_joints):
                    if n not in node_to_name:
                        preceding = [c for c in chosen_indices if c < idx]
                        prec_idx = preceding[-1] if preceding else 0
                        raw_to_canonical[G.nodes[n].get("raw_name", n)] = names[chosen_indices.index(prec_idx)] + sfx

    parent_map = {c: p for p, c in nx.bfs_edges(G, hips_node)}
    head_extras = 0
    limb_extras = 0
    for n in nx.bfs_tree(G, hips_node):
        if G.nodes[n].get("is_tip", False) or n in node_to_name:
            continue
        p = parent_map.get(n, hips_node)
        while p not in node_to_name and p in parent_map:
            p = parent_map[p]
        p_name = node_to_name.get(p, "Spine1")
        sfx = ".L" if G.nodes[n]["pos"] @ A["left"] > 0 else ".R"

        if p_name.startswith("Head") or p_name.startswith("Neck"):
            head_extras += 1
            node_to_name[n] = f"HeadX{head_extras}{sfx}"
            node_to_role[n] = "head"
            node_to_kind[n] = "flex"
        else:
            limb_extras += 1
            node_to_name[n] = f"Pair{limb_extras}_1{sfx}"
            node_to_role[n] = "chain"
            node_to_kind[n] = "limb"

        kids = [c for c in G.neighbors(n) if c != p]
        if kids:
            node_to_tail[n] = G.nodes[kids[0]]["pos"]
        else:
            d_vec = G.nodes[n]["pos"] - G.nodes[p]["pos"]
            node_to_tail[n] = G.nodes[n]["pos"] + (d_vec / max(np.linalg.norm(d_vec), 1e-6)) * (0.05 * height)

    output_bones: dict[str, dict[str, Any]] = {}
    root_pos = np.array([G.nodes[hips_node]["pos"][0], G.nodes[hips_node]["pos"][1], floor])
    output_bones["Root"] = {
        "name": "Root",
        "head": root_pos.tolist(),
        "tail": (root_pos + np.array([0, 0, 0.1 * height])).tolist(),
        "rest_head_world": root_pos.tolist(),
        "rest_tail_world": (root_pos + np.array([0, 0, 0.1 * height])).tolist(),
        "parent": None,
        "role": "root",
        "kind": "root_Hips",
        "canonical_name": "Root",
        "raw_name": "Root",
    }

    for n in nx.bfs_tree(G, hips_node):
        if G.nodes[n].get("is_tip", False):
            continue
        c_name = node_to_name.get(n, f"Bone_{n}")
        p_node = parent_map.get(n)
        p_name = "Root" if n == hips_node else node_to_name.get(p_node, "Root")
        head = G.nodes[n]["pos"]
        tail = node_to_tail.get(n, head + np.array([0, 0, 0.05 * height]))
        role = node_to_role.get(n, "limb")
        kind = node_to_kind.get(n, "limb")

        canonical_name = c_name
        sub_role = "forearm_hinge" if ("LowerLeg" in c_name or "LowerArm" in c_name or "Shin" in c_name) else None
        if "UpperLeg" in c_name:
            canonical_name = c_name.replace("UpperLeg", "Thigh")
        elif "LowerLeg" in c_name:
            canonical_name = c_name.replace("LowerLeg", "Shin")
        elif "Toe" in c_name and "Toes" not in c_name:
            canonical_name = c_name.replace("Toe", "Toes")

        b_entry = {
            "name": c_name,
            "head": [float(x) for x in head],
            "tail": [float(x) for x in tail],
            "rest_head_world": [float(x) for x in head],
            "rest_tail_world": [float(x) for x in tail],
            "parent": p_name,
            "role": role,
            "kind": kind,
            "canonical_name": canonical_name,
            "raw_name": G.nodes[n].get("raw_name", n),
        }
        if sub_role:
            b_entry["sub_role"] = sub_role
        output_bones[c_name] = b_entry
        raw_to_canonical[G.nodes[n].get("raw_name", n)] = c_name

    return output_bones, raw_to_canonical


def postprocess_skintokens_skeleton(
    bones: Any,
    *args: Any,
    shell: Shell | None = None,
    fold_digits: bool = False,
    facing: str = "-Y",
    **kwargs: Any,
) -> PostprocessResult:
    """Analyze and post-process SkinTokens raw bones into canonical v3 skeleton."""
    verts: np.ndarray | None = None
    if len(args) >= 2:
        verts = np.asarray(args[0], float)
        shell = args[1]
    elif len(args) == 1:
        if isinstance(args[0], Shell):
            shell = args[0]
        else:
            verts = np.asarray(args[0], float)

    if isinstance(bones, dict):
        bone_list = [{"name": k, **v} if "name" not in v else v for k, v in bones.items()]
    elif isinstance(bones, list):
        bone_list = bones
    else:
        raise ValueError(f"Unsupported bones structure: {type(bones)}")

    if len(bone_list) > 0 and all(parse_anatomical_name(str(b.get("name", ""))) is not None for b in bone_list):
        names_set = {str(b["name"]) for b in bone_list}
        output_bones: dict[str, dict[str, Any]] = {}
        weight_fold_map: dict[str, str] = {}
        for b in bone_list:
            nm = str(b["name"])
            h = [float(x) for x in b.get("head", b.get("rest_head_world", [0, 0, 0]))]
            t = [float(x) for x in b.get("tail", b.get("rest_tail_world", [0, 0, 0.05]))]
            parsed = parse_anatomical_name(nm) or {}
            is_org_root = (nm == "Root" and "Hips" in names_set)
            entry = dict(b)
            entry.update(
                name=nm,
                head=h,
                tail=t,
                rest_head_world=h,
                rest_tail_world=t,
                parent=b.get("parent"),
                kind=b.get("kind", parsed.get("kind", "limb")),
                side=b.get("side", parsed.get("side", "C")),
                role=b.get("role", "root" if is_org_root else parsed.get("kind", "limb")),
                canonical_name=b.get("canonical_name", nm),
                raw_name=nm,
            )
            output_bones[nm] = entry
            weight_fold_map[nm] = nm
        report = {
            "bones": output_bones,
            "archetype": "humanoid",
            "body": "biped",
            "facing": facing,
            "weight_fold_map": weight_fold_map,
            "raw_to_canonical": weight_fold_map,
            "dropped_outside": 0,
            "pruned_spurs": 0,
            "folded_branches": 0,
            "transplanted_limbs": 0,
            "fold_digits": fold_digits,
            "removed": {},
        }
        return PostprocessResult(output_bones, report)

    if shell is None:
        if verts is None:
            heads = [np.asarray(b.get("head", b.get("rest_head_world", [0, 0, 0])), float) for b in bone_list]
            verts = np.asarray(heads, float)
        shell = voxelize(verts, None)

    height = max(
        float(np.ptp(shell.verts[:, 2])) if shell.verts is not None else float(shell.solid.shape[2] * shell.step),
        1e-4,
    )
    floor = float(shell.verts[:, 2].min()) if shell.verts is not None else float(shell.origin[2])

    facing_key = facing.get("forward", "-Y") if isinstance(facing, dict) else str(facing)
    fwd = AXES.get(facing_key, AXES["-Y"])
    left = np.cross([0.0, 0.0, 1.0], fwd)
    ijk = np.argwhere(shell.solid)
    mid = float(np.median((shell.origin + (ijk + 0.5) * shell.step) @ left)) if len(ijk) else 0.0

    # 1. Build joint tree
    G, root, tip_to_raw = _build_joint_tree(bone_list, height, floor)

    # 2. Clean tree
    G, dropped_outside, pruned_spurs = _clean_joint_tree(
        G, root, shell, height, floor, fwd, left, mid
    )

    # 3. Analyze tree topology
    A = _analyse_tree(G, height, floor, facing_key, mid=mid)

    # 4. Label skeleton with canonical v3 names
    output_bones, raw_to_canonical = _label_skeleton(G, A, height)

    # 5. Compute complete weight_fold_map for EVERY input raw bone
    all_raw_names = [b["name"] for b in bone_list]
    raw_parents = {b["name"]: b.get("parent") for b in bone_list}
    weight_fold_map: dict[str, str] = {}
    weight_fold_map["Root"] = "Root"

    for r_name in all_raw_names:
        if r_name in raw_to_canonical:
            weight_fold_map[r_name] = raw_to_canonical[r_name]
        else:
            curr = r_name
            found = None
            visited = set()
            while curr and curr not in visited:
                visited.add(curr)
                p = raw_parents.get(curr)
                if p in raw_to_canonical:
                    found = raw_to_canonical[p]
                    break
                curr = p
            weight_fold_map[r_name] = found if found else "Hips"

    report = {
        "bones": output_bones,
        "archetype": A["body"],
        "body": A["body"],
        "facing": facing,
        "weight_fold_map": weight_fold_map,
        "raw_to_canonical": weight_fold_map,
        "dropped_outside": dropped_outside,
        "pruned_spurs": pruned_spurs,
        "folded_branches": 0,
        "transplanted_limbs": 0,
        "fold_digits": fold_digits,
        "removed": {r: c for r, c in weight_fold_map.items() if r not in output_bones},
    }

    return PostprocessResult(output_bones, report)
