"""V3 articulated FK with analytic derivatives and robust multi-camera fitting.

Root rotations use world-space exponential coordinates. Descendant rotations use Blender
XYZ local Euler channels, preserving the exact rest offsets, including disconnected joints.
"""

import math
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits
from studio_camera import triangulate


def skew(v):
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def exp_so3(w):
    w = np.asarray(w, float)
    t = np.linalg.norm(w)
    K = skew(w)
    a = np.sinc(t / np.pi)
    b = 0.5 * np.sinc(t / (2 * np.pi)) ** 2
    return np.eye(3) + a * K + b * (K @ K)


def right_jacobian_so3(w):
    w = np.asarray(w, float)
    t = np.linalg.norm(w)
    K = skew(w)
    if t < 1e-5:
        return np.eye(3) - 0.5 * K + (1 / 6 - t * t / 120) * (K @ K)
    return np.eye(3) - (1 - math.cos(t)) / t**2 * K + (t - math.sin(t)) / t**3 * (K @ K)


def rotation_between(source, target):
    a = np.asarray(source, float)
    b = np.asarray(target, float)
    if min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-12:
        return np.eye(3)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cosine = float(np.clip(a @ b, -1, 1))
    cross = np.cross(a, b)
    if cosine > -1 + 1e-8:
        K = skew(cross)
        return np.eye(3) + K + K @ K / (1 + cosine)
    axis = np.cross(a, np.eye(3)[np.argmin(abs(a))])
    axis /= np.linalg.norm(axis)
    return exp_so3(axis * np.pi)


def unwrap_rotvecs(values):
    out = np.asarray(values, float).copy()
    for i in range(1, len(out)):
        t = np.linalg.norm(out[i])
        axis = (
            out[i] / t
            if t > 1e-10
            else (out[i - 1] / max(np.linalg.norm(out[i - 1]), 1e-10))
        )
        choices = [out[i] + 2 * np.pi * k * axis for k in range(-3, 4)]
        out[i] = min(choices, key=lambda v: np.linalg.norm(v - out[i - 1]))
    return out


def _euler(a):
    x, y, z = a
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    X = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    dX = np.array([[0, 0, 0], [0, -sx, -cx], [0, cx, -sx]])
    dY = np.array([[-sy, 0, cy], [0, 0, 0], [-cy, 0, -sy]])
    dZ = np.array([[-sz, -cz, 0], [cz, -sz, 0], [0, 0, 0]])
    return Z @ Y @ X, [Z @ Y @ dX, Z @ dY @ X, dZ @ Y @ X]


def aligned_rest_rotation(head, tail, left=(1, 0, 0)):
    y = np.asarray(tail) - head
    y = y / max(np.linalg.norm(y), 1e-10)
    x = np.asarray(left, float) - y * np.dot(left, y)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0, 0, 1.0]) - y * y[2]
    x /= max(np.linalg.norm(x), 1e-10)
    return np.column_stack([x, y, np.cross(x, y)])


def _fit_structure(meta, skin=None):
    """Keep weighted ancestors, pruning only empty organization above the pelvis."""
    parents = {
        n: b.get("parent") if b.get("parent") in meta else None
        for n, b in meta.items()
    }
    if skin is None:
        active = {
            n
            for n, b in meta.items()
            if b.get("kind") != "rigid" and not b.get("fixed_local")
        }
        return active, {n for n in active if parents[n] is None}
    mass = np.zeros(len(skin["bone_names"]))
    indices = np.asarray(skin["skin_indices"])
    weights = np.asarray(skin["skin_weights"])
    np.add.at(mass, indices.ravel(), weights.ravel())
    active = set()
    direct_mass = dict(zip(skin["bone_names"], mass))
    for i, (n, m) in enumerate(zip(skin["bone_names"], mass / max(mass.sum(), 1e-12))):
        is_jaw = n == "Jaw" or n.startswith("Jaw") or meta.get(n, {}).get("canonical_name") == "Jaw"
        has_jaw_support = is_jaw and (
            m > 0.0
            or "jaw_hinge" in meta.get(n, {})
            or int(np.count_nonzero(np.any((indices == i) & (weights > 0.3), axis=1))) >= 1
        )
        # Small weighted parts (fingers, toes, eyelids) still deform the model.
        # A whole-character mass cutoff silently freezes them in synthesis/fitting.
        if m <= 0.0 and not has_jaw_support:
            continue
        while n in meta and n not in active:
            active.add(n)
            n = meta[n].get("parent")
    active = {
        n
        for n in active
        if not meta[n].get("fixed_local") and meta[n].get("kind") != "rigid"
    }
    roots = {n for n in active if parents[n] is None}

    def ancestors(name):
        chain = []
        while name in meta and name not in chain:
            chain.append(name)
            name = parents[name]
        return chain

    for hips in sorted(active):
        if hips != "Hips" and meta[hips].get("canonical_name") != "Hips":
            continue
        chain = ancestors(hips)
        root = chain[-1]
        if len(chain) == 1 or root not in roots:
            continue
        # Even a tiny positive influence outside this subtree prevents pruning:
        # the common root still couples those weighted branches physically.
        weighted = [
            n for n, m in direct_mass.items() if m > 0 and root in ancestors(n)
        ]
        if any(hips not in ancestors(n) for n in weighted):
            continue
        active.difference_update(chain[1:])
        roots.remove(root)
        roots.add(hips)
    return active, roots


def fit_bones(meta, skin=None):
    return _fit_structure(meta, skin)[0]


def _orient_transverse_axis(axis_2d, bone_name, R_world):
    """Deterministic mirror-consistent orientation for off-axis side vectors (outward, facing, up)."""
    outward = np.array([-1.0 if str(bone_name).endswith(".R") else 1.0, 0.0, 0.0])
    facing = np.array([0.0, -1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    refs_2d = [(R_world.T @ r)[[0, 2]] for r in (outward, facing, up)]
    ref = max(refs_2d, key=lambda r: abs(float(r @ axis_2d)))
    if float(ref @ axis_2d) < -1e-12:
        return -axis_2d
    return axis_2d


def surface_keypoints(
    meta,
    names,
    skin,
    *,
    snap_to_vertices=False,
    round_prop_only_mid=False,
):
    """Append skin tips and off-axis cross-section quantiles after all 2N endpoints."""
    from pose_features import keypoint_mode

    if skin is None or keypoint_mode() == "legacy":
        return []
    vertices = np.asarray(skin["rest_verts"])
    indices = np.asarray(skin["skin_indices"])
    weights = np.asarray(skin["skin_weights"])
    skin_index = {n: i for i, n in enumerate(skin["bone_names"])}
    parents = {b.get("parent") for b in meta.values()}
    extra = []
    for n in names:
        if n not in skin_index:
            continue
        i = skin_index[n]
        dominant = np.any((indices == i) & (weights > 0.3), axis=1)
        pts = vertices[dominant]
        if len(pts) < 4:
            continue
        b = meta[n]
        h = np.asarray(b["rest_head_world"])
        R = np.asarray(
            b.get("rest_rot_mat_world", aligned_rest_rotation(h, b["rest_tail_world"]))
        )
        local = (pts - h) @ R
        if n not in parents:
            q = np.array([0.0, np.quantile(local[:, 1], 0.98), 0.0])
            extra.append({"name": n + ".tip", "bone": n, "local": q.tolist()})
        if any(
            n.startswith(p)
            for p in (
                "Hand",
                "Foot",
                "FrontPaw",
                "HindPaw",
                "Head",
                "WingHand",
                "Weapon",
                "Prop_",
            )
        ):
            mid = np.median(local, axis=0)
            xy = local[:, [0, 2]] - mid[[0, 2]]
            _, axes = np.linalg.eigh(xy.T @ xy / max(1, len(xy)))
            ordered_axes = [
                _orient_transverse_axis(axis, n, R) for axis in axes.T[::-1]
            ]
            for idx_ax, (label, axis) in enumerate(zip(("sidea", "sideb"), ordered_axes)):
                other_axis = ordered_axes[1 - idx_ax]
                projected = xy @ axis
                other_proj = xy @ other_axis
                for j, q in enumerate((0.01, 0.99), 1):
                    target = float(np.quantile(projected, q))
                    if snap_to_vertices:
                        pick = int(np.argmin((projected - target) ** 2 + 0.25 * (other_proj ** 2)))
                        point = local[pick].copy()
                    else:
                        point = mid.copy()
                        point[[0, 2]] += axis * target
                    extra.append(
                        {"name": f"{n}.{label}{j}", "bone": n, "local": point.tolist()}
                    )
        if n.startswith("Prop_"):
            lo, hi = np.quantile(local[:, 1], [0.01, 0.99])
            centered = local - local.mean(0)
            singular = np.linalg.svd(centered, compute_uv=False) if len(local) >= 3 else np.array([1.0, 0.0, 0.0])
            is_long_prop = float(singular[0]) >= 1.8 * float(singular[1]) if len(singular) >= 2 else True
            prop_labels = (
                (("mid", (lo + hi) / 2),)
                if (round_prop_only_mid and not is_long_prop)
                else (("end1", lo), ("end2", hi), ("mid", (lo + hi) / 2))
            )
            for label, q in prop_labels:
                extra.append(
                    {"name": n + "." + label, "bone": n, "local": [0.0, float(q), 0.0]}
                )
    return extra


class RigModel:
    def __init__(self, meta, names=None, skin=None, extra_points=None):
        self.meta = meta
        self.skin = skin
        self.names = list(names or sorted(meta))
        self.index = {n: i for i, n in enumerate(self.names)}
        self.parents = {
            n: b.get("parent") if b.get("parent") in meta else None
            for n, b in meta.items()
        }
        self.order = []
        pending = set(self.names)
        while pending:
            ready = sorted(
                n
                for n in pending
                if self.parents[n] is None or self.parents[n] in self.order
            )
            if not ready:
                raise ValueError("Cyclic or incomplete bone hierarchy")
            self.order.extend(ready)
            pending.difference_update(ready)
        self.h = {n: np.asarray(meta[n]["rest_head_world"], float) for n in self.names}
        self.t = {n: np.asarray(meta[n]["rest_tail_world"], float) for n in self.names}
        from rig_pose import rigid_rotation, rigid_pose_basis

        self.R = {
            n: rigid_rotation(
                meta[n].get(
                    "rest_rot_mat_world", aligned_rest_rotation(self.h[n], self.t[n])
                ),
                n,
            )
            for n in self.names
        }
        self.height = max(
            np.ptp(np.array(list(self.h.values()) + list(self.t.values()))[:, 2]), 1e-4
        )
        if skin is not None:
            self.height = float(
                skin.get("height", np.ptp(np.asarray(skin["rest_verts"])[:, 2]))
            )
            if not np.isfinite(self.height) or self.height <= 0:
                raise ValueError(
                    "Rig skinning requires a positive finite reference height"
                )
        active, self.motion_roots = _fit_structure(meta, skin)
        self.basis = {
            n: rigid_pose_basis(meta[n].get("pose_basis", np.eye(4)), n)
            for n in self.names
        }
        self.slices = {}
        lower = []
        upper = []
        for n in self.order:
            if n in self.motion_roots:
                self.slices[n] = slice(len(lower), len(lower) + 6)
                lower.extend([-np.inf] * 6)
                upper.extend([np.inf] * 6)
            elif n in active:
                lim = np.asarray(
                    meta[n].get("limits_rad", [[-np.pi, np.pi]] * 3), float
                )
                # Fixed channels have no optimization degree of freedom.
                channels = np.flatnonzero(lim[:, 1] - lim[:, 0] > 1e-8)
                self.slices[n] = (
                    np.arange(len(lower), len(lower) + len(channels)),
                    channels,
                )
                lower.extend(lim[channels, 0])
                upper.extend(lim[channels, 1])
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.size = len(lower)
        self.extra = list(
            extra_points
            if extra_points is not None
            else surface_keypoints(meta, self.names, skin, snap_to_vertices=True)
        )
        self.point_names = [
            n + "." + end for n in self.names for end in ("head", "tail")
        ] + [p["name"] for p in self.extra]
        self.point_bones = [n for n in self.names for _ in range(2)] + [
            p["bone"] for p in self.extra
        ]
        self.local_points = [
            np.zeros(3) if end == 0 else self.R[n].T @ (self.t[n] - self.h[n])
            for n in self.names
            for end in (0, 1)
        ] + [np.asarray(p["local"]) for p in self.extra]
        point_lookup = {name: k for k, name in enumerate(self.point_names)}
        self.end_pairs = [
            (point_lookup[f"{n}.end1"], point_lookup[f"{n}.end2"])
            for n in self.names
            if f"{n}.end1" in point_lookup and f"{n}.end2" in point_lookup
        ]
        self.held_guards = []
        self.floor = (
            float(skin.get("floor_z", np.min(np.asarray(skin["rest_verts"])[:, 2])))
            if skin is not None
            else min(p[2] for p in list(self.h.values()) + list(self.t.values()))
        )
        self.capsules = []
        if skin is not None:
            v = np.asarray(skin["rest_verts"])
            ids = np.asarray(skin["skin_indices"])
            ws = np.asarray(skin["skin_weights"])
            skin_index = {n: i for i, n in enumerate(skin["bone_names"])}
            for n in self.names:
                if n not in skin_index:
                    continue
                i = skin_index[n]
                pts = v[np.any((ids == i) & (ws > 0.3), axis=1)]
                if not len(pts):
                    continue
                d = self.t[n] - self.h[n]
                L = np.dot(d, d)
                bone_len = float(np.sqrt(max(L, 1e-10)))
                if meta[n].get("kind") in ("root", "spine", "head") and not n.startswith("Prop_"):
                    nearest = (
                        self.h[n]
                        + np.clip((pts - self.h[n]) @ d / max(L, 1e-10), 0, 1)[:, None] * d
                    )
                    radius = np.median(np.linalg.norm(pts - nearest, axis=1))
                    self.capsules.append((n, float(radius)))
                elif n.startswith(("Hand", "FrontPaw", "LowerArm", "Forearm")) and len(pts) >= 4:
                    dists = np.linalg.norm(pts - self.h[n], axis=1)
                    if float(dists.max()) > 2.5 * bone_len and bone_len > 1e-6:
                        far = pts[int(np.argmax(dists))]
                        local_tip = self.R[n].T @ (far - self.h[n])
                        self.held_guards.append((n, local_tip))
                        self.held_guards.append((n, 0.5 * local_tip))
        self.capsule_exemptions = set()
        if self.capsules:
            p0 = self.start_params()
            X0, J0, _, _ = self.fk(p0, jac=True)
            samples0, _, owners0 = self._guard_samples(X0, J0)
            for cap_name, radius in self.capsules:
                if radius <= 0:
                    continue
                ai = 2 * self.index[cap_name]
                a0, b0 = X0[ai : ai + 2]
                ab0 = b0 - a0
                for k, owner in enumerate(owners0):
                    d0 = samples0[k] - a0
                    t0 = float(np.clip(np.dot(d0, ab0) / max(float(np.dot(ab0, ab0)), 1e-10), 0.0, 1.0))
                    dist0 = float(np.linalg.norm(d0 - t0 * ab0))
                    if dist0 < radius + 0.02 * self.height:
                        self.capsule_exemptions.add((cap_name, k))

    def zero(self):
        return np.clip(np.zeros(self.size), self.lower, self.upper)

    def prior_weights(self, one_view: bool = False) -> np.ndarray:
        """Role-specific rest-pose prior weights (video2rig/creature.py:prior).

        - twist (local Y / axis 1): 25.0
        - hinge non-bend axes: 60.0
        - spine: 15.0
        - clavicle / shoulder: 40.0
        - toe: 30.0
        - loose / tail / secondary: 6.0
        - primary bend / default: 2.0
        - root rotation: 2.0 (single view) or 0.2 (multi-view); root translation: 0.0
        """
        w = np.full(self.size, 2.0, dtype=float)
        for n in self.order:
            entry = self.slices.get(n)
            if entry is None:
                continue
            if isinstance(entry, slice):
                w[entry.start : entry.start + 3] = 2.0 if one_view else 0.2
                w[entry.start + 3 : entry.stop] = 0.0
                continue
            ix, ch = entry
            b = self.meta.get(n, {})
            role = str(b.get("canonical_name", n))
            kind = str(b.get("kind", ""))
            hinge_info = b.get("hinge")
            hinge_axis = None
            if isinstance(hinge_info, dict) and "axis" in hinge_info:
                hinge_axis = int(hinge_info["axis"])
            elif role.startswith(("Shin", "Forearm", "LowerArm", "Tibia")) or b.get("sub_role") == "forearm_hinge":
                roles_map = b.get("axis_roles", {})
                hinge_axis = next(
                    (a for a, key in enumerate(("rx", "ry", "rz")) if roles_map.get(key) == "bend"),
                    0,
                )
            for slot, axis in zip(ix, ch):
                axis = int(axis)
                if axis == 1:
                    w[slot] = 25.0
                elif hinge_axis is not None and axis != hinge_axis:
                    w[slot] = 60.0
                elif kind == "spine" or role.startswith("Spine"):
                    w[slot] = 15.0
                elif role.startswith(("Clavicle", "Shoulder")):
                    w[slot] = 40.0
                elif role.startswith("Toe"):
                    w[slot] = 30.0
                elif kind == "loose" or role.startswith(("Tail", "Ear", "Cape", "Chain")):
                    w[slot] = 6.0
                else:
                    w[slot] = 2.0
        return w

    def start_params(self):
        """Saved p0 rotations and root translations, expressed in the solver's channels."""
        from rig_pose import start_pose_matrices

        matrices = start_pose_matrices(
            {
                n: {**b, "rest_rot_mat_world": self.R[n].tolist(),
                    "pose_basis": self.basis[n].tolist()}
                for n, b in self.meta.items()
            }
        )
        params = self.zero()
        for name in self.order:
            entry = self.slices.get(name)
            if isinstance(entry, slice):
                rotation = matrices[name][:3, :3] @ self.R[name].T
                params[entry.start : entry.start + 3] = Rotation.from_matrix(
                    rotation
                ).as_rotvec()
                params[entry.start + 3 : entry.stop] = (
                    matrices[name][:3, 3] - self.h[name]
                )
            elif entry is not None:
                basis = self.basis[name]
                # Blender stores pose matrices as float32; parent-rest rebasing
                # can leave sub-micrometre local translation roundoff.
                if not np.allclose(basis[:3, 3], 0, atol=1e-6 * self.height):
                    raise ValueError(
                        f"{name}: animated non-root translation {basis[:3, 3].tolist()} needs a rigid prop/root channel"
                    )
                indices, channels = entry
                angles = Rotation.from_matrix(basis[:3, :3]).as_euler("xyz")
                params[indices] = angles[channels]
        return params

    def height_px(self, camera):
        """Reference creature height at this panel's scale, excluding independent props."""
        if camera.type == "ortho":
            return self.height * camera.height / camera.scale
        X = self.fk(self.start_params())[0]
        body = [
            i for i, name in enumerate(self.point_bones) if not name.startswith("Prop_")
        ]
        center = X[body].mean(0) if body else X.mean(0)
        return self.height * camera.focal_px / max(float(camera.depth(center)), 1e-7)

    def fk(self, params, jac=False):
        p = np.asarray(params, float)
        heads = {}
        rots = {}
        dh = {}
        dr = {}
        for n in self.order:
            parent = self.parents[n]
            JH = np.zeros((3, self.size))
            JR = np.zeros((3, 3, self.size))
            if n in self.motion_roots:
                sl = self.slices[n]
                w = p[sl.start : sl.start + 3]
                R = exp_so3(w) @ self.R[n]
                h = self.h[n] + p[sl.start + 3 : sl.stop]
                if jac:
                    E = exp_so3(w)
                    Jr = right_jacobian_so3(w)
                    for k in range(3):
                        JR[:, :, sl.start + k] = E @ skew(Jr[:, k]) @ self.R[n]
                    JH[:, sl.start + 3 : sl.stop] = np.eye(3)
            elif parent is None:
                # Fixed/unweighted independent roots still provide deterministic FK.
                R = self.R[n] @ self.basis[n][:3, :3]
                h = self.h[n] + self.R[n] @ self.basis[n][:3, 3]
            else:
                rel = self.R[parent].T @ self.R[n]
                offset = self.R[parent].T @ (self.h[n] - self.h[parent])
                a = np.zeros(3)
                entry = self.slices.get(n)
                lim = np.asarray(
                    self.meta[n].get("limits_rad", [[0.0, 0.0]] * 3), dtype=float
                )
                a = (
                    np.clip(a, lim[:, 0], lim[:, 1])
                    if not self.meta[n].get("fixed_local")
                    else a
                )
                if entry is not None:
                    ix, ch = entry
                    a[ch] = p[ix]
                if entry is None:
                    offset = offset + rel @ self.basis[n][:3, 3]
                    delta = self.basis[n][:3, :3]
                else:
                    delta, deltas = _euler(a)
                h = heads[parent] + rots[parent] @ offset
                R = rots[parent] @ rel @ delta
                if jac:
                    JH = dh[parent] + np.einsum("ijk,j->ik", dr[parent], offset)
                    JR = np.einsum("ijk,jl->ilk", dr[parent], rel @ delta)
                    if entry is not None:
                        for ixk, c in zip(ix, ch):
                            JR[:, :, ixk] += rots[parent] @ rel @ deltas[c]
            heads[n] = h
            rots[n] = R
            dh[n] = JH
            dr[n] = JR
        pts = np.array(
            [
                heads[n] + rots[n] @ local
                for n, local in zip(self.point_bones, self.local_points)
            ]
        )
        self._last_fk_state = (heads, rots, dh, dr)
        if not jac:
            return pts, heads, rots
        J = np.array(
            [
                dh[n] + np.einsum("ijk,j->ik", dr[n], local)
                for n, local in zip(self.point_bones, self.local_points)
            ]
        )
        return pts, J, heads, rots

    bone_points = fk

    def joint_axes(self, name, params, rotations):
        """Instantaneous world angular axes, one column per local rotation channel."""
        entry = self.slices.get(name)
        if isinstance(entry, slice):
            w = params[entry.start : entry.start + 3]
            return exp_so3(w) @ right_jacobian_so3(w)
        limits = np.asarray(self.meta[name].get("limits_rad", [[0, 0]] * 3))
        angles = (
            np.zeros(3)
            if self.meta[name].get("fixed_local")
            else np.clip(np.zeros(3), limits[:, 0], limits[:, 1])
        )
        if entry is not None:
            angles[entry[1]] = params[entry[0]]
        delta, derivatives = _euler(angles)
        base = rotations[name] @ delta.T
        columns = []
        for derivative in derivatives:
            omega = base @ derivative @ rotations[name].T
            columns.append([omega[2, 1], omega[0, 2], omega[1, 0]])
        return np.asarray(columns).T

    def vector_jacobian(self, params, rotations, name, local):
        """A bone-local direction and its analytic derivative, without adding Reader slots."""
        vector = rotations[name] @ np.asarray(local, float)
        J = np.zeros((3, self.size))
        ancestor = name
        while ancestor is not None:
            entry = self.slices.get(ancestor)
            axes = self.joint_axes(ancestor, params, rotations)
            if isinstance(entry, slice):
                J[:, entry.start : entry.start + 3] = np.cross(axes.T, vector).T
            elif entry is not None:
                indices, channels = entry
                for index, channel in zip(indices, channels):
                    J[:, index] = np.cross(axes[:, channel], vector)
            # A promoted Hips is a world motion root even with an organizational parent.
            if ancestor in self.motion_roots:
                break
            ancestor = self.parents[ancestor]
        return vector, J

    def align_bone_to_tip(self, name, target, params):
        """Fit one direction in its current parent frame, retaining unobserved twist."""
        p = np.asarray(params, float).copy()
        entry = self.slices.get(name)
        if entry is None or isinstance(entry, slice):
            return p
        parent = self.parents[name]
        _, heads, rotations = self.fk(p)
        base = rotations[parent] @ self.R[parent].T @ self.R[name]
        desired = base.T @ (np.asarray(target) - heads[name])
        current = base.T @ rotations[name]
        delta = rotation_between(current[:, 1], desired) @ current
        angles = Rotation.from_matrix(delta).as_euler("xyz")
        ix, ch = entry
        p[ix] = np.clip(angles[ch], self.lower[ix], self.upper[ix])
        return p

    def read_start(self, points, previous=None):
        """Multi-anchor Kabsch roots and parent-local direction alignment initialize the nonlinear fit."""
        p = self.zero() if previous is None else previous.copy()
        pts = np.asarray(points)
        for n in self.order:
            i = self.index[n]
            entry = self.slices.get(n)
            if entry is None:
                continue
            if isinstance(entry, slice):
                anchors = [
                    2 * self.index[a]
                    for a in self.order
                    if a == n or self.parents[a] == n
                ]
                anchors += [2 * i + 1]
                if n.startswith("Prop_"):
                    anchors = [
                        k for k, owner in enumerate(self.point_bones) if owner == n
                    ]
                valid = [a for a in anchors if np.isfinite(pts[a]).all()]
                if len(valid) >= 2:
                    source = np.array(
                        [
                            self.h[self.point_bones[a]]
                            + self.R[self.point_bones[a]] @ self.local_points[a]
                            for a in valid
                        ]
                    )
                    target = pts[valid]
                    A = source - source.mean(0)
                    B = target - target.mean(0)
                    U, singular, V = np.linalg.svd(A.T @ B)
                    if singular[1] > max(singular[0] * 1e-6, 1e-14 * self.height**2):
                        R = V.T @ np.diag([1, 1, np.linalg.det(V.T @ U.T)]) @ U.T
                    else:
                        # A line of anchors cannot observe rotation about that line.
                        # Keep the previous twist rather than accepting an arbitrary SVD spin.
                        R = exp_so3(p[entry.start : entry.start + 3])
                        a, b = np.unravel_index(
                            np.linalg.norm(
                                source[:, None] - source[None], axis=-1
                            ).argmax(),
                            (len(source), len(source)),
                        )
                        R = (
                            rotation_between(
                                R @ (source[b] - source[a]), target[b] - target[a]
                            )
                            @ R
                        )
                    p[entry.start : entry.start + 3] = Rotation.from_matrix(
                        R
                    ).as_rotvec()
                    p[entry.start + 3 : entry.stop] = (
                        target.mean(0) - R @ (source.mean(0) - self.h[n]) - self.h[n]
                    )
                elif np.isfinite(pts[2 * i]).all():
                    p[entry.start + 3 : entry.stop] = pts[2 * i] - self.h[n]
            elif np.isfinite(pts[2 * i : 2 * i + 2]).all():
                # Off-axis surface readings and child attachment offsets observe
                # parent twist. A tip direction alone cannot initialize it.
                anchors = [k for k, owner in enumerate(self.point_bones) if owner == n]
                anchors += [2 * self.index[c] for c in self.order
                            if self.parents[c] == n and c not in self.motion_roots]
                valid = [k for k in anchors if np.isfinite(pts[k]).all()]
                source = np.array([self.h[self.point_bones[k]]
                                   + self.R[self.point_bones[k]] @ self.local_points[k]
                                   for k in valid])
                target = pts[valid]
                A, B = source - source.mean(0), target - target.mean(0)
                U, singular, V = np.linalg.svd(A.T @ B)
                if singular[1] <= max(singular[0] * 1e-6, 1e-14 * self.height**2):
                    p = self.align_bone_to_tip(n, pts[2 * i + 1], p)
                    continue
                world_delta = V.T @ np.diag([1, 1, np.linalg.det(V.T @ U.T)]) @ U.T
                _, _, rotations = self.fk(p)
                parent = self.parents[n]
                base = rotations[parent] @ self.R[parent].T @ self.R[n]
                desired = base.T @ world_delta @ self.R[n]
                angles = Rotation.from_matrix(desired).as_euler("xyz")
                ix, ch = entry
                limits = np.asarray(self.meta[n].get("limits_rad", [[-np.pi, np.pi]] * 3), dtype=float)
                options = []
                for candidate in (angles, np.array([angles[0] + np.pi,
                                                    np.pi - angles[1], angles[2] + np.pi])):
                    current = limits[:, 0].copy()
                    for slot, channel in zip(ix, ch):
                        choices = candidate[channel] + 2 * np.pi * np.arange(-2, 3)
                        current[channel] = min(choices, key=lambda v: (
                            max(self.lower[slot] - v, v - self.upper[slot], 0),
                            abs(v - p[slot])))
                    trial = p.copy()
                    trial[ix] = np.clip(current[ch], self.lower[ix], self.upper[ix])
                    current[ch] = trial[ix]
                    options.append((np.linalg.norm(_euler(current)[0] - desired), trial))
                p = min(options, key=lambda item: item[0])[1]
        return np.clip(p, self.lower, self.upper)

    def refine_read_start(self, points, previous):
        """Resolve chained hinge/twist ambiguity using triangulated readings only.

        Direction alignment cannot observe a thigh's roll until the knee bends.
        Fit the complete observed chain before the robust pixel objective, whose
        wide Cauchy plateaus can otherwise trap a badly folded initial chain.
        """
        target = np.asarray(points, dtype=float)
        valid = np.isfinite(target).all(axis=1)
        if not self.size or not np.any(valid):
            return previous.copy()
        start_points = self.fk(previous)[0]
        if np.max(np.linalg.norm(start_points[valid] - target[valid], axis=1)) < 1e-7 * self.height:
            return previous.copy()
        evaluated = [None, None]

        def objective(q):
            if evaluated[0] is None or not np.array_equal(q, evaluated[0]):
                X, J, _, _ = self.fk(q, jac=True)
                evaluated[:] = [q.copy(), ((X[valid] - target[valid]).ravel() / self.height,
                                          J[valid].reshape(-1, self.size) / self.height)]
            return tuple(value.copy() for value in evaluated[1])

        result = least_squares(
            lambda q: objective(q)[0], np.clip(previous, self.lower, self.upper),
            jac=lambda q: objective(q)[1], bounds=(self.lower, self.upper),
            tr_solver="lsmr", tr_options={"atol": 1e-10, "btol": 1e-10},
            x_scale="jac", max_nfev=80, ftol=1e-10, xtol=1e-10, gtol=1e-10,
        )
        return result.x

    def _guard_samples(self, points, J):
        samples = list(points)
        derivatives = list(J)
        owners = list(self.point_bones)
        for i, name in enumerate(self.names):
            samples.append(0.5 * (points[2 * i] + points[2 * i + 1]))
            derivatives.append(0.5 * (J[2 * i] + J[2 * i + 1]))
            owners.append(name)
        heads, rots, dh, dr = getattr(self, "_last_fk_state", (None, None, None, None))
        for item in getattr(self, "held_guards", ()):
            if len(item) == 3:
                owner, idx_pair, w_pair = item
                samples.append(w_pair[0] * points[idx_pair[0]] + w_pair[1] * points[idx_pair[1]])
                derivatives.append(w_pair[0] * J[idx_pair[0]] + w_pair[1] * J[idx_pair[1]])
                owners.append(owner)
            else:
                owner, local_vec = item
                local_v = np.asarray(local_vec, dtype=float)
                bi = self.index[owner]
                if (
                    heads is not None
                    and owner in heads
                    and np.allclose(points[2 * bi], heads[owner], atol=1e-9)
                ):
                    samples.append(heads[owner] + rots[owner] @ local_v)
                    derivatives.append(dh[owner] + np.einsum("ijk,j->ik", dr[owner], local_v))
                else:
                    bone_len = max(float(np.linalg.norm(self.t[owner] - self.h[owner])), 1e-6)
                    alpha = float(local_v[1]) / bone_len
                    samples.append((1.0 - alpha) * points[2 * bi] + alpha * points[2 * bi + 1])
                    derivatives.append((1.0 - alpha) * J[2 * bi] + alpha * J[2 * bi + 1])
                owners.append(owner)
        return np.asarray(samples), np.asarray(derivatives), owners

    def guard(self, points, J):
        # Endpoints alone miss a long limb or blade passing all the way through a torso.
        # Midpoints and held-weapon reaches (.held_tip / .held_mid) are internal constraint samples.
        samples, derivatives, owners = self._guard_samples(points, J)
        exempt = getattr(self, "capsule_exemptions", ())
        residual = []
        rows = []
        for name, radius in self.capsules:
            if radius <= 0:
                continue
            ai = 2 * self.index[name]
            a, b = points[ai : ai + 2]
            ab = b - a
            for k, owner in enumerate(owners):
                if (
                    owner == name
                    or self.parents.get(owner) == name
                    or self.parents.get(name) == owner
                    or (name, k) in exempt
                ):
                    continue
                if self.meta[owner].get("kind") in ("root", "spine", "head"):
                    continue
                d = samples[k] - a
                t = np.clip(np.dot(d, ab) / max(np.dot(ab, ab), 1e-10), 0, 1)
                diff = d - t * ab
                distance = np.linalg.norm(diff)
                normal = diff / max(distance, 1e-10)
                relative_J = derivatives[k] - (1 - t) * J[ai] - t * J[ai + 1]
                if distance < 1e-10:
                    # Distance has no unique gradient on the capsule centreline. Pick
                    # a deterministic feasible escape; a fixed rest-X normal can be
                    # orthogonal to every movable channel and leave a blade stuck.
                    escape = relative_J.copy()
                    if 0 < t < 1:
                        axis = ab / max(np.linalg.norm(ab), 1e-10)
                        escape -= np.outer(axis, axis @ escape)
                    U, singular, _ = np.linalg.svd(escape, full_matrices=False)
                    normal = (
                        U[:, 0]
                        if len(singular) and singular[0] > 1e-10
                        else self.R[name][:, 0]
                    )
                    nonzero = np.flatnonzero(abs(normal) > 1e-10)
                    if len(nonzero) and normal[nonzero[0]] < 0:
                        normal = -normal
                residual.append(max(0, radius - distance) * 3000 / self.height)
                rows.append(
                    -normal @ relative_J * 3000 / self.height
                    if distance < radius
                    else np.zeros(self.size)
                )
        for point, derivative in zip(samples, derivatives):
            residual.append(max(0, self.floor - point[2]) * 3000 / self.height)
            rows.append(
                -derivative[2] * 3000 / self.height
                if point[2] < self.floor
                else np.zeros(self.size)
            )
        return np.asarray(residual), np.asarray(rows).reshape(-1, self.size)

    def frames(self, sequence, confidence):
        out = []
        for t, p in enumerate(sequence):
            points, heads, rots = self.fk(p)
            bones = {}
            for i, n in enumerate(self.names):
                bones[n] = {
                    "head_3d": points[2 * i].tolist(),
                    "tail_3d": points[2 * i + 1].tolist(),
                    "rot_mat_world": rots[n].tolist(),
                    "confidence": float(np.mean(confidence[t, :, 2 * i : 2 * i + 2])),
                    "kind": self.meta[n].get("kind"),
                    "canonical_name": n,
                }
            out.append({"frame": t, "bones": bones})
        return out


def _geometry_pose(meta):
    """Read saved p0 before skin masses decide which organizational root owns motion."""
    from rig_pose import start_pose_matrices

    rig = RigModel(meta)
    matrices = start_pose_matrices(
        {
            name: {**bone, "rest_rot_mat_world": rig.R[name].tolist()}
            for name, bone in meta.items()
        }
    )
    p0 = rig.zero()
    rotations = {name: matrix[:3, :3] for name, matrix in matrices.items()}
    for name, entry in rig.slices.items():
        if isinstance(entry, slice):
            p0[entry.start : entry.start + 3] = Rotation.from_matrix(
                rotations[name] @ rig.R[name].T
            ).as_rotvec()
        else:
            angles = Rotation.from_matrix(
                np.asarray(meta[name].get("pose_basis", np.eye(4)))[:3, :3]
            ).as_euler("xyz")
            p0[entry[0]] = angles[entry[1]]
    points = np.array(
        [
            matrices[name][:3, 3] + rotations[name] @ local
            for name, local in zip(rig.point_bones, rig.local_points)
        ]
    )
    return rig, p0, points, rotations


def axis_roles(meta, facing=(0, -1, 0), up=(0, 0, 1)):
    """Measure bend/twist/side roles in the saved starting pose, including rolled axes."""
    rig, p0, X, rotations = _geometry_pose(meta)
    up, facing = np.asarray(up, float), np.asarray(facing, float)
    lateral = np.cross(up, facing)
    if (
        not np.isfinite([*up, *facing]).all()
        or min(np.linalg.norm(up), np.linalg.norm(lateral)) < 1e-10
    ):
        raise ValueError(
            "Hinge up and facing directions must be finite and independent"
        )
    up = up / np.linalg.norm(up)
    lateral /= np.linalg.norm(lateral)
    roles = {}
    for name in rig.names:
        if name in rig.motion_roots:
            measured = dict.fromkeys(("rx", "ry", "rz"), "global")
        else:
            i = 2 * rig.index[name]
            along = X[i + 1] - X[i]
            along /= max(np.linalg.norm(along), 1e-10)
            reference = up if abs(along @ lateral) > 0.7 else lateral
            axes = rig.joint_axes(name, p0, rotations)
            bend = max((0, 2), key=lambda axis: abs(axes[:, axis] @ reference))
            measured = {
                key: "twist" if axis == 1 else "bend" if axis == bend else "side"
                for axis, key in enumerate(("rx", "ry", "rz"))
            }
        roles[name] = measured
        meta[name]["axis_roles"] = measured.copy()
        meta[name]["axis_role_source"] = "geometry_start_pose"
    return roles


def find_hinges(meta, facing=(0, -1, 0), up=(0, 0, 1)):
    """Find the geometric bend channel, straight angle and one-way range in final p0."""
    roles = axis_roles(meta, facing, up)
    rig, p0, X, rotations = _geometry_pose(meta)
    facing = np.asarray(facing, float)
    facing /= np.linalg.norm(facing)
    out = {}
    for name, bone in meta.items():
        role = bone.get("canonical_name", name)
        if bone.get("fixed_local") or not (
            role.startswith(("Shin", "Forearm", "LowerArm", "Tibia"))
            or bone.get("sub_role") == "forearm_hinge"
        ):
            continue
        parent = rig.parents[name]
        if parent is None:
            continue
        rest_parent = rig.t[parent] - rig.h[parent]
        rest_child = rig.t[name] - rig.h[name]
        normal = np.cross(rest_parent, rest_child)
        if np.linalg.norm(normal) > math.sin(math.radians(8)) * np.linalg.norm(
            rest_parent
        ) * np.linalg.norm(rest_child):
            axis = max((0, 2), key=lambda a: abs(rig.R[name][:, a] @ normal))
            axis_source = "rest_bend_plane"
        else:
            axis = next(
                a
                for a, key in enumerate(("rx", "ry", "rz"))
                if roles[name][key] == "bend"
            )
            axis_source = "start_pose_axis_role"
        entry = rig.slices.get(name)
        if entry is None or isinstance(entry, slice) or axis not in entry[1]:
            continue
        existing = np.asarray(bone.get("limits_rad", [[-np.pi, np.pi]] * 3), float)
        angles = np.clip(np.zeros(3), existing[:, 0], existing[:, 1])
        angles[entry[1]] = p0[entry[0]]
        base = rotations[name] @ _euler(angles)[0].T
        shaft = rig.R[name].T @ rest_child
        shaft /= max(np.linalg.norm(shaft), 1e-10)
        j = 2 * rig.index[parent]
        parent_direction = X[j + 1] - X[j]
        parent_direction /= max(np.linalg.norm(parent_direction), 1e-10)

        def direction(value):
            trial = angles.copy()
            trial[axis] = value
            return base @ _euler(trial)[0] @ shaft

        # Holding the other two Euler channels fixed gives A*cos(a)+B*sin(a)+C.
        # Its maximum is the exact straightening angle, without half-degree quantization.
        d0 = float(direction(0) @ parent_direction)
        dpi = float(direction(np.pi) @ parent_direction)
        cosine = (d0 - dpi) / 2
        sine = float(direction(np.pi / 2) @ parent_direction) - (d0 + dpi) / 2
        if math.hypot(cosine, sine) < 1e-8:
            continue
        straight = math.atan2(sine, cosine)
        start_angle = angles[axis]
        bend = math.acos(
            float(np.clip(direction(start_angle) @ parent_direction, -1, 1))
        )
        knee = role.startswith(("Shin", "Tibia"))
        plus = direction(start_angle + 0.15)
        minus = direction(start_angle - 0.15)
        if bend > math.radians(12) + 1e-10:
            sign = 1 if plus @ parent_direction < minus @ parent_direction else -1
            direction_source = "start_pose_fold"
        else:
            displacement = float((plus - minus) @ facing)
            sign = (
                (1 if displacement < 0 else -1)
                if knee
                else (1 if displacement > 0 else -1)
            )
            direction_source = "knee_back" if knee else "elbow_front"
        intervals = []
        for winding in (-1, 0, 1):
            center = straight + 2 * np.pi * winding
            limits = sorted((center - sign * 0.05, center + sign * 2.7))
            gap = max(limits[0] - start_angle, start_angle - limits[1], 0)
            intervals.append((gap, abs(center - start_angle), center, limits))
        _, _, straight, anatomical = min(intervals, key=lambda value: value[:2])
        old = np.asarray(bone.get("limits_rad", [[-np.pi, np.pi]] * 3), float).copy()
        bounds = (
            anatomical.copy()
            if bone.get("limits_source") in ("canonical_default", "stage1_heuristic")
            else [max(anatomical[0], old[axis, 0]), min(anatomical[1], old[axis, 1])]
        )
        if bounds[1] <= bounds[0]:
            raise ValueError(
                f"{name}: geometric hinge and existing limits do not intersect"
            )
        old[axis] = bounds
        bone["limits_rad"] = old.tolist()
        out[name] = {
            "axis": int(axis),
            "axis_name": ("rx", "ry", "rz")[axis],
            "axis_source": axis_source,
            "direction_source": direction_source,
            "straight": float(straight),
            "rest_bend": bend,
            "start_angle": float(start_angle),
            "hinge_sign": sign,
            "anatomical_limits": anatomical,
            "limits": bounds,
        }
        bone["hinge"] = out[name]
    configure_jaw_hinge(meta, up=up)
    return out


def configure_jaw_hinge(meta, up=(0, 0, 1)):
    """One-way downward opening bounds [-0.05, +0.70] rad on the Jaw opening axis."""
    rig_arg = meta if isinstance(meta, RigModel) else None
    if rig_arg is not None:
        meta = rig_arg.meta
    up_vec = np.asarray(up, float)
    up_norm = np.linalg.norm(up_vec)
    if up_norm < 1e-10:
        return {}
    up_vec = up_vec / up_norm
    rig, p0, X, rotations = _geometry_pose(meta)
    jaw_out = {}
    for name, bone in meta.items():
        role = bone.get("canonical_name", name)
        if bone.get("fixed_local") or not (name == "Jaw" or role == "Jaw" or name.startswith("Jaw.")):
            continue
        entry = rig.slices.get(name)
        if entry is None or isinstance(entry, slice):
            continue
        existing = np.asarray(bone.get("limits_rad", [[-np.pi, np.pi]] * 3), float)
        angles = np.clip(np.zeros(3), existing[:, 0], existing[:, 1])
        angles[entry[1]] = p0[entry[0]]
        base = rotations[name] @ _euler(angles)[0].T
        rest_vec = rig.t[name] - rig.h[name]
        if np.linalg.norm(rest_vec) < 1e-8:
            continue
        shaft = rig.R[name].T @ rest_vec

        def tip_offset(axis_idx, delta_rad):
            trial = angles.copy()
            trial[axis_idx] += delta_rad
            return base @ _euler(trial)[0] @ shaft - base @ _euler(angles)[0] @ shaft

        candidates = [a for a in (0, 2) if a in entry[1]]
        if not candidates:
            continue
        open_axis = max(
            candidates,
            key=lambda a: abs(float((tip_offset(a, 0.15) - tip_offset(a, -0.15)) @ (-up_vec))),
        )
        downward_disp = float((tip_offset(open_axis, 0.15) - tip_offset(open_axis, -0.15)) @ (-up_vec))
        sign = 1 if downward_disp >= 0.0 else -1
        center = float(angles[open_axis])
        open_bounds = sorted((center - sign * 0.05, center + sign * 0.70))
        new_lims = np.array([[-0.08, 0.08], [-0.08, 0.08], [-0.08, 0.08]], dtype=float)
        for ch_i in range(3):
            c_ang = float(angles[ch_i])
            new_lims[ch_i] = [c_ang - 0.08, c_ang + 0.08]
        new_lims[open_axis] = open_bounds
        if bone.get("limits_source") in ("canonical_default", "stage1_heuristic", None):
            bone["limits_rad"] = new_lims.tolist()
            bone["limits_source"] = "geometry_jaw_hinge"
            if rig_arg is not None and name in rig_arg.slices:
                r_idx, r_ch = rig_arg.slices[name]
                for i_slot, c_slot in zip(r_idx, r_ch):
                    rig_arg.lower[i_slot] = new_lims[c_slot, 0]
                    rig_arg.upper[i_slot] = new_lims[c_slot, 1]
        jaw_out[name] = {
            "axis": int(open_axis),
            "axis_name": ("rx", "ry", "rz")[open_axis],
            "hinge_sign": int(sign),
            "anatomical_limits": open_bounds,
            "limits": open_bounds,
        }
        bone["jaw_hinge"] = jaw_out[name]
        bone["hinge"] = jaw_out[name]
    return jaw_out


def _cauchy(r, scale):
    q = r / scale
    val = scale * np.sqrt(np.log1p(q * q))
    transformed = np.sign(r) * val
    derivative = np.ones_like(r)
    nz = np.abs(r) > 1e-8
    derivative[nz] = np.abs(r[nz]) / ((1 + q[nz] ** 2) * np.maximum(val[nz], 1e-12))
    return transformed, derivative


def robust_cauchy_loss(c_scale, n_robust: int | None = None):
    """SciPy `least_squares` Cauchy loss callable (video2rig/reader/fit.py:robust).

    Returns a callable `loss(z)` mapping squared residuals `z = f_i^2` to a `(3, M)`
    array `[rho(z), rho'(z), rho''(z)]`:
      - `rho(z) = c^2 * log(1 + z / c^2)`
      - `rho'(z) = 1 / (1 + z / c^2)`
      - `rho''(z) = -1 / (c^2 * (1 + z / c^2)^2)`
    When `n_robust` is provided (hybrid mode), only the first `n_robust` data
    residuals use the Cauchy kernel while physical/prior constraints `z[n_robust:]`
    remain exact quadratic L2 (`rho(z) = z`, `rho'(z) = 1`, `rho''(z) = 0`).
    """
    c_arr = np.asarray(c_scale, dtype=float)

    def _loss(z):
        z = np.asarray(z, dtype=float)
        out = np.empty((3, len(z)), dtype=float)
        if n_robust is None:
            c2 = np.broadcast_to(c_arr * c_arr, z.shape)
            u = 1.0 + z / np.maximum(c2, 1e-12)
            out[0] = c2 * np.log(u)
            out[1] = 1.0 / u
            out[2] = -1.0 / (np.maximum(c2, 1e-12) * u * u)
            return out
        n = min(int(n_robust), len(z))
        if n > 0:
            z_r = z[:n]
            c2 = np.broadcast_to(c_arr * c_arr, z_r.shape)
            u = 1.0 + z_r / np.maximum(c2, 1e-12)
            out[0, :n] = c2 * np.log(u)
            out[1, :n] = 1.0 / u
            out[2, :n] = -1.0 / (np.maximum(c2, 1e-12) * u * u)
        if n < len(z):
            out[0, n:] = z[n:]
            out[1, n:] = 1.0
            out[2, n:] = 0.0
        return out

    return _loss


def _stable_readings(rig, cameras, uv, confidence):
    """Only sure, smooth tracks widen the robust kernel; scale motion by creature height."""
    stable = np.zeros_like(confidence, dtype=bool)
    if len(uv) < 3:
        return stable
    height_px = np.array([rig.height_px(camera) for camera in cameras])
    acceleration = np.linalg.norm(uv[2:] - 2 * uv[1:-1] + uv[:-2], axis=-1)
    stable[1:-1] = (
        (acceleration < 0.02 * height_px[:, None])
        & (confidence[:-2] >= 0.6)
        & (confidence[1:-1] >= 0.6)
        & (confidence[2:] >= 0.6)
    )
    stable[0], stable[-1] = stable[1], stable[-2]
    return stable


def _flat_foot_vectors(rig, reference):
    """Unit lateral axes of feet resting flat in p0, including feet with no surface labels."""
    X, _, rotations = rig.fk(reference)
    feet = []
    for name in rig.names:
        if not name.startswith(("Foot.", "FrontPaw.", "HindPaw.", "Tarsus")):
            continue
        i = 2 * rig.index[name]
        direction = X[i + 1] - X[i]
        length = np.linalg.norm(direction)
        if (
            length < 1e-10
            or X[i + 1, 2] - rig.floor >= 0.05 * rig.height
            or abs(direction[2]) / length > 0.8
        ):
            continue
        lateral = np.cross([0, 0, 1], direction / length)
        lateral /= np.linalg.norm(lateral)
        feet.append((name, rotations[name].T @ lateral))
    return feet


def _foot_roll_observed(rig, name, readings, confidence):
    """A flat-contact prior must not override multi-view evidence of foot roll."""
    anchors = [k for k, owner in enumerate(rig.point_bones) if owner == name]
    anchors += [2 * rig.index[n] for n in rig.order
                if rig.parents[n] == name and n not in rig.motion_roots]
    valid = [k for k in anchors if np.isfinite(readings[k]).all()
             and np.count_nonzero(confidence[:, k] >= 0.6) >= 2]
    if len(valid) < 3:
        return False
    source = np.array([rig.h[rig.point_bones[k]]
                       + rig.R[rig.point_bones[k]] @ rig.local_points[k]
                       for k in valid])
    singular = np.linalg.svd(source - source.mean(0), compute_uv=False)
    return bool(singular[1] > max(singular[0] * 1e-6, rig.height * 1e-8))


def _resolve_frame_end_pairs(rig, cameras, uv_t, conf_t, previous_points):
    """Dynamic symmetric prop endpoint (end1/end2) swap resolution per view."""
    end_pairs = getattr(rig, "end_pairs", ())
    if not end_pairs:
        return uv_t, conf_t
    uv_out = uv_t.copy()
    conf_out = conf_t.copy()
    for i1, i2 in end_pairs:
        vis_views = [v for v in range(len(cameras)) if conf_out[v, i1] > 0.2 and conf_out[v, i2] > 0.2]
        if len(vis_views) >= 2:
            def _pair_triang_err(u_cand, c_cand):
                X_tri = triangulate(cameras, u_cand, c_cand, previous_points)
                if not np.isfinite(X_tri[[i1, i2]]).all():
                    return float("inf")
                errs = [
                    np.linalg.norm(cameras[v].project(X_tri[[i1, i2]]) - u_cand[v, [i1, i2]], axis=-1).mean()
                    for v in vis_views
                ]
                return float(np.mean(errs))

            base_err = _pair_triang_err(uv_out, conf_out)
            if base_err > 6.0:
                best_u, best_c, best_err = uv_out, conf_out, base_err
                for v_swap in vis_views:
                    u_try = uv_out.copy()
                    c_try = conf_out.copy()
                    u_try[v_swap, [i1, i2]] = u_try[v_swap, [i2, i1]]
                    c_try[v_swap, [i1, i2]] = c_try[v_swap, [i2, i1]]
                    err_try = _pair_triang_err(u_try, c_try)
                    if err_try + 2.0 < best_err:
                        best_u, best_c, best_err = u_try, c_try, err_try
                uv_out, conf_out = best_u, best_c
            # Align 3D end1/end2 label orientation to the closer previous 3D endpoints.
            X_tri = triangulate(cameras, uv_out, conf_out, previous_points)
            owner = rig.point_bones[i1]
            anchors = [k for k, name in enumerate(rig.point_bones)
                       if name == owner and k not in (i1, i2)
                       and np.count_nonzero(conf_out[:, k] >= 0.5) >= 2
                       and np.isfinite(X_tri[k]).all()]
            if anchors:
                local = rig.local_points
                def shape_error(first, second):
                    return sum(abs(np.linalg.norm(X_tri[k] - X_tri[observed])
                                   - np.linalg.norm(local[k] - local[expected]))
                               for k in anchors
                               for observed, expected in ((first, i1), (second, i2)))
                keep_error = shape_error(i1, i2)
                flip_error = shape_error(i2, i1)
                if abs(keep_error - flip_error) > 1e-4 * rig.height:
                    if flip_error < keep_error:
                        uv_out[:, [i1, i2]] = uv_out[:, [i2, i1]]
                        conf_out[:, [i1, i2]] = conf_out[:, [i2, i1]]
                    continue
            if np.isfinite(X_tri[[i1, i2]]).all() and np.isfinite(previous_points[[i1, i2]]).all():
                d_keep = np.linalg.norm(X_tri[i1] - previous_points[i1]) + np.linalg.norm(X_tri[i2] - previous_points[i2])
                d_flip = np.linalg.norm(X_tri[i2] - previous_points[i1]) + np.linalg.norm(X_tri[i1] - previous_points[i2])
                if d_flip + 0.05 * rig.height < d_keep:
                    uv_out[:, [i1, i2]] = uv_out[:, [i2, i1]]
                    conf_out[:, [i1, i2]] = conf_out[:, [i2, i1]]
        elif len(vis_views) == 1 and np.isfinite(previous_points[[i1, i2]]).all():
            v0 = vis_views[0]
            proj_prev = cameras[v0].project(previous_points[[i1, i2]])
            d_keep = np.linalg.norm(uv_out[v0, i1] - proj_prev[0]) + np.linalg.norm(uv_out[v0, i2] - proj_prev[1])
            d_flip = np.linalg.norm(uv_out[v0, i2] - proj_prev[0]) + np.linalg.norm(uv_out[v0, i1] - proj_prev[1])
            if d_flip + 4.0 < d_keep:
                uv_out[v0, [i1, i2]] = uv_out[v0, [i2, i1]]
                conf_out[v0, [i1, i2]] = conf_out[v0, [i2, i1]]
    return uv_out, conf_out


def body_picture(rig, cameras, p, uv_t, scale=0.5, window=2):
    """Mesh Z-buffer bone-ownership check at pose `p` (video2rig/reader/fit.py:body_picture).
    Returns `(elsewhere, own)` boolean arrays of shape `(V, K)`.
    """
    from pose_features import lbs_deform_vertices_numpy
    from reader_v3 import bone_neighbours, rasterize_triangles_numpy, triangle_bone_owners

    V = len(cameras)
    K = len(rig.point_names)
    elsewhere = np.zeros((V, K), dtype=bool)
    own = np.zeros((V, K), dtype=bool)
    skin = getattr(rig, "skin", None)
    if skin is None or not len(skin.get("triangles", ())):
        return elsewhere, own
    _, heads, rots = rig.fk(p)
    skin_bones = list(skin.get("bone_names", rig.names))
    posed = lbs_deform_vertices_numpy(
        np.asarray(skin["rest_verts"], dtype=float),
        np.asarray(skin["skin_indices"], dtype=int),
        np.asarray(skin["skin_weights"], dtype=float),
        np.asarray(skin["rest_heads"], dtype=float),
        np.asarray(skin["rest_R"], dtype=float),
        np.array([heads.get(n, rig.h.get(n, np.zeros(3))) for n in skin_bones], dtype=float),
        np.array([rots.get(n, rig.R.get(n, np.eye(3))) for n in skin_bones], dtype=float),
    )
    tris = np.asarray(skin["triangles"], dtype=int)
    tri_owner = triangle_bone_owners(skin, num_verts=len(posed))
    name_to_idx = {str(n): i for i, n in enumerate(skin_bones)}
    actor_bone_ids = {i for i, n in enumerate(skin_bones) if not str(n).startswith("Prop_")}
    allowed_by_k = [
        {
            name_to_idx[n]
            for n in bone_neighbours(rig.meta, str(rig.point_bones[k]))
            if n in name_to_idx
        }
        for k in range(K)
    ]
    for v, cam in enumerate(cameras):
        W = max(int(round(cam.width * scale)), 1)
        H = max(int(round(cam.height * scale)), 1)
        tri_buf, _, _ = rasterize_triangles_numpy(
            cam.project(posed) * scale, cam.depth(posed), tris, (W, H)
        )
        owner_buf = np.where(tri_buf >= 0, tri_owner[np.maximum(tri_buf, 0)], -1)
        for k in range(K):
            if not np.isfinite(uv_t[v, k]).all():
                continue
            x, y = int(uv_t[v, k, 0] * scale), int(uv_t[v, k, 1] * scale)
            if not (0 <= x < W and 0 <= y < H):
                continue
            win = owner_buf[
                max(0, y - window) : min(H, y + window + 1),
                max(0, x - window) : min(W, x + window + 1),
            ]
            win_pos = win[win >= 0]
            own[v, k] = bool(len(win_pos)) and any(int(o) in allowed_by_k[k] for o in win_pos)
            body_win = [int(o) for o in win_pos if int(o) in actor_bone_ids]
            elsewhere[v, k] = bool(len(body_win)) and not any(o in allowed_by_k[k] for o in body_win)
    return elsewhere, own


def _apply_body_picture_weights(
    rig,
    cameras,
    prev_p_or_X,
    uv_t,
    conf_t,
    weights,
    hidden_weight=0.1,
    zero_contradicted=False,
):
    """Downweight or zero far-side limb keypoints landing on another body part (`elsewhere`) when another view sees them."""
    if len(cameras) < 2:
        return weights
    out_w = weights.copy()
    arr = np.asarray(prev_p_or_X, dtype=float)
    if arr.ndim == 1 and len(arr) == rig.size:
        prev_p = arr
        X_ref = rig.fk(prev_p)[0]
    else:
        prev_p = rig.start_params()
        X_ref = arr

    skin = getattr(rig, "skin", None)
    if skin is not None and len(skin.get("triangles", ())):
        elsewhere, own = body_picture(rig, cameras, prev_p, uv_t)
        for v in range(len(cameras)):
            for k, owner in enumerate(rig.point_bones):
                if rig.meta[owner].get("kind") in ("root", "spine", "head") or owner.startswith("Prop_"):
                    continue
                if not elsewhere[v, k] or conf_t[v, k] <= 0.0:
                    continue
                other_sees = any(
                    conf_t[v2, k] >= 0.3 and (own[v2, k] or not elsewhere[v2, k])
                    for v2 in range(len(cameras))
                    if v2 != v
                )
                if not other_sees:
                    continue
                X_tri = triangulate(cameras, uv_t, conf_t)
                disagrees = True
                if np.isfinite(X_tri[k]).all():
                    reproj = max(
                        float(np.linalg.norm(cameras[v2].project(X_tri[k]) - uv_t[v2, k]))
                        for v2 in range(len(cameras))
                        if conf_t[v2, k] >= 0.3
                    )
                    disagrees = reproj > 0.03 * rig.height_px(cameras[v])
                if disagrees or conf_t[v, k] < 0.5:
                    out_w[v, k] = 0.0 if zero_contradicted else out_w[v, k] * hidden_weight
        return out_w

    if not getattr(rig, "capsules", None):
        return out_w
    for cap_name, radius in rig.capsules:
        if radius <= 0:
            continue
        ai = 2 * rig.index[cap_name]
        a3, b3 = X_ref[ai], X_ref[ai + 1]
        for v, cam in enumerate(cameras):
            pa, pb = cam.project(np.array([a3, b3]))
            ab2 = pb - pa
            ab2_len2 = float(np.dot(ab2, ab2))
            rad_px = radius * rig.height_px(cam) / max(rig.height, 1e-6)
            for k, owner in enumerate(rig.point_bones):
                if rig.meta[owner].get("kind") in ("root", "spine", "head") or owner.startswith("Prop_"):
                    continue
                if conf_t[v, k] <= 0.1 or not any(conf_t[v2, k] >= 0.5 for v2 in range(len(cameras)) if v2 != v):
                    continue
                s = float(np.clip(np.dot(uv_t[v, k] - pa, ab2) / max(ab2_len2, 1e-10), 0.0, 1.0))
                dist_2d = float(np.linalg.norm((uv_t[v, k] - pa) - s * ab2))
                if dist_2d >= 0.85 * rad_px:
                    continue
                cap_pt = (1.0 - s) * a3 + s * b3
                if float(cam.depth(X_ref[k])) > float(cam.depth(cap_pt)) + 0.35 * radius:
                    X_tri = triangulate(cameras, uv_t, conf_t)
                    disagrees = conf_t[v, k] < 0.5
                    if not disagrees and np.isfinite(X_tri[k]).all():
                        reproj = max(
                            float(np.linalg.norm(cameras[v2].project(X_tri[k]) - uv_t[v2, k]))
                            for v2 in range(len(cameras))
                            if conf_t[v2, k] >= 0.3
                        )
                        disagrees = reproj > 0.03 * rig.height_px(cam)
                    if disagrees:
                        out_w[v, k] *= 0.25
    return out_w


def fit_clip_2d_reprojection(
    rig,
    cameras,
    uv,
    confidence,
    cauchy_px=6.0,
    stable_scale=2.0,
    flat_feet=120.0,
    read_one_panel=True,
    max_nfev=100,
    who_sees="model",
    w_prior: float = 0.0,
    depth: np.ndarray | None = None,
    w_depth: float = 0.2,
    depth_where: str = "unseen",
    robust_mode: str = "analytic_cauchy",
    zero_contradicted_elsewhere: bool = False,
):
    """Fit pixels directly; independent starts compete on the same complete robust objective."""
    cameras = list(cameras)
    if (
        not np.isfinite([cauchy_px, stable_scale, flat_feet]).all()
        or cauchy_px <= 0
        or stable_scale < 1
        or flat_feet < 0
    ):
        raise ValueError(
            "Solver scales must be finite: Cauchy > 0, stable >= 1, feet >= 0"
        )
    uv = np.asarray(uv, float)
    raw_confidence = np.asarray(confidence, float)
    conf = np.where(np.isfinite(raw_confidence), np.clip(raw_confidence, 0, 1), 0.0)
    if (
        uv.ndim != 4
        or uv.shape[:3] != conf.shape
        or uv.shape[1] != len(cameras)
        or uv.shape[2] != len(rig.point_names)
    ):
        raise ValueError(
            "Expected observations [T,V,K,2], confidences [T,V,K] matching the rig"
        )
    conf = np.where(np.isfinite(uv).all(-1), conf, 0).copy()
    uv = np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0).copy()
    depth_arr = np.asarray(depth, dtype=float) if depth is not None else None
    params = []
    diagnostics = []
    reference = rig.start_params()
    previous_points = rig.fk(reference)[0]
    previous_error = np.inf
    stable = _stable_readings(rig, cameras, uv, conf)
    feet = _flat_foot_vectors(rig, reference) if flat_feet else []
    prop_roots = [n for n in rig.order if n in rig.motion_roots and n.startswith("Prop_")]
    prior_w = (
        rig.prior_weights(one_view=(len(cameras) == 1)) * (float(w_prior) / 2.0)
        if w_prior > 0
        else None
    )
    body_indices = np.array(
        [
            k
            for k, owner in enumerate(rig.point_bones)
            if rig.meta.get(owner, {}).get("kind") in ("root", "spine", "head")
        ],
        dtype=int,
    )
    if not len(body_indices):
        body_indices = np.arange(min(2 * len(rig.names), len(rig.point_names)), dtype=int)
    kpx_by_cam = [float(rig.height_px(c)) / max(float(rig.height), 1e-6) for c in cameras]

    with threadpool_limits(limits=1):
        for t in range(len(uv)):
            prev = params[-1] if params else reference
            if getattr(rig, "end_pairs", ()):
                uv[t], conf[t] = _resolve_frame_end_pairs(
                    rig, cameras, uv[t], conf[t], previous_points
                )
            scales = np.where(stable[t], cauchy_px * stable_scale, cauchy_px)[..., None]
            # Kernel widening alone can still prefer a higher-confidence
            # jittering panel. Smooth sure tracks also receive the configured
            # evidence weight, so independent starts use the intended objective.
            weights = (conf[t] * np.where(stable[t], stable_scale, 1.0))[..., None]
            if who_sees == "model" and (getattr(rig, "capsules", ()) or getattr(rig, "skin", None) is not None):
                weights = _apply_body_picture_weights(
                    rig,
                    cameras,
                    prev,
                    uv[t],
                    conf[t],
                    weights,
                    zero_contradicted=zero_contradicted_elsewhere,
                )
            read = triangulate(
                cameras, uv[t], conf[t], previous_points if read_one_panel else None
            )
            # Triangulation fills unseen slots from the previous frame. Those
            # placeholders are not observations of a bone's roll or attachment.
            read[np.max(conf[t], axis=0) <= 0] = np.nan
            planted = []
            for name, local in feet:
                tip = 2 * rig.index[name] + 1
                # Hold contact membership fixed during a solve: tilting/lifting cannot turn
                # the residual off. Visible airborne feet are exempt in their current frame.
                z = (
                    read[tip, 2]
                    if np.isfinite(read[tip]).all()
                    else previous_points[tip, 2]
                )
                if (z - rig.floor < 0.05 * rig.height
                        and not _foot_roll_observed(rig, name, read, conf[t])):
                    planted.append((name, local))

            use_scipy_loss = robust_mode in ("scipy_hybrid", "scipy_video2rig")
            data_scales = []
            for v in range(len(cameras)):
                data_scales.extend(np.broadcast_to(scales[v], (len(rig.point_names), 2)).ravel())
            if depth_arr is not None and w_depth > 0:
                for v in range(len(cameras)):
                    data_scales.extend(scales[v, :, 0].ravel())
            data_scales_arr = np.asarray(data_scales, dtype=float)
            n_data_residuals = len(data_scales_arr)
            evaluated = [None, None]

            def objective(p):
                if evaluated[0] is not None and np.array_equal(p, evaluated[0]):
                    return tuple(value.copy() for value in evaluated[1])
                X, J, _, rotations = rig.fk(p, jac=True)
                residual = []
                rows = []
                for v, c in enumerate(cameras):
                    u, P = c.project(X, jac=True)
                    r = u - uv[t, v]
                    if use_scipy_loss:
                        residual.extend((r * weights[v]).ravel())
                        rows.extend(
                            (
                                np.einsum("kij,kjl->kil", P, J)
                                * weights[v, :, None]
                            ).reshape(-1, rig.size)
                        )
                    else:
                        cr, derivative = _cauchy(r, np.broadcast_to(scales[v], r.shape))
                        residual.extend((cr * weights[v]).ravel())
                        rows.extend(
                            (
                                np.einsum("kij,kjl->kil", P, J)
                                * derivative[..., None]
                                * weights[v, :, None]
                            ).reshape(-1, rig.size)
                        )
                if depth_arr is not None and w_depth > 0:
                    mid = X[body_indices].mean(axis=0)
                    J_mid = J[body_indices].mean(axis=0)
                    for v, c in enumerate(cameras):
                        cam_fwd = np.asarray(c.R[2], dtype=float)
                        cam_fwd = cam_fwd / max(float(np.linalg.norm(cam_fwd)), 1e-12)
                        kpx = kpx_by_cam[v]
                        d_obs = np.nan_to_num(depth_arr[t, v], nan=0.0, posinf=0.0, neginf=0.0)
                        dw = conf[t, v] * float(w_depth)
                        if depth_where == "unseen" and len(cameras) >= 2:
                            other_max = np.max(
                                [conf[t, v2] for v2 in range(len(cameras)) if v2 != v],
                                axis=0,
                            )
                            dw = dw * np.clip(1.0 - other_max / 0.35, 0.0, 1.0)
                        r_d = kpx * ((X - mid) @ cam_fwd - d_obs * rig.height)
                        J_d = kpx * np.einsum("j,kjl->kl", cam_fwd, J - J_mid[None, :, :])
                        if use_scipy_loss:
                            residual.extend((r_d * dw).ravel())
                            rows.extend((J_d * dw[:, None]).reshape(-1, rig.size))
                        else:
                            cr_d, deriv_d = _cauchy(r_d, scales[v, :, 0])
                            residual.extend((cr_d * dw).ravel())
                            rows.extend((J_d * (deriv_d * dw)[:, None]).reshape(-1, rig.size))
                guard, G = rig.guard(X, J)
                residual.extend(guard)
                rows.extend(G)
                for name, local in planted:
                    vector, VJ = rig.vector_jacobian(p, rotations, name, local)
                    residual.append(flat_feet * vector[2])
                    rows.append(flat_feet * VJ[2])
                if prior_w is not None:
                    residual.extend(prior_w * (p - reference))
                    rows.extend(np.diag(prior_w))
                residual.extend((p - prev) * 0.03)
                rows.extend(np.eye(rig.size) * 0.03)
                result = np.asarray(residual), np.asarray(rows)
                evaluated[:] = [p.copy(), tuple(value.copy() for value in result)]
                return result

            read_cand = rig.refine_read_start(read, rig.read_start(read, prev))
            starts = [("prev", prev), ("read", read_cand)]
            if not np.allclose(prev, reference, atol=1e-8, rtol=0):
                starts.append(("reference", reference))
            # A preceding extreme pose can preserve the wrong unobserved twist
            # in a line-only chain. Reinitialize from readings and saved p0 too.
            read_reference = rig.refine_read_start(read, rig.read_start(read, reference))
            if not np.allclose(read_reference, read_cand, atol=1e-8, rtol=0):
                starts.append(("read_reference", read_reference))
            if prop_roots:
                prop_placed = prev.copy()
                changed_prop = False
                for pr in prop_roots:
                    sl = rig.slices[pr]
                    if np.linalg.norm(read_cand[sl] - prev[sl]) > 1e-4:
                        prop_placed[sl] = read_cand[sl]
                        changed_prop = True
                if changed_prop:
                    starts.append(("prop_place", prop_placed))

            loss_fn = (
                robust_cauchy_loss(
                    data_scales_arr if robust_mode == "scipy_hybrid" else cauchy_px,
                    n_robust=n_data_residuals if robust_mode == "scipy_hybrid" else None,
                )
                if use_scipy_loss
                else "linear"
            )

            def run(start):
                label, p = start
                ls_kwargs = {}
                if use_scipy_loss:
                    ls_kwargs["loss"] = loss_fn
                    ls_kwargs["x_scale"] = "jac"
                result = least_squares(
                    lambda q: objective(q)[0],
                    np.clip(p, rig.lower, rig.upper),
                    jac=lambda q: objective(q)[1],
                    bounds=(rig.lower, rig.upper),
                    max_nfev=max_nfev,
                    **ls_kwargs,
                )
                return float(result.cost), label, result.x

            # A complete, high-confidence multi-view fit already at subpixel
            # precision needs no costly escape starts. Noisy/hidden readings
            # retain all independent starts and the identical objective.
            read_result = run(("read", read_cand))
            read_points, read_jac, _, _ = rig.fk(read_result[2], jac=True)
            read_guard = rig.guard(read_points, read_jac)[0]
            dense_fit = (len(cameras) >= 2 and np.all(conf[t] >= .99)
                         and prior_w is None
                         and max(float(np.linalg.norm(c.project(read_points) - uv[t, v], axis=-1).max())
                                 for v, c in enumerate(cameras)) < .01
                         and float(np.max(read_guard, initial=0.0)) < 1e-6)
            candidates = [read_result]
            if not dense_fit:
                candidates.extend(run(s) for s in starts if s[0] != "read")
            best = min(candidates, key=lambda x: x[0])

            def error(p):
                X = rig.fk(p)[0]
                errors = np.stack(
                    [
                        np.linalg.norm(c.project(X) - uv[t, v], axis=-1)
                        for v, c in enumerate(cameras)
                    ]
                )
                return (
                    float(np.median(errors[conf[t] > 0.5]))
                    if np.any(conf[t] > 0.5)
                    else float("inf")
                )

            current_error = error(best[2])
            if not dense_fit and (current_error > 1.5 * previous_error or not params):
                reset = prev.copy()
                for entry in rig.slices.values():
                    if not isinstance(entry, slice):
                        reset[entry[0]] = reference[entry[0]]
                candidates.append(run(("reset", reset)))
                best = min(candidates, key=lambda x: x[0])
                current_error = error(best[2])
            pre_turn_error = current_error
            root = next(
                (
                    n
                    for n in rig.order
                    if n in rig.motion_roots and not n.startswith("Prop_")
                ),
                None,
            )
            if current_error > 8.0 and np.any(conf[t] > 0.5) and root is not None:
                sl = rig.slices[root]
                escape_base = best[2].copy()
                for angle in (-45, 45, -90, 90, 180):
                    start = escape_base.copy()
                    R = exp_so3([0, 0, math.radians(angle)]) @ exp_so3(
                        start[sl.start : sl.start + 3]
                    )
                    start[sl.start : sl.start + 3] = Rotation.from_matrix(R).as_rotvec()
                    candidates.append(run((f"turn_{angle}", start)))
                best = min(candidates, key=lambda x: x[0])
                current_error = error(best[2])
            params.append(best[2])
            previous_points = rig.fk(best[2])[0]
            previous_error = current_error
            diagnostics.append(
                {
                    "start": best[1],
                    "median_error_px": current_error
                    if np.isfinite(current_error)
                    else None,
                    "observed_points": int(np.count_nonzero(conf[t] > 0.5)),
                    "stable_points": int(np.count_nonzero(stable[t])),
                    "planted_feet": [n for n, _ in planted],
                    "pre_turn_error_px": pre_turn_error
                    if np.isfinite(pre_turn_error)
                    else None,
                    "starts": [c[1] for c in candidates],
                    "candidate_costs": {c[1]: c[0] for c in candidates},
                }
            )
    values = np.array(params).reshape(len(uv), rig.size)
    for n, sl in rig.slices.items():
        if isinstance(sl, slice) and len(values):
            values[:, sl.start : sl.start + 3] = unwrap_rotvecs(
                values[:, sl.start : sl.start + 3]
            )
    return values, {
        "solver": "v3-cauchy-2d",
        "frames": diagnostics,
        "blas_threads": 1,
        "determinism": "fixed inputs/environment; cross-platform bitwise identity is not promised",
    }


def sample_v3_pose(
    meta, eulers, rng, skin=None, cameras=None, events=None, diagnostics=None,
    root_offsets=None,
):
    """Sample anatomical contacts and calibrated root motion with explicit event evidence."""
    from pose_sampling_v3 import (
        sampling_events,
        body_surface_targets,
        posed_surface,
        sample_roaming_translation,
    )

    rig = RigModel(
        meta, names=skin["bone_names"] if skin is not None else None, skin=skin
    )
    report = {} if diagnostics is None else diagnostics
    events = dict(sampling_events(rng) if events is None else events)
    report["events"] = events
    report["motion_roots"] = sorted(rig.motion_roots)
    report["fitted_bones"] = sorted(
        n for n, entry in rig.slices.items()
        if isinstance(entry, slice) or len(entry[0]) > 0
    )
    p = rig.zero()
    root = next(
        n for n in rig.order if n in rig.motion_roots and not n.startswith("Prop_")
    )
    for n, entry in rig.slices.items():
        if not isinstance(entry, slice):
            ix, ch = entry
            p[ix] = np.clip(
                np.asarray(eulers.get(n, [0, 0, 0]))[ch], rig.lower[ix], rig.upper[ix]
            )
    hands = [n for n in rig.names if n.startswith(("Hand", "FrontPaw", "WingHand"))]
    targets = {}
    surface = (
        body_surface_targets(rig, p, skin, rng)
        if hands and events.get("reach")
        else np.empty((0, 3))
    )
    X = rig.fk(p)[0]
    if hands and events.get("reach") == "together":
        chest = [n for n in rig.names if meta[n].get("kind") == "spine"]
        body = rig.t[chest[-1]] if chest else rig.t[root]
        if len(surface):
            nearby = surface[abs(surface[:, 2] - body[2]) < 0.20 * rig.height]
            if len(nearby):
                body = nearby[np.argmin(nearby[:, 1])]
        center = body + np.array([0, -0.02, 0]) * rig.height
        report["reach_center"] = center.tolist()
        for n in hands:
            targets[2 * rig.index[n] + 1] = (
                center
                + np.array([0.02 if n.endswith(".L") else -0.02, 0, 0]) * rig.height
            )
    elif hands and events.get("reach") == "onto_body" and len(surface):
        n = rng.choice(hands)
        point = 2 * rig.index[n] + 1
        # Closest of sampled real surface points biases toward anatomically reachable contact.
        target = surface[np.argmin(np.linalg.norm(surface - X[point], axis=-1))]
        targets[point] = target
    moving = []
    for hand in hands:
        name = hand
        while name in rig.meta and name not in rig.motion_roots:
            entry = rig.slices.get(name)
            if entry is not None and not isinstance(entry, slice):
                moving.extend(entry[0])
            name = rig.parents[name]
    moving = np.unique(moving).astype(int)
    if targets and len(moving):
        initial = p.copy()

        def objective(q):
            full = initial.copy()
            full[moving] = q
            X, J, _, _ = rig.fk(full, True)
            residual = np.r_[
                np.concatenate(
                    [(X[k] - target) / rig.height for k, target in targets.items()]
                ),
                0.01 * (q - initial[moving]),
            ]
            jac = np.vstack(
                [
                    np.concatenate(
                        [J[k][:, moving] / rig.height for k in targets], axis=0
                    ),
                    np.eye(len(moving)) * 0.01,
                ]
            )
            return residual, jac

        solved = least_squares(
            lambda q: objective(q)[0],
            p[moving],
            jac=lambda q: objective(q)[1],
            bounds=(rig.lower[moving], rig.upper[moving]),
            max_nfev=60,
        )
        p[moving] = solved.x
        actual = rig.fk(p)[0]
        report["reach_targets"] = {
            rig.point_names[k]: target.tolist() for k, target in targets.items()
        }
        report["reach_error_H"] = {
            rig.point_names[k]: float(np.linalg.norm(actual[k] - target) / rig.height)
            for k, target in targets.items()
        }
    sl = rig.slices[root]
    root_euler = np.array(eulers.get(root, [0, 0, 0]), float)
    if events.get("yaw"):
        root_euler[2] = rng.uniform(-np.pi, np.pi)
    if events.get("pitch"):
        root_euler[0] = float(np.clip(rng.gauss(0, math.radians(30)), -math.radians(75), math.radians(75)))
    p[sl.start : sl.start + 3] = Rotation.from_euler("xyz", root_euler).as_rotvec()
    translation = np.zeros(3)
    if skin is not None:
        vertices = posed_surface(rig, p, skin)
        actor_bones = np.array([not n.startswith("Prop_") for n in rig.names])
        actor_mass = (
            np.asarray(skin["skin_weights"])
            * actor_bones[np.asarray(skin["skin_indices"])]
        ).sum(-1)
        vertices = vertices[actor_mass > 0.5]
        if not len(vertices):
            raise ValueError("Sampling requires a deforming character surface")
        # Root tilting changes the body's lowest point; jumping is measured above its grounded pose.
        translation[2] = max(0.0, rig.floor - float(vertices[:, 2].min()))
        center = vertices.mean(0) + translation
    else:
        center = np.mean(rig.fk(p)[0], axis=0)
    jump = rng.uniform(0, 0.30) * rig.height if events.get("jump") else 0.0
    translation[2] += jump
    center[2] += jump
    if events.get("roam"):
        if cameras is None:
            raise ValueError("V3 roaming requires calibrated cameras")
        translation += sample_roaming_translation(
            list(cameras), center, rig.height, rng
        )
    if root_offsets is not None:
        translation = np.asarray(root_offsets.get(root, [0, 0, 0]), dtype=float)
    p[sl.start + 3 : sl.stop] = translation
    report["roaming_center_world"] = (center + np.r_[translation[:2], 0.0]).tolist()
    if events.get("jaw"):
        for n in rig.names:
            entry = rig.slices.get(n)
            if (
                n.startswith("Jaw")
                and entry is not None
                and not isinstance(entry, slice)
            ):
                ix, ch = entry
                jaw_meta = rig.meta.get(n, {}).get("jaw_hinge") or rig.meta.get(n, {}).get("hinge") or {}
                open_ax = int(jaw_meta.get("axis", 0))
                if open_ax not in ch and len(ch):
                    open_ax = int(ch[0])
                if open_ax in ch:
                    slot = int(ix[np.flatnonzero(ch == open_ax)[0]])
                    p[slot] = rng.uniform(rig.lower[slot], rig.upper[slot])
    X, heads, rots = rig.fk(p)
    report.update(
        root_world_euler=root_euler.tolist(),
        motion_root_world_eulers={root: root_euler.tolist()},
        jump_height_H=jump / rig.height,
        translation=translation.tolist(),
    )
    props = {}
    held = {}
    for n in rig.names:
        if not n.startswith("Prop_") or n not in rig.motion_roots:
            continue
        entry = rig.slices[n]
        if root_offsets is not None:
            target = rig.h[n] + np.asarray(root_offsets.get(n, [0, 0, 0]), dtype=float)
            prop_euler = np.asarray(eulers.get(n, [0, 0, 0]), dtype=float)
            rotation = Rotation.from_euler("xyz", prop_euler).as_matrix()
            report["motion_root_world_eulers"][n] = prop_euler.tolist()
        elif events.get("hold_prop") and hands:
            hand = rng.choice(hands)
            target = (
                X[2 * rig.index[hand] + 1]
                + np.array([rng.uniform(-0.015, 0.015) for _ in range(3)]) * rig.height
            )
            rotation = rots[hand]
            held[n] = {
                "hand": hand,
                "grip_distance_H": float(
                    np.linalg.norm(target - X[2 * rig.index[hand] + 1]) / rig.height
                ),
            }
        else:
            target = (
                heads[root]
                + np.array([rng.uniform(-0.4, 0.4) for _ in range(3)]) * rig.height
            )
            rotation = Rotation.from_euler(
                "xyz", [rng.uniform(-np.pi, np.pi) for _ in range(3)]
            ).as_matrix()
        if root_offsets is None:
            report["motion_root_world_eulers"][n] = Rotation.from_matrix(rotation).as_euler("xyz").tolist()
        p[entry.start : entry.start + 3] = Rotation.from_matrix(
            rotation @ rig.R[n].T
        ).as_rotvec()
        p[entry.start + 3 : entry.stop] = target - rig.h[n]
        props[n] = (target - rig.h[n]).tolist()
    _, heads, rots = rig.fk(p)
    out = {}
    locations = {}
    for n in rig.names:
        parent = rig.parents[n]
        base = rots[parent] @ rig.R[parent].T @ rig.R[n] if parent else rig.R[n]
        base_head = (
            heads[parent] + rots[parent] @ rig.R[parent].T @ (rig.h[n] - rig.h[parent])
            if parent else rig.h[n]
        )
        out[n] = Rotation.from_matrix(base.T @ rots[n]).as_euler("xyz").tolist()
        location = base.T @ (heads[n] - base_head)
        if np.linalg.norm(location) > 1e-12:
            locations[n] = location.tolist()
    report["held_props"] = held
    report["local_translations"] = locations
    return out, translation.tolist(), props
