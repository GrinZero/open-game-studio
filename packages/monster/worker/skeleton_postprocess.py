"""Geometry-only skeleton preparation, independent of bpy, torch and cloud services."""

from copy import deepcopy
from dataclasses import dataclass
import heapq
import math
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree


@dataclass
class Shell:
    surface: np.ndarray
    solid: np.ndarray
    origin: np.ndarray
    step: float
    distance: np.ndarray

    def indices(self, points):
        return np.rint((np.asarray(points) - self.origin) / self.step).astype(int)

    def contains(self, points):
        i = self.indices(points)
        good = ((i >= 0) & (i < np.array(self.solid.shape))).all(axis=-1)
        answer = np.zeros(i.shape[:-1], bool)
        answer[good] = self.solid[tuple(i[good].T)]
        return answer

    def radius(self, points):
        i = np.clip(self.indices(points), 0, np.array(self.solid.shape) - 1)
        return self.distance[tuple(i.T)] * self.step

    def mirror_overlap(self, axis, plane=None):
        """Volume IoU against reflection, including reflected voxels outside this grid."""
        occupied = np.argwhere(self.solid)
        if not len(occupied):
            return 0.0
        coordinate = (
            (occupied[:, axis].min() + occupied[:, axis].max()) / 2
            if plane is None
            else (plane - self.origin[axis]) / self.step
        )
        reflected = np.rint(2 * coordinate - np.arange(self.solid.shape[axis])).astype(
            int
        )
        valid = (reflected >= 0) & (reflected < self.solid.shape[axis])
        shape = [1] * 3
        shape[axis] = len(valid)
        mirror = np.take(
            self.solid, np.clip(reflected, 0, self.solid.shape[axis] - 1), axis=axis
        ) & valid.reshape(shape)
        intersection = np.count_nonzero(mirror & self.solid)
        count = np.count_nonzero(self.solid)
        return float(intersection / max(1, 2 * count - intersection))

    def save(self, path, vertices, triangles):
        np.savez_compressed(
            path,
            vertices=vertices,
            triangles=triangles,
            thin=self.surface,
            solid=self.solid,
            origin=self.origin,
            voxel=self.step,
        )


def voxelize(vertices, triangles, voxel=None):
    """Conservative triangle sampling plus closing/flood fill; retain the original thin shell."""
    v, f = np.asarray(vertices, float), np.asarray(triangles, int)
    if len(v) < 3 or not np.isfinite(v).all():
        raise ValueError("A finite surface mesh is required")
    height = max(float(np.ptp(v[:, 2])), 1e-5)
    step = float(voxel or 0.01 * height)
    origin = v.min(0) - 3 * step
    shape = np.ceil((v.max(0) - origin) / step).astype(int) + 4
    if np.prod(shape, dtype=np.int64) > 64_000_000:
        raise ValueError("Shell grid exceeds 64M cells; increase voxel size")
    surface = np.zeros(tuple(shape), bool)
    vertex_indices = np.rint((v - origin) / step).astype(int)
    surface[tuple(vertex_indices.T)] = True
    for tri in f:
        a, b, c = v[tri]
        n = max(
            1,
            int(
                np.ceil(
                    max(
                        np.linalg.norm(a - b),
                        np.linalg.norm(a - c),
                        np.linalg.norm(b - c),
                    )
                    / (step * 0.6)
                )
            ),
        )
        # Vectorize one barycentric row; memory is bounded even for very long triangles.
        for i in range(n + 1):
            u = i / n
            vv = np.arange(n - i + 1) / n
            pts = a + (b - a) * u + vv[:, None] * (c - a)
            ix = np.rint((pts - origin) / step).astype(int)
            surface[tuple(ix.T)] = True
    closed = ndimage.binary_closing(surface, iterations=1)
    solid = ndimage.binary_fill_holes(closed) | surface
    return Shell(surface, solid, origin, step, ndimage.distance_transform_edt(solid))


def find_facing(vertices, triangles=None, shell=None):
    v = np.asarray(vertices, float)
    H = max(np.ptp(v[:, 2]), 1e-5)
    mid = np.median(v, axis=0)
    # A bilateral axis has the smallest normalized mirrored nearest-neighbour error.
    tree = cKDTree(v)
    scores = []
    for axis in (0, 1):
        mirror = v.copy()
        mirror[:, axis] = 2 * mid[axis] - mirror[:, axis]
        scores.append(np.mean(tree.query(mirror)[0]) / H)
    if shell is None:
        shell = voxelize(
            v, np.empty((0, 3), dtype=int) if triangles is None else triangles, 0.01 * H
        )
    overlap = [shell.mirror_overlap(axis) for axis in (0, 1)]
    lateral = int(np.argmax(overlap))
    forward = 1 - lateral
    z = (v[:, 2] - v[:, 2].min()) / H
    top, feet, ankle = (
        v[z >= 0.97, forward],
        v[z <= 0.04, forward],
        v[(z >= 0.06) & (z <= 0.15), forward],
    )
    head = np.median(top) - mid[forward] if len(top) else 0
    toe = (np.mean(feet) - np.mean(ankle)) if len(feet) and len(ankle) else 0
    sign = 1 if head + toe >= 0 else -1
    return {
        "forward": ("+" if sign > 0 else "-") + "XY"[forward],
        "yaw_to_minus_y_deg": float(
            (
                -90
                - math.degrees(
                    math.atan2(sign if forward == 1 else 0, sign if forward == 0 else 0)
                )
                + 180
            )
            % 360
            - 180
        ),
        "mirror_error": float(min(scores)),
        "symmetry": float(overlap[lateral]),
        "mirror_overlap_by_axis": {"X": overlap[0], "Y": overlap[1]},
        "lateral_axis": "XY"[lateral],
    }


def _footprints(shell, *, thin_legs=False):
    occupied = np.argwhere(shell.solid)
    if not len(occupied):
        return []
    floor = occupied[:, 2].min()
    slab = shell.solid[:, :, floor : floor + 3].any(2)
    if not thin_legs:
        # Split floor contacts using actual volumetric columns before labeling.
        # Labeling the thin skin first merges feet joined by a skirt hem, then
        # rejects the whole merged region because its median lies in the cloth.
        slab &= (shell.distance[:, :, floor : floor + 8] >= 1.8).any(2)
    labels, count = ndimage.label(slab)
    feet = []
    for k in range(1, count + 1):
        xy = np.argwhere(labels == k)
        if len(xy) < (1 if thin_legs else 3):
            continue
        center = np.rint(np.median(xy, axis=0)).astype(int)
        # Reject 1-2 voxel sheets (skirts); require a volumetric interior above the floor.
        column = shell.distance[
            max(0, center[0] - 1) : center[0] + 2,
            max(0, center[1] - 1) : center[1] + 2,
            floor : floor + 8,
        ]
        if column.size and column.max() >= (0.9 if thin_legs else 1.8):
            feet.append(shell.origin + np.r_[center, floor + 1] * shell.step)
    return feet


def centre_tree(shell, root, ends, *, volume=None):
    """One Dijkstra tree through the medial skeleton, with interior fallback for thin limbs."""
    from skimage.morphology import skeletonize

    center = skeletonize(shell.solid if volume is None else volume)
    medial = np.argwhere(center)
    if len(medial) == 0:
        return [] if volume is None else _interior_paths(shell, root, ends)
    tree = cKDTree(medial)
    start = tuple(medial[tree.query(shell.indices(root))[1]])
    targets = [tuple(medial[tree.query(shell.indices(e))[1]]) for e in ends]
    # Skeleton voxels occasionally disconnect at a one-voxel diagonal. 26 neighbours preserve them.
    offsets = [
        np.array([x, y, z])
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        for z in (-1, 0, 1)
        if (x, y, z) != (0, 0, 0)
    ]
    distances, prev, queue = {start: 0.0}, {}, [(0.0, start)]
    remaining = set(targets)
    while queue and remaining:
        dist, a = heapq.heappop(queue)
        if dist != distances.get(a):
            continue
        remaining.discard(a)
        for off in offsets:
            b_arr = np.array(a) + off
            if (b_arr < 0).any() or (b_arr >= np.array(center.shape)).any():
                continue
            b = tuple(b_arr)
            if not center[b]:
                continue
            w = np.linalg.norm(off) * (
                1 + 3 / (0.5 * (shell.distance[a] + shell.distance[b]) + 0.5)
            )
            new = dist + w
            if new < distances.get(b, float("inf")):
                distances[b] = new
                prev[b] = a
                heapq.heappush(queue, (new, b))
    paths = []
    for end, goal in zip(targets, ends):
        if end not in distances:
            if volume is not None:
                return _interior_paths(shell, root, ends)
            continue
        path = [end]
        while path[-1] != start:
            path.append(prev[path[-1]])
        path = shell.origin + np.array(path[::-1]) * shell.step
        if volume is not None:
            # A medial endpoint lies inside a rounded paw. Keep the actual
            # contact as the transplanted endpoint, rather than stopping high.
            length = float(np.linalg.norm(path[-1] - goal))
            height = np.ptp(np.argwhere(shell.solid)[:, 2]) * shell.step
            bridge = np.linspace(
                path[-1], goal, max(2, int(np.ceil(length / shell.step)) + 1)
            )
            if (
                length > max(4 * shell.step, 0.08 * height)
                or not shell.contains(bridge).all()
            ):
                return _interior_paths(shell, root, ends)
            path = np.r_[path, bridge[1:]]
        paths.append(path)
    return paths


def _resample(path, fractions):
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    return np.array(
        [np.interp(np.asarray(fractions) * d[-1], d, path[:, i]) for i in range(3)]
    ).T


def _interior_paths(shell, root, ends):
    """Surface-repelled shortest paths through the volume, not a thinned graph."""
    occupied = np.argwhere(shell.solid)
    if not len(occupied) or not len(ends):
        return []
    ids = np.full(shell.solid.shape, -1, dtype=np.int32)
    ids[tuple(occupied.T)] = np.arange(len(occupied))
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
        costs.append(
            np.linalg.norm(delta) * (1 + 3.0 / (0.5 * (depth[a] + depth[b]) + 0.5))
        )
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


def _leg_joint_fractions(path, step, height):
    """Find one significant bend near each planned joint, measured along the arc."""
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
        result.append(
            float(fractions[best]) if bends[best] > math.radians(12) else float(desired)
        )
    return np.asarray(result)


def _paired_leg_paths(left, right, plane, height):
    """Monotone body-root matching leaves missing counterparts as explicit gaps."""
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


def free_limb_paths(shell, paths):
    """Remove shared trunk routes and the thick body interior before joint fitting."""
    counts = {}
    for path in paths:
        for point in np.unique(shell.indices(path), axis=0):
            key = tuple(point)
            counts[key] = counts.get(key, 0) + 1
    output = []
    for path in paths:
        common = [i for i, p in enumerate(shell.indices(path)) if counts[tuple(p)] > 1]
        radius = shell.radius(path)
        # A limb can branch off inside a thick torso. Keep the branch's outward
        # section, rather than spending its knee/ankle slots traversing the torso.
        core = max(
            float(np.quantile(radius[: max(1, len(path) // 3)], 0.8)), 2 * shell.step
        )
        distal = float(np.median(radius[max(0, len(path) * 2 // 3) :]))
        thick = (
            np.flatnonzero(
                (radius >= 0.65 * core) & (np.arange(len(path)) < 0.8 * len(path))
            )
            if core > 1.5 * max(distal, shell.step)
            else []
        )
        start = max(common[-1] if common else 0, int(thick[-1]) if len(thick) else 0)
        start = min(start, max(0, len(path) - 5))
        output.append(path[start:])
    return output


def manylegs_build(shell, root):
    feet = _footprints(shell, thin_legs=True)
    if len(feet) < 6:
        return {}
    paths = _interior_paths(shell, root, feet)
    if len(paths) < 6:
        return {}
    # Extract the free limb after the body junction, preserving each side's measured arc shape.
    paths = free_limb_paths(shell, paths)
    height = max(np.ptp(np.argwhere(shell.solid)[:, 2]) * shell.step, shell.step)
    # Grow the thick body core to include crowded limb roots. A local radius
    # threshold alone leaves long tapered trunk segments in the front/back legs.
    core = shell.distance >= 0.5 * shell.distance.max()
    body = (
        ndimage.distance_transform_edt(~core) <= 1.2 * shell.distance.max()
    ) & shell.solid
    for i, path in enumerate(paths):
        inside = np.flatnonzero(body[tuple(shell.indices(path).T)])
        if not len(inside):
            continue
        start = int(inside[-1])
        remainder = np.linalg.norm(np.diff(path[start:], axis=0), axis=1).sum()
        # Uniform-width branching creatures have no distinct thick torso;
        # retain their measured shared-branch cut rather than erasing the legs.
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


def rest_coords(
    posed_vertices, skin_indices, skin_weights, rest_matrices, posed_matrices
):
    """Invert the blended affine transform per vertex (inverse LBS, not blended inverses)."""
    transforms = np.asarray(posed_matrices) @ np.linalg.inv(rest_matrices)
    blend = (
        transforms[np.asarray(skin_indices)] * np.asarray(skin_weights)[..., None, None]
    ).sum(1)
    v = np.c_[posed_vertices, np.ones(len(posed_vertices))]
    if np.any(np.abs(np.linalg.det(blend[:, :3, :3])) < 1e-10):
        raise ValueError(
            "Singular inverse LBS transform; preserve source and repair weights"
        )
    return np.linalg.solve(blend, v[..., None])[:, :3, 0]


def cross_section_flatness(
    shell, head, tail, trunk_points=(), volume_points=None, reference_height=None
):
    """Median width/thickness over four thin cuts through the limb volume.

    Whole-segment covariance confuses a swept/bent arm with a wing. Each cut is
    only three voxels deep; trunk-nearer voxels never contribute to the limb.
    """
    head, tail = np.asarray(head), np.asarray(tail)
    axis = tail - head
    length = np.linalg.norm(axis)
    if length < shell.step:
        return 1.0
    axis = axis / length
    voxels = (
        shell.origin + np.argwhere(shell.solid) * shell.step
        if volume_points is None
        else volume_points
    )
    if len(voxels) < 5:
        return 1.0
    height = (
        reference_height
        if reference_height is not None
        else np.ptp(np.argwhere(shell.solid)[:, 2]) * shell.step
    )
    delta = voxels - head
    along = delta @ axis
    radial = np.linalg.norm(delta - along[:, None] * axis, axis=1)
    use = (along >= 0) & (along <= length) & (radial < 0.35 * height)
    if len(trunk_points):
        use &= radial < cKDTree(trunk_points).query(voxels)[0]
    values = []
    for fraction in (0.30, 0.45, 0.60, 0.75):
        cut = delta[use & (abs(along - fraction * length) <= 1.5 * shell.step)]
        if len(cut) < 5:
            continue
        cut = cut - (cut @ axis)[:, None] * axis
        eigen = np.linalg.eigvalsh(np.cov(cut, rowvar=False))
        values.append(float(np.sqrt(eigen[-1] / max(eigen[-2], shell.step**2 / 12))))
    return float(np.median(values)) if values else 1.0


def jaw_from_mouth_volume(shell, head, tail, mid_x):
    """Find a front-open air pocket bounded by actual upper and lower lip volume.

    Candidate space is near the head segment on the midsagittal plane. Sealed
    cavities, the air above/below the skull, and sparse surface vertex gaps do
    not qualify. Coordinates here follow the cleaned character's -Y front.
    """
    head, tail = np.asarray(head, float), np.asarray(tail, float)
    length = np.linalg.norm(tail - head)
    if length < 4 * shell.step:
        return None
    x = int(
        np.clip(
            round((mid_x - shell.origin[0]) / shell.step), 1, shell.solid.shape[0] - 2
        )
    )
    occupied = shell.solid[x - 1 : x + 2].any(0)
    iy, iz = np.indices(occupied.shape)
    points = np.stack(
        [
            np.full_like(iy, mid_x, dtype=float),
            shell.origin[1] + iy * shell.step,
            shell.origin[2] + iz * shell.step,
        ],
        -1,
    )
    direction = tail - head
    alpha = np.clip(((points - head) @ direction) / max(length**2, 1e-10), 0, 1)
    near = (
        np.linalg.norm(points - (head + alpha[..., None] * direction), axis=-1)
        < 0.8 * length
    ) & (points[..., 1] < head[1] + 0.25 * length)
    lower = np.maximum.accumulate(np.where(occupied, iz, -1), axis=1)
    upper = np.minimum.accumulate(
        np.where(occupied, iz, occupied.shape[1])[..., ::-1], axis=1
    )[..., ::-1]
    bounded = (lower >= 0) & (upper < occupied.shape[1])
    gap = (~occupied) & bounded & near
    labels, count = ndimage.label(gap)
    choices = []
    for label in range(1, count + 1):
        ys, zs = np.where(labels == label)
        if len(ys) < 6 or (ys.max() - ys.min() + 1) * shell.step < 0.25 * length:
            continue
        front = ys.min()
        # At least half the opening rows must have a clear path out of the face.
        opening = zs[ys == front]
        if np.mean(~occupied[:front, opening].any(0)) < 0.5:
            continue
        back = ys.max()
        hinge = np.array(
            [
                mid_x,
                shell.origin[1] + back * shell.step,
                shell.origin[2] + zs[ys == back].mean() * shell.step,
            ]
        )
        floor = int(lower[front, opening.min()])
        if floor < 0:
            continue
        # Follow the same lower lip forward; do not extend down into a chest.
        chin = front
        while chin > 0 and occupied[chin - 1, max(0, floor - 1) : floor + 2].any():
            chin -= 1
        tip = np.array(
            [
                mid_x,
                shell.origin[1] + chin * shell.step,
                shell.origin[2] + floor * shell.step,
            ]
        )
        if (
            np.linalg.norm(tip - hinge) >= 0.3 * length
            and np.linalg.norm(tip - tail) < 0.8 * length
        ):
            choices.append((len(ys), hinge, tip))
    if not choices:
        return None
    _, hinge, tip = max(choices, key=lambda x: x[0])
    return hinge, tip


def align_quadruped_hips(meta, shell, height):
    """Place each hind hip above its knee, just below the local spinal axis."""
    role = lambda n: str(meta[n].get("canonical_name", n))
    tails = [n for n in meta if role(n).startswith("Tail")]
    spine = [
        n
        for n in meta
        if meta[n].get("kind") in ("root", "spine") and not role(n).startswith("Root")
    ]
    if not spine:
        spine = [n for n in meta if meta[n].get("kind") in ("root", "spine")]
    if not tails or not spine:
        return []
    rear = min(float(meta[n]["rest_head_world"][1]) for n in tails)
    changed = []
    for name in meta:
        if not role(name).startswith("Thigh"):
            continue
        head = np.array(meta[name]["rest_head_world"], float)
        knee = np.array(meta[name]["rest_tail_world"], float)
        y = min(knee[1], rear - 0.02 * height)
        locations = []
        for n in spine:
            a = np.asarray(meta[n]["rest_head_world"])
            d = np.asarray(meta[n]["rest_tail_world"]) - a
            fraction = np.clip((y - a[1]) / d[1], 0, 1) if abs(d[1]) > 1e-10 else 0.5
            p = a + fraction * d
            locations.append((abs(p[1] - y), p))
        axis = min(locations, key=lambda item: item[0])[1]
        radius = float(shell.radius(axis[None])[0])
        head[1] = y
        head[2] = axis[2] - 0.25 * radius
        meta[name]["rest_head_world"] = head.tolist()
        changed.append(name)
    return changed


def refine_long_tail(meta, height):
    """Repartition a rear axial extension and/or a curved tail by measured arc length."""
    role = lambda n: str(meta[n].get("canonical_name", n))
    tails = {n for n in meta if role(n).startswith("Tail")}
    starts = [n for n in tails if meta[n].get("parent") not in tails]
    # Multiple independent tails keep their topology for anatomy-specific treatment.
    if len(starts) > 1:
        return None
    chain = []
    if starts:
        name = starts[0]
        while name:
            chain.append(name)
            children = [n for n in tails if meta[n].get("parent") == name]
            if len(children) > 1:
                return None
            name = children[0] if children else None
    axial = None
    hind = [
        b["rest_head_world"][1] for n, b in meta.items() if role(n).startswith("Thigh")
    ]
    if hind:
        candidates = []
        for n, b in meta.items():
            if role(n) != "Hips" and not role(n).startswith("Spine"):
                continue
            a, t = (
                np.array(b["rest_head_world"], float),
                np.array(b["rest_tail_world"], float),
            )
            if t[1] <= a[1] or t[1] <= max(hind):
                continue
            fraction = np.clip((max(hind) + 0.08 * height - a[1]) / (t[1] - a[1]), 0, 1)
            start = a + fraction * (t - a)
            if np.linalg.norm(t - start) > height:
                candidates.append((np.linalg.norm(t - start), n, start, t))
        if candidates:
            _, n, start, end = max(candidates, key=lambda item: item[0])
            if not chain or meta[chain[0]].get("parent") == n:
                axial = (n, start, end)
    if not chain and axial is None:
        return None
    points = []
    if axial:
        points = [axial[1], axial[2]]
    for n in chain:
        a = np.array(meta[n]["rest_head_world"], float)
        t = np.array(meta[n]["rest_tail_world"], float)
        if not points or np.linalg.norm(points[-1] - a) > 1e-8:
            points.append(a)
        points.append(t)
    points = np.array(points)
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if length <= height:
        return None
    count = min(10, max(2, int(np.ceil(length / (0.16 * height)))))
    sampled = _resample(points, np.linspace(0, 1, count + 1))
    parent = axial[0] if axial else meta[chain[0]].get("parent")
    if axial:
        meta[parent]["rest_tail_world"] = sampled[0].tolist()
    template = deepcopy(meta[chain[0]]) if chain else {"kind": "tail", "side": "C"}
    attachments = {
        n: np.array(b["rest_head_world"])
        for n, b in meta.items()
        if b.get("parent") in chain and n not in chain
    }
    for n in chain:
        del meta[n]
    names = [f"Tail{i + 1}" for i in range(count)]
    for i, name in enumerate(names):
        meta[name] = {
            **deepcopy(template),
            "canonical_name": name,
            "parent": parent if i == 0 else names[i - 1],
            "rest_head_world": sampled[i].tolist(),
            "rest_tail_world": sampled[i + 1].tolist(),
        }
    for n, point in attachments.items():
        nearest = int(
            np.linalg.norm((sampled[:-1] + sampled[1:]) / 2 - point, axis=1).argmin()
        )
        meta[n]["parent"] = names[nearest]
    return {
        "old_names": chain,
        "new_names": names,
        "arc_length": length,
        "axial": None if axial is None else axial[0],
    }


def _remove_bone(meta, name, removed):
    parent = meta[name].get("parent")
    removed[name] = parent
    for bone in meta.values():
        if bone.get("parent") == name:
            bone["parent"] = parent
    del meta[name]


def fold_short_digits(meta, height):
    """Fold complete short digit branches; long fingers keep every segment and attachment."""
    removed = {}
    children = {n: [c for c in meta if meta[c].get("parent") == n] for n in meta}
    roots = [
        n
        for n in meta
        if meta[n].get("kind") == "digit"
        and meta.get(meta[n].get("parent"), {}).get("kind") != "digit"
    ]
    folded = set()
    tips = {}
    for root in roots:
        stack = [(root, 0.0)]
        branch, furthest, endpoint = [], 0.0, None
        while stack:
            name, distance = stack.pop()
            bone = meta[name]
            head, tail = (
                np.asarray(bone["rest_head_world"]),
                np.asarray(bone["rest_tail_world"]),
            )
            distance += float(np.linalg.norm(tail - head))
            branch.append(name)
            if distance >= furthest:
                furthest, endpoint = distance, tail.copy()
            for child in children[name]:
                if meta[child].get("kind") == "digit":
                    gap = np.linalg.norm(
                        np.asarray(meta[child]["rest_head_world"]) - tail
                    )
                    stack.append((child, distance + gap))
        if furthest < 0.22 * height - 1e-12 * height:
            folded.add(root)
            tips[root] = endpoint
            for name in branch:
                _remove_bone(meta, name, removed)
    # A fully folded finger fan ends at the mean of its distal tips, not at the wrist.
    for parent, branches in children.items():
        if parent in meta and len(branches) >= 2 and all(n in folded for n in branches):
            meta[parent]["rest_tail_world"] = np.mean(
                [tips[n] for n in branches], axis=0
            ).tolist()
    return removed


def drop_outside(meta, shell):
    """Measure outside arc share; retract protruding tips while retaining at least 30% length."""
    removed, retracted = {}, []
    for name in list(meta):
        bone = meta[name]
        head, tail = (
            np.asarray(bone["rest_head_world"], float),
            np.asarray(bone["rest_tail_world"], float),
        )
        delta = tail - head
        count = max(
            20, min(1024, 2 * math.ceil(np.linalg.norm(delta) / (2 * shell.step)))
        )
        midpoint_inside = shell.contains(
            head + ((np.arange(count) + 0.5) / count)[:, None] * delta
        )
        if midpoint_inside.mean() < 0.5 and bone.get("parent"):
            _remove_bone(meta, name, removed)
            continue
        if shell.contains(tail[None])[0]:
            continue
        locations = np.linspace(0, 1, count + 1)
        inside = shell.contains(head + locations[:, None] * delta)
        if not inside.any():
            continue
        last = int(np.flatnonzero(inside)[-1])
        lower, upper = locations[last], locations[min(last + 1, count)]
        for _ in range(16):
            middle = (lower + upper) / 2
            if shell.contains((head + middle * delta)[None])[0]:
                lower = middle
            else:
                upper = middle
        fraction = max(0.3, lower)
        bone["rest_tail_world"] = (head + fraction * delta).tolist()
        retracted.append(name)
    return removed, retracted


def cut_held_prop(points, radii, height, voxel):
    """Split a long contiguous thin suffix after the fist; short claws are never weapons."""
    points, radii = np.asarray(points, float), np.asarray(radii, float)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if len(points) < 3 or distance[-1] <= 0.1 * height:
        return None
    proximal = radii[distance <= min(0.1 * height, 0.5 * distance[-1])]
    arm_radius = (
        float(np.quantile(proximal, 0.75)) if len(proximal) else float(radii[0])
    )
    thin = radii <= min(1.5 * voxel, 0.5 * arm_radius)
    start = len(thin)
    while start and thin[start - 1]:
        start -= 1
    if start == len(thin) or start == 0:
        return None
    length = distance[-1] - distance[start]
    if length <= 0.1 * height + 1e-12 * height:
        return None
    cut_distance = min(distance[-1], distance[start] + 0.03 * height)
    cut = np.array(
        [np.interp(cut_distance, distance, points[:, axis]) for axis in range(3)]
    )
    return {
        "point": cut.tolist(),
        "thin_length": float(length),
        "arm_radius": arm_radius,
    }


def extend_soles(meta, vertices, height):
    """Assign each floor-near vertex to its nearest foot segment and locate its 97% toe tip."""
    vertices = np.asarray(vertices, float)
    floor = float(vertices[:, 2].min())
    eligible = [
        name
        for name, bone in meta.items()
        if any(
            word in str(bone.get("canonical_name", name))
            for word in ("Foot", "Paw", "Tarsus", "Toes")
        )
        and bone["rest_tail_world"][2] < floor + 0.05 * height
        and not any(
            other.get("parent") == name
            and other.get("kind") not in ("digit", "rigid", "flex")
            for other in meta.values()
        )
    ]
    low = vertices[vertices[:, 2] < floor + 0.03 * height]
    if not eligible or not len(low):
        return []
    distances = []
    for name in eligible:
        head, tail = (
            np.array(meta[name]["rest_head_world"]),
            np.array(meta[name]["rest_tail_world"]),
        )
        delta = tail[:2] - head[:2]
        fraction = np.clip(
            (low[:, :2] - head[:2]) @ delta / max(delta @ delta, 1e-12), 0, 1
        )
        distances.append(
            np.linalg.norm(low[:, :2] - head[:2] - fraction[:, None] * delta, axis=1)
        )
    distances = np.array(distances)
    owner = distances.argmin(0)
    moved = []
    for i, name in enumerate(eligible):
        own = low[(owner == i) & (distances[i] < 0.12 * height)]
        if len(own) < 8:
            continue
        threshold = np.quantile(-own[:, 1], 0.97)
        tip = own[-own[:, 1] >= threshold].mean(0)
        if meta[name]["rest_head_world"][1] - tip[1] < 0.02 * height:
            continue
        meta[name]["rest_tail_world"] = tip.tolist()
        moved.append(name)
    return moved


def symmetrize_skeleton(meta, shell, plane):
    """Treat complete paired trees together, preserving connections during length equalization."""
    symmetry = shell.mirror_overlap(0, plane)
    by_role = {
        str(bone.get("canonical_name", name)): name for name, bone in meta.items()
    }
    children = {n: [c for c in meta if meta[c].get("parent") == n] for n in meta}
    events = []
    for role, root in by_role.items():
        if not role.endswith(".L"):
            continue
        parent_role = str(
            meta.get(meta[root].get("parent"), {}).get(
                "canonical_name", meta[root].get("parent", "")
            )
        )
        if parent_role.endswith(".L"):
            continue
        left, queue = [], [root]
        while queue:
            name = queue.pop(0)
            left.append(name)
            queue.extend(children[name])
        roles = [str(meta[n].get("canonical_name", n)) for n in left]
        if any(not r.endswith(".L") or r[:-2] + ".R" not in by_role for r in roles):
            continue
        right = [by_role[r[:-2] + ".R"] for r in roles]
        right_tree, queue = [], [right[0]]
        while queue:
            name = queue.pop(0)
            right_tree.append(name)
            queue.extend(children[name])
        if set(right_tree) != set(right):
            continue
        # Paired names alone do not prove matching topology.
        mapping = dict(zip(left, right))
        if any(
            meta[r].get("parent")
            != mapping.get(meta[l].get("parent"), meta[l].get("parent"))
            for l, r in zip(left, right)
        ):
            continue
        lengths = lambda names: np.array(
            [
                np.linalg.norm(
                    np.array(meta[n]["rest_tail_world"]) - meta[n]["rest_head_world"]
                )
                for n in names
            ]
        )
        a, b = lengths(left), lengths(right)
        ratio = min(a.sum(), b.sum()) / max(a.sum(), b.sum(), 1e-10)
        if symmetry >= 0.8 - 1e-12:
            for l, r in zip(left, right):
                P = np.array([meta[l]["rest_head_world"], meta[l]["rest_tail_world"]])
                Q = np.array([meta[r]["rest_head_world"], meta[r]["rest_tail_world"]])
                Q[:, 0] = 2 * plane - Q[:, 0]
                P = (P + Q) / 2
                Q = P.copy()
                Q[:, 0] = 2 * plane - Q[:, 0]
                for name, points in ((l, P), (r, Q)):
                    meta[name]["rest_head_world"], meta[name]["rest_tail_world"] = (
                        points.tolist()
                    )
            events.append(["symmetrize", root])
        elif (
            ratio >= 0.75 - 1e-12
            and np.min(np.minimum(a, b) / np.maximum(np.maximum(a, b), 1e-10)) >= 0.4
        ):
            for l, r in zip(left, right):
                target = (
                    sum(
                        np.linalg.norm(
                            np.array(meta[n]["rest_tail_world"])
                            - meta[n]["rest_head_world"]
                        )
                        for n in (l, r)
                    )
                    / 2
                )
                for name in (l, r):
                    head, tail = (
                        np.array(meta[name]["rest_head_world"]),
                        np.array(meta[name]["rest_tail_world"]),
                    )
                    new = head + (tail - head) * target / max(
                        np.linalg.norm(tail - head), 1e-10
                    )
                    delta = new - tail
                    meta[name]["rest_tail_world"] = new.tolist()
                    descendants = list(children[name])
                    while descendants:
                        child = descendants.pop(0)
                        for endpoint in ("rest_head_world", "rest_tail_world"):
                            meta[child][endpoint] = (
                                np.asarray(meta[child][endpoint]) + delta
                            ).tolist()
                        descendants.extend(children[child])
            events.append(["extend_to", root])
    return symmetry, events


def postprocess_skintokens_skeleton(bones, vertices, shell, fold_digits=False):
    """Twelve measured repairs; returns edited metadata plus an explicit removal/reparent map."""
    meta = deepcopy(bones)
    v = np.asarray(vertices)
    H = max(np.ptp(v[:, 2]), 1e-5)
    removed, events = {}, []

    def role(n):
        return str(meta[n].get("canonical_name", n))

    def remove(n):
        _remove_bone(meta, n, removed)

    if fold_digits:
        folded = fold_short_digits(meta, H)
        removed.update(folded)
        events.extend([["fold_digits", n] for n in folded])
    dropped, retracted = drop_outside(meta, shell)
    removed.update(dropped)
    events.extend([["drop_outside", n] for n in dropped])
    events.extend([["retract_tip", n] for n in retracted])
    roots = [n for n in meta if meta[n].get("parent") not in meta]
    if not roots:
        raise ValueError("Skeleton has no root after cleanup")
    root = roots[0]
    root_pos = np.asarray(meta[root]["rest_head_world"])
    feet = _footprints(shell)
    # A one/two-voxel sheet can tangle the medial tree and make a grounded leg
    # look like a wing. An opening removes those sheets while retaining tubes.
    limb_solid = (
        ndimage.binary_opening(
            shell.solid, structure=ndimage.generate_binary_structure(3, 1)
        )
        if feet
        else None
    )
    ends = np.array([b["rest_tail_world"] for b in meta.values()])
    missing = [p for p in feet if np.linalg.norm(ends - p, axis=1).min() > 0.18 * H]
    paths = (
        free_limb_paths(shell, centre_tree(shell, root_pos, feet, volume=limb_solid))
        if missing
        else []
    )
    trunk = [n for n in meta if meta[n].get("kind") in ("root", "spine", "neck")]
    grounded = set()
    for i, path in enumerate(paths):
        if not any(np.linalg.norm(path[-1] - foot) < 0.08 * H for foot in missing):
            continue
        points = _resample(path, [0, 0.35, 0.7, 1])

        def attachment_distance(n):
            head = np.asarray(meta[n]["rest_head_world"])
            delta = np.asarray(meta[n]["rest_tail_world"]) - head
            projection = (
                head
                + np.clip(
                    np.dot(points[0] - head, delta) / max(np.dot(delta, delta), 1e-10),
                    0,
                    1,
                )
                * delta
            )
            return np.linalg.norm(points[0] - projection)

        parent = min(trunk or [root], key=attachment_distance)
        for j in range(3):
            n = f"__transplant_{i}_{j}"
            meta[n] = {
                "parent": parent,
                "kind": "limb",
                "side": "L" if points[-1, 0] > root_pos[0] else "R",
                "rest_head_world": points[j].tolist(),
                "rest_tail_world": points[j + 1].tolist(),
            }
            grounded.add(n)
            parent = n
        events.append(["transplant_missing_legs", parent])
    volume_points = shell.origin + np.argwhere(shell.solid) * shell.step
    floor = float(v[:, 2].min())
    for n, b in meta.items():
        if (
            role(n).startswith(
                (
                    "Foot",
                    "FrontPaw",
                    "HindPaw",
                    "Toes",
                    "FrontToes",
                    "HindToes",
                    "Tarsus",
                )
            )
            and b["rest_tail_world"][2] < floor + 0.05 * H
        ):
            while n in meta and n not in grounded:
                grounded.add(n)
                n = meta[n].get("parent")
    limb_names = [n for n, b in meta.items() if b.get("kind") in ("limb", "wing")]
    limb_index = {n: i for i, n in enumerate(limb_names)}
    samples, labels = [], []
    for n in limb_names:
        a, t = (
            np.array(meta[n]["rest_head_world"]),
            np.array(meta[n]["rest_tail_world"]),
        )
        count = max(3, min(2048, int(np.ceil(np.linalg.norm(t - a) / shell.step)) + 1))
        samples.append(a + np.linspace(0, 1, count)[:, None] * (t - a))
        labels.extend([limb_index[n]] * count)
    owners = (
        np.asarray(labels)[cKDTree(np.concatenate(samples)).query(volume_points)[1]]
        if samples
        else np.full(len(volume_points), -1)
    )
    leg_volume = (
        limb_solid[shell.solid]
        if limb_solid is not None
        else np.ones(len(volume_points), bool)
    )
    trunk_points = (
        np.concatenate(
            [
                np.asarray(b["rest_head_world"])
                + np.linspace(0, 1, 12)[:, None]
                * (np.asarray(b["rest_tail_world"]) - b["rest_head_world"])
                for b in meta.values()
                if b.get("kind") in ("root", "spine", "neck")
            ]
        )
        if any(b.get("kind") in ("root", "spine", "neck") for b in meta.values())
        else []
    )
    for n in list(meta):
        if n not in meta:
            continue
        b = meta[n]
        h = np.asarray(b["rest_head_world"])
        t = np.asarray(b["rest_tail_world"])
        d = t - h
        L = np.linalg.norm(d)
        if L < 1e-8:
            continue
        if b.get("kind") in ("limb", "wing"):
            # Opposite legs and clothing are not part of this bone's local
            # cross-section, even when they share its transverse slice.
            own = owners == limb_index[n]
            if n in grounded:
                own &= leg_volume
            flatness = cross_section_flatness(
                shell,
                h,
                t,
                trunk_points,
                volume_points[own],
                reference_height=H,
            )
            b["flatness"] = flatness
            if flatness > 5 and b.get("kind") == "limb":
                b["kind"] = "wing"
                events.append(["flatness", n])
        if any(x in role(n) for x in ("Hand", "FrontPaw")) and L > 0.1 * H:
            count = max(30, min(2048, math.ceil(L / (shell.step * 0.5))))
            points = h + np.linspace(0, 1, count + 1)[:, None] * d
            split = cut_held_prop(points, shell.radius(points), H, shell.step)
            if split is not None:
                cut = split["point"]
                b["rest_tail_world"] = cut
                side = b.get("side", "L")
                name = f"Weapon.{side}"
                if name not in meta:
                    meta[name] = {
                        "parent": n,
                        "rest_head_world": cut,
                        "rest_tail_world": t.tolist(),
                        "kind": "rigid",
                        "side": side,
                        "fixed_local": True,
                        "weapon": True,
                        "identified_thin_length": split["thin_length"],
                    }
                    events.append(["cut_held_prop", n])
        if role(n).startswith("Head"):
            if abs(t[0] - root_pos[0]) > 0.08 * H:
                old_head = h.copy()
                h[0] = root_pos[0]
                t[0] = root_pos[0]
                b["rest_head_world"] = h.tolist()
                b["rest_tail_world"] = t.tolist()
                parent = b.get("parent")
                if (
                    parent in meta
                    and np.linalg.norm(
                        np.asarray(meta[parent]["rest_tail_world"]) - old_head
                    )
                    < 0.02 * H
                ):
                    meta[parent]["rest_tail_world"] = h.tolist()
                events.append(["head_to_face", n])
            ch = [c for c in meta if meta[c].get("parent") == n]
            for c in ch:
                ct = np.asarray(meta[c]["rest_tail_world"])
                mirror = ct.copy()
                mirror[0] = 2 * root_pos[0] - ct[0]
                child_head = np.asarray(meta[c]["rest_head_world"])
                mirror_head = child_head.copy()
                mirror_head[0] = 2 * root_pos[0] - mirror_head[0]
                paired = any(
                    np.linalg.norm(np.asarray(meta[a]["rest_tail_world"]) - mirror)
                    < 0.05 * H
                    and np.linalg.norm(
                        np.asarray(meta[a]["rest_head_world"]) - mirror_head
                    )
                    < 0.05 * H
                    for a in ch
                    if a != c and a in meta
                )
                if (
                    abs(ct[0] - root_pos[0]) > 0.08 * H
                    and np.linalg.norm(ct - child_head) < 0.25 * H
                    and meta[c].get("kind") == "head"
                    and not paired
                    and not any(meta[a].get("parent") == c for a in meta)
                ):
                    remove(c)
                    events.append(["lone_head_ends", c])
            if not any(role(a) == "Jaw" for a in meta):
                mouth = jaw_from_mouth_volume(shell, h, t, root_pos[0])
                if mouth is not None:
                    hinge, chin = mouth
                    meta["Jaw"] = {
                        "parent": n,
                        "rest_head_world": hinge.tolist(),
                        "rest_tail_world": chin.tolist(),
                        "kind": "jaw",
                        "side": "C",
                    }
                    events.append(["jaw_at_mouth", n])
    if len(feet) >= 4:
        for name in align_quadruped_hips(meta, shell, H):
            events.append(["quadruped_hip", name])
    tail_report = refine_long_tail(meta, H)
    if tail_report:
        for old_name in tail_report["old_names"]:
            if old_name not in meta:
                removed[old_name] = tail_report["new_names"][0]
        events.append(["long_tail", tail_report])
    events.extend([["soles", name] for name in extend_soles(meta, v, H)])
    symmetry, paired_events = symmetrize_skeleton(meta, shell, float(root_pos[0]))
    events.extend(paired_events)
    for n, b in meta.items():
        b["rest_length"] = float(
            np.linalg.norm(np.asarray(b["rest_tail_world"]) - b["rest_head_world"])
        )
        b["children"] = [c for c in meta if meta[c].get("parent") == n]
    return meta, {
        "fold_digits": fold_digits,
        "removed": removed,
        "events": events,
        "facing": find_facing(v, shell=shell),
        "symmetry_overlap": symmetry,
    }
