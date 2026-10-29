"""Calibrated anatomical sampling shared by Blender synthesis and CPU verification."""

import math
from copy import deepcopy
import numpy as np

SAMPLING_CONTRACT = "monster-calibrated-sampling-v2"
POSE_POLICY = "video2rig-start-pose-v1"
POSE_REGIMES = ("video2rig_calm", "video2rig_active")


def sampling_audit_inputs(records, bones):
    """Audit fitted channels and world-root angles without altering training labels/p0."""
    reports = [r.get("v3_sampling") or {} for r in records]
    if not any("fitted_bones" in report for report in reports):
        return records, bones
    fitted = set(reports[0]["fitted_bones"])
    meta = deepcopy(bones)
    for name in meta:
        if name not in fitted:
            meta[name]["fixed_local"] = True
    if reports[0].get("sampling_policy") == POSE_POLICY:
        # Motion roots are solved in world SO(3), with no local Euler box.
        # Euler components below are reporting coordinates, not joint limits.
        for name in reports[0].get("motion_roots", ()):
            meta[name]["limits_rad"] = [[-math.pi, math.pi] for _ in range(3)]
            meta[name]["limits_source"] = "agent_custom"
            meta[name]["allow_leaf_twist"] = True
    audit = []
    for record, report in zip(records, reports):
        if set(report.get("fitted_bones", ())) != fitted:
            raise ValueError("Sampling changed fitted channels within one dataset")
        joints = {
            n: {"euler_xyz": deepcopy(b["euler_xyz"])}
            for n, b in record["joints"].items()
        }
        roots = report.get("motion_roots", ())
        world = report.get("motion_root_world_eulers") or {}
        for name in roots:
            if name not in world:
                raise ValueError(
                    f"Sampling audit lacks actual world-root angles for {name}"
                )
            joints[name]["euler_xyz"] = deepcopy(world[name])
        audit.append(
            {"sampling_regime": record.get("sampling_regime"), "joints": joints}
        )
    return audit, meta


def sampling_events(rng, *, calm=False, motion_blur=False):
    reach = rng.random()
    return {
        "reach": "together"
        if reach < 0.18
        else ("onto_body" if reach < 0.35 else None),
        "yaw": not calm and rng.random() < 0.25,
        "pitch": not calm and rng.random() < 0.20,
        "roam": not calm and rng.random() < 0.50,
        "jump": not calm and rng.random() < 0.30,
        "jaw": rng.random() < 0.50,
        "hold_prop": rng.random() < 0.75,
        "strike": motion_blur and rng.random() < 0.25,
        "lunge": motion_blur and rng.random() < 0.30,
    }


def sample_start_pose(rig, rng):
    """Port video2rig.reader.poses.sample's local distribution to this rig contract.

    Python Random.gauss is used instead of numpy.normal; fixed seeds are stable
    within this implementation, not bit-identical to the source repository.
    """
    from scipy.spatial.transform import Rotation

    calm = rng.random() < .3
    scale = .4 if calm else 1.0
    p0 = rig.start_params()
    eulers = {}
    for name in rig.order:
        entry = rig.slices.get(name)
        initial = Rotation.from_matrix(rig.basis[name][:3, :3]).as_euler("xyz")
        if isinstance(entry, slice):
            initial = Rotation.from_rotvec(p0[entry.start:entry.start + 3]).as_euler("xyz")
            # World Z is up here; the source uses its measured up/facing axes.
            delta = np.radians([
                np.clip(rng.gauss(0, 8), -25, 25),
                np.clip(rng.gauss(0, 4), -12, 12),
                np.clip(rng.gauss(0, 12), -30, 30),
            ]) * scale
            initial = (Rotation.from_euler("xyz", delta) * Rotation.from_euler("xyz", initial)).as_euler("xyz")
        elif entry is not None:
            indices, channels = entry
            info = rig.meta[name]
            hinge = info.get("hinge") or {}
            roles = info.get("axis_roles", {})
            for slot, axis in zip(indices, channels):
                if axis == 1 and name not in rig.parents.values() and not info.get("allow_leaf_twist", False):
                    initial[axis] = p0[slot]
                    continue
                lo, hi, start = rig.lower[slot], rig.upper[slot], p0[slot]
                role = roles.get(("rx", "ry", "rz")[axis], "twist" if axis == 1 else "bend")
                factor, cap = {"bend": (.5, 1.6), "twist": (.3, 1.2), "side": (.4, 1.2)}.get(role, (.5, 1.6))
                sd = factor * min(max(hi - start, start - lo), cap)
                value = rng.uniform(lo, hi) if hinge.get("axis") == int(axis) else start + rng.gauss(0, sd)
                initial[axis] = np.clip(start + scale * (value - start), lo, hi)
            limits = np.asarray(info.get("limits_rad", [[-np.pi, np.pi]] * 3))
            frozen = limits[:, 1] - limits[:, 0] <= 1e-8
            initial[frozen] = limits[frozen, 0]
        eulers[name] = initial.tolist()
    return eulers, calm


def posed_surface(rig, params, skin):
    from pose_features import lbs_deform_vertices_numpy

    _, heads, rotations = rig.fk(params)
    return lbs_deform_vertices_numpy(
        skin["rest_verts"],
        skin["skin_indices"],
        skin["skin_weights"],
        skin["rest_heads"],
        skin["rest_R"],
        np.array([heads[n] for n in rig.names]),
        np.array([rotations[n] for n in rig.names]),
    )


def body_surface_targets(rig, params, skin, rng, count=32):
    """Area-weighted barycentric points on actual posed trunk triangles."""
    if skin is None:
        return np.empty((0, 3))
    vertices = posed_surface(rig, params, skin)
    ids = np.asarray(skin["skin_indices"])
    weights = np.asarray(skin["skin_weights"])
    trunk = np.array(
        [
            rig.meta[n].get("kind") in ("root", "spine", "neck")
            and not n.startswith("Prop_")
            for n in rig.names
        ]
    )
    ownership = (weights * trunk[ids]).sum(-1)
    triangles = np.asarray(skin["triangles"], int)
    eligible = triangles[ownership[triangles].mean(-1) > 0.6]
    if not len(eligible):
        return np.empty((0, 3))
    faces = vertices[eligible]
    area = np.linalg.norm(
        np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=-1
    )
    if area.sum() < 1e-10:
        return np.empty((0, 3))
    cumulative = np.cumsum(area) / area.sum()
    output = []
    for _ in range(count):
        tri = faces[min(np.searchsorted(cumulative, rng.random()), len(faces) - 1)]
        u = math.sqrt(rng.random())
        v = rng.random()
        output.append((1 - u) * tri[0] + u * (1 - v) * tri[1] + u * v * tri[2])
    return np.asarray(output)


def _clip_polygon(polygon, normal, bound):
    if not len(polygon):
        return polygon
    output = []
    previous = polygon[-1]
    old = bound - float(normal @ previous)
    for point in polygon:
        distance = bound - float(normal @ point)
        if (old >= 0) != (distance >= 0):
            output.append(previous + (point - previous) * (old / (old - distance)))
        if distance >= 0:
            output.append(point)
        previous, old = point, distance
    return np.asarray(output).reshape(-1, 2)


def sample_roaming_translation(cameras, center, height, rng, margin=0.03):
    """Uniform point in the common calibrated frame polygon on the horizontal plane.

    Pinhole pixel inequalities become linear half-planes after multiplication by
    positive depth. This preserves every panel's [3%,97%] range with lens shift,
    oblique views, and horizontal/stacked layouts alike.
    """
    if not cameras:
        raise ValueError("Roaming needs calibrated cameras")
    center = np.asarray(center, float)
    extent = max(
        [height * 2]
        + [
            c.scale
            if c.type == "ortho"
            else max(c.width, c.height) * abs(float(c.depth(center))) / c.focal_px
            for c in cameras
        ]
    )
    polygon = np.array(
        [[-extent, -extent], [extent, -extent], [extent, extent], [-extent, extent]],
        float,
    )
    for camera in cameras:
        principal = np.array([camera.width / 2, camera.height / 2]) + camera.shift_px
        for axis, size in ((0, camera.width), (1, camera.height)):
            for sign, pixel in ((1, (1 - margin) * size), (-1, margin * size)):
                edge = pixel - principal[axis]
                if camera.type == "ortho":
                    normal = sign * camera.height / camera.scale * camera.R[axis]
                    constant = sign * (
                        camera.height / camera.scale * camera.t[axis] - edge
                    )
                else:
                    normal = sign * (
                        camera.focal_px * camera.R[axis] - edge * camera.R[2]
                    )
                    constant = sign * (
                        camera.focal_px * camera.t[axis] - edge * camera.t[2]
                    )
                polygon = _clip_polygon(
                    polygon, normal[:2], -constant - normal @ center
                )
        if camera.type == "persp":
            normal = -camera.R[2]
            polygon = _clip_polygon(
                polygon, normal[:2], camera.t[2] - 0.001 * height - normal @ center
            )
    if len(polygon) < 3:
        raise ValueError(
            "Calibrated panels have no shared roaming region at this height"
        )
    centroid = polygon.mean(0)
    triangles = np.stack([polygon, np.roll(polygon, -1, axis=0)], axis=1)
    a = triangles[:, 0] - centroid
    b = triangles[:, 1] - centroid
    areas = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    if areas.sum() < 1e-12:
        raise ValueError("Degenerate calibrated roaming region")
    index = min(
        np.searchsorted(np.cumsum(areas) / areas.sum(), rng.random()), len(areas) - 1
    )
    u = math.sqrt(rng.random())
    v = rng.random()
    point = (
        (1 - u) * centroid
        + u * (1 - v) * triangles[index, 0]
        + u * v * triangles[index, 1]
    )
    return np.r_[point, 0.0]
