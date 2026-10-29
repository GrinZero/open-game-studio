"""The saved starting pose is separate from the armature's bind/rest transforms."""

import numpy as np

START_POSE_CONTRACT = "monster-start-pose-v1"


def rigid_rotation(value, name="rotation"):
    """Remove Blender float32 drift, while rejecting actual scale/shear/reflection."""
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"{name}: invalid rigid rotation")
    u, singular, vt = np.linalg.svd(rotation)
    if np.linalg.det(rotation) <= 0 or np.max(np.abs(singular - 1.0)) > 1e-4:
        raise ValueError(f"{name}: scale, shear or reflection is not a rigid rotation")
    return u @ vt


def rigid_pose_basis(value, name="pose_basis"):
    local = np.asarray(value, dtype=float).copy()
    if (
        local.shape != (4, 4)
        or not np.isfinite(local).all()
        or not np.allclose(local[3], [0, 0, 0, 1])
    ):
        raise ValueError(f"{name}: invalid saved pose_basis")
    local[:3, :3] = rigid_rotation(local[:3, :3], name)
    return local


def start_pose_matrices(bones):
    """Evaluate Blender's parent-relative matrix_basis contract in world coordinates.

    Rest matrices are built from exported world heads and axes, so an armature object
    transform is already accounted for. No pose is inferred from a current frame number.
    """
    rest = {}
    basis = {}
    for name, bone in bones.items():
        matrix = np.eye(4)
        matrix[:3, :3] = rigid_rotation(bone["rest_rot_mat_world"], name)
        matrix[:3, 3] = bone["rest_head_world"]
        rest[name] = matrix
        basis[name] = rigid_pose_basis(bone.get("pose_basis", np.eye(4)), name)
    posed = {}
    pending = set(bones)
    while pending:
        ready = sorted(
            n
            for n in pending
            if bones[n].get("parent") not in bones or bones[n]["parent"] in posed
        )
        if not ready:
            raise ValueError("Starting-pose hierarchy is cyclic")
        for name in ready:
            parent = bones[name].get("parent")
            base = (
                posed[parent] @ np.linalg.solve(rest[parent], rest[name])
                if parent in bones
                else rest[name]
            )
            posed[name] = base @ basis[name]
        pending.difference_update(ready)
    return posed


def start_pose_vertices(skin, bones):
    """Use the measured frame-zero mesh when present, or reproduce it from saved bases."""
    vertices = np.asarray(skin["rest_verts"], dtype=float)
    if "start_verts" in skin:
        start = np.asarray(skin["start_verts"], dtype=float)
        if start.shape != vertices.shape or not np.isfinite(start).all():
            raise ValueError(
                "Saved frame-zero mesh differs from the rest mesh topology"
            )
        return start
    names = list(skin["bone_names"])
    posed = start_pose_matrices(bones)
    transforms = []
    for name in names:
        rest = np.eye(4)
        rest[:3, :3] = rigid_rotation(bones[name]["rest_rot_mat_world"], name)
        rest[:3, 3] = bones[name]["rest_head_world"]
        transforms.append(posed[name] @ np.linalg.inv(rest))
    ids = np.asarray(skin["skin_indices"], dtype=int)
    weights = np.asarray(skin["skin_weights"], dtype=float)
    matrices = (np.asarray(transforms)[ids] * weights[..., None, None]).sum(axis=1)
    return (matrices @ np.c_[vertices, np.ones(len(vertices))][..., None])[:, :3, 0]
