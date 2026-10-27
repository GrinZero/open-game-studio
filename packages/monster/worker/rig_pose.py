"""The saved starting pose is separate from the armature's bind/rest transforms."""

import numpy as np

START_POSE_CONTRACT = "monster-start-pose-v1"


def start_pose_matrices(bones):
    """Evaluate Blender's parent-relative matrix_basis contract in world coordinates.

    Rest matrices are built from exported world heads and axes, so an armature object
    transform is already accounted for. No pose is inferred from a current frame number.
    """
    rest = {}
    basis = {}
    for name, bone in bones.items():
        matrix = np.eye(4)
        matrix[:3, :3] = np.asarray(bone["rest_rot_mat_world"], dtype=float)
        matrix[:3, 3] = bone["rest_head_world"]
        rest[name] = matrix
        local = np.asarray(bone.get("pose_basis", np.eye(4)), dtype=float)
        if (
            local.shape != (4, 4)
            or not np.isfinite(local).all()
            or not np.allclose(local[3], [0, 0, 0, 1])
        ):
            raise ValueError(f"{name}: invalid saved pose_basis")
        basis[name] = local
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
        rest[:3, :3] = bones[name]["rest_rot_mat_world"]
        rest[:3, 3] = bones[name]["rest_head_world"]
        transforms.append(posed[name] @ np.linalg.inv(rest))
    ids = np.asarray(skin["skin_indices"], dtype=int)
    weights = np.asarray(skin["skin_weights"], dtype=float)
    matrices = (np.asarray(transforms)[ids] * weights[..., None, None]).sum(axis=1)
    return (matrices @ np.c_[vertices, np.ones(len(vertices))][..., None])[:, :3, 0]
