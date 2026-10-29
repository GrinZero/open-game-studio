"""Blender boundary for v3 geometry, dual-mode skin binding, camera export and review."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

# Auto-add local blender site-packages if available
_EXTRA_SITE = "/Users/bugyaluwang/.local/share/codex-integrations/game-studio-local/blender-python-5.2.3/lib/python3.13/site-packages"
if _EXTRA_SITE not in sys.path and Path(_EXTRA_SITE).is_dir():
    sys.path.append(_EXTRA_SITE)

from reprojection_solver import aligned_rest_rotation, find_hinges
from skeleton_postprocess import (
    find_facing,
    manylegs_build,
    parse_anatomical_name,
    postprocess_skintokens_skeleton,
    rest_coords,
    voxelize,
)
from strip_renderer import StripRenderer  # compatibility export
from studio_camera import Camera


def mesh_arrays(meshes: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenated vertex world coordinates and triangle indices."""
    vertices = []
    triangles = []
    offset = 0
    valid_meshes = [
        obj
        for obj in meshes
        if not any(c.name == "glTF_not_exported" for c in obj.users_collection)
        and not obj.name.startswith("glTF_")
    ]
    for obj in valid_meshes:
        v = np.array([list(obj.matrix_world @ p.co) for p in obj.data.vertices])
        obj.data.calc_loop_triangles()
        vertices.extend(v)
        triangles.extend(
            [[offset + int(i) for i in tri.vertices] for tri in obj.data.loop_triangles]
        )
        offset += len(v)
    return np.asarray(vertices, float), np.asarray(triangles, int)


def _remesh_watertight_shell_arrays(meshes: list[Any], voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Build video2rig's smooth closed watertight shell via Blender's native VOXEL REMESH modifier."""
    import bpy

    char_meshes = [
        obj
        for obj in meshes
        if not obj.name.startswith("Prop_")
        and not any(c.name == "glTF_not_exported" for c in obj.users_collection)
        and not obj.name.startswith("glTF_")
        and len(obj.data.polygons) > 0
    ]
    if not char_meshes:
        char_meshes = [
            obj
            for obj in meshes
            if not any(c.name == "glTF_not_exported" for c in obj.users_collection)
            and not obj.name.startswith("glTF_")
            and len(obj.data.polygons) > 0
        ]
    if not char_meshes:
        return np.zeros((0, 3), float), np.zeros((0, 3), int)

    prev_active = bpy.context.view_layer.objects.active
    prev_selected = [obj for obj in bpy.context.scene.objects if obj.select_get()]
    copy_names = []
    temp_mesh_names = []
    shell_obj_name = None
    try:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in char_meshes:
            copy_data = obj.data.copy()
            temp_mesh_names.append(copy_data.name)
            copy_obj = obj.copy()
            copy_obj.data = copy_data
            copy_obj.modifiers.clear()
            bpy.context.collection.objects.link(copy_obj)
            copy_obj.select_set(True)
            copy_names.append(copy_obj.name)
        bpy.context.view_layer.objects.active = bpy.data.objects[copy_names[0]]
        if len(copy_names) > 1:
            bpy.ops.object.join()
        shell_obj = bpy.context.view_layer.objects.active
        shell_obj_name = shell_obj.name
        if shell_obj.data is not None:
            temp_mesh_names.append(shell_obj.data.name)
        mod = shell_obj.modifiers.new("Remesh", "REMESH")
        mod.mode = "VOXEL"
        mod.voxel_size = float(max(voxel_size, 1e-4))
        mod.use_smooth_shade = False
        bpy.ops.object.modifier_apply(modifier=mod.name)
        if shell_obj.data is not None:
            temp_mesh_names.append(shell_obj.data.name)
        v, f = mesh_arrays([shell_obj])
        return v, f
    finally:
        for obj_name in [shell_obj_name, *copy_names]:
            if obj_name:
                rem_obj = bpy.data.objects.get(obj_name)
                if rem_obj is not None:
                    if rem_obj.data is not None:
                        temp_mesh_names.append(rem_obj.data.name)
                    bpy.data.objects.remove(rem_obj, do_unlink=True)
        for mesh_name in set(temp_mesh_names):
            rem_mesh = bpy.data.meshes.get(mesh_name)
            if rem_mesh is not None and rem_mesh.users == 0:
                bpy.data.meshes.remove(rem_mesh)
        bpy.ops.object.select_all(action="DESELECT")
        for obj in prev_selected:
            if obj.name in bpy.data.objects:
                bpy.data.objects[obj.name].select_set(True)
        if prev_active and prev_active.name in bpy.data.objects:
            bpy.context.view_layer.objects.active = bpy.data.objects[prev_active.name]


def export_shell(meshes: list[Any], path: str | Path, voxel_share: float = 0.01, facing: str = "-Y"):
    """Export 1.0% H watertight shell once for the character."""
    sv, sf = mesh_arrays(meshes)
    lo = sv.min(axis=0)
    hi = sv.max(axis=0)
    height = max(float(np.ptp(sv[:, 2])), 1e-4)
    voxel = float(voxel_share * height)
    remesh_v, remesh_f = _remesh_watertight_shell_arrays(meshes, voxel)
    if len(remesh_v) >= 3 and len(remesh_f) >= 1:
        combined_v = np.vstack([remesh_v, sv])
        combined_f = np.vstack([remesh_f, sf + len(remesh_v)])
    else:
        remesh_v, remesh_f = sv, sf
        combined_v, combined_f = sv, sf

    shell = voxelize(combined_v, combined_f, voxel)
    shell.remesh_verts = remesh_v
    shell.remesh_faces = remesh_f
    shell.save(
        path,
        sv,
        sf,
        lo=lo.astype(np.float64),
        hi=hi.astype(np.float64),
        height=np.float64(height),
        facing=np.array(facing),
        meshes=np.array([obj.name for obj in meshes]),
    )
    return shell, sv, sf


def _surface_component_sizes(vertices: np.ndarray, edges: np.ndarray, tolerance: float) -> np.ndarray:
    """Count geometric components across duplicate UV/normal seam vertices.

    This adjacency is only used for skin repair; mesh topology and UVs stay intact.
    """
    vertices = np.asarray(vertices, dtype=float)
    parent = np.arange(len(vertices))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        parent[root(int(a))] = root(int(b))
    if len(vertices):
        for a, b in cKDTree(vertices).query_pairs(tolerance, output_type="ndarray"):
            parent[root(int(a))] = root(int(b))
    ids = np.array([root(i) for i in range(len(vertices))], dtype=int)
    return np.bincount(ids, minlength=len(vertices))[ids]


def loose_weights(meshes: list[Any], arm: Any, height: float) -> dict[str, Any]:
    """Normalize and verify deform weights so every vertex has at most four normalized influences."""
    total = 0
    covered = 0
    for obj in meshes:
        if obj.name.startswith("Prop_") or not len(obj.data.vertices):
            continue
        vgs = {g.index: g.name for g in obj.vertex_groups}
        for v in obj.data.vertices:
            total += 1
            if any(g.weight > 1e-5 for g in v.groups if g.group in vgs):
                covered += 1
    return {
        "total_vertices": total,
        "covered_vertices": covered,
        "coverage_ratio": float(covered / max(1, total)),
    }


def _evaluated_vertices_world(obj: Any) -> np.ndarray:
    """Read evaluated deformed vertices in world coordinates."""
    import bpy

    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        return np.array([list(evaluated.matrix_world @ v.co) for v in mesh.vertices])
    finally:
        evaluated.to_mesh_clear()


def ensure_prop_bones(meshes: list[Any], arm: Any) -> None:
    """Explicit Prop_* objects remain independent rigid bodies with their own channels."""
    import bpy
    from mathutils import Vector

    props = [o for o in meshes if o.name.startswith("Prop_")]
    if not props:
        return
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    inv = arm.matrix_world.inverted()
    for obj in props:
        if arm.data.edit_bones.get(obj.name):
            continue
        points = np.array([list(obj.matrix_world @ v.co) for v in obj.data.vertices])
        center = points.mean(0)
        _, _, axes = np.linalg.svd(points - center, full_matrices=False)
        axis = axes[0]
        half = max(np.ptp((points - center) @ axis) / 2, 1e-4)
        bone = arm.data.edit_bones.new(obj.name)
        bone.head = inv @ Vector(center - axis * half)
        bone.tail = inv @ Vector(center + axis * half)
    bpy.ops.object.mode_set(mode="OBJECT")
    for obj in props:
        for group in list(obj.vertex_groups):
            obj.vertex_groups.remove(group)
        vg = obj.vertex_groups.new(name=obj.name)
        vg.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
        modifiers = [m for m in obj.modifiers if m.type == "ARMATURE"]
        mod = modifiers[0] if modifiers else obj.modifiers.new("V3 Prop", "ARMATURE")
        mod.object = arm


def _parent_heat_excluding_props(arm: Any) -> Any:
    """Run Blender's native Bone Heat while temporarily excluding Prop_* bones from body heat."""
    import bpy

    props = [b for b in arm.data.bones if b.name.startswith("Prop_") and b.use_deform]
    for b in props:
        b.use_deform = False
    try:
        return bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    finally:
        for b in props:
            b.use_deform = True


def _bind_welded_bone_heat(meshes: list[Any], arm: Any, height: float | None = None) -> dict[str, Any]:
    """Port of video2rig welded Bone Heat: weld UV seams on temp copy, ARMATURE_AUTO, and transfer weights."""
    import bmesh
    import bpy
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    char_meshes = [
        obj
        for obj in meshes
        if not obj.name.startswith("Prop_")
        and not any(c.name == "glTF_not_exported" for c in obj.users_collection)
        and not obj.name.startswith("glTF_")
        and len(obj.data.vertices) > 0
    ]
    if not char_meshes:
        return {"method": "mesh_heat", "skinning_mode": "bone_heat", "fallback": False, "direct_coverage": 0.0, "mesh_weighted_fraction": 0.0}

    if height is None:
        sv, _ = mesh_arrays(char_meshes)
        height = max(float(np.ptp(sv[:, 2])), 1e-4) if len(sv) else 1.0

    # 1. Duplicate & join all mesh objects into tmp_mesh
    prev_pose_position = arm.data.pose_position
    arm.data.pose_position = "REST"
    tmp_objs = []
    tmp_obj_names: list[str] = ["__v3_heat_shell"]
    tmp_data_names: list[str] = ["__v3_heat_shell"]
    try:
        for obj in char_meshes:
            mw = obj.matrix_world.copy()
            tmp = obj.copy()
            tmp.data = obj.data.copy()
            tmp.parent = None
            tmp.matrix_world = mw
            tmp.modifiers.clear()
            tmp.vertex_groups.clear()
            bpy.context.collection.objects.link(tmp)
            tmp_objs.append(tmp)
            tmp_obj_names.append(tmp.name)
            if tmp.data is not None:
                tmp_data_names.append(tmp.data.name)

        bpy.ops.object.select_all(action="DESELECT")
        for o in tmp_objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = tmp_objs[0]
        if len(tmp_objs) > 1:
            bpy.ops.object.join()
        tmp_mesh = bpy.context.view_layer.objects.active
        tmp_mesh.name = "__v3_heat_shell"
        tmp_obj_names.append(tmp_mesh.name)
        if tmp_mesh.data is not None:
            tmp_mesh.data.name = "__v3_heat_shell"
            tmp_data_names.append(tmp_mesh.data.name)

        # 2. Weld UV seams with remove_doubles, remove degenerate zero-area faces and loose verts
        bm = bmesh.new()
        bm.from_mesh(tmp_mesh.data)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
        zero_faces = [f for f in bm.faces if f.calc_area() < 1e-6]
        if zero_faces:
            bmesh.ops.delete(bm, geom=zero_faces, context="FACES")
        loose_v = [v for v in bm.verts if len(v.link_edges) == 0]
        if loose_v:
            bmesh.ops.delete(bm, geom=loose_v, context="VERTS")
        bm.to_mesh(tmp_mesh.data)
        bm.free()
        tmp_mesh.data.update()

        # 3. Filter connected components >= 2% of height
        n_v = len(tmp_mesh.data.vertices)
        edges = np.array([e.vertices[:] for e in tmp_mesh.data.edges], dtype=int) if tmp_mesh.data.edges else np.empty((0, 2), int)
        if len(edges) > 0:
            adj = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_v, n_v))
            adj = adj + adj.T
            n_comp, labels = connected_components(adj, directed=False)
        else:
            n_comp, labels = n_v, np.arange(n_v)

        welded_pts = np.array([list(tmp_mesh.matrix_world @ v.co) for v in tmp_mesh.data.vertices])
        drop_indices = []
        for c in range(n_comp):
            c_pts = welded_pts[labels == c]
            if len(c_pts) > 0 and float(np.ptp(c_pts, axis=0).max()) < 0.02 * height:
                drop_indices.extend(np.flatnonzero(labels == c).tolist())

        if drop_indices and len(drop_indices) < n_v:
            bm = bmesh.new()
            bm.from_mesh(tmp_mesh.data)
            del_verts = [bm.verts[i] for i in drop_indices if i < len(bm.verts)]
            bmesh.ops.delete(bm, geom=del_verts, context="VERTS")
            bm.to_mesh(tmp_mesh.data)
            bm.free()
            tmp_mesh.data.update()

        # 4. Parent to Armature with ARMATURE_AUTO (excluding Prop_* bones)
        tmp_mesh.vertex_groups.clear()
        bpy.ops.object.select_all(action="DESELECT")
        tmp_mesh.select_set(True)
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm
        _parent_heat_excluding_props(arm)

        # 5. Read weights from welded body
        deform_names = [b.name for b in arm.data.bones if b.use_deform and not b.name.startswith("Prop_")]
        gid_to_name = {g.index: g.name for g in tmp_mesh.vertex_groups}
        col_map = {n: i for i, n in enumerate(deform_names)}
        W_welded = np.zeros((len(tmp_mesh.data.vertices), len(deform_names)), dtype=np.float32)
        for i, v in enumerate(tmp_mesh.data.vertices):
            for g in v.groups:
                nm = gid_to_name.get(g.group)
                if nm in col_map:
                    W_welded[i, col_map[nm]] = g.weight

        welded_coords = np.array([list(tmp_mesh.matrix_world @ v.co) for v in tmp_mesh.data.vertices])
        kd = cKDTree(welded_coords)

        # 6. Transfer weights to each original mesh
        total_verts = 0
        direct_covered = 0
        per_mesh_cov: dict[str, float] = {}

        for obj in char_meshes:
            v_coords = np.array([list(obj.matrix_world @ v.co) for v in obj.data.vertices])
            n_verts = len(v_coords)
            total_verts += n_verts
            obj.vertex_groups.clear()
            group_objs = {n: obj.vertex_groups.new(name=n) for n in deform_names}

            # Query KDTree
            k_query = min(8, len(welded_coords))
            dists, indices = kd.query(v_coords, k=k_query)
            if k_query == 1:
                dists = dists[:, None]
                indices = indices[:, None]

            exact = dists[:, 0] <= 1e-4
            obj_exact = int(np.count_nonzero(exact))
            direct_covered += obj_exact
            per_mesh_cov[obj.name] = float(obj_exact / max(1, n_verts))

            W_mesh = np.zeros((n_verts, len(deform_names)), dtype=np.float32)
            W_mesh[exact] = W_welded[indices[exact, 0]]
            not_exact = ~exact
            if np.any(not_exact):
                d_ne = dists[not_exact]
                idx_ne = indices[not_exact]
                weights_idw = 1.0 / np.maximum(d_ne, 1e-5) ** 2
                weights_idw /= weights_idw.sum(axis=1, keepdims=True)
                W_mesh[not_exact] = np.sum(W_welded[idx_ne] * weights_idw[:, :, None], axis=1)

            # Fallback for unweighted points: assign to nearest bone head
            unweighted = np.where(W_mesh.sum(axis=1) <= 1e-5)[0]
            if len(unweighted) > 0 and len(deform_names) > 0:
                bone_heads = np.array([list(arm.matrix_world @ arm.data.bones[b_nm].head_local) for b_nm in deform_names])
                b_kd = cKDTree(bone_heads)
                _, nearest_b = b_kd.query(v_coords[unweighted], k=1)
                W_mesh[unweighted, nearest_b] = 1.0

            # Keep top 4 and normalize
            top4_idx = np.argsort(-W_mesh, axis=1)[:, :4]
            rows = np.arange(n_verts)[:, None]
            top4_w = W_mesh[rows, top4_idx]
            s = top4_w.sum(axis=1, keepdims=True)
            s[s < 1e-6] = 1.0
            top4_w /= s

            for col_i, name in enumerate(deform_names):
                mask = (top4_idx == col_i) & (top4_w > 1e-4)
                vert_indices = np.where(mask.any(axis=1))[0]
                if len(vert_indices) == 0:
                    continue
                weights = top4_w[mask]
                vg = group_objs[name]
                for vi, w in zip(vert_indices, weights):
                    vg.add([int(vi)], float(w), "REPLACE")

            # Ensure single ARMATURE modifier on mesh
            modifiers = [m for m in obj.modifiers if m.type == "ARMATURE"]
            for extra_mod in modifiers[1:]:
                obj.modifiers.remove(extra_mod)
            mod = modifiers[0] if modifiers else obj.modifiers.new("Armature", "ARMATURE")
            mod.object = arm
    finally:
        arm.data.pose_position = prev_pose_position
        for obj_name in set(tmp_obj_names):
            rem_obj = bpy.data.objects.get(obj_name)
            if rem_obj is not None:
                if rem_obj.data is not None:
                    tmp_data_names.append(rem_obj.data.name)
                bpy.data.objects.remove(rem_obj, do_unlink=True)
        for data_name in set(tmp_data_names):
            rem_data = bpy.data.meshes.get(data_name)
            if rem_data is not None and rem_data.users == 0:
                bpy.data.meshes.remove(rem_data)
    coverage = float(direct_covered / max(1, total_verts))
    return {
        "method": "mesh_heat",
        "skinning_mode": "bone_heat",
        "fallback": False,
        "direct_coverage": coverage,
        "mesh_weighted_fraction": coverage,
        "per_mesh_weighted_fraction": per_mesh_cov,
        "total_vertices": total_verts,
        "welded_vertices": len(welded_coords),
    }


heat_bind_updated_skeleton = _bind_welded_bone_heat


def _bind_skintokens_fold(meshes: list[Any], arm: Any, weight_fold_map: dict[str, str]) -> dict[str, Any]:
    """Fold and normalize existing SkinTokens vertex groups to canonical bone names."""
    char_meshes = [
        obj
        for obj in meshes
        if not obj.name.startswith("Prop_")
        and not any(c.name == "glTF_not_exported" for c in obj.users_collection)
        and not obj.name.startswith("glTF_")
        and len(obj.data.vertices) > 0
    ]
    total_verts = 0
    covered_verts = 0

    for obj in char_meshes:
        names = {g.index: g.name for g in obj.vertex_groups}
        n_v = len(obj.data.vertices)
        total_verts += n_v
        per_vert_accum: list[dict[str, float]] = [{} for _ in range(n_v)]

        for i, vertex in enumerate(obj.data.vertices):
            has_w = False
            for g in vertex.groups:
                vg_name = names.get(g.group)
                if not vg_name:
                    continue
                target_name = weight_fold_map.get(vg_name, "Hips")
                per_vert_accum[i][target_name] = per_vert_accum[i].get(target_name, 0.0) + g.weight
                has_w = True
            if has_w:
                covered_verts += 1

        obj.vertex_groups.clear()
        per_group_verts: dict[str, list[int]] = {}
        per_group_weights: dict[str, list[float]] = {}

        for i, accum in enumerate(per_vert_accum):
            if not accum:
                accum = {"Hips": 1.0}
            sorted_items = sorted(accum.items(), key=lambda x: x[1], reverse=True)[:4]
            total_w = sum(w for _, w in sorted_items)
            for target_name, w in sorted_items:
                norm_w = float(w / max(total_w, 1e-8))
                if norm_w > 1e-4:
                    per_group_verts.setdefault(target_name, []).append(i)
                    per_group_weights.setdefault(target_name, []).append(norm_w)

        for target_name, v_list in per_group_verts.items():
            vg = obj.vertex_groups.new(name=target_name)
            w_list = per_group_weights[target_name]
            for vi, w in zip(v_list, w_list):
                vg.add([vi], w, "REPLACE")

        modifiers = [m for m in obj.modifiers if m.type == "ARMATURE"]
        mod = modifiers[0] if modifiers else obj.modifiers.new("Armature", "ARMATURE")
        mod.object = arm

    return {
        "method": "skintokens",
        "skinning_mode": "skintokens",
        "direct_coverage": float(covered_verts / max(1, total_verts)),
        "total_vertices": total_verts,
    }


fold_skintokens_weights = _bind_skintokens_fold


def prepare_candidate(
    cand_or_meshes: Any,
    *args: Any,
    out_blend_path: str | Path | None = None,
    skinning_mode: str | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Clean, re-rig and skin candidate character supporting welded bone_heat or skintokens folding."""
    import bpy
    from mathutils import Matrix, Vector
    from bone_taxonomy import classify_rig_bones

    explicit_skinning_mode = skinning_mode
    skinning_mode = (
        skinning_mode or os.environ.get("MONSTER_SKINNING_MODE", "bone_heat")
    ).strip().lower()
    if skinning_mode not in ("bone_heat", "skintokens"):
        skinning_mode = "bone_heat"

    # Support prepare_candidate(cand_path, out_blend_path, ...)
    if isinstance(cand_or_meshes, (str, Path)):
        cand_path = Path(cand_or_meshes)
        out_dir = Path(args[0]) if len(args) > 0 else (cand_path.parent / "prepared")
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(cand_path))
        bpy.context.view_layer.update()
        meshes = [
            o
            for o in bpy.context.scene.objects
            if o.type == "MESH"
            and not any(c.name == "glTF_not_exported" for c in o.users_collection)
            and not o.name.startswith("glTF_")
        ]
        arms = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
        if not arms:
            raise ValueError(f"No Armature found in {cand_path}")
        arm = arms[0]
        raw_bones = [
            {
                "name": b.name,
                "head": list(b.head_local),
                "tail": list(b.tail_local),
                "parent": b.parent.name if b.parent else None,
                "children": [c.name for c in b.children],
            }
            for b in arm.data.bones
        ]
        fold_digits = False
    else:
        # Support prepare_candidate(meshes, arm, out_dir, bones, fold_digits)
        meshes = [
            o
            for o in cand_or_meshes
            if not any(c.name == "glTF_not_exported" for c in o.users_collection)
            and not o.name.startswith("glTF_")
        ]
        arm = args[0] if len(args) > 0 else kwargs.get("arm")
        out_dir = Path(args[1] if len(args) > 1 else kwargs.get("out_dir", "prepared"))
        bones = args[2] if len(args) > 2 else kwargs.get("bones")
        fold_digits = args[3] if len(args) > 3 else kwargs.get("fold_digits", False)
        if bones is not None:
            raw_bones = bones
        else:
            raw_bones = [
                {
                    "name": b.name,
                    "head": list(b.head_local),
                    "tail": list(b.tail_local),
                    "parent": b.parent.name if b.parent else None,
                }
                for b in arm.data.bones
            ]

    out_dir.mkdir(parents=True, exist_ok=True)

    # Capture start pose deformation in case the input armature is posed (p0 != I)
    had_nonzero_pose = any(
        np.max(np.abs(np.array(pb.matrix_basis) - np.eye(4))) > 1e-6
        for pb in arm.pose.bones
    )
    initial_deform_local = {
        pb.name: (pb.matrix.copy() @ arm.data.bones[pb.name].matrix_local.inverted())
        for pb in arm.pose.bones
    }
    target_eval_verts = (
        {obj.name: _evaluated_vertices_world(obj).copy() for obj in meshes if not obj.name.startswith("Prop_")}
        if had_nonzero_pose
        else {}
    )

    # 1. Reset pose to rest (p0 = 0) during rest skeleton edit & binding
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()

    # 2. Build 1.0% H shell ONCE
    shell, vertices, faces = export_shell(meshes, out_dir / "shell.npz", voxel_share=0.01)
    height = max(float(np.ptp(vertices[:, 2])), 1e-4)

    raw_names = (
        list(raw_bones.keys())
        if isinstance(raw_bones, dict)
        else [str(b.get("name", "")) for b in raw_bones]
    )
    all_canonical = len(raw_names) > 0 and all(parse_anatomical_name(n) is not None for n in raw_names)

    # 3. Detect facing and run postprocess_skintokens_skeleton
    facing = find_facing(vertices, faces, shell=shell, bones=raw_bones)
    processed, report = postprocess_skintokens_skeleton(
        raw_bones, shell=shell, facing=facing, fold_digits=fold_digits
    )

    # Multileg arthropod support (>=6 grounded legs)
    roots = [n for n in processed if processed[n].get("parent") not in processed]
    root_name = roots[0] if roots else next(iter(processed))
    many = manylegs_build(shell, np.array(processed[root_name]["rest_head_world"]))
    if many:
        old_limbs = [n for n, b in processed.items() if b.get("kind") == "limb"]
        for n in old_limbs:
            report["removed"][n] = root_name
            processed.pop(n)
        trunk = [
            n
            for n, b in processed.items()
            if b.get("kind") in ("root", "root_Hips", "pelvis", "spine", "neck", "head", "tail")
        ]

        def _nearest_trunk(point: np.ndarray) -> str:
            def _dist(t_name: str) -> float:
                head_pt = np.asarray(processed[t_name]["rest_head_world"])
                direction = np.asarray(processed[t_name]["rest_tail_world"]) - head_pt
                frac = np.clip(
                    np.dot(point - head_pt, direction) / max(np.dot(direction, direction), 1e-10),
                    0.0,
                    1.0,
                )
                return float(np.linalg.norm(point - head_pt - frac * direction))

            return min(trunk or [root_name], key=_dist)

        for n, b in many.items():
            if b["parent"] is None:
                b["parent"] = _nearest_trunk(np.asarray(b["rest_head_world"]))
            b["head"] = list(b["rest_head_world"])
            b["tail"] = list(b["rest_tail_world"])
            b["role"] = "limb"
            b["canonical_name"] = n
            processed[n] = b
        report["manylegs"] = len(many) // 4

    # Snap leaf bone tails/heads inside shell when post-processing raw anonymous bones
    if not all_canonical:
        for name, b_info in processed.items():
            if b_info.get("role") == "root":
                continue
            h = np.array(b_info["head"])
            t = np.array(b_info["tail"])
            if not bool(np.asarray(shell.contains(t)).flat[0]):
                t = shell.snap_inside(h, t, max_ratio=0.85)
                b_info["tail"] = t.tolist()
                b_info["rest_tail_world"] = b_info["tail"]
            if not bool(np.asarray(shell.contains(h)).flat[0]):
                h = shell.snap_inside(t, h, max_ratio=0.85)
                b_info["head"] = h.tolist()
                b_info["rest_head_world"] = b_info["head"]

    # 4. Apply cleaned canonical skeleton to Armature in Edit Mode
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    inv = arm.matrix_world.inverted()
    rot = arm.matrix_world.to_3x3().inverted()

    # Remove deleted edit bones
    for b in list(arm.data.edit_bones):
        if b.name not in processed:
            arm.data.edit_bones.remove(b)

    # Add/update canonical bones
    for name, b_info in processed.items():
        eb = arm.data.edit_bones.get(name) or arm.data.edit_bones.new(name)
        eb.head = inv @ Vector(b_info["head"])
        eb.tail = inv @ Vector(b_info["tail"])
        if eb.length < 1e-5:
            eb.tail = eb.head + Vector((0, 1e-4, 0))
        R = aligned_rest_rotation(np.asarray(b_info["head"]), np.asarray(b_info["tail"]))
        eb.align_roll(rot @ Vector(R[:, 2]))
        eb.use_connect = False
        eb.use_deform = (b_info.get("role") != "root")

    for name, b_info in processed.items():
        p_name = b_info.get("parent")
        arm.data.edit_bones[name].parent = (
            arm.data.edit_bones.get(p_name) if p_name and p_name in arm.data.edit_bones else None
        )

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()

    # Check existing weight coverage if input was already canonical
    valid_v = total_v = 0
    for obj in meshes:
        if obj.name.startswith("Prop_") or not len(obj.data.vertices):
            continue
        vg_names = {g.index: g.name for g in obj.vertex_groups}
        for vertex in obj.data.vertices:
            total_v += 1
            valid_v += any(vg_names.get(g.group) in processed and g.weight > 1e-6 for g in vertex.groups)
    added_deform = set(processed) - set(raw_names)

    # 5. Dual-Mode Skinning Binding
    if all_canonical and not added_deform and (valid_v / max(1, total_v)) >= 0.5 and explicit_skinning_mode is None:
        binding_report = {
            "method": "preserved_input_weights",
            "skinning_mode": skinning_mode,
            "weighted_fraction": float(valid_v / max(1, total_v)),
            "direct_coverage": float(valid_v / max(1, total_v)),
        }
    elif skinning_mode == "bone_heat":
        binding_report = _bind_welded_bone_heat(meshes, arm, height)
    else:  # "skintokens"
        binding_report = _bind_skintokens_fold(meshes, arm, report["weight_fold_map"])

    report["binding"] = binding_report
    report["skinning_mode"] = skinning_mode
    report["weights"] = loose_weights(meshes, arm, height)
    ensure_prop_bones(meshes, arm)

    # Restore non-zero start pose p0 if the input armature was posed
    if had_nonzero_pose:
        topo_bones: list[str] = []

        def _visit_bone(b_obj: Any) -> None:
            topo_bones.append(b_obj.name)
            for c_obj in b_obj.children:
                _visit_bone(c_obj)

        for b_obj in arm.data.bones:
            if b_obj.parent is None:
                _visit_bone(b_obj)
        for b_name in topo_bones:
            if b_name in initial_deform_local and b_name in arm.pose.bones:
                pb = arm.pose.bones[b_name]
                pb.matrix = initial_deform_local[b_name] @ arm.data.bones[b_name].matrix_local
                bpy.context.view_layer.update()

        surface_max_error = 0.0
        for obj in meshes:
            if obj.name in target_eval_verts:
                err = float(np.linalg.norm(_evaluated_vertices_world(obj) - target_eval_verts[obj.name], axis=1).max())
                surface_max_error = max(surface_max_error, err)

        # If re-binding changed weights while posed, invert LBS so start pose vertices match targets
        if surface_max_error > 1e-5:
            bone_order = [b.name for b in arm.data.bones]
            bone_idx_map = {n: i for i, n in enumerate(bone_order)}
            rest_mats = np.array([np.array(arm.matrix_world @ arm.data.bones[n].matrix_local) for n in bone_order])
            pose_mats = np.array([np.array(arm.matrix_world @ arm.pose.bones[n].matrix) for n in bone_order])
            for obj in meshes:
                if obj.name not in target_eval_verts:
                    continue
                vg_to_b = {
                    g.index: bone_idx_map[g.name]
                    for g in obj.vertex_groups
                    if g.name in bone_idx_map
                }
                V_cnt = len(obj.data.vertices)
                s_idx = np.zeros((V_cnt, 4), dtype=int)
                s_w = np.zeros((V_cnt, 4), dtype=float)
                for vi, v in enumerate(obj.data.vertices):
                    pairs = sorted(
                        [(vg_to_b[g.group], float(g.weight)) for g in v.groups if g.group in vg_to_b and g.weight > 1e-6],
                        key=lambda x: x[1],
                        reverse=True,
                    )[:4]
                    if not pairs:
                        s_idx[vi, 0] = 0
                        s_w[vi, 0] = 1.0
                    else:
                        w_sum = max(1e-8, sum(w for _, w in pairs))
                        for k_i, (bi, w) in enumerate(pairs):
                            s_idx[vi, k_i] = bi
                            s_w[vi, k_i] = w / w_sum
                new_rest_world = rest_coords(target_eval_verts[obj.name], s_idx, s_w, rest_mats, pose_mats)
                inv_mw = np.array(obj.matrix_world.inverted())
                local_co = (
                    inv_mw
                    @ np.pad(new_rest_world, ((0, 0), (0, 1)), constant_values=1.0)[..., None]
                )[..., :3, 0]
                obj.data.vertices.foreach_set("co", local_co.astype(np.float32).ravel())
                obj.data.update()
            bpy.context.view_layer.update()
            surface_max_error = max(
                float(np.linalg.norm(_evaluated_vertices_world(obj) - target_eval_verts[obj.name], axis=1).max())
                for obj in meshes
                if obj.name in target_eval_verts
            )
        report["start_pose_rebind"] = {
            "surface_max_error": float(surface_max_error),
            "method": "pose_basis_roll_transport",
        }
    else:
        report["start_pose_rebind"] = {
            "surface_max_error": 0.0,
            "method": "rest_identity",
        }

    for name, info in processed.items():
        bone = arm.data.bones[name]
        info["rest_rot_mat_world"] = np.asarray(
            (arm.matrix_world @ bone.matrix_local).to_3x3()
        ).tolist()
        info["pose_basis"] = np.array(arm.pose.bones[name].matrix_basis).tolist()
        info["matrix_local"] = np.array(bone.matrix_local).tolist()

    processed = classify_rig_bones(
        processed, vertices.min(0).tolist(), vertices.max(0).tolist()
    )
    for name, info in processed.items():
        info["pose_basis"] = np.array(arm.pose.bones[name].matrix_basis).tolist()
        info["matrix_local"] = np.array(arm.data.bones[name].matrix_local).tolist()

    report["hinges"] = find_hinges(processed)
    (out_dir / "skeleton_postprocess.json").write_text(json.dumps(report, indent=2))

    if out_blend_path:
        bpy.ops.wm.save_mainfile(filepath=str(out_blend_path))

    return processed, report


def attach_reader_points(
    record: dict[str, Any],
    arm: Any,
    names: list[str],
    extra: list[dict[str, Any]],
    cameras: dict[str, Any],
    visibility: Any,
    calib: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    """Export 3D joint reader points and project to camera views."""
    from mathutils import Vector

    points = []
    owners = []
    for n in names:
        pb = arm.pose.bones[n]
        points.extend([arm.matrix_world @ pb.head, arm.matrix_world @ pb.tail])
        owners.extend([n, n])
    for p in extra:
        points.append(
            arm.matrix_world @ arm.pose.bones[p["bone"]].matrix @ Vector(p["local"])
        )
        owners.append(p["bone"])
    pts_3d = np.array([list(p) for p in points], dtype=float)
    record["reader_points"] = {}
    for view, obj in cameras.items():
        camera = Camera.from_blender(obj, calib["panel_width"], calib["panel_height"])
        record["reader_points"][view] = {
            "uv": camera.project(pts_3d).tolist(),
            "visibility": [
                float(visibility.label(p, n, obj, calib)["visibility_ratio"])
                for p, n in zip(points, owners)
            ],
        }
    return pts_3d, owners


def render_v3_review(
    blend_path: str | Path,
    meta_path: str | Path,
    trajectory_path: str | Path,
    video_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Actual GLB bake/readback and calibrated original/3/4/silhouette review frames."""
    import bpy
    import cv2
    import math
    import subprocess
    from PIL import Image
    from mathutils import Matrix, Vector
    from studio_camera import cameras_from_calibration
    from studio_bpy_worker import bake_3d_joints_to_animated_glb, load_glb_and_find_rig
    from video_timing import VideoTiming

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(trajectory_path).read_text())
    timing = VideoTiming.from_trajectory(data)
    calib = data["calibration"]
    if not data.get("frames") or not data.get("gate_passed"):
        raise ValueError("G3 requires a nonempty measured trajectory that passed synchronization")

    baked = out / "stage4_exam_reconstructed.glb"
    bake_3d_joints_to_animated_glb(
        str(blend_path), str(trajectory_path), str(baked), meta_path=str(meta_path)
    )
    meshes, arm = load_glb_and_find_rig(str(baked), fps=timing.fps)
    expected = set(data["bones_meta"])
    actual = {b.name for b in arm.data.bones}
    if not expected.issubset(actual):
        raise ValueError("Baked GLB readback lost expected bones")

    scene = bpy.context.scene
    W, H = int(calib["panel_width"]), int(calib["panel_height"])
    scene.render.resolution_percentage = 100
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    cameras = cameras_from_calibration(calib)
    bpycams = {}
    C = np.diag([1.0, -1.0, -1.0])
    for name, c in cameras.items():
        cd = bpy.data.cameras.new("__review_" + name)
        cd.type = "ORTHO" if c.type == "ortho" else "PERSP"
        cd.sensor_fit = "VERTICAL"
        cd.ortho_scale = c.scale
        cd.lens = c.focal_px * cd.sensor_height / H
        cd.shift_x = -c.shift_px[0] / H
        cd.shift_y = c.shift_px[1] / H
        cam = bpy.data.objects.new(cd.name, cd)
        bpy.context.collection.objects.link(cam)
        M = np.eye(4)
        M[:3, :3] = c.R.T @ C
        M[:3, 3] = -c.R.T @ c.t
        cam.matrix_world = Matrix(M.tolist())
        bpycams[name] = cam

    center = np.asarray(calib["center"])
    dist = calib["ortho_scale"] * 2.5
    yaw, pitch = np.deg2rad([50, 18])
    cd = bpy.data.cameras.new("__review_three_quarter")
    cd.type = "ORTHO"
    cd.ortho_scale = calib["ortho_scale"]
    cd.sensor_fit = "VERTICAL"
    quarter = bpy.data.objects.new(cd.name, cd)
    bpy.context.collection.objects.link(quarter)
    quarter.location = Vector(
        center
        + dist
        * np.array(
            [np.sin(yaw) * np.cos(pitch), -np.cos(yaw) * np.cos(pitch), np.sin(pitch)]
        )
    )
    quarter.rotation_euler = (
        (Vector(center) - quarter.location).to_track_quat("-Z", "Y").to_euler()
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Cannot open the independent G3 source video")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0:
        source_fps = timing.fps
    preview_fps = timing.preview_fps(source_fps)
    source_times = timing.map_source_frames(data["cadence"]["frame_times"])
    directories = {n: out / n for n in ("sync_frames", "quarter_frames", "silhouette_frames")}
    for folder in directories.values():
        folder.mkdir(exist_ok=True)

    reports = []
    i = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok or i >= len(data["cadence"]["frame_times"]):
                break
            source = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            time = source_times[i]
            scene.frame_set(math.floor(time), subframe=time % 1)
            scene.render.resolution_x = W
            scene.render.resolution_y = H
            panels = []
            masks = []
            for name in calib["view_order"]:
                scene.camera = bpycams[name]
                file = out / "_review.png"
                scene.render.filepath = str(file)
                bpy.ops.render.render(write_still=True)
                with Image.open(file) as image:
                    rgba = np.asarray(image.convert("RGBA"))
                    panels.append(Image.fromarray(rgba[:, :, :3]))
                    masks.append(rgba[:, :, 3] > 127)
            stacked = calib.get("layout") == "stacked"
            rendered = np.concatenate([np.asarray(p) for p in panels], axis=0 if stacked else 1)
            mask = np.concatenate(masks, axis=0 if stacked else 1)
            if source.shape != rendered.shape:
                raise ValueError("Readback render dimensions differ from source video calibration")
            compare = Image.fromarray(np.concatenate([source, rendered], axis=1))
            compare.save(directories["sync_frames"] / f"sync_frame_{i:04d}.png")
            border = np.concatenate([source[:5].reshape(-1, 3), source[-5:].reshape(-1, 3)])
            bg = np.median(border, axis=0)
            target = np.linalg.norm(source.astype(float) - bg, axis=-1) > 25
            union = np.count_nonzero(mask | target)
            iou = np.count_nonzero(mask & target) / max(1, union)
            overlay = source.astype(float) * 0.5
            overlay[mask] += [0, 100, 0]
            overlay[target] += [100, 0, 0]
            Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
                directories["silhouette_frames"] / f"silhouette_{i:04d}.png"
            )
            scene.camera = quarter
            scene.render.filepath = str(directories["quarter_frames"] / f"quarter_{i:04d}.png")
            bpy.ops.render.render(write_still=True)
            reports.append(
                {
                    "source_frame": i,
                    "baked_frame": time,
                    "baked_seconds": timing.seconds(time),
                    "silhouette_iou": float(iou),
                }
            )
            i += 1
    finally:
        capture.release()

    if not reports:
        raise RuntimeError("No actual GLB readback frames were rendered")

    for folder, pattern, filename in (
        ("sync_frames", "sync_frame_%04d.png", "g3_comparison_sync.mp4"),
        ("quarter_frames", "quarter_%04d.png", "g3_three_quarter.mp4"),
        ("silhouette_frames", "silhouette_%04d.png", "g3_silhouette_overlay.mp4"),
    ):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                f"{preview_fps:.12g}",
                "-i",
                str(directories[folder] / pattern),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(out / filename),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    report = {
        "actual_glb_readback": True,
        "timing": timing.as_dict(),
        "source_video_fps": source_fps,
        "preview_fps": preview_fps,
        "glb": str(baked),
        "frames": reports,
        "three_quarter": {"yaw_deg": 50, "pitch_deg": 18},
        "silhouette_method": "background_segmented_video_vs_rendered_alpha",
        "silhouette_mean_iou": float(np.mean([r["silhouette_iou"] for r in reports])),
    }
    (out / "glb_readback.json").write_text(json.dumps(report, indent=2))
    return report
