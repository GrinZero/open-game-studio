#!/usr/bin/env python3
"""studio_bpy_worker.py — Headless Blender 4.2 (bpy) Worker for Colab G4 & Local Monster Studio.

Bridges VAST-AI SkinTokens output (.glb) with:
  0. [Pre-Stage 1] `sanitize_input_mesh_for_skintokens` (`--mode sanitize_input`):
     Automatically joins split sub-meshes, applies all unapplied Object Transforms (`Ctrl+A`),
     removes loose vertices/degenerate faces, recalculates outward-consistent face normals,
     and centers the mesh on the ground plane (`X=0, Y=0, Z_min=0`) before feeding to SkinTokens!
  1. [Stage 1] 4-Candidate Geometric Audit (multi-ray parity + surface-distance containment check,
     4-quadrant limb extremity coverage supporting raised front claws, left-right symmetry check,
     and Air-Bridging Chord Repair), Finger-Weight Merging & Pruning, and Bone Classification.
  2. [Stage 2] Dual-Orthographic Camera Studio (`1344x768` or 2K `2560x1440` split-screen)
     synthesizing 5,000 stratified decoupled poses (40% synergistic, 30% decoupled, 15% anti-phase/extreme,
     15% cross-midline self-occlusion; 50% crisp static + 50% inter-frame shutter blur) with an automatic
     Pose Coverage & Decoupling Rank Audit Gate.
  3. [Stage 4] Procedural Constrained Random Continuous Animation Synthesizer (96 frames @ 24 FPS)
     using sparse keyframes + Blender native `AUTO_CLAMPED` Bezier spline interpolation.
  4. [Stage 6] Two-Stage Swing + Child-Plane Twist Quaternion IK Retargeter (`bake_3d_joints_to_animated_glb`):
     - Resets pose to Rest Pose at each frame to eliminate multi-frame holonomy twist accumulation;
     - Uses child-bone hinge plane normals (`d_parent x d_child`) to recover exact axial Twist on
       multi-segment limbs (hind legs, knees, ankles, forearms).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pose_features import (
    bone_motion_blur_speed_factor,
    compute_mesh_edge_stretch_metric,
    dejitter_velocity_gated,
    features_enabled,
    dense_surface_enabled,
    dense_grid_shape,
    lbs_deform_vertices_numpy,
    mocap_priors_enabled,
    rasterize_ortho_dense_rest_map_numpy,
    DATASET_SCHEMA,
    dataset_signature,
    studio_resolution,
    idle_split_filename,
    three_view_enabled,
    studio_view_order,
)
from bone_taxonomy import (
    BONE_TAXONOMY_SCHEMA,
    DEFAULT_SYNTH_SAMPLES,
    audit_pose_dataset_coverage,
    build_bounded_local_euler_rotation,
    canonicalize_rig_bone_names_in_blender,
    classify_rig_bones,
    compute_stratified_motion_metrics,
    effective_sampling_limits_for_bone,
    reconstruct_frames_with_kind_solvers,
    sample_root_translation_offset,
    sample_stratified_pose_euler_dict,
    select_pose_sampling_regime,
    solve_bone_rotation_by_kind,
    validate_canonical_rig_bone_names,
)
from pathlib import Path
from typing import Any

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

try:
    import bpy
    from mathutils import Euler, Matrix, Quaternion, Vector
    from mathutils.bvhtree import BVHTree

    HAS_BPY = True
except ImportError:
    bpy = None  # type: ignore
    Euler = Matrix = Quaternion = Vector = BVHTree = Any  # type: ignore
    HAS_BPY = False


def require_bpy() -> None:
    if not HAS_BPY:
        raise RuntimeError(
            "studio_bpy_worker.py requires `bpy==4.2.0` for 3D scene operations "
            "(e.g. `/content/skintokens-service/venv/bin/python` on Colab G4 or local Blender Python)."
        )


# ==============================================================================
# 0. Pre-Stage 1: Automatic Input Mesh Sanitizer & Orientation Normalizer
# ==============================================================================

def clear_scene() -> None:
    """Resets the Blender scene to a clean empty state."""
    require_bpy()
    bpy.ops.wm.read_factory_settings(use_empty=True)


def sanitize_input_mesh_for_skintokens(
    input_glb_path: str,
    out_glb_path: str,
    yaw_deg: float = 0.0,
) -> dict[str, Any]:
    """Prepares a raw or decimated proxy 3D mesh for VAST-AI SkinTokens (`Michelangelo` + `3DShape2VecSet`)."""
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(input_glb_path))
    mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    if not mesh_objs:
        raise ValueError(f"No MESH objects found in {input_glb_path}")

    for obj in list(bpy.data.objects):
        if obj.type != "MESH":
            bpy.data.objects.remove(obj, do_unlink=True)

    bpy.ops.object.select_all(action="DESELECT")
    for m in mesh_objs:
        m.parent = None
        m.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    if len(mesh_objs) > 1:
        bpy.ops.object.join()
    mesh_obj = bpy.context.view_layer.objects.active

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    if abs(yaw_deg) > 1e-3:
        mesh_obj.rotation_euler = Euler((0.0, 0.0, math.radians(yaw_deg)), "XYZ")
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
    bpy.ops.mesh.remove_doubles(threshold=1e-4)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    verts = np.array([[v.co.x, v.co.y, v.co.z] for v in mesh_obj.data.vertices], dtype=np.float64)
    bmin = verts.min(axis=0)
    bmax = verts.max(axis=0)
    cx = 0.5 * (bmin[0] + bmax[0])
    cy = 0.5 * (bmin[1] + bmax[1])
    z0 = bmin[2]
    for v in mesh_obj.data.vertices:
        v.co.x -= cx
        v.co.y -= cy
        v.co.z -= z0
    mesh_obj.data.update()

    Path(out_glb_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(out_glb_path), export_format="GLB")
    report = {
        "input_glb": str(input_glb_path),
        "sanitized_glb": str(out_glb_path),
        "num_vertices": len(mesh_obj.data.vertices),
        "num_faces": len(mesh_obj.data.polygons),
        "bbox_extent_xyz": [round(float(x), 4) for x in (bmax - bmin)],
        "yaw_applied_deg": yaw_deg,
    }
    print(f"[Mesh Sanitizer] Prepared clean mesh for SkinTokens: {report}")
    return report


# ==============================================================================
# 1. Stage 1: Candidate Geometric Audit, Air-Chord Repair & Bone Classifier
# ==============================================================================

def ensure_mesh_has_visible_material(mesh_obj: Any) -> None:
    """Assigns a high-contrast studio clay PBR material if the imported mesh lacks materials."""
    if mesh_obj.data.materials and any(m is not None for m in mesh_obj.data.materials):
        return
    mat = bpy.data.materials.get("MonsterStudioClay") or bpy.data.materials.new(name="MonsterStudioClay")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.72, 0.68, 0.62, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.55
    mesh_obj.data.materials.append(mat)


def load_glb_and_find_rig(glb_path: str) -> tuple[Any, Any]:
    """Imports a .glb file, removes `glTF_not_exported` empties, and returns (mesh_obj, armature_obj)."""
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    arm_objs = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not mesh_objs or not arm_objs:
        raise ValueError(f"GLB {glb_path} missing MESH or ARMATURE object.")
    bpy.ops.object.select_all(action="DESELECT")
    for m in mesh_objs:
        m.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    if len(mesh_objs) > 1:
        bpy.ops.object.join()
    active_mesh = bpy.context.view_layer.objects.active
    ensure_mesh_has_visible_material(active_mesh)
    return active_mesh, arm_objs[0]


def is_point_inside_or_near_mesh(bvh: Any, pt_local: Any, surface_tol: float) -> bool:
    """Robustly tests whether a point is inside the mesh or within `surface_tol` of thin structures."""
    location, normal, _, dist = bvh.find_nearest(pt_local)
    if location is None:
        return False
    if dist is not None and float(dist) <= surface_tol:
        return True
    if normal is not None and (pt_local - location).dot(normal) <= 0.0:
        return True
    inside_votes = 0
    for axis_dir in (
        Vector((1.0, 0.0, 0.0)),
        Vector((-1.0, 0.0, 0.0)),
        Vector((0.0, 1.0, 0.0)),
        Vector((0.0, -1.0, 0.0)),
        Vector((0.0, 0.0, 1.0)),
        Vector((0.0, 0.0, -1.0)),
    ):
        hit_loc, hit_norm, _, _ = bvh.ray_cast(pt_local, axis_dir)
        if hit_loc is not None and hit_norm is not None and axis_dir.dot(hit_norm) > 0.0:
            inside_votes += 1
    return inside_votes >= 3


def score_candidate_rig(glb_path: str) -> dict[str, Any]:
    """Evaluates a SkinTokens candidate .glb on 4 geometric criteria."""
    mesh_obj, arm_obj = load_glb_and_find_rig(glb_path)
    bpy.context.view_layer.update()

    depsgraph = bpy.context.evaluated_depsgraph_get()
    bvh = BVHTree.FromObject(mesh_obj, depsgraph)
    mw_arm = arm_obj.matrix_world
    mw_mesh_inv = mesh_obj.matrix_world.inverted()

    verts_w = [mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices]
    zs = np.array([v.z for v in verts_w])
    xs = np.array([v.x for v in verts_w])
    ys = np.array([v.y for v in verts_w])
    z_min, z_max = float(zs.min()), float(zs.max())
    height = max(1e-4, z_max - z_min)
    surface_tol = 0.028 * height

    bones = arm_obj.data.bones
    if len(bones) < 6:
        return {"path": str(glb_path), "valid": False, "score": -999.0, "reason": "Too few bones (<6)"}

    total_samples = 0
    outside_samples = 0
    protruding_bones: list[str] = []
    air_bridging_bones: list[str] = []

    for bone in bones:
        h_w = mw_arm @ bone.head_local
        t_w = mw_arm @ bone.tail_local
        b_len = float((t_w - h_w).length)
        bone_out = 0
        mid_out = 0
        for idx_a, alpha in enumerate(np.linspace(0.15, 0.92, 6)):
            pt_w = h_w.lerp(t_w, float(alpha))
            pt_m = mw_mesh_inv @ pt_w
            total_samples += 1
            if not is_point_inside_or_near_mesh(bvh, pt_m, surface_tol):
                outside_samples += 1
                bone_out += 1
                if 1 <= idx_a <= 4:
                    mid_out += 1
        head_in = is_point_inside_or_near_mesh(bvh, mw_mesh_inv @ h_w, surface_tol * 1.4)
        tail_in = is_point_inside_or_near_mesh(bvh, mw_mesh_inv @ t_w, surface_tol * 1.4)
        if head_in and tail_in and mid_out >= 3 and b_len > 0.18 * height:
            air_bridging_bones.append(bone.name)
        elif bone_out >= 4 or not tail_in:
            protruding_bones.append(bone.name)

    outside_ratio = outside_samples / max(1, total_samples)

    y_mid = float(0.5 * (ys.min() + ys.max()))
    bone_pts = [mw_arm @ b.tail_local for b in bones] + [mw_arm @ b.head_local for b in bones]

    quadrants_covered = 0
    for x_sign in (-1.0, 1.0):
        for is_front in (True, False):
            mask = (xs * x_sign > 0.04 * height) & ((ys < y_mid) if is_front else (ys >= y_mid))
            if not np.any(mask):
                quadrants_covered += 1
                continue
            q_z_min = float(zs[mask].min())
            has_limb_bone = any(
                (p.x * x_sign > 0.02 * height)
                and ((p.y < y_mid + 0.08 * height) if is_front else (p.y >= y_mid - 0.08 * height))
                and (p.z <= q_z_min + 0.25 * height)
                for p in bone_pts
            )
            if has_limb_bone:
                quadrants_covered += 1

    grounded_ok = quadrants_covered >= 3

    heads = np.array([[p.x, p.y, p.z] for p in [mw_arm @ b.head_local for b in bones]])
    mirrored = heads.copy()
    mirrored[:, 0] *= -1.0
    dists = np.linalg.norm(heads[:, None, :] - mirrored[None, :, :], axis=-1)
    sym_chamfer = float(np.mean(np.min(dists, axis=1)) / height)

    count_penalty = 0.0
    if len(bones) < 18:
        count_penalty = 35.0
    elif len(bones) > 105:
        count_penalty = 15.0

    score = (
        25.0 * quadrants_covered
        - 120.0 * outside_ratio
        - 45.0 * sym_chamfer
        - 8.0 * len(air_bridging_bones)
        - count_penalty
    )
    return {
        "path": str(glb_path),
        "valid": grounded_ok and outside_ratio < 0.38 and len(bones) >= 15,
        "score": round(score, 3),
        "grounded_ok": grounded_ok,
        "quadrants_covered": f"{quadrants_covered}/4",
        "outside_ratio": round(outside_ratio, 4),
        "sym_chamfer": round(sym_chamfer, 4),
        "num_bones": len(bones),
        "protruding_bones": protruding_bones,
        "air_bridging_bones": air_bridging_bones,
    }


def repair_air_bridging_chords_and_prune(
    mesh_obj: Any,
    arm_obj: Any,
    protruding_bones: list[str],
    air_bridging_bones: list[str],
) -> None:
    """Repairs `air_bridging_bones` and prunes `protruding_bones` with lossless weight merging."""
    if air_bridging_bones:
        bpy.context.view_layer.objects.active = arm_obj
        bpy.ops.object.mode_set(mode="EDIT")
        for b_name in air_bridging_bones:
            eb = arm_obj.data.edit_bones.get(b_name)
            if not eb:
                continue
            eb.use_connect = False
            vec = eb.tail - eb.head
            if vec.length > 1e-4:
                eb.tail = eb.head + vec.normalized() * min(vec.length * 0.22, 0.06)
        bpy.ops.object.mode_set(mode="OBJECT")

    merge_child_weights_and_prune(mesh_obj, arm_obj, protruding_bones)


def merge_child_weights_and_prune(mesh_obj: Any, arm_obj: Any, bones_to_remove: list[str]) -> None:
    """Safely folds/deletes finger or protruding terminal bones by accumulating their
    vertex_group skinning weights into their parent bone BEFORE deleting the bones.
    """
    if not bones_to_remove:
        return

    def bone_depth(b_name: str) -> int:
        d = 0
        b = arm_obj.data.bones.get(b_name)
        while b and b.parent:
            d += 1
            b = b.parent
        return d

    sorted_remove = sorted(set(bones_to_remove), key=bone_depth, reverse=True)
    for b_name in sorted_remove:
        bone = arm_obj.data.bones.get(b_name)
        if not bone or not bone.parent:
            continue
        p_name = bone.parent.name
        if b_name in mesh_obj.vertex_groups:
            vg_child = mesh_obj.vertex_groups[b_name]
            vg_parent = (
                mesh_obj.vertex_groups.get(p_name)
                or mesh_obj.vertex_groups.new(name=p_name)
            )
            child_idx = vg_child.index
            for v in mesh_obj.data.vertices:
                for g in v.groups:
                    if g.group == child_idx and g.weight > 1e-5:
                        vg_parent.add([v.index], g.weight, "ADD")
            mesh_obj.vertex_groups.remove(vg_child)

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")
    for b_name in sorted_remove:
        if b_name in arm_obj.data.edit_bones:
            arm_obj.data.edit_bones.remove(arm_obj.data.edit_bones[b_name])
    bpy.ops.object.mode_set(mode="OBJECT")


def classify_anonymous_bones_and_assign_limits(
    mesh_obj: Any,
    arm_obj: Any,
    prune_digits: bool | None = None,
    canonicalize_names: bool | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Solves SkinTokens' anonymous bone naming (`bone_0, bone_1...`) via 3D graph topology
    and enriches metadata with the v2 Canonical Bone Taxonomy (`<KIND>_<SIDE>_<GROUP>_<INDEX>`).
    IMPORTANT: In Blender PoseBone local space, the Y axis (`euler[1]`) is the longitudinal
    twist/roll axis along the bone shaft. Because a 2-point `(head, tail)` segment cannot observe
    pure longitudinal shaft spin on terminal bones, we constrain local Y roll (`limits[1]`) tightly
    on hinges, feet, and distal chains!
    """
    mw_arm = arm_obj.matrix_world
    verts_w = [mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices]
    bbox_min = np.min([[v.x, v.y, v.z] for v in verts_w], axis=0)
    bbox_max = np.max([[v.x, v.y, v.z] for v in verts_w], axis=0)
    extent = bbox_max - bbox_min
    height = float(max(1e-4, extent[2]))
    width = float(max(1e-4, extent[0]))

    if prune_digits is None:
        prune_digits = os.environ.get("MONSTER_PRUNE_DIGITS", "1").strip().lower() in ("1", "true", "on")
    if canonicalize_names is None:
        canonicalize_names = os.environ.get("MONSTER_CANONICALIZE_BONE_NAMES", "1").strip().lower() in ("1", "true", "on")

    # Step A: Optionally detect dense distal finger/toe clusters to fold into the hand/foot bone
    finger_bones: list[str] = []
    if prune_digits:
        for bone in arm_obj.data.bones:
            if len(bone.children) >= 3:
                short_children = [
                    c for c in bone.children if (mw_arm @ c.tail_local - mw_arm @ c.head_local).length < 0.085 * height
                ]
                if len(short_children) >= 3:
                    for sc in short_children:
                        finger_bones.append(sc.name)
                        for desc in sc.children_recursive:
                            finger_bones.append(desc.name)
        merge_child_weights_and_prune(mesh_obj, arm_obj, finger_bones)

    # Step B: Classify skeleton and assign biomechanical Euler limits (in radians)
    # Note: limits = [(X_pitch_lo, X_pitch_hi), (Y_roll_lo, Y_roll_hi), (Z_yaw_lo, Z_yaw_hi)]
    bone_meta: dict[str, Any] = {}
    deg = math.radians
    for bone in arm_obj.data.bones:
        h_w = mw_arm @ bone.head_local
        t_w = mw_arm @ bone.tail_local
        vec = t_w - h_w
        depth = 0
        curr = bone
        while curr.parent:
            depth += 1
            curr = curr.parent

        if bone.parent is None:
            role = "root"
            limits = [(-deg(12), deg(12)), (-deg(6), deg(6)), (-deg(15), deg(15))]
            chain_corr = 0.0
        elif abs(h_w.x) < 0.10 * width and vec.y > 0.02 * height and h_w.y > 0.10 * height and len(bone.children) <= 1:
            role = "tail"
            limits = [(-deg(18), deg(18)), (-deg(3), deg(3)), (-deg(14), deg(14))]
            chain_corr = 0.75
        elif abs(h_w.x) < 0.08 * width and abs(t_w.x) < 0.08 * width:
            # Midline sagittal chain (spine, chest, neck, head, jaw facing -Y toward Cam_Front)
            role = "spine"
            limits = [(-deg(15), deg(15)), (-deg(3), deg(3)), (-deg(15), deg(15))]
            chain_corr = 0.65
        elif bone.parent and abs((mw_arm @ bone.parent.head_local).x) < 0.10 * width:
            # Hip / Shoulder ball-and-socket: allow X/Z swing, keep longitudinal Y roll moderate
            role = "ball_socket"
            limits = [(-deg(45), deg(55)), (-deg(6), deg(6)), (-deg(25), deg(25))]
            chain_corr = 0.0
        elif len(bone.children) >= 1 and depth >= 2:
            # Mid-limb joint (Knee / Hock / Elbow): allow bidirectional swing for uncalibrated bone rolls
            role = "hinge"
            limits = [(-deg(45), deg(95)), (-deg(25), deg(25)), (-deg(35), deg(35))]
            chain_corr = 0.0
        else:
            # Terminal foot / paw / claw / tentacle segment: lock unobservable Y shaft roll to [-2°, +2°]!
            role = "distal_chain"
            limits = [(-deg(25), deg(30)), (-deg(2), deg(2)), (-deg(10), deg(10))]
            chain_corr = 0.50

        bone_meta[bone.name] = {
            "parent": bone.parent.name if bone.parent else None,
            "children": [c.name for c in bone.children],
            "role": role,
            "depth": depth,
            "rest_head_world": [round(float(h_w.x), 5), round(float(h_w.y), 5), round(float(h_w.z), 5)],
            "rest_tail_world": [round(float(t_w.x), 5), round(float(t_w.y), 5), round(float(t_w.z), 5)],
            "rest_length": round(float(vec.length), 5),
            "rest_rot_mat_world": [[round(float(v), 5) for v in row] for row in (mw_arm @ bone.matrix_local).to_3x3()],
            "limits_rad": [[round(lo, 4), round(hi, 4)] for (lo, hi) in limits],
            "chain_corr": chain_corr,
        }

    bmin_list = [round(float(x), 5) for x in bbox_min]
    bmax_list = [round(float(x), 5) for x in bbox_max]
    bone_meta = classify_rig_bones(bone_meta, bbox_min=bmin_list, bbox_max=bmax_list)

    rename_map: dict[str, str] = {}
    if canonicalize_names:
        rename_map = canonicalize_rig_bone_names_in_blender(arm_obj, [mesh_obj], bone_meta)
        remapped_meta: dict[str, Any] = {}
        for old_name, info in bone_meta.items():
            new_name = rename_map.get(old_name, old_name)
            info["legacy_name"] = old_name
            info["canonical_name"] = new_name
            if info.get("parent") in rename_map:
                info["parent"] = rename_map[info["parent"]]
            info["children"] = [rename_map.get(c, c) for c in info.get("children", [])]
            remapped_meta[new_name] = info
        bone_meta = remapped_meta

    return finger_bones, {
        "bone_taxonomy_schema": BONE_TAXONOMY_SCHEMA,
        "bbox_min": bmin_list,
        "bbox_max": bmax_list,
        "num_active_bones": len(bone_meta),
        "folded_finger_bones": finger_bones,
        "canonical_rename_map": rename_map,
        "bone_naming_validation": validate_canonical_rig_bone_names(bone_meta),
        "bones": bone_meta,
    }


def render_candidates_contact_sheet(reports: list[dict[str, Any]], out_jpg_path: Path) -> None:
    """Renders a 2x2 visual contact sheet overlaying individual bone segments (`head -> tail`)."""
    from PIL import Image, ImageDraw

    panels: list[Image.Image] = []
    for idx, rep in enumerate(reports[:4]):
        p_w, p_h = 480, 480
        canvas = Image.new("RGB", (p_w, p_h), (18, 22, 30))
        draw = ImageDraw.Draw(canvas)
        try:
            mesh_obj, arm_obj = load_glb_and_find_rig(rep["path"])
            verts_w = [mesh_obj.matrix_world @ v.co for v in mesh_obj.data.vertices]
            bmin = np.min([[v.x, v.y, v.z] for v in verts_w], axis=0).tolist()
            bmax = np.max([[v.x, v.y, v.z] for v in verts_w], axis=0).tolist()
            calib = {
                "center": [0.5 * (bmin[i] + bmax[i]) for i in range(3)],
                "ortho_scale": float(max(bmax[i] - bmin[i] for i in range(3)) * 1.35),
                "panel_width": p_w,
                "panel_height": p_h,
            }
            mw = arm_obj.matrix_world
            for bone in arm_obj.data.bones:
                uv_h, _ = project_3d_to_dual_ortho_pixels(mw @ bone.head_local, calib)
                uv_t, _ = project_3d_to_dual_ortho_pixels(mw @ bone.tail_local, calib)
                if bone.name in rep.get("protruding_bones", []):
                    color = (235, 75, 75)
                elif bone.name in rep.get("air_bridging_bones", []):
                    color = (240, 130, 255)
                else:
                    color = (80, 220, 165)
                draw.line([(uv_h[0], uv_h[1]), (uv_t[0], uv_t[1])], fill=color, width=3)
                draw.ellipse([uv_h[0] - 4, uv_h[1] - 4, uv_h[0] + 4, uv_h[1] + 4], fill=(255, 215, 90))
        except Exception:
            pass
        header = (
            f"Candidate #{idx} | Score: {rep.get('score')} | Valid: {rep.get('valid')}\n"
            f"Bones: {rep.get('num_bones')} | Outside: {rep.get('outside_ratio')} | Limbs: {rep.get('quadrants_covered')}"
        )
        draw.rectangle([0, 0, p_w, 46], fill=(10, 12, 18))
        draw.text((10, 8), header, fill=(240, 244, 255))
        panels.append(canvas)

    while len(panels) < 4:
        panels.append(Image.new("RGB", (480, 480), (18, 22, 30)))

    grid = Image.new("RGB", (960, 960), (12, 14, 20))
    grid.paste(panels[0], (0, 0))
    grid.paste(panels[1], (480, 0))
    grid.paste(panels[2], (0, 480))
    grid.paste(panels[3], (480, 480))
    out_jpg_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_jpg_path, quality=92)


# ==============================================================================
# 2. Shared-Scale Dual-Orthographic Camera Rig & Closed-Form Projection Math
# ==============================================================================

def configure_blender_gpu_rendering(scene: Any, samples: int | None = None) -> dict[str, Any]:
    """Configures hardware-accelerated Cycles GPU rendering (`OPTIX` / `CUDA` / `METAL` / `HIP` / `ONEAPI`)
    with GPU OpenImageDenoise (`denoising_use_gpu=True`), adaptive sampling, and persistent scene data
    so GPU VRAM retains geometry/BVH across frames and camera switches without falling back to CPU.
    """
    require_bpy()
    if samples is None:
        samples = int(os.environ.get("MONSTER_CYCLES_SAMPLES", "16"))
    preferred_engine = os.environ.get("MONSTER_RENDER_ENGINE", "CYCLES").strip().upper()
    active_backend: str | None = None
    gpu_enabled = False

    if preferred_engine in ("CYCLES", "AUTO"):
        try:
            scene.render.engine = "CYCLES"
            scene.cycles.samples = int(samples)
            scene.cycles.use_adaptive_sampling = True
            scene.cycles.use_denoising = True
            scene.cycles.denoiser = "OPENIMAGEDENOISE"
            if hasattr(scene.cycles, "denoising_use_gpu"):
                scene.cycles.denoising_use_gpu = True
            if hasattr(scene.cycles, "denoising_prefilter"):
                scene.cycles.denoising_prefilter = "ACCURATE"
            if hasattr(scene.cycles, "denoising_quality"):
                scene.cycles.denoising_quality = "HIGH"
            scene.render.use_persistent_data = True

            cycles_addon = bpy.context.preferences.addons.get("cycles")
            if cycles_addon and hasattr(cycles_addon, "preferences"):
                prefs = cycles_addon.preferences
                for backend in ("OPTIX", "CUDA", "METAL", "HIP", "ONEAPI"):
                    try:
                        prefs.compute_device_type = backend
                        if hasattr(prefs, "get_devices"):
                            prefs.get_devices()
                        elif hasattr(prefs, "refresh_devices"):
                            prefs.refresh_devices()
                        matched = [d for d in prefs.devices if getattr(d, "type", None) == backend]
                        if matched:
                            # Enable only GPU devices for the chosen backend; disable CPU hybrid device
                            # so slow CPU threads never steal tiles or bottleneck GPU frame completion.
                            for d in prefs.devices:
                                d.use = getattr(d, "type", None) == backend
                            if backend == "METAL" and hasattr(prefs, "kernel_optimization_level"):
                                prefs.kernel_optimization_level = "OFF"
                            scene.cycles.device = "GPU"
                            active_backend = backend
                            gpu_enabled = True
                            break
                    except Exception:
                        continue
        except Exception:
            pass

    if scene.render.engine != "CYCLES" or (not gpu_enabled and preferred_engine == "EEVEE"):
        for eng in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
            try:
                scene.render.engine = eng
                break
            except TypeError:
                continue

    for vt in ("AgX", "Filmic", "Standard"):
        try:
            scene.view_settings.view_transform = vt
            if vt == "AgX":
                scene.view_settings.look = "AgX - Medium High Contrast"
            break
        except TypeError:
            continue

    return {
        "engine": str(scene.render.engine),
        "device": str(getattr(scene.cycles, "device", "CPU")) if scene.render.engine == "CYCLES" else "GPU",
        "backend": active_backend,
        "samples": int(getattr(scene.cycles, "samples", samples)) if scene.render.engine == "CYCLES" else 0,
        "denoiser": str(getattr(scene.cycles, "denoiser", "")) if scene.render.engine == "CYCLES" else None,
        "denoising_use_gpu": bool(getattr(scene.cycles, "denoising_use_gpu", False))
        if scene.render.engine == "CYCLES"
        else False,
        "use_persistent_data": bool(getattr(scene.render, "use_persistent_data", False)),
    }


def setup_dual_orthographic_studio(
    bbox_min: list[float],
    bbox_max: list[float],
    panel_w: int | None = None,
    panel_h: int | None = None,
    enable_3view: bool | None = None,
) -> dict[str, Any]:
    """Configures GPU-accelerated renderer (Cycles OptiX/CUDA/Metal + GPU OpenImageDenoise),
    aimed studio lighting, and strictly calibrated orthographic cameras (`Cam_Front` looking +Y,
    `Cam_Side` looking -X, and optional `Cam_Right` looking +X when 3-view mode is enabled)
    with `sensor_fit="VERTICAL"`.
    """
    require_bpy()
    use_3view = three_view_enabled() if enable_3view is None else bool(enable_3view)
    num_views = 3 if use_3view else 2
    view_order = list(studio_view_order(use_3view))
    if panel_w is None or panel_h is None:
        def_pw, def_ph, _, _ = studio_resolution(enable_3view=use_3view)
        panel_w = def_pw if panel_w is None else int(panel_w)
        panel_h = def_ph if panel_h is None else int(panel_h)
    scene = bpy.context.scene
    render_config = configure_blender_gpu_rendering(scene)

    scene.render.resolution_x = panel_w
    scene.render.resolution_y = panel_h
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "JPEG"
    scene.render.image_settings.quality = 94

    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs[0].default_value = (0.035, 0.04, 0.055, 1.0)
        bg_node.inputs[1].default_value = 1.0

    bmin = np.array(bbox_min, dtype=float)
    bmax = np.array(bbox_max, dtype=float)
    center = 0.5 * (bmin + bmax)
    center_vec = Vector((float(center[0]), float(center[1]), float(center[2])))
    ortho_scale = float(np.max(bmax - bmin) * 1.48)
    dist = ortho_scale * 2.5

    for l_name, loc, energy in [
        ("KeyLight", (2.5, -3.5, 4.0), 900.0),
        ("FillLight", (-3.0, 2.5, 3.0), 550.0),
        ("RimLight", (0.0, 3.5, 4.5), 700.0),
    ]:
        ldata = bpy.data.lights.new(name=l_name, type="AREA")
        ldata.energy = energy
        ldata.size = 3.0
        lobj = bpy.data.objects.new(name=l_name, object_data=ldata)
        bpy.context.collection.objects.link(lobj)
        lobj.location = loc
        lobj.rotation_euler = (center_vec - lobj.location).to_track_quat("-Z", "Y").to_euler()

    def make_ortho_cam(name: str, loc: tuple[float, float, float], rot: tuple[float, float, float]) -> Any:
        cdata = bpy.data.cameras.new(name=name)
        cdata.type = "ORTHO"
        cdata.sensor_fit = "VERTICAL"
        cdata.ortho_scale = ortho_scale
        cdata.clip_start = 0.05
        cdata.clip_end = dist * 4.0
        cobj = bpy.data.objects.new(name=name, object_data=cdata)
        bpy.context.collection.objects.link(cobj)
        cobj.location = loc
        cobj.rotation_euler = Euler(rot, "XYZ")
        return cobj

    cam_front = make_ortho_cam(
        "Cam_Front",
        loc=(float(center[0]), float(center[1] - dist), float(center[2])),
        rot=(math.pi / 2.0, 0.0, 0.0),
    )
    cam_side = make_ortho_cam(
        "Cam_Side",
        loc=(float(center[0] + dist), float(center[1]), float(center[2])),
        rot=(math.pi / 2.0, 0.0, math.pi / 2.0),
    )
    cam_right = (
        make_ortho_cam(
            "Cam_Right",
            loc=(float(center[0] - dist), float(center[1]), float(center[2])),
            rot=(math.pi / 2.0, 0.0, -math.pi / 2.0),
        )
        if use_3view
        else None
    )
    bpy.context.view_layer.update()

    return {
        "cam_front": cam_front,
        "cam_side": cam_side,
        "cam_right": cam_right,
        "render_config": render_config,
        "calibration": {
            "center": [round(float(c), 5) for c in center],
            "ortho_scale": round(ortho_scale, 5),
            "panel_width": panel_w,
            "panel_height": panel_h,
            "composite_width": panel_w * num_views,
            "composite_height": panel_h,
            "num_views": num_views,
            "view_order": view_order,
            "render_config": render_config,
        },
    }


def project_3d_to_dual_ortho_pixels(
    pt_world: Any, calib: dict[str, Any]
) -> tuple[list[float], list[float]]:
    """Closed-form projection from 3D world coordinates (X, Y, Z) to Front and Side view pixels."""
    if hasattr(pt_world, "x"):
        px, py, pz = float(pt_world.x), float(pt_world.y), float(pt_world.z)
    else:
        px, py, pz = float(pt_world[0]), float(pt_world[1]), float(pt_world[2])

    cx, cy, cz = calib["center"]
    S = float(calib["ortho_scale"])
    W = float(calib["panel_width"])
    H = float(calib["panel_height"])
    span_x = S * (W / H)
    span_z = S

    u_F = ((px - cx) / span_x + 0.5) * W
    v_F = (0.5 - (pz - cz) / span_z) * H

    u_S = ((py - cy) / span_x + 0.5) * W
    v_S = (0.5 - (pz - cz) / span_z) * H

    return [round(float(u_F), 3), round(float(v_F), 3)], [round(float(u_S), 3), round(float(v_S), 3)]


def project_3d_to_right_ortho_pixels(pt_world: Any, calib: dict[str, Any]) -> list[float]:
    """Closed-form projection from 3D world coordinates (X, Y, Z) to Right view (`Cam_Right` at -X looking +X) pixels."""
    if hasattr(pt_world, "x"):
        _, py, pz = float(pt_world.x), float(pt_world.y), float(pt_world.z)
    else:
        _, py, pz = float(pt_world[0]), float(pt_world[1]), float(pt_world[2])

    _, cy, cz = calib["center"]
    S = float(calib["ortho_scale"])
    W = float(calib["panel_width"])
    H = float(calib["panel_height"])
    span_x = S * (W / H)
    span_z = S

    u_R = ((cy - py) / span_x + 0.5) * W
    v_R = (0.5 - (pz - cz) / span_z) * H
    return [round(float(u_R), 3), round(float(v_R), 3)]


def triangulate_dual_ortho_pixels_to_3d(
    uv_front: list[float] | np.ndarray,
    uv_side: list[float] | np.ndarray,
    calib: dict[str, Any],
    conf_front: float = 1.0,
    conf_side: float = 1.0,
    uv_right: list[float] | np.ndarray | None = None,
    conf_right: float = 1.0,
) -> list[float]:
    """Closed-form inverse of `project_3d_to_dual_ortho_pixels` (and optional `project_3d_to_right_ortho_pixels`)."""
    cx, cy, cz = calib["center"]
    S = float(calib["ortho_scale"])
    W = float(calib["panel_width"])
    H = float(calib["panel_height"])
    span_x = S * (W / H)
    span_z = S

    u_F, v_F = float(uv_front[0]), float(uv_front[1])
    u_S, v_S = float(uv_side[0]), float(uv_side[1])
    w_F = max(1e-4, float(conf_front)) ** 2
    w_S = max(1e-4, float(conf_side)) ** 2

    X = cx + (u_F / W - 0.5) * span_x
    Y_S = cy + (u_S / W - 0.5) * span_x
    Z_F = cz + (0.5 - v_F / H) * span_z
    Z_S = cz + (0.5 - v_S / H) * span_z

    if uv_right is not None:
        u_R, v_R = float(uv_right[0]), float(uv_right[1])
        w_R = max(1e-4, float(conf_right)) ** 2
        Y_R = cy - (u_R / W - 0.5) * span_x
        Z_R = cz + (0.5 - v_R / H) * span_z
        Y = (w_S * Y_S + w_R * Y_R) / (w_S + w_R)
        Z = (w_F * Z_F + w_S * Z_S + w_R * Z_R) / (w_F + w_S + w_R)
    else:
        Y = Y_S
        Z = (w_F * Z_F + w_S * Z_S) / (w_F + w_S)
    return [round(float(X), 5), round(float(Y), 5), round(float(Z), 5)]


def render_split_screen_frame(
    cam_front: Any,
    cam_side: Any,
    out_png_path: Path,
    tmp_dir: Path,
    cam_right: Any = None,
) -> None:
    """Renders studio cameras and horizontally concatenates them into a composite frame:
      - 2-view mode (`cam_right is None`): `[ Front | Side ]` (`W * 2`)
      - 3-view mode (`cam_right is not None`): `[ Left (cam_side) | Front (cam_front) | Right (cam_right) ]` (`W * 3`)
    Uses Linux RAM disk (`/dev/shm`) when available to avoid disk/FUSE writes during synthesis.
    """
    from PIL import Image

    scene = bpy.context.scene
    shm_dir = Path("/dev/shm")
    scratch_dir = shm_dir if (shm_dir.is_dir() and os.access(shm_dir, os.W_OK)) else tmp_dir
    pid = os.getpid()
    f_path = scratch_dir / f"_tmp_front_{pid}.jpg"
    s_path = scratch_dir / f"_tmp_side_{pid}.jpg"

    scene.camera = cam_front
    scene.render.filepath = str(f_path)
    bpy.ops.render.render(write_still=True)

    scene.camera = cam_side
    scene.render.filepath = str(s_path)
    bpy.ops.render.render(write_still=True)

    im_f = Image.open(f_path).convert("RGB")
    im_s = Image.open(s_path).convert("RGB")

    if cam_right is not None:
        r_path = scratch_dir / f"_tmp_right_{pid}.jpg"
        scene.camera = cam_right
        scene.render.filepath = str(r_path)
        bpy.ops.render.render(write_still=True)
        im_r = Image.open(r_path).convert("RGB")
        comp = Image.new("RGB", (im_s.width + im_f.width + im_r.width, im_f.height))
        comp.paste(im_s, (0, 0))
        comp.paste(im_f, (im_s.width, 0))
        comp.paste(im_r, (im_s.width + im_f.width, 0))
        comp.save(out_png_path)
        return

    comp = Image.new("RGB", (im_f.width + im_s.width, im_f.height))
    comp.paste(im_f, (0, 0))
    comp.paste(im_s, (im_f.width, 0))
    comp.save(out_png_path)


# ==============================================================================
# 3. Stage 2 (5,000 Stratified Poses + 50% Motion Blur + Coverage Audit) & Stage 4 (96-Frame Exam)
# ==============================================================================

def sample_in_limit_pose(
    arm_obj: Any,
    bones_meta: dict[str, Any],
    rng: random.Random,
    sample_idx: int | None = None,
    sample_root_location: bool = False,
    rig_height: float = 1.0,
) -> dict[str, list[float]]:
    """Samples a biomechanically valid pose using the 4-Regime Stratified Decoupled Sampler
    (`synergistic`, `decoupled`, `antiphase_extreme`, `workspace_cross_occlusion`) and applies
    it to `arm_obj.pose.bones`.
    """
    sampled_raw, regime = sample_stratified_pose_euler_dict(
        bones_meta=bones_meta,
        rng=rng,
        sample_idx=sample_idx,
        enforce_anatomical_floor=True,
    )
    sampled_eulers: dict[str, list[float]] = {}
    for b_name, raw in sampled_raw.items():
        pbone = arm_obj.pose.bones.get(b_name)
        if not pbone:
            continue
        rot = list(raw)
        if not pbone.children:
            rot[1] = 0.0
        pbone.rotation_mode = "XYZ"
        pbone.rotation_euler = Euler(rot, "XYZ")
        if pbone.parent is None:
            if sample_root_location:
                dx, dy, dz = sample_root_translation_offset(bones_meta=bones_meta, rng=rng, sample_idx=sample_idx)
                delta_arm = arm_obj.matrix_world.to_3x3().inverted() @ Vector((dx, dy, dz))
                pbone.location = pbone.bone.matrix_local.to_3x3().inverted() @ delta_arm
            else:
                pbone.location = Vector((0.0, 0.0, 0.0))
        sampled_eulers[b_name] = rot
    bpy.context.view_layer.update()
    return sampled_eulers


def extract_rig_lbs_skinning_and_verify_gate(
    mesh_obj: Any,
    arm_obj: Any,
    bone_names: list[str],
    calib: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extracts rest-pose world vertices `(V, 3)`, top-4 LBS bone skinning indices `(V, 4)` and
    normalized weights `(V, 4)`, triangle indices `(F, 3)`, and rest-pose bone transforms
    `(rest_heads, rest_R)` for analytical/differentiable LBS surface correspondence refinement.
    Also enforces `use_deform_preserve_volume = False` on Armature modifiers so Blender's rendered
    surface matches pure Linear Blend Skinning (`lbs_deform_vertices_numpy`).
    """
    require_bpy()
    for mod in getattr(mesh_obj, "modifiers", []):
        if getattr(mod, "type", None) == "ARMATURE" and hasattr(mod, "use_deform_preserve_volume"):
            mod.use_deform_preserve_volume = False

    for pbone in arm_obj.pose.bones:
        pbone.rotation_mode = "XYZ"
        pbone.rotation_euler = (0.0, 0.0, 0.0)
        pbone.location = (0.0, 0.0, 0.0)
        pbone.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()

    mw_mesh = mesh_obj.matrix_world
    mw_arm = arm_obj.matrix_world
    verts_list = [[float(w.x), float(w.y), float(w.z)] for w in (mw_mesh @ v.co for v in mesh_obj.data.vertices)]
    rest_verts = np.asarray(verts_list, dtype=np.float32)
    V = int(rest_verts.shape[0])

    mesh_obj.data.calc_loop_triangles()
    triangles = np.asarray(
        [[int(lt.vertices[0]), int(lt.vertices[1]), int(lt.vertices[2])] for lt in mesh_obj.data.loop_triangles],
        dtype=np.int32,
    )

    bone_to_idx = {b_name: idx for idx, b_name in enumerate(bone_names)}
    vg_to_bone: dict[int, int] = {}
    for vg in mesh_obj.vertex_groups:
        if vg.name in bone_to_idx:
            vg_to_bone[int(vg.index)] = int(bone_to_idx[vg.name])

    rest_heads = np.zeros((len(bone_names), 3), dtype=np.float64)
    rest_tails = np.zeros((len(bone_names), 3), dtype=np.float64)
    rest_R = np.zeros((len(bone_names), 3, 3), dtype=np.float64)
    for b_idx, b_name in enumerate(bone_names):
        db = arm_obj.data.bones[b_name]
        hw = mw_arm @ db.head_local
        tw = mw_arm @ db.tail_local
        rm = (mw_arm @ db.matrix_local).to_3x3()
        rest_heads[b_idx] = [float(hw.x), float(hw.y), float(hw.z)]
        rest_tails[b_idx] = [float(tw.x), float(tw.y), float(tw.z)]
        rest_R[b_idx] = [[float(v) for v in row] for row in rm]

    skin_indices = np.zeros((V, 4), dtype=np.int32)
    skin_weights = np.zeros((V, 4), dtype=np.float32)
    bone_mids = 0.5 * (rest_heads + rest_tails)

    for v_idx, v in enumerate(mesh_obj.data.vertices):
        pairs: list[tuple[int, float]] = []
        for g in v.groups:
            b_idx = vg_to_bone.get(int(g.group))
            w_val = float(g.weight)
            if b_idx is not None and w_val > 1e-6:
                pairs.append((b_idx, w_val))
        if not pairs:
            dists = np.linalg.norm(bone_mids - rest_verts[v_idx].astype(np.float64)[None, :], axis=-1)
            nearest_b = int(np.argmin(dists))
            skin_indices[v_idx, 0] = nearest_b
            skin_weights[v_idx, 0] = 1.0
            continue
        pairs.sort(key=lambda item: item[1], reverse=True)
        top_pairs = pairs[:4]
        w_sum = max(1e-8, sum(item[1] for item in top_pairs))
        for k_i, (b_idx, w_val) in enumerate(top_pairs):
            skin_indices[v_idx, k_i] = int(b_idx)
            skin_weights[v_idx, k_i] = float(w_val / w_sum)

    lbs_rest = lbs_deform_vertices_numpy(
        rest_verts, skin_indices, skin_weights, rest_heads, rest_R, rest_heads, rest_R
    )
    rest_identity_err = float(np.max(np.linalg.norm(lbs_rest - rest_verts, axis=-1))) if V > 0 else 0.0
    if calib is not None:
        calib["lbs_skinning_gate"] = {
            "num_vertices": V,
            "num_triangles": int(triangles.shape[0]),
            "rest_identity_max_err": round(rest_identity_err, 7),
            "preserve_volume_disabled": True,
        }
    return {
        "bone_names": list(bone_names),
        "rest_verts": rest_verts,
        "skin_indices": skin_indices,
        "skin_weights": skin_weights,
        "triangles": triangles,
        "rest_heads": rest_heads.astype(np.float32),
        "rest_R": rest_R.astype(np.float32),
    }


def generate_stage2_synthetic_dataset(
    blend_path: str,
    meta_path: str,
    out_dir: str,
    num_samples: int = DEFAULT_SYNTH_SAMPLES,
    seed: int = 42,
) -> None:
    """Renders the Stage 2 synthetic dataset (default 5,000 stratified decoupled dual-view images)
    and verifies the dataset with `audit_pose_dataset_coverage`.
    """
    require_bpy()
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    arm_obj = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
    mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    mesh_obj = mesh_objs[0] if mesh_objs else None
    studio = setup_dual_orthographic_studio(meta["bbox_min"], meta["bbox_max"])
    cam_f, cam_s, cam_r, calib = (
        studio["cam_front"],
        studio["cam_side"],
        studio.get("cam_right"),
        studio["calibration"],
    )
    meta["calibration"] = calib
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    out_path = Path(out_dir)
    img_dir = out_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for pbone in arm_obj.pose.bones:
        pbone.rotation_mode = "XYZ"
        pbone.rotation_euler = (0.0, 0.0, 0.0)
    bpy.context.view_layer.update()
    idle_name = idle_split_filename(
        calib["composite_width"],
        calib["composite_height"],
        enable_3view=(cam_r is not None),
    )
    render_split_screen_frame(cam_f, cam_s, out_path / idle_name, out_path, cam_right=cam_r)

    enhanced = features_enabled()
    dense_enabled = bool(enhanced and dense_surface_enabled() and mesh_obj is not None)
    rig_skinning: dict[str, Any] | None = None
    dense_dir = out_path / "dense"
    dense_hm_h, dense_hm_w = dense_grid_shape(int(calib["panel_width"]), int(calib["panel_height"]))
    sorted_bones_for_skin = sorted(meta["bones"])
    if enhanced:
        from pose_visibility import SurfaceVisibilitySampler, randomize_studio_lighting
        calib["depth_scale"] = max(1e-4, float(np.max(np.array(meta["bbox_max"])-np.array(meta["bbox_min"]))))
        calib["depth_reference"] = "fixed_studio_center"
        calib["camera_contract"] = (
            "left_minus_x_front_plus_y_right_plus_x_orthographic"
            if cam_r is not None
            else "front_plus_y_side_minus_x_orthographic"
        )
        calib["camera_world_matrices"] = {
            "front": [list(row) for row in cam_f.matrix_world],
            "side": [list(row) for row in cam_s.matrix_world],
        }
        if cam_r is not None:
            calib["camera_world_matrices"]["right"] = [list(row) for row in cam_r.matrix_world]
        if dense_enabled and mesh_obj is not None:
            dense_dir.mkdir(parents=True, exist_ok=True)
            rig_skinning = extract_rig_lbs_skinning_and_verify_gate(
                mesh_obj, arm_obj, sorted_bones_for_skin, calib=calib
            )
            np.savez_compressed(
                out_path / "rig_skinning.npz",
                bone_names=np.asarray(rig_skinning["bone_names"]),
                rest_verts=rig_skinning["rest_verts"],
                skin_indices=rig_skinning["skin_indices"],
                skin_weights=rig_skinning["skin_weights"],
                triangles=rig_skinning["triangles"],
                rest_heads=rig_skinning["rest_heads"],
                rest_R=rig_skinning["rest_R"],
            )
        meta["calibration"] = calib
        Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        visibility = SurfaceVisibilitySampler(arm_obj, sorted(meta["bones"]), calib["depth_scale"])
    signature = dataset_signature(blend_path, meta_path, num_samples, seed)
    rng = random.Random(seed)
    light_rng = random.Random(seed+10000)
    scene = bpy.context.scene
    if enhanced:
        scene.render.motion_blur_position = "CENTER"
    bones_meta = classify_rig_bones(
        meta["bones"],
        bbox_min=meta.get("bbox_min"),
        bbox_max=meta.get("bbox_max"),
    )
    bone_names = list(bones_meta.keys())
    records = []

    for idx in range(num_samples):
        use_blur = idx >= (num_samples // 2)
        arm_obj.animation_data_clear()
        arm_obj.animation_data_create()
        scene.frame_set(1)

        regime = select_pose_sampling_regime(idx, rng)
        pose_f1 = sample_in_limit_pose(arm_obj, bones_meta, rng, sample_idx=idx)
        if enhanced and use_blur:
            # Symmetric shutter samples around the labeled pose at frame 1.
            # Put keyframes on both sides; labels always refer to the center pose.
            use_speed_prior = mocap_priors_enabled()
            for b_name, rot in pose_f1.items():
                pb = arm_obj.pose.bones[b_name]
                b_info = bones_meta[b_name]
                lims = effective_sampling_limits_for_bone(b_info, enforce_anatomical_floor=True)
                speed_mul = (
                    0.55
                    * bone_motion_blur_speed_factor(
                        str(b_info.get("kind", "limb")),
                        str(b_info.get("sub_role", "")),
                        int(b_info.get("chain_index", 0)),
                    )
                    if use_speed_prior
                    else 1.0
                )
                delta = [rng.uniform(-0.5, 0.5) * speed_mul for _ in range(3)]
                if not pb.children or b_info.get("fixed_local") or b_info.get("kind") == "rigid":
                    delta[1] = 0.0
                if b_info.get("fixed_local") or b_info.get("kind") == "rigid":
                    delta = [0.0, 0.0, 0.0]
                for frame, sign in ((0, -1), (1, 0), (2, 1)):
                    pb.rotation_euler = Euler([float(np.clip(rot[i]+sign*delta[i], lims[i][0], lims[i][1])) for i in range(3)], "XYZ")
                    pb.keyframe_insert(data_path="rotation_euler", frame=frame)
            scene.frame_set(1)
            scene.render.use_motion_blur = True
            scene.render.motion_blur_shutter = light_rng.uniform(0.35, 0.9)
        elif use_blur:
            for b_name, rot in pose_f1.items():
                pb = arm_obj.pose.bones[b_name]
                pb.rotation_euler = Euler(rot, "XYZ")
                pb.keyframe_insert(data_path="rotation_euler", frame=1)
            scene.frame_set(2)
            for b_name in bone_names:
                info = bones_meta[b_name]
                pb = arm_obj.pose.bones[b_name]
                if info.get("role") not in ("root", "spine") and not info.get("fixed_local") and rng.random() < 0.45:
                    lims = effective_sampling_limits_for_bone(info, enforce_anatomical_floor=True)
                    delta = [rng.uniform(-0.35, 0.35) for _ in range(3)]
                    if not pb.children:
                        delta[1] = 0.0
                    rot2 = [
                        float(np.clip(pose_f1[b_name][i] + delta[i], lims[i][0], lims[i][1]))
                        for i in range(3)
                    ]
                    pb.rotation_euler = Euler(rot2, "XYZ")
                pb.keyframe_insert(data_path="rotation_euler", frame=2)
            scene.frame_set(1)
            scene.render.use_motion_blur = True
            scene.render.motion_blur_shutter = 0.65
        else:
            scene.render.use_motion_blur = False

        bpy.context.view_layer.update()
        lighting = randomize_studio_lighting(studio, light_rng) if enhanced else None
        bpy.context.view_layer.update()
        if enhanced:
            visibility.update()
        img_name = f"pose_{idx:05d}.jpg"
        render_split_screen_frame(cam_f, cam_s, img_dir / img_name, out_path, cam_right=cam_r)

        joints_gt = {}
        mw = arm_obj.matrix_world
        for b_name in bone_names:
            pb = arm_obj.pose.bones[b_name]
            head_w = mw @ pb.head
            tail_w = mw @ pb.tail
            uv_f_h, uv_s_h = project_3d_to_dual_ortho_pixels(head_w, calib)
            uv_f_t, uv_s_t = project_3d_to_dual_ortho_pixels(tail_w, calib)
            joints_gt[b_name] = {
                "head_3d": [round(float(head_w.x), 5), round(float(head_w.y), 5), round(float(head_w.z), 5)],
                "tail_3d": [round(float(tail_w.x), 5), round(float(tail_w.y), 5), round(float(tail_w.z), 5)],
                "front_uv_head": uv_f_h,
                "side_uv_head": uv_s_h,
                "front_uv_tail": uv_f_t,
                "side_uv_tail": uv_s_t,
                "euler_xyz": [round(float(x), 5) for x in pose_f1[b_name]],
            }
            if cam_r is not None:
                joints_gt[b_name]["right_uv_head"] = project_3d_to_right_ortho_pixels(head_w, calib)
                joints_gt[b_name]["right_uv_tail"] = project_3d_to_right_ortho_pixels(tail_w, calib)
            if enhanced:
                aux_cameras = [("front", cam_f), ("side", cam_s)]
                if cam_r is not None:
                    aux_cameras.append(("right", cam_r))
                for view, camera in aux_cameras:
                    for endpoint, point in (("head", head_w), ("tail", tail_w)):
                        joints_gt[b_name][f"{view}_{endpoint}_aux"] = visibility.label(point, b_name, camera, calib)
        record_item: dict[str, Any] = {
            "image": img_name,
            "motion_blur": use_blur,
            "sampling_regime": regime,
            "joints": joints_gt,
            "label_frame": 1,
            "shutter": float(scene.render.motion_blur_shutter) if use_blur else 0.0,
            "lighting": lighting,
        }
        if dense_enabled and rig_skinning is not None:
            posed_heads = np.zeros((len(sorted_bones_for_skin), 3), dtype=np.float64)
            posed_R = np.zeros((len(sorted_bones_for_skin), 3, 3), dtype=np.float64)
            for b_idx, b_name in enumerate(sorted_bones_for_skin):
                pb = arm_obj.pose.bones[b_name]
                hw = mw @ pb.head
                rm = (mw @ pb.matrix).to_3x3()
                posed_heads[b_idx] = [float(hw.x), float(hw.y), float(hw.z)]
                posed_R[b_idx] = [[float(v) for v in row] for row in rm]
            posed_verts = lbs_deform_vertices_numpy(
                rig_skinning["rest_verts"],
                rig_skinning["skin_indices"],
                rig_skinning["skin_weights"],
                rig_skinning["rest_heads"],
                rig_skinning["rest_R"],
                posed_heads,
                posed_R,
            )
            dense_payload: dict[str, np.ndarray] = {}
            dense_views_to_render = ["front", "side"] + (["right"] if cam_r is not None else [])
            for v_key in dense_views_to_render:
                d_map = rasterize_ortho_dense_rest_map_numpy(
                    posed_verts,
                    rig_skinning["rest_verts"],
                    rig_skinning["triangles"],
                    calib,
                    view=v_key,
                    grid_w=dense_hm_w,
                    grid_h=dense_hm_h,
                )
                dense_payload[f"{v_key}_mask"] = (d_map["mask"] > 0.5).astype(np.uint8)
                dense_payload[f"{v_key}_rest_xyz"] = d_map["rest_xyz"].astype(np.float16)
            dense_file = f"pose_{idx:05d}_dense.npz"
            np.savez_compressed(dense_dir / dense_file, **dense_payload)
            record_item["dense_map"] = dense_file
            if mocap_priors_enabled():
                stretch_metric = compute_mesh_edge_stretch_metric(
                    rig_skinning["rest_verts"], posed_verts, rig_skinning["triangles"]
                )
                record_item["edge_stretch_passed"] = stretch_metric["passed"]
                record_item["bad_edge_ratio"] = stretch_metric["bad_edge_ratio"]
        records.append(record_item)

    coverage_audit = audit_pose_dataset_coverage(records, bones_meta, enforce_anatomical_floor=True)
    with open(out_path / "pose_coverage_audit.json", "w", encoding="utf-8") as f:
        json.dump(coverage_audit, f, indent=2, ensure_ascii=False)
    if not coverage_audit.get("passed", False):
        raise RuntimeError(
            f"[Stage 2 Pose Coverage Audit Gate Failed] {coverage_audit.get('failures')}"
        )

    with open(out_path / "keypoints_gt.json", "w", encoding="utf-8") as f:
        json.dump({"schema": DATASET_SCHEMA if enhanced else "monster-pose-legacy-v1",
                   "signature": signature, "depth_visibility_enabled": enhanced,
                   "dense_surface_enabled": dense_enabled,
                   "bone_names": sorted(bone_names), "visibility_method": "skin_weight_surface_proxy_v1" if enhanced else None,
                   "coverage_audit": coverage_audit,
                   "calibration": calib, "samples": records}, f, indent=2, ensure_ascii=False)


def synthesize_stage4_exam_sequence(
    blend_path: str, meta_path: str, out_dir: str, total_frames: int = 96, key_step: int = 12, seed: int = 100
) -> None:
    """Synthesizes a smooth, continuous constrained random animation sequence (96 frames = 4s @ 24 FPS)."""
    require_bpy()
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    arm_obj = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
    saved_calib = meta.get("calibration", {})
    saved_3view = (int(saved_calib.get("num_views", 2)) == 3) if "num_views" in saved_calib else None
    studio = setup_dual_orthographic_studio(
        meta["bbox_min"],
        meta["bbox_max"],
        panel_w=saved_calib.get("panel_width"),
        panel_h=saved_calib.get("panel_height"),
        enable_3view=saved_3view,
    )
    cam_f, cam_s, cam_r, calib = (
        studio["cam_front"],
        studio["cam_side"],
        studio.get("cam_right"),
        studio["calibration"],
    )

    rng = random.Random(seed)
    bones_meta = meta["bones"]
    arm_obj.animation_data_clear()
    arm_obj.animation_data_create()

    for f_idx in range(0, total_frames + 1, key_step):
        bpy.context.scene.frame_set(f_idx)
        sample_in_limit_pose(arm_obj, bones_meta, rng)
        for pbone in arm_obj.pose.bones:
            pbone.keyframe_insert(data_path="rotation_euler", frame=f_idx)

    if arm_obj.animation_data and arm_obj.animation_data.action:
        for fc in arm_obj.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"
                kp.handle_left_type = "AUTO_CLAMPED"
                kp.handle_right_type = "AUTO_CLAMPED"

    out_path = Path(out_dir)
    frames_dir = out_path / "exam_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.use_motion_blur = True
    bpy.context.scene.render.motion_blur_shutter = 0.5

    gt_trajectory = []
    mw = arm_obj.matrix_world
    for f_idx in range(total_frames):
        bpy.context.scene.frame_set(f_idx)
        bpy.context.view_layer.update()
        frame_file = f"exam_{f_idx:04d}.jpg"
        render_split_screen_frame(cam_f, cam_s, frames_dir / frame_file, out_path, cam_right=cam_r)

        frame_bones = {}
        for b_name in bones_meta:
            pb = arm_obj.pose.bones[b_name]
            h_w = mw @ pb.head
            t_w = mw @ pb.tail
            rot_mat = (mw @ pb.matrix).to_3x3()
            frame_bones[b_name] = {
                "head_3d": [round(float(h_w.x), 5), round(float(h_w.y), 5), round(float(h_w.z), 5)],
                "tail_3d": [round(float(t_w.x), 5), round(float(t_w.y), 5), round(float(t_w.z), 5)],
                "rot_mat_world": [[round(float(v), 5) for v in row] for row in rot_mat],
            }
        gt_trajectory.append({"frame": f_idx, "image": frame_file, "bones": frame_bones})

    with open(out_path / "exam_gt.json", "w", encoding="utf-8") as f:
        json.dump({"calibration": calib, "frames": gt_trajectory}, f, indent=2, ensure_ascii=False)


# ==============================================================================
# 4. Stage 6: Bounded Local-Euler + Multi-Child Observability IK & Savitzky-Golay Baker
# ==============================================================================

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


def minimum_swing_matrix_numpy(curr_dir: np.ndarray, target_dir: np.ndarray) -> np.ndarray:
    """Computes the minimum-angle rotation matrix in SO(3) mapping `curr_dir -> target_dir`."""
    u = curr_dir / max(1e-8, float(np.linalg.norm(curr_dir)))
    v = target_dir / max(1e-8, float(np.linalg.norm(target_dir)))
    cross = np.cross(u, v)
    sin_a = float(np.linalg.norm(cross))
    cos_a = float(np.clip(np.dot(u, v), -1.0, 1.0))
    if sin_a < 1e-6:
        return np.eye(3, dtype=np.float64) if cos_a >= 0.0 else -np.eye(3, dtype=np.float64)
    axis = cross / sin_a
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + sin_a * K + (1.0 - cos_a) * (K @ K)


def solve_rx_rz_given_ry(d_loc: np.ndarray, ry: float = 0.0) -> tuple[float, float]:
    """Exact closed-form inverse of Blender's `'XYZ'` PoseBone local Euler matrix:
    `Rz(rz) @ Ry(ry) @ Rx(rx) @ [0, 1, 0]^T = d_loc`.
    Recovers the exact `(rx, rz)` pitch and yaw angles given longitudinal roll `ry`,
    automatically accounting for the Baker-Campbell-Hausdorff `1/2 * rx * rz` Euler coupling twist!
    """
    cy = math.cos(ry)
    sy = math.sin(ry)
    sx = float(np.clip(d_loc[2] / max(1e-6, cy), -1.0, 1.0))
    rx = math.asin(sx)
    cx = math.cos(rx)
    a = sy * sx
    b = cx
    sin_rz = a * float(d_loc[1]) - b * float(d_loc[0])
    cos_rz = a * float(d_loc[0]) + b * float(d_loc[1])
    rz = math.atan2(sin_rz, cos_rz)
    return rx, rz


def build_clamped_local_euler_rotation(
    d_loc: np.ndarray,
    ry: float,
    rx_lims: tuple[float, float] | list[float],
    rz_lims: tuple[float, float] | list[float],
) -> np.ndarray:
    """Constructs the PoseBone local rotation matrix `R_loc` whose Y-axis points along unit vector
    `d_loc` while locking longitudinal roll to `ry` and clamping `(rx, rz)` within anatomical limits
    (`rx_lims`, `rz_lims`) across both inverse-sine branches without any post-clamp `minimum_swing` override.
    """
    return build_bounded_local_euler_rotation(d_loc, ry, rx_lims, rz_lims)


def compute_swing_and_child_plane_twist_matrix(
    curr_dir: np.ndarray,
    target_dir: np.ndarray,
    curr_child_dir: np.ndarray | None = None,
    target_child_dir: np.ndarray | None = None,
    max_twist_rad: float = math.radians(6.0),
) -> np.ndarray:
    """Observability-damped Swing + Twist rotation matrix in SO(3).
    Uses smooth quadratic observability weighting near collinearity (`n_curr, n_targ > 0.16`)
    and a conservative anatomical twist cap so near-collinear limbs never snap by +/-32 deg.
    """
    v = target_dir / max(1e-8, float(np.linalg.norm(target_dir)))
    R_swing = minimum_swing_matrix_numpy(curr_dir, v)
    if curr_child_dir is None or target_child_dir is None or max_twist_rad <= 1e-6:
        return R_swing

    c_curr = R_swing @ (curr_child_dir / max(1e-8, float(np.linalg.norm(curr_child_dir))))
    c_targ = target_child_dir / max(1e-8, float(np.linalg.norm(target_child_dir)))

    p_curr = c_curr - np.dot(c_curr, v) * v
    p_targ = c_targ - np.dot(c_targ, v) * v
    n_curr = float(np.linalg.norm(p_curr))
    n_targ = float(np.linalg.norm(p_targ))
    if n_curr < 0.16 or n_targ < 0.16:
        return R_swing

    obs = min(1.0, ((min(n_curr, n_targ) - 0.16) / 0.25) ** 2)
    p_curr /= n_curr
    p_targ /= n_targ
    twist_cos = float(np.clip(np.dot(p_curr, p_targ), -1.0, 1.0))
    twist_sin = float(np.dot(v, np.cross(p_curr, p_targ)))
    twist_angle = float(np.clip(obs * math.atan2(twist_sin, twist_cos), -max_twist_rad, max_twist_rad))

    Kv = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    R_twist = (
        np.eye(3, dtype=np.float64)
        + math.sin(twist_angle) * Kv
        + (1.0 - math.cos(twist_angle)) * (Kv @ Kv)
    )
    return R_twist @ R_swing


def solve_bounded_bone_rotation_matrix(
    b_name: str,
    R_init: np.ndarray,
    frame_bones: dict[str, Any],
    bones_meta: dict[str, Any],
    rest_R: dict[str, np.ndarray],
    children_map: dict[str, list[str]],
    curr_ry_map: dict[str, float],
    prev_ry_map: dict[str, float],
    raw_obs_bones: dict[str, dict[str, Any]] | None = None,
) -> tuple[np.ndarray, float]:
    """Solves bone `b_name`'s 3x3 world rotation matrix `R_final` using Layer 3 Kind-Specific
    Solvers (`solve_bone_rotation_by_kind`) when taxonomy metadata is present, or falls back
    to Bounded Local-Euler + Multi-Child Observability & Rigid Child-Head Lever-Arm IK.
    """
    info = bones_meta.get(b_name, {})
    if "kind" in info or "fixed_local" in info:
        return solve_bone_rotation_by_kind(
            b_name=b_name,
            R_init=R_init,
            frame_bones=frame_bones,
            bones_meta=bones_meta,
            rest_R=rest_R,
            children_map=children_map,
            curr_ry_map=curr_ry_map,
            prev_ry_map=prev_ry_map,
            raw_obs_bones=raw_obs_bones,
        )
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
    lims = info.get("limits_rad", default_lims)
    rx_lims = (float(lims[0][0]), float(lims[0][1]))
    ry_lo, ry_hi = float(lims[1][0]), float(lims[1][1])
    rz_lims = (float(lims[2][0]), float(lims[2][1]))

    p_name = info.get("parent")
    corr = float(info.get("chain_corr", 0.0))
    ry_prior = float(np.clip(corr * curr_ry_map.get(p_name, 0.0), ry_lo, ry_hi)) if (corr > 0.0 and p_name in curr_ry_map) else 0.0
    if b_name in prev_ry_map:
        ry_prior = 0.55 * prev_ry_map[b_name] + 0.45 * ry_prior

    valid_children = [c for c in children_map.get(b_name, []) if c in frame_bones and c in rest_R]
    if not valid_children or (ry_hi - ry_lo) < 1e-4:
        ry_best = 0.0 if not valid_children else ry_prior
        R_loc = build_clamped_local_euler_rotation(d_loc, ry_best, rx_lims, rz_lims)
        return R_init @ R_loc, ry_best

    rest_h_b = np.array(info.get("rest_head_world", h_b), dtype=np.float64)
    rest_t_b = np.array(info.get("rest_tail_world", t_b), dtype=np.float64)

    grid_ry = np.linspace(ry_lo, ry_hi, 11)
    best_ry = ry_prior
    best_cost = 1e18
    for ry_cand in grid_ry:
        ry_f = float(ry_cand)
        R_cand = R_init @ build_clamped_local_euler_rotation(d_loc, ry_f, rx_lims, rz_lims)
        cost = 1.25 * ((ry_f - ry_prior) ** 2)
        for c in valid_children:
            c_info = bones_meta.get(c, {})
            c_data = frame_bones[c]
            h_c = np.array(c_data["head_3d"], dtype=np.float64)
            t_c = np.array(c_data["tail_3d"], dtype=np.float64)

            # A) Disconnected child-head rigid lever-arm constraint (only when child head is offset from parent tail)
            if "rest_head_world" in c_info and "rest_head_world" in info and "rest_tail_world" in info:
                rest_h_c = np.array(c_info["rest_head_world"], dtype=np.float64)
                gap_rest = float(np.linalg.norm(rest_h_c - rest_t_b))
                if gap_rest > 0.008:
                    r_rest = rest_h_c - rest_h_b
                    r_curr = (R_cand @ rest_R[b_name].T) @ r_rest
                    r_targ = h_c - h_b
                    p_c = r_curr - np.dot(r_curr, v) * v
                    p_t = r_targ - np.dot(r_targ, v) * v
                    nc, nt = float(np.linalg.norm(p_c)), float(np.linalg.norm(p_t))
                    if nc > 0.015 and (0.45 * nc <= nt <= 1.85 * nc):
                        ang_diff = math.atan2(
                            float(np.dot(v, np.cross(p_c / nc, p_t / nt))),
                            float(np.clip(np.dot(p_c / nc, p_t / nt), -1.0, 1.0)),
                        )
                        w_lever = min(12.0, (nc / 0.06) ** 2)
                        cost += w_lever * (ang_diff ** 2)

            # B) Child shaft anatomical limit & hinge-plane constraint
            vec_c = t_c - h_c
            norm_c = float(np.linalg.norm(vec_c))
            if norm_c < 1e-6:
                continue
            v_c = vec_c / norm_c
            sin_perp = float(np.linalg.norm(v_c - np.dot(v_c, v) * v))
            if sin_perp > 0.16:
                c_lims = c_info.get("limits_rad", default_lims)
                R_c_init = R_cand @ (rest_R[b_name].T @ rest_R[c])
                d_c_loc = R_c_init.T @ v_c
                sx_c = float(np.clip(d_c_loc[2], -1.0, 1.0))
                rx1 = math.asin(sx_c)
                rx2 = (math.pi - rx1) if rx1 >= 0.0 else (-math.pi - rx1)
                c_best_pen = 1e18
                c_rz_span = float(c_lims[2][1]) - float(c_lims[2][0])
                for rxc in (rx1, rx2):
                    pen_x = max(0.0, float(c_lims[0][0]) - rxc) + max(0.0, rxc - float(c_lims[0][1]))
                    cxc = math.cos(rxc)
                    rzc = math.atan2(-cxc * float(d_c_loc[0]), cxc * float(d_c_loc[1]))
                    pen_z = max(0.0, float(c_lims[2][0]) - rzc) + max(0.0, rzc - float(c_lims[2][1]))
                    hinge_pull = 0.25 * (rzc ** 2) if c_rz_span <= math.radians(15.0) else 0.0
                    pen = (pen_x ** 2) + 8.0 * (pen_z ** 2) + hinge_pull
                    if pen < c_best_pen:
                        c_best_pen = pen
                cost += (sin_perp ** 2) * c_best_pen * 6.0

        if cost < best_cost:
            best_cost = cost
            best_ry = ry_f

    R_loc_best = build_clamped_local_euler_rotation(d_loc, best_ry, rx_lims, rz_lims)
    return R_init @ R_loc_best, best_ry


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
    """Converts a quaternion `[w, x, y, z]` to a 3x3 rotation matrix."""
    w, x, y, z = q / max(1e-12, float(np.linalg.norm(q)))
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def smooth_rotation_matrices_savgol(R_seq: np.ndarray, gaussian_passes: int = 1) -> np.ndarray:
    """Applies zero-phase 5-tap quadratic Savitzky-Golay filtering (`[-3, 12, 17, 12, -3]/35`)
    plus optional 3-tap Gaussian smoothing on antipodal-aligned unit quaternions.
    Preserves full motion amplitude while eliminating frame-to-frame high-frequency jitter.
    """
    T = R_seq.shape[0]
    if T < 5:
        return R_seq
    qs = np.zeros((T, 4), dtype=np.float64)
    for t in range(T):
        q = mat_to_quat_wxyz_numpy(R_seq[t])
        if t > 0 and float(np.dot(q, qs[t - 1])) < 0.0:
            q = -q
        qs[t] = q
    pad5 = np.pad(qs, ((2, 2), (0, 0)), mode="edge")
    qs = (-3.0 * pad5[:-4] + 12.0 * pad5[1:-3] + 17.0 * pad5[2:-2] + 12.0 * pad5[3:-1] - 3.0 * pad5[4:]) / 35.0
    for ts in range(T):
        qs[ts] /= max(1e-12, float(np.linalg.norm(qs[ts])))
    if mocap_priors_enabled() and gaussian_passes > 0:
        qs = dejitter_velocity_gated(qs, passes=max(1, gaussian_passes), strength=0.45)
        for ts in range(T):
            qs[ts] /= max(1e-12, float(np.linalg.norm(qs[ts])))
    else:
        for _ in range(gaussian_passes):
            pad3 = np.pad(qs, ((1, 1), (0, 0)), mode="edge")
            qs = 0.22 * pad3[:-2] + 0.56 * pad3[1:-1] + 0.22 * pad3[2:]
            for ts in range(T):
                qs[ts] /= max(1e-12, float(np.linalg.norm(qs[ts])))
    out = np.zeros_like(R_seq)
    for t in range(T):
        out[t] = quat_wxyz_to_mat_numpy(qs[t])
    return out


def bake_3d_joints_to_animated_glb(
    blend_path: str,
    joints_3d_json: str,
    out_glb_path: str,
    fps: int = 24,
    meta_path: str | None = None,
) -> None:
    """Retargets triangulated 3D joint trajectories onto the Blender Armature in parent-to-child
    topological order using:
      1. Frame-start Rest-Pose Reset (eliminates multi-frame quaternion holonomy twist drift);
      2. Bounded Local-Euler + Multi-Child Observability & Rigid Child-Head Lever-Arm IK Solver;
      3. Zero-phase 5-Frame Quaternion Savitzky-Golay Temporal Filtering;
      4. Continuous Quaternion Sign Alignment (`q.dot(prev_q) >= 0`) for smooth glTF interpolation.
    """
    require_bpy()
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    with open(joints_3d_json, "r", encoding="utf-8") as f:
        traj_data = json.load(f)

    bones_meta: dict[str, Any] = {}
    if meta_path and Path(meta_path).exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            bones_meta = json.load(f).get("bones", {})
    elif isinstance(traj_data.get("bones_meta"), dict):
        bones_meta = traj_data["bones_meta"]
    else:
        cand_meta = Path(blend_path).parent / "studio_rig_meta.json"
        if cand_meta.exists():
            with open(cand_meta, "r", encoding="utf-8") as f:
                bones_meta = json.load(f).get("bones", {})

    frames_list = traj_data["frames"]
    total_frames = len(frames_list)
    scene = bpy.context.scene
    scene.render.fps = fps
    scene.frame_start = 0
    scene.frame_end = max(0, total_frames - 1)

    arm_obj = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
    mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    if not bones_meta and mesh_objs:
        _, auto_meta = classify_anonymous_bones_and_assign_limits(
            mesh_objs[0], arm_obj, canonicalize_names=False
        )
        bones_meta = auto_meta.get("bones", {})

    arm_obj.animation_data_clear()
    arm_obj.animation_data_create()

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")

    topo_bones: list[Any] = []
    visited: set[str] = set()

    def visit(pb: Any) -> None:
        if pb.name in visited:
            return
        if pb.parent:
            visit(pb.parent)
        visited.add(pb.name)
        topo_bones.append(pb)

    for pb in arm_obj.pose.bones:
        visit(pb)

    mw_arm = arm_obj.matrix_world
    mw_inv = mw_arm.inverted()

    rest_R: dict[str, np.ndarray] = {}
    children_map: dict[str, list[str]] = {pb.name: [c.name for c in pb.children] for pb in topo_bones}
    for pb in topo_bones:
        b_name = pb.name
        R0 = np.array((mw_arm @ pb.bone.matrix_local).to_3x3(), dtype=np.float64)
        rest_R[b_name] = R0
        info = bones_meta.setdefault(b_name, {})
        info.setdefault("parent", pb.parent.name if pb.parent else None)
        info["children"] = children_map[b_name]
        h_w = mw_arm @ pb.bone.head_local
        t_w = mw_arm @ pb.bone.tail_local
        info.setdefault("rest_head_world", [float(h_w.x), float(h_w.y), float(h_w.z)])
        info.setdefault("rest_tail_world", [float(t_w.x), float(t_w.y), float(t_w.z)])
        info.setdefault("rest_length", float((t_w - h_w).length))
        info.setdefault("rest_rot_mat_world", [[float(v) for v in row] for row in R0])

    bones_meta = classify_rig_bones(bones_meta)

    first_frame_bones = frames_list[0].get("bones", {}) if frames_list else {}
    any_precomputed_rot = any(
        isinstance(b_info, dict) and "rot_mat_world" in b_info
        for b_info in first_frame_bones.values()
    )
    if total_frames > 0 and not any_precomputed_rot:
        sorted_bnames = sorted(bones_meta.keys())
        num_b = len(sorted_bnames)
        pts_3d = np.zeros((total_frames, 2 * num_b, 3), dtype=np.float64)
        trust = np.full((total_frames, 2 * num_b), 0.90, dtype=np.float64)
        for t_i, f_item in enumerate(frames_list):
            fb = f_item.get("bones", {})
            for b_i, bn in enumerate(sorted_bnames):
                b_obs = fb.get(bn, {})
                h3 = b_obs.get("head_3d", bones_meta[bn].get("rest_head_world", [0.0, 0.0, 0.0]))
                t3 = b_obs.get("tail_3d", bones_meta[bn].get("rest_tail_world", [0.0, 0.0, 0.04]))
                c_val = float(b_obs.get("confidence", 0.90))
                pts_3d[t_i, 2 * b_i] = h3
                pts_3d[t_i, 2 * b_i + 1] = t3
                trust[t_i, 2 * b_i] = c_val
                trust[t_i, 2 * b_i + 1] = c_val
        frames_solved, _ = reconstruct_frames_with_kind_solvers(
            pts_3d=pts_3d,
            trust=trust,
            bone_names=sorted_bnames,
            classified_bones=bones_meta,
        )
        frames_list = frames_solved

    # Step 1: Solve bounded world rotation matrices across all frames in topological order
    solved_world_R: dict[str, np.ndarray] = {
        pb.name: np.zeros((total_frames, 3, 3), dtype=np.float64) for pb in topo_bones
    }
    has_precomputed_rot = False
    prev_ry_map: dict[str, float] = {}
    for f_idx, frame_item in enumerate(frames_list):
        b_dict = frame_item["bones"]
        curr_ry_map: dict[str, float] = {}
        posed_R: dict[str, np.ndarray] = {}
        for pb in topo_bones:
            b_name = pb.name
            p_name = pb.parent.name if pb.parent else None
            if p_name and p_name in posed_R:
                R_init = posed_R[p_name] @ (rest_R[p_name].T @ rest_R[b_name])
            else:
                R_init = rest_R[b_name]
            if b_name not in b_dict:
                posed_R[b_name] = R_init
                solved_world_R[b_name][f_idx] = R_init
                continue
            if "rot_mat_world" in b_dict[b_name]:
                R_pre = np.array(b_dict[b_name]["rot_mat_world"], dtype=np.float64)
                posed_R[b_name] = R_pre
                solved_world_R[b_name][f_idx] = R_pre
                has_precomputed_rot = True
                continue
            R_final, ry_best = solve_bounded_bone_rotation_matrix(
                b_name=b_name,
                R_init=R_init,
                frame_bones=b_dict,
                bones_meta=bones_meta,
                rest_R=rest_R,
                children_map=children_map,
                curr_ry_map=curr_ry_map,
                prev_ry_map=prev_ry_map,
                raw_obs_bones=b_dict,
            )
            posed_R[b_name] = R_final
            curr_ry_map[b_name] = ry_best
            prev_ry_map[b_name] = ry_best
            solved_world_R[b_name][f_idx] = R_final

    # Step 2: Apply 5-frame Quaternion Savitzky-Golay temporal smoothing when not already smoothed
    if total_frames >= 5 and not has_precomputed_rot:
        for pb in topo_bones:
            b_name = pb.name
            depth = int(bones_meta.get(b_name, {}).get("depth", 1))
            passes = 1 if depth >= 2 else 0
            solved_world_R[b_name] = smooth_rotation_matrices_savgol(
                solved_world_R[b_name], gaussian_passes=passes
            )

    # Step 3: Bake smoothed world rotation matrices into Blender PoseBones
    prev_quats: dict[str, Any] = {}
    for f_idx, frame_item in enumerate(frames_list):
        scene.frame_set(f_idx)
        b_dict = frame_item["bones"]

        for pb in topo_bones:
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
            if pb.parent is None:
                pb.location = Vector((0.0, 0.0, 0.0))
        bpy.context.view_layer.update()

        for pbone in topo_bones:
            b_name = pbone.name
            if b_name not in b_dict:
                continue
            head_gt = Vector(b_dict[b_name]["head_3d"])
            if pbone.parent is None:
                delta_arm = mw_inv @ head_gt - pbone.bone.head_local
                pbone.location = pbone.bone.matrix_local.to_3x3().inverted() @ delta_arm
                pbone.keyframe_insert(data_path="location", frame=f_idx)

            R_targ_w = solved_world_R[b_name][f_idx]
            R_curr_w = np.array((mw_arm @ pbone.matrix).to_3x3(), dtype=np.float64)
            R_local_delta = R_curr_w.T @ R_targ_w
            qw, qx, qy, qz = mat_to_quat_wxyz_numpy(R_local_delta)
            local_delta = Quaternion((float(qw), float(qx), float(qy), float(qz)))
            q_new = pbone.rotation_quaternion @ local_delta

            if b_name in prev_quats and q_new.dot(prev_quats[b_name]) < 0.0:
                q_new = -q_new
            prev_quats[b_name] = q_new.copy()

            pbone.rotation_quaternion = q_new
            pbone.keyframe_insert(data_path="rotation_quaternion", frame=f_idx)
            bpy.context.view_layer.update()

    bpy.ops.object.mode_set(mode="OBJECT")
    Path(out_glb_path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(out_glb_path),
        export_format="GLB",
        export_animations=True,
        export_frame_range=True,
    )


def verify_exam_glb_roundtrip(
    baked_glb_path: str,
    exam_gt_json_path: str,
    out_report_path: str,
) -> dict[str, Any]:
    """Re-imports a baked Stage 4 exam `.glb` into Blender, steps through all frames, and computes
    BOTH the 3D Bone Vector Directional Error (`directional_mpjre_deg`) AND the Full SO(3) 3x3
    Rotation Matrix Geodesic Error (`full_rot_mpjre_deg`, including Axial Twist / Roll) against
    `exam_gt.json`.
    """
    require_bpy()
    _, arm_obj = load_glb_and_find_rig(baked_glb_path)
    with open(exam_gt_json_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    dir_errors: dict[str, list[float]] = {}
    full_rot_errors: dict[str, list[float]] = {}
    non_leaf_rot_errors: dict[str, list[float]] = {}

    mw = arm_obj.matrix_world
    leaf_bones = {b.name for b in arm_obj.data.bones if len(b.children) == 0}

    for f_item in gt_data["frames"]:
        f_idx = int(f_item["frame"])
        bpy.context.scene.frame_set(f_idx)
        bpy.context.view_layer.update()
        for b_name, b_gt in f_item["bones"].items():
            pb = arm_obj.pose.bones.get(b_name)
            if not pb:
                continue
            h_pr = mw @ pb.head
            t_pr = mw @ pb.tail
            v_pr = np.array([h_pr.x, h_pr.y, h_pr.z]) - np.array([t_pr.x, t_pr.y, t_pr.z])
            v_pr = -v_pr  # tail - head
            v_gt = np.array(b_gt["tail_3d"], dtype=np.float64) - np.array(b_gt["head_3d"], dtype=np.float64)
            n_gt = float(np.linalg.norm(v_gt))
            n_pr = float(np.linalg.norm(v_pr))
            if n_gt > 1e-6 and n_pr > 1e-6:
                cos_d = float(np.clip(np.dot(v_gt / n_gt, v_pr / n_pr), -1.0, 1.0))
                dir_errors.setdefault(b_name, []).append(math.degrees(math.acos(cos_d)))

            if "rot_mat_world" in b_gt:
                R_gt = np.array(b_gt["rot_mat_world"], dtype=np.float64)
                R_pr = np.array((mw @ pb.matrix).to_3x3(), dtype=np.float64)
                tr = float(np.clip((np.trace(R_gt.T @ R_pr) - 1.0) * 0.5, -1.0, 1.0))
                ang_so3 = math.degrees(math.acos(tr))
                full_rot_errors.setdefault(b_name, []).append(ang_so3)
                if b_name not in leaf_bones:
                    non_leaf_rot_errors.setdefault(b_name, []).append(ang_so3)

    per_bone_dir = {k: round(float(np.mean(v)), 3) for k, v in dir_errors.items()}
    per_bone_full = {k: round(float(np.mean(v)), 3) for k, v in full_rot_errors.items()}
    per_bone_non_leaf = {k: round(float(np.mean(v)), 3) for k, v in non_leaf_rot_errors.items()}

    dir_mpjre = round(float(np.mean(list(per_bone_dir.values()))), 3) if per_bone_dir else 999.0
    dir_max = round(float(np.max(list(per_bone_dir.values()))), 3) if per_bone_dir else 999.0
    full_mpjre = round(float(np.mean(list(per_bone_full.values()))), 3) if per_bone_full else dir_mpjre
    full_max = round(float(np.max(list(per_bone_full.values()))), 3) if per_bone_full else dir_max
    non_leaf_mpjre = round(float(np.mean(list(per_bone_non_leaf.values()))), 3) if per_bone_non_leaf else full_mpjre

    report = {
        "glb_roundtrip_verified": True,
        "baked_glb": str(baked_glb_path),
        "directional_mpjre_deg": dir_mpjre,
        "directional_max_bone_deg": dir_max,
        "full_rot_mpjre_deg": full_mpjre,
        "full_rot_max_bone_deg": full_max,
        "non_leaf_full_rot_mpjre_deg": non_leaf_mpjre,
        "exam_passed": (dir_mpjre < 3.5) and (non_leaf_mpjre < 3.5) and (full_max < 8.0),
        "per_bone_directional_deg": per_bone_dir,
        "per_bone_full_rot_deg": per_bone_full,
    }
    Path(out_report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(
        f"[GLB Round-Trip Exam] Dir MPJRE={dir_mpjre:.2f}° | Full SO(3) MPJRE={full_mpjre:.2f}° "
        f"(Non-Leaf={non_leaf_mpjre:.2f}°, Max={full_max:.2f}°)"
    )
    return report


def sop_preflight_check(input_glb: str, out_dir: str, monster_id: str = "monster") -> dict[str, Any]:
    """SOP Stage 1: Headless Blender 7-point model preflight and decoration separation check."""
    require_bpy()
    clean_scene()
    bpy.ops.import_scene.gltf(filepath=str(input_glb))
    bpy.context.view_layer.update()

    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    anims = [a.name for a in getattr(bpy.data, "actions", [])]

    all_verts = []
    uv_layers = 0
    material_count = 0
    decor_entries = []
    for m_idx, m in enumerate(meshes):
        mw = m.matrix_world
        pts = [[float((mw @ v.co).x), float((mw @ v.co).y), float((mw @ v.co).z)] for v in m.data.vertices]
        all_verts.extend(pts)
        uv_layers += len(getattr(m.data, "uv_layers", []))
        material_count += len(getattr(m.data, "materials", []))
        if m_idx > 0:
            decor_entries.append(
                {
                    "name": m.name,
                    "vertex_count": len(pts),
                    "matrix_world": [[float(v) for v in row] for row in mw],
                }
            )

    if all_verts:
        arr = np.asarray(all_verts, dtype=np.float64)
        b_min = [round(float(x), 4) for x in np.min(arr, axis=0)]
        b_max = [round(float(x), 4) for x in np.max(arr, axis=0)]
    else:
        b_min, b_max = [-0.5, -0.5, 0.0], [0.5, 0.5, 1.8]

    bone_names = [b.name for b in arms[0].data.bones] if arms else []
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    report = {
        "monster_id": monster_id,
        "mesh_count": len(meshes),
        "armature_count": len(arms),
        "animation_clips": anims,
        "bone_names": bone_names,
        "uv_layer_count": uv_layers,
        "material_count": material_count,
        "bbox_min": b_min,
        "bbox_max": b_max,
        "decorations": decor_entries,
    }
    (out_p / "bpy_preflight.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def sop_prepare_g1_candidates(
    candidates: list[str],
    out_dir: str,
    decor_manifest: str | None = None,
    archetype: str = "humanoid",
) -> dict[str, Any]:
    """SOP Stage 2: Headless Blender 4-candidate canonical bone naming & Octahedral bone view render."""
    require_bpy()
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    results = []
    for idx, c_path in enumerate(candidates[:4]):
        mesh_obj, arm_obj = load_glb_and_find_rig(c_path)
        arm_obj.data.display_type = "OCTAHEDRAL"
        arm_obj.show_in_front = True
        _, rig_meta = classify_anonymous_bones_and_assign_limits(mesh_obj, arm_obj)
        c_dir = out_p / f"candidate_{idx}"
        c_dir.mkdir(parents=True, exist_ok=True)
        c_meta_path = c_dir / "studio_rig_meta.json"
        c_meta_path.write_text(json.dumps(rig_meta, indent=2, ensure_ascii=False), encoding="utf-8")
        results.append(
            {
                "candidate_id": f"candidate_{idx}",
                "path": str(c_path),
                "meta_path": str(c_meta_path),
                "bones": sorted(rig_meta.get("bones", {}).keys()),
                "archetype": archetype,
            }
        )
    summary = {"candidates": results, "decor_manifest": decor_manifest, "archetype": archetype}
    (out_p / "bpy_g1_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def sop_verify_delivery_glb(glb_path: str, is_loop: bool = False, out_report: str | None = None) -> dict[str, Any]:
    """SOP Stage 5: Headless Blender GLB round-trip & loop seam verification."""
    require_bpy()
    clean_scene()
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    bpy.context.view_layer.update()
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        report = {"glb_path": str(glb_path), "valid": False, "reason": "no_armature"}
    else:
        arm_obj = arms[0]
        scene = bpy.context.scene
        f_start = int(scene.frame_start)
        f_end = int(scene.frame_end)
        scene.frame_set(f_start)
        bpy.context.view_layer.update()
        mw = arm_obj.matrix_world
        first_pos = {pb.name: np.array([float((mw @ pb.head).x), float((mw @ pb.head).y), float((mw @ pb.head).z)]) for pb in arm_obj.pose.bones}
        scene.frame_set(f_end)
        bpy.context.view_layer.update()
        mw = arm_obj.matrix_world
        last_pos = {pb.name: np.array([float((mw @ pb.head).x), float((mw @ pb.head).y), float((mw @ pb.head).z)]) for pb in arm_obj.pose.bones}
        jumps = [float(np.linalg.norm(last_pos[k] - first_pos[k])) for k in first_pos if k in last_pos]
        max_jump = round(max(jumps), 5) if jumps else 0.0
        report = {
            "glb_path": str(glb_path),
            "valid": True,
            "is_loop": bool(is_loop),
            "frame_start": f_start,
            "frame_end": f_end,
            "max_joint_seam_jump": max_jump,
            "seam_passed": (max_jump <= 0.12) if is_loop else True,
        }
    if out_report:
        Path(out_report).parent.mkdir(parents=True, exist_ok=True)
        Path(out_report).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless Blender 4.2 Studio Worker for Monster Pipeline")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "sanitize_input",
            "audit",
            "render_3000",
            "render_5000",
            "render_g2_50",
            "render_exam",
            "bake_ik",
            "verify_exam_glb",
            "preflight_check",
            "prepare_g1_candidates",
            "verify_delivery_glb",
        ],
    )
    parser.add_argument("--input_glb", help="Raw input GLB for sanitize_input or preflight_check mode")
    parser.add_argument("--monster_id", default="monster", help="Monster ID for SOP modes")
    parser.add_argument("--yaw_deg", type=float, default=0.0, help="Optional Z-axis rotation in degrees")
    parser.add_argument("--candidates", nargs="*", help="Candidate GLB files for Stage 1 audit or G1 pack")
    parser.add_argument("--decor_manifest", help="Path to decorations_manifest.json for G1 restoration")
    parser.add_argument("--archetype", default="humanoid", help="Motion archetype for G1 test motion")
    parser.add_argument("--blend_path", help="Path to rigged_pruned.blend")
    parser.add_argument("--meta_path", help="Path to studio_rig_meta.json")
    parser.add_argument("--out_dir", help="Output directory for renders / metadata")
    parser.add_argument("--joints_json", help="Input 3D joint trajectory JSON for Stage 6 IK baking")
    parser.add_argument("--out_glb", help="Output GLB path for sanitize_input or Stage 6 bake_ik")
    parser.add_argument("--glb_path", help="GLB path for verify_delivery_glb mode")
    parser.add_argument("--is_loop", type=int, default=0, help="Whether the action clip is a loop (1 or 0)")
    parser.add_argument("--exam_gt_json", help="Path to exam_gt.json for verify_exam_glb mode")
    parser.add_argument("--out_report", help="Output JSON report path for verify_exam_glb or verify_delivery_glb mode")
    parser.add_argument("--num_samples", type=int, default=DEFAULT_SYNTH_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exam_frames", type=int, default=96)
    args = parser.parse_args()

    if args.mode == "sanitize_input":
        sanitize_input_mesh_for_skintokens(args.input_glb, args.out_glb, yaw_deg=args.yaw_deg)

    elif args.mode == "preflight_check":
        sop_preflight_check(args.input_glb, args.out_dir, monster_id=args.monster_id)

    elif args.mode == "prepare_g1_candidates":
        sop_prepare_g1_candidates(
            args.candidates or [],
            args.out_dir,
            decor_manifest=args.decor_manifest,
            archetype=args.archetype,
        )

    elif args.mode == "audit":
        reports = [score_candidate_rig(c) for c in (args.candidates or [])]
        contact_path = Path(args.meta_path).parent / "candidates_contact_sheet.jpg"
        render_candidates_contact_sheet(reports, contact_path)

        best = max(reports, key=lambda r: r["score"])
        mesh_obj, arm_obj = load_glb_and_find_rig(best["path"])
        repair_air_bridging_chords_and_prune(
            mesh_obj,
            arm_obj,
            best["protruding_bones"],
            best.get("air_bridging_bones", []),
        )
        _, rig_meta = classify_anonymous_bones_and_assign_limits(mesh_obj, arm_obj)
        rig_meta["audit_reports"] = reports
        rig_meta["selected_candidate"] = best["path"]
        rig_meta["contact_sheet_path"] = str(contact_path)

        Path(args.blend_path).parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.blend_path))
        with open(args.meta_path, "w", encoding="utf-8") as f:
            json.dump(rig_meta, f, indent=2, ensure_ascii=False)
        print(f"[Stage 1 Audit Complete] Best candidate: {best['path']} (score={best['score']})")

    elif args.mode in ("render_3000", "render_5000", "render_g2_50"):
        samples = 50 if args.mode == "render_g2_50" and args.num_samples == DEFAULT_SYNTH_SAMPLES else args.num_samples
        generate_stage2_synthetic_dataset(args.blend_path, args.meta_path, args.out_dir, samples, seed=args.seed)
        print(f"[Stage 2 Render Complete] Generated {samples} dual-view pairs in {args.out_dir}")

    elif args.mode == "render_exam":
        synthesize_stage4_exam_sequence(args.blend_path, args.meta_path, args.out_dir, args.exam_frames)
        print(f"[Stage 4 Exam Sequence Complete] Rendered {args.exam_frames} continuous frames in {args.out_dir}")

    elif args.mode == "bake_ik":
        bake_3d_joints_to_animated_glb(args.blend_path, args.joints_json, args.out_glb, meta_path=args.meta_path)
        print(f"[Stage 6 IK Bake Complete] Exported {args.out_glb}")

    elif args.mode == "verify_exam_glb":
        verify_exam_glb_roundtrip(args.out_glb or args.input_glb, args.exam_gt_json, args.out_report)

    elif args.mode == "verify_delivery_glb":
        sop_verify_delivery_glb(args.glb_path or args.out_glb or args.input_glb, bool(args.is_loop), args.out_report)


if __name__ == "__main__":
    main()