"""Blender boundary for v3 geometry, skin binding, camera export and strip rendering."""

import json
from pathlib import Path
import numpy as np
from skeleton_postprocess import (
    voxelize,
    postprocess_skintokens_skeleton,
    manylegs_build,
    find_facing,
)
from reprojection_solver import aligned_rest_rotation, find_hinges, surface_keypoints
from studio_camera import Camera, export_cameras


def mesh_arrays(meshes):
    vertices = []
    triangles = []
    offset = 0
    for obj in meshes:
        v = np.array([list(obj.matrix_world @ p.co) for p in obj.data.vertices])
        obj.data.calc_loop_triangles()
        vertices.extend(v)
        triangles.extend(
            [[offset + int(i) for i in tri.vertices] for tri in obj.data.loop_triangles]
        )
        offset += len(v)
    return np.asarray(vertices, float), np.asarray(triangles, int)


def export_shell(meshes, path):
    v, f = mesh_arrays(meshes)
    height = max(np.ptp(v[:, 2]), 1e-5)
    shell = voxelize(v, f, 0.012 * height)
    shell.save(path, v, f)
    return shell, v, f


def loose_weights(meshes, arm, height):
    """Repair loose fragments and cross-limb leakage, retaining at most four normalized influences."""
    from scipy.spatial import cKDTree

    bones = list(arm.data.bones)
    names = [b.name for b in bones]
    h = np.array([list(arm.matrix_world @ b.head_local) for b in bones])
    t = np.array([list(arm.matrix_world @ b.tail_local) for b in bones])
    d = t - h

    def distances(point):
        alpha = np.clip(
            np.einsum("ij,ij->i", point - h, d)
            / np.maximum(np.sum(d * d, axis=1), 1e-10),
            0,
            1,
        )
        return np.linalg.norm(point - h - alpha[:, None] * d, axis=-1)

    repaired = 0
    rigid = 0
    total_character_vertices = sum(len(obj.data.vertices) for obj in meshes if not obj.name.startswith("Prop_"))
    for obj in meshes:
        groups = {g.index: g.name for g in obj.vertex_groups}
        verts = list(obj.data.vertices)
        # Connected components distinguish a small independent fragment from the main body.
        parent = np.arange(len(verts))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for edge in obj.data.edges:
            a, b = map(int, edge.vertices)
            parent[root(a)] = root(b)
        ids = np.array([root(i) for i in range(len(verts))])
        counts = np.bincount(ids, minlength=len(verts))
        for vertex in verts:
            point = np.array(obj.matrix_world @ vertex.co)
            dist = distances(point)
            prop_owner = (
                names.index(obj.name)
                if obj.name.startswith("Prop_") and obj.name in names
                else None
            )
            if prop_owner is None:
                dist[[i for i, n in enumerate(names) if n.startswith("Prop_")]] = np.inf
            nearest = int(dist.argmin())
            pairs = [
                (names.index(groups[g.group]), g.weight)
                for g in vertex.groups
                if groups.get(g.group) in names and g.weight > 1e-6
            ]
            small = counts[ids[vertex.index]] < 0.02 * max(1, total_character_vertices)
            weapon = [
                i
                for i, n in enumerate(names)
                if n.startswith("Weapon") and dist[i] < 0.025 * height
            ]
            if prop_owner is not None:
                pairs = [(prop_owner, 1.0)]
            elif weapon:
                i = min(weapon, key=lambda k: dist[k])
                pairs = [(i, 1.0)]
                rigid += 1
            elif not pairs or small:
                chosen = np.argsort(dist)[:4]
                chosen = chosen[np.isfinite(dist[chosen])]
                w = 1 / np.maximum(dist[chosen], 0.002 * height) ** 4
                pairs = list(zip(chosen, w / w.sum()))
                repaired += 1
            else:
                # A distant opposite limb cannot own a vertex across the body.
                pairs = [
                    (i, w)
                    for i, w in pairs
                    if dist[i] <= max(0.12 * height, 3 * dist[nearest])
                ]
                if not pairs:
                    pairs = [(nearest, 1.0)]
                    repaired += 1
            pairs = sorted(pairs, key=lambda item: item[1], reverse=True)[:4]
            total = sum(w for _, w in pairs)
            for group in obj.vertex_groups:
                if group.name in names:
                    group.remove([vertex.index])
            for i, w in pairs:
                vg = obj.vertex_groups.get(names[i]) or obj.vertex_groups.new(
                    name=names[i]
                )
                vg.add([vertex.index], float(w / total), "REPLACE")
    return {"repaired_vertices": repaired, "rigid_weapon_vertices": rigid}


def _parent_heat_excluding_props(arm):
    """Independent rigid roots must never receive weights from the character shell."""
    import bpy

    properties = {
        bone: bone.use_deform
        for bone in arm.data.bones
        if bone.name.startswith("Prop_")
    }
    try:
        for bone in properties:
            bone.use_deform = False
        return bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    finally:
        for bone, enabled in properties.items():
            bone.use_deform = enabled


def solid_copy(meshes, arm):
    """100-cell outer shell Bone Heat, followed by nearest-surface weight transfer."""
    import bpy
    from skimage.measure import marching_cubes
    from scipy.spatial import cKDTree

    meshes = [obj for obj in meshes if not obj.name.startswith("Prop_")]
    if not meshes:
        raise ValueError("Outer shell Heat requires a character mesh")
    v, f = mesh_arrays(meshes)
    step = max(np.ptp(v, axis=0)) / 100
    shell = voxelize(v, f, step)
    verts, faces, _, _ = marching_cubes(shell.solid.astype(float), level=0.5)
    verts = shell.origin + verts * step
    data = bpy.data.meshes.new("__v3_heat_shell")
    data.from_pydata(verts.tolist(), [], faces.tolist())
    data.update()
    obj = bpy.data.objects.new("__v3_heat_shell", data)
    bpy.context.collection.objects.link(obj)
    try:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm
        _parent_heat_excluding_props(arm)
        group_names = {g.index: g.name for g in obj.vertex_groups}
        tree = cKDTree(verts)
        if not any(p.groups for p in obj.data.vertices):
            raise RuntimeError("Outer-shell Bone Heat produced no weights")
        for mesh in meshes:
            positions = np.array(
                [list(mesh.matrix_world @ p.co) for p in mesh.data.vertices]
            )
            indices = tree.query(positions)[1]
            for vertex, near in zip(mesh.data.vertices, indices):
                for group in list(mesh.vertex_groups):
                    if arm.data.bones.get(group.name):
                        group.remove([vertex.index])
                for group in obj.data.vertices[int(near)].groups:
                    name = group_names[group.group]
                    vg = mesh.vertex_groups.get(name) or mesh.vertex_groups.new(
                        name=name
                    )
                    vg.add([vertex.index], float(group.weight), "REPLACE")
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.meshes.remove(data)
    return {"method": "100_cell_outer_shell_heat", "shell_vertices": len(verts)}


def heat_bind_updated_skeleton(meshes, arm):
    """Try the actual input mesh first; use the outer shell below 50% coverage."""
    import bpy

    meshes = [obj for obj in meshes if not obj.name.startswith("Prop_")]
    if not meshes:
        raise ValueError("Mesh Heat requires a character mesh")

    position = arm.data.pose_position
    worlds = {obj: obj.matrix_world.copy() for obj in meshes}
    report = {"method": "mesh_heat", "fallback": False}
    try:
        arm.data.pose_position = "REST"
        bpy.context.view_layer.update()
        for obj in meshes:
            for group in obj.vertex_groups:
                if arm.data.bones.get(group.name):
                    group.remove(list(range(len(obj.data.vertices))))
        bpy.ops.object.select_all(action="DESELECT")
        for obj in meshes:
            obj.select_set(True)
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm
        try:
            _parent_heat_excluding_props(arm)
        except RuntimeError as exc:
            report["mesh_heat_error"] = str(exc)
        total = weighted = 0
        for obj in meshes:
            obj.matrix_world = worlds[obj]
            deform = {g.index for g in obj.vertex_groups if arm.data.bones.get(g.name)}
            total += len(obj.data.vertices)
            weighted += sum(
                any(g.group in deform and g.weight > 1e-6 for g in vertex.groups)
                for vertex in obj.data.vertices
            )
            modifiers = [
                m for m in obj.modifiers if m.type == "ARMATURE" and m.object == arm
            ]
            for modifier in modifiers[1:]:
                obj.modifiers.remove(modifier)
        report["mesh_weighted_fraction"] = weighted / max(total, 1)
        if report["mesh_weighted_fraction"] < 0.5:
            report.update(fallback=True, outer_shell=solid_copy(meshes, arm))
    finally:
        arm.data.pose_position = position
        for obj in meshes:
            obj.matrix_world = worlds[obj]
        bpy.context.view_layer.update()
    return report


def _evaluated_vertices_world(obj):
    import bpy

    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        return np.array([list(evaluated.matrix_world @ v.co) for v in mesh.vertices])
    finally:
        evaluated.to_mesh_clear()


def _capture_start_pose(meshes, arm):
    import bpy

    for obj in meshes:
        for modifier in obj.modifiers:
            if modifier.type == "ARMATURE" and modifier.object == arm:
                modifier.use_deform_preserve_volume = False
    bpy.context.view_layer.update()
    return {
        "rest": {
            b.name: np.array(arm.matrix_world @ b.matrix_local) for b in arm.data.bones
        },
        "posed": {
            b.name: np.array(arm.matrix_world @ b.matrix) for b in arm.pose.bones
        },
        "basis": {b.name: np.array(b.matrix_basis) for b in arm.pose.bones},
        "vertices": {obj.name: _evaluated_vertices_world(obj) for obj in meshes},
    }


def _restore_start_after_rest_edit(meshes, arm, start):
    """Transport p0 across axis/geometry edits and rebind the mesh by inverse LBS.

    Copying old matrix_basis onto newly aligned axes changes the starting gesture.
    We preserve each source bone's world deformation, recompute its parent-relative
    basis on the new bind skeleton, and solve the new rest coordinates for the
    measured frame-zero surface after the new weights have been finalized.
    """
    import bpy
    from mathutils import Matrix
    from skeleton_postprocess import rest_coords

    names = [b.name for b in arm.data.bones]
    old = start["rest"]
    rest = {b.name: np.array(arm.matrix_world @ b.matrix_local) for b in arm.data.bones}
    order = sorted(arm.pose.bones, key=lambda b: len(b.parent_recursive))
    arm_inverse = arm.matrix_world.inverted()
    for pb in arm.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    for pb in order:
        name = pb.name
        if name not in old:
            continue
        deformation = start["posed"][name] @ np.linalg.inv(old[name])
        target = deformation @ rest[name]
        if pb.parent is not None:
            parent = pb.parent.name
            local_shift = np.eye(4)
            local_shift[:3, 3] = np.linalg.solve(
                rest[name][:3, :3], old[name][:3, :3] @ start["basis"][name][:3, 3]
            )
            head = (
                np.array(arm.matrix_world @ pb.parent.matrix)
                @ np.linalg.solve(rest[parent], rest[name])
                @ local_shift
            )
            target[:3, 3] = head[:3, 3]
        pb.matrix = arm_inverse @ Matrix(target.tolist())
        bpy.context.view_layer.update()
    posed = np.array(
        [np.array(arm.matrix_world @ arm.pose.bones[n].matrix) for n in names]
    )
    bind = np.array([rest[n] for n in names])
    index = {n: i for i, n in enumerate(names)}
    for obj in meshes:
        target = start["vertices"][obj.name]
        if len(target) != len(obj.data.vertices):
            raise ValueError("Start-pose rebind requires unchanged vertex topology")
        group_names = {g.index: g.name for g in obj.vertex_groups}
        ids = np.zeros((len(target), 4), dtype=int)
        weights = np.zeros((len(target), 4))
        for i, vertex in enumerate(obj.data.vertices):
            influences = sorted(
                [
                    (index[group_names[g.group]], float(g.weight))
                    for g in vertex.groups
                    if group_names.get(g.group) in index and g.weight > 1e-8
                ],
                key=lambda item: item[1],
                reverse=True,
            )
            if not influences or len(influences) > 4:
                raise ValueError(
                    "Start-pose rebind requires normalized top-four skinning weights"
                )
            total = sum(w for _, w in influences)
            for slot, (bone, weight) in enumerate(influences):
                ids[i, slot] = bone
                weights[i, slot] = weight / total
        recovered = rest_coords(target, ids, weights, bind, posed)
        inverse = np.array(obj.matrix_world.inverted())
        local = (inverse @ np.c_[recovered, np.ones(len(recovered))].T).T[:, :3]
        for vertex, point in zip(obj.data.vertices, local):
            vertex.co = point
        obj.data.update()
    bpy.context.view_layer.update()
    error = max(
        (
            np.linalg.norm(
                _evaluated_vertices_world(obj) - start["vertices"][obj.name], axis=1
            ).max(initial=0)
            for obj in meshes
        ),
        default=0.0,
    )
    if error > 1e-4:
        raise ValueError(
            f"Start-pose surface changed during rebind: {error:g} world units"
        )
    return {
        "surface_max_error": float(error),
        "method": "world_deformation_transport_and_inverse_lbs",
    }


def prepare_candidate(meshes, arm, out_dir, bones, fold_digits=False):
    import bpy
    from mathutils import Vector
    from bone_taxonomy import classify_rig_bones

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _, vertices, faces = export_shell(meshes, out / "shell.npz")
    # Geometry decisions use the specified 1% H grid. The serialized dual shell
    # independently retains its 1.2% H contract for downstream skinning/export.
    shell = voxelize(vertices, faces, 0.01 * max(np.ptp(vertices[:, 2]), 1e-5))
    start = _capture_start_pose(meshes, arm)
    processed, report = postprocess_skintokens_skeleton(
        bones, vertices, shell, fold_digits
    )
    roots = [n for n in processed if processed[n].get("parent") not in processed]
    root = roots[0]
    many = manylegs_build(shell, np.array(processed[root]["rest_head_world"]))
    if many:
        old = [n for n, b in processed.items() if b.get("kind") == "limb"]
        # Keep digits, attaching to the nearest new distal chain rather than discarding them.
        for n in old:
            report["removed"][n] = root
            processed.pop(n)
        trunk = [
            n for n, b in processed.items()
            if b.get("kind") in ("root", "spine", "neck", "head", "tail")
        ]

        def nearest_trunk(point):
            def distance(name):
                head = np.asarray(processed[name]["rest_head_world"])
                direction = np.asarray(processed[name]["rest_tail_world"]) - head
                fraction = np.clip(
                    np.dot(point - head, direction) / max(np.dot(direction, direction), 1e-10), 0, 1
                )
                return np.linalg.norm(point - head - fraction * direction)

            return min(trunk or [root], key=distance)

        for n, b in many.items():
            if b["parent"] is None:
                b["parent"] = nearest_trunk(np.asarray(b["rest_head_world"]))
            processed[n] = b
        for n, b in processed.items():
            if b.get("parent") in old:
                point = np.array(b["rest_head_world"])
                b["parent"] = min(
                    many,
                    key=lambda a: np.linalg.norm(
                        np.asarray(many[a]["rest_tail_world"]) - point
                    ),
                )
        report["manylegs"] = len(many) // 4
    # Merge deleted groups before deleting their deform bones.
    for name, parent in report["removed"].items():
        while parent in report["removed"]:
            parent = report["removed"][parent]
        if parent not in processed:
            parent = root
        for obj in meshes:
            source = obj.vertex_groups.get(name)
            if source is None:
                continue
            weights = [
                (v.index, g.weight)
                for v in obj.data.vertices
                for g in v.groups
                if g.group == source.index
            ]
            target = obj.vertex_groups.get(parent) or obj.vertex_groups.new(name=parent)
            for i, w in weights:
                target.add([i], float(w), "ADD")
            obj.vertex_groups.remove(source)
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    inv = arm.matrix_world.inverted()
    rot = arm.matrix_world.to_3x3().inverted()
    for bone in list(arm.data.edit_bones):
        if bone.name not in processed:
            arm.data.edit_bones.remove(bone)
    for n, b in processed.items():
        bone = arm.data.edit_bones.get(n) or arm.data.edit_bones.new(n)
        bone.use_connect = False
        bone.head = inv @ Vector(b["rest_head_world"])
        bone.tail = inv @ Vector(b["rest_tail_world"])
        if bone.length < 1e-6:
            bone.tail = bone.head + Vector((0, 1e-5, 0))
        R = aligned_rest_rotation(
            np.asarray(b["rest_head_world"]), np.asarray(b["rest_tail_world"])
        )
        bone.align_roll(rot @ Vector(R[:, 2]))
    for n, b in processed.items():
        arm.data.edit_bones[n].parent = (
            arm.data.edit_bones.get(b.get("parent", "")) if b.get("parent") else None
        )
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.update()
    for n, b in processed.items():
        bone = arm.data.bones[n]
        b["rest_rot_mat_world"] = np.asarray(
            (arm.matrix_world @ bone.matrix_local).to_3x3()
        ).tolist()
        b["pose_basis"] = np.array(arm.pose.bones[n].matrix_basis).tolist()
    processed = classify_rig_bones(
        processed, vertices.min(0).tolist(), vertices.max(0).tolist()
    )
    valid = total = 0
    for obj in meshes:
        names = {g.index: g.name for g in obj.vertex_groups}
        for vertex in obj.data.vertices:
            total += 1
            valid += any(
                names.get(g.group) in processed and g.weight > 1e-6
                for g in vertex.groups
            )
    added_deform = set(processed) - set(bones)
    if valid / max(1, total) < 0.5 or added_deform:
        report["binding"] = heat_bind_updated_skeleton(meshes, arm)
        if report["binding"]["fallback"]:
            report["binding_fallback"] = report["binding"]["outer_shell"]
    else:
        report["binding"] = {
            "method": "preserved_input_weights",
            "weighted_fraction": valid / max(1, total),
        }
    report["weights"] = loose_weights(meshes, arm, max(np.ptp(vertices[:, 2]), 1e-5))
    report["start_pose_rebind"] = _restore_start_after_rest_edit(meshes, arm, start)
    for name, info in processed.items():
        info["pose_basis"] = np.array(arm.pose.bones[name].matrix_basis).tolist()
        info["matrix_local"] = np.array(arm.data.bones[name].matrix_local).tolist()
    report["hinges"] = find_hinges(processed)
    (out / "skeleton_postprocess.json").write_text(json.dumps(report, indent=2))
    return processed, report


def attach_reader_points(record, arm, names, extra, cameras, visibility, calib):
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


from strip_renderer import StripRenderer  # compatibility import for existing callers


def render_v3_review(blend_path, meta_path, trajectory_path, video_path, out_dir):
    """Actual GLB bake/readback and calibrated original/3/4/silhouette review frames."""
    import bpy
    import cv2
    import math
    import subprocess
    from PIL import Image
    from mathutils import Matrix, Vector
    from studio_camera import cameras_from_calibration
    from studio_bpy_worker import bake_3d_joints_to_animated_glb, load_glb_and_find_rig

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(trajectory_path).read_text())
    from video_timing import VideoTiming

    timing = VideoTiming.from_trajectory(data)
    calib = data["calibration"]
    if not data.get("frames") or not data.get("gate_passed"):
        raise ValueError(
            "G3 requires a nonempty measured trajectory that passed synchronization"
        )
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
    directories = {
        n: out / n for n in ("sync_frames", "quarter_frames", "silhouette_frames")
    }
    for folder in directories.values():
        folder.mkdir(exist_ok=True)
    reports = []
    i = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if i >= len(data["cadence"]["frame_times"]):
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
            rendered = np.concatenate(
                [np.asarray(p) for p in panels], axis=0 if stacked else 1
            )
            mask = np.concatenate(masks, axis=0 if stacked else 1)
            if source.shape != rendered.shape:
                raise ValueError(
                    "Readback render dimensions differ from source video calibration"
                )
            compare = Image.fromarray(np.concatenate([source, rendered], axis=1))
            compare.save(directories["sync_frames"] / f"sync_frame_{i:04d}.png")
            border = np.concatenate(
                [source[:5].reshape(-1, 3), source[-5:].reshape(-1, 3)]
            )
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
            scene.render.filepath = str(
                directories["quarter_frames"] / f"quarter_{i:04d}.png"
            )
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


def ensure_prop_bones(meshes, arm):
    """Explicit Prop_* objects remain independent rigid bodies with their own root channels."""
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
