"""Native sanitizer hierarchy and four-heading canonical rig-frame checks."""

from pathlib import Path
import json
import math
import sys

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / "packages/monster/worker"))
sys.path.append(
    str(
        repo
        / "packages/monster/.venv/lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
)
import bpy
import numpy as np
from scipy.spatial import cKDTree
from mathutils import Matrix
from reprojection_solver import aligned_rest_rotation, RigModel
from video2rig_blender import prepare_candidate, _evaluated_vertices_world
import studio_bpy_worker as worker

out = Path(sys.argv[sys.argv.index("--") + 1])
out.mkdir(parents=True, exist_ok=True)
results = []


def cube(name, center, size):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.name, obj.scale = name, size
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def check_rig(directory, center, scale):
    worker.clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(directory / "canonical.glb"))
    mesh = next(o for o in bpy.context.scene.objects if o.type == "MESH")
    arm = bpy.data.objects.new("Rig", bpy.data.armatures.new("Rig"))
    bpy.context.collection.objects.link(arm)
    arm.matrix_world = Matrix.Translation((0.21, -0.17, 0.08)) @ Matrix.Rotation(
        0.43, 4, "Z"
    )
    bpy.context.view_layer.update()
    inverse = arm.matrix_world.inverted()
    points = {
        "Root": (None, [0, 0, 0.6], [0, 0, 1.2], "root"),
        "Head": ("Root", [0, 0, 1.25], [0, -0.3, 1.95], "head"),
        "Arm.L": ("Root", [0.15, 0, 1.25], [0.77, 0, 1.25], "limb"),
        "Arm.R": ("Root", [-0.15, 0, 1.25], [-0.77, 0, 1.25], "limb"),
        "Rigid1.L": ("Arm.L", [0.77, 0, 1.25], [0.77, 0, 1.32], "rigid"),
        "Rigid1.R": ("Arm.R", [-0.77, 0, 1.25], [-0.77, 0, 1.32], "rigid"),
        "Foot.L": ("Root", [0.16, 0, 0.65], [0.16, -0.16, 0.03], "limb"),
        "Foot.R": ("Root", [-0.16, 0, 0.65], [-0.16, -0.16, 0.03], "limb"),
    }
    from mathutils import Vector

    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for i, (name, (parent, head, tail, kind)) in enumerate(points.items()):
        b = arm.data.edit_bones.new(name)
        b.head = inverse @ Vector((np.array(head) - center) * scale)
        b.tail = inverse @ Vector((np.array(tail) - center) * scale)
        b.roll = 0.17 * (i + 1)
    for name, (parent, *_rest) in points.items():
        if parent:
            arm.data.edit_bones[name].parent = arm.data.edit_bones[parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh.parent = arm
    mesh.matrix_parent_inverse = arm.matrix_world.inverted()
    mesh.matrix_world = Matrix.Identity(4)
    mesh.modifiers.new("Armature", "ARMATURE").object = arm
    for n in points:
        mesh.vertex_groups.new(name=n)
    for vertex in mesh.data.vertices:
        x, y, z = vertex.co
        name = (
            "Arm.L"
            if x > 0.3 * scale
            else "Arm.R"
            if x < -0.3 * scale
            else "Head"
            if z > 1.5 * scale
            else ("Foot.L" if x > 0 else "Foot.R")
            if z < 0.7 * scale
            else "Root"
        )
        mesh.vertex_groups[name].add([vertex.index], 1, "REPLACE")
    for name, angles in {
        "Root": [0.06, -0.04, 0.12],
        "Arm.L": [0.08, 0, -0.04],
        "Arm.R": [-0.05, 0, 0.07],
        "Head": [0.03, 0, -0.02],
    }.items():
        pb = arm.pose.bones[name]
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = angles
    bpy.context.view_layer.update()
    before = _evaluated_vertices_world(mesh).copy()
    raw = {}
    for name, (_parent, _head, _tail, kind) in points.items():
        b = arm.data.bones[name]
        m = np.array(arm.matrix_world @ b.matrix_local)
        raw[name] = dict(
            parent=b.parent.name if b.parent else None,
            kind=kind,
            fixed_local=kind == "rigid",
            rest_head_world=m[:3, 3].tolist(),
            rest_tail_world=list(arm.matrix_world @ b.tail_local),
            rest_rot_mat_world=m[:3, :3].tolist(),
            pose_basis=np.array(arm.pose.bones[name].matrix_basis).tolist(),
            rest_length=b.length,
        )
    prepared, report = prepare_candidate([mesh], arm, directory / "prepared", raw)
    assert set(prepared) == set(points), sorted(prepared)
    frame_error = orthogonality_error = 0.0
    for name, b in prepared.items():
        actual = np.array(
            (arm.matrix_world @ arm.data.bones[name].matrix_local).to_3x3()
        )
        expected = aligned_rest_rotation(
            np.array(b["rest_head_world"]), np.array(b["rest_tail_world"])
        )
        frame_error = max(frame_error, float(abs(actual - expected).max()))
        orthogonality_error = max(
            orthogonality_error, float(abs(actual.T @ actual - np.eye(3)).max())
        )
        # Bone matrices and align_roll use Blender float32 internally.
        np.testing.assert_allclose(actual, expected, atol=1e-5, err_msg=name)
        np.testing.assert_allclose(b["rest_rot_mat_world"], actual, atol=1e-5)
        np.testing.assert_allclose(
            actual.T @ actual, np.eye(3), atol=1e-5, err_msg=name
        )
        assert np.linalg.det(actual) > 0.99999
    for name in ("Arm.L", "Arm.R"):
        np.testing.assert_allclose(
            np.array(prepared[name]["rest_rot_mat_world"])[:, 0], [0, 0, 1], atol=1e-5
        )
    meta = worker._classify_and_canonicalize_candidate_rig(
        [mesh], arm, prepared_bones=prepared
    )
    skin = worker.extract_complete_rig_skinning([mesh], arm, sorted(meta["bones"]))
    rig = RigModel(meta["bones"], skin=skin)
    _, heads, rotations = rig.fk(rig.start_params())
    for name in rig.names:
        actual = np.array(arm.matrix_world @ arm.pose.bones[name].matrix)
        np.testing.assert_allclose(heads[name], actual[:3, 3], atol=3e-5, err_msg=name)
        np.testing.assert_allclose(
            rotations[name], actual[:3, :3], atol=3e-5, err_msg=name
        )
    error = float(
        np.linalg.norm(_evaluated_vertices_world(mesh) - before, axis=1).max()
    )
    assert error < 3e-5, error
    bpy.ops.wm.save_as_mainfile(filepath=str(directory / "axes.blend"))
    return dict(
        bones=len(prepared),
        start_surface_max_error=error,
        parallel_axis_fallback="+Z",
        nonzero_armature_transform=True,
        frame_max_error=frame_error,
        orthogonality_max_error=orthogonality_error,
    )


for yaw in (0, 90, 180, 270):
    directory = out / str(yaw)
    directory.mkdir(exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    meshes = [
        cube("Torso", (0, 0, 1), (0.5, 0.35, 1.2)),
        cube("Head", (0, -0.3, 1.75), (0.32, 0.38, 0.5)),
    ]
    for sign in (1, -1):
        meshes += [
            cube("Leg", (sign * 0.16, 0, 0.35), (0.14, 0.16, 0.7)),
            cube("Toe", (sign * 0.16, -0.15, 0.05), (0.2, 0.45, 0.1)),
            cube("Arm", (sign * 0.5, 0, 1.25), (0.6, 0.14, 0.14)),
        ]
    base = np.concatenate(
        [np.array([v.co for v in obj.data.vertices]) for obj in meshes]
    )
    center = (base.min(0) + base.max(0)) / 2
    center[2] = base[:, 2].min()
    scale = 1.25
    expected = (base - center) * scale
    parent = bpy.data.objects.new("WorldPlacement", None)
    child = bpy.data.objects.new("ModelGroup", None)
    bpy.context.collection.objects.link(parent)
    bpy.context.collection.objects.link(child)
    parent.location = (2.3, -1.1, 0.6)
    parent.rotation_euler = (0, 0, math.radians(yaw))
    parent.scale = (scale,) * 3
    child.parent = parent
    for obj in meshes:
        obj.parent = child
    bpy.context.view_layer.update()
    source = directory / "hierarchy.glb"
    bpy.ops.export_scene.gltf(filepath=str(source), export_format="GLB")
    report = worker.sanitize_input_mesh_for_skintokens(
        str(source), str(directory / "canonical.glb")
    )
    canonical_meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    actual = np.concatenate(
        [
            np.array([o.matrix_world @ v.co for v in o.data.vertices])
            for o in canonical_meshes
        ]
    )
    error = max(
        cKDTree(actual).query(expected)[0].max(),
        cKDTree(expected).query(actual)[0].max(),
    )
    assert error < 2e-5, (
        yaw,
        error,
        actual.min(0),
        actual.max(0),
        expected.min(0),
        expected.max(0),
    )
    results.append(
        dict(
            source_heading=yaw,
            geometry_max_error=float(error),
            sanitizer=report,
            rig=check_rig(directory, center, scale),
        )
    )

(out / "rig-axes.json").write_text(json.dumps(results, indent=2))
print("CANONICAL_RIG_AXES", json.dumps(results))
