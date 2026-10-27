"""Native coverage-policy fault injection plus real double-wall shell Heat/transfer."""

from pathlib import Path
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

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
from mathutils import Matrix
import video2rig_blender as bridge

out = Path(sys.argv[sys.argv.index("--") + 1])
out.mkdir(parents=True, exist_ok=True)


def cube(name, center, size):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.name, obj.scale = name, size
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def rig():
    arm = bpy.data.objects.new("Rig", bpy.data.armatures.new("Rig"))
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for n, h, t in [
        ("Root", (0, 0, 0.3), (0, 0, 1.05)),
        ("Head", (0, 0, 1.05), (0, 0, 1.8)),
        ("Prop_tool", (0.8, 0, 0.5), (0.8, 0, 1.5)),
    ]:
        b = arm.data.edit_bones.new(n)
        b.head, b.tail = h, t
    arm.data.edit_bones["Head"].parent = arm.data.edit_bones["Root"]
    bpy.ops.object.mode_set(mode="OBJECT")
    arm.matrix_world = Matrix.Translation((0.21, -0.16, 0.13)) @ Matrix.Rotation(
        0.31, 4, "Z"
    )
    arm.pose.bones["Root"].rotation_mode = "XYZ"
    arm.pose.bones["Root"].rotation_euler = (0.03, -0.02, 0.04)
    bpy.context.view_layer.update()
    return arm


def bind(obj, arm, name="Root"):
    obj.parent = arm
    obj.matrix_world = arm.matrix_world.copy()
    obj.modifiers.new("Armature", "ARMATURE").object = arm
    obj.vertex_groups.new(name=name).add(
        list(range(len(obj.data.vertices))), 1, "REPLACE"
    )


def bone_weights(obj, arm):
    groups = {g.index for g in obj.vertex_groups if g.name in arm.data.bones}
    return np.array(
        [
            sum(g.weight for g in v.groups if g.group in groups)
            for v in obj.data.vertices
        ]
    )


def policy_case(covered, *, mesh_error=False, shell_error=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    actor = cube("Actor", (0, 0, 1), (0.6, 0.5, 2))
    bpy.context.view_layer.objects.active = actor
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=3)
    bpy.ops.object.mode_set(mode="OBJECT")
    vertices = [list(v.co) for v in actor.data.vertices] + [[0, 0, 0.4], [0, 0, 1.5]]
    faces = [list(p.vertices) for p in actor.data.polygons]
    actor.data.clear_geometry()
    actor.data.from_pydata(vertices, [], faces)
    actor.data.update()
    assert len(vertices) == 100
    prop = cube("Prop_tool", (0.8, 0, 1), (0.1, 0.1, 0.7))
    arm = rig()
    bind(actor, arm)
    bind(prop, arm, "Prop_tool")
    actor.vertex_groups.new(name="DecorationMask").add(list(range(100)), 0.5, "REPLACE")
    world = actor.matrix_world.copy()
    basis = arm.pose.bones["Root"].matrix_basis.copy()
    original = bridge._parent_heat_excluding_props
    calls = []

    def partial_heat(_arm):
        calls.append(True)
        if len(calls) == 1:
            actor.vertex_groups["Root"].add(list(range(covered)), 1, "REPLACE")
            actor.modifiers.new("DuplicateArmature", "ARMATURE").object = arm
            actor.location.x += 3
            if mesh_error:
                raise RuntimeError("injected input Heat failure")
            return {"FINISHED"}
        if shell_error:
            raise RuntimeError("injected shell Heat failure")
        return original(_arm)

    try:
        with patch.object(
            bridge, "_parent_heat_excluding_props", side_effect=partial_heat
        ):
            report = bridge.heat_bind_updated_skeleton([actor, prop], arm)
    except RuntimeError as error:
        assert shell_error and "injected shell Heat failure" in str(error)
        report = dict(expected_failure=True)
    else:
        assert not shell_error
        assert report["mesh_weighted_fraction"] == covered / 100, report
        assert report["fallback"] == (covered < 50), report
        assert len(calls) == (2 if covered < 50 else 1), calls
        if covered < 50:
            assert np.all(bone_weights(actor, arm) > 0.99)
    np.testing.assert_allclose(actor.matrix_world, world, atol=1e-6)
    np.testing.assert_allclose(arm.pose.bones["Root"].matrix_basis, basis, atol=1e-7)
    assert arm.data.pose_position == "POSE"
    assert sum(m.type == "ARMATURE" and m.object == arm for m in actor.modifiers) == 1
    np.testing.assert_allclose(bone_weights(prop, arm), 1)
    assert arm.data.bones["Prop_tool"].use_deform
    assert not any(o.name.startswith("__v3_heat_shell") for o in bpy.data.objects)
    assert not any(m.name.startswith("__v3_heat_shell") for m in bpy.data.meshes)
    return dict(
        covered=covered,
        report=report,
        mesh_error=mesh_error,
        shell_error=shell_error,
        pose_and_world_restored=True,
        prop_preserved=True,
    )


policies = [policy_case(k) for k in (49, 50, 51)]
policies += [policy_case(0, mesh_error=True), policy_case(49, shell_error=True)]

# Directly observe the existing Prop exclusion wrapper on success and failure.
for fail in (False, True):
    arm = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
    observed = []

    def operation(**kwargs):
        observed.append(arm.data.bones["Prop_tool"].use_deform)
        assert kwargs["type"] == "ARMATURE_AUTO"
        if fail:
            raise RuntimeError("injected operator failure")
        return {"FINISHED"}

    with patch.object(
        bpy, "ops", SimpleNamespace(object=SimpleNamespace(parent_set=operation))
    ):
        try:
            bridge._parent_heat_excluding_props(arm)
        except RuntimeError:
            assert fail
    assert observed == [False]
    assert arm.data.bones["Prop_tool"].use_deform

# Real Heat on a new exterior, including UV seams, inner walls and a separate piece.
bpy.ops.wm.read_factory_settings(use_empty=True)
source = cube("Template", (0, 0, 1), (0.6, 0.5, 2))
corners = np.array([v.co for v in source.data.vertices])
polygons = [list(p.vertices) for p in source.data.polygons]
vertices, faces = [], []
for shrink in (1, 0.86):
    for poly in polygons:
        start = len(vertices)
        vertices.extend(
            (np.array([0, 0, 1]) + (corners[poly] - [0, 0, 1]) * shrink).tolist()
        )
        faces.append(list(range(start, start + 4))[:: 1 if shrink == 1 else -1])
bpy.data.objects.remove(source, do_unlink=True)
data = bpy.data.meshes.new("DoubleWall")
data.from_pydata(vertices, [], faces)
data.update()
actor = bpy.data.objects.new("Actor", data)
bpy.context.collection.objects.link(actor)
fragment = cube("DisconnectedPlate", (0.48, 0, 1), (0.04, 0.04, 0.1))
prop = cube("Prop_tool", (0.8, 0, 1), (0.1, 0.1, 0.7))
arm = rig()
bind(actor, arm)
bind(fragment, arm)
bind(prop, arm, "Prop_tool")
meshes = [actor, fragment, prop]
start = bridge._capture_start_pose(meshes, arm)
worlds = {o: np.array(o.matrix_world) for o in meshes}
arm.data.pose_position = "REST"
outer = bridge.solid_copy(meshes, arm)
repair = bridge.loose_weights(meshes, arm, 2)
arm.data.pose_position = "POSE"
restore = bridge._restore_start_after_rest_edit(meshes, arm, start)
for obj in meshes:
    np.testing.assert_allclose(obj.matrix_world, worlds[obj], atol=1e-6)
    np.testing.assert_allclose(bone_weights(obj, arm), 1, atol=1e-6)
    np.testing.assert_allclose(
        bridge._evaluated_vertices_world(obj), start["vertices"][obj.name], atol=3e-5
    )
    for vertex in obj.data.vertices:
        assert (
            sum(
                g.weight > 1e-6 and obj.vertex_groups[g.group].name in arm.data.bones
                for g in vertex.groups
            )
            <= 4
        )
    if obj != prop:
        indices = {g.index for g in obj.vertex_groups if g.name.startswith("Prop_")}
        assert not any(
            g.weight > 1e-6 and g.group in indices
            for v in obj.data.vertices
            for g in v.groups
        )
assert not any(o.name.startswith("__v3_heat_shell") for o in bpy.data.objects)
bpy.ops.wm.save_as_mainfile(filepath=str(out / "double-wall.blend"))
result = dict(
    passed=True,
    policies=policies,
    exterior=outer,
    repairs=repair,
    start_pose=restore,
    double_wall_vertices=len(vertices),
    input_disconnected_face_components=12,
    separate_fragment_vertices=len(fragment.data.vertices),
    real_shell_heat=True,
    prop_operator_flag_restored=True,
)
(out / "heat.json").write_text(json.dumps(result, indent=2))
print("REAL_HEAT_BINDING", json.dumps(result))
