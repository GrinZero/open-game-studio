"""Native sampling and GLB roundtrip with a posed, rolled organization chain."""

from pathlib import Path
import json
import math
import random
import sys
from unittest.mock import patch

repo = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo / "packages/monster/worker"))
sys.path.insert(0, str(repo / "packages/monster/tests"))
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
from scipy.spatial.transform import Rotation
from mathutils import Matrix
from test_v3_motion_roots import nested_rig
from reprojection_solver import RigModel
from video2rig_blender import _evaluated_vertices_world
from pose_sampling_v3 import posed_surface
import studio_bpy_worker as worker

out = Path(sys.argv[sys.argv.index("--") + 1])
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
meta, _ = nested_rig()
rename = {"Org": "Spine0", "Spine1": "UpperArm.L"}
meta = {
    rename.get(n, n): {**b, "parent": rename.get(b["parent"], b["parent"])}
    for n, b in meta.items()
}
meta["UpperArm.L"]["kind"] = "limb"
arm = bpy.data.objects.new("NestedRig", bpy.data.armatures.new("NestedRig"))
bpy.context.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
arm.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
for i, (name, bone) in enumerate(meta.items()):
    eb = arm.data.edit_bones.new(name)
    eb.head, eb.tail = bone["rest_head_world"], bone["rest_tail_world"]
    eb.roll = 0.13 * (i + 1)
    if bone["parent"]:
        eb.parent = arm.data.edit_bones[bone["parent"]]
bpy.ops.object.mode_set(mode="OBJECT")
arm.location = (0.14, -0.11, 0.23)
arm.rotation_euler = (0.1, -0.15, 0.2)
bpy.context.view_layer.update()
meshes = []
for name in ("Hips", "UpperArm.L", "Rigid1"):
    center = np.array(meta[name]["rest_head_world"]) + [0, 0, 0.12]
    bpy.ops.mesh.primitive_cube_add(size=0.15, location=center)
    obj = bpy.context.object
    obj.name = name + "Mesh"
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    obj.parent = arm
    obj.matrix_world = arm.matrix_world.copy()
    obj.vertex_groups.new(name=name).add(
        list(range(len(obj.data.vertices))), 1, "REPLACE"
    )
    obj.modifiers.new("Armature", "ARMATURE").object = arm
    meshes.append(obj)
for name, bone in meta.items():
    pb = arm.pose.bones[name]
    pb.matrix_basis = Matrix(bone["pose_basis"])
    rest = np.array(arm.matrix_world @ pb.bone.matrix_local)
    bone.update(
        rest_head_world=rest[:3, 3].tolist(),
        rest_tail_world=list(arm.matrix_world @ pb.bone.tail_local),
        rest_rot_mat_world=rest[:3, :3].tolist(),
        anatomical_role=name,
    )
bpy.context.view_layer.update()
skin = worker.extract_complete_rig_skinning(meshes, arm, sorted(meta))
rig = RigModel(meta, skin=skin)
assert rig.motion_roots == {"Hips"} and rig.size == 9
p0 = rig.start_params()
max_error = 0.0


def verify_pose(params, objects, actual_arm, tolerance=3e-5):
    global max_error
    _, heads, rotations = rig.fk(params)
    for name in rig.names:
        actual = np.array(actual_arm.matrix_world @ actual_arm.pose.bones[name].matrix)
        error = max(
            np.max(abs(actual[:3, 3] - heads[name])),
            np.max(abs(actual[:3, :3] - rotations[name])),
        )
        max_error = max(max_error, float(error))
        assert error < tolerance, (name, error, actual, heads[name], rotations[name])
    actual = np.concatenate([_evaluated_vertices_world(o) for o in objects])
    # glTF may reorder vertices, so compare the nearest vertices in both directions.
    from scipy.spatial import cKDTree

    expected = posed_surface(rig, params, skin)
    error = max(
        cKDTree(actual).query(expected)[0].max(),
        cKDTree(expected).query(actual)[0].max(),
    )
    max_error = max(max_error, float(error))
    assert error < tolerance, (
        error,
        actual.min(0),
        actual.max(0),
        expected.min(0),
        expected.max(0),
        [
            (
                o.name,
                list(o.matrix_world.translation),
                [
                    (
                        m.type,
                        getattr(m, "object", None).name
                        if getattr(m, "object", None)
                        else None,
                    )
                    for m in o.modifiers
                ],
            )
            for o in objects
        ],
    )


verify_pose(p0, meshes, arm)
bpy.ops.wm.save_as_mainfile(filepath=str(out / "nested.blend"))
report = {}
seed_eulers = {n: [0.12, 0.07, -0.09] for n in meta}
boundary_report = {}
with patch.object(
    worker, "sample_stratified_pose_euler_dict", return_value=(seed_eulers, "fixture")
):
    worker.sample_in_limit_pose(
        arm,
        meta,
        random.Random(5),
        sampling_skin=skin,
        boundary_exploration=True,
        sampling_report=boundary_report,
    )
boundary_params = rig.zero()
boundary_root = rig.slices["Hips"]
boundary_params[boundary_root.start : boundary_root.start + 3] = Rotation.from_euler(
    "xyz", boundary_report["root_world_euler"]
).as_rotvec()
boundary_params[boundary_root.start + 3 : boundary_root.stop] = boundary_report[
    "translation"
]
ix, channels = rig.slices["UpperArm.L"]
boundary_params[ix] = np.asarray(seed_eulers["UpperArm.L"])[channels]
verify_pose(boundary_params, meshes, arm)
for name in ("Root", "Spine0", "Rigid1"):
    np.testing.assert_allclose(
        arm.pose.bones[name].matrix_basis, meta[name]["pose_basis"], atol=5e-6
    )
with patch.object(
    worker, "sample_stratified_pose_euler_dict", return_value=(seed_eulers, "fixture")
):
    sampled = worker.sample_in_limit_pose(
        arm,
        meta,
        random.Random(5),
        sampling_skin=skin,
        sampling_events={"yaw": True, "pitch": True, "jump": True},
        sampling_report=report,
    )
p1 = rig.zero()
root = rig.slices["Hips"]
p1[root.start : root.start + 3] = Rotation.from_euler(
    "xyz", report["root_world_euler"]
).as_rotvec()
p1[root.start + 3 : root.stop] = report["translation"]
indices, channels = rig.slices["UpperArm.L"]
p1[indices] = np.array(seed_eulers["UpperArm.L"])[channels]
verify_pose(p1, meshes, arm)
for name in ("Root", "Spine0", "Rigid1"):
    np.testing.assert_allclose(
        arm.pose.bones[name].matrix_basis, meta[name]["pose_basis"], atol=5e-6
    )
assert sampled.keys() == meta.keys()
shutter_deltas = {}
for mode in ("normal", "strike", "lunge"):
    arm.animation_data_clear()
    arm.animation_data_create()
    worker.keyframe_pose_shutter(
        arm,
        meta,
        sampled,
        random.Random(7),
        reader_v3=True,
        events={mode: True},
        report=report,
    )
    bpy.context.view_layer.update()
    verify_pose(p1, meshes, arm)
    ends = []
    for frame in (0, 2):
        bpy.context.scene.frame_set(frame)
        for name in ("Root", "Spine0", "Rigid1"):
            np.testing.assert_allclose(
                arm.pose.bones[name].matrix_basis, meta[name]["pose_basis"], atol=5e-6
            )
        ends.append(
            {
                name: np.array(arm.pose.bones[name].rotation_euler)
                for name in ("Hips", "UpperArm.L")
            }
        )
    shutter_deltas[mode] = {name: ends[1][name] - ends[0][name] for name in ends[0]}
np.testing.assert_allclose(
    shutter_deltas["strike"]["UpperArm.L"],
    2.5 * shutter_deltas["normal"]["UpperArm.L"],
    atol=1e-6,
)
np.testing.assert_allclose(
    shutter_deltas["lunge"]["Hips"], 2.5 * shutter_deltas["normal"]["Hips"], atol=1e-6
)
assert np.linalg.norm(shutter_deltas["normal"]["UpperArm.L"]) > 0.01
sequence = [p0, p1]
frames = rig.frames(sequence, np.ones((2, 2, len(rig.point_names))))
frames[1]["frame"] = 2.5
trajectory = out / "trajectory.json"
trajectory.write_text(
    json.dumps(
        dict(
            reader_contract="monster-reader-v3", fps=24, frames=frames, bones_meta=meta
        )
    )
)
(out / "meta.json").write_text(json.dumps({"bones": meta}))
profile = {
    "schema": "monster-joint-limits-v1",
    "bones": {
        name: {"limits_rad": bone["limits_rad"], "fixed_local": bone["fixed_local"]}
        for name, bone in meta.items()
    },
}
(out / "explicit-limits.json").write_text(json.dumps(profile))
worker.bake_3d_joints_to_animated_glb(
    str(out / "nested.blend"),
    str(trajectory),
    str(out / "nested.glb"),
    meta_path=str(out / "meta.json"),
    limits_path=str(out / "explicit-limits.json"),
)
for frame, params in zip((0, 2.5), sequence):
    bpy.context.scene.frame_set(math.floor(frame), subframe=frame % 1)
    verify_pose(
        params,
        [o for o in bpy.context.scene.objects if o.type == "MESH"],
        next(o for o in bpy.context.scene.objects if o.type == "ARMATURE"),
    )
objects, imported_arm = worker.load_glb_and_find_rig(str(out / "nested.glb"), fps=24)
objects = [
    o
    for o in bpy.context.scene.objects
    if o.type == "MESH"
    and any(m.type == "ARMATURE" and m.object == imported_arm for m in o.modifiers)
]
for frame, params in zip((0, 2.5), sequence):
    bpy.context.scene.frame_set(math.floor(frame), subframe=frame % 1)
    verify_pose(params, objects, imported_arm, tolerance=6e-5)
evidence = dict(
    passed=True,
    motion_roots=sorted(rig.motion_roots),
    degrees_of_freedom=rig.size,
    bones=len(meta),
    meshes=len(meshes),
    max_error=max_error,
    sample_report=report,
    boundary_sample_report=boundary_report,
    glb_frames=[0, 2.5],
    shutter_frozen_bases_preserved=True,
    strike_and_lunge_speed_ratio=2.5,
)
(out / "motion-roots.json").write_text(json.dumps(evidence, indent=2))
print("NESTED_ROOT_ROUNDTRIP", json.dumps(evidence))
