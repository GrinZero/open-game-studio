"""Real Blender regression for a posed, rolled, split-mesh source. No GPU is required."""

from pathlib import Path
import json
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
from mathutils import Euler
import studio_bpy_worker as worker
from video2rig_blender import prepare_candidate, _evaluated_vertices_world
from rig_pose import start_pose_vertices
from reprojection_solver import RigModel

out = Path(sys.argv[sys.argv.index("--") + 1])
out.mkdir(exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 1))
body = bpy.context.object
body.name = "Body"
body.scale = (0.5, 0.4, 2)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0.6, 0, 1.1))
part = bpy.context.object
part.name = "SeparatePart"
part.scale = (0.08, 0.06, 0.2)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
arm = bpy.data.objects.new("PosedRig", bpy.data.armatures.new("PosedRig"))
bpy.context.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
arm.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
root = arm.data.edit_bones.new("Hips")
root.head = (0, 0, 0.5)
root.tail = (0, 0, 1.2)
root.roll = -0.35
empty = arm.data.edit_bones.new("Root")
empty.head = (0, 0, 0.1)
empty.tail = (0, 0, 0.2)
root.parent = empty
head = arm.data.edit_bones.new("Head")
head.head = (0, 0, 1.2)
head.tail = (0, 0, 1.8)
head.parent = root
head.roll = 0.6
upper = arm.data.edit_bones.new("UpperArm.L")
upper.head = (0.1, 0, 1.2)
upper.tail = (0.15, 0, 1.05)
upper.parent = root
upper.roll = -0.4
lower = arm.data.edit_bones.new("LowerArm.L")
lower.head = (0.15, 0, 1.05)
lower.tail = (0.2, -0.08, 0.95)
lower.parent = upper
lower.roll = 0.7
bpy.ops.object.mode_set(mode="OBJECT")
for obj in (body, part):
    for name in ("Hips", "Head", "UpperArm.L", "LowerArm.L"):
        obj.vertex_groups.new(name=name)
    for vertex in obj.data.vertices:
        if obj == part:
            obj.vertex_groups["LowerArm.L"].add([vertex.index], 0.8, "REPLACE")
            obj.vertex_groups["Hips"].add([vertex.index], 0.2, "REPLACE")
        else:
            obj.vertex_groups["Head" if vertex.co.z > 1.2 else "Hips"].add(
                [vertex.index], 1.0, "REPLACE"
            )
    modifier = obj.modifiers.new("Armature", "ARMATURE")
    modifier.object = arm
arm.pose.bones["Hips"].rotation_mode = "XYZ"
arm.pose.bones["Hips"].rotation_euler = Euler((0.2, 0.1, -0.18))
arm.pose.bones["Hips"].location = (0.04, -0.03, 0.05)
arm.pose.bones["Head"].rotation_mode = "XYZ"
arm.pose.bones["Head"].rotation_euler = Euler((0.3, -0.1, 0.1))
arm.pose.bones["UpperArm.L"].rotation_mode = "XYZ"
arm.pose.bones["UpperArm.L"].rotation_euler = Euler((-0.1, 0.2, 0.1))
arm.pose.bones["LowerArm.L"].rotation_mode = "XYZ"
arm.pose.bones["LowerArm.L"].rotation_euler = Euler((0.45, 0.1, -0.1))
bpy.context.view_layer.update()
meshes = [body, part]
expected = {obj.name: _evaluated_vertices_world(obj) for obj in meshes}
_, raw = worker.classify_anonymous_bones_and_assign_limits(
    body, arm, prune_digits=False, canonicalize_names=False
)
prepared, report = prepare_candidate(meshes, arm, out / "prepared", raw["bones"])
hinge = report["hinges"]["LowerArm.L"]
# The hinge must observe the restored p0, not the intermediate edited-rest armature.
child = arm.pose.bones["LowerArm.L"]
parent = child.parent
a = np.array(arm.matrix_world.to_3x3() @ (child.tail - child.head), dtype=float)
b = np.array(arm.matrix_world.to_3x3() @ (parent.tail - parent.head), dtype=float)
actual_bend = np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))
np.testing.assert_allclose(hinge["rest_bend"], actual_bend, atol=1e-6)
for obj in meshes:
    np.testing.assert_allclose(
        _evaluated_vertices_world(obj), expected[obj.name], atol=3e-6
    )
assert report["start_pose_rebind"]["surface_max_error"] < 3e-6
meta = worker._classify_and_canonicalize_candidate_rig(
    meshes, arm, prepared_bones=prepared
)
meta["reader_contract"] = "monster-reader-v3"
assert meta["bones"]["LowerArm.L"]["hinge"] == hinge
assert meta["bones"]["LowerArm.L"]["axis_roles"] == prepared["LowerArm.L"]["axis_roles"]
# Exporting all parts must be a read-only operation on the saved p0.
before = {pb.name: np.array(pb.matrix_basis) for pb in arm.pose.bones}
names = sorted(meta["bones"])
skin = worker.extract_complete_rig_skinning(meshes, arm, names)
for pb in arm.pose.bones:
    np.testing.assert_allclose(pb.matrix_basis, before[pb.name], atol=2e-7)
np.testing.assert_allclose(
    skin["start_verts"], np.concatenate(list(expected.values())), atol=3e-6
)
assert np.max(np.abs(skin["start_verts"] - skin["rest_verts"])) > 0.05
# The pure CPU contract reproduces the same start mesh without stored frame-zero vertices.
reconstructed = start_pose_vertices(
    {k: v for k, v in skin.items() if k != "start_verts"}, meta["bones"]
)
np.testing.assert_allclose(reconstructed, skin["start_verts"], atol=2e-5)
rig = RigModel(meta["bones"], names, skin)
_, heads, rotations = rig.fk(rig.start_params())
for name in names:
    pb = arm.pose.bones[name]
    actual = np.array(arm.matrix_world @ pb.matrix)
    np.testing.assert_allclose(heads[name], actual[:3, 3], atol=2e-5)
    np.testing.assert_allclose(rotations[name], actual[:3, :3], atol=2e-5)
profile = worker.probe_rig_joint_limits(meta["bones"], rig_skinning=skin)
profile_hinge = profile["bones"]["LowerArm.L"]
axis = hinge["axis"]
assert profile_hinge["hinge"] == hinge
assert profile_hinge["axis_roles"] == prepared["LowerArm.L"]["axis_roles"]
merged = worker.apply_joint_limits_profile(meta["bones"], profile)
merged_rig = RigModel(merged, names, skin)
_, merged_heads, merged_rotations = merged_rig.fk(merged_rig.start_params())
for name in names:
    matrix = np.array(arm.matrix_world @ arm.pose.bones[name].matrix)
    np.testing.assert_allclose(merged_heads[name], matrix[:3, 3], atol=2e-5)
    np.testing.assert_allclose(merged_rotations[name], matrix[:3, :3], atol=2e-5)
limits = merged["LowerArm.L"]["limits_rad"][axis]
assert limits[0] >= hinge["limits"][0] and limits[1] <= hinge["limits"][1]
safe = np.radians(profile_hinge["probe_safe_limits_deg"][axis])
assert limits[0] >= safe[0] and limits[1] <= safe[1]
(out / "joint_limits.json").write_text(json.dumps(profile))
(out / "meta.json").write_text(json.dumps(meta))
bpy.ops.wm.save_as_mainfile(filepath=str(out / "posed.blend"))
# Observe the production first-frame render before any random pose is sampled.
original_render = worker.render_split_screen_frame
observations = []


def render(*args, **kwargs):
    if not observations:
        for name, vertices in expected.items():
            np.testing.assert_allclose(
                _evaluated_vertices_world(bpy.data.objects[name]), vertices, atol=3e-6
            )
    observations.append(True)
    return original_render(*args, **kwargs)


def workbench(scene, samples=None):
    scene.render.engine = "BLENDER_WORKBENCH"
    return {"engine": "BLENDER_WORKBENCH", "test_fixture": True}


worker.render_split_screen_frame = render
worker.configure_blender_gpu_rendering = workbench
worker.audit_pose_dataset_coverage = lambda *args, **kwargs: {
    "passed": True,
    "test_fixture": True,
}
worker.generate_stage2_synthetic_dataset(
    str(out / "posed.blend"),
    str(out / "meta.json"),
    str(out / "dataset"),
    num_samples=2,
)
with np.load(out / "dataset/rig_skinning.npz") as archive:
    np.testing.assert_allclose(archive["start_verts"], skin["start_verts"], atol=3e-6)
    assert archive["start_pose_basis"].shape == (len(names), 4, 4)
    assert not np.allclose(archive["start_pose_basis"], np.eye(4)[None])
print("POSED_START_PRESERVED", json.dumps(report["start_pose_rebind"]))
labels = json.loads((out / "dataset/keypoints_gt.json").read_text())
for sample in labels["samples"]:
    assert sample["v3_sampling"]["motion_roots"] == ["Hips"]
    np.testing.assert_allclose(
        sample["joints"]["LowerArm.L"]["euler_xyz"][1],
        profile_hinge["frozen_start_channels_rad"]["ry"],
        atol=1e-5,
    )
    np.testing.assert_allclose(
        sample["joints"]["Hips"]["head_3d"],
        np.array(meta["bones"]["Hips"]["rest_head_world"])
        + sample["v3_sampling"]["translation"],
        atol=1e-5,
    )
# Baking must preserve the movable pelvis below a fixed organizational Root.
parameters = rig.start_params()
sl = rig.slices["Hips"]
parameters[sl.start : sl.start + 3] = [0.2, -0.3, 1.4]
parameters[sl.start + 3 : sl.stop] += [0.1, 0.3, 0.15]
frames = rig.frames(
    np.array([parameters, parameters]), np.ones((2, 2, len(rig.point_names)))
)
frames[0]["frame"] = 0.0
frames[1]["frame"] = 1.5
trajectory = {
    "reader_contract": "monster-reader-v3",
    "frames": frames,
    "bones_meta": meta["bones"],
}
(out / "hips-trajectory.json").write_text(json.dumps(trajectory))
worker.bake_3d_joints_to_animated_glb(
    str(out / "posed.blend"),
    str(out / "hips-trajectory.json"),
    str(out / "hips.glb"),
    meta_path=str(out / "meta.json"),
)
_, imported = worker.load_glb_and_find_rig(str(out / "hips.glb"))
bpy.context.scene.frame_set(1, subframe=0.5)
bpy.context.view_layer.update()
_, expected_heads, expected_rotations = rig.fk(parameters)
matrix = np.array(imported.matrix_world @ imported.pose.bones["Hips"].matrix)
np.testing.assert_allclose(matrix[:3, 3], expected_heads["Hips"], atol=2e-5)
np.testing.assert_allclose(matrix[:3, :3], expected_rotations["Hips"], atol=2e-5)
print("ACTUAL_CHILD_HIPS_WORLD_POSE_READBACK_OK")

# A low-coverage source must try real mesh Bone Heat before the shell fallback.
bpy.ops.wm.open_mainfile(filepath=str(out / "posed.blend"))
arm = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
for obj in meshes:
    for group in obj.vertex_groups:
        group.remove(list(range(len(obj.data.vertices))))
bpy.context.view_layer.update()
targets = {obj.name: _evaluated_vertices_world(obj) for obj in meshes}
_, rebound = prepare_candidate(meshes, arm, out / "rebound", meta["bones"])
assert rebound["binding"]["method"] == "mesh_heat"
assert rebound["binding"]["mesh_weighted_fraction"] >= 0.5, rebound["binding"]
for obj in meshes:
    np.testing.assert_allclose(
        _evaluated_vertices_world(obj), targets[obj.name], atol=3e-6
    )
    assert all(vertex.groups for vertex in obj.data.vertices)
print("DIRECT_MESH_HEAT_PRESERVES_P0", json.dumps(rebound["binding"]))
