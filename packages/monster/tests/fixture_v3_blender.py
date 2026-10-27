import sys, os, json, math
from pathlib import Path

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
import bpy, numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
import studio_bpy_worker as worker
from studio_camera import Camera
from reprojection_solver import aligned_rest_rotation, RigModel
from video2rig_blender import StripRenderer, solid_copy, render_v3_review

if os.environ.get("MONSTER_TEST_LAYOUT") == "stacked":
    os.environ["MONSTER_STUDIO_LAYOUT"] = "stacked"

out = Path(sys.argv[sys.argv.index("--") + 1])
out.mkdir(exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 1))
mesh = bpy.context.object
mesh.scale = (0.5, 0.4, 2)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
armdata = bpy.data.armatures.new("TestRig")
arm = bpy.data.objects.new("TestRig", armdata)
bpy.context.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
arm.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
root = armdata.edit_bones.new("Root")
root.head = (0, 0, 0.5)
root.tail = (0, 0, 1.2)
head = armdata.edit_bones.new("Head")
head.head = (0, 0, 1.2)
head.tail = (0, 0, 1.8)
head.parent = root
prop = armdata.edit_bones.new("Prop_sword")
prop.head = (0, 0.6, 1.0)
prop.tail = (0, 0.6, 1.7)
for bone in armdata.edit_bones:
    bone.align_roll(
        Vector(aligned_rest_rotation(np.array(bone.head), np.array(bone.tail))[:, 2])
    )
bpy.ops.object.mode_set(mode="OBJECT")
for name in ("Root", "Head"):
    mesh.vertex_groups.new(name=name)
for vertex in mesh.data.vertices:
    mesh.vertex_groups["Head" if vertex.co.z > 1.2 else "Root"].add(
        [vertex.index], 1.0, "REPLACE"
    )
mod = mesh.modifiers.new("Armature", "ARMATURE")
mod.object = arm
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0.6, 1.35))
prop_mesh = bpy.context.object
prop_mesh.name = "Prop_sword"
prop_mesh.scale = (0.04, 0.04, 0.7)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
prop_mesh.vertex_groups.new(name="Prop_sword").add(
    list(range(len(prop_mesh.data.vertices))), 1.0, "REPLACE"
)
prop_modifier = prop_mesh.modifiers.new("Armature", "ARMATURE")
prop_modifier.object = arm
meta = {
    "reader_contract": "monster-reader-v3",
    "bone_taxonomy_schema": "monster-bone-taxonomy-v3",
    "bbox_min": [-0.25, -0.2, 0],
    "bbox_max": [0.25, 0.65, 2],
    "bones": {},
}
for bone in armdata.bones:
    meta["bones"][bone.name] = {
        "parent": bone.parent.name if bone.parent else None,
        "children": [c.name for c in bone.children],
        "rest_head_world": list(bone.head_local),
        "rest_tail_world": list(bone.tail_local),
        "rest_length": bone.length,
        "rest_rot_mat_world": np.array(bone.matrix_local.to_3x3()).tolist(),
        "kind": "root"
        if bone.name == "Root" or bone.name.startswith("Prop_")
        else "head",
        "limits_rad": [[-0.4, 0.4]] * 3,
        "sub_role": "root"
        if bone.name == "Root" or bone.name.startswith("Prop_")
        else "head_leaf",
        "group_id": "0",
        "chain_index": 1,
        "side": "C",
        "anatomical_role": bone.name,
    }
(out / "studio_rig_meta.json").write_text(json.dumps(meta))
bpy.ops.wm.save_as_mainfile(filepath=str(out / "rig.blend"))
# Pinhole and ortho must match Blender's actual projection including sensor fit and shift.
for kind in ("ORTHO", "PERSP"):
    cd = bpy.data.cameras.new("ProjectionTest")
    cd.type = kind
    cd.sensor_fit = "VERTICAL"
    cd.ortho_scale = 3
    cd.lens = 50
    cd.shift_x = 0.07
    cd.shift_y = -0.03
    cam = bpy.data.objects.new(cd.name, cd)
    bpy.context.collection.objects.link(cam)
    cam.location = (3, -4, 3)
    cam.rotation_euler = (
        (Vector((0, 0, 1)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    )
    bpy.context.scene.render.resolution_x = 672
    bpy.context.scene.render.resolution_y = 768
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.view_layer.update()
    c = Camera.from_blender(cam, 672, 768)
    for p in (Vector((0.1, 0.2, 1)), Vector((-0.2, -0.1, 1.4))):
        uv = world_to_camera_view(bpy.context.scene, cam, p)
        expected = np.array([uv.x * 672, (1 - uv.y) * 768])
        actual = c.project(np.array(p))
        assert np.max(np.abs(actual - expected)) < 0.01, (kind, actual, expected)
print("ACTUAL_CAMERA_PROJECTION_OK")
from studio_camera import export_cameras, cameras_from_calibration

custom = export_cameras(out / "custom_cameras.json", {"custom_scene": cam}, 672, 768)
consumed = cameras_from_calibration(
    json.loads((out / "custom_cameras.json").read_text())
)
np.testing.assert_allclose(
    consumed["custom_scene"].project(np.array([0.1, 0.2, 1.0])),
    c.project(np.array([0.1, 0.2, 1.0])),
    atol=1e-10,
)
print("CUSTOM_CAMERA_EXPORTED_AND_CONSUMED")
# Force the fallback path explicitly on this disposable fixture and verify all vertices get weights.
print("SOLID_COPY", solid_copy([mesh], arm))
assert all(v.groups for v in mesh.data.vertices)
prop_group = mesh.vertex_groups.get("Prop_sword")
if prop_group is not None:
    assert all(
        all(g.group != prop_group.index or g.weight < 1e-8 for g in vertex.groups)
        for vertex in mesh.data.vertices
    )
assert arm.data.bones["Prop_sword"].use_deform
bpy.ops.wm.save_as_mainfile(filepath=str(out / "rig.blend"))


def workbench(scene, samples=None):
    scene.render.engine = "BLENDER_WORKBENCH"
    return {"engine": "BLENDER_WORKBENCH", "test_fixture": True}


worker.configure_blender_gpu_rendering = workbench
worker.audit_pose_dataset_coverage = lambda *args, **kwargs: {
    "passed": True,
    "test_fixture": True,
}
worker.generate_stage2_synthetic_dataset(
    str(out / "rig.blend"),
    str(out / "studio_rig_meta.json"),
    str(out / "dataset"),
    num_samples=2,
)
labels = json.loads((out / "dataset/keypoints_gt.json").read_text())
assert (
    labels["reader_contract"] == "monster-reader-v3"
    and labels["training_calibration"]["panel_width"] == 896
)
assert labels["calibration"]["panel_width"] == 672
for sample in labels["samples"]:
    assert (out / "dataset/images" / sample["image"]).is_file()
    with np.load(out / "dataset/dense" / sample["dense_map"]) as fields:
        for view in labels["training_calibration"]["view_order"]:
            assert fields[view + "_part"].max() > 0
if os.environ.get("MONSTER_TEST_LAYOUT") == "stacked":
    from PIL import Image

    training = labels["training_calibration"]
    assert training["view_order"] == ["side", "top"]
    cameras = cameras_from_calibration(training)
    probe = np.array([[0.1, 0.2, 1.0], [-0.2, -0.15, 0.8]])
    np.testing.assert_allclose(
        cameras["side"].project(probe)[:, 0],
        cameras["top"].project(probe)[:, 0],
        atol=1e-3,
        rtol=0,
    )
    with Image.open(out / "dataset" / labels["idle_reference"]) as image:
        assert image.size == (672, 1536), image.size
    from pose_features import dataset_cache_matches

    assert dataset_cache_matches(
        out / "dataset", out / "rig.blend", out / "studio_rig_meta.json", 2
    )
    print("ACTUAL_STACKED_LABELS_AND_CACHE_OK")
print("ACTUAL_STRIP_AND_DENSE_LABELS_OK")
# Exercise a complete 16-copy strip independently of the 2-sample data fixture.
arm = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
for pb in arm.pose.bones:
    pb.rotation_euler = (0, 0, 0)
    pb.location = (0, 0, 0)
bpy.context.scene.render.resolution_x = 96
bpy.context.scene.render.resolution_y = 96
camera = next(
    o
    for o in bpy.context.scene.objects
    if o.type == "CAMERA" and o.name.startswith("Cam_Front")
)
strip = StripRenderer({"front": camera}, out, count=16)
for k in range(16):
    strip.enqueue(out / f"strip16_{k:02d}.jpg")
assert all((out / f"strip16_{k:02d}.jpg").is_file() for k in range(16))
print("ACTUAL_FULL_16_COPY_STRIP_OK")
# Actual baked GLB and fractional source timestamps, followed by re-import and review rendering.
rig = RigModel(meta["bones"])
p = rig.zero()
root_slice = rig.slices["Root"]
p[root_slice.start + 3 : root_slice.stop] = [0, 0, 0.1]
sequence = np.array([p, p.copy(), p.copy()])
sequence[1, root_slice.start + 2] = 0.1
sequence[2, root_slice.start + 2] = 0.2
prop_slice = rig.slices["Prop_sword"]
sequence[:, prop_slice.start : prop_slice.start + 3] = [
    [0.1, -0.2, 0.3],
    [0.4, 0.3, -0.2],
    [-0.2, 0.5, 0.6],
]
sequence[:, prop_slice.start + 3 : prop_slice.stop] = [
    [0.1, 0.05, 0.1],
    [-0.1, 0.12, 0.2],
    [0.2, -0.1, 0.3],
]
confidence = np.ones((3, 2, len(rig.point_names)))
from video_timing import VideoTiming

is_stacked = os.environ.get("MONSTER_TEST_LAYOUT") == "stacked"
timing = VideoTiming(30 if is_stacked else 24, 0.5 if is_stacked else 1)
source_times = [0.0, 1.5, 2.5]
baked_times = timing.map_source_frames(source_times)
frames = rig.frames(sequence, confidence)
for f, t in zip(frames, baked_times):
    f["frame"] = t
traj = {
    "reader_contract": "monster-reader-v3",
    "gate_passed": True,
    "frames": frames,
    "bones_meta": meta["bones"],
    "calibration": labels["calibration"],
    "cadence": {"frame_times": source_times},
    "time_scale": timing.time_scale,
    "fps": timing.fps,
    "timing": timing.as_dict(),
}
(out / "trajectory.json").write_text(json.dumps(traj))
import subprocess

idle = out / "dataset" / labels["idle_reference"]
subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        str(idle),
        "-frames:v",
        "3",
        "-r",
        str(timing.fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out / "source.mp4"),
    ],
    check=True,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
)
report = render_v3_review(
    out / "rig.blend",
    out / "studio_rig_meta.json",
    out / "trajectory.json",
    out / "source.mp4",
    out / "review",
)
assert report["actual_glb_readback"] and len(report["frames"]) == 3
print("ACTUAL_GLB_REVIEW_OK", json.dumps(report))

# Compare evaluated imported matrices at the actual fractional key times, not just report timestamps.
readback_arm = next(obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE")
for parameters, time in zip(sequence, baked_times):
    bpy.context.scene.frame_set(math.floor(time), subframe=time % 1)
    bpy.context.view_layer.update()
    _, heads, rotations = rig.fk(parameters)
    matrix = np.array(
        readback_arm.matrix_world @ readback_arm.pose.bones["Prop_sword"].matrix
    )
    np.testing.assert_allclose(matrix[:3, 3], heads["Prop_sword"], atol=2e-5)
    np.testing.assert_allclose(matrix[:3, :3], rotations["Prop_sword"], atol=2e-5)
print("ACTUAL_PROP_SIX_DOF_FRACTIONAL_READBACK_OK")

# glTF stores seconds independently of the Blender display frame rate.
import struct

with open(report["glb"], "rb") as stream:
    binary = stream.read()
chunk_length = struct.unpack_from("<I", binary, 12)[0]
document = json.loads(binary[20 : 20 + chunk_length])
inputs = {
    sampler["input"]
    for animation in document["animations"]
    for sampler in animation["samplers"]
}
maximum = max(document["accessors"][index]["max"][0] for index in inputs)
assert abs(maximum - timing.seconds(baked_times[-1])) < 1e-7, (
    maximum,
    timing.seconds(baked_times[-1]),
)
assert abs(report["preview_fps"] - timing.preview_fps(timing.fps)) < 1e-6
assert (
    abs(report["frames"][1]["baked_seconds"] - timing.seconds(baked_times[1])) < 1e-10
)
probe = subprocess.run(
    [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "json",
        str(out / "review/g3_comparison_sync.mp4"),
    ],
    check=True,
    capture_output=True,
    text=True,
)
rate = json.loads(probe.stdout)["streams"][0]["avg_frame_rate"]
numerator, denominator = map(float, rate.split("/"))
assert abs(numerator / denominator - report["preview_fps"]) < 1e-6
print("ACTUAL_FPS_SCALE_GLB_SECONDS_AND_PREVIEW_OK", timing.as_dict())

roundtrip = worker.verify_exam_glb_roundtrip(
    report["glb"],
    str(out / "trajectory.json"),
    str(out / "review/kinematic-timing-check.json"),
)
assert roundtrip["directional_mpjre_deg"] < 0.05, roundtrip
assert roundtrip["full_rot_mpjre_deg"] < 0.05, roundtrip
print("ACTUAL_FRACTIONAL_TIMING_IN_FORMAL_ROUNDTRIP_OK")
