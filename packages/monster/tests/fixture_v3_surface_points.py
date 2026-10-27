"""Native rolled, posed, split-mesh landmarks through production skin and label export."""

from pathlib import Path
import json
import sys
import numpy as np

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
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Euler, Vector
from studio_bpy_worker import extract_complete_rig_skinning
from video2rig_blender import attach_reader_points
from reprojection_solver import RigModel, surface_keypoints
from pose_visibility import SurfaceVisibilitySampler
from studio_camera import export_cameras, cameras_from_calibration

out = Path(sys.argv[sys.argv.index("--") + 1])
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
arm = bpy.data.objects.new("Rig", bpy.data.armatures.new("Rig"))
bpy.context.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
arm.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
parents = {
    "Hips": None,
    "Hand.L": "Hips",
    "Foot.L": "Hips",
    "FrontPaw.R": "Hips",
    "HindPaw.L": "Hips",
    "WingHand.R": "Hips",
    "Head": "Hips",
    "Finger1_1.L": "Hand.L",
    "Jaw": "Head",
    "Prop_tool": None,
}
for i, (name, parent) in enumerate(parents.items()):
    b = arm.data.edit_bones.new(name)
    b.head = (0.2 * i, 0, 0.7 + 0.05 * i)
    b.tail = b.head + Vector((0, 0.5, 0))
    b.roll = 0.11 * (i + 1)
    if parent:
        b.parent = arm.data.edit_bones[parent]
bpy.ops.object.mode_set(mode="OBJECT")
arm.matrix_world = Euler((0.21, -0.14, 0.31)).to_matrix().to_4x4()
arm.matrix_world.translation = Vector((0.2, -0.1, 0.15))
bpy.context.view_layer.update()
meta = {}
for name, parent in parents.items():
    b = arm.data.bones[name]
    M = np.array(arm.matrix_world @ b.matrix_local)
    meta[name] = dict(
        parent=parent,
        kind="root" if parent is None else "limb",
        rest_head_world=M[:3, 3].tolist(),
        rest_tail_world=list(arm.matrix_world @ b.tail_local),
        rest_rot_mat_world=M[:3, :3].tolist(),
        children=[c.name for c in b.children],
    )

meshes = []
expected_support = {name: [] for name in parents}
for part in range(2):
    points, groups, faces = [], [], []
    for i, name in enumerate(parents):
        if i % 2 != part:
            continue
        # Full-rank, non-circular transverse surface, with exactly known quantiles.
        local = np.array(
            [
                [x, y, z]
                for x in np.linspace(-0.12, 0.12, 51)
                for y in (0.1, 0.5)
                for z in (-0.04, 0.04)
            ]
        )
        M = np.array(arm.data.bones[name].matrix_local)
        coords = M[:3, 3] + local @ M[:3, :3].T
        start = len(points)
        points.extend(coords)
        groups.extend([name] * len(coords))
        # Triangles give the real SurfaceVisibilitySampler owned surface support.
        for j in range(0, len(coords) - 3, 4):
            faces.extend(
                [
                    (start + j, start + j + 1, start + j + 2),
                    (start + j + 1, start + j + 3, start + j + 2),
                ]
            )
        expected_support[name] = local
    mesh = bpy.data.meshes.new(f"Part{part}")
    mesh.from_pydata(points, [], faces)
    obj = bpy.data.objects.new(mesh.name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.matrix_world = arm.matrix_world.copy()
    for name in parents:
        ids = [j for j, owner in enumerate(groups) if owner == name]
        if ids:
            obj.vertex_groups.new(name=name).add(ids, 1, "REPLACE")
    obj.modifiers.new("Armature", "ARMATURE").object = arm
    meshes.append(obj)
for i, pb in enumerate(arm.pose.bones):
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (0.07 * (i % 3), 0.13 * (i + 1), -0.03 * i)
arm.pose.bones["Hips"].location = (0.04, -0.02, 0.03)
arm.pose.bones["Prop_tool"].location = (-0.1, 0.07, 0.09)
bpy.context.view_layer.update()
for name, b in meta.items():
    b["pose_basis"] = np.array(arm.pose.bones[name].matrix_basis).tolist()
names = sorted(meta)
skin = extract_complete_rig_skinning(meshes, arm, list(reversed(names)))
extra = surface_keypoints(meta, names, skin)
lookup = {p["name"]: p for p in extra}
assert (
    "Hips.tip" not in lookup and "Hand.L.tip" not in lookup and "Head.tip" not in lookup
)
assert "Finger1_1.L.tip" in lookup and "Jaw.tip" in lookup
for name, cloud in expected_support.items():
    if name + ".tip" in lookup:
        np.testing.assert_allclose(
            lookup[name + ".tip"]["local"],
            [0, np.quantile(cloud[:, 1], 0.98), 0],
            atol=2e-6,
        )
    if name + ".sidea1" in lookup:
        sidea = np.array([lookup[f"{name}.sidea{j}"]["local"] for j in (1, 2)])
        sideb = np.array([lookup[f"{name}.sideb{j}"]["local"] for j in (1, 2)])
        np.testing.assert_allclose(
            np.sort(sidea[:, 0]), np.quantile(cloud[:, 0], [0.01, 0.99]), atol=2e-6
        )
        np.testing.assert_allclose(sidea[:, 1:], [[0.3, 0], [0.3, 0]], atol=2e-6)
        np.testing.assert_allclose(np.sort(sideb[:, 2]), [-0.04, 0.04], atol=2e-6)
        np.testing.assert_allclose(sideb[:, :2], [[0, 0.3], [0, 0.3]], atol=2e-6)
for label, y in [("end1", 0.1), ("end2", 0.5), ("mid", 0.3)]:
    np.testing.assert_allclose(
        lookup["Prop_tool." + label]["local"], [0, y, 0], atol=2e-6
    )
rig = RigModel(meta, names, skin)
points = rig.fk(rig.start_params())[0]
expected_world = []
for name in names:
    pb = arm.pose.bones[name]
    expected_world.extend(
        [list(arm.matrix_world @ pb.head), list(arm.matrix_world @ pb.tail)]
    )
for p in extra:
    expected_world.append(
        list(arm.matrix_world @ arm.pose.bones[p["bone"]].matrix @ Vector(p["local"]))
    )
np.testing.assert_allclose(points, expected_world, atol=2e-6)
scene = bpy.context.scene
scene.render.resolution_percentage = 100
calib = dict(center=[1, 0, 1], depth_scale=3)
cameras = {}
for kind, position in [("ortho", (1, -5, 3)), ("persp", (5, -2, 4))]:
    data = bpy.data.cameras.new(kind)
    data.type = "ORTHO" if kind == "ortho" else "PERSP"
    data.ortho_scale = 4
    data.lens = 45
    data.shift_x, data.shift_y = 0.07, -0.05
    cam = bpy.data.objects.new(kind, data)
    bpy.context.collection.objects.link(cam)
    cam.location = position
    cam.rotation_euler = (
        (Vector(calib["center"]) - cam.location).to_track_quat("-Z", "Y").to_euler()
    )
    cameras[kind] = cam
bpy.context.view_layer.update()
visibility = SurfaceVisibilitySampler(arm, names, calib["depth_scale"])
visibility.update()
error = 0.0
cases = []
for width, height in [(640, 480), (480, 640)]:
    scene.render.resolution_x, scene.render.resolution_y = width, height
    calib.update(panel_width=width, panel_height=height)
    for fit in ("AUTO", "VERTICAL", "HORIZONTAL"):
        for cam in cameras.values():
            cam.data.sensor_fit = fit
        bpy.context.view_layer.update()
        record = {}
        attach_reader_points(record, arm, names, extra, cameras, visibility, calib)
        path = out / f"cameras-{width}-{fit}.json"
        export_cameras(path, cameras, width, height)
        exported = cameras_from_calibration(json.loads(path.read_text()))
        for kind, cam in cameras.items():
            oracle = []
            for p in expected_world:
                uv = world_to_camera_view(scene, cam, Vector(p))
                oracle.append([uv.x * width, (1 - uv.y) * height])
            actual = np.array(record["reader_points"][kind]["uv"])
            case_error = float(np.max(np.linalg.norm(actual - oracle, axis=1)))
            error = max(error, case_error)
            assert case_error < 0.001, (width, height, fit, kind, case_error)
            np.testing.assert_allclose(
                exported[kind].project(points), actual, atol=0.001
            )
            assert len(record["reader_points"][kind]["visibility"]) == len(points)
            cases.append(
                dict(
                    width=width,
                    height=height,
                    fit=fit,
                    type=kind,
                    pixel_error=case_error,
                )
            )
report = dict(
    meshes=len(skin["measured_mesh_names"]),
    endpoints=2 * len(names),
    points=len(points),
    pixel_error=error,
    keypoint_names=rig.point_names,
    camera_cases=cases,
)
(out / "native-surface-points.json").write_text(json.dumps(report, indent=2))
print("SURFACE_POINTS_NATIVE_PASSED")
