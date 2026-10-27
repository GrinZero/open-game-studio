"""Native cloth/missing-leg, quadruped hip and curved-tail candidate evidence."""

from pathlib import Path
import json
import sys

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
from scipy import ndimage
from skimage.measure import marching_cubes
from test_v3_body_repairs import robed_legs, incomplete_biped
from skeleton_postprocess import voxelize
from video2rig_blender import (
    prepare_candidate,
    heat_bind_updated_skeleton,
    _evaluated_vertices_world,
)
from reprojection_solver import RigModel
import studio_bpy_worker as worker

out = Path(sys.argv[sys.argv.index("--") + 1])
out.mkdir(parents=True, exist_ok=True)


def bone(parent, head, tail, kind="limb", **extra):
    return dict(
        parent=parent,
        rest_head_world=list(head),
        rest_tail_world=list(tail),
        kind=kind,
        **extra,
    )


def tube_mesh(center, radii, tubes, step=0.025):
    anchors = np.concatenate(
        [np.array([center]) - radii, np.array([center]) + radii]
        + [np.array(p) for p, _ in tubes]
    )
    margin = max(r for _, r in tubes) + 3 * step
    origin = np.floor((anchors.min(0) - margin) / step) * step
    shape = np.ceil((anchors.max(0) + margin - origin) / step).astype(int) + 1
    xyz = np.indices(shape)
    solid = (
        sum(((origin[i] + xyz[i] * step - center[i]) / radii[i]) ** 2 for i in range(3))
        <= 1
    )
    by_radius = {}
    for path, radius in tubes:
        r = max(1, round(radius / step))
        lines = by_radius.setdefault(r, np.zeros(tuple(shape), bool))
        for a, b in zip(np.asarray(path), np.asarray(path)[1:]):
            n = max(2, int(np.ceil(np.linalg.norm(b - a) / step * 3)))
            points = np.rint(
                (a + np.linspace(0, 1, n)[:, None] * (b - a) - origin) / step
            ).astype(int)
            lines[tuple(points.T)] = True
    for r, lines in by_radius.items():
        x, y, z = np.mgrid[-r : r + 1, -r : r + 1, -r : r + 1]
        solid |= ndimage.binary_dilation(
            lines, structure=x * x + y * y + z * z <= r * r
        )
    vertices, faces, _, _ = marching_cubes(np.pad(solid.astype(float), 1), 0.5)
    return origin + (vertices - 1) * step, faces


def candidate(name, vertices, faces, raw, *, automatic=False, pose=None):
    directory = out / name
    directory.mkdir(exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.update()
    mesh = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(mesh)
    arm = bpy.data.objects.new("Rig", bpy.data.armatures.new("Rig"))
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    for i, (n, b) in enumerate(raw.items()):
        item = arm.data.edit_bones.new(n)
        item.head, item.tail, item.roll = (
            b["rest_head_world"],
            b["rest_tail_world"],
            0.07 * i,
        )
    for n, b in raw.items():
        if b.get("parent"):
            arm.data.edit_bones[n].parent = arm.data.edit_bones[b["parent"]]
    bpy.ops.object.mode_set(mode="OBJECT")
    mesh.parent = arm
    mesh.modifiers.new("Armature", "ARMATURE").object = arm
    if automatic:
        heat_bind_updated_skeleton([mesh], arm)
    else:
        mesh.vertex_groups.new(name="Root").add(
            list(range(len(mesh.data.vertices))), 1, "REPLACE"
        )
    for n, angles in (pose or {}).items():
        arm.pose.bones[n].rotation_mode = "XYZ"
        arm.pose.bones[n].rotation_euler = angles
    bpy.context.view_layer.update()
    for n, b in raw.items():
        db = arm.data.bones[n]
        b.update(
            rest_rot_mat_world=np.array(db.matrix_local)[:3, :3].tolist(),
            rest_length=db.length,
            pose_basis=np.array(arm.pose.bones[n].matrix_basis).tolist(),
        )
    original = _evaluated_vertices_world(mesh).copy()
    bpy.ops.wm.save_as_mainfile(filepath=str(directory / "source.blend"))
    prepared, report = prepare_candidate([mesh], arm, directory / "prepared", raw)
    error = np.linalg.norm(_evaluated_vertices_world(mesh) - original, axis=1).max()
    assert error < 3e-5, (name, error)
    meta = worker._classify_and_canonicalize_candidate_rig(
        [mesh], arm, prepared_bones=prepared
    )
    skin = worker.extract_complete_rig_skinning([mesh], arm, sorted(meta["bones"]))
    rig = RigModel(meta["bones"], skin=skin)
    _, heads, rots = rig.fk(rig.start_params())
    for n in rig.names:
        actual = np.array(arm.matrix_world @ arm.pose.bones[n].matrix)
        np.testing.assert_allclose(
            heads[n], actual[:3, 3], atol=3e-5, err_msg=(name, n)
        )
        np.testing.assert_allclose(
            rots[n], actual[:3, :3], atol=3e-5, err_msg=(name, n)
        )
    bpy.ops.wm.save_as_mainfile(filepath=str(directory / "prepared.blend"))
    (directory / "meta.json").write_text(json.dumps(meta))
    return (
        mesh,
        arm,
        prepared,
        report,
        dict(
            surface_max_error=float(error),
            bones=len(rig.names),
            vertices=len(mesh.data.vertices),
            binding=report["binding"],
        ),
    )


results = {}
shell = robed_legs(True, width=1)
v, f, _, _ = marching_cubes(np.pad(shell.solid.astype(float), 1), 0.5)
v = shell.origin + (v - 1) * shell.step
mesh, arm, prepared, report, results["robed"] = candidate(
    "robed", v, f, incomplete_biped()
)
assert sum(e[0] == "transplant_missing_legs" for e in report["events"]) == 1, report
right = [n for n in arm.data.bones.keys() if n.endswith(".R")]
assert len(right) == 3 and "Foot.R" in right, sorted(arm.data.bones.keys())
assert not any(n.startswith("Wing") for n in arm.data.bones.keys())
nearest = np.argsort(np.linalg.norm(v - [-0.2, 0, 0.02], axis=1))[:12]
groups = {mesh.vertex_groups[n].index for n in right}
ownership = [
    sum(g.weight for g in mesh.data.vertices[int(i)].groups if g.group in groups)
    for i in nearest
]
assert np.mean(ownership) > 0.5, ownership
results["robed"].update(
    transplanted_bones=right, foot_ownership=float(np.mean(ownership))
)

raw = {
    "Root": bone(None, [0, 0, 1.0], [0, 0, 1.1], "root"),
    "Hips": bone("Root", [0, -0.5, 1.05], [0, 0.6, 1.05], "spine"),
    "Tail1": bone("Hips", [0, 0.6, 1.05], [0, 1.15, 1.05], "tail"),
}
tubes = [([[0, 0.5, 1.05], [0, 1.15, 1.05]], 0.06)]
for side, sign in (("L", 1), ("R", -1)):
    for rear, y in ((True, 0.35), (False, -0.4)):
        roles = (
            ("Thigh", "Shin", "HindPaw", "HindToes")
            if rear
            else ("UpperArm", "Forearm", "FrontPaw", "FrontToes")
        )
        points = [
            [sign * 0.2, y, 1.0],
            [sign * 0.2, y, 0.6],
            [sign * 0.2, y, 0.15],
            [sign * 0.2, y - 0.08, 0.04],
            [sign * 0.2, y - 0.16, 0.025],
        ]
        tubes.append((points, 0.065))
        parent = "Hips"
        for j, (a, b) in enumerate(zip(points, points[1:])):
            head = [a[0], -0.1, 0.95] if rear and j == 0 else a
            n = roles[j] + "." + side
            raw[n] = bone(parent, head, b)
            parent = n
v, f = tube_mesh(np.array([0, 0, 0.95]), np.array([0.32, 0.7, 0.3]), tubes)
H = np.ptp(v[:, 2])
geometry = voxelize(v, f, 0.01 * H)
expected_hips = [sign * 0.2 for sign in (1, -1)]
expected_z = 1.05 - 0.25 * geometry.radius(np.array([[0, 0.35, 1.05]]))[0]
front_before = {
    n: list(raw[n]["rest_head_world"]) for n in ("UpperArm.L", "UpperArm.R")
}
mesh, arm, prepared, report, results["quadruped"] = candidate(
    "quadruped", v, f, raw, automatic=True, pose={"Root": [0.02, -0.03, 0.05]}
)
for n, x in zip(("Thigh.L", "Thigh.R"), expected_hips):
    np.testing.assert_allclose(
        prepared[n]["rest_head_world"], [x, 0.35, expected_z], atol=2e-5
    )
    np.testing.assert_allclose(
        arm.data.bones[n].head_local, prepared[n]["rest_head_world"], atol=2e-5
    )
for n, point in front_before.items():
    np.testing.assert_allclose(prepared[n]["rest_head_world"], point, atol=2e-5)
results["quadruped"].update(
    expected_hip_z=float(expected_z), fixed_hind_hips=2, front_limbs_unchanged=True
)

raw = {
    "Root": bone(None, [0, 0, 0.8], [0, 0, 0.95], "root"),
    "Hips": bone("Root", [0, -0.15, 0.8], [0, 0.15, 0.8], "spine"),
    "Tail1": bone("Hips", [0, 0.15, 0.8], [0, 0.9, 0.8], "tail"),
    "Tail2": bone("Tail1", [0, 0.9, 0.8], [0.5, 1.65, 0.9], "tail"),
    "Rigid1": bone(
        "Tail2", [0.5, 1.65, 0.9], [0.5, 1.65, 1.05], "rigid", fixed_local=True
    ),
    "Foot.L": bone("Root", [0.12, -0.05, 0.7], [0.12, -0.05, 0.02]),
    "Foot.R": bone("Root", [-0.12, -0.05, 0.7], [-0.12, -0.05, 0.02]),
}
tubes = [
    ([[0, 0.1, 0.8], [0, 0.9, 0.8], [0.5, 1.65, 0.9]], 0.065),
    ([[0.5, 1.65, 0.9], [0.5, 1.65, 1.05]], 0.03),
]
for sign in (1, -1):
    tubes.append(([[sign * 0.12, -0.05, 0.75], [sign * 0.12, -0.05, 0.04]], 0.055))
v, f = tube_mesh(np.array([0, -0.05, 0.8]), np.array([0.27, 0.35, 0.3]), tubes)
mesh, arm, prepared, report, results["tail"] = candidate(
    "tail",
    v,
    f,
    raw,
    automatic=True,
    pose={"Root": [0.02, -0.03, 0.05], "Tail1": [0.08, 0.03, -0.02]},
)
names = [n for n in prepared if n.startswith("Tail")]
assert len(names) == 10, names
assert prepared["Tail1"]["parent"] == "Hips"
assert prepared["Rigid1"]["parent"] == "Tail10"
for i in range(2, 11):
    assert prepared[f"Tail{i}"]["parent"] == f"Tail{i - 1}"
    np.testing.assert_allclose(
        prepared[f"Tail{i}"]["rest_head_world"],
        prepared[f"Tail{i - 1}"]["rest_tail_world"],
    )
np.testing.assert_allclose(
    prepared["Tail10"]["rest_tail_world"], [0.5, 1.65, 0.9], atol=2e-5
)
results["tail"].update(
    tail_bones=10, attachment_parent="Tail10", saved_nonzero_pose=True
)
(out / "body-repairs.json").write_text(json.dumps(results, indent=2))
print("BODY_REPAIRS_CANDIDATE", json.dumps(results))
