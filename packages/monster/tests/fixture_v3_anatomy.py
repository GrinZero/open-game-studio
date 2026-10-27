"""Actual candidate semantic handoff, bone/group/action renaming and pose preservation."""

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
from test_v3_anatomy import paired_case
import studio_bpy_worker as worker
from video2rig_blender import _evaluated_vertices_world

out = Path(sys.argv[sys.argv.index("--") + 1])
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
meta, logical = paired_case(wing=True, common_digit_heads=True, distal_only=True)
arm = bpy.data.objects.new("Rig", bpy.data.armatures.new("Rig"))
bpy.context.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
arm.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
for i, (name, info) in enumerate(meta.items()):
    b = arm.data.edit_bones.new(name)
    b.head, b.tail = info["rest_head_world"], info["rest_tail_world"]
    b.roll = 0.07 * (i % 5)
for name, info in meta.items():
    if info.get("parent"):
        arm.data.edit_bones[name].parent = arm.data.edit_bones[info["parent"]]
bpy.ops.object.mode_set(mode="OBJECT")
meshes = []
cube_faces = [
    (0, 1, 3, 2),
    (4, 6, 7, 5),
    (0, 4, 5, 1),
    (2, 3, 7, 6),
    (0, 2, 6, 4),
    (1, 5, 7, 3),
]
for part in range(2):
    vertices, faces, groups = [], [], {}
    for i, (name, b) in enumerate(meta.items()):
        if i % 2 != part:
            continue
        start = len(vertices)
        center = 0.5 * (np.array(b["rest_head_world"]) + b["rest_tail_world"])
        vertices.extend(
            center + [x, y, z]
            for x in (-0.008, 0.008)
            for y in (-0.008, 0.008)
            for z in (-0.008, 0.008)
        )
        faces.extend(tuple(start + v for v in face) for face in cube_faces)
        groups[name] = list(range(start, start + 8))
    data = bpy.data.meshes.new(f"Part{part}")
    data.from_pydata(vertices, [], faces)
    obj = bpy.data.objects.new(data.name, data)
    bpy.context.collection.objects.link(obj)
    for name, ids in groups.items():
        obj.vertex_groups.new(name=name).add(ids, 1.0, "REPLACE")
    obj.modifiers.new("Rig", "ARMATURE").object = arm
    meshes.append(obj)
for i, pb in enumerate(arm.pose.bones):
    pb.rotation_mode = "XYZ"
    for frame, factor in ((1, 1), (3, -1)):
        pb.rotation_euler = (
            factor * 0.03 * (i % 3),
            factor * 0.04 * (i % 5),
            factor * 0.02,
        )
        pb.keyframe_insert("rotation_euler", frame=frame)
expected_vertices, expected_matrices = {}, {}
for frame in (1, 3):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    expected_vertices[frame] = {
        obj.name: _evaluated_vertices_world(obj) for obj in meshes
    }
    expected_matrices[frame] = {
        pb.name: np.array(arm.matrix_world @ pb.matrix) for pb in arm.pose.bones
    }
bpy.context.scene.frame_set(1)
for name, info in meta.items():
    info["pose_basis"] = np.array(arm.pose.bones[name].matrix_basis).tolist()
result = worker._classify_and_canonicalize_candidate_rig(
    meshes, arm, prepared_bones=meta
)
renames = result["canonical_rename_map"]
for side in ("L", "R"):
    for j, role in enumerate(
        ("WingShoulder", "WingUpper", "WingForearm", "WingHand"), 1
    ):
        new = renames[logical[("wing", side, j)]]
        assert new == role + "." + side, (new, role)
        assert result["bones"][new]["kind"] == "wing"
        np.testing.assert_allclose(
            np.array(result["bones"][new]["limits_rad"])[[0, 2]],
            np.round(np.radians([[-40, 45], [-35, 35]]), 4),
            atol=1e-12,
        )
for family in ("wing_digit", "leg_digit"):
    for finger in (1, 2, 3):
        for joint in (1, 2):
            left = renames[logical[(family, "L", finger, joint)]]
            right = renames[logical[(family, "R", finger, joint)]]
            assert left[:-2] + ".R" == right, (left, right)
surface_error = 0.0
matrix_error = 0.0
for frame in (1, 3):
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    for obj in meshes:
        surface_error = max(
            surface_error,
            float(
                np.max(
                    abs(
                        _evaluated_vertices_world(obj)
                        - expected_vertices[frame][obj.name]
                    )
                )
            ),
        )
        assert all(g.name in result["bones"] for g in obj.vertex_groups)
        for vertex in obj.data.vertices:
            assert len(vertex.groups) == 1 and vertex.groups[0].weight == 1.0
    for old, new in renames.items():
        matrix_error = max(
            matrix_error,
            float(
                np.max(
                    abs(
                        np.array(arm.matrix_world @ arm.pose.bones[new].matrix)
                        - expected_matrices[frame][old]
                    )
                )
            ),
        )
        actual_parent = arm.data.bones[new].parent
        assert result["bones"][new]["parent"] == (
            actual_parent.name if actual_parent else None
        )
assert surface_error < 2e-6 and matrix_error < 2e-6
report = dict(
    wing_bones=sum(b["kind"] == "wing" for b in result["bones"].values()),
    surface_error=surface_error,
    matrix_error=matrix_error,
    renames=renames,
    bone_count=len(meta),
)
(out / "native-anatomy.json").write_text(json.dumps(report, indent=2))
print("ANATOMY_NATIVE_PASSED")
