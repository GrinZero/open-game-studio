"""Actual multileg candidate preparation, trunk attachment and distal skin ownership."""

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
from skimage.measure import marching_cubes
from test_v3_manylegs import creature
from video2rig_blender import prepare_candidate
import studio_bpy_worker as worker

out = Path(sys.argv[sys.argv.index("--") + 1])
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
shell, root, tips = creature()
vertices, triangles, _, _ = marching_cubes(np.pad(shell.solid.astype(float), 1), 0.5)
vertices = shell.origin + (vertices - 1) * shell.step
data = bpy.data.meshes.new("SixLegMesh")
data.from_pydata(vertices, [], triangles)
data.update()
mesh = bpy.data.objects.new("SixLegBody", data)
bpy.context.collection.objects.link(mesh)
arm = bpy.data.objects.new("Rig", bpy.data.armatures.new("Rig"))
bpy.context.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
arm.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
root_bone = arm.data.edit_bones.new("Root")
root_bone.head, root_bone.tail = (0, 0, 1.18), (0, 0, 1.23)
spine = arm.data.edit_bones.new("Spine1")
spine.head, spine.tail = (0, 0.5, 1.3), (0, -0.5, 1.3)
spine.parent = root_bone
bpy.ops.object.mode_set(mode="OBJECT")
mesh.parent = arm
mesh.vertex_groups.new(name="Root").add(
    list(range(len(mesh.data.vertices))), 1, "REPLACE"
)
mesh.modifiers.new("Armature", "ARMATURE").object = arm
raw = {}
for bone in arm.data.bones:
    matrix = np.array(bone.matrix_local)
    raw[bone.name] = dict(
        parent=bone.parent.name if bone.parent else None,
        kind="root" if bone.name == "Root" else "spine",
        rest_head_world=list(bone.head_local),
        rest_tail_world=list(bone.tail_local),
        rest_rot_mat_world=matrix[:3, :3].tolist(),
        rest_length=bone.length,
        pose_basis=np.eye(4).tolist(),
    )
prepared, report = prepare_candidate([mesh], arm, out / "prepared", raw)
assert report["manylegs"] == 6, report
meta = worker._classify_and_canonicalize_candidate_rig(
    [mesh], arm, prepared_bones=prepared
)
names = [
    f"{role}{k}.{side}"
    for side in ("L", "R")
    for k in (1, 2, 3)
    for role in ("Coxa", "Femur", "Tibia", "Tarsus")
]
assert all(n in meta["bones"] for n in names), sorted(meta["bones"])
for side in ("L", "R"):
    for k in (1, 2, 3):
        name = f"Coxa{k}.{side}"
        assert meta["bones"][name]["parent"] == "Spine1", (
            name,
            meta["bones"][name]["parent"],
        )
        assert arm.data.bones[name].parent.name == "Spine1"
        for a, b in zip(("Coxa", "Femur", "Tibia"), ("Femur", "Tibia", "Tarsus")):
            assert arm.data.bones[f"{b}{k}.{side}"].parent.name == f"{a}{k}.{side}"
        np.testing.assert_allclose(
            prepared[name]["paired_arc_fractions"],
            prepared[f"Coxa{k}.R"]["paired_arc_fractions"],
        )
ownership = {}
for side, tip_group in (("L", tips[:3]), ("R", tips[3:])):
    for k, tip in enumerate(tip_group, 1):
        chosen = sorted(
            mesh.data.vertices, key=lambda v: np.linalg.norm(np.array(v.co) - tip)
        )[:12]
        groups = {
            mesh.vertex_groups[f"{role}{k}.{side}"].index
            for role in ("Coxa", "Femur", "Tibia", "Tarsus")
        }
        weights = [sum(g.weight for g in v.groups if g.group in groups) for v in chosen]
        ownership[f"{k}.{side}"] = float(np.mean(weights))
        assert np.mean(weights) > 0.5, (side, k, weights)
assert all(sum(g.weight for g in v.groups) > 0.99 for v in mesh.data.vertices)
(out / "meta.json").write_text(json.dumps(meta))
bpy.ops.wm.save_as_mainfile(filepath=str(out / "six-leg.blend"))
result = dict(
    passed=True,
    legs=6,
    limb_bones=len(names),
    vertices=len(mesh.data.vertices),
    root_parent="Spine1",
    foot_ownership=ownership,
    binding=report["binding"],
)
(out / "manylegs.json").write_text(json.dumps(result, indent=2))
print("MANYLEGS_CANDIDATE", json.dumps(result))
