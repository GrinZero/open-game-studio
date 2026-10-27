"""Native Blender proof of folded group transfer, rigid blade weights and small fragments."""

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
from video2rig_blender import prepare_candidate, loose_weights

out = Path(sys.argv[sys.argv.index("--") + 1])
out.mkdir(parents=True, exist_ok=True)


def bone(parent, h, t, kind="limb"):
    return dict(
        parent=parent, kind=kind, rest_head_world=list(h), rest_tail_world=list(t)
    )


def armature(meta):
    arm = bpy.data.objects.new("Rig", bpy.data.armatures.new("Rig"))
    bpy.context.collection.objects.link(arm)
    bpy.ops.object.select_all(action="DESELECT")
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for name, info in meta.items():
        item = arm.data.edit_bones.new(name)
        item.head, item.tail = info["rest_head_world"], info["rest_tail_world"]
    for name, info in meta.items():
        if info.get("parent"):
            arm.data.edit_bones[name].parent = arm.data.edit_bones[info["parent"]]
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm


def cube(name, center, scale):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def assign(obj, name, ids, weight=1):
    group = obj.vertex_groups.get(name) or obj.vertex_groups.new(name=name)
    group.add(list(ids), weight, "REPLACE")


def weights(obj, index):
    return {
        obj.vertex_groups[group.group].name: group.weight
        for group in obj.data.vertices[index].groups
        if group.weight > 1e-7
    }


# The entire long finger remains even though each of its individual bones is short.
for fold in (False, True):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    body = cube("Body", (0, 0, 0.5), (1, 1, 1))
    meta = {
        "Root": bone(None, [0, 0, 0], [0, 0, 0.6], "root"),
        "Hand.L": bone("Root", [0.15, 0, 0.6], [0.2, 0, 0.65]),
        "Foot.L": bone("Root", [0, 0, 0.1], [0, 0, 0.01]),
    }
    for label, lengths, parent in [
        ("Finger1", (0.05, 0.05), "Hand.L"),
        ("Finger2", (0.12, 0.12), "Hand.L"),
        ("Finger3", (0.11, 0.11), "Hand.L"),
        ("Toe1", (0.04, 0.05), "Foot.L"),
    ]:
        head = np.array(meta[parent]["rest_tail_world"])
        for j, length in enumerate(lengths, 1):
            name = f"{label}_{j}.L"
            tail = head + [0, length, 0]
            meta[name] = bone(parent, head, tail, "digit")
            parent, head = name, tail
    meta["Rigid1.L"] = bone("Finger1_2.L", [0.2, 0.1, 0.65], [0.2, 0.1, 0.7], "rigid")
    arm = armature(meta)
    for name in meta:
        body.vertex_groups.new(name=name)
    assign(body, "Finger1_1.L", [0], 0.3)
    assign(body, "Finger1_2.L", [0], 0.7)
    for i, name in enumerate(
        [
            "Finger2_1.L",
            "Finger2_2.L",
            "Rigid1.L",
            "Root",
            "Foot.L",
            "Finger3_1.L",
            "Finger3_2.L",
        ],
        1,
    ):
        assign(body, name, [i])
    body.modifiers.new("Armature", "ARMATURE").object = arm
    processed, report = prepare_candidate(
        [body], arm, out / ("fold" if fold else "keep"), meta, fold_digits=fold
    )
    short = {"Finger1_1.L", "Finger1_2.L", "Toe1_1.L", "Toe1_2.L"}
    expected = set(meta) - short if fold else set(meta)
    assert set(b.name for b in arm.data.bones) == expected
    assert set(processed) == expected
    if fold:
        assert weights(body, 0) == {"Hand.L": 1.0}, weights(body, 0)
        assert arm.data.bones["Rigid1.L"].parent.name == "Hand.L"
        assert all(body.vertex_groups.get(n) is None for n in short)
    else:
        np.testing.assert_allclose(
            [weights(body, 0)["Finger1_1.L"], weights(body, 0)["Finger1_2.L"]],
            [0.3, 0.7],
            atol=1e-7,
        )
    assert weights(body, 1) == {"Finger2_1.L": 1.0}

# A real thin blade must be split by the production shell/geometry path and receive rigid weights.
bpy.ops.wm.read_factory_settings(use_empty=True)
body = cube("Body", (0.1, 0, 0.5), (0.6, 0.4, 1))
blade = cube("Blade", (0.8, 0, 0.5), (0.8, 0.01, 0.01))
meta = {
    "Root": bone(None, [0, 0, 0], [0, 0, 0.6], "root"),
    "Foot.L": bone("Root", [0, 0, 0.1], [0.1, 0, 0.01]),
    "Hand.L": bone("Root", [0.2, 0, 0.5], [1.2, 0, 0.5]),
}
arm = armature(meta)
for obj, name in [(body, "Root"), (blade, "Hand.L")]:
    assign(obj, name, range(len(obj.data.vertices)))
    obj.modifiers.new("Armature", "ARMATURE").object = arm
processed, report = prepare_candidate([body, blade], arm, out / "weapon", meta)
assert "Weapon.L" in processed, report
assert processed["Weapon.L"]["fixed_local"] and processed["Weapon.L"]["weapon"]
assert processed["Weapon.L"]["identified_thin_length"] > 0.1
assert processed["Weapon.L"]["parent"] == "Hand.L"
assert arm.data.bones["Weapon.L"].parent.name == "Hand.L"
for vertex in blade.data.vertices:
    if vertex.co.x > 0.5:
        assert weights(blade, vertex.index) == {"Weapon.L": 1.0}, weights(
            blade, vertex.index
        )

# Native connected components across separate mesh objects use the whole character denominator.
fragment_results = []
for count, small in [(147, False), (148, True), (200, True)]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    meta = {
        "Root": bone(None, [0, 0, 0], [0, 0, 1], "root"),
        "Hand.L": bone("Root", [0.3, 0, 0.5], [0.4, 0, 0.5]),
        "Hand.R": bone("Root", [-0.3, 0, 0.5], [-0.4, 0, 0.5]),
        "Prop_remote": bone(None, [0.2, 0.15, 0], [0.2, 0.15, 1], "root"),
    }
    arm = armature(meta)
    data = bpy.data.meshes.new("Body")
    data.from_pydata(
        [[0, 0.25, 0.2 + i * 0.6 / count] for i in range(count)],
        [(i, i + 1) for i in range(count - 1)],
        [],
    )
    body = bpy.data.objects.new("Body", data)
    bpy.context.collection.objects.link(body)
    assign(body, "Root", range(count))
    # A clear opposite-limb leak must be removed on the main connected component too.
    body.data.vertices[0].co = (0.35, 0, 0.5)
    body.vertex_groups["Root"].remove([0])
    assign(body, "Hand.R", [0])
    data = bpy.data.meshes.new("Fragment")
    data.from_pydata(
        [[0.2, 0.15, 0.5], [0.201, 0.15, 0.5], [0.2, 0.151, 0.5]], [], [[0, 1, 2]]
    )
    fragment = bpy.data.objects.new("Fragment", data)
    bpy.context.collection.objects.link(fragment)
    assign(fragment, "Hand.R", range(3))
    # Hundreds of independent prop vertices do not make a 2% body fragment look smaller.
    data = bpy.data.meshes.new("Prop_remote")
    data.from_pydata([[0.2, 0.15, i / 500] for i in range(500)], [], [])
    prop = bpy.data.objects.new("Prop_remote", data)
    bpy.context.collection.objects.link(prop)
    loose_weights([body, fragment, prop], arm, 1)
    result = weights(fragment, 0)
    if small:
        distances = np.array([0.25, np.hypot(0.1, 0.15), np.hypot(0.5, 0.15)])
        expected = distances**-4
        expected /= expected.sum()
        np.testing.assert_allclose(
            [result[n] for n in ("Root", "Hand.L", "Hand.R")], expected, atol=1e-6
        )
    else:
        assert result == {"Hand.R": 1.0}, result
    assert weights(body, 0) == {"Hand.L": 1.0}
    assert weights(prop, 0) == {"Prop_remote": 1.0}
    assert "Prop_remote" not in result
    fragment_results.append({"body_vertices": count, "small": small, "weights": result})

(out / "native-cleanup.json").write_text(
    json.dumps(
        {"fragments": fragment_results, "weapon": True, "digit_fold": True}, indent=2
    )
)
print("CLEANUP_NATIVE_PASSED")
