import sys
from pathlib import Path
import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from pose_mesh_guard import PoseMeshGuard

bpy.ops.wm.read_factory_settings(use_empty=True)
def cube(name, position, groups):
    bpy.ops.mesh.primitive_cube_add(size=1, location=position)
    obj = bpy.context.object
    obj.name = name
    for group in groups:
        obj.vertex_groups.new(name=group).add(list(range(8)), 1 / len(groups), "REPLACE")
    return obj

a = cube("A", (0, 0, 0), ["Prop_a"])
b = cube("B", (2, 0, 0), ["Prop_b"])
c = cube("C", (0, 0, 0), ["Prop_c"])
soft = cube("String", (-2, 0, 0), ["Prop_a", "Prop_c"])
guard = PoseMeshGuard([a, b, c, soft])
assert guard.evaluate()["passed"]  # Original assembly contact stays allowed.
b.location = (0, 0, 0)
collision = guard.evaluate()
assert collision["new_hard_intersection_pairs"] == 2, collision
assert not collision["passed"]
b.location = (-2, 0, 0)
contact = guard.evaluate()
assert contact["new_hard_intersection_pairs"] == 0, contact
assert contact["new_flexible_intersection_pairs"] == 1, contact
assert contact["passed"]
print("MEASURED_RIGID_COLLISIONS_AND_FLEXIBLE_CONTACT_OK")
