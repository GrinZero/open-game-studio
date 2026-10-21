"""Skin-surface visibility proxy, deliberately not ray tests to internal bone centers."""
import numpy as np
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view


class SurfaceVisibilitySampler:
    def __init__(self, armature, bone_names, scale):
        self.armature = armature
        self.names = set(bone_names)
        self.scale = float(scale)
        self.adjacent = {}
        self._topology_cache = {}
        for name in bone_names:
            bone = armature.data.bones[name]
            self.adjacent[name] = {c.name for c in bone.children}
            if bone.parent:
                self.adjacent[name].add(bone.parent.name)

    def _get_static_mesh_topology(self, obj, mesh):
        key = (
            obj.name,
            len(mesh.vertices),
            len(mesh.loop_triangles),
            tuple(g.name for g in obj.vertex_groups),
        )
        cached = self._topology_cache.get(key)
        if cached is not None:
            return cached
        group_names = {g.index: g.name for g in obj.vertex_groups if g.name in self.names}
        vertex_owners = []
        for vertex in mesh.vertices:
            weights = sorted(
                ((g.weight, group_names[g.group]) for g in vertex.groups if g.group in group_names),
                reverse=True,
            )
            owner = None
            if weights and weights[0][0] >= 0.45 and (len(weights) < 2 or weights[0][0] - weights[1][0] >= 0.15):
                owner = weights[0][1]
            vertex_owners.append(owner)
        local_polygons = []
        tri_owners = []
        owned_tris = []
        for tri in mesh.loop_triangles:
            v0, v1, v2 = int(tri.vertices[0]), int(tri.vertices[1]), int(tri.vertices[2])
            local_polygons.append((v0, v1, v2))
            o0, o1, o2 = vertex_owners[v0], vertex_owners[v1], vertex_owners[v2]
            # Ambiguous skin seams are unknown, not false negatives.
            owner = o0 if (o0 and o0 == o1 and o0 == o2) else None
            tri_owners.append(owner)
            if owner:
                owned_tris.append((owner, v0, v1, v2))
        cached = (local_polygons, tri_owners, owned_tris)
        self._topology_cache[key] = cached
        return cached

    def update(self):
        verts, polygons, owners = [], [], []
        self.support = {name: [] for name in self.names}
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for obj in bpy.context.scene.objects:
            if obj.type != "MESH" or obj.hide_render:
                continue
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            try:
                mesh.calc_loop_triangles()
                offset = len(verts)
                world = evaluated.matrix_world
                positions = [world @ vertex.co for vertex in mesh.vertices]
                verts.extend(positions)
                local_polygons, tri_owners, owned_tris = self._get_static_mesh_topology(obj, mesh)
                if offset == 0:
                    polygons.extend(local_polygons)
                else:
                    polygons.extend((offset + v0, offset + v1, offset + v2) for (v0, v1, v2) in local_polygons)
                owners.extend(tri_owners)
                for owner, v0, v1, v2 in owned_tris:
                    # Interior surface points avoid ray misses on silhouette vertices/edges.
                    self.support[owner].append((positions[v0] + positions[v1] + positions[v2]) / 3)
            finally:
                evaluated.to_mesh_clear()
        self.owners = owners
        self.bvh = BVHTree.FromPolygons(verts, polygons, all_triangles=True) if polygons else None

    def label(self, point, bone_name, camera, calibration):
        scene = bpy.context.scene
        projected = world_to_camera_view(scene, camera, point)
        center = Vector(calibration["center"])
        ref_depth = world_to_camera_view(scene, camera, center).z
        in_frame = bool(projected.z > 0 and 0 <= projected.x <= 1 and 0 <= projected.y <= 1)
        result = {"depth_camera": float(projected.z),
                  "depth_relative": float((projected.z-ref_depth)/self.scale),
                  "in_frame": in_frame, "label_valid": bool(projected.z > 0),
                  "visibility_ratio": 0.0, "visibility_valid": False}
        if not in_frame:
            result["visibility_valid"] = True
            return result
        support = self.support.get(bone_name, [])
        if not support or self.bvh is None:
            return result
        # A neighborhood on this bone's skin, never the internal joint itself.
        support = sorted(support, key=lambda p: (p-point).length_squared)[:12]
        axis = (camera.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()
        visible, known = 0, 0
        for surface in support:
            if camera.data.type == "ORTHO":
                depth = (surface-camera.matrix_world.translation).dot(axis)
                origin = surface-axis*depth
                direction = axis
            else:
                origin = camera.matrix_world.translation
                direction = (surface-origin).normalized()
            length = (surface-origin).length
            hit, _, index, distance = self.bvh.ray_cast(origin, direction, length+self.scale*0.002)
            if hit is None:
                continue
            owner = self.owners[index]
            if owner == bone_name:
                known += 1
                visible += 1
            elif owner is not None and owner not in self.adjacent[bone_name]:
                known += 1
            # Unweighted surfaces / adjacent joint seams remain unknown.
        if known >= 3:
            result["visibility_ratio"] = float(visible/known)
            result["visibility_valid"] = True
        return result


def randomize_studio_lighting(studio, rng):
    """Real illumination variation; no geometric change to the fixed camera contract."""
    center = None
    if isinstance(studio, dict) and "calibration" in studio and "center" in studio["calibration"]:
        center = Vector(studio["calibration"]["center"])
    for name, energy in (("KeyLight", 900.0), ("FillLight", 550.0), ("RimLight", 700.0)):
        light = bpy.data.objects.get(name)
        if light is None:
            continue
        if "feature_base_location" not in light:
            light["feature_base_location"] = list(light.location)
        base = Vector(light["feature_base_location"])
        light.location = base + Vector(tuple(rng.uniform(-0.5, 0.5) for _ in range(3)))
        if center is not None:
            light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
        light.data.energy = energy*rng.uniform(0.7, 1.3)
        light.data.color = tuple(rng.uniform(0.85, 1.0) for _ in range(3))
    return {name: {"energy": float(bpy.data.objects[name].data.energy),
                    "color": list(bpy.data.objects[name].data.color),
                    "location": list(bpy.data.objects[name].location)}
            for name in ("KeyLight", "FillLight", "RimLight") if name in bpy.data.objects}
