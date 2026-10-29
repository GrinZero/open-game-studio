"""Measured G2 pose acceptance: rigid part intersections and complete mesh stretch."""
import itertools
import numpy as np

from pose_features import (
    compute_mesh_edge_stretch_metric, extract_unique_edges_from_faces,
    lbs_deform_vertices_numpy,
)


class PoseMeshGuard:
    def __init__(self, objects, armature=None, skin=None):
        self.armature, self.skin = armature, skin
        self.groups = {}
        for obj in objects:
            if obj.type != "MESH" or obj.hide_render:
                continue
            used = {obj.vertex_groups[g.group].name
                    for v in obj.data.vertices for g in v.groups if g.weight > 1e-6}
            # Parts on the same rigid bone form one assembly. The flexible
            # two-bone string is included with soft geometry, not a rigid bow.
            if len(used) == 1 and next(iter(used)).startswith(("Prop_", "Weapon")):
                key = next(iter(used))
            elif used:
                key = "__flexible__"
            else:
                continue
            self.groups.setdefault(key, []).append(obj)
        self.baseline = self._intersections()
        if skin is not None:
            self.edges = extract_unique_edges_from_faces(skin["triangles"])
            mixed = np.max(skin["skin_weights"], axis=1) < .999
            self.flex_edges = self.edges[np.any(mixed[self.edges], axis=1)]

    def _intersections(self):
        import bpy
        from mathutils.bvhtree import BVHTree
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        trees = {}
        for key, objects in self.groups.items():
            vertices, faces = [], []
            for obj in objects:
                evaluated = obj.evaluated_get(graph)
                mesh = evaluated.to_mesh()
                try:
                    mesh.calc_loop_triangles()
                    offset = len(vertices)
                    vertices.extend(evaluated.matrix_world @ v.co for v in mesh.vertices)
                    faces.extend(tuple(offset + i for i in f.vertices) for f in mesh.loop_triangles)
                finally:
                    evaluated.to_mesh_clear()
            if faces:
                trees[key] = BVHTree.FromPolygons(vertices, faces, all_triangles=True, epsilon=1e-7)
        return {tuple(sorted((a, b))) for a, b in itertools.combinations(trees, 2)
                if trees[a].overlap(trees[b])}

    def evaluate(self):
        new = self._intersections() - self.baseline
        hard = [list(pair) for pair in sorted(new) if "__flexible__" not in pair]
        soft = [list(pair) for pair in sorted(new) if "__flexible__" in pair]
        metrics = {"new_hard_intersection_pairs": len(hard),
                   "new_flexible_intersection_pairs": len(soft),
                   "new_hard_pairs": hard, "new_flexible_pairs": soft}
        if self.skin is not None:
            skin, arm = self.skin, self.armature
            matrices = [arm.matrix_world @ arm.pose.bones[n].matrix for n in skin["bone_names"]]
            heads = np.array([list(m.translation) for m in matrices])
            rotations = np.array([np.array(m.to_3x3()) for m in matrices])
            vertices = lbs_deform_vertices_numpy(
                skin["rest_verts"], skin["skin_indices"], skin["skin_weights"],
                skin["rest_heads"], skin["rest_R"], heads, rotations)
            metrics["whole_mesh"] = compute_mesh_edge_stretch_metric(
                skin["rest_verts"], vertices, self.edges)
            metrics["flexible_connections"] = compute_mesh_edge_stretch_metric(
                skin["rest_verts"], vertices, self.flex_edges)
        metrics["passed"] = not hard and metrics.get("whole_mesh", {}).get("passed", True)
        return metrics
