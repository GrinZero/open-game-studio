"""Geometry-derived hinge axes/signs and measured LBS safety intersections."""

import copy
import math
import random
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from reprojection_solver import (
    RigModel,
    aligned_rest_rotation,
    axis_roles,
    exp_so3,
    find_hinges,
    sample_v3_pose,
)
from bone_taxonomy import (
    apply_joint_limits_profile,
    probe_rig_joint_limits,
    effective_sampling_limits_for_bone,
)
from pose_features import lbs_deform_vertices_numpy, compute_mesh_edge_stretch_metric


def leg(bend_deg=30, roll_deg=0, yaw=0, pose_deg=0):
    angle = math.radians(bend_deg)
    meta = {
        "Root": dict(
            parent=None,
            kind="root",
            rest_head_world=[0, 0, 0.8],
            rest_tail_world=[0, 0, 1.5],
        ),
        "Thigh.L": dict(
            parent="Root",
            kind="limb",
            rest_head_world=[0.2, 0, 1],
            rest_tail_world=[0.2, 0, 0.5],
        ),
        "Shin.L": dict(
            parent="Thigh.L",
            kind="limb",
            rest_head_world=[0.2, 0, 0.5],
            rest_tail_world=[0.2, 0.45 * math.sin(angle), 0.5 - 0.45 * math.cos(angle)],
        ),
    }
    E = exp_so3([0, 0, yaw])
    for name, bone in meta.items():
        h, t = np.array(bone["rest_head_world"]), np.array(bone["rest_tail_world"])
        R = aligned_rest_rotation(h, t)
        if name == "Shin.L":
            R = R @ exp_so3([0, math.radians(roll_deg), 0])
        bone.update(
            rest_head_world=(E @ h).tolist(),
            rest_tail_world=(E @ t).tolist(),
            rest_rot_mat_world=(E @ R).tolist(),
            limits_rad=[[-math.pi, math.pi]] * 3,
            limits_source="canonical_default",
        )
        B = np.eye(4)
        if name == "Shin.L":
            axis = 2 if abs(roll_deg) == 90 else 0
            B[:3, :3] = exp_so3(np.eye(3)[axis] * math.radians(pose_deg))
        bone["pose_basis"] = B.tolist()
    return meta, E @ np.array([0, -1, 0.0])


def joint_angle(rig, p, name="Shin.L"):
    X = rig.fk(p)[0]
    i, j = 2 * rig.index[name], 2 * rig.index[rig.parents[name]]
    a, b = X[i + 1] - X[i], X[j + 1] - X[j]
    return math.acos(
        float(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))
    )


class HingeGeometryTests(unittest.TestCase):
    def test_forward_reverse_and_rolled_axes_under_all_world_headings(self):
        for bend in (30, -30):
            for roll in (0, 90, -90):
                for yaw in (0, np.pi / 2, np.pi, -np.pi / 2):
                    with self.subTest(bend=bend, roll=roll, yaw=yaw):
                        meta, facing = leg(bend, roll, yaw)
                        before = RigModel(meta)
                        report = find_hinges(meta, facing)
                        hinge = report["Shin.L"]
                        self.assertEqual(hinge["axis"], 0 if roll == 0 else 2)
                        self.assertEqual(hinge["axis_source"], "rest_bend_plane")
                        self.assertAlmostEqual(
                            abs(hinge["straight"]), math.radians(30), places=10
                        )
                        self.assertAlmostEqual(
                            hinge["rest_bend"], math.radians(30), places=10
                        )
                        p = before.start_params()
                        index = before.slices["Shin.L"][0][hinge["axis"]]
                        q = p.copy()
                        q[index] += 0.15 * hinge["hinge_sign"]
                        self.assertGreater(
                            joint_angle(before, q), joint_angle(before, p)
                        )
                        q[index] = hinge["straight"]
                        self.assertLess(joint_angle(before, q), 2e-8)
                        self.assertAlmostEqual(
                            hinge["anatomical_limits"][1]
                            - hinge["anatomical_limits"][0],
                            2.75,
                        )

    def test_twelve_degree_boundary_uses_named_direction_until_exceeded(self):
        results = []
        for bend in (-11.9999, -12, -12.0001):
            meta, facing = leg(bend)
            results.append(find_hinges(meta, facing)["Shin.L"])
        self.assertEqual([h["hinge_sign"] for h in results], [1, 1, -1])
        self.assertEqual(
            [h["direction_source"] for h in results],
            ["knee_back", "knee_back", "start_pose_fold"],
        )

    def test_non_rest_start_pose_decides_fold_direction(self):
        for pose in (-25, 25):
            meta, facing = leg(0, 90, pose_deg=pose)
            result = find_hinges(meta, facing)["Shin.L"]
            self.assertEqual(result["axis"], 2)
            self.assertEqual(result["direction_source"], "start_pose_fold")
            self.assertAlmostEqual(result["rest_bend"], math.radians(25))
            self.assertAlmostEqual(result["straight"], 0, places=12)
            self.assertEqual(result["hinge_sign"], 1 if pose > 0 else -1)
            self.assertAlmostEqual(result["start_angle"], math.radians(pose))

    def test_straight_elbows_move_forward_on_both_sides(self):
        for side, sign in (("L", 1), ("R", -1)):
            meta = {
                "Root": dict(
                    parent=None,
                    kind="root",
                    rest_head_world=[0, 0, 0.5],
                    rest_tail_world=[0, 0, 1],
                ),
                f"UpperArm.{side}": dict(
                    parent="Root",
                    kind="limb",
                    rest_head_world=[sign * 0.1, 0, 1],
                    rest_tail_world=[sign * 0.5, 0, 1],
                ),
                f"LowerArm.{side}": dict(
                    parent=f"UpperArm.{side}",
                    kind="limb",
                    rest_head_world=[sign * 0.5, 0, 1],
                    rest_tail_world=[sign * 0.9, 0, 1],
                ),
            }
            rig = RigModel(meta)
            report = find_hinges(meta)[f"LowerArm.{side}"]
            p = rig.zero()
            q = p.copy()
            q[rig.slices[f"LowerArm.{side}"][0][report["axis"]]] += (
                0.1 * report["hinge_sign"]
            )
            tip = 2 * rig.index[f"LowerArm.{side}"] + 1
            self.assertLess((rig.fk(q)[0][tip] - rig.fk(p)[0][tip])[1], -0.03)
            self.assertEqual(report["direction_source"], "elbow_front")

    def test_explicit_limits_are_intersected_and_empty_intersections_fail(self):
        meta, facing = leg(30)
        meta["Shin.L"].update(
            limits_source="agent_custom", limits_rad=[[-0.3, 0.4], [-1, 1], [-1, 1]]
        )
        result = find_hinges(meta, facing)["Shin.L"]
        np.testing.assert_allclose(result["limits"], [-0.3, 0.4])
        meta, facing = leg(30)
        meta["Shin.L"].update(
            limits_source="agent_custom", limits_rad=[[-2, -1], [-1, 1], [-1, 1]]
        )
        with self.assertRaisesRegex(ValueError, "do not intersect"):
            find_hinges(meta, facing)

    def test_axis_roles_follow_rolled_geometry_and_do_not_change_pose(self):
        meta, facing = leg(0, 90, np.pi / 2)
        before = RigModel(meta).fk(RigModel(meta).start_params())[0]
        roles = axis_roles(meta, facing)
        self.assertEqual(roles["Shin.L"], {"rx": "side", "ry": "twist", "rz": "bend"})
        np.testing.assert_array_equal(
            before, RigModel(meta).fk(RigModel(meta).start_params())[0]
        )
        with self.assertRaises(ValueError):
            axis_roles(meta, up=(0, 0, 0))


class SafeHingeProbeTests(unittest.TestCase):
    @staticmethod
    def mesh_fixture(roll=False):
        meta = {
            "Root": dict(
                parent=None,
                kind="root",
                rest_head_world=[0, 0, 0],
                rest_tail_world=[0, 0, 1],
            ),
            "LowerArm.L": dict(
                parent="Root",
                kind="limb",
                rest_head_world=[0, 0, 1],
                rest_tail_world=[0, 0, 2],
                limits_source="canonical_default",
                limits_rad=[[-math.pi, math.pi]] * 3,
            ),
        }
        for name, bone in meta.items():
            R = aligned_rest_rotation(
                np.array(bone["rest_head_world"]), np.array(bone["rest_tail_world"])
            )
            if roll and name != "Root":
                R = R @ exp_so3([0, math.pi / 2, 0])
            bone["rest_rot_mat_world"] = R.tolist()
        find_hinges(meta)
        skin = {
            "bone_names": list(meta),
            "rest_heads": [b["rest_head_world"] for b in meta.values()],
            "rest_R": [b["rest_rot_mat_world"] for b in meta.values()],
            "rest_verts": [[0, 0, 2], [0, 0.05, 2], [0.05, 0, 2]],
            "triangles": [[0, 1, 2]],
            "skin_indices": [[0] * 4, [0] * 4, [1, 0, 0, 0]],
            "skin_weights": [[1, 0, 0, 0]] * 3,
        }
        return meta, skin

    @staticmethod
    def deformed_metric(meta, skin, axis, angle):
        rig = RigModel(meta)
        params = rig.zero()
        params[rig.slices["LowerArm.L"][0][axis]] = angle
        _, heads, rotations = rig.fk(params)
        posed = lbs_deform_vertices_numpy(
            np.asarray(skin["rest_verts"]),
            np.asarray(skin["skin_indices"]),
            np.asarray(skin["skin_weights"]),
            np.asarray(skin["rest_heads"]),
            np.asarray(skin["rest_R"]),
            np.array([heads[n] for n in skin["bone_names"]]),
            np.array([rotations[n] for n in skin["bone_names"]]),
        )
        return compute_mesh_edge_stretch_metric(
            np.asarray(skin["rest_verts"]),
            posed,
            np.asarray(skin["triangles"]),
            max_bad_ratio=0,
            max_stretch=1.8,
        )

    def test_first_fifteen_degree_failure_never_claims_fifteen_is_safe(self):
        for rolled in (False, True):
            meta, skin = self.mesh_fixture(rolled)
            axis = meta["LowerArm.L"]["hinge"]["axis"]
            self.assertEqual(axis, 2 if rolled else 0)
            self.assertFalse(
                self.deformed_metric(meta, skin, axis, math.radians(15))["passed"]
            )
            profile = probe_rig_joint_limits(
                meta, rig_skinning=skin, max_bad_edge_ratio=0
            )
            bone = profile["bones"]["LowerArm.L"]
            maximum = bone["probe_safe_limits_deg"][axis][1]
            self.assertGreater(maximum, 0)
            self.assertLess(maximum, 5)
            self.assertTrue(
                self.deformed_metric(meta, skin, axis, math.radians(maximum))["passed"]
            )
            self.assertFalse(
                self.deformed_metric(meta, skin, axis, math.radians(maximum + 0.2))[
                    "passed"
                ]
            )
            limits = bone["limits_rad"][axis]
            anatomical = meta["LowerArm.L"]["hinge"]["limits"]
            self.assertGreaterEqual(limits[0], anatomical[0])
            self.assertLessEqual(limits[1], math.radians(maximum))
            merged = apply_joint_limits_profile(meta, profile)
            final = merged["LowerArm.L"]["limits_rad"][axis]
            self.assertGreaterEqual(final[0], anatomical[0])
            self.assertLessEqual(final[1], math.radians(maximum))
            self.assertEqual(
                merged["LowerArm.L"]["axis_roles"], meta["LowerArm.L"]["axis_roles"]
            )
            self.assertEqual(merged["LowerArm.L"]["hinge"]["axis"], axis)

    def test_profile_generation_does_not_mutate_input_geometry(self):
        meta, skin = self.mesh_fixture(True)
        saved = copy.deepcopy(meta)
        probe_rig_joint_limits(meta, rig_skinning=skin, max_bad_edge_ratio=0)
        self.assertEqual(meta, saved)

    def test_frozen_leaf_twist_retains_saved_p0_through_probe_merge_and_sampling(self):
        meta, skin = self.mesh_fixture()
        angle = 0.123456789
        basis = np.eye(4)
        basis[:3, :3] = exp_so3([0, angle, 0])
        meta["LowerArm.L"]["pose_basis"] = basis.tolist()
        find_hinges(meta)
        before = RigModel(meta)
        profile = probe_rig_joint_limits(meta, rig_skinning=skin)
        bone = profile["bones"]["LowerArm.L"]
        self.assertAlmostEqual(
            bone["frozen_start_channels_rad"]["ry"], angle, places=12
        )
        self.assertTrue(bone["allow_leaf_twist"])
        merged = apply_joint_limits_profile(meta, profile)
        after = RigModel(merged)
        self.assertNotIn(
            1, after.slices["LowerArm.L"][1]
        )  # zero new optimization freedom
        np.testing.assert_allclose(
            before.fk(before.start_params())[0],
            after.fk(after.start_params())[0],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            before.fk(before.start_params())[2]["LowerArm.L"],
            after.fk(after.start_params())[2]["LowerArm.L"],
            atol=1e-12,
        )
        np.testing.assert_allclose(
            effective_sampling_limits_for_bone(merged["LowerArm.L"])[1],
            [angle, angle],
            atol=1e-12,
        )
        eulers, _, _ = sample_v3_pose(merged, {}, random.Random(4), events={})
        self.assertAlmostEqual(eulers["LowerArm.L"][1], angle, places=12)
        again = probe_rig_joint_limits(merged, rig_skinning=skin)["bones"]["LowerArm.L"]
        np.testing.assert_allclose(again["limits_rad"][1], [angle, angle], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
