"""Weighted volume routes and asymmetric multileg anatomical joint placement."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from skeleton_postprocess import (
    Shell,
    _interior_paths,
    _leg_joint_fractions,
    free_limb_paths,
    manylegs_build,
)


def make_shell(solid, step=0.02, origin=(-0.8, -1, 0)):
    return Shell(
        solid & ~ndimage.binary_erosion(solid),
        solid,
        np.array(origin),
        step,
        ndimage.distance_transform_edt(solid),
    )


def creature(left=3, right=3, radius=3):
    shape = (81, 101, 91)
    xyz = np.indices(shape)
    center = np.array([40, 50, 65])
    torso = (
        sum(
            ((xyz[i] - center[i]) / radius) ** 2
            for i, radius in enumerate((12, 34, 14))
        )
        <= 1
    )
    lines = np.zeros(shape, bool)
    tips = []
    for sign, count in ((1, left), (-1, right)):
        for y in np.linspace(12, 88, count):
            attach_y = 50 + 0.6 * (y - 50)
            path = np.array(
                [
                    [40 + sign * 7, attach_y, 64],
                    [
                        40 + sign * (24 if sign > 0 else 27),
                        attach_y,
                        76 if sign > 0 else 70,
                    ],
                    [40 + sign * 32, y, 27],
                    [40 + sign * 33, y, 3],
                ]
            )
            for a, b in zip(path, path[1:]):
                samples = np.rint(a + np.linspace(0, 1, 180)[:, None] * (b - a)).astype(
                    int
                )
                lines[tuple(samples.T)] = True
            tips.append(path[-1])
    x, y, z = np.mgrid[-radius : radius + 1, -radius : radius + 1, -radius : radius + 1]
    solid = torso | ndimage.binary_dilation(
        lines, structure=x * x + y * y + z * z <= radius**2
    )
    shell = make_shell(solid)
    return (
        shell,
        shell.origin + center * shell.step,
        shell.origin + np.array(tips) * shell.step,
    )


def bent_path(fractions, side=1, offset=0):
    distances = np.diff(np.r_[0, fractions, 1])
    directions = np.array([[1, 0, 0], [0, 0, -1], [1, 0, 0], [0, 0, -1]])
    corners = np.r_[
        np.zeros((1, 3)), np.cumsum(distances[:, None] * directions, axis=0)
    ]
    points = np.concatenate(
        [
            a
            + np.linspace(0, 1, max(2, int(np.ceil(length / 0.002))), endpoint=False)[
                :, None
            ]
            * (b - a)
            for a, b, length in zip(corners, corners[1:], distances)
        ]
        + [corners[-1:]]
    )
    points[:, 0] = side * (points[:, 0] + 0.1)
    points += [0, offset, 1]
    return points


class ManyLegTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MONSTER_TEST_BLENDER"),
        "Set MONSTER_TEST_BLENDER for real six-leg binding",
    )
    def test_native_candidate_attaches_to_spine_and_weights_all_six_feet(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    os.environ["MONSTER_TEST_BLENDER"],
                    "--background",
                    "--factory-startup",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(Path(__file__).with_name("fixture_v3_manylegs.py")),
                    "--",
                    temporary,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout[-14000:])
            self.assertIn("MANYLEGS_CANDIDATE", result.stdout)

    def test_volume_route_prefers_long_wide_interior_over_short_surface_tube(self):
        solid = np.zeros((75, 48, 31), bool)
        solid[10:65, 17, 15] = True
        solid[8:15, 17:32, 12:19] = True
        solid[60:67, 17:32, 12:19] = True
        solid[8:67, 25:34, 11:20] = True
        shell = make_shell(solid, step=1, origin=(0, 0, 0))
        path = _interior_paths(shell, np.array([11, 17, 15]), np.array([[63, 17, 15]]))[
            0
        ]
        self.assertGreater(
            path[:, 1].max(),
            25,
            "The narrow direct tube is cheaper in distance, but not in the required surface cost",
        )
        self.assertTrue(shell.contains(path).all())
        np.testing.assert_equal(path[0], [11, 17, 15])
        np.testing.assert_equal(path[-1], [63, 17, 15])

    def test_disconnected_contact_is_not_given_a_fabricated_route(self):
        solid = np.zeros((30, 10, 10), bool)
        solid[2:9, 3:7, 3:7] = True
        solid[20:27, 3:7, 3:7] = True
        shell = make_shell(solid, step=1, origin=(0, 0, 0))
        self.assertEqual(_interior_paths(shell, [4, 4, 4], [[24, 4, 4]]), [])

    def test_local_joint_windows_follow_three_separate_bends(self):
        expected = np.array([0.15, 0.55, 0.85])
        measured = _leg_joint_fractions(bent_path(expected), 0.005, 1)
        np.testing.assert_allclose(measured, expected, atol=0.025)
        self.assertTrue((np.diff(measured) > 0.15).all())

    def test_paired_fractions_are_mean_of_both_measured_shapes(self):
        left = np.array([0.15, 0.49, 0.84])
        right = np.array([0.29, 0.57, 0.74])
        paths = [bent_path(left, 1, y) for y in (-0.3, 0, 0.3)] + [
            bent_path(right, -1, y) for y in (-0.3, 0, 0.3)
        ]
        solid = np.ones((170, 90, 120), bool)
        shell = make_shell(solid, step=0.01, origin=(-0.8, -0.45, 0))
        shell.distance[:] = 1
        with (
            patch("skeleton_postprocess._footprints", return_value=np.zeros((6, 3))),
            patch("skeleton_postprocess._interior_paths", return_value=paths),
        ):
            bones = manylegs_build(shell, np.array([0, 0, 1]))
        for k in (1, 2, 3):
            a = bones[f"Coxa{k}.L"]["paired_arc_fractions"]
            b = bones[f"Coxa{k}.R"]["paired_arc_fractions"]
            np.testing.assert_allclose(a, b, atol=1e-12)
            np.testing.assert_allclose(a[1:-1], (left + right) / 2, atol=0.025)

    def test_real_volume_keeps_six_asymmetric_legs_outside_shared_torso(self):
        shell, root, tips = creature()
        bones = manylegs_build(shell, root)
        self.assertEqual(len(bones), 24)
        for side, sign in (("L", 1), ("R", -1)):
            for k in (1, 2, 3):
                co = bones[f"Coxa{k}.{side}"]
                self.assertGreater(sign * (co["rest_head_world"][0] - root[0]), 0.12)
                self.assertGreater(co["rest_head_world"][2], 0.9)
                end = np.array(bones[f"Tarsus{k}.{side}"]["rest_tail_world"])
                self.assertLess(np.linalg.norm(tips - end, axis=1).min(), 0.07)
                for a, b in zip(
                    ("Coxa", "Femur", "Tibia"), ("Femur", "Tibia", "Tarsus")
                ):
                    self.assertEqual(
                        bones[f"{b}{k}.{side}"]["parent"], f"{a}{k}.{side}"
                    )
                    np.testing.assert_allclose(
                        bones[f"{a}{k}.{side}"]["rest_tail_world"],
                        bones[f"{b}{k}.{side}"]["rest_head_world"],
                    )
                np.testing.assert_allclose(
                    co["paired_arc_fractions"],
                    bones[f"Coxa{k}.R"]["paired_arc_fractions"],
                )

    def test_unpaired_legs_are_preserved_instead_of_zip_truncation(self):
        shell, root, _ = creature(left=4, right=2)
        bones = manylegs_build(shell, root)
        self.assertEqual(len(bones), 24)
        self.assertTrue(
            all(
                f"{role}{k}.L" in bones
                for role in ("Coxa", "Femur", "Tibia", "Tarsus")
                for k in (1, 2, 3, 4)
            )
        )
        self.assertTrue(all(f"Coxa{k}.R" in bones for k in (1, 4)))
        self.assertNotIn("Coxa2.R", bones)
        self.assertNotIn("Coxa3.R", bones)
        np.testing.assert_allclose(
            bones["Coxa4.L"]["paired_arc_fractions"],
            bones["Coxa4.R"]["paired_arc_fractions"],
        )

    def test_empty_volume_and_five_grounded_legs_do_not_enable_manylegs(self):
        empty = make_shell(np.zeros((5, 5, 5), bool))
        self.assertEqual(manylegs_build(empty, [0, 0, 0]), {})
        shell, root, _ = creature(left=3, right=2)
        self.assertEqual(manylegs_build(shell, root), {})

    def test_one_voxel_contact_and_thin_legs_are_not_rejected_as_skirt_sheets(self):
        shell, root, _ = creature(radius=1)
        bones = manylegs_build(shell, root)
        self.assertEqual(len(bones), 24)
        self.assertTrue(
            all(
                np.linalg.norm(np.array(b["rest_tail_world"]) - b["rest_head_world"])
                > 0.03
                for b in bones.values()
            )
        )

    def test_dense_sampling_does_not_count_one_path_as_multiple_shared_branches(self):
        shell = Mock(step=1)
        shell.indices.side_effect = lambda points: np.rint(points).astype(int)
        shell.radius.side_effect = lambda points: np.full(len(points), 2.0)
        trunk = np.array([[0, 0, z] for z in np.arange(0, 11, 0.1)])
        left = np.array([[x, 0, 11] for x in np.arange(0, 17, 0.1)])
        right = left.copy()
        right[:, 0] *= -1
        result = free_limb_paths(shell, [np.r_[trunk, left], np.r_[trunk, right]])
        for path in result:
            self.assertLessEqual(abs(path[0, 0]), 0.5)
            self.assertEqual(path[0, 2], 11)
            self.assertGreater(abs(path[-1, 0]), 16)


if __name__ == "__main__":
    unittest.main()
