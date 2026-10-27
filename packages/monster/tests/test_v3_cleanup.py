"""Full digit trees and numerical boundaries for skeleton geometry repairs."""

import copy
import json
import os
import subprocess
import tempfile
from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from skeleton_postprocess import (
    Shell,
    cut_held_prop,
    drop_outside,
    extend_soles,
    fold_short_digits,
    postprocess_skintokens_skeleton,
    symmetrize_skeleton,
)


def shell(solid, origin=(0, 0, 0), step=0.01):
    return Shell(
        solid.copy(),
        solid,
        np.array(origin, float),
        step,
        distance_transform_edt(solid),
    )


def bone(parent, head, tail, kind="limb"):
    return dict(
        parent=parent, kind=kind, rest_head_world=list(head), rest_tail_world=list(tail)
    )


def digit_tree():
    meta = {
        "Root": bone(None, [0, 0, 0], [0, 0, 0.7], "root"),
        "Hand.L": bone("Root", [0.2, 0, 0.7], [0.2, 0.05, 0.7]),
        "Foot.L": bone("Root", [0, 0, 0.1], [0, 0, 0.01]),
    }
    for digit, lengths, parent in [
        ("Finger1", (0.05, 0.06), "Hand.L"),
        ("Finger2", (0.12, 0.12), "Hand.L"),
        ("Finger3", (0.11, 0.11), "Hand.L"),
        ("Toe1", (0.04, 0.05), "Foot.L"),
    ]:
        h = np.array(meta[parent]["rest_tail_world"])
        for j, length in enumerate(lengths, 1):
            name = f"{digit}_{j}.L"
            t = h + [0, length, 0]
            meta[name] = bone(parent, h, t, "digit")
            parent, h = name, t
    meta["Rigid1.L"] = bone("Finger1_2.L", [0.2, 0.16, 0.7], [0.2, 0.16, 0.72], "rigid")
    return meta


def overlap_volume(pairs, singles):
    grid = np.zeros((3, pairs + singles + 2, 3), bool)
    grid[0, :pairs, 1] = grid[2, :pairs, 1] = True
    grid[0, pairs : pairs + singles, 1] = True
    return shell(grid, origin=(-0.01, 0, 0))


class CleanupTests(unittest.TestCase):
    def test_default_preserves_complete_digits_and_explicit_fold_uses_branch_length(
        self,
    ):
        meta = digit_tree()
        # Real solid grid, height exactly one metre; contact at the root is already covered.
        solid = np.ones((101, 101, 51), bool)
        volume = shell(solid, (-1, -1, 0), 0.02)
        vertices = np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (0.0, 1.0)]
        )
        keep, report = postprocess_skintokens_skeleton(meta, vertices, volume)
        self.assertFalse(report["fold_digits"])
        self.assertEqual(set(keep), set(meta))
        folded, report = postprocess_skintokens_skeleton(
            meta, vertices, volume, fold_digits=True
        )
        short = {"Finger1_1.L", "Finger1_2.L", "Toe1_1.L", "Toe1_2.L"}
        self.assertEqual(set(folded), set(meta) - short)
        self.assertTrue(
            all(
                n in folded
                for n in ("Finger2_1.L", "Finger2_2.L", "Finger3_1.L", "Finger3_2.L")
            )
        )
        self.assertEqual(folded["Rigid1.L"]["parent"], "Hand.L")
        self.assertEqual({report["removed"][n] for n in short}, {"Hand.L", "Foot.L"})
        self.assertEqual(meta, digit_tree())

    def test_folded_fan_keeps_distal_mean_and_attachment_map(self):
        meta = digit_tree()
        for n in ["Finger2_1.L", "Finger2_2.L", "Finger3_1.L", "Finger3_2.L"]:
            del meta[n]
        meta["Finger4_1.L"] = bone(
            "Hand.L", [0.2, 0.05, 0.7], [0.3, 0.12, 0.7], "digit"
        )
        tips = np.array(
            [
                meta["Finger1_2.L"]["rest_tail_world"],
                meta["Finger4_1.L"]["rest_tail_world"],
            ]
        )
        removed = fold_short_digits(meta, 1)
        np.testing.assert_allclose(meta["Hand.L"]["rest_tail_world"], tips.mean(0))
        self.assertEqual(removed["Finger1_2.L"], "Hand.L")
        self.assertEqual(meta["Rigid1.L"]["parent"], "Hand.L")

    def test_outside_share_strict_half_and_tip_retraction_minimum(self):
        for occupied, survives in [(50, True), (49, False)]:
            grid = np.zeros((101, 5, 5), bool)
            grid[:occupied] = True
            volume = shell(grid, (0.005, 0, 0))
            meta = {
                "Root": bone(None, [0.1, 0.02, 0.01], [0.1, 0.02, 0.03], "root"),
                "Branch.L": bone("Root", [0, 0.02, 0.02], [1, 0.02, 0.02]),
                "Attachment": bone(
                    "Branch.L", [0.1, 0.02, 0.02], [0.2, 0.02, 0.02], "rigid"
                ),
            }
            removed, retract = drop_outside(meta, volume)
            self.assertEqual("Branch.L" in meta, survives)
            if survives:
                self.assertIn("Branch.L", retract)
                self.assertAlmostEqual(
                    meta["Branch.L"]["rest_tail_world"][0], 0.5, places=6
                )
                self.assertTrue(
                    volume.contains(
                        np.array(meta["Branch.L"]["rest_tail_world"])[None]
                    )[0]
                )
            else:
                self.assertEqual(removed["Branch.L"], "Root")
                self.assertEqual(meta["Attachment"]["parent"], "Root")
        grid = np.zeros((101, 5, 5), bool)
        grid[:20] = True
        meta = {"Root": bone(None, [0, 0.02, 0.02], [1, 0.02, 0.02], "root")}
        drop_outside(meta, shell(grid, (0.005, 0, 0)))
        self.assertAlmostEqual(meta["Root"]["rest_tail_world"][0], 0.3)

    def test_only_long_contiguous_thin_suffix_is_a_held_weapon(self):
        points = np.c_[np.linspace(0, 1, 1001), np.zeros((1001, 2))]
        for thin_length, expected in [
            (0.09, False),
            (0.10, False),
            (0.10001, True),
            (0.11, True),
            (0.7, True),
        ]:
            # Exact arclength boundary; include its transition point in the sample path.
            x = np.unique(np.r_[points[:, 0], 1 - thin_length])
            path = np.c_[x, np.zeros((len(x), 2))]
            radii = np.where(x >= 1 - thin_length, 0.01, 0.06)
            result = cut_held_prop(path, radii, 1, 0.01)
            self.assertEqual(result is not None, expected, thin_length)
            if expected:
                self.assertAlmostEqual(result["thin_length"], thin_length)
                self.assertAlmostEqual(result["point"][0], 1 - thin_length + 0.03)
        radii = np.where(points[:, 0] > 0.6, 0.016, 0.06)
        self.assertIsNone(cut_held_prop(points, radii, 1, 0.01))  # exceeds 1.5 voxels
        radii = np.where(points[:, 0] > 0.6, 0.011, 0.02)
        self.assertIsNone(
            cut_held_prop(points, radii, 1, 0.01)
        )  # exceeds half arm radius
        radii = np.where(points[:, 0] > 0.6, 0.015, 0.03)
        self.assertIsNotNone(
            cut_held_prop(points, radii, 1, 0.01)
        )  # both inclusive thickness limits
        radii = np.where(points[:, 0] > 0.6, 0.01, 0.06)
        radii[900] = 0.06
        self.assertIsNone(
            cut_held_prop(points, radii, 1, 0.01)
        )  # no long suffix after interruption

    def test_floor_sole_uses_nearest_owner_and_97th_percentile_forward(self):
        meta = {
            "Foot.L": bone(None, [0.04, 0, 0.02], [0.04, -0.03, 0.01]),
            "Foot.R": bone(None, [-0.04, 0, 0.02], [-0.04, -0.03, 0.01]),
            "FootOther.L": bone(None, [0.3, 0, 0.2], [0.3, -0.03, 0.2]),
        }
        left = np.array(
            [
                [0.04 + x, y, 0.005 + 0.001 * i % 0.01]
                for i, y in enumerate(np.linspace(-0.1, 0.05, 101))
                for x in (-0.005, 0.005)
            ]
        )
        right = np.array(
            [
                [-0.04 + x, y, 0.006]
                for y in np.linspace(-0.055, 0.04, 101)
                for x in (-0.005, 0.005)
            ]
        )
        vertices = np.r_[left, right, [[0.5, 0.5, 1], [0.5, 0.5, 0]]]
        moved = extend_soles(meta, vertices, 1)
        self.assertEqual(set(moved), {"Foot.L", "Foot.R"})
        for name, own in [("Foot.L", left), ("Foot.R", right)]:
            expected = own[-own[:, 1] >= np.quantile(-own[:, 1], 0.97)].mean(0)
            np.testing.assert_allclose(meta[name]["rest_tail_world"], expected)
        self.assertEqual(meta["FootOther.L"]["rest_tail_world"], [0.3, -0.03, 0.2])

    def test_head_centering_preserves_paired_ears_and_real_flex_but_drops_short_lone_head_branch(
        self,
    ):
        meta = {
            "Root": bone(None, [0, 0, 0], [0, 0, 0.7], "root"),
            "Head": bone("Root", [0.1, 0, 0.7], [0.12, -0.1, 0.9], "head"),
        }
        for side, sign in [("L", 1), ("R", -1)]:
            meta[f"EarStub.{side}"] = bone(
                "Head", [sign * 0.12, -0.05, 0.9], [sign * 0.2, -0.05, 0.97], "head"
            )
        meta["Brow"] = bone("Head", [0.14, -0.06, 0.8], [0.18, -0.08, 0.84], "head")
        meta["FalseTwin"] = bone("Head", [-0.7, 0.4, 0.3], [-0.18, -0.08, 0.84], "head")
        meta["LongHorn"] = bone("Head", [0.2, 0.5, 0.4], [0.2, -0.05, 0.9], "head")
        meta["Ear1_1.L"] = bone("Head", [0.15, -0.05, 0.9], [0.22, -0.05, 0.98], "flex")
        grid = np.ones((101, 101, 51), bool)
        vertices = np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (0.0, 1.0)]
        )
        result, report = postprocess_skintokens_skeleton(
            meta, vertices, shell(grid, (-1, -1, 0), 0.02)
        )
        self.assertEqual(result["Head"]["rest_head_world"][0], 0)
        self.assertEqual(result["Head"]["rest_tail_world"][0], 0)
        # The brow's own mirror cannot be supplied by a different long bone's endpoint.
        self.assertNotIn("Brow", result)
        self.assertIn("LongHorn", result)
        self.assertIn("Ear1_1.L", result)
        self.assertIn("EarStub.L", result)
        self.assertIn("EarStub.R", result)
        self.assertIn(["lone_head_ends", "Brow"], report["events"])


class SymmetryTests(unittest.TestCase):
    def test_exact_eighty_percent_volume_mirrors_complete_connected_trees(self):
        volume = overlap_volume(4, 1)
        self.assertEqual(volume.mirror_overlap(0, 0), 0.8)
        meta = {
            "Root": bone(None, [0, 0, 0.6], [0, 0, 1], "root"),
            "Arm.L": bone("Root", [0.2, 0, 1], [0.5, 0, 1]),
            "LowerArm.L": bone("Arm.L", [0.5, 0, 1], [0.7, -0.1, 1]),
            "Arm.R": bone("Root", [-0.3, 0.1, 1], [-0.6, 0.1, 1]),
            "LowerArm.R": bone("Arm.R", [-0.6, 0.1, 1], [-0.8, 0.2, 1]),
        }
        saved = copy.deepcopy(meta)
        symmetry, events = symmetrize_skeleton(meta, volume, 0)
        self.assertEqual(symmetry, 0.8)
        self.assertEqual(events, [["symmetrize", "Arm.L"]])
        for l, r in [("Arm.L", "Arm.R"), ("LowerArm.L", "LowerArm.R")]:
            for endpoint in ("rest_head_world", "rest_tail_world"):
                expected = np.array(saved[r][endpoint]) * [-1, 1, 1]
                expected = (expected + saved[l][endpoint]) / 2
                np.testing.assert_allclose(meta[l][endpoint], expected)
                np.testing.assert_allclose(
                    np.array(meta[r][endpoint]) * [-1, 1, 1], expected
                )
        self.assertEqual(
            meta["LowerArm.L"]["rest_head_world"], meta["Arm.L"]["rest_tail_world"]
        )

    def test_asymmetric_equalization_respects_75_percent_and_keeps_directions_and_junctions(
        self,
    ):
        volume = overlap_volume(79, 21)
        self.assertEqual(volume.mirror_overlap(0, 0), 0.79)
        for ratio, changes in [(0.75, True), (0.7499, False)]:
            meta = {"Root": bone(None, [0, 0, 0.8], [0, 0, 1], "root")}
            for side in ("L", "R"):
                h = np.array([0.2 if side == "L" else -0.2, 0, 0.8])
                for name in ("Arm", "LowerArm"):
                    delta = (
                        np.array([0, 0, -0.4])
                        if side == "L"
                        else np.array([0, 0.4 * ratio, 0])
                    )
                    meta[f"{name}.{side}"] = bone(
                        "Root" if name == "Arm" else f"Arm.{side}", h, h + delta
                    )
                    h += delta
            saved = copy.deepcopy(meta)
            _, events = symmetrize_skeleton(meta, volume, 0)
            if not changes:
                self.assertEqual(meta, saved)
                self.assertEqual(events, [])
                continue
            self.assertEqual(events, [["extend_to", "Arm.L"]])
            for side in ("L", "R"):
                for name in ("Arm", "LowerArm"):
                    key = f"{name}.{side}"
                    before = (
                        np.array(saved[key]["rest_tail_world"])
                        - saved[key]["rest_head_world"]
                    )
                    after = (
                        np.array(meta[key]["rest_tail_world"])
                        - meta[key]["rest_head_world"]
                    )
                    np.testing.assert_allclose(
                        after / np.linalg.norm(after), before / np.linalg.norm(before)
                    )
                    self.assertAlmostEqual(np.linalg.norm(after), 0.35)
                self.assertEqual(
                    meta[f"LowerArm.{side}"]["rest_head_world"],
                    meta[f"Arm.{side}"]["rest_tail_world"],
                )

    def test_unmatched_extra_branch_keeps_both_trees_unchanged(self):
        meta = {
            "Root": bone(None, [0, 0, 0.5], [0, 0, 1], "root"),
            "Arm.L": bone("Root", [0.2, 0, 1], [0.5, 0, 1]),
            "Arm.R": bone("Root", [-0.2, 0, 1], [-0.5, 0, 1]),
            "Rigid1.R": bone("Arm.R", [-0.5, 0, 1], [-0.5, 0.1, 1], "rigid"),
        }
        saved = copy.deepcopy(meta)
        _, events = symmetrize_skeleton(meta, overlap_volume(4, 1), 0)
        self.assertEqual(events, [])
        self.assertEqual(meta, saved)


class NativeCleanupTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MONSTER_TEST_BLENDER"),
        "Set MONSTER_TEST_BLENDER for native weight verification",
    )
    def test_native_group_transfer_weapon_and_fragment_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    os.environ["MONSTER_TEST_BLENDER"],
                    "--background",
                    "--factory-startup",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(Path(__file__).with_name("fixture_v3_cleanup.py")),
                    "--",
                    temporary,
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("CLEANUP_NATIVE_PASSED", result.stdout)
            report = json.loads((Path(temporary) / "native-cleanup.json").read_text())
            self.assertEqual(len(report["fragments"]), 3)
            self.assertTrue(report["weapon"] and report["digit_fold"])


if __name__ == "__main__":
    unittest.main()
