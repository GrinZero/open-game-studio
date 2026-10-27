"""Body cleanup must keep real contacts when cloth joins their floor patches."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from skeleton_postprocess import (
    Shell,
    _footprints,
    cross_section_flatness,
    postprocess_skintokens_skeleton,
)


def robed_legs(hem=True, width=1):
    xyz = np.indices((81, 81, 91))
    solid = (
        sum(((xyz[i] - [40, 40, 68][i]) / r) ** 2 for i, r in enumerate((14, 10, 17)))
        <= 1
    )
    lines = np.zeros_like(solid)
    for x in (30, 50):
        lines[x, 40, 4:61] = True
    x, y, z = np.mgrid[-4:5, -4:5, -4:5]
    solid |= ndimage.binary_dilation(lines, structure=x * x + y * y + z * z <= 16)
    if hem:
        solid[18:63, 40 : 40 + width, 0:61] = True
    shell = Shell(
        solid & ~ndimage.binary_erosion(solid),
        solid,
        np.array([-0.8, -0.8, 0]),
        0.02,
        ndimage.distance_transform_edt(solid),
    )
    return shell


def incomplete_biped():
    def bone(parent, head, tail, kind):
        return dict(
            parent=parent,
            rest_head_world=list(head),
            rest_tail_world=list(tail),
            kind=kind,
        )

    return {
        "Root": bone(None, [0, 0, 1.3], [0, 0, 1.5], "root"),
        "Thigh.L": bone("Root", [0.2, 0, 1.1], [0.2, 0, 0.65], "limb"),
        "Shin.L": bone("Thigh.L", [0.2, 0, 0.65], [0.2, 0, 0.2], "limb"),
        "Foot.L": bone("Shin.L", [0.2, 0, 0.2], [0.2, 0, 0.02], "limb"),
    }


class BodyRepairTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MONSTER_TEST_BLENDER"),
        "Set MONSTER_TEST_BLENDER for real body candidate repairs",
    )
    def test_native_robed_leg_quadruped_hips_and_ten_segment_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    os.environ["MONSTER_TEST_BLENDER"],
                    "--background",
                    "--factory-startup",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(Path(__file__).with_name("fixture_v3_body_repairs.py")),
                    "--",
                    temporary,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout[-14000:])
            self.assertIn("BODY_REPAIRS_CANDIDATE", result.stdout)

    def test_filtered_wing_cross_section_keeps_character_height_and_empty_support_is_neutral(
        self,
    ):
        solid = np.zeros((101, 61, 91), bool)
        solid[45:56, 25:36, 20:76] = True
        solid[55:96, 10:51, 55:57] = True
        shell = Shell(
            solid.copy(),
            solid,
            np.array([-1.0, -0.6, 0]),
            0.02,
            ndimage.distance_transform_edt(solid),
        )
        indices = np.argwhere(solid)
        wing = shell.origin + indices[indices[:, 0] > 60] * shell.step
        self.assertGreater(
            cross_section_flatness(
                shell, [0.2, 0, 1.1], [0.9, 0, 1.1], volume_points=wing
            ),
            5,
        )
        self.assertEqual(
            cross_section_flatness(
                shell, [0.2, 0, 1.1], [0.9, 0, 1.1], volume_points=np.empty((0, 3))
            ),
            1,
        )

    def test_thin_hem_connecting_feet_does_not_erase_two_real_contacts(self):
        bare = np.array(_footprints(robed_legs(False)))
        self.assertEqual(len(bare), 2)
        for width in (1, 2):
            robed = np.array(_footprints(robed_legs(True, width)))
            self.assertEqual(len(robed), 2)
            np.testing.assert_allclose(robed, bare, atol=0.02)

    def test_sheet_without_volumetric_feet_does_not_create_a_contact(self):
        solid = np.zeros((40, 40, 60), bool)
        for width in (1, 2):
            solid[8:32, 20 : 20 + width, :45] = True
            shell = Shell(
                solid.copy(),
                solid,
                np.zeros(3),
                0.02,
                ndimage.distance_transform_edt(solid),
            )
            self.assertEqual(_footprints(shell), [])

    def test_only_the_missing_right_leg_is_transplanted_through_robed_volume(self):
        shell = robed_legs()
        vertices = shell.origin + np.argwhere(shell.surface) * shell.step
        source = incomplete_biped()
        prepared, report = postprocess_skintokens_skeleton(source, vertices, shell)
        additions = [n for n in prepared if n not in source]
        self.assertEqual(len(additions), 3, report)
        self.assertEqual(
            sum(e[0] == "transplant_missing_legs" for e in report["events"]), 1
        )
        root = next(n for n in additions if prepared[n]["parent"] not in additions)
        self.assertEqual(prepared[root]["parent"], "Root")
        self.assertLess(prepared[root]["rest_head_world"][0], -0.05)
        self.assertGreater(prepared[root]["rest_head_world"][2], 0.8)
        end = next(
            n
            for n in additions
            if not any(prepared[c]["parent"] == n for c in additions)
        )
        self.assertLess(
            np.linalg.norm(
                np.array(prepared[end]["rest_tail_world"]) - [-0.2, 0, 0.02]
            ),
            0.06,
        )
        self.assertEqual(prepared["Foot.L"]["parent"], "Shin.L")
        self.assertTrue(
            all(
                prepared[n]["kind"] == "limb"
                for n in additions + ["Thigh.L", "Shin.L", "Foot.L"]
            )
        )

    def test_narrow_ankle_survives_when_sheet_filter_disconnects_the_medial_tree(self):
        shell = robed_legs(False)
        shell.solid[:, :, 30:33] = False
        for x in (30, 50):
            shell.solid[x, 40, 30:33] = True
        shell.distance = ndimage.distance_transform_edt(shell.solid)
        shell.surface = shell.solid & ~ndimage.binary_erosion(shell.solid)
        vertices = shell.origin + np.argwhere(shell.surface) * shell.step
        prepared, report = postprocess_skintokens_skeleton(
            incomplete_biped(), vertices, shell
        )
        additions = [n for n in prepared if n.startswith("__transplant")]
        self.assertEqual(len(additions), 3, report)
        end = next(
            n
            for n in additions
            if not any(prepared[c]["parent"] == n for c in additions)
        )
        np.testing.assert_allclose(
            prepared[end]["rest_tail_world"], [-0.2, 0, 0.02], atol=0.02
        )


if __name__ == "__main__":
    unittest.main()
