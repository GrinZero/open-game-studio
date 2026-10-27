"""Canonical frames and native sanitizer/Heat binding contracts."""

from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from reprojection_solver import aligned_rest_rotation


class RigPreparationTests(unittest.TestCase):
    def test_frames_follow_heading_and_parallel_left_has_a_proper_up_fallback(self):
        head = np.array([0.3, -0.4, 0.8])
        for yaw in (0, np.pi / 2, np.pi, -np.pi / 2):
            E = Rotation.from_rotvec([0, 0, yaw]).as_matrix()
            left = E @ [1.0, 0, 0]
            for direction in (
                [0, 0, 1.0],
                [0, -1, 0.2],
                [1.0, 0, 0],
                [-1.0, 0, 0],
                [1.0, 1e-8, 0],
                [1.0, 1e-4, 0.2],
            ):
                base = aligned_rest_rotation(head, head + direction)
                translated = E @ head + [0.2, -0.1, 0.3]
                frame = aligned_rest_rotation(
                    translated, translated + E @ direction, left
                )
                np.testing.assert_allclose(frame, E @ base, atol=1e-12)
                np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
                self.assertAlmostEqual(np.linalg.det(frame), 1.0, places=12)
                y = E @ direction / np.linalg.norm(direction)
                np.testing.assert_allclose(frame[:, 1], y, atol=1e-12)
                projected = left - y * (left @ y)
                if np.linalg.norm(projected) < 1e-6:
                    np.testing.assert_allclose(frame[:, 0], [0, 0, 1], atol=1e-7)
                else:
                    np.testing.assert_allclose(
                        frame[:, 0], projected / np.linalg.norm(projected), atol=1e-12
                    )

    def native(self, script, marker):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    os.environ["MONSTER_TEST_BLENDER"],
                    "--background",
                    "--factory-startup",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(Path(__file__).with_name(script)),
                    "--",
                    temporary,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout[-16000:])
            self.assertIn(marker, result.stdout)

    @unittest.skipUnless(
        os.environ.get("MONSTER_TEST_BLENDER"),
        "Set MONSTER_TEST_BLENDER for native frame preparation",
    )
    def test_four_heading_hierarchy_normalization_and_actual_bone_frames(self):
        self.native("fixture_v3_rig_axes.py", "CANONICAL_RIG_AXES")

    @unittest.skipUnless(
        os.environ.get("MONSTER_TEST_BLENDER"),
        "Set MONSTER_TEST_BLENDER for native Heat preparation",
    )
    def test_coverage_boundary_real_exterior_heat_and_failure_cleanup(self):
        self.native("fixture_v3_heat.py", "REAL_HEAT_BINDING")


if __name__ == "__main__":
    unittest.main()
