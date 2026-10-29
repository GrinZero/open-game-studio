import os
from pathlib import Path
import subprocess
import unittest


class PoseMeshGuardTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("MONSTER_TEST_BLENDER"), "Requires native Blender")
    def test_new_rigid_collision_fails_and_original_or_flexible_contacts_remain_distinct(self):
        result = subprocess.run([
            os.environ["MONSTER_TEST_BLENDER"], "--background", "--factory-startup",
            "--python-exit-code", "1", "--python",
            str(Path(__file__).with_name("fixture_pose_mesh_guard.py")),
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.assertEqual(result.returncode, 0, result.stdout[-8000:])
        self.assertIn("MEASURED_RIGID_COLLISIONS_AND_FLEXIBLE_CONTACT_OK", result.stdout)
