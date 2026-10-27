"""Start-pose and collision requirements that identity-only fixtures cannot exercise."""

from copy import deepcopy
from pathlib import Path
import sys
import os
import tempfile
import subprocess
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from reprojection_solver import RigModel, aligned_rest_rotation


def posed_rig():
    meta = {
        "Root": {
            "parent": None,
            "kind": "root",
            "rest_head_world": [0, 0, 0.2],
            "rest_tail_world": [0, 0, 1.2],
        },
        "Socket.L": {
            "parent": "Root",
            "kind": "limb",
            "rest_head_world": [-1, 0.06, 0.7],
            "rest_tail_world": [-1, 0.06, 0.9],
            "limits_rad": [[-1.4, 1.4]] * 3,
        },
        "Arm.L": {
            "parent": "Socket.L",
            "kind": "limb",
            "rest_head_world": [-1, 0.06, 0.7],
            "rest_tail_world": [1, 0.06, 0.7],
            "limits_rad": [[-1.4, 1.4]] * 3,
        },
        "Rigid1.L": {
            "parent": "Arm.L",
            "kind": "rigid",
            "fixed_local": True,
            "rest_head_world": [1, 0.06, 0.7],
            "rest_tail_world": [1, 0.3, 0.7],
        },
    }
    for n, b in meta.items():
        b["rest_rot_mat_world"] = aligned_rest_rotation(
            np.asarray(b["rest_head_world"]), np.asarray(b["rest_tail_world"])
        ).tolist()
        B = np.eye(4)
        if n == "Root":
            B[:3, :3] = Rotation.from_euler("xyz", [0.15, -0.1, 0.2]).as_matrix()
            B[:3, 3] = [0.08, -0.03, 0.04]
        if n == "Arm.L":
            B[:3, :3] = Rotation.from_euler("xyz", [0.3, -0.2, 0.1]).as_matrix()
        b["pose_basis"] = B.tolist()
    return meta


class StartPoseTests(unittest.TestCase):
    def test_start_params_preserve_parent_relative_pose_and_root_translation(self):
        meta = posed_rig()
        rig = RigModel(meta)
        points, heads, rots = rig.fk(rig.start_params())
        transforms = {}
        for n in rig.order:
            rest = np.eye(4)
            rest[:3, :3] = rig.R[n]
            rest[:3, 3] = rig.h[n]
            parent = rig.parents[n]
            if parent:
                parent_rest = np.eye(4)
                parent_rest[:3, :3] = rig.R[parent]
                parent_rest[:3, 3] = rig.h[parent]
                posed = (
                    transforms[parent]
                    @ np.linalg.inv(parent_rest)
                    @ rest
                    @ meta[n]["pose_basis"]
                )
            else:
                posed = rest @ meta[n]["pose_basis"]
            transforms[n] = posed
            np.testing.assert_allclose(heads[n], posed[:3, 3], atol=1e-12)
            np.testing.assert_allclose(rots[n], posed[:3, :3], atol=1e-12)
        self.assertGreater(np.linalg.norm(points - rig.fk(rig.zero())[0]), 0.1)

    def test_dense_reference_height_is_the_exported_mesh_height(self):
        meta = posed_rig()
        names = sorted(meta)
        skin = {
            "bone_names": names,
            "rest_verts": np.array([[0, 0, 0], [0, 0, 4], [0.2, 0, 0], [0.2, 0, 4]]),
            "skin_indices": np.full((4, 1), names.index("Root")),
            "skin_weights": np.ones((4, 1)),
            "height": 4.0,
        }
        rig = RigModel(meta, names, skin)
        self.assertEqual(rig.height, 4.0)

    def test_prop_read_start_uses_off_axis_points_for_full_rigid_orientation(self):
        meta = {
            "Prop_tool": {
                "parent": None,
                "kind": "root",
                "rest_head_world": [0, 0, 1],
                "rest_tail_world": [0, 1, 1],
                "rest_rot_mat_world": np.eye(3).tolist(),
            }
        }
        rig = RigModel(
            meta,
            extra_points=[
                {
                    "name": "Prop_tool.sidea1",
                    "bone": "Prop_tool",
                    "local": [0.15, 0.5, 0],
                },
                {
                    "name": "Prop_tool.sideb1",
                    "bone": "Prop_tool",
                    "local": [0, 0.5, 0.12],
                },
            ],
        )
        params = np.array([0.4, -0.2, 0.7, 0.15, -0.08, 0.03])
        target, _, rotations = rig.fk(params)
        recovered = rig.read_start(target)
        np.testing.assert_allclose(rig.fk(recovered)[0], target, atol=1e-12)
        np.testing.assert_allclose(
            rig.fk(recovered)[2]["Prop_tool"], rotations["Prop_tool"], atol=1e-12
        )

    def test_bone_midpoint_penetration_has_analytic_guard_jacobian(self):
        meta = posed_rig()
        for b in meta.values():
            b["pose_basis"] = np.eye(4).tolist()
        rig = RigModel(meta)
        rig.capsules = [("Root", 0.2)]
        params = rig.zero()
        params[5] = 0.3
        X, J, _, _ = rig.fk(params, True)
        residual, jac = rig.guard(X, J)
        self.assertGreater(
            residual.max(),
            1.0,
            "The segment crosses the body even though both endpoints are clear",
        )
        epsilon = 1e-6

        def evaluate(p):
            X, J, _, _ = rig.fk(p, True)
            return rig.guard(X, J)[0]

        numeric = np.stack(
            [
                (
                    evaluate(params + np.eye(rig.size)[i] * epsilon)
                    - evaluate(params - np.eye(rig.size)[i] * epsilon)
                )
                / (2 * epsilon)
                for i in range(rig.size)
            ],
            -1,
        )
        np.testing.assert_allclose(jac, numeric, atol=1e-6)


class MirrorOverlapTests(unittest.TestCase):
    def test_overlap_counts_unmatched_and_out_of_grid_reflections(self):
        from skeleton_postprocess import Shell
        from scipy.ndimage import distance_transform_edt

        solid = np.zeros((5, 5, 5), bool)
        solid[1, 2, 2] = True
        solid[3, 2, 2] = True
        shell = Shell(
            solid.copy(),
            solid,
            np.array([-2.0, -2.0, -2.0]),
            1.0,
            distance_transform_edt(solid),
        )
        self.assertEqual(shell.mirror_overlap(0, 0), 1.0)
        solid[3, 3, 2] = True
        self.assertEqual(shell.mirror_overlap(0, 0), 0.5)
        # Reflection outside the grid still belongs to the union, rather than disappearing.
        self.assertEqual(shell.mirror_overlap(0, -2), 0.0)
        threshold = np.zeros((13, 4, 3), bool)
        for offset in range(1, 5):
            threshold[6 - offset, 0, 1] = True
            threshold[6 + offset, 0, 1] = True
        threshold[11, 1, 1] = True
        boundary = Shell(
            threshold.copy(),
            threshold,
            np.zeros(3),
            1.0,
            distance_transform_edt(threshold),
        )
        self.assertAlmostEqual(boundary.mirror_overlap(0, 6), 0.80)
        threshold[11, 2, 1] = True
        self.assertLess(boundary.mirror_overlap(0, 6), 0.80)

    def test_head_and_foot_facing_rotates_through_all_four_horizontal_axes(self):
        from skeleton_postprocess import find_facing

        vertices = []
        faces = []
        tris = np.array(
            [
                [0, 1, 3],
                [0, 3, 2],
                [4, 6, 7],
                [4, 7, 5],
                [0, 4, 5],
                [0, 5, 1],
                [2, 3, 7],
                [2, 7, 6],
                [0, 2, 6],
                [0, 6, 4],
                [1, 5, 7],
                [1, 7, 3],
            ]
        )
        for center, size in (
            ((0, 0, 1), (0.5, 0.4, 1.2)),
            ((0, -0.15, 1.8), (0.35, 0.5, 0.4)),
            ((-0.2, -0.2, 0.06), (0.2, 0.6, 0.12)),
            ((0.2, -0.2, 0.06), (0.2, 0.6, 0.12)),
        ):
            corner = (
                np.array(
                    [
                        [x, y, z]
                        for x in (-0.5, 0.5)
                        for y in (-0.5, 0.5)
                        for z in (-0.5, 0.5)
                    ]
                )
                * size
                + center
            )
            faces.extend(tris + len(vertices))
            vertices.extend(corner)
        for angle, expected in ((0, "-Y"), (90, "+X"), (180, "+Y"), (270, "-X")):
            rotated = (
                np.asarray(vertices)
                @ Rotation.from_euler("z", angle, degrees=True).as_matrix().T
            )
            result = find_facing(rotated, triangles=np.asarray(faces))
            self.assertEqual(result["forward"], expected)
            # A one-percent voxel grid quantizes triangle edges; the Plan's
            # acceptance threshold is .80, not exact continuous-surface symmetry.
            self.assertGreater(result["symmetry"], 0.90)


class BlenderStartPoseTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MONSTER_TEST_BLENDER"),
        "Set MONSTER_TEST_BLENDER for posed split-mesh regression",
    )
    def test_posed_source_survives_rebind_export_and_first_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "blender.log"
            with log.open("w") as output:
                result = subprocess.run(
                    [
                        os.environ["MONSTER_TEST_BLENDER"],
                        "--background",
                        "--factory-startup",
                        "--python-exit-code",
                        "1",
                        "--python",
                        str(Path(__file__).with_name("fixture_v3_posed.py")),
                        "--",
                        temporary,
                    ],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
            self.assertEqual(result.returncode, 0, log.read_text()[-12000:])
            self.assertIn("POSED_START_PRESERVED", log.read_text())


if __name__ == "__main__":
    unittest.main()
