import copy
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from rig_pose import rigid_pose_basis, start_pose_matrices, start_pose_vertices
from reprojection_solver import RigModel
from v3_verification import verify_v3_numerics


def exported_chain():
    bones = {}
    for i in range(12):
        name = "Root" if i == 0 else f"Spine{i}"
        rotation = Rotation.from_rotvec([0.07 * i, -0.03, 0.11]).as_matrix()
        # Float32 export plus parent-space rebasing can accumulate this drift.
        rotation = (rotation * (1.0 + 1.2e-5)).astype(np.float32)
        basis = np.eye(4, dtype=np.float32)
        basis[:3, :3] = Rotation.from_rotvec([0.03, -0.02, 0.01]).as_matrix()
        basis[:3, :3] *= np.float32(1.0 - 5e-6)
        bones[name] = {
            "parent": None if i == 0 else ("Root" if i == 1 else f"Spine{i-1}"),
            "kind": "root" if i == 0 else "spine",
            "rest_head_world": [0.01 * i, 0.0, 0.1 * i],
            "rest_tail_world": [0.01 * i, 0.0, 0.1 * i + 0.1],
            "rest_rot_mat_world": rotation.tolist(),
            "pose_basis": basis.tolist(),
            "limits_rad": [[-0.5, 0.5]] * 3,
        }
    return bones


class RigidTransformPrecisionTests(unittest.TestCase):
    def test_float32_chain_keeps_exact_bone_lengths_and_so3(self):
        bones = exported_chain()
        original = copy.deepcopy(bones)
        report = verify_v3_numerics(bones)
        self.assertTrue(report["passed"])
        rig = RigModel(bones)
        points, _, rotations = rig.fk(rig.start_params())
        saved = start_pose_matrices(bones)
        for i, name in enumerate(rig.names):
            np.testing.assert_allclose(points[2 * i], saved[name][:3, 3], atol=1e-12)
            np.testing.assert_allclose(rotations[name], saved[name][:3, :3], atol=1e-12)
        self.assertEqual(bones, original)

    def test_basis_translation_and_bind_identity_are_preserved(self):
        basis = np.eye(4)
        basis[:3, :3] = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix() * (1 + 1e-5)
        basis[:3, 3] = [0.2, -0.3, 0.4]
        np.testing.assert_array_equal(rigid_pose_basis(basis)[:3, 3], basis[:3, 3])
        bones = exported_chain()
        for bone in bones.values():
            bone["pose_basis"] = np.eye(4).tolist()
        rest = np.array([[0.2, 0.3, 0.4], [0.4, 0.5, 0.6]])
        skin = {"rest_verts": rest, "bone_names": ["Root"],
                "skin_indices": np.zeros((2, 1), dtype=int),
                "skin_weights": np.ones((2, 1))}
        np.testing.assert_allclose(start_pose_vertices(skin, bones), rest, atol=1e-12)

    def test_material_scale_shear_reflection_and_nonfinite_are_rejected(self):
        for bad in [np.diag([1.02, 1, 1]), np.diag([-1, 1, 1]),
                    np.array([[1, 0.02, 0], [0, 1, 0], [0, 0, 1]]),
                    np.full((3, 3), np.nan)]:
            bones = exported_chain()
            bones["Spine4"]["rest_rot_mat_world"] = bad.tolist()
            with self.assertRaises(ValueError):
                RigModel(bones)
            with self.assertRaises(ValueError):
                start_pose_matrices(bones)


if __name__ == "__main__":
    unittest.main()
