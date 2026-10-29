"""Weighted hierarchy roots and saved poses through redundant organization chains."""

from copy import deepcopy
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from reprojection_solver import RigModel, aligned_rest_rotation, fit_bones
from rig_pose import start_pose_matrices


def nested_rig():
    meta = {}
    for i, name in enumerate(("Root", "Org", "Hips", "Spine1", "Rigid1")):
        head = np.array([0.03 * i, 0.02 * i, 0.2 + 0.3 * i])
        tail = head + [0.02, -0.04, 0.25]
        rotation = (
            aligned_rest_rotation(head, tail)
            @ Rotation.from_euler("y", 0.11 * i).as_matrix()
        )
        basis = np.eye(4)
        basis[:3, :3] = Rotation.from_euler(
            "xyz", [0.08 * (i + 1), -0.03 * i, 0.05]
        ).as_matrix()
        if i < 2 or i == 4:
            basis[:3, 3] = [0.03, -0.02, 0.01]
        meta[name] = dict(
            parent=None if i == 0 else list(meta)[-1],
            kind="rigid" if i == 4 else "root" if i < 3 else "spine",
            fixed_local=i == 4,
            rest_head_world=head.tolist(),
            rest_tail_world=tail.tolist(),
            rest_rot_mat_world=rotation.tolist(),
            pose_basis=basis.tolist(),
            limits_rad=[[-1.4, 1.4]] * 3,
        )
    names = sorted(meta)
    skin = dict(
        bone_names=names,
        rest_verts=np.array([[0, 0, 0], [0, 0, 2], [0.1, 0, 1], [-0.1, 0, 1]]),
        skin_indices=np.array(
            [[names.index(n)] for n in ("Hips", "Spine1", "Spine1", "Rigid1")]
        ),
        skin_weights=np.ones((4, 1)),
        height=2.0,
    )
    return meta, skin


class MotionRootTests(unittest.TestCase):
    def test_small_weighted_digits_keep_channels_below_global_mass_cutoff(self):
        meta, skin = nested_rig()
        meta["Finger2_1.L"] = dict(
            parent="Spine1", kind="digit", rest_head_world=[.2, 0, 1.3],
            rest_tail_world=[.3, 0, 1.3], limits_rad=[[-1, 1], [0, 0], [-.2, .2]])
        skin["bone_names"].append("Finger2_1.L")
        # One small but real skin influence, below 0.4% of the full model.
        skin["skin_indices"] = np.vstack([skin["skin_indices"], [[len(skin["bone_names"]) - 1]]])
        skin["skin_weights"] = np.vstack([skin["skin_weights"], [[.001]]])
        skin["rest_verts"] = np.vstack([skin["rest_verts"], [[.3, 0, 1.3]]])
        rig = RigModel(meta, skin=skin)
        self.assertIn("Finger2_1.L", rig.slices)
        self.assertEqual(len(rig.slices["Finger2_1.L"][0]), 2)

    def test_nested_empty_ancestors_have_no_channels_and_hips_has_six(self):
        meta, skin = nested_rig()
        rig = RigModel(meta, skin=skin)
        self.assertEqual(fit_bones(meta, skin), {"Hips", "Spine1"})
        self.assertEqual(rig.motion_roots, {"Hips"})
        self.assertEqual(rig.size, 9)
        self.assertEqual(rig.slices["Hips"].stop - rig.slices["Hips"].start, 6)

    def test_nonzero_frozen_bases_survive_start_pose_and_analytic_derivatives(self):
        meta, skin = nested_rig()
        rig = RigModel(meta, skin=skin)
        params = rig.start_params()
        points, jac, heads, rotations = rig.fk(params, True)
        expected = start_pose_matrices(meta)
        for name in meta:
            np.testing.assert_allclose(heads[name], expected[name][:3, 3], atol=1e-12)
            np.testing.assert_allclose(
                rotations[name], expected[name][:3, :3], atol=1e-12
            )
        epsilon = 1e-6
        numeric = np.stack(
            [
                (rig.fk(params + step)[0] - rig.fk(params - step)[0]) / (2 * epsilon)
                for step in np.eye(rig.size) * epsilon
            ],
            axis=-1,
        )
        np.testing.assert_allclose(jac, numeric, atol=1e-8)
        self.assertTrue(np.isfinite(points).all())
        for name in meta:
            vector, analytic = rig.vector_jacobian(
                params, rotations, name, [0.2, 0.3, 0.1]
            )
            numeric = np.stack(
                [
                    (
                        rig.fk(params + step)[2][name] @ [0.2, 0.3, 0.1]
                        - rig.fk(params - step)[2][name] @ [0.2, 0.3, 0.1]
                    )
                    / (2 * epsilon)
                    for step in np.eye(rig.size) * epsilon
                ],
                axis=-1,
            )
            np.testing.assert_allclose(analytic, numeric, atol=1e-8)
            self.assertTrue(np.isfinite(vector).all())

    def test_common_root_with_weighted_sibling_is_a_valid_ancestor(self):
        meta, skin = nested_rig()
        meta["Spine1"]["parent"] = "Root"
        meta["Org"]["pose_basis"] = np.eye(4).tolist()
        rig = RigModel(meta, skin=skin)
        self.assertEqual(rig.motion_roots, {"Root"})
        self.assertIn("Org", fit_bones(meta, skin))
        params = rig.start_params()
        moved = params.copy()
        moved[rig.slices["Root"].start + 3] += 0.1
        before, after = rig.fk(params)[1], rig.fk(moved)[1]
        for name in meta:
            np.testing.assert_allclose(
                after[name] - before[name], [0.1, 0, 0], atol=1e-12
            )

    def test_even_small_direct_ancestor_weight_prevents_promotion(self):
        meta, skin = nested_rig()
        # This ancestor is now fitted; local animation translations are outside
        # the rotational joint contract, unlike the frozen organization case.
        meta["Org"]["pose_basis"] = np.eye(4).tolist()
        skin["skin_indices"][0, 0] = skin["bone_names"].index("Org")
        skin["skin_weights"][0, 0] = 0.0001
        self.assertEqual(RigModel(meta, skin=skin).motion_roots, {"Root"})

    def test_explicit_fixed_root_is_not_replaced_by_a_free_child(self):
        meta, skin = nested_rig()
        meta["Org"]["pose_basis"] = np.eye(4).tolist()
        meta["Root"]["fixed_local"] = True
        self.assertEqual(RigModel(meta, skin=skin).motion_roots, set())

    def test_fitting_mass_is_invariant_to_common_weight_scale(self):
        meta, skin = nested_rig()
        reference = fit_bones(meta, skin)
        for scale in (0.0001, 0.1, 10):
            scaled = deepcopy(skin)
            scaled["skin_weights"] *= scale
            self.assertEqual(fit_bones(meta, scaled), reference)

    @unittest.skipUnless(
        os.environ.get("MONSTER_TEST_BLENDER"),
        "Set MONSTER_TEST_BLENDER for native sampling/GLB roundtrip",
    )
    def test_native_sampling_and_fractional_frame_glb_roundtrip(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    os.environ["MONSTER_TEST_BLENDER"],
                    "--background",
                    "--factory-startup",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(Path(__file__).with_name("fixture_v3_motion_roots.py")),
                    "--",
                    temporary,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout[-14000:])
            self.assertIn("NESTED_ROOT_ROUNDTRIP", result.stdout)


if __name__ == "__main__":
    unittest.main()
