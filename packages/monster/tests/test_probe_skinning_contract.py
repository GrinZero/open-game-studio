import hashlib
import sys
import tempfile
import unittest
from unittest.mock import patch
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "skills/monster-workflow/scripts"))
from rig_skinning_artifact import load_probe_skinning
from reprojection_solver import RigModel, aligned_rest_rotation, surface_keypoints
import sop_joint_limits


class ProbeSkinningContractTests(unittest.TestCase):
    def test_bone_only_limit_edit_preserves_v3_root_envelope(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            limits, meta, blend = base / "joint_limits.json", base / "meta.json", base / "rig.blend"
            profile = {"bones": {"Root": {"kind": "root"}},
                       "root_motion": {"translation_fraction_xyz": [.08, .08, .1]},
                       "global_root": {"rotation_limits_deg": [[-75, 75], [-180, 180], [-75, 75]]}}
            limits.write_text(json.dumps(profile))
            meta.write_text('{"bones": {}}')
            blend.write_bytes(b"TEST FIXTURE")
            with (patch.object(sop_joint_limits, "load_ledger", return_value={}),
                  patch.object(sop_joint_limits, "assert_gate", return_value={}),
                  patch.object(sop_joint_limits, "_resolve_g1_candidate_files", return_value=("0", meta, blend)),
                  patch.object(sop_joint_limits, "resolve_joint_limits_path", return_value=limits),
                  patch.object(sop_joint_limits, "derive_invalidations", return_value=[]),
                  patch.object(sop_joint_limits, "save_ledger"),
                  patch.object(sop_joint_limits, "verify_rig_formal_gate", return_value={"passed": True}),
                  patch.object(sop_joint_limits, "resolve_scratch_dirs", return_value={"checks_dir": base})):
                sop_joint_limits.apply_custom_joint_limits("fixture", project_root=base,
                    set_bone_specs=["Root=0:0,0:0,0:0"])
            actual = json.loads(limits.read_text())
            self.assertEqual(actual["global_root"], profile["global_root"] | {
                "translation_fraction_xyz": [.08, .08, .1]})

    def test_measured_weights_are_bound_to_the_approved_blend(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            blend, artifact = base / "rig.blend", base / "skin.npz"
            blend.write_bytes(b"approved rig fixture")
            np.savez(artifact, bone_names=np.array(["Root"]), rest_heads=np.zeros((1, 3)),
                     rest_verts=np.array([[0, 0, 0]]), skin_indices=np.array([[0]]),
                     skin_weights=np.array([[1.0]]))
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            profile = {"rig_skinning": {"path": str(artifact), "sha256": digest(artifact),
                                        "blend_sha256": digest(blend)}}
            bones = {"Root": {"rest_head_world": [0, 0, 0]}}
            skin = load_probe_skinning(profile, blend, bones)
            np.testing.assert_array_equal(skin["skin_weights"], [[1]])
            blend.write_bytes(b"another rig")
            with self.assertRaisesRegex(ValueError, "another approved blend"):
                load_probe_skinning(profile, blend, bones)
            blend.write_bytes(b"approved rig fixture")
            with self.assertRaisesRegex(ValueError, "joint origins"):
                load_probe_skinning(profile, blend, {"Root": {"rest_head_world": [1, 0, 0]}})
            artifact.write_bytes(b"corrupt archive")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_probe_skinning(profile, blend, bones)

    def test_foot_quantile_labels_cannot_create_phantom_floor_penetration(self):
        points = np.array([
            [.115091, -.273549, 0], [.048711, -.106555, .032352],
            [-.104998, -.188899, .001705], [.002357, -.150207, .063984],
            [-.171433, -.199355, .083526], [.080035, -.109774, .009792],
            [.010690, -.129522, .120842], [.081859, -.114866, .084795],
        ])
        bone = {"parent": None, "kind": "root", "rest_head_world": [0, 0, .2],
                "rest_tail_world": [0, -.25, .2]}
        R = aligned_rest_rotation(bone["rest_head_world"], bone["rest_tail_world"])
        bone["rest_rot_mat_world"] = R.tolist()
        bones = {"Foot.L": bone}
        skin = {"rest_verts": points, "bone_names": ["Foot.L"],
                "skin_indices": np.zeros((8, 1), dtype=int), "skin_weights": np.ones((8, 1))}
        phantom = surface_keypoints(bones, ["Foot.L"], skin)
        projected = np.array([bone["rest_head_world"] + R @ p["local"] for p in phantom])
        self.assertLess(projected[:, 2].min(), -.01)
        rig = RigModel(bones, skin=skin)
        X, J, _, _ = rig.fk(rig.start_params(), jac=True)
        residual, _ = rig.guard(X, J)
        self.assertGreaterEqual(X[:, 2].min(), -1e-12)
        self.assertLess(residual.max(), 1e-9)


if __name__ == "__main__":
    unittest.main()
