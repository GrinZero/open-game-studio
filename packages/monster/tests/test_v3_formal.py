"""V3 formal gates must check world roots/p0 using the current FK and pixel solver."""

from copy import deepcopy
from pathlib import Path
import sys, unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "skills/monster-workflow/scripts"))
from test_v3_motion_roots import nested_rig
from test_v3_reader_boundaries import point_rig
from reprojection_solver import RigModel
from bone_taxonomy import classify_rig_bones
from sop_formal_gate import verify_rig_formal_gate, verify_trajectory_formal_gate


class V3FormalTests(unittest.TestCase):
    def test_v3_delivery_requires_nonempty_poses_and_actual_blender_readback(self):
        import json, struct, tempfile
        import sop_delivery

        meta, _, _, frames, context = self.posed()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            raw = json.dumps(
                {"asset": {"version": "2.0"}, "skins": [{}], "animations": [{}]}
            ).encode()
            raw += b" " * ((-len(raw)) % 4)
            glb = base / "fixture.glb"
            glb.write_bytes(
                struct.pack("<4sII", b"glTF", 2, 20 + len(raw))
                + struct.pack("<I4s", len(raw), b"JSON")
                + raw
            )
            with self.assertRaisesRegex(ValueError, "actual reconstructed trajectory"):
                sop_delivery.verify_animated_glb_and_loop_seam(
                    glb, None, is_loop=False, rig_bones=meta
                )
            trajectory = base / "trajectory.json"
            trajectory.write_text(
                json.dumps({**context, "frames": [], "gate_passed": True})
            )
            with self.assertRaisesRegex(ValueError, "nonempty v3 poses"):
                sop_delivery.verify_animated_glb_and_loop_seam(
                    glb, trajectory, is_loop=False, rig_bones=meta
                )
            trajectory.write_text(
                json.dumps(
                    {
                        **context,
                        "frames": frames,
                        "bones_meta": meta,
                        "gate_passed": True,
                        "vertical_trajectory_corr": None,
                    }
                )
            )
            with patch.object(sop_delivery, "_worker_command", return_value=None):
                with self.assertRaisesRegex(
                    RuntimeError, "actual Blender GLB verification"
                ):
                    sop_delivery.verify_animated_glb_and_loop_seam(
                        glb, trajectory, is_loop=False, rig_bones=meta
                    )

    def posed(self):
        meta, skin = nested_rig()
        rename={'Org':'Spine2'}
        meta={rename.get(n,n):{**b,'parent':rename.get(b['parent'],b['parent'])} for n,b in meta.items()}
        skin['bone_names']=[rename.get(n,n) for n in skin['bone_names']]
        for n, b in meta.items():
            b["anatomical_role"] = n
        rig = RigModel(meta, skin=skin)
        frames = rig.frames(
            np.array([rig.start_params()]), np.ones((1, 2, len(rig.point_names)))
        )
        context = {
            "reader_contract": "monster-reader-v3",
            "motion_structure": {
                "fitted_bones": ["Hips", "Spine1"],
                "motion_roots": ["Hips"],
            },
        }
        return meta, skin, rig, frames, context

    def test_weighted_frozen_p0_uses_v3_solver_and_jacobian_instead_of_legacy_neutral(
        self,
    ):
        meta, skin, _, frames, context = self.posed()
        with patch(
            "bone_taxonomy.reconstruct_frames_with_kind_solvers",
            side_effect=AssertionError("V3 must not use legacy hierarchy rescue"),
        ):
            report = verify_rig_formal_gate(meta, rig_skinning=skin, num_random_poses=4)
        self.assertTrue(report["passed"])
        self.assertEqual(report["motion_structure"], context["motion_structure"])
        self.assertLess(report["v3_fk_camera"]["fk_jacobian_max_error"], 1e-6)
        self.assertLess(
            report["p2_oracle_completeness"]["neutral_hierarchy_max_mm"], 1e-5
        )
        self.assertTrue(
            verify_trajectory_formal_gate(frames, meta, trajectory_meta=context)[
                "passed"
            ]
        )

    def test_fixed_root_basis_missing_pose_nan_tail_frame_and_limits_are_checked(self):
        meta, _, rig, frames, context = self.posed()
        variants = []
        variants.append(([], "P1_EMPTY_TRAJECTORY"))
        f = deepcopy(frames)
        f[0]["bones"]["Root"]["rot_mat_world"] = np.eye(3).tolist()
        variants.append((f, "P1_FIXED_LOCAL"))
        f = deepcopy(frames)
        f[0]["bones"]["Root"]["head_3d"][0] += 0.01
        variants.append((f, "P1_HEAD_OFFSET"))
        f = deepcopy(frames)
        del f[0]["bones"]["Rigid1"]
        variants.append((f, "P1_MISSING_POSE"))
        f = deepcopy(frames)
        f[0]["bones"]["Spine1"]["head_3d"][0] = float("nan")
        variants.append((f, "P1_INVALID_POSE"))
        f = deepcopy(frames)
        b = f[0]["bones"]["Spine1"]
        v = np.array(b["tail_3d"]) - b["head_3d"]
        b["tail_3d"] = (
            np.array(b["head_3d"]) + Rotation.from_euler("z", 0.3).apply(v)
        ).tolist()
        variants.append((f, "P1_TAIL_FRAME"))
        f = deepcopy(frames)
        b = f[0]["bones"]["Spine1"]
        parent = np.array(f[0]["bones"]["Hips"]["rot_mat_world"])
        base = parent @ rig.R["Hips"].T @ rig.R["Spine1"]
        r = base @ Rotation.from_euler("x", 2.0).as_matrix()
        b["rot_mat_world"] = r.tolist()
        b["tail_3d"] = (
            np.array(b["head_3d"])
            + r @ rig.R["Spine1"].T @ (rig.t["Spine1"] - rig.h["Spine1"])
        ).tolist()
        variants.append((f, "P1_LIMIT_BOX"))
        for f, kind in variants:
            with self.subTest(kind=kind):
                report = verify_trajectory_formal_gate(
                    f, meta, trajectory_meta=context, assert_passed=False
                )
                self.assertFalse(report["passed"])
                self.assertTrue(
                    any(v.startswith(kind) for v in report["violations"]), report
                )

    def test_independent_prop_world_pose_is_free_and_structure_cannot_promote_a_limb(
        self,
    ):
        base = point_rig()
        rig = RigModel(classify_rig_bones(base.meta), extra_points=base.extra)
        p = rig.start_params()
        s = rig.slices["Prop_tool"]
        p[s.start : s.start + 3] = [0.6, -0.4, 0.9]
        p[s.start + 3 : s.stop] = [0.5, 0.2, -0.1]
        frames = rig.frames(np.array([p]), np.ones((1, 2, len(rig.point_names))))
        context = {
            "reader_contract": "monster-reader-v3",
            "motion_structure": {
                "fitted_bones": list(rig.slices),
                "motion_roots": sorted(rig.motion_roots),
            },
        }
        self.assertTrue(
            verify_trajectory_formal_gate(frames, rig.meta, trajectory_meta=context)[
                "passed"
            ]
        )
        invalid = deepcopy(context)
        invalid["motion_structure"]["motion_roots"].append("Hand.L")
        with self.assertRaises(ValueError):
            verify_trajectory_formal_gate(frames, rig.meta, trajectory_meta=invalid)


if __name__ == "__main__":
    unittest.main()
