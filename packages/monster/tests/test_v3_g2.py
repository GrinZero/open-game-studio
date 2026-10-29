"""Actual v3 G2 boundary labels, camera panels, and reviewed revision binding."""

from copy import deepcopy
import json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "skills/monster-workflow/scripts"))
import sop_g2_pack as g2
import sop_gate as gate
import sop_delivery
from pose_sampling_v3 import sampling_audit_inputs
from studio_camera import cameras_from_calibration


class ActualG2Tests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MONSTER_TEST_BLENDER"),
        "Set MONSTER_TEST_BLENDER for actual G2 labels",
    )
    def test_actual_stacked_perspective_labels_are_the_reviewed_pose_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.check_native(Path(temporary))

    def check_native(self, base):
        blender = os.environ["MONSTER_TEST_BLENDER"]
        source = base / "source"
        source.mkdir()
        result = subprocess.run(
            [
                blender,
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                str(ROOT / "tests/fixture_v3_blender.py"),
                "--",
                str(source),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                **os.environ,
                "MONSTER_TEST_LAYOUT": "stacked",
                "MONSTER_STUDIO_LAYOUT": "stacked",
                "MONSTER_CAMERA_LENS": "0",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout[-16000:])
        work = base / "work"
        checks = base / "checks"
        work.mkdir()
        (checks / "g2_review").mkdir(parents=True)
        # A cached crop from an earlier view layout must not survive this revision.
        Image.new("RGB", (1, 1), "magenta").save(checks / "g2_review/pose_001_top.png")
        ledger = gate.load_ledger("native-g2", base, base / "drive")
        gate.record_user_decision(
            ledger,
            "G1",
            decision="approved",
            user_quote="TEST FIXTURE ONLY",
            selected_candidate_id=0,
            rig_meta_path=str(source / "studio_rig_meta.json"),
        )
        ledger["gates"]["G1"]["materials"] = {
            "selected_blend_path": str(source / "rig.blend"),
            "selected_candidate_id": "0",
        }
        gate.save_ledger(ledger, base, base / "drive")

        def command(args):
            return [
                blender,
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                str(ROOT / "tests/fixture_v3_g2_worker.py"),
                "--",
                *args,
            ]

        approved_metadata = (source / "studio_rig_meta.json").read_bytes()
        probe = subprocess.run(
            [blender, "--background", "--factory-startup", "--python-exit-code", "1",
             "--python", str(ROOT / "tests/fixture_v3_g2_worker.py"), "--",
             "--mode", "probe_joint_limits", "--blend_path", str(source / "rig.blend"),
             "--meta_path", str(source / "studio_rig_meta.json"),
             "--limits_path", str(work / "joint_limits.json")],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        self.assertEqual(probe.returncode, 0, probe.stdout[-16000:])

        with (
            patch.object(
                g2,
                "resolve_scratch_dirs",
                return_value={"work_dir": work, "checks_dir": checks},
            ),
            patch.object(g2, "_worker_command", side_effect=command),
            patch.object(
                g2,
                "sample_boundary_exploration_pose_euler_dict",
                side_effect=AssertionError("Review must consume actual v3 poses"),
            ),
            patch.object(
                g2,
                "compute_posed_bone_endpoints_fk",
                side_effect=AssertionError("Review must consume actual FK labels"),
            ),
            patch.object(
                g2,
                "_project_ortho_panel",
                side_effect=AssertionError(
                    "Perspective review must consume calibrated labels"
                ),
            ),
            patch.dict(
                os.environ,
                {"MONSTER_STUDIO_LAYOUT": "stacked", "MONSTER_CAMERA_LENS": "50"},
            ),
        ):
            result = g2.build_g2_review_pack(
                "native-g2", project_root=base, drive_root=base / "drive", num_poses=50
            )
        self.assertTrue(result["boundary_exploration_passed"])
        self.assertTrue(result["coverage_passed"])
        output = checks / "g2_review"
        self.assertEqual((source / "studio_rig_meta.json").read_bytes(), approved_metadata)
        self.assertTrue((output / "studio_rig_meta.json").is_file())
        data = json.loads((output / "keypoints_gt.json").read_text())
        self.assertEqual(data["sampling_contract"], "monster-calibrated-sampling-v2")
        self.assertFalse(data["coverage_audit"]["boundary_exploration_audit"]["boundary_coverage_required"])
        self.assertEqual({r["sampling_regime"] for r in data["samples"]}, {"video2rig_calm", "video2rig_active"})
        self.assertTrue(all(r["v3_sampling"]["sampling_policy"] == "video2rig-start-pose-v1" for r in data["samples"]))
        summary = json.loads((output / "g2_pose_summary.json").read_text())
        records = json.loads((output / "pose_records.json").read_text())
        self.assertEqual(len(records), 50)
        self.assertEqual(summary["calibration"], data["training_calibration"])
        self.assertEqual(summary["reader_contract"], "monster-reader-v3")
        cameras = cameras_from_calibration(summary["calibration"])
        self.assertEqual(list(cameras), ["side", "top"])
        self.assertTrue(all(c.type == "persp" for c in cameras.values()))
        for index in (0, 17, 49):
            original = data["samples"][index]
            record = records[index]
            self.assertEqual(
                record["joints_rad"],
                {
                    n: [round(float(x), 5) for x in b["euler_xyz"]]
                    for n, b in original["joints"].items()
                },
            )
            self.assertEqual(record["reader_points"], original["reader_points"])
            for view, camera in cameras.items():
                for n, bone in original["joints"].items():
                    for end in ("head", "tail"):
                        uv = record["uv_labels"][n][f"{view}_uv_{end}"]
                        slot = data["keypoint_names"].index(n + "." + end)
                        np.testing.assert_allclose(
                            uv, original["reader_points"][view]["uv"][slot], atol=1e-10
                        )
                        np.testing.assert_allclose(
                            uv, camera.project(bone[end + "_3d"]), atol=0.02
                        )
        raw = np.asarray(Image.open(output / "pose_001.png"))
        pw, ph = (
            summary["calibration"]["panel_width"],
            summary["calibration"]["panel_height"],
        )
        np.testing.assert_array_equal(
            np.asarray(Image.open(output / "pose_001_top.png")), raw[ph : 2 * ph, :pw]
        )
        ledger = gate.load_ledger("native-g2", base, base / "drive")
        self.assertEqual(ledger["gates"]["G2"]["user_decision"], "pending")
        self.assertEqual(
            len(ledger["gates"]["G2"]["materials"]["g2_review_artifacts"]), 307
        )
        gate.record_user_decision(
            ledger, "G2", decision="approved", user_quote="TEST FIXTURE ONLY", seed=42
        )
        gate.assert_gate(ledger, "G2")
        crop = output / "pose_001_top.png"
        original_crop = crop.read_bytes()
        try:
            crop.write_bytes(b"changed crop")
            with self.assertRaisesRegex(RuntimeError, "G2 审查材料"):
                gate.assert_gate(ledger, "G2")
        finally:
            crop.write_bytes(original_crop)
        self.assertEqual(ledger["gates"]["G1"]["status"], "approved")
        self.assertEqual(ledger["gates"]["G2"]["status"], "invalidated")
        trajectory = json.loads((source / "trajectory.json").read_text())
        trajectory["vertical_trajectory_corr"] = None
        delivery_trajectory = source / "delivery-trajectory.json"
        delivery_trajectory.write_text(json.dumps(trajectory))
        with patch.dict(os.environ, {"MONSTER_BLENDER": blender}):
            delivered = sop_delivery.verify_animated_glb_and_loop_seam(
                source / "review/stage4_exam_reconstructed.glb",
                delivery_trajectory,
                is_loop=False,
                rig_bones=trajectory["bones_meta"],
            )
        self.assertTrue(delivered["valid"])
        self.assertTrue(delivered["blender_roundtrip_verified"])
        self.assertIsNone(delivered["vertical_trajectory_corr"])
        self.assertEqual(delivered['blender_readback']['trajectory_frames_checked'],3)
        self.assertLess(delivered['blender_readback']['pose_max_head_error'],2e-4)
        self.assertAlmostEqual(delivered['blender_readback']['frame_start'],0,places=5)
        self.assertAlmostEqual(delivered['blender_readback']['frame_end'],1.25,places=5)

    def test_sampling_audit_preserves_labels_and_excludes_unfitted_organization(self):
        bones = {
            "Root": {"kind": "root"},
            "Hips": {"kind": "root"},
            "Arm.L": {"kind": "limb"},
        }
        record = {
            "sampling_regime": "fixture",
            "joints": {
                "Root": {"euler_xyz": [0.2, 0, 0]},
                "Hips": {"euler_xyz": [0.1, 0.2, 0.3]},
                "Arm.L": {"euler_xyz": [0.4, 0, 0]},
            },
            "v3_sampling": {
                "fitted_bones": ["Hips", "Arm.L"],
                "motion_roots": ["Hips"],
                "motion_root_world_eulers": {"Hips": [0.7, 0.8, 0.9]},
            },
        }
        before = deepcopy(record)
        a, m = sampling_audit_inputs([record], bones)
        self.assertEqual(record, before)
        self.assertTrue(m["Root"]["fixed_local"])
        self.assertNotIn("fixed_local", bones["Root"])
        self.assertEqual(a[0]["joints"]["Hips"]["euler_xyz"], [0.7, 0.8, 0.9])
        invalid = deepcopy(record)
        invalid["v3_sampling"]["motion_root_world_eulers"] = {}
        with self.assertRaises(ValueError):
            sampling_audit_inputs([invalid], bones)


if __name__ == "__main__":
    unittest.main()
