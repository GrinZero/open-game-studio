"""G3 diagnostic visibility and decisions bound to actual review artifacts."""

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "skills/monster-workflow/scripts"))
sys.path.insert(0, str(ROOT / "worker"))
import sop_gate as gate
import sop_g3_exam as exam
import sop_preflight
from reprojection_solver import RigModel, aligned_rest_rotation
from reading_pipeline import at_limit


def measured_prediction():
    meta = {
        "Root": dict(
            parent=None,
            kind="root",
            rest_head_world=[0, 0, 0.5],
            rest_tail_world=[0, 0, 1],
        ),
        "Arm.L": dict(
            parent="Root",
            kind="limb",
            rest_head_world=[0, 0, 1],
            rest_tail_world=[0.5, 0, 1],
            limits_rad=[[-1, 1]] * 3,
        ),
    }
    for b in meta.values():
        b["rest_rot_mat_world"] = aligned_rest_rotation(
            np.array(b["rest_head_world"]), np.array(b["rest_tail_world"])
        ).tolist()
    rig = RigModel(meta)
    params = np.tile(rig.start_params(), (100, 1))
    axis = rig.slices["Arm.L"][0][0]
    params[:26, axis] = rig.upper[axis] - np.deg2rad(2)
    warnings = at_limit(rig, params)
    frames = []
    for i, p in enumerate(params):
        points, _, heads, rotations = rig.fk(p, True)
        frames.append(
            {
                "frame": i,
                "bones": {
                    n: {
                        "head_3d": heads[n].tolist(),
                        "tail_3d": points[2 * rig.names.index(n) + 1].tolist(),
                        "rot_mat_world": rotations[n].tolist(),
                        "confidence": 0.95,
                    }
                    for n in rig.names
                },
            }
        )
    return meta, {
        "reader_contract": "monster-reader-v3",
        "gate_passed": True,
        "calibration": {},
        "frames": frames,
        "taxonomy_diagnostics": {"at_limit": warnings},
        "mean_epipolar_diff_px": 0,
    }


class ReviewBindingTests(unittest.TestCase):
    def test_safe_shutdown_consumes_current_v3_review_fingerprints(self):
        import sop_colab_guard

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ledger, result = self.prepare(base)
            gate.record_user_decision(
                ledger, "G3", decision="approved", user_quote="TEST FIXTURE ONLY"
            )
            gate.mark_awaiting_review(
                ledger, "final_batch", review_pack_path=result["review_html"]
            )
            gate.record_user_decision(
                ledger,
                "final_batch",
                decision="approved",
                user_quote="TEST FIXTURE ONLY",
            )
            gate.save_ledger(ledger, base, base / "drive")
            Path(result["exam_report_json"]).write_text("{}")
            with patch.dict(os.environ, {"MONSTER_SKIP_PS_SCAN": "1"}):
                check = sop_colab_guard.evaluate_safe_shutdown(
                    "review-fixture",
                    "final_batch_approved",
                    project_root=base,
                    drive_root=base / "drive",
                    worker_status={"status": "IDLE"},
                )
            self.assertFalse(check["allowed"])
            self.assertTrue(any("G3 审查材料" in value for value in check["blockers"]))
            saved = gate.load_ledger("review-fixture", base, base / "drive")
            self.assertEqual(saved["gates"]["G1"]["status"], "approved")
            self.assertEqual(saved["gates"]["G2"]["status"], "approved")
            self.assertEqual(saved["gates"]["G3"]["status"], "invalidated")

    def prepare(self, base):
        meta, prediction = measured_prediction()
        work, checks = base / "work", base / "checks"
        candidates = work / "g1_candidates"
        candidates.mkdir(parents=True)
        meta_file = candidates / "candidate_0_rig_meta.json"
        meta_file.write_text(
            json.dumps({"bones": meta, "reader_contract": "monster-reader-v3"})
        )
        blend = candidates / "candidate_0_full.blend"
        blend.write_bytes(b"disposable renderer input")
        source = base / "source.mp4"
        source.write_bytes(b"disposable source video")
        trajectory = base / "trajectory.json"
        trajectory.write_text(json.dumps(prediction))
        g3 = checks / "g3_review"
        g3.mkdir(parents=True)
        # Media generation is already covered by native Blender tests. This fixture
        # substitutes only its outputs to isolate report/ledger behavior.
        for name in (
            "stage4_exam_reconstructed.glb",
            "g3_comparison_sync.mp4",
            "g3_three_quarter.mp4",
            "g3_silhouette_overlay.mp4",
        ):
            (g3 / name).write_bytes(("disposable " + name).encode())
        (g3 / "glb_readback.json").write_text(
            json.dumps({"passed": True, "fixture": "review binding only"})
        )
        (g3 / "sync_frames").mkdir()
        (g3 / "sync_frames/sync_frame_0000.png").write_bytes(b"not decoded")
        ledger = gate.load_ledger("review-fixture", base, base / "drive")
        gate.record_user_decision(
            ledger,
            "G1",
            decision="approved",
            user_quote="TEST FIXTURE ONLY",
            selected_candidate_id=0,
            rig_meta_path=str(meta_file),
        )
        gate.record_user_decision(
            ledger, "G2", decision="approved", user_quote="TEST FIXTURE ONLY", seed=42
        )
        gate.save_ledger(ledger, base, base / "drive")
        with (
            patch.object(
                exam,
                "resolve_scratch_dirs",
                return_value={"work_dir": work, "checks_dir": checks},
            ),
            patch.object(
                sop_preflight,
                "_worker_command",
                return_value=["disposable-test-renderer"],
            ),
            patch.object(exam.subprocess, "run"),
        ):
            result = exam.build_g3_exam_pack(
                "review-fixture",
                project_root=base,
                drive_root=base / "drive",
                exam_video=source,
                pred_traj_json=trajectory,
                sync_u2=False,
            )
        return gate.load_ledger("review-fixture", base, base / "drive"), result

    def test_measured_limit_warning_reaches_report_html_and_pending_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger, result = self.prepare(Path(tmp))
            report = json.loads(Path(result["exam_report_json"]).read_text())
            self.assertEqual(
                report["joint_limit_diagnostics"],
                {
                    "available": True,
                    "near_limit_degrees": 2,
                    "warning_share_above": 0.25,
                    "warnings": [{"bone": "Arm.L", "share": 0.26}],
                },
            )
            text = Path(result["review_html"]).read_text()
            self.assertIn("关节极限卡住诊断", text)
            self.assertIn("Arm.L", text)
            self.assertIn("26.0%", text)
            info = ledger["gates"]["G3"]
            self.assertEqual(info["user_decision"], "pending")
            self.assertEqual(
                info["bound_fingerprints"]["g3_review_sha256"],
                gate.sha256_json(info["materials"]["g3_review_artifacts"]),
            )
            self.assertEqual(len(info["materials"]["g3_review_artifacts"]), 8)
            # A warning is review evidence, not an automatic fabricated approval or rejection.
            gate.record_user_decision(
                ledger, "G3", decision="approved", user_quote="TEST FIXTURE ONLY"
            )
            self.assertEqual(
                gate.assert_gate(ledger, "G3")["user_decision"], "approved"
            )

    def test_artifact_edit_or_deletion_invalidates_g3_and_final_preserving_upstream(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            ledger, _ = self.prepare(Path(tmp))
            gate.record_user_decision(
                ledger, "G3", decision="approved", user_quote="TEST FIXTURE ONLY"
            )
            gate.mark_awaiting_review(
                ledger,
                "final_batch",
                review_pack_path=ledger["gates"]["G3"]["review_pack_path"],
            )
            gate.record_user_decision(
                ledger,
                "final_batch",
                decision="approved",
                user_quote="TEST FIXTURE ONLY",
            )
            for field, item in ledger["gates"]["G3"]["materials"][
                "g3_review_artifacts"
            ].items():
                for delete in (False, True):
                    with self.subTest(field=field, delete=delete):
                        current = deepcopy(ledger)
                        path = Path(item["path"])
                        original = path.read_bytes()
                        try:
                            if delete:
                                path.unlink()
                            else:
                                path.write_bytes(original + b"changed")
                            with self.assertRaisesRegex(RuntimeError, "G3 审查材料"):
                                gate.assert_gate(current, "final_batch")
                            self.assertEqual(
                                current["gates"]["G3"]["status"], "invalidated"
                            )
                            self.assertEqual(
                                current["gates"]["final_batch"]["status"], "invalidated"
                            )
                            self.assertEqual(
                                current["gates"]["G1"]["status"], "approved"
                            )
                            self.assertEqual(
                                current["gates"]["G2"]["status"], "approved"
                            )
                        finally:
                            path.write_bytes(original)
            current = deepcopy(ledger)
            new_artifacts = deepcopy(
                current["gates"]["G3"]["materials"]["g3_review_artifacts"]
            )
            new_artifacts["exam_report_json"]["sha256"] = "new-result"
            result = gate.derive_invalidations(
                current,
                new_fingerprints={"g3_review_sha256": gate.sha256_json(new_artifacts)},
            )
            self.assertEqual({v["gate"] for v in result}, {"G3", "final_batch"})

    def test_changed_report_cannot_be_approved_and_unknown_is_not_reported_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger, result = self.prepare(Path(tmp))
            Path(result["exam_report_json"]).write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "G3 审查材料"):
                gate.record_user_decision(
                    ledger, "G3", decision="approved", user_quote="TEST FIXTURE ONLY"
                )
            self.assertEqual(ledger["gates"]["G3"]["user_decision"], "pending")
            report = json.loads(
                (Path(tmp) / "checks/g3_review/glb_readback.json").read_text()
            )
            self.assertTrue(report["passed"])
            limits = exam.joint_limit_review({})
            self.assertFalse(limits["available"])
            self.assertIsNone(limits["warnings"])
            clear = exam.joint_limit_review({"taxonomy_diagnostics": {"at_limit": []}})
            self.assertTrue(clear["available"])
            self.assertEqual(clear["warnings"], [])
            for warning in (
                {"bone": "Arm.L", "share": 0.25},
                {"bone": "Arm.L", "share": float("nan")},
                {"bone": "", "share": 0.26},
            ):
                with self.assertRaises(ValueError):
                    exam.joint_limit_review(
                        {"taxonomy_diagnostics": {"at_limit": [warning]}}
                    )


if __name__ == "__main__":
    unittest.main()
