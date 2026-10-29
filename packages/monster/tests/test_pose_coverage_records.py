"""Coverage receipts must measure the rendered nested records, not constants."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from bone_taxonomy import audit_pose_dataset_coverage, audit_boundary_exploration_coverage


class CoverageRecordsTests(unittest.TestCase):
    def fixture(self):
        bones = {"Root": {"parent": None, "kind": "root", "fixed_local": True,
                            "rest_head_world": [0, 0, 0], "rest_tail_world": [0, 0, 1]}}
        for i in range(1, 6):
            bones[f"Finger{i}_1.L"] = {
                "parent": "Root", "kind": "digit", "rest_head_world": [i * .1, 0, 1],
                "rest_tail_world": [i * .1, .2, 1], "limits_rad": [[-1, 1]] * 3,
                "allow_leaf_twist": True, "limits_source": "agent_custom",
            }
        rng = np.random.default_rng(71)
        samples = [{"sampling_regime": "decoupled", "joints": {
            n: {"euler_xyz": rng.uniform(-1, 1, 3).tolist()}
            for n in bones}} for _ in range(200)]
        return bones, samples

    def test_nested_rendered_records_match_arrays_and_measure_actual_metrics(self):
        bones, records = self.fixture()
        nested = audit_pose_dataset_coverage(records, bones)
        arrays = {n: np.array([r["joints"][n]["euler_xyz"] for r in records]) for n in bones}
        plain = audit_pose_dataset_coverage(arrays, bones)
        self.assertTrue(nested["passed"], nested["failures"])
        self.assertEqual(nested["decoupling_metrics"], plain["decoupling_metrics"])
        self.assertNotEqual(nested["decoupling_metrics"]["max_inter_finger_abs_corr"], .15)
        self.assertTrue(audit_boundary_exploration_coverage(records, bones)["passed"])

    def test_midrange_samples_do_not_claim_boundary_coverage(self):
        bones, records = self.fixture()
        for rec in records:
            for point in rec["joints"].values():
                point["euler_xyz"] = [0, 0, 0]
        report = audit_boundary_exploration_coverage(records, bones)
        self.assertFalse(report["passed"])
        self.assertEqual(report["dual_boundary_touch_rate"], 0)

    def test_locked_fingers_are_detected_from_nested_records(self):
        bones, records = self.fixture()
        for i, rec in enumerate(records):
            for n in bones:
                rec["joints"][n]["euler_xyz"] = [2 * i / 199 - 1, 0, 0]
        report = audit_pose_dataset_coverage(records, bones)
        self.assertFalse(report["passed"])
        self.assertGreater(report["decoupling_metrics"]["digit_pc1_variance_ratio"], .99)


if __name__ == "__main__":
    unittest.main()
