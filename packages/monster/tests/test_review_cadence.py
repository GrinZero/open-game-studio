"""Content cadence must preserve holds and small articulated motions."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from reading_pipeline import cadence_from_flow_steps, detect_video_cadence_optical_flow


class CadenceReviewTests(unittest.TestCase):
    def test_low_motion_is_not_evidence_of_a_missing_time_interval(self):
        report = cadence_from_flow_steps([0.0, 0.001, 0.049, 0.0], fps=30)
        np.testing.assert_array_equal(report["frame_times"], np.arange(5))
        self.assertIsNone(report["period"])
        self.assertEqual(report["cadence"], 1.0)

    def test_hold_between_movements_preserves_its_entire_duration(self):
        report = cadence_from_flow_steps([1.0] * 8 + [0.0] * 24 + [1.0] * 8)
        np.testing.assert_array_equal(report["frame_times"], np.arange(41))
        self.assertEqual(report["frame_times"][32] - report["frame_times"][8], 24)

    def test_stationary_subject_and_empty_background_preserve_encoded_time(self):
        for foreground in (False, True):
            with self.subTest(foreground=foreground):
                image = np.full((192, 320, 3), 230, dtype=np.uint8)
                if foreground:
                    image[40:160, 85:195] = 90
                report = detect_video_cadence_optical_flow([image.copy() for _ in range(12)])
                np.testing.assert_array_equal(report["frame_times"], np.arange(12))

    def test_small_moving_part_beside_stationary_trunk_keeps_every_frame(self):
        rng = np.random.default_rng(2)
        trunk = rng.integers(60, 160, (120, 110, 3), dtype=np.uint8)
        images = []
        for t in range(20):
            image = np.full((192, 320, 3), 230, dtype=np.uint8)
            image[40:160, 85:195] = trunk
            image[65:80, 205 + 3 * t : 220 + 3 * t] = [10, 30, 90]
            images.append(image)
        self.assertEqual(len({image.tobytes() for image in images}), 20)
        report = detect_video_cadence_optical_flow(images)
        self.assertEqual(len(set(report["frame_times"])), 20)
        np.testing.assert_allclose(report["frame_times"], np.arange(20))
        self.assertGreater(float(np.mean(report["flow_speed"])), 0.01)

    def test_periodic_four_and_twenty_four_step_drop_cadences_still_detected(self):
        for period, content_fps in ((4, 30.0), (24, 25.0)):
            with self.subTest(period=period):
                speeds = np.ones(period * 6)
                speeds[period - 1 :: period] = 2
                report = cadence_from_flow_steps(speeds)
                self.assertEqual(report["period"], period)
                self.assertEqual(report["phase"], period - 1)
                self.assertEqual(report["content_fps"], content_fps)
                expected = np.full(len(speeds), 24.0 / content_fps)
                expected[period - 1 :: period] *= 2
                np.testing.assert_allclose(np.diff(report["frame_times"]), expected)

    def test_single_strike_and_gradual_acceleration_do_not_rewrite_time(self):
        for speeds in (np.linspace(0.2, 4.0, 100), np.r_[np.ones(20), 8.0, np.ones(20)]):
            report = cadence_from_flow_steps(speeds)
            self.assertIsNone(report["period"])
            np.testing.assert_array_equal(report["frame_times"], np.arange(len(speeds) + 1))


if __name__ == "__main__":
    unittest.main()
