"""Timing acceptance includes configuration errors, fractional frame rates and generation metadata."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from video_timing import VideoTiming, set_scene_fps


class TimingAcceptanceTests(unittest.TestCase):
    def test_slow_generation_and_explicit_playback_override_share_one_contract(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(VideoTiming.from_environment().time_scale, 1)
        with patch.dict(
            "os.environ",
            {"MONSTER_VIDEO_FPS": "30", "MONSTER_SLOW_MOTION_SCALE": ".5"},
            clear=True,
        ):
            timing = VideoTiming.from_environment()
            self.assertEqual(timing.time_scale, 2)
            self.assertEqual(timing.map_source_frames([0, 1.5, 2.5]), [0, 0.75, 1.25])
            self.assertAlmostEqual(timing.seconds(1.25), 1.25 / 30)
            self.assertEqual(timing.preview_fps(30), 60)
            data = {
                "fps": timing.fps,
                "time_scale": timing.time_scale,
                "timing": timing.as_dict(),
            }
            self.assertEqual(VideoTiming.from_trajectory(data), timing)
        with patch.dict(
            "os.environ",
            {"MONSTER_SLOW_MOTION_SCALE": ".5", "MONSTER_TIME_SCALE": "1.25"},
            clear=True,
        ):
            self.assertEqual(VideoTiming.from_environment().time_scale, 1.25)

    def test_invalid_configuration_and_conflicting_saved_values_fail_early(self):
        for name, values in [
            ("MONSTER_VIDEO_FPS", ["0", "-1", "nan", "inf", "no"]),
            ("MONSTER_SLOW_MOTION_SCALE", ["0", "-1", "1.01", "nan", "inf"]),
            ("MONSTER_TIME_SCALE", ["0", "-1", "nan", "inf"]),
        ]:
            for value in values:
                with (
                    self.subTest(name=name, value=value),
                    patch.dict("os.environ", {name: value}, clear=True),
                    self.assertRaises(ValueError),
                ):
                    VideoTiming.from_environment()
        for kwargs in (
            {"fps": True},
            {"time_scale": False},
            {"slow_motion_scale": True},
        ):
            with self.assertRaises(ValueError):
                VideoTiming(**kwargs)
        with self.assertRaisesRegex(ValueError, "Conflicting fps"):
            VideoTiming.from_trajectory(
                {"fps": 30, "timing": {**VideoTiming().as_dict()}}
            )
        for times in ([0, -1], [0, float("nan")], [2, 1], [0, True]):
            with self.assertRaises(ValueError):
                VideoTiming().map_source_frames(times)

    def test_fractional_fps_uses_fps_base_without_integer_truncation(self):
        for fps in (24, 30, 29.97, 23.976, 59.94):
            scene = SimpleNamespace(render=SimpleNamespace(fps=0, fps_base=0))
            set_scene_fps(scene, fps)
            self.assertAlmostEqual(scene.render.fps / scene.render.fps_base, fps)
        self.assertEqual(VideoTiming.from_trajectory({}).fps, 24)
        self.assertEqual(
            VideoTiming.from_trajectory({"cadence": {"scene_fps": 30}}).fps, 30
        )

    def test_prompt_and_graph_keep_stacked_geometry_and_configured_fps(self):
        import pipeline_runner

        with patch.dict(
            "os.environ",
            {
                "MONSTER_STUDIO_LAYOUT": "stacked",
                "MONSTER_ENABLE_3VIEW": "0",
                "MONSTER_ENABLE_2K": "0",
                "MONSTER_VIDEO_FPS": "30",
                "MONSTER_SLOW_MOTION_SCALE": ".5",
            },
            clear=False,
        ):
            prompt = pipeline_runner.build_split_screen_prompt("bite")
            self.assertIn("Upper panel: side profile view", prompt)
            self.assertIn("Lower panel: top view", prompt)
            self.assertIn("0.5 times normal speed", prompt)
            self.assertIn("No smoke", prompt)
            graph = pipeline_runner.build_minimax_h3_comfy_graph(
                prompt, "reference.png"
            )
            self.assertEqual(graph["15"]["inputs"]["frame_rate"], 30)
            self.assertEqual(
                (graph["7"]["inputs"]["width"], graph["7"]["inputs"]["height"]),
                (768, 1344),
            )
            self.assertEqual(
                graph["31"]["inputs"]["mode"],
                {"mode": "target dimensions", "width": 672, "height": 1536},
            )
        with (
            patch.dict(
                "os.environ",
                {"MONSTER_STUDIO_LAYOUT": "stacked", "MONSTER_ENABLE_3VIEW": "1"},
                clear=False,
            ),
            self.assertRaisesRegex(ValueError, "two views"),
        ):
            pipeline_runner.build_split_screen_prompt("bite")


if __name__ == "__main__":
    unittest.main()
