"""One timing contract for generation, source-frame fitting, GLB baking, and review."""

from dataclasses import dataclass
import math
import os

TIMING_CONTRACT = "monster-video-timing-v1"


def _positive(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


@dataclass(frozen=True)
class VideoTiming:
    fps: float = 24.0
    slow_motion_scale: float = 1.0
    time_scale: float | None = None

    def __post_init__(self):
        fps = _positive(self.fps, "MONSTER_VIDEO_FPS")
        slow = _positive(self.slow_motion_scale, "MONSTER_SLOW_MOTION_SCALE")
        if slow > 1:
            raise ValueError("MONSTER_SLOW_MOTION_SCALE must be in (0,1]")
        scale = _positive(
            1 / slow if self.time_scale is None else self.time_scale,
            "MONSTER_TIME_SCALE",
        )
        if not math.isfinite(fps * scale):
            raise ValueError("Combined playback FPS is not finite")
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "slow_motion_scale", slow)
        object.__setattr__(self, "time_scale", scale)

    @classmethod
    def from_environment(cls):
        return cls(
            os.environ.get("MONSTER_VIDEO_FPS", 24),
            os.environ.get("MONSTER_SLOW_MOTION_SCALE", 1),
            os.environ.get("MONSTER_TIME_SCALE"),
        )

    @classmethod
    def from_trajectory(cls, data, fallback_fps=24):
        saved = data.get("timing", {})
        if saved and saved.get("contract") != TIMING_CONTRACT:
            raise ValueError("Unknown video timing contract")
        clock_fps = data.get("cadence", {}).get("scene_fps", fallback_fps)
        timing = cls(
            data.get("fps", saved.get("fps", clock_fps)),
            saved.get("slow_motion_scale", 1),
            data.get("time_scale", saved.get("time_scale", 1)),
        )
        for key in ("fps", "time_scale"):
            if key in saved and not math.isclose(
                _positive(saved[key], key), getattr(timing, key), rel_tol=1e-9
            ):
                raise ValueError(f"Conflicting {key} in trajectory timing")
        return timing

    def as_dict(self):
        return {
            "contract": TIMING_CONTRACT,
            "fps": self.fps,
            "slow_motion_scale": self.slow_motion_scale,
            "time_scale": self.time_scale,
            "frame_units": "scaled_scene_frames",
        }

    def map_source_frames(self, times):
        result = []
        previous = -1.0
        for value in times:
            if isinstance(value, bool):
                raise ValueError("Source times must be finite, nonnegative and ordered")
            value = float(value)
            if not math.isfinite(value) or value < 0 or value < previous:
                raise ValueError("Source times must be finite, nonnegative and ordered")
            result.append(value / self.time_scale)
            previous = value
        return result

    def seconds(self, scene_frame):
        return float(scene_frame) / self.fps

    def preview_fps(self, source_fps=None):
        rate = (
            self.fps
            if source_fps is None
            else _positive(source_fps, "Source video FPS")
        )
        result = rate * self.time_scale
        if not math.isfinite(result):
            raise ValueError("Preview FPS is not finite")
        return result


def set_scene_fps(scene, fps):
    """Blender represents fractional frame rates as an integer FPS divided by fps_base."""
    fps = _positive(fps, "Scene FPS")
    scene.render.fps = max(1, min(32767, round(fps)))
    scene.render.fps_base = scene.render.fps / fps
    actual = scene.render.fps / scene.render.fps_base
    if not math.isclose(actual, fps, rel_tol=1e-6):
        raise ValueError(f"Blender cannot represent scene FPS {fps:g}")
