"""Sampling acceptance: measured body contacts, calibrated framing, and event distributions."""

import math
import random
import sys
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from pose_sampling_v3 import (
    sampling_events,
    sample_roaming_translation,
    body_surface_targets,
)
from reprojection_solver import RigModel, aligned_rest_rotation, sample_v3_pose
from studio_camera import cameras_from_calibration


def sampling_fixture():
    meta = {
        "Root": {
            "parent": None,
            "kind": "root",
            "rest_head_world": [0, 0, 0],
            "rest_tail_world": [0, 0, 2],
        },
        "Prop_sword": {
            "parent": None,
            "kind": "root",
            "rest_head_world": [0, 0.6, 1],
            "rest_tail_world": [0, 0.6, 1.7],
        },
    }
    for side, sign in (("L", 1), ("R", -1)):
        parent = "Root"
        for name, a, b in [
            ("Arm", 0.2, 0.65),
            ("Forearm", 0.65, 1.0),
            ("Hand", 1.0, 1.2),
        ]:
            key = name + "." + side
            meta[key] = {
                "parent": parent,
                "kind": "limb",
                "rest_head_world": [sign * a, 0, 1.3],
                "rest_tail_world": [sign * b, 0, 1.3],
                "limits_rad": [[-3.0, 3.0]] * 3,
                "allow_leaf_twist": True,
            }
            parent = key
    names = sorted(meta)
    vertices = []
    triangles = []
    owners = []
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ]
    )
    for owner, center, scale in [
        ("Root", [0, 0, 1], [0.5, 0.4, 2]),
        ("Hand.L", [1.1, 0, 1.3], [0.2, 0.1, 0.1]),
        ("Hand.R", [-1.1, 0, 1.3], [0.2, 0.1, 0.1]),
        ("Prop_sword", [0, 0.6, 1.35], [0.04, 0.04, 0.7]),
    ]:
        verts = (
            np.array(
                [
                    [x, y, z]
                    for x in (-0.5, 0.5)
                    for y in (-0.5, 0.5)
                    for z in (-0.5, 0.5)
                ]
            )
            * scale
            + center
        )
        triangles.extend((faces + len(vertices)).tolist())
        vertices.extend(verts.tolist())
        owners.extend([names.index(owner)] * 8)
    for b in meta.values():
        b["rest_rot_mat_world"] = aligned_rest_rotation(
            np.array(b["rest_head_world"]), np.array(b["rest_tail_world"])
        ).tolist()
    skin = {
        "bone_names": names,
        "rest_verts": np.array(vertices),
        "triangles": np.array(triangles),
        "skin_indices": np.array(owners)[:, None],
        "skin_weights": np.ones((32, 1)),
        "rest_heads": np.array([meta[n]["rest_head_world"] for n in names]),
        "rest_R": np.array([meta[n]["rest_rot_mat_world"] for n in names]),
        "height": 2.0,
    }
    rig = RigModel(meta, names, skin)
    return meta, skin, rig


class SamplingAcceptanceTests(unittest.TestCase):
    def test_event_probabilities_are_per_pose_and_motion_blur_conditioned(self):
        rng = random.Random(123)
        counts = {
            key: 0
            for key in [
                "together",
                "onto_body",
                "yaw",
                "pitch",
                "roam",
                "jump",
                "jaw",
                "hold_prop",
                "strike",
                "lunge",
            ]
        }
        for _ in range(50000):
            event = sampling_events(rng, motion_blur=True)
            if event["reach"]:
                counts[event["reach"]] += 1
            for name in (
                "yaw",
                "pitch",
                "roam",
                "jump",
                "jaw",
                "hold_prop",
                "strike",
                "lunge",
            ):
                counts[name] += event[name]
        for name, expected in [
            ("together", 0.18),
            ("onto_body", 0.17),
            ("yaw", 0.25),
            ("pitch", 0.20),
            ("roam", 0.50),
            ("jump", 0.30),
            ("jaw", 0.50),
            ("hold_prop", 0.75),
            ("strike", 0.25),
            ("lunge", 0.30),
        ]:
            self.assertAlmostEqual(
                counts[name] / 50000, expected, delta=0.006, msg=name
            )
        for _ in range(100):
            event = sampling_events(rng, calm=True, motion_blur=False)
            self.assertFalse(
                any(
                    event[k]
                    for k in ("yaw", "pitch", "roam", "jump", "strike", "lunge")
                )
            )

    def test_roaming_stays_inside_every_shifted_panel_for_all_camera_modes(self):
        for order in (["front", "side"], ["left", "front", "right"], ["side", "top"]):
            native = list(
                cameras_from_calibration(
                    {
                        "center": [0, 0, 1],
                        "panel_width": 640,
                        "panel_height": 512,
                        "ortho_scale": 3.0,
                        "view_order": order,
                    }
                ).values()
            )
            for kind in ("ortho", "persp"):
                cameras = [
                    replace(c, type=kind, focal_px=700, shift_px=(17, -11))
                    for c in native
                ]
                rng = random.Random(7)
                xy = []
                for _ in range(200):
                    translation = sample_roaming_translation(
                        cameras, [0, 0, 1.15], 2.0, rng
                    )
                    xy.append(translation[:2])
                    for c in cameras:
                        uv = c.project(np.array([0, 0, 1.15]) + translation) / [
                            c.width,
                            c.height,
                        ]
                        self.assertTrue(
                            ((uv >= 0.03 - 1e-9) & (uv <= 0.97 + 1e-9)).all(),
                            (kind, order, uv),
                        )
                self.assertGreater(np.ptp(xy, axis=0).min(), 0.6)

    def test_contact_points_lie_on_posed_trunk_surface_and_exclude_props(self):
        meta, skin, rig = sampling_fixture()
        p = rig.zero()
        root = rig.slices["Root"]
        p[root.start : root.start + 3] = [0.2, -0.1, 0.3]
        p[root.start + 3 : root.stop] = [0.2, 0.4, 0.1]
        points = body_surface_targets(rig, p, skin, random.Random(5), count=200)
        _, heads, rots = rig.fk(p)
        rest = (points - heads["Root"]) @ rots["Root"] @ rig.R["Root"].T + rig.h["Root"]
        normalized = (rest - [0, 0, 1]) / [0.25, 0.2, 1]
        self.assertTrue((abs(normalized) <= 1 + 1e-10).all())
        np.testing.assert_allclose(abs(normalized).max(1), 1, atol=1e-10)

    def test_reacher_uses_actual_surface_and_together_targets_are_close(self):
        meta, skin, rig = sampling_fixture()
        eulers = {n: [0, 0, 0] for n in meta}
        for mode in ("onto_body", "together"):
            report = {}
            sample_v3_pose(
                meta,
                eulers,
                random.Random(4),
                skin=skin,
                events={"reach": mode},
                diagnostics=report,
            )
            self.assertEqual(
                len(report["reach_targets"]), 1 if mode == "onto_body" else 2
            )
            self.assertLess(max(report["reach_error_H"].values()), 0.035, report)
            targets = np.array(list(report["reach_targets"].values()))
            if mode == "onto_body":
                normalized = (targets - [0, 0, 1]) / [0.25, 0.2, 1]
                np.testing.assert_allclose(abs(normalized).max(1), 1, atol=1e-10)
            else:
                self.assertAlmostEqual(
                    np.linalg.norm(targets[0] - targets[1]) / 2, 0.04
                )

    def test_held_props_have_independent_rotation_and_sub_three_percent_grip_distance(
        self,
    ):
        meta, skin, rig = sampling_fixture()
        report = {}
        eulers, translation, props = sample_v3_pose(
            meta,
            {n: [0.1, 0.2, 0.3] for n in meta},
            random.Random(12),
            skin=skin,
            events={"hold_prop": True, "yaw": True, "pitch": True, "jump": True},
            diagnostics=report,
        )
        self.assertIn("Prop_sword", props)
        self.assertLess(report["held_props"]["Prop_sword"]["grip_distance_H"], 0.03)
        self.assertGreater(np.linalg.norm(eulers["Prop_sword"]), 0.01)
        self.assertLessEqual(report["jump_height_H"], 0.3)
        self.assertLessEqual(abs(report["root_world_euler"][0]), math.radians(75))
        self.assertLessEqual(abs(report["root_world_euler"][2]), math.pi)

    def test_reading_permutation_follows_primary_view_not_composite_slot(self):
        from video2rig_pipeline import observation_view_permutation

        for order, expected in (
            (["front", "side"], [0, 1]),
            (["left", "front", "right"], [1, 0, 2]),
            (["side", "top"], [1, 0]),
        ):
            c = list(
                cameras_from_calibration(
                    {
                        "center": [0, 0, 1],
                        "panel_width": 448,
                        "panel_height": 448,
                        "ortho_scale": 3.0,
                        "view_order": order,
                    }
                ).values()
            )
            self.assertEqual(
                observation_view_permutation({"view_order": order}, c), expected
            )


class ReadingLayoutAcceptanceTests(unittest.TestCase):
    def test_stacked_protocol_uses_original_network_ids_and_repairs_side_swaps(self):
        import tempfile
        from unittest.mock import patch
        from PIL import Image
        from test_video2rig_v3 import fixture
        from video2rig_pipeline import infer_v3_frames
        import reader_v3, dinov2_pose_trainer, pose_finetuning

        rig, _ = fixture()
        order = ["side", "top"]
        calibration = {
            "center": [0, 0, 1],
            "panel_width": 64,
            "panel_height": 64,
            "ortho_scale": 3.0,
            "view_order": order,
            "layout": "stacked",
        }
        cameras = list(cameras_from_calibration(calibration).values())
        p = rig.zero()
        p[0:3] = [0.2, 0.3, 0.7]
        X = rig.fk(p)[0]
        clean = np.array([c.project(X) for c in cameras])
        observed = clean.copy()
        for left, right in [("Arm.L", "Arm.R")]:
            ids = [
                2 * rig.index[left],
                2 * rig.index[left] + 1,
                2 * rig.index[right],
                2 * rig.index[right] + 1,
            ]
            observed[0, ids] = observed[0, [ids[2], ids[3], ids[0], ids[1]]]
        calls = []

        def predict(model, panels, view, device, *args, **kwargs):
            calls.append(view)
            if not kwargs.get('coarse_only'):
                self.assertEqual(kwargs['window'], checkpoint['reader_window'])
            self.assertEqual(int(panels[0][0, 0, 0]), 17 if view == 0 else 83)
            return (
                observed[view][None],
                np.ones((1, len(X))),
                [] if kwargs.get("coarse_only") else {"oracle": True},
            )

        class Probe:
            def __init__(self, **kwargs):
                pass

            def to(self, *args):
                return self

            def eval(self):
                return self

        checkpoint = {
            "start_pose_contract": "monster-start-pose-v1",
            "reader_window": {"contract": "monster-reader-training-p95-window-v1", "side_px": 64, "panel_width": 64, "panel_height": 64},
            "calibration": calibration,
            "bone_names": rig.names,
            "bones_meta": rig.meta,
            "extra_points": rig.extra,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.png"
            image = np.zeros((128, 64, 3), np.uint8)
            image[:64] = 17
            image[64:] = 83
            Image.fromarray(image).save(path)
            with (
                patch.object(reader_v3, "infer_reader_sequence", side_effect=predict),
                patch.object(dinov2_pose_trainer, "MonsterDINOv2PoseProbe", Probe),
                patch.object(pose_finetuning, "load_probe_weights"),
            ):
                result = infer_v3_frames(checkpoint, [str(path)])
        self.assertEqual(calls, [0, 1, 0, 1])
        self.assertTrue(result["gate_passed"], result)
        frame = result["frames"][0]
        for name in ("Arm.L", "Arm.R"):
            i = 2 * rig.index[name]
            np.testing.assert_allclose(
                frame["bones"][name]["reader_uv_side_tail"], clean[0, i + 1], atol=1e-9
            )
            np.testing.assert_allclose(
                frame["bones"][name]["reader_uv_top_tail"], clean[1, i + 1], atol=1e-9
            )
        self.assertEqual(result["calibration"]["view_order"], order)

    def test_solver_treats_nonfinite_readings_as_missing_and_holds_an_occluded_pose(
        self,
    ):
        from test_video2rig_v3 import fixture
        from reprojection_solver import fit_clip_2d_reprojection

        rig, cameras = fixture()
        X = rig.fk(rig.zero())[0]
        uv = np.array([[c.project(X) for c in cameras]] * 4)
        confidence = np.ones(uv.shape[:-1])
        uv[1, 0, :2] = np.nan
        uv[2, 0] = np.inf
        uv[3] = np.nan
        confidence[3] = np.nan
        parameters, report = fit_clip_2d_reprojection(
            rig, cameras, uv, confidence, max_nfev=30
        )
        self.assertTrue(np.isfinite(parameters).all())
        np.testing.assert_allclose(parameters, 0, atol=1e-7)
        self.assertEqual(report["frames"][3]["observed_points"], 0)
        self.assertIsNone(report["frames"][3]["median_error_px"])
        self.assertFalse(any("turn" in name for name in report["frames"][3]["starts"]))
        import json

        json.dumps(report, allow_nan=False)

    def test_overlap_threshold_only_downweights_the_secondary_view(self):
        from test_video2rig_v3 import fixture
        from reading_pipeline import downweight_overlapping_twins

        rig, _ = fixture()
        left = 2 * rig.index["Arm.L"] + 1
        right = 2 * rig.index["Arm.R"] + 1
        uv = np.zeros((1, 2, len(rig.point_names), 2))
        confidence = np.ones(uv.shape[:-1])
        for distance, expected in ((5.99, 0.4), (6.0, 1.0), (6.01, 1.0)):
            uv[0, 1, right] = [distance, 0]
            out = downweight_overlapping_twins(rig, uv, confidence, 100)
            self.assertEqual(out[0, 1, left], expected)
            self.assertEqual(out[0, 1, right], expected)
            self.assertEqual(out[0, 0, left], 1.0)

    def test_jump_repair_transports_confidence_with_its_observation(self):
        from test_video2rig_v3 import fixture
        from reading_pipeline import jump_pass_constant_velocity

        rig, _ = fixture()
        l = [2 * rig.index["Arm.L"], 2 * rig.index["Arm.L"] + 1]
        r = [2 * rig.index["Arm.R"], 2 * rig.index["Arm.R"] + 1]
        uv = np.zeros((5, 2, len(rig.point_names), 2))
        confidence = np.ones(uv.shape[:-1])
        for t in range(5):
            uv[t, :, l, 0] = t
            uv[t, :, r, 0] = 40 + t
            confidence[t, :, l] = 0.8
            confidence[t, :, r] = 0.9
        clean = uv.copy()
        expected = confidence.copy()
        uv[2, 1, l + r] = uv[2, 1, r + l]
        confidence[2, 1, l + r] = confidence[2, 1, r + l]
        out, weights = jump_pass_constant_velocity(rig, uv, confidence)
        np.testing.assert_array_equal(out, clean)
        np.testing.assert_array_equal(weights, expected)


if __name__ == "__main__":
    unittest.main()
