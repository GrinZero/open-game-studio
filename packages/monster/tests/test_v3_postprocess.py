"""Counterexamples for real fast motion, planted-foot translation and spring dynamics."""

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from reprojection_solver import RigModel, aligned_rest_rotation
from reading_pipeline import (
    despike_reading_backed,
    recover_root_translation_from_steps,
    simulate_damped_spring_chains,
)


def leg_rig():
    meta = {
        "Root": {
            "parent": None,
            "kind": "root",
            "rest_head_world": [0, 0, 0.9],
            "rest_tail_world": [0, 0, 1.4],
        },
        "Hand.L": {
            "parent": "Root",
            "kind": "rigid",
            "fixed_local": True,
            "rest_head_world": [0.35, 0, 1.1],
            "rest_tail_world": [0.4, -0.2, 1.1],
        },
        "Prop_held": {
            "parent": None,
            "kind": "root",
            "rest_head_world": [0.4, -0.2, 1.1],
            "rest_tail_world": [0.4, -0.2, 1.6],
        },
        "Prop_world": {
            "parent": None,
            "kind": "root",
            "rest_head_world": [2, 2, 0.2],
            "rest_tail_world": [2, 2, 0.8],
        },
    }
    for side, x in [("L", 0.2), ("R", -0.2)]:
        parent = "Root"
        for name, h, t in [
            ("Thigh", [x, 0, 0.8], [x, 0.2, 0.45]),
            ("Shin", [x, 0.2, 0.45], [x, 0, 0.05]),
            ("Foot", [x, 0, 0.05], [x, -0.15, 0.05]),
        ]:
            key = name + "." + side
            meta[key] = {
                "parent": parent,
                "kind": "limb",
                "rest_head_world": h,
                "rest_tail_world": t,
                "limits_rad": [[-math.pi, math.pi]] * 3,
            }
            parent = key
    for b in meta.values():
        b["rest_rot_mat_world"] = aligned_rest_rotation(
            np.array(b["rest_head_world"]), np.array(b["rest_tail_world"])
        ).tolist()
    return RigModel(meta)


def planted_sequence(rig, ankle_y, root_y=None):
    result = []
    for y in ankle_y:
        target = rig.fk(rig.zero())[0]
        for side in ("L", "R"):
            hip = rig.h["Thigh." + side]
            ankle = np.array([hip[0], y, 0.05])
            delta = ankle - hip
            d = np.linalg.norm(delta)
            L1 = np.linalg.norm(rig.t["Thigh." + side] - hip)
            L2 = np.linalg.norm(rig.t["Shin." + side] - rig.h["Shin." + side])
            a = (L1 * L1 - L2 * L2 + d * d) / (2 * d)
            normal = np.array([0, -delta[2], delta[1]]) / d
            knee = hip + a * delta / d + np.sqrt(max(0, L1 * L1 - a * a)) * normal
            for name, h, t in [
                ("Thigh", hip, knee),
                ("Shin", knee, ankle),
                ("Foot", ankle, ankle + [0, -0.15, 0]),
            ]:
                i = 2 * rig.index[name + "." + side]
                target[i : i + 2] = [h, t]
        p = rig.read_start(target)
        # Verify the test really uses a constrained articulated stance, not independent point motion.
        for side in ("L", "R"):
            i = 2 * rig.index["Foot." + side] + 1
            np.testing.assert_allclose(rig.fk(p)[0][i], target[i], atol=1e-10)
        result.append(p)
    result = np.array(result)
    if root_y is not None:
        s = rig.slices["Root"]
        result[:, s.start + 4] = root_y
    return result


class DespikeAcceptanceTests(unittest.TestCase):
    def test_true_fast_single_limb_motion_is_preserved_but_wrong_spike_is_removed(self):
        from test_video2rig_v3 import fixture

        rig, cameras = fixture()
        parameters = np.zeros((3, rig.size))
        indices, _ = rig.slices["Arm.L"]
        parameters[1, indices[0]] = 0.4
        for real_motion in (True, False):
            truth = parameters if real_motion else np.zeros_like(parameters)
            uv = np.array([[c.project(rig.fk(p)[0]) for c in cameras] for p in truth])
            output = despike_reading_backed(
                rig, parameters, cameras, uv, np.ones(uv.shape[:-1])
            )
            np.testing.assert_allclose(
                output, parameters if real_motion else 0.0, atol=1e-12
            )

    def test_translation_is_preserved_when_a_joint_spike_is_removed(self):
        from test_video2rig_v3 import fixture

        rig, cameras = fixture()
        p = np.zeros((3, rig.size))
        s = rig.slices["Root"]
        p[1, s.start + 3] = 0.1
        truth = p.copy()
        ix, _ = rig.slices["Arm.L"]
        p[1, ix[0]] = 0.4
        uv = np.array([[c.project(rig.fk(q)[0]) for c in cameras] for q in truth])
        out = despike_reading_backed(rig, p, cameras, uv, np.ones(uv.shape[:-1]))
        np.testing.assert_allclose(out, truth, atol=1e-12)

    def test_six_degree_boundary_does_not_filter(self):
        from test_video2rig_v3 import fixture

        rig, cameras = fixture()
        p = np.zeros((3, rig.size))
        ix, _ = rig.slices["Arm.L"]
        p[1, ix[0]] = math.radians(6)
        uv = np.array([[c.project(rig.fk(rig.zero())[0]) for c in cameras]] * 3)
        np.testing.assert_array_equal(
            despike_reading_backed(rig, p, cameras, uv, np.ones(uv.shape[:-1])), p
        )


class RootStepsAcceptanceTests(unittest.TestCase):
    def test_in_place_stance_recovers_root_and_moves_only_held_props(self):
        rig = leg_rig()
        t = np.linspace(0, 1, 31)
        p = planted_sequence(rig, 0.1 - 0.2 * t)
        out, report = recover_root_translation_from_steps(rig, p, 30, frame_times=t)
        self.assertTrue(report["applied"], report)
        self.assertAlmostEqual(report["displacement"][1], 0.2, places=8)
        foot = 2 * rig.index["Foot.L"] + 1
        positions = np.array([rig.fk(q)[0][foot] for q in out])
        np.testing.assert_allclose(
            positions, positions[[0]].repeat(len(t), axis=0), atol=1e-8
        )
        held = rig.slices["Prop_held"]
        world = rig.slices["Prop_world"]
        self.assertIn("Prop_held", report["translated_props"])
        self.assertNotIn("Prop_world", report["translated_props"])
        self.assertAlmostEqual(out[-1, held.start + 4], 0.2, places=8)
        np.testing.assert_array_equal(out[:, world], p[:, world])

    def test_existing_root_translation_is_not_cancelled_or_duplicated(self):
        rig = leg_rig()
        t = np.linspace(0, 1, 31)
        for y, root in ((np.zeros(31), -0.2 * t), (0.1 - 0.2 * t, 0.2 * t)):
            p = planted_sequence(rig, y, root)
            out, report = recover_root_translation_from_steps(rig, p, 30, frame_times=t)
            self.assertFalse(report["applied"], report)
            np.testing.assert_array_equal(out, p)

    def test_fifty_five_percent_and_speed_thresholds(self):
        rig = leg_rig()
        t = np.linspace(0, 1, 101)
        for moving, expected in ((54, False), (55, True)):
            y = 0.1 - np.minimum(np.arange(101), moving) * 0.2 / 100
            p = planted_sequence(rig, y)
            _, report = recover_root_translation_from_steps(rig, p, 100, frame_times=t)
            self.assertEqual(report["applied"], expected, report)
        p = planted_sequence(rig, 0.05 - 0.079 * rig.height * t)
        _, report = recover_root_translation_from_steps(rig, p, 100, frame_times=t)
        self.assertFalse(report["applied"])

    def test_direction_consistency_boundary_and_two_hertz_filter(self):
        from reading_pipeline import _lowpass_in_seconds

        rig = leg_rig()
        times = np.linspace(0, 1, 101)
        for backward, expected in ((95, False), (96, True)):
            velocity = np.r_[-np.ones(backward), np.ones(100 - backward)] * 0.2
            y = 0.1 + np.r_[0, np.cumsum(velocity) / 100]
            _, report = recover_root_translation_from_steps(
                rig, planted_sequence(rig, y), 100, frame_times=times
            )
            self.assertAlmostEqual(
                report["direction_consistency"], abs(1 - 2 * backward / 100), places=10
            )
            self.assertEqual(report["applied"], expected, report)
        t = np.arange(1200) / 120
        velocity = np.column_stack(
            [
                0.2
                + 0.03 * np.sin(2 * np.pi * 0.5 * t)
                + 0.03 * np.sin(2 * np.pi * 8 * t),
                np.zeros(len(t)),
            ]
        )
        filtered = _lowpass_in_seconds(velocity, t, 120)
        middle = (t >= 1) & (t < 9)

        def amplitude(f):
            return 2 * abs(
                np.mean(
                    (filtered[middle, 0] - 0.2) * np.exp(-2j * np.pi * f * t[middle])
                )
            )

        self.assertGreater(amplitude(0.5), 0.03 * 0.97)
        self.assertLess(amplitude(8), 0.03 * 0.01)

    def test_irregular_and_duplicate_times_do_not_create_extra_root_travel(self):
        rig = leg_rig()
        t = np.array(
            [
                0,
                0.04,
                0.04,
                0.08,
                0.13,
                0.18,
                0.22,
                0.26,
                0.30,
                0.35,
                0.4,
                0.45,
                0.5,
                0.55,
                0.6,
                0.65,
                0.7,
                0.75,
                0.8,
                0.85,
                0.9,
                0.95,
                1.0,
            ]
        )
        p = planted_sequence(rig, 0.1 - 0.2 * t)
        out, report = recover_root_translation_from_steps(rig, p, 24, frame_times=t)
        self.assertTrue(report["applied"], report)
        self.assertAlmostEqual(report["displacement"][1], 0.2, places=8)
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_array_equal(out[1], out[2])


def spring_rig():
    meta = {
        "Root": {
            "parent": None,
            "kind": "root",
            "rest_head_world": [0, 0, 0],
            "rest_tail_world": [0, 0, 1],
        },
        "Tail1": {
            "parent": "Root",
            "kind": "tail",
            "rest_head_world": [0, 0, 0.7],
            "rest_tail_world": [0, 1, 0.7],
            "limits_rad": [[-1.5, 1.5]] * 3,
        },
    }
    return RigModel(meta)


class SpringAcceptanceTests(unittest.TestCase):
    def test_static_rotated_body_keeps_unobserved_twist_and_has_no_spurious_motion(
        self,
    ):
        rig = spring_rig()
        p = np.zeros((4, rig.size))
        root = rig.slices["Root"]
        p[:, root.start + 2] = 0.7
        p[:, rig.slices["Tail1"][0][1]] = 0.2
        out = simulate_damped_spring_chains(rig, p, 24)
        np.testing.assert_allclose(out, p, atol=1e-12)
        guess = rig.read_start(rig.fk(p[0])[0], p[0])
        np.testing.assert_allclose(rig.fk(guess)[0], rig.fk(p[0])[0], atol=1e-12)
        np.testing.assert_allclose(
            guess[root.start : root.start + 3],
            p[0, root.start : root.start + 3],
            atol=1e-12,
        )

    def test_four_hertz_point_five_five_damping_and_exact_length(self):
        from scipy.signal import find_peaks

        rig = spring_rig()
        fps = 120
        p = np.zeros((180, rig.size))
        root = rig.slices["Root"]
        p[20:, root.start + 3] = 0.03
        out = simulate_damped_spring_chains(rig, p, fps)
        points = np.array([rig.fk(q)[0] for q in out])
        index = 2 * rig.index["Tail1"]
        np.testing.assert_allclose(
            np.linalg.norm(points[:, index + 1] - points[:, index], axis=-1),
            1.0,
            atol=1e-12,
        )
        error = points[:, index + 1, 0] - 0.03
        self.assertLess(error[20], -0.02)  # The appendage lags the moving parent.
        peaks = find_peaks(error[20:])[0] + 20
        self.assertGreaterEqual(len(peaks), 3)
        measured_frequency = fps / np.mean(np.diff(peaks[:3]))
        expected_frequency = 4 * np.sqrt(1 - 0.55**2)
        self.assertAlmostEqual(measured_frequency, expected_frequency, delta=0.08)
        measured_ratio = error[peaks[1]] / error[peaks[0]]
        expected_ratio = np.exp(-2 * np.pi * 0.55 / np.sqrt(1 - 0.55**2))
        self.assertAlmostEqual(measured_ratio, expected_ratio, delta=0.003)
        self.assertLess(abs(error[-1]), 1e-6)
        self.assertTrue(
            (out >= rig.lower - 1e-12).all() and (out <= rig.upper + 1e-12).all()
        )

    def test_playback_time_and_duplicate_frames_do_not_double_the_spring_frequency(
        self,
    ):
        rig = spring_rig()
        root = rig.slices["Root"]
        times = np.arange(90) / 60
        params = np.zeros((90, rig.size))
        params[10:, root.start + 3] = 0.03
        actual = simulate_damped_spring_chains(rig, params, 30, frame_times=times)
        expected = simulate_damped_spring_chains(rig, params, 60)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        duplicate_times = np.insert(times, 20, times[20])
        duplicate_params = np.insert(params, 20, params[20], axis=0)
        duplicate = simulate_damped_spring_chains(
            rig, duplicate_params, 30, frame_times=duplicate_times
        )
        np.testing.assert_allclose(
            np.delete(duplicate, 20, axis=0), expected, atol=1e-12
        )

    def test_local_bounds_and_bone_lengths_survive_large_parent_movement(self):
        rig = spring_rig()
        tail_indices, _ = rig.slices["Tail1"]
        rig.lower[tail_indices] = -0.2
        rig.upper[tail_indices] = 0.2
        p = np.zeros((30, rig.size))
        s = rig.slices["Root"]
        p[3:, s.start + 3] = 0.6
        out = simulate_damped_spring_chains(rig, p, 24)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(
            (out[:, tail_indices] <= 0.2 + 1e-12).all()
            and (out[:, tail_indices] >= -0.2 - 1e-12).all()
        )
        points = np.array([rig.fk(q)[0] for q in out])
        k = 2 * rig.index["Tail1"]
        np.testing.assert_allclose(
            np.linalg.norm(points[:, k + 1] - points[:, k], axis=-1), 1.0, atol=1e-12
        )

    def test_nested_tail_ears_and_flex_follow_parent_without_breaking_attachment(self):
        rig = spring_rig()
        meta = rig.meta.copy()
        for name, parent, head, tail in [
            ("Tail2", "Tail1", [0, 1, 0.7], [0, 1.6, 0.7]),
            ("Ear1_1.L", "Root", [0.1, 0, 0.9], [0.2, 0, 1.1]),
            ("Flex1_1.R", "Root", [-0.1, 0, 0.8], [-0.1, -0.4, 0.8]),
        ]:
            meta[name] = {
                "parent": parent,
                "kind": "flex",
                "rest_head_world": head,
                "rest_tail_world": tail,
                "limits_rad": [[-1.5, 1.5]] * 3,
            }
        rig = RigModel(meta)
        p = np.zeros((120, rig.size))
        s = rig.slices["Root"]
        time = np.arange(120) / 60
        p[:, s.start + 2] = 0.3 * np.sin(2 * np.pi * 0.5 * time)
        p[:, s.start + 3] = 0.03 * np.sin(2 * np.pi * 0.2 * time)
        out = simulate_damped_spring_chains(rig, p, 60)
        original = np.array([rig.fk(q)[0] for q in p])
        points = np.array([rig.fk(q)[0] for q in out])
        for name in ("Tail1", "Tail2", "Ear1_1.L", "Flex1_1.R"):
            i = 2 * rig.index[name]
            length = np.linalg.norm(rig.t[name] - rig.h[name])
            np.testing.assert_allclose(
                np.linalg.norm(points[:, i + 1] - points[:, i], axis=-1),
                length,
                atol=1e-12,
            )
            self.assertGreater(
                np.linalg.norm(points[:, i + 1] - original[:, i + 1], axis=-1).max(),
                1e-4,
            )
        np.testing.assert_allclose(
            points[:, 2 * rig.index["Tail2"]],
            points[:, 2 * rig.index["Tail1"] + 1],
            atol=1e-12,
        )


class Video2RigTwoPathAndCompletionTests(unittest.TestCase):
    def test_two_path_feature_flags_and_legacy_mode_switching(self):
        import os
        from unittest.mock import patch
        from pose_features import (
            backbone_mode,
            clean_motion_enabled,
            decode_mode,
            export_npz_labels_enabled,
            fold_digits_enabled,
            keypoint_mode,
            no_cut_tracks_enabled,
            panel_drift_enabled,
            pipeline_feature_flags,
            pipeline_mode,
            rig_contract_v3_enabled,
            solver_mode,
            surface_contact_enabled,
            twoway_jump_enabled,
            video2rig_mode_enabled,
            visibility_mode,
        )

        with patch.dict(os.environ, {}, clear=False):
            for k in (
                "MONSTER_PIPELINE_MODE",
                "MONSTER_RIG_CONTRACT",
                "MONSTER_FOLD_DIGITS",
                "MONSTER_KEYPOINT_MODE",
                "MONSTER_VISIBILITY_MODE",
                "MONSTER_DECODE_MODE",
                "MONSTER_BACKBONE",
                "MONSTER_SOLVER_MODE",
                "MONSTER_CLEAN_MOTION",
                "MONSTER_SURFACE_CONTACT",
                "MONSTER_EXPORT_NPZ_LABELS",
                "MONSTER_NO_CUT_TRACKS",
                "MONSTER_TWOWAY_JUMP",
                "MONSTER_PANEL_DRIFT",
            ):
                os.environ.pop(k, None)
            self.assertEqual(pipeline_mode(), "video2rig")
            self.assertTrue(video2rig_mode_enabled())
            self.assertTrue(rig_contract_v3_enabled())
            self.assertFalse(fold_digits_enabled())
            self.assertEqual(keypoint_mode(), "video2rig")
            self.assertEqual(visibility_mode(), "hybrid")
            self.assertEqual(decode_mode(), "parabola")
            self.assertEqual(backbone_mode(), "dinov2-small")
            self.assertEqual(solver_mode(), "reprojection_v3")
            self.assertTrue(clean_motion_enabled())
            self.assertTrue(surface_contact_enabled())
            self.assertTrue(export_npz_labels_enabled())
            self.assertTrue(no_cut_tracks_enabled())
            self.assertTrue(twoway_jump_enabled())
            self.assertTrue(panel_drift_enabled())

        with patch.dict(os.environ, {"MONSTER_PIPELINE_MODE": "legacy"}, clear=False):
            self.assertEqual(pipeline_mode(), "legacy")
            self.assertFalse(video2rig_mode_enabled())
            self.assertFalse(rig_contract_v3_enabled())
            self.assertEqual(keypoint_mode(), "legacy")
            self.assertEqual(visibility_mode(), "bvh")
            self.assertEqual(decode_mode(), "dsnt")
            self.assertEqual(backbone_mode(), "dinov2_vitb14_reg")
            self.assertEqual(solver_mode(), "legacy_3d_kind")
            self.assertFalse(clean_motion_enabled())
            self.assertFalse(surface_contact_enabled())
            self.assertFalse(export_npz_labels_enabled())
            self.assertFalse(no_cut_tracks_enabled())
            self.assertFalse(twoway_jump_enabled())
            self.assertFalse(panel_drift_enabled())
            snap = pipeline_feature_flags()
            self.assertEqual(snap["pipeline_mode"], "legacy")
            self.assertEqual(snap["solver_mode"], "legacy_3d_kind")

    def test_mirror_consistent_surface_keypoints_and_jaw_hinge_and_held_guards(self):
        from reprojection_solver import configure_jaw_hinge, surface_keypoints

        meta = {
            "Root": {
                "parent": None,
                "kind": "root",
                "rest_head_world": [0.0, 0.0, 0.0],
                "rest_tail_world": [0.0, 0.0, 1.0],
            },
            "Head": {
                "parent": "Root",
                "kind": "head",
                "rest_head_world": [0.0, 0.0, 1.0],
                "rest_tail_world": [0.0, 0.0, 1.4],
                "limits_rad": [[-0.5, 0.5]] * 3,
            },
            "Jaw": {
                "parent": "Head",
                "kind": "jaw",
                "rest_head_world": [0.0, -0.05, 1.1],
                "rest_tail_world": [0.0, -0.25, 1.05],
                "limits_rad": [[-1.0, 1.0]] * 3,
            },
            "Hand.L": {
                "parent": "Root",
                "kind": "limb",
                "rest_head_world": [0.4, 0.0, 0.8],
                "rest_tail_world": [0.5, 0.0, 0.8],
                "limits_rad": [[-1.0, 1.0]] * 3,
            },
            "Hand.R": {
                "parent": "Root",
                "kind": "limb",
                "rest_head_world": [-0.4, 0.0, 0.8],
                "rest_tail_world": [-0.5, 0.0, 0.8],
                "limits_rad": [[-1.0, 1.0]] * 3,
            },
        }
        names = ["Root", "Head", "Jaw", "Hand.L", "Hand.R"]
        # Build symmetric vertices for Hand.L and Hand.R, plus extended held-weapon vertices on Hand.L
        verts_l = np.array(
            [
                [0.44, -0.08, 0.80],
                [0.44, 0.08, 0.80],
                [0.46, -0.08, 0.80],
                [0.46, 0.08, 0.80],
                [0.45, 0.00, 0.77],
                [0.45, 0.00, 0.83],
                [0.85, 0.00, 0.80],
                [0.95, 0.00, 0.80],
            ],
            dtype=float,
        )
        verts_r = verts_l.copy()
        verts_r[:, 0] *= -1.0
        verts_jaw = np.array(
            [
                [-0.04, -0.15, 1.07],
                [0.04, -0.15, 1.07],
                [-0.04, -0.22, 1.05],
                [0.04, -0.22, 1.05],
                [0.0, -0.25, 1.04],
                [0.0, -0.12, 1.08],
            ],
            dtype=float,
        )
        all_verts = np.vstack([verts_l, verts_r, verts_jaw])
        skin_idx = np.zeros((len(all_verts), 4), dtype=int)
        skin_w = np.zeros((len(all_verts), 4), dtype=float)
        skin_idx[:8, 0] = 3  # Hand.L
        skin_w[:8, 0] = 1.0
        skin_idx[8:16, 0] = 4  # Hand.R
        skin_w[8:16, 0] = 1.0
        skin_idx[16:, 0] = 2  # Jaw
        skin_w[16:, 0] = 1.0
        skin = {
            "bone_names": names,
            "rest_verts": all_verts,
            "skin_indices": skin_idx,
            "skin_weights": skin_w,
            "triangles": np.zeros((0, 3), dtype=int),
        }
        pts = surface_keypoints(meta, names, skin)
        by_name = {p["name"]: p for p in pts}
        self.assertIn("Hand.L.sidea1", by_name)
        self.assertIn("Hand.R.sidea1", by_name)
        rig = RigModel(meta, names, skin, pts)
        X0 = rig.fk(rig.start_params())[0]
        idx_l = rig.point_names.index("Hand.L.sidea1")
        idx_r = rig.point_names.index("Hand.R.sidea1")
        # Mirror-consistent orientation: Hand.L.sidea1 and Hand.R.sidea1 should have same Y sign (facing)
        self.assertAlmostEqual(
            np.sign(X0[idx_l, 1]),
            np.sign(X0[idx_r, 1]),
        )
        self.assertGreaterEqual(len(rig.held_guards), 4)
        jaw_cfg = configure_jaw_hinge(rig)
        self.assertIn("Jaw", jaw_cfg)
        ax = jaw_cfg["Jaw"]["axis"]
        jaw_indices, _ = rig.slices["Jaw"]
        self.assertAlmostEqual(float(rig.lower[jaw_indices[ax]]), -0.05)
        self.assertAlmostEqual(float(rig.upper[jaw_indices[ax]]), 0.70)

    def test_parabola_decode_zbuffer_visibility_and_npz_label_export(self):
        import tempfile
        from studio_camera import Camera
        from reader_v3 import (
            compute_zbuffer_visibility,
            decode_heatmap_parabola_numpy,
            export_video2rig_npz_labels,
        )

        logits = np.full((1, 2, 16, 16), -6.0, dtype=float)
        logits[0, 0, 8, 5] = 4.0
        logits[0, 0, 8, 6] = 2.0
        logits[0, 0, 8, 4] = 0.0
        coords, conf = decode_heatmap_parabola_numpy(logits)
        self.assertEqual(coords.shape, (1, 2, 2))
        self.assertGreater(coords[0, 0, 0], 5.0 / 15.0)
        self.assertLess(coords[0, 0, 0], 5.5 / 15.0)
        self.assertGreater(conf[0, 0], 0.95)

        rig = leg_rig()
        camera = Camera(np.eye(3), [0.0, 0.0, 3.0], 64, 64, "ortho", 2.0)
        verts = np.array(
            [
                [-0.2, 0.0, 0.1],
                [0.2, 0.0, 0.1],
                [0.0, 0.2, 0.8],
            ],
            dtype=float,
        )
        skin = {
            "bone_names": rig.names,
            "skin_indices": np.full((3, 1), rig.index["Root"], dtype=int),
            "skin_weights": np.ones((3, 1), dtype=float),
            "triangles": np.array([[0, 1, 2]], dtype=int),
        }
        pts3d = np.array([[0.0, 0.05, 0.3], [10.0, 10.0, 0.3]], dtype=float)
        vis = compute_zbuffer_visibility(
            verts, skin, camera, pts3d, ["Root", "Root"], rig.meta, bone_names=rig.names
        )
        self.assertEqual(vis[0], 1.0)
        self.assertEqual(vis[1], 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            records = [
                {
                    "joints": {
                        n: {"head_3d": list(rig.h[n]), "tail_3d": list(rig.t[n])}
                        for n in rig.names
                    },
                    "reader_points": {
                        "front": {
                            "uv": [[10.0, 10.0]] * (2 * len(rig.names)),
                            "visibility": [1.0] * (2 * len(rig.names)),
                        }
                    },
                }
            ]
            exported = export_video2rig_npz_labels(
                tmp,
                records,
                rig.meta,
                rig.names,
                [],
                {"view_order": ["front"]},
            )
            with np.load(exported["labels_npz"]) as lab:
                self.assertEqual(lab["heads"].shape, (1, len(rig.names), 3))
            with np.load(exported["visible_npz"]) as vis_npz:
                self.assertEqual(vis_npz["vis"].shape, (1, 1, 2 * len(rig.names)))

    def test_clean_motion_eight_step_postprocessing_pipeline(self):
        from studio_camera import Camera
        from reading_pipeline import clean_motion

        rig = leg_rig()
        cameras = [
            Camera(np.eye(3), [0.0, 0.0, 3.0], 256, 256, "ortho", 2.0),
            Camera(
                np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
                [3.0, 0.0, 0.0],
                256,
                256,
                "ortho",
                2.0,
            ),
        ]
        T = 24
        t = np.linspace(0, 1, T)
        p = planted_sequence(rig, 0.05 - 0.12 * t)
        uv = np.array([[c.project(rig.fk(q)[0]) for c in cameras] for q in p])
        conf = np.ones(uv.shape[:-1])
        # Inject a single-frame root yaw spike at frame 10
        p[10, 2] += 0.35
        cleaned, report = clean_motion(
            rig, cameras, p, uv, conf, fps=24.0, frame_times=t
        )
        self.assertEqual(cleaned.shape, p.shape)
        self.assertTrue(np.isfinite(cleaned).all())
        self.assertEqual(len(report["steps_applied"]), 8)
        self.assertIn("root_from_steps", report)
        self.assertIn("locked_feet", report)
        self.assertIn("keep_out_repairs", report)

    def test_review_counterexamples_perspective_jacobian_and_legacy_contracts(self):
        from bone_taxonomy import classify_rig_bones
        from reprojection_solver import surface_keypoints
        from studio_camera import Camera

        cam = Camera(
            np.eye(3), [0.0, 0.0, 0.0], 256, 256, "persp", focal_px=300.0
        )
        behind = np.array([[0.2, -0.3, -1.5]], dtype=float)
        uv0, J = cam.project(behind, jac=True)
        eps = 1e-6
        uv_dz = cam.project(behind + np.array([[0.0, 0.0, eps]]))
        num_dz = (uv_dz[0] - uv0[0]) / eps
        np.testing.assert_allclose(J[0, :, 2], num_dz, atol=1e-7)
        np.testing.assert_allclose(J[0, :, 2], 0.0, atol=1e-12)

        raw_bones = {
            "b0": {
                "parent": None,
                "children": ["b1"],
                "rest_head_world": [0.0, 0.0, 0.5],
                "rest_tail_world": [0.0, 0.0, 1.0],
                "rest_len": 0.5,
            },
            "b1": {
                "parent": "b0",
                "children": [],
                "rest_head_world": [0.0, 0.0, 1.0],
                "rest_tail_world": [0.0, 0.0, 1.5],
                "rest_len": 0.5,
            },
        }
        with mock.patch.dict(os.environ, {"MONSTER_PIPELINE_MODE": "legacy"}, clear=False):
            meta_leg = classify_rig_bones(raw_bones)
            self.assertEqual(meta_leg["b0"]["canonical_name"], "ROOT_C_00_00")
            skin_dummy = {
                "bone_names": ["b0", "b1"],
                "rest_verts": np.zeros((8, 3)),
                "skin_indices": np.zeros((8, 1), dtype=int),
                "skin_weights": np.ones((8, 1), dtype=float),
            }
            self.assertEqual(surface_keypoints(meta_leg, ["b0", "b1"], skin_dummy), [])

    def test_review_counterexamples_zbuffer_visibility_and_poses_npz(self):
        from reader_v3 import compute_zbuffer_visibility, export_video2rig_npz_labels
        from studio_camera import Camera

        rig = leg_rig()
        camera = Camera(np.eye(3), [0.0, 0.0, 3.0], 128, 128, "ortho", 2.0)
        root_idx = rig.index["Root"]
        shin_idx = rig.index["Shin.L"]
        # Triangle where vertex 0 belongs to Root, but vertices 1 and 2 belong to Shin.L
        verts = np.array(
            [
                [-0.3, -0.3, 0.2],
                [0.3, -0.3, 0.2],
                [0.0, 0.3, 0.2],
            ],
            dtype=float,
        )
        skin = {
            "bone_names": rig.names,
            "skin_indices": np.array([[root_idx], [shin_idx], [shin_idx]], dtype=int),
            "skin_weights": np.array([[1.0], [1.0], [1.0]], dtype=float),
            "triangles": np.array([[0, 1, 2]], dtype=int),
        }
        # Point 0 projects onto the triangle (owned by Shin.L=2); Point 1 is in-frame on empty background
        pts3d = np.array([[0.0, -0.05, 0.2], [0.7, 0.7, 0.2]], dtype=float)
        vis_shin = compute_zbuffer_visibility(
            verts, skin, camera, pts3d, ["Shin.L", "Shin.L"], rig.meta, bone_names=rig.names
        )
        self.assertEqual(vis_shin[0], 1.0)
        # In-frame background pixel (zbuf == inf) must NOT be treated as visible!
        self.assertEqual(vis_shin[1], 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            records = [
                {
                    "joints": {
                        n: {
                            "head_3d": (rig.h[n] + np.array([0.1, -0.2, 0.05])).tolist(),
                            "tail_3d": (rig.t[n] + np.array([0.1, -0.2, 0.05])).tolist(),
                            "euler_xyz": [0.25, -0.15, 0.35],
                        }
                        for n in rig.names
                    },
                    "v3_sampling": {"translation": [0.1, -0.2, 0.05]},
                    "reader_points": {
                        "front": {
                            "uv": [[10.0, 10.0]] * (2 * len(rig.names)),
                            "visibility": [1.0] * (2 * len(rig.names)),
                        }
                    },
                }
            ]
            exported = export_video2rig_npz_labels(
                tmp, records, rig.meta, rig.names, [], {"view_order": ["front"]}
            )
            with np.load(exported["poses_npz"]) as p_npz:
                self.assertFalse(np.allclose(p_npz["quats"][0, 0], [1.0, 0.0, 0.0, 0.0]))
                np.testing.assert_allclose(p_npz["root_offsets"][0], [0.1, -0.2, 0.05], atol=1e-5)
                self.assertIn("params", p_npz)
                self.assertGreater(np.linalg.norm(p_npz["params"][0]), 0.1)

    def test_review_counterexamples_offaxis_held_guard_capsule_exemption_and_prop_floor(self):
        from reading_pipeline import floor_bend_and_smooth_lift
        from reprojection_solver import RigModel, configure_jaw_hinge, sample_v3_pose

        meta = {
            "Root": {
                "parent": None,
                "kind": "root",
                "rest_head_world": [0.0, 0.0, 0.8],
                "rest_tail_world": [0.0, 0.0, 1.3],
            },
            "Hand.L": {
                "parent": "Root",
                "kind": "limb",
                "rest_head_world": [0.05, 0.0, 1.0],
                "rest_tail_world": [0.05, 0.0, 1.1],
                "limits_rad": [[-1.0, 1.0], [-1.5, 1.5], [-1.0, 1.0]],
            },
            "Jaw": {
                "parent": "Root",
                "kind": "jaw",
                "rest_head_world": [0.0, -0.1, 1.25],
                "rest_tail_world": [0.0, -0.25, 1.20],
                " rest_rot_mat_world": [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
                "rest_rot_mat_world": [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]],
                "limits_rad": [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
                "limits_source": "canonical_default",
            },
            "Prop_sword": {
                "parent": None,
                "kind": "weapon",
                "rest_head_world": [0.5, 0.0, 0.9],
                "rest_tail_world": [0.5, 0.0, 1.3],
            },
        }
        # Hand.L holds an off-axis blade (x = +0.35, y = +0.40) and starts inside Root's capsule
        rest_verts = np.array(
            [
                [-0.2, -0.2, 0.80],
                [0.2, -0.2, 0.95],
                [-0.2, 0.2, 1.15],
                [0.2, 0.2, 1.25],
                [0.05, 0.0, 1.02],
                [0.05, 0.0, 1.05],
                [0.05, 0.0, 1.08],
                [0.40, 0.0, 1.40],  # off-axis weapon tip!
                [0.50, 0.0, 1.00],  # Prop_sword vertex
            ],
            dtype=float,
        )
        skin = {
            "bone_names": ["Root", "Hand.L", "Jaw", "Prop_sword"],
            "rest_verts": rest_verts,
            "skin_indices": np.array([[0], [0], [0], [0], [1], [1], [1], [1], [3]], dtype=int),
            "skin_weights": np.ones((9, 1), dtype=float),
            "rest_heads": np.array([meta[n]["rest_head_world"] for n in ["Root", "Hand.L", "Jaw", "Prop_sword"]]),
            "rest_R": np.array([np.eye(3)] * 4),
            "triangles": np.array([[0, 1, 2], [1, 3, 2]], dtype=int),
        }
        configure_jaw_hinge(meta)
        self.assertEqual(meta["Jaw"]["jaw_hinge"]["axis"], 2)
        self.assertIn("hinge", meta["Jaw"])

        rig = RigModel(meta, names=["Root", "Hand.L", "Jaw", "Prop_sword"], skin=skin)
        self.assertEqual(len(rig.held_guards), 2)
        p0 = rig.start_params()
        X0, J0, _, _ = rig.fk(p0, jac=True)
        samples0, derivs0, _ = rig._guard_samples(X0, J0)
        # Off-axis held tip x-coordinate must be 0.40, not on the Hand.L bone axis (x=0.05)
        self.assertAlmostEqual(float(samples0[-2][0]), 0.40, places=5)
        hand_ry_slot = int(rig.slices["Hand.L"][0][1])
        self.assertGreater(float(np.linalg.norm(derivs0[-2][:, hand_ry_slot])), 0.1)
        # Starting-pose p0 sample inside Root capsule is exempted at p0
        res0, _ = rig.guard(X0, J0)
        self.assertEqual(float(np.max(res0)), 0.0)

        # Sampling Jaw with rolled axis=2 opens channel 2, not channel 0
        import random

        eulers_sampled, _, _ = sample_v3_pose(
            meta, {}, random.Random(7), skin=skin, events={"jaw": True}
        )
        self.assertNotAlmostEqual(eulers_sampled["Jaw"][2], 0.0)

        # Prop_sword below floor must NOT lift Root Z in floor_bend_and_smooth_lift
        seq = np.tile(p0, (4, 1))
        prop_sl = rig.slices["Prop_sword"]
        seq[:, prop_sl.start + 5] = -2.0  # drop prop far below rig.floor
        lifted = floor_bend_and_smooth_lift(rig, seq)
        root_sl = rig.slices["Root"]
        np.testing.assert_allclose(lifted[:, root_sl.start + 5], seq[:, root_sl.start + 5], atol=1e-7)

    def test_review_counterexamples_body_picture_and_geometric_consistency(self):
        from reprojection_solver import RigModel, _apply_body_picture_weights, body_picture
        from studio_camera import Camera
        from video2rig_pipeline import geometric_consistency

        rig = leg_rig()
        root_idx = rig.index["Root"]
        shin_idx = rig.index["Shin.L"]
        # Ortho front camera has +y down, so z maps to -y if R rotates Z->Y, or in identity R camera projects (x, y)
        verts = np.array(
            [
                [-0.25, -0.25, 0.40],
                [0.25, -0.25, 0.40],
                [-0.25, 0.25, 0.40],
                [0.25, 0.25, 0.40],
                [-0.10, 0.0, 0.05],
                [-0.10, 0.0, 0.25],
            ],
            dtype=float,
        )
        skin = {
            "bone_names": rig.names,
            "rest_verts": verts,
            "skin_indices": np.array(
                [[root_idx], [root_idx], [root_idx], [root_idx], [shin_idx], [shin_idx]],
                dtype=int,
            ),
            "skin_weights": np.ones((6, 1), dtype=float),
            "rest_heads": np.array([rig.h[n] for n in rig.names]),
            "rest_R": np.array([rig.R[n] for n in rig.names]),
            "triangles": np.array([[0, 1, 2], [1, 3, 2]], dtype=int),
        }
        rig_skinned = RigModel(rig.meta, rig.names, skin=skin)
        cameras = [
            Camera(np.eye(3), [0.0, 0.0, 3.0], 128, 128, "ortho", 2.0),
            Camera(
                np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
                [3.0, 0.0, 0.0],
                128,
                128,
                "ortho",
                2.0,
            ),
        ]
        p0 = rig_skinned.start_params()
        X0 = rig_skinned.fk(p0)[0]
        uv_t = np.stack([c.project(X0) for c in cameras], axis=0)
        # Put Shin.L tail reading in view 0 onto Root's torso mesh (0.0, 0.15, 0.40) with high confidence 0.8
        shin_tail = 2 * rig_skinned.index["Shin.L"] + 1
        uv_t[0, shin_tail] = cameras[0].project(np.array([0.0, 0.15, 0.40]))
        conf_t = np.full((2, len(rig_skinned.point_names)), 0.8, dtype=float)
        elsewhere, own = body_picture(rig_skinned, cameras, p0, uv_t)
        self.assertTrue(bool(elsewhere[0, shin_tail]))
        w_out = _apply_body_picture_weights(
            rig_skinned, cameras, p0, uv_t, conf_t, conf_t[..., None].copy()
        )
        self.assertLess(float(w_out[0, shin_tail, 0]), 0.2)

        gate = geometric_consistency(cameras, uv_t[None], conf_t[None], 128)
        self.assertIn("vertical_trajectory_corr", gate)

    def test_pass2_review_counterexamples_promoted_hips_surface_contact_and_poses_roundtrip(self):
        from scipy.spatial.transform import Rotation
        from reader_v3 import export_video2rig_npz_labels
        from reading_pipeline import panel_drift
        from reprojection_solver import RigModel
        from studio_camera import Camera
        from surface_contact import apply_surface_floor_constraints
        from video2rig_pipeline import geometric_consistency

        # 1. Promoted Hips beneath unweighted organizational Root in surface_contact
        meta = {
            "Root": {
                "parent": None,
                "kind": "root",
                "rest_head_world": [0.0, 0.0, 0.0],
                "rest_tail_world": [0.0, 0.0, 0.2],
            },
            "Hips": {
                "parent": "Root",
                "kind": "root",
                "rest_head_world": [0.0, 0.0, 0.8],
                "rest_tail_world": [0.0, 0.0, 1.1],
            },
            "Foot.L": {
                "parent": "Hips",
                "kind": "limb",
                "rest_head_world": [0.2, 0.0, 0.1],
                "rest_tail_world": [0.2, -0.15, 0.02],
                "limits_rad": [[-1.0, 1.0]] * 3,
            },
        }
        verts = np.array(
            [
                [0.0, 0.0, 0.85],
                [0.0, 0.0, 1.05],
                [0.2, -0.1, 0.0],
                [0.2, -0.15, 0.02],
            ],
            dtype=float,
        )
        skin = {
            "bone_names": ["Root", "Hips", "Foot.L"],
            "rest_verts": verts,
            "skin_indices": np.array([[1], [1], [2], [2]], dtype=int),
            "skin_weights": np.ones((4, 1), dtype=float),
        }
        rig = RigModel(meta, ["Root", "Hips", "Foot.L"], skin=skin)
        self.assertEqual(rig.motion_roots, {"Hips"})
        p0 = rig.zero()
        frames = rig.frames(p0[None], np.ones((1, 2, len(rig.point_names))))
        cfg = {
            "enabled": True,
            "floor_z": 0.0,
            "support_sets": [
                {
                    "bone": "Foot.L",
                    "samples_local": [((verts[2] - rig.h["Foot.L"]) @ rig.R["Foot.L"]).tolist()],
                }
            ],
        }
        report = apply_surface_floor_constraints(
            frames,
            rig.order,
            "Hips",
            meta,
            rig.h,
            rig.t,
            rig.R,
            {n: meta[n].get("limits_rad", [[-np.pi, np.pi]] * 3) for n in rig.names},
            cfg,
        )
        self.assertTrue(report["contact_quality_passed"])
        self.assertAlmostEqual(frames[0]["bones"]["Root"]["head_3d"][2], 0.0, places=5)
        self.assertAlmostEqual(frames[0]["bones"]["Hips"]["head_3d"][2], 0.8, places=2)

        # 2. Exported poses.npz params round-trip with rotated root rest basis
        R_root = Rotation.from_euler("xyz", [0.4, -0.3, 0.7]).as_matrix()
        meta_rot = {
            "Root": {
                "parent": None,
                "kind": "root",
                "rest_head_world": [0.0, 0.0, 0.8],
                "rest_tail_world": (np.array([0.0, 0.0, 0.8]) + R_root[:, 1] * 0.5).tolist(),
                "rest_rot_mat_world": R_root.tolist(),
            },
            "Shin.L": {
                "parent": "Root",
                "kind": "limb",
                "rest_head_world": [0.2, 0.0, 0.5],
                "rest_tail_world": [0.2, 0.0, 0.1],
                "limits_rad": [[-1.5, 1.5]] * 3,
            },
        }
        rig_rot = RigModel(meta_rot, ["Root", "Shin.L"])
        loc_eul_root = [0.35, -0.25, 0.5]
        loc_eul_shin = [0.2, -0.1, 0.15]
        R_world_root = R_root @ Rotation.from_euler("xyz", loc_eul_root).as_matrix()
        p_true = rig_rot.zero()
        p_true[:3] = Rotation.from_matrix(R_world_root @ R_root.T).as_rotvec()
        p_true[3:6] = [0.1, -0.15, 0.05]
        p_true[6:9] = loc_eul_shin
        X_true = rig_rot.fk(p_true)[0]
        with tempfile.TemporaryDirectory() as tmp:
            rec = {
                "joints": {
                    "Root": {
                        "head_3d": X_true[0].tolist(),
                        "tail_3d": X_true[1].tolist(),
                        "euler_xyz": loc_eul_root,
                    },
                    "Shin.L": {
                        "head_3d": X_true[2].tolist(),
                        "tail_3d": X_true[3].tolist(),
                        "euler_xyz": loc_eul_shin,
                    },
                },
                "v3_sampling": {"translation": [0.1, -0.15, 0.05]},
                "reader_points": {"front": {"visibility": [1.0, 1.0, 1.0, 1.0]}},
            }
            exp = export_video2rig_npz_labels(
                tmp, [rec], meta_rot, ["Root", "Shin.L"], [], {"view_order": ["front"]}
            )
            with np.load(exp["poses_npz"]) as pz:
                X_recon = rig_rot.fk(pz["params"][0])[0]
                np.testing.assert_allclose(X_recon, X_true, atol=1e-5)

        # 3. Panel drift correction before geometric_consistency recovers 30% H shift
        rig_leg = leg_rig()
        cams = [
            Camera(np.eye(3), [0.0, 0.0, 3.0], 256, 256, "ortho", 2.0),
            Camera(
                np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
                [0.0, 0.0, 3.0],
                256,
                256,
                "ortho",
                2.0,
            ),
        ]
        X_leg = rig_leg.fk(rig_leg.zero())[0]
        uv_seq = np.array([[c.project(X_leg) for c in cams] for _ in range(6)], dtype=float)
        conf_seq = np.ones(uv_seq.shape[:-1], dtype=float)
        uv_seq[:, 1, :, 1] += 45.0  # 30% H vertical drift in panel 1
        self.assertFalse(geometric_consistency(cams, uv_seq, conf_seq, 256)["gate_passed"])
        uv_corr, _ = panel_drift(rig_leg, cams, uv_seq, conf_seq)
        self.assertTrue(geometric_consistency(cams, uv_corr, conf_seq, 256)["gate_passed"])


if __name__ == "__main__":
    unittest.main()
