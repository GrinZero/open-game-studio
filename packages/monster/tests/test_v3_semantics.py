"""Behavioral counterexamples for geometry, timing, and image-space calibration."""

from pathlib import Path
import sys
import unittest
import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from reading_pipeline import cadence_from_flow_steps
from skeleton_postprocess import Shell, cross_section_flatness, jaw_from_mouth_volume
from studio_camera import Camera
from reprojection_solver import RigModel


class CadenceTests(unittest.TestCase):
    def test_real_acceleration_and_one_strike_do_not_retime_the_clip(self):
        speeds = np.linspace(0.4, 6, 120)
        speeds[55] *= 3
        result = cadence_from_flow_steps(speeds, 60)
        np.testing.assert_allclose(result["frame_times"], np.arange(121))
        self.assertIsNone(result["period"])
        self.assertEqual(result["content_fps"], 60)

    def test_periodic_drops_are_separate_from_the_local_motion_trend(self):
        for period, rate in ((4, 30), (24, 25)):
            speeds = np.linspace(1, 3, 120)
            phase = period - 1
            speeds[np.arange(120) % period == phase] *= 2
            result = cadence_from_flow_steps(speeds, 24)
            self.assertEqual((result["period"], result["phase"]), (period, phase))
            expected = np.ones(120) * 24 / rate
            expected[np.arange(120) % period == phase] *= 2
            np.testing.assert_allclose(np.diff(result["frame_times"]), expected)
            self.assertTrue(any(t % 1 for t in result["frame_times"]))

    def test_no_motion_and_empty_steps_are_finite(self):
        np.testing.assert_array_equal(
            cadence_from_flow_steps(np.zeros(10))["frame_times"], np.arange(11)
        )
        self.assertEqual(cadence_from_flow_steps([])["frame_times"], [0])
        with self.assertRaises(ValueError):
            cadence_from_flow_steps([1], 0)


def volume(solid, step=0.01, origin=(0, 0, 0)):
    return Shell(
        solid.copy(),
        solid,
        np.array(origin, float),
        step,
        distance_transform_edt(solid),
    )


class VolumeTests(unittest.TestCase):
    def test_cross_sections_distinguish_thin_wing_and_round_arm(self):
        for thickness, wing in ((3, True), (17, False)):
            voxels = np.zeros((65, 110, 105), bool)
            voxels[:5, :5, :] = (
                True  # Character H is one metre, independent of wing thickness.
            )
            # Same length and broad sweep: only local thickness changes.
            for y in range(10, 95):
                center = 32 + int(7 * np.sin(y / 20))
                voxels[
                    center - 15 : center + 16,
                    y,
                    50 - thickness // 2 : 51 + thickness // 2,
                ] = True
            result = cross_section_flatness(
                volume(voxels), np.array([0.32, 0.10, 0.5]), np.array([0.32, 0.94, 0.5])
            )
            self.assertEqual(result > 5, wing, result)

    def test_open_mouth_needs_both_lips_and_a_front_open_cavity(self):
        solid = np.zeros((31, 81, 81), bool)
        solid[5:26, 15:66, 20:61] = True
        head = np.array([0, 0.55, 0.40])
        tip = np.array([0, 0.16, 0.40])
        # Closed low-poly surface: vertex sparsity has no bearing on a volume query.
        self.assertIsNone(
            jaw_from_mouth_volume(volume(solid, origin=(-0.15, 0, 0)), head, tip, 0)
        )
        solid[5:26, 15:47, 35:43] = False
        result = jaw_from_mouth_volume(
            volume(solid, origin=(-0.15, 0, 0)), head, tip, 0
        )
        self.assertIsNotNone(result)
        hinge, chin = result
        self.assertGreater(hinge[1], chin[1])
        self.assertLess(chin[2], hinge[2])
        solid[5:26, 15:18, 35:43] = (
            True  # Close the front: internal cavity is not a mouth.
        )
        self.assertIsNone(
            jaw_from_mouth_volume(volume(solid, origin=(-0.15, 0, 0)), head, tip, 0)
        )


class LimbAndTailTests(unittest.TestCase):
    def test_free_limb_starts_at_shared_branch_without_arbitrary_third(self):
        from skeleton_postprocess import free_limb_paths
        from unittest.mock import Mock

        trunk = np.array([[0, 0, z] for z in range(12)])
        left = np.array([[x, 0, 11] for x in range(1, 18)])
        right = np.array([[-x, 0, 11] for x in range(1, 18)])
        shell = Mock(step=1)
        shell.indices.side_effect = lambda points: np.asarray(points, int)
        shell.radius.side_effect = lambda points: np.ones(len(points)) * 2
        paths = free_limb_paths(shell, [np.r_[trunk, left], np.r_[trunk, right]])
        np.testing.assert_array_equal(paths[0][0], trunk[-1])
        np.testing.assert_array_equal(paths[0][-1], left[-1])
        np.testing.assert_array_equal(paths[1][0], trunk[-1])

    def test_hind_hips_reference_knee_spine_and_body_radius(self):
        from skeleton_postprocess import align_quadruped_hips
        from unittest.mock import Mock

        meta = {
            "Hips": {
                "kind": "spine",
                "rest_head_world": [0, 0, 1],
                "rest_tail_world": [0, 1, 1.2],
            },
            "Tail1": {
                "parent": "Hips",
                "rest_head_world": [0, 0.8, 1.16],
                "rest_tail_world": [0, 1.2, 1.1],
            },
            "Thigh.L": {
                "rest_head_world": [0.2, -0.1, 0.8],
                "rest_tail_world": [0.2, 0.6, 0.4],
            },
            "UpperArm.L": {
                "rest_head_world": [0.2, -0.5, 0.8],
                "rest_tail_world": [0.2, -0.6, 0.4],
            },
        }
        arm = list(meta["UpperArm.L"]["rest_head_world"])
        shell = Mock()
        shell.radius.return_value = np.array([0.2])
        self.assertEqual(align_quadruped_hips(meta, shell, 1), ["Thigh.L"])
        np.testing.assert_allclose(
            meta["Thigh.L"]["rest_head_world"], [0.2, 0.6, 1.12 - 0.25 * 0.2]
        )
        self.assertEqual(meta["UpperArm.L"]["rest_head_world"], arm)

    def test_curved_multisegment_tail_resamples_arc_and_keeps_attachment(self):
        from skeleton_postprocess import refine_long_tail

        meta = {
            "Hips": {
                "kind": "root",
                "rest_head_world": [0, 0, 0],
                "rest_tail_world": [0, 0, 1],
            },
            "Tail1": {
                "parent": "Hips",
                "kind": "tail",
                "rest_head_world": [0, 0, 1],
                "rest_tail_world": [0, 0.8, 1],
            },
            "Tail2": {
                "parent": "Tail1",
                "kind": "tail",
                "rest_head_world": [0, 0.8, 1],
                "rest_tail_world": [0.7, 1.6, 1],
            },
            "Rigid1": {
                "parent": "Tail2",
                "kind": "rigid",
                "rest_head_world": [0.7, 1.6, 1],
                "rest_tail_world": [0.7, 1.6, 1.2],
            },
        }
        report = refine_long_tail(meta, 1)
        self.assertEqual(len(report["new_names"]), 10)
        self.assertEqual(meta["Tail1"]["parent"], "Hips")
        self.assertEqual(meta["Rigid1"]["parent"], "Tail10")
        np.testing.assert_allclose(meta["Tail10"]["rest_tail_world"], [0.7, 1.6, 1])
        lengths = [
            np.linalg.norm(
                np.array(meta[n]["rest_tail_world"]) - meta[n]["rest_head_world"]
            )
            for n in report["new_names"]
        ]
        self.assertGreater(
            min(lengths), 0.16
        )  # Arc subdivision respects the ten-channel cap.

    def test_long_rear_hips_are_shortened_and_split_without_existing_tail(self):
        from skeleton_postprocess import refine_long_tail

        meta = {
            "Hips": {
                "kind": "root",
                "rest_head_world": [0, 0, 1],
                "rest_tail_world": [0, 2, 1],
            },
            "Thigh.L": {
                "parent": "Hips",
                "rest_head_world": [0.1, 0.1, 1],
                "rest_tail_world": [0.1, 0.1, 0],
            },
            "Thigh.R": {
                "parent": "Hips",
                "rest_head_world": [-0.1, 0.1, 1],
                "rest_tail_world": [-0.1, 0.1, 0],
            },
        }
        report = refine_long_tail(meta, 1)
        self.assertEqual(report["axial"], "Hips")
        self.assertEqual(len(report["new_names"]), 10)
        np.testing.assert_allclose(meta["Hips"]["rest_tail_world"], [0, 0.18, 1])
        self.assertEqual(meta["Thigh.L"]["parent"], "Hips")
        np.testing.assert_allclose(meta["Tail10"]["rest_tail_world"], [0, 2, 1])


class TrainingGeometryTests(unittest.TestCase):
    def test_context_is_two_articulated_poses_with_constant_bone_lengths(self):
        from reader_v3 import preceding_pose_points

        meta = {
            "Root": {
                "parent": None,
                "kind": "root",
                "rest_head_world": [0, 0, 0],
                "rest_tail_world": [0, 0, 1],
            },
            "Arm.L": {
                "parent": "Root",
                "kind": "limb",
                "rest_head_world": [0, 0, 1],
                "rest_tail_world": [1, 0, 1],
            },
        }
        rig = RigModel(meta)
        joints = {
            n: {"euler_xyz": [0.1, 0.2, 0.3], "head_3d": rig.h[n].tolist()}
            for n in rig.names
        }
        X = preceding_pose_points(rig, joints, np.random.default_rng(12))
        self.assertEqual(X.shape, (2, 4, 3))
        self.assertGreater(np.linalg.norm(X[0] - X[1]), 0.01)
        for i, n in enumerate(rig.names):
            np.testing.assert_allclose(
                np.linalg.norm(X[:, 2 * i + 1] - X[:, 2 * i], axis=-1),
                np.linalg.norm(rig.t[n] - rig.h[n]),
                atol=1e-12,
            )
        # Attached joint stays attached across both perturbations.
        np.testing.assert_allclose(
            X[:, 0], X[:, 3], atol=1e-12
        )  # sorted Arm.L then Root

    def test_dense_center_follows_world_translation_in_both_views(self):
        from video2rig_pipeline import multiview_dense_centers
        from studio_camera import cameras_from_calibration
        from reprojection_solver import aligned_rest_rotation

        meta = {
            "Root": {
                "parent": None,
                "kind": "root",
                "rest_head_world": [0, 0, 0],
                "rest_tail_world": [0, 0, 1],
            },
            "Socket.L": {
                "parent": "Root",
                "kind": "rigid",
                "fixed_local": True,
                "rest_head_world": [0.2, 0, 0.7],
                "rest_tail_world": [0.3, 0, 0.7],
            },
            "Socket.R": {
                "parent": "Root",
                "kind": "rigid",
                "fixed_local": True,
                "rest_head_world": [-0.2, 0, 0.7],
                "rest_tail_world": [-0.3, 0, 0.7],
            },
        }
        for b in meta.values():
            b["rest_rot_mat_world"] = aligned_rest_rotation(
                np.array(b["rest_head_world"]), np.array(b["rest_tail_world"])
            ).tolist()
        names = sorted(meta)
        vertices = np.array([[-0.1, 0, 0], [0.1, 0, 0], [0, 0.2, 1], [0, -0.2, 1]])
        skin = {
            "bone_names": names,
            "rest_verts": vertices,
            "skin_indices": np.full((4, 1), names.index("Root")),
            "skin_weights": np.ones((4, 1)),
            "rest_heads": np.array([meta[n]["rest_head_world"] for n in names]),
            "rest_R": np.array([meta[n]["rest_rot_mat_world"] for n in names]),
            "height": 1.0,
        }
        rig = RigModel(meta, names, skin, [])
        cameras = list(
            cameras_from_calibration(
                {
                    "center": [0, 0, 0.5],
                    "panel_width": 448,
                    "panel_height": 448,
                    "ortho_scale": 3,
                }
            ).values()
        )
        shifts = np.array([[0, 0, 0], [0.2, 0.7, 0.15], [-0.4, 1.1, 0.3]])
        parameters = np.zeros((3, rig.size))
        parameters[:, 3:6] = shifts
        uv = np.array([[c.project(rig.fk(p)[0]) for c in cameras] for p in parameters])
        centers = multiview_dense_centers(
            rig, cameras, uv, np.ones(uv.shape[:-1]), skin
        )
        np.testing.assert_allclose(centers, vertices.mean(0) + shifts, atol=1e-12)

    def test_empty_root_gives_pelvis_world_channels_without_redundant_joint(self):
        meta = {
            "Root": {
                "parent": None,
                "kind": "root",
                "rest_head_world": [0, 0, 0],
                "rest_tail_world": [0, 0, 0.5],
            },
            "Hips": {
                "parent": "Root",
                "kind": "root",
                "rest_head_world": [0, 0, 0.5],
                "rest_tail_world": [0, 0, 1],
            },
            "Unused": {
                "parent": None,
                "kind": "root",
                "rest_head_world": [5, 0, 0],
                "rest_tail_world": [5, 0, 1],
            },
        }
        names = sorted(meta)
        skin = {
            "bone_names": names,
            "rest_verts": np.array([[0, 0, 0], [0, 0, 1]]),
            "skin_indices": np.full((2, 1), names.index("Hips")),
            "skin_weights": np.ones((2, 1)),
        }
        rig = RigModel(meta, names, skin, [])
        self.assertEqual(rig.size, 6)
        self.assertEqual(set(rig.slices), {"Hips"})
        p = np.array([0.2, -0.3, 0.1, 0.1, 0.2, 0.3])
        points, J, heads, _ = rig.fk(p, True)
        np.testing.assert_allclose(heads["Hips"], np.array([0, 0, 0.5]) + p[3:])
        np.testing.assert_allclose(heads["Root"], [0, 0, 0])
        numeric = np.stack(
            [
                (
                    rig.fk(p + np.eye(6)[i] * 1e-6)[0]
                    - rig.fk(p - np.eye(6)[i] * 1e-6)[0]
                )
                / 2e-6
                for i in range(6)
            ],
            -1,
        )
        np.testing.assert_allclose(J, numeric, atol=1e-9)

    def test_perspective_crop_extent_uses_current_depth(self):
        from reader_v3 import projected_body_height

        c = Camera(np.eye(3), np.zeros(3), 896, 896, "persp", focal_px=700)
        self.assertEqual(projected_body_height(c, 2, [0, 0, 10]), 140)
        self.assertEqual(projected_body_height(c, 2, [0, 0, 5]), 280)

    def test_foot_shadow_is_at_contact_and_applied_in_rgb_space(self):
        import torch
        from reader_v3 import inject_contact_shadow_gpu

        image = torch.zeros(
            1, 3, 101, 101
        )  # normalized: zero means RGB mean, not black.
        feet = torch.tensor([[[0.25, 0.55]]])
        valid = torch.ones(1, 1)
        output = inject_contact_shadow_gpu(
            image, feet, valid, torch.tensor([0.6]), probability=1
        )
        self.assertLess(float(output[0, 0, 55, 25]), -0.6)
        self.assertAlmostEqual(float(output[0, 0, 91, 50]), 0, places=6)
        np.testing.assert_array_equal(
            inject_contact_shadow_gpu(image, feet, valid * 0, probability=1), image
        )
        self.assertTrue(torch.equal(image, torch.zeros_like(image)))

    def test_smear_extent_scales_with_body_height(self):
        import torch
        from reader_v3 import smear_limbs_gpu

        image = torch.zeros(1, 3, 128, 128)
        image[:, :, 60:68, 60:68] = 1
        mask = torch.zeros(1, 1, 128, 128)
        mask[:, :, 60:68, 60:68] = 1
        widths = []
        for height in (0.25, 0.75):
            result = smear_limbs_gpu(
                image,
                mask,
                probability=1,
                generator=torch.Generator().manual_seed(9),
                body_height=torch.tensor([height]),
            )
            columns = torch.where(result[0].abs().sum((0, 1)) > 1e-4)[0]
            widths.append(int(columns[-1] - columns[0]))
        self.assertGreater(widths[1], widths[0] * 1.8)

    def test_context_disabled_is_zero_and_uses_previous_pose_when_enabled(self):
        import torch
        from unittest.mock import patch
        from reader_v3 import temporal_training_context

        coords = torch.ones(1, 1, 2) * 0.2
        previous = torch.ones(1, 2, 1, 2) * 0.7
        self.assertEqual(
            float(
                temporal_training_context(
                    coords, torch.ones(1, 1), False, previous
                ).sum()
            ),
            0,
        )
        with patch("reader_v3.CONTEXT_SHARE", 1):
            context = temporal_training_context(
                coords, torch.ones(1, 1), True, previous
            )
        peak = int(context[0, 0].argmax())
        self.assertGreater(peak // 128, 75)
        self.assertGreater(peak % 128, 75)


if __name__ == "__main__":
    unittest.main()
