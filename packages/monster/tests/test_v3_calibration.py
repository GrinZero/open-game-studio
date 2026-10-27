"""Known rigid geometry tests for camera alignment and far-limb confidence selection."""

from dataclasses import replace
import copy
import json
from pathlib import Path
import sys
import unittest
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from reading_pipeline import (
    align_frame0_camera_chamfer,
    downweight_contradicting_far_limb,
)
from reprojection_solver import RigModel, exp_so3
from studio_camera import Camera, cameras_from_calibration, triangulate


POINTS = np.array(
    [
        [-0.65, -0.35, 0],
        [0.45, -0.3, 0],
        [0.6, 0.3, 0],
        [-0.4, 0.45, 0],
        [-0.4, -0.25, 1.2],
        [0.25, -0.3, 1],
        [0.38, 0.2, 0.8],
        [-0.3, 0.3, 1.4],
    ]
)
FACES = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]
)


def silhouette(camera):
    out = np.zeros((camera.height, camera.width), np.uint8)
    uv = camera.project(POINTS)
    for face in FACES:
        cv2.fillConvexPoly(out, np.rint(uv[face]).astype(np.int32), 1)
    return out.astype(bool)


def limb_fixture():
    rig = RigModel(
        {
            "Root": dict(
                parent=None,
                kind="root",
                rest_head_world=[0, 0, 0.8],
                rest_tail_world=[0, 0, 1.4],
            ),
            "Foot.L": dict(
                parent="Root",
                kind="limb",
                rest_head_world=[0.25, 0, 0.05],
                rest_tail_world=[0.25, -0.2, 0.05],
            ),
            "Foot.R": dict(
                parent="Root",
                kind="limb",
                rest_head_world=[-0.25, 0, 0.05],
                rest_tail_world=[-0.25, -0.2, 0.05],
            ),
        }
    )
    cameras = list(
        cameras_from_calibration(
            {
                "center": [0, 0, 0.8],
                "panel_width": 448,
                "panel_height": 448,
                "ortho_scale": 3,
            }
        ).values()
    )
    X = rig.fk(rig.zero())[0]
    uv = np.array([[c.project(X) for c in cameras]])
    return rig, cameras, uv


class CameraAlignmentTests(unittest.TestCase):
    def test_recovers_known_six_dof_perspective_offset_and_ortho_projection(self):
        for kind in ("ortho", "persp"):
            with self.subTest(camera=kind):
                camera = Camera(
                    exp_so3([0.15, -0.1, 0.08]),
                    np.array([-0.03, 0.04, 3]),
                    128,
                    128,
                    kind,
                    2.5,
                    170,
                )
                E = exp_so3([0.035, -0.025, 0.045])
                target = replace(
                    camera, R=E @ camera.R, t=E @ camera.t + [0.06, -0.035, 0.045]
                )
                mask = silhouette(target)
                fitted, report = align_frame0_camera_chamfer(
                    camera, POINTS, mask, FACES, maxiter=20
                )
                self.assertTrue(report["accepted"])
                self.assertLess(report["silhouette_iou_before"], 0.7)
                self.assertGreater(report["silhouette_iou_after"], 0.99)
                self.assertLess(report["chamfer_px_after"], 0.1)
                self.assertLess(
                    np.linalg.norm(
                        fitted.project(POINTS) - target.project(POINTS), axis=-1
                    ).mean(),
                    1,
                )
                if kind == "ortho":
                    self.assertEqual(fitted.t[2], camera.t[2])
                    self.assertEqual(report["unobservable_parameters"], ["tz"])
                else:
                    self.assertEqual(len(report["optimized_parameters"]), 6)
                    self.assertLess(abs(fitted.t[2] - target.t[2]), 0.02)
                json.dumps(report, allow_nan=False)

    def test_already_aligned_and_unobservable_ortho_depth_do_not_drift(self):
        camera = Camera(
            exp_so3([0.15, -0.1, 0.08]), [-0.03, 0.04, 3], 128, 128, "ortho", 2.5
        )
        mask = silhouette(camera)
        deeper = replace(camera, t=camera.t + [0, 0, 7])
        np.testing.assert_array_equal(mask, silhouette(deeper))
        fitted, report = align_frame0_camera_chamfer(
            deeper, POINTS, mask, FACES, maxiter=5
        )
        self.assertIs(fitted, deeper)
        self.assertFalse(report["accepted"])
        self.assertEqual(report["before"], 0)
        self.assertEqual(report["after"], 0)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            align_frame0_camera_chamfer(camera, POINTS, np.zeros_like(mask), FACES)

    def test_more_than_5000_faces_remain_filled_silhouettes(self):
        camera = Camera(
            exp_so3([0.15, -0.1, 0.08]), [-0.03, 0.04, 3], 128, 128, "ortho", 2.5
        )
        faces = np.tile(FACES, (418, 1))
        fitted, report = align_frame0_camera_chamfer(
            camera, POINTS, silhouette(camera), faces, maxiter=1
        )
        self.assertIs(fitted, camera)
        self.assertEqual(report["rasterized_faces"], 5016)
        self.assertEqual(report["silhouette_iou_before"], 1)
        self.assertEqual(report["chamfer_px_before"], 0)


class FarLimbSelectionTests(unittest.TestCase):
    def test_only_contradictory_far_branch_loses_weight(self):
        rig, cameras, uv = limb_fixture()
        original = uv.copy()
        conf = np.ones(uv.shape[:-1]) * 0.9
        for name in ("Foot.L", "Foot.R", "Root"):
            point = 2 * rig.index[name] + 1
            uv[0, 1, point, 1] += 90
            conf[0, 1, point] = 0.4
        saved = conf.copy()
        heights = [rig.height_px(c) for c in cameras]
        result = downweight_contradicting_far_limb(rig, cameras, uv, conf, heights)
        far = 2 * rig.index["Foot.R"] + 1
        self.assertAlmostEqual(result[0, 1, far], 0.04)
        expected = saved.copy()
        expected[0, 1, far] = 0.04
        np.testing.assert_allclose(result, expected)
        np.testing.assert_array_equal(conf, saved)
        self.assertFalse(
            np.array_equal(uv, original)
        )  # fixture contains actual conflicting pixels

    def test_confidence_and_twelve_percent_boundary_preserve_valid_readings(self):
        rig, cameras, base = limb_fixture()
        point = 2 * rig.index["Foot.R"] + 1
        H = rig.height_px(cameras[1])
        for front, side, residual_share, expected_factor in [
            (0.9, 0.4, 0.12, 1),
            (0.9, 0.4, 0.12001, 0.1),
            (0.7, 0.4, 0.2, 1),
            (0.9, 0.5, 0.2, 1),
        ]:
            with self.subTest(front=front, side=side, residual=residual_share):
                uv = base.copy()
                conf = np.ones(uv.shape[:-1])
                conf[0, 0, point], conf[0, 1, point] = front, side
                # Weighted orthographic ray intersection gives this exact shared-axis residual.
                uv[0, 1, point, 1] += residual_share * H * (1 + (side / front) ** 2)
                X = triangulate(cameras, uv[0], conf[0])
                actual = (
                    np.linalg.norm(cameras[1].project(X)[point] - uv[0, 1, point]) / H
                )
                self.assertAlmostEqual(actual, residual_share, places=12)
                result = downweight_contradicting_far_limb(rig, cameras, uv, conf, H)
                self.assertAlmostEqual(result[0, 1, point], side * expected_factor)

    def test_turning_reverses_far_side_and_unclear_primary_cannot_veto(self):
        rig, cameras, _ = limb_fixture()
        p = rig.zero()
        p[2] = np.pi
        uv = np.array([[c.project(rig.fk(p)[0]) for c in cameras]])
        conf = np.ones(uv.shape[:-1]) * 0.9
        for name in ("Foot.L", "Foot.R"):
            point = 2 * rig.index[name] + 1
            uv[0, 1, point, 1] += 90
            conf[0, 1, point] = 0.4
        result = downweight_contradicting_far_limb(
            rig, cameras, uv, conf, rig.height_px(cameras[1])
        )
        self.assertAlmostEqual(result[0, 1, 2 * rig.index["Foot.L"] + 1], 0.04)
        self.assertAlmostEqual(result[0, 1, 2 * rig.index["Foot.R"] + 1], 0.4)
        conf[:, 0] = 0.3
        np.testing.assert_array_equal(
            downweight_contradicting_far_limb(
                rig, cameras, uv, conf, rig.height_px(cameras[1])
            ),
            conf,
        )

    def test_perspective_pixel_height_uses_depth_and_excludes_distant_props(self):
        rig, cameras, _ = limb_fixture()
        c = replace(cameras[0], type="persp", focal_px=500)
        height = rig.height_px(c)
        shifted = replace(
            c, t=c.t + [0, 0, float(c.depth(rig.fk(rig.zero())[0].mean(0)))]
        )
        self.assertAlmostEqual(rig.height_px(shifted), height / 2)
        meta = copy.deepcopy(rig.meta)
        meta["Prop_remote"] = dict(
            parent=None,
            kind="root",
            rest_head_world=[0, 100, 0.6],
            rest_tail_world=[0, 100, 0.9],
        )
        self.assertAlmostEqual(RigModel(meta).height_px(c), height)


if __name__ == "__main__":
    unittest.main()
