"""Observable solver counterexamples: turns, hidden feet, smooth tracks and contacts."""

import copy
import math
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from reading_pipeline import at_limit
from reprojection_solver import (
    RigModel,
    _apply_body_picture_weights,
    aligned_rest_rotation,
    exp_so3,
    fit_clip_2d_reprojection,
)
from studio_camera import cameras_from_calibration


def cameras(width=448):
    return list(
        cameras_from_calibration(
            {
                "center": [0, 0, 1],
                "panel_width": width,
                "panel_height": width,
                "ortho_scale": 3,
            }
        ).values()
    )


def rig_from(meta):
    for bone in meta.values():
        bone["rest_rot_mat_world"] = aligned_rest_rotation(
            np.array(bone["rest_head_world"]), np.array(bone["rest_tail_world"])
        ).tolist()
    return RigModel(meta)


def radial_rig(foot=False):
    # Direct root anchors are collinear; only descendant endpoints observe yaw.
    # A 180-degree turn is a genuine stationary point of the robust local fit.
    meta = {
        "Root": dict(
            parent=None,
            kind="root",
            rest_head_world=[0, 0, 0.6],
            rest_tail_world=[0, 0, 1.2],
        ),
        "Spine1": dict(
            parent="Root",
            kind="spine",
            fixed_local=True,
            rest_head_world=[0, 0, 1.2],
            rest_tail_world=[0, 0, 1.6],
        ),
    }
    for i, angle in enumerate(np.arange(8) * math.pi / 4):
        h = np.array([0.4 * np.cos(angle), 0.4 * np.sin(angle), 1.2])
        meta[f"Rigid{i + 1}"] = dict(
            parent="Spine1",
            kind="rigid",
            fixed_local=True,
            rest_head_world=h.tolist(),
            rest_tail_world=(
                h + [0.3 * np.cos(angle), 0.3 * np.sin(angle), 0.15]
            ).tolist(),
        )
    if foot:
        meta["Foot.L"] = dict(
            parent="Spine1",
            kind="limb",
            rest_head_world=[0.2, 0, 0.05],
            rest_tail_world=[0.2, -0.3, 0.05],
            limits_rad=[[0, 0], [-1.2, 1.2], [0, 0]],
        )
        meta["Toe1_1.L"] = dict(
            parent="Foot.L",
            kind="rigid",
            fixed_local=True,
            rest_head_world=[0.32, -0.1, 0.05],
            rest_tail_world=[0.32, -0.2, 0.05],
        )
    return rig_from(meta)


def observations(rig, cams, params):
    return np.array([[camera.project(rig.fk(p)[0]) for camera in cams] for p in params])


class SolverBehaviorTests(unittest.TestCase):
    def test_prior_capsule_occlusion_keeps_confident_current_readings(self):
        rig = rig_from(
            {
                "Root": dict(
                    parent=None,
                    kind="root",
                    rest_head_world=[0, 0, 0.2],
                    rest_tail_world=[0, 0, 1.4],
                ),
                "Hand.L": dict(
                    parent="Root",
                    kind="limb",
                    rest_head_world=[0, 0.3, 0.8],
                    rest_tail_world=[0, 0.4, 0.9],
                ),
            }
        )
        rig.capsules = [("Root", 0.25)]
        cams = cameras()
        X = rig.fk(rig.zero())[0]
        uv = np.array([c.project(X) for c in cams])
        key = 2 * rig.index["Hand.L"]
        for confidence, other, factor in [
            (1, 1, 1),
            (0.5, 1, 1),
            (0.4999, 1, 0.25),
            (0.1, 1, 1),
            (0.4, 0.49, 1),
        ]:
            conf = np.ones(uv.shape[:-1])
            conf[0, key] = confidence
            conf[1, key] = other
            result = _apply_body_picture_weights(
                rig, cams, X, uv, conf, conf[..., None]
            )
            self.assertAlmostEqual(result[0, key, 0], confidence * factor)
            self.assertEqual(result[1, key, 0], other)
            self.assertEqual(conf[0, key], confidence)

    def test_half_turn_requires_and_wins_five_escape_starts(self):
        rig, cams = radial_rig(), cameras()
        p = rig.zero()
        q = p.copy()
        q[2] = np.pi
        uv = observations(rig, cams, [p, q])
        result, report = fit_clip_2d_reprojection(rig, cams, uv, np.ones(uv.shape[:-1]))
        frame = report["frames"][1]
        self.assertGreater(frame["pre_turn_error_px"], 8)
        self.assertEqual(
            {s for s in frame["starts"] if s.startswith("turn_")},
            {"turn_-45", "turn_45", "turn_-90", "turn_90", "turn_180"},
        )
        self.assertTrue(frame["start"].startswith("turn_"))
        costs = frame["candidate_costs"]
        self.assertLess(
            costs[frame["start"]], 0.001 * min(costs["prev"], costs["read"])
        )
        self.assertLess(frame["median_error_px"], 0.001)
        np.testing.assert_allclose(exp_so3(result[1, :3]), exp_so3(q[:3]), atol=1e-6)
        self.assertFalse(
            any(s.startswith("turn_") for s in report["frames"][0]["starts"])
        )

    def test_hidden_foot_roll_flattens_on_landing_and_is_free_in_air(self):
        rig, cams = radial_rig(foot=True), cameras()
        self.assertEqual(rig.extra, [])  # flat feet must work without surface labels
        twist = rig.slices["Foot.L"][0][0]
        air = rig.zero()
        air[twist], air[5] = 0.7, 0.3
        landing = air.copy()
        landing[5] = 0
        uv = observations(rig, cams, [air, landing, air])
        conf = np.ones(uv.shape[:-1])
        toe = 2 * rig.index["Toe1_1.L"]
        conf[1, :, toe : toe + 2] = 0  # no roll information when the foot lands
        unconstrained, _ = fit_clip_2d_reprojection(rig, cams, uv, conf, flat_feet=0)
        result, report = fit_clip_2d_reprojection(rig, cams, uv, conf)
        self.assertGreater(unconstrained[1, twist], 0.65)
        self.assertLess(abs(result[1, twist]), 0.001)
        np.testing.assert_allclose(result[[0, 2], twist], 0.7, atol=2e-5)
        self.assertEqual(
            [d["planted_feet"] for d in report["frames"]], [[], ["Foot.L"], []]
        )
        self.assertLess(max(d["median_error_px"] for d in report["frames"]), 0.001)

    def test_vertical_hoof_is_not_forced_to_be_a_flat_foot(self):
        original = radial_rig(foot=True)
        meta = copy.deepcopy(original.meta)
        meta["Foot.L"]["rest_tail_world"] = [0.2, 0, 0.3]
        rig = rig_from(meta)
        p = rig.zero()
        uv = observations(rig, cameras(), [p])
        _, report = fit_clip_2d_reprojection(rig, cameras(), uv, np.ones(uv.shape[:-1]))
        self.assertEqual(report["frames"][0]["planted_feet"], [])

    def test_observed_foot_roll_near_floor_overrides_flat_contact_prior(self):
        rig, cams = radial_rig(foot=True), cameras()
        twist = rig.slices["Foot.L"][0][0]
        target = rig.zero()
        target[twist] = 0.7
        uv = observations(rig, cams, [target])
        result, report = fit_clip_2d_reprojection(rig, cams, uv, np.ones(uv.shape[:-1]))
        self.assertEqual(report["frames"][0]["planted_feet"], [])
        self.assertAlmostEqual(result[0, twist], 0.7, places=4)

    def test_lateral_vector_jacobian_in_rotated_ancestor_frames(self):
        meta = copy.deepcopy(radial_rig(foot=True).meta)
        meta["Spine1"].update(
            kind="spine", fixed_local=False, limits_rad=[[-1, 1], [0.15, 0.15], [-1, 1]]
        )
        meta["Foot.L"]["limits_rad"] = [[-0.8, 0.8]] * 3
        rig = rig_from(meta)
        p = np.linspace(-0.24, 0.31, rig.size)
        vector = np.array([0.7, -0.2, 0.4])
        rotations = rig.fk(p)[2]
        actual, jac = rig.vector_jacobian(p, rotations, "Foot.L", vector)
        numeric = np.stack(
            [
                (
                    rig.fk(p + np.eye(rig.size)[k] * 1e-6)[2]["Foot.L"] @ vector
                    - rig.fk(p - np.eye(rig.size)[k] * 1e-6)[2]["Foot.L"] @ vector
                )
                / 2e-6
                for k in range(rig.size)
            ],
            -1,
        )
        np.testing.assert_allclose(actual, rotations["Foot.L"] @ vector, atol=1e-12)
        np.testing.assert_allclose(jac, numeric, atol=3e-9)
        np.testing.assert_array_equal(
            jac[:, 3:6], 0
        )  # a direction ignores root translation

    def test_sure_smooth_view_overcomes_jittering_contradictory_panel(self):
        rig = rig_from(
            {
                "Root": dict(
                    parent=None,
                    kind="root",
                    rest_head_world=[0, 0, 0.6],
                    rest_tail_world=[0, 0, 1.2],
                )
            }
        )
        cams = cameras()
        uv = observations(rig, cams, [rig.zero()] * 9)
        conf = np.zeros(uv.shape[:-1])
        conf[:, :, 0] = 1
        conf[:, 0, 0] = 0.65
        for t in range(9):
            uv[t, 0, 0, 1] -= 448 / 3 * (0.3 if t < 2 else 0)
            uv[t, 1, 0, 1] -= (
                448 / 3 * (0.3 if t < 2 else (0.36 if t % 2 == 0 else 0.15))
            )
        normal, _ = fit_clip_2d_reprojection(rig, cams, uv, conf, stable_scale=1)
        widened, report = fit_clip_2d_reprojection(rig, cams, uv, conf)
        self.assertGreater(np.mean(abs(normal[3:, 5])), 0.2)
        self.assertLess(np.max(abs(widened[3:, 5])), 0.04)
        self.assertTrue(all(d["stable_points"] == 1 for d in report["frames"][3:]))
        # Resolution changes preserve the geometric smoothness classification.
        scaled = observations(rig, cameras(896), [rig.zero()] * 9)
        scaled += 2 * (uv - observations(rig, cams, [rig.zero()] * 9))
        _, report2 = fit_clip_2d_reprojection(rig, cameras(896), scaled, conf)
        self.assertEqual(
            [d["stable_points"] for d in report["frames"]],
            [d["stable_points"] for d in report2["frames"]],
        )

    def test_long_total_occlusion_holds_pose_and_reacquires_without_poisoning(self):
        rig, cams = radial_rig(), cameras()
        initial = rig.zero()
        initial[:3] = [0.1, -0.05, 0.4]
        initial[3:6] = [0.1, -0.04, 0.12]
        final = initial.copy()
        final[2] = 1.3
        final[3] = 0.2
        uv = observations(rig, cams, [initial] * 36 + [final] * 2)
        conf = np.ones(uv.shape[:-1])
        conf[1:36] = 0
        uv[1:36] = np.nan
        result, report = fit_clip_2d_reprojection(rig, cams, uv, conf)
        np.testing.assert_allclose(
            result[1:36], np.tile(result[0], (35, 1)), atol=1e-10
        )
        for frame in report["frames"][1:36]:
            self.assertEqual(frame["stable_points"], 0)
            self.assertIsNone(frame["median_error_px"])
            self.assertFalse(any(s.startswith("turn_") for s in frame["starts"]))
        self.assertLess(report["frames"][36]["median_error_px"], 0.001)
        np.testing.assert_allclose(rig.fk(result[36])[0], rig.fk(final)[0], atol=1e-5)

    def test_complete_fit_exits_capsule_centreline_and_keeps_floor(self):
        rig = rig_from(
            {
                "Root": dict(
                    parent=None,
                    kind="root",
                    rest_head_world=[0, 0, 0],
                    rest_tail_world=[0, 0, 1],
                ),
                "Spine1": dict(
                    parent="Root",
                    kind="spine",
                    fixed_local=True,
                    rest_head_world=[0, 0, 1],
                    rest_tail_world=[0, 0, 1.3],
                ),
                "Arm.L": dict(
                    parent="Spine1",
                    kind="limb",
                    rest_head_world=[-0.7, 0, 0.5],
                    rest_tail_world=[0.7, 0, 0.5],
                    limits_rad=[[-1.4, 1.4]] * 3,
                ),
            }
        )
        rig.capsules = [("Root", 0.25)]
        uv = observations(rig, cameras(), [rig.zero()])
        result, report = fit_clip_2d_reprojection(
            rig, cameras(), uv, np.ones(uv.shape[:-1]), max_nfev=150
        )
        X = rig.fk(result[0])[0]
        body = 2 * rig.index["Root"]
        limb = 2 * rig.index["Arm.L"]
        a, b = X[body : body + 2]
        midpoint = X[limb : limb + 2].mean(0)
        nearest = a + np.clip((midpoint - a) @ (b - a) / np.dot(b - a, b - a), 0, 1) * (
            b - a
        )
        self.assertGreater(np.linalg.norm(midpoint - nearest), 0.25 - 1e-4 * rig.height)
        self.assertLess(report["frames"][0]["median_error_px"], 1)
        # Even a confidently read location below the floor cannot drag the whole body down.
        uv2 = observations(rig, cameras(), [rig.zero()])
        uv2[..., 1] += 448 / 3 * 0.1
        result2, _ = fit_clip_2d_reprojection(
            rig, cameras(), uv2, np.ones(uv2.shape[:-1])
        )
        self.assertGreater(rig.fk(result2[0])[0][:, 2].min(), -1e-4 * rig.height)

    def test_at_limit_requires_over_quarter_and_includes_two_degree_boundary(self):
        rig = radial_rig(foot=True)
        ix = rig.slices["Foot.L"][0][0]
        params = np.tile(rig.zero(), (100, 1))
        params[:25, ix] = rig.upper[ix] - np.deg2rad(2)
        self.assertEqual(at_limit(rig, params), [])
        params[25, ix] = rig.lower[ix] + np.deg2rad(2)
        report = at_limit(rig, params)
        self.assertEqual(report, [{"bone": "Foot.L", "share": 0.26}])
        params[:26, ix] += np.where(params[:26, ix] > 0, -1e-5, 1e-5)
        self.assertEqual(at_limit(rig, params), [])
        params[:, ix] = rig.upper[ix]
        rig.meta["Foot.L"]["hinge"] = {"hinge_sign": 1}
        self.assertEqual(at_limit(rig, params), [])


if __name__ == "__main__":
    unittest.main()
