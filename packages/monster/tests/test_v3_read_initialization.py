import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from reprojection_solver import RigModel, aligned_rest_rotation, _resolve_frame_end_pairs
from studio_camera import cameras_from_calibration


class ReadingInitializationTests(unittest.TestCase):
    def test_complete_readings_resolve_twist_before_a_bounded_knee(self):
        bones = {
            "Root": {"parent": None, "kind": "root", "rest_head_world": [0, 0, 1],
                     "rest_tail_world": [0, 0, 1.2]},
            "Thigh.L": {"parent": "Root", "kind": "limb",
                        "rest_head_world": [.2, 0, 1], "rest_tail_world": [.2, 0, .6],
                        "limits_rad": [[-.2, .2], [-1, 1], [-.2, .2]]},
            "Shin.L": {"parent": "Thigh.L", "kind": "limb",
                       "rest_head_world": [.2, 0, .6], "rest_tail_world": [.2, 0, .2],
                       "limits_rad": [[0, 1.4], [-.05, .05], [-.05, .05]]},
            "Foot.L": {"parent": "Shin.L", "kind": "rigid", "fixed_local": True,
                       "rest_head_world": [.2, 0, .2], "rest_tail_world": [.2, .2, .2]},
        }
        for b in bones.values():
            b["rest_rot_mat_world"] = aligned_rest_rotation(
                b["rest_head_world"], b["rest_tail_world"]).tolist()
        rig = RigModel(bones)
        target = rig.zero()
        target[rig.slices["Thigh.L"][0][1]] = .8
        target[rig.slices["Shin.L"][0][0]] = 1.0
        points = rig.fk(target)[0]
        initial = rig.read_start(points, rig.zero())
        self.assertGreater(np.linalg.norm(rig.fk(initial)[0] - points, axis=1).max(), .05)
        refined = rig.refine_read_start(points, initial)
        np.testing.assert_allclose(rig.fk(refined)[0], points, atol=1e-5)

    def test_asymmetric_prop_landmark_prevents_false_half_turn_endpoint_swap(self):
        bones = {"Prop_arrow": {"parent": None, "kind": "root",
                 "rest_head_world": [0, 0, 1], "rest_tail_world": [0, 1, 1],
                 "rest_rot_mat_world": np.eye(3).tolist()}}
        rig = RigModel(bones, extra_points=[
            {"name": "Prop_arrow.end1", "bone": "Prop_arrow", "local": [0, 0, 0]},
            {"name": "Prop_arrow.end2", "bone": "Prop_arrow", "local": [0, 1, 0]},
            {"name": "Prop_arrow.tip", "bone": "Prop_arrow", "local": [0, .9, 0]},
        ])
        cameras = list(cameras_from_calibration({"center": [0, .5, 1],
                        "ortho_scale": 3, "panel_width": 448, "panel_height": 448}).values())
        previous = rig.fk(rig.zero())[0]
        target = rig.zero()
        target[:] = [0, 0, np.pi, 0, 1, 0]
        points = rig.fk(target)[0]
        observed = np.array([c.project(points) for c in cameras])
        confidence = np.ones(observed.shape[:-1])
        actual, _ = _resolve_frame_end_pairs(rig, cameras, observed, confidence, previous)
        np.testing.assert_array_equal(actual, observed)
        i1, i2 = rig.end_pairs[0]
        reversed_ends = observed.copy()
        reversed_ends[:, [i1, i2]] = reversed_ends[:, [i2, i1]]
        actual, _ = _resolve_frame_end_pairs(rig, cameras, reversed_ends, confidence, previous)
        np.testing.assert_array_equal(actual, observed)

    def test_child_attachment_observes_parent_twist_without_truth_parameters(self):
        bones = {
            "Root": {"parent": None, "kind": "root",
                     "rest_head_world": [0, 0, 1], "rest_tail_world": [0, 0, 1.2]},
            "Arm.L": {"parent": "Root", "kind": "limb",
                      "rest_head_world": [0.2, 0, 1.2], "rest_tail_world": [0.7, 0, 1.2],
                      "limits_rad": [[-1.5, 1.5]] * 3},
            "Rigid1.L": {"parent": "Arm.L", "kind": "rigid", "fixed_local": True,
                         "rest_head_world": [0.45, 0.2, 1.32],
                         "rest_tail_world": [0.5, 0.3, 1.4]},
        }
        for b in bones.values():
            b["rest_rot_mat_world"] = aligned_rest_rotation(
                b["rest_head_world"], b["rest_tail_world"]).tolist()
        rig = RigModel(bones)
        target_params = rig.zero()
        root = rig.slices["Root"]
        target_params[root] = [0.3, 0.1, -0.2, 0.15, -0.1, 0.2]
        indices, _ = rig.slices["Arm.L"]
        target_params[indices] = [0.7, 0.8, -0.6]
        points, _, rotations = rig.fk(target_params)
        recovered = rig.read_start(points, rig.zero())
        actual, _, solved_rotations = rig.fk(recovered)
        np.testing.assert_allclose(actual, points, atol=1e-12)
        np.testing.assert_allclose(solved_rotations["Arm.L"], rotations["Arm.L"], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
