"""Measured skin quantiles, stable slot ownership and observable terminal roll."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from reprojection_solver import (
    RigModel,
    aligned_rest_rotation,
    fit_clip_2d_reprojection,
    surface_keypoints,
)
from studio_camera import cameras_from_calibration


def bone(parent, head, tail, **kwargs):
    return dict(
        parent=parent,
        kind="limb",
        rest_head_world=list(head),
        rest_tail_world=list(tail),
        rest_rot_mat_world=aligned_rest_rotation(
            np.array(head), np.array(tail)
        ).tolist(),
        **kwargs,
    )


def local_cloud():
    # Independent transverse coordinates with a unique, known principal axis.
    return np.array(
        [
            [x, y, z]
            for x in np.linspace(-0.12, 0.12, 101)
            for y in (0.1, 0.5)
            for z in (-0.04, 0.04)
        ]
    )


def skin_for(meta, names=None):
    names = names or list(meta)
    points, indices = [], []
    for i, name in enumerate(names):
        b = meta[name]
        points.extend(
            np.array(b["rest_head_world"])
            + local_cloud() @ np.array(b["rest_rot_mat_world"]).T
        )
        indices.extend([i] * len(local_cloud()))
    return dict(
        bone_names=names,
        rest_verts=np.array(points),
        skin_indices=np.array(indices)[:, None],
        skin_weights=np.ones((len(points), 1)),
        height=2.0,
        floor_z=0.0,  # This fixture contains only the hand, not the actor's sole.
    )


class SurfacePointTests(unittest.TestCase):
    def test_quantiles_all_terminal_families_and_slots_follow_named_skin_ownership(
        self,
    ):
        terminals = [
            "Hand.L",
            "Foot.L",
            "FrontPaw.R",
            "HindPaw.L",
            "Head",
            "WingHand.R",
            "Prop_tool",
        ]
        meta = {"Root": bone(None, [0, 0, 0], [0, 0, 1])}
        for i, name in enumerate(terminals):
            meta[name] = bone("Root", [i, 0, 1], [i, 0.4, 1])
        meta["Root"]["kind"] = "root"
        meta["Head"]["kind"] = "head"
        # Neither missing nor stale cached children lists may invent a leaf.
        meta["Root"]["children"] = []
        meta["Hand.L"]["children"] = []
        meta["Finger1_1.L"] = bone(
            "Hand.L", [0, 0.4, 1], [0, 0.6, 1], children=["deleted"]
        )
        skin = skin_for(meta, list(reversed(meta)))
        names = sorted(meta)
        before = copy.deepcopy(meta)
        points = surface_keypoints(meta, names, skin)
        lookup = {p["name"]: p for p in points}
        self.assertNotIn("Root.tip", lookup)
        self.assertNotIn("Hand.L.tip", lookup)
        self.assertIn("Finger1_1.L.tip", lookup)
        cloud = local_cloud()
        for name in terminals:
            # Side labels can have either eigenvector sign, but both exact quantiles exist.
            side = np.array([lookup[f"{name}.sidea{j}"]["local"] for j in (1, 2)])
            np.testing.assert_allclose(
                np.sort(side[:, 0]), np.quantile(cloud[:, 0], [0.01, 0.99]), atol=1e-12
            )
            np.testing.assert_allclose(
                side[:, 1:], np.tile([0.3, 0], (2, 1)), atol=1e-12
            )
            side = np.array([lookup[f"{name}.sideb{j}"]["local"] for j in (1, 2)])
            np.testing.assert_allclose(np.sort(side[:, 2]), [-0.04, 0.04], atol=1e-12)
            np.testing.assert_allclose(
                side[:, :2], np.tile([0, 0.3], (2, 1)), atol=1e-12
            )
            if name != "Hand.L":
                np.testing.assert_allclose(
                    lookup[name + ".tip"]["local"],
                    [0, np.quantile(cloud[:, 1], 0.98), 0],
                )
        for label, y in [("end1", 0.1), ("end2", 0.5), ("mid", 0.3)]:
            np.testing.assert_allclose(lookup["Prop_tool." + label]["local"], [0, y, 0])
        rig = RigModel(meta, names, skin)
        self.assertEqual(
            rig.point_names[: 2 * len(names)],
            [n + "." + end for n in names for end in ("head", "tail")],
        )
        self.assertEqual(rig.point_names[2 * len(names) :], [p["name"] for p in points])
        self.assertEqual(rig.point_bones[2 * len(names) :], [p["bone"] for p in points])
        self.assertEqual(meta, before)
        # Capsules must use the same named mapping instead of the requested slot order.
        self.assertEqual({n for n, _ in rig.capsules}, {"Root", "Head"})
        for _, radius in rig.capsules:
            self.assertLess(radius, 0.55)

    def test_strict_point_three_skin_threshold_and_sparse_or_absent_bones(self):
        meta = {
            "Hand.L": bone(None, [0, 0, 0], [0, 1, 0]),
            "Foot.R": bone(None, [2, 0, 0], [2, 1, 0]),
        }
        local = local_cloud()
        skin = dict(
            bone_names=["Hand.L", "Foot.R"],
            rest_verts=np.vstack(
                [local, [[100, 100, 100], [200, 200, 200]], [[0, 0.8, 0]]]
            ),
            skin_indices=np.zeros((len(local) + 3, 1), int),
            skin_weights=np.r_[np.ones(len(local)), 0.3, 0.299, 0.30001][:, None],
        )
        points = {
            p["name"]: p for p in surface_keypoints(meta, list(reversed(meta)), skin)
        }
        expected = np.quantile(np.r_[local[:, 1], 0.8], 0.98)
        self.assertAlmostEqual(points["Hand.L.tip"]["local"][1], expected)
        self.assertFalse(any(n.startswith("Foot.R.") for n in points))
        # Missing support must not use another bone's vertices or create bogus landmarks.
        skin["bone_names"] = ["Hand.L"]
        self.assertEqual(surface_keypoints(meta, ["Foot.R"], skin), [])
        skin["rest_verts"] = local[:3]
        skin["skin_indices"] = np.zeros((3, 1), int)
        skin["skin_weights"] = np.ones((3, 1))
        self.assertEqual(surface_keypoints(meta, ["Hand.L"], skin), [])

    def test_surface_sides_resolve_terminal_roll_and_parent_twist_in_pixel_fit(self):
        meta = {
            "Root": bone(None, [0, 0, 0], [0, 0, 1], fixed_local=True),
            "UpperArm.L": bone(
                "Root",
                [0.2, 0, 1],
                [0.2, 0.5, 1],
                limits_rad=[[0, 0], [-1.2, 1.2], [0, 0]],
            ),
            "Hand.L": bone(
                "UpperArm.L",
                [0.2, 0.5, 1],
                [0.2, 0.7, 1.1],
                limits_rad=[[0, 0], [-1.2, 1.2], [0, 0]],
            ),
        }
        skin = skin_for({"Hand.L": meta["Hand.L"]})
        rig = RigModel(meta, sorted(meta), skin)
        cameras = list(
            cameras_from_calibration(
                dict(
                    center=[0, 0.4, 1], panel_width=448, panel_height=448, ortho_scale=2
                )
            ).values()
        )
        expected = rig.zero()
        expected[rig.slices["UpperArm.L"][0]] = -0.45
        expected[rig.slices["Hand.L"][0]] = 0.7
        points, J, _, _ = rig.fk(expected, jac=True)
        endpoint_count = 2 * len(meta)
        hand_channel = rig.slices["Hand.L"][0][0]
        # Endpoint-only observations cannot measure the hand's own axial rotation.
        np.testing.assert_allclose(J[:endpoint_count, :, hand_channel], 0, atol=1e-12)
        self.assertGreater(np.linalg.norm(J[endpoint_count:, :, hand_channel]), 0.1)
        uv = np.array([[c.project(points) for c in cameras]])
        endpoint_confidence = np.ones(uv.shape[:-1])
        endpoint_confidence[:, :, endpoint_count:] = 0
        endpoint_fit, _ = fit_clip_2d_reprojection(
            rig, cameras, uv, endpoint_confidence, flat_feet=0
        )
        self.assertLess(abs(endpoint_fit[0, hand_channel]), 1e-5)
        fit, _ = fit_clip_2d_reprojection(
            rig, cameras, uv, np.ones(uv.shape[:-1]), flat_feet=0
        )
        np.testing.assert_allclose(fit[0], expected, atol=1e-4)
        actual_uv = np.array([c.project(rig.fk(fit[0])[0]) for c in cameras])
        self.assertLess(np.max(np.linalg.norm(actual_uv - uv[0], axis=-1)), 0.001)


class NativeSurfacePointTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MONSTER_TEST_BLENDER"),
        "Set MONSTER_TEST_BLENDER for native surface point verification",
    )
    def test_real_split_mesh_skin_quantiles_and_reader_projection(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = subprocess.run(
                [
                    os.environ["MONSTER_TEST_BLENDER"],
                    "--background",
                    "--factory-startup",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(Path(__file__).with_name("fixture_v3_surface_points.py")),
                    "--",
                    temporary,
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("SURFACE_POINTS_NATIVE_PASSED", result.stdout)
            report = json.loads(
                (Path(temporary) / "native-surface-points.json").read_text()
            )
            self.assertEqual(report["meshes"], 2)
            self.assertLess(report["pixel_error"], 0.001)
            self.assertGreater(report["points"], report["endpoints"])


if __name__ == "__main__":
    unittest.main()
