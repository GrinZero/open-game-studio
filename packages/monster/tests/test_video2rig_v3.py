"""V3 contract tests: camera/FK derivatives, occlusion, learning targets and actual integrations."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))
from studio_camera import Camera, cameras_from_calibration, triangulate
from reprojection_solver import (
    RigModel,
    aligned_rest_rotation,
    fit_clip_2d_reprojection,
    exp_so3,
    surface_keypoints,
    unwrap_rotvecs,
    find_hinges,
    fit_bones,
)
from reading_pipeline import (
    fix_twins_viterbi,
    bridge_unsure_stretches,
    despike_reading_backed,
    simulate_damped_spring_chains,
    learn_frame0_shadow_table,
    strip_shadow,
    detect_video_cadence_optical_flow,
)
from skeleton_postprocess import voxelize, rest_coords, postprocess_skintokens_skeleton
from anatomical_names import parse_anatomical_name
from bone_taxonomy import classify_rig_bones, validate_canonical_rig_bone_names


def fixture():
    meta = {
        "Root": {
            "parent": None,
            "kind": "root",
            "rest_head_world": [0, 0, 0.7],
            "rest_tail_world": [0, 0, 1.2],
        },
        "Arm.L": {
            "parent": "Root",
            "kind": "limb",
            "rest_head_world": [0.2, 0, 1.2],
            "rest_tail_world": [0.65, 0, 1.2],
            "limits_rad": [[-1.4, 1.4]] * 3,
        },
        "Arm.R": {
            "parent": "Root",
            "kind": "limb",
            "rest_head_world": [-0.2, 0, 1.2],
            "rest_tail_world": [-0.65, 0, 1.2],
            "limits_rad": [[-1.4, 1.4]] * 3,
        },
        "Rigid1.L": {
            "parent": "Arm.L",
            "kind": "rigid",
            "fixed_local": True,
            "rest_head_world": [0.65, 0, 1.2],
            "rest_tail_world": [0.65, 0.2, 1.2],
        },
    }
    for b in meta.values():
        b["rest_rot_mat_world"] = aligned_rest_rotation(
            np.array(b["rest_head_world"]), np.array(b["rest_tail_world"])
        ).tolist()
    rig = RigModel(
        meta,
        extra_points=[
            {"name": "Arm.L.sidea1", "bone": "Arm.L", "local": [0.1, 0.4, 0.08]}
        ],
    )
    cams = list(
        cameras_from_calibration(
            {
                "center": [0, 0, 1],
                "panel_width": 448,
                "panel_height": 448,
                "ortho_scale": 3,
            }
        ).values()
    )
    return rig, cams


class CameraAndFKTests(unittest.TestCase):
    def test_projection_jacobians_and_depth_inverse(self):
        for kind in ("ortho", "persp"):
            c = Camera(
                exp_so3([0.2, -0.1, 0.05]),
                np.array([0.1, -0.2, 3]),
                640,
                480,
                kind,
                2.1,
                700,
                (5, -7),
            )
            X = np.array([[0.2, 0.3, 0.4], [-0.2, 0.1, 0.5]])
            uv, J = c.project(X, True)
            numeric = np.stack(
                [
                    (
                        c.project(X + np.eye(3)[j] * 1e-6)
                        - c.project(X - np.eye(3)[j] * 1e-6)
                    )
                    / 2e-6
                    for j in range(3)
                ],
                -1,
            )
            np.testing.assert_allclose(J, numeric, atol=1e-7)
            np.testing.assert_allclose(c.ray_at_depth(uv, c.depth(X)), X, atol=1e-12)

    def test_fk_analytic_jacobian_including_skin_offset_and_rigid_child(self):
        rig, _ = fixture()
        p = np.linspace(-0.18, 0.22, rig.size)
        X, J, _, _ = rig.fk(p, True)
        numeric = np.stack(
            [
                (
                    rig.fk(p + np.eye(rig.size)[j] * 1e-6)[0]
                    - rig.fk(p - np.eye(rig.size)[j] * 1e-6)[0]
                )
                / 2e-6
                for j in range(rig.size)
            ],
            -1,
        )
        np.testing.assert_allclose(J, numeric, atol=3e-9)
        for i, n in enumerate(rig.names):
            self.assertAlmostEqual(
                np.linalg.norm(X[2 * i + 1] - X[2 * i]),
                np.linalg.norm(rig.t[n] - rig.h[n]),
                places=10,
            )

    def test_perspective_triangulation_and_one_panel_previous_depth(self):
        rig, cams = fixture()
        cams = [
            Camera(c.R, c.t, c.width, c.height, "persp", c.scale, 500) for c in cams
        ]
        X = rig.fk(rig.zero())[0]
        uv = np.array([c.project(X) for c in cams])
        w = np.ones(uv.shape[:-1])
        np.testing.assert_allclose(triangulate(cams, uv, w), X, atol=1e-10)
        w[1, 0] = 0
        uv[0, 0] += [3, 0]
        read = triangulate(cams, uv, w, X)
        self.assertAlmostEqual(cams[0].depth(read[0]), cams[0].depth(X[0]))
        np.testing.assert_allclose(cams[0].project(read[0]), uv[0, 0])

    def test_multistart_fit_reduces_pixels_and_is_repeatable(self):
        rig, cams = fixture()
        p = rig.zero()
        p[:3] = [0.18, -0.12, 0.4]
        p[3:6] = [0.03, 0.01, 0.06]
        X = rig.fk(p)[0]
        uv = np.array([[c.project(X) for c in cams]])
        weights = np.ones(uv.shape[:-1])
        a, diag = fit_clip_2d_reprojection(rig, cams, uv, weights, max_nfev=100)
        b, _ = fit_clip_2d_reprojection(rig, cams, uv, weights, max_nfev=100)
        np.testing.assert_array_equal(a, b)
        error = np.mean(
            [
                np.linalg.norm(c.project(rig.fk(a[0])[0]) - uv[0, v], axis=-1).mean()
                for v, c in enumerate(cams)
            ]
        )
        self.assertLess(error, 0.05)
        self.assertEqual(diag["blas_threads"], 1)

    def test_rotvec_unwrap_preserves_rotation(self):
        a = np.array([[0, 0, 3.1], [0, 0, -3.1], [0, 0, -2.9]])
        b = unwrap_rotvecs(a)
        self.assertLess(np.linalg.norm(b[1] - b[0]), 0.1)
        for x, y in zip(a, b):
            np.testing.assert_allclose(exp_so3(x), exp_so3(y), atol=1e-12)


class GeometryContractTests(unittest.TestCase):
    def test_uv_seams_do_not_split_surface_into_loose_fragments(self):
        from video2rig_blender import _surface_component_sizes

        # Two triangles share geometric positions, but keep separate UV vertices.
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                             [1, 0, 0], [1, 1, 0], [0, 1, 0],
                             [3, 0, 0], [4, 0, 0], [3, 1, 0]], float)
        edges = np.array([[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3],
                          [6, 7], [7, 8], [8, 6]])
        before = vertices.copy()
        sizes = _surface_component_sizes(vertices, edges, 1e-6)
        np.testing.assert_array_equal(sizes, [6] * 6 + [3] * 3)
        np.testing.assert_array_equal(vertices, before)

    def test_separate_toes_do_not_count_as_extra_grounded_limbs(self):
        from scipy import ndimage
        from skeleton_postprocess import Shell, _footprints

        grid = np.zeros((65, 65, 65), bool)
        grid[25:40, 25:40, 40:55] = True
        for x, y in ((12, 12), (12, 52), (52, 12), (52, 52)):
            grid[x - 3:x - 1, y - 1:y + 1, 1:6] = True
            grid[x + 1:x + 3, y - 1:y + 1, 1:6] = True
            grid[x - 3:x + 3, y - 1:y + 1, 4:8] = True
            grid[x - 1:x + 1, y - 1:y + 1, 6:35] = True
        shell = Shell(grid, grid, np.zeros(3), 1.0, ndimage.distance_transform_edt(grid))
        self.assertEqual(len(_footprints(shell, thin_legs=True)), 4)

    def test_midline_limb_names_remain_valid_without_inventing_a_side(self):
        from anatomical_names import anatomical_names

        meta = {
            "root": {"kind": "root", "side": "C", "parent": None},
            "midline": {"kind": "limb", "side": "C", "parent": "root"},
            "left": {"kind": "limb", "side": "L", "parent": "root"},
            "right": {"kind": "limb", "side": "R", "parent": "root"},
        }
        names = anatomical_names(meta, 2.0)
        self.assertTrue(names["midline"].startswith("Limb"))
        self.assertEqual(names["left"], "Hand.L")
        self.assertEqual(names["right"], "Hand.R")
        self.assertTrue(all(parse_anatomical_name(name) for name in names.values()))

    def test_v3_names_include_digits_and_legacy_is_readable(self):
        rig, _ = fixture()
        names = classify_rig_bones(rig.meta)
        self.assertTrue(
            all(parse_anatomical_name(b["canonical_name"]) for b in names.values())
        )
        self.assertTrue(validate_canonical_rig_bone_names(["ROOT_C_00_00"])["valid"])
        for n in (
            "Finger1_1.L",
            "Toe3_2.R",
            "WingHand.L",
            "Coxa3.R",
            "Prop_sword",
            "Ear1_2.L",
        ):
            self.assertIsNotNone(parse_anatomical_name(n))
        for n in ("bone_0", "Thigh", "Root.L", "Toes"):
            self.assertIsNone(parse_anatomical_name(n))

    def test_manylegs_pairs_share_arc_fractions(self):
        from scipy import ndimage
        from skeleton_postprocess import Shell, manylegs_build

        grid = np.zeros((65, 65, 65), bool)
        root = np.array([32, 32, 54])
        for sign in (-1, 1):
            for y in (16, 32, 48):
                path = [
                    root,
                    np.array([32 + sign * 18, y, 46]),
                    np.array([32 + sign * 24, y, 25]),
                    np.array([32 + sign * 25, y, 3]),
                ]
                for a, b in zip(path, path[1:]):
                    points = np.rint(
                        a + np.linspace(0, 1, 100)[:, None] * (b - a)
                    ).astype(int)
                    grid[tuple(points.T)] = True
        xx, yy, zz = np.mgrid[-3:4, -3:4, -3:4]
        ball = xx * xx + yy * yy + zz * zz <= 9
        solid = ndimage.binary_dilation(grid, structure=ball)
        distance = ndimage.distance_transform_edt(solid)
        shell = Shell(
            solid & ~ndimage.binary_erosion(solid),
            solid,
            np.array([-0.64, -0.64, 0]),
            0.02,
            distance,
        )
        legs = manylegs_build(shell, shell.origin + root * shell.step)
        self.assertEqual(len(legs), 24)
        for k in range(1, 4):
            self.assertEqual(
                legs[f"Coxa{k}.L"]["paired_arc_fractions"],
                legs[f"Coxa{k}.R"]["paired_arc_fractions"],
            )

    def test_inverse_lbs_inverts_blended_matrices(self):
        rest = np.repeat(np.eye(4)[None], 2, 0)
        posed = rest.copy()
        posed[1, :3, :3] = exp_so3([0, 0, 0.5])
        posed[0, :3, 3] = [0.2, 0, 0]
        v = np.array([[0.4, 0.2, 0.3]])
        ids = np.array([[0, 1]])
        weights = np.array([[0.6, 0.4]])
        blend = (posed[ids] * weights[..., None, None]).sum(1)
        actual = (blend @ np.c_[v, np.ones(1)][..., None])[:, :3, 0]
        np.testing.assert_allclose(
            rest_coords(actual, ids, weights, rest, posed), v, atol=1e-12
        )

    def test_shell_keeps_thin_surface_and_fills_closed_interior(self):
        v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
        f = np.array(
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
        shell = voxelize(v, f, 0.05)
        self.assertTrue(shell.contains(np.array([[0.5, 0.5, 0.5]]))[0])
        self.assertGreater(shell.solid.sum(), shell.surface.sum())
        bones = {
            "Root": {
                "parent": None,
                "kind": "root",
                "rest_head_world": [0.5, 0.5, 0.3],
                "rest_tail_world": [0.5, 0.5, 0.7],
            },
            "Finger1_1.L": {
                "parent": "Root",
                "kind": "digit",
                "rest_head_world": [0.5, 0.5, 0.7],
                "rest_tail_world": [0.6, 0.5, 0.7],
            },
            "outside": {
                "parent": "Root",
                "kind": "limb",
                "rest_head_world": [2, 2, 2],
                "rest_tail_world": [3, 3, 3],
            },
        }
        out, report = postprocess_skintokens_skeleton(bones, v, shell)
        self.assertIn("Finger1_1.L", out)
        self.assertNotIn("outside", out)
        self.assertFalse(report["fold_digits"])


class TemporalTests(unittest.TestCase):
    def test_twin_viterbi_repairs_side_swap(self):
        rig, cams = fixture()
        p = rig.zero()
        p[rig.slices["Arm.L"][0][2]] = 0.8
        p[rig.slices["Arm.R"][0][2]] = -0.6
        X = rig.fk(p)[0]
        uv = np.array([[c.project(X) for c in cams]] * 12)
        conf = np.ones(uv.shape[:-1])
        nameidx = {n: i for i, n in enumerate(rig.point_names)}
        l = [nameidx["Arm.L.head"], nameidx["Arm.L.tail"]]
        r = [nameidx["Arm.R.head"], nameidx["Arm.R.tail"]]
        expected = uv.copy()
        ids = l + r
        flip = r + l
        uv[:, 1, ids] = uv[:, 1, flip]
        repaired, _, diag = fix_twins_viterbi(rig, cams, uv, conf)
        self.assertTrue(diag)
        np.testing.assert_allclose(repaired[:, :, l + r], expected[:, :, l + r])

    def test_bridge_respects_short_and_long_gaps(self):
        uv = np.arange(40.0)[:, None, None, None] * np.ones((40, 2, 1, 2))
        conf = np.ones((40, 2, 1))
        conf[3:8, 0] = 0
        conf[10:38, 1] = 0
        uv[3:8, 0] = 999
        uv[10:38, 1] = 999
        out, w = bridge_unsure_stretches(uv, conf)
        np.testing.assert_allclose(out[3:8, 0, 0, 0], np.arange(3, 8))
        self.assertTrue((out[10:38, 1] == 999).all())
        self.assertLess(w[4, 0, 0], 0.4)

    def test_cadence_preserves_encoded_duration_of_repeated_frames(self):
        frames = []
        for dx in (0, 3, 3, 6, 9):
            im = np.zeros((192, 320, 3), np.uint8)
            im[50:130, 80 + dx : 180 + dx] = 255
            frames.append(im)
        report = detect_video_cadence_optical_flow(frames)
        np.testing.assert_allclose(report["frame_times"], np.arange(len(frames)))
        self.assertEqual(report["cadence"], 1.0)

    def test_shadow_colors_do_not_remove_object_colors(self):
        render = np.full((20, 20, 3), 200, np.uint8)
        render[5:15, 5:15] = [200, 30, 30]
        mask = np.zeros((20, 20), bool)
        mask[5:15, 5:15] = True
        video = render.copy()
        video[16:20] = 80
        table = learn_frame0_shadow_table(render, video, mask)
        clean = strip_shadow(video, table)
        np.testing.assert_array_equal(clean[mask], video[mask])
        self.assertTrue((clean[18] == 200).all())


class ReaderLearningTests(unittest.TestCase):
    def test_reader_heads_context_gradients_and_dense_losses(self):
        import torch
        from reader_v3 import ReaderHeads, reader_losses, temporal_training_context

        heads = ReaderHeads(2, 5)
        feature = torch.randn(2, 192, 16, 16)
        coords = torch.rand(2, 5, 2)
        # This tests gradients conditional on context being present. The 50%
        # training gate can correctly disable both batch elements at random.
        with patch("reader_v3.CONTEXT_SHARE", 1.0):
            context = temporal_training_context(coords, torch.ones(2, 5))
        logits, aux = heads(feature, context)
        labels = {
            "visibility_ratio": torch.ones(2, 5),
            "in_frame": torch.ones(2, 5),
            "label_valid": torch.ones(2, 5),
            "part": torch.ones(2, 16, 16, dtype=torch.long),
            "local": torch.zeros(2, 3, 16, 16),
            "relative_depth": torch.zeros(2, 1, 16, 16),
        }
        loss = reader_losses(logits, aux, coords, labels)["total"]
        loss.backward()
        for module in (heads.context[0], heads.phead, heads.lhead, heads.dhead):
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(module.weight.grad.abs().sum(), 0)

    def test_context_share_gate_is_one_decision_for_both_lags(self):
        import torch
        from reader_v3 import temporal_training_context

        with torch.random.fork_rng():
            torch.manual_seed(37)
            coords = torch.zeros(10000, 1, 2)
            valid = torch.ones(10000, 1)
            # Isolate the actual per-example gate from heatmap rasterization.
            with patch(
                "reader_v3.heatmaps", side_effect=lambda x: x.new_ones(len(x), 1, 1, 1)
            ):
                context = temporal_training_context(coords, valid)
            first, second = context[:, 0, 0, 0], context[:, 1, 0, 0]
            self.assertTrue(torch.equal(first, second))
            self.assertLess(abs(float(first.mean()) - 0.5), 0.03)
            self.assertTrue(torch.all((first == 0) | (first == 1)))

    def test_local_field_svd_recovers_all_attached_points(self):
        from reader_v3 import joints_from_dense_fields

        rig, cams = fixture()
        c = cams[0]
        H = W = 16
        yy, xx = np.mgrid[:H, :W]
        pixels = np.stack([(xx + 0.5) * 448 / W, (yy + 0.5) * 448 / H], -1)
        X = c.ray_at_depth(pixels, np.full((H, W), c.depth(rig.h["Arm.L"])))
        i = rig.index["Arm.L"]
        scale = max(np.linalg.norm(rig.t["Arm.L"] - rig.h["Arm.L"]), 0.05 * rig.height)
        local = (X - rig.h["Arm.L"]) @ rig.R["Arm.L"] / scale
        fields = np.zeros((len(rig.names), 3, H, W))
        fields[i] = local.transpose(2, 0, 1)
        logits = np.full((len(rig.names) + 1, H, W), -20.0)
        logits[i + 1] = 20
        points, conf = joints_from_dense_fields(
            logits, fields, np.zeros((1, H, W)), c, rig, c.depth(rig.h["Arm.L"])
        )
        for k, owner in enumerate(rig.point_bones):
            if owner == "Arm.L":
                np.testing.assert_allclose(
                    points[k],
                    rig.h[owner] + rig.R[owner] @ rig.local_points[k],
                    atol=1e-8,
                )

    def test_moving_windows_tta_inverse_and_temporal_context(self):
        import torch
        import torch.nn.functional as F
        from reader_v3 import infer_reader_sequence
        from dinov2_pose_trainer import MonsterDINOv2PoseProbe

        class Oracle:
            num_keypoints = 3
            contexts = []
            _decode_logits_dsnt = staticmethod(
                MonsterDINOv2PoseProbe._decode_logits_dsnt
            )

            def __call__(self, x, ids, return_aux=True, context=None):
                if context is not None:
                    self.contexts.append(float(context.abs().sum()))
                rgb = (
                    x * x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
                    + x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
                )
                hm = F.interpolate(
                    rgb, size=(64, 64), mode="bilinear", align_corners=False
                )
                logits = torch.logit((hm * 0.98 + 0.005).clamp(0.005, 0.995))
                uv, conf = self._decode_logits_dsnt(logits)
                return uv, conf, logits, {}

        frames = []
        expected = []
        for t in range(2):
            image = np.zeros((448, 448, 3), np.uint8)
            points = np.array(
                [[100 + t * 4, 100], [200 + t * 4, 200], [300 + t * 4, 130]]
            )
            for k, (x, y) in enumerate(points):
                image[y - 4 : y + 5, x - 4 : x + 5, k] = 255
            frames.append(image)
            expected.append(points)
        oracle = Oracle()
        uv, conf, report = infer_reader_sequence(oracle, frames, 0, "cpu")
        np.testing.assert_allclose(uv, expected, atol=4.0)
        self.assertEqual(report["tta"], 8)
        self.assertEqual(report["context_offsets"], [-1, 1])
        self.assertTrue(all(c > 0 for c in oracle.contexts[:8]))
        self.assertGreater(oracle.contexts[-1], 0)
        # Existing checkpoints retain their original causal boundary padding.
        from reader_v3 import LEGACY_TEMPORAL_CONTEXT_CONTRACT

        legacy = Oracle()
        legacy.contexts = []
        _, _, legacy_report = infer_reader_sequence(
            legacy, frames, 0, "cpu",
            temporal_context_contract=LEGACY_TEMPORAL_CONTEXT_CONTRACT,
        )
        self.assertEqual(legacy_report["context_offsets"], [-1, -2])
        self.assertTrue(all(c == 0 for c in legacy.contexts[:8]))

    def test_model_v3_checkpoint_reload_includes_context_head(self):
        import torch
        from torch import nn
        from dinov2_pose_trainer import MonsterDINOv2PoseProbe
        from pose_finetuning import load_probe_weights

        class Backbone(nn.Module):
            embed_dim = 16

            def __init__(self):
                super().__init__()
                self.proj = nn.Conv2d(3, 16, 14, stride=14)

            def forward_features(self, x):
                return {"x_norm_patchtokens": self.proj(x).flatten(2).transpose(1, 2)}

        model = MonsterDINOv2PoseProbe(
            5,
            custom_backbone=Backbone(),
            enable_depth_visibility=True,
            enable_dense_surface=True,
            reader_v3=True,
            num_bones=2,
        ).eval()
        x = torch.randn(1, 3, 56, 56)
        ids = torch.tensor([0])
        output = model(x, ids, True)
        self.assertEqual(output[3]["part_logits"].shape[1], 3)
        checkpoint = {
            "reader_contract": "monster-reader-v3",
            "head_state_dict": model.head.state_dict(),
            "aux_state_dict": model.aux_head.state_dict(),
            "reader_heads_state_dict": model.reader_heads.state_dict(),
            "backbone_state_dict": model.backbone.state_dict(),
        }
        for name in (
            "dino_multi_fuse",
            "rgb_detail_branch",
            "rgb_kp_refine",
            "dense_head",
        ):
            checkpoint[name + "_state_dict"] = getattr(model, name).state_dict()
        clone = MonsterDINOv2PoseProbe(
            5,
            custom_backbone=Backbone(),
            enable_depth_visibility=True,
            enable_dense_surface=True,
            reader_v3=True,
            num_bones=2,
        ).eval()
        load_probe_weights(clone, checkpoint)
        torch.testing.assert_close(output[0], clone(x, ids, True)[0])


class BlenderV3IntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("MONSTER_TEST_BLENDER"),
        "Set MONSTER_TEST_BLENDER for v3 actual render/bake/readback regression",
    )
    def test_actual_dense_labels_full_strip_fallback_and_glb_review(self):
        self._actual_layout("horizontal")

    @unittest.skipUnless(
        os.environ.get("MONSTER_TEST_BLENDER"),
        "Set MONSTER_TEST_BLENDER for stacked integration",
    )
    def test_actual_stacked_labels_training_inference_and_glb_review(self):
        self._actual_layout("stacked")

    def _actual_layout(self, layout):
        import subprocess

        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "blender.log"
            with log.open("w") as stream:
                result = subprocess.run(
                    [
                        os.environ["MONSTER_TEST_BLENDER"],
                        "--background",
                        "--factory-startup",
                        "--python-exit-code",
                        "1",
                        "--python",
                        str(Path(__file__).with_name("fixture_v3_blender.py")),
                        "--",
                        temporary,
                    ],
                    env={
                        **os.environ,
                        "MONSTER_TEST_LAYOUT": layout,
                        "MONSTER_STUDIO_LAYOUT": layout,
                    },
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            self.assertEqual(result.returncode, 0, log.read_text()[-10000:])
            report = json.loads(
                (Path(temporary) / "review/glb_readback.json").read_text()
            )
            self.assertTrue(report["actual_glb_readback"])
            self.assertEqual(
                report["frames"][1]["baked_frame"], 0.75 if layout == "stacked" else 1.5
            )
            self.assertEqual(len(list(Path(temporary).glob("strip16_*.jpg"))), 16)
            # Consume the actual rendered v3 labels in the production training/checkpoint path.
            import torch
            import dinov2_pose_trainer as trainer
            from test_pose_finetuning import TinyBackbone
            from pose_finetuning import load_probe_weights
            from test_v3_reader_boundaries import check_native_training_samples

            expected_window = check_native_training_samples(self, Path(temporary) / "dataset")

            original = trainer.MonsterDINOv2PoseProbe
            initial = {}

            def factory(**kwargs):
                model = original(**kwargs, custom_backbone=TinyBackbone())
                if not initial:
                    initial["context"] = (
                        model.reader_heads.context[0].weight.detach().clone()
                    )
                    initial["rgb"] = model.rgb_detail_branch[0].weight.detach().clone()
                return model

            torch.set_num_threads(1)
            with (
                patch.object(trainer, "MonsterDINOv2PoseProbe", side_effect=factory),
                patch.object(trainer, "_query_telemetry", return_value={}),
                patch.dict(os.environ, {"MONSTER_DATALOADER_WORKERS": "0"}),
                patch("reader_v3.CONTEXT_SHARE", 1.0),
            ):
                trainer.train_monster_dinov2_probe(
                    str(Path(temporary) / "dataset"),
                    str(Path(temporary) / "studio_rig_meta.json"),
                    str(Path(temporary) / "reader.pt"),
                    epochs=2,
                    batch_size=2,
                    training_config={
                        "unfreeze_last_blocks": 2,
                        "gradient_checkpointing": False,
                        "accumulation_steps": 1,
                    },
                )
            ckpt = torch.load(Path(temporary) / "reader.pt", weights_only=False)
            self.assertEqual(ckpt["reader_contract"], "monster-reader-v3")
            self.assertEqual(
                ckpt["temporal_context_contract"],
                "monster-reader-bidirectional-context-v2",
            )
            self.assertEqual(ckpt['reader_window'], expected_window)
            self.assertFalse(
                torch.equal(
                    initial["context"],
                    ckpt["reader_heads_state_dict"]["context.0.weight"],
                )
            )
            self.assertFalse(
                torch.equal(
                    initial["rgb"], ckpt["rgb_detail_branch_state_dict"]["0.weight"]
                )
            )
            self.assertIn("frame0_reference", ckpt)
            clone = factory(
                num_keypoints=ckpt["num_keypoints"],
                enable_depth_visibility=True,
                enable_dense_surface=True,
                reader_v3=True,
                num_bones=len(ckpt["bone_names"]),
            )
            load_probe_weights(clone, ckpt)
            self.assertTrue(all(torch.isfinite(p).all() for p in clone.parameters()))
            if layout == "stacked":
                from video2rig_pipeline import infer_v3_frames

                labels = json.loads(
                    (Path(temporary) / "dataset/keypoints_gt.json").read_text()
                )
                result = infer_v3_frames(
                    ckpt,
                    [str(Path(temporary) / "dataset" / labels["idle_reference"])],
                    custom_backbone=TinyBackbone(),
                )
                self.assertEqual(result["reader_contract"], "monster-reader-v3")
                self.assertEqual(result["calibration"]["layout"], "stacked")
                self.assertIn("ACTUAL_STACKED_LABELS_AND_CACHE_OK", log.read_text())
                # Pass real readback artifacts through G3 without replacing their 60 FPS encode.
                import shutil

                sys.path.insert(0, str(ROOT / "skills/monster-workflow/scripts"))
                import sop_g3_exam, sop_preflight

                base = Path(temporary)
                work = base / "sop-work"
                checks = base / "sop-checks"
                g3 = checks / "g3_review"
                (work / "g1_candidates").mkdir(parents=True)
                shutil.copyfile(
                    base / "studio_rig_meta.json",
                    work / "g1_candidates/candidate_0_rig_meta.json",
                )
                shutil.copytree(base / "review", g3)
                before = (g3 / "g3_comparison_sync.mp4").read_bytes()
                with (
                    patch.object(sop_g3_exam, "load_ledger", return_value={}),
                    patch.object(
                        sop_g3_exam,
                        "assert_gate",
                        return_value={"selected_candidate_id": "0"},
                    ),
                    patch.object(
                        sop_g3_exam,
                        "resolve_scratch_dirs",
                        return_value={"work_dir": work, "checks_dir": checks},
                    ),
                    patch.object(
                        sop_g3_exam,
                        "resolve_g1_candidate_files",
                        return_value=(
                            "0",
                            base / "studio_rig_meta.json",
                            base / "rig.blend",
                        ),
                    ),
                    patch.object(
                        sop_preflight,
                        "_worker_command",
                        return_value=["disposable-native-test"],
                    ),
                    patch.object(sop_g3_exam.subprocess, "run"),
                    patch.object(
                        sop_g3_exam,
                        "_encode_sync_comparison_video",
                        side_effect=AssertionError(
                            "Actual timed video must not be re-encoded"
                        ),
                    ),
                    patch.object(sop_g3_exam, "mark_awaiting_review"),
                    patch.object(sop_g3_exam, "save_ledger"),
                ):
                    result = sop_g3_exam.build_g3_exam_pack(
                        "timing-fixture",
                        project_root=base,
                        drive_root=base / "drive",
                        exam_video=base / "source.mp4",
                        pred_traj_json=base / "trajectory.json",
                        sync_u2=False,
                    )
                self.assertEqual(before, (g3 / "g3_comparison_sync.mp4").read_bytes())
                self.assertEqual(result["status"], "awaiting_exam_review")


if __name__ == "__main__":
    unittest.main()
