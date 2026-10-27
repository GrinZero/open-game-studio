"""Reader camera/crop, visibility, temporal and augmentation acceptance probes."""

from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from reader_v3 import (
    joints_from_dense_fields,
    infer_reader_sequence,
    heatmaps,
    smear_limbs_gpu,
    inject_contact_shadow_gpu,
    reader_losses,
    temporal_training_context,
)
from reader_v3 import ReaderDataset, temporal_pose_points, map_points, crop_rgb
from reader_v3 import training_window, scaled_window_side, WINDOW_CONTRACT
from reprojection_solver import RigModel, aligned_rest_rotation
from studio_camera import Camera
from dinov2_pose_trainer import MonsterDINOv2PoseProbe


def check_native_training_samples(case, dataset):
    """Called by the existing native horizontal/stacked render-and-train tests."""
    from PIL import Image

    data = ReaderDataset(
        dataset,
        sorted(json.loads((Path(dataset) / "keypoints_gt.json").read_text())["bones"]),
        augment=True,
    )
    padded = False
    for index, view in enumerate(data.view_order):
        worlds = []

        def capture(rig, joints, rng, contract):
            points = temporal_pose_points(rig, joints, rng, contract)
            worlds.append(points)
            return points

        # Keep the renderer's measured crop while retaining actual articulated
        # previous-pose sampling, normalization and dense target resampling.
        with (
            patch("reader_v3.np.random.rand", return_value=1.0),
            patch("reader_v3.np.random.uniform", side_effect=[1.0, np.zeros(2)]),
            patch("reader_v3.temporal_pose_points", side_effect=capture),
        ):
            image, coords, labels = data[index]
        item = data.samples[0]
        uv = np.asarray(item["reader_points"][view]["uv"])
        W, H = data.render_calib["panel_width"], data.render_calib["panel_height"]
        inside = np.isfinite(uv).all(-1) & (uv >= 0).all(-1) & (uv <= [W, H]).all(-1)
        center = (
            np.median(uv[inside], axis=0)
            if inside.sum() >= 3
            else np.array([W / 2, H / 2])
        )
        box = np.r_[
            center - data.window["side_px"] / 2, center + data.window["side_px"] / 2
        ]
        # end_pairs use a consistent left-to-right slot ordering in the dataset.
        expected = map_points(uv, box, 1)
        previous = map_points(data.cameras[view].project(worlds[0]), box, 1)
        visibility = np.asarray(item["reader_points"][view]["visibility"]).copy()
        for a, b in data.rig.end_pairs:
            if expected[a, 0] > expected[b, 0]:
                expected[[a, b]] = expected[[b, a]]
                previous[:, [a, b]] = previous[:, [b, a]]
                visibility[[a, b]] = visibility[[b, a]]
        np.testing.assert_allclose(coords, expected, atol=2e-6)
        np.testing.assert_allclose(labels["previous_coords"], previous, atol=2e-6)
        np.testing.assert_allclose(
            labels["previous_valid"],
            ((previous >= 0) & (previous <= 1)).all(-1) * visibility[None],
            atol=1e-6,
        )
        case.assertGreater(float(np.max(np.abs(previous - expected[None]))), 1e-5)
        for i, n in enumerate(data.rig.names):
            np.testing.assert_allclose(
                np.linalg.norm(worlds[0][:, 2 * i + 1] - worlds[0][:, 2 * i], axis=-1),
                np.linalg.norm(data.rig.t[n] - data.rig.h[n]),
                atol=1e-8,
            )
        with Image.open(Path(dataset) / "images" / item["image"]) as source:
            W, H = data.render_calib["panel_width"], data.render_calib["panel_height"]
            panel = np.asarray(
                source.crop(
                    (0, index * H, W, (index + 1) * H)
                    if data.render_calib.get("layout") == "stacked"
                    else (index * W, 0, (index + 1) * W, H)
                )
            )
        rgb = image.permute(1, 2, 0).numpy() * [0.229, 0.224, 0.225] + [
            0.485,
            0.456,
            0.406,
        ]
        np.testing.assert_allclose(rgb, crop_rgb(panel, box) / 255, atol=1e-6)
        yy, xx = np.mgrid[:128, :128] / 127
        x = box[0] + xx * (box[2] - box[0])
        y = box[1] + yy * (box[3] - box[1])
        outside = (
            (x < -W / 128) | (x > W + W / 128) | (y < -H / 128) | (y > H + H / 128)
        )
        if outside.any():
            padded = True
            case.assertFalse(bool(labels["part"][outside].any()))
        with patch("reader_v3.CONTEXT_SHARE", 1.0):
            context = temporal_training_context(
                coords[None],
                (labels["visibility_ratio"] * labels["in_frame"])[None],
                True,
                labels["previous_coords"][None],
                labels["previous_valid"][None],
            )
        case.assertTrue(bool(context.any()))
        for lag in range(2):
            invisible = labels["previous_valid"][lag] == 0
            case.assertFalse(
                bool(
                    context[0, lag * len(coords) : (lag + 1) * len(coords)][
                        invisible
                    ].any()
                )
            )
    case.assertTrue(padded, "Native generated crops must exercise real panel padding")
    return data.window


def point_rig():
    specs = {
        "Root": (None, [0, 0, 0.4], [0, 0, 1], "root"),
        "Hand.L": ("Root", [0.15, 0, 1], [0.4, 0, 1], "limb"),
        "Finger1_1.L": ("Hand.L", [0.4, 0, 1], [0.5, 0, 1], "digit"),
        "Finger1_2.L": ("Finger1_1.L", [0.5, 0, 1], [0.6, 0, 1], "digit"),
        "Foot.R": ("Root", [-0.15, 0, 0.35], [-0.15, -0.2, 0.1], "limb"),
        "Toe1_1.R": ("Foot.R", [-0.15, -0.2, 0.1], [-0.15, -0.3, 0.1], "digit"),
        "WingHand.R": ("Root", [-0.2, 0, 1], [-0.45, 0, 1], "wing"),
        "Finger2_1.R": ("WingHand.R", [-0.45, 0, 1], [-0.7, 0, 1], "digit"),
        "Tail1": ("Root", [0, 0.1, 0.5], [0, 0.4, 0.5], "tail"),
        "Prop_tool": (None, [0, 0.4, 0.5], [0, 0.4, 1], "rigid"),
    }
    meta = {
        n: dict(parent=p, rest_head_world=h, rest_tail_world=t, kind=k)
        for n, (p, h, t, k) in specs.items()
    }
    for b in meta.values():
        b["rest_rot_mat_world"] = aligned_rest_rotation(
            np.array(b["rest_head_world"]), np.array(b["rest_tail_world"])
        ).tolist()
    extra = [
        dict(name=n + suffix, bone=n, local=local)
        for n in meta
        for suffix, local in (
            (".tip", [0, 0.3, 0]),
            (".sidea1", [0.1, 0.1, 0]),
            (".sideb2", [0, 0.1, -0.08]),
        )
    ]
    extra += [
        dict(name="Prop_tool." + name, bone="Prop_tool", local=p)
        for name, p in (
            ("end1", [0, 0, 0]),
            ("end2", [0, 0.5, 0]),
            ("mid", [0, 0.25, 0]),
        )
    ]
    return RigModel(meta, extra_points=extra)


class ReaderBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)

    def test_p95_is_across_training_pictures_and_scales_from_saved_checkpoint(self):
        samples = []
        extents = list(np.linspace(100, 120, 20)) + [300]
        for extent in extents:
            samples.append(
                {
                    "reader_points": {
                        "front": {
                            "uv": [
                                [448 - extent / 2, 448],
                                [448 + extent / 2, 448],
                                [448, 450],
                            ],
                            "visibility": [0, 0, 0],
                        }
                    }
                }
            )
        window = training_window(samples, ["front"], 896, 896)
        self.assertEqual(window["side_px"], 134)
        self.assertEqual(window["contract"], WINDOW_CONTRACT)
        self.assertEqual(scaled_window_side(window, 448, 448), 67)
        self.assertEqual(scaled_window_side(window, 40, 448), 40)
        for invalid in (
            None,
            {},
            dict(window, side_px=float("nan")),
            dict(window, side_px=897),
        ):
            with self.assertRaises(ValueError):
                scaled_window_side(invalid, 448, 448)

    def test_dense_fields_recover_every_point_for_shifted_non_square_cameras(self):
        rig = point_rig()
        p = rig.start_params()
        p[:3] = [0.2, -0.15, 0.35]
        p[3:6] = [0.12, -0.07, 0.1]
        expected, _, heads, rotations = rig.fk(p, True)
        for typ in ("ortho", "persp"):
            for width, height in ((640, 320), (320, 640)):
                c = Camera(
                    np.eye(3),
                    np.array([0, 0, 4.0]),
                    width,
                    height,
                    typ,
                    scale=3,
                    focal_px=500,
                    shift_px=(31, -17),
                )
                hh, ww = 24, 5 * len(rig.names)
                yy, xx = np.mgrid[:hh, :ww]
                uv = np.stack([(xx + 0.5) * width / ww, (yy + 0.5) * height / hh], -1)
                X = c.ray_at_depth(uv, np.full((hh, ww), 4.5))
                logits = np.full((len(rig.names) + 1, hh, ww), -20.0)
                local = np.zeros((len(rig.names), 3, hh, ww))
                for i, n in enumerate(rig.names):
                    section = slice(i * 5, (i + 1) * 5)
                    logits[i + 1, :, section] = 20
                    scale = max(np.linalg.norm(rig.t[n] - rig.h[n]), 0.05 * rig.height)
                    local[i, :, :, section] = (
                        (X[:, section] - heads[n]) @ rotations[n] / scale
                    ).transpose(2, 0, 1)
                result, confidence = joints_from_dense_fields(
                    logits, local, np.zeros((1, hh, ww)), c, rig, 4.5
                )
                np.testing.assert_allclose(result, expected, atol=1e-10)
                self.assertTrue((confidence > 0.99).all())
                # Background-only output cannot invent rigid transforms or point confidence.
                logits[:] = -20
                logits[0] = 20
                result, confidence = joints_from_dense_fields(
                    logits, local, np.zeros((1, hh, ww)), c, rig, 4.5
                )
                self.assertTrue(np.isnan(result).all())
                self.assertFalse(confidence.any())

    def test_non_square_edge_motion_inverse_tta_and_start_context_lags(self):
        class Oracle:
            num_keypoints = 3
            _decode_logits_dsnt = staticmethod(
                MonsterDINOv2PoseProbe._decode_logits_dsnt
            )

            def __init__(self):
                self.contexts = []

            def __call__(self, x, ids, return_aux=True, context=None):
                if context is not None:
                    self.contexts.append(context.detach().clone())
                rgb = (
                    x * x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
                    + x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
                )
                hm = F.interpolate(
                    rgb, size=(96, 96), mode="bilinear", align_corners=False
                )
                logits = torch.logit((hm * 0.98 + 0.005).clamp(0.005, 0.995))
                uv, conf = self._decode_logits_dsnt(logits)
                return uv, conf, logits, None

        for width, height in ((320, 192), (192, 320)):
            frames = []
            expected = []
            for t in range(5):
                angle = t * np.pi / 4
                R = np.array(
                    [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                )
                points = np.rint(
                    np.array([[-70, -15], [-18, 22], [60, -8]]) @ R.T
                    + [80 + t * 40, 80]
                ).astype(int)
                if width < height:
                    points = points[:, ::-1]
                image = np.zeros((height, width, 3), np.uint8)
                for k, (x, y) in enumerate(points):
                    image[y - 4 : y + 5, x - 4 : x + 5, k] = 255
                # One limb is genuinely hidden, without fabricating an image measurement.
                if t == 2:
                    image[:, :, 1] = 0
                frames.append(image)
                expected.append(points)
            oracle = Oracle()
            window = {
                "contract": WINDOW_CONTRACT,
                "side_px": 180,
                "panel_width": width,
                "panel_height": height,
            }
            uv, confidence, report = infer_reader_sequence(
                oracle, frames, 0, "cpu", window=window
            )
            valid = np.ones((5, 3), bool)
            valid[2, 1] = False
            np.testing.assert_allclose(uv[valid], np.array(expected)[valid], atol=4)
            self.assertLess(confidence[2, 1], 0.1)
            self.assertEqual(report["tta"], 8)
            np.testing.assert_allclose(
                np.diff(np.array(report["windows"])[:, [0, 2]], axis=1), 180, atol=1e-6
            )
            self.assertEqual(report["training_window"], window)
            self.assertEqual(len(oracle.contexts), 40)
            self.assertEqual(report["context_offsets"], [-1, 1])
            self.assertTrue(
                all(bool(c[:, :3].any()) and bool(c[:, 3:].any()) for c in oracle.contexts)
            )
            boxes = np.array(report["windows"])
            self.assertTrue((boxes[0, :2] < np.array(expected)[0].min(0)).all())

    def test_default_augmentation_event_rates_and_thirteen_normal_copies(self):
        B = 1000
        image = torch.zeros(B, 3, 17, 17)
        image[:, :, 8, 8] = 1
        mask = torch.zeros(B, 1, 17, 17)
        mask[:, :, 8, 8] = 1
        result = smear_limbs_gpu(
            image,
            mask,
            generator=torch.Generator().manual_seed(27),
            body_height=torch.ones(B) * 0.5,
        )
        changed = (result - image).abs().flatten(1).sum(1) > 0
        self.assertLess(abs(float(changed.float().mean()) - 0.35), 0.035)
        normalized = torch.zeros(10000, 3, 5, 5)
        shadow = inject_contact_shadow_gpu(
            normalized,
            torch.full((10000, 1, 2), 0.5),
            torch.ones(10000, 1),
            torch.ones(10000),
            generator=torch.Generator().manual_seed(28),
        )
        changed = shadow.abs().flatten(1).sum(1) > 1e-6
        self.assertLess(abs(float(changed.float().mean()) - 0.45), 0.02)
        one = image[:1]
        m = mask[:1]
        with patch("reader_v3.F.grid_sample", wraps=F.grid_sample) as sampler:
            output = smear_limbs_gpu(
                one,
                m,
                directions=torch.tensor([[[1.0, 0.0]]]),
                probability=1,
                generator=torch.Generator().manual_seed(1),
                body_height=torch.tensor([0.5]),
            )
        self.assertEqual(
            sampler.call_count, 26
        )  # 13 RGB samples + 13 corresponding masks.
        self.assertTrue(torch.equal(output[:, :, :, :8], one[:, :, :, :8]))
        self.assertTrue(torch.equal(image[:, :, 8, 8], torch.ones(B, 3)))

    def test_local_reader_refines_nested_fingers_toes_wing_sides_only_above_confidence(
        self,
    ):
        rig = point_rig()
        camera = Camera(np.eye(3), np.array([0, 0, 4.0]), 320, 192, scale=3)
        K = len(rig.point_names)

        class ConstantReader:
            num_keypoints = K
            _decode_logits_dsnt = staticmethod(
                MonsterDINOv2PoseProbe._decode_logits_dsnt
            )

            def __init__(self):
                self.local_calls = 0

            def __call__(self, x, ids, return_aux=True, context=None):
                if x.shape[-1] == 224:
                    self.local_calls += 1
                    coords = torch.tensor([0.55, 0.45]).repeat(1, K, 1)
                    confidence = torch.full((1, K), 0.51)
                    confidence[0, rig.point_names.index("Finger1_1.L.head")] = 0.5
                    confidence[0, rig.point_names.index("Toe1_1.R.tail")] = 0.49
                    return coords, confidence, None, None
                logits = torch.logit(
                    (heatmaps(torch.full((1, K, 2), 0.5), 32) * 0.98 + 0.005).clamp(
                        0.005, 0.995
                    )
                )
                uv, conf = self._decode_logits_dsnt(logits)
                return uv, conf, logits, None

        first = (
            [camera.project(rig.fk(rig.start_params())[0])],
            [np.ones(K)],
            [torch.zeros(1, K, 32, 32)],
        )
        image = np.zeros((192, 320, 3), np.uint8)
        baseline, base_conf, _ = infer_reader_sequence(
            ConstantReader(), [image], 0, "cpu", first_pass=first
        )
        model = ConstantReader()
        result, confidence, _ = infer_reader_sequence(
            model, [image], 0, "cpu", rig, camera, first_pass=first
        )
        self.assertEqual(
            model.local_calls, 3
        )  # Hand, foot and wing hand, each at 224px.
        expected_changed = []
        for k, (name, owner) in enumerate(zip(rig.point_names, rig.point_bones)):
            inside = owner.startswith(("Finger", "Toe")) or (
                owner.startswith(("Hand", "Foot", "WingHand")) and ".side" in name
            )
            eligible = inside and name not in ("Finger1_1.L.head", "Toe1_1.R.tail")
            if eligible:
                expected_changed.append(name)
                self.assertGreater(
                    np.linalg.norm(result[0, k] - baseline[0, k]), 0.1, name
                )
                self.assertAlmostEqual(float(confidence[0, k]), 0.51, places=6)
            else:
                np.testing.assert_allclose(
                    result[0, k], baseline[0, k], atol=1e-6, err_msg=name
                )
                self.assertAlmostEqual(
                    float(confidence[0, k]), float(base_conf[0, k]), places=6
                )
        self.assertIn("Finger1_2.L.tail", expected_changed)
        self.assertIn("WingHand.R.sidea1", expected_changed)
        self.assertIn("Finger2_1.R.tip", expected_changed)

    def test_rgb_and_context_receive_gradients_and_occluded_heatmap_target_is_zero(
        self,
    ):
        from test_pose_finetuning import TinyBackbone

        model = MonsterDINOv2PoseProbe(
            5,
            custom_backbone=TinyBackbone(),
            enable_depth_visibility=True,
            enable_dense_surface=True,
            reader_v3=True,
            num_bones=2,
        )
        coords = torch.tensor(
            [[[0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.3, 0.7], [0.8, 0.3]]]
        )
        visible = torch.tensor([[1.0, 0, 1, 1, 1]])
        previous = coords[:, None].repeat(1, 2, 1, 1) + 0.05
        with patch("reader_v3.CONTEXT_SHARE", 1):
            context = temporal_training_context(
                coords, visible, True, previous, visible[:, None].repeat(1, 2, 1)
            )
        _, _, logits, aux = model(
            torch.randn(1, 3, 56, 56), torch.tensor([0]), True, context
        )
        logits.retain_grad()
        h, w = logits.shape[-2:]
        labels = {
            "visibility_ratio": visible,
            "in_frame": torch.ones_like(visible),
            "label_valid": torch.ones_like(visible),
            "part": torch.ones(1, h, w, dtype=torch.long),
            "local": torch.zeros(1, 3, h, w),
            "relative_depth": torch.zeros(1, 1, h, w),
        }
        loss = reader_losses(logits, aux, coords, labels)["total"]
        loss.backward()
        self.assertTrue((logits.grad[:, 1] > 0).all())
        self.assertTrue((logits.grad[:, 0] < 0).any())
        for module in (model.rgb_detail_branch[0], model.reader_heads.context[0]):
            self.assertTrue(torch.isfinite(module.weight.grad).all())
            self.assertGreater(float(module.weight.grad.abs().sum()), 0)
        self.assertFalse(bool(context[:, 1].any()))
        self.assertFalse(bool(context[:, 6].any()))


if __name__ == "__main__":
    unittest.main()
