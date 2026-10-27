"""Absolute visibility evidence must survive V3 decoding, TTA and local crops."""
from pathlib import Path
import os
import sys
import unittest
from unittest.mock import patch
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'worker'))
from dinov2_pose_trainer import MonsterDINOv2PoseProbe, dsnt_soft_argmax_numpy
from reader_v3 import infer_reader_sequence, heatmaps
from reprojection_solver import RigModel
from studio_camera import Camera
from test_pose_finetuning import TinyBackbone

class ConfidenceRegressionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        environment = patch.dict(os.environ, {'MONSTER_DECODE_MODE': 'dsnt'})
        environment.start()
        self.addCleanup(environment.stop)

    def make_model(self, keypoints, local_weak=False, weak=False):
        model = MonsterDINOv2PoseProbe(
            keypoints, custom_backbone=TinyBackbone(), enable_depth_visibility=True,
            enable_dense_surface=True, reader_v3=True, num_bones=max(1, keypoints // 2),
        ).eval()
        with torch.no_grad():
            model.head[-1].weight.zero_()
            model.head[-1].bias.zero_()
            model.rgb_kp_refine.weight.zero_()
            model.rgb_kp_refine.bias.zero_()
        def fields(fused, context):
            B, _, H, W = fused.shape
            is_local = W == 64  # 224px input, rather than the 448px full reader.
            center = 0.8 if is_local and local_weak else 0.5
            amplitude = 0.0003353501 if weak or (is_local and local_weak) else 0.95
            coords = torch.full((B, keypoints, 2), center, device=fused.device)
            logits = torch.logit((heatmaps(coords, W, sigma=3) * amplitude).clamp_min(1e-11))
            return logits, {}
        model.reader_heads.forward = fields
        return model

    def test_v3_peak_height_does_not_promote_near_zero_peaked_maps(self):
        logits = torch.full((1, 3, 128, 128), -25.)
        logits[0, 0, 64, 64] = -8.
        logits[0, 1] = -1000.
        logits[0, 2] = -12.
        coords, confidence = MonsterDINOv2PoseProbe._decode_logits_dsnt(
            logits, peak_height_confidence=True)
        torch.testing.assert_close(confidence, logits.sigmoid().amax((-2, -1)))
        self.assertLess(float(confidence.max()), 0.001)
        self.assertEqual(float(confidence[0, 1]), 0.)
        self.assertTrue(bool(torch.isfinite(coords).all()))
        legacy_coords, legacy_conf = MonsterDINOv2PoseProbe._decode_logits_dsnt(logits)
        torch.testing.assert_close(coords, legacy_coords)
        self.assertEqual(float(legacy_conf[0, 0]), 1.)
        _, reference_conf = dsnt_soft_argmax_numpy(logits.numpy())
        np.testing.assert_allclose(legacy_conf, reference_conf)

    def test_visible_peak_and_parabola_retain_response_height(self):
        logits = torch.full((1, 1, 128, 128), -25.)
        logits[0, 0, 50, 70] = torch.logit(torch.tensor(.95))
        for mode in ('dsnt', 'parabola'):
            with self.subTest(mode=mode), patch.dict(os.environ, {'MONSTER_DECODE_MODE': mode}):
                _, confidence = MonsterDINOv2PoseProbe._decode_logits_dsnt(logits, peak_height_confidence=True)
                self.assertAlmostEqual(float(confidence[0, 0]), .95, places=6)

    def test_actual_model_coarse_tta_and_224_forward_keep_low_visibility(self):
        model = self.make_model(1, weak=True)
        image = np.zeros((448, 448, 3), np.uint8)
        with torch.inference_mode():
            _, local_conf, local_logits, _ = model(torch.zeros(1, 3, 224, 224), torch.tensor([0]), return_aux=True)
        torch.testing.assert_close(local_conf, local_logits.sigmoid().amax((-2, -1)))
        coarse = infer_reader_sequence(model, [image], 0, 'cpu', coarse_only=True)
        self.assertLess(float(coarse[1].max()), .001)
        uv, confidence, report = infer_reader_sequence(model, [image], 0, 'cpu', first_pass=coarse)
        self.assertEqual(report['tta'], 8)
        self.assertEqual(report['passes'], 2)
        self.assertLess(float(confidence.max()), .001)
        self.assertTrue(bool(np.isfinite(uv).all()))

    def test_weak_actual_local_reader_cannot_override_visible_finger_reading(self):
        meta = {
            'Root': dict(parent=None, kind='root', rest_head_world=[0, 0, .2], rest_tail_world=[0, 0, .8]),
            'Hand.L': dict(parent='Root', kind='limb', rest_head_world=[.1, 0, .8], rest_tail_world=[.3, 0, .8]),
            'Finger1_1.L': dict(parent='Hand.L', kind='digit', rest_head_world=[.3, 0, .8], rest_tail_world=[.4, 0, .8]),
        }
        rig = RigModel(meta, extra_points=[])
        model = self.make_model(len(rig.point_names), local_weak=True)
        class Wrapper:
            num_keypoints = model.num_keypoints
            _decode_logits_dsnt = staticmethod(MonsterDINOv2PoseProbe._decode_logits_dsnt)
            local_calls = 0
            def __call__(self, x, *args, **kwargs):
                if x.shape[-1] == 224:
                    self.local_calls += 1
                uv, conf, logits, _ = model(x, *args, **kwargs)
                return uv, conf, logits, None  # Isolate confidence from dense geometry.
        camera = Camera(np.eye(3), np.array([0, 0, 4.]), 448, 448, scale=2.)
        first = ([camera.project(rig.fk(rig.start_params())[0])], [np.ones(model.num_keypoints)],
                 [torch.zeros(1, model.num_keypoints, 128, 128)])
        image = np.zeros((448, 448, 3), np.uint8)
        baseline, base_conf, _ = infer_reader_sequence(Wrapper(), [image], 0, 'cpu', first_pass=first)
        wrapped = Wrapper()
        observed, confidence, _ = infer_reader_sequence(wrapped, [image], 0, 'cpu', rig, camera, first_pass=first)
        self.assertEqual(wrapped.local_calls, 1)
        finger_slots = [i for i, owner in enumerate(rig.point_bones) if owner == 'Finger1_1.L']
        np.testing.assert_allclose(observed[:, finger_slots], baseline[:, finger_slots])
        np.testing.assert_allclose(confidence[:, finger_slots], base_conf[:, finger_slots])

if __name__ == '__main__':
    unittest.main()
