"""Offline temporal channels preserve upstream neighbours and saved legacy semantics."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'worker'))
from reader_v3 import (
    TEMPORAL_CONTEXT_CONTRACT, LEGACY_TEMPORAL_CONTEXT_CONTRACT,
    resolve_temporal_context_contract, temporal_pose_points, preceding_pose_points,
    infer_reader_sequence, WINDOW_CONTRACT,
)
from dinov2_pose_trainer import MonsterDINOv2PoseProbe
from test_v3_reader_boundaries import point_rig


class TemporalContextReviewTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_checkpoint_contract_keeps_old_causal_semantics_and_rejects_unknown(self):
        self.assertEqual(resolve_temporal_context_contract({}), LEGACY_TEMPORAL_CONTEXT_CONTRACT)
        for contract in (TEMPORAL_CONTEXT_CONTRACT, LEGACY_TEMPORAL_CONTEXT_CONTRACT):
            self.assertEqual(resolve_temporal_context_contract({'temporal_context_contract': contract}), contract)
        for invalid in (None, '', 'unknown-reader-context'):
            with self.assertRaisesRegex(ValueError, 'Unknown Reader temporal'):
                resolve_temporal_context_contract({'temporal_context_contract': invalid})

    def test_training_neighbours_share_velocity_on_opposite_sides_of_current_pose(self):
        rig = point_rig()
        joints = {n: {'euler_xyz': [0., 0., 0.], 'head_3d': rig.h[n].tolist()} for n in rig.names}

        class FixedVelocity:
            def normal(self, mean, std, count):
                return np.full(count, 0.02)

        velocity = np.full(rig.size, 0.02)
        for entry in rig.slices.values():
            if isinstance(entry, slice):
                velocity[entry.start + 3:entry.stop] *= rig.height * 0.2
        expected = np.array([rig.fk(np.clip(rig.zero() + offset * velocity, rig.lower, rig.upper))[0]
                             for offset in (-1, 1)])
        actual = temporal_pose_points(rig, joints, FixedVelocity())
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertGreater(np.linalg.norm(actual[0] - actual[1]), 0.01)
        legacy = preceding_pose_points(rig, joints, FixedVelocity())
        np.testing.assert_allclose(legacy[0], actual[0], atol=1e-12)
        np.testing.assert_allclose(legacy[1], rig.fk(np.clip(rig.zero() - 2 * velocity, rig.lower, rig.upper))[0], atol=1e-12)

    def test_inference_uses_future_frame_and_upstream_edge_clamping_with_legacy_opt_in(self):
        class ContextRecorder:
            num_keypoints = 3
            _decode_logits_dsnt = staticmethod(MonsterDINOv2PoseProbe._decode_logits_dsnt)

            def __init__(self):
                self.contexts = []

            def __call__(self, image, ids, return_aux=True, context=None):
                self.contexts.append(context.detach().clone())
                logits = torch.full((1, 3, 32, 32), -3.)
                uv, conf = self._decode_logits_dsnt(logits)
                return uv, conf, logits, {}

        panels = [np.zeros((448, 448, 3), np.uint8) for _ in range(3)]
        coarse = np.tile([[200, 200], [224, 224], [248, 248]], (3, 1, 1))
        first_pass = (coarse, np.ones((3, 3)), [torch.full((1, 3, 32, 32), v) for v in (0.1, 0.2, 0.3)])
        window = {'contract': WINDOW_CONTRACT, 'side_px': 448, 'panel_width': 448, 'panel_height': 448}
        expectations = {
            TEMPORAL_CONTEXT_CONTRACT: [(0.1, 0.2), (0.1, 0.3), (0.2, 0.3)],
            LEGACY_TEMPORAL_CONTEXT_CONTRACT: [(0, 0), (0.1, 0), (0.2, 0.1)],
        }
        for contract, expected in expectations.items():
            with self.subTest(contract=contract):
                model = ContextRecorder()
                _, _, report = infer_reader_sequence(model, panels, 0, 'cpu', first_pass=first_pass,
                    window=window, temporal_context_contract=contract)
                self.assertEqual(report['temporal_context_contract'], contract)
                self.assertEqual(report['context_offsets'], [-1, 1] if contract == TEMPORAL_CONTEXT_CONTRACT else [-1, -2])
                for t, values in enumerate(expected):
                    ctx = model.contexts[t * 8]
                    np.testing.assert_allclose(ctx[0, [0, 3], 16, 16], values, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
