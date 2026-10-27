"""The v3 oracle gate must reject a solver that only returns its saved starting pose."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'worker'), str(ROOT / 'skills/monster-workflow/scripts')]
import test_v3_formal as formal_fixtures
from reprojection_solver import RigModel, aligned_rest_rotation
from sop_formal_gate import verify_rig_formal_gate, _v3_oracle_probes


class ReviewOracleTests(unittest.TestCase):
    def test_constant_start_pose_solver_is_rejected_for_actual_nonzero_observations(self):
        meta, skin, *_ = formal_fixtures.V3FormalTests().posed()
        calls = []

        def ignores_readings(rig, cameras, uv, confidence, *args, **kwargs):
            # Neither this stub nor the real fitting API receives true channels.
            self.assertEqual(args, ())
            self.assertNotIn('initial_params', kwargs)
            calls.append(uv.copy())
            return np.repeat(rig.start_params()[None], len(uv), axis=0), {}

        with patch('reprojection_solver.fit_clip_2d_reprojection', side_effect=ignores_readings):
            report = verify_rig_formal_gate(meta, rig_skinning=skin, num_random_poses=3, assert_passed=False)
        self.assertFalse(report['passed'])
        self.assertFalse(report['p2_oracle_completeness']['passed'])
        self.assertTrue(any(v.startswith('P2_V3_PIXEL_REACHABILITY') for v in report['violations']))
        self.assertGreater(np.max(np.abs(calls[0][1:] - calls[0][:1])), 1)
        self.assertEqual(len(calls), 2)
        self.assertGreater(len(calls[0]), 1)
        self.assertEqual(len(calls[1]), 1)

    def test_empty_or_nonfinite_solver_output_fails_closed(self):
        meta, skin, *_ = formal_fixtures.V3FormalTests().posed()
        rig = RigModel(meta, skin=skin)
        for bad in (np.zeros((0, rig.size)), np.full((1, rig.size), np.nan)):
            with self.subTest(shape=bad.shape), patch('reprojection_solver.fit_clip_2d_reprojection', return_value=(bad, {})):
                with self.assertRaisesRegex(ValueError, 'P2_V3_INVALID_SOLUTION'):
                    verify_rig_formal_gate(meta, rig_skinning=skin, num_random_poses=0)

    def test_invalid_hidden_frame_solution_fails_closed(self):
        meta, skin, *_ = formal_fixtures.V3FormalTests().posed()

        def invalid_hidden(model, cameras, uv, confidence, *args, **kwargs):
            if confidence.any():
                return np.repeat(model.start_params()[None], len(uv), axis=0), {}
            return np.full((1, model.size), np.nan), {}

        with patch('reprojection_solver.fit_clip_2d_reprojection', side_effect=invalid_hidden):
            with self.assertRaisesRegex(ValueError, 'P3_V3_INVALID_SOLUTION'):
                verify_rig_formal_gate(meta, rig_skinning=skin, num_random_poses=0)

    def test_screening_out_all_observable_motion_cannot_make_the_gate_pass(self):
        meta, skin, rig, *_ = formal_fixtures.V3FormalTests().posed()
        neutral = rig.fk(rig.start_params())[0]
        original_guard = RigModel.guard

        def only_neutral_allowed(model, points, jac):
            residual, derivative = original_guard(model, points, jac)
            forbidden = float(np.max(np.abs(points - neutral)) > 1e-8)
            return np.r_[residual, forbidden], np.vstack([derivative, np.zeros(model.size)])

        with patch.object(RigModel, 'guard', only_neutral_allowed):
            report = verify_rig_formal_gate(meta, rig_skinning=skin, num_random_poses=1, assert_passed=False)
        self.assertFalse(report['passed'])
        self.assertTrue(any(v.startswith('P2_V3_INSUFFICIENT_COVERAGE') for v in report['violations']))
        self.assertEqual(report['p2_oracle_completeness']['coverage']['nonneutral_observable_poses'], 0)

    def test_real_solver_recovers_sampled_boundaries_and_random_p0_rig(self):
        meta, skin, *_ = formal_fixtures.V3FormalTests().posed()
        before = deepcopy(meta)
        report = verify_rig_formal_gate(meta, rig_skinning=skin, num_random_poses=4)
        oracle = report['p2_oracle_completeness']
        labels = [p['pose'] for p in oracle['poses']]
        self.assertTrue(report['passed'])
        self.assertEqual(meta, before)
        self.assertTrue(any(n.startswith('boundary_') and n.endswith('_lower') for n in labels))
        self.assertTrue(any(n.startswith('boundary_') and n.endswith('_upper') for n in labels))
        self.assertTrue(any(n.startswith('random_') for n in labels))
        self.assertGreater(oracle['coverage']['nonneutral_observable_poses'], 0)
        self.assertEqual(oracle['coverage']['random_checked'], 4)
        self.assertLess(oracle['max_pose_point_error_mm'], 1.5)
        self.assertLess(oracle['max_identifiable_rotation_error_deg'], .35)
        self.assertIn('Rigid1', set.union(*(set(p['identifiable_rotation_bones']) for p in oracle['poses'])))
        self.assertTrue(report['p1_recovered_soundness']['passed'])
        json.dumps(report)  # Receipts written by G2 must remain JSON serializable.

    def test_unobservable_leaf_roll_is_reported_without_claiming_rotation_recovery(self):
        meta = {
            'Root': dict(parent=None, fixed_local=True, kind='root', rest_head_world=[0, 0, 1], rest_tail_world=[0, 0, 1.2]),
            'Head': dict(parent='Root', kind='head', rest_head_world=[0, 0, 1.2], rest_tail_world=[0, 0, 1.6], limits_rad=[[-.35, .35]] * 3),
        }
        for info in meta.values():
            info['rest_rot_mat_world'] = aligned_rest_rotation(np.array(info['rest_head_world']), np.array(info['rest_tail_world'])).tolist()
        report = verify_rig_formal_gate(meta, num_random_poses=2)
        self.assertTrue(report['passed'])
        oracle = report['p2_oracle_completeness']
        self.assertTrue(all('Head' in p['unobservable_rotation_bones'] for p in oracle['poses']))
        self.assertLess(oracle['max_pose_point_error_mm'], 1.5)
        self.assertFalse(oracle['coverage']['exhaustive'])

    def test_guard_infeasible_boundaries_are_excluded_and_auditable(self):
        meta, skin, *_ = formal_fixtures.V3FormalTests().posed()
        rig = RigModel(meta, skin=skin)
        original_guard = rig.guard

        def reject_one_boundary(points, jac):
            residual, derivative = original_guard(points, jac)
            # This guard adapter rejects a visible subset of nonneutral targets.
            slot = 2 * rig.index['Spine1'] + 1
            if np.linalg.norm(points[slot] - rig.fk(rig.start_params())[0][slot]) > .05:
                return np.r_[residual, 1.0], np.vstack([derivative, np.zeros(rig.size)])
            return residual, derivative

        with patch.object(rig, 'guard', side_effect=reject_one_boundary):
            probes, coverage = _v3_oracle_probes(rig, 11, 2)
        self.assertTrue(coverage['excluded_poses'])
        self.assertTrue(all(p['reason'] == 'guard_infeasible' for p in coverage['excluded_poses']))
        excluded = {p['pose'] for p in coverage['excluded_poses']}
        self.assertFalse(excluded.intersection(label for label, _ in probes))


if __name__ == '__main__':
    unittest.main()
