"""Contact correction preserves saved local bases, including frozen translations."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'worker'))
from reprojection_solver import RigModel
from surface_contact import apply_surface_floor_constraints
from test_v3_motion_roots import nested_rig


class ContactBasisReviewTests(unittest.TestCase):
    def fixture(self):
        meta, skin = nested_rig()
        # This independent, unfitted prop also carries a non-identity local basis.
        basis = np.eye(4)
        basis[:3, :3] = Rotation.from_euler('xyz', [.3, -.2, .4]).as_matrix()
        basis[:3, 3] = [.05, .02, -.03]
        meta['Prop_stone'] = dict(
            parent=None, kind='rigid', fixed_local=True,
            rest_head_world=[2., .3, .5], rest_tail_world=[2., .3, .8],
            rest_rot_mat_world=np.eye(3).tolist(), pose_basis=basis.tolist(),
            limits_rad=[[0., 0.]] * 3,
        )
        rig = RigModel(meta, skin=skin)
        frame = rig.frames([rig.start_params()], np.ones((1, 2, len(rig.point_names))))[0]
        limits = {n: meta[n].get('limits_rad', [[-np.pi, np.pi]] * 3) for n in rig.names}
        config = {
            'floor_z': -10.,
            'fitted_bones': list(rig.slices),
            'support_sets': [{'bone': 'Rigid1', 'samples_local': [[0., 0., 0.]]}],
        }
        return rig, frame, limits, config

    def apply(self, rig, frame, limits, config):
        return apply_surface_floor_constraints(
            [frame], rig.order, 'Hips', rig.meta, rig.h, rig.t, rig.R, limits, config,
        )

    def local_state(self, frame, bone, parent):
        b, p = frame['bones'][bone], frame['bones'][parent]
        R = np.asarray(p['rot_mat_world'])
        return (R.T @ np.asarray(b['rot_mat_world']),
                R.T @ (np.asarray(b['head_3d']) - p['head_3d']))

    def test_no_contact_is_a_fixed_point_with_nonzero_rigid_basis_and_promoted_hips(self):
        rig, frame, limits, config = self.fixture()
        before = deepcopy(frame)
        result = self.apply(rig, frame, limits, config)
        self.assertTrue(result['contact_quality_passed'])
        self.assertLess(result['root_contact_max_correction'], 1e-12)
        for n in rig.names:
            for key in ('head_3d', 'tail_3d', 'rot_mat_world'):
                np.testing.assert_allclose(frame['bones'][n][key], before['bones'][n][key], atol=1e-12)

    def test_small_contact_repair_preserves_frozen_local_rotation_and_translation(self):
        rig, frame, limits, config = self.fixture()
        before = deepcopy(frame)
        original = self.local_state(frame, 'Rigid1', 'Spine1')
        config['floor_z'] = frame['bones']['Rigid1']['head_3d'][2] + .006
        result = self.apply(rig, frame, limits, config)
        self.assertTrue(result['contact_quality_passed'])
        self.assertGreater(np.linalg.norm(np.asarray(frame['bones']['Rigid1']['head_3d'])
                                         - before['bones']['Rigid1']['head_3d']), 1e-5)
        after = self.local_state(frame, 'Rigid1', 'Spine1')
        for a, b in zip(after, original):
            np.testing.assert_allclose(a, b, atol=1e-12)
        self.assertLessEqual(result['surface_floor_max_penetration'], .002 * rig.height)
        for n in ('Root', 'Org', 'Prop_stone'):
            for key in ('head_3d', 'tail_3d', 'rot_mat_world'):
                np.testing.assert_allclose(frame['bones'][n][key], before['bones'][n][key], atol=1e-12)

    def test_clear_motion_sequence_is_not_smoothed_by_the_previous_contact_pose(self):
        rig, _, limits, config = self.fixture()
        params = np.repeat(rig.start_params()[None], 3, axis=0)
        indices, _ = rig.slices['Spine1']
        params[:, indices[0]] += [0., .2, -.1]
        frames = rig.frames(params, np.ones((3, 2, len(rig.point_names))))
        before = deepcopy(frames)
        result = apply_surface_floor_constraints(
            frames, rig.order, 'Hips', rig.meta, rig.h, rig.t, rig.R, limits, config,
        )
        self.assertTrue(result['contact_quality_passed'])
        for actual, expected in zip(frames, before):
            for n in rig.names:
                for key in ('head_3d', 'tail_3d', 'rot_mat_world'):
                    np.testing.assert_allclose(actual['bones'][n][key], expected['bones'][n][key], atol=1e-12)

    def test_unfitted_non_rigid_bone_is_not_added_as_a_contact_control(self):
        rig, frame, limits, config = self.fixture()
        # A lightweight bone can be frozen by fit_bones even without fixed_local.
        config['fitted_bones'] = ['Hips']
        before = deepcopy(frame)
        config['floor_z'] = frame['bones']['Rigid1']['head_3d'][2] + .006
        result = self.apply(rig, frame, limits, config)
        self.assertTrue(result['contact_quality_passed'])
        for n, parent in (('Spine1', 'Hips'), ('Rigid1', 'Spine1')):
            for a, b in zip(self.local_state(frame, n, parent), self.local_state(before, n, parent)):
                np.testing.assert_allclose(a, b, atol=1e-12)
        self.assertGreater(result['root_contact_max_correction'], .005)


if __name__ == '__main__':
    unittest.main()
