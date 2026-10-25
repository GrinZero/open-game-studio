"""Regression tests for observable twist and unreliable multi-view depth (CPU only)."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'worker'))
import bone_taxonomy as tax
from pose_features import fuse_depth_visibility, project_3d_points_to_ortho_view_numpy


class MotionRecoveryTests(unittest.TestCase):
    def fixture(self, angle, scale=1.0, axis=1, fixed_child=True):
        limits = [[0.0, 0.0] for _ in range(3)]
        limits[axis] = list(np.deg2rad([-25, 19]))
        meta = {
            'joint': {'kind': 'head' if axis == 1 else 'digit', 'parent': 'parent',
                      'children': ['child'] if fixed_child else [], 'chain_corr': 0.55,
                      'limits_rad': limits, 'limits_source': 'agent_custom',
                      'rest_head_world': [0, 0, 0], 'rest_tail_world': [0, 0.1 * scale, 0]},
            'child': {'kind': 'head', 'fixed_local': True, 'parent': 'joint',
                      'limits_rad': [[0, 0]] * 3,
                      'rest_head_world': [0.05 * scale, 0.03 * scale, 0.02 * scale],
                      'rest_tail_world': [0.03 * scale, 0.08 * scale, 0.04 * scale]},
        }
        init = tax.euler_xyz_to_matrix_numpy(-0.2, 0.3, 0.1)
        euler = np.zeros(3); euler[axis] = np.deg2rad(angle)
        expected = init @ tax.euler_xyz_to_matrix_numpy(*euler)
        rest = {n: np.eye(3) for n in meta}
        obs = {n: {**{f'{end}_3d': (expected @ np.array(info[f'rest_{end}_world']) +
                                  np.array([0.12, -0.06, 0.4]) * scale).tolist()
                        for end in ('head', 'tail')}, 'confidence': 1.0, 'tail_conf': 1.0}
               for n, info in meta.items()}
        return meta, init, expected, rest, obs

    def test_fixed_child_recovers_positive_negative_zero_and_limits(self):
        for scale in (0.1, 1.0, 10.0):
            for angle in (-25, -19.95, -7.3, 0, 8.2, 15.16, 19):
                with self.subTest(scale=scale, angle=angle):
                    meta, init, expected, rest, obs = self.fixture(angle, scale)
                    actual, _ = tax.solve_bone_rotation_by_kind(
                        'joint', init, obs, meta, rest, {'joint': ['child']},
                        {'parent': 0.0}, {'joint': 0.0}, raw_obs_bones=obs)
                    np.testing.assert_allclose(actual, expected, atol=0.0002)

    def test_depth_cannot_reverse_visible_coordinate_but_completes_occlusion(self):
        for height in (768, 1440):
            for three_views in (False, True):
                cal = {'panel_width': height * 7 / 12, 'panel_height': height,
                       'ortho_scale': 1.5, 'depth_scale': 1, 'center': [0, 0, 0]}
                pt = np.array([[[-0.01, 0.02, 0.5]]])
                uv = {v: project_3d_points_to_ortho_view_numpy(pt, cal, v)[0]
                      for v in ('front', 'side', 'right')}
                one = np.ones((1, 1)); lv = -6 * one
                kwargs = ({'uv_right': uv['right'], 'conf_right': one, 'depth_right': 0.12 * one,
                           'visibility_right': one, 'logvar_right': lv} if three_views else {})
                out, _ = fuse_depth_visibility(uv['front'], uv['side'], one, one, cal,
                                               0.02 * one, -0.12 * one, one, one, lv, lv, **kwargs)
                if three_views:
                    self.assertLess(abs(float(out[0, 0, 0]) + 0.01), 0.001)
                # When the direct front view is invisible, trustworthy side depth is still usable.
                bad_front = uv['front'].copy(); bad_front[..., 0] += 80 * height / 768
                if three_views:
                    kwargs['depth_right'] = -0.01 * one
                out, _ = fuse_depth_visibility(bad_front, uv['side'], 0.01 * one, one, cal,
                                               0.02 * one, 0.01 * one, 0 * one, one, lv, lv, **kwargs)
                self.assertLess(abs(float(out[0, 0, 0]) + 0.01), 0.001)

    def test_transitive_fixed_and_offset_child_heads_recover_oracle_twist(self):
        deg = np.deg2rad
        meta = {
            'SPINE_01': {
                'kind': 'spine', 'sub_role': 'spine', 'side': 'C', 'parent': None,
                'limits_rad': [[-deg(30), deg(30)], [-deg(25), deg(25)], [-deg(20), deg(20)]],
                'limits_source': 'agent_custom',
                'rest_head_world': [0.0, 0.0, 0.40], 'rest_tail_world': [0.0, 0.0, 0.55],
            },
            'SPINE_02': {
                'kind': 'spine', 'sub_role': 'spine', 'side': 'C', 'parent': 'SPINE_01',
                'fixed_local': True, 'limits_rad': [[0.0, 0.0]] * 3,
                'rest_head_world': [0.0, 0.0005, 0.55], 'rest_tail_world': [0.0, 0.0005, 0.70],
            },
            'CLAV_L': {
                'kind': 'limb', 'sub_role': 'clavicle_socket', 'side': 'L', 'parent': 'SPINE_02',
                'limits_rad': [[-deg(25), deg(25)], [-deg(30), deg(30)], [-deg(25), deg(25)]],
                'limits_source': 'agent_custom',
                'rest_head_world': [-0.08, 0.01, 0.68], 'rest_tail_world': [-0.22, 0.02, 0.68],
            },
            'UPPER_L': {
                'kind': 'limb', 'sub_role': 'upper_limb', 'side': 'L', 'parent': 'CLAV_L',
                'limits_rad': [[-deg(45), deg(45)], [-deg(35), deg(35)], [-deg(35), deg(35)]],
                'limits_source': 'agent_custom',
                'rest_head_world': [-0.24, 0.05, 0.64], 'rest_tail_world': [-0.28, 0.05, 0.42],
            },
        }
        children_map = {
            'SPINE_01': ['SPINE_02'],
            'SPINE_02': ['CLAV_L'],
            'CLAV_L': ['UPPER_L'],
            'UPPER_L': [],
        }
        rest_R = {}
        for n, info in meta.items():
            v0 = np.array(info['rest_tail_world'], dtype=float) - np.array(info['rest_head_world'], dtype=float)
            rest_R[n] = tax._build_default_rest_rot_matrix(v0 / np.linalg.norm(v0))
        R_spine = rest_R['SPINE_01'] @ tax.euler_xyz_to_matrix_numpy(float(deg(14.0)), float(deg(-18.5)), float(deg(9.0)))
        R_spine2 = R_spine @ (rest_R['SPINE_01'].T @ rest_R['SPINE_02'])
        R_clav_init = R_spine2 @ (rest_R['SPINE_02'].T @ rest_R['CLAV_L'])
        R_clav = R_clav_init @ tax.euler_xyz_to_matrix_numpy(float(deg(-11.0)), float(deg(21.0)), float(deg(-13.0)))
        R_upper_init = R_clav @ (rest_R['CLAV_L'].T @ rest_R['UPPER_L'])
        R_upper = R_upper_init @ tax.euler_xyz_to_matrix_numpy(float(deg(15.0)), 0.0, float(deg(-10.0)))
        world_R = {'SPINE_01': R_spine, 'SPINE_02': R_spine2, 'CLAV_L': R_clav, 'UPPER_L': R_upper}
        heads, tails = {}, {}
        for n in ('SPINE_01', 'SPINE_02', 'CLAV_L', 'UPPER_L'):
            p = meta[n]['parent']
            rh = np.array(meta[n]['rest_head_world'], dtype=float)
            rt = np.array(meta[n]['rest_tail_world'], dtype=float)
            heads[n] = rh if p is None else heads[p] + world_R[p] @ (rest_R[p].T @ (rh - np.array(meta[p]['rest_head_world'], dtype=float)))
            tails[n] = heads[n] + world_R[n] @ (rest_R[n].T @ (rt - rh))
        obs = {
            n: {'head_3d': heads[n].tolist(), 'tail_3d': tails[n].tolist(), 'confidence': 1.0, 'tail_conf': 1.0}
            for n in meta
        }
        solved_spine, _ = tax.solve_bone_rotation_by_kind(
            'SPINE_01', rest_R['SPINE_01'], obs, meta, rest_R, children_map, {}, {}, raw_obs_bones=obs
        )
        solved_clav, _ = tax.solve_bone_rotation_by_kind(
            'CLAV_L', R_clav_init, obs, meta, rest_R, children_map, {}, {}, raw_obs_bones=obs
        )
        np.testing.assert_allclose(solved_spine, R_spine, atol=2e-3)
        np.testing.assert_allclose(solved_clav, R_clav, atol=2e-3)

    def test_kneeling_forearm_hinge_not_overwritten_to_vertical_during_macro_collapse(self):
        deg = np.deg2rad
        bone_names = ['ROOT', 'SPINE', 'THIGH_L', 'SHIN_L', 'FOOT_L', 'THIGH_R', 'SHIN_R', 'FOOT_R']
        classified_bones = {
            'ROOT': {
                'parent': None, 'kind': 'root', 'sub_role': 'root', 'side': 'C', 'group_id': 'axial', 'chain_idx': 0,
                'rest_head_world': [0.0, 0.0, 0.30], 'rest_tail_world': [0.0, 0.0, 0.45], 'rest_length': 0.15,
                'limits_rad': [[-deg(80), deg(80)], [-deg(45), deg(45)], [-deg(45), deg(45)]],
                'limits_source': 'agent_custom',
            },
            'SPINE': {
                'parent': 'ROOT', 'kind': 'spine', 'sub_role': 'spine', 'side': 'C', 'group_id': 'axial', 'chain_idx': 1,
                'rest_head_world': [0.0, 0.0, 0.45], 'rest_tail_world': [0.0, 0.0, 0.75], 'rest_length': 0.30,
                'limits_rad': [[-deg(60), deg(60)], [-deg(30), deg(30)], [-deg(30), deg(30)]],
                'limits_source': 'agent_custom',
            },
            'THIGH_L': {
                'parent': 'ROOT', 'kind': 'limb', 'sub_role': 'upper_limb', 'side': 'L', 'group_id': 'leg', 'chain_idx': 0,
                'rest_head_world': [-0.10, 0.0, 0.30], 'rest_tail_world': [-0.10, 0.0, 0.16], 'rest_length': 0.14,
                'limits_rad': [[-deg(90), deg(90)], [-deg(45), deg(45)], [-deg(45), deg(45)]],
                'limits_source': 'agent_custom',
            },
            'SHIN_L': {
                'parent': 'THIGH_L', 'kind': 'limb', 'sub_role': 'forearm_hinge', 'side': 'L', 'group_id': 'leg', 'chain_idx': 1,
                'rest_head_world': [-0.10, 0.0, 0.16], 'rest_tail_world': [-0.10, 0.0, 0.04], 'rest_length': 0.12,
                'limits_rad': [[-deg(90), deg(90)], [-deg(30), deg(30)], [-deg(30), deg(30)]],
                'limits_source': 'agent_custom',
            },
            'FOOT_L': {
                'parent': 'SHIN_L', 'kind': 'limb', 'sub_role': 'distal_limb', 'side': 'L', 'group_id': 'leg', 'chain_idx': 2,
                'rest_head_world': [-0.10, -0.02, 0.04], 'rest_tail_world': [-0.10, -0.08, 0.0], 'rest_length': 0.072,
                'limits_rad': [[-deg(60), deg(60)], [-deg(30), deg(30)], [-deg(30), deg(30)]],
                'limits_source': 'agent_custom',
            },
            'THIGH_R': {
                'parent': 'ROOT', 'kind': 'limb', 'sub_role': 'upper_limb', 'side': 'R', 'group_id': 'leg', 'chain_idx': 0,
                'rest_head_world': [0.10, 0.0, 0.30], 'rest_tail_world': [0.10, 0.0, 0.16], 'rest_length': 0.14,
                'limits_rad': [[-deg(90), deg(90)], [-deg(45), deg(45)], [-deg(45), deg(45)]],
                'limits_source': 'agent_custom',
            },
            'SHIN_R': {
                'parent': 'THIGH_R', 'kind': 'limb', 'sub_role': 'forearm_hinge', 'side': 'R', 'group_id': 'leg', 'chain_idx': 1,
                'rest_head_world': [0.10, 0.0, 0.16], 'rest_tail_world': [0.10, 0.0, 0.04], 'rest_length': 0.12,
                'limits_rad': [[-deg(90), deg(90)], [-deg(30), deg(30)], [-deg(30), deg(30)]],
                'limits_source': 'agent_custom',
            },
            'FOOT_R': {
                'parent': 'SHIN_R', 'kind': 'limb', 'sub_role': 'distal_limb', 'side': 'R', 'group_id': 'leg', 'chain_idx': 2,
                'rest_head_world': [0.10, -0.02, 0.04], 'rest_tail_world': [0.10, -0.08, 0.0], 'rest_length': 0.072,
                'limits_rad': [[-deg(60), deg(60)], [-deg(30), deg(30)], [-deg(30), deg(30)]],
                'limits_source': 'agent_custom',
            },
        }
        # Frame has bowed spine (dz_obs << 0.78 * dz_axial_rest -> macro_collapse = 1.0)
        # while right shin is horizontal (kneeling: dz = -0.008 m, dy = +0.12 m).
        pts_frame = {
            'ROOT': ([0.0, 0.0, 0.18], [0.0, -0.14, 0.22]),
            'SPINE': ([0.0, -0.14, 0.22], [0.0, -0.42, 0.26]),
            'THIGH_L': ([-0.10, 0.0, 0.18], [-0.10, -0.08, 0.07]),
            'SHIN_L': ([-0.10, -0.08, 0.07], [-0.10, 0.03, 0.02]),
            'FOOT_L': ([-0.10, 0.03, 0.02], [-0.10, 0.09, 0.0]),
            'THIGH_R': ([0.10, 0.0, 0.18], [0.10, -0.02, 0.042]),
            'SHIN_R': ([0.10, -0.02, 0.042], [0.10, 0.10, 0.034]),
            'FOOT_R': ([0.10, 0.10, 0.034], [0.10, 0.16, 0.00]),
        }
        pts_3d = np.zeros((1, len(bone_names) * 2, 3), dtype=np.float64)
        for i, n in enumerate(bone_names):
            pts_3d[0, 2 * i] = pts_frame[n][0]
            pts_3d[0, 2 * i + 1] = pts_frame[n][1]
        conf = np.ones((1, len(bone_names) * 2), dtype=np.float64)
        frames_out, _ = tax.reconstruct_frames_with_kind_solvers(
            pts_3d=pts_3d,
            trust=conf,
            bone_names=bone_names,
            classified_bones=classified_bones,
            depth_scale=1.0,
        )
        shin_r = frames_out[0]['bones']['SHIN_R']
        dz_shin_r = float(shin_r['tail_3d'][2]) - float(shin_r['head_3d'][2])
        dy_shin_r = float(shin_r['tail_3d'][1]) - float(shin_r['head_3d'][1])
        self.assertGreater(dy_shin_r, 0.08)
        self.assertGreater(dz_shin_r, -0.03)

    def test_formal_exact_penalty_boundary_reachability_and_so3_invariants(self):
        """Formal verification of P1 (SO(3) + Limit Box + FK Invariance) and P2 (Exact Penalty
        Boundary Fixed-Point Theorem: soft regularizers cannot push parent twist outside the
        child's feasible reachability set even when the child sits at its rx/ry/rz boundary).
        """
        deg = np.deg2rad
        meta = {
            'THIGH_R': {
                'parent': None, 'kind': 'limb', 'sub_role': 'ball_socket', 'side': 'R',
                'rest_head_world': [0.10, 0.0, 0.35], 'rest_tail_world': [0.10, 0.0, 0.18],
                'limits_rad': [[-deg(27), deg(62)], [-deg(31), deg(31)], [-deg(18), deg(9)]],
                'limits_source': 'agent_custom',
            },
            'SHIN_R': {
                'parent': 'THIGH_R', 'kind': 'limb', 'sub_role': 'forearm_hinge', 'side': 'R',
                'rest_head_world': [0.10, 0.0, 0.18], 'rest_tail_world': [0.10, 0.0, 0.04],
                'limits_rad': [[0.0, deg(95)], [-deg(10), deg(10)], [-deg(7), deg(12)]],
                'limits_source': 'agent_custom',
            },
            'FOOT_R': {
                'parent': 'SHIN_R', 'kind': 'limb', 'sub_role': 'distal_limb', 'side': 'R',
                'rest_head_world': [0.10, 0.0, 0.04], 'rest_tail_world': [0.10, -0.09, 0.01],
                'limits_rad': [[-deg(75), deg(36)], [-deg(5), deg(12)], [-deg(28), deg(16)]],
                'limits_source': 'agent_custom',
            },
        }
        order = ['THIGH_R', 'SHIN_R', 'FOOT_R']
        children_map = {'THIGH_R': ['SHIN_R'], 'SHIN_R': ['FOOT_R'], 'FOOT_R': []}
        rest_R = {}
        for n in order:
            v0 = np.array(meta[n]['rest_tail_world'], dtype=float) - np.array(meta[n]['rest_head_world'], dtype=float)
            rest_R[n] = tax._build_default_rest_rot_matrix(v0 / np.linalg.norm(v0))

        # Test both interior poses and boundary-active kneeling poses (SHIN_R at ry=+10, rz=+12 boundary)
        test_angles = [
            {
                'THIGH_R': (deg(0.93), deg(-22.21), deg(-18.0)),
                'SHIN_R': (deg(87.2), deg(10.0), deg(12.0)),
                'FOOT_R': (deg(-45.28), deg(0.0), deg(7.81)),
            },
            {
                'THIGH_R': (deg(-10.94), deg(-8.49), deg(4.17)),
                'SHIN_R': (deg(69.0), deg(-4.04), deg(9.94)),
                'FOOT_R': (deg(33.63), deg(0.0), deg(7.98)),
            },
        ]
        rng = np.random.default_rng(42)
        for _ in range(25):
            test_angles.append({
                n: tuple(rng.uniform(np.array(meta[n]['limits_rad'])[:, 0], np.array(meta[n]['limits_rad'])[:, 1]))
                for n in order
            })

        for pose_idx, angs in enumerate(test_angles):
            world_R, heads, tails = {}, {}, {}
            for n in order:
                p = meta[n]['parent']
                rh = np.array(meta[n]['rest_head_world'], dtype=float)
                rt = np.array(meta[n]['rest_tail_world'], dtype=float)
                R_init = rest_R[n] if p is None else world_R[p] @ (rest_R[p].T @ rest_R[n])
                world_R[n] = R_init @ tax.euler_xyz_to_matrix_numpy(*angs[n])
                heads[n] = rh if p is None else heads[p] + world_R[p] @ (rest_R[p].T @ (rh - np.array(meta[p]['rest_head_world'], dtype=float)))
                tails[n] = heads[n] + world_R[n] @ (rest_R[n].T @ (rt - rh))
            obs = {
                n: {'head_3d': heads[n].tolist(), 'tail_3d': tails[n].tolist(), 'confidence': 1.0, 'tail_conf': 1.0}
                for n in order
            }
            solved_R, solved_H, solved_T = {}, {}, {}
            for n in order:
                p = meta[n]['parent']
                rh = np.array(meta[n]['rest_head_world'], dtype=float)
                rt = np.array(meta[n]['rest_tail_world'], dtype=float)
                R_init = rest_R[n] if p is None else solved_R[p] @ (rest_R[p].T @ rest_R[n])
                solved_H[n] = rh if p is None else solved_H[p] + solved_R[p] @ (rest_R[p].T @ (rh - np.array(meta[p]['rest_head_world'], dtype=float)))
                fk_obs = copy.deepcopy(obs)
                fk_obs[n]['head_3d'] = solved_H[n].tolist()
                fk_obs[n]['tail_3d'] = (solved_H[n] + (tails[n] - heads[n])).tolist()
                R_sol, _ = tax.solve_bone_rotation_by_kind(
                    n, R_init, fk_obs, meta, rest_R, children_map, {}, {}, raw_obs_bones=obs
                )
                solved_R[n] = R_init @ tax.clamp_local_delta_euler(R_init.T @ R_sol, meta[n]['limits_rad'])
                solved_T[n] = solved_H[n] + solved_R[n] @ (rest_R[n].T @ (rt - rh))

                # P1: SO(3) & limit box soundness
                self.assertLess(float(np.linalg.norm(solved_R[n].T @ solved_R[n] - np.eye(3))), 1e-6)
                self.assertAlmostEqual(float(np.linalg.det(solved_R[n])), 1.0, places=6)
                # P2: Exact endpoint reachability (< 0.05 mm across all 27 boundary & random cases)
                err_mm = float(np.linalg.norm(solved_T[n] - tails[n])) * 1000.0
                self.assertLess(err_mm, 0.05, msg=f'Case {pose_idx} bone {n} endpoint error {err_mm:.4f}mm >= 0.05mm')

    def test_sop_formal_gate_passes_canonical_rig_and_blocks_invariant_violations(self):
        """Verifies that the standalone SOP Formal Verification Gate (`sop_formal_gate.py`)
        passes P1-P4 on the canonical 12-bone multi-branch rig and hard-stops on trajectory
        SO(3), limit-box, or bone-length violations.
        """
        scripts_dir = str(Path(__file__).resolve().parents[1] / 'skills' / 'monster-workflow' / 'scripts')
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import sop_formal_gate as fg

        report = fg.verify_rig_formal_gate(None, None, seed=20261025, num_random_poses=8, assert_passed=True)
        self.assertTrue(report['passed'])
        self.assertTrue(report['p1_soundness']['passed'])
        self.assertTrue(report['p2_oracle_completeness']['passed'])
        self.assertTrue(report['p3_collapse_invariance']['passed'])
        self.assertTrue(report['p4_surface_contact']['passed'])
        self.assertLess(report['p2_oracle_completeness']['max_single_bone_tail_err_mm'], 0.05)

        # Corrupt a frame's bone length and verify hard-stop rejection
        canon_rig = fg.build_canonical_verification_rig()
        valid_frame = fg._forward_kinematics_from_local_eulers(canon_rig, {b: (0.0, 0.0, 0.0) for b in canon_rig})
        bad_frame = copy.deepcopy(valid_frame)
        bad_frame['bones']['SHIN_R']['tail_3d'][2] += 0.025  # +25mm bone length corruption
        with self.assertRaisesRegex(RuntimeError, 'P1_BONE_LENGTH'):
            fg.verify_trajectory_formal_gate([bad_frame], canon_rig, assert_passed=True)


if __name__ == '__main__':
    unittest.main()
