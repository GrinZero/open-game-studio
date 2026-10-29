"""CPU regressions for surface-based contact and branch-local collision correction."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'worker'))
sys.path.insert(0, str(ROOT/'skills/monster-workflow/scripts'))
import bone_taxonomy as tax
import sop_joint_limits as limits
import studio_bpy_worker as bpy_worker
from surface_contact import apply_surface_floor_constraints, assert_surface_contact_quality


class SurfaceContactTests(unittest.TestCase):
    def test_probe_covers_all_bound_mesh_parts_and_offsets_triangles(self):
        arm=object()
        parts=[SimpleNamespace(name=n,modifiers=[SimpleNamespace(type='ARMATURE',object=arm)]) for n in ['body','sole']]
        parts.append(SimpleNamespace(name='unbound_ground',modifiers=[]))
        data={'bone_names':['root'],'rest_verts':np.zeros((3,3)),
              'rest_coords':np.zeros((3,3)),'start_verts':np.ones((3,3)),
              'skin_indices':np.zeros((3,4),dtype=int),'skin_weights':np.ones((3,4))*.25,
              'triangles':np.array([[0,1,2]]),'rest_heads':np.zeros((1,3)),'rest_R':np.eye(3)[None]}
        with patch.object(bpy_worker,'extract_rig_lbs_skinning_and_verify_gate',return_value=data) as extract:
            merged=bpy_worker.extract_complete_rig_skinning(parts,arm,['root'])
        self.assertEqual(extract.call_count,2)
        self.assertEqual(merged['measured_mesh_names'],['body','sole'])
        self.assertEqual(merged['rest_verts'].shape,(6,3))
        self.assertEqual(merged['start_verts'].shape,(6,3))
        np.testing.assert_array_equal(merged['start_verts'],np.ones((6,3)))
        np.testing.assert_array_equal(merged['triangles'],[[0,1,2],[3,4,5]])

    def test_view_sync_cannot_approve_an_unresolved_contact(self):
        with self.assertRaisesRegex(RuntimeError, 'hard-stopping before IK bake'):
            assert_surface_contact_quality({'gate_passed':True,'taxonomy_diagnostics':{'contact_quality_passed':False}},'kneel')
        assert_surface_contact_quality({'gate_passed':True,'taxonomy_diagnostics':{'contact_quality_passed':True}})

    def fixture(self, sample_z=.1):
        order=['root','foot']
        heads={'root':np.array([0.,0.,.5]),'foot':np.array([0.,0.,.1])}
        tails={'root':np.array([0.,0.,.7]),'foot':np.array([0.,0.,-.1])}
        rotations={n:np.eye(3) for n in order}
        meta={'root':{'parent':None,'kind':'root'},
              'foot':{'parent':'root','kind':'limb','fixed_local':True}}
        frames=[{'bones':{n:{'head_3d':(heads[n]-[0,0,.2]).tolist(),
                               'tail_3d':(tails[n]-[0,0,.2]).tolist(),
                               'rot_mat_world':np.eye(3).tolist()} for n in order}}
                for _ in range(4)]
        config={'floor_z':0.,'max_iterations':12,'support_sets':[
            {'bone':'foot','lock_xy':True,'samples_local':[[0,0,sample_z],[.02,0,sample_z]]}]}
        return frames,order,'root',meta,heads,tails,rotations,{n:[[0,0]]*3 for n in order},config

    def test_valid_crouch_uses_visible_sole_instead_of_submerged_bone_tip(self):
        args=self.fixture()
        result=apply_surface_floor_constraints(*args)
        self.assertTrue(result['contact_quality_passed'])
        self.assertEqual(result['foot_contact_locked_frames'],4)
        self.assertLess(result['root_contact_max_correction'],1e-5)
        for frame in args[0]:self.assertAlmostEqual(frame['bones']['root']['head_3d'][2],.3,places=5)

    def test_stalled_fit_uses_only_remaining_root_budget(self):
        args=self.fixture(sample_z=.09)
        args[-1]['max_iterations']=0
        result=apply_surface_floor_constraints(*args)
        self.assertTrue(result['contact_quality_passed'])
        self.assertGreater(result['contact_frames'][0]['bounded_fallback_lift'],.009)
        self.assertLessEqual(result['root_contact_max_correction'],.012+1e-8)
        for frame in args[0]:
            self.assertLess(frame['bones']['root']['head_3d'][2],.313)

    def test_output_lift_does_not_seed_the_next_frame(self):
        args=self.fixture(sample_z=.09)
        args[-1]['max_iterations']=0
        for bone in args[0][1]['bones'].values():
            for key in ('head_3d','tail_3d'):
                bone[key][2] += .005
        apply_surface_floor_constraints(*args)
        # The second frame needs half the first frame's correction. Reusing
        # the fallback as the optimizer warm start would leave it floating.
        self.assertAlmostEqual(args[0][1]['bones']['root']['head_3d'][2],.310006,places=5)

    def test_unreachable_floor_is_reported_without_erasing_root_drop(self):
        args=self.fixture(sample_z=-.1)
        result=apply_surface_floor_constraints(*args)
        self.assertFalse(result['contact_quality_passed'])
        self.assertEqual(result['surface_floor_unresolved_frames'],4)
        self.assertLessEqual(result['root_contact_max_correction'],.02*.6+1e-6)
        self.assertLess(args[0][-1]['bones']['root']['head_3d'][2],.32)

    def test_nonfoot_guard_cannot_pass_with_a_submerged_part(self):
        args=self.fixture()
        args[-1]['support_sets'].append({'bone':'root','adjust_controls':False,'samples_local':[[0,0,-.6]]})
        result=apply_surface_floor_constraints(*args)
        self.assertFalse(result['contact_quality_passed'])
        self.assertGreater(result['surface_floor_max_penetration'],.25)

    def test_weighted_support_keeps_the_serializable_configuration_immutable(self):
        args=self.fixture()
        args[-1]['support_sets']=[{'bone':'foot','vertex_count':2,'lock_xy':True,'influences':[
            {'bone':'foot','indices':[0,1],'samples_local':[[0,0,.1],[.02,0,.1]],'weights':[1.,1.]}]}]
        serialized=json.dumps(args[-1],sort_keys=True)
        result=apply_surface_floor_constraints(*args)
        self.assertTrue(result['contact_quality_passed'])
        self.assertEqual(json.dumps(args[-1],sort_keys=True),serialized)

    def test_root_fraction_change_refreshes_both_absolute_consumers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'joint_limits.json'
            profile={'schema':'monster-joint-limits-v1','bones':{},'bbox_min':[-.5,-.2,0],
                     'bbox_max':[.5,.2,2.],
                     'global_root':{'max_translation_world_xyz':[.08,.032,.2]},
                     'root_motion':{'global_pitch_deg':[-20,20],'global_yaw_deg':[-20,20],
                                    'global_roll_deg':[-20,20],'translation_fraction_xyz':[.08,.08,.1],
                                    'max_translation_world_xyz':[.08,.032,.2]}}
            path.write_text(json.dumps(profile))
            with patch.object(limits,'load_ledger',return_value={}), \
                 patch.object(limits,'assert_gate',return_value={'selected_candidate_id':'0'}), \
                 patch.object(limits,'_resolve_g1_candidate_files',return_value=('0',Path(tmp)/'meta',Path(tmp)/'blend')), \
                 patch.object(limits,'resolve_joint_limits_path',return_value=path), \
                 patch.object(limits,'derive_invalidations',return_value=[]), \
                 patch.object(limits,'save_ledger'):
                limits.apply_custom_joint_limits('fixture',set_root_trans='.08,.08,.18',project_root=tmp)
            updated=json.loads(path.read_text())
            self.assertEqual(updated['global_root']['max_translation_world_xyz'],[.08,.032,.36])
            self.assertEqual(updated['root_motion']['max_translation_world_xyz'],[.08,.032,.36])

    def test_kneeling_pose_with_minor_toe_penetration_is_not_reset_to_midpoint_seed(self):
        order = ['root', 'thigh', 'shin', 'foot']
        rest_H = {
            'root': np.array([0.0, 0.0, 0.50]),
            'thigh': np.array([0.0, 0.0, 0.40]),
            'shin': np.array([0.0, 0.0, 0.22]),
            'foot': np.array([0.0, 0.0, 0.06]),
        }
        rest_T = {
            'root': np.array([0.0, 0.0, 0.70]),
            'thigh': np.array([0.0, 0.0, 0.22]),
            'shin': np.array([0.0, 0.0, 0.06]),
            'foot': np.array([0.0, -0.08, 0.00]),
        }
        rest_R = {n: np.eye(3) for n in order}
        meta = {
            'root': {'parent': None, 'kind': 'root'},
            'thigh': {'parent': 'root', 'kind': 'limb'},
            'shin': {'parent': 'thigh', 'kind': 'limb'},
            'foot': {'parent': 'shin', 'kind': 'limb'},
        }
        deg = np.deg2rad
        eff_limits = {
            'root': [[0.0, 0.0]] * 3,
            'thigh': [[float(-deg(85)), float(deg(85))], [0.0, 0.0], [0.0, 0.0]],
            'shin': [[float(-deg(85)), float(deg(85))], [0.0, 0.0], [0.0, 0.0]],
            'foot': [[float(-deg(65)), float(deg(65))], [0.0, 0.0], [0.0, 0.0]],
        }
        # Kneeling pose: root lowered by 0.16m, thigh pitched forward (-60 deg),
        # shin pitched backward (+75 deg), foot slightly penetrating floor (-0.012m).
        R_root = np.eye(3)
        R_thigh = tax.euler_xyz_to_matrix_numpy(float(-deg(60)), 0.0, 0.0)
        R_shin = R_thigh @ tax.euler_xyz_to_matrix_numpy(float(deg(75)), 0.0, 0.0)
        R_foot = R_shin @ tax.euler_xyz_to_matrix_numpy(float(-deg(25)), 0.0, 0.0)
        rots = {'root': R_root, 'thigh': R_thigh, 'shin': R_shin, 'foot': R_foot}
        heads = {'root': np.array([0.0, 0.0, 0.34])}
        tails = {}
        for b in order:
            p = meta[b]['parent']
            if p:
                heads[b] = heads[p] + rots[p] @ (rest_H[b] - rest_H[p])
            tails[b] = heads[b] + rots[b] @ (rest_T[b] - rest_H[b])
        # Support sample on foot sits 12mm below floor_z=0.0 at initial_x, while
        # midpoint seed (all angles 0) would hang deep below floor or require standing reset.
        foot_sample_world = np.array([0.0, -0.05, -0.012])
        sample_local = (R_foot.T @ (foot_sample_world - heads['foot'])).tolist()
        frames = [{
            'bones': {
                b: {
                    'head_3d': heads[b].tolist(),
                    'tail_3d': tails[b].tolist(),
                    'rot_mat_world': rots[b].tolist(),
                }
                for b in order
            }
        }]
        config = {
            'floor_z': 0.0,
            'max_iterations': 24,
            'support_sets': [{'bone': 'foot', 'lock_xy': False, 'samples_local': [sample_local]}],
        }
        orig_knee_Head = heads['shin'].copy()
        result = apply_surface_floor_constraints(
            frames, order, 'root', meta, rest_H, rest_T, rest_R, eff_limits, config
        )
        self.assertTrue(result['contact_quality_passed'])
        new_knee_head = np.array(frames[0]['bones']['shin']['head_3d'], dtype=float)
        # Knee position must stay near the observed kneeling posture (<25mm shift),
        # not jump back toward the 0-degree midpoint pose (>100mm away).
        self.assertLess(float(np.linalg.norm(new_knee_head - orig_knee_Head)), 0.025)

    def test_formal_surface_contact_fixed_point_and_idempotence(self):
        """Formal verification of P4a (Fixed-Point Identity when no surface penetrates the floor)
        and P4b (Idempotence: re-applying surface contact to a resolved trajectory is stable).
        """
        args = self.fixture(sample_z=0.35)
        before = json.loads(json.dumps(args[0]))
        res1 = apply_surface_floor_constraints(*args)
        self.assertTrue(res1['contact_quality_passed'])
        self.assertEqual(res1['root_contact_max_correction'], 0.0)
        for f_before, f_after in zip(before, args[0]):
            for b in f_before['bones']:
                np.testing.assert_allclose(f_after['bones'][b]['head_3d'], f_before['bones'][b]['head_3d'], atol=1e-12)
                np.testing.assert_allclose(f_after['bones'][b]['tail_3d'], f_before['bones'][b]['tail_3d'], atol=1e-12)

        # Idempotence on penetrating crouch fixture
        crouch_args = self.fixture(sample_z=0.09)
        res_first = apply_surface_floor_constraints(*crouch_args)
        self.assertTrue(res_first['contact_quality_passed'])
        after_first = json.loads(json.dumps(crouch_args[0]))
        res_second = apply_surface_floor_constraints(*crouch_args)
        self.assertTrue(res_second['contact_quality_passed'])
        self.assertLess(res_second['root_contact_max_correction'], 5e-5)
        for f1, f2 in zip(after_first, crouch_args[0]):
            for b in f1['bones']:
                np.testing.assert_allclose(f2['bones'][b]['head_3d'], f1['bones'][b]['head_3d'], atol=5e-5)
                np.testing.assert_allclose(f2['bones'][b]['tail_3d'], f1['bones'][b]['tail_3d'], atol=5e-5)


if __name__=='__main__':unittest.main()
