"""Behavioral checks for action contact targets, stale inputs and fixed branches."""
import copy
import itertools
import json
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'worker'))
from motion_contact import refine_trajectory, frames_fingerprint


class MotionContactTests(unittest.TestCase):
    def fixture(self):
        meta={}
        for name,parent,head,tail in [('root',None,[0,0,.5],[0,0,.6]),('arm','root',[0,0,.5],[0,.2,.5]),('foot','root',[.2,0,0],[.2,.1,0])]:
            meta[name]={'parent':parent,'depth':0 if parent is None else 1,'kind':'root' if parent is None else 'limb','sub_role':'fixture','group_id':'01','rest_head_world':head,'rest_tail_world':tail,'rest_rot_mat_world':np.eye(3).tolist(),'limits_rad':[[-.7,.7],[0,0],[0,0]] if name=='arm' else [[0,0]]*3,'fixed_local':name!='arm'}
        frames=[{'frame':f,'bones':{b:{'head_3d':v['rest_head_world'],'tail_3d':v['rest_tail_world'],'rot_mat_world':v['rest_rot_mat_world']} for b,v in meta.items()}} for f in range(9)]
        cube=np.array(list(itertools.product([-.02,.02],repeat=3)))
        groups=[]
        for name,bone,center in [('hammer','arm',[0,.25,-.45]),('sole','foot',[0,0,.02])]:
            groups.append({'id':name,'bone':bone,'vertex_count':8,'influences':[{'bone':bone,'indices':list(range(8)),'weights':[1.]*8,'samples_local':(cube+center).tolist()}]})
        profile={'bones':{},'surface_contact':{'support_sets':groups,'floor_z':0.,'penetration_tolerance_fraction':.002}}
        trajectory={'frames':frames,'taxonomy_diagnostics':{}}
        event={'window':[0,8],'phase':[1,3,5,7],'controls':['arm'],'surface':'hammer','surface_axis_local':[0,1,0],'surface_axis_world':[0,1,0],'center_xy':[0,.25],'center_tolerance':[.1,.1],'separation_pairs':[{'a':'hammer','b':'sole','axis':[1,0,0],'gap':.002}],'max_nfev':80}
        config={'schema':'monster-motion-contact-v1','frame_count':9,'input_frames_sha256':frames_fingerprint(frames),'events':[event]}
        return trajectory,meta,profile,config

    def test_contact_reaches_floor_preserves_foot_limits_and_boundary(self):
        args=self.fixture();before=json.dumps(args,sort_keys=True);out=refine_trajectory(*args)
        self.assertEqual(json.dumps(args,sort_keys=True),before)
        report=out['taxonomy_diagnostics']['motion_contact']
        self.assertTrue(report['targets_passed']);self.assertTrue(report['floor_passed'])
        self.assertLess(report['events'][0]['max_contact_height_error'],.003)
        for f in range(9):self.assertEqual(out['frames'][f]['bones']['foot'],args[0]['frames'][f]['bones']['foot'])
        self.assertEqual(out['frames'][0],args[0]['frames'][0]);self.assertEqual(out['frames'][-1],args[0]['frames'][-1])

    def test_unreachable_target_is_not_a_successful_export(self):
        args=self.fixture();args[3]['events'][0]['floor_z']=-2.
        out=refine_trajectory(*args)
        self.assertFalse(out['taxonomy_diagnostics']['contact_quality_passed'])
        self.assertFalse(out['taxonomy_diagnostics']['motion_contact']['targets_passed'])

    def test_no_event_is_an_exact_noop(self):
        args=self.fixture();self.assertIs(refine_trajectory(args[0],args[1],args[2],{}),args[0])

    def test_stale_trajectory_rejected(self):
        args=self.fixture();args[0]['frames'][2]['bones']['arm']['head_3d']=[0,0,.6]
        with self.assertRaisesRegex(ValueError,'fingerprint'):refine_trajectory(*args)

    def test_wrong_window_and_weighted_target_fail_explicitly(self):
        args=self.fixture();args[3]['events'][0]['window']=[-1,8]
        with self.assertRaisesRegex(ValueError,'window'):refine_trajectory(*args)
        args=self.fixture();args[2]['surface_contact']['support_sets'][0]['influences'][0]['weights'][0]=.5
        with self.assertRaisesRegex(ValueError,'rigid surface'):refine_trajectory(*args)

    def test_overlapping_events_rejected(self):
        args=self.fixture();args[3]['events'].append(copy.deepcopy(args[3]['events'][0]))
        with self.assertRaisesRegex(ValueError,'Overlapping'):refine_trajectory(*args)

if __name__=='__main__':unittest.main()
