"""Regression: broad legal boxes must not become all-body boundary poses."""
import random
import unittest
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'worker'))
from pose_sampling_v3 import sample_start_pose
from reprojection_solver import RigModel
from bone_taxonomy import audit_boundary_exploration_coverage

class StartPoseSamplingTests(unittest.TestCase):
    def rig(self):
        meta = {'Root': dict(parent=None, kind='root', rest_head_world=[0,0,0], rest_tail_world=[0,0,1])}
        for i, name in enumerate(('Neck1','Head','LowerArm.L','Rigid1')):
            basis = np.eye(4)
            basis[:3,:3] = Rotation.from_euler('xyz',[.25,.1,-.1]).as_matrix()
            meta[name] = dict(parent='Root', kind='rigid' if name=='Rigid1' else 'head',
                fixed_local=name=='Rigid1', rest_head_world=[i*.2,0,1], rest_tail_world=[i*.2,0,1.2],
                pose_basis=basis.tolist(), limits_source='agent_custom', limits_rad=[[-.4,1.57],[.1,.1],[-.4,.4]],
                axis_roles={'rx':'bend','ry':'twist','rz':'side'}, allow_leaf_twist=True)
        meta['LowerArm.L']['hinge']={'axis':0}
        return RigModel(meta)

    def test_distribution_preserves_start_and_limits_without_joint_extreme_pileup(self):
        rig=self.rig(); rng=random.Random(42)
        samples=[sample_start_pose(rig,rng) for _ in range(2000)]
        calm=np.array([c for _,c in samples]); self.assertTrue(.26 < calm.mean() < .34)
        neck=np.array([e['Neck1'] for e,_ in samples]); head=np.array([e['Head'] for e,_ in samples])
        self.assertLess(np.mean((neck[:,0]>1.3)&(head[:,0]>1.3)), .01)
        self.assertLess(abs(np.median(neck[:,0])-.25), .08)
        self.assertLess(neck[calm,0].std(), .6*neck[~calm,0].std())
        self.assertTrue(np.all(neck[:,0]>=-.4)); self.assertTrue(np.all(neck[:,0]<=1.57))
        np.testing.assert_allclose(neck[:,1],.1)
        np.testing.assert_allclose(np.array([e['Rigid1'] for e,_ in samples]),np.tile([.25,.1,-.1],(2000,1)))
        hinge=np.array([e['LowerArm.L'][0] for e,_ in samples])
        self.assertLess(hinge.min(),-.35); self.assertGreater(hinge.max(),1.5)
        self.assertEqual(sample_start_pose(rig,random.Random(9)),sample_start_pose(rig,random.Random(9)))

    def test_new_audit_reports_sparse_boundaries_but_rejects_frozen_and_overlimit(self):
        rig=self.rig(); rng=random.Random(42)
        records=[]
        for _ in range(100):
            e, calm=sample_start_pose(rig,rng)
            e['Root']=[0,0,0]
            records.append(dict(sampling_regime='video2rig_calm' if calm else 'video2rig_active',joints=e))
        meta=rig.meta.copy(); meta['Root']={**meta['Root'],'fixed_local':True}
        report=audit_boundary_exploration_coverage(records,meta)
        self.assertTrue(report['passed'],report['failures'])
        self.assertFalse(report['boundary_coverage_required'])
        records[0]['joints']['Neck1'][0]=3
        self.assertFalse(audit_boundary_exploration_coverage(records,meta)['passed'])
        for r in records:r['joints']['Neck1']=[.25,.1,-.1]
        self.assertFalse(audit_boundary_exploration_coverage(records,meta)['passed'])
