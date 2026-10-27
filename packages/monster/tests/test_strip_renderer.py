"""Render policy, cache invalidation, crop transforms and optional real EEVEE regression."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'worker'))
from reader_v3 import creature_crop_box, map_points, crop_rgb
from strip_renderer import strip_settings, StripRenderer
from pose_features import dataset_signature


class StripPolicyTests(unittest.TestCase):
    def test_full_mesh_crop_retains_outliers_and_maps_roundtrip(self):
        # Long tail and held weapon must survive even if they are occluded/unlabeled.
        points = np.r_[np.full((100,2),448.), [[10.,120.],[880.,850.]]]
        box = creature_crop_box(points)
        normalized = map_points(points, box, 448) / 448
        self.assertTrue(((normalized >= 0) & (normalized <= 1)).all())
        np.testing.assert_allclose(map_points(normalized*448,box,448,inverse=True),points)
        self.assertAlmostEqual(box[2]-box[0],box[3]-box[1])
        self.assertEqual(crop_rgb(np.zeros((896,896,3),np.uint8),box).shape,(448,448,3))
        with self.assertRaises(ValueError): creature_crop_box(np.empty((0,2)))

    def test_render_changes_invalidate_existing_dataset(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ,{},clear=True):
            blend=Path(temporary)/'rig.blend'; blend.write_bytes(b'fixture')
            meta=Path(temporary)/'meta.json'; meta.write_text('{"bones": {}}')
            original=dataset_signature(blend,meta,5000)
            for key,value in [('MONSTER_RENDER_ENGINE','CYCLES'),('MONSTER_RENDER_STRIP','0'),
                              ('MONSTER_EEVEE_SAMPLES','16'),('MONSTER_STRIP_COPIES','8')]:
                with patch.dict(os.environ,{key:value}):
                    self.assertNotEqual(original,dataset_signature(blend,meta,5000),key)
            meta.write_text('{"bones": {}, "calibration": {"temporary": true}}')
            self.assertEqual(original,dataset_signature(blend,meta,5000))

    def test_eevee_42_is_blocked_before_render(self):
        from types import SimpleNamespace
        fake = SimpleNamespace(app=SimpleNamespace(version=(4,2,0)),
            context=SimpleNamespace(scene=SimpleNamespace(render=SimpleNamespace(engine='BLENDER_EEVEE_NEXT'))))
        camera = SimpleNamespace(data=SimpleNamespace(type='ORTHO'))
        with patch.dict(sys.modules,{'bpy':fake}), self.assertRaisesRegex(RuntimeError,'Cell 2'):
            StripRenderer({'front':camera},'.')

    def test_batch_policy_defaults_and_bounds(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(strip_settings()['copies'],16)
            self.assertTrue(strip_settings()['enabled'])
        with patch.dict(os.environ,{'MONSTER_STRIP_COPIES':'100'}):
            self.assertEqual(strip_settings()['copies'],16)

    @unittest.skipUnless(os.environ.get('MONSTER_TEST_BLENDER'),'Set MONSTER_TEST_BLENDER for EEVEE rendering')
    def test_actual_eevee_rendering(self):
        with tempfile.TemporaryDirectory() as temporary:
            result=subprocess.run([os.environ['MONSTER_TEST_BLENDER'],'--background','--factory-startup',
                '--python-exit-code','1','--python',str(Path(__file__).with_name('fixture_strip_render.py')),
                '--',temporary],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            self.assertEqual(result.returncode,0,result.stdout[-8000:])
            report=json.loads((Path(temporary)/'report.json').read_text())
            self.assertTrue(report['failure_cleanup'])
            self.assertEqual(report['full_batch']['batch_sizes'],[16,1])


if __name__=='__main__': unittest.main()
