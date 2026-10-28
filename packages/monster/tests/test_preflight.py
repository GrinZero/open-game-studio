"""Preflight failure, artifact provenance and real Blender regression coverage."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/monster-workflow/scripts'))
import sop_preflight as preflight
import sop_g1_pack as g1_pack
import sop_gate


def source_glb(path):
    body = json.dumps({'asset': {'version': '2.0'}, 'meshes': [], 'nodes': []}).encode()
    body += b' ' * (-len(body) % 4)
    path.write_bytes(struct.pack('<4sIIII', b'glTF', 2, 20 + len(body), len(body), 0x4E4F534A) + body)
    return path


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.source = source_glb(self.project / 'source.glb')
        self.dirs = preflight.resolve_scratch_dirs(self.project, 'test')
        self.report_path = self.dirs['checks_dir'] / 'preflight_report.json'
        self.drive = self.project / 'drive'

    def fake_worker(self, command, **kwargs):
        args = command
        stage = Path(args[args.index('--out_dir') + 1])
        for name in [
            'rig_input_variant.glb',
            'shell.npz',
            'watertight_shell.glb',
            'preflight_front_side.png',
            'preflight_watertight_shell.png',
            'decoration_manifest.json',
            'decorations_backup.blend',
        ]:
            (stage / name).write_bytes(b'test-only artifact, not Blender acceptance')
        source_glb(stage / 'rig_input_variant.glb')
        source_glb(stage / 'watertight_shell.glb')
        png_bytes = b'\x89PNG\r\n\x1a\n' + struct.pack('>I4sII', 13, b'IHDR', 1344, 768) + b'test-only PNG fixture'
        (stage / 'preflight_front_side.png').write_bytes(png_bytes)
        (stage / 'preflight_watertight_shell.png').write_bytes(png_bytes)
        (stage / 'bpy_preflight.json').write_text(json.dumps({
            'automatic_ready': True, 'checks': {'side_limb_occlusion_ratio': None, 'watertight_shell_ready': True, 'watertight_shell_rendered': True},
            'watertight_shell': {
                'path': str(stage / 'shell.npz'),
                'mesh_glb_path': str(stage / 'watertight_shell.glb'),
                'preview_image_path': str(stage / 'preflight_watertight_shell.png'),
                'watertight_ready': True,
                'solid_voxels': 128,
                'shell_mesh_vertices': 64,
                'shell_mesh_faces': 120,
            },
            'bone_naming_validation': {'valid': True, 'total_bones': 0},
        }))

    def run_fake_preflight(self):
        with patch.object(preflight, '_worker_command', side_effect=lambda args: args), patch.object(preflight.subprocess, 'run', side_effect=self.fake_worker):
            return preflight.run_preflight('test', self.source, project_root=self.project)

    def test_no_backend_invalidates_previous_success_and_blocks_u1(self):
        self.report_path.write_text('{"passed": true}')
        with patch.object(preflight, '_worker_command', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'bpy/Blender'):
                preflight.run_preflight('test', self.source, project_root=self.project)
        self.assertFalse(json.loads(self.report_path.read_text())['passed'])
        self.assertFalse((self.dirs['work_dir'] / 'rig_input_variant.glb').exists())
        with self.assertRaises(RuntimeError):
            preflight.sync_u1_to_drive('test', project_root=self.project, drive_root=self.drive)
        self.assertFalse(self.drive.exists())

    def test_worker_error_invalidates_previous_success(self):
        self.run_fake_preflight()
        with patch.object(preflight, '_worker_command', return_value=['worker']), patch.object(preflight.subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'worker')):
            with self.assertRaises(subprocess.CalledProcessError):
                preflight.run_preflight('test', self.source, project_root=self.project)
        self.assertEqual(json.loads(self.report_path.read_text())['status'], 'failed')
        with self.assertRaises(RuntimeError):
            preflight.sync_u1_to_drive('test', project_root=self.project, drive_root=self.drive)

    def test_missing_outputs_cannot_reuse_previous_artifacts(self):
        self.run_fake_preflight()
        with patch.object(preflight, '_worker_command', return_value=['worker']), patch.object(preflight.subprocess, 'run'):
            with self.assertRaises(FileNotFoundError):
                preflight.run_preflight('test', self.source, project_root=self.project)
        self.assertFalse(json.loads(self.report_path.read_text())['passed'])

    def test_visual_review_required_before_sync(self):
        result = self.run_fake_preflight()
        self.assertFalse(result['passed'])
        self.assertTrue(result['human_in_the_loop_required'])
        self.assertTrue(Path(result['watertight_shell_glb']).is_file())
        self.assertTrue(Path(result['watertight_shell_check_image']).is_file())
        self.assertTrue(Path(result['preflight_review_html']).is_file())
        self.assertIsNone(result['checks']['side_limb_occlusion_ratio'])
        with self.assertRaises(RuntimeError):
            preflight.sync_u1_to_drive('test', project_root=self.project, drive_root=self.drive)
        with self.assertRaises(ValueError):
            preflight.review_preflight('test', project_root=self.project, note=' ')
        # CLI --review-visual must block if --user-quote (Human-in-the-Loop user confirmation) is omitted
        with self.assertRaisesRegex(SystemExit, 'Human-in-the-Loop'):
            preflight.main([
                '--monster-id', 'test',
                '--project-root', str(self.project),
                '--review-visual',
                '--review-note', 'Agent tried to approve without user confirmation',
            ])
        rc = preflight.main([
            '--monster-id', 'test',
            '--project-root', str(self.project),
            '--drive-root', str(self.drive),
            '--review-visual',
            '--review-note', 'Test operator evidence only',
            '--user-quote', '预检和水密壳结果确认 OK，继续下一步',
            '--sync-u1',
        ])
        self.assertEqual(rc, 0)
        result = preflight.sync_u1_to_drive('test', project_root=self.project, drive_root=self.drive)
        self.assertTrue(Path(result['drive_u1_glb']).is_file())
        self.assertTrue(Path(result['drive_u1_shell_npz']).is_file())
        self.assertTrue(Path(result['drive_u1_shell_glb']).is_file())

    def test_changed_source_or_input_or_image_blocks_review_and_u1(self):
        for key in ['source', 'rig_input', 'shell', 'shell_glb', 'front_side', 'shell_preview', 'decorations', 'backup']:
            with self.subTest(key=key):
                self.source = source_glb(self.source)
                result = self.run_fake_preflight()
                preflight.review_preflight('test', project_root=self.project, note='Test evidence')
                artifact = Path(result['evidence'][key]['path'])
                artifact.write_bytes(artifact.read_bytes() + b'changed')
                with self.assertRaisesRegex(RuntimeError, '已改变或缺失'):
                    preflight.sync_u1_to_drive('test', project_root=self.project, drive_root=self.drive)
                with self.assertRaises(RuntimeError):
                    preflight.review_preflight('test', project_root=self.project, note='Stale evidence')

    def test_legacy_passed_report_is_rejected(self):
        (self.dirs['work_dir'] / 'rig_input_variant.glb').write_bytes(b'legacy')
        self.report_path.write_text('{"passed": true}')
        with self.assertRaisesRegex(RuntimeError, '真实 Blender'):
            preflight.sync_u1_to_drive('test', project_root=self.project, drive_root=self.drive)

    def test_geometric_failure_cannot_be_overridden_by_visual_review(self):
        self.run_fake_preflight()
        report = json.loads(self.report_path.read_text())
        report['automatic_ready'] = False
        self.report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(RuntimeError, '几何错误'):
            preflight.review_preflight('test', project_root=self.project, note='Looks fine')

    def test_preflight_cannot_overwrite_its_own_source(self):
        output = source_glb(self.dirs['work_dir'] / 'rig_input_variant.glb')
        original = output.read_bytes()
        with self.assertRaisesRegex(ValueError, '原地覆盖'):
            preflight.run_preflight('test', output, project_root=self.project)
        self.assertEqual(output.read_bytes(), original)

    @unittest.skipUnless(os.environ.get('MONSTER_TEST_BLENDER'), 'Set MONSTER_TEST_BLENDER for the real export/reimport/render regression')
    def test_real_blender_preserves_parented_parts_and_roundtrips(self):
        blender = os.environ['MONSTER_TEST_BLENDER']
        expression = f'''import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
parent=bpy.data.objects.new('Assembly',None)
bpy.context.collection.objects.link(parent)
parent.location=(5,2,3)
for x,name in [(0,'Body'),(3,'ornament')]:
    bpy.ops.mesh.primitive_cube_add(size=1, location=(x,0,0))
    obj=bpy.context.object
    obj.name=name
    obj.parent=parent
bpy.ops.export_scene.gltf(filepath={str(self.source)!r},export_format='GLB')
'''
        subprocess.run([blender, '--background', '--factory-startup', '--python-exit-code', '1', '--python-expr', expression], check=True, stdout=subprocess.DEVNULL)
        with patch.dict(os.environ, {'MONSTER_BLENDER': blender}), patch.object(preflight, '_find_bpy_python', return_value=None):
            report = preflight.run_preflight('test', self.source, project_root=self.project, yaw_deg=90)
        actual = report['bpy_preflight']
        self.assertTrue(report['automatic_ready'], json.dumps(report['bpy_preflight'], indent=2))
        self.assertFalse(report['passed'])
        self.assertEqual(actual['mesh_count'], 2)
        self.assertTrue(actual['checks']['glb_roundtrip_verified'])
        self.assertAlmostEqual(actual['bbox_max'][1] - actual['bbox_min'][1], 4.0, places=4)
        self.assertTrue(all(m['boundary_edges'] == 0 for m in actual['mesh_metrics']))
        png = Path(report['front_side_check_image']).read_bytes()
        self.assertEqual(struct.unpack('>II', png[16:24]), (1344, 768))
        shell_png = Path(report['watertight_shell_check_image']).read_bytes()
        self.assertEqual(struct.unpack('>II', shell_png[16:24]), (1344, 1536))
        manifest = json.loads(Path(report['decoration_manifest']).read_text())
        self.assertIn('ornament', manifest['review_candidates'])
        self.assertEqual(manifest['detached_decorations'], [])
        self.assertTrue(Path(report['watertight_shell_npz']).is_file())
        self.assertTrue(Path(report['watertight_shell_glb']).is_file())
        self.assertTrue(Path(report['preflight_review_html']).is_file())
        self.assertTrue(report['watertight_shell']['watertight_ready'])
        self.assertGreater(report['watertight_shell']['solid_voxels'], 0)
        self.assertGreater(report['watertight_shell']['shell_mesh_faces'], 0)

    @unittest.skipUnless(os.environ.get('MONSTER_TEST_BLENDER'), 'Set MONSTER_TEST_BLENDER for the real watertight shell & sanitization regression')
    def test_real_blender_auto_sanitizes_defects_and_watertight_shell_backs_open_non_manifold_sheets(self):
        blender = os.environ['MONSTER_TEST_BLENDER']
        expression = f'''import bpy, bmesh
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.5))
body = bpy.context.object
body.name = 'ThornHunter_Body'
bm = bmesh.new()
bm.from_mesh(body.data)
bm.verts.ensure_lookup_table()
bm.faces.ensure_lookup_table()
f0_coords = [v.co.copy() for v in bm.faces[0].verts]
v_top0 = bm.verts[0]
v_top1 = bm.verts[1]
# 1) Add an open single-sided leaf sheet attached to two top-edge vertices -> creates a 3-face non-manifold T-junction + open boundary edges
v_leaf0 = bm.verts.new((v_top0.co.x + 0.35, v_top0.co.y, v_top0.co.z + 0.25))
v_leaf1 = bm.verts.new((v_top1.co.x + 0.35, v_top1.co.y, v_top1.co.z + 0.25))
bm.faces.new((v_top0, v_top1, v_leaf1, v_leaf0))
# 2) Add an exact duplicate face over the first cube face
dup_verts = [bm.verts.new(co) for co in f0_coords]
bm.faces.new(dup_verts)
# 3) Add a zero-area degenerate collinear triangle and a collapsed 1e-8 triangle (which would become a wire edge/loose vert on weld)
dv0 = bm.verts.new((0.1, 0.1, 0.2))
dv1 = bm.verts.new((0.2, 0.1, 0.2))
dv2 = bm.verts.new((0.3, 0.1, 0.2))
bm.faces.new((dv0, dv1, dv2))
cv0 = bm.verts.new((0.4, 0.4, 0.4))
cv1 = bm.verts.new((0.4 + 1e-8, 0.4, 0.4))
cv2 = bm.verts.new((0.4, 0.4 + 1e-8, 0.4))
bm.faces.new((cv0, cv1, cv2))
bm.to_mesh(body.data)
body.data.update()
bm.free()
bpy.ops.export_scene.gltf(filepath={str(self.source)!r}, export_format='GLB')
'''
        subprocess.run([blender, '--background', '--factory-startup', '--python-exit-code', '1', '--python-expr', expression], check=True, stdout=subprocess.DEVNULL)
        with patch.dict(os.environ, {'MONSTER_BLENDER': blender}), patch.object(preflight, '_find_bpy_python', return_value=None):
            report = preflight.run_preflight('test', self.source, project_root=self.project)
        actual = report['bpy_preflight']
        self.assertTrue(report['automatic_ready'], json.dumps(actual, indent=2))
        self.assertTrue(actual['checks']['watertight_shell_ready'])
        self.assertTrue(actual['checks']['watertight_shell_fallback_active'])
        self.assertFalse(actual['checks']['strict_surface_manifold'])
        m0 = actual['mesh_metrics'][0]
        self.assertEqual(m0['loose_vertices'], 0)
        self.assertEqual(m0['wire_edges'], 0)
        self.assertEqual(m0['degenerate_faces'], 0)
        self.assertEqual(m0['duplicate_faces'], 0)
        self.assertGreater(m0['boundary_edges'], 0)
        self.assertGreater(m0['non_manifold_junctions'], 0)
        self.assertGreater(actual['watertight_shell']['solid_voxels'], 0)
        preflight.review_preflight('test', project_root=self.project, note='Open foliage sheet backed by watertight shell')
        u1 = preflight.sync_u1_to_drive('test', project_root=self.project, drive_root=self.drive)
        self.assertTrue(Path(u1['drive_u1_glb']).is_file())
        self.assertTrue(Path(u1['drive_u1_shell_npz']).is_file())
    @unittest.skipUnless(os.environ.get('MONSTER_TEST_BLENDER'), 'Set MONSTER_TEST_BLENDER for real rig and delivery verification')
    def test_real_blender_renames_actual_bones_and_delivery_uses_clear_scene(self):
        blender = os.environ['MONSTER_TEST_BLENDER']
        expression = f"""import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.object.armature_add()
arm=bpy.context.object
arm.data.bones[0].name='Bone.001'
for x in [0,2]:
    bpy.ops.mesh.primitive_cube_add(size=1,location=(x,0,0))
    obj=bpy.context.object
    group=obj.vertex_groups.new(name='Bone.001')
    group.add(list(range(len(obj.data.vertices))),1.0,'REPLACE')
    obj.modifiers.new('Rig','ARMATURE').object=arm
    obj.parent=arm
bpy.ops.export_scene.gltf(filepath={str(self.source)!r},export_format='GLB')
"""
        subprocess.run([blender, '--background', '--factory-startup', '--python-exit-code', '1', '--python-expr', expression], check=True, stdout=subprocess.DEVNULL)
        with patch.dict(os.environ, {'MONSTER_BLENDER': blender}), patch.object(preflight, '_find_bpy_python', return_value=None):
            report = preflight.run_preflight('test', self.source, project_root=self.project)
        self.assertTrue(report['automatic_ready'], json.dumps(report['bpy_preflight'], indent=2))
        self.assertTrue(report['bpy_preflight']['bone_naming_validation']['valid'])
        self.assertNotIn('Bone.001', report['bpy_preflight']['bone_names'])
        delivery_report = self.project / 'delivery.json'
        args = ['--mode', 'verify_delivery_glb', '--glb_path', report['rig_input_variant_glb'], '--out_report', str(delivery_report)]
        with patch.dict(os.environ, {'MONSTER_BLENDER': blender}), patch.object(preflight, '_find_bpy_python', return_value=None):
            command = preflight._worker_command(args)
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        self.assertTrue(json.loads(delivery_report.read_text())['valid'])

    def test_g1_pack_rejects_missing_backend_and_placeholder_stubs(self):
        cand_dir = self.dirs['work_dir'] / 'candidates'
        cand_dir.mkdir(parents=True, exist_ok=True)
        for cid in range(4):
            source_glb(cand_dir / f'candidate_{cid}.glb')

        with patch.object(g1_pack, '_worker_command', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'bpy/Blender'):
                g1_pack.build_g1_review_pack('test', archetype='humanoid', project_root=self.project, drive_root=self.drive)

        g1_report = json.loads((self.dirs['checks_dir'] / 'g1_review' / 'g1_report.json').read_text())
        self.assertFalse(g1_report['passed'])
        self.assertEqual(g1_report['status'], 'failed')
        ledger = sop_gate.load_ledger('test', project_root=self.project, drive_root=self.drive)
        self.assertNotEqual(ledger['gates']['G1']['status'], 'awaiting_rig_review')

        # Explicitly verify 24-byte ftyp stub, fake blend stub, and Zstd compressed blend are rejected
        stub_mp4 = self.project / 'stub.mp4'
        stub_mp4.write_bytes(b'\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom')
        with self.assertRaisesRegex(RuntimeError, '24 字节|过小'):
            g1_pack._validate_mp4_file(stub_mp4)

        fake_blend = self.project / 'fake.blend'
        fake_blend.write_bytes(b'BLENDER-v402-G1-CANDIDATE-0' + b'\x00' * 2048)
        with self.assertRaisesRegex(RuntimeError, '伪造的占位'):
            g1_pack._validate_blend_file(fake_blend)

        zstd_blend = self.project / 'compressed.blend'
        zstd_blend.write_bytes(b'\x28\xb5\x2f\xfd' + b'\x00' * 2048)
        with self.assertRaisesRegex(RuntimeError, '压缩'):
            g1_pack._validate_blend_file(zstd_blend)

    @unittest.skipUnless(os.environ.get('MONSTER_TEST_BLENDER'), 'Set MONSTER_TEST_BLENDER for real G1 Blender + FFmpeg regression')
    def test_real_blender_g1_pack_renders_octahedral_views_and_real_mp4(self):
        blender = os.environ['MONSTER_TEST_BLENDER']
        cand_dir = self.dirs['work_dir'] / 'candidates'
        cand_dir.mkdir(parents=True, exist_ok=True)
        for cid in range(4):
            c_glb = cand_dir / f'candidate_{cid}.glb'
            expr = f"""import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.object.armature_add(location=(0,0,0))
arm=bpy.context.object
bpy.ops.object.mode_set(mode='EDIT')
eb0=arm.data.edit_bones[0]
eb0.name='bone_0'
eb0.head=(0,0,0)
eb0.tail=(0,0,0.6)
for name,h,t in [
    ('bone_1',(0,0,0.6),(0,0,1.2)),
    ('bone_2',(0,0,1.2),(0,-0.1,1.6)),
    ('bone_3',(0.15,0,1.1),(0.45,0,0.8)),
    ('bone_4',(-0.15,0,1.1),(-0.45,0,0.8)),
    ('bone_5',(0.15,0,0.5),(0.25,0,0.0)),
    ('bone_6',(-0.15,0,0.5),(-0.25,0,0.0)),
]:
    eb=arm.data.edit_bones.new(name)
    eb.head=h
    eb.tail=t
    eb.parent=eb0
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.mesh.primitive_cube_add(size=1.2, location=(0,0,0.8))
mesh=bpy.context.object
for i,b in enumerate(arm.data.bones):
    vg=mesh.vertex_groups.new(name=b.name)
    vg.add([i % len(mesh.data.vertices)], 1.0, 'REPLACE')
vg0=mesh.vertex_groups['bone_0']
vg0.add(list(range(len(mesh.data.vertices))), 0.5, 'ADD')
mod=mesh.modifiers.new('Armature','ARMATURE')
mod.object=arm
mesh.parent=arm
bpy.ops.export_scene.gltf(filepath={str(c_glb)!r}, export_format='GLB')
"""
            subprocess.run([blender, '--background', '--factory-startup', '--python-exit-code', '1', '--python-expr', expr], check=True, stdout=subprocess.DEVNULL)

        with patch.dict(os.environ, {'MONSTER_BLENDER': blender}), patch.object(preflight, '_find_bpy_python', return_value=None):
            res = g1_pack.build_g1_review_pack('test', archetype='humanoid', project_root=self.project, drive_root=self.drive)

        self.assertEqual(res['status'], 'awaiting_rig_review')
        self.assertEqual(res['octahedral_views_rendered'], 16)
        g1_rep = json.loads(Path(res['g1_report_json']).read_text())
        g1_pack.validate_g1_evidence(g1_rep)
        for v_path in res['sync_videos'].values():
            box_info = g1_pack._validate_mp4_file(Path(v_path))
            self.assertGreater(box_info['bytes'], 1000)
            self.assertGreater(box_info['mdat_payload_bytes'], 0)
        for c_info in res['candidates']:
            g1_pack._validate_blend_file(Path(c_info['blend_file']))
            self.assertTrue(c_info['bone_naming_validation']['valid'])


if __name__ == '__main__':
    unittest.main()
