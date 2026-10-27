"""Actual EEVEE batch/individual parity, pose/blur isolation and failure cleanup."""
import json
from unittest.mock import patch
import math
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / '.venv/lib' / f'python{sys.version_info.major}.{sys.version_info.minor}/site-packages'))
import bpy
import numpy as np
from PIL import Image
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'worker'))
from strip_renderer import StripRenderer
from studio_bpy_worker import configure_blender_gpu_rendering

out = Path(sys.argv[sys.argv.index('--') + 1]); out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
config = configure_blender_gpu_rendering(scene)
assert config['engine'].startswith('BLENDER_EEVEE') and config['samples'] == 8, config
scene.render.resolution_x = scene.render.resolution_y = 96
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
scene.world = bpy.data.worlds.new('World'); scene.world.color = (.1,.1,.1)
armdata = bpy.data.armatures.new('Rig'); arm = bpy.data.objects.new('Rig',armdata)
scene.collection.objects.link(arm); bpy.context.view_layer.objects.active = arm; arm.select_set(True)
bpy.ops.object.mode_set(mode='EDIT'); bone = armdata.edit_bones.new('Root')
bone.head = (0,0,0); bone.tail = (0,0,1); bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.mesh.primitive_cube_add(size=1, location=(0,0,.5))
mesh = bpy.context.object; mesh.scale = (.28,.18,.8)
bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
mesh.vertex_groups.new(name='Root').add(list(range(8)),1,'REPLACE')
modifier=mesh.modifiers.new('skin','ARMATURE'); modifier.object=arm
mat=bpy.data.materials.new('Material'); mat.diffuse_color=(.9,.2,.05,1); mesh.data.materials.append(mat)
for name,loc in [('key',(2,-3,3)),('fill',(-2,1,2))]:
    data=bpy.data.lights.new(name,'AREA'); data.energy=180; data.size=2
    obj=bpy.data.objects.new(name,data); scene.collection.objects.link(obj); obj.location=loc
    obj.rotation_euler=(Vector((0,0,.5))-obj.location).to_track_quat('-Z','Y').to_euler()
cameras={}
for name,loc in [('front',(0,-8,.5)),('side',(8,0,.5))]:
    data=bpy.data.cameras.new(name); data.type='ORTHO'; data.sensor_fit='VERTICAL'; data.ortho_scale=2
    obj=bpy.data.objects.new(name,data); scene.collection.objects.link(obj); obj.location=loc
    obj.rotation_euler=(Vector((0,0,.5))-obj.location).to_track_quat('-Z','Y').to_euler(); cameras[name]=obj
scene.camera=cameras['front']; bpy.context.view_layer.update()
matrices={c:np.array(c.matrix_world) for c in cameras.values()}
base_counts=(len(bpy.data.objects),len(bpy.data.actions),len(bpy.data.armatures),len(bpy.data.lights))

def pose(k,blur):
    arm.animation_data_clear()
    pb=arm.pose.bones['Root']; pb.rotation_mode='XYZ'
    for frame in (0,1,2):
        pb.rotation_euler=(0,(k-8)*.035+(frame-1)*(.8 if blur else 0),0)
        pb.keyframe_insert('rotation_euler',frame=frame)
    scene.frame_set(1); scene.render.use_motion_blur=blur; scene.render.motion_blur_shutter=.5
    bpy.context.view_layer.update()

# Warmup shader compilation before comparing timings.
pose(0,False); scene.render.filepath=str(out/'warmup.png'); bpy.ops.render.render(write_still=True)
t0=time.perf_counter()
reference=[]
for k in range(17):
    pose(k,k>=8)
    parts=[]
    for name,camera in cameras.items():
        scene.camera=camera; scene.render.filepath=str(out/f'single_{k}_{name}.png'); bpy.ops.render.render(write_still=True)
        with Image.open(scene.render.filepath) as im: parts.append(np.array(im.convert('RGB')))
    reference.append(np.concatenate(parts,axis=1))
single_seconds=time.perf_counter()-t0
renderer=StripRenderer(cameras,out,count=16)
t0=time.perf_counter()
for k in range(17):
    pose(k,k>=8); renderer.enqueue(out/f'batch_{k:02}.jpg')
renderer.flush(); batch_seconds=time.perf_counter()-t0
errors=[]
for k,expected in enumerate(reference):
    actual=np.array(Image.open(out/f'batch_{k:02}.jpg').convert('RGB'))
    errors.append(float(np.abs(actual.astype(float)-expected).mean()))
    assert actual.shape==expected.shape
assert max(errors)<3,errors
assert renderer.stats['batch_sizes']==[8,9],renderer.stats
for camera,matrix in matrices.items(): np.testing.assert_allclose(np.array(camera.matrix_world),matrix,atol=1e-6)
assert not any(o.name.startswith('__strip_') for o in scene.objects)
assert not any(c.name.startswith('__strip_') for c in bpy.data.collections)
# Source action remains live; independent copied armatures/actions/lights are released.
assert len(bpy.data.armatures)==base_counts[2] and len(bpy.data.lights)==base_counts[3]
# A blurred source must actually differ from the same sharp pose.
pose(8,False); scene.camera=cameras['front']; scene.render.filepath=str(out/'sharp_08.png')
bpy.ops.render.render(write_still=True)
sharp=np.array(Image.open(out/'sharp_08.png').convert('RGB')).astype(float)
blur_delta=float(np.abs(sharp-reference[8][:,:96]).mean())
assert blur_delta > .05, blur_delta
import os
with patch.dict(os.environ,{'MONSTER_STRIP_MAX_WIDTH':'500'}):
    assert StripRenderer(cameras,out,count=16).count == 3
# Exact 16-pose batch, partial tail, stacked output, and cap splitting.
scene.render.use_motion_blur=False
full=StripRenderer(cameras,out,count=16,layout='stacked')
for k in range(17):
    pose(k,False); full.enqueue(out/f'stacked_{k:02}.jpg')
full.flush()
assert full.stats['batch_sizes']==[16,1]
assert Image.open(out/'stacked_00.jpg').size==(96,192)
# Force a render exception after creating the copies, then check full restoration.
from unittest.mock import patch
saved=(scene.camera,scene.render.filepath,scene.render.image_settings.file_format)
(out/'must_not_exist.jpg').unlink(missing_ok=True)
failed=StripRenderer(cameras,out,count=16); failed.enqueue(out/'must_not_exist.jpg')
from types import SimpleNamespace
from unittest.mock import Mock
with patch.object(bpy, 'ops', SimpleNamespace(render=SimpleNamespace(render=Mock(side_effect=RuntimeError('injected render failure'))))):
    try: failed.flush()
    except RuntimeError: pass
    else: raise AssertionError('Expected injected failure')
assert saved==(scene.camera,scene.render.filepath,scene.render.image_settings.file_format)
assert not (out/'must_not_exist.jpg').exists()
assert not any(o.name.startswith('__strip_') for o in scene.objects)
report={'engine':config,'single_seconds':single_seconds,'batch_seconds':batch_seconds,
        'speedup':single_seconds/batch_seconds,'pixel_mae':errors,'blur_pixel_delta':blur_delta,'mixed_blur':renderer.stats,
        'full_batch':full.stats,'failure_cleanup':True,'blender':bpy.app.version_string}
(out/'report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report))
