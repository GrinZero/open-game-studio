"""Batched orthographic rendering with independent animated rigs and calibrated tiles."""
import math
import os
import time
from pathlib import Path

STRIP_CONTRACT = "monster-animated-strip-v1"


def strip_settings():
    return {
        "contract": STRIP_CONTRACT,
        "enabled": os.environ.get("MONSTER_RENDER_STRIP", "1") == "1",
        "copies": max(1, min(16, int(os.environ.get("MONSTER_STRIP_COPIES", "16")))),
        "max_width": max(1, int(os.environ.get("MONSTER_STRIP_MAX_WIDTH", "16384"))),
        "gutter_ratio": 0.5,
        "depth_stagger": 2.5,
    }


def _copy_action(owner):
    if owner.animation_data and owner.animation_data.action:
        owner.animation_data.action = owner.animation_data.action.copy()


def _remap_pointers(owner, mapping):
    for prop in owner.bl_rna.properties:
        if prop.type == 'POINTER' and not prop.is_readonly:
            target = getattr(owner, prop.identifier)
            if getattr(target, 'bl_rna', None) and target in mapping:
                setattr(owner, prop.identifier, mapping[target])


class StripRenderer:
    """Queue independent pose/action snapshots; render each camera once per batch.

    Camera intrinsics and per-pose label coordinates stay in the original panel.
    Half-panel gutters and alternating depth keep neighboring geometry apart.
    Perspective views must use the individual rendering path.
    """
    def __init__(self, cameras, output_dir, count=None, layout="horizontal"):
        import bpy
        if not cameras or any(c.data.type != "ORTHO" for c in cameras.values()):
            raise ValueError("Strip rendering requires orthographic cameras")
        if bpy.context.scene.render.engine.startswith('BLENDER_EEVEE') and bpy.app.version < (4, 3, 0):
            raise RuntimeError("EEVEE strips require Blender >=4.3 light/shadow linking; rerun Cell 2 for pinned bpy 5.2.3, or set MONSTER_RENDER_STRIP=0")
        self.cameras = cameras
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.settings = strip_settings()
        self.count = self.settings['copies'] if count is None else max(1, min(16, int(count)))
        self.layout = layout
        W = bpy.context.scene.render.resolution_x
        self.slot = math.ceil(W * (1 + self.settings['gutter_ratio']))
        self.count = min(self.count, max(1, self.settings['max_width'] // self.slot))
        if W > self.settings['max_width']:
            raise ValueError("Panel exceeds MONSTER_STRIP_MAX_WIDTH")
        self.pending = []
        self.stats = {**self.settings, "effective_copies": self.count, "batches": 0,
                      "poses": 0, "render_calls": 0, "render_seconds": 0.0,
                      "batch_sizes": [], "layout": layout}
        self.blur = None

    def enqueue(self, path):
        import bpy
        scene = bpy.context.scene
        blur = (bool(scene.render.use_motion_blur), float(scene.render.motion_blur_shutter))
        if self.pending and blur != self.blur:
            self.flush()
        self.blur = blur
        originals = [obj for obj in scene.objects if not obj.name.startswith('__strip_')
                     and obj.type in ('ARMATURE', 'MESH', 'LIGHT', 'EMPTY')
                     and not obj.hide_render]
        for obj in list(originals):
            parent = obj.parent
            while parent is not None:
                if parent not in originals:
                    originals.append(parent)
                parent = parent.parent
        mapping = {}
        collection = bpy.data.collections.new('__strip_slot')
        scene.collection.children.link(collection)
        try:
            for source in originals:
                copy = source.copy()
                copy.name = '__strip_' + source.name
                # Meshes can share immutable geometry; armature poses and light state cannot.
                if source.type in ('ARMATURE', 'LIGHT') or (source.type == 'MESH' and (source.data.shape_keys or source.data.animation_data)):
                    copy.data = source.data.copy()
                    _copy_action(copy.data)
                    if source.type == 'MESH' and copy.data.shape_keys:
                        _copy_action(copy.data.shape_keys)
                _copy_action(copy)
                collection.objects.link(copy)
                copy.hide_render = True
                mapping[source] = copy
            for source, copy in mapping.items():
                copy.parent = mapping.get(source.parent, source.parent)
                copy.matrix_world = source.matrix_world.copy()
                for owner in [*copy.modifiers, *copy.constraints]:
                    _remap_pointers(owner, mapping)
                if copy.type == 'ARMATURE':
                    for pb in copy.pose.bones:
                        for constraint in pb.constraints:
                            _remap_pointers(constraint, mapping)
                for owner in (copy, copy.data, getattr(copy.data, 'shape_keys', None)):
                    animation = getattr(owner, 'animation_data', None)
                    if animation:
                        for driver in animation.drivers:
                            for variable in driver.driver.variables:
                                for target in variable.targets:
                                    if target.id in mapping:
                                        target.id = mapping[target.id]
                # Light/shadow linking prevents adjacent slots illuminating or shadowing one another.
                if copy.type == 'LIGHT' and hasattr(copy, 'light_linking'):
                    copy.light_linking.receiver_collection = collection
                    copy.light_linking.blocker_collection = collection
            anchor = bpy.data.objects.new('__strip_anchor', None)
            collection.objects.link(anchor)
            for source, copy in mapping.items():
                if source.parent not in mapping:
                    world = copy.matrix_world.copy()
                    copy.parent = anchor
                    copy.matrix_world = world
            self.pending.append((Path(path), collection, anchor))
        except Exception:
            self._remove_collection(collection)
            raise
        if len(self.pending) >= self.count:
            self.flush()

    @staticmethod
    def _remove_collection(collection):
        import bpy
        actions = set()
        for obj in list(collection.objects):
            data = obj.data
            for owner in (obj, data, getattr(data, 'shape_keys', None)):
                animation = getattr(owner, 'animation_data', None)
                if animation and animation.action:
                    actions.add(animation.action)
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and data.users == 0:
                bpy.data.batch_remove([data])
        bpy.data.collections.remove(collection)
        for action in actions:
            if action.users == 0:
                bpy.data.actions.remove(action)

    def flush(self):
        import bpy
        from mathutils import Matrix
        from PIL import Image
        if not self.pending:
            return
        scene = bpy.context.scene
        W, H = scene.render.resolution_x, scene.render.resolution_y
        settings = {key: getattr(scene.render, key) for key in (
            'resolution_x', 'resolution_y', 'resolution_percentage', 'filepath',
            'use_motion_blur', 'motion_blur_shutter', 'use_border', 'use_crop_to_border')}
        fmt = scene.render.image_settings.file_format
        color = scene.render.image_settings.color_mode
        active = scene.camera
        cameras = {cam: cam.matrix_world.copy() for cam in self.cameras.values()}
        clip_ends = {cam: cam.data.clip_end for cam in self.cameras.values()}
        hidden = [(obj, obj.hide_render) for obj in scene.objects
                  if not obj.name.startswith('__strip_') and obj.type in ('MESH', 'LIGHT')]
        count = len(self.pending)
        slot = self.slot if count > 1 else W
        gutter = (slot - W) // 2
        tiles = [[] for _ in self.pending]
        temporary = self.output / '_strip_render.png'
        start = time.perf_counter()
        try:
            for obj, _ in hidden:
                obj.hide_render = True
            scene.render.use_motion_blur, scene.render.motion_blur_shutter = self.blur
            scene.render.resolution_percentage = 100
            scene.render.use_border = scene.render.use_crop_to_border = False
            for view, camera in self.cameras.items():
                matrix = cameras[camera]
                camera.data.clip_end = clip_ends[camera] + self.settings['depth_stagger']
                right = matrix.to_3x3().col[0]
                forward = -matrix.to_3x3().col[2]
                pixels_per_meter = H / float(camera.data.ortho_scale)
                for k, (_, collection, anchor) in enumerate(self.pending):
                    # Tile centers include integer gutter rounding, so saved pixels match labels exactly.
                    offset = (k * slot + gutter + W / 2 - count * slot / 2) / pixels_per_meter
                    move = Matrix.Translation(right * offset + forward * ((k % 2) * 2.5))
                    for obj in collection.objects:
                        obj.hide_render = False
                    anchor.matrix_world = move
                scene.camera = camera
                scene.render.resolution_x = count * slot
                scene.render.image_settings.file_format = 'PNG'
                scene.render.image_settings.color_mode = 'RGB'
                scene.render.filepath = str(temporary)
                bpy.context.view_layer.update()
                bpy.ops.render.render(write_still=True)
                self.stats['render_calls'] += 1
                with Image.open(temporary) as raw:
                    for k in range(count):
                        x = k * slot + gutter
                        tiles[k].append(raw.crop((x, 0, x + W, H)).convert('RGB'))
            for (path, _, _), parts in zip(self.pending, tiles):
                stacked = self.layout == 'stacked'
                image = Image.new('RGB', (W, H * len(parts)) if stacked else (W * len(parts), H))
                for i, part in enumerate(parts):
                    image.paste(part, (0, i * H) if stacked else (i * W, 0))
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path, quality=94)
            self.stats['batches'] += 1
            self.stats['poses'] += count
            self.stats['batch_sizes'].append(count)
        finally:
            self.stats['render_seconds'] += time.perf_counter() - start
            for cam, matrix in cameras.items():
                cam.matrix_world = matrix
                cam.data.clip_end = clip_ends[cam]
            scene.camera = active
            for key, value in settings.items():
                setattr(scene.render, key, value)
            scene.render.image_settings.file_format = fmt
            scene.render.image_settings.color_mode = color
            for obj, hide in hidden:
                obj.hide_render = hide
            for _, collection, _ in self.pending:
                self._remove_collection(collection)
            self.pending = []
            temporary.unlink(missing_ok=True)
            bpy.context.view_layer.update()
