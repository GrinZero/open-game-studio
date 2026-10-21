#!/usr/bin/env python3
"""tier3_full_blender.py — Tier 3 End-to-End Headless Blender GLB Roundtrip & 2K Video Comparison Runner."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.datasets.loader import load_benchmark_fixture
from evals.runners.tier1_fast_solver import RUNS_ROOT, execute_tier1_fast

MONSTER_ROOT = Path(__file__).resolve().parents[2]
WORKER_DIR = MONSTER_ROOT / "worker"


def resolve_blender_binary() -> str:
    """Finds the Blender executable via `BLENDER_BIN`, PATH, or macOS `/Applications/Blender.app`."""
    env_bin = os.environ.get("BLENDER_BIN")
    if env_bin and Path(env_bin).is_file():
        return env_bin
    which_bin = shutil.which("blender") or shutil.which("Blender")
    if which_bin:
        return which_bin
    mac_app = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    if mac_app.is_file():
        return str(mac_app)
    raise FileNotFoundError(
        "Blender executable not found. Set BLENDER_BIN=/path/to/Blender or install /Applications/Blender.app."
    )


_BLENDER_BAKE_AND_ROUNDTRIP_SCRIPT = r"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.kdtree import KDTree

cfg = json.loads(Path(os.environ["MONSTER_EVAL_BLENDER_CFG"]).read_text(encoding="utf-8"))
sys.path.insert(0, cfg["worker_dir"])

blend_path = Path(cfg["blend_path"])
meta = json.loads(Path(cfg["meta_path"]).read_text(encoding="utf-8"))
traj = json.loads(Path(cfg["trajectory_path"]).read_text(encoding="utf-8"))
glb_out = Path(cfg["glb_out_path"])
bake_report_out = Path(cfg["bake_report_path"])
render_frames_dir = Path(cfg["render_frames_dir"]) if cfg.get("render_frames_dir") else None

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

orig_sha = sha256_file(blend_path)
names = sorted(meta["bones"].keys())
frames = traj["frames"]
T = len(frames)

bpy.ops.wm.open_mainfile(filepath=str(blend_path))
scene = bpy.context.scene
rig = next(o for o in scene.objects if o.type == "ARMATURE")
meshes = [o for o in scene.objects if o.type == "MESH" and any(m.type == "ARMATURE" for m in o.modifiers)]

rig.animation_data_clear()
for act in list(bpy.data.actions):
    bpy.data.actions.remove(act)
for obj in list(scene.objects):
    if obj != rig and obj not in meshes:
        bpy.data.objects.remove(obj, do_unlink=True)

rig.animation_data_create()
action = bpy.data.actions.new("MonsterEval_Reconstructed")
rig.animation_data.action = action
scene.render.fps = 24
scene.frame_start = 0
scene.frame_end = T - 1

topo = sorted(rig.pose.bones, key=lambda pb: int(meta["bones"][pb.name].get("depth", 0)))
prev_quat = {}
audit_phases = [p for p in (0, 60, 131, 211, T - 1) if 0 <= p < T]
audit_meshes = {}

def sample_mesh_points(ms):
    pts = []
    deps = bpy.context.evaluated_depsgraph_get()
    for obj in ms:
        ev = obj.evaluated_get(deps)
        m = ev.to_mesh()
        pts.extend(ev.matrix_world @ v.co for v in m.vertices)
        ev.to_mesh_clear()
    return pts

for fi, item in enumerate(frames):
    scene.frame_set(fi)
    for pb in topo:
        pb.rotation_mode = "QUATERNION"
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    bmap = item["bones"]
    for pb in topo:
        b_info = meta["bones"].get(pb.name, {})
        if pb.name == "Core_Rigid" or b_info.get("fixed_local", False):
            pb.matrix_basis = Matrix.Identity(4)
        else:
            pred = bmap[pb.name]
            world = Matrix(pred["rot_mat_world"]).to_4x4()
            world.translation = Vector(pred["head_3d"]) if not pb.parent else (rig.matrix_world @ pb.head)
            pb.matrix = rig.matrix_world.inverted() @ world
            if pb.parent:
                pb.location = (0.0, 0.0, 0.0)
            pb.scale = (1.0, 1.0, 1.0)
        q = pb.rotation_quaternion.copy()
        if pb.name in prev_quat and q.dot(prev_quat[pb.name]) < 0.0:
            q.negate()
        pb.rotation_quaternion = q
        prev_quat[pb.name] = q.copy()
        pb.keyframe_insert(data_path="rotation_quaternion", frame=fi, group=pb.name)
        if not pb.parent:
            pb.keyframe_insert(data_path="location", frame=fi, group=pb.name)
        bpy.context.view_layer.update()
    if fi in audit_phases:
        audit_meshes[fi] = sample_mesh_points(meshes)

for obj in bpy.context.selected_objects:
    obj.select_set(False)
rig.select_set(True)
for obj in meshes:
    obj.select_set(True)
bpy.context.view_layer.objects.active = rig
glb_out.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(
    filepath=str(glb_out),
    export_format="GLB",
    use_selection=True,
    export_animations=True,
    export_animation_mode="ACTIVE_ACTIONS",
    export_frame_range=True,
    export_force_sampling=True,
    export_skins=True,
    export_morph=True,
    export_def_bones=False,
)

# Clean factory re-import & 5-phase skinning roundtrip check
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = 24
bpy.ops.import_scene.gltf(filepath=str(glb_out))
rig = next(o for o in scene.objects if o.type == "ARMATURE")
meshes = [o for o in scene.objects if o.type == "MESH"]
act = bpy.data.actions[0]
for trk in rig.animation_data.nla_tracks:
    trk.mute = True
rig.animation_data.action = act
if hasattr(act, "slots") and len(act.slots):
    rig.animation_data.action_slot = act.slots[0]
start_frame = float(act.frame_range[0])

skin_errors = []
for fi, ref_pts in audit_meshes.items():
    scene.frame_set(int(start_frame) + fi)
    bpy.context.view_layer.update()
    curr_pts = sample_mesh_points(meshes)
    kd = KDTree(len(curr_pts))
    for idx, pt in enumerate(curr_pts):
        kd.insert(pt, idx)
    kd.balance()
    errs = [kd.find(pt)[2] for pt in ref_pts]
    skin_errors.append({
        "frame": fi,
        "max": float(max(errs)) if errs else 0.0,
        "mean": float(sum(errs) / max(1, len(errs))),
    })

max_skin_err = max((x["max"] for x in skin_errors), default=0.0)
source_unchanged = (sha256_file(blend_path) == orig_sha)
bake_report = {
    "valid": bool(max_skin_err < 1e-4 and source_unchanged),
    "frame_count": T,
    "fps": 24,
    "bones": len(names),
    "source_unchanged": source_unchanged,
    "glb_sha256": sha256_file(glb_out),
    "gltf_animation_frame_origin": start_frame,
    "skin_roundtrip_max": max_skin_err,
    "skin_roundtrip_checks": skin_errors,
}
bake_report_out.write_text(json.dumps(bake_report, indent=2) + "\n", encoding="utf-8")
"""


def execute_tier3_full_blender(
    benchmark_id: str = "fantasy_creature_2k_v2",
    fixture_dir: Path | None = None,
    save_baseline: bool = False,
) -> dict[str, Any]:
    """Executes Tier 3 (`full`): runs Tier 1 solver, then launches headless Blender to bake
    `<benchmark_id>_reconstructed.glb`, re-imports it in a clean scene, verifies the 5-phase
    skinning roundtrip (`< 1e-4`), and updates Stage 5 in the final `eval_report.json`.
    """
    fixture = load_benchmark_fixture(benchmark_id=benchmark_id, fixture_dir=fixture_dir)
    blender_bin = resolve_blender_binary()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / f"{ts}_{benchmark_id}_full_taxonomy_kind_v2"
    run_dir.mkdir(parents=True, exist_ok=True)

    # First run Tier 1 to generate predicted_trajectory.json
    tier1_out = execute_tier1_fast(
        benchmark_id=benchmark_id,
        fixture_dir=fixture_dir,
        tier_label="full",
        run_dir=run_dir,
        save_baseline=False,
    )

    glb_out_path = run_dir / f"{benchmark_id}_reconstructed.glb"
    bake_report_path = run_dir / "bake_report.json"
    meta_tmp_path = run_dir / "studio_rig_meta.json"
    meta_tmp_path.write_text(json.dumps(fixture["rig_meta"]), encoding="utf-8")

    with tempfile.TemporaryDirectory(prefix="monster_eval_blender_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        cfg_path = tmp_root / "blender_cfg.json"
        script_path = tmp_root / "run_bake_roundtrip.py"
        cfg_path.write_text(
            json.dumps(
                {
                    "worker_dir": str(WORKER_DIR),
                    "blend_path": str(fixture["paths"]["rig_blend"]),
                    "meta_path": str(meta_tmp_path),
                    "trajectory_path": tier1_out["trajectory_path"],
                    "glb_out_path": str(glb_out_path),
                    "bake_report_path": str(bake_report_path),
                }
            ),
            encoding="utf-8",
        )
        script_path.write_text(_BLENDER_BAKE_AND_ROUNDTRIP_SCRIPT, encoding="utf-8")

        env = os.environ.copy()
        env["MONSTER_EVAL_BLENDER_CFG"] = str(cfg_path)
        proc = subprocess.run(
            [
                blender_bin,
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                str(script_path),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        (run_dir / "blender_bake.log").write_text(proc.stdout, encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"Headless Blender GLB roundtrip failed (exit {proc.returncode}). See {run_dir / 'blender_bake.log'}")

    bake_report = json.loads(bake_report_path.read_text(encoding="utf-8"))

    # Re-evaluate with Stage 5 bake_report populated
    return execute_tier1_fast(
        benchmark_id=benchmark_id,
        fixture_dir=fixture_dir,
        tier_label="full",
        run_dir=run_dir,
        save_baseline=save_baseline,
        bake_report=bake_report,
    )
