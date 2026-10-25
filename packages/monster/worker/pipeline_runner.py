#!/usr/bin/env python3
"""pipeline_runner.py — Production Colab G4 Orchestrator for the 3D Monster Animation Pipeline.

Unifies the three runtime environments on a single Google Colab G4 (96GB VRAM) instance:
  - Runtime 1 (`/content/skintokens-service/venv/bin/python`): VAST-AI SkinTokens (`demo.run_rig` +
    `demo.start_bpy_server()`) + Headless Blender 4.2 (`bpy==4.2.0`)
  - Runtime 2 (Colab system Python / PyTorch): Frozen Meta DINOv2 Probe Trainer & 3D Triangulator
  - Runtime 3 (`/content/h3-lab/ComfyUI` @ `http://127.0.0.1:8193`): MiniMax H3 FL2VA 8-Step Turbo
    15-Node ComfyUI Workflow Engine (`MiniMaxH3ImageToVideo` + `VHS_VideoCombine`)

Key v2.1 Rigging Fixes (from `thorn_hunter_proxy_v5` diagnostic):
  1. Pre-SkinTokens Mesh Sanitization (`--mode sanitize_input`): applies all transforms, centers XY,
     places Z_min at 0, removes loose vertices, and recalculates consistent outward normals.
  2. Beam-Stabilized Stochastic Sampling (`num_beams=10`, `temperatures=[0.75, 0.85, 0.95, 1.05]`,
     `use_postprocess=True` for 196^3 voxel skinning refinement) to prevent Qwen3-0.6B autoregressive
     skeleton tree collapse or mid-air joint hallucination.
"""

from __future__ import annotations

import argparse
import json
from pose_features import (
    features_enabled,
    dataset_cache_matches,
    resolution_2k_enabled,
    three_view_enabled,
    studio_resolution,
    idle_split_filename,
)
import os
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any
from surface_contact import assert_surface_contact_quality


DRIVE_ROOT = Path(os.environ.get("MONSTER_DRIVE_ROOT", "/content/drive/MyDrive/MonsterStudio"))
WORK_ROOT = Path(os.environ.get("MONSTER_WORK_ROOT", "/content/monster-studio"))
SKINTOKENS_ROOT = Path(os.environ.get("SKINTOKENS_ROOT", "/content/skintokens-service"))
SKINTOKENS_PY = SKINTOKENS_ROOT / "venv" / "bin" / "python"
H3_BASE = Path(os.environ.get("H3_LAB_DIR", "/content/h3-lab"))
COMFY_DIR = H3_BASE / "ComfyUI"
COMFY_PORT = int(os.environ.get("H3_COMFY_PORT", "8193"))
COMFY_URL = os.environ.get("H3_COMFY_URL", f"http://127.0.0.1:{COMFY_PORT}")

H3_PROFILES: dict[str, dict[str, str]] = {
    "int8": {
        "model": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "text": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video": "minimax_h3_video_vae_int8_convrot.safetensors",
        "audio": "minimax_h3_audio_vae_fp32.safetensors",
        "lora": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    },
    "full": {
        "model": "minimax_h3_fl2va_pruned_bf16.safetensors",
        "text": "qwen3vl_32b_minimax_h3_bf16.safetensors",
        "video": "minimax_h3_video_vae_fp16.safetensors",
        "audio": "minimax_h3_audio_vae_fp32.safetensors",
        "lora": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    },
}


def run_cmd(cmd: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    """Executes a subprocess with live logging and raises on non-zero exit."""
    print(f"\n[EXEC] {' '.join(str(c) for c in cmd)}", flush=True)
    merged_env = {**os.environ, **(env or {})}
    subprocess.run([str(c) for c in cmd], env=merged_env, cwd=str(cwd) if cwd else None, check=True)


def validate_rigged_glb_binary(path: Path) -> dict[str, Any]:
    """Verifies that a SkinTokens output `.glb` has valid glTF 2.0 binary headers, non-empty `skins`,
    and `JOINTS_0` + `WEIGHTS_0` attributes on every mesh primitive (ported from `SkinTokens-G4.ipynb`).
    """
    with open(path, "rb") as f:
        magic, version, size = struct.unpack("<4sII", f.read(12))
        if magic != b"glTF" or version != 2 or size != path.stat().st_size:
            raise ValueError(f"Output {path} is not a complete glTF 2.0 binary (.glb)")
        length, kind = struct.unpack("<II", f.read(8))
        if kind != 0x4E4F534A:
            raise ValueError(f"GLB JSON chunk missing in {path}")
        gltf = json.loads(f.read(length))
    skins = gltf.get("skins", [])
    primitives = [p for m in gltf.get("meshes", []) for p in m.get("primitives", [])]
    if not skins or not all("JOINTS_0" in p.get("attributes", {}) and "WEIGHTS_0" in p.get("attributes", {}) for p in primitives):
        raise ValueError(f"Output {path} is missing skeleton skins or JOINTS_0/WEIGHTS_0 attributes")
    return {
        "file": path.name,
        "bytes": size,
        "skins": len(skins),
        "joints": sum(len(s.get("joints", [])) for s in skins),
        "primitives": len(primitives),
    }


def free_comfyui_vram() -> None:
    """Asks the local ComfyUI MiniMax H3 server to unload models from VRAM when switching stages."""
    try:
        req = urllib.request.Request(
            f"{COMFY_URL}/free",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
        print("[VRAM Guard] Released ComfyUI MiniMax H3 VRAM cache.")
    except Exception:
        pass


# ==============================================================================
# Phase A: One-Time Monster Registration (Stages 1 -> 2 -> 3 -> 4)
# ==============================================================================

SKINTOKENS_BATCH_RUNNER_CODE = r"""
import json, os, shutil, sys, time
from pathlib import Path
import torch

repo_dir = Path(sys.argv[1]).resolve()
input_glb = Path(sys.argv[2]).resolve()
cand_dir = Path(sys.argv[3]).resolve()
num_candidates = int(sys.argv[4])

os.chdir(repo_dir)
sys.path.insert(0, str(repo_dir))
import demo

_orig_get_model = demo.get_model
def _eval_get_model(*args, **kwargs):
    m = _orig_get_model(*args, **kwargs)
    m.eval()
    return m
demo.get_model = _eval_get_model

demo.start_bpy_server()
demo.wait_for_bpy_server(timeout=120)
demo.load_model(demo.MODEL_CKPTS[0], None)
demo.model.eval()

# Safe stochastic beam-search schedule:
# Keep num_beams=10 (official TokenRig default) so Qwen3-0.6B never collapses mid-sequence,
# while varying seed + gentle temperature (0.75..1.05) + voxel postprocessing!
TEMPS = [0.75, 0.85, 0.95, 1.05]
BEAMS = [10, 10, 8, 12]

cand_dir.mkdir(parents=True, exist_ok=True)
with torch.inference_mode():
    for i in range(num_candidates):
        out_glb = cand_dir / f"candidate_{i}.glb"
        if out_glb.exists() and out_glb.stat().st_size > 1024:
            continue
        temp = TEMPS[i % len(TEMPS)]
        beams = BEAMS[i % len(BEAMS)]
        src_copy = cand_dir / f"_input_seed_{i}{input_glb.suffix.lower()}"
        shutil.copy2(input_glb, src_copy)
        torch.manual_seed(100 + i * 37)
        demo.run_rig(
            [src_copy],
            5,           # top_k (official default: 5)
            0.95,        # top_p (official default: 0.95)
            float(temp), # temperature (gentle range around 0.85-1.0)
            2.0,         # repetition_penalty (official default: 2.0)
            int(beams),  # num_beams=10 (CRITICAL: prevents autoregressive tree collapse!)
            False,       # use_skeleton
            True,        # use_transfer (preserve original UVs, textures, and scale)
            True,        # use_postprocess (196^3 voxel_skin refinement to prevent weight bleeding)
            [out_glb],
            demo.MODEL_CKPTS[0],
            None,
        )
        src_copy.unlink(missing_ok=True)
        torch.cuda.synchronize()
print("[SkinTokens Batch Runner] Completed all candidates with Beam Search + Voxel Postprocess.")
"""


def run_stage1_skintokens_candidates(input_glb: Path, cand_dir: Path, num_candidates: int = 4) -> list[Path]:
    """Pre-sanitizes the input `.glb` (applying transforms, outward normals, centering, and optional
    yaw alignment) and runs VAST-AI SkinTokens (`num_beams=10`, `use_postprocess=True`, `use_transfer=True`)
    across 4 stochastic seeds.
    """
    cand_dir.mkdir(parents=True, exist_ok=True)
    bpy_worker = Path(__file__).resolve().parent / "studio_bpy_worker.py"
    py_bpy = str(SKINTOKENS_PY) if SKINTOKENS_PY.exists() else sys.executable

    sanitized_glb = cand_dir / "_sanitized_input.glb"
    if not sanitized_glb.exists() and bpy_worker.exists():
        try:
            run_cmd(
                [
                    py_bpy,
                    str(bpy_worker),
                    "--mode",
                    "sanitize_input",
                    "--input_glb",
                    str(input_glb.resolve()),
                    "--out_glb",
                    str(sanitized_glb.resolve()),
                ]
            )
        except Exception as exc:
            print(f"[Mesh Sanitizer Warning] Falling back to raw input GLB: {exc}")
            shutil.copy2(input_glb, sanitized_glb)
    elif not sanitized_glb.exists():
        shutil.copy2(input_glb, sanitized_glb)

    candidates = [cand_dir / f"candidate_{i}.glb" for i in range(num_candidates)]
    repo_dir = SKINTOKENS_ROOT / "upstream"

    if not all(c.exists() and c.stat().st_size > 1024 for c in candidates):
        if repo_dir.exists() and SKINTOKENS_PY.exists():
            run_cmd(
                [
                    str(SKINTOKENS_PY),
                    "-c",
                    SKINTOKENS_BATCH_RUNNER_CODE,
                    str(repo_dir),
                    str(sanitized_glb.resolve()),
                    str(cand_dir.resolve()),
                    str(num_candidates),
                ],
                cwd=repo_dir,
            )
        else:
            raise RuntimeError(
                f"SkinTokens environment not found at {SKINTOKENS_ROOT}. "
                "Run Cell 2 of MonsterStudio_Colab_G4.ipynb to bootstrap SkinTokens + bpy 4.2."
            )

    for c in candidates:
        if c.exists():
            summary = validate_rigged_glb_binary(c)
            print(f"[Stage 1 GLB Verified] {summary}")
    return candidates


def register_new_monster(monster_id: str, input_glb: Path, num_synth: int = 5000, epochs: int = 15) -> None:
    """Executes Phase A (Stages 1-4) for a new 3D monster asset and archives outputs to Google Drive."""
    free_comfyui_vram()

    monster_drive = DRIVE_ROOT / "monsters" / monster_id
    monster_drive.mkdir(parents=True, exist_ok=True)
    monster_work = WORK_ROOT / "work" / monster_id
    monster_work.mkdir(parents=True, exist_ok=True)

    archived_input = monster_drive / "input_mesh.glb"
    if input_glb.resolve() != archived_input.resolve():
        shutil.copy2(input_glb, archived_input)

    blend_path = monster_drive / "rigged_pruned.blend"
    meta_path = monster_drive / "studio_rig_meta.json"
    limits_path = monster_drive / "joint_limits.json"
    ckpt_path = monster_drive / "dinov2_head.pt"
    exam_report_path = monster_drive / "exam_report.json"

    bpy_worker = Path(__file__).resolve().parent / "studio_bpy_worker.py"
    dino_worker = Path(__file__).resolve().parent / "dinov2_pose_trainer.py"
    py_bpy = str(SKINTOKENS_PY) if SKINTOKENS_PY.exists() else sys.executable

    # Stage 1: Sanitize mesh + Generate 4 SkinTokens candidates + Geometric Audit + Chord/Finger Pruning
    cands = run_stage1_skintokens_candidates(archived_input, monster_work / "candidates", num_candidates=4)
    run_cmd(
        [
            py_bpy,
            str(bpy_worker),
            "--mode",
            "audit",
            "--candidates",
            *[str(c) for c in cands],
            "--blend_path",
            str(blend_path),
            "--meta_path",
            str(meta_path),
        ]
    )
    if not limits_path.exists():
        run_cmd(
            [
                py_bpy,
                str(bpy_worker),
                "--mode",
                "probe_joint_limits",
                "--blend_path",
                str(blend_path),
                "--meta_path",
                str(meta_path),
                "--limits_path",
                str(limits_path),
                "--monster_id",
                str(monster_id),
            ]
        )

    # Stage 2: Render 3,000 Constrained Poses (50% with inter-frame shutter motion blur)
    synth_dir = monster_work / "synth_3000"
    run_cmd(
        [
            py_bpy,
            str(bpy_worker),
            "--mode",
            "render_3000",
            "--blend_path",
            str(blend_path),
            "--meta_path",
            str(meta_path),
            "--limits_path",
            str(limits_path),
            "--out_dir",
            str(synth_dir),
            "--num_samples",
            str(num_synth),
        ]
    )
    idle_name = idle_split_filename()
    shutil.copy2(synth_dir / idle_name, monster_drive / idle_name)

    # Stage 3: Train Frozen DINOv2 + Deconv Heatmap Probe (~12 mins on Colab G4)
    run_cmd(
        [
            sys.executable,
            str(dino_worker),
            "--mode",
            "train",
            "--synth_dir",
            str(synth_dir),
            "--meta_path",
            str(meta_path),
            "--ckpt_path",
            str(ckpt_path),
            "--epochs",
            str(epochs),
            "--progress_file",
            str(monster_work / "training_progress.json"),
        ]
    )

    # Stage 4: Synthesize 96-Frame Continuous Random Motion & Run Bone-by-Bone Studio Exam
    exam_dir = monster_work / "studio_exam"
    run_cmd(
        [
            py_bpy,
            str(bpy_worker),
            "--mode",
            "render_exam",
            "--blend_path",
            str(blend_path),
            "--meta_path",
            str(meta_path),
            "--limits_path",
            str(limits_path),
            "--out_dir",
            str(exam_dir),
            "--exam_frames",
            "96",
        ]
    )
    run_cmd(
        [
            sys.executable,
            str(dino_worker),
            "--mode",
            "exam",
            "--ckpt_path",
            str(ckpt_path),
            "--exam_dir",
            str(exam_dir),
            "--out_json",
            str(exam_report_path),
        ]
    )

    shutil.rmtree(synth_dir, ignore_errors=True)
    print(f"\n[PHASE A COMPLETE] Monster '{monster_id}' registered and certified in {monster_drive}")



def reexam_existing_monster(monster_id: str, rerender_exam: bool = False) -> dict[str, Any]:
    """Fast Stage 4 Re-Exam & GLB Round-Trip Verification (0 synthetic renders, 0 DINOv2 retraining!).
    Uses the upgraded v2.2 Squared-Confidence Side-Occlusion Triangulator + Two-Stage Swing &
    Child-Plane Twist IK Solver to re-score `exam_gt.json` and bake `stage4_reexam_fixed.glb`.
    """
    monster_drive = DRIVE_ROOT / "monsters" / monster_id
    monster_work = WORK_ROOT / "work" / monster_id
    blend_path = monster_drive / "rigged_pruned.blend"
    meta_path = monster_drive / "studio_rig_meta.json"
    limits_path = monster_drive / "joint_limits.json"
    ckpt_path = monster_drive / "dinov2_head.pt"
    exam_dir = monster_work / "studio_exam"
    exam_report_path = monster_drive / "exam_report.json"
    reexam_glb_path = monster_drive / "stage4_reexam_fixed.glb"
    roundtrip_report_path = monster_drive / "exam_glb_roundtrip_report.json"

    bpy_worker = Path(__file__).resolve().parent / "studio_bpy_worker.py"
    dino_worker = Path(__file__).resolve().parent / "dinov2_pose_trainer.py"
    py_bpy = str(SKINTOKENS_PY) if SKINTOKENS_PY.exists() else sys.executable

    if rerender_exam or not (exam_dir / "exam_gt.json").exists():
        render_exam_cmd = [
            py_bpy,
            str(bpy_worker),
            "--mode",
            "render_exam",
            "--blend_path",
            str(blend_path),
            "--meta_path",
            str(meta_path),
            "--out_dir",
            str(exam_dir),
            "--exam_frames",
            "96",
        ]
        if limits_path.exists():
            render_exam_cmd.extend(["--limits_path", str(limits_path)])
        run_cmd(render_exam_cmd)

    run_cmd(
        [
            sys.executable,
            str(dino_worker),
            "--mode",
            "exam",
            "--ckpt_path",
            str(ckpt_path),
            "--exam_dir",
            str(exam_dir),
            "--out_json",
            str(exam_report_path),
        ]
    )

    pred_traj_json = exam_dir / "exam_pred_traj.json"
    if pred_traj_json.exists() and blend_path.exists():
        bake_cmd = [
            py_bpy,
            str(bpy_worker),
            "--mode",
            "bake_ik",
            "--blend_path",
            str(blend_path),
            "--meta_path",
            str(meta_path),
            "--joints_json",
            str(pred_traj_json),
            "--out_glb",
            str(reexam_glb_path),
        ]
        if limits_path.exists():
            bake_cmd.extend(["--limits_path", str(limits_path)])
        run_cmd(bake_cmd)
        run_cmd(
            [
                py_bpy,
                str(bpy_worker),
                "--mode",
                "verify_exam_glb",
                "--out_glb",
                str(reexam_glb_path),
                "--exam_gt_json",
                str(exam_dir / "exam_gt.json"),
                "--out_report",
                str(roundtrip_report_path),
            ]
        )
    return json.loads(exam_report_path.read_text(encoding="utf-8"))


# ==============================================================================
# Phase B: Action Video Generation via MiniMax H3 FL2VA + 3D IK Baking (Stages 5-6)
# ==============================================================================

def build_split_screen_prompt(action_desc: str, enable_3view: bool | None = None) -> str:
    """Constructs the structured split-screen orthographic prompt for MiniMax H3."""
    use_3view = three_view_enabled() if enable_3view is None else bool(enable_3view)
    if use_3view:
        return (
            "Split-screen three-panel technical character reference sheet. "
            "Left panel: left side profile orthographic view. "
            "Center panel: front orthographic view. "
            "Right panel: right side profile orthographic view. "
            f"All three panels show the exact same creature performing a synchronized {action_desc} motion "
            "in place on a dark studio floor. Static locked camera, no camera pan or zoom."
        )
    return (
        "Split-screen two-panel technical character reference sheet. "
        "Left panel: front orthographic view. Right panel: side profile orthographic view. "
        f"Both panels show the exact same creature performing a synchronized {action_desc} motion "
        "in place on a dark studio floor. Static locked camera, no camera pan or zoom."
    )


def build_minimax_h3_comfy_graph(
    prompt_text: str,
    first_frame_name: str,
    last_frame_name: str | None = None,
    width: int | None = None,
    height: int | None = None,
    frames: int = 124,
    steps: int = 8,
    seed: int = 424242,
    profile: str | None = None,
    use_turbo_lora: bool = True,
    filename_prefix: str = "monster_h3",
) -> dict[str, Any]:
    """Constructs the ComfyUI MiniMax H3 workflow graph.
    - When `width <= 1344 and height <= 768` (default 768P mode): runs the single-stage 15-node
      `MiniMaxH3ImageToVideo` workflow at `1344x768`.
    - When `width > 1344 or height > 768` (2K mode, `2560x1440`): runs the official two-stage
      768P -> 2K workflow:
        Pass 1: Generate global motion & dual-panel anatomy at native `1344x768` (`steps=8, denoise=1.0`).
        Pass 2: Separate AV latent (`LTXVSeparateAVLatent`), upscale 24-ch video latent to `2560x1440`
                with the trained 3D neural latent upscaler (`MinimaxH3LatentUpscaler3D`), re-attach audio
                (`LTXVConcatAVLatent`), encode native `2560x1440` keyframe conditioning, and run tiled
                Hires-Fix refinement (`MMH3SplitUpscale`, `steps=4, denoise=0.30`).
    """
    if width is None or height is None:
        _, _, def_w, def_h = studio_resolution()
        width = def_w if width is None else int(width)
        height = def_h if height is None else int(height)
    if int(width) % 32 != 0 or int(height) % 32 != 0:
        raise ValueError(f"MiniMax H3 resolution {width}x{height} must be divisible by 32.")
    active_profile = profile or os.environ.get("H3_PROFILE", "full")
    files = H3_PROFILES.get(active_profile, H3_PROFILES["full"])
    two_stage_2k = int(width) > 1344 or int(height) > 768
    base_w = 1344 if two_stage_2k else int(width)
    base_h = 768 if two_stage_2k else int(height)

    graph: dict[str, Any] = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": files["model"], "weight_dtype": "default"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": files["text"], "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": files["video"]}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": files["audio"]}},
        "7": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["3", 0],
                "vae": ["4", 0],
                "prompt": prompt_text,
                "width": base_w,
                "height": base_h,
                "length": int(frames),
            },
        },
        "8": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["7", 0]}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "11": {
            "class_type": "BasicScheduler",
            "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": int(steps), "denoise": 1.0},
        },
        "12": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["9", 0],
                "guider": ["8", 0],
                "sampler": ["10", 0],
                "sigmas": ["11", 0],
                "latent_image": ["7", 1],
            },
        },
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["5", 0]}},
        "15": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["13", 0],
                "audio": ["14", 0],
                "frame_rate": 24,
                "loop_count": 0,
                "filename_prefix": filename_prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 18,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
            },
        },
    }
    active_model_ref = ["1", 0]
    if use_turbo_lora:
        graph["2"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": files["lora"], "strength_model": 1.0},
        }
        active_model_ref = ["2", 0]
        graph["8"]["inputs"]["model"] = active_model_ref
        graph["11"]["inputs"]["model"] = active_model_ref

    if first_frame_name:
        graph["20"] = {"class_type": "LoadImage", "inputs": {"image": first_frame_name}}
        graph["7"]["inputs"]["first_frame"] = ["20", 0]
    if last_frame_name:
        graph["21"] = {"class_type": "LoadImage", "inputs": {"image": last_frame_name}}
        graph["7"]["inputs"]["last_frame"] = ["21", 0]

    if two_stage_2k:
        graph["30"] = {
            "class_type": "LTXVSeparateAVLatent",
            "inputs": {"av_latent": ["12", 0]},
        }
        graph["31"] = {
            "class_type": "MinimaxH3LatentUpscaler3D",
            "inputs": {
                "latent": ["30", 0],
                "model_name": "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors",
                "mode": {"mode": "target dimensions", "width": int(width), "height": int(height)},
                "align": 32,
                "enable_temporal_chunking": True,
                "force_unload": True,
                "device": "cuda",
                "precision": "bf16" if active_profile == "full" else "fp16",
            },
        }
        graph["32"] = {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {"video_latent": ["31", 0], "audio_latent": ["30", 1]},
        }
        cond_2k_inputs: dict[str, Any] = {
            "clip": ["3", 0],
            "vae": ["4", 0],
            "prompt": prompt_text,
            "width": int(width),
            "height": int(height),
            "length": int(frames),
        }
        if first_frame_name:
            cond_2k_inputs["first_frame"] = ["20", 0]
        if last_frame_name:
            cond_2k_inputs["last_frame"] = ["21", 0]
        graph["33"] = {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": cond_2k_inputs,
        }
        graph["34"] = {
            "class_type": "BasicScheduler",
            "inputs": {"model": active_model_ref, "scheduler": "simple", "steps": 4, "denoise": 0.30},
        }
        graph["35"] = {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": int(seed) + 1},
        }
        graph["36"] = {
            "class_type": "MMH3TemporalSplitParamsV10",
            "inputs": {
                "chunk_frames": 73,
                "temporal_overlap_frames": 22,
                "anchor_strength": 0.999,
                "motion_anchor_frames": "22",
                "identity_anchor_frames": 24,
            },
        }
        graph["37"] = {
            "class_type": "MMH3SpatialSplitParamsV10",
            "inputs": {
                "tile_width": 1344,
                "tile_height": 768,
                "overlap_ratio": 0.25,
                "fade_ratio": 0.50,
                "min_tile_size": 384,
                "seam_denoise": 0.70,
            },
        }
        graph["38"] = {
            "class_type": "MMH3SplitUpscale",
            "inputs": {
                "model": active_model_ref,
                "conditioning": ["33", 0],
                "latent": ["32", 0],
                "noise": ["35", 0],
                "sampler": ["10", 0],
                "sigmas": ["34", 0],
                "cfg": 1.0,
                "temporal_split_param": ["36", 0],
                "spatial_split_param": ["37", 0],
                "seam_polish": "auto",
                "color_match": True,
            },
        }
        graph["13"]["inputs"]["samples"] = ["38", 0]
        graph["14"]["inputs"]["samples"] = ["38", 0]
    return graph


def ensure_comfyui_server_running(require_2k_nodes: bool = False) -> None:
    """Checks if the local ComfyUI MiniMax H3 server is listening on 127.0.0.1:8193; if not,
    launches `/content/h3-lab/ComfyUI/main.py` on loopback and waits for `MiniMaxH3ImageToVideo`.
    """
    import requests

    required_nodes = ["MiniMaxH3ImageToVideo", "VHS_VideoCombine", "VAEDecodeAudio"]
    if require_2k_nodes:
        required_nodes.extend(["MinimaxH3LatentUpscaler3D", "MMH3SplitUpscale"])

    try:
        r = requests.get(f"{COMFY_URL}/system_stats", timeout=2)
        if r.ok:
            if require_2k_nodes:
                nodes = requests.get(f"{COMFY_URL}/object_info", timeout=10).json()
                for req_node in required_nodes:
                    if req_node not in nodes:
                        raise RuntimeError(
                            f"Required ComfyUI 2K node '{req_node}' missing. Re-run setup-comfy.py."
                        )
            return
    except RuntimeError:
        raise
    except Exception:
        pass

    if not (COMFY_DIR / "main.py").exists():
        raise RuntimeError(
            f"ComfyUI not found at {COMFY_DIR}. Run Cell 3 of MonsterStudio_Colab_G4.ipynb first."
        )
    H3_BASE.mkdir(parents=True, exist_ok=True)
    log_file = (H3_BASE / "comfy.log").open("a")
    proc = subprocess.Popen(
        [
            sys.executable,
            str(COMFY_DIR / "main.py"),
            "--listen",
            "127.0.0.1",
            "--port",
            str(COMFY_PORT),
            "--disable-auto-launch",
        ],
        cwd=str(COMFY_DIR),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    for _ in range(150):
        if proc.poll() is not None:
            tail = (H3_BASE / "comfy.log").read_text(errors="replace")[-4000:]
            raise RuntimeError(f"ComfyUI server exited prematurely:\n{tail}")
        try:
            if requests.get(f"{COMFY_URL}/system_stats", timeout=2).ok:
                nodes = requests.get(f"{COMFY_URL}/object_info", timeout=10).json()
                for req_node in required_nodes:
                    if req_node not in nodes:
                        raise RuntimeError(f"Required ComfyUI node '{req_node}' missing.")
                return
        except Exception:
            time.sleep(2)
    raise TimeoutError("ComfyUI MiniMax H3 server did not become ready within 300 seconds.")


def submit_minimax_h3_comfy_job(
    idle_split_png: Path,
    prompt_text: str,
    loop_lock: bool,
    seed: int,
    out_mp4: Path,
    profile: str | None = None,
    width: int | None = None,
    height: int | None = None,
    timeout_sec: int = 2400,
) -> Path:
    """Stages the reference split-screen PNG (`1344x768` or `2560x1440`) into `/content/h3-lab/ComfyUI/input/`,
    builds the single-stage (768P) or two-stage (768P -> 2K) `MiniMaxH3ImageToVideo` workflow graph,
    submits `POST /prompt`, and polls `GET /history/{prompt_id}`.
    """
    import requests

    if width is None or height is None:
        _, _, def_w, def_h = studio_resolution()
        width = def_w if width is None else int(width)
        height = def_h if height is None else int(height)
    active_profile = profile or os.environ.get("H3_PROFILE", "full")

    ensure_comfyui_server_running(require_2k_nodes=(int(width) > 1344 or int(height) > 768))

    ref_name = f"h3ref_{uuid.uuid4().hex[:12]}.png"
    comfy_input_dir = COMFY_DIR / "input"
    if comfy_input_dir.exists():
        shutil.copy2(idle_split_png, comfy_input_dir / ref_name)
    else:
        with open(idle_split_png, "rb") as f:
            up_resp = requests.post(
                f"{COMFY_URL}/upload/image",
                files={"image": (ref_name, f, "image/png")},
                data={"overwrite": "true"},
                timeout=30,
            )
            up_resp.raise_for_status()
            ref_name = up_resp.json()["name"]

    job_tag = uuid.uuid4().hex[:10]
    graph = build_minimax_h3_comfy_graph(
        prompt_text=prompt_text,
        first_frame_name=ref_name,
        last_frame_name=ref_name if loop_lock else None,
        width=width,
        height=height,
        frames=124,
        steps=8,
        seed=seed,
        profile=active_profile,
        use_turbo_lora=True,
        filename_prefix=f"monster_h3/{job_tag}",
    )

    client_id = f"monster-h3-{job_tag}"
    resp = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": graph, "client_id": client_id},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"ComfyUI /prompt HTTP {resp.status_code}: {resp.text[:2000]}")
    resp_json = resp.json()
    if resp_json.get("node_errors") or "prompt_id" not in resp_json:
        raise RuntimeError(f"ComfyUI graph error: {json.dumps(resp_json, ensure_ascii=False)}")

    prompt_id = resp_json["prompt_id"]
    print(
        f"[Stage 5 MiniMax H3] Submitted prompt_id={prompt_id} "
        f"({width}x{height}, 124 frames, 8-step Turbo, profile={active_profile}, loop_lock={loop_lock}, seed={seed})"
    )

    started = time.monotonic()
    video_info: dict[str, Any] | None = None
    while time.monotonic() - started < timeout_sec:
        hist_resp = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=30)
        if hist_resp.ok:
            hist = hist_resp.json()
            res = hist.get(prompt_id)
            if res:
                status_str = res.get("status", {}).get("status_str")
                if status_str == "error":
                    msgs = res.get("status", {}).get("messages", [])
                    raise RuntimeError(f"MiniMax H3 execution error: {json.dumps(msgs, ensure_ascii=False)[-2500:]}")
                gifs = res.get("outputs", {}).get("15", {}).get("gifs", [])
                video_info = next((x for x in gifs if str(x.get("filename", "")).endswith(".mp4")), None)
                if video_info:
                    break
                if res.get("status", {}).get("completed"):
                    raise RuntimeError("ComfyUI workflow completed without returning an MP4 output in node 15.")
        time.sleep(2.0)
    else:
        requests.post(f"{COMFY_URL}/interrupt", timeout=10)
        raise TimeoutError(f"MiniMax H3 generation timed out after {timeout_sec}s.")

    src_mp4 = (COMFY_DIR / "output" / video_info.get("subfolder", "") / video_info["filename"]).resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_mp4, out_mp4)
    print(
        f"[Stage 5 MiniMax H3 Complete] Saved {out_mp4} "
        f"({out_mp4.stat().st_size / 1024:.1f} KB in {time.monotonic() - started:.1f}s)"
    )
    return out_mp4


def animate_monster_actions(
    monster_id: str,
    action_specs: list[str],
    max_retries: int = 3,
    base_seed: int = 424242,
    bake_glb_on_cloud: bool = True,
) -> list[str]:
    """Executes Phase B (Stages 5-6) for one or more actions (`<action_name>:<loop|once>`)."""
    monster_drive = DRIVE_ROOT / "monsters" / monster_id
    blend_path = monster_drive / "rigged_pruned.blend"
    ckpt_path = monster_drive / "dinov2_head.pt"
    meta_path = monster_drive / "studio_rig_meta.json"
    _, _, target_comp_w, target_comp_h = studio_resolution()
    target_3view = three_view_enabled()
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        calib = meta.get("calibration", {})
        comp_w = int(calib.get("composite_width", target_comp_w))
        comp_h = int(calib.get("composite_height", target_comp_h))
        calib_3view = int(calib.get("num_views", 3 if target_3view else 2)) == 3
        if (comp_w, comp_h) != (target_comp_w, target_comp_h) or calib_3view != target_3view:
            raise RuntimeError(
                f"Monster '{monster_id}' was trained at {comp_w}x{comp_h} (3view={calib_3view}), "
                f"but current worker resolution is {target_comp_w}x{target_comp_h} "
                f"(MONSTER_ENABLE_2K={'1' if resolution_2k_enabled() else '0'}, "
                f"MONSTER_ENABLE_3VIEW={'1' if target_3view else '0'}). "
                "Re-run train_monster_probe at the matching studio configuration before generating video."
            )
    else:
        comp_w, comp_h, calib_3view = target_comp_w, target_comp_h, target_3view
    idle_png = monster_drive / idle_split_filename(comp_w, comp_h, enable_3view=calib_3view)
    anim_out_dir = monster_drive / "animations"
    anim_out_dir.mkdir(parents=True, exist_ok=True)

    bpy_worker = Path(__file__).resolve().parent / "studio_bpy_worker.py"
    dino_worker = Path(__file__).resolve().parent / "dinov2_pose_trainer.py"
    py_bpy = str(SKINTOKENS_PY) if SKINTOKENS_PY.exists() else sys.executable
    completed_glbs: list[str] = []

    for spec in action_specs:
        parts = spec.split(":")
        action_name = parts[0].strip()
        loop_lock = (len(parts) < 2) or (parts[1].strip().lower() == "loop")
        prompt_text = build_split_screen_prompt(action_name.replace("_", " "), enable_3view=calib_3view)

        action_work = WORK_ROOT / "work" / monster_id / "actions" / action_name
        frames_dir = action_work / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        traj_json = action_work / "joints_3d_traj.json"

        gate_passed = False
        last_res: dict[str, Any] = {}
        for attempt in range(max_retries):
            seed = base_seed + attempt * 101
            mp4_path = action_work / f"raw_seed_{seed}.mp4"
            if not any(frames_dir.glob("*.jpg")):
                submit_minimax_h3_comfy_job(
                    idle_png, prompt_text, loop_lock, seed, mp4_path, width=comp_w, height=comp_h
                )
                if mp4_path.exists():
                    run_cmd(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            str(mp4_path),
                            "-q:v",
                            "2",
                            str(frames_dir / "frame_%04d.jpg"),
                        ]
                    )

            run_cmd(
                [
                    sys.executable,
                    str(dino_worker),
                    "--mode",
                    "triangulate_video",
                    "--ckpt_path",
                    str(ckpt_path),
                    "--video_frames_dir",
                    str(frames_dir),
                    "--out_json",
                    str(traj_json),
                ] + (["--contact_events", str(action_work / "contact_events.json")]
                     if (action_work / "contact_events.json").is_file() else [])
            )
            with open(traj_json, "r", encoding="utf-8") as f:
                last_res = json.load(f)
            if last_res.get("gate_passed", False):
                assert_surface_contact_quality(last_res, action_name)
                gate_passed = True
                print(f"[Stage 5 Gate PASSED] Action '{action_name}' passed on seed {seed}!")
                break
            print(
                f"[Stage 5 Gate REJECTED] Split-screen desync detected "
                f"(v_diff={last_res.get('mean_epipolar_diff_px')}px, corr={last_res.get('vertical_trajectory_corr')}). "
                f"Retrying with next seed..."
            )
            for old_f in frames_dir.glob("*.jpg"):
                old_f.unlink()

        if not gate_passed:
            raise RuntimeError(
                f"[Stage 5 Gate FAILED] Action '{action_name}' failed vertical epipolar sync check after "
                f"{max_retries} attempts (last v_diff={last_res.get('mean_epipolar_diff_px')}px, "
                f"corr={last_res.get('vertical_trajectory_corr')}). Hard-stopping before IK bake per SOP §6.2."
            )

        free_comfyui_vram()
        if bake_glb_on_cloud:
            out_glb = anim_out_dir / f"{action_name}_animated.glb"
            bake_cmd = [
                py_bpy,
                str(bpy_worker),
                "--mode",
                "bake_ik",
                "--blend_path",
                str(blend_path),
                "--meta_path",
                str(monster_drive / "studio_rig_meta.json"),
                "--joints_json",
                str(traj_json),
                "--out_glb",
                str(out_glb),
            ]
            limits_path = monster_drive / "joint_limits.json"
            if limits_path.exists():
                bake_cmd.extend(["--limits_path", str(limits_path)])
            run_cmd(bake_cmd)
            completed_glbs.append(out_glb.name)
            print(f"[PHASE B COMPLETE] Exported animated 3D asset: {out_glb}")
    return completed_glbs


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab G4 All-in-One Monster Studio Orchestrator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_reg = sub.add_parser("register", help="Phase A: Register a new 3D monster (Stages 1-4)")
    p_reg.add_argument("--monster_id", required=True)
    p_reg.add_argument("--input_glb", required=True, type=Path)
    p_reg.add_argument("--num_synth", type=int, default=5000)
    p_reg.add_argument("--epochs", type=int, default=15)

    p_re = sub.add_parser("reexam", help="Fast Stage 4 Re-Exam & Swing+Twist GLB Round-Trip (0 retraining)")
    p_re.add_argument("--monster_id", required=True)
    p_re.add_argument("--rerender_exam", action="store_true")

    p_anim = sub.add_parser("animate", help="Phase B: Generate & retarget new animations (Stages 5-6)")
    p_anim.add_argument("--monster_id", required=True)
    p_anim.add_argument("--actions", nargs="+", required=True, help="e.g. idle:loop walk:loop slash:once")
    p_anim.add_argument("--max_retries", type=int, default=3)
    p_anim.add_argument("--bake_glb_on_cloud", type=int, default=1, help="1 to bake GLB on cloud, 0 for trajectory only")

    args = parser.parse_args()
    if args.cmd == "register":
        register_new_monster(args.monster_id, args.input_glb, args.num_synth, args.epochs)
    elif args.cmd == "reexam":
        reexam_existing_monster(args.monster_id, args.rerender_exam)
    elif args.cmd == "animate":
        animate_monster_actions(
            args.monster_id,
            args.actions,
            args.max_retries,
            bake_glb_on_cloud=bool(args.bake_glb_on_cloud),
        )


if __name__ == "__main__":
    main()
