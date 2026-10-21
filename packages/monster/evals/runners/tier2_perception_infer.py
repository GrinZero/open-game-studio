#!/usr/bin/env python3
"""tier2_perception_infer.py — Tier 2 Native 2K DINOv2 Inference + Kinematic Evaluation Runner."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

MONSTER_ROOT = Path(__file__).resolve().parents[2]
WORKER_DIR = MONSTER_ROOT / "worker"
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from evals.datasets.loader import load_benchmark_fixture  # noqa: E402
from evals.runners.tier1_fast_solver import RUNS_ROOT, execute_tier1_fast  # noqa: E402
from pose_features import dinov2_panel_padding  # noqa: E402


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dinov2_2k_inference(
    video_path: Path,
    checkpoint_path: Path,
    studio_rig_meta: dict[str, Any],
    output_npz_path: Path,
    receipt_path: Path,
    dinov2_repo_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Extracts 2K (`2560x1440`) dual-orthographic frames via ffmpeg and runs `dinov2_head.pt`
    inference on MPS/CUDA/CPU to produce `observations.npz`.
    """
    import torch
    from PIL import Image
    import dinov2_pose_trainer as trainer

    calib = studio_rig_meta.get("calibration", {})
    panel_w = int(calib.get("panel_width", 1280))
    panel_h = int(calib.get("panel_height", 1440))
    comp_w = int(calib.get("composite_width", panel_w * 2))
    comp_h = int(calib.get("composite_height", panel_h))
    pad_top, pad_bottom, pad_left, pad_right = dinov2_panel_padding(panel_w, panel_h)
    padded_w = panel_w + pad_left + pad_right
    padded_h = panel_h + pad_top + pad_bottom

    bone_names = sorted(studio_rig_meta.get("bones", {}).keys())
    num_kp = 2 * len(bone_names)

    if torch.cuda.is_available():
        device = "cuda"
        use_autocast = True
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
        use_autocast = True
    else:
        device = "cpu"
        use_autocast = False

    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Resolve local DINOv2 hub repo if available to avoid network access
    repo_candidates = [
        dinov2_repo_dir,
        Path(os.environ.get("MONSTER_DINOV2_REPO", "")) if os.environ.get("MONSTER_DINOV2_REPO") else None,
        Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main",
        Path("/Users/bugyaluwang/project/star-maker/.asset-workbench/local/runs/fantasy_creature_notebook_enhanced_v0003_20261021/dinov2_repo"),
    ]
    local_repo = next((p for p in repo_candidates if p is not None and p.is_dir()), None)
    if local_repo is not None:
        backbone = torch.hub.load(str(local_repo), "dinov2_vitb14_reg", source="local", pretrained=True)
    else:
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg", pretrained=True)

    model = trainer.MonsterDINOv2PoseProbe(
        num_kp,
        custom_backbone=backbone,
        enable_depth_visibility=True,
    )
    model.head.load_state_dict(ck["head_state_dict"])
    model.aux_head.load_state_dict(ck["aux_state_dict"])
    model = model.to(device).eval()

    # Probe frame count or default to 288
    num_frames = int(studio_rig_meta.get("exam_frame_count", 288))
    arrays: dict[str, np.ndarray] = {
        f"{key}_{view}_all": np.zeros((num_frames, num_kp, 2) if key == "uv" else (num_frames, num_kp), np.float32)
        for view in ("front", "side")
        for key in ("uv", "conf", "depth", "visibility", "logvar")
    }

    frame_bytes = comp_w * comp_h * 3
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(video_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    t0 = time.time()
    assert proc.stdout is not None

    with torch.inference_mode():
        for fi in range(num_frames):
            raw = proc.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                raise RuntimeError(f"Short ffmpeg frame read at frame {fi}: expected {frame_bytes}, got {len(raw)}")
            im = Image.frombytes("RGB", (comp_w, comp_h), raw)
            for vi, v in enumerate(("front", "side")):
                crop = im.crop((vi * panel_w, 0, (vi + 1) * panel_w, panel_h))
                x = trainer.preprocess_panel_crop(crop, augment=False)[None].to(device)
                if use_autocast:
                    with torch.autocast(device, dtype=torch.float16):
                        uv, cf, _, aux = model(x, torch.tensor([vi], device=device), return_aux=True)
                else:
                    uv, cf, _, aux = model(x, torch.tensor([vi], device=device), return_aux=True)

                uv_np = uv.float().cpu().numpy()[0]
                arrays[f"uv_{v}_all"][fi, :, 0] = uv_np[:, 0] * padded_w - pad_left
                arrays[f"uv_{v}_all"][fi, :, 1] = uv_np[:, 1] * padded_h - pad_top
                arrays[f"conf_{v}_all"][fi] = cf.float().cpu().numpy()[0]

                for prefix, key in (("depth", "depth"), ("visibility", "visibility_logits"), ("logvar", "depth_logvar")):
                    val = aux[key].float()
                    if prefix == "visibility":
                        val = torch.sigmoid(val)
                    arrays[f"{prefix}_{v}_all"][fi] = val.cpu().numpy()[0]

    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with status {rc}")

    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz_path, **arrays)

    receipt = {
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "input_sha256": _sha256_file(video_path),
        "observations_sha256": _sha256_file(output_npz_path),
        "native_panel": [panel_w, panel_h],
        "padded_panel": [padded_w, padded_h],
        "device": f"{device.upper()} fp16 autocast" if use_autocast else device.upper(),
        "elapsed_s": round(time.time() - t0, 2),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return arrays


def execute_tier2_infer(
    benchmark_id: str = "fantasy_creature_2k_v2",
    fixture_dir: Path | None = None,
    save_baseline: bool = False,
) -> dict[str, Any]:
    """Executes Tier 2 (`infer`): runs DINOv2 2K perception inference on `exam_input_2k_v0001.mp4`
    and then evaluates the resulting observations through Tier 1.
    """
    fixture = load_benchmark_fixture(benchmark_id=benchmark_id, fixture_dir=fixture_dir)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_ROOT / f"{ts}_{benchmark_id}_infer_taxonomy_kind_v2"
    run_dir.mkdir(parents=True, exist_ok=True)

    obs_path = run_dir / "observations.npz"
    receipt_path = run_dir / "inference_receipt.json"
    inferred_obs = run_dinov2_2k_inference(
        video_path=fixture["paths"]["exam_video"],
        checkpoint_path=fixture["paths"]["dinov2_head"],
        studio_rig_meta=fixture["rig_meta"],
        output_npz_path=obs_path,
        receipt_path=receipt_path,
    )

    return execute_tier1_fast(
        benchmark_id=benchmark_id,
        fixture_dir=fixture_dir,
        observations_override=inferred_obs,
        tier_label="infer",
        run_dir=run_dir,
        save_baseline=save_baseline,
    )
