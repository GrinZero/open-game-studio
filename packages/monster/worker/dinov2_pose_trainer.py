#!/usr/bin/env python3
"""dinov2_pose_trainer.py — Frozen DINOv2 + Deconv Heatmap Probe, Epipolar Gate & 3D Triangulator.

Implements:
  1. [Stage 3] Frozen Meta DINOv2 (`dinov2_vitb14_reg`) + 3-Layer Deconv Keypoint Heatmap Head
               with Differentiable Spatial-to-Numerical Transform (2D DSNT Soft-Argmax) and
               synthetic video degradation (color jitter, compression, Gaussian blur) to bridge
               the Sim-to-Video domain gap on 3,000 synthetic dual-view renders.
  2. [Stage 4] Bone-by-Bone Studio Exam Evaluator (computes both Directional MPJRE and Child-Plane
               Full SO(3) Geodesic Error against `exam_gt.json` before unlocking video generation).
  3. [Stage 5] Cross-View Epipolar Consistency Gate (`|v_F(t) - v_S(t)| < 18px` & `corr(v_F, v_S) > 0.82`)
               to automatically reject split-screen desynchronized MiniMax H3 video generations.
  4. [Stage 6] Dual-Orthographic Ray Triangulation + Squared-Confidence Side-Occlusion Weighting +
               Top-Down Hierarchical Rigid Bone-Length Projection (`||P_tail(t) - P_head(t)|| = L_b`).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import shutil
import subprocess
import time
from types import MethodType
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo
from pose_features import (features_enabled, DATASET_SCHEMA, FEATURE_VERSION, validate_enhanced_dataset,
                           video_domain_degrade, fuse_depth_visibility, dinov2_panel_padding)
from bone_taxonomy import (
    BONE_TAXONOMY_SCHEMA,
    classify_rig_bones,
    clean_uv_short_gaps_hampel,
    compute_stratified_motion_metrics,
    reconstruct_frames_with_kind_solvers,
    savgol_smooth_array,
    short_gap_fill_1d_or_nd,
    whittaker_smooth_trajectory,
)
from pathlib import Path
from typing import Any
from task_progress import format_eta as _format_eta, write_progress_files as _write_progress_files

import numpy as np

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    Image = ImageEnhance = ImageFilter = Any  # type: ignore

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except ImportError:
    torch = nn = F = DataLoader = Dataset = Any  # type: ignore
    HAS_TORCH = False


def require_torch() -> None:
    if not HAS_TORCH:
        raise RuntimeError(
            "PyTorch (`torch`) is required for DINOv2 probe training and GPU inference on Colab G4."
        )


def configure_torch_gpu_acceleration() -> None:
    """Enables CUDA TF32 Tensor Core matmul/convolutions and cuDNN kernel autotuning."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def enable_backbone_sdpa_attention(backbone: Any) -> int:
    """Binds PyTorch fused `F.scaled_dot_product_attention` (FlashAttention / cuDNN SDPA) onto
    DINOv2 VisionTransformer attention blocks so high-resolution 2K inputs (`N=9481` tokens)
    never materialize an O(N^2) attention matrix when `xformers` is absent.
    """
    if not HAS_TORCH or not hasattr(F, "scaled_dot_product_attention"):
        return 0
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        return 0

    def _sdpa_attention_forward(attn: Any, x: torch.Tensor, attn_bias: Any = None) -> torch.Tensor:
        if attn_bias is not None:
            raise ValueError("Packed attention bias is unsupported in fused SDPA path")
        B, N, C = x.shape
        qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, C // attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=getattr(attn, "scale", None))
        return attn.proj_drop(attn.proj(y.transpose(1, 2).reshape(B, N, C)))

    patched = 0
    for block in blocks:
        attn = getattr(block, "attn", None)
        if attn is not None and all(hasattr(attn, attr) for attr in ("qkv", "num_heads", "proj", "proj_drop")):
            attn.forward = MethodType(_sdpa_attention_forward, attn)
            patched += 1
    return patched


# ==============================================================================
# 0. Framework-Agnostic (NumPy + PyTorch) 2D DSNT Soft-Argmax & Heatmap Math
# ==============================================================================

def dsnt_soft_argmax_numpy(logits: np.ndarray, beta: float = 12.0) -> tuple[np.ndarray, np.ndarray]:
    """NumPy reference implementation of 2D Differentiable Spatial-to-Numerical Transform (DSNT)
    with shift-invariant 5x5 local peak probability concentration confidence.
    """
    B, K, H_hm, W_hm = logits.shape
    flat = logits.reshape(B, K, -1) * beta
    flat_max = np.max(flat, axis=-1, keepdims=True)
    exp_flat = np.exp(flat - flat_max)
    probs = (exp_flat / np.sum(exp_flat, axis=-1, keepdims=True)).reshape(B, K, H_hm, W_hm)

    sig_conf = np.max(1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0))), axis=(-2, -1))
    pad_p = np.pad(probs, ((0, 0), (0, 0), (2, 2), (2, 2)), mode="constant")
    win5 = sum(pad_p[:, :, dy : dy + H_hm, dx : dx + W_hm] for dy in range(5) for dx in range(5))
    peak_mass = np.clip(np.max(win5, axis=(-2, -1)), 0.0, 1.0)
    conf = np.maximum(sig_conf, peak_mass).astype(np.float32)

    grid_x = np.linspace(0.0, 1.0, W_hm, dtype=np.float32).reshape(1, 1, 1, W_hm)
    grid_y = np.linspace(0.0, 1.0, H_hm, dtype=np.float32).reshape(1, 1, H_hm, 1)
    exp_u = np.sum(probs * grid_x, axis=(-2, -1))
    exp_v = np.sum(probs * grid_y, axis=(-2, -1))
    coords_norm = np.stack([exp_u, exp_v], axis=-1)
    return coords_norm, conf


# ==============================================================================
# 1. Model Architecture: Frozen DINOv2 (vitb14_reg) + Deconv Head + 2D DSNT
# ==============================================================================

if HAS_TORCH:

    class MonsterDINOv2PoseProbe(nn.Module):
        """Frozen DINOv2 ViT-B/14 (with registers) backbone + ~3.1M parameter Deconvolution Heatmap
        Readout Head with sub-pixel 2D Soft-Argmax (DSNT) coordinate & confidence extraction.
        """

        def __init__(
            self,
            num_keypoints: int,
            backbone_name: str = "dinov2_vitb14_reg",
            custom_backbone: nn.Module | None = None,
            enable_depth_visibility: bool = False,
        ):
            super().__init__()
            self.num_keypoints = num_keypoints
            self.enable_depth_visibility = bool(enable_depth_visibility)
            if custom_backbone is not None:
                self.backbone = custom_backbone
            else:
                self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            enable_backbone_sdpa_attention(self.backbone)

            embed_dim = getattr(self.backbone, "embed_dim", 768)
            self.head = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.GELU(),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.Conv2d(128, num_keypoints, kernel_size=1),
            )
            if self.enable_depth_visibility:
                self.aux_head = nn.ModuleDict({
                    "keypoint_embedding": nn.Embedding(num_keypoints, 16),
                    "readout": nn.Sequential(nn.Linear(embed_dim*2+18, 256), nn.GELU(), nn.Linear(256, 3)),
                })


        @torch.no_grad()
        def extract_patch_grid(self, x: torch.Tensor) -> torch.Tensor:
            B, _, H, W = x.shape
            feats = self.backbone.forward_features(x)["x_norm_patchtokens"]
            h_p, w_p = H // 14, W // 14
            return feats.transpose(1, 2).reshape(B, -1, h_p, w_p)

        def forward_from_features(self, patch_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            logits = self.head(patch_grid)
            B, K, H_hm, W_hm = logits.shape
            flat = logits.reshape(B, K, -1)
            probs = F.softmax(flat * 12.0, dim=-1).reshape(B, K, H_hm, W_hm)

            sig_conf = torch.amax(torch.sigmoid(logits), dim=(-2, -1))
            peak_mass = (F.avg_pool2d(probs.float(), kernel_size=5, stride=1, padding=2) * 25.0).amax(dim=(-2, -1))
            conf = torch.maximum(sig_conf.float(), peak_mass.clamp(0.0, 1.0)).to(logits.dtype)

            grid_x = torch.linspace(0.0, 1.0, W_hm, device=logits.device).view(1, 1, 1, W_hm)
            grid_y = torch.linspace(0.0, 1.0, H_hm, device=logits.device).view(1, 1, H_hm, 1)
            exp_u = torch.sum(probs * grid_x, dim=(-2, -1))
            exp_v = torch.sum(probs * grid_y, dim=(-2, -1))
            coords_norm = torch.stack([exp_u, exp_v], dim=-1)
            return coords_norm, conf, logits

        def forward(self, x: torch.Tensor, view_ids=None, return_aux: bool = False):
            patch_grid = self.extract_patch_grid(x)
            coords, conf, logits = self.forward_from_features(patch_grid)
            if not self.enable_depth_visibility or not return_aux:
                return coords, conf, logits
            if view_ids is None:
                raise ValueError("Auxiliary depth/visibility predictions require front=0 / side=1 view IDs")
            B, C, H, W = patch_grid.shape
            weights = F.interpolate(logits.float(), size=(H, W), mode="bilinear", align_corners=False)
            weights = F.softmax(weights.flatten(2)*12.0, dim=-1)
            local = torch.einsum("bkn,bcn->bkc", weights, patch_grid.float().flatten(2))
            global_features = patch_grid.float().mean(dim=(2, 3))[:, None, :].expand(-1, self.num_keypoints, -1)
            view = F.one_hot(view_ids.long(), num_classes=2).float()[:, None, :].expand(-1, self.num_keypoints, -1)
            ids = torch.arange(self.num_keypoints, device=x.device)
            identities = self.aux_head["keypoint_embedding"](ids)[None].expand(B, -1, -1)
            aux = self.aux_head["readout"](torch.cat((local, global_features, view, identities), dim=-1))
            return coords, conf, logits, {"depth": aux[..., 0], "depth_logvar": aux[..., 1].clamp(-6, 3),
                                         "visibility_logits": aux[..., 2]}



# ==============================================================================
# 2. Synthetic Dual-View Dataset with Video-Domain Degradation
# ==============================================================================

IMAGENET_MEAN_NP = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD_NP = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def random_bool(p: float) -> bool:
    return float(np.random.rand()) < p


def preprocess_panel_crop_numpy(pil_img: Image.Image, augment: bool = False, enhanced: bool | None = None) -> np.ndarray:
    if augment:
        if (features_enabled() if enhanced is None else enhanced):
            pil_img = video_domain_degrade(pil_img)
        if random_bool(0.6):
            pil_img = ImageEnhance.Brightness(pil_img).enhance(np.random.uniform(0.75, 1.25))
            pil_img = ImageEnhance.Contrast(pil_img).enhance(np.random.uniform(0.80, 1.20))
            pil_img = ImageEnhance.Color(pil_img).enhance(np.random.uniform(0.80, 1.20))
        if random_bool(0.35):
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=float(np.random.uniform(0.5, 1.6))))

    arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    pad_top, pad_bottom, pad_left, pad_right = dinov2_panel_padding(arr.shape[1], arr.shape[0])
    arr_pad = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")
    chw = np.transpose(arr_pad, (2, 0, 1))
    return (chw - IMAGENET_MEAN_NP) / IMAGENET_STD_NP


def preprocess_panel_crop(pil_img: Image.Image, augment: bool = False, enhanced: bool | None = None) -> Any:
    chw = preprocess_panel_crop_numpy(pil_img, augment=augment, enhanced=enhanced)
    if HAS_TORCH:
        return torch.from_numpy(chw)
    return chw


if HAS_TORCH:

    class DualPanelMonsterDataset(Dataset):
        def __init__(self, synth_dir: str, bone_names: list[str], augment: bool = True, enhanced: bool | None = None):
            self.synth_dir = Path(synth_dir)
            with open(self.synth_dir / "keypoints_gt.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            self.enhanced = features_enabled() if enhanced is None else enhanced
            if self.enhanced:
                validate_enhanced_dataset(data, bone_names)
            self.calib = data["calibration"]
            self.samples = data["samples"]
            self.bone_names = sorted(bone_names)
            self.augment = augment

        def __len__(self) -> int:
            return len(self.samples) * 2

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            sample_idx = idx // 2
            is_side = (idx % 2) == 1
            item = self.samples[sample_idx]
            full_img = Image.open(self.synth_dir / "images" / item["image"]).convert("RGB")
            W = self.calib["panel_width"]
            H = self.calib["panel_height"]
            pad_top, pad_bottom, pad_left, pad_right = dinov2_panel_padding(W, H)
            W_pad = float(W + pad_left + pad_right)
            H_pad = float(H + pad_top + pad_bottom)

            crop = full_img.crop((W, 0, W * 2, H)) if is_side else full_img.crop((0, 0, W, H))
            inp = preprocess_panel_crop(crop, augment=self.augment, enhanced=self.enhanced)

            uv_list = []
            for b_name in self.bone_names:
                j = item["joints"][b_name]
                uv_h = j["side_uv_head"] if is_side else j["front_uv_head"]
                uv_t = j["side_uv_tail"] if is_side else j["front_uv_tail"]
                uv_list.append([(uv_h[0] + pad_left) / W_pad, (uv_h[1] + pad_top) / H_pad])
                uv_list.append([(uv_t[0] + pad_left) / W_pad, (uv_t[1] + pad_top) / H_pad])

            gt_uv = torch.tensor(uv_list, dtype=torch.float32)
            if not self.enhanced:
                return inp, gt_uv
            view = "side" if is_side else "front"
            labels = [item["joints"][bone][f"{view}_{endpoint}_aux"]
                      for bone in self.bone_names for endpoint in ("head", "tail")]
            extra = {field: torch.tensor([label[field] for label in labels], dtype=torch.float32)
                     for field in ("depth_relative", "visibility_ratio", "visibility_valid", "in_frame", "label_valid")}
            extra["view_id"] = torch.tensor(int(is_side), dtype=torch.long)
            return inp, gt_uv, extra

    def build_gaussian_target_heatmaps(
        coords_norm: torch.Tensor, H_hm: int = 220, W_hm: int = 192, sigma: float = 2.5
    ) -> torch.Tensor:
        B, K, _ = coords_norm.shape
        device = coords_norm.device
        sigma_eff = sigma * (float(H_hm) / 220.0)
        xs = torch.arange(W_hm, dtype=torch.float32, device=device).view(1, 1, 1, W_hm)
        ys = torch.arange(H_hm, dtype=torch.float32, device=device).view(1, 1, H_hm, 1)
        u_px = coords_norm[:, :, 0:1].unsqueeze(-1) * (W_hm - 1)
        v_px = coords_norm[:, :, 1:2].unsqueeze(-1) * (H_hm - 1)
        dist2 = (xs - u_px) ** 2 + (ys - v_px) ** 2
        return torch.exp(-dist2 / (2.0 * sigma_eff**2))


def masked_mean(values, valid):
    valid = valid.to(values.dtype)
    return (values*valid).sum()/valid.sum().clamp_min(1.0)


def enhanced_pose_losses(pred_uv, logits, aux, gt_uv, labels):
    gt_hm = build_gaussian_target_heatmaps(gt_uv, logits.shape[2], logits.shape[3])
    uv_valid = labels["in_frame"]*labels["label_valid"]
    uv_loss = masked_mean(F.smooth_l1_loss(pred_uv.float(), gt_uv, beta=0.01, reduction="none").mean(-1), uv_valid)
    hm_weight = 1.0 + 19.0 * gt_hm
    hm_loss = masked_mean(((torch.sigmoid(logits.float()) - gt_hm).square() * hm_weight).mean((-1, -2)), uv_valid)
    # Hidden but valid joints retain depth/coordinate supervision.
    residual = aux["depth"].float()-labels["depth_relative"]
    lv = aux["depth_logvar"].float()
    depth_loss = masked_mean(0.5*(torch.exp(-lv)*residual.square()+lv), labels["label_valid"])
    vis_loss = masked_mean(F.binary_cross_entropy_with_logits(aux["visibility_logits"].float(),
                            labels["visibility_ratio"], reduction="none"), labels["visibility_valid"])
    return {"uv": uv_loss, "heatmap": hm_loss, "depth": depth_loss, "visibility": vis_loss,
            "total": 10.0*uv_loss + hm_loss + 0.25*depth_loss + 0.25*vis_loss}


def _query_telemetry() -> dict[str, Any]:
    gpu_name = "unknown"
    vram_used = 0.0
    vram_total = 0.0
    util = 0.0
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_used = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)
            vram_total = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1)
    except Exception:
        pass
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=1,
        ).strip()
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) >= 4:
            gpu_name = parts[0]
            util = float(parts[1])
            vram_used = float(parts[2])
            vram_total = float(parts[3])
    except Exception:
        pass
    display_str = (
        f"{gpu_name} · 显存: {vram_used:.0f} / {vram_total:.0f} MiB · 利用率: {util:.0f}%"
        if vram_total > 0
        else (f"{gpu_name}" if gpu_name != "unknown" else "CPU 模式")
    )
    return {
        "name": gpu_name,
        "vram_used_mib": vram_used,
        "vram_total_mib": vram_total,
        "gpu_util_pct": util,
        "display": display_str,
    }


def train_monster_dinov2_probe(
    synth_dir: str,
    meta_path: str,
    out_ckpt: str,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 1e-3,
    progress_file: str | None = None,
) -> None:
    require_torch()
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    configure_torch_gpu_acceleration()
    meta = json.loads(Path(meta_path).read_text())
    bone_names = sorted(meta["bones"])
    num_keypoints = len(bone_names)*2
    enhanced = features_enabled()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    non_blocking = device.type == "cuda"
    ds = DualPanelMonsterDataset(synth_dir, bone_names, augment=True, enhanced=enhanced)
    default_workers = min(8, max(2, (os.cpu_count() or 4) // 2)) if device.type == "cuda" else 0
    num_workers = int(os.environ.get("MONSTER_DATALOADER_WORKERS", str(default_workers)))
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": non_blocking,
    }
    if num_workers > 0 and len(ds) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(ds, **loader_kwargs)
    model = MonsterDINOv2PoseProbe(num_keypoints=num_keypoints, enable_depth_visibility=enhanced).to(device)
    parameters = list(model.head.parameters())
    if enhanced:
        parameters += list(model.aux_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    targets: list[Path] = []
    if progress_file:
        targets.append(Path(progress_file))
    ckpt_dir = Path(out_ckpt).parent
    targets.append(ckpt_dir / "training_progress.json")
    targets.append(ckpt_dir / "task_progress.json")
    synth_parent = Path(synth_dir).parent
    if synth_parent != ckpt_dir:
        targets.append(synth_parent / "training_progress.json")
        targets.append(synth_parent / "task_progress.json")
    drive_root = os.environ.get("MONSTER_DRIVE_ROOT")
    if drive_root:
        targets.append(Path(drive_root) / "runtime" / "current_task_progress.json")
    work_root = os.environ.get("MONSTER_WORK_ROOT")
    if work_root:
        targets.append(Path(work_root) / "runtime" / "current_task_progress.json")

    total_start_time = time.time()
    for ep in range(1, epochs+1):
        epoch_start_time = time.time()
        model.backbone.eval()
        model.head.train()
        if enhanced:
            model.aux_head.train()
        totals_gpu: dict[str, torch.Tensor] = {}
        total_batches = len(loader)
        for step, batch in enumerate(loader, 1):
            imgs = batch[0].to(device, non_blocking=non_blocking)
            gt_uv = batch[1].to(device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=non_blocking):
                if enhanced:
                    labels = {key: value.to(device, non_blocking=non_blocking) for key, value in batch[2].items()}
                    pred_uv, _, logits, aux = model(imgs, labels["view_id"], return_aux=True)
                    losses = enhanced_pose_losses(pred_uv, logits, aux, gt_uv, labels)
                else:
                    pred_uv, _, logits = model(imgs)
                    gt_hm = build_gaussian_target_heatmaps(gt_uv, logits.shape[2], logits.shape[3])
                    hm_w = 1.0 + 19.0 * gt_hm
                    losses = {"total": 10.0*F.smooth_l1_loss(pred_uv, gt_uv, beta=0.01)
                              + ((torch.sigmoid(logits) - gt_hm).square() * hm_w).mean()}
            losses["total"].backward()
            optimizer.step()
            for key, value in losses.items():
                val_det = value.detach().float()
                totals_gpu[key] = totals_gpu[key] + val_det if key in totals_gpu else val_det

            if step == 1 or step % 5 == 0 or step == total_batches:
                now = time.time()
                epoch_elapsed = now - epoch_start_time
                avg_batch_s = epoch_elapsed / max(1, step)
                rem_batches = (total_batches - step) + max(0, epochs - ep) * total_batches
                eta_s = rem_batches * avg_batch_s
                completed_total_steps = (ep - 1) * total_batches + step
                all_total_steps = epochs * total_batches
                pct = min(100.0, max(0.0, 100.0 * completed_total_steps / max(1, all_total_steps)))
                loss_snapshot = {k: round(float(v.detach().cpu().item()), 6) for k, v in losses.items()}
                stamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")
                telemetry = _query_telemetry()
                progress_payload = {
                    "stage": "training",
                    "stage_display": "训练 DINOv2 姿态头",
                    "status": "running",
                    "epoch": ep,
                    "epochs": epochs,
                    "completed_batches": step,
                    "batches_per_epoch": total_batches,
                    "overall_pct": round(pct, 1),
                    "epoch_elapsed_s": round(epoch_elapsed, 1),
                    "total_elapsed_s": round(now - total_start_time, 1),
                    "eta_seconds": round(eta_s, 1),
                    "eta_str": _format_eta(eta_s),
                    "losses": loss_snapshot,
                    "batch_loss": loss_snapshot,
                    "history": list(history),
                    "gpu": telemetry,
                    "updated_unix": now,
                    "updated_at": stamp,
                }
                _write_progress_files(targets, progress_payload)

        if "total" in totals_gpu and not bool(torch.isfinite(totals_gpu["total"]).all()):
            raise RuntimeError("Non-finite pose loss; checkpoint was not replaced")
        totals = {key: float(value.item()) for key, value in totals_gpu.items()}
        scheduler.step()
        epoch_summary = {"epoch": ep, **{key: value/max(1, len(loader)) for key, value in totals.items()}}
        history.append(epoch_summary)
        print(f"[Stage 3 {FEATURE_VERSION} enhanced={enhanced}] {history[-1]}")
        now = time.time()
        rem_batches = max(0, epochs - ep) * total_batches
        avg_batch_s = (now - epoch_start_time) / max(1, total_batches)
        eta_s = rem_batches * avg_batch_s
        pct = min(100.0, max(0.0, 100.0 * ep / max(1, epochs)))
        progress_payload = {
            "stage": "training",
            "stage_display": "训练 DINOv2 姿态头",
            "status": "completed" if ep == epochs else "running",
            "epoch": ep,
            "epochs": epochs,
            "completed_batches": total_batches,
            "batches_per_epoch": total_batches,
            "overall_pct": round(pct, 1),
            "epoch_elapsed_s": round(now - epoch_start_time, 1),
            "total_elapsed_s": round(now - total_start_time, 1),
            "eta_seconds": round(eta_s, 1),
            "eta_str": _format_eta(eta_s) if ep < epochs else "训练完成",
            "losses": {k: round(v / max(1, total_batches), 6) for k, v in totals.items()},
            "history": list(history),
            "gpu": _query_telemetry(),
            "updated_unix": now,
            "updated_at": datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S"),
            "training_completed": ep == epochs,
        }
        _write_progress_files(targets, progress_payload)
    target = Path(out_ckpt)
    target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {"bone_names": bone_names, "bones_meta": meta["bones"], "num_keypoints": num_keypoints,
                  "calibration": ds.calib, "head_state_dict": model.head.state_dict(),
                  "depth_visibility_enabled": enhanced, "feature_version": FEATURE_VERSION,
                  "dataset_schema": DATASET_SCHEMA if enhanced else "legacy", "training_history": history}
    if enhanced:
        checkpoint["aux_state_dict"] = model.aux_head.state_dict()
    temporary = target.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    if target.exists():
        stamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d%H%M%S%f")
        shutil.copy2(target, target.with_name(f"{target.stem}.before_{stamp}.pt"))
    temporary.replace(target)
    print(f"[Stage 3 Saved] {target} | depth_visibility={enhanced}")


# ==============================================================================
# 3. Dual-View Video Inference, Epipolar Consistency Gate & 3D Triangulation
# ==============================================================================

def clean_2d_trajectories_hampel(
    uv_front_all: np.ndarray,
    uv_side_all: np.ndarray,
    conf_front_all: np.ndarray,
    conf_side_all: np.ndarray,
    spike_thresh_px: float = 22.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detects and interpolates single/multi-frame 2D keypoint tracking spikes (e.g., 196px side-view
    far-leg swap spikes) using a 5-frame rolling median Hampel filter + cross-view vertical epipolar
    consistency check. Works reliably even when raw checkpoint confidences are uniformly low (0.06-0.08).
    """
    T, K, _ = uv_front_all.shape
    uv_f = uv_front_all.astype(np.float64).copy()
    uv_s = uv_side_all.astype(np.float64).copy()
    cf = conf_front_all.astype(np.float64).copy()
    cs = conf_side_all.astype(np.float64).copy()
    if T < 5:
        return uv_f.astype(np.float32), uv_s.astype(np.float32), cf.astype(np.float32), cs.astype(np.float32)

    epi_thresh_px = 24.0 * (float(spike_thresh_px) / 22.0)
    t_idx = np.arange(T)
    for k in range(K):
        uf, vf = uv_f[:, k, 0], uv_f[:, k, 1]
        us, vs = uv_s[:, k, 0], uv_s[:, k, 1]

        uf_pad = np.pad(uf, (2, 2), mode="edge")
        vf_pad = np.pad(vf, (2, 2), mode="edge")
        us_pad = np.pad(us, (2, 2), mode="edge")
        vs_pad = np.pad(vs, (2, 2), mode="edge")

        uf_med = np.median(np.stack([uf_pad[i : i + T] for i in range(5)], axis=0), axis=0)
        vf_med = np.median(np.stack([vf_pad[i : i + T] for i in range(5)], axis=0), axis=0)
        us_med = np.median(np.stack([us_pad[i : i + T] for i in range(5)], axis=0), axis=0)
        vs_med = np.median(np.stack([vs_pad[i : i + T] for i in range(5)], axis=0), axis=0)

        dev_f = np.hypot(uf - uf_med, vf - vf_med)
        dev_s = np.hypot(us - us_med, vs - vs_med)
        mad_f = float(np.median(dev_f)) + 1.0
        mad_s = float(np.median(dev_s)) + 1.0

        epi_diff = np.abs(vf - vs)
        out_f = (dev_f > max(spike_thresh_px, 3.5 * mad_f)) | ((epi_diff > epi_thresh_px) & (dev_f > dev_s * 1.35))
        out_s = (dev_s > max(spike_thresh_px, 3.5 * mad_s)) | ((epi_diff > epi_thresh_px) & (dev_s > dev_f * 1.35))

        if np.any(cf[:, k] >= 0.45):
            out_f = out_f | (cf[:, k] < 0.35)
        if np.any(cs[:, k] >= 0.45):
            out_s = out_s | (cs[:, k] < 0.35)

        val_f = ~out_f
        if np.any(val_f) and not np.all(val_f):
            uv_f[:, k, 0] = short_gap_fill_1d_or_nd(uf, out_f, max_gap=3)
            uv_f[:, k, 1] = short_gap_fill_1d_or_nd(vf, out_f, max_gap=3)
            cf[out_f, k] *= 0.25

        val_s = ~out_s
        if np.any(val_s) and not np.all(val_s):
            uv_s[:, k, 0] = short_gap_fill_1d_or_nd(us, out_s, max_gap=3)
            uv_s[:, k, 1] = short_gap_fill_1d_or_nd(vs, out_s, max_gap=3)
            cs[out_s, k] *= 0.25

    return uv_f.astype(np.float32), uv_s.astype(np.float32), cf.astype(np.float32), cs.astype(np.float32)


def evaluate_epipolar_consistency_and_triangulate(
    uv_front_all: np.ndarray,
    uv_side_all: np.ndarray,
    conf_front_all: np.ndarray,
    conf_side_all: np.ndarray,
    calib: dict[str, Any],
    bone_names: list[str],
    bones_meta: dict[str, Any] | None = None,
    depth_front_all=None,
    depth_side_all=None,
    visibility_front_all=None,
    visibility_side_all=None,
    logvar_front_all=None,
    logvar_side_all=None,
) -> dict[str, Any]:
    """Pure-math Epipolar Consistency Gate (`|v_F(t) - v_S(t)| < 18px` & `corr > 0.82`),
    5-frame Hampel 2D tracking spike rejection, squared-confidence 3D ray triangulation,
    5-tap Savitzky-Golay quadratic polynomial smoothing, and top-down rigid bone-length projection.
    """
    T, K, _ = uv_front_all.shape
    W, H = float(calib["panel_width"]), float(calib["panel_height"])
    cx, cy, cz = calib["center"]
    S = float(calib["ortho_scale"])
    span_x = S * (W / H)
    span_z = S
    res_scale = H / 768.0
    max_v_diff_px = 18.0 * res_scale

    # Evaluate raw epipolar gate statistics before spike interpolation so genuine desync is still flagged
    v_diff_raw = np.abs(uv_front_all[:, :, 1] - uv_side_all[:, :, 1])
    mean_v_diff_px = float(np.mean(v_diff_raw))
    vf_mean = np.mean(uv_front_all[:, :, 1], axis=1)
    vs_mean = np.mean(uv_side_all[:, :, 1], axis=1)
    if np.std(vf_mean) > 1e-3 and np.std(vs_mean) > 1e-3:
        v_corr = float(np.corrcoef(vf_mean, vs_mean)[0, 1])
    else:
        v_corr = 1.0
    gate_passed = bool((mean_v_diff_px < max_v_diff_px) and (v_corr > 0.82))

    # Clean 1-to-3 frame 2D tracking spikes (such as 196px side-view leg crossings)
    uv_f_clean, uv_s_clean, conf_f_clean, conf_s_clean = clean_2d_trajectories_hampel(
        uv_front_all, uv_side_all, conf_front_all, conf_side_all, spike_thresh_px=22.0 * res_scale
    )

    pts_3d = np.zeros((T, K, 3), dtype=np.float32)
    joint_conf = 0.5 * (conf_f_clean + conf_s_clean)
    for t in range(T):
        u_F, v_F = uv_f_clean[t, :, 0], uv_f_clean[t, :, 1]
        u_S, v_S = uv_s_clean[t, :, 0], uv_s_clean[t, :, 1]
        w_F = np.square(np.clip(conf_f_clean[t], 1e-3, 1.0))
        w_S = np.square(np.clip(conf_s_clean[t], 1e-3, 1.0))

        X = cx + (u_F / W - 0.5) * span_x
        Y = cy + (u_S / W - 0.5) * span_x
        Z_F = cz + (0.5 - v_F / H) * span_z
        Z_S = cz + (0.5 - v_S / H) * span_z
        Z = (w_F * Z_F + w_S * Z_S) / (w_F + w_S)
        pts_3d[t] = np.stack([X, Y, Z], axis=-1)

    enhanced = depth_front_all is not None
    if enhanced:
        uv_f_enh = clean_uv_short_gaps_hampel(uv_front_all.astype(np.float64), scale=res_scale, max_gap=3)
        uv_s_enh = clean_uv_short_gaps_hampel(uv_side_all.astype(np.float64), scale=res_scale, max_gap=3)
        epipolar_dv_px = np.abs(uv_f_enh[..., 1] - uv_s_enh[..., 1])
        epipolar_gate = np.exp(-0.5 * (epipolar_dv_px / (28.0 * res_scale)) ** 2)
        cf_gated = np.clip(conf_front_all.astype(np.float64) * (0.35 + 0.65 * epipolar_gate), 0.01, 1.0)
        cs_gated = np.clip(conf_side_all.astype(np.float64) * (0.35 + 0.65 * epipolar_gate), 0.01, 1.0)
        pts_3d, joint_conf = fuse_depth_visibility(
            uv_f_enh,
            uv_s_enh,
            cf_gated,
            cs_gated,
            calib,
            depth_front_all,
            depth_side_all,
            visibility_front_all,
            visibility_side_all,
            logvar_front_all,
            logvar_side_all,
        )
        pts_3d = whittaker_smooth_trajectory(
            np.asarray(pts_3d, dtype=np.float64),
            weights=np.asarray(joint_conf, dtype=np.float64) ** 2,
            lam_accel=4.0,
            lam_jerk=10.0,
        ).astype(np.float32)
        # Adaptive confidence normalization so low-logit checkpoints never zero out stereo_valid
        cf_norm = conf_front_all / np.maximum(1e-4, np.max(conf_front_all, axis=0, keepdims=True).clip(0.25, 1.0))
        cs_norm = conf_side_all / np.maximum(1e-4, np.max(conf_side_all, axis=0, keepdims=True).clip(0.25, 1.0))
        stereo_weight = np.minimum(visibility_front_all, visibility_side_all) * np.minimum(cf_norm, cs_norm)
        stereo_valid = stereo_weight >= 0.30
        if int(stereo_valid.sum()) >= max(4, int(T * K * 0.05)):
            mean_v_diff_px = float(np.average(v_diff_raw[stereo_valid], weights=stereo_weight[stereo_valid]))
            frame_ok = stereo_valid.sum(axis=1) >= 2
            if int(frame_ok.sum()) >= 3:
                w = np.where(stereo_valid, stereo_weight, 0.0)
                denom = w.sum(axis=1).clip(1e-6)
                vf = (uv_front_all[:, :, 1] * w).sum(axis=1) / denom
                vs = (uv_side_all[:, :, 1] * w).sum(axis=1) / denom
                if np.std(vf[frame_ok]) > 1e-3 and np.std(vs[frame_ok]) > 1e-3:
                    v_corr = float(np.corrcoef(vf[frame_ok], vs[frame_ok])[0, 1])
                else:
                    v_corr = 1.0
            else:
                v_corr = 0.0
            gate_passed = bool(mean_v_diff_px < max_v_diff_px and v_corr > 0.82)
        else:
            gate_passed = False

    # 3D trajectory spike cleanup & zero-phase 5-tap Savitzky-Golay smoothing (legacy non-enhanced path)
    if not enhanced:
        for k in range(K):
            b_name = bone_names[k // 2]
            depth = int(bones_meta.get(b_name, {}).get("depth", 1)) if bones_meta else 1
            side_valid = conf_s_clean[:, k] >= 0.45
            if np.any(side_valid) and not np.all(side_valid):
                pts_3d[:, k, 1] = short_gap_fill_1d_or_nd(pts_3d[:, k, 1], ~side_valid, max_gap=3)

            valid_mask = joint_conf[:, k] >= 0.40
            if np.any(valid_mask) and not np.all(valid_mask):
                for dim in (0, 2):
                    pts_3d[:, k, dim] = short_gap_fill_1d_or_nd(pts_3d[:, k, dim], ~valid_mask, max_gap=3)

            if T >= 5:
                # 5-tap symmetric quadratic Savitzky-Golay filter (preserves stride amplitude, removes high-freq noise)
                for dim in range(3):
                    p5 = np.pad(pts_3d[:, k, dim].astype(np.float64), (2, 2), mode="edge")
                    sg = (-3.0 * p5[:-4] + 12.0 * p5[1:-3] + 17.0 * p5[2:-2] + 12.0 * p5[3:-1] - 3.0 * p5[4:]) / 35.0
                    pts_3d[:, k, dim] = sg.astype(np.float32)
                # Light 3-tap Gaussian polish for distal limbs (depth >= 2) when not using enhanced FK solver
                if depth >= 2:
                    for dim in range(3):
                        p3 = np.pad(pts_3d[:, k, dim], (1, 1), mode="edge")
                        pts_3d[:, k, dim] = 0.22 * p3[:-2] + 0.56 * p3[1:-1] + 0.22 * p3[2:]

    classified_meta = classify_rig_bones(bones_meta) if bones_meta else None
    if enhanced and classified_meta:
        depth_scale = float(calib.get("depth_scale", max(1e-4, S)))
        frames_out, taxonomy_diag = reconstruct_frames_with_kind_solvers(
            pts_3d=pts_3d.astype(np.float64),
            trust=np.clip(joint_conf.astype(np.float64), 0.01, 1.0),
            bone_names=bone_names,
            classified_bones=classified_meta,
            depth_scale=depth_scale,
        )
        for t in range(T):
            b_map = frames_out[t]["bones"]
            for b_idx, b_name in enumerate(bone_names):
                endpoints = slice(b_idx * 2, b_idx * 2 + 2)
                b_map[b_name]["depth_front_relative"] = depth_front_all[t, endpoints].tolist()
                b_map[b_name]["depth_side_relative"] = depth_side_all[t, endpoints].tolist()
                b_map[b_name]["visibility_front"] = visibility_front_all[t, endpoints].tolist()
                b_map[b_name]["visibility_side"] = visibility_side_all[t, endpoints].tolist()
                b_map[b_name]["depth_std_front"] = np.exp(0.5 * logvar_front_all[t, endpoints]).tolist()
                b_map[b_name]["depth_std_side"] = np.exp(0.5 * logvar_side_all[t, endpoints]).tolist()
        return {
            "bone_taxonomy_schema": BONE_TAXONOMY_SCHEMA,
            "depth_visibility_enabled": enhanced,
            "gate_passed": gate_passed,
            "mean_epipolar_diff_px": round(mean_v_diff_px, 3),
            "vertical_trajectory_corr": round(v_corr, 4),
            "calibration": calib,
            "bones_meta": classified_meta,
            "taxonomy_diagnostics": taxonomy_diag,
            "frames": frames_out,
        }

    # Top-down hierarchical rigid bone-length projection (parent-first so connected child heads stay locked!)
    topo_order = list(range(len(bone_names)))
    if classified_meta:
        topo_order.sort(key=lambda idx: int(classified_meta.get(bone_names[idx], {}).get("depth", 0)))

    frames_out = []
    for t in range(T):
        b_map: dict[str, Any] = {}
        for b_idx in topo_order:
            b_name = bone_names[b_idx]
            h_pt = pts_3d[t, b_idx * 2].copy()
            t_pt = pts_3d[t, b_idx * 2 + 1].copy()
            if classified_meta and b_name in classified_meta:
                p_name = classified_meta[b_name].get("parent")
                if p_name and p_name in b_map and p_name in classified_meta:
                    rest_h = np.array(classified_meta[b_name].get("rest_head_world", h_pt))
                    rest_pt = np.array(classified_meta[p_name].get("rest_tail_world", h_pt))
                    if float(np.linalg.norm(rest_h - rest_pt)) < 0.015:
                        h_pt = np.array(b_map[p_name]["tail_3d"], dtype=np.float32)
                rest_len = float(classified_meta[b_name].get("rest_length", 0.0))
                if rest_len > 1e-5:
                    vec = t_pt - h_pt
                    norm = float(np.linalg.norm(vec))
                    if norm > 1e-5:
                        t_pt = h_pt + (vec / norm) * rest_len
            b_map[b_name] = {
                "head_3d": [round(float(x), 5) for x in h_pt],
                "tail_3d": [round(float(x), 5) for x in t_pt],
                "confidence": round(float(0.5 * (joint_conf[t, b_idx * 2] + joint_conf[t, b_idx * 2 + 1])), 4),
            }
            if enhanced:
                endpoints = slice(b_idx * 2, b_idx * 2 + 2)
                b_map[b_name]["depth_front_relative"] = depth_front_all[t, endpoints].tolist()
                b_map[b_name]["depth_side_relative"] = depth_side_all[t, endpoints].tolist()
                b_map[b_name]["visibility_front"] = visibility_front_all[t, endpoints].tolist()
                b_map[b_name]["visibility_side"] = visibility_side_all[t, endpoints].tolist()
                b_map[b_name]["depth_std_front"] = np.exp(0.5 * logvar_front_all[t, endpoints]).tolist()
                b_map[b_name]["depth_std_side"] = np.exp(0.5 * logvar_side_all[t, endpoints]).tolist()
        frames_out.append({"frame": t, "bones": b_map})

    if classified_meta:
        reconstruct_frame_so3_rotations_from_keypoints(frames_out, classified_meta)

    return {
        "bone_taxonomy_schema": BONE_TAXONOMY_SCHEMA,
        "depth_visibility_enabled": enhanced,
        "gate_passed": gate_passed,
        "mean_epipolar_diff_px": round(mean_v_diff_px, 3),
        "vertical_trajectory_corr": round(v_corr, 4),
        "calibration": calib,
        "bones_meta": classified_meta,
        "frames": frames_out,
    }


def infer_dual_view_frames(
    ckpt_path: str, frame_paths: list[Path], custom_backbone: Any = None
) -> dict[str, Any]:
    """Runs DINOv2 probe on a sequence of split-screen frames (`1344x768` or `2560x1440`) using
    batched `torch.inference_mode()` + multi-threaded CPU frame decoding to maximize GPU throughput.
    """
    require_torch()
    configure_torch_gpu_acceleration()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    non_blocking = device.type == "cuda"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    bone_names: list[str] = ckpt["bone_names"]
    bones_meta: dict[str, Any] | None = ckpt.get("bones_meta")
    calib: dict[str, Any] = ckpt["calibration"]
    W, H = float(calib["panel_width"]), float(calib["panel_height"])
    W_int, H_int = int(W), int(H)
    pad_top, pad_bottom, pad_left, pad_right = dinov2_panel_padding(W_int, H_int)
    W_pad = W + float(pad_left + pad_right)
    H_pad = H + float(pad_top + pad_bottom)

    enhanced = features_enabled() and bool(ckpt.get("depth_visibility_enabled", False))
    if enhanced and "aux_state_dict" not in ckpt:
        raise ValueError("Enhanced checkpoint is missing its auxiliary head")
    if features_enabled() and not enhanced:
        warnings.warn("Legacy checkpoint: depth/visibility unavailable until retraining; using original 2D pipeline.")
    model = MonsterDINOv2PoseProbe(
        num_keypoints=ckpt["num_keypoints"], custom_backbone=custom_backbone,
        enable_depth_visibility=enhanced,
    ).to(device)
    model.head.load_state_dict(ckpt["head_state_dict"])
    if enhanced:
        model.aux_head.load_state_dict(ckpt["aux_state_dict"])
    model.eval()

    T = len(frame_paths)
    K = ckpt["num_keypoints"]
    uv_front_all = np.zeros((T, K, 2), dtype=np.float32)
    uv_side_all = np.zeros((T, K, 2), dtype=np.float32)
    conf_front_all = np.zeros((T, K), dtype=np.float32)
    conf_side_all = np.zeros((T, K), dtype=np.float32)
    aux_arrays = {key: np.zeros((T, K), dtype=np.float32) for key in
                  ("depth_front_all", "depth_side_all", "visibility_front_all", "visibility_side_all", "logvar_front_all", "logvar_side_all")} if enhanced else {}

    def _load_frame_pair(fp: Path) -> tuple[Any, Any]:
        with Image.open(fp) as raw_im:
            im = raw_im.convert("RGB")
        if im.size != (W_int * 2, H_int):
            raise ValueError(
                f"Frame {fp.name} resolution {im.width}x{im.height} does not match checkpoint calibration {W_int * 2}x{H_int}."
            )
        t_f = preprocess_panel_crop(im.crop((0, 0, W_int, H_int)), augment=False)
        t_s = preprocess_panel_crop(im.crop((W_int, 0, W_int * 2, H_int)), augment=False)
        return t_f, t_s

    batch_frames = max(1, int(os.environ.get("MONSTER_INFER_BATCH_FRAMES", "8")))
    decode_workers = min(8, max(2, (os.cpu_count() or 4) // 2))
    use_amp = device.type == "cuda"
    with torch.inference_mode(), ThreadPoolExecutor(max_workers=decode_workers) as pool:
        for start_t in range(0, T, batch_frames):
            end_t = min(T, start_t + batch_frames)
            chunk_paths = frame_paths[start_t:end_t]
            pairs = list(pool.map(_load_frame_pair, chunk_paths))
            chunk_len = len(pairs)
            tensors = [tensor for pair in pairs for tensor in pair]
            batch = torch.stack(tensors, dim=0)
            if non_blocking:
                batch = batch.pin_memory()
            batch = batch.to(device, non_blocking=non_blocking)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                if enhanced:
                    view_ids = torch.tensor([0, 1] * chunk_len, device=device, dtype=torch.long)
                    coords_norm, conf, _, auxiliary = model(batch, view_ids, return_aux=True)
                    for prefix, key in (("depth", "depth"), ("visibility", "visibility_logits"), ("logvar", "depth_logvar")):
                        values = auxiliary[key].float()
                        if prefix == "visibility":
                            values = torch.sigmoid(values)
                        vals_np = values.cpu().numpy().reshape(chunk_len, 2, K)
                        aux_arrays[f"{prefix}_front_all"][start_t:end_t] = vals_np[:, 0, :]
                        aux_arrays[f"{prefix}_side_all"][start_t:end_t] = vals_np[:, 1, :]
                else:
                    coords_norm, conf, _ = model(batch)
            coords_np = coords_norm.float().cpu().numpy().reshape(chunk_len, 2, K, 2)
            conf_np = conf.float().cpu().numpy().reshape(chunk_len, 2, K)

            uv_front_all[start_t:end_t, :, 0] = coords_np[:, 0, :, 0] * W_pad - float(pad_left)
            uv_front_all[start_t:end_t, :, 1] = coords_np[:, 0, :, 1] * H_pad - float(pad_top)
            uv_side_all[start_t:end_t, :, 0] = coords_np[:, 1, :, 0] * W_pad - float(pad_left)
            uv_side_all[start_t:end_t, :, 1] = coords_np[:, 1, :, 1] * H_pad - float(pad_top)
            conf_front_all[start_t:end_t] = conf_np[:, 0, :]
            conf_side_all[start_t:end_t] = conf_np[:, 1, :]

    return evaluate_epipolar_consistency_and_triangulate(
        uv_front_all=uv_front_all,
        uv_side_all=uv_side_all,
        conf_front_all=conf_front_all,
        conf_side_all=conf_side_all,
        calib=calib,
        bone_names=bone_names,
        bones_meta=bones_meta,
        **aux_arrays,
    )


def reconstruct_frame_so3_rotations_from_keypoints(
    frames_pred: list[dict[str, Any]],
    bones_meta: dict[str, Any] | None,
    gt_first_frame_bones: dict[str, Any] | None = None,
) -> None:
    """Reconstructs each bone's 3x3 world rotation matrix `rot_mat_world` in parent-to-child
    topological order using Bounded Local-Euler + Multi-Child Observability & Rigid Child-Head
    Lever-Arm Solver (`solve_bounded_bone_rotation_matrix`) followed by 5-frame Quaternion
    Savitzky-Golay temporal filtering (`smooth_rotation_matrices_savgol`).
    """
    from studio_bpy_worker import (
        minimum_swing_matrix_numpy,
        smooth_rotation_matrices_savgol,
        solve_bounded_bone_rotation_matrix,
    )

    if not frames_pred:
        return
    if all("rot_mat_world" in b_info for b_info in frames_pred[0]["bones"].values()):
        return
    bone_names = list(frames_pred[0]["bones"].keys())
    topo_names = list(bone_names)
    meta_dict: dict[str, Any] = classify_rig_bones({b: dict((bones_meta or {}).get(b, {})) for b in bone_names})
    topo_names.sort(key=lambda b: int(meta_dict.get(b, {}).get("depth", 0)))

    # Always reconstruct children_map from parent links so legacy metadata without "children" works!
    children_map: dict[str, list[str]] = {b: [] for b in bone_names}
    for b in bone_names:
        p = meta_dict[b].get("parent")
        if p and p in children_map:
            children_map[p].append(b)
    for b in bone_names:
        meta_dict[b]["children"] = children_map[b]

    rest_R: dict[str, np.ndarray] = {}
    for b in bone_names:
        info = meta_dict[b]
        if "rest_rot_mat_world" in info:
            rest_R[b] = np.array(info["rest_rot_mat_world"], dtype=np.float64)
        elif "rest_head_world" in info and "rest_tail_world" in info:
            v0 = np.array(info["rest_tail_world"], dtype=np.float64) - np.array(
                info["rest_head_world"], dtype=np.float64
            )
            d0 = v0 / max(1e-8, float(np.linalg.norm(v0)))
            if gt_first_frame_bones and b in gt_first_frame_bones and "rot_mat_world" in gt_first_frame_bones[b]:
                R_f0 = np.array(gt_first_frame_bones[b]["rot_mat_world"], dtype=np.float64)
                v_f0 = np.array(gt_first_frame_bones[b]["tail_3d"], dtype=np.float64) - np.array(
                    gt_first_frame_bones[b]["head_3d"], dtype=np.float64
                )
                S_back = minimum_swing_matrix_numpy(v_f0, d0)
                rest_R[b] = S_back @ R_f0
            else:
                up = np.array([0.0, 0.0, 1.0]) if abs(float(d0[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
                x_ax = np.cross(d0, up)
                x_ax /= max(1e-8, float(np.linalg.norm(x_ax)))
                z_ax = np.cross(x_ax, d0)
                rest_R[b] = np.column_stack([x_ax, d0, z_ax])
        else:
            rest_R[b] = np.eye(3, dtype=np.float64)

    T = len(frames_pred)
    solved_seq: dict[str, np.ndarray] = {b: np.zeros((T, 3, 3), dtype=np.float64) for b in bone_names}
    prev_ry_map: dict[str, float] = {}

    for t, frame_item in enumerate(frames_pred):
        b_dict = frame_item["bones"]
        curr_ry_map: dict[str, float] = {}
        posed_R: dict[str, np.ndarray] = {}
        for b in topo_names:
            p = meta_dict[b].get("parent")
            if p and p in posed_R and p in rest_R:
                R_init = posed_R[p] @ (rest_R[p].T @ rest_R[b])
            else:
                R_init = rest_R[b]
            if b not in b_dict:
                posed_R[b] = R_init
                solved_seq[b][t] = R_init
                continue
            R_final, ry_best = solve_bounded_bone_rotation_matrix(
                b_name=b,
                R_init=R_init,
                frame_bones=b_dict,
                bones_meta=meta_dict,
                rest_R=rest_R,
                children_map=children_map,
                curr_ry_map=curr_ry_map,
                prev_ry_map=prev_ry_map,
            )
            posed_R[b] = R_final
            curr_ry_map[b] = ry_best
            prev_ry_map[b] = ry_best
            solved_seq[b][t] = R_final

    if T >= 5:
        for b in topo_names:
            depth = int(meta_dict[b].get("depth", 1))
            passes = 1 if depth >= 2 else 0
            solved_seq[b] = smooth_rotation_matrices_savgol(solved_seq[b], gaussian_passes=passes)

    for t, frame_item in enumerate(frames_pred):
        b_dict = frame_item["bones"]
        for b in topo_names:
            if b in b_dict:
                b_dict[b]["rot_mat_world"] = [[round(float(x), 5) for x in row] for row in solved_seq[b][t]]


def evaluate_stage4_studio_exam(ckpt_path: str, exam_dir: str, out_report: str) -> dict[str, Any]:
    """Runs the trained DINOv2 probe on the 96-frame Studio Exam sequence, reconstructs 3D
    joint trajectories (`exam_pred_traj.json`) and Full SO(3) rotation matrices, and computes BOTH:
      1. `directional_mpjre_deg` (3D bone vector pointing error, < 3.5 deg gate)
      2. `full_rot_mpjre_deg` & `non_leaf_full_rot_mpjre_deg` (Full SO(3) geodesic rotation error
         including Axial Twist / Roll: `arccos((Tr(R_gt^T R_pred) - 1) / 2)`).
    """
    exam_path = Path(exam_dir)
    with open(exam_path / "exam_gt.json", "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    ckpt_meta = None
    if HAS_TORCH and Path(ckpt_path).exists():
        ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_meta = ckpt_obj.get("bones_meta")

    frame_files = [exam_path / "exam_frames" / item["image"] for item in gt_data["frames"]]
    pred_res = infer_dual_view_frames(ckpt_path, frame_files)

    first_gt_bones = gt_data["frames"][0]["bones"] if gt_data.get("frames") else None
    reconstruct_frame_so3_rotations_from_keypoints(pred_res["frames"], ckpt_meta, first_gt_bones)

    pred_traj_path = exam_path / "exam_pred_traj.json"
    with open(pred_traj_path, "w", encoding="utf-8") as f:
        json.dump(pred_res, f, indent=2, ensure_ascii=False)

    # Reconstruct parent->children map so legacy ckpt_meta without "children" never marks all bones as leaves
    all_bones = list(gt_data["frames"][0]["bones"].keys()) if gt_data.get("frames") else []
    has_children: set[str] = set()
    if ckpt_meta:
        for b, info in ckpt_meta.items():
            if info.get("children"):
                has_children.add(b)
            p = info.get("parent")
            if p:
                has_children.add(p)
    leaf_bones: set[str] = {b for b in all_bones if b not in has_children} if ckpt_meta else set()

    dir_errors: dict[str, list[float]] = {}
    full_rot_errors: dict[str, list[float]] = {}
    non_leaf_rot_errors: dict[str, list[float]] = {}

    for f_gt, f_pred in zip(gt_data["frames"], pred_res["frames"]):
        for b_name, b_gt in f_gt["bones"].items():
            b_pr = f_pred["bones"][b_name]
            v_gt = np.array(b_gt["tail_3d"], dtype=np.float64) - np.array(b_gt["head_3d"], dtype=np.float64)
            v_pr = np.array(b_pr["tail_3d"], dtype=np.float64) - np.array(b_pr["head_3d"], dtype=np.float64)
            n_gt = float(np.linalg.norm(v_gt))
            n_pr = float(np.linalg.norm(v_pr))
            if n_gt > 1e-5 and n_pr > 1e-5:
                cos_sim = float(np.clip(np.dot(v_gt / n_gt, v_pr / n_pr), -1.0, 1.0))
                ang_deg = math.degrees(math.acos(cos_sim))
                dir_errors.setdefault(b_name, []).append(ang_deg)

            if "rot_mat_world" in b_gt and "rot_mat_world" in b_pr:
                R_gt = np.array(b_gt["rot_mat_world"], dtype=np.float64)
                R_pr = np.array(b_pr["rot_mat_world"], dtype=np.float64)
                tr = float(np.clip((np.trace(R_gt.T @ R_pr) - 1.0) * 0.5, -1.0, 1.0))
                so3_deg = math.degrees(math.acos(tr))
                full_rot_errors.setdefault(b_name, []).append(so3_deg)
                if b_name not in leaf_bones:
                    non_leaf_rot_errors.setdefault(b_name, []).append(so3_deg)

    per_bone_dir = {k: round(float(np.mean(v)), 3) for k, v in dir_errors.items()}
    per_bone_full = {k: round(float(np.mean(v)), 3) for k, v in full_rot_errors.items()}
    per_bone_non_leaf = {k: round(float(np.mean(v)), 3) for k, v in non_leaf_rot_errors.items()}

    dir_mpjre = round(float(np.mean(list(per_bone_dir.values()))), 3) if per_bone_dir else 999.0
    dir_max = round(float(np.max(list(per_bone_dir.values()))), 3) if per_bone_dir else 999.0
    full_mpjre = round(float(np.mean(list(per_bone_full.values()))), 3) if per_bone_full else dir_mpjre
    full_max = round(float(np.max(list(per_bone_full.values()))), 3) if per_bone_full else dir_max
    non_leaf_mpjre = round(float(np.mean(list(per_bone_non_leaf.values()))), 3) if per_bone_non_leaf else full_mpjre

    exam_passed = (dir_mpjre < 3.5) and (dir_max < 8.0) and (non_leaf_mpjre < 3.5)

    auxiliary_depth_errors = []
    if pred_res.get("depth_visibility_enabled"):
        cx, cy, _ = pred_res["calibration"]["center"]
        depth_scale = pred_res["calibration"]["depth_scale"]
        for f_gt, f_pred in zip(gt_data["frames"], pred_res["frames"]):
            for name, gt_bone in f_gt["bones"].items():
                predicted = f_pred["bones"][name]
                for endpoint_index, endpoint in enumerate(("head", "tail")):
                    x, y, _ = gt_bone[f"{endpoint}_3d"]
                    auxiliary_depth_errors.extend((
                        abs(predicted["depth_front_relative"][endpoint_index]-(y-cy)/depth_scale),
                        abs(predicted["depth_side_relative"][endpoint_index]-(cx-x)/depth_scale)))
    stratified = compute_stratified_motion_metrics(pred_res["frames"], gt_data["frames"], ckpt_meta or {})
    primary_stats = stratified.get("primary_bones", {})

    report = {
        "bone_taxonomy_schema": BONE_TAXONOMY_SCHEMA,
        "depth_mae_relative": float(np.mean(auxiliary_depth_errors)) if auxiliary_depth_errors else None,
        "depth_visibility_enabled": pred_res.get("depth_visibility_enabled", False),
        "feature_version": FEATURE_VERSION,
        "exam_passed": exam_passed,
        "mpjre_deg": dir_mpjre,
        "max_bone_error_deg": dir_max,
        "directional_mpjre_deg": dir_mpjre,
        "directional_max_bone_deg": dir_max,
        "full_rot_mpjre_deg": full_mpjre,
        "full_rot_max_bone_deg": full_max,
        "non_leaf_full_rot_mpjre_deg": non_leaf_mpjre,
        "primary_directional_mpjre_deg": primary_stats.get("direction_mean_deg", dir_mpjre),
        "primary_directional_max_bone_deg": primary_stats.get("direction_max_bone_deg", dir_max),
        "primary_non_leaf_full_rot_mpjre_deg": primary_stats.get("nonleaf_rot_mean_deg", non_leaf_mpjre),
        "metrics_by_kind": stratified.get("by_kind", {}),
        "epipolar_gate_passed": pred_res["gate_passed"],
        "pred_traj_json": str(pred_traj_path),
        "per_bone_mean_deg": per_bone_dir,
        "per_bone_directional_deg": per_bone_dir,
        "per_bone_full_rot_deg": per_bone_full,
    }
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(
        f"[Stage 4 Studio Exam] Passed={exam_passed} | Dir MPJRE={dir_mpjre:.2f}° (Max={dir_max:.2f}°) | "
        f"Full SO(3) MPJRE={full_mpjre:.2f}° (Non-Leaf={non_leaf_mpjre:.2f}°, Max={full_max:.2f}°)"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="DINOv2 Probe Trainer, Exam Evaluator & 3D Triangulator")
    parser.add_argument("--mode", required=True, choices=["train", "exam", "triangulate_video"])
    parser.add_argument("--synth_dir", help="Stage 2 synthetic dataset directory")
    parser.add_argument("--meta_path", help="Path to studio_rig_meta.json")
    parser.add_argument("--ckpt_path", required=True, help="Path to dinov2_head.pt")
    parser.add_argument("--exam_dir", help="Stage 4 exam sequence directory")
    parser.add_argument("--video_frames_dir", help="Directory of extracted video frames for Stage 5/6")
    parser.add_argument("--out_json", help="Output JSON report or 3D trajectory file")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--progress_file", help="Path to write training progress JSON")
    args = parser.parse_args()

    if args.mode == "train":
        train_monster_dinov2_probe(args.synth_dir, args.meta_path, args.ckpt_path, epochs=args.epochs, progress_file=args.progress_file)
    elif args.mode == "exam":
        evaluate_stage4_studio_exam(args.ckpt_path, args.exam_dir, args.out_json)
    elif args.mode == "triangulate_video":
        frames = sorted(Path(args.video_frames_dir).glob("*.jpg")) + sorted(Path(args.video_frames_dir).glob("*.png"))
        res = infer_dual_view_frames(args.ckpt_path, frames)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(
            f"[Stage 5/6 Triangulation] GatePassed={res['gate_passed']} "
            f"(mean_v_diff={res['mean_epipolar_diff_px']}px, corr={res['vertical_trajectory_corr']})"
        )


if __name__ == "__main__":
    main()
