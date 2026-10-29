"""Reproducible partial DINOv2 adaptation and checkpoint compatibility helpers."""
from dataclasses import asdict, dataclass, fields
import math
import os
import random


@dataclass(frozen=True)
class TrainingConfig:
    mode: str = "partial"
    unfreeze_last_blocks: int = 8
    backbone_lr: float = 1e-4
    warmup_epochs: int = 1
    batch_size: int = 2
    accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    validation_fraction: float = 0.05
    seed: int = 42
    max_grad_norm: float = 1.0

    def __post_init__(self):
        if self.mode not in {"frozen", "partial"}:
            raise ValueError("mode must be frozen or partial")
        for name in ("unfreeze_last_blocks", "batch_size", "accumulation_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.warmup_epochs) is not int or self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be a nonnegative integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be a boolean")
        for name in ("backbone_lr", "max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.validation_fraction, bool) or not isinstance(self.validation_fraction, (int, float)) or not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")

    @classmethod
    def from_environment(cls, overrides=None):
        values = {}
        for field in fields(cls):
            raw = os.environ.get("MONSTER_TRAIN_" + field.name.upper())
            if raw is None:
                continue
            if field.type is bool:
                if raw.lower() not in {"true", "false", "1", "0"}:
                    raise ValueError(f"Invalid boolean for {field.name}")
                values[field.name] = raw.lower() in {"true", "1"}
            else:
                values[field.name] = field.type(raw)
        if overrides is not None:
            if not isinstance(overrides, dict):
                raise ValueError("training_config must be an object")
            unknown = set(overrides) - {field.name for field in fields(cls)}
            if unknown:
                raise ValueError(f"Unknown training settings: {sorted(unknown)}")
            values.update(overrides)
        return cls(**values)

    def as_dict(self):
        return asdict(self)


def split_pose_indices(num_poses, num_views, validation_fraction, seed):
    """Keep all views of a pose together; never use the G3 exam to select weights."""
    if num_poses < 2 or num_views < 1:
        raise ValueError("Training needs at least two poses and one view")
    poses = list(range(num_poses))
    random.Random(seed).shuffle(poses)
    count = min(num_poses - 1, max(1, round(num_poses * validation_fraction)))
    def panels(items):
        return [pose * num_views + view for pose in sorted(items) for view in range(num_views)]
    return panels(poses[count:]), panels(poses[:count])


def configure_backbone(backbone, config):
    """Freeze the prefix, patch/token embeddings, and registers; adapt the suffix + norm."""
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone.eval()
    if config.mode == "frozen":
        return []
    blocks = list(getattr(backbone, "blocks", ()))
    norm = getattr(backbone, "norm", None)
    if not blocks and hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        blocks = list(backbone.encoder.layer)
        norm = getattr(backbone, "layernorm", norm)
    if not blocks or config.unfreeze_last_blocks > len(blocks):
        raise ValueError("unfreeze_last_blocks exceeds the backbone block count")
    selected = blocks[-config.unfreeze_last_blocks:]
    if config.gradient_checkpointing:
        from torch.utils.checkpoint import checkpoint
        for block in selected:
            if getattr(block, "_monster_checkpointed", False):
                continue
            original_forward = block.forward
            def forward(*args, _original=original_forward, _block=block, **kwargs):
                import torch
                if _block.training and torch.is_grad_enabled():
                    return checkpoint(_original, *args, use_reentrant=False, **kwargs)
                return _original(*args, **kwargs)
            block.forward = forward
            block._monster_checkpointed = True
    if norm is not None:
        selected.append(norm)
    for module in selected:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return [p for p in backbone.parameters() if p.requires_grad]


def load_probe_weights(model, checkpoint):
    """Load adapted features as well as heads; legacy head-only checkpoints remain supported."""
    adapted = checkpoint.get("backbone_finetuned", False)
    if adapted and "backbone_state_dict" not in checkpoint:
        raise ValueError("Fine-tuned checkpoint is missing backbone_state_dict")
    if "backbone_state_dict" in checkpoint:
        model.backbone.load_state_dict(checkpoint["backbone_state_dict"], strict=True)
    model.head.load_state_dict(checkpoint["head_state_dict"])
    if model.enable_depth_visibility:
        model.aux_head.load_state_dict(checkpoint["aux_state_dict"])
    if getattr(model, "reader_v3", False) and (
        "reader_heads_state_dict" in checkpoint or "reader_contract" in checkpoint
    ):
        if checkpoint.get("reader_contract") != "monster-reader-v3":
            raise ValueError("V3 Reader weights require a v3 checkpoint; regenerate labels and retrain")
        rh_state = dict(checkpoint["reader_heads_state_dict"])
        if (
            "dhead.weight" in rh_state
            and rh_state["dhead.weight"].shape[0] == 1
            and model.reader_heads.dhead.weight.shape[0] > 1
        ):
            k_ch = model.reader_heads.dhead.weight.shape[0]
            rh_state["dhead.weight"] = rh_state["dhead.weight"].expand(k_ch, -1, -1, -1).clone()
            if "dhead.bias" in rh_state and rh_state["dhead.bias"].shape[0] == 1:
                rh_state["dhead.bias"] = rh_state["dhead.bias"].expand(k_ch).clone()
        model.reader_heads.load_state_dict(rh_state, strict=True)
    if model.enable_dense_surface:
        for name in ("dino_multi_fuse", "rgb_detail_branch", "rgb_kp_refine", "dense_head"):
            getattr(model, name).load_state_dict(checkpoint[name + "_state_dict"])
