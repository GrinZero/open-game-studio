"""CPU tests with a tiny backbone: gradients, warm-up, grouped splits and reload parity."""

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from pose_finetuning import (
    TrainingConfig,
    configure_backbone,
    load_probe_weights,
    split_pose_indices,
)

try:
    import torch
    from torch import nn
    import dinov2_pose_trainer as trainer
except ImportError:
    torch = None


class TrainingConfigurationTests(unittest.TestCase):
    def test_colab_defaults_reach_the_eight_block_training_configuration(self):
        import ast
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "scripts/colab_runtime.py"
        spec = importlib.util.spec_from_file_location("v3_colab_defaults_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.dict(os.environ, {}),
            patch.object(module, "stop_previous_worker"),
            patch.object(module, "WORKER_MODULES", ()),
            patch.object(module.Path, "mkdir"),
        ):
            module.configure(drive_root=temporary, work_root=temporary)
            config = TrainingConfig.from_environment()
            self.assertEqual(config.unfreeze_last_blocks, 8)
            self.assertEqual(config.backbone_lr, 1e-4)
        notebook = json.loads((path.parents[1] / "notebooks/colab.ipynb").read_text())
        constants = {}
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.parse("".join(cell["source"])).body:
                if isinstance(node, ast.Assign) and isinstance(
                    node.value, ast.Constant
                ):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            constants[target.id] = node.value.value
        self.assertEqual(constants["UNFREEZE_LAST_BLOCKS"], 8)
        self.assertEqual(constants["BACKBONE_LR"], 1e-4)

    def test_pose_views_never_leak_and_split_is_reproducible(self):
        for views in (2, 3):
            train, val = split_pose_indices(100, views, 0.05, 42)
            self.assertEqual((train, val), split_pose_indices(100, views, 0.05, 42))
            self.assertEqual(len(val), 5 * views)
            self.assertFalse({i // views for i in train} & {i // views for i in val})
            self.assertEqual(sorted(train + val), list(range(100 * views)))

    def test_settings_fail_closed_and_environment_can_select_baseline(self):
        for values in (
            {"mode": "full"},
            {"backbone_lr": float("nan")},
            {"warmup_epochs": -1},
            {"batch_size": 0},
            {"seed": True},
            {"gradient_checkpointing": "false"},
            {"validation_fraction": 0},
            {"unexpected": 1},
        ):
            with (
                self.subTest(values=values),
                self.assertRaises((ValueError, TypeError)),
            ):
                TrainingConfig.from_environment(values)
        with patch.dict(
            os.environ,
            {
                "MONSTER_TRAIN_MODE": "frozen",
                "MONSTER_TRAIN_GRADIENT_CHECKPOINTING": "0",
            },
        ):
            config = TrainingConfig.from_environment()
            self.assertEqual(config.mode, "frozen")
            self.assertFalse(config.gradient_checkpointing)


if torch is not None:

    class TinyBackbone(nn.Module):
        embed_dim = 16

        def __init__(self):
            super().__init__()
            self.patch_embed = nn.Conv2d(3, 16, 14, stride=14)
            self.blocks = nn.ModuleList(
                [nn.Sequential(nn.Linear(16, 16), nn.GELU()) for _ in range(4)]
            )
            self.norm = nn.LayerNorm(16)

        def get_intermediate_layers(self, x, n, norm=True):
            x = self.patch_embed(x).flatten(2).transpose(1, 2)
            result = []
            for i, block in enumerate(self.blocks):
                x = x + block(x)
                if i in n:
                    result.append(self.norm(x) if norm else x)
            return result

        def forward_features(self, x):
            return {"x_norm_patchtokens": self.get_intermediate_layers(x, [3])[-1]}

    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self, *args, **kwargs):
            self.samples = list(range(5))
            self.num_views = 2
            self.view_order = ("front", "side")
            self.calib = {"panel_width": 28, "panel_height": 28}
            self.dense_surface = False

        def __len__(self):
            return 10

        def __getitem__(self, idx):
            image = torch.rand(3, 28, 28, generator=torch.Generator().manual_seed(idx))
            return image, torch.tensor([[0.25, 0.5], [0.65, 0.75]])


@unittest.skipIf(
    torch is None,
    "Optional PyTorch CPU checks; standard library migration checks still run",
)
class FineTuningGradientTests(unittest.TestCase):
    def test_twelve_block_structure_freezes_embeddings_and_first_four_blocks(self):
        backbone = TinyBackbone()
        backbone.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(16, 16), nn.GELU()) for _ in range(12)]
        )
        parameters = configure_backbone(backbone, TrainingConfig())
        backbone.train()
        output = backbone.get_intermediate_layers(torch.randn(1, 3, 28, 28), [11])[-1]
        (output.square().mean() + output[..., 0].mean()).backward()
        self.assertTrue(
            all(
                p.grad is None and not p.requires_grad
                for p in backbone.patch_embed.parameters()
            )
        )
        for i, block in enumerate(backbone.blocks):
            for p in block.parameters():
                self.assertEqual(p.requires_grad, i >= 4)
                self.assertEqual(p.grad is None, i < 4)
        self.assertEqual(
            {id(p) for p in parameters},
            {id(p) for p in backbone.parameters() if p.requires_grad},
        )

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)

    def test_only_suffix_updates_in_plain_and_dense_paths_and_reload_matches(self):
        for dense in (False, True):
            with self.subTest(dense=dense):
                model = trainer.MonsterDINOv2PoseProbe(
                    2,
                    custom_backbone=TinyBackbone(),
                    enable_depth_visibility=True,
                    enable_dense_surface=dense,
                )
                config = TrainingConfig(unfreeze_last_blocks=2, warmup_epochs=0)
                params = configure_backbone(model.backbone, config)
                before = copy.deepcopy(model.backbone.state_dict())
                model.train()
                optimizer = torch.optim.SGD(params, lr=0.1)
                image = torch.rand(2, 3, 28, 28)
                _, _, logits, aux = model(image, torch.tensor([0, 1]), return_aux=True)
                loss = logits.square().mean() + aux["depth"].square().mean()
                if dense:
                    loss += aux["dense_rest_xyz"].mean()
                loss.backward()
                self.assertTrue(
                    all(p.grad is None for p in model.backbone.blocks[0].parameters())
                )
                self.assertTrue(
                    all(p.grad is None for p in model.backbone.patch_embed.parameters())
                )
                self.assertTrue(
                    any(
                        p.grad is not None and bool(p.grad.abs().sum() > 0)
                        for p in params
                    )
                )
                optimizer.step()
                after = model.backbone.state_dict()
                self.assertTrue(
                    torch.equal(before["blocks.0.0.weight"], after["blocks.0.0.weight"])
                )
                self.assertFalse(
                    torch.equal(before["blocks.3.0.weight"], after["blocks.3.0.weight"])
                )
                checkpoint = {
                    "backbone_finetuned": True,
                    "backbone_state_dict": after,
                    "head_state_dict": model.head.state_dict(),
                    "aux_state_dict": model.aux_head.state_dict(),
                }
                if dense:
                    for name in (
                        "dino_multi_fuse",
                        "rgb_detail_branch",
                        "rgb_kp_refine",
                        "dense_head",
                    ):
                        checkpoint[name + "_state_dict"] = getattr(
                            model, name
                        ).state_dict()
                restored = trainer.MonsterDINOv2PoseProbe(
                    2,
                    custom_backbone=TinyBackbone(),
                    enable_depth_visibility=True,
                    enable_dense_surface=dense,
                )
                load_probe_weights(restored, checkpoint)
                model.eval()
                restored.eval()
                with torch.inference_mode():
                    torch.testing.assert_close(
                        model(image)[0], restored(image)[0], rtol=1e-6, atol=1e-7
                    )
                del checkpoint["backbone_state_dict"]
                with self.assertRaisesRegex(ValueError, "missing backbone"):
                    load_probe_weights(restored, checkpoint)

    def test_frozen_mode_keeps_backbone_and_loads_legacy_head(self):
        model = trainer.MonsterDINOv2PoseProbe(2, custom_backbone=TinyBackbone())
        self.assertEqual(
            configure_backbone(model.backbone, TrainingConfig(mode="frozen")), []
        )
        model(torch.rand(2, 3, 28, 28))[2].sum().backward()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        load_probe_weights(model, {"head_state_dict": model.head.state_dict()})

    def test_training_loop_warmup_checkpoint_and_initialization(self):
        model_class = trainer.MonsterDINOv2PoseProbe
        created = []

        def factory(**kwargs):
            result = model_class(**kwargs, custom_backbone=TinyBackbone())
            created.append(result)
            return result

        phases = []

        def progress(_targets, payload):
            phases.append(
                (
                    payload["epoch"],
                    payload["backbone_adapting"],
                    any(p.requires_grad for p in created[-1].backbone.parameters()),
                )
            )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(trainer, "DualPanelMonsterDataset", TinyDataset),
            patch.object(trainer, "MonsterDINOv2PoseProbe", side_effect=factory),
            patch.object(trainer, "features_enabled", return_value=False),
            patch.object(trainer, "_query_telemetry", return_value={}),
            patch.object(trainer, "_write_progress_files", side_effect=progress),
            patch.dict(os.environ, {"MONSTER_DATALOADER_WORKERS": "0"}),
        ):
            root = Path(tmp)
            meta = root / "meta.json"
            meta.write_text(json.dumps({"bones": {"joint": {}}}))
            (root / "keypoints_gt.json").write_text("{}")
            ckpt = root / "probe.pt"
            ckpt.write_bytes(b"original checkpoint")
            trainer.train_monster_dinov2_probe(
                tmp,
                str(meta),
                str(ckpt),
                epochs=2,
                training_config={"unfreeze_last_blocks": 2, "accumulation_steps": 3},
            )
            saved = torch.load(ckpt, weights_only=False)
            self.assertTrue(saved["backbone_finetuned"])
            self.assertEqual(saved["selected_epoch"], 2)
            self.assertIn((1, False, False), phases)
            self.assertIn((2, True, True), phases)
            self.assertEqual(
                next(root.glob("probe.before_*.pt")).read_bytes(),
                b"original checkpoint",
            )
            self.assertTrue(ckpt.with_suffix(".training.json").is_file())
            baseline = root / "baseline.pt"
            trainer.train_monster_dinov2_probe(
                tmp,
                str(meta),
                str(baseline),
                epochs=1,
                init_checkpoint=str(ckpt),
                training_config={"mode": "frozen"},
            )
            baseline_saved = torch.load(baseline, weights_only=False)
            # A frozen run initialized from adapted weights must retain those features too.
            self.assertTrue(baseline_saved["backbone_finetuned"])
            self.assertTrue(
                all(
                    torch.equal(
                        saved["backbone_state_dict"][k],
                        baseline_saved["backbone_state_dict"][k],
                    )
                    for k in saved["backbone_state_dict"]
                )
            )
            previous = ckpt.read_bytes()

            def invalid_item(_self, _idx):
                return torch.full((3, 28, 28), float("nan")), torch.zeros(2, 2)

            with patch.object(TinyDataset, "__getitem__", invalid_item):
                with self.assertRaisesRegex(RuntimeError, "Non-finite pose loss"):
                    trainer.train_monster_dinov2_probe(
                        tmp,
                        str(meta),
                        str(ckpt),
                        epochs=2,
                        training_config={"unfreeze_last_blocks": 2},
                    )
            self.assertEqual(ckpt.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
