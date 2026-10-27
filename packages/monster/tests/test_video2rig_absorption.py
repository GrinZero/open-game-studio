import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worker"))

from dinov2_pose_trainer import MonsterDINOv2PoseProbe
from pose_features import (
    backbone_mode,
    pipeline_feature_flags,
    reader_crops_enabled,
    reader_dense_enabled,
)
from pose_finetuning import configure_backbone, load_probe_weights
from reader_v3 import (
    ReaderHeads,
    Video2RigReaderNet,
    corrupt_context,
    crop_sides,
    dense_reader_losses,
    end_groups,
    joints_from_dense_fields,
    rasterize_dense_fields,
    read_crops,
    reader_losses,
    sample_maps,
    sight_depth_targets,
    smear_limbs_capsules_gpu,
)
from reading_pipeline import (
    downweight_contradicting_far_limb,
    downweight_flipped_limbs,
    downweight_overlapping_twins,
    fix_twins_viterbi,
    jump_pass_constant_velocity,
)
from reprojection_solver import (
    RigModel,
    _apply_body_picture_weights,
    axis_roles,
    find_hinges,
    fit_clip_2d_reprojection,
    robust_cauchy_loss,
)
from software_rasterizer import (
    rasterize_triangles_torch,
    render_fast_review_frames,
    render_mesh_software,
    vertex_normals,
)
from studio_camera import cameras_from_calibration


def _test_cameras(width=512, height=512, scale=2.2, center=(0.0, 0.0, 0.8)):
    cams = cameras_from_calibration(
        {
            "center": list(center),
            "panel_width": width,
            "panel_height": height,
            "ortho_scale": scale,
            "view_order": ["front", "left"],
        }
    )
    return cams["front"], cams["left"]


class _DummyDINOv2Small(nn.Module):
    """Lightweight DINOv2-small (d=384, 12 blocks, patch=14) mock for unit tests."""

    def __init__(self, embed_dim: int = 384, depth: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        self.embeddings = nn.Conv2d(3, embed_dim, kernel_size=14, stride=14)
        self.blocks = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward_features(self, x: torch.Tensor):
        tokens = self.embeddings(x).flatten(2).transpose(1, 2)
        for blk in self.blocks:
            tokens = tokens + 0.01 * blk(tokens)
        tokens = self.norm(tokens)
        return {"x_norm_patchtokens": tokens}


def _build_test_rig():
    meta = {
        "Hips": {
            "parent": None,
            "kind": "root",
            "canonical_name": "Hips",
            "rest_head_world": [0.0, 0.0, 1.0],
            "rest_tail_world": [0.0, 0.0, 1.2],
        },
        "Spine": {
            "parent": "Hips",
            "kind": "spine",
            "canonical_name": "Spine",
            "rest_head_world": [0.0, 0.0, 1.2],
            "rest_tail_world": [0.0, 0.0, 1.6],
            "limits_rad": [[-0.5, 0.5], [-0.4, 0.4], [-0.4, 0.4]],
        },
        "Clavicle.L": {
            "parent": "Spine",
            "kind": "limb",
            "canonical_name": "Clavicle.L",
            "rest_head_world": [0.05, 0.0, 1.55],
            "rest_tail_world": [0.20, 0.0, 1.55],
            "limits_rad": [[-0.4, 0.4], [-0.3, 0.3], [-0.4, 0.4]],
        },
        "UpperArm.L": {
            "parent": "Clavicle.L",
            "kind": "limb",
            "canonical_name": "UpperArm.L",
            "rest_head_world": [0.20, 0.0, 1.55],
            "rest_tail_world": [0.45, 0.0, 1.35],
            "limits_rad": [[-1.2, 1.2], [-0.8, 0.8], [-1.2, 1.2]],
        },
        "Forearm.L": {
            "parent": "UpperArm.L",
            "kind": "limb",
            "canonical_name": "Forearm.L",
            "rest_head_world": [0.45, 0.0, 1.35],
            "rest_tail_world": [0.68, -0.05, 1.15],
            "limits_rad": [[-2.0, 2.0], [-0.5, 0.5], [-1.5, 1.5]],
        },
        "Hand.L": {
            "parent": "Forearm.L",
            "kind": "limb",
            "canonical_name": "Hand.L",
            "rest_head_world": [0.68, -0.05, 1.15],
            "rest_tail_world": [0.78, -0.08, 1.05],
            "limits_rad": [[-0.8, 0.8], [-0.5, 0.5], [-0.8, 0.8]],
        },
        "Thigh.L": {
            "parent": "Hips",
            "kind": "leg",
            "canonical_name": "Thigh.L",
            "rest_head_world": [0.15, 0.0, 1.0],
            "rest_tail_world": [0.15, 0.02, 0.55],
            "limits_rad": [[-1.4, 1.4], [-0.6, 0.6], [-0.8, 0.8]],
        },
        "Shin.L": {
            "parent": "Thigh.L",
            "kind": "leg",
            "canonical_name": "Shin.L",
            "rest_head_world": [0.15, 0.02, 0.55],
            "rest_tail_world": [0.15, 0.08, 0.12],
            "limits_rad": [[-2.2, 2.2], [-0.4, 0.4], [-0.5, 0.5]],
        },
        "Foot.L": {
            "parent": "Shin.L",
            "kind": "leg",
            "canonical_name": "Foot.L",
            "rest_head_world": [0.15, 0.08, 0.12],
            "rest_tail_world": [0.15, -0.10, 0.02],
            "limits_rad": [[-0.8, 0.8], [-0.4, 0.4], [-0.5, 0.5]],
        },
        "Toe.L": {
            "parent": "Foot.L",
            "kind": "leg",
            "canonical_name": "Toe.L",
            "rest_head_world": [0.15, -0.10, 0.02],
            "rest_tail_world": [0.15, -0.18, 0.01],
            "limits_rad": [[-0.6, 0.6], [-0.2, 0.2], [-0.3, 0.3]],
        },
        "Thigh.R": {
            "parent": "Hips",
            "kind": "leg",
            "canonical_name": "Thigh.R",
            "rest_head_world": [-0.15, 0.0, 1.0],
            "rest_tail_world": [-0.15, 0.02, 0.55],
            "limits_rad": [[-1.4, 1.4], [-0.6, 0.6], [-0.8, 0.8]],
        },
        "Shin.R": {
            "parent": "Thigh.R",
            "kind": "leg",
            "canonical_name": "Shin.R",
            "rest_head_world": [-0.15, 0.02, 0.55],
            "rest_tail_world": [-0.15, 0.08, 0.12],
            "limits_rad": [[-2.2, 2.2], [-0.4, 0.4], [-0.5, 0.5]],
        },
        "Foot.R": {
            "parent": "Shin.R",
            "kind": "leg",
            "canonical_name": "Foot.R",
            "rest_head_world": [-0.15, 0.08, 0.12],
            "rest_tail_world": [-0.15, -0.10, 0.02],
            "limits_rad": [[-0.8, 0.8], [-0.4, 0.4], [-0.5, 0.5]],
        },
        "Tail.001": {
            "parent": "Hips",
            "kind": "loose",
            "canonical_name": "Tail.001",
            "rest_head_world": [0.0, 0.10, 0.98],
            "rest_tail_world": [0.0, 0.40, 0.85],
            "limits_rad": [[-0.8, 0.8], [-0.6, 0.6], [-0.8, 0.8]],
        },
    }
    axis_roles(meta)
    find_hinges(meta)
    return RigModel(meta)


class Video2RigAbsorptionTests(unittest.TestCase):
    def test_item1_dinov2_small_reader_net_and_probe(self):
        from dinov2_pose_trainer import resolve_checkpoint_backbone_name

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MONSTER_BACKBONE", None)
            os.environ.pop("MONSTER_PIPELINE_MODE", None)
            self.assertEqual(backbone_mode(), "dinov2-small")
            flags = pipeline_feature_flags()
            self.assertEqual(flags["backbone_mode"], "dinov2-small")
            self.assertEqual(flags["decode_mode"], "parabola")
            self.assertTrue(flags["no_cut_tracks"])
            self.assertTrue(flags["twoway_jump"])
            self.assertTrue(flags["panel_drift"])
            # Legacy 768-dim checkpoint without backbone_name auto-resolves to dinov2_vitb14_reg
            self.assertEqual(
                resolve_checkpoint_backbone_name({"head_state_dict": {"0.weight": torch.zeros((768, 256, 4, 4))}}),
                "dinov2_vitb14_reg",
            )
            self.assertEqual(
                resolve_checkpoint_backbone_name({"head_state_dict": {"0.weight": torch.zeros((384, 256, 4, 4))}}),
                "dinov2-small",
            )

        dummy_backbone = _DummyDINOv2Small(embed_dim=384, depth=12)
        net = Video2RigReaderNet(
            num_keypoints=10,
            dense_bones=5,
            custom_backbone=dummy_backbone,
            depth=True,
        )
        # Embeddings and first 4 blocks must be frozen (FROZEN=4), blocks 4..11 trainable
        for p in net.backbone.embeddings.parameters():
            self.assertFalse(p.requires_grad)
        for i in range(4):
            for p in net.backbone.blocks[i].parameters():
                self.assertFalse(p.requires_grad)
        for i in range(4, 12):
            for p in net.backbone.blocks[i].parameters():
                self.assertTrue(p.requires_grad)

        img = torch.zeros((2, 3, 448, 448), dtype=torch.float32)
        ctx = torch.zeros((2, 20, 112, 112), dtype=torch.float32)
        vid = torch.tensor([0, 1], dtype=torch.long)
        coords, conf, logits, aux = net(img, view_ids=vid, return_aux=True, context=ctx)
        self.assertEqual(tuple(logits.shape), (2, 10, 112, 112))
        self.assertEqual(tuple(coords.shape), (2, 10, 2))
        self.assertEqual(tuple(aux["relative_depth"].shape), (2, 10, 112, 112))
        self.assertEqual(tuple(aux["point_depth"].shape), (2, 10))
        self.assertEqual(tuple(aux["part_logits"].shape), (2, 6, 112, 112))
        self.assertEqual(tuple(aux["local_fields"].shape), (2, 3, 112, 112))

        probe = MonsterDINOv2PoseProbe(
            num_keypoints=10,
            custom_backbone=_DummyDINOv2Small(embed_dim=384, depth=12),
            backbone_name="dinov2-small",
            enable_depth_visibility=True,
            enable_dense_surface=True,
            num_views=2,
            reader_v3=True,
            num_bones=5,
        )
        self.assertEqual(probe.embed_dim, 384)
        _, _, _, p_aux = probe(img, view_ids=vid, return_aux=True, context=ctx)
        self.assertEqual(tuple(p_aux["point_depth"].shape), (2, 10))

    def test_item2_k_channel_depth_height_normalized_dense_and_modular_crops(self):
        self.assertFalse(reader_crops_enabled())
        self.assertFalse(reader_dense_enabled())

        # K-channel depth head & sample_maps & sight_depth_targets
        rig = _build_test_rig()
        X0 = rig.fk(rig.start_params())[0]
        cam, _ = _test_cameras(width=512, height=512, scale=2.2, center=(0.0, 0.0, 0.8))
        d_targets = sight_depth_targets(rig, cam, X0)
        self.assertEqual(d_targets.shape, (len(rig.point_names),))
        self.assertTrue(np.isfinite(d_targets).all())

        # Legacy 1-channel dhead weight expansion into K-channel ReaderHeads
        probe = MonsterDINOv2PoseProbe(
            num_keypoints=6,
            custom_backbone=_DummyDINOv2Small(embed_dim=384, depth=12),
            backbone_name="dinov2-small",
            reader_v3=True,
            num_bones=3,
        )
        rh_sd = probe.reader_heads.state_dict()
        rh_sd["dhead.weight"] = rh_sd["dhead.weight"][:1].clone()
        rh_sd["dhead.bias"] = rh_sd["dhead.bias"][:1].clone()
        load_probe_weights(
            probe,
            {
                "head_state_dict": probe.head.state_dict(),
                "reader_heads_state_dict": rh_sd,
                "reader_contract": "monster-reader-v3",
            },
        )
        self.assertEqual(tuple(probe.reader_heads.dhead.weight.shape)[0], 6)

        # Height-H normalized dense fields & round-trip joint reconstruction
        meta = {
            "Bone0": {
                "parent": None,
                "kind": "root",
                "rest_head_world": [0.0, 0.0, 0.0],
                "rest_tail_world": [0.0, 0.0, 0.4],
            }
        }
        rig_simple = RigModel(meta)
        rig_simple.height = 2.0
        verts = np.array(
            [
                [-0.2, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.2, 0.0, 0.4],
                [-0.2, 0.0, 0.4],
            ],
            dtype=float,
        )
        tris = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)
        skin = {
            "bone_names": ["Bone0"],
            "rest_verts": verts,
            "triangles": tris,
            "skin_indices": np.zeros((4, 1), dtype=int),
            "skin_weights": np.ones((4, 1), dtype=float),
            "rest_heads": np.array([[0.0, 0.0, 0.0]]),
            "rest_R": np.array([rig_simple.R["Bone0"]]),
            "rest_lengths": np.array([0.4]),
            "height": 2.0,
        }
        dense_fields = rasterize_dense_fields(
            verts, skin, cam, grid=64, local_norm="height"
        )
        part = dense_fields["part"]
        local = dense_fields["local"]
        depth = dense_fields["depth"]
        fg = part == 1
        self.assertTrue(np.any(fg))
        # Since bone length is 0.4 and height is 2.0, max Y in height units is <= 0.4 / 2.0 = 0.2
        self.assertLessEqual(float(np.max(local[1, fg])), 0.25)
        part_logits = np.zeros((2, 64, 64), dtype=float)
        part_logits[0, ~fg] = 8.0
        part_logits[1, fg] = 8.0
        pts_h, conf_h = joints_from_dense_fields(
            part_logits,
            local,
            depth,
            cam,
            rig_simple,
            float(cam.depth(verts.mean(0))),
            local_norm="height",
        )
        self.assertTrue(np.allclose(pts_h[0], [0.0, 0.0, 0.0], atol=0.08))
        self.assertTrue(np.allclose(pts_h[1], [0.0, 0.0, 0.4], atol=0.08))

        # Modular crops helpers: end_groups, crop_sides, read_crops, dense_reader_losses
        groups = end_groups(rig)
        self.assertTrue(len(groups) >= 2)
        sides = np.asarray(crop_sides(rig, [cam], groups), dtype=float)
        self.assertEqual(sides.shape, (len(groups),))
        self.assertTrue(np.all((sides >= 72.0) & (sides <= 320.0)))

        d_loss = dense_reader_losses(
            torch.zeros((2, 2, 16, 16), requires_grad=True),
            torch.zeros((2, 3, 16, 16), requires_grad=True),
            torch.ones((2, 16, 16), dtype=torch.long),
            torch.zeros((2, 3, 16, 16)),
        )
        self.assertTrue(torch.isfinite(d_loss["total"]))
        self.assertIn("part", d_loss)
        self.assertIn("local", d_loss)

    def test_item3_corrupt_context_and_smear_limbs_capsules(self):
        gen = torch.Generator().manual_seed(123)
        uv = np.full((2, 8, 2), 224.0, dtype=np.float32)
        vis = np.ones((2, 8), dtype=np.float32)
        twins = np.array([[2, 3], [4, 5]], dtype=np.int64)
        # Run 60 trials to verify full dropout (~20%), twin swap (~25%), and far outliers (~5%)
        full_zeros = 0
        swaps = 0
        for _ in range(60):
            u_in = uv.copy()
            u_in[:, 2] = [100.0, 100.0]
            u_in[:, 3] = [300.0, 300.0]
            u_c, v_c = corrupt_context(
                u_in,
                vis,
                size=448,
                twins=twins,
                normalized=False,
                generator=gen,
                return_coords=True,
            )
            u_c_np = u_c.numpy()
            v_c_np = v_c.numpy()
            if np.all(v_c_np[0] == 0.0) or np.all(v_c_np[1] == 0.0):
                full_zeros += 1
            if np.linalg.norm(u_c_np[0, 2] - [300.0, 300.0]) < 35.0:
                swaps += 1
        self.assertGreater(full_zeros, 2)
        self.assertGreater(swaps, 2)

        # smear_limbs_capsules_gpu (requires >= 4 visible keypoints)
        imgs = torch.full((2, 3, 128, 128), 0.2, dtype=torch.float32)
        imgs[:, :, 50:78, 50:78] = 0.9
        uv_t = torch.tensor(
            [[[56.0, 56.0], [68.0, 62.0], [64.0, 74.0], [74.0, 70.0]]] * 2,
            dtype=torch.float32,
        )
        vis_t = torch.tensor([[1.0, 1.0, 1.0, 1.0]] * 2, dtype=torch.float32)
        smeared = smear_limbs_capsules_gpu(imgs, uv_t, vis_t, p=1.0)
        self.assertEqual(tuple(smeared.shape), (2, 3, 128, 128))
        self.assertTrue(torch.all((smeared >= 0.0) & (smeared <= 1.0)))
        self.assertGreater(float((smeared - imgs).abs().max()), 1e-3)

    def test_item4_twins_viterbi_jump_pass_overlapping_flipped_and_pick_panels(self):
        rig = _build_test_rig()
        cam_front, cam_side = _test_cameras(width=512, height=512, scale=2.2, center=(0.0, 0.0, 0.8))
        cameras = [cam_front, cam_side]
        T = 10
        K = len(rig.point_names)
        X0 = rig.fk(rig.start_params())[0]
        uv = np.zeros((T, 2, K, 2), dtype=float)
        conf = np.full((T, 2, K), 0.9, dtype=float)
        for t in range(T):
            for v, c in enumerate(cameras):
                uv[t, v] = c.project(X0)

        # 1) jump_pass_constant_velocity: inject a 1-frame left/right thigh swap at t=5 in view 0
        li = rig.point_names.index("Thigh.L.head")
        ri = rig.point_names.index("Thigh.R.head")
        uv_sub1 = uv.copy()
        for t in range(T):
            uv_sub1[t, 0, li] = [180.0 + 2.0 * t, 300.0]
            uv_sub1[t, 0, ri] = [320.0 + 2.0 * t, 300.0]
        uv_swapped = uv_sub1.copy()
        uv_swapped[5, 0, [li, ri]] = uv_swapped[5, 0, [ri, li]]
        fixed_uv, _ = jump_pass_constant_velocity(rig, uv_swapped, conf, twoway=True)
        self.assertTrue(np.allclose(fixed_uv[5, 0, li], uv_sub1[5, 0, li], atol=1e-4))
        self.assertTrue(np.allclose(fixed_uv[5, 0, ri], uv_sub1[5, 0, ri], atol=1e-4))

        # 2) downweight_overlapping_twins: standing legs with separated knees should NOT be downweighted
        uv_overlap = uv.copy()
        conf_overlap = conf.copy()
        # Make Foot.L.tail and Foot.R.tail overlap in view 1, while Foot.L.head & Foot.R.head are 12px apart (<= 3*near)
        f_lh = rig.point_names.index("Foot.L.head")
        f_rh = rig.point_names.index("Foot.R.head")
        lt = rig.point_names.index("Foot.L.tail")
        rt = rig.point_names.index("Foot.R.tail")
        uv_overlap[0, 1, lt] = [250.0, 450.0]
        uv_overlap[0, 1, rt] = [251.0, 450.0]
        uv_overlap[0, 1, f_lh] = [245.0, 420.0]
        uv_overlap[0, 1, f_rh] = [257.0, 420.0]
        c_protected = downweight_overlapping_twins(rig, cameras, uv_overlap, conf_overlap)
        self.assertAlmostEqual(float(c_protected[0, 1, lt]), 0.9, places=5)
        # Now separate the adjacent heads by 80px (> 3*near): tail collapse SHOULD be downweighted
        uv_overlap[0, 1, f_lh] = [200.0, 420.0]
        uv_overlap[0, 1, f_rh] = [290.0, 420.0]
        c_down = downweight_overlapping_twins(rig, cameras, uv_overlap, conf_overlap)
        self.assertAlmostEqual(float(c_down[0, 1, lt]), 0.9 * 0.4, places=5)

        # 3) downweight_flipped_limbs: temporary >60 deg flip at t=3 under still upper joint
        uv_flip = uv.copy()
        conf_flip = conf.copy()
        sh_h = rig.point_names.index("Shin.L.head")
        sh_t = rig.point_names.index("Shin.L.tail")
        for t in range(T):
            uv_flip[t, 0, sh_h] = [200.0, 300.0]
            uv_flip[t, 0, sh_t] = [200.0, 360.0]
        # Flip Shin.L.tail upward at t=3 (>90 deg flip) while Shin.L.head stays still
        uv_flip[3, 0, sh_t] = [200.0, 240.0]
        c_flipped = downweight_flipped_limbs(rig, cameras, uv_flip, conf_flip)
        self.assertAlmostEqual(float(c_flipped[3, 0, sh_t]), 0.9 * 0.1, places=5)
        self.assertAlmostEqual(float(c_flipped[2, 0, sh_t]), 0.9, places=5)

        # 4) downweight_contradicting_far_limb with window_expand=True
        uv_far = uv.copy()
        conf_far = conf.copy()
        # Displace Foot.R.head in left-side view (view 1, where right leg is far) at t=4 with conf < 0.5
        conf_far[3:6, 1, f_rh] = 0.45
        uv_far[4, 1, f_rh] += np.array([0.0, -120.0])
        c_far = downweight_contradicting_far_limb(
            rig, cameras, uv_far, conf_far, window_expand=True
        )
        self.assertLess(float(c_far[4, 1, f_rh]), 0.1)
        self.assertLess(float(c_far[3, 1, f_rh]), 0.1)
        self.assertLess(float(c_far[5, 1, f_rh]), 0.1)

    def test_item5_solver_prior_weights_and_robust_cauchy_loss(self):
        rig = _build_test_rig()
        w_multi = rig.prior_weights(one_view=False)
        w_single = rig.prior_weights(one_view=True)
        self.assertEqual(len(w_multi), rig.size)
        # Root rotation: 0.2 in multi-view, 2.0 in single-view; root translation: 0.0
        root_sl = rig.slices["Hips"]
        self.assertTrue(np.allclose(w_multi[root_sl.start : root_sl.start + 3], 0.2))
        self.assertTrue(np.allclose(w_single[root_sl.start : root_sl.start + 3], 2.0))
        self.assertTrue(np.allclose(w_multi[root_sl.start + 3 : root_sl.stop], 0.0))

        # Check role-specific weights
        def _bone_weights(bname):
            ix, ch = rig.slices[bname]
            return dict(zip(ch, w_multi[ix]))

        spine_w = _bone_weights("Spine")
        self.assertEqual(spine_w[1], 25.0)  # twist
        self.assertTrue(spine_w[0] == 15.0 or spine_w[2] == 15.0)

        clav_w = _bone_weights("Clavicle.L")
        self.assertEqual(clav_w[1], 25.0)
        self.assertTrue(clav_w[0] == 40.0 or clav_w[2] == 40.0)

        toe_w = _bone_weights("Toe.L")
        self.assertEqual(toe_w[1], 25.0)
        self.assertTrue(toe_w[0] == 30.0 or toe_w[2] == 30.0)

        tail_w = _bone_weights("Tail.001")
        self.assertEqual(tail_w[1], 25.0)
        self.assertTrue(tail_w[0] == 6.0 or tail_w[2] == 6.0)

        shin_w = _bone_weights("Shin.L")
        self.assertIn(60.0, shin_w.values())  # non-bend hinge axis
        self.assertIn(2.0, shin_w.values())   # primary bend axis

        # Check robust_cauchy_loss callable & finite-difference derivatives
        loss_fn = robust_cauchy_loss(np.array([6.0, 12.0]), n_robust=2)
        z = np.array([18.0, 72.0, 50.0, 100.0])
        rho = loss_fn(z)
        self.assertEqual(rho.shape, (3, 4))
        eps = 1e-6
        rho_plus = loss_fn(z + eps)
        rho_minus = loss_fn(z - eps)
        fd_d1 = (rho_plus[0] - rho_minus[0]) / (2 * eps)
        fd_d2 = (rho_plus[1] - rho_minus[1]) / (2 * eps)
        self.assertTrue(np.allclose(rho[1], fd_d1, atol=1e-5))
        self.assertTrue(np.allclose(rho[2], fd_d2, atol=1e-5))
        # Constraints after n_robust=2 stay linear: rho(z)=z, rho'(z)=1, rho''(z)=0
        self.assertTrue(np.allclose(rho[0, 2:], z[2:]))
        self.assertTrue(np.allclose(rho[1, 2:], 1.0))
        self.assertTrue(np.allclose(rho[2, 2:], 0.0))

        # Test fit_clip_2d_reprojection with w_prior, depth, and scipy_hybrid mode
        cam_front, cam_side = _test_cameras(width=512, height=512, scale=2.2, center=(0.0, 0.0, 0.8))
        cameras = [cam_front, cam_side]
        p_gt = rig.start_params()
        X_gt = rig.fk(p_gt)[0]
        uv = np.stack([[c.project(X_gt) for c in cameras]], axis=0)
        conf = np.ones((1, 2, len(rig.point_names)), dtype=float)
        depth = np.stack([[sight_depth_targets(rig, c, X_gt) for c in cameras]], axis=0)
        solved_p, diag = fit_clip_2d_reprojection(
            rig,
            cameras,
            uv,
            conf,
            w_prior=2.0,
            depth=depth,
            w_depth=0.2,
            robust_mode="scipy_hybrid",
        )
        self.assertEqual(solved_p.shape, (1, rig.size))
        self.assertTrue(np.allclose(solved_p[0], p_gt, atol=1e-3))

    def test_item6_software_rasterizer_and_fast_review_strip(self):
        pts2d = np.array([[16.0, 16.0], [48.0, 16.0], [32.0, 48.0]], dtype=np.float32)
        depths = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        tris = np.array([[0, 1, 2]], dtype=np.int64)
        pix_tri, pix_bary, zbuf = rasterize_triangles_torch(pts2d, depths, tris, (64, 64))
        self.assertEqual(pix_tri.shape, (64, 64))
        self.assertEqual(pix_bary.shape, (64, 64, 3))
        self.assertEqual(int(pix_tri[30, 32]), 0)
        self.assertAlmostEqual(float(pix_bary[30, 32].sum()), 1.0, places=5)
        self.assertAlmostEqual(float(zbuf[30, 32]), 2.0, places=5)

        verts3d = np.array(
            [
                [-0.3, 0.0, 0.2],
                [0.3, 0.0, 0.2],
                [0.0, -0.1, 0.8],
            ],
            dtype=np.float32,
        )
        vn = vertex_normals(verts3d, tris)
        self.assertEqual(vn.shape, (3, 3))
        self.assertTrue(np.allclose(np.linalg.norm(vn, axis=1), 1.0, atol=1e-4))

        cam, _ = _test_cameras(width=64, height=64, scale=1.5, center=(0.0, 0.0, 0.5))
        rgb, mask, _, _, _ = render_mesh_software(verts3d, tris, cam, width=64, height=64)
        self.assertEqual(rgb.shape, (64, 64, 3))
        self.assertEqual(mask.shape, (64, 64))
        self.assertTrue(np.any(mask))

        # Fast 4-column review strip
        meta = {
            "Root": {
                "parent": None,
                "kind": "root",
                "rest_head_world": [0.0, 0.0, 0.2],
                "rest_tail_world": [0.0, 0.0, 0.8],
            }
        }
        rig = RigModel(meta)
        skin = {
            "bone_names": ["Root"],
            "rest_verts": verts3d,
            "triangles": tris,
            "skin_indices": np.zeros((3, 1), dtype=int),
            "skin_weights": np.ones((3, 1), dtype=float),
            "rest_heads": np.array([[0.0, 0.0, 0.2]]),
            "rest_R": np.array([rig.R["Root"]]),
            "height": 0.6,
        }
        panel_frames = [[rgb.copy(), rgb.copy()]]
        params = np.vstack([rig.start_params(), rig.start_params()])
        review = render_fast_review_frames(
            rig, [cam], params, skin=skin, video_panels=panel_frames
        )
        self.assertEqual(len(review["frames"]), 2)
        # 4 columns of width 64 -> total width 256
        self.assertEqual(review["frames"][0].shape, (64, 256, 3))
        self.assertGreater(review["mean_iou"], 0.9)


if __name__ == "__main__":
    unittest.main()
