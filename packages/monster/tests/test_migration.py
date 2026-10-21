import ast
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release = load("monster_package_release", "scripts/package-release.py")
installer = load("monster_install_release", "scripts/install-release.py")
client_module = load("monster_client", "integration/monster_client.py")
runtime = load("monster_colab_runtime", "scripts/colab_runtime.py")


class MigrationTests(unittest.TestCase):
    def test_release_installs_and_replaces_without_touching_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = release.build_release(root / "first")
            data = root / "user-data" / "existing.blend"
            data.parent.mkdir()
            data.write_bytes(b"existing project")
            package = installer.install_release(first["archive"], root / "versions")
            self.assertEqual(installer.verify_package(package)["release_id"], first["release_id"])
            (package / "worker/pipeline_runner.py").write_text("changed")
            reinstalled = installer.install_release(first["archive"], root / "versions")
            self.assertEqual(reinstalled, package)
            self.assertNotEqual((package / "worker/pipeline_runner.py").read_text(), "changed")
            self.assertEqual(data.read_bytes(), b"existing project")
            notebook = json.loads(Path(first["notebook"]).read_text())
            source = "".join(notebook["cells"][1]["source"])
            self.assertIn(Path(first["archive"]).name, source)
            self.assertNotIn('RELEASE_ARCHIVE = ""', source)

    def test_missing_archive_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "not found"):
                installer.install_release(root / "missing.tar.gz", root / "versions")
            self.assertFalse((root / "versions").exists())

    def test_generated_notebook_loads_the_packaged_runtime(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = release.build_release(root / "release")
            notebook = json.loads(Path(result["notebook"]).read_text())
            source = "".join(notebook["cells"][1]["source"])
            drive_path = "/content/drive/MyDrive/MonsterStudio/releases/" + Path(result["archive"]).name
            source = source.replace(drive_path, result["archive"])
            source = source.replace("/content/monster-releases", str(root / "versions"))
            tree = ast.parse(source)
            # The final configure call sets real Colab paths; verify loading independently.
            self.assertIsInstance(tree.body[-1], ast.Expr)
            tree.body.pop()
            google, colab = ModuleType("google"), ModuleType("google.colab")
            colab.drive = SimpleNamespace(mount=lambda directory: None)
            with patch.dict(sys.modules, {"google": google, "google.colab": colab}):
                namespace = {"__name__": "__notebook_test__"}
                exec(compile(tree, "generated-colab.ipynb", "exec"), namespace)
            package = namespace["MONSTER_PACKAGE"]
            self.assertTrue((package / "worker/colab_g4_worker_server.py").is_file())
            self.assertEqual(namespace["monster_runtime"].ROOT, package)
            self.assertEqual(installer.verify_package(package)["release_id"], result["release_id"])

    def test_template_notebook_clones_from_github_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook = json.loads((ROOT / "notebooks/colab.ipynb").read_text())
            source = "".join(notebook["cells"][1]["source"])
            source = source.replace("/content/monster-releases", str(root / "versions"))
            tree = ast.parse(source)
            self.assertIsInstance(tree.body[-1], ast.Expr)
            tree.body.pop()
            google, colab = ModuleType("google"), ModuleType("google.colab")
            colab.drive = SimpleNamespace(mount=lambda directory: None)

            def fake_clone(cmd, check):
                target = Path(cmd[-1]) / "packages/monster/scripts"
                target.mkdir(parents=True, exist_ok=True)
                (target / "colab_runtime.py").write_text((ROOT / "scripts/colab_runtime.py").read_text())

            with patch.dict(sys.modules, {"google": google, "google.colab": colab}), \
                    patch("subprocess.run", side_effect=fake_clone) as run_mock:
                namespace = {"__name__": "__notebook_github_test__"}
                exec(compile(tree, "colab.ipynb", "exec"), namespace)
            self.assertTrue(run_mock.called)
            self.assertEqual(namespace["MONSTER_PACKAGE"], root / "versions/github-checkout/packages/monster")

    def test_release_does_not_package_credentials_or_caches(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = release.build_release(temporary)
            with tarfile.open(result["archive"]) as tar:
                names = tar.getnames()
                self.assertEqual(tar.extractfile("monster/LICENSE").read(), (ROOT / "LICENSE").read_bytes())
                self.assertIn(b"SkinTokens", tar.extractfile("monster/THIRD_PARTY.md").read())
                self.assertIn(b"AGPL-3.0-only", tar.extractfile("monster/LICENSING.md").read())
                self.assertIn(b"VAST-AI-Research", tar.extractfile("monster/licenses/SkinTokens-MIT.txt").read())
                self.assertEqual(tar.extractfile("monster/skills/monster-workflow/LICENSE.txt").read(), (ROOT / "LICENSE").read_bytes())
                self.assertEqual(json.load(tar.extractfile("monster/package.json"))["license"], "AGPL-3.0-only")
            self.assertFalse(any("endpoint_registry" in name or "__pycache__" in name for name in names))
            self.assertIn("monster/contracts/protocol.json", names)
            self.assertIn("monster/scripts/bootstrap-skintokens.sh", names)

    def test_archive_path_escape_and_links_rejected(self):
        for name, kind in [("../escape.py", tarfile.REGTYPE), ("monster/link", tarfile.SYMTYPE)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive = root / "bad.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    item = tarfile.TarInfo(name)
                    item.type = kind
                    item.size = 1 if kind == tarfile.REGTYPE else 0
                    item.linkname = "../outside"
                    tar.addfile(item, io.BytesIO(b"x"))
                with self.assertRaisesRegex(ValueError, "Invalid archive entry"):
                    installer.install_release(archive, root / "versions")
                self.assertEqual(list((root / "versions").iterdir()), [])
                self.assertFalse((root / "escape.py").exists())

    def test_running_task_prevents_reload(self):
        old = SimpleNamespace(GUARD=SimpleNamespace(busy=True, stopped=False))
        with patch.dict(sys.modules, {"colab_g4_worker_server": old}):
            with self.assertRaisesRegex(RuntimeError, "正在执行任务"):
                runtime.stop_previous_worker()
        self.assertFalse(old.GUARD.stopped)

    def test_idle_worker_closes_before_reload(self):
        calls = []
        old = SimpleNamespace(GUARD=SimpleNamespace(busy=False, stopped=False),
                              APP=SimpleNamespace(close=lambda: calls.append("closed")))
        with patch.dict(sys.modules, {"colab_g4_worker_server": old}):
            runtime.stop_previous_worker()
        self.assertTrue(old.GUARD.stopped)
        self.assertEqual(calls, ["closed"])

    def test_resolution_2k_feature_flag_and_default_full_weights(self):
        sys.path.insert(0, str(ROOT / "worker"))
        try:
            pose_features = load("monster_pose_features", "worker/pose_features.py")
            pipeline_runner = load("monster_pipeline_runner", "worker/pipeline_runner.py")
            with patch.dict("os.environ", {"MONSTER_ENABLE_2K": "0", "H3_PROFILE": "full"}, clear=False):
                self.assertEqual(pose_features.studio_resolution(), (672, 768, 1344, 768))
                self.assertEqual(pose_features.idle_split_filename(), "idle_split_1344x768.png")
                self.assertEqual(pose_features.dinov2_panel_padding(672, 768), (1, 1, 0, 0))
                graph_default = pipeline_runner.build_minimax_h3_comfy_graph("walk", "ref.png")
                self.assertEqual(graph_default["7"]["inputs"]["width"], 1344)
                self.assertEqual(graph_default["7"]["inputs"]["height"], 768)
                self.assertNotIn("31", graph_default)
                self.assertNotIn("38", graph_default)
                self.assertEqual(graph_default["13"]["inputs"]["samples"], ["12", 0])
                self.assertEqual(
                    graph_default["1"]["inputs"]["unet_name"],
                    "minimax_h3_fl2va_pruned_bf16.safetensors",
                )
                self.assertEqual(
                    graph_default["3"]["inputs"]["clip_name"],
                    "qwen3vl_32b_minimax_h3_bf16.safetensors",
                )
                self.assertEqual(
                    graph_default["4"]["inputs"]["vae_name"],
                    "minimax_h3_video_vae_fp16.safetensors",
                )
            with patch.dict("os.environ", {"MONSTER_ENABLE_2K": "1"}, clear=False):
                self.assertEqual(pose_features.studio_resolution(), (1280, 1440, 2560, 1440))
                self.assertEqual(pose_features.idle_split_filename(), "idle_split_2560x1440.png")
                pad_t, pad_b, pad_l, pad_r = pose_features.dinov2_panel_padding(1280, 1440)
                self.assertEqual((pad_t, pad_b, pad_l, pad_r), (1, 1, 4, 4))
                self.assertEqual((1280 + pad_l + pad_r) % 14, 0)
                self.assertEqual((1440 + pad_t + pad_b) % 14, 0)
                graph_2k = pipeline_runner.build_minimax_h3_comfy_graph("walk", "ref.png", "ref.png")
                # Pass 1 generates base global motion at 1344x768
                self.assertEqual(graph_2k["7"]["inputs"]["width"], 1344)
                self.assertEqual(graph_2k["7"]["inputs"]["height"], 768)
                # Pass 2 upscales 24-ch video latent via 3D Conv Upscaler + Tiled Hires-Fix to 2560x1440
                self.assertEqual(graph_2k["30"]["class_type"], "LTXVSeparateAVLatent")
                self.assertEqual(graph_2k["31"]["class_type"], "MinimaxH3LatentUpscaler3D")
                self.assertEqual(
                    graph_2k["31"]["inputs"]["mode"],
                    {"mode": "target dimensions", "width": 2560, "height": 1440},
                )
                self.assertEqual(graph_2k["32"]["class_type"], "LTXVConcatAVLatent")
                self.assertEqual(graph_2k["33"]["inputs"]["width"], 2560)
                self.assertEqual(graph_2k["33"]["inputs"]["height"], 1440)
                self.assertEqual(graph_2k["38"]["class_type"], "MMH3SplitUpscale")
                self.assertEqual(graph_2k["13"]["inputs"]["samples"], ["38", 0])
                self.assertEqual(graph_2k["14"]["inputs"]["samples"], ["38", 0])
        finally:
            sys.path.remove(str(ROOT / "worker"))

    def test_task_progress_panel_and_worker_persistence(self):
        notebook = json.loads((ROOT / "notebooks/colab.ipynb").read_text())
        cell_ids = [c.get("metadata", {}).get("id") for c in notebook["cells"]]
        self.assertIn("monster-progress", cell_ids)
        prog_cell = [c for c in notebook["cells"] if c.get("metadata", {}).get("id") == "monster-progress"][0]
        self.assertIn("monster_runtime.show_progress()", "".join(prog_cell["source"]))

        # Test colab_runtime progress functions
        data = runtime.get_progress_data()
        self.assertIn("stage", data)
        self.assertIn("gpu", data)
        self.assertIn("eta_str", data)

        # Mock data and verify HTML output
        mock_data = {
            "monster_id": "test_monster",
            "stage": "training",
            "stage_display": "训练 DINOv2 姿态头",
            "status": "running",
            "epoch": 3,
            "epochs": 15,
            "completed_batches": 45,
            "batches_per_epoch": 313,
            "overall_pct": 18.5,
            "eta_str": "约 12 分钟",
            "losses": {"uv": 0.01234, "heatmap": 0.04567, "visibility": 0.00891, "total": 0.17523},
            "history": [{"epoch": 1, "uv": 0.05, "heatmap": 0.08, "visibility": 0.02, "total": 0.5},
                        {"epoch": 2, "uv": 0.03, "heatmap": 0.06, "visibility": 0.01, "total": 0.3}],
            "detail": "第 3 / 15 轮 · 本轮 45 / 313 批次 · 预计剩余：约 12 分钟",
            "gpu": {"display": "Tesla T4 · 显存: 4200 / 15360 MiB · 利用率: 76%"},
            "updated_at": "2026-10-21 17:40:00",
            "updated_unix": 1729500000.0,
        }
        html_out = runtime.render_progress_html(mock_data)
        self.assertIn("训练 DINOv2 姿态头", html_out)
        self.assertIn("第 3 / 15 轮", html_out)
        self.assertIn("45 / 313", html_out)
        self.assertIn("约 12 分钟", html_out)
        self.assertIn("Tesla T4", html_out)
        self.assertIn("0.01234", html_out)
        self.assertIn("关键点 UV loss", html_out)

        # Test write_task_progress and dinov2 trainer progress writing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            sys.path.insert(0, str(ROOT / "worker"))
            try:
                server = load("monster_server", "worker/colab_g4_worker_server.py")
                dino = load("monster_dino", "worker/dinov2_pose_trainer.py")
                with patch.object(server, "DRIVE_ROOT", temp_path / "drive"), \
                     patch.object(server, "WORK_ROOT", temp_path / "work"):
                    server.write_task_progress(
                        monster_id="swamp_troll",
                        stage="training",
                        stage_display="训练 DINOv2 姿态头",
                        status="running",
                        detail="测试中",
                        epochs=15,
                    )
                    runtime_prog = temp_path / "drive/runtime/current_task_progress.json"
                    self.assertTrue(runtime_prog.is_file())
                    saved = json.loads(runtime_prog.read_text(encoding="utf-8"))
                    self.assertEqual(saved["monster_id"], "swamp_troll")
                    self.assertEqual(saved["stage"], "training")

                self.assertEqual(dino._format_eta(45), "约 45 秒")
                self.assertEqual(dino._format_eta(360), "约 6 分钟")
                self.assertEqual(dino._format_eta(0), "即将完成")

                target_prog = temp_path / "custom_progress.json"
                dino._write_progress_files([target_prog], {"epoch": 1, "test": True})
                self.assertTrue(target_prog.is_file())
                self.assertTrue(json.loads(target_prog.read_text())["test"])
            finally:
                sys.path.remove(str(ROOT / "worker"))

    def test_pose_and_ik_optimizations_present(self):
        dino_src = (ROOT / "worker/dinov2_pose_trainer.py").read_text(encoding="utf-8")
        bpy_src = (ROOT / "worker/studio_bpy_worker.py").read_text(encoding="utf-8")
        vis_src = (ROOT / "worker/pose_visibility.py").read_text(encoding="utf-8")
        dino_tree = ast.parse(dino_src)
        bpy_tree = ast.parse(bpy_src)
        dino_funcs = {n.name for n in dino_tree.body if isinstance(n, ast.FunctionDef)}
        bpy_funcs = {n.name for n in bpy_tree.body if isinstance(n, ast.FunctionDef)}
        self.assertIn("clean_2d_trajectories_hampel", dino_funcs)
        self.assertIn("configure_torch_gpu_acceleration", dino_funcs)
        self.assertIn("enable_backbone_sdpa_attention", dino_funcs)
        self.assertIn("scaled_dot_product_attention", dino_src)
        self.assertIn("1.0 + 19.0 * gt_hm", dino_src)
        self.assertIn("peak_mass", dino_src)
        self.assertIn("configure_blender_gpu_rendering", bpy_funcs)
        self.assertIn("denoising_use_gpu = True", bpy_src)
        self.assertIn("use_persistent_data = True", bpy_src)
        self.assertIn('sensor_fit = "VERTICAL"', bpy_src)
        self.assertIn("_get_static_mesh_topology", vis_src)
        for fn in (
            "euler_xyz_to_matrix_numpy",
            "minimum_swing_matrix_numpy",
            "build_clamped_local_euler_rotation",
            "solve_bounded_bone_rotation_matrix",
            "mat_to_quat_wxyz_numpy",
            "quat_wxyz_to_mat_numpy",
            "smooth_rotation_matrices_savgol",
        ):
            self.assertIn(fn, bpy_funcs)
        try:
            import numpy as np
        except ImportError:
            return
        sys.path.insert(0, str(ROOT / "worker"))
        try:
            dino_mod = load("monster_dinov2_trainer", "worker/dinov2_pose_trainer.py")
            bpy_mod = load("monster_bpy_worker", "worker/studio_bpy_worker.py")
            logits = np.full((1, 2, 16, 16), -6.0, dtype=np.float32)
            logits[0, 0, 8, 8] = 0.0
            coords, conf = dino_mod.dsnt_soft_argmax_numpy(logits)
            self.assertEqual(coords.shape, (1, 2, 2))
            self.assertGreater(float(conf[0, 0]), 0.8)
            uv_f = np.zeros((7, 1, 2), dtype=np.float32)
            uv_s = np.zeros((7, 1, 2), dtype=np.float32)
            uv_s[3, 0, 0] = 196.0
            cf = np.full((7, 1), 0.8, dtype=np.float32)
            cs = np.full((7, 1), 0.8, dtype=np.float32)
            _, uv_s_clean, _, cs_clean = dino_mod.clean_2d_trajectories_hampel(uv_f, uv_s, cf, cs)
            self.assertLess(abs(float(uv_s_clean[3, 0, 0])), 1.0)
            self.assertLess(float(cs_clean[3, 0]), 0.3)
            r_seq = np.repeat(np.eye(3, dtype=np.float64)[None, :, :], 7, axis=0)
            r_smooth = bpy_mod.smooth_rotation_matrices_savgol(r_seq, gaussian_passes=1)
            self.assertTrue(np.allclose(r_smooth, r_seq, atol=1e-6))
        finally:
            sys.path.remove(str(ROOT / "worker"))

    def test_canonical_bone_taxonomy_and_kind_solvers(self):
        try:
            import numpy as np
        except ImportError:
            return
        sys.path.insert(0, str(ROOT / "worker"))
        try:
            tax = load("monster_bone_taxonomy", "worker/bone_taxonomy.py")
            # 1. Canonical naming parse & format
            formatted = tax.format_canonical_bone_name("limb", "L", 1, 3)
            self.assertEqual(formatted, "LIMB_L_01_03")
            parsed = tax.parse_canonical_bone_name(formatted)
            self.assertIsNotNone(parsed)
            self.assertEqual(parsed["kind"], "limb")
            self.assertEqual(parsed["side"], "L")
            self.assertEqual(parsed["group_id"], "01")
            self.assertEqual(parsed["chain_index"], 3)

            # 2. Hybrid classification on an anonymous rig with torso, arm hub, 3 fingers, rigid & flex
            def make_bone(parent, head, tail, limits_rad=None, fixed_local=False):
                h = np.asarray(head, dtype=np.float64)
                t = np.asarray(tail, dtype=np.float64)
                L = float(np.linalg.norm(t - h))
                d0 = (t - h) / max(1e-8, L)
                up = np.array([0.0, 0.0, 1.0]) if abs(float(d0[2])) < 0.9 else np.array([1.0, 0.0, 0.0])
                x_ax = np.cross(d0, up)
                x_ax /= max(1e-8, float(np.linalg.norm(x_ax)))
                z_ax = np.cross(x_ax, d0)
                rest_r = np.column_stack([x_ax, d0, z_ax])
                info = {
                    "parent": parent,
                    "rest_head_world": h.tolist(),
                    "rest_tail_world": t.tolist(),
                    "rest_length": L,
                    "rest_rot_mat_world": rest_r.tolist(),
                }
                if limits_rad is not None:
                    info["limits_rad"] = limits_rad
                if fixed_local:
                    info["fixed_local"] = True
                return info

            rig_bones = {
                "bone_0": make_bone(None, [0.0, 0.0, 0.50], [0.0, 0.0, 0.60]),
                "bone_1": make_bone("bone_0", [0.0, 0.0, 0.60], [0.0, 0.0, 0.75]),
                "bone_2": make_bone("bone_1", [0.0, 0.0, 0.75], [0.0, 0.0, 0.95]),
                "bone_3": make_bone("bone_1", [0.05, 0.0, 0.72], [0.15, 0.0, 0.72]),
                "bone_4": make_bone("bone_3", [0.15, 0.0, 0.72], [0.30, 0.0, 0.60]),
                "bone_5": make_bone("bone_4", [0.30, 0.0, 0.60], [0.42, 0.0, 0.50]),
                "bone_6": make_bone("bone_5", [0.42, 0.0, 0.50], [0.442, 0.0, 0.485]),
                "bone_7": make_bone("bone_6", [0.442, 0.02, 0.485], [0.465, 0.02, 0.470]),
                "bone_8": make_bone("bone_7", [0.465, 0.02, 0.470], [0.485, 0.02, 0.455]),
                "bone_9": make_bone("bone_6", [0.442, 0.00, 0.485], [0.465, 0.00, 0.470]),
                "bone_10": make_bone("bone_9", [0.465, 0.00, 0.470], [0.485, 0.00, 0.455]),
                "bone_11": make_bone("bone_6", [0.442, -0.02, 0.485], [0.465, -0.02, 0.470]),
                "bone_12": make_bone("bone_11", [0.465, -0.02, 0.470], [0.485, -0.02, 0.455]),
                "Core_Rigid": make_bone(
                    "bone_0",
                    [0.0, 0.0, 0.55],
                    [0.0, 0.0, 0.65],
                    limits_rad=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                    fixed_local=True,
                ),
                "Flex_Cape_0": make_bone("bone_0", [0.0, 0.05, 0.50], [0.0, 0.05, 0.25]),
            }
            classified = tax.classify_rig_bones(rig_bones)
            self.assertEqual(classified["bone_0"]["kind"], "root")
            self.assertEqual(classified["bone_1"]["kind"], "spine")
            self.assertEqual(classified["bone_3"]["kind"], "limb")
            self.assertEqual(classified["bone_3"]["sub_role"], "clavicle_socket")
            self.assertEqual(classified["bone_4"]["kind"], "limb")
            self.assertEqual(classified["bone_4"]["sub_role"], "upper_limb")
            self.assertEqual(classified["bone_5"]["kind"], "limb")
            self.assertEqual(classified["bone_5"]["sub_role"], "forearm_hinge")
            self.assertEqual(classified["bone_6"]["kind"], "limb")
            self.assertEqual(classified["bone_6"]["sub_role"], "hand_hub")
            for d_name in ("bone_7", "bone_8", "bone_9", "bone_10", "bone_11", "bone_12"):
                self.assertEqual(classified[d_name]["kind"], "digit")
            self.assertEqual(classified["Core_Rigid"]["kind"], "rigid")
            self.assertEqual(classified["Flex_Cape_0"]["kind"], "flex")

            # 3. Short-gap 2D fill (<= 3 frames filled; > 3 frames NOT filled)
            vals = np.linspace(0.0, 10.0, 11, dtype=np.float64)[:, None]
            vals_corrupt = vals.copy()
            vals_corrupt[2:4, 0] = -999.0
            vals_corrupt[6:10, 0] = -999.0
            bad_mask = np.zeros((11,), dtype=bool)
            bad_mask[2:4] = True
            bad_mask[6:10] = True
            filled = tax.short_gap_fill_1d_or_nd(vals_corrupt, bad_mask, max_gap=3)
            self.assertTrue(np.allclose(filled[2:4, 0], vals[2:4, 0], atol=1e-5))
            self.assertEqual(float(filled[7, 0]), -999.0)

            # 4. Two-branch Euler clamping without post-clamp minimum_swing override
            r_big = tax.euler_xyz_to_matrix_numpy(*np.deg2rad([60.0, -50.0, 70.0]))
            lim = [np.deg2rad([-25.0, 25.0]).tolist(), np.deg2rad([-20.0, 20.0]).tolist(), np.deg2rad([-30.0, 30.0]).tolist()]
            r_clamped = tax.clamp_local_delta_euler(r_big, lim)
            e_clamped = tax.matrix_to_euler_xyz_numpy(r_clamped)
            self.assertLessEqual(abs(float(np.rad2deg(e_clamped[0]))), 25.0 + 1e-4)
            self.assertLessEqual(abs(float(np.rad2deg(e_clamped[1]))), 20.0 + 1e-4)
            self.assertLessEqual(abs(float(np.rad2deg(e_clamped[2]))), 30.0 + 1e-4)
            self.assertAlmostEqual(float(np.linalg.det(r_clamped)), 1.0, places=5)

            # 5. End-to-end reconstruct_frames_with_kind_solvers & compute_stratified_motion_metrics
            bone_names = sorted(classified.keys())
            n_bones = len(bone_names)
            pts_3d = np.zeros((5, n_bones * 2, 3), dtype=np.float64)
            for bi, b_name in enumerate(bone_names):
                pts_3d[:, bi * 2, :] = classified[b_name]["rest_head_world"]
                pts_3d[:, bi * 2 + 1, :] = classified[b_name]["rest_tail_world"]
            trust = np.full((5, n_bones * 2), 0.9, dtype=np.float64)
            frames, diag = tax.reconstruct_frames_with_kind_solvers(
                pts_3d, trust, bone_names, classified, depth_scale=1.0
            )
            self.assertEqual(len(frames), 5)
            self.assertEqual(diag["short_gap_max_frames"], 3)
            gt_frames = [
                {
                    "frame": fi,
                    "bones": {
                        b_name: {
                            "head_3d": classified[b_name]["rest_head_world"],
                            "tail_3d": classified[b_name]["rest_tail_world"],
                            "rot_mat_world": frames[fi]["bones"][b_name]["rot_mat_world"],
                        }
                        for b_name in bone_names
                    },
                }
                for fi in range(5)
            ]
            strat = tax.compute_stratified_motion_metrics(frames, gt_frames, classified)
            self.assertLess(strat["all_bones"]["direction_mean_deg"], 0.1)
            self.assertLess(strat["primary_bones"]["full_rot_mean_deg"], 0.1)
            self.assertIn("limb", strat["by_kind"])
            self.assertIn("digit", strat["by_kind"])

            # 6. Stratified 5,000-sample 4-regime decoupled pose sampler & coupled-curl audit gate
            import random
            self.assertEqual(tax.DEFAULT_SYNTH_SAMPLES, 5000)
            rng = random.Random(42)
            samples_500 = []
            for idx in range(500):
                eulers, regime = tax.sample_stratified_pose_euler_dict(
                    classified, rng=rng, sample_idx=idx, enforce_anatomical_floor=True
                )
                samples_500.append({"sampling_regime": regime, "joints": eulers})
            cov_good = tax.audit_pose_dataset_coverage(samples_500, classified, enforce_anatomical_floor=True)
            self.assertTrue(cov_good["passed"], msg=f"Unexpected failures: {cov_good['failures']}")
            self.assertEqual(cov_good["regime_counts"]["synergistic"], 200)
            self.assertEqual(cov_good["regime_counts"]["decoupled"], 150)
            self.assertEqual(cov_good["regime_counts"]["antiphase_extreme"], 75)
            self.assertEqual(cov_good["regime_counts"]["workspace_cross_occlusion"], 75)
            self.assertLess(cov_good["decoupling_metrics"]["digit_pc1_variance_ratio"], 0.45)
            self.assertLess(cov_good["decoupling_metrics"]["max_inter_finger_abs_corr"], 0.45)
            self.assertGreaterEqual(cov_good["decoupling_metrics"]["antiphase_sign_flip_ratio"], 0.15)

            # Simulate a defective "fixed-ratio hand curl" dataset where all fingers curl by 1 scalar
            bad_samples = []
            for idx in range(200):
                global_curl = -0.25 + 0.80 * (idx / 199.0)
                jmap = {}
                for b_name in bone_names:
                    if classified[b_name]["kind"] == "digit":
                        jmap[b_name] = [global_curl, 0.0, 0.0]
                    else:
                        jmap[b_name] = samples_500[idx]["joints"][b_name]
                bad_samples.append({"sampling_regime": "synergistic", "joints": jmap})
            cov_bad = tax.audit_pose_dataset_coverage(bad_samples, classified, enforce_anatomical_floor=True)
            self.assertFalse(cov_bad["passed"])
            self.assertTrue(any("Coupled hand/digit curl" in f for f in cov_bad["failures"]))
            self.assertTrue(any("Inter-finger fixed-ratio coupling" in f for f in cov_bad["failures"]))
        finally:
            sys.path.remove(str(ROOT / "worker"))


class FakeTransport:
    protocol_version = 1

    def __init__(self, url, **kwargs):
        self.calls = []
        self.options = kwargs

    def predict(self, *args, api_name):
        self.calls.append((api_name, args))
        if api_name == "/heartbeat":
            return json.dumps({"protocol_version": self.protocol_version, "status": "READY"})
        if api_name == "/rig_candidates":
            return ("{}", "downloaded-rig.zip")
        return "{}"


class ClientTests(unittest.TestCase):
    registry = {"url": "https://example.gradio.live", "username": "monster", "token": "test-only-secret"}

    def connect(self, directory, factory=FakeTransport):
        return client_module.MonsterClient(self.registry, directory, client_factory=factory,
                                           file_handler=lambda file: {"upload": file})

    def test_authentication_handshake_and_file_upload_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = self.connect(temporary)
            source = Path(temporary) / "input.glb"
            source.write_bytes(b"test fixture, not a production model")
            result = client.rig_candidates("creature_01", source)
            self.assertEqual(result[1], "downloaded-rig.zip")
            self.assertEqual(client._client.options["auth"], ("monster", "test-only-secret"))
            self.assertEqual(client._client.calls, [
                ("/heartbeat", ()),
                ("/rig_candidates", ("creature_01", {"upload": str(source.resolve())}, True)),
            ])

    def test_legacy_or_incompatible_worker_cannot_submit(self):
        class OldWorker(FakeTransport):
            protocol_version = None
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "Incompatible"):
                self.connect(temporary, OldWorker)

    def test_remote_plain_http_and_embedded_credentials_rejected(self):
        for url in ["http://example.com", "https://u:p@example.com", "https://example.com?token=secret"]:
            with self.subTest(url=url), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaises(ValueError):
                    client_module.MonsterClient({**self.registry, "url": url}, temporary,
                                                client_factory=FakeTransport)

    def test_no_automatic_resubmission_after_unknown_result(self):
        class LostResponse(FakeTransport):
            def predict(self, *args, api_name):
                if api_name != "/heartbeat":
                    self.calls.append((api_name, args))
                    raise TimeoutError("remote result unknown")
                return super().predict(*args, api_name=api_name)
        with tempfile.TemporaryDirectory() as temporary:
            client = self.connect(temporary, LostResponse)
            with self.assertRaises(TimeoutError):
                client.generate_actions_batch("creature", "idle:loop")
            self.assertEqual(len(client._client.calls), 2)
            self.assertEqual(client._client.calls[-1],
                             ("/generate_actions_batch", ("creature", "idle:loop", True, False)))

    def test_invalid_inputs_never_submit(self):
        with tempfile.TemporaryDirectory() as temporary:
            client = self.connect(temporary)
            with self.assertRaises(ValueError):
                client.generate_actions_batch("../outside", "idle:loop")
            with self.assertRaises(ValueError):
                client.train_monster_probe("creature", epochs=0)
            with self.assertRaises(ValueError):
                client.generate_actions_batch("creature", "")
            self.assertEqual(client._client.calls, [("/heartbeat", ())])

    def test_auto_discovery_and_one_line_connect_pairing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            drive_reg = root / "Library/CloudStorage/GoogleDrive-user/MyDrive/MonsterStudio/runtime/endpoint_registry.json"
            drive_reg.parent.mkdir(parents=True, exist_ok=True)
            drive_reg.write_text(json.dumps(self.registry))
            discovered = client_module.discover_registry_path(home=root)
            self.assertEqual(discovered, drive_reg.resolve())
            saved_path = root / "var/monster/endpoint_registry.json"
            receipt = client_module.save_registry(
                json.dumps({"url": "https://example.gradio.live", "token": "test-only-secret"}),
                destination=saved_path,
                output_dir=root / "downloads",
                client_factory=FakeTransport,
                file_handler=lambda file: {"upload": file},
            )
            self.assertEqual(receipt["status"], "CONNECTED")
            self.assertNotIn("token", receipt)
            self.assertEqual(json.loads(saved_path.read_text())["username"], "monster")

    def test_eval_suite_manifest_and_synthetic_pipeline(self):
        pkg_script = (ROOT / "scripts/package-release.py").read_text()
        self.assertNotIn('"evals"', pkg_script)

        manifest_path = ROOT / "evals/benchmarks/fantasy_creature_2k_v2/manifest.json"
        golden_path = ROOT / "evals/benchmarks/fantasy_creature_2k_v2/golden_frames.json"
        baseline_path = ROOT / "evals/benchmarks/fantasy_creature_2k_v2/baseline_latest.json"
        self.assertTrue(manifest_path.is_file())
        self.assertTrue(golden_path.is_file())
        self.assertTrue(baseline_path.is_file())

        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["resolution_mode"], "2k")
        self.assertEqual(manifest["composite_resolution"], [2560, 1440])
        self.assertEqual(manifest["panel_resolution"], [1280, 1440])
        self.assertEqual(manifest["bone_count"], 77)

        baseline = json.loads(baseline_path.read_text())
        self.assertEqual(baseline["solver_name"], "taxonomy_kind_v2")
        self.assertTrue(baseline["golden_frames"]["all_passed"])

        try:
            import numpy as np
        except ImportError:
            return

        from evals.metrics.evaluator import diagnose_single_frame, evaluate_full_pipeline
        from evals.reporters.markdown_reporter import render_eval_summary_markdown
        from evals.solvers.latest_solver import run_latest_solver

        studio_meta = {
            "bbox_min": [-0.3, -0.2, 0.0],
            "bbox_max": [0.3, 0.2, 1.0],
            "calibration": {
                "center": [0.0, 0.0, 0.5],
                "panel_width": 1280,
                "panel_height": 1440,
                "composite_width": 2560,
                "composite_height": 1440,
                "ortho_scale": 1.2,
                "depth_scale": 0.8,
            },
            "bones": {
                "ROOT_C_00_00": {
                    "parent": None,
                    "rest_head_world": [0.0, 0.0, 0.50],
                    "rest_tail_world": [0.0, 0.0, 0.60],
                    "rest_length": 0.10,
                    "rest_rot_mat_world": [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                },
                "RIGID_C_01_01": {
                    "parent": "ROOT_C_00_00",
                    "rest_head_world": [0.0, 0.0, 0.50],
                    "rest_tail_world": [0.0, 0.0, 0.58],
                    "rest_length": 0.08,
                    "fixed_local": True,
                    "rest_rot_mat_world": [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                },
            },
        }
        T, K = 6, 4
        obs = {
            "uv_front_all": np.full((T, K, 2), [640.0, 720.0], dtype=np.float32),
            "uv_side_all": np.full((T, K, 2), [640.0, 720.0], dtype=np.float32),
            "conf_front_all": np.full((T, K), 0.95, dtype=np.float32),
            "conf_side_all": np.full((T, K), 0.95, dtype=np.float32),
            "depth_front_all": np.zeros((T, K), dtype=np.float32),
            "depth_side_all": np.zeros((T, K), dtype=np.float32),
            "visibility_front_all": np.ones((T, K), dtype=np.float32),
            "visibility_side_all": np.ones((T, K), dtype=np.float32),
            "logvar_front_all": np.full((T, K), -4.0, dtype=np.float32),
            "logvar_side_all": np.full((T, K), -4.0, dtype=np.float32),
        }
        # Set tail v slightly higher (smaller pixel y)
        obs["uv_front_all"][:, 1::2, 1] = 600.0
        obs["uv_side_all"][:, 1::2, 1] = 600.0

        solver_res = run_latest_solver(obs, studio_meta)
        gt = {"frames": solver_res.stage4_frames}
        golden = {
            "probes": [
                {"frame": 0, "name": "f0", "description": "start", "max_root_rot_deg": 5.0, "max_primary_dir_deg": 5.0}
            ]
        }
        report = evaluate_full_pipeline("synth_2k", "fast", solver_res, gt, golden)
        self.assertTrue(report["golden_frames"]["all_passed"])
        self.assertAlmostEqual(report["stratified_kinematics"]["primary_bones"]["direction_mean_deg"], 0.0, places=2)
        diag = diagnose_single_frame(solver_res, gt, frame_idx=0)
        self.assertEqual(diag["frame"], 0)
        md = render_eval_summary_markdown(report)
        self.assertIn("Monster Studio Eval Report", md)


if __name__ == "__main__":
    unittest.main()
