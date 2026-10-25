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
                # Exercise the trainer's shared helpers without importing NumPy or torch.
                progress = load("monster_task_progress", "worker/task_progress.py")
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

                self.assertEqual(progress.format_eta(45), "约 45 秒")
                self.assertEqual(progress.format_eta(360), "约 6 分钟")
                self.assertEqual(progress.format_eta(0), "即将完成")

                target_prog = temp_path / "custom_progress.json"
                progress.write_progress_files([target_prog], {"epoch": 1, "test": True})
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

            # 2b. Mandatory preflight bone naming validation gate
            anon_check = tax.validate_canonical_rig_bone_names(classified)
            self.assertFalse(anon_check["valid"])
            self.assertEqual(len(anon_check["non_canonical_bones"]), len(classified))
            canon_bones = {
                info["canonical_name"]: {
                    **info,
                    "parent": (
                        classified[info["parent"]]["canonical_name"]
                        if info.get("parent") in classified
                        else info.get("parent")
                    ),
                }
                for info in classified.values()
            }
            canon_check = tax.validate_canonical_rig_bone_names(canon_bones)
            self.assertTrue(canon_check["valid"])
            self.assertEqual(canon_check["non_canonical_bones"], [])
            self.assertEqual(canon_check["canonical_count"], len(canon_bones))
            skill_md = (ROOT / "skills/monster-workflow/SKILL.md").read_text(encoding="utf-8")
            sop_md = (ROOT / "skills/monster-workflow/references/workflow-and-audit.md").read_text(encoding="utf-8")
            self.assertIn("validate_canonical_rig_bone_names", skill_md)
            self.assertIn("骨骼规范命名（强制）", sop_md)

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

            # 7. Universal 3-Layer Joint Coordination (intra_bone_coupling + inter_bone_couplings + body_part_volumes)
            jl_prof = tax.probe_rig_joint_limits(canon_bones, monster_id="test_coord_monster", archetype="humanoid")
            self.assertIn("coordination", jl_prof)
            self.assertEqual(jl_prof["coordination"]["schema"], "monster-joint-coordination-v1")
            self.assertGreater(len(jl_prof["coordination"]["inter_bone_couplings"]), 0)
            self.assertIn("body_part_volumes", jl_prof["coordination"])
            self.assertGreater(len(jl_prof["coordination"]["body_part_volumes"].get("bones", {})), 0)
            canon_classified = tax.apply_joint_limits_profile(dict(canon_bones), jl_prof)
            coord_rng = random.Random(108)
            coord_samples = []
            for idx in range(50):
                e_rad, reg, b_info = tax.sample_boundary_exploration_pose_euler_dict(
                    canon_classified,
                    rng=coord_rng,
                    sample_idx=idx,
                    total_samples=50,
                    joint_limits_profile=jl_prof,
                )
                self.assertIn("coordination_report", b_info)
                coord_samples.append({"sampling_regime": reg, "joints": e_rad})
            b_cov = tax.audit_boundary_exploration_coverage(coord_samples, canon_classified)
            self.assertTrue(b_cov["passed"], msg=f"Boundary audit failed under 3-layer coordination: {b_cov}")
            self.assertGreaterEqual(b_cov["dual_boundary_touch_rate"], 0.95)
            # Verify all 50 poses keep ROOT_C_00_00 yaw front-facing (|ry| <= 22 deg << 90 deg)
            root_yaws_deg = [abs(tax.math.degrees(s["joints"]["ROOT_C_00_00"][1])) for s in coord_samples]
            self.assertTrue(all(y <= 22.5 for y in root_yaws_deg), msg=f"Unexpected root yaws: {root_yaws_deg}")
            # Even in non-root regimes, ROOT yaw stays within the studio front window
            non_root_yaws = [
                abs(tax.math.degrees(s["joints"]["ROOT_C_00_00"][1]))
                for s in coord_samples
                if s["sampling_regime"] != "full_body_dynamic_root_boundary"
            ]
            self.assertTrue(all(y <= 16.0 for y in non_root_yaws))
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

    def test_three_view_feature_flag_and_occlusion_recovery(self):
        sys.path.insert(0, str(ROOT / "worker"))
        try:
            pose_features = load("monster_pose_features_3v", "worker/pose_features.py")
            pipeline_runner = load("monster_pipeline_runner_3v", "worker/pipeline_runner.py")
            bpy_mod = load("monster_bpy_worker_3v", "worker/studio_bpy_worker.py")

            with patch.dict("os.environ", {"MONSTER_ENABLE_2K": "0", "MONSTER_ENABLE_3VIEW": "0"}, clear=False):
                self.assertFalse(pose_features.three_view_enabled())
                self.assertEqual(pose_features.studio_view_order(), ("front", "side"))
                self.assertEqual(pose_features.studio_resolution(), (672, 768, 1344, 768))
                self.assertEqual(pose_features.idle_split_filename(), "idle_split_1344x768.png")
                prompt_2v = pipeline_runner.build_split_screen_prompt("slash")
                self.assertIn("two-panel", prompt_2v)

            with patch.dict("os.environ", {"MONSTER_ENABLE_2K": "0", "MONSTER_ENABLE_3VIEW": "1"}, clear=False):
                self.assertTrue(pose_features.three_view_enabled())
                self.assertEqual(pose_features.studio_view_order(), ("left", "front", "right"))
                self.assertEqual(pose_features.studio_resolution(), (448, 768, 1344, 768))
                self.assertEqual(pose_features.idle_split_filename(), "idle_3view_1344x768.png")
                self.assertEqual(pose_features.dinov2_panel_padding(448, 768), (1, 1, 0, 0))
                prompt_3v = pipeline_runner.build_split_screen_prompt("slash")
                self.assertIn("three-panel", prompt_3v)
                self.assertIn("Center panel: front orthographic view", prompt_3v)

            with patch.dict("os.environ", {"MONSTER_ENABLE_2K": "1", "MONSTER_ENABLE_3VIEW": "1"}, clear=False):
                self.assertEqual(pose_features.studio_resolution(), (896, 1440, 2688, 1440))
                self.assertEqual(pose_features.idle_split_filename(), "idle_3view_2688x1440.png")
                self.assertEqual(pose_features.dinov2_panel_padding(896, 1440), (1, 1, 0, 0))
                graph_3v_2k = pipeline_runner.build_minimax_h3_comfy_graph("walk", "ref3v.png")
                self.assertEqual(
                    graph_3v_2k["31"]["inputs"]["mode"],
                    {"mode": "target dimensions", "width": 2688, "height": 1440},
                )
                self.assertEqual(graph_3v_2k["33"]["inputs"]["width"], 2688)
                self.assertEqual(graph_3v_2k["33"]["inputs"]["height"], 1440)

            try:
                import numpy as np
            except ImportError:
                return

            from evals.solvers.latest_solver import run_latest_solver

            calib_3v = {
                "center": [0.0, 0.0, 0.5],
                "panel_width": 896,
                "panel_height": 1440,
                "composite_width": 2688,
                "composite_height": 1440,
                "num_views": 3,
                "view_order": ["left", "front", "right"],
                "ortho_scale": 1.2,
                "depth_scale": 0.8,
            }
            pt_gt = (-0.18, 0.22, 0.65)
            uv_f, uv_s = bpy_mod.project_3d_to_dual_ortho_pixels(pt_gt, calib_3v)
            uv_r = bpy_mod.project_3d_to_right_ortho_pixels(pt_gt, calib_3v)
            # Verify opposing lateral horizontal coordinates sum to panel_width: u_S + u_R == W
            self.assertAlmostEqual(uv_s[0] + uv_r[0], 896.0, places=2)
            self.assertAlmostEqual(uv_s[1], uv_r[1], places=2)

            # Simulate severe Left-view occlusion where uv_s is corrupted by +180px (wrong limb swap)
            uv_s_occluded = [uv_s[0] + 180.0, uv_s[1]]
            rec_2v = bpy_mod.triangulate_dual_ortho_pixels_to_3d(uv_f, uv_s_occluded, calib_3v, 0.9, 0.15)
            rec_3v = bpy_mod.triangulate_dual_ortho_pixels_to_3d(
                uv_f, uv_s_occluded, calib_3v, 0.9, 0.15, uv_right=uv_r, conf_right=0.95
            )
            err_y_2v = abs(rec_2v[1] - pt_gt[1])
            err_y_3v = abs(rec_3v[1] - pt_gt[1])
            self.assertGreaterEqual(err_y_2v, 0.149)
            self.assertLess(err_y_3v, 0.01)

            # Verify 3-view fuse_depth_visibility and run_latest_solver under occlusion
            T, K = 6, 2
            uf_arr = np.tile(np.array(uv_f, dtype=np.float64), (T, K, 1))
            us_arr = np.tile(np.array(uv_s_occluded, dtype=np.float64), (T, K, 1))
            ur_arr = np.tile(np.array(uv_r, dtype=np.float64), (T, K, 1))
            cf_arr = np.full((T, K), 0.92, dtype=np.float64)
            cs_arr = np.full((T, K), 0.20, dtype=np.float64)
            cr_arr = np.full((T, K), 0.95, dtype=np.float64)
            df_arr = np.full((T, K), (pt_gt[1] - 0.0) / 0.8, dtype=np.float64)
            ds_arr = np.full((T, K), (0.0 - pt_gt[0]) / 0.8, dtype=np.float64)
            dr_arr = np.full((T, K), (pt_gt[0] - 0.0) / 0.8, dtype=np.float64)
            vf_arr = np.full((T, K), 0.90, dtype=np.float64)
            vs_arr = np.full((T, K), 0.05, dtype=np.float64)
            vr_arr = np.full((T, K), 0.95, dtype=np.float64)
            lv_f = np.full((T, K), -2.0, dtype=np.float64)
            lv_s = np.full((T, K), 1.5, dtype=np.float64)
            lv_r = np.full((T, K), -3.5, dtype=np.float64)

            pts_2v, _ = pose_features.fuse_depth_visibility(
                uf_arr, us_arr, cf_arr, cs_arr, calib_3v, df_arr, ds_arr, vf_arr, vs_arr, lv_f, lv_s
            )
            pts_3v, _ = pose_features.fuse_depth_visibility(
                uf_arr, us_arr, cf_arr, cs_arr, calib_3v, df_arr, ds_arr, vf_arr, vs_arr, lv_f, lv_s,
                uv_right=ur_arr, conf_right=cr_arr, depth_right=dr_arr, visibility_right=vr_arr, logvar_right=lv_r,
            )
            self.assertLess(
                float(np.mean(np.abs(pts_3v[..., 1] - pt_gt[1]))),
                float(np.mean(np.abs(pts_2v[..., 1] - pt_gt[1]))) * 0.25,
            )
            self.assertLess(float(np.mean(np.abs(pts_3v[..., 1] - pt_gt[1]))), 0.01)

            studio_meta_3v = {
                "bbox_min": [-0.3, -0.3, 0.0],
                "bbox_max": [0.3, 0.3, 1.0],
                "calibration": calib_3v,
                "bones": {
                    "ROOT_C_00_00": {
                        "parent": None,
                        "rest_head_world": [pt_gt[0], pt_gt[1], pt_gt[2]],
                        "rest_tail_world": [pt_gt[0], pt_gt[1], pt_gt[2] + 0.10],
                        "rest_length": 0.10,
                        "rest_rot_mat_world": [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                    },
                },
            }
            obs_3v = {
                "uv_front_all": uf_arr,
                "uv_side_all": us_arr,
                "uv_right_all": ur_arr,
                "conf_front_all": cf_arr,
                "conf_side_all": cs_arr,
                "conf_right_all": cr_arr,
                "depth_front_all": df_arr,
                "depth_side_all": ds_arr,
                "depth_right_all": dr_arr,
                "visibility_front_all": vf_arr,
                "visibility_side_all": vs_arr,
                "visibility_right_all": vr_arr,
                "logvar_front_all": lv_f,
                "logvar_side_all": lv_s,
                "logvar_right_all": lv_r,
            }
            res_3v = run_latest_solver(obs_3v, studio_meta_3v)
            self.assertIn("uv_right_clean", res_3v.stage2_cleaned_2d)
            self.assertLess(float(np.mean(np.abs(res_3v.stage3_triangulation["pts_3d_raw"][..., 1] - pt_gt[1]))), 0.01)
        finally:
            sys.path.remove(str(ROOT / "worker"))

    def test_dense_surface_correspondence_and_lbs_roll_refinement(self):
        dino_src = (ROOT / "worker/dinov2_pose_trainer.py").read_text(encoding="utf-8")
        for sym in (
            "extract_multi_layer_patch_grid",
            "dino_multi_fuse",
            "rgb_detail_branch",
            "rgb_kp_refine",
            "dense_head",
            "dense_mask_logits",
            "dense_rest_xyz",
            "dense_logvar",
        ):
            self.assertIn(sym, dino_src)

        sys.path.insert(0, str(ROOT / "worker"))
        try:
            pose_features = load("monster_pose_features_dense", "worker/pose_features.py")
            tax = load("monster_bone_taxonomy_dense", "worker/bone_taxonomy.py")

            with tempfile.TemporaryDirectory() as tmp:
                b_file = Path(tmp) / "rig.blend"
                m_file = Path(tmp) / "meta.json"
                b_file.write_bytes(b"blend")
                m_file.write_text("{}")
                with patch.dict("os.environ", {}, clear=False):
                    import os
                    os.environ.pop("MONSTER_ENABLE_DENSE_SURFACE", None)
                    self.assertFalse(pose_features.dense_surface_enabled())
                with patch.dict("os.environ", {"MONSTER_ENABLE_DENSE_SURFACE": "0"}, clear=False):
                    self.assertFalse(pose_features.dense_surface_enabled())
                    sig_off = pose_features.dataset_signature(b_file, m_file, 100, 42)
                with patch.dict("os.environ", {"MONSTER_ENABLE_DENSE_SURFACE": "1"}, clear=False):
                    self.assertTrue(pose_features.dense_surface_enabled())
                    sig_on = pose_features.dataset_signature(b_file, m_file, 100, 42)
                    self.assertNotEqual(sig_on, sig_off)
                    self.assertEqual(pose_features.dense_grid_shape(672, 768), (220, 192))
                    self.assertEqual(pose_features.dense_grid_shape(448, 768), (220, 128))

            try:
                import numpy as np
            except ImportError:
                return

            from evals.solvers.latest_solver import run_latest_solver

            # Build a 2-bone rig where LIMB_L_01_01 is a horizontal leaf bone along +X
            # with off-axis surface vertices (a wide blade/paddle in the X-Y plane).
            # Pure axial roll (ry = +26 deg around local Y = +X world) leaves head_3d & tail_3d stationary!
            root_R = np.eye(3, dtype=np.float64)
            # Local Y along +X world, Local X along -Y world, Local Z along +Z world (det = +1)
            limb_R0 = np.array(
                [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            studio_meta_base = {
                "bbox_min": [-0.35, -0.35, 0.15],
                "bbox_max": [0.45, 0.35, 0.85],
                "bones": {
                    "ROOT_C_00_00": {
                        "parent": None,
                        "rest_head_world": [0.0, 0.0, 0.45],
                        "rest_tail_world": [0.0, 0.0, 0.60],
                        "rest_length": 0.15,
                        "rest_rot_mat_world": root_R.tolist(),
                    },
                    "LIMB_L_01_01": {
                        "parent": "ROOT_C_00_00",
                        "rest_head_world": [0.05, 0.0, 0.50],
                        "rest_tail_world": [0.35, 0.0, 0.50],
                        "rest_length": 0.30,
                        "rest_rot_mat_world": limb_R0.tolist(),
                    },
                },
            }

            # Construct a dense surface grid on the paddle attached to LIMB_L_01_01 (bone index 0 in sorted order)
            # Sorted bone_names = ["LIMB_L_01_01", "ROOT_C_00_00"]
            xs = np.linspace(0.06, 0.34, 14)
            ys = np.linspace(-0.14, 0.14, 14)
            verts_list = []
            for x in xs:
                for y in ys:
                    verts_list.append([float(x), float(y), 0.50])
            # Also add a few root vertices
            for z in np.linspace(0.42, 0.58, 4):
                for x in (-0.02, 0.02):
                    verts_list.append([float(x), 0.0, float(z)])
            rest_verts = np.asarray(verts_list, dtype=np.float32)
            V_limb = 14 * 14
            V_total = int(rest_verts.shape[0])

            tris = []
            for ix in range(13):
                for iy in range(13):
                    i0 = ix * 14 + iy
                    i1 = i0 + 1
                    i2 = (ix + 1) * 14 + iy
                    i3 = i2 + 1
                    tris.append([i0, i2, i1])
                    tris.append([i1, i2, i3])
            triangles = np.asarray(tris, dtype=np.int32)

            skin_indices = np.zeros((V_total, 4), dtype=np.int32)
            skin_weights = np.zeros((V_total, 4), dtype=np.float32)
            # Bone 0 = LIMB_L_01_01, Bone 1 = ROOT_C_00_00
            skin_indices[:V_limb, 0] = 0
            skin_weights[:V_limb, 0] = 1.0
            skin_indices[V_limb:, 0] = 1
            skin_weights[V_limb:, 0] = 1.0

            rest_heads = np.array([[0.05, 0.0, 0.50], [0.0, 0.0, 0.45]], dtype=np.float32)
            rest_R = np.stack([limb_R0, root_R], axis=0).astype(np.float32)
            rig_skinning = {
                "bone_names": ["LIMB_L_01_01", "ROOT_C_00_00"],
                "rest_verts": rest_verts,
                "skin_indices": skin_indices,
                "skin_weights": skin_weights,
                "triangles": triangles,
                "rest_heads": rest_heads,
                "rest_R": rest_R,
            }

            # Apply a +26 deg pure axial roll (ry) on LIMB_L_01_01
            true_ry = float(np.deg2rad(26.0))
            R_local_roll = tax.euler_xyz_to_matrix_numpy(0.0, true_ry, 0.0)
            limb_R_posed = limb_R0 @ R_local_roll
            posed_heads = rest_heads.copy()
            posed_R = np.stack([limb_R_posed, root_R], axis=0)
            posed_verts = pose_features.lbs_deform_vertices_numpy(
                rest_verts, skin_indices, skin_weights, rest_heads, rest_R, posed_heads, posed_R
            )

            # Test both 2-view ("front", "side") and 3-view ("left", "front", "right") studio configurations
            for mode_3view in (False, True):
                calib = {
                    "center": [0.05, 0.0, 0.50],
                    "panel_width": 448 if mode_3view else 672,
                    "panel_height": 768,
                    "composite_width": 1344,
                    "composite_height": 768,
                    "num_views": 3 if mode_3view else 2,
                    "view_order": ["left", "front", "right"] if mode_3view else ["front", "side"],
                    "ortho_scale": 0.9,
                    "depth_scale": 0.7,
                }
                hm_h, hm_w = pose_features.dense_grid_shape(calib["panel_width"], calib["panel_height"])
                dense_views = {}
                for v_name in calib["view_order"]:
                    rast = pose_features.rasterize_ortho_dense_rest_map_numpy(
                        posed_verts, rest_verts, triangles, calib, view=v_name, grid_w=hm_w, grid_h=hm_h
                    )
                    dense_views[v_name] = {
                        "mask_prob": rast["mask"],
                        "rest_xyz": rast["rest_xyz"],
                        "logvar": np.full((hm_h, hm_w), -3.0, dtype=np.float32),
                    }
                matched_list = pose_features.match_dense_surface_correspondences(dense_views, rig_skinning, calib)
                self.assertEqual(len(matched_list), 1)
                matched = matched_list[0]
                self.assertIn("front", matched)
                self.assertIn("side", matched)
                if mode_3view:
                    self.assertIn("right", matched)

                # Build 7-frame observation sequence where keypoints (head, tail) are completely stationary
                T, K = 7, 4
                bone_endpoints_3d = np.array(
                    [
                        [0.05, 0.0, 0.50],  # LIMB_L_01_01 head
                        [0.35, 0.0, 0.50],  # LIMB_L_01_01 tail
                        [0.0, 0.0, 0.45],   # ROOT_C_00_00 head
                        [0.0, 0.0, 0.60],   # ROOT_C_00_00 tail
                    ],
                    dtype=np.float64,
                )
                uv_f, df = pose_features.project_3d_points_to_ortho_view_numpy(bone_endpoints_3d, calib, "front")
                uv_s, ds = pose_features.project_3d_points_to_ortho_view_numpy(bone_endpoints_3d, calib, "side")
                obs = {
                    "uv_front_all": np.tile(uv_f[None, :, :], (T, 1, 1)),
                    "uv_side_all": np.tile(uv_s[None, :, :], (T, 1, 1)),
                    "conf_front_all": np.full((T, K), 0.95, dtype=np.float64),
                    "conf_side_all": np.full((T, K), 0.95, dtype=np.float64),
                    "depth_front_all": np.tile(df[None, :], (T, 1)),
                    "depth_side_all": np.tile(ds[None, :], (T, 1)),
                    "visibility_front_all": np.ones((T, K), dtype=np.float64),
                    "visibility_side_all": np.ones((T, K), dtype=np.float64),
                    "logvar_front_all": np.full((T, K), -4.0, dtype=np.float64),
                    "logvar_side_all": np.full((T, K), -4.0, dtype=np.float64),
                    "dense_correspondences": [matched for _ in range(T)],
                }
                if mode_3view:
                    uv_r, dr = pose_features.project_3d_points_to_ortho_view_numpy(bone_endpoints_3d, calib, "right")
                    obs.update({
                        "uv_right_all": np.tile(uv_r[None, :, :], (T, 1, 1)),
                        "conf_right_all": np.full((T, K), 0.95, dtype=np.float64),
                        "depth_right_all": np.tile(dr[None, :], (T, 1)),
                        "visibility_right_all": np.ones((T, K), dtype=np.float64),
                        "logvar_right_all": np.full((T, K), -4.0, dtype=np.float64),
                    })

                studio_meta = {**studio_meta_base, "calibration": calib, "rig_skinning": rig_skinning}

                def _geodesic_deg(R_a, R_b):
                    tr = float(np.clip((np.trace(R_a.T @ R_b) - 1.0) * 0.5, -1.0, 1.0))
                    return float(np.rad2deg(np.arccos(tr)))

                # 1) With feature flag OFF: keypoint-only solver cannot observe leaf axial roll (~26 deg error)
                with patch.dict("os.environ", {"MONSTER_ENABLE_DENSE_SURFACE": "0"}, clear=False):
                    res_off = run_latest_solver(obs, studio_meta)
                    self.assertFalse(res_off.stage4_diagnostics["dense_surface_refined"])
                    R_pred_off = np.asarray(res_off.stage4_frames[3]["bones"]["LIMB_L_01_01"]["rot_mat_world"])
                    err_off_deg = _geodesic_deg(R_pred_off, limb_R_posed)
                    self.assertGreater(err_off_deg, 20.0)

                # 2) With feature flag ON: multi-view dense LBS refiner unlocks and recovers the +26 deg axial roll!
                with patch.dict("os.environ", {"MONSTER_ENABLE_DENSE_SURFACE": "1"}, clear=False):
                    res_on = run_latest_solver(obs, studio_meta)
                    self.assertTrue(res_on.stage4_diagnostics["dense_surface_refined"])
                    self.assertGreater(res_on.stage4_diagnostics["dense_matched_points_total"], 0)
                    self.assertIn("LIMB_L_01_01", res_on.stage4_diagnostics["unlocked_roll_bones"])
                    self.assertLess(
                        res_on.stage4_diagnostics["mean_reproj_error_px_after"],
                        res_on.stage4_diagnostics["mean_reproj_error_px_before"],
                    )
                    R_pred_on = np.asarray(res_on.stage4_frames[3]["bones"]["LIMB_L_01_01"]["rot_mat_world"])
                    err_on_deg = _geodesic_deg(R_pred_on, limb_R_posed)
                    self.assertLess(err_on_deg, 2.0)
        finally:
            sys.path.remove(str(ROOT / "worker"))

    def test_mocap_priors_feature_flag_and_xsxb_mechanisms(self):
        import os

        sys.path.insert(0, str(ROOT / "worker"))
        try:
            pose_features = load("monster_pose_features_priors", "worker/pose_features.py")
            mocap_priors = load("monster_mocap_priors", "worker/mocap_priors.py")
            tax = load("monster_bone_taxonomy_priors", "worker/bone_taxonomy.py")

            with patch.dict("os.environ", {}, clear=False):
                os.environ.pop("MONSTER_ENABLE_MOCAP_PRIORS", None)
                self.assertTrue(pose_features.mocap_priors_enabled())
            with patch.dict("os.environ", {"MONSTER_ENABLE_MOCAP_PRIORS": "0"}, clear=False):
                self.assertFalse(pose_features.mocap_priors_enabled())
            with patch.dict("os.environ", {"MONSTER_ENABLE_MOCAP_PRIORS": "invalid"}, clear=False):
                with self.assertRaises(ValueError):
                    pose_features.mocap_priors_enabled()

            try:
                import numpy as np
            except ImportError:
                return

            with patch.dict("os.environ", {"MONSTER_ENABLE_MOCAP_PRIORS": "1"}, clear=False):
                # 1. Per-Frame View Scale & Vertical Drift Calibration
                rng = np.random.default_rng(123)
                T, K = 14, 10
                calib = {"panel_width": 1280.0, "panel_height": 1440.0}
                uv_side = rng.uniform(240.0, 1180.0, size=(T, K, 2))
                uv_front_clean = uv_side.copy()
                uv_front_clean[..., 0] = rng.uniform(220.0, 1020.0, size=(T, K))
                conf = np.full((T, K), 0.92, dtype=np.float64)
                vis = np.full((T, K), 0.90, dtype=np.float64)

                # Zero drift -> deadband keeps exact coordinates
                uv_f0, _, diag0 = mocap_priors.calibrate_view_scale_drift(
                    uv_front_clean, uv_side, conf, conf, calib, vis_front=vis, vis_side=vis
                )
                self.assertFalse(diag0["view_drift_calibrated"])
                self.assertTrue(np.allclose(uv_f0, uv_front_clean))

                # True camera zoom (1.035x) + vertical drift (+20px) -> calibrated accurately
                u0, v0 = 640.0, 720.0
                uv_front_drifted = uv_front_clean.copy()
                uv_front_drifted[..., 0] = (uv_front_clean[..., 0] - u0) / 1.035 + u0
                uv_front_drifted[..., 1] = (uv_front_clean[..., 1] - 20.0 - v0) / 1.035 + v0
                uv_f1, _, diag1 = mocap_priors.calibrate_view_scale_drift(
                    uv_front_drifted, uv_side, conf, conf, calib, vis_front=vis, vis_side=vis
                )
                self.assertTrue(diag1["view_drift_calibrated"])
                v_err_after = float(np.mean(np.abs(uv_f1[..., 1] - uv_side[..., 1])))
                self.assertLess(v_err_after, 1.5)

                # 2. Local Windowed Peak Soft-Argmax (Multimodal Secondary Lobe Protection)
                H_hm, W_hm = 32, 32
                gx = np.arange(W_hm, dtype=np.float64) * 4.0
                gy = np.arange(H_hm, dtype=np.float64) * 4.0
                prob = np.zeros((1, 1, H_hm, W_hm), dtype=np.float64)
                prob[0, 0, 7, 9] = 0.72    # Primary joint peak at (36.0, 28.0)
                prob[0, 0, 25, 27] = 0.28  # Secondary occlusion peak at (108.0, 100.0)
                u_g = np.sum(prob * gx[None, None, None, :], axis=(-2, -1))
                v_g = np.sum(prob * gy[None, None, :, None], axis=(-2, -1))
                uv_g = np.stack([u_g, v_g], axis=-1)
                uv_w = mocap_priors.windowed_peak_dsnt_numpy(prob, gx, gy, uv_g)
                self.assertGreater(float(np.linalg.norm(uv_g[0, 0] - np.array([36.0, 28.0]))), 15.0)
                self.assertLess(float(np.linalg.norm(uv_w[0, 0] - np.array([36.0, 28.0]))), 1.0)

                # 3. IRLS Charbonnier Acceleration Smoother & Velocity-Gated De-Jitter
                sig = np.zeros((25, 3), dtype=np.float64)
                sig[11:14, 0] = [4.5, 10.0, 4.5]
                sm_quad = mocap_priors.whittaker_charbonnier_smooth_trajectory(sig, irls_iters=1)
                sm_charb = mocap_priors.whittaker_charbonnier_smooth_trajectory(
                    sig, irls_iters=4, charbonnier_sigma=2.0
                )
                self.assertGreater(float(sm_charb[12, 0]), float(sm_quad[12, 0]) + 1.5)

                # Stationary micro-jitter vs fast motion peak under dejitter_velocity_gated
                jitter_sig = np.zeros((20, 3), dtype=np.float64)
                jitter_sig[:10, 0] = np.array([0.02, -0.02] * 5, dtype=np.float64)
                jitter_sig[14, 0] = 5.0
                dejit = mocap_priors.dejitter_velocity_gated(jitter_sig, v_fast=0.5, passes=2, strength=0.8)
                self.assertLess(float(np.std(dejit[2:8, 0])), float(np.std(jitter_sig[2:8, 0])) * 0.5)
                self.assertGreater(float(dejit[14, 0]), 4.8)

                # 4. Floor Penetration Clamp & Contact-Aware Foot Anti-Skate Locking
                order = ["ROOT_C_00_00", "LIMB_L_01_01"]
                classified = {
                    "ROOT_C_00_00": {"kind": "root", "side": "C", "parent": None, "rest_length": 0.20, "fixed_local": False},
                    "LIMB_L_01_01": {"kind": "limb", "side": "L", "parent": "ROOT_C_00_00", "rest_length": 0.40, "fixed_local": False},
                }
                rest_H = {
                    "ROOT_C_00_00": np.array([0.0, 0.0, 0.40]),
                    "LIMB_L_01_01": np.array([0.0, 0.0, 0.40]),
                }
                rest_T = {
                    "ROOT_C_00_00": np.array([0.0, 0.0, 0.60]),
                    "LIMB_L_01_01": np.array([0.0, 0.0, 0.00]),  # Foot tip at z_floor = 0.0
                }
                R_root0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
                R_limb0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
                rest_R = {"ROOT_C_00_00": R_root0, "LIMB_L_01_01": R_limb0}
                eff_lims = {
                    "ROOT_C_00_00": [[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]],
                    "LIMB_L_01_01": [[-0.8, 0.8], [-0.5, 0.5], [-0.8, 0.8]],
                }
                frames_test = []
                for f in range(6):
                    # Frame 5 dips 0.08 below floor; frames 0..4 have tiny horizontal foot skate
                    z_dip = -0.08 if f == 5 else 0.0
                    skate_x = 0.002 * (f - 2) if f < 5 else 0.0
                    v_dir = np.array([skate_x, 0.0, -0.40], dtype=np.float64)
                    v_dir /= np.linalg.norm(v_dir)
                    R_sw = tax.minimum_swing_matrix_numpy(R_limb0[:, 1], v_dir)
                    R_w_limb = R_sw @ R_limb0
                    head_p = [0.0, 0.0, 0.40 + z_dip]
                    tail_p = (np.array(head_p) + R_w_limb[:, 1] * 0.40).tolist()
                    frames_test.append({
                        "frame": f,
                        "bones": {
                            "ROOT_C_00_00": {
                                "head_3d": head_p,
                                "tail_3d": [0.0, 0.0, 0.60 + z_dip],
                                "rot_mat_world": R_root0.tolist(),
                                "confidence": 0.9,
                            },
                            "LIMB_L_01_01": {
                                "head_3d": head_p,
                                "tail_3d": tail_p,
                                "rot_mat_world": R_w_limb.tolist(),
                                "confidence": 0.9,
                            },
                        },
                    })
                c_diag = mocap_priors.apply_floor_and_foot_contact_constraints(
                    frames_out=frames_test,
                    order=order,
                    root="ROOT_C_00_00",
                    classified_bones=classified,
                    rest_H=rest_H,
                    rest_T=rest_T,
                    rest_R=rest_R,
                    effective_base_limits=eff_lims,
                    frame_macro_collapse=np.zeros(6, dtype=np.float64),
                    clamp_local_delta_euler_fn=tax.clamp_local_delta_euler,
                    minimum_swing_matrix_fn=tax.minimum_swing_matrix_numpy,
                )
                self.assertEqual(c_diag["floor_clamped_frames"], 1)
                self.assertGreaterEqual(c_diag["foot_contact_locked_frames"], 3)
                self.assertGreaterEqual(float(frames_test[5]["bones"]["LIMB_L_01_01"]["tail_3d"][2]), -0.015)

                # 5. Stage 2 Edge Stretch Metric & Ghost Afterimage Compositing
                v_rest = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
                faces = np.array([[0, 1, 2]], dtype=np.int64)
                m_ok = mocap_priors.compute_mesh_edge_stretch_metric(v_rest, v_rest * 1.05, faces)
                self.assertTrue(m_ok["passed"])
                v_torn = v_rest.copy()
                v_torn[1, 0] = 3.2  # 3.2x stretch > 1.8 threshold
                m_bad = mocap_priors.compute_mesh_edge_stretch_metric(v_rest, v_torn, faces)
                self.assertFalse(m_bad["passed"])

                main_rgba = np.zeros((8, 8, 4), dtype=np.uint8)
                main_rgba[2:6, 2:6] = [220, 40, 40, 255]
                ghost_rgba = np.zeros((8, 8, 4), dtype=np.uint8)
                ghost_rgba[2:6, 4:8] = [40, 200, 40, 255]
                comp_rgba = mocap_priors.blend_limb_afterimage_ghost_rgba(main_rgba, [ghost_rgba], ghost_opacity=0.35)
                # Opaque main region stays intact while trailing ghost region becomes translucent
                self.assertTrue(np.array_equal(comp_rgba[3, 3], [220, 40, 40, 255]))
                self.assertGreater(int(comp_rgba[3, 7, 1]), 150)
                self.assertGreater(int(comp_rgba[3, 7, 3]), 50)
                self.assertLess(int(comp_rgba[3, 7, 3]), 200)
        finally:
            sys.path.remove(str(ROOT / "worker"))

    def test_quadruped_articulation_and_root_translation_regression(self):
        import random
        import sys
        import numpy as np

        sys.path.insert(0, str(ROOT / "worker"))
        try:
            import bone_taxonomy as tax
            import mocap_priors

            # 1. Quadruped taxonomy: curled tail propagation, midline neck/head/jaw split, and 4-segment hind leg
            quad_meta = {
                "bone_0": {
                    "parent": None,
                    "role": "root",
                    "rest_head_world": [0.0, -0.35, 1.50],
                    "rest_tail_world": [0.0, -0.58, 1.50],
                },
                "bone_1": {
                    "parent": "bone_0",
                    "role": "spine",
                    "rest_head_world": [0.0, -0.35, 1.50],
                    "rest_tail_world": [0.0, 0.14, 1.48],
                },
                "bone_2": {
                    "parent": "bone_1",
                    "role": "pelvis",
                    "rest_head_world": [0.0, 0.14, 1.48],
                    "rest_tail_world": [0.0, 0.45, 1.47],
                },
                # Curled tail chain whose distal segments bend laterally past midline_tol_x
                "bone_3": {
                    "parent": "bone_2",
                    "role": "tail_base",
                    "rest_head_world": [0.0, 0.45, 1.47],
                    "rest_tail_world": [0.02, 0.72, 1.38],
                },
                "bone_4": {
                    "parent": "bone_3",
                    "role": "tail",
                    "rest_head_world": [0.02, 0.72, 1.38],
                    "rest_tail_world": [0.18, 0.95, 1.25],
                },
                "bone_5": {
                    "parent": "bone_4",
                    "role": "tail",
                    "rest_head_world": [0.18, 0.95, 1.25],
                    "rest_tail_world": [0.32, 1.10, 1.18],
                },
                # Midline neck branching into upper head + lower jaw
                "bone_29": {
                    "parent": "bone_0",
                    "role": "neck",
                    "rest_head_world": [0.0, -0.58, 1.50],
                    "rest_tail_world": [0.0, -0.85, 1.69],
                },
                "bone_30": {
                    "parent": "bone_29",
                    "role": "head",
                    "rest_head_world": [0.0, -0.85, 1.69],
                    "rest_tail_world": [0.0, -1.03, 1.71],
                },
                "bone_32": {
                    "parent": "bone_29",
                    "role": "jaw",
                    "rest_head_world": [0.0, -0.85, 1.69],
                    "rest_tail_world": [0.0, -1.02, 1.51],
                },
                # 4-segment left hind leg (hip -> knee -> hock -> paw)
                "bone_14": {
                    "parent": "bone_2",
                    "role": "pelvis_attachment_L",
                    "fixed_local": True,
                    "rest_head_world": [0.0, 0.45, 1.47],
                    "rest_tail_world": [0.22, 0.43, 1.44],
                },
                "bone_15": {
                    "parent": "bone_14",
                    "role": "hind_hip_L",
                    "rest_head_world": [0.22, 0.43, 1.44],
                    "rest_tail_world": [0.29, 0.34, 1.10],
                },
                "bone_16": {
                    "parent": "bone_15",
                    "role": "hind_knee_L",
                    "rest_head_world": [0.29, 0.34, 1.10],
                    "rest_tail_world": [0.33, 0.56, 0.72],
                },
                "bone_17": {
                    "parent": "bone_16",
                    "role": "hind_hock_L",
                    "rest_head_world": [0.33, 0.56, 0.72],
                    "rest_tail_world": [0.33, 0.43, 0.25],
                },
                "bone_18": {
                    "parent": "bone_17",
                    "role": "hind_paw_L",
                    "rest_head_world": [0.33, 0.43, 0.25],
                    "rest_tail_world": [0.34, 0.22, 0.05],
                },
            }
            classified = tax.classify_rig_bones(
                quad_meta, bbox_min=[-0.6, -1.2, 0.0], bbox_max=[0.6, 1.2, 2.3]
            )
            self.assertEqual(classified["bone_3"]["kind"], "tail")
            self.assertEqual(classified["bone_4"]["kind"], "tail")
            self.assertEqual(classified["bone_5"]["kind"], "tail")
            self.assertEqual(classified["bone_29"]["kind"], "head")
            self.assertEqual(classified["bone_30"]["kind"], "head")
            self.assertEqual(classified["bone_32"]["kind"], "jaw")
            self.assertEqual(classified["bone_15"]["kind"], "limb")
            self.assertEqual(classified["bone_15"]["sub_role"], "ball_socket")
            self.assertEqual(classified["bone_16"]["sub_role"], "forearm_hinge")
            self.assertEqual(classified["bone_17"]["sub_role"], "forearm_hinge")
            self.assertEqual(classified["bone_18"]["sub_role"], "distal_limb")

            # 2. Fallback rest_R (+X-aligned) and ball_socket child-plane twist recovery
            d0_hip = np.array([0.29 - 0.22, 0.34 - 0.43, 1.10 - 1.44], dtype=np.float64)
            d0_hip /= np.linalg.norm(d0_hip)
            R0_hip = tax._build_default_rest_rot_matrix(d0_hip)
            self.assertAlmostEqual(float(np.linalg.det(R0_hip)), 1.0, places=6)
            self.assertGreater(float(R0_hip[0, 0]), 0.90)

            # Apply a +22 deg axial twist on bone_15 (ball_socket) and -35 deg knee flexion on bone_16
            R_twist = tax.euler_xyz_to_matrix_numpy(np.deg2rad(15.0), np.deg2rad(22.0), np.deg2rad(-8.0))
            R_hip_gt = R0_hip @ R_twist
            d0_knee = np.array([0.33 - 0.29, 0.56 - 0.34, 0.72 - 1.10], dtype=np.float64)
            len_knee = float(np.linalg.norm(d0_knee))
            R0_knee = tax._build_default_rest_rot_matrix(d0_knee / len_knee)
            R_knee_rel = R0_hip.T @ R0_knee
            R_knee_gt = (R_hip_gt @ R_knee_rel) @ tax.euler_xyz_to_matrix_numpy(np.deg2rad(-35.0), 0.0, 0.0)

            hip_head = np.array([0.22, 0.43, 1.44], dtype=np.float64)
            len_hip = float(np.linalg.norm([0.29 - 0.22, 0.34 - 0.43, 1.10 - 1.44]))
            hip_tail = hip_head + R_hip_gt[:, 1] * len_hip
            knee_tail = hip_tail + R_knee_gt[:, 1] * len_knee

            sorted_qnames = sorted(classified.keys())
            pts_3d_q = np.zeros((1, 2 * len(sorted_qnames), 3), dtype=np.float64)
            trust_q = np.full((1, 2 * len(sorted_qnames)), 0.95, dtype=np.float64)
            for b_i, bn in enumerate(sorted_qnames):
                pts_3d_q[0, 2 * b_i] = quad_meta[bn]["rest_head_world"]
                pts_3d_q[0, 2 * b_i + 1] = quad_meta[bn]["rest_tail_world"]
            # Inject slight separation at the coincident hip_tail / knee_head joint to test joint fusion
            idx_15 = sorted_qnames.index("bone_15")
            idx_16 = sorted_qnames.index("bone_16")
            idx_17 = sorted_qnames.index("bone_17")
            pts_3d_q[0, 2 * idx_15 + 1] = hip_tail + np.array([0.008, -0.006, 0.004])
            pts_3d_q[0, 2 * idx_16] = hip_tail - np.array([0.008, -0.006, 0.004])
            pts_3d_q[0, 2 * idx_16 + 1] = knee_tail
            pts_3d_q[0, 2 * idx_17] = knee_tail

            frames_limb, _ = tax.reconstruct_frames_with_kind_solvers(
                pts_3d=pts_3d_q,
                trust=trust_q,
                bone_names=sorted_qnames,
                classified_bones=classified,
            )
            R_hip_sol = np.array(frames_limb[0]["bones"]["bone_15"]["rot_mat_world"], dtype=np.float64)
            tr_hip = float(np.clip((np.trace(R_hip_sol.T @ R_hip_gt) - 1.0) * 0.5, -1.0, 1.0))
            err_hip_deg = float(np.rad2deg(np.arccos(tr_hip)))
            self.assertLess(err_hip_deg, 3.0)

            # 3. Stationary-root floor clamp gating: zero root drop must not lift root upward
            stationary_frames = [
                {
                    "frame": f,
                    "bones": {
                        b: {
                            "head_3d": list(quad_meta[b]["rest_head_world"]),
                            "tail_3d": list(quad_meta[b]["rest_tail_world"]),
                            "rot_mat_world": np.eye(3).tolist(),
                            "confidence": 0.95,
                        }
                        for b in quad_meta
                    },
                }
                for f in range(4)
            ]
            # Dip distal paw slightly below floor while root stays at rest height
            stationary_frames[2]["bones"]["bone_18"]["tail_3d"][2] = 0.01
            order_q = sorted(classified.keys(), key=lambda b: int(classified[b].get("depth", 0)))
            rest_H_q = {b: np.array(classified[b]["rest_head_world"], dtype=np.float64) for b in classified}
            rest_T_q = {b: np.array(classified[b]["rest_tail_world"], dtype=np.float64) for b in classified}
            rest_R_q = {b: np.eye(3, dtype=np.float64) for b in classified}
            eff_lims_q = {b: classified[b]["limits_rad"] for b in classified}
            diag_stat = mocap_priors.apply_floor_and_foot_contact_constraints(
                frames_out=stationary_frames,
                order=order_q,
                root="bone_0",
                classified_bones=classified,
                rest_H=rest_H_q,
                rest_T=rest_T_q,
                rest_R=rest_R_q,
                effective_base_limits=eff_lims_q,
                frame_macro_collapse=np.zeros(4, dtype=np.float64),
                clamp_local_delta_euler_fn=tax.clamp_local_delta_euler,
                minimum_swing_matrix_fn=tax.minimum_swing_matrix_numpy,
            )
            self.assertEqual(diag_stat["floor_clamped_frames"], 0)
            self.assertAlmostEqual(float(stationary_frames[2]["bones"]["bone_0"]["head_3d"][2]), 1.50, places=5)

            # 4. Root translation sampling helper stays within biomechanical workspace bounds
            rng = random.Random(42)
            for s_idx in range(8):
                dx, dy, dz = tax.sample_root_translation_offset(classified, rng, sample_idx=s_idx)
                self.assertLessEqual(abs(dx), 0.16 + 1e-6)
                self.assertLessEqual(abs(dy), 0.20 + 1e-6)
                self.assertLessEqual(abs(dz), 0.20 + 1e-6)
        finally:
            sys.path.remove(str(ROOT / "worker"))


class MonsterSopExecutionSuiteTests(unittest.TestCase):
    """End-to-end tests for the solidified SOP execution scripts in skills/monster-workflow/scripts/."""

    def _create_minimal_glb(self, path: Path, nodes: list[dict] | None = None, meshes: list[dict] | None = None) -> Path:
        import struct

        gltf = {
            "asset": {"version": "2.0"},
            "meshes": meshes or [{"name": "BodyMesh", "primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}}]}],
            "materials": [{"name": "Mat0"}],
            "nodes": nodes
            or [
                {"name": "ROOT_C_00_00", "children": [1]},
                {"name": "SPINE_C_00_01", "translation": [0.0, 0.0, 0.4]},
            ],
            "skins": [{"joints": [0, 1]}],
            "accessors": [{"min": [-0.4, -0.3, 0.0], "max": [0.4, 0.3, 1.8]}],
        }
        json_bytes = json.dumps(gltf).encode("utf-8")
        pad = (4 - (len(json_bytes) % 4)) % 4
        json_bytes += b" " * pad
        total_len = 12 + 8 + len(json_bytes)
        header = struct.pack("<4sII", b"glTF", 2, total_len)
        chunk_hdr = struct.pack("<I4s", len(json_bytes), b"JSON")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(header + chunk_hdr + json_bytes)
        return path

    def test_sop_gate_ledger_fingerprints_and_10_invalidation_rules(self):
        scripts_dir = str(ROOT / "skills/monster-workflow/scripts")
        sys.path.insert(0, scripts_dir)
        try:
            import sop_gate

            with tempfile.TemporaryDirectory() as tmp:
                proj = Path(tmp) / "project"
                drv = Path(tmp) / "drive"
                ledger = sop_gate.load_ledger("troll_01", project_root=proj, drive_root=drv)
                self.assertEqual(ledger["gates"]["G1"]["user_decision"], "pending")

                # Unapproved G1 must block downstream G2
                with self.assertRaisesRegex(RuntimeError, "SOP 门禁拦截"):
                    sop_gate.assert_gate(ledger, "G1")

                # Anonymous bones must fail canonical bone gate
                meta_bad = Path(tmp) / "bad_meta.json"
                meta_bad.write_text(json.dumps({"bones": {"bone_0": {}, "Bone.001": {}}}))
                bad_bones = sop_gate.verify_rig_meta_canonical_bones(meta_bad)
                self.assertFalse(bad_bones["valid"])

                meta_good = Path(tmp) / "good_meta.json"
                meta_good.write_text(json.dumps({"bones": {"ROOT_C_00_00": {}, "SPINE_C_00_01": {}}}))
                good_bones = sop_gate.verify_rig_meta_canonical_bones(meta_good)
                self.assertTrue(good_bones["valid"])

                # Record user approval for G1, G2, G3
                sop_gate.record_user_decision(
                    ledger,
                    "G1",
                    decision="approved",
                    user_quote="选定 candidate_0，骨骼命名与装饰随动确认通过",
                    selected_candidate_id="0",
                    rig_meta_path=str(meta_good),
                )
                self.assertEqual(sop_gate.assert_gate(ledger, "G1")["user_decision"], "approved")

                sop_gate.record_user_decision(
                    ledger,
                    "G2",
                    decision="approved",
                    user_quote="50 个随机姿势幅度与重抽策略通过",
                    seed=42,
                )
                sop_gate.record_user_decision(
                    ledger,
                    "G3",
                    decision="approved",
                    user_quote="已有视频考试还原效果通过",
                )

                # Verify graded invalidation: sampler_script_or_seed_change preserves G1, invalidates G2 & G3
                inv_sample = sop_gate.derive_invalidations(ledger, change_type="sampler_script_or_seed_change")
                inv_gates = {item["gate"] for item in inv_sample}
                self.assertEqual(ledger["gates"]["G1"]["user_decision"], "approved")
                self.assertIn("G2", inv_gates)
                self.assertIn("G3", inv_gates)
                self.assertEqual(ledger["gates"]["G2"]["user_decision"], "pending")

                # Re-approve G2 & G3, then verify mocap_priors_change preserves G1 & G2 and only invalidates G3
                sop_gate.record_user_decision(ledger, "G2", decision="approved", user_quote="重审通过", seed=42)
                sop_gate.record_user_decision(ledger, "G3", decision="approved", user_quote="重审通过")
                inv_mocap = sop_gate.derive_invalidations(ledger, change_type="mocap_priors_change")
                self.assertEqual({item["gate"] for item in inv_mocap}, {"G3"})
                self.assertEqual(ledger["gates"]["G1"]["user_decision"], "approved")
                self.assertEqual(ledger["gates"]["G2"]["user_decision"], "approved")
                self.assertEqual(ledger["gates"]["G3"]["user_decision"], "pending")
        finally:
            sys.path.remove(scripts_dir)

    def test_sop_preflight_redlines_and_g1_g2_g3_delivery_colab_guard(self):
        import os
        import shutil

        scripts_dir = str(ROOT / "skills/monster-workflow/scripts")
        sys.path.insert(0, scripts_dir)
        os.environ["MONSTER_SKIP_PS_SCAN"] = "1"
        try:
            import monster_sop
            import sop_colab_guard
            import sop_delivery
            import sop_g1_pack
            import sop_g2_pack
            import sop_g3_exam
            import sop_gate
            import sop_preflight

            with tempfile.TemporaryDirectory() as tmp:
                proj_root = Path(tmp) / "project"
                drive_root = Path(tmp) / "drive"
                src_glb = self._create_minimal_glb(proj_root / "scratch/raw/troll_source.glb")

                # No Blender: a structural GLB fixture is never a valid preflight/U1 result.
                with patch.object(sop_preflight, "_worker_command", return_value=None):
                    with self.assertRaisesRegex(RuntimeError, "bpy/Blender"):
                        sop_preflight.run_preflight("troll_01", src_glb, project_root=proj_root)
                report_path = sop_preflight.resolve_scratch_dirs(proj_root, "troll_01")["checks_dir"] / "preflight_report.json"
                self.assertFalse(json.loads(report_path.read_text())["passed"])
                with self.assertRaises(RuntimeError):
                    sop_preflight.sync_u1_to_drive("troll_01", project_root=proj_root, drive_root=drive_root)

                # Verify art_source redline violation is caught if intermediate file is placed in art_source/
                bad_dir = proj_root / "art_source/troll_01/v1"
                bad_dir.mkdir(parents=True, exist_ok=True)
                (bad_dir / "candidate_0.glb").write_bytes(b"leak")
                redline_fail = sop_preflight.check_workspace_redlines(proj_root, "troll_01")
                self.assertFalse(redline_fail["passed"])
                shutil.rmtree(proj_root / "art_source")

                # 2. Stage 2 G1 Pack
                dirs = sop_preflight.resolve_scratch_dirs(proj_root, "troll_01")
                work_dir = dirs["work_dir"]
                for idx in range(4):
                    self._create_minimal_glb(work_dir / "candidates" / f"candidate_{idx}.glb")

                # No Blender: G1 pack must fail and never fabricate stick figures or 24-byte MP4s.
                with patch.object(sop_g1_pack, "_worker_command", return_value=None):
                    with self.assertRaisesRegex(RuntimeError, "bpy/Blender"):
                        sop_g1_pack.build_g1_review_pack(
                            monster_id="troll_01",
                            archetype="humanoid",
                            project_root=proj_root,
                            drive_root=drive_root,
                        )

                def _fake_g1_worker(cmd, **kwargs):
                    from PIL import Image

                    stage_out = Path(cmd[cmd.index("--out_dir") + 1])
                    raw_bones = {
                        "bone_0": {"parent": None, "rest_head_world": [0.0, 0.0, 0.50], "rest_tail_world": [0.0, 0.0, 0.65], "rest_length": 0.15},
                        "bone_1": {"parent": "bone_0", "rest_head_world": [0.0, 0.0, 0.65], "rest_tail_world": [0.0, 0.0, 0.85], "rest_length": 0.20},
                        "bone_2": {"parent": "bone_1", "rest_head_world": [0.0, 0.0, 0.85], "rest_tail_world": [0.0, -0.08, 1.02], "rest_length": 0.19},
                        "bone_3": {"parent": "bone_1", "rest_head_world": [0.08, 0.0, 0.80], "rest_tail_world": [0.24, 0.0, 0.68], "rest_length": 0.20},
                        "bone_4": {"parent": "bone_1", "rest_head_world": [-0.08, 0.0, 0.80], "rest_tail_world": [-0.24, 0.0, 0.68], "rest_length": 0.20},
                    }
                    classified = sop_preflight.classify_rig_bones(raw_bones, bbox_min=[-0.45, -0.35, 0.0], bbox_max=[0.45, 0.35, 1.15])
                    canonical_bones = {}
                    rename_map = {}
                    for old_n, info in classified.items():
                        cn = info["canonical_name"]
                        rename_map[old_n] = cn
                        canonical_bones[cn] = {
                            **info,
                            "legacy_name": old_n,
                            "parent": classified[info["parent"]]["canonical_name"] if info.get("parent") in classified else None,
                            "children": [classified[c]["canonical_name"] for c in info.get("children", []) if c in classified],
                        }
                    ft_issues = {}
                    for cid in range(4):
                        ck = f"candidate_{cid}"
                        (stage_out / f"{ck}_full.blend").write_bytes(b"BLENDER-v402" + b"\x00" * 2048)
                        meta = {
                            "schema_version": 2,
                            "bbox_min": [-0.45, -0.35, 0.0],
                            "bbox_max": [0.45, 0.35, 1.15],
                            "canonical_rename_map": rename_map,
                            "bone_naming_validation": sop_preflight.validate_canonical_rig_bone_names(canonical_bones),
                            "audit_report": {"score": 82.5, "outside_ratio": 0.0, "quadrants_covered": "4/4"},
                            "restored_decorations": [],
                            "bones": canonical_bones,
                        }
                        (stage_out / f"{ck}_rig_meta.json").write_text(json.dumps(meta))
                        for v in sop_g1_pack.G1_VIEWS:
                            Image.new("RGB", (640, 640), (28, 34, 46)).save(stage_out / f"cand_{cid}_{v}_octahedral.png")
                        ft_issues[ck] = {
                            "status": "PASS",
                            "max_vertex_displacement_norm": 0.12,
                            "max_edge_stretch_ratio": 1.18,
                            "max_bad_edge_ratio": 0.0,
                            "unweighted_vertices": 0,
                            "total_vertices": 64,
                            "geometric_score": 82.5,
                            "outside_ratio": 0.0,
                            "issues": [],
                        }
                    for sv in sop_g1_pack.G1_SYNC_VIEWS:
                        fdir = stage_out / "sync_frames" / sv
                        fdir.mkdir(parents=True, exist_ok=True)
                        for f_i in range(8):
                            Image.new("RGB", (1280, 720), (20 + f_i * 4, 28, 40)).save(fdir / f"sync_frame_{f_i:04d}.png")
                    (stage_out / "bpy_g1_summary.json").write_text(
                        json.dumps({"schema_version": 2, "render_engine": "BLENDER_WORKBENCH", "fps": 24, "follow_through_issues": ft_issues})
                    )

                real_subproc_run = sop_g1_pack.subprocess.run

                def _dispatch_run(cmd, *a, **kw):
                    if "--mode" in cmd and "prepare_g1_candidates" in cmd:
                        return _fake_g1_worker(cmd, **kw)
                    return real_subproc_run(cmd, *a, **kw)

                with patch.object(sop_g1_pack, "_worker_command", side_effect=lambda args: args), \
                     patch.object(sop_g1_pack.subprocess, "run", side_effect=_dispatch_run), \
                     patch.object(sop_g2_pack, "_worker_command", return_value=None):
                    g1_rep = sop_g1_pack.build_g1_review_pack(
                        monster_id="troll_01",
                        archetype="humanoid",
                        project_root=proj_root,
                        drive_root=drive_root,
                    )
                self.assertEqual(g1_rep["status"], "awaiting_rig_review")
                self.assertEqual(g1_rep["octahedral_views_rendered"], 16)
                self.assertEqual(len(g1_rep["candidates"]), 4)
                self.assertTrue(Path(g1_rep["review_html"]).is_file())
                sop_g1_pack.validate_g1_evidence(json.loads(Path(g1_rep["g1_report_json"]).read_text()))

                # Approve G1 via CLI dispatcher
                rc_g1 = monster_sop.main([
                    "gate",
                    "--monster-id", "troll_01",
                    "--project-root", str(proj_root),
                    "--drive-root", str(drive_root),
                    "approve",
                    "--gate", "G1",
                    "--candidate-id", "0",
                    "--rig-meta", g1_rep["candidates"][0]["meta_file"],
                    "--quote", "选定 candidate_0 通过 G1",
                ])
                self.assertEqual(rc_g1, 0)

                # 2b. Post-G1 Joint Limits Probe & Agent Customizer
                rc_jl_probe = monster_sop.main([
                    "joint-limits",
                    "--monster-id", "troll_01",
                    "--project-root", str(proj_root),
                    "--drive-root", str(drive_root),
                    "probe",
                    "--archetype", "humanoid",
                ])
                self.assertEqual(rc_jl_probe, 0)
                rc_jl_apply = monster_sop.main([
                    "joint-limits",
                    "--monster-id", "troll_01",
                    "--project-root", str(proj_root),
                    "--drive-root", str(drive_root),
                    "apply",
                    "--set-bone", "SPINE_C_00_01=-28:32,-20:20,-35:35",
                    "--set-root-pitch", "-24:24",
                    "--set-root-roll", "-18:18",
                    "--set-root-yaw", "-180:180",
                    "--notes", "Agent 定制动作无关完整运动包络",
                ])
                self.assertEqual(rc_jl_apply, 0)

                # 3. Stage 3 G2 50-pose Pack (Boundary Exploration)
                with patch.object(sop_g2_pack, "_worker_command", return_value=None):
                    g2_rep = sop_g2_pack.build_g2_review_pack(
                        monster_id="troll_01",
                        seed=42,
                        num_poses=50,
                        project_root=proj_root,
                        drive_root=drive_root,
                    )
                self.assertEqual(g2_rep["num_poses"], 50)
                self.assertTrue(g2_rep["boundary_exploration_passed"])
                self.assertTrue(Path(g2_rep["joint_limits_path"]).is_file())
                self.assertTrue(Path(g2_rep["overview_jpg"]).is_file())
                self.assertTrue(Path(g2_rep["review_html"]).is_file())
                g2_html_text = Path(g2_rep["review_html"]).read_text(encoding="utf-8")
                self.assertIn("G2 50 姿势审阅工作台 · MonsterStudio", g2_html_text)
                self.assertIn("inspectorDrawer", g2_html_text)
                self.assertTrue((Path(g2_rep["review_html"]).parent / "pose_001_front.png").is_file())
                self.assertTrue((Path(g2_rep["review_html"]).parent / "pose_records.json").is_file())

                # Approve G2
                rc_g2 = monster_sop.main([
                    "gate",
                    "--monster-id", "troll_01",
                    "--project-root", str(proj_root),
                    "--drive-root", str(drive_root),
                    "approve",
                    "--gate", "G2",
                    "--seed", "42",
                    "--quote", "50 姿势随机分布确认通过 G2",
                ])
                self.assertEqual(rc_g2, 0)

                # 4. Stage 4 G3 Exam (verify unlabeled_video never fabricates 3D MPJPE / SO(3))
                g3_unlabeled = sop_g3_exam.build_g3_exam_pack(
                    monster_id="troll_01",
                    project_root=proj_root,
                    drive_root=drive_root,
                )
                self.assertEqual(g3_unlabeled["ground_truth_mode"], "unlabeled_video")
                g3_report_data = json.loads(Path(g3_unlabeled["exam_report_json"]).read_text(encoding="utf-8"))
                self.assertIsNone(g3_report_data["stratified_3d_metrics"])
                self.assertIsNone(g3_report_data["reprojection_2d_gt_metrics"])

                # Second U2 sync with unchanged files should skip re-uploading
                u2_second = sop_g3_exam.sync_u2_to_drive(
                    "troll_01",
                    project_root=proj_root,
                    drive_root=drive_root,
                )
                self.assertTrue(u2_second["skipped_unchanged_blend"] and u2_second["skipped_unchanged_meta"])

                # Approve G3
                ckpt_file = drive_root / "monsters/troll_01/dinov2_head.pt"
                ckpt_file.parent.mkdir(parents=True, exist_ok=True)
                ckpt_file.write_bytes(b"ckpt_weights_v1")
                rc_g3 = monster_sop.main([
                    "gate",
                    "--monster-id", "troll_01",
                    "--project-root", str(proj_root),
                    "--drive-root", str(drive_root),
                    "approve",
                    "--gate", "G3",
                    "--checkpoint", str(ckpt_file),
                    "--quote", "已有视频考试还原通过 G3",
                ])
                self.assertEqual(rc_g3, 0)

                # 5. Stage 5 Delivery: verify epipolar sync failure hard stop & current-batch manifest
                traj_idle = {
                    "gate_passed": True,
                    "mean_epipolar_diff_px": 1.8,
                    "vertical_trajectory_corr": 0.96,
                    "frames": [
                        {"frame": 0, "bones": {"ROOT_C_00_00": {"head_3d": [0, 0, 1], "tail_3d": [0, 0, 1.2]}}},
                        {"frame": 1, "bones": {"ROOT_C_00_00": {"head_3d": [0, 0, 1.01], "tail_3d": [0, 0, 1.21]}}},
                    ],
                }
                act_dir = work_dir / "actions/idle"
                act_dir.mkdir(parents=True, exist_ok=True)
                (act_dir / "joints_3d_traj.json").write_text(json.dumps(traj_idle))
                self._create_minimal_glb(drive_root / "monsters/troll_01/animations/idle_animated.glb")
                # Plant a legacy action file to confirm it is excluded from current batch_manifest
                self._create_minimal_glb(drive_root / "monsters/troll_01/animations/legacy_old_animated.glb")

                deliv_rep = sop_delivery.verify_and_pack_delivery_batch(
                    monster_id="troll_01",
                    actions="idle:loop",
                    project_root=proj_root,
                    drive_root=drive_root,
                )
                self.assertTrue(deliv_rep["all_valid"])
                batch_manifest = json.loads(Path(deliv_rep["batch_manifest_json"]).read_text(encoding="utf-8"))
                batch_names = [a["name"] for a in batch_manifest["actions"]]
                self.assertEqual(batch_names, ["idle"])
                self.assertNotIn("legacy_old", batch_names)

                # 6. Stage 6 Colab Guard: block release during video iteration, allow after final_batch_approved
                sop_colab_guard.acquire_session_lease(
                    monster_id="troll_01",
                    session_id="sess_a",
                    stage="stage5_video_iteration",
                    workspace_dir=proj_root,
                    drive_root=drive_root,
                )
                blocked_iter = sop_colab_guard.evaluate_safe_shutdown(
                    monster_id="troll_01",
                    reason="final_batch_approved",
                    session_id="sess_a",
                    project_root=proj_root,
                    drive_root=drive_root,
                )
                self.assertFalse(blocked_iter["allowed"])

                # Approve final_batch clip and gate, then verify safe release
                ledger = sop_gate.load_ledger("troll_01", project_root=proj_root, drive_root=drive_root)
                sop_gate.record_user_decision(
                    ledger,
                    "final_batch",
                    decision="approved",
                    user_quote="idle:loop 动作验收满意，同意收尾释放",
                    clip_decisions={"idle": "approved"},
                )
                sop_gate.save_ledger(ledger, project_root=proj_root, drive_root=drive_root)

                allowed_final = sop_colab_guard.evaluate_safe_shutdown(
                    monster_id="troll_01",
                    reason="final_batch_approved",
                    session_id="sess_a",
                    project_root=proj_root,
                    drive_root=drive_root,
                )
                self.assertTrue(allowed_final["allowed"])
                sop_colab_guard.release_session_lease("sess_a", drive_root=drive_root)
        finally:
            os.environ.pop("MONSTER_SKIP_PS_SCAN", None)
            sys.path.remove(scripts_dir)

    def test_post_g1_joint_limits_probe_customizer_and_g2_boundary_exploration(self):
        import math
        import random

        sys.path.insert(0, str(ROOT / "worker"))
        try:
            import bone_taxonomy as tax

            raw_bones = {
                "ROOT_C_00_00": {
                    "parent": None,
                    "rest_head_world": [0.0, 0.0, 0.50],
                    "rest_tail_world": [0.0, 0.0, 0.65],
                    "rest_length": 0.15,
                    "limits_rad": [[-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2]],
                    "limits_source": "stage1_heuristic",
                },
                "SPINE_C_00_01": {
                    "parent": "ROOT_C_00_00",
                    "rest_head_world": [0.0, 0.0, 0.65],
                    "rest_tail_world": [0.0, 0.0, 0.90],
                    "rest_length": 0.25,
                },
                "LIMB_L_01_00": {
                    "parent": "SPINE_C_00_01",
                    "rest_head_world": [0.10, 0.0, 0.85],
                    "rest_tail_world": [0.28, 0.0, 0.72],
                    "rest_length": 0.22,
                },
                "LIMB_L_01_01": {
                    "parent": "LIMB_L_01_00",
                    "rest_head_world": [0.28, 0.0, 0.72],
                    "rest_tail_world": [0.42, -0.05, 0.56],
                    "rest_length": 0.21,
                },
                "LIMB_R_01_00": {
                    "parent": "SPINE_C_00_01",
                    "rest_head_world": [-0.10, 0.0, 0.85],
                    "rest_tail_world": [-0.28, 0.0, 0.72],
                    "rest_length": 0.22,
                },
                "LIMB_R_01_01": {
                    "parent": "LIMB_R_01_00",
                    "rest_head_world": [-0.28, 0.0, 0.72],
                    "rest_tail_world": [-0.42, -0.05, 0.56],
                    "rest_length": 0.21,
                },
            }
            classified = tax.classify_rig_bones(
                raw_bones, bbox_min=[-0.5, -0.4, 0.0], bbox_max=[0.5, 0.4, 1.2]
            )
            # 1. Verify stage1_heuristic limits are upgraded to canonical defaults during classify_rig_bones
            self.assertEqual(classified["ROOT_C_00_00"]["limits_source"], "canonical_default")

            # 2. Run Geometric & LBS Axis Probe
            profile = tax.probe_rig_joint_limits(
                classified,
                monster_id="behemoth_01",
                archetype="humanoid",
                bbox_min=[-0.5, -0.4, 0.0],
                bbox_max=[0.5, 0.4, 1.2],
            )
            self.assertEqual(profile["schema"], tax.JOINT_LIMITS_SCHEMA)
            self.assertIn("root_motion", profile)
            self.assertIn("LIMB_L_01_00", profile["bones"])
            self.assertIn("rx", profile["bones"]["LIMB_L_01_00"]["axis_semantics"])

            # 3. Apply custom Agent-authored narrower limits on SPINE_C_00_01 and verify
            # effective_sampling_limits_for_bone respects agent_custom without expanding to 65% floor
            profile["bones"]["SPINE_C_00_01"]["limits_deg"] = {
                "rx": [-12.0, 14.0],
                "ry": [-8.0, 8.0],
                "rz": [-10.0, 10.0],
            }
            profile["bones"]["SPINE_C_00_01"]["source"] = "agent_custom"
            applied = tax.apply_joint_limits_profile(classified, profile)
            eff_spine = tax.effective_sampling_limits_for_bone(
                applied["SPINE_C_00_01"], enforce_anatomical_floor=True
            )
            self.assertAlmostEqual(math.degrees(eff_spine[0][0]), -12.0, places=1)
            self.assertAlmostEqual(math.degrees(eff_spine[0][1]), 14.0, places=1)

            # 4. Run 50-pose Stratified Boundary-Exploration Random Suite and verify dual-sided boundary coverage
            rng = random.Random(42)
            samples = []
            regimes_seen = set()
            for idx in range(50):
                eulers, regime, b_info = tax.sample_boundary_exploration_pose_euler_dict(
                    applied,
                    rng=rng,
                    sample_idx=idx,
                    total_samples=50,
                    joint_limits_profile=profile,
                )
                regimes_seen.add(regime)
                # Every sampled angle must stay strictly within configured limits
                for b_n, rot in eulers.items():
                    lims = tax.effective_sampling_limits_for_bone(applied[b_n])
                    for ax_i in range(3):
                        self.assertGreaterEqual(rot[ax_i], lims[ax_i][0] - 1e-4)
                        self.assertLessEqual(rot[ax_i], lims[ax_i][1] + 1e-4)
                samples.append({"sampling_regime": regime, "joints": eulers, "boundary_info": b_info})

            self.assertEqual(len(regimes_seen), 5)
            b_audit = tax.audit_boundary_exploration_coverage(samples, applied)
            self.assertTrue(b_audit["passed"])
            self.assertEqual(b_audit["dual_boundary_touch_rate"], 1.0)

            # 5. Verify downstream kinematic solver & triangulation coupling with joint_limits.json
            with tempfile.TemporaryDirectory() as tmp_d:
                lim_file = Path(tmp_d) / "joint_limits.json"
                profile["bones"]["LIMB_L_01_01"]["limits_deg"] = {
                    "rx": [-10.0, 10.0],
                    "ry": [-5.0, 5.0],
                    "rz": [-8.0, 8.0],
                }
                profile["bones"]["LIMB_L_01_01"]["source"] = "agent_custom"
                lim_file.write_text(json.dumps(profile), encoding="utf-8")

                resolved_prof, resolved_path = tax.resolve_joint_limits_profile_from_paths(
                    Path(tmp_d) / "studio_rig_meta.json"
                )
                self.assertIsNotNone(resolved_prof)
                self.assertEqual(Path(resolved_path), lim_file)

                coupled_bones = tax.apply_joint_limits_profile(classified, resolved_prof)
                bone_names = sorted(coupled_bones.keys())
                T = 6
                pts_3d = tax.np.zeros((T, 2 * len(bone_names), 3), dtype=tax.np.float64)
                trust = tax.np.full((T, 2 * len(bone_names)), 0.95, dtype=tax.np.float64)
                for b_i, bn in enumerate(bone_names):
                    pts_3d[:, 2 * b_i] = coupled_bones[bn]["rest_head_world"]
                    pts_3d[:, 2 * b_i + 1] = coupled_bones[bn]["rest_tail_world"]
                # Deflect LIMB_L_01_01 tail strongly in +Z/+Y (well beyond 10 deg)
                l_idx = bone_names.index("LIMB_L_01_01")
                pts_3d[:, 2 * l_idx + 1] = tax.np.array([0.42, 0.25, 0.92], dtype=tax.np.float64)

                frames_solved, diag = tax.reconstruct_frames_with_kind_solvers(
                    pts_3d=pts_3d,
                    trust=trust,
                    bone_names=bone_names,
                    classified_bones=coupled_bones,
                )
                self.assertLessEqual(diag["final_limit_max_violation_rad"], 1e-4)
                R_parent = tax.np.asarray(frames_solved[-1]["bones"]["LIMB_L_01_00"]["rot_mat_world"])
                R_child = tax.np.asarray(frames_solved[-1]["bones"]["LIMB_L_01_01"]["rot_mat_world"])
                v0_p = tax.np.asarray(coupled_bones["LIMB_L_01_00"]["rest_tail_world"]) - tax.np.asarray(
                    coupled_bones["LIMB_L_01_00"]["rest_head_world"]
                )
                v0_c = tax.np.asarray(coupled_bones["LIMB_L_01_01"]["rest_tail_world"]) - tax.np.asarray(
                    coupled_bones["LIMB_L_01_01"]["rest_head_world"]
                )
                R_rest_p = tax._build_default_rest_rot_matrix(v0_p / float(tax.np.linalg.norm(v0_p)))
                R_rest_c = tax._build_default_rest_rot_matrix(v0_c / float(tax.np.linalg.norm(v0_c)))
                R_init_c = R_parent @ (R_rest_p.T @ R_rest_c)
                eul_c = tax.matrix_to_euler_xyz_numpy(R_init_c.T @ R_child)
                self.assertLessEqual(abs(math.degrees(float(eul_c[0]))), 10.05)
                self.assertLessEqual(abs(math.degrees(float(eul_c[1]))), 5.05)
                self.assertLessEqual(abs(math.degrees(float(eul_c[2]))), 8.05)

                # 6. Verify multi-digit claw fan (3+ digits, rest_length < 0.035) and hand_hub
                #    reconstruct full custom joint_limits.json angles (>30 deg) instead of 2.5 deg / 25 deg caps,
                #    and verify macro_collapse never expands custom spine/root limits.
                claw_raw_bones = {
                    "ROOT_C_00_00": {
                        "parent": None,
                        "rest_head_world": [0.0, 0.0, 0.40],
                        "rest_tail_world": [0.0, 0.0, 0.60],
                        "rest_length": 0.20,
                    },
                    "SPINE_C_00_01": {
                        "parent": "ROOT_C_00_00",
                        "rest_head_world": [0.0, 0.0, 0.60],
                        "rest_tail_world": [0.0, 0.0, 0.90],
                        "rest_length": 0.30,
                    },
                    "LIMB_L_01_00": {
                        "parent": "SPINE_C_00_01",
                        "rest_head_world": [0.15, 0.0, 0.85],
                        "rest_tail_world": [0.35, 0.0, 0.85],
                        "rest_length": 0.20,
                    },
                    "HAND_L_01_01": {
                        "parent": "LIMB_L_01_00",
                        "rest_head_world": [0.35, 0.0, 0.85],
                        "rest_tail_world": [0.40, 0.0, 0.85],
                        "rest_length": 0.05,
                    },
                    "DIGIT_L_01_00": {
                        "parent": "HAND_L_01_01",
                        "rest_head_world": [0.40, 0.02, 0.85],
                        "rest_tail_world": [0.425, 0.02, 0.85],
                        "rest_length": 0.025,
                    },
                    "DIGIT_L_02_00": {
                        "parent": "HAND_L_01_01",
                        "rest_head_world": [0.40, 0.0, 0.85],
                        "rest_tail_world": [0.425, 0.0, 0.85],
                        "rest_length": 0.025,
                    },
                    "DIGIT_L_03_00": {
                        "parent": "HAND_L_01_01",
                        "rest_head_world": [0.40, -0.02, 0.85],
                        "rest_tail_world": [0.425, -0.02, 0.85],
                        "rest_length": 0.025,
                    },
                }
                claw_cls = tax.classify_rig_bones(
                    claw_raw_bones, bbox_min=[-0.5, -0.4, 0.0], bbox_max=[0.5, 0.4, 1.2]
                )
                self.assertEqual(claw_cls["DIGIT_L_01_00"]["kind"], "digit")
                claw_prof = tax.probe_rig_joint_limits(
                    claw_cls,
                    monster_id="claw_beast",
                    archetype="humanoid",
                    bbox_min=[-0.5, -0.4, 0.0],
                    bbox_max=[0.5, 0.4, 1.2],
                )
                # Set narrow custom limits on ROOT/SPINE and wide custom limits on DIGIT/HAND
                claw_prof["bones"]["ROOT_C_00_00"]["limits_deg"] = {"rx": [-8.0, 8.0], "ry": [-8.0, 8.0], "rz": [-8.0, 8.0]}
                claw_prof["bones"]["ROOT_C_00_00"]["source"] = "agent_custom"
                claw_prof["root_motion"]["global_pitch_deg"] = [-8.0, 8.0]
                claw_prof["bones"]["SPINE_C_00_01"]["limits_deg"] = {"rx": [-10.0, 10.0], "ry": [-8.0, 8.0], "rz": [-8.0, 8.0]}
                claw_prof["bones"]["SPINE_C_00_01"]["source"] = "agent_custom"
                for dn in ("DIGIT_L_01_00", "DIGIT_L_02_00", "DIGIT_L_03_00"):
                    claw_prof["bones"][dn]["limits_deg"] = {"rx": [-60.0, 45.0], "ry": [-15.0, 15.0], "rz": [-50.0, 50.0]}
                    claw_prof["bones"][dn]["source"] = "agent_custom"
                claw_applied = tax.apply_joint_limits_profile(claw_cls, claw_prof)
                self.assertTrue(claw_applied["DIGIT_L_01_00"].get("allow_leaf_twist"))

                c_names = sorted(claw_applied.keys())
                T_c = 8
                pts_c = tax.np.zeros((T_c, 2 * len(c_names), 3), dtype=tax.np.float64)
                trust_c = tax.np.full((T_c, 2 * len(c_names)), 0.92, dtype=tax.np.float64)
                for b_i, bn in enumerate(c_names):
                    pts_c[:, 2 * b_i] = claw_applied[bn]["rest_head_world"]
                    pts_c[:, 2 * b_i + 1] = claw_applied[bn]["rest_tail_world"]

                # Deflect DIGIT_L_01_00 down by 42 deg in Z (from +X towards -Z)
                d1_idx = c_names.index("DIGIT_L_01_00")
                ang_rad = math.radians(42.0)
                pts_c[:, 2 * d1_idx + 1] = tax.np.array(
                    [0.40 + 0.025 * math.cos(ang_rad), 0.02, 0.85 - 0.025 * math.sin(ang_rad)],
                    dtype=tax.np.float64,
                )
                # Trigger macro_collapse on SPINE_C_00_01 by lowering its observed tail Z strongly
                sp_idx = c_names.index("SPINE_C_00_01")
                pts_c[:, 2 * sp_idx + 1] = tax.np.array([0.0, -0.30, 0.55], dtype=tax.np.float64)

                frames_c, diag_c = tax.reconstruct_frames_with_kind_solvers(
                    pts_3d=pts_c,
                    trust=trust_c,
                    bone_names=c_names,
                    classified_bones=claw_applied,
                )
                self.assertLessEqual(diag_c["final_limit_max_violation_rad"], 1e-4)

                # Verify DIGIT_L_01_00 achieved >35 deg rotation instead of being capped at 2.5 deg
                R_hand = tax.np.asarray(frames_c[-1]["bones"]["HAND_L_01_01"]["rot_mat_world"])
                R_d1 = tax.np.asarray(frames_c[-1]["bones"]["DIGIT_L_01_00"]["rot_mat_world"])
                v0_h = tax.np.asarray(claw_applied["HAND_L_01_01"]["rest_tail_world"]) - tax.np.asarray(
                    claw_applied["HAND_L_01_01"]["rest_head_world"]
                )
                v0_d1 = tax.np.asarray(claw_applied["DIGIT_L_01_00"]["rest_tail_world"]) - tax.np.asarray(
                    claw_applied["DIGIT_L_01_00"]["rest_head_world"]
                )
                R_rest_h = tax._build_default_rest_rot_matrix(v0_h / float(tax.np.linalg.norm(v0_h)))
                R_rest_d1 = tax._build_default_rest_rot_matrix(v0_d1 / float(tax.np.linalg.norm(v0_d1)))
                R_init_d1 = R_hand @ (R_rest_h.T @ R_rest_d1)
                delta_d1 = R_init_d1.T @ R_d1
                geod_d1_deg = math.degrees(
                    math.acos(max(-1.0, min(1.0, (float(tax.np.trace(delta_d1)) - 1.0) * 0.5)))
                )
                geod_d1_world_deg = math.degrees(
                    math.acos(max(-1.0, min(1.0, (float(tax.np.trace(R_rest_d1.T @ R_d1)) - 1.0) * 0.5)))
                )
                self.assertGreater(geod_d1_deg, 28.0)
                self.assertGreater(geod_d1_world_deg, 40.0)

                # Verify SPINE_C_00_01 did NOT exceed its custom [-10, 10] deg limit despite macro_collapse
                R_root_w = tax.np.asarray(frames_c[-1]["bones"]["ROOT_C_00_00"]["rot_mat_world"])
                R_sp_w = tax.np.asarray(frames_c[-1]["bones"]["SPINE_C_00_01"]["rot_mat_world"])
                v0_rt = tax.np.asarray(claw_applied["ROOT_C_00_00"]["rest_tail_world"]) - tax.np.asarray(
                    claw_applied["ROOT_C_00_00"]["rest_head_world"]
                )
                v0_sp = tax.np.asarray(claw_applied["SPINE_C_00_01"]["rest_tail_world"]) - tax.np.asarray(
                    claw_applied["SPINE_C_00_01"]["rest_head_world"]
                )
                R_rest_rt = tax._build_default_rest_rot_matrix(v0_rt / float(tax.np.linalg.norm(v0_rt)))
                R_rest_sp = tax._build_default_rest_rot_matrix(v0_sp / float(tax.np.linalg.norm(v0_sp)))
                R_init_sp = R_root_w @ (R_rest_rt.T @ R_rest_sp)
                eul_sp = tax.matrix_to_euler_xyz_numpy(R_init_sp.T @ R_sp_w)
                self.assertLessEqual(abs(math.degrees(float(eul_sp[0]))), 10.05)
        finally:
            sys.path.remove(str(ROOT / "worker"))


if __name__ == "__main__":
    unittest.main()
