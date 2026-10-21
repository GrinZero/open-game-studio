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


if __name__ == "__main__":
    unittest.main()
