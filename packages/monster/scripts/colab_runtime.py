"""Colab lifecycle entry points. Heavy dependencies are imported only by setup/launch."""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER_MODULES = (
    "pose_features", "pose_visibility", "dinov2_pose_trainer", "pipeline_runner",
    "studio_bpy_worker", "runtime_metadata", "colab_g4_worker_server",
)


def stop_previous_worker():
    old = sys.modules.get("colab_g4_worker_server")
    if old is None:
        return
    if old.GUARD.busy:
        raise RuntimeError("Worker 正在执行任务，请任务完成后再切换版本或重载。")
    if getattr(old, "APP", None) is None:
        raise RuntimeError("旧 Worker 未提供可关闭的 APP；请先停止旧服务再切换版本。")
    old.GUARD.stopped = True
    old.APP.close()


def configure(*, depth_visibility=True, enable_2k=False, profile="full", cache_models=True,
              drive_root="/content/drive/MyDrive/MonsterStudio",
              work_root="/content/monster-studio"):
    if profile not in {"int8", "full"}:
        raise ValueError("H3 profile must be int8 or full")
    stop_previous_worker()
    for name in WORKER_MODULES:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    for name, value in {
        "MONSTER_DRIVE_ROOT": drive_root,
        "MONSTER_WORK_ROOT": work_root,
        "MONSTER_ENABLE_DEPTH_VISIBILITY": "1" if depth_visibility else "0",
        "MONSTER_ENABLE_2K": "1" if enable_2k else "0",
        "MONSTER_CACHE_MODELS_ON_DRIVE": "1" if cache_models else "0",
        "H3_PROFILE": profile,
        "MONSTER_IDLE_TIMEOUT": "600",
        "SKINTOKENS_ROOT": "/content/skintokens-service",
        "SKINTOKENS_CACHE": "/content/drive/MyDrive/SkinTokens/cache",
        "SKINTOKENS_ARCHIVE": "/content/drive/MyDrive/SkinTokens/results",
    }.items():
        os.environ[name] = str(value)
    for directory in (drive_root, work_root, os.environ["SKINTOKENS_CACHE"],
                      os.environ["SKINTOKENS_ARCHIVE"]):
        Path(directory).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "worker"))
    from runtime_metadata import runtime_metadata
    print(json.dumps(runtime_metadata(), ensure_ascii=False))
    print("代码从版本目录加载；Drive 中原有模型、训练结果及 src 备份均保留。")


def setup_skintokens():
    bootstrap = ROOT / "scripts/bootstrap-skintokens.sh"
    runtime = Path(os.environ["SKINTOKENS_ROOT"])
    if (runtime / "ready.json").is_file():
        print("SkinTokens / bpy 已就绪，复用已有环境。")
        return
    subprocess.run(["bash", str(bootstrap)], env=os.environ.copy(), check=True)


def setup_comfy():
    subprocess.run([sys.executable, str(ROOT / "scripts/setup-comfy.py")],
                   env=os.environ.copy(), check=True)


def launch_worker(port=7865):
    stop_previous_worker()
    for name in WORKER_MODULES:
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    from colab_g4_worker_server import REGISTRY_PATH, launch_headless_worker
    launch_headless_worker(port=port)
    registry_file = Path(REGISTRY_PATH)
    if registry_file.is_file():
        info = json.loads(registry_file.read_text(encoding="utf-8"))
        pairing = json.dumps({
            "url": info["url"],
            "username": info["username"],
            "token": info["token"],
            "protocol_version": info.get("protocol_version"),
        }, ensure_ascii=False, separators=(",", ":"))
        print(
            "[本机配对] 若本机已挂载 Google Drive 桌面版可直接使用；否则可在本机执行：\n"
            f"npm run monster:call -- connect '{pairing}'",
            flush=True,
        )


def shutdown():
    from colab_g4_worker_server import GUARD
    return GUARD.hard_unassign_colab(reason="notebook_manual_stop_button")
