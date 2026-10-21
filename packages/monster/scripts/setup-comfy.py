# @title 3. 安装/复用 ComfyUI + MiniMax H3 + DINOv2 依赖（v2.3；功能开关在 Cell 1）
# 自动将 ~50GB 的 MiniMax H3 权重软链缓存至 Google Drive (/content/drive/MyDrive/MonsterStudio/cache/h3_models)，避免每次重下！
import json, os, shutil, subprocess, sys, time
from pathlib import Path

CACHE_MODELS_ON_DRIVE = os.environ.get("MONSTER_CACHE_MODELS_ON_DRIVE", "1") == "1"
H3_PROFILE = os.environ.get("H3_PROFILE", "full")

H3_BASE = Path(os.environ.get("H3_LAB_DIR", "/content/h3-lab"))
COMFY = H3_BASE / "ComfyUI"
PORT = 8193
COMFY_COMMIT = "7f7fd918ed0815c5d5ef2ccfe44bfc8e72cfd217"
VHS_COMMIT = "4d907bee61e92c2e65af3bd6383a4e4d356126d1"
UPSCALER_COMMIT = "40316cf008b2fd8663263270669eb4da23f89d2c"
MODEL_REVISION = "e5eb578a89295337b8ff433a035929ce0279e0b6"
UPSCALER_MODEL_REPO = "LBH-123-AI/Minimax_h3_latent_Upscaler"
UPSCALER_MODEL_REVISION = "3f941d5d182014dd5c0a5e16330420ee2d4aa0c6"
UPSCALER_MODEL_REL = "minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"
UPSCALER_MODEL_FILENAME = "minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors"

PROFILES = {
    "int8": {
        "audio": "vae/minimax_h3_audio_vae_fp32.safetensors",
        "lora": "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "model": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "text": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "video": "vae/minimax_h3_video_vae_int8_convrot.safetensors",
    },
    "full": {
        "audio": "vae/minimax_h3_audio_vae_fp32.safetensors",
        "lora": "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "model": "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors",
        "text": "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
        "video": "vae/minimax_h3_video_vae_fp16.safetensors",
    },
}

def checkout_repo(url: str, dest: Path, commit: str) -> None:
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", url], check=True)
    cur = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True)
    if cur.returncode != 0 or cur.stdout.strip() != commit:
        subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "origin", commit], check=True)
        subprocess.run(["git", "-C", str(dest), "checkout", "--detach", commit], check=True)

print("[Runtime 3] Preparing ComfyUI + VideoHelperSuite + H3 Latent Upscaler + Gradio Worker dependencies...")
checkout_repo("https://github.com/Comfy-Org/ComfyUI.git", COMFY, COMFY_COMMIT)
vhs_dir = COMFY / "custom_nodes" / "ComfyUI-VideoHelperSuite"
checkout_repo("https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git", vhs_dir, VHS_COMMIT)
upscaler_dir = COMFY / "custom_nodes" / "Comfyui_Minimax_h3_latent_Upscaler"
checkout_repo("https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git", upscaler_dir, UPSCALER_COMMIT)

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "-r", str(COMFY / "requirements.txt"),
    "-r", str(vhs_dir / "requirements.txt"),
    "-r", str(Path(__file__).resolve().parents[1] / "worker/requirements.txt"),
], check=True)

from huggingface_hub import hf_hub_download
drive_model_cache = Path(os.environ["MONSTER_DRIVE_ROOT"]) / "cache/h3_models"
if CACHE_MODELS_ON_DRIVE:
    drive_model_cache.mkdir(parents=True, exist_ok=True)

selected_files = list(PROFILES[H3_PROFILE].values())
for idx, rel_path in enumerate(selected_files, 1):
    comfy_target = COMFY / "models" / rel_path
    comfy_target.parent.mkdir(parents=True, exist_ok=True)
    if comfy_target.exists():
        print(f"  [{idx}/{len(selected_files)}] Already present: {rel_path}")
        continue
    if CACHE_MODELS_ON_DRIVE:
        cached_file = drive_model_cache / rel_path
        if not cached_file.exists():
            print(f"  [{idx}/{len(selected_files)}] Downloading {rel_path} to Google Drive cache...")
            hf_hub_download("Comfy-Org/MiniMax-H3", rel_path, revision=MODEL_REVISION, local_dir=str(drive_model_cache))
        if comfy_target.is_symlink() or comfy_target.exists():
            comfy_target.unlink()
        comfy_target.symlink_to(cached_file)
        print(f"  [{idx}/{len(selected_files)}] Linked from Drive cache: {rel_path}")
    else:
        print(f"  [{idx}/{len(selected_files)}] Downloading {rel_path} to local NVMe...")
        hf_hub_download("Comfy-Org/MiniMax-H3", rel_path, revision=MODEL_REVISION, local_dir=str(COMFY / "models"))

upscaler_target = COMFY / "models" / "latent_upscale_models" / UPSCALER_MODEL_FILENAME
upscaler_target.parent.mkdir(parents=True, exist_ok=True)
if not upscaler_target.exists():
    if CACHE_MODELS_ON_DRIVE:
        cached_up = drive_model_cache / UPSCALER_MODEL_REL
        if not cached_up.exists():
            print(f"  [2K Latent Upscaler] Downloading {UPSCALER_MODEL_REL} to Google Drive cache...")
            hf_hub_download(
                UPSCALER_MODEL_REPO,
                UPSCALER_MODEL_REL,
                revision=UPSCALER_MODEL_REVISION,
                local_dir=str(drive_model_cache),
            )
        if upscaler_target.is_symlink() or upscaler_target.exists():
            upscaler_target.unlink()
        upscaler_target.symlink_to(cached_up)
    else:
        dl_path = Path(
            hf_hub_download(
                UPSCALER_MODEL_REPO,
                UPSCALER_MODEL_REL,
                revision=UPSCALER_MODEL_REVISION,
                local_dir=str(COMFY / "models" / "latent_upscale_models"),
            )
        )
        if dl_path != upscaler_target:
            shutil.copy2(dl_path, upscaler_target)

# Start ComfyUI loopback server on 127.0.0.1:8193
import requests
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))
from pipeline_runner import ensure_comfyui_server_running
ensure_comfyui_server_running()
print("[Runtime 3 Ready] ComfyUI MiniMax H3 loopback server is LIVE at http://127.0.0.1:8193")
