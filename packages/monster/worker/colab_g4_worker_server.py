#!/usr/bin/env python3
"""colab_g4_worker_server.py — Headless Colab G4 GPU Compute Worker with Anti-Idle Watchdog.

Designed for Pattern B (Local MCP Server / Blender Plugin / Codex CLI + Remote Colab G4 Worker):
  1. Exposes 5 stateless/resumable RPC endpoints via Gradio / FastAPI (`share=True` tunnel):
     - `rig_candidates`: Pre-sanitizes mesh, runs VAST-AI SkinTokens (`num_beams=10`,
                         `use_postprocess=True`, 4 seeds), repairs air-bridging chords, and returns
                         4 `.glb` candidates + `candidates_contact_sheet.jpg` + `rigged_pruned.blend`
                         and `studio_rig_meta.json`.
     - `train_monster_probe`: Renders 3,000 dual-ortho poses in headless `bpy==4.2.0`, trains the
                              frozen DINOv2 + Deconv Probe, and runs the 96-frame Studio Exam.
     - `generate_actions_batch`: Drives the 15-node ComfyUI MiniMax H3 FL2VA workflow (8-step Turbo,
                                 loop-locked), runs the DINOv2 Epipolar Consistency Gate, and returns
                                 lightweight 3D joint trajectories (`joints_3d_traj.json`) + `.glb`s.
     - `heartbeat`: Refreshes the activity timer and returns GPU VRAM / job progress telemetry.
     - `shutdown`: Immediately flushes Drive buffers and calls `google.colab.runtime.unassign()`
                   to hard-release the Colab G4 GPU VM and stop Compute Unit billing.
"""

from __future__ import annotations

import json
from pose_features import (
    features_enabled,
    dataset_cache_matches,
    FEATURE_VERSION,
    resolution_2k_enabled,
    studio_resolution,
    idle_split_filename,
)
from runtime_metadata import runtime_metadata
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


DRIVE_ROOT = Path(os.environ.get("MONSTER_DRIVE_ROOT", "/content/drive/MyDrive/MonsterStudio"))
WORK_ROOT = Path(os.environ.get("MONSTER_WORK_ROOT", "/content/monster-studio"))
SKINTOKENS_ROOT = Path(os.environ.get("SKINTOKENS_ROOT", "/content/skintokens-service"))
SKINTOKENS_PY = SKINTOKENS_ROOT / "venv" / "bin" / "python"
COMFY_URL = os.environ.get("H3_COMFY_URL", "http://127.0.0.1:8193")
REGISTRY_PATH = DRIVE_ROOT / "runtime" / "endpoint_registry.json"


class ColabLifecycleGuard:
    """Thread-safe GPU mutex + activity tracker + automatic idle VM self-termination."""

    def __init__(self, idle_timeout_sec: int = 600):
        self.idle_timeout_sec = idle_timeout_sec
        self.lock = threading.Lock()
        self.busy = False
        self.current_job = "IDLE"
        self.last_activity_ts = time.time()
        self.stopped = False
        self._watchdog = threading.Thread(target=self._loop, daemon=True)
        self._watchdog.start()

    def touch(self, job_label: str | None = None) -> None:
        self.last_activity_ts = time.time()
        if job_label is not None:
            self.current_job = job_label

    def idle_seconds(self) -> float:
        if self.busy:
            self.last_activity_ts = time.time()
            return 0.0
        return max(0.0, time.time() - self.last_activity_ts)

    def _loop(self) -> None:
        while not self.stopped:
            time.sleep(15)
            idle_s = self.idle_seconds()
            if not self.busy and idle_s >= self.idle_timeout_sec:
                print(
                    f"\n[IdleWatchdog] Worker idle for {idle_s:.0f}s (>= {self.idle_timeout_sec}s). "
                    "Triggering automatic Colab G4 runtime.unassign() to prevent CU bleed!"
                )
                self.hard_unassign_colab(reason=f"auto_idle_timeout_{self.idle_timeout_sec}s")
                break

    def hard_unassign_colab(self, reason: str = "client_requested_shutdown") -> dict[str, Any]:
        """Flushes status to Google Drive and releases the Colab GPU VM immediately."""
        self.stopped = True
        try:
            if REGISTRY_PATH.exists():
                reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
                reg["status"] = "TERMINATED"
                reg["terminated_reason"] = reason
                reg["terminated_at"] = time.time()
                REGISTRY_PATH.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

        def _delayed_kill() -> None:
            time.sleep(2.0)
            try:
                from google.colab import runtime  # type: ignore

                runtime.unassign()
            except Exception as exc:
                print(f"[ColabLifecycleGuard] google.colab.runtime.unassign() fallback: {exc}")

        threading.Thread(target=_delayed_kill, daemon=True).start()
        return {"status": "TERMINATING", "reason": reason, "timestamp": time.time()}


def free_comfyui_vram() -> None:
    """Releases MiniMax H3 weights from VRAM via ComfyUI's /free endpoint before running PyTorch/bpy."""
    try:
        req = urllib.request.Request(
            f"{COMFY_URL}/free",
            data=json.dumps({"unload_models": True, "free_memory": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def query_gpu_telemetry() -> dict[str, Any]:
    """Queries nvidia-smi for live GPU name, VRAM usage, and utilization."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        name, used, total, util = [x.strip() for x in out.splitlines()[0].split(",")]
        return {
            "gpu": name,
            "vram_used_mib": float(used),
            "vram_total_mib": float(total),
            "gpu_util_pct": float(util),
        }
    except Exception:
        return {"gpu": "unknown", "vram_used_mib": 0.0, "vram_total_mib": 0.0, "gpu_util_pct": 0.0}


GUARD = ColabLifecycleGuard(idle_timeout_sec=int(os.environ.get("MONSTER_IDLE_TIMEOUT", "600")))


def save_stage_state(monster_id: str, stage: str, meta: dict[str, Any]) -> None:
    """Persists atomic checkpoint state to Google Drive so preempted sessions resume cleanly."""
    state_file = DRIVE_ROOT / "monsters" / monster_id / "stage_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    current = {}
    if state_file.exists():
        try:
            current = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current[stage] = {"updated_at": time.time(), **meta}
    state_file.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")


def write_task_progress(
    monster_id: str | None = None,
    stage: str = "idle",
    stage_display: str = "就绪",
    status: str = "idle",
    detail: str = "",
    epochs: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    stamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")
    telemetry = query_gpu_telemetry()
    payload = {
        "monster_id": monster_id or "",
        "stage": stage,
        "stage_display": stage_display,
        "status": status,
        "detail": detail,
        "epochs": epochs or 0,
        "gpu": telemetry,
        "updated_unix": time.time(),
        "updated_at": stamp,
        **(extra or {}),
    }
    targets = [
        DRIVE_ROOT / "runtime" / "current_task_progress.json",
        WORK_ROOT / "runtime" / "current_task_progress.json",
    ]
    if monster_id:
        targets.extend([
            DRIVE_ROOT / "monsters" / monster_id / "task_progress.json",
            WORK_ROOT / "work" / monster_id / "task_progress.json",
        ])
    for target in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(target)
        except Exception:
            pass


def api_heartbeat() -> str:
    """Endpoint 0: Refreshes the anti-idle timer and returns worker status + GPU telemetry."""
    GUARD.touch()
    _, _, comp_w, comp_h = studio_resolution()
    payload = {
        **runtime_metadata(),
        "status": "BUSY" if GUARD.busy else "READY",
        "current_job": GUARD.current_job,
        "feature_version": FEATURE_VERSION,
        "depth_visibility_enabled": features_enabled(),
        "resolution_2k_enabled": resolution_2k_enabled(),
        "studio_resolution": f"{comp_w}x{comp_h}",
        "h3_profile": os.environ.get("H3_PROFILE", "full"),
        "idle_seconds": round(GUARD.idle_seconds(), 1),
        "idle_timeout_sec": GUARD.idle_timeout_sec,
        "telemetry": query_gpu_telemetry(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def api_rig_candidates(monster_id: str, input_glb_file: str, run_auto_audit: bool = True) -> tuple[str, str]:
    """Endpoint 1: Pre-sanitizes mesh, runs SkinTokens 4-seed stochastic beam-search rigging
    (`num_beams=10`, `use_postprocess=True`), repairs air-bridging chords, and packages
    `candidate_0..3.glb` + `candidates_contact_sheet.jpg` + `rigged_pruned.blend` + `studio_rig_meta.json`.
    """
    from pipeline_runner import run_stage1_skintokens_candidates

    with GUARD.lock:
        GUARD.busy = True
        GUARD.touch(f"Stage1_SkinTokens:{monster_id}")
        write_task_progress(
            monster_id=monster_id,
            stage="rigging",
            stage_display="候选骨骼生成",
            status="running",
            detail=f"正在为 {monster_id} 生成 4 组候选骨骼与拓扑修剪...",
        )
        try:
            free_comfyui_vram()
            m_drive = DRIVE_ROOT / "monsters" / monster_id
            m_drive.mkdir(parents=True, exist_ok=True)
            m_work = WORK_ROOT / "work" / monster_id
            cand_dir = m_work / "candidates"
            cand_dir.mkdir(parents=True, exist_ok=True)

            src_glb = Path(input_glb_file)
            archived_input = m_drive / "input_mesh.glb"
            shutil.copy2(src_glb, archived_input)

            cands = run_stage1_skintokens_candidates(archived_input, cand_dir, num_candidates=4)
            GUARD.touch()

            py_bpy = str(SKINTOKENS_PY) if SKINTOKENS_PY.exists() else sys.executable
            blend_path = m_drive / "rigged_pruned.blend"
            meta_path = m_drive / "studio_rig_meta.json"
            if run_auto_audit:
                bpy_worker = Path(__file__).resolve().parent / "studio_bpy_worker.py"
                subprocess.run(
                    [
                        py_bpy,
                        str(bpy_worker),
                        "--mode",
                        "audit",
                        "--candidates",
                        *[str(c) for c in cands],
                        "--blend_path",
                        str(blend_path),
                        "--meta_path",
                        str(meta_path),
                    ],
                    check=True,
                )

            bundle_zip = m_work / f"{monster_id}_stage1_rig_bundle.zip"
            contact_sheet = m_drive / "candidates_contact_sheet.jpg"
            with zipfile.ZipFile(bundle_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for c in cands:
                    if c.exists():
                        zf.write(c, c.name)
                if blend_path.exists():
                    zf.write(blend_path, blend_path.name)
                if meta_path.exists():
                    zf.write(meta_path, meta_path.name)
                if contact_sheet.exists():
                    zf.write(contact_sheet, contact_sheet.name)

            save_stage_state(monster_id, "stage1_rig", {"status": "completed", "bundle": str(bundle_zip)})
            write_task_progress(
                monster_id=monster_id,
                stage="stage1_completed",
                stage_display="候选骨骼生成完成",
                status="completed",
                detail=f"候选骨骼生成与评审大图已保存，等待 G1 选型。",
            )
            return meta_path.read_text(encoding="utf-8") if meta_path.exists() else "{}", str(bundle_zip)
        finally:
            GUARD.busy = False
            GUARD.touch("IDLE")
            write_task_progress(
                stage="idle",
                stage_display="就绪（等待任务）",
                status="idle",
                detail="Worker 正在运行并监听请求，随时可提交新任务。",
            )


def api_train_monster_probe(
    monster_id: str,
    custom_blend_file: str | None = None,
    custom_meta_json: str | None = None,
    num_samples: int = 5000,
    epochs: int = 15,
    auto_shutdown_after: bool = False,
) -> str:
    """Endpoint 2: Renders 5,000 stratified poses, trains the DINOv2 probe, and runs the 96-frame Studio Exam."""
    with GUARD.lock:
        GUARD.busy = True
        GUARD.touch(f"Stage2_3_4_TrainExam:{monster_id}")
        try:
            free_comfyui_vram()
            m_drive = DRIVE_ROOT / "monsters" / monster_id
            m_drive.mkdir(parents=True, exist_ok=True)
            m_work = WORK_ROOT / "work" / monster_id
            m_work.mkdir(parents=True, exist_ok=True)

            blend_path = m_drive / "rigged_pruned.blend"
            meta_path = m_drive / "studio_rig_meta.json"
            if custom_blend_file and Path(custom_blend_file).exists():
                shutil.copy2(custom_blend_file, blend_path)
            if custom_meta_json and custom_meta_json.strip():
                meta_path.write_text(custom_meta_json, encoding="utf-8")

            py_bpy = str(SKINTOKENS_PY) if SKINTOKENS_PY.exists() else sys.executable
            here = Path(__file__).resolve().parent
            bpy_worker = here / "studio_bpy_worker.py"
            dino_worker = here / "dinov2_pose_trainer.py"
            synth_dir = m_work / "synth_3000"
            ckpt_path = m_drive / "dinov2_head.pt"
            exam_dir = m_work / "studio_exam"
            exam_report_path = m_drive / "exam_report.json"
            progress_file = m_work / "training_progress.json"

            if not dataset_cache_matches(synth_dir, blend_path, meta_path, num_samples):
                GUARD.touch(f"Stage2_Render3000:{monster_id}")
                write_task_progress(
                    monster_id=monster_id,
                    stage="synthesizing",
                    stage_display="合成训练图",
                    status="running",
                    detail=f"正在生成 {num_samples} 姿态双视图...",
                    epochs=epochs,
                )
                subprocess.run(
                    [
                        py_bpy,
                        str(bpy_worker),
                        "--mode",
                        "render_3000",
                        "--blend_path",
                        str(blend_path),
                        "--meta_path",
                        str(meta_path),
                        "--out_dir",
                        str(synth_dir),
                        "--num_samples",
                        str(int(num_samples)),
                    ],
                    check=True,
                )
                idle_name = idle_split_filename()
                shutil.copy2(synth_dir / idle_name, m_drive / idle_name)
                _, _, comp_w, comp_h = studio_resolution()
                save_stage_state(
                    monster_id,
                    "stage2_synth",
                    {
                        "num_samples": int(num_samples),
                        "depth_visibility_enabled": features_enabled(),
                        "resolution_2k_enabled": resolution_2k_enabled(),
                        "studio_resolution": f"{comp_w}x{comp_h}",
                        "feature_version": FEATURE_VERSION,
                    },
                )

            GUARD.touch(f"Stage3_DINOv2Train:{monster_id}")
            write_task_progress(
                monster_id=monster_id,
                stage="training",
                stage_display="训练 DINOv2 姿态头",
                status="running",
                detail=f"开始 DINOv2 姿态估计训练（共 {epochs} 轮）...",
                epochs=epochs,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(dino_worker),
                    "--mode",
                    "train",
                    "--synth_dir",
                    str(synth_dir),
                    "--meta_path",
                    str(meta_path),
                    "--ckpt_path",
                    str(ckpt_path),
                    "--epochs",
                    str(int(epochs)),
                    "--progress_file",
                    str(progress_file),
                ],
                check=True,
            )
            save_stage_state(monster_id, "stage3_dinov2", {"ckpt": str(ckpt_path), "epochs": int(epochs), "depth_visibility_enabled": features_enabled(), "feature_version": FEATURE_VERSION})

            GUARD.touch(f"Stage4_StudioExam:{monster_id}")
            write_task_progress(
                monster_id=monster_id,
                stage="exam_rendering",
                stage_display="视频考试渲染",
                status="running",
                detail="正在合成 96 帧连续动作视频测试集...",
                epochs=epochs,
            )
            subprocess.run(
                [
                    py_bpy,
                    str(bpy_worker),
                    "--mode",
                    "render_exam",
                    "--blend_path",
                    str(blend_path),
                    "--meta_path",
                    str(meta_path),
                    "--out_dir",
                    str(exam_dir),
                    "--exam_frames",
                    "96",
                ],
                check=True,
            )
            write_task_progress(
                monster_id=monster_id,
                stage="exam_inference",
                stage_display="已有视频考试推理",
                status="running",
                detail="正在执行双视图 3D 姿态重构与骨骼误差评估...",
                epochs=epochs,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(dino_worker),
                    "--mode",
                    "exam",
                    "--ckpt_path",
                    str(ckpt_path),
                    "--exam_dir",
                    str(exam_dir),
                    "--out_json",
                    str(exam_report_path),
                ],
                check=True,
            )
            shutil.rmtree(synth_dir, ignore_errors=True)
            report_str = exam_report_path.read_text(encoding="utf-8")
            save_stage_state(monster_id, "stage4_exam", json.loads(report_str))
            write_task_progress(
                monster_id=monster_id,
                stage="awaiting_exam_review",
                stage_display="考试完成，等待 G3 审阅",
                status="completed",
                detail="Studio Exam 模拟考评估完成，结果已保存至 Drive，等待用户 G3 审阅。",
                epochs=epochs,
            )
            return report_str
        finally:
            GUARD.busy = False
            GUARD.touch("IDLE")
            write_task_progress(
                stage="idle",
                stage_display="就绪（等待任务）",
                status="idle",
                detail="Worker 正在运行并监听请求，随时可提交新任务。",
            )
            if auto_shutdown_after:
                GUARD.hard_unassign_colab(reason="auto_shutdown_after_train_monster_probe")



def api_reexam_only(monster_id: str, rerender_exam: bool = False, auto_shutdown_after: bool = False) -> str:
    """Endpoint 2b: Fast Stage 4 Re-Exam + Swing+Twist IK GLB Round-Trip Verification (0 retraining!)."""
    from pipeline_runner import reexam_existing_monster

    with GUARD.lock:
        GUARD.busy = True
        GUARD.touch(f"Stage4_FastReExam:{monster_id}")
        try:
            report = reexam_existing_monster(monster_id, rerender_exam=rerender_exam)
            save_stage_state(monster_id, "stage4_reexam_v2_2", report)
            return json.dumps(report, indent=2, ensure_ascii=False)
        finally:
            GUARD.busy = False
            GUARD.touch("IDLE")
            if auto_shutdown_after:
                GUARD.hard_unassign_colab(reason="auto_shutdown_after_reexam_only")


def api_generate_actions_batch(
    monster_id: str,
    actions_csv: str,
    bake_glb_on_cloud: bool = True,
    auto_shutdown_after: bool = True,
) -> tuple[str, str]:
    """Endpoint 3: Batch-generates multiple actions via the 15-node ComfyUI MiniMax H3 FL2VA graph."""
    with GUARD.lock:
        GUARD.busy = True
        GUARD.touch(f"Stage5_6_AnimateBatch:{monster_id}")
        write_task_progress(
            monster_id=monster_id,
            stage="animating",
            stage_display="MiniMax H3 动作生成",
            status="running",
            detail=f"正在为 {monster_id} 批量生成动作: {actions_csv}...",
        )
        try:
            specs = [s.strip() for s in actions_csv.replace(",", " ").split() if s.strip()]
            runner = Path(__file__).resolve().parent / "pipeline_runner.py"
            subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "animate",
                    "--monster_id",
                    monster_id,
                    "--actions",
                    *specs,
                ],
                check=True,
            )
            m_drive = DRIVE_ROOT / "monsters" / monster_id
            m_work = WORK_ROOT / "work" / monster_id
            out_zip = m_work / f"{monster_id}_animations_bundle.zip"
            summary = []
            with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for glb_p in (m_drive / "animations").glob("*.glb"):
                    zf.write(glb_p, f"animations/{glb_p.name}")
                    summary.append(glb_p.name)
                for tj_p in (m_work / "actions").rglob("joints_3d_traj.json"):
                    act_name = tj_p.parent.name
                    zf.write(tj_p, f"trajectories/{act_name}_joints_3d.json")

            res_json = json.dumps(
                {
                    "monster_id": monster_id,
                    "completed_actions": summary,
                    "auto_shutdown_triggered": auto_shutdown_after,
                },
                ensure_ascii=False,
                indent=2,
            )
            write_task_progress(
                monster_id=monster_id,
                stage="animating_completed",
                stage_display="动作生成与烘焙完成",
                status="completed",
                detail=f"成功完成 {len(summary)} 个动作生成与 GLB 烘焙。",
            )
            return res_json, str(out_zip)
        finally:
            GUARD.busy = False
            GUARD.touch("IDLE")
            write_task_progress(
                stage="idle",
                stage_display="就绪（等待任务）",
                status="idle",
                detail="Worker 正在运行并监听请求，随时可提交新任务。",
            )
            if auto_shutdown_after:
                GUARD.hard_unassign_colab(reason="auto_shutdown_after_animate_batch")


def api_shutdown(reason: str = "local_scheduler_explicit_shutdown") -> str:
    """Endpoint 4: Immediately releases the Colab G4 VM (`runtime.unassign()`)."""
    return json.dumps(GUARD.hard_unassign_colab(reason=reason), ensure_ascii=False, indent=2)


def launch_headless_worker(port: int = 7865) -> None:
    """Launches the Gradio / REST Worker Server and registers its live URL + Token to Google Drive."""
    import gradio as gr
    global APP

    token = os.environ.get("MONSTER_WORKER_TOKEN") or secrets.token_urlsafe(20)
    with gr.Blocks(title="MonsterStudio · Colab G4 Headless Worker") as app:
        gr.Markdown(
            "# 🦖 MonsterStudio · Colab G4 无头算力服务端 (Worker v2.3 · Depth/Visibility)\n"
            f"- **自动断连看门狗已启动**：空闲超过 `{GUARD.idle_timeout_sec}s` 自动调用 `runtime.unassign()` 释放 G4 停止扣点。"
        )
        hb_btn = gr.Button("刷新心跳 / 查看 GPU 遥测")
        hb_out = gr.Code(language="json", label="Worker Telemetry")
        hb_btn.click(api_heartbeat, inputs=[], outputs=[hb_out], api_name="heartbeat")

        with gr.Tab("1. SkinTokens 候选生成"):
            m_id1 = gr.Textbox(label="Monster ID", value="swamp_troll_01")
            in_glb = gr.File(label="Input Static GLB", type="filepath")
            auto_aud = gr.Checkbox(label="Run Auto Geometric Audit & Pruning", value=True)
            rig_btn = gr.Button("Run SkinTokens (4 Seeds)")
            rig_meta_out = gr.Code(language="json", label="studio_rig_meta.json")
            rig_zip_out = gr.File(label="Stage 1 Bundle ZIP")
            rig_btn.click(
                api_rig_candidates,
                inputs=[m_id1, in_glb, auto_aud],
                outputs=[rig_meta_out, rig_zip_out],
                api_name="rig_candidates",
            )

        with gr.Tab("2. 5000 姿态渲染 + DINOv2 训练 + 模拟考"):
            _, _, comp_w, comp_h = studio_resolution()
            gr.Markdown(
                f"**{FEATURE_VERSION}** · 深度/可见性监督：**{'开启' if features_enabled() else '关闭'}** · "
                f"画布分辨率：**{comp_w}x{comp_h} ({'2K 特性开关已开启' if resolution_2k_enabled() else '默认 768P'})** · "
                f"H3 权重：**{os.environ.get('H3_PROFILE', 'full')}**。"
                "在 Notebook Cell 1 修改总开关后重新运行 Cell 1 和 Cell 4。"
            )
            m_id2 = gr.Textbox(label="Monster ID", value="swamp_troll_01")
            c_blend = gr.File(label="Optional Locally Adjusted .blend", type="filepath")
            c_meta = gr.Textbox(label="Optional Locally Adjusted studio_rig_meta.json", lines=4)
            n_samp = gr.Number(label="Num Synthetic Poses", value=5000)
            n_ep = gr.Number(label="DINOv2 Epochs", value=15)
            auto_kill_train = gr.Checkbox(label="训练与模拟考完成后立即自动释放 G4 GPU", value=False)
            tr_btn = gr.Button("Run Synthetic Studio + DINOv2 + Exam")
            tr_out = gr.Code(language="json", label="Studio Exam Report")
            tr_btn.click(
                api_train_monster_probe,
                inputs=[m_id2, c_blend, c_meta, n_samp, n_ep, auto_kill_train],
                outputs=[tr_out],
                api_name="train_monster_probe",
            )

        with gr.Tab("3. MiniMax H3 批量动作生成 + 自动关机"):
            m_id3 = gr.Textbox(label="Monster ID", value="swamp_troll_01")
            acts = gr.Textbox(label="Action Specs", value="idle:loop walk:loop run:loop slash:once")
            bake_cloud = gr.Checkbox(label="Bake GLB on Cloud", value=True)
            auto_kill_anim = gr.Checkbox(label="批量生成完成后立即调用 runtime.unassign() 释放 G4", value=True)
            an_btn = gr.Button("Run MiniMax H3 Batch & Auto-Shutdown")
            an_json = gr.Code(language="json", label="Batch Summary")
            an_zip = gr.File(label="Animations & 3D Trajectories ZIP")
            an_btn.click(
                api_generate_actions_batch,
                inputs=[m_id3, acts, bake_cloud, auto_kill_anim],
                outputs=[an_json, an_zip],
                api_name="generate_actions_batch",
            )

        with gr.Tab("4. 立即释放 GPU"):
            kill_reason = gr.Textbox(label="Reason", value="manual_or_client_shutdown")
            kill_btn = gr.Button("立即销毁并释放 Colab G4 实例", variant="stop")
            kill_out = gr.Code(language="json")
            kill_btn.click(api_shutdown, inputs=[kill_reason], outputs=[kill_out], api_name="shutdown")

    _, local_url, share_url = app.queue(max_size=8, default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=port,
        share=True,
        auth=("monster", token),
        prevent_thread_lock=True,
    )

    APP = app
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _, _, comp_w, comp_h = studio_resolution()
    reg_info = {
        **runtime_metadata(),
        "feature_version": FEATURE_VERSION,
        "depth_visibility_enabled": features_enabled(),
        "resolution_2k_enabled": resolution_2k_enabled(),
        "studio_resolution": f"{comp_w}x{comp_h}",
        "h3_profile": os.environ.get("H3_PROFILE", "full"),
        "status": "ONLINE",
        "url": share_url or local_url,
        "username": "monster",
        "token": token,
        "idle_timeout_sec": GUARD.idle_timeout_sec,
        "started_at": time.time(),
        "telemetry": query_gpu_telemetry(),
    }
    REGISTRY_PATH.write_text(json.dumps(reg_info, indent=2, ensure_ascii=False), encoding="utf-8")
    write_task_progress(
        stage="idle",
        stage_display="就绪（等待任务）",
        status="idle",
        detail="Worker 正在运行并监听请求，随时可提交新任务。",
    )
    print(f"[Worker Online] Public API URL: {share_url} | Registry saved to {REGISTRY_PATH}", flush=True)


if __name__ == "__main__":
    launch_headless_worker()