"""Colab lifecycle entry points. Heavy dependencies are imported only by setup/launch."""
from __future__ import annotations

import html
import importlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
WORKER_MODULES = (
    "mocap_priors", "pose_features", "pose_visibility", "dinov2_pose_trainer", "pipeline_runner",
    "studio_bpy_worker", "runtime_metadata", "task_progress", "colab_g4_worker_server",
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


def configure(*, depth_visibility=True, dense_surface=False, mocap_priors=True, enable_2k=False, enable_3view=False, profile="full", cache_models=True,
              idle_timeout=0,
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
        "MONSTER_ENABLE_DENSE_SURFACE": "1" if dense_surface else "0",
        "MONSTER_ENABLE_MOCAP_PRIORS": "1" if mocap_priors else "0",
        "MONSTER_ENABLE_2K": "1" if enable_2k else "0",
        "MONSTER_ENABLE_3VIEW": "1" if enable_3view else "0",
        "MONSTER_CACHE_MODELS_ON_DRIVE": "1" if cache_models else "0",
        "H3_PROFILE": profile,
        "MONSTER_IDLE_TIMEOUT": str(int(idle_timeout)),
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


_monster_progress_stop: threading.Event | None = None


def _read_json_safe(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _query_live_gpu_telemetry() -> dict[str, Any]:
    name = "unknown"
    used = 0.0
    total = 0.0
    util = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            used = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 1)
            total = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 1)
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
            name = parts[0]
            util = float(parts[1])
            used = float(parts[2])
            total = float(parts[3])
    except Exception:
        pass
    display = (
        f"{name} · 显存: {used:.0f} / {total:.0f} MiB · 利用率: {util:.0f}%"
        if total > 0
        else (f"{name} (CPU)" if name != "unknown" else "CPU 模式")
    )
    return {
        "name": name,
        "vram_used_mib": used,
        "vram_total_mib": total,
        "gpu_util_pct": util,
        "display": display,
    }


def get_progress_data(monster_id: str | None = None) -> dict[str, Any]:
    drive_root = Path(os.environ.get("MONSTER_DRIVE_ROOT", "/content/drive/MyDrive/MonsterStudio"))
    work_root = Path(os.environ.get("MONSTER_WORK_ROOT", "/content/monster-studio"))

    task_data: dict[str, Any] = {}
    train_data: dict[str, Any] = {}

    for p in (work_root / "runtime/current_task_progress.json", drive_root / "runtime/current_task_progress.json"):
        if p.is_file():
            task_data = _read_json_safe(p)
            if task_data:
                break

    active_id = monster_id or task_data.get("monster_id")
    if not active_id:
        candidates = []
        for parent in (work_root / "work", drive_root / "monsters"):
            if parent.is_dir():
                for sub in parent.iterdir():
                    if sub.is_dir() and ((sub / "training_progress.json").exists() or (sub / "task_progress.json").exists()):
                        candidates.append(sub)
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            active_id = candidates[0].name

    if active_id:
        for p in (
            work_root / "work" / active_id / "training_progress.json",
            drive_root / "monsters" / active_id / "training_progress.json",
        ):
            if p.is_file():
                t = _read_json_safe(p)
                if t:
                    train_data = t
                    break

        if not task_data:
            for p in (
                work_root / "work" / active_id / "task_progress.json",
                drive_root / "monsters" / active_id / "task_progress.json",
            ):
                if p.is_file():
                    task_data = _read_json_safe(p)
                    if task_data:
                        break

    stage = train_data.get("stage") or task_data.get("stage") or "idle"
    stage_display = train_data.get("stage_display") or task_data.get("stage_display")
    if not stage_display:
        stages_map = {
            "synthesizing": "合成训练图",
            "training": "训练 DINOv2 姿态头",
            "stage3_dinov2_done": "DINOv2 姿态头训练完成",
            "exam_rendering": "视频考试渲染",
            "exam_inference": "已有视频考试推理",
            "awaiting_exam_review": "考试完成，等待 G3 审阅",
            "rigging": "候选骨骼生成",
            "stage1_completed": "候选骨骼生成完成",
            "animating": "MiniMax H3 动作生成",
            "animating_completed": "动作生成与烘焙完成",
            "idle": "就绪（等待任务）",
        }
        stage_display = stages_map.get(stage, stage)

    status = train_data.get("status") or task_data.get("status") or ("running" if stage in ("training", "synthesizing", "exam_rendering", "exam_inference", "rigging", "animating") else "idle")
    epoch = int(train_data.get("epoch") or task_data.get("epoch") or 0)
    epochs = int(train_data.get("epochs") or task_data.get("epochs") or 15)
    completed_batches = int(train_data.get("completed_batches") or task_data.get("completed_batches") or 0)
    batches_per_epoch = int(train_data.get("batches_per_epoch") or task_data.get("batches_per_epoch") or 0)

    overall_pct = float(train_data.get("overall_pct") or task_data.get("overall_pct") or 0.0)
    if not overall_pct and epochs > 0 and epoch > 0:
        overall_pct = min(100.0, 100.0 * (max(0, epoch - 1) + completed_batches / max(1, batches_per_epoch or 1)) / epochs)
    if train_data.get("training_completed") or status == "completed":
        overall_pct = 100.0

    eta_str = train_data.get("eta_str") or task_data.get("eta_str")
    if not eta_str:
        if stage == "training" and completed_batches > 0:
            elapsed = float(train_data.get("epoch_elapsed_s") or 0)
            if elapsed > 0:
                avg = elapsed / completed_batches
                rem = (max(1, batches_per_epoch) - completed_batches) + max(0, epochs - epoch) * max(1, batches_per_epoch)
                sec = rem * avg
                mins = round(sec / 60)
                eta_str = f"约 {mins} 分钟" if mins >= 1 else f"约 {round(sec)} 秒"
            else:
                eta_str = "计算中"
        elif status == "completed" or train_data.get("training_completed"):
            eta_str = "训练完成"
        else:
            eta_str = "等待任务"

    losses = train_data.get("losses") or train_data.get("batch_loss") or task_data.get("losses") or {}
    history = train_data.get("history") or task_data.get("history") or []
    detail = train_data.get("detail") or task_data.get("detail") or ""
    if not detail and stage == "training":
        detail = f"第 {epoch} / {epochs} 轮 · 本轮 {completed_batches} / {max(1, batches_per_epoch)} 批次 · 预计剩余：{eta_str}"

    gpu_telemetry = _query_live_gpu_telemetry()
    updated_unix = float(train_data.get("updated_unix") or task_data.get("updated_unix") or time.time())
    updated_at = train_data.get("updated_at") or task_data.get("updated_at") or datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "monster_id": active_id or "",
        "stage": stage,
        "stage_display": stage_display,
        "status": status,
        "epoch": epoch,
        "epochs": epochs,
        "completed_batches": completed_batches,
        "batches_per_epoch": batches_per_epoch,
        "overall_pct": round(overall_pct, 1),
        "eta_str": eta_str,
        "losses": losses,
        "history": history,
        "detail": detail,
        "gpu": gpu_telemetry,
        "updated_unix": updated_unix,
        "updated_at": updated_at,
        "drive_path": str(drive_root / "monsters" / active_id) if active_id else str(drive_root),
    }


def render_progress_html(data: dict[str, Any] | None = None, monster_id: str | None = None) -> str:
    if data is None:
        data = get_progress_data(monster_id=monster_id)

    stage = data.get("stage", "idle")
    stage_display = data.get("stage_display", "就绪")
    status = data.get("status", "idle")
    detail = data.get("detail", "")
    epoch = data.get("epoch", 0)
    epochs = data.get("epochs", 15)
    done_batches = data.get("completed_batches", 0)
    total_batches = data.get("batches_per_epoch", 0)
    pct = data.get("overall_pct", 0.0)
    eta_str = data.get("eta_str", "—")
    losses = data.get("losses", {})
    history = data.get("history", [])
    gpu_info = data.get("gpu", {})
    gpu_display = gpu_info.get("display") if isinstance(gpu_info, dict) else str(gpu_info)
    updated_at = data.get("updated_at", "")
    updated_unix = data.get("updated_unix", time.time())
    drive_path = data.get("drive_path", "")

    badge_colors = {
        "running": "background:#1e40af;color:#93c5fd",
        "completed": "background:#065f46;color:#6ee7b7",
        "failed": "background:#991b1b;color:#fca5a5",
        "idle": "background:#374151;color:#d1d5db",
    }
    badge_style = badge_colors.get(status, badge_colors["idle"])
    status_text_map = {"running": "运行中", "completed": "已完成", "failed": "运行失败", "idle": "空闲就绪"}
    status_text = status_text_map.get(status, status)

    loss_chips = ""
    if losses:
        labels = [("关键点 UV loss", "uv"), ("热图 loss", "heatmap"), ("可见性 loss", "visibility")]
        if "dense_xyz" in losses or "dense_mask" in losses:
            labels.extend([("表面静止坐标 loss", "dense_xyz"), ("前景掩码 loss", "dense_mask")])
        labels.append(("总 loss", "total"))
        chips_html = []
        for name, key in labels:
            val = losses.get(key)
            val_str = f"{float(val):.5f}" if val is not None else "—"
            chips_html.append(f'<div class="chip"><span class="k">{html.escape(name)}</span><span class="v">{val_str}</span></div>')
        loss_chips = f'<div class="chips-row">{"".join(chips_html)}</div>'

    loss_table_html = ""
    if history:
        rows = []
        for item in history[-15:]:
            ep_val = item.get("epoch", "—")
            cols = [str(ep_val)]
            for k in ("uv", "heatmap", "visibility", "total"):
                v = item.get(k)
                cols.append(f"{float(v):.5f}" if v is not None else "—")
            rows.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cols) + "</tr>")
        loss_table_html = f'''
        <div class="table-container">
          <table>
            <thead>
              <tr><th>轮次 (Epoch)</th><th>关键点 UV loss</th><th>热图 loss</th><th>可见性 loss</th><th>总 loss</th></tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
          </table>
        </div>
        <p class="meta">总 loss 包含可取负值的深度不确定性惩罚；以实际视频考试指标与 3D 轨迹为准。</p>
        '''

    stale_html = ""
    if status == "running" and (time.time() - updated_unix > 180):
        stale_html = '<p class="warning-banner">⚠️ 进度已超过 3 分钟未更新，可能正在执行长时间渲染或等待批次，请检查日志。</p>'

    error_html = ""
    if status == "failed" and data.get("error"):
        error_html = f'<div class="error-banner"><strong>错误信息：</strong><pre>{html.escape(str(data["error"]))}</pre></div>'

    progress_bar_html = f'''
    <div class="track"><i style="width:{min(100.0, max(0.0, pct)):.1f}%"></i></div>
    <div class="progress-meta">
      <span>整体进度：<strong>{pct:.1f}%</strong></span>
      <span>预计剩余时间：<strong style="color:#70d1be">{html.escape(eta_str)}</strong></span>
    </div>
    '''

    return f'''<div class="monster-progress-root">
<style>
.monster-progress-root {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background: #0e1726;
  color: #e4edf9;
  padding: 24px;
  border-radius: 14px;
  border: 1px solid #1f2e46;
  max-width: 820px;
  margin: 12px 0;
  box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}}
.monster-progress-root .header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}}
.monster-progress-root h2 {{
  margin: 0;
  font-size: 20px;
  font-weight: 600;
  color: #ffffff;
}}
.monster-progress-root .badge {{
  font-size: 12px;
  font-weight: 600;
  padding: 3px 10px;
  border-radius: 9999px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}}
.monster-progress-root .detail {{
  margin: 6px 0 14px;
  color: #94a3b8;
  font-size: 14px;
}}
.monster-progress-root .track {{
  height: 12px;
  background: #1e293b;
  border-radius: 8px;
  overflow: hidden;
  margin: 12px 0 6px;
  position: relative;
}}
.monster-progress-root .track i {{
  display: block;
  height: 100%;
  background: linear-gradient(90deg, #38bdf8, #70d1be);
  border-radius: 8px;
  transition: width 0.4s ease;
}}
.monster-progress-root .progress-meta {{
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  color: #cbd5e1;
  margin-bottom: 14px;
}}
.monster-progress-root .chips-row {{
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 14px 0;
}}
.monster-progress-root .chip {{
  background: #162238;
  border: 1px solid #283a58;
  border-radius: 8px;
  padding: 8px 12px;
  display: flex;
  flex-direction: column;
  min-width: 110px;
}}
.monster-progress-root .chip .k {{
  font-size: 11px;
  color: #94a3b8;
  margin-bottom: 2px;
}}
.monster-progress-root .chip .v {{
  font-size: 15px;
  font-weight: 600;
  color: #38bdf8;
  font-feature-settings: "tnum";
}}
.monster-progress-root .gpu-row {{
  background: #141f33;
  padding: 9px 14px;
  border-radius: 8px;
  font-size: 13px;
  color: #a5b4fc;
  margin: 10px 0;
  display: flex;
  align-items: center;
}}
.monster-progress-root .table-container {{
  max-height: 220px;
  overflow-y: auto;
  margin-top: 14px;
  border: 1px solid #1f2e46;
  border-radius: 8px;
}}
.monster-progress-root table {{
  width: 100%;
  font-size: 12px;
  border-collapse: collapse;
}}
.monster-progress-root th {{
  position: sticky;
  top: 0;
  background: #162238;
  color: #cbd5e1;
  padding: 8px;
  text-align: left;
  border-bottom: 1px solid #283a58;
}}
.monster-progress-root td {{
  padding: 6px 8px;
  border-bottom: 1px solid #1b263b;
  color: #e2e8f0;
  font-feature-settings: "tnum";
}}
.monster-progress-root tr:hover td {{
  background: #1a2742;
}}
.monster-progress-root .meta {{
  font-size: 11px;
  color: #64748b;
  margin: 8px 0 0;
}}
.monster-progress-root .footer {{
  margin-top: 16px;
  padding-top: 12px;
  border-top: 1px solid #1e293b;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 11px;
  color: #64748b;
}}
.monster-progress-root .warning-banner {{
  background: rgba(245, 158, 11, 0.15);
  color: #fbbf24;
  border: 1px solid rgba(245, 158, 11, 0.3);
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 12px;
  margin: 10px 0;
}}
.monster-progress-root .error-banner {{
  background: rgba(239, 68, 68, 0.15);
  color: #fca5a5;
  border: 1px solid rgba(239, 68, 68, 0.3);
  padding: 10px 12px;
  border-radius: 6px;
  font-size: 12px;
  margin: 10px 0;
}}
</style>
<div class="header">
  <h2>Monster · {html.escape(stage_display)}</h2>
  <span class="badge" style="{badge_style}">{status_text}</span>
</div>
<div class="detail">{html.escape(detail)}</div>
{progress_bar_html}
{loss_chips}
<div class="gpu-row">🖥️ <strong>GPU 状态：</strong>&nbsp;{html.escape(gpu_display)}</div>
{loss_table_html}
{stale_html}
{error_html}
<div class="footer">
  <span>台北时间 {html.escape(updated_at)} · 实时只读监控</span>
  <span>{html.escape(data.get("monster_id", ""))}</span>
</div>
</div>'''


def show_progress(monster_id: str | None = None, interval: int = 5, snapshot_only: bool = False) -> Any:
    global _monster_progress_stop
    if _monster_progress_stop is not None:
        _monster_progress_stop.set()
    _monster_progress_stop = threading.Event()

    html_content = render_progress_html(monster_id=monster_id)
    try:
        from IPython.display import HTML, display
        handle = display(HTML(html_content), display_id=True)
        if snapshot_only:
            return handle

        def _refresh_loop(stop_ev: threading.Event, disp_handle: Any) -> None:
            while not stop_ev.wait(interval):
                try:
                    updated = render_progress_html(monster_id=monster_id)
                    disp_handle.update(HTML(updated))
                except Exception:
                    pass

        t = threading.Thread(target=_refresh_loop, args=(_monster_progress_stop, handle), daemon=True)
        t.start()
        return handle
    except ImportError:
        print("[Monster Task Progress]")
        print(json.dumps(get_progress_data(monster_id), indent=2, ensure_ascii=False))
        return html_content
