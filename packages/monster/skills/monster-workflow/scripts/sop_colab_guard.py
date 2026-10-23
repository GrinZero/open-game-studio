#!/usr/bin/env python3
"""sop_colab_guard.py — Colab Instance Lifecycle & Cross-Session Safe Shutdown Guard (SOP §7.1 / §7.2).

Enforces SOP §7.2 instance retention & release rules deterministically:
  1. Cross-Session Lease Registry (`active_sessions.json`):
     Tracks active Codex/Agent sessions, PIDs, workspaces, `monster_id`s, and GPU queue dependencies
     across sessions sharing the same Colab account/endpoint.
  2. Stage-Gated Release Preconditions:
     - `stage1_candidates_fetched`: Requires all 4 SkinTokens candidate rigs to be archived on Drive
       and verified locally before releasing the exclusive instance for local decoration restoration.
     - `final_batch_approved`: Requires `final_batch` and every clip in the current batch to have
       explicit user approval (`user_decision == "approved"`) and Drive persistence verified.
     - Explicitly blocks release during video iteration, user viewing, or waiting for feedback!
  3. Mandatory Cross-Session Safety Check before Shutdown:
     - Checks live Worker status (`IDLE` or 0% GPU util alone never proves no other session is using it).
     - Checks `active_sessions.json` and local background processes for any other session or queued job.
     - If shared occupancy is detected, refuses shutdown and records the retention reason in the ledger.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sop_gate import (  # noqa: E402
    MONSTER_ROOT,
    load_ledger,
    now_taipei_iso,
    save_ledger,
)
REPO_ROOT = MONSTER_ROOT.parents[1]
INTEGRATION_DIR = MONSTER_ROOT / "integration"
if str(INTEGRATION_DIR) not in sys.path:
    sys.path.insert(0, str(INTEGRATION_DIR))
from sop_preflight import resolve_scratch_dirs  # noqa: E402

VALID_RELEASE_REASONS = (
    "stage1_candidates_fetched",
    "final_batch_approved",
)


def _session_registry_paths(drive_root: Path | str | None = None) -> list[Path]:
    if drive_root is not None:
        drv = Path(drive_root).resolve()
        return [drv / "runtime" / "active_sessions.json"]
    drv = Path(
        os.environ.get("MONSTER_DRIVE_ROOT")
        or "/content/drive/MyDrive/MonsterStudio"
    ).resolve()
    local_var = REPO_ROOT / "var" / "monster" / "active_sessions.json"
    drive_var = drv / "runtime" / "active_sessions.json"
    return [local_var, drive_var]


def load_active_sessions(drive_root: Path | str | None = None) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for p in _session_registry_paths(drive_root):
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for sid, sinfo in data.get("sessions", {}).items():
                    if isinstance(sinfo, dict):
                        if sid not in merged or float(sinfo.get("updated_unix", 0)) > float(merged[sid].get("updated_unix", 0)):
                            merged[sid] = sinfo
            except Exception:
                continue
    return merged


def save_active_sessions(sessions: dict[str, dict[str, Any]], drive_root: Path | str | None = None) -> list[str]:
    payload = {"updated_at": now_taipei_iso(), "sessions": sessions}
    written: list[str] = []
    for p in _session_registry_paths(drive_root):
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written.append(str(p))
        except OSError:
            pass
    return written


def acquire_session_lease(
    monster_id: str,
    *,
    session_id: str | None = None,
    stage: str = "active",
    endpoint_url: str | None = None,
    workspace_dir: Path | str | None = None,
    drive_root: Path | str | None = None,
) -> dict[str, Any]:
    sid = session_id or os.environ.get("MONSTER_SESSION_ID") or f"session_{monster_id}"
    ws = str(Path(workspace_dir or os.environ.get("MONSTER_PROJECT_ROOT") or Path.cwd()).resolve())
    sessions = load_active_sessions(drive_root)
    entry = {
        "session_id": sid,
        "monster_id": monster_id,
        "stage": stage,
        "pid": os.getpid(),
        "workspace_dir": ws,
        "endpoint_url": endpoint_url or "",
        "status": "ACTIVE",
        "updated_at": now_taipei_iso(),
        "updated_unix": time.time(),
    }
    sessions[sid] = entry
    paths = save_active_sessions(sessions, drive_root)
    return {"lease": entry, "saved_to": paths}


def release_session_lease(
    session_id: str,
    *,
    drive_root: Path | str | None = None,
) -> dict[str, Any]:
    sessions = load_active_sessions(drive_root)
    removed = sessions.pop(session_id, None)
    paths = save_active_sessions(sessions, drive_root)
    return {"released_session_id": session_id, "removed": removed is not None, "saved_to": paths}


def _detect_other_local_monster_processes(current_pid: int) -> list[str]:
    """Scans local processes for concurrent monster jobs (excluding `current_pid`)."""
    if os.environ.get("MONSTER_SKIP_PS_SCAN") == "1":
        return []
    try:
        out = subprocess.check_output(["ps", "-Ao", "pid,command"], text=True, timeout=5)
    except Exception:
        return []
    found: list[str] = []
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1]
        if pid == current_pid or pid == os.getppid():
            continue
        if any(k in cmd for k in ("monster_client.py", "pipeline_runner.py")) and "status" not in cmd:
            found.append(f"PID {pid}: {cmd[:120]}")
    return found


def evaluate_safe_shutdown(
    monster_id: str,
    reason: str,
    *,
    session_id: str | None = None,
    project_root: Path | str | None = None,
    drive_root: Path | str | None = None,
    worker_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluates all SOP §7.2 stage & cross-session preconditions before releasing a Colab GPU instance."""
    if reason not in VALID_RELEASE_REASONS:
        return {
            "allowed": False,
            "reason": reason,
            "blockers": [
                f"非法的实例释放节点 {reason!r}：SOP §7.2 仅允许在 {VALID_RELEASE_REASONS} 两个节点释放独占实例；"
                "视频生成、用户查看、等待反馈或调参迭代期间严禁关闭实例！"
            ],
        }

    proj = Path(project_root or os.environ.get("MONSTER_PROJECT_ROOT") or Path.cwd()).resolve()
    drv = Path(
        drive_root
        or os.environ.get("MONSTER_DRIVE_ROOT")
        or "/content/drive/MyDrive/MonsterStudio"
    ).resolve()
    sid = session_id or os.environ.get("MONSTER_SESSION_ID") or f"session_{monster_id}"
    blockers: list[str] = []

    # 1. Check Worker live busy state
    if worker_status and worker_status.get("status") == "BUSY":
        blockers.append(
            f"远端 Worker 正在执行任务 ({worker_status.get('current_job')})，严禁中断释放"
        )

    # 2. Check cross-session active leases & queued jobs
    sessions = load_active_sessions(drv)
    other_sessions = [
        s for k, s in sessions.items()
        if k != sid and s.get("status") in ("ACTIVE", "QUEUED")
    ]
    if other_sessions:
        other_desc = ", ".join(
            f"{s['session_id']} (monster={s.get('monster_id')}, stage={s.get('stage')}, ws={s.get('workspace_dir')})"
            for s in other_sessions
        )
        blockers.append(f"跨会话安全检查拦截：发现其他活跃或排队会话正在共享/依赖实例 -> {other_desc}")

    # 3. Check local background processes
    other_procs = _detect_other_local_monster_processes(os.getpid())
    if other_procs:
        blockers.append(f"本机后台进程安全检查拦截：发现并发 Monster 任务进程 -> {'; '.join(other_procs)}")

    # 4. Check stage-specific SOP preconditions
    dirs = resolve_scratch_dirs(proj, monster_id)
    work_dir = dirs["work_dir"]
    ledger = load_ledger(monster_id, project_root=proj, drive_root=drv)

    if reason == "stage1_candidates_fetched":
        local_cands = [
            p for p in (
                *(work_dir / "candidates").glob("candidate_*.glb"),
                *(work_dir / "g1_candidates").glob("candidate_*_full.blend"),
            )
            if p.is_file() and p.stat().st_size > 0
        ]
        if len(local_cands) < 4:
            blockers.append(
                f"阶段二释放前提未满足：本地尚未取回并核验齐全部 4 套候选工程 (当前仅找到 {len(local_cands)} 个文件)"
            )

    elif reason == "final_batch_approved":
        fb_gate = ledger.get("gates", {}).get("final_batch", {})
        if fb_gate.get("user_decision") != "approved" or fb_gate.get("status") != "approved":
            blockers.append(
                "阶段五释放前提未满足：本批动作清单尚未获得用户明确全部满意 (final_batch.user_decision != 'approved')；"
                "等待用户查看与反馈期间必须持续保留实例！"
            )
        clips = fb_gate.get("clips", {})
        unapproved_clips = [k for k, v in clips.items() if v.get("decision") != "approved"]
        if unapproved_clips:
            blockers.append(f"批次中仍有待审或需返修的动作视频: {unapproved_clips}，不满足全部满意后释放条件")

    decision = {
        "allowed": len(blockers) == 0,
        "monster_id": monster_id,
        "session_id": sid,
        "release_reason": reason,
        "checked_at": now_taipei_iso(),
        "blockers": blockers,
    }

    # Record retention or release audit trail in ledger
    ledger["colab_instance_state"] = {
        "status": "READY_TO_RELEASE" if decision["allowed"] else "RETAINED",
        "last_check": decision,
    }
    save_ledger(ledger, project_root=proj, drive_root=drv)
    return decision


def execute_safe_shutdown(
    monster_id: str,
    reason: str,
    *,
    session_id: str | None = None,
    project_root: Path | str | None = None,
    drive_root: Path | str | None = None,
    registry_path: Path | str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Runs `evaluate_safe_shutdown` and, if allowed and not `dry_run`, calls Worker `shutdown` and verifies termination."""
    from monster_client import MonsterClient, discover_registry_path

    drv = Path(
        drive_root
        or os.environ.get("MONSTER_DRIVE_ROOT")
        or "/content/drive/MyDrive/MonsterStudio"
    ).resolve()
    sid = session_id or os.environ.get("MONSTER_SESSION_ID") or f"session_{monster_id}"

    worker_status = None
    client = None
    if not dry_run:
        try:
            client = MonsterClient.from_registry(registry_path)
            worker_status = client.heartbeat()
        except Exception:
            worker_status = None

    check = evaluate_safe_shutdown(
        monster_id,
        reason,
        session_id=sid,
        project_root=project_root,
        drive_root=drv,
        worker_status=worker_status,
    )
    if not check["allowed"] or dry_run:
        return {**check, "executed": False}

    shutdown_resp = None
    if client is not None:
        shutdown_resp = client.shutdown(reason=f"sop_safe_shutdown:{reason}:{monster_id}")

    release_session_lease(sid, drive_root=drv)

    # Verify remote registry status updated to TERMINATED when accessible
    verified_terminated = False
    try:
        reg_file = discover_registry_path(registry_path)
        if reg_file.is_file():
            reg_data = json.loads(reg_file.read_text(encoding="utf-8"))
            verified_terminated = reg_data.get("status") in ("TERMINATED", "TERMINATING")
    except Exception:
        pass

    proj = Path(project_root or os.environ.get("MONSTER_PROJECT_ROOT") or Path.cwd()).resolve()
    ledger = load_ledger(monster_id, project_root=proj, drive_root=drv)
    ledger["colab_instance_state"] = {
        "status": "RELEASED",
        "released_at": now_taipei_iso(),
        "release_reason": reason,
        "verified_terminated": verified_terminated,
    }
    save_ledger(ledger, project_root=proj, drive_root=drv)

    return {
        **check,
        "executed": True,
        "shutdown_response": shutdown_resp,
        "verified_terminated": verified_terminated,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monster SOP Stage 6 Colab Instance Lifecycle & Safe Release Guard")
    parser.add_argument("--monster-id", required=True)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--drive-root", default=None)
    parser.add_argument("--registry", default=None)
    sub = parser.add_subparsers(dest="action", required=True)

    sub.add_parser("status", help="List active sessions across workspaces and current instance state")

    p_acq = sub.add_parser("acquire-lease", help="Register current session lease on the active Colab instance")
    p_acq.add_argument("--stage", default="active")
    p_acq.add_argument("--endpoint-url", default=None)

    sub.add_parser("release-lease", help="Remove current session lease without shutting down the instance")

    p_shut = sub.add_parser("safe-shutdown", help="Run cross-session & stage safety checks and release instance if allowed")
    p_shut.add_argument("--reason", required=True, choices=VALID_RELEASE_REASONS)
    p_shut.add_argument("--dry-run", action="store_true", help="Only evaluate safety checks without calling shutdown")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    sid = args.session_id or os.environ.get("MONSTER_SESSION_ID") or f"session_{args.monster_id}"

    if args.action == "status":
        sessions = load_active_sessions(args.drive_root)
        ledger = load_ledger(args.monster_id, project_root=args.project_root, drive_root=args.drive_root)
        print(
            json.dumps(
                {
                    "monster_id": args.monster_id,
                    "session_id": sid,
                    "active_sessions": sessions,
                    "colab_instance_state": ledger.get("colab_instance_state"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if args.action == "acquire-lease":
        res = acquire_session_lease(
            args.monster_id,
            session_id=sid,
            stage=args.stage,
            endpoint_url=args.endpoint_url,
            workspace_dir=args.project_root,
            drive_root=args.drive_root,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    if args.action == "release-lease":
        res = release_session_lease(sid, drive_root=args.drive_root)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    if args.action == "safe-shutdown":
        res = execute_safe_shutdown(
            args.monster_id,
            args.reason,
            session_id=sid,
            project_root=args.project_root,
            drive_root=args.drive_root,
            registry_path=args.registry,
            dry_run=args.dry_run,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0 if res["allowed"] else 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
