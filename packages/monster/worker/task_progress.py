"""Training progress helpers that require only the Python standard library."""
import json
from pathlib import Path
from typing import Any


def format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "即将完成"
    sec = int(round(seconds))
    if sec < 60:
        return f"约 {sec} 秒"
    mins = round(sec / 60)
    if mins < 60:
        return f"约 {mins} 分钟"
    hours = mins // 60
    rem_mins = mins % 60
    return f"约 {hours} 小时 {rem_mins} 分钟"


def write_progress_files(targets: list[Path], data: dict[str, Any]) -> None:
    for target in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(target)
        except Exception:
            pass
