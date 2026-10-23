#!/usr/bin/env python3
"""monster_sop.py — Unified Executable SOP Entry Point for the Monster Workflow Skill.

Provides a single deterministic CLI covering all 7 MonsterStudio SOP modules so the Agent never
needs to write ad-hoc Python/Blender/FFmpeg/JSON glue code:

  1. `gate`         -> `sop_gate.py`         (Audit ledger, SHA-256 fingerprints, graded invalidation, hard gates)
  2. `preflight`    -> `sop_preflight.py`    (Stage 1 workspace redline guard, model preflight, bone naming gate, U1 sync)
  3. `g1-pack`      -> `sop_g1_pack.py`      (Stage 2 4-candidate decoration restore, canonical naming, 16 views, 4-archetype videos, G1 HTML)
  4. `joint-limits` -> `sop_joint_limits.py` (Post-G1 geometric & LBS axis probe, Agent-authored `joint_limits.json` customizer)
  5. `g2-pack`      -> `sop_g2_pack.py`      (Stage 3 50 boundary-exploration random poses, 2D overlays, 50-thumb grid, redraw log, G2 HTML)
  6. `g3-exam`      -> `sop_g3_exam.py`      (Stage 4 U2 incremental sync, existing-video exam, 3-boundary report, GLB comparison, G3 HTML)
  7. `deliver`      -> `sop_delivery.py`     (Stage 5 sync hard-stop check, current-batch manifest, local GLB + loop seam check, delivery HTML)
  8. `colab-guard`  -> `sop_colab_guard.py`  (Stage 6 cross-session lease registry & stage-gated safe GPU shutdown)
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _ensure_venv_site_packages() -> None:
    """Adds `packages/monster/.venv` site-packages to `sys.path` when invoked via system `python3`."""
    try:
        import numpy  # noqa: F401
        return
    except ImportError:
        pass

    candidate_roots = [
        SCRIPT_DIR.parents[2],  # packages/monster
        Path.home() / ".local/share/codex-integrations/game-studio-local/runtime/packages/monster",
        Path("/Users/bugyaluwang/project/codex-game-studio/packages/monster"),
    ]
    for root in candidate_roots:
        venv_lib = root / ".venv" / "lib"
        if venv_lib.is_dir():
            for sp in sorted(venv_lib.glob("python*/site-packages")):
                if sp.is_dir() and str(sp) not in sys.path:
                    sys.path.insert(0, str(sp))


_ensure_venv_site_packages()

DISPATCH_MODULES = {
    "gate": "sop_gate",
    "preflight": "sop_preflight",
    "g1-pack": "sop_g1_pack",
    "joint-limits": "sop_joint_limits",
    "g2-pack": "sop_g2_pack",
    "g3-exam": "sop_g3_exam",
    "deliver": "sop_delivery",
    "colab-guard": "sop_colab_guard",
}


def print_usage() -> None:
    print(
        "Usage: monster_sop.py <command> [args...]\n\n"
        "Available SOP commands:\n"
        "  gate          Audit ledger, version fingerprints, graded invalidation & G1/G2/G3 hard gates\n"
        "  preflight     Stage 1 workspace redline guard, model/bone preflight & U1 Drive sync\n"
        "  g1-pack       Stage 2 4-candidate decoration restore, canonical bone naming, 16 views, motion videos & G1 HTML\n"
        "  joint-limits  Post-G1 geometric & LBS axis probe and Agent-authored joint_limits.json customizer\n"
        "  g2-pack       Stage 3 50 boundary-exploration random poses, 2D overlays, 50-thumb overview, redraw log & G2 HTML\n"
        "  g3-exam       Stage 4 U2 incremental sync, existing-video exam, 3-boundary report, GLB re-render & G3 HTML\n"
        "  deliver       Stage 5 current-batch verification, local GLB round-trip & loop-seam check\n"
        "  colab-guard   Stage 6 cross-session lease registry & stage-gated safe Colab shutdown\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print_usage()
        return 0
    cmd = args[0]
    rest = args[1:]
    if cmd not in DISPATCH_MODULES:
        print(f"Unknown SOP command: {cmd!r}\n", file=sys.stderr)
        print_usage()
        return 1
    mod = importlib.import_module(DISPATCH_MODULES[cmd])
    return int(mod.main(rest))


if __name__ == "__main__":
    raise SystemExit(main())
