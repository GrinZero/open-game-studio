#!/usr/bin/env python3
"""monster_sop.py — Unified Executable SOP Entry Point for the Monster Workflow Skill.

Provides a single deterministic CLI covering all 7 MonsterStudio SOP modules so the Agent never
needs to write ad-hoc Python/Blender/FFmpeg/JSON glue code:

  1. `gate`        -> `sop_gate.py`        (Audit ledger, SHA-256 fingerprints, graded invalidation, hard gates)
  2. `preflight`   -> `sop_preflight.py`   (Stage 1 workspace redline guard, model preflight, bone naming gate, U1 sync)
  3. `g1-pack`     -> `sop_g1_pack.py`     (Stage 2 4-candidate decoration restore, canonical naming, 16 views, 4-archetype videos, G1 HTML)
  4. `g2-pack`     -> `sop_g2_pack.py     (Stage 3 50 random poses, 2D overlays, 50-thumb grid, redraw log, G2 HTML)
  5. `g3-exam`     -> `sop_g3_exam.py`     (Stage 4 U2 incremental sync, existing-video exam, 3-boundary report, GLB comparison, G3 HTML)
  6. `deliver`     -> `sop_delivery.py`    (Stage 5 sync hard-stop check, current-batch manifest, local GLB + loop seam check, delivery HTML)
  7. `colab-guard` -> `sop_colab_guard.py` (Stage 6 cross-session lease registry & stage-gated safe GPU shutdown)
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sop_colab_guard  # noqa: E402
import sop_delivery  # noqa: E402
import sop_g1_pack  # noqa: E402
import sop_g2_pack  # noqa: E402
import sop_g3_exam  # noqa: E402
import sop_gate  # noqa: E402
import sop_preflight  # noqa: E402

DISPATCH = {
    "gate": sop_gate.main,
    "preflight": sop_preflight.main,
    "g1-pack": sop_g1_pack.main,
    "g2-pack": sop_g2_pack.main,
    "g3-exam": sop_g3_exam.main,
    "deliver": sop_delivery.main,
    "colab-guard": sop_colab_guard.main,
}


def print_usage() -> None:
    print(
        "Usage: monster_sop.py <command> [args...]\n\n"
        "Available SOP commands:\n"
        "  gate         Audit ledger, version fingerprints, graded invalidation & G1/G2/G3 hard gates\n"
        "  preflight    Stage 1 workspace redline guard, model/bone preflight & U1 Drive sync\n"
        "  g1-pack      Stage 2 4-candidate decoration restore, canonical bone naming, 16 views, motion videos & G1 HTML\n"
        "  g2-pack      Stage 3 50 true random poses, 2D overlays, 50-thumb overview, redraw log & G2 HTML\n"
        "  g3-exam      Stage 4 U2 incremental sync, existing-video exam, 3-boundary report, GLB re-render & G3 HTML\n"
        "  deliver      Stage 5 current-batch verification, local GLB round-trip & loop-seam check\n"
        "  colab-guard  Stage 6 cross-session lease registry & stage-gated safe Colab shutdown\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print_usage()
        return 0
    cmd = args[0]
    rest = args[1:]
    if cmd not in DISPATCH:
        print(f"Unknown SOP command: {cmd!r}\n", file=sys.stderr)
        print_usage()
        return 1
    return int(DISPATCH[cmd](rest))


if __name__ == "__main__":
    raise SystemExit(main())
