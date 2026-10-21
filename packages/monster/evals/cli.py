#!/usr/bin/env python3
"""cli.py — Unified Command-Line Interface for Monster Studio Video-to-Animation Evaluation.

Usage examples:
  npm run monster:eval -- dataset pack-from-starmaker
  npm run monster:eval -- dataset verify fantasy_creature_2k_v1
  npm run monster:eval -- dataset pull fantasy_creature_2k_v1 --drive-file-id <ID>
  npm run monster:eval -- run fantasy_creature_2k_v1 --tier fast
  npm run monster:eval -- run fantasy_creature_2k_v1 --tier fast --save-baseline
  npm run monster:eval -- diagnose fantasy_creature_2k_v1 --frame 17
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

MONSTER_ROOT = Path(__file__).resolve().parents[1]
WORKER_DIR = MONSTER_ROOT / "worker"
if str(MONSTER_ROOT) not in sys.path:
    sys.path.insert(0, str(MONSTER_ROOT))
if str(WORKER_DIR) not in sys.path:
    sys.path.insert(0, str(WORKER_DIR))

from bone_taxonomy import (  # noqa: E402
    DEFAULT_SYNTH_SAMPLES,
    audit_pose_dataset_coverage,
    classify_rig_bones,
    sample_stratified_pose_euler_dict,
)

from evals.datasets.drive_bundle import (  # noqa: E402
    DEFAULT_BENCHMARK_ID,
    pack_from_starmaker_2k,
    pull_fixture_bundle,
)
from evals.datasets.loader import (  # noqa: E402
    load_benchmark_fixture,
    verify_fixture_directory,
)
from evals.metrics.evaluator import diagnose_single_frame  # noqa: E402
from evals.reporters.markdown_reporter import render_frame_diagnosis_markdown  # noqa: E402
from evals.runners.tier1_fast_solver import execute_tier1_fast  # noqa: E402
from evals.runners.tier2_perception_infer import execute_tier2_infer  # noqa: E402
from evals.runners.tier3_full_blender import execute_tier3_full_blender  # noqa: E402
from evals.solvers.latest_solver import run_latest_solver  # noqa: E402


def _cmd_dataset(args: argparse.Namespace) -> int:
    action = args.dataset_action
    if action == "pack-from-starmaker":
        result = pack_from_starmaker_2k(
            benchmark_id=args.benchmark_id,
            share_url=args.drive_file_id or "",
        )
        print(json.dumps(result, indent=2))
        return 0

    if action == "verify":
        report = verify_fixture_directory(benchmark_id=args.benchmark_id)
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] else 1

    if action == "pull":
        archive_p = Path(args.archive_path).resolve() if args.archive_path else None
        result = pull_fixture_bundle(
            benchmark_id=args.benchmark_id,
            url_or_id=args.drive_file_id,
            from_archive=archive_p,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["valid"] else 1

    raise ValueError(f"Unknown dataset action: {action!r}")


def _cmd_run(args: argparse.Namespace) -> int:
    tier = args.tier
    benchmark_id = args.benchmark_id

    if tier == "fast":
        out = execute_tier1_fast(
            benchmark_id=benchmark_id,
            save_baseline=args.save_baseline,
        )
    elif tier == "infer":
        out = execute_tier2_infer(
            benchmark_id=benchmark_id,
            save_baseline=args.save_baseline,
        )
    elif tier == "full":
        out = execute_tier3_full_blender(
            benchmark_id=benchmark_id,
            save_baseline=args.save_baseline,
        )
    else:
        raise ValueError(f"Unsupported tier: {tier!r}")

    if args.json:
        print(
            json.dumps(
                {
                    "run_dir": out["run_dir"],
                    "report_path": out["report_path"],
                    "summary_path": out["summary_path"],
                    "html_path": out["html_path"],
                    "baseline_saved_path": out["baseline_saved_path"],
                    "primary_bones": out["report"]["stratified_kinematics"]["primary_bones"],
                    "all_bones": out["report"]["stratified_kinematics"]["all_bones"],
                    "golden_frames_passed": out["report"]["golden_frames"]["all_passed"],
                },
                indent=2,
            )
        )
    else:
        print(out["summary_markdown"])
        print(f"Run artifacts written to: {out['run_dir']}")
        print(f"  - JSON Report : {out['report_path']}")
        print(f"  - Markdown    : {out['summary_path']}")
        print(f"  - HTML Review : {out['html_path']}")
        if out["baseline_saved_path"]:
            print(f"  - Baseline    : {out['baseline_saved_path']}")

    return 0 if out["report"]["golden_frames"]["all_passed"] else 2


def _cmd_diagnose(args: argparse.Namespace) -> int:
    fixture = load_benchmark_fixture(benchmark_id=args.benchmark_id)
    solver_res = run_latest_solver(
        observations=fixture["observations"],
        studio_rig_meta=fixture["rig_meta"],
        constraints_meta=fixture["constraints"],
    )
    diag = diagnose_single_frame(
        solver_res=solver_res,
        exam_gt=fixture["exam_gt"],
        frame_idx=args.frame,
        top_k=args.top_k,
    )
    if args.json:
        print(json.dumps(diag, indent=2))
    else:
        print(render_frame_diagnosis_markdown(diag))
    return 0


def _cmd_audit_sampling(args: argparse.Namespace) -> int:
    fixture = load_benchmark_fixture(benchmark_id=args.benchmark_id)
    rig_meta = fixture["rig_meta"]
    classified = classify_rig_bones(
        rig_meta.get("bones", {}),
        bbox_min=rig_meta.get("bbox_min"),
        bbox_max=rig_meta.get("bbox_max"),
        constraints_meta=fixture.get("constraints"),
    )
    rng = random.Random(int(args.seed))
    records = []
    for idx in range(int(args.samples)):
        eulers, regime = sample_stratified_pose_euler_dict(
            classified,
            rng=rng,
            sample_idx=idx,
            enforce_anatomical_floor=True,
        )
        records.append({"sampling_regime": regime, "joints": eulers})
    report = audit_pose_dataset_coverage(records, classified, enforce_anatomical_floor=True)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="monster-eval",
        description="Monster Studio Video-to-Animation Reconstruction Evaluation Suite (2K Benchmark, Latest Solver)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # dataset subcommands
    p_ds = sub.add_parser("dataset", help="Pack, pull, or verify 2K evaluation fixture bundles")
    ds_sub = p_ds.add_subparsers(dest="dataset_action", required=True)

    p_pack = ds_sub.add_parser(
        "pack-from-starmaker",
        help="Pack the 2K benchmark fixture from star-maker into var/monster/evals/archives/fantasy_creature_2k_v1.tar.gz",
    )
    p_pack.add_argument("--benchmark-id", default=DEFAULT_BENCHMARK_ID)
    p_pack.add_argument("--drive-file-id", default=None, help="Optional Google Drive file ID to record in manifest.json")

    p_verify = ds_sub.add_parser("verify", help="Verify local fixture files against manifest.json SHA-256 digests")
    p_verify.add_argument("benchmark_id", nargs="?", default=DEFAULT_BENCHMARK_ID)

    p_pull = ds_sub.add_parser("pull", help="Extract or download a fixture archive and verify SHA-256 checksums")
    p_pull.add_argument("benchmark_id", nargs="?", default=DEFAULT_BENCHMARK_ID)
    p_pull.add_argument("--archive-path", default=None, help="Local .tar.gz path (defaults to var/monster/evals/archives/<id>.tar.gz)")
    p_pull.add_argument("--drive-file-id", default=None, help="Google Drive file ID override")
    p_pull.add_argument("--force", action="store_true", help="Re-download and re-extract even if already verified")

    # run subcommand
    p_run = sub.add_parser("run", help="Run evaluation on a benchmark fixture")
    p_run.add_argument("benchmark_id", nargs="?", default=DEFAULT_BENCHMARK_ID)
    p_run.add_argument(
        "--tier",
        choices=("fast", "infer", "full"),
        default="fast",
        help="Evaluation tier: fast (CPU ~1.5s), infer (MPS/CUDA DINOv2 + solver), full (Headless Blender GLB roundtrip)",
    )
    p_run.add_argument("--save-baseline", action="store_true", help="Update baseline_latest.json in benchmarks/<id>/")
    p_run.add_argument("--json", action="store_true", help="Print compact JSON summary to stdout")

    # diagnose subcommand
    p_diag = sub.add_parser("diagnose", help="Inspect stage-by-stage error attribution on a single frame")
    p_diag.add_argument("benchmark_id", nargs="?", default=DEFAULT_BENCHMARK_ID)
    p_diag.add_argument("--frame", type=int, default=17, help="Frame index to diagnose (default: 17)")
    p_diag.add_argument("--top-k", type=int, default=12, help="Number of top primary bones to list")
    p_diag.add_argument("--json", action="store_true", help="Output raw JSON diagnosis")

    # audit-sampling subcommand
    p_aud = sub.add_parser(
        "audit-sampling",
        help="Dry-run and audit the 5,000-sample 4-regime decoupled pose sampler on a benchmark rig",
    )
    p_aud.add_argument("benchmark_id", nargs="?", default=DEFAULT_BENCHMARK_ID)
    p_aud.add_argument("--samples", type=int, default=DEFAULT_SYNTH_SAMPLES, help="Number of poses to sample (default: 5000)")
    p_aud.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "dataset":
        return _cmd_dataset(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "diagnose":
        return _cmd_diagnose(args)
    if args.command == "audit-sampling":
        return _cmd_audit_sampling(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
