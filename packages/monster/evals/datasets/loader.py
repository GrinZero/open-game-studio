"""Fixture loader and SHA-256 integrity verifier for Monster Studio Evals."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

from .drive_bundle import EVALS_ROOT, VAR_EVALS_ROOT, sha256_file


def load_manifest(benchmark_id: str = "fantasy_creature_2k_v2") -> dict[str, Any]:
    manifest_path = EVALS_ROOT / "benchmarks" / benchmark_id / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Benchmark manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_golden_frames(benchmark_id: str = "fantasy_creature_2k_v2") -> dict[str, Any]:
    golden_path = EVALS_ROOT / "benchmarks" / benchmark_id / "golden_frames.json"
    if not golden_path.is_file():
        return {"frames": []}
    return json.loads(golden_path.read_text(encoding="utf-8"))


def load_baseline_metrics(benchmark_id: str = "fantasy_creature_2k_v2") -> dict[str, Any] | None:
    baseline_path = EVALS_ROOT / "benchmarks" / benchmark_id / "baseline_latest.json"
    if not baseline_path.is_file():
        return None
    return json.loads(baseline_path.read_text(encoding="utf-8"))


def verify_fixture_directory(
    benchmark_id: str = "fantasy_creature_2k_v2",
    fixture_dir: Path | None = None,
) -> dict[str, Any]:
    """Verifies that all benchmark files exist locally under `var/monster/evals/fixtures/<benchmark_id>/`
    and match the exact byte counts and SHA-256 hashes in `manifest.json`.
    """
    manifest = load_manifest(benchmark_id)
    root = (fixture_dir or (VAR_EVALS_ROOT / "fixtures" / benchmark_id)).resolve()
    checks: dict[str, Any] = {}
    all_valid = True

    for key, spec in manifest["files"].items():
        fpath = root / spec["rel_path"]
        if not fpath.is_file():
            checks[key] = {"valid": False, "reason": "missing", "path": str(fpath)}
            all_valid = False
            continue
        actual_bytes = fpath.stat().st_size
        actual_sha = sha256_file(fpath)
        ok = (actual_bytes == spec["bytes"]) and (actual_sha == spec["sha256"])
        if not ok:
            all_valid = False
        checks[key] = {
            "valid": ok,
            "path": str(fpath),
            "bytes": actual_bytes,
            "expected_bytes": spec["bytes"],
            "sha256": actual_sha,
            "expected_sha256": spec["sha256"],
        }

    return {
        "benchmark_id": benchmark_id,
        "fixture_dir": str(root),
        "valid": all_valid,
        "files": checks,
    }


def load_benchmark_fixture(
    benchmark_id: str = "fantasy_creature_2k_v2",
    fixture_dir: Path | None = None,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    """Loads all metadata, cached observations, and ground truth for `benchmark_id`."""
    import numpy as np

    manifest = load_manifest(benchmark_id)
    root = (fixture_dir or (VAR_EVALS_ROOT / "fixtures" / benchmark_id)).resolve()
    if verify_hashes:
        ver = verify_fixture_directory(benchmark_id, fixture_dir=root)
        if not ver["valid"]:
            raise RuntimeError(
                f"Fixture directory {root} is missing or corrupted. "
                f"Run `npm run monster:eval -- dataset pull {benchmark_id}` or `dataset pack-from-starmaker` first.\n"
                f"Details: {json.dumps(ver, indent=2)}"
            )

    files = manifest["files"]
    rig_meta = json.loads((root / files["studio_rig_meta"]["rel_path"]).read_text(encoding="utf-8"))
    constraints = json.loads((root / files["constraints"]["rel_path"]).read_text(encoding="utf-8"))
    motion_spec = json.loads((root / files["motion_spec"]["rel_path"]).read_text(encoding="utf-8"))
    inference_receipt = json.loads((root / files["inference_receipt"]["rel_path"]).read_text(encoding="utf-8"))

    gt_gz_path = root / files["exam_gt"]["rel_path"]
    with gzip.open(gt_gz_path, "rt", encoding="utf-8") as gz_in:
        exam_gt = json.load(gz_in)

    obs_path = root / files["observations_npz"]["rel_path"]
    observations = dict(np.load(obs_path))

    return {
        "benchmark_id": benchmark_id,
        "fixture_dir": root,
        "manifest": manifest,
        "golden_frames": load_golden_frames(benchmark_id),
        "baseline_latest": load_baseline_metrics(benchmark_id),
        "rig_meta": rig_meta,
        "constraints": constraints,
        "motion_spec": motion_spec,
        "inference_receipt": inference_receipt,
        "exam_gt": exam_gt,
        "observations": observations,
        "paths": {
            "rig_blend": root / files["rig_blend"]["rel_path"],
            "studio_rig_meta": root / files["studio_rig_meta"]["rel_path"],
            "constraints": root / files["constraints"]["rel_path"],
            "dinov2_head": root / files["dinov2_head"]["rel_path"],
            "exam_video": root / files["exam_video"]["rel_path"],
            "motion_spec": root / files["motion_spec"]["rel_path"],
            "exam_gt_gz": gt_gz_path,
            "observations_npz": obs_path,
            "inference_receipt": root / files["inference_receipt"]["rel_path"],
        },
    }
