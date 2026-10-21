"""Google Drive fixture bundle packager, downloader, and SHA-256 verifier for Monster Studio Evals."""
from __future__ import annotations

import gzip
import hashlib
import http.cookiejar
import io
import json
import os
import re
import shutil
import tarfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

EVALS_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_ROOT = EVALS_ROOT / "benchmarks"
DEFAULT_BENCHMARK_ID = "fantasy_creature_2k_v1"
MONSTER_ROOT = EVALS_ROOT.parent
REPO_ROOT = MONSTER_ROOT.parents[1]
VAR_EVALS_ROOT = Path(os.environ.get("MONSTER_EVAL_VAR_ROOT", str(REPO_ROOT / "var" / "monster" / "evals")))

STARMAKER_DEFAULT_ROOT = Path("/Users/bugyaluwang/project/star-maker/.asset-workbench/local/runs")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_drive_file_id(url_or_id: str) -> str:
    """Extracts a Google Drive file ID from a share URL or raw ID string."""
    raw = url_or_id.strip()
    if not raw:
        raise ValueError("Empty Google Drive share URL or file ID")
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]{10,})",
        r"[?&]id=([a-zA-Z0-9_-]{10,})",
        r"/d/([a-zA-Z0-9_-]{10,})",
    ]
    for pat in patterns:
        m = re.search(pat, raw)
        if m:
            return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", raw):
        return raw
    raise ValueError(f"Could not parse Google Drive file ID from: {url_or_id!r}")


def download_google_drive_file(url_or_id: str, destination: Path) -> Path:
    """Downloads a publicly shared Google Drive file (including large files requiring virus-scan confirmation)."""
    file_id = extract_drive_file_id(url_or_id)
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = destination.with_suffix(destination.suffix + ".part")

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    headers = {"User-Agent": "MonsterStudioEval/1.0"}

    # Try direct usercontent endpoint first (works for large shared binaries with confirm=t)
    direct_url = (
        f"https://drive.usercontent.google.com/download?id={urllib.parse.quote(file_id)}"
        "&export=download&confirm=t"
    )
    req = urllib.request.Request(direct_url, headers=headers)
    with opener.open(req, timeout=120) as resp:
        content_type = resp.headers.get("Content-Type", "")
        first_chunk = resp.read(65536)
        if b"<!DOCTYPE html" in first_chunk[:512] or "text/html" in content_type.lower():
            html_bytes = first_chunk + resp.read()
            html_text = html_bytes.decode("utf-8", errors="ignore")
            m_uuid = re.search(r'name="uuid"\s+value="([^"]+)"', html_text)
            m_confirm = re.search(r'name="confirm"\s+value="([^"]+)"', html_text)
            confirm_val = m_confirm.group(1) if m_confirm else "t"
            uuid_param = f"&uuid={urllib.parse.quote(m_uuid.group(1))}" if m_uuid else ""
            confirm_url = (
                f"https://drive.usercontent.google.com/download?id={urllib.parse.quote(file_id)}"
                f"&export=download&confirm={urllib.parse.quote(confirm_val)}{uuid_param}"
            )
            req2 = urllib.request.Request(confirm_url, headers=headers)
            with opener.open(req2, timeout=300) as resp2, tmp_dest.open("wb") as out:
                shutil.copyfileobj(resp2, out, length=1024 * 1024)
        else:
            with tmp_dest.open("wb") as out:
                out.write(first_chunk)
                shutil.copyfileobj(resp, out, length=1024 * 1024)

    tmp_dest.replace(destination)
    return destination


def safe_extract_tar_gz(archive_path: Path, target_dir: Path) -> list[str]:
    """Extracts a `.tar.gz` archive into `target_dir`, blocking path traversal and symlinks."""
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Unsupported link or device in archive: {member.name}")
            dest = (target_dir / member.name).resolve()
            if not dest.is_relative_to(target_dir):
                raise ValueError(f"Archive path traversal blocked: {member.name}")
        for member in tar.getmembers():
            if not member.isfile():
                continue
            dest = (target_dir / member.name).resolve()
            dest.parent.mkdir(parents=True, exist_ok=True)
            src_stream = tar.extractfile(member)
            if src_stream is None:
                continue
            with src_stream, dest.open("wb") as out_f:
                shutil.copyfileobj(src_stream, out_f, length=1024 * 1024)
            extracted.append(member.name)
    return sorted(extracted)


def build_deterministic_tar_gz(source_dir: Path, archive_path: Path) -> dict[str, Any]:
    """Packages all regular files under `source_dir` into a deterministic `.tar.gz` archive."""
    source_dir = source_dir.resolve()
    archive_path = archive_path.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_archive = archive_path.with_suffix(archive_path.suffix + ".tmp")

    rel_files = sorted(
        p.relative_to(source_dir).as_posix()
        for p in source_dir.rglob("*")
        if p.is_file() and not p.name.startswith(".") and "__pycache__" not in p.parts
    )
    with tmp_archive.open("wb") as raw_out, gzip.GzipFile(filename="", fileobj=raw_out, mode="wb", mtime=0) as gz_out:
        with tarfile.open(fileobj=gz_out, mode="w") as tar:
            for rel_posix in rel_files:
                full_path = source_dir / rel_posix
                stat = full_path.stat()
                info = tarfile.TarInfo(name=rel_posix)
                info.size = stat.st_size
                info.mtime = 0
                info.mode = 0o644
                with full_path.open("rb") as f_in:
                    tar.addfile(info, f_in)

    tmp_archive.replace(archive_path)
    return {
        "archive_path": str(archive_path),
        "archive_name": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "file_count": len(rel_files),
    }


def pack_from_starmaker_2k(
    benchmark_id: str = "fantasy_creature_2k_v1",
    starmaker_runs_root: Path = STARMAKER_DEFAULT_ROOT,
    share_url: str = "",
) -> dict[str, Any]:
    """Collects the validated 2K exam & model fixtures from `star-maker` into `var/monster/evals/`,
    creates `var/monster/evals/archives/fantasy_creature_2k_v1.tar.gz` for Google Drive sharing,
    and writes the exact SHA-256 lock manifest to `packages/monster/evals/benchmarks/<benchmark_id>/manifest.json`.
    """
    r2k = starmaker_runs_root / "fantasy_creature_train_2k_v0001_20261021"
    rv9 = starmaker_runs_root / "fantasy_creature_train_v9_20261021"
    o3 = r2k / "optimized_v0003"

    src_blend = rv9 / "input/design/models/characters/fantasy_creature/work/source/fantasy_creature_rig.blend"
    src_meta = r2k / "studio_rig_meta.json"
    src_constraints = o3 / "constraints.json"
    src_ckpt = r2k / "dinov2_head.pt"
    src_video = r2k / "exam/exam_input_2k_v0001.mp4"
    src_motion_spec = o3 / "scoring_reference/motion_spec.json"
    src_gt_json = o3 / "scoring_reference/exam_gt.json"
    src_obs = o3 / "observations.npz"
    src_receipt = o3 / "inference_receipt.json"

    for required in (src_blend, src_meta, src_constraints, src_ckpt, src_video, src_motion_spec, src_gt_json, src_obs, src_receipt):
        if not required.is_file():
            raise FileNotFoundError(f"Required 2K source artifact missing: {required}")

    fixture_dir = VAR_EVALS_ROOT / "fixtures" / benchmark_id
    for sub in ("source", "checkpoints", "exam", "cached_observations"):
        (fixture_dir / sub).mkdir(parents=True, exist_ok=True)

    copy_map = {
        "rig_blend": (src_blend, "source/rig.blend"),
        "studio_rig_meta": (src_meta, "source/studio_rig_meta.json"),
        "constraints": (src_constraints, "source/constraints.json"),
        "dinov2_head": (src_ckpt, "checkpoints/dinov2_head.pt"),
        "exam_video": (src_video, "exam/exam_input_2k_v0001.mp4"),
        "motion_spec": (src_motion_spec, "exam/motion_spec.json"),
        "observations_npz": (src_obs, "cached_observations/observations.npz"),
        "inference_receipt": (src_receipt, "cached_observations/inference_receipt.json"),
    }

    files_manifest: dict[str, dict[str, Any]] = {}
    for key, (src_path, rel_target) in copy_map.items():
        dst_path = fixture_dir / rel_target
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if not dst_path.exists() or dst_path.stat().st_size != src_path.stat().st_size:
            shutil.copy2(src_path, dst_path)
        files_manifest[key] = {
            "rel_path": rel_target,
            "sha256": sha256_file(dst_path),
            "bytes": dst_path.stat().st_size,
        }

    # Compress exam_gt.json deterministically to exam/exam_gt.json.gz
    gt_raw_bytes = src_gt_json.read_bytes()
    gt_uncompressed_sha = sha256_bytes(gt_raw_bytes)
    gt_gz_rel = "exam/exam_gt.json.gz"
    gt_gz_path = fixture_dir / gt_gz_rel
    with gt_gz_path.open("wb") as raw_f, gzip.GzipFile(filename="", fileobj=raw_f, mode="wb", mtime=0) as gz_f:
        gz_f.write(gt_raw_bytes)
    files_manifest["exam_gt"] = {
        "rel_path": gt_gz_rel,
        "sha256": sha256_file(gt_gz_path),
        "bytes": gt_gz_path.stat().st_size,
        "uncompressed_sha256": gt_uncompressed_sha,
    }

    archive_path = VAR_EVALS_ROOT / "archives" / f"{benchmark_id}.tar.gz"
    archive_info = build_deterministic_tar_gz(fixture_dir, archive_path)

    manifest_path = EVALS_ROOT / "benchmarks" / benchmark_id / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing_share_url = share_url
    existing_file_id = ""
    if manifest_path.exists() and not existing_share_url:
        try:
            old_m = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing_share_url = old_m.get("drive_bundle", {}).get("share_url", "")
            existing_file_id = old_m.get("drive_bundle", {}).get("file_id", "")
        except Exception:
            pass
    if existing_share_url and not existing_file_id:
        try:
            existing_file_id = extract_drive_file_id(existing_share_url)
        except ValueError:
            existing_file_id = ""

    manifest_data = {
        "schema": "monster-eval-benchmark-manifest-v1",
        "benchmark_id": benchmark_id,
        "monster_id": "fantasy_creature",
        "description": "Native 2K (2560x1440, 24 FPS, 288 frames, 77 bones) dual-orthographic video-to-animation reconstruction benchmark for fantasy_creature v0009.",
        "resolution_mode": "2k",
        "composite_resolution": [2560, 1440],
        "panel_resolution": [1280, 1440],
        "padded_panel_resolution": [1288, 1442],
        "dino_patch_grid": [92, 103],
        "fps": 24.0,
        "frame_count": 288,
        "bone_count": 77,
        "drive_bundle": {
            "archive_name": archive_info["archive_name"],
            "archive_sha256": archive_info["archive_sha256"],
            "archive_bytes": archive_info["archive_bytes"],
            "share_url": existing_share_url,
            "file_id": existing_file_id,
        },
        "files": files_manifest,
        "thresholds": {
            "epipolar_max_v_diff_px": 33.75,
            "epipolar_min_corr": 0.82,
            "direction_mean_deg": 3.5,
            "direction_max_bone_deg": 8.0,
            "nonleaf_so3_mean_deg": 3.5,
            "skin_roundtrip_max_units": 0.0001,
        },
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "benchmark_id": benchmark_id,
        "fixture_dir": str(fixture_dir),
        "archive_path": archive_info["archive_path"],
        "archive_bytes": archive_info["archive_bytes"],
        "archive_sha256": archive_info["archive_sha256"],
        "manifest_path": str(manifest_path),
    }


def pull_fixture_bundle(
    benchmark_id: str = "fantasy_creature_2k_v1",
    url_or_id: str | None = None,
    from_archive: str | Path | None = None,
) -> dict[str, Any]:
    """Downloads or imports `<benchmark_id>.tar.gz` into `var/monster/evals/fixtures/<benchmark_id>/`
    and verifies both the bundle SHA-256 and every extracted file's SHA-256 against `manifest.json`.
    """
    from .loader import load_manifest, verify_fixture_directory

    manifest_path = EVALS_ROOT / "benchmarks" / benchmark_id / "manifest.json"
    manifest = load_manifest(benchmark_id)
    bundle_meta = manifest["drive_bundle"]
    archive_path = VAR_EVALS_ROOT / "archives" / bundle_meta["archive_name"]
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    if from_archive is not None:
        src_archive = Path(from_archive).expanduser().resolve()
        if not src_archive.is_file():
            raise FileNotFoundError(f"Archive not found: {src_archive}")
        if src_archive != archive_path.resolve():
            shutil.copy2(src_archive, archive_path)
    elif not archive_path.is_file() or sha256_file(archive_path) != bundle_meta["archive_sha256"]:
        target_url = (url_or_id or bundle_meta.get("share_url") or bundle_meta.get("file_id") or "").strip()
        if not target_url:
            raise ValueError(
                f"No Google Drive share_url configured in {manifest_path} and no --url / --from-archive provided."
            )
        download_google_drive_file(target_url, archive_path)

    actual_archive_sha = sha256_file(archive_path)
    expected_archive_sha = bundle_meta.get("archive_sha256", "")
    if expected_archive_sha and actual_archive_sha != expected_archive_sha:
        raise ValueError(
            f"Archive SHA-256 mismatch for {archive_path}: expected {expected_archive_sha}, got {actual_archive_sha}"
        )

    fixture_dir = VAR_EVALS_ROOT / "fixtures" / benchmark_id
    safe_extract_tar_gz(archive_path, fixture_dir)

    if url_or_id:
        bundle_meta["share_url"] = url_or_id.strip()
        try:
            bundle_meta["file_id"] = extract_drive_file_id(url_or_id)
        except ValueError:
            pass
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    verification = verify_fixture_directory(benchmark_id, fixture_dir=fixture_dir)
    if not verification["valid"]:
        raise RuntimeError(f"Extracted fixture verification failed: {json.dumps(verification, indent=2)}")
    return verification
