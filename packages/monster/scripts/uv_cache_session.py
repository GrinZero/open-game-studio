"""Restore a Drive snapshot to a local uv cache; publish only after uv exits."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cache_entries(cache):
    for directory, dirs, files in os.walk(cache):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".tmp"))
        for name in sorted([*dirs, *files]):
            if name.startswith(".tmp") or name.endswith((".lock", ".partial")):
                continue
            if name in {"credentials.toml", "credentials.json"}:
                continue
            path = Path(directory) / name
            yield path


def fingerprint(cache):
    result = hashlib.sha256()
    for path in cache_entries(cache):
        st = path.lstat()
        row = [path.relative_to(cache).as_posix(), st.st_size if not path.is_dir() else 0,
               int(st.st_mtime) if not path.is_dir() else 0,
               os.path.relpath(path.resolve(), cache.resolve()) if path.is_symlink() else ""]
        result.update(json.dumps(row, ensure_ascii=True).encode())
    return result.hexdigest()


def records(store):
    for path in sorted(store.glob("*.json"), reverse=True):
        try:
            record = json.loads(path.read_text())
            name = record["archive"]
            if Path(name).name != name or not name.endswith(".tar.gz"):
                continue
            yield record
        except (ValueError, KeyError, OSError):
            continue


def restore(cache, store):
    cache.mkdir(parents=True, exist_ok=True)
    if any(cache_entries(cache)):
        return {"status": "local_cache_reused"}
    for record in records(store):
        archive = store / record["archive"]
        try:
            if archive.stat().st_size != record["bytes"] or digest(archive) != record["sha256"]:
                continue
            with tempfile.TemporaryDirectory(dir=cache.parent, prefix="uv-restore-") as tmp:
                stage = Path(tmp)
                with tarfile.open(archive, "r:gz") as tar:
                    if any(not m.name.startswith("uv/") and m.name != "uv" for m in tar.getmembers()):
                        raise ValueError("Unexpected snapshot root")
                    tar.extractall(stage, filter="data")
                restored = stage / "uv"
                if fingerprint(restored) != record["fingerprint"]:
                    continue
                cache.rmdir()  # cache is still empty; never replace an active cache.
                restored.rename(cache)
            return {"status": "restored", "archive": str(archive)}
        except (OSError, ValueError, tarfile.TarError):
            continue
    return {"status": "cache_miss"}


def save(cache, store):
    cache.mkdir(parents=True, exist_ok=True)
    before = fingerprint(cache)
    for record in records(store):
        if record.get("fingerprint") == before:
            archive = store / record["archive"]
            if archive.is_file() and archive.stat().st_size == record.get("bytes"):
                return {"status": "unchanged", "archive": str(archive)}
            break
    store.mkdir(parents=True, exist_ok=True)
    generation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    with tempfile.TemporaryDirectory(dir=cache.parent, prefix="uv-snapshot-") as tmp:
        archive = Path(tmp) / (generation + ".tar.gz")
        with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
            tar.add(cache, arcname="uv", recursive=False)
            for path in cache_entries(cache):
                name = "uv/" + path.relative_to(cache).as_posix()
                info = tar.gettarinfo(str(path), arcname=name)
                if info.issym():
                    target = path.resolve()
                    if not target.is_relative_to(cache.resolve()):
                        raise ValueError("Cache symlink escapes the local cache")
                    info.linkname = os.path.relpath(target, path.parent.resolve())
                if info.isfile():
                    with path.open("rb") as stream:
                        tar.addfile(info, stream)
                else:
                    tar.addfile(info)
        if fingerprint(cache) != before:
            raise RuntimeError("uv cache changed during snapshot; refusing to publish")
        record = {"schema_version": 1, "archive": archive.name, "bytes": archive.stat().st_size,
                  "sha256": digest(archive), "fingerprint": before}
        destination = store / archive.name
        shutil.copyfile(archive, destination)
        if destination.stat().st_size != record["bytes"] or digest(destination) != record["sha256"]:
            raise RuntimeError("Drive snapshot readback failed; previous snapshots retained")
        # Immutable generations avoid DriveFS atomic-rename/locking requirements and
        # concurrent VM writers overwriting one another's only good snapshot.
        (store / (generation + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    return {"status": "saved", **record, "directory": str(store)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("run", "restore", "save"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--drive-root", type=Path, required=True)
    parser.add_argument("--key", required=True)
    argv = sys.argv[1:]
    boundary = argv.index("--") if "--" in argv else len(argv)
    command = argv[boundary + 1:]
    args = parser.parse_args(argv[:boundary])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.key) or args.key in {".", ".."}:
        parser.error("Invalid cache key")
    cache, store = args.cache_dir.resolve(), args.drive_root.resolve() / args.key
    if cache.is_relative_to(args.drive_root.resolve().parent):
        parser.error("uv must run in a local cache, outside the persistent Drive cache tree")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with (cache.parent / ".monster-uv-cache-session.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.operation != "save":
            print(json.dumps(restore(cache, store)), flush=True)
        if args.operation == "restore":
            return 0
        if args.operation == "run":
            if not command:
                parser.error("run requires a command after --")
            env = {**os.environ, "UV_CACHE_DIR": str(cache), "MONSTER_UV_CACHE_SESSION": "1"}
            # No snapshot while uv is alive; exceptional interruption skips publication.
            result = subprocess.run(command, env=env)
        else:
            result = None
        print(json.dumps(save(cache, store)), flush=True)
        return result.returncode if result else 0


if __name__ == "__main__":
    raise SystemExit(main())
