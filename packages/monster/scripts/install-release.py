"""Unpack a Monster deployment into an isolated version directory."""
from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def verify_package(package):
    package = Path(package).resolve()
    manifest = json.loads((package / "release-manifest.json").read_text())
    package_json = json.loads((package / "package.json").read_text())
    protocol = json.loads((package / "contracts/protocol.json").read_text())
    if (manifest.get("release_id") != f"monster-{package_json['version']}"
            or manifest.get("package_version") != package_json["version"]
            or manifest.get("protocol_version") != protocol["protocol_version"]):
        raise ValueError("Release metadata does not match its contents")
    return manifest


def install_release(archive, releases_root):
    archive = Path(archive)
    if not archive.is_file():
        raise ValueError("Deployment archive not found")
    releases_root = Path(releases_root).resolve()
    releases_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".install-", dir=releases_root))
    try:
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            if len({m.name for m in members}) != len(members):
                raise ValueError("Duplicate archive entries")
            if sum(m.size for m in members) > 32 * 1024 * 1024:
                raise ValueError("Deployment archive exceeds 32 MiB")
            for member in members:
                name = PurePosixPath(member.name)
                if (not member.isfile() or name.is_absolute() or ".." in name.parts
                        or not name.parts or name.parts[0] != "monster"):
                    raise ValueError("Invalid archive entry")
            tar.extractall(staging, members=members, filter="data")
        manifest = verify_package(staging / "monster")
        destination = releases_root / manifest["release_id"]
        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination / "monster"
