"""Create a Colab deployment bundle, without caches, credentials or outputs."""
from __future__ import annotations

import argparse
import gzip
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# An explicit source allowlist keeps runtime credentials / model caches out of releases.
SOURCE_DIRS = {
    "worker": {".py", ".txt"},
    "integration": {".py", ".txt"},
    "contracts": {".json"},
    "scripts": {".py", ".sh", ".mjs"},
    "notebooks": {".ipynb"},
    "skills": {".md", ".yaml", ".txt", ".py"},
    "licenses": {".txt"},
}


def source_files(root=ROOT):
    files = [root / name for name in ("package.json", "README.md", "LICENSE", "LICENSING.md", "THIRD_PARTY.md")]
    for directory, suffixes in SOURCE_DIRS.items():
        files.extend(p for p in (root / directory).rglob("*")
                     if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts)
    if any(p.is_symlink() or not p.is_relative_to(root) for p in files):
        raise ValueError("Release sources must be regular files inside the package")
    return sorted(files)


def build_release(output, root=ROOT):
    contents = {p.relative_to(root).as_posix(): p.read_bytes() for p in source_files(root)}
    file_count = len(contents)
    version = json.loads(contents["package.json"])["version"]
    release_id = f"monster-{version}"
    manifest = {
        "release_id": release_id,
        "package_version": version,
        "protocol_version": json.loads(contents["contracts/protocol.json"])["protocol_version"],
    }
    contents["release-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{release_id}.tar.gz"
    with archive.open("wb") as stream, gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode="w") as tar:
            for name, data in sorted(contents.items()):
                info = tarfile.TarInfo("monster/" + name)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                tar.addfile(info, io.BytesIO(data))
    notebook = json.loads(contents["notebooks/colab.ipynb"])
    for cell in notebook["cells"]:
        cell["source"] = [line.replace('RELEASE_ARCHIVE = ""',
            f'RELEASE_ARCHIVE = "/content/drive/MyDrive/MonsterStudio/releases/{archive.name}"')
            for line in cell["source"]]
    notebook_path = output / f"{release_id}.ipynb"
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")
    return {"release_id": release_id, "archive": str(archive),
            "notebook": str(notebook_path), "file_count": file_count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "dist"))
    print(json.dumps(build_release(parser.parse_args().output), ensure_ascii=False, indent=2))
