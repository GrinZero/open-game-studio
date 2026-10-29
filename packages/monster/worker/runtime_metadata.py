"""Deployment metadata independent of the algorithm's FEATURE_VERSION."""
import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def runtime_metadata():
    package = json.loads((PACKAGE_ROOT / "package.json").read_text())
    protocol = json.loads((PACKAGE_ROOT / "contracts/protocol.json").read_text())
    manifest_path = PACKAGE_ROOT / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    blender = json.loads((PACKAGE_ROOT / "contracts/blender-runtime.json").read_text())
    return {
        "package_version": package["version"],
        "blender_target": blender["blender_version"],
        "blender_python": blender["python_version"],
        "protocol_version": protocol["protocol_version"],
        "orientation_contract": protocol["orientation_contract"],
        "release_id": manifest.get("release_id", "source-checkout"),
    }
