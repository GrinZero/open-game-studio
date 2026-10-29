"""Load the measured mesh/weights used by the approved rig's numerical checks."""

import hashlib
from pathlib import Path

import numpy as np


def load_probe_skinning(profile, blend_path, bones=None):
    artifact = (profile or {}).get("rig_skinning")
    if artifact is None:
        return None
    path = Path(artifact["path"])
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if digest(blend_path) != artifact["blend_sha256"]:
        raise ValueError("Joint-limit skinning belongs to another approved blend")
    if digest(path) != artifact["sha256"]:
        raise ValueError("Joint-limit skinning artifact checksum differs")
    with np.load(path, allow_pickle=False) as archive:
        skin = {name: archive[name] for name in archive.files}
    skin["bone_names"] = skin["bone_names"].tolist()
    if bones is not None:
        if set(skin["bone_names"]) != set(bones):
            raise ValueError("Joint-limit skinning bone names differ from the approved rig")
        expected = np.array([bones[n]["rest_head_world"] for n in skin["bone_names"]])
        if not np.allclose(skin["rest_heads"], expected, atol=2e-6, rtol=0):
            raise ValueError("Joint-limit skinning joint origins differ from the approved rig")
    return skin
