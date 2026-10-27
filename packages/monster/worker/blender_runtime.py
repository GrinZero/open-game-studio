"""Resolve the dedicated Blender Python without mixing SkinTokens' Python ABI."""
import json
import os
import sys
from pathlib import Path

CONTRACT = json.loads((Path(__file__).resolve().parents[1] / 'contracts/blender-runtime.json').read_text())
BPY_VERSION = CONTRACT['blender_version']
PYTHON_VERSION = CONTRACT['python_version']


def bpy_python_candidates():
    root = Path(os.environ.get('MONSTER_BPY_ROOT', CONTRACT['colab_root']))
    local = Path.home() / '.local/share/codex-integrations/game-studio-local' / f'blender-python-{BPY_VERSION}'
    return [os.environ.get('MONSTER_BPY_PYTHON'), str(root / 'venv/bin/python'),
            str(local / 'bin/python'), sys.executable]


def resolve_bpy_python():
    explicit = os.environ.get('MONSTER_BPY_PYTHON')
    if explicit and not Path(explicit).is_file():
        raise FileNotFoundError(f'MONSTER_BPY_PYTHON does not exist: {explicit}')
    for candidate in bpy_python_candidates():
        if candidate and Path(candidate).is_file():
            # On Colab the system interpreter belongs to training, never silently
            # route rendering back to that interpreter or the SkinTokens venv.
            if candidate == sys.executable and Path('/content').is_dir():
                break
            return candidate
    raise RuntimeError(f'Blender {BPY_VERSION} rendering environment missing; run Cell 2 or set MONSTER_BPY_PYTHON')
