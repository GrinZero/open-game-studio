#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
BOOTSTRAP_PYTHON="${MONSTER_BOOTSTRAP_PYTHON:-python3}"
BPY_ROOT="${MONSTER_BPY_ROOT:-/content/monster-bpy}"
read -r BPY_VERSION BPY_PYTHON < <("$BOOTSTRAP_PYTHON" - "$PACKAGE_DIR/contracts/blender-runtime.json" <<'PY'
import json,sys
spec=json.load(open(sys.argv[1]))
print(spec['blender_version'],spec['python_version'])
PY
)
command -v uv >/dev/null || "$BOOTSTRAP_PYTHON" -m pip install 'uv==0.9.5'
export UV_CACHE_DIR="${MONSTER_BPY_CACHE_DIR:-$BPY_ROOT/cache/uv}"
# A version-specific environment keeps the preceding environment recoverable.
ENV_DIR="$BPY_ROOT/envs/$BPY_VERSION"
mkdir -p "$BPY_ROOT/envs"
if [ ! -x "$ENV_DIR/bin/python" ]; then uv venv --python "$BPY_PYTHON" --seed "$ENV_DIR"; fi
uv pip install --python "$ENV_DIR/bin/python" -r "$PACKAGE_DIR/worker/blender-requirements.txt"
"$ENV_DIR/bin/python" - "$BPY_VERSION" "$BPY_PYTHON" <<'PY'
import sys,bpy,numpy,scipy,PIL,skimage,cv2
assert bpy.app.version_string.split()[0] == sys.argv[1], bpy.app.version_string
assert '.'.join(map(str,sys.version_info[:2])) == sys.argv[2], sys.version
print('Blender renderer ready:',bpy.app.version_string,'Python',sys.version.split()[0],'NumPy',numpy.__version__)
PY
# Replace only our launcher symlink, never a pre-existing user directory.
if [ -e "$BPY_ROOT/venv" ] && [ ! -L "$BPY_ROOT/venv" ]; then
  echo "Refusing to overwrite non-symlink $BPY_ROOT/venv" >&2
  exit 1
fi
ln -sfn "$ENV_DIR" "$BPY_ROOT/venv"
uv pip freeze --python "$ENV_DIR/bin/python" > "$BPY_ROOT/installed-packages.txt"
