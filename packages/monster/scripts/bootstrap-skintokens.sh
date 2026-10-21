#!/usr/bin/env bash
set -euo pipefail
ROOT="${SKINTOKENS_ROOT:-/content/skintokens-service}"
CACHE="${SKINTOKENS_CACHE:-$ROOT/cache}"
UPSTREAM=273b691d35989d71cd17ff2895fdc735097b92d1
FLASH=965c9212e6be33c821ee9f2b461564c5c91a2503
mkdir -p "$ROOT"
mkdir -p "$CACHE/wheels"
export UV_CACHE_DIR="$CACHE/uv"
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
if [[ "$GPU_NAME" != *"6000 Blackwell"* ]]; then
  echo '请选择 G4 GPU 后重新启动此环境。'
  exit 1
fi
command -v uv >/dev/null || python -m pip install 'uv==0.9.5'
if [ ! -d upstream/.git ]; then
  git clone https://github.com/VAST-AI-Research/SkinTokens.git upstream
fi
git -C upstream checkout "$UPSTREAM"
if [ ! -x "$ROOT/venv/bin/python" ]; then
  uv venv --python 3.11 --seed "$ROOT/venv"
fi
PY="$ROOT/venv/bin/python"
export CUDA_HOME=/usr/local/cuda
export PATH="$ROOT/venv/bin:$CUDA_HOME/bin:$PATH"
export MAX_JOBS=8 NVCC_THREADS=2 FLASH_ATTN_CUDA_ARCHS=120 TORCH_CUDA_ARCH_LIST=12.0
export FLASH_ATTENTION_FORCE_BUILD=TRUE
uv pip install --python "$PY" torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$PY" -r upstream/requirements.txt \
  transformers==4.57.6 diffusers==0.35.2 gradio==5.49.1 \
  numpy==1.26.4 scipy==1.15.3 bpy==4.2.0 lightning==2.5.5 ninja packaging psutil wheel setuptools
if [ ! -f flash-ready ]; then
  WHEEL_DIR="$CACHE/wheels/py311-torch291-cu130-$FLASH"
  mkdir -p "$WHEEL_DIR"
  if ! compgen -G "$WHEEL_DIR/flash_attn*.whl" > /dev/null; then
    if [ ! -d flash-src/.git ]; then
      git clone --filter=blob:none https://github.com/Dao-AILab/flash-attention.git flash-src
    fi
    git -C flash-src checkout "$FLASH"
    git -C flash-src submodule update --init --recursive
    "$PY" -m pip wheel --no-deps --no-build-isolation "$ROOT/flash-src" --wheel-dir "$WHEEL_DIR"
  fi
  uv pip install --python "$PY" "$WHEEL_DIR"/flash_attn*.whl
  "$PY" - <<'PY'
import torch
from flash_attn import flash_attn_func
from torch.nn.functional import scaled_dot_product_attention
assert torch.cuda.get_device_capability() == (12, 0), 'This environment targets G4 (SM120)'
q = torch.randn(1, 64, 4, 64, device='cuda', dtype=torch.bfloat16)
out = flash_attn_func(q, q, q)
ref = scaled_dot_product_attention(q.transpose(1,2), q.transpose(1,2), q.transpose(1,2)).transpose(1,2)
torch.testing.assert_close(out, ref, atol=.03, rtol=.03)
assert torch.isfinite(out).all()
print('G4 BF16 FlashAttention verified:', torch.cuda.get_device_name())
PY
  touch flash-ready
fi
# The Blender helper deserializes local tensors; it must only listen on loopback.
"$PY" - <<'PY'
from pathlib import Path
p = Path('upstream/src/server/bpy_server.py')
s = p.read_text()
s = s.replace("host='0.0.0.0'", "host='127.0.0.1'")
p.write_text(s)
# Upstream clears indices 0..989 even on smaller parts (e.g. hair).
# bpy 4.2 can segfault on out-of-range vertex indices.
p = Path('upstream/src/rig_package/parser/bpy.py')
s = p.read_text()
old = 'ob.vertex_groups[name].remove(range(990))'
new = 'ob.vertex_groups[name].remove(range(len(ob.data.vertices)))'
assert old in s or new in s, 'Upstream vertex-group cleanup changed; inspect before patching'
p.write_text(s.replace(old, new))
PY
cd upstream
if [ "$CACHE" != "$ROOT/cache" ]; then
  for name in experiments models; do
    mkdir -p "$CACHE/$name"
    if [ ! -e "$name" ]; then ln -s "$CACHE/$name" "$name"; fi
  done
fi
"$PY" download.py --model
cd "$ROOT"
uv pip freeze --python "$PY" > installed-packages.txt
"$PY" - <<'PY'
import json, torch, platform, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path('upstream').resolve()))
import demo
Path('ready.json').write_text(json.dumps({
    'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
    'vram_gib': torch.cuda.get_device_properties(0).total_memory / 2**30,
    'python': platform.python_version(), 'torch': torch.__version__,
    'upstream': subprocess.check_output(['git', '-C', 'upstream', 'rev-parse', 'HEAD'], text=True).strip(),
}, indent=2))
print(Path('ready.json').read_text())
PY
