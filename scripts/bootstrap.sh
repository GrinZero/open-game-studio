#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${GAME_STUDIO_REPO_URL:-https://github.com/GrinZero/codex-game-studio.git}"
REPO_REF="${GAME_STUDIO_REPO_REF:-main}"
BASE_DIR="${HOME}/.local/share/codex-integrations/game-studio-local"
SRC_DIR="${GAME_STUDIO_SOURCE_DIR:-${BASE_DIR}/source}"

for cmd in git node npm codex; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "缺少必要命令：$cmd（需要 macOS、Node.js 24+、npm 与 Codex CLI）" >&2
    exit 1
  fi
done

mkdir -p "$BASE_DIR"
if [ -d "$SRC_DIR/.git" ]; then
  git -C "$SRC_DIR" fetch --depth 1 origin "$REPO_REF"
  git -C "$SRC_DIR" checkout -f FETCH_HEAD
else
  rm -rf "$SRC_DIR"
  git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$SRC_DIR"
fi

cd "$SRC_DIR"
npm ci
npm run install:local
