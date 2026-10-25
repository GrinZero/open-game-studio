import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const relVenvBin = process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python";
const venvPython = path.join(root, relVenvBin);
const fallbackVenvPython = "/Users/bugyaluwang/project/codex-game-studio/packages/monster/" + relVenvBin;

function resolvePython(scriptArg = "") {
  if (process.env.MONSTER_PYTHON) return process.env.MONSTER_PYTHON;
  if (existsSync(venvPython)) return venvPython;
  if (existsSync(fallbackVenvPython)) return fallbackVenvPython;
  const needsClientDeps =
    scriptArg.includes("monster_client.py") &&
    process.env.MONSTER_SKIP_AUTO_VENV !== "1";
  if (!needsClientDeps) return "python3";
  const probe = spawnSync("python3", ["-c", "import gradio_client"], {
    cwd: root,
    stdio: "ignore",
  });
  if (probe.status === 0) return "python3";
  console.error("[Monster] 首次运行客户端，正在自动准备 packages/monster/.venv ...");
  const create = spawnSync("python3", ["-m", "venv", path.join(root, ".venv")], {
    cwd: root,
    stdio: "inherit",
  });
  if (create.status !== 0) return "python3";
  const install = spawnSync(
    venvPython,
    ["-m", "pip", "install", "-q", "-r", "integration/requirements.txt"],
    { cwd: root, stdio: "inherit" },
  );
  if (install.status !== 0) {
    throw new Error("自动安装 Monster 客户端依赖失败，请检查网络连接或 Python 环境。");
  }
  return venvPython;
}

const args = process.argv.slice(2);
const result = spawnSync(resolvePython(args[0]), args, {
  cwd: root,
  stdio: "inherit",
});
if (result.error) throw result.error;
process.exit(result.status ?? 1);
