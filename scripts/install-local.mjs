import {
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { spawnSync } from "node:child_process";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { runtimeCopyFilter } from "./runtime-copy.mjs";

// The plugin cache contains only skills and configuration. Its runtime lives
// outside the cache so Codex can replace the cache without losing dependencies.
const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const base = path.join(
  os.homedir(),
  ".local/share/codex-integrations/game-studio-local",
);
const runtime = path.join(base, "runtime");
const plugin = path.join(base, "plugins/game-studio");
const studioHome = path.resolve(
  process.env.STUDIO_HOME || path.join(os.homedir(), ".game-studio"),
);
const port = Number(process.env.PORT || 4310);
const label = "local.codex.game-studio";
if (process.platform !== "darwin")
  throw new Error("This local installer currently supports macOS.");
if (Number(process.versions.node.split(".")[0]) < 24)
  throw new Error("Node 24+ is required.");
if (!Number.isInteger(port) || port < 1024 || port > 65535)
  throw new Error("Invalid PORT.");
function run(command, args, cwd = repo) {
  const result = spawnSync(command, args, { cwd, stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0)
    throw new Error(`${command} failed (${result.status}).`);
}
function json(file, value) {
  mkdirSync(path.dirname(file), { recursive: true });
  writeFileSync(file, JSON.stringify(value, null, 2) + "\n");
}
const xml = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
if (!existsSync(path.join(repo, "node_modules/tsx/dist/cli.mjs")))
  run("npm", ["ci"]);
run("npm", ["run", "build"]);
mkdirSync(base, { recursive: true });
const stagedRuntime = mkdtempSync(path.join(base, "runtime-next-"));
for (const entry of [
  "src",
  "packages",
  "assets",
  "dist",
  "node_modules",
  "package.json",
  "package-lock.json",
  ...readdirSync(repo).filter((file) => /^README(?:\.[\w-]+)?\.md$/.test(file)),
  "CONTRIBUTING.md",
  "LICENSE",
  "docs",
  "licenses",
  "AGENTS.md",
  "samples",
  "scripts",
]) {
  const destination = path.join(stagedRuntime, entry);
  mkdirSync(path.dirname(destination), { recursive: true });
  cpSync(path.join(repo, entry), destination, {
    recursive: true,
    verbatimSymlinks: true,
    filter: (source) => runtimeCopyFilter(entry, source, repo),
  });
}
// Finish the new copy before stopping this installer's service. Replacing the
// whole owned runtime also removes stale code/dependencies during upgrades.
const domain = `gui/${process.getuid()}`;
spawnSync("launchctl", ["bootout", `${domain}/${label}`], { stdio: "ignore" });
const previousRuntime = path.join(base, "runtime-previous");
rmSync(previousRuntime, { recursive: true, force: true });
if (existsSync(runtime)) renameSync(runtime, previousRuntime);
renameSync(stagedRuntime, runtime);
rmSync(previousRuntime, { recursive: true, force: true });
cpSync(path.join(repo, "skills"), path.join(plugin, "skills"), {
  recursive: true,
});
mkdirSync(path.join(plugin, "assets"), { recursive: true });
cpSync(
  path.join(repo, "assets/icon.svg"),
  path.join(plugin, "assets/icon.svg"),
);
const portable = JSON.parse(
  readFileSync(path.join(repo, "plugin.json"), "utf8"),
);
json(path.join(plugin, ".codex-plugin/plugin.json"), {
  name: portable.name,
  version: portable.version,
  description: portable.description,
  skills: "./skills/",
  mcpServers: "./.mcp.json",
  interface: {
    ...portable.extensions["com.openai"].interface,
    composerIcon: "./assets/icon.svg",
    logo: "./assets/icon.svg",
    longDescription:
      "本地 Alpha：Codex 侧边栏与聊天内嵌工作台，共用项目、资源版本、自由画布与人工角色制作流程。自动模型生成与 Blender/Godot worker 尚未接入。",
    category: "Design",
    capabilities: ["Interactive", "Read", "Write"],
    defaultPrompt: [
      "打开 Open Game Studio 工作台，让我体验示例项目。",
      "列出 Open Game Studio 项目和画布。",
    ],
  },
});
const args = [
  path.join(runtime, "node_modules/tsx/dist/cli.mjs"),
  path.join(runtime, "src/server/mcp.ts"),
];
json(path.join(plugin, ".mcp.json"), {
  mcpServers: {
    "game-studio": {
      command: process.execPath,
      args,
      cwd: runtime,
      env: { STUDIO_HOME: studioHome, PORT: String(port) },
      startup_timeout_sec: 30,
    },
  },
});
json(path.join(base, ".agents/plugins/marketplace.json"), {
  name: "game-studio-local",
  interface: { displayName: "Open Game Studio 本地体验" },
  plugins: [
    {
      name: "game-studio",
      source: { source: "local", path: "./plugins/game-studio" },
      policy: { installation: "AVAILABLE", authentication: "ON_USE" },
      category: "Design",
    },
  ],
});
run("codex", ["plugin", "marketplace", "add", base, "--json"]);
run("codex", ["plugin", "add", "game-studio@game-studio-local", "--json"]);
mkdirSync(studioHome, { recursive: true });
const agent = path.join(os.homedir(), "Library/LaunchAgents", `${label}.plist`);
mkdirSync(path.dirname(agent), { recursive: true });
writeFileSync(
  agent,
  `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>${label}</string>
<key>ProgramArguments</key><array><string>${xml(process.execPath)}</string><string>${xml(args[0])}</string><string>${xml(path.join(runtime, "src/server/http.ts"))}</string><string>--production</string></array>
<key>WorkingDirectory</key><string>${xml(runtime)}</string>
<key>EnvironmentVariables</key><dict><key>STUDIO_HOME</key><string>${xml(studioHome)}</string><key>PORT</key><string>${port}</string></dict>
<key>RunAtLoad</key><true/><key>KeepAlive</key><true/><key>ThrottleInterval</key><integer>10</integer>
<key>StandardOutPath</key><string>${xml(path.join(studioHome, "http.log"))}</string>
<key>StandardErrorPath</key><string>${xml(path.join(studioHome, "http.error.log"))}</string>
</dict></plist>
`,
);
run("launchctl", ["bootstrap", domain, agent]);
for (let attempt = 0; attempt < 30; attempt++) {
  try {
    const response = await fetch(`http://127.0.0.1:${port}/api/health`);
    if (response.ok && (await response.json()).schemaVersion === 1) {
      console.log(
        `Open Game Studio installed and running: http://127.0.0.1:${port}`,
      );
      console.log(
        `Runtime: ${runtime}\nService: ${agent}\nRegistry: ${studioHome}`,
      );
      process.exit(0);
    }
  } catch {}
  await new Promise((resolve) => setTimeout(resolve, 500));
}
throw new Error(
  `Workbench did not become ready; inspect ${studioHome}/http.error.log.`,
);
