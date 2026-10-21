// Update the owned runtime without reinstalling the plugin or touching project data.
import {
  cpSync,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import os from "node:os";
import { runtimeCopyFilter } from "./runtime-copy.mjs";
import { exportPackageSkills } from "./export-package-skills.mjs";
const repo = process.cwd();
const base = path.join(
  os.homedir(),
  ".local/share/codex-integrations/game-studio-local",
);
const runtime = path.join(base, "runtime");
const configPath = path.join(base, "plugins/game-studio/.mcp.json");
const config = JSON.parse(readFileSync(configPath, "utf8")).mcpServers[
  "game-studio"
];
if (
  config.cwd !== runtime ||
  !config.args.includes(path.join(runtime, "src/server/mcp.ts"))
)
  throw new Error("Runtime configuration differs; inspect before updating.");
if (!existsSync(path.join(repo, "dist/mcp/index.html")))
  throw new Error("Build first.");
exportPackageSkills(repo);
const backup = path.join(base, "backups", `resources-${Date.now()}`);
mkdirSync(backup, { recursive: true });
for (const entry of [
  "src",
  "packages",
  "dist",
  "package.json",
  "package-lock.json",
  ...readdirSync(repo).filter((file) => /^README(?:\.[\w-]+)?\.md$/.test(file)),
  "CONTRIBUTING.md",
  "LICENSE",
  "docs",
  "licenses",
  "scripts",
  "skills",
  "AGENTS.md",
]) {
  if (existsSync(path.join(runtime, entry)))
    cpSync(path.join(runtime, entry), path.join(backup, entry), {
      recursive: true,
    });
  cpSync(path.join(repo, entry), path.join(runtime, entry), {
    recursive: true,
    filter: (source) => runtimeCopyFilter(entry, source, repo),
  });
}
const sync = spawnSync(
  "/usr/bin/rsync",
  ["-a", path.join(repo, "node_modules/"), path.join(runtime, "node_modules/")],
  { stdio: "inherit" },
);
if (sync.status !== 0) throw new Error("Dependency sync failed.");
const cache = path.join(
  os.homedir(),
  ".codex/plugins/cache/game-studio-local/game-studio",
  JSON.parse(readFileSync(path.join(repo, "package.json"), "utf8")).version,
);
for (const plugin of [path.join(base, "plugins/game-studio"), cache]) {
  if (!existsSync(plugin)) continue;
  const installed = JSON.parse(
    readFileSync(path.join(plugin, ".mcp.json"), "utf8"),
  ).mcpServers["game-studio"];
  if (installed.cwd !== runtime)
    throw new Error(`Plugin runtime differs: ${plugin}`);
  cpSync(path.join(repo, "skills"), path.join(plugin, "skills"), {
    recursive: true,
  });
}
const stamp = {
  build: readFileSync(path.join(repo, "src/shared/version.ts"), "utf8").match(
    /studioVersionMarker = "([^"]+)"/,
  )[1],
  runtime,
  backup,
  syncedAt: new Date().toISOString(),
};
writeFileSync(
  path.join(runtime, "runtime-update.json"),
  JSON.stringify(stamp, null, 2),
);
console.log(JSON.stringify(stamp));
