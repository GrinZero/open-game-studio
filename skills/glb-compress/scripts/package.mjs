#!/usr/bin/env node
import { mkdir } from "node:fs/promises";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
const root = fileURLToPath(new URL("..", import.meta.url));
const output = path.resolve(
  process.argv[2] ?? path.join(root, "dist/glb-compress-skill.tar.gz"),
);
await mkdir(path.dirname(output), { recursive: true });
const result = spawnSync(
  "tar",
  [
    "--exclude=node_modules",
    "--exclude=dist",
    "--exclude=.DS_Store",
    "-czf",
    output,
    "-C",
    path.dirname(root),
    path.basename(root),
  ],
  { stdio: "inherit" },
);
if (result.status !== 0) throw Error("Could not create skill archive");
console.log(output);
