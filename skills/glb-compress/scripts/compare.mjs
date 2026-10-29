#!/usr/bin/env node
import { mkdir, writeFile } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { parseArgs } from "node:util";
import { projectContext } from "./policy.mjs";
const run = promisify(execFile);
const cli = fileURLToPath(new URL("./compress.mjs", import.meta.url));
const variants = [
  ["all-off", { preset: "none", textures: "off", resizeMax: 0 }],
  ["webp-8k", { preset: "none", textures: "webp", resizeMax: 0 }],
  ["webp-2k", { preset: "none", textures: "webp", resizeMax: 2048 }],
  ["webp-1k", { preset: "none", textures: "webp", resizeMax: 1024 }],
  ["resize-1k", { preset: "none", textures: "off", resizeMax: 1024 }],
  ["draco", { preset: "balanced", textures: "off", resizeMax: 0 }],
  [
    "simplify-75",
    { preset: "none", textures: "off", resizeMax: 0, simplify: "balanced" },
  ],
  [
    "simplify-60",
    { preset: "none", textures: "off", resizeMax: 0, simplify: "aggressive" },
  ],
  ["balanced", {}],
];
try {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: { project: { type: "string" }, engine: { type: "string" } },
  });
  const [input, destination, ...extra] = positionals;
  if (!input || !destination || extra.length)
    throw Error(
      "Usage: node scripts/compare.mjs INPUT.glb NEW_OUTPUT_DIRECTORY [--project ROOT] [--engine auto|godot|generic]",
    );
  const context = projectContext({
    project: values.project,
    input,
    engine: values.engine,
  });
  const dir = path.resolve(destination);
  // A fresh directory prevents stale or overwritten comparison artifacts.
  await mkdir(dir);
  const reports = [];
  for (const [name, options] of variants) {
    if (name === "draco" && context.engine === "godot") continue;
    console.error(`Compressing ${name}…`);
    const args = Object.entries(options).flatMap(([k, v]) => [
      `--${k}`,
      String(v),
    ]);
    if (values.project) args.push("--project", values.project);
    if (values.engine) args.push("--engine", values.engine);
    const { stdout } = await run(
      process.execPath,
      [cli, path.resolve(input), path.join(dir, `${name}.glb`), ...args],
      { maxBuffer: 8 * 1024 * 1024 },
    );
    reports.push({ name, ...JSON.parse(stdout) });
  }
  await writeFile(
    path.join(dir, "comparison.json"),
    JSON.stringify(reports, null, 2) + "\n",
  );
  console.log(
    JSON.stringify(
      reports.map((r) => ({
        name: r.name,
        bytes: r.after.bytes,
        triangles: r.after.triangles,
        textures: r.after.textures,
        warnings: r.warnings,
      })),
      null,
      2,
    ),
  );
} catch (e) {
  console.error(e.stderr || e.message);
  process.exitCode = 1;
}
