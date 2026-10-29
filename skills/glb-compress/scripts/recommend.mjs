#!/usr/bin/env node
import { parseArgs } from "node:util";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import path from "node:path";
import { projectContext, recommendCompression } from "./policy.mjs";
try {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      project: { type: "string" },
      engine: { type: "string", default: "auto" },
      usage: { type: "string", default: "standard" },
      "plan-dir": { type: "string" },
      help: { type: "boolean" },
    },
  });
  if (values.help || positionals.length !== 1) {
    console.log(
      "Usage: node scripts/recommend.mjs INPUT.glb [--project ROOT] [--engine auto|godot|generic] [--usage standard|close|distant] [--plan-dir NEW_DIRECTORY]",
    );
    process.exit(values.help ? 0 : 1);
  }
  const input = path.resolve(positionals[0]);
  const { getIO, stats, parseGLB } = await import("./engine.mjs");
  const bytes = await readFile(input);
  parseGLB(bytes);
  const context = projectContext({
    project: values.project,
    input,
    engine: values.engine,
  });
  const analysis = await stats(
    await (await getIO()).readBinary(new Uint8Array(bytes)),
    bytes.length,
  );
  const plan = recommendCompression(analysis, context, { usage: values.usage });
  plan.input = input;
  if (values["plan-dir"]) {
    const dir = path.resolve(values["plan-dir"]);
    await mkdir(dir, { recursive: false });
    for (const c of plan.candidates)
      await writeFile(
        path.join(dir, c.id + ".json"),
        JSON.stringify(c.options, null, 2) + "\n",
        { flag: "wx" },
      );
    await writeFile(
      path.join(dir, "plan.json"),
      JSON.stringify(plan, null, 2) + "\n",
      { flag: "wx" },
    );
  }
  console.log(JSON.stringify(plan, null, 2));
} catch (e) {
  console.error(`glb-compress recommendation: ${e.message}`);
  process.exitCode = 1;
}
