#!/usr/bin/env node
import { parseArgs } from "node:util";
import {
  readFile,
  writeFile,
  mkdir,
  realpath,
  stat,
  rename,
  link,
  unlink,
} from "node:fs/promises";
import path from "node:path";
import { randomUUID, createHash } from "node:crypto";
import { defaults, resolveOptions, schema } from "./options.mjs";
import { projectContext, enforceProjectPolicy } from "./policy.mjs";

const help = `Local GLB compression (Node.js 22+)
  node scripts/compress.mjs INPUT.glb OUTPUT.glb [options]
  node scripts/compress.mjs --inspect INPUT.glb
  node scripts/compress.mjs --schema

All processing is local. First install dependencies: npm ci --prefix SKILL_DIR
Compression options (also accepted as camelCase keys in --config JSON):
  --preset balanced|max|fast|none       default balanced; none forces geometry off
  --geometry on|off                    default on (Draco)
  --textures webp|jpeg|off|ktx2         default ktx2; creates a candidate pending user review
  --resizeMax N                       default 0 (source size); no upscaling
  --simplify off|light|balanced|aggressive|true|false
  --simplifyRatio N                   used with simplify=true (0 < N <= 1)
  --simplifyError N                   default 0.0001, relative error limit
  --weld off|light|standard|aggressive|true|false
  --weldTolerance N                   used with weld=true; mesh-local units
  --keepNames true|false              default true
  --stripAnimations true|false        default false
  --stripTangents true|false          default false
  --mipmaps true|false                default false; embedded levels for KTX2
  --preservePrecision true|false      default false; disables lossy geometry

Execution:
  --config FILE.json   CLI values override JSON
  --report FILE.json   default OUTPUT.glb.report.json
  --toktx PATH         KTX2 encoder, otherwise TOKTX env or toktx on PATH
  --project ROOT       Target project root (also detects ROOT/game/project.godot)
  --engine auto|godot|generic  Default auto; Godot always disables Draco
  --dry-run            print resolved options without reading/writing a model
  --force              replace output/report; never the source
  --help               this help
JSON report is printed on stdout; diagnostics go to stderr.
`;
async function identity(p) {
  try {
    const s = await stat(p);
    return `${s.dev}:${s.ino}`;
  } catch (e) {
    if (e.code === "ENOENT") return null;
    throw e;
  }
}
async function canonical(p) {
  try {
    return await realpath(p);
  } catch (e) {
    if (e.code !== "ENOENT") throw e;
    return path.join(await realpath(path.dirname(p)), path.basename(p));
  }
}
async function writeAtomic(file, data, force) {
  const tmp = path.join(
    path.dirname(file),
    `.${path.basename(file)}.${randomUUID()}.tmp`,
  );
  try {
    await writeFile(tmp, data, { flag: "wx" });
    if (force) await rename(tmp, file);
    else {
      await link(tmp, file);
      await unlink(tmp);
    }
  } finally {
    await unlink(tmp).catch((e) => {
      if (e.code !== "ENOENT") throw e;
    });
  }
}
try {
  const options = Object.fromEntries(
    Object.keys(defaults).map((k) => [k, { type: "string" }]),
  );
  for (const k of ["config", "report", "toktx", "project", "engine"])
    options[k] = { type: "string" };
  for (const k of ["help", "schema", "inspect", "dry-run", "force"])
    options[k] = { type: "boolean" };
  const { values, positionals } = parseArgs({
    options,
    allowPositionals: true,
    strict: true,
  });
  if (values.help) {
    console.log(help);
    process.exit(0);
  }
  if (values.schema) {
    console.log(JSON.stringify(schema, null, 2));
    process.exit(0);
  }
  let inputOptions = values.config
    ? JSON.parse(await readFile(values.config, "utf8"))
    : {};
  if (
    !inputOptions ||
    typeof inputOptions !== "object" ||
    Array.isArray(inputOptions)
  )
    throw Error("Config must be a JSON object");
  for (const [key, value] of Object.entries(values))
    if (Object.hasOwn(defaults, key)) {
      let parsed = value;
      if (typeof defaults[key] === "number") parsed = Number(value);
      if (
        typeof defaults[key] === "boolean" ||
        ["simplify", "weld"].includes(key)
      ) {
        if (value === "true") parsed = true;
        if (value === "false") parsed = false;
      }
      inputOptions[key] = parsed;
    }
  const context = projectContext({
    project: values.project,
    input: positionals[0],
    engine: values.engine,
  });
  inputOptions = enforceProjectPolicy(inputOptions, context);
  const resolved = { ...resolveOptions(inputOptions), context };
  if (values["dry-run"]) {
    console.log(JSON.stringify(resolved, null, 2));
    process.exit(0);
  }
  if (positionals.length !== (values.inspect ? 1 : 2)) throw Error(help);
  const input = path.resolve(positionals[0]);
  const bytes = await readFile(input);
  let engine;
  try {
    engine = await import("./engine.mjs");
  } catch (e) {
    if (e.code === "ERR_MODULE_NOT_FOUND")
      throw Error(
        "Missing dependencies. Run npm ci --prefix /absolute/path/to/glb-compress first.",
      );
    throw e;
  }
  if (values.inspect) {
    engine.parseGLB(bytes);
    console.log(
      JSON.stringify(
        await engine.stats(
          await (await engine.getIO()).readBinary(new Uint8Array(bytes)),
          bytes.length,
        ),
        null,
        2,
      ),
    );
    process.exit(0);
  }
  const output = path.resolve(positionals[1]),
    reportPath = path.resolve(values.report ?? `${output}.report.json`);
  if (path.extname(output).toLowerCase() !== ".glb")
    throw Error("Output must have .glb extension");
  await mkdir(path.dirname(output), { recursive: true });
  await mkdir(path.dirname(reportPath), { recursive: true });
  const paths = [input, output, reportPath],
    canonicalPaths = await Promise.all(paths.map(canonical)),
    ids = await Promise.all(paths.map(identity));
  for (let i = 0; i < 3; i++)
    for (let j = i + 1; j < 3; j++) {
      if (
        canonicalPaths[i] === canonicalPaths[j] ||
        (ids[i] && ids[i] === ids[j])
      )
        throw Error(
          "Input, output and report must be different files (including symlinks/hardlinks)",
        );
    }
  if (!values.force && (ids[1] || ids[2]))
    throw Error(
      "Output/report already exists; choose new paths or use --force",
    );
  const { output: result, report } = await engine.compressBytes(
    bytes,
    inputOptions,
    { toktx: values.toktx },
  );
  report.projectPolicy = context;
  report.files = {
    input,
    output,
    report: reportPath,
    inputSha256: createHash("sha256").update(bytes).digest("hex"),
    outputSha256: createHash("sha256").update(result).digest("hex"),
  };
  // Refuse writing if the source or existing destination changed while encoding.
  const current = await readFile(input);
  if (!bytes.equals(current))
    throw Error("Input changed during compression; no output written");
  const now = await Promise.all(paths.map(identity));
  if (now.some((id, i) => id !== ids[i]))
    throw Error("A destination changed during compression; no output written");
  await writeAtomic(output, result, values.force);
  await writeAtomic(
    reportPath,
    JSON.stringify(report, null, 2) + "\n",
    values.force,
  );
  console.log(JSON.stringify(report, null, 2));
} catch (e) {
  console.error(`glb-compress: ${e.message}`);
  process.exitCode = 1;
}
