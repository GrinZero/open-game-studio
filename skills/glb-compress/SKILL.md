---
name: glb-compress
description: Compress local GLB models with Draco, WebP/JPEG/KTX2, texture resizing, simplification and vertex welding. Load by default after a model is complete with UVs, materials and textures, and before delivery; also use for offline size reduction and visual comparisons. Detect Godot projects, disable Draco, and recommend texture/geometry candidates from measured model statistics.
---

# Local GLB compression

Use the bundled Node.js CLI. Model processing stays local; it does not need Open Game Studio, Blender, a website account, API credentials, or a running MCP server. Package installation needs internet once; subsequent compression runs offline.

## Default completion policy

As soon as a model is complete **including UVs, materials and textures**, read [compression policy v1](references/compression-policy.md) before delivery. Re-evaluate after texture or geometry revisions. This applies to standalone creation and the bundled resource/character/delivery workflows; it does not compress unfinished models. Default loading is a skill workflow rule, not a background filesystem watcher.

Use the project-aware recommendation command below. It measures triangle count, image share and source resolution, and writes candidate configs without claiming visual acceptance. Godot projects always use geometry=off; this also applies to explicit settings and high-poly models. The CLI detects an ancestor project.godot or a repository's game/project.godot; pass --project for a different target project or --engine godot when the engine is known but no marker exists.

```sh
node "$SKILL_DIR/scripts/recommend.mjs" /absolute/complete-model.glb \
  --project /absolute/game-project --usage standard \
  --plan-dir /absolute/dedicated-temp/compression-plan
node "$SKILL_DIR/scripts/compress.mjs" /absolute/complete-model.glb \
  /absolute/dedicated-temp/webp2k.glb --project /absolute/game-project \
  --config /absolute/dedicated-temp/compression-plan/webp2k.json
```

Start with 2K, compare 1K where appropriate, and retain 4K/original when closer views need it. Low-poly assets keep topology. Recommendations leave `selected=null` until actual renders have been compared. All temporary models, configs, screenshots and reports for a task belong in a dedicated folder. Keep originals untouched.

## Setup and execution

Resolve `SKILL_DIR` to this skill's actual installed directory. Require Node.js 22+ and npm. If dependencies are missing, run:

```sh
npm ci --prefix "$SKILL_DIR" --no-audit --no-fund
node "$SKILL_DIR/scripts/compress.mjs" --help
```

For KTX2, also install [KTX-Software](https://github.com/KhronosGroup/KTX-Software/releases) with its libraries. The CLI resolves the encoder from `--toktx`, `TOKTX`, `~/.local/share/glb-compress/bin/toktx`, then PATH. Run `toktx --version` to verify the installation. WebP, JPEG, resizing and Draco need no extra executable. Do not silently substitute another format if KTX2 is unavailable.

1. Inspect the input when choosing settings: `node "$SKILL_DIR/scripts/compress.mjs" --inspect /absolute/input.glb`.
2. Apply [compression policy v1](references/compression-policy.md). Use the recommended candidates rather than the website's unconditional 1K + Draco default. Preserve explicit settings subject to the project's Godot no-Draco rule. If 1K/2K degrade visible detail, compare source resolution or retain the original. Keep skin, animation, tangents, names and morphs unless a separate task explicitly changes them.
3. Write a new output path. The source is never overwritten, even with `--force`. Existing outputs/reports require a new path or explicit replacement via `--force`.
4. Read the JSON report and summarize actual bytes saved, texture dimensions, triangles, required extensions and warnings. A requested simplification ratio is a target, not a guaranteed result. Structural read-back does not establish visual quality or engine acceptance.

```sh
# Raw encoder defaults for a generic project (use recommendations for production).
node "$SKILL_DIR/scripts/compress.mjs" /absolute/source.glb /absolute/balanced.glb

# 2K WebP without Draco, retaining geometry and animation.
node "$SKILL_DIR/scripts/compress.mjs" /absolute/source.glb /absolute/webp-2k.glb \
  --geometry off --textures webp --resizeMax 2048

# Explicit configuration; command-line values override JSON.
node "$SKILL_DIR/scripts/compress.mjs" /absolute/source.glb /absolute/output.glb \
  --config "$SKILL_DIR/references/balanced.json"

# Diagnostic matrix matching the original study; Godot skips the Draco variant.
# Parent directory must exist; the output directory must be new.
node "$SKILL_DIR/scripts/compare.mjs" /absolute/source.glb /absolute/new-comparison
```

For observed examples read [seven-model visual study](references/policy-study.md). Use [configuration and implementation boundaries](references/options.md) for every available setting, exact preset encoding, KTX2/mipmap semantics and precision rules. `--schema` prints machine-readable defaults and enumerations; `--dry-run` prints effective settings before processing. Stdout is JSON; stderr is diagnostic text. Exit code 1 is failure. Every successful compression writes `OUTPUT.glb.report.json` unless `--report` chooses another path.

Do not promise byte-identical website output or equivalent visual quality: its backend parameters are private. This implementation uses the same controls with documented local algorithms. Preserve the original, and compare in the target renderer before replacing production assets. Do not import, register, publish or replace existing project assets merely because compression succeeded.

## External distribution

Copy this entire `glb-compress` folder into an agent's skill directory, for example `~/.codex/skills/glb-compress`, or `~/.agents/skills/glb-compress`. Then run the setup above. Copying only `SKILL.md` is insufficient. The skill has no absolute workspace dependency.

Build a portable source archive with `node "$SKILL_DIR/scripts/package.mjs" /absolute/glb-compress-skill.tar.gz`. The archive excludes installed dependencies and generated output; recipients install platform-appropriate dependencies from the lockfile. The optional KTX executable is installed separately. See [validation record](references/verification.md) for the tested scope and [third-party notices](references/third-party.md).
