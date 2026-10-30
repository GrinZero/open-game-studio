---
name: glb-compress
description: Compress local GLB models with Draco, WebP/JPEG/KTX2, texture resizing, simplification and vertex welding. Load by default after a model is complete with UVs, materials and textures, and before delivery; also use for offline size reduction and visual comparisons. Detect Godot projects, disable Draco, and recommend texture/geometry candidates from measured model statistics.
---

# Local GLB compression

Use the bundled Node.js CLI. Model processing stays local; it does not need Open Game Studio, Blender, a website account, API credentials, or a running MCP server. Package installation needs internet once; subsequent compression runs offline.

## Default completion policy

As soon as a model is complete **including UVs, materials and textures**, read [asset-specific assessment policy](references/compression-policy.md) before delivery. **KTX2 is enabled by default as a candidate and requires human visual comparison before adoption.** Read [the mandatory KTX2 human review](references/ktx2-human-review.md). For Godot loading, read [the KTX2 loading adapter](references/godot-ktx2-loading.md) and use its bundled runtime helper when the probe detects pre-linearized color images. Re-evaluate after texture or geometry revisions. This applies to standalone creation and the bundled resource/character/delivery workflows; it does not compress unfinished models. Default loading is a skill workflow rule, not a background filesystem watcher.

Use the project-aware recommendation command below. It measures triangle count, image share and source resolution, and writes candidate configs without claiming visual acceptance. Godot projects always use geometry=off; this also applies to explicit settings and high-poly models. The CLI detects an ancestor project.godot or a repository's game/project.godot; pass --project for a different target project or --engine godot when the engine is known but no marker exists.

```sh
node "$SKILL_DIR/scripts/recommend.mjs" /absolute/complete-model.glb \
  --project /absolute/game-project --usage standard \
  --plan-dir /absolute/dedicated-temp/compression-plan
# Default: KTX2 at source resolution. Generate review material, then wait for
# the user to choose this exact candidate before adopting it.
node "$SKILL_DIR/scripts/compress.mjs" /absolute/complete-model.glb \
  /absolute/dedicated-temp/candidate.glb --project /absolute/game-project \
  --config /absolute/dedicated-temp/compression-plan/ktx2.json
```

**根据实际资产处理，不预设只压颜色图，也不统一降分辨率；默认 KTX2 输出仅为待用户对比的候选。** 先检查每张贴图的体积、编码、分辨率和实际材质槽，再针对主要载荷比较保留原图、原尺寸无损重编码、降尺寸或有损编码的收益与效果。法线、粗糙度、金属度、AO、颜色和自发光都可以成为处理对象；是否处理由这次的事实决定。

Read [PBR observations](references/pbr-study.md) and the [large material-map counterexample](references/harbor-pbr-study.md). `recommend.mjs` ranks actual image payloads and sets `preferredTrial=ktx2` when textures exist and keeps `selected=null` pending user choice. Its whole-file configs are exploratory codec trials, not automatically appropriate choices for every map. The CLI applies texture settings to all images: use it only when that scope fits the assessed asset. If different maps need different treatment, use a targeted script or the authoring tool rather than pretending the bulk CLI selects individual maps. Keep originals and place trial outputs/reports in a dedicated folder.

## Setup and execution

Resolve `SKILL_DIR` to this skill's actual installed directory. Require Node.js 22+ and npm. If dependencies are missing, run:

```sh
npm ci --prefix "$SKILL_DIR" --no-audit --no-fund
node "$SKILL_DIR/scripts/compress.mjs" --help
```

Default compression uses KTX2, so install [KTX-Software](https://github.com/KhronosGroup/KTX-Software/releases) with its libraries. The CLI resolves the encoder from `--toktx`, `TOKTX`, `~/.local/share/glb-compress/bin/toktx`, then PATH. Run `toktx --version` to verify the installation. WebP, JPEG, resizing and Draco need no extra executable. Do not silently substitute another format if KTX2 is unavailable.

1. Inspect the input when choosing settings: `node "$SKILL_DIR/scripts/compress.mjs" --inspect /absolute/input.glb`.
2. Apply [asset-specific assessment policy](references/compression-policy.md). Inspect the ranked per-texture analysis and choose measured trials; do not treat raw presets as a production recommendation. Preserve explicit settings subject to the project's Godot no-Draco rule. If 1K/2K degrade visible detail, compare source resolution or retain the original. Keep skin, animation, tangents, names and morphs unless a separate task explicitly changes them.
3. Write a new output path. The source is never overwritten, even with `--force`. Existing outputs/reports require a new path or explicit replacement via `--force`.
4. **Godot loading:** use the [probe-driven KTX2 loader](references/godot-ktx2-loading.md) for runtime GLTFDocument loading; confirm the actual display path before judging compression quality.
5. **Human review:** show actual same-camera original/KTX2 comparisons and measured size differences, then wait for the user to choose adopt/original/revise. Keep the exact candidate pending; never automatically import or mark it accepted. Follow [the review protocol](references/ktx2-human-review.md).
6. Read the JSON report and summarize actual bytes saved, texture dimensions, triangles, required extensions and warnings. A requested simplification ratio is a target, not a guaranteed result. Structural read-back does not establish visual quality or engine acceptance.

```sh
# Default KTX2 candidate at source resolution; user visual review required.
node "$SKILL_DIR/scripts/compress.mjs" /absolute/source.glb /absolute/balanced.glb

# 2K WebP without Draco, retaining geometry and animation.
node "$SKILL_DIR/scripts/compress.mjs" /absolute/source.glb /absolute/webp-2k.glb \
  --geometry off --textures webp --resizeMax 2048

# Explicit configuration; command-line values override JSON.
node "$SKILL_DIR/scripts/compress.mjs" /absolute/source.glb /absolute/output.glb \
  --config "$SKILL_DIR/references/balanced.json"

# Optional bulk diagnostic matrix; assess suitability first. Godot skips Draco.
# Parent directory must exist; the output directory must be new.
node "$SKILL_DIR/scripts/compare.mjs" /absolute/source.glb /absolute/new-comparison
```

For observed examples read [seven-model visual study](references/policy-study.md). Use [configuration and implementation boundaries](references/options.md) for every available setting, exact preset encoding, KTX2/mipmap semantics and precision rules. `--schema` prints machine-readable defaults and enumerations; `--dry-run` prints effective settings before processing. Stdout is JSON; stderr is diagnostic text. Exit code 1 is failure. Every successful compression writes `OUTPUT.glb.report.json` unless `--report` chooses another path.

Do not promise byte-identical website output or equivalent visual quality: its backend parameters are private. This implementation uses the same controls with documented local algorithms. Preserve the original, and compare in the target renderer before replacing production assets. Do not import, register, publish or replace existing project assets merely because compression succeeded.

## External distribution

Copy this entire `glb-compress` folder into an agent's skill directory, for example `~/.codex/skills/glb-compress`, or `~/.agents/skills/glb-compress`. Then run the setup above. Copying only `SKILL.md` is insufficient. The skill has no absolute workspace dependency.

Build a portable source archive with `node "$SKILL_DIR/scripts/package.mjs" /absolute/glb-compress-skill.tar.gz`. The archive excludes installed dependencies and generated output; recipients install platform-appropriate dependencies from the lockfile. The KTX executable is installed separately and is required for the default KTX2 encoding. See [validation record](references/verification.md) for the tested scope and [third-party notices](references/third-party.md).
