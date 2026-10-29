# Configuration contract

Reference UI: [3D AI Studio GLB Compression](https://www.3daistudio.com/Tools/GLBCompression), inspected 2026-10-30. The linked conversation “解析 GLB 模型压缩机制” measured actual website downloads. They identified glTF Transform 4.5.1 as the generator, but did not disclose backend encoder settings. This tool pins that version and implements its own explicit preset mapping.

Production entrypoint: [compression policy](compression-policy.md). `recommend.mjs` selects trial configurations from actual statistics; the raw codec defaults below remain available for generic projects. `compress.mjs` detects Godot from input/cwd ancestors or `--project`, forces geometry off, and rejects explicit geometry=on. `--engine godot` supplies engine context without a project file; `--engine generic` cannot override a detected Godot project.

CLI names and JSON keys are identical, including camelCase. Boolean CLI values must be `true` or `false`; JSON uses actual booleans. Unknown keys and invalid values fail before encoding.

| Website control / submitted key | Local CLI / JSON | Default | Meaning |
| --- | --- | --- | --- |
| Balanced / Maximum / Fast / Minimal | `preset`: `balanced`, `max`, `fast`, `none` | `balanced` | Select local encoder profile. Minimal forces Draco off; still converts/resizes textures. |
| Geometry compression | `geometry`: `on`, `off` | `on` | Draco geometry encoding; off also decodes existing Draco/Meshopt input. |
| Textures | `textures`: `webp`, `jpeg`, `off`, `ktx2` | `webp` | Re-encode textures. Off preserves encoded bytes unless resizing is required. |
| Max Texture Size | `resizeMax`: integer | `1024` | UI choices 0 / 512 / 1024 / 2048; CLI additionally permits integers up to 32768. 0 is no limit. Fit inside the square, preserve aspect ratio, never upscale. |
| Simplify Geometry | `simplify`: `off`, `light`, `balanced`, `aggressive` | `off` | Retain target 90% / 75% / 60% of triangles. Also accepts boolean for request-style configuration. |
| Simplify ratio | `simplifyRatio`: number >0, <=1 | `0.75` | Used with `simplify=true`. Named levels select their corresponding ratio. |
| Weld Vertices | `weld`: `off`, `light`, `standard`, `aggressive` | `off` | Tolerance 1e-5 / 1e-4 / 1e-3 in mesh-local units. Also accepts boolean. |
| Weld tolerance | `weldTolerance`: number >0, <=1 | `0.0001` | Used with `weld=true`. Named levels select their tolerance. |
| Keep object names | `keepNames`: boolean | `true` | False removes glTF property names, not arbitrary extras. |
| Strip animations | `stripAnimations`: boolean | `false` | Remove animation channels, samplers and orphan accessors; retain skin/bones. |
| Strip tangents (submitted fixed value) | `stripTangents`: boolean | `false` | Remove base and morph tangent attributes; renderer must regenerate if needed. |
| Mipmaps (submitted fixed value) | `mipmaps`: boolean | `false` | KTX2: embed actual mip levels; other images: request trilinear runtime filtering and emit a warning, since these formats cannot store mip chains in glTF. |
| Preserve precision (submitted fixed value) | `preservePrecision`: boolean | `false` | Local definition: disable Draco; reject simultaneous tolerance welding/simplification. Texture settings remain independent. This cannot recover precision lost before input. |
| Additional local control | `simplifyError`: 0…1 | `0.0001` | Meshoptimizer relative error limit; may stop before target ratio. |

The current website UI does not expose every submitted field: geometry follows the preset; mipmaps, stripTangents and preservePrecision are fixed values. Local CLI exposes them explicitly. Website `stream` and `async` are HTTP transport/job controls, not compression settings; there is no server upload or remote job in this local tool. Use the invoking process's background execution if needed.

## Local encoder profiles

These are local definitions, **not reverse-engineered website values**. Individual feature switches remain unchanged when choosing a preset, except Minimal's geometry-off behavior.

| Preset | WebP/JPEG quality | WebP effort | Draco encode/decode speed | Position / normal / UV / color / generic quantization bits | KTX ETC1S quality |
| --- | --- | --- | --- | --- | --- |
| balanced | 80 | 4 | 5 / 5 | 14 / 10 / 12 / 8 / 12 | 128 |
| max | 70 | 6 | 0 / 0 | 12 / 8 / 10 / 8 / 10 | 96 |
| fast | 80 | 0 | 10 / 10 | 14 / 10 / 12 / 8 / 12 | 128 |
| none | 80 | 4 | disabled | disabled | 128 |

JPEG uses 4:4:4 sampling. Images with an alpha channel remain PNG when JPEG is requested, with an explicit warning. PNG resize writes lossless PNG. KTX2 uses KTX-Software `toktx`, Basis ETC1S (`--encode etc1s`, BasisLZ payload), compression level 1 for Fast, 2 otherwise. Material color slots use sRGB; data textures use linear transfer. A texture shared between color and data slots must be split before KTX2 encoding. Base dimensions must be divisible by 4 for KHR_texture_basisu; incompatible dimensions are rejected instead of distorting the aspect ratio. KTX2 existing input can be preserved with textures=off, resizeMax=0, mipmaps=false; transcoding existing KTX2 is not implemented.

## Processing order and invariants

1. Read a complete self-contained GLB; decode input Draco/Meshopt. Reject unknown extensions, external files and URLs instead of silently dropping data or fetching resources.
2. Apply explicitly requested animation/tangent removal.
3. Weld within position tolerance. Keep all other attributes exactly equal, including UV, normals, colors, skin weights/joints and every morph delta. Search neighboring spatial cells; no global position rounding. glTF Transform's current exact-only weld is not used as a pretend tolerance weld.
4. Simplify with meshoptimizer and the declared error bound. For skinned/morph models, weights, joint indices and morph attributes are remapped to the retained vertices, and animation channels remain. The error bound measures rest-pose geometry, not animated deformation; the report calls out the need to review poses in the target renderer.
5. Resize and encode textures; rebuild texture extension declarations. Set mip filter when requested.
6. Remove names if requested, remove unreferenced accessors, consolidate buffers, then encode Draco if enabled. Topology/order can change through Draco even when welding is off.
7. Serialize and read back output before writing it, reporting decoded counts, dimensions and required extensions. No dedup/prune/flatten/animation resampling is silently applied.

“All off” means `preset=none`, `textures=off`, `resizeMax=0`, with other defaults. It preserves model payloads but the GLB JSON/container is reserialized, so whole-file byte identity is not promised. Geometry compression does not imply fewer triangles; image file size does not measure VRAM or frame rate. Use the report's actual retained triangle ratio, including cases where 75% and 60% requests reach the same result.

Both output GLB and report must be distinct from source, including symlinks and hardlinks. Output/report must be distinct from each other. Existing destinations fail without `--force`. Writes use a temporary sibling file and atomic rename/link. Output and report are individually atomic, not a multi-file transaction; if a final report write fails, a validated output may already exist and the CLI still exits nonzero.

## API use

The byte-only API does not infer a filesystem project: Node callers must use `projectContext` and `enforceProjectPolicy` from `scripts/policy.mjs` when targeting a project. The CLI applies them automatically.

For external Node.js callers, import `compressBytes` from `scripts/engine.mjs`. It returns `{output: Uint8Array, report}` and does not write files. `resolveOptions` from `scripts/options.mjs` validates and expands settings. Concurrent callers should use separate worker processes: Draco I/O modules are shared within a process. CLI and comparison runner process jobs sequentially.

## Primary implementation references

- [glTF Transform](https://gltf-transform.dev/): NodeIO, extension codecs, simplification and texture/material APIs.
- [Draco glTF extension](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_draco_mesh_compression).
- [Meshoptimizer simplifier](https://github.com/zeux/meshoptimizer/tree/master/js).
- [Sharp image output](https://sharp.pixelplumbing.com/api-output/).
- [KTX-Software](https://github.com/KhronosGroup/KTX-Software).
