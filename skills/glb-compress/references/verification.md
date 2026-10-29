# Verification record — 2026-10-30

Tested locally on macOS arm64 with Node.js 24.14.0, pinned Node dependencies, and KTX-Software 4.4.2. No model data was uploaded. Runtime compression uses no network API.

## Automated behavior checks

Run `npm test` from the skill directory. Set `TOKTX=/absolute/path/to/toktx` to include real KTX2 encoding; without it, that test explicitly skips, while missing-encoder failure is still tested.

Covered: all preset profiles, default Draco/WebP read-back, no-op payload preservation, resize/aspect ratio/no-upscale, JPEG/alpha preservation, actual simplification and ratios, tolerance welding with UV/normal/joint/morph seams, skin/morph/animation retention through compression and simplification, removal of animation/tangent/name data, exact-geometry precision mode, runtime mip filter, unknown extension rejection, real ETC1S/BasisLZ and mip levels, JSON config/CLI/report/inspect behavior, output collision and input symlink/hardlink protections.

## Real model comparison

Sample: the original study's wooden stairs, 14,356,436 bytes, 996 triangles, 1044 vertices, one 8192×8192 JPEG. The model and website downloads are not included in the distributable skill.

| Local variant | Bytes | Triangles | Texture |
| --- | ---: | ---: | --- |
| All options off | 14,356,408 | 996 | JPEG 8192 |
| WebP, no resize | 3,018,880 | 996 | WebP 8192 |
| WebP 2K, no Draco | 341,448 | 996 | WebP 2048 |
| WebP 1K, no Draco | 129,840 | 996 | WebP 1024 |
| Resize 1K, texture format off | 220,300 | 996 | JPEG 1024 |
| Draco, original texture | 14,322,944 | 996 | JPEG 8192 |
| Simplify target 75% | 14,354,028 | 898 | JPEG 8192 |
| Simplify target 60% | 14,354,028 | 898 | JPEG 8192 |
| Balanced defaults | 96,328 | 996 | WebP 1024 + Draco |
| KTX2 + mipmaps, no Draco | 201,012 | 996 | ETC1S 1024, 11 mip levels |

Both simplify requests retained 90.16% due to the chosen error/topology constraints, matching the earlier website triangle count. The report explicitly warns that the target was not achieved. Local Balanced is 94.07 KiB; the website study's Balanced was 143.68 KiB. Different encoder settings mean different size/quality; the smaller result is not evidence of equal visual quality.

All ten outputs were read back. Khronos glTF Validator 2.0.0-dev.3.10 reported zero errors. Nine ordinary outputs had zero warnings; the KTX2 output produced two image-format warnings because that validator does not support KHR_texture_basisu. The extracted KTX2 payload separately passed KTX-Software `ktx validate --gltf-basisu --warnings-as-errors`, with `valid: true` and no messages. This verifies the actual compressed texture, including BasisLZ and mip storage, rather than trusting the MIME label.

Those ten encoder checks establish structure and encoding, not visual quality. A subsequent [seven-model study](policy-study.md) compares this implementation's actual non-Draco output in Godot Forward+ and Mobile renderer on desktop: 17 candidates and 120 screenshots, including two representative animated poses. Its conclusions and limits are documented separately. Verify the chosen result in the intended project before replacing an asset.

## Completion policy checks

The suite now includes 19 behavior tests. Policy cases cover Godot detection outside the project working directory, explicit Draco rejection through both CLI and JSON config, no-Draco serialized output, the comparison matrix skipping Draco, low-poly texture prioritization, source-size-aware candidates, geometry-heavy assets, preservation of skins/animation/morphs, and recommendations remaining unselected before visual review. The portable recommendation command writes reusable candidate configs without modifying the model.
