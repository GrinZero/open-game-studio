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
| Historical Balanced (WebP 1K) | 96,328 | 996 | WebP 1024 + Draco |
| KTX2 + mipmaps, no Draco | 201,012 | 996 | ETC1S 1024, 11 mip levels |

Both simplify requests retained 90.16% due to the chosen error/topology constraints, matching the earlier website triangle count. The report explicitly warns that the target was not achieved. The historical local Balanced result is 94.07 KiB; the website study's Balanced was 143.68 KiB. Different encoder settings mean different size/quality; the smaller result is not evidence of equal visual quality.

All ten outputs were read back. Khronos glTF Validator 2.0.0-dev.3.10 reported zero errors. Nine ordinary outputs had zero warnings; the KTX2 output produced two image-format warnings because that validator does not support KHR_texture_basisu. The extracted KTX2 payload separately passed KTX-Software `ktx validate --gltf-basisu --warnings-as-errors`, with `valid: true` and no messages. This verifies the actual compressed texture, including BasisLZ and mip storage, rather than trusting the MIME label.

Those ten encoder checks establish structure and encoding, not visual quality. A subsequent [seven-model study](policy-study.md) compares this implementation's actual non-Draco output in Godot Forward+ and Mobile renderer on desktop: 17 candidates and 120 screenshots, including two representative animated poses. Its conclusions and limits are documented separately. Verify the chosen result in the intended project before replacing an asset.

## Completion policy checks

The suite now includes 21 behavior tests. Policy cases cover Godot detection outside the project working directory, explicit Draco rejection through both CLI and JSON config, no-Draco serialized output, the comparison matrix skipping Draco, low-poly texture prioritization, source-size-aware candidates, geometry-heavy assets, preservation of skins/animation/morphs, and recommendations remaining unselected before visual review. The portable recommendation command writes reusable candidate configs without modifying the model.

## Asset-specific assessment

Recommendations rank actual per-image payloads, slots, dimensions and encoding, and prefer a source-resolution KTX2 candidate and leave final selection unset for human comparison. A regression case verifies that a large metallic/roughness PNG ranks before a small color image and suggests measuring original-size lossless re-encoding. The earlier color-only restriction has been withdrawn. The bulk encoder still processes all images when executed; the skill requires diagnosing suitability first.

## Default KTX2 review regression

The real encoder test exercises omitted texture/resolution options, verifies actual BasisLZ/mip output, and asserts that successful encoding remains awaiting_user_comparison with no decision or selection. Recommendation/CLI tests verify that the default KTX2 plan retains source dimensions and requires human approval. Non-KTX2 encoder tests choose their format explicitly.

A real 4K harbor-wall GLB was encoded with omitted texture/resolution options: KTX2/BasisLZ at 4096×4096, Godot geometry off, awaiting_user_comparison. Godot 4.7.2 loaded that actual file with color, normal and metallic/roughness slots present; 16 Forward+ original/candidate screenshots cover two distances and four light directions. This is comparison material, with no human acceptance recorded.

## Bundled Godot runtime loader

The probe-driven godot_ktx2_loader.gd is validated in Godot 4.7.2. Runtime tests check the synthetic sRGB probe, compensation, no modification of linear maps, pass-through when compensation is disabled, and unchanged source bytes. Real GLB captures cover original, color-only/data-only KTX2, all-KTX2 and decoded-PNG controls in Forward+ and Mobile desktop. User acceptance remains pending. A codec regression checks sRGB/BT709 for color and linear/unspecified primaries for normal/MR maps.
