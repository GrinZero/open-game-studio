---
name: game-asset-delivery
description: Export fixed Open Game Studio asset versions and truthful handoff evidence for a game project.
---

Read workflow_get and asset_inspect. The completed character route requires matching GLB and Blend versions and an explicit deformation review. User decisions are stored separately from technical and engine evidence.

Call export_bundle with the exact delivery inputs, runId, expectedRevision and stable idempotencyKey. The Core copies fixed files under the project's `.game-studio/exports`, verifies hashes and writes manifest.json. It does not overwrite a game directory or run Godot.

Read the returned bundle and actual manifest. Technical passed in this version covers file integrity and import format structure only. Blend opening, mesh budget, skinning, materials and engine behavior require real tools; do not reinterpret these as passed. The engine field remains not_verified until a future adapter performs real import and records evidence.

Complete with the bundle path, version set and remaining checks. Exporting arbitrary resources without a run is supported but does not create visual approval.

## Textured model completion

After a complete model has geometry, UVs, materials and textures, load [glb-compress](../glb-compress/SKILL.md) and its compression policy by default before delivery. Re-run this step after texture changes. Generate project-aware recommendations and visual candidates in one dedicated temporary directory; keep the complete source and fixed production versions intact. Godot projects use geometry=off. Choose preservation, lossless re-encoding, resizing or lossy encoding from the actual payload and same-camera results; no fixed resolution or texture-type rule applies. Do not apply this step to unfinished meshes or replace a reviewed rig/training source with a compressed derivative.

Inspect actual per-texture size, encoding, dimensions and material role before choosing compression. Compare measured savings and target-renderer appearance; do not hard-code color-only or all-texture lossy processing.

Generate a KTX2 compression candidate by default at source texture resolution. Follow glb-compress references/ktx2-human-review.md: show same-camera original/candidate visuals and sizes, then wait for the user to choose adopt, keep original or revise. Do not replace/import a production asset or mark it accepted before this explicit choice; regenerate the comparison after candidate changes.
