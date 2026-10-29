---
name: game-asset-delivery
description: Export fixed Open Game Studio asset versions and truthful handoff evidence for a game project.
---

Read workflow_get and asset_inspect. The completed character route requires matching GLB and Blend versions and an explicit deformation review. User decisions are stored separately from technical and engine evidence.

Call export_bundle with the exact delivery inputs, runId, expectedRevision and stable idempotencyKey. The Core copies fixed files under the project's `.game-studio/exports`, verifies hashes and writes manifest.json. It does not overwrite a game directory or run Godot.

Read the returned bundle and actual manifest. Technical passed in this version covers file integrity and import format structure only. Blend opening, mesh budget, skinning, materials and engine behavior require real tools; do not reinterpret these as passed. The engine field remains not_verified until a future adapter performs real import and records evidence.

Complete with the bundle path, version set and remaining checks. Exporting arbitrary resources without a run is supported but does not create visual approval.

## Textured model completion

After a complete model has geometry, UVs, materials and textures, load [glb-compress](../glb-compress/SKILL.md) and its compression policy by default before delivery. Re-run this step after texture changes. Generate project-aware recommendations and visual candidates in one dedicated temporary directory; keep the complete source and fixed production versions intact. Godot projects use geometry=off. Choose between WebP 1K/2K (or source resolution when detail requires it) using actual size and same-camera renders, not triangle count alone. Do not apply this step to unfinished meshes or replace a reviewed rig/training source with a compressed derivative.
