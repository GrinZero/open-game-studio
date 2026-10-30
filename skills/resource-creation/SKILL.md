---
name: resource-creation
description: Create and edit canvas documents, reuse pinned assets, or run persistent image/video/model/animation/audio production through the configured Codex tools.
---

Read project_open and canvas_open first. For Markdown use document_save with current graph revision; editing also requires the exact nodeId, assetId and original expectedLatestRevisionId. Saving creates an immutable version and updates only that node. Existing documents and other resources use asset nodes with an explicit assetId and revisionId, and may be reused across canvases.

For resource production use generation_create with an explicit kind, prompt, settings, selected inputNodeIds and a stable idempotencyKey. Input versions and text are frozen. Animation needs a model reference. Read generation_get to see the frozen text and actual reference files. Creation makes a draft, not a provider submission. generation_start claims the draft once and launches the configured local Codex executor. Never repeatedly submit when status is running, failed, interrupted or outcome is unclear. Look up the original provider receipt in the task output directory. Use actual supported, authenticated production tools, obey their confirmation requirements, and never fabricate generated assets.

Real results are downloaded to the task output directory, validated, imported and attached to the original canvas node. generation_adopt can attach an already-imported exact output revision after manual production or recovery. Animation outputs must have a GLB skeleton and animation clips; structural checks do not establish deformation or game readiness. generation_control stops local execution and retains history; it does not retract remote submissions. Missing tools, login or required user confirmation must be reported plainly.

## Textured model completion

After a complete model has geometry, UVs, materials and textures, load [glb-compress](../glb-compress/SKILL.md) and its compression policy by default before delivery. Re-run this step after texture changes. Generate project-aware recommendations and visual candidates in one dedicated temporary directory; keep the complete source and fixed production versions intact. Godot projects use geometry=off. Choose preservation, lossless re-encoding, resizing or lossy encoding from the actual payload and same-camera results; no fixed resolution or texture-type rule applies. Do not apply this step to unfinished meshes or replace a reviewed rig/training source with a compressed derivative.

Inspect actual per-texture size, encoding, dimensions and material role before choosing compression. Compare measured savings and target-renderer appearance; do not hard-code color-only or all-texture lossy processing.

Generate a KTX2 compression candidate by default at source texture resolution. Follow glb-compress references/ktx2-human-review.md: show same-camera original/candidate visuals and sizes, then wait for the user to choose adopt, keep original or revise. Do not replace/import a production asset or mark it accepted before this explicit choice; regenerate the comparison after candidate changes.
