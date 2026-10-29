---
name: character-refinement
description: Repair a specified character version while preserving unrelated results and review history.
---

Inputs: projectId, runId, resource version, body part and concrete feedback. Animation defects also need animation name and time. Read workflow_get and asset_inspect before starting.

For animation repairs validated against reference footage, use [frame-overlay-review](../frame-overlay-review/SKILL.md) to compare synchronized frames in fixed camera coordinates, inspect each view and report paired before/after scores with coverage.

Use workflow_decide request_changes for a pending review, recording the exact version and observation. To restart a production stage, use job_control action=rerun with fromStep, current expectedRevision and a stable idempotencyKey. Optional inputRevisionIds explicitly replaces that stage's inputs. Completed upstream steps are reused; this initial linear template does not provide independent head/body branches.

Produce real authorized repairs with local tools or manual handoff, import a new immutable version and submit it. Old hashes remain unchanged. Record only checks actually run. If Blender, deformation preview or engine inspection was not executed, report it as unverified. Do not auto-accept a new revision using an old visual decision.

## Textured model completion

After a complete model has geometry, UVs, materials and textures, load [glb-compress](../glb-compress/SKILL.md) and its compression policy by default before delivery. Re-run this step after texture changes. Generate project-aware recommendations and visual candidates in one dedicated temporary directory; keep the complete source and fixed production versions intact. Godot projects use geometry=off. Choose between WebP 1K/2K (or source resolution when detail requires it) using actual size and same-camera renders, not triangle count alone. Do not apply this step to unfinished meshes or replace a reviewed rig/training source with a compressed derivative.
