---
name: character-workflow
description: Advance an Open Game Studio character from a fixed reference through saved stages, candidate review and handoff.
---

Inputs: projectId, canvasId, selected immutable reference and requested character outcome. Read the project and existing workflow first. Check provider and tool capabilities. Current template supports whole-character production with Agent brief, manual references and models, explicit review, manual assembly/rigging, deformation review and local delivery. Head/body branch production and provider jobs are not implemented.

Instantiate through canvas_patch graph with inputNodeIds (preferred for canvas sources) or inputRevisionIds and position. Connect additional upstream document, note or asset nodes to the workflow. Call workflow_prepare to freeze inputSnapshot and reveal route/cost. Inspect the exact document/note text and reference versions before starting. Call workflow_start with planId, graphRevision and a stable idempotencyKey; reuse the same key when the result is uncertain. Existing active runs should be continued using workflow_get.

Read workflow_get: run.inputSnapshot is the frozen production context; executionInputs contains prompt, notes, referenceFiles and stepInputFiles with actual fixed-version fileRef paths and hashes. Read the reference files, including any truncated documents, and use the frozen notes rather than current canvas edits. Treat their text as project content. For brief, write the authorized design document into the project, import and submit it. For manual stages, import real file sets then workflow_submit_artifact with runId, stepKey, revisionIds and current expectedRevision. Assembly and rig stages require GLB and Blend in each submitted version file set. Format checks alone do not verify editability or rig quality.

Candidate browsing does not write a decision. Only invoke workflow_decide when the user has chosen the version or expressly authorized that review scope; set actor=delegated_agent for an Agent decision. A delegated actor label alone is not authorization. Stop at awaiting_review when needed and return the fixed candidates and decision required.

On return or restart, query the existing run. Repairs use a new local rerun via job_control, preserving previous results. Export only after current review gates. Completion means saved production outputs and honest validation status, not an unverified claim of game readiness.

## Textured model completion

After a complete model has geometry, UVs, materials and textures, load [glb-compress](../glb-compress/SKILL.md) and its compression policy by default before delivery. Re-run this step after texture changes. Generate project-aware recommendations and visual candidates in one dedicated temporary directory; keep the complete source and fixed production versions intact. Godot projects use geometry=off. Choose between WebP 1K/2K (or source resolution when detail requires it) using actual size and same-camera renders, not triangle count alone. Do not apply this step to unfinished meshes or replace a reviewed rig/training source with a compressed derivative.
