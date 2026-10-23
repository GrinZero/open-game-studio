---
name: studio-project
description: Create, link, rename or restore a local Open Game Studio project, locate unfinished production, or help users optionally adopt an engine-neutral project file guide.
---

Inputs are an intended project name, explicit directory or existing projectId. Use studio_open to list projects and resolve ambiguity. Do not choose among same-name projects by guesswork.

For create, pass an absolute new directory. For link, use an existing directory; registration adds `.game-studio` but does not reorganize original files. project_manage update changes name, engine, optional collection or archive status. Relocate requires a directory containing the exact existing project identity.

Read project_open after mutation and report actual path availability, canvases, pending runs and capability limits. Resume the existing run before creating a new one. Project archive never means deleting external files.

## Optional project file guide

When the user asks about project file organization, naming conventions, or adopting the bundled guide, read [the project file guide](references/project-file-guide.md). Explain or summarize it in the user's language. Reading or discussing the guide does not opt the project in. Creating or linking a project does not adopt it automatically.

The guide has a quick start and a complete reference, not weaker and stronger sets of rules. Before applying naming rules, read [asset naming](references/asset-naming.md); before engine-specific organization or technical checks, read [engine profiles](references/engine-profiles.md). Use [source coverage and decisions](references/guide-sources.md) when explaining the relationship to Allar and bakneko. Do not reduce an adoption request to creating a few generic folders: apply the selected rules for asset families, content ownership, core/shared boundaries, dependencies and relevant resource types.

For version history, synchronization, large files, file locks, restoration or backend choices, read [version control and collaboration](references/version-control.md). Engine and VCS are independent selections. Recognize the user's existing Git, Diversion, Perforce, Unity Version Control or other storage workflow using available read-only evidence; do not assume Git, initialize a repository or configure LFS merely because the guide is adopted. Record provider, repository/storage identity, fixed revision and authoritative file scopes, including hybrid code/art storage when applicable. Keep local saves, cloud workspace sync, submitted changes, branch integration, publication and backups distinct; inspect actual sync and lock capabilities rather than assume them.

After the user explicitly chooses to adopt all or part of the guide, inspect the project's existing conventions and engine. For a general adoption request, use the standard scheme with only the relevant resource types and create directories as they become necessary. Preserve prior selected rules and backend choices unless the user asks to expand or change them. Use normal workspace file tools to create only the needed directories and record the chosen version, rule scope, engine mapping, technical settings, version-control approach and exceptions in `PROJECT_STRUCTURE.md`, or update the existing equivalent document. A request to adopt is sufficient authorization for these additive changes; do not ask again for the same scope. Resolve an unknown target directory before writing, preserve existing documents, and keep `.game-studio` internals untouched. Do not call project registration tools just to discuss the guide or write a file convention.

For existing projects, apply chosen conventions to new files by default. Keep in-progress, unapproved production drafts, pre-flight rig inputs and candidate iterations in `scratch/work/` outside version control and formal manifests; only promote editable sources to `art_source/` (at a single stable path per asset, without nested stage/revision folders) and register them in VCS/LFS/DVC or source manifests after explicit user confirmation or formal game adoption. A request for advice or adoption alone does not authorize bulk relocation or renaming. Prepare a concrete path and reference impact list before a user-requested migration, use engine-aware operations, and verify affected resources in the actual engine. If engine verification is unavailable, report it as unverified.

This is optional conversation guidance, not a workbench toggle, scaffolding API, automatic linter or migration service. Users can adopt selected rules or stop following them without losing plugin functionality. Report the actual files written and selected scope; do not claim engine compatibility from naming checks.

When auditing a project, report the rule ID, affected path, observed evidence, recommendation and verification scope. Respect recorded aliases and exceptions; distinguish a naming deviation, a broken dependency, a technical problem and a visual-quality judgment. Do not label an unperformed engine check as passed.
