---
name: asset-curation
description: Import, find, name and version Open Game Studio resources with explicit provenance.
---

Inputs: explicit projectId, authorized files and optional target assetId. Inspect asset_search and asset_inspect to distinguish a new logical resource from a new revision.

asset_import reads only files within the registered project after realpath boundary checking. For files elsewhere, use the browser upload interface or copy the explicitly authorized files into the project before importing. No remote path string implies upload or local access.

Submit related GLB and Blend files together as one version file set. Independent candidates get separate imports. Record sourceRef through Core import options when available. Old version files and hashes remain unchanged. Importing an in-progress draft or candidate into `.game-studio` for workbench preview does not promote it to `art_source/` or authorize registering it in project VCS/LFS/DVC or formal source/deployment manifests; keep unapproved working files in `scratch/work/` so obsolete drafts can be overwritten or deleted freely without breaking repository checks.

Naming, tags and reversible archive use asset_manage with expectedRevision. After import read asset_inspect and verify files, source and immutable version. File header and hash checks do not prove geometry quality, Blend editability or engine compatibility.
