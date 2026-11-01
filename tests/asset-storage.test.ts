import { test } from "node:test";
import assert from "node:assert/strict";
import {
  mkdtempSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
  rmSync,
  symlinkSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";
import { StudioCore } from "../src/server/core.js";
import { projectChangeVersion } from "../src/server/codex-collaboration.js";

function setup() {
  const dir = mkdtempSync(path.join(os.tmpdir(), "studio-storage-"));
  const core = new StudioCore(path.join(dir, "config"));
  const project = core.projectManage({
    action: "create",
    name: "历史迁移",
    root: path.join(dir, "project"),
  });
  const first = core.importFiles(project.id, [
    { name: "old.md", bytes: Buffer.from("# 历史版本") },
  ]);
  const second = core.importFiles(
    project.id,
    [{ name: "new.md", bytes: Buffer.from("# 当前版本") }],
    { assetId: first.assetId },
  );
  const entries = [first, second].map((revision) => {
    const file = revision.files[0];
    const to = `local-archive/studio_assets/${revision.id}/${file.name}`;
    mkdirSync(path.dirname(path.join(project.root, to)), { recursive: true });
    writeFileSync(
      path.join(project.root, to),
      readFileSync(path.join(project.root, file.relativePath)),
    );
    return {
      revisionId: revision.id,
      fileIndex: 0,
      from: file.relativePath,
      to,
      sha256: file.sha256,
      sizeBytes: file.sizeBytes,
    };
  });
  return {
    dir,
    core,
    project,
    first,
    second,
    entries,
    clean() {
      core.close();
      rmSync(dir, { recursive: true, force: true });
    },
  };
}

test("storage relocation preserves versions and provenance, updates frozen inputs and recovery state, and is repeatable", () => {
  const x = setup();
  try {
    const canvas = x.core.snapshot(x.project.id).canvases[0];
    x.core.canvasPatch({
      projectId: x.project.id,
      canvasId: canvas.id,
      target: "graph",
      expectedRevision: canvas.graphRevision,
      ops: [
        {
          op: "add",
          node: {
            id: "old-version",
            type: "asset",
            position: { x: 0, y: 0 },
            data: {
              title: "历史",
              assetId: x.first.assetId,
              revisionId: x.first.id,
            },
          },
        },
      ],
    });
    const task = x.core.generationCreate({
      projectId: x.project.id,
      canvasId: canvas.id,
      expectedRevision: x.core.snapshot(x.project.id).canvases[0].graphRevision,
      kind: "image",
      name: "历史输入",
      prompt: "使用旧版本",
      inputNodeIds: ["old-version"],
      position: { x: 300, y: 0 },
      idempotencyKey: "storage-relocation-task",
    });
    const before = x.core.snapshot(x.project.id);
    const beforeVersion = projectChangeVersion(x.core, x.project.id).version;
    const result = x.core.relocateAssetFiles(x.project.id, x.entries);
    assert.equal(result.changedFiles, 2);
    assert.equal(result.changedRecords.revisions, 2);
    assert.equal(result.changedRecords.tasks, 1);
    assert.notEqual(
      projectChangeVersion(x.core, x.project.id).version,
      beforeVersion,
    );
    const after = x.core.snapshot(x.project.id);
    assert.deepEqual(
      after.assets.map((asset) => ({ ...asset, revisions: undefined })),
      before.assets.map((asset) => ({ ...asset, revisions: undefined })),
    );
    assert.deepEqual(after.canvases, before.canvases);
    for (const original of [x.first, x.second]) {
      const migrated = after.assets[0].revisions.find(
        (revision) => revision.id === original.id,
      )!;
      assert.deepEqual(
        { ...migrated, files: original.files },
        JSON.parse(JSON.stringify(original)),
      );
      assert.equal(migrated.files[0].sha256, original.files[0].sha256);
      assert.equal(
        migrated.files[0].relativePath,
        x.entries.find((entry) => entry.revisionId === original.id)!.to,
      );
      assert.equal(
        readFileSync(
          x.core.assetFile(x.project.id, original.id, 0).file,
          "utf8",
        ),
        original.id === x.first.id ? "# 历史版本" : "# 当前版本",
      );
    }
    const inputFile = after.tasks!.find((item) => item.id === task.id)!
      .inputSnapshot.items[0].files![0];
    assert.equal(inputFile.relativePath, x.entries[0].to);
    assert.equal(inputFile.sha256, x.first.files[0].sha256);
    const state = JSON.parse(
      readFileSync(
        path.join(x.project.root, ".game-studio/state.json"),
        "utf8",
      ),
    );
    assert.deepEqual(
      state.revisions.map((r: any) => r.files[0].relativePath).sort(),
      x.entries.map((entry) => entry.to).sort(),
    );
    for (const entry of x.entries)
      rmSync(path.join(x.project.root, entry.from));
    assert.equal(
      x.core.relocateAssetFiles(x.project.id, x.entries).changedFiles,
      0,
    );
  } finally {
    x.clean();
  }
});

test("a mismatched archive or stale plan rejects the entire relocation without changing history", () => {
  const x = setup();
  try {
    const before = x.core.snapshot(x.project.id);
    writeFileSync(path.join(x.project.root, x.entries[1].to), "wrong content");
    assert.throws(
      () => x.core.relocateAssetFiles(x.project.id, x.entries),
      /历史文件内容/,
    );
    assert.deepEqual(x.core.snapshot(x.project.id), before);
    const stale = x.entries.map((entry) => ({ ...entry }));
    stale[0].sha256 = "0".repeat(64);
    assert.throws(
      () => x.core.relocateAssetFiles(x.project.id, stale),
      /资源记录已变化/,
    );
    assert.deepEqual(x.core.snapshot(x.project.id), before);
  } finally {
    x.clean();
  }
});

test("archive references cannot traverse or follow a symlink outside the project", () => {
  const x = setup();
  try {
    const bytes = readFileSync(path.join(x.project.root, x.entries[0].from));
    const outside = path.join(x.dir, "outside.md");
    writeFileSync(outside, bytes);
    const link = path.join(x.project.root, "local-archive/escape.md");
    symlinkSync(outside, link);
    assert.throws(
      () =>
        x.core.relocateAssetFiles(x.project.id, [
          { ...x.entries[0], to: "local-archive/escape.md" },
        ]),
      /项目目录之外/,
    );
    assert.throws(
      () =>
        x.core.relocateAssetFiles(x.project.id, [
          { ...x.entries[0], to: "local-archive/../../outside.md" },
        ]),
      /规范的项目相对路径/,
    );
    assert.equal(
      x.core
        .snapshot(x.project.id)
        .assets[0].revisions.find((revision) => revision.id === x.first.id)!
        .files[0].relativePath,
      x.entries[0].from,
    );
  } finally {
    x.clean();
  }
});
