import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { StudioCore } from "../src/server/core";
import { createStandaloneCharacterRun } from "../src/server/character-workflow";
import {
  prepareCharacterSession,
  openCharacterSession,
  projectChangeVersion,
} from "../src/server/codex-collaboration";
import { characterWorkbenchUrl } from "../src/shared/codex-collaboration";

function fixture() {
  const dir = mkdtempSync(path.join(os.tmpdir(), "studio-codex-session-"));
  const core = new StudioCore(path.join(dir, "registry"));
  const project = core.projectManage({
    action: "create",
    name: "角色协作",
    root: path.join(dir, "project"),
  });
  const source = core.importFiles(project.id, [
    {
      name: "source.png",
      bytes: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aM2cAAAAASUVORK5CYII=",
        "base64",
      ),
    },
  ]);
  const run = createStandaloneCharacterRun(core, {
    projectId: project.id,
    sourceImageRevisionId: source.id,
  });
  const input = {
    runId: run.id,
    nodeKey: "source_illustration",
    slotKey: "original",
    expectedRevision: run.revision,
    referenceRevisionIds: [source.id],
    prompt: "换成蓝色外套",
  };
  return {
    dir,
    core,
    project,
    source,
    run,
    input,
    clean() {
      core.close();
      rmSync(dir, { recursive: true, force: true });
    },
  };
}

test("打开 Codex 新对话同时携带项目、固定参考、目标槽位和右侧角色工作台", async () => {
  const x = fixture();
  try {
    const session = prepareCharacterSession(
      x.core,
      x.project.id,
      x.input,
      4321,
    );
    const url = new URL(session.url);
    assert.equal(url.protocol, "codex:");
    assert.equal(url.hostname, "new");
    assert.equal(url.searchParams.get("path"), x.project.root);
    assert.equal(url.searchParams.get("browserUrl"), session.workbenchUrl);
    assert.ok(url.searchParams.get("prompt")!.includes(session.contextFile));
    assert.equal(
      new URL(session.workbenchUrl).searchParams.get("runId"),
      x.run.id,
    );
    assert.equal(
      new URL(session.workbenchUrl).searchParams.get("view"),
      "character",
    );
    assert.equal(new URL(session.workbenchUrl).port, "4321");
    const task = readFileSync(session.contextFile, "utf8");
    for (const value of [
      x.source.id,
      '"slotKey":"original"',
      "换成蓝色外套",
      "真实产物",
      "edit-view",
    ])
      assert.ok(task.includes(value));
    let opened = "";
    const result = await openCharacterSession(
      x.core,
      x.project.id,
      x.input,
      async (link) => {
        opened = link;
      },
    );
    assert.equal(result.status, "open_requested");
    assert.equal(opened, result.url);
    assert.equal(x.core.snapshot(x.project.id).runs.length, 1);
  } finally {
    x.clean();
  }
});

test("会话入口拒绝过期角色、不匹配槽位与不存在的参考，系统拒绝打开时明确报错", async () => {
  const x = fixture();
  try {
    assert.throws(
      () =>
        prepareCharacterSession(x.core, x.project.id, {
          ...x.input,
          expectedRevision: x.run.revision + 1,
        }),
      /进度已更新/,
    );
    assert.throws(
      () =>
        prepareCharacterSession(x.core, x.project.id, {
          ...x.input,
          slotKey: "head_front",
        }),
      /不匹配/,
    );
    assert.throws(
      () =>
        prepareCharacterSession(x.core, x.project.id, {
          ...x.input,
          referenceRevisionIds: ["missing"],
        }),
      /参考版本/,
    );
    await assert.rejects(
      openCharacterSession(x.core, x.project.id, x.input, async () => {
        throw new Error("launch rejected");
      }),
      /launch rejected/,
    );
  } finally {
    x.clean();
  }
});

test("协作版本能检测另一个服务进程的项目写入", () => {
  const x = fixture();
  const other = new StudioCore(path.join(x.dir, "registry"));
  try {
    const before = projectChangeVersion(x.core, x.project.id).version;
    other.assetManage({
      projectId: x.project.id,
      assetId: x.source.assetId,
      expectedRevision: other
        .snapshot(x.project.id)
        .assets.find((asset) => asset.id === x.source.assetId)!.revision,
      name: "新原画名称",
    });
    assert.notEqual(projectChangeVersion(x.core, x.project.id).version, before);
    assert.equal(
      characterWorkbenchUrl(
        x.project.id,
        x.run.id,
        "head_multiview",
      ).startsWith("http://127.0.0.1:4310/"),
      true,
    );
  } finally {
    other.close();
    x.clean();
  }
});
