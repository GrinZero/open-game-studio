import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { StudioCore } from "../src/server/core.js";
import { dispatch } from "../src/server/operations.js";
import { createStandaloneCharacterRun } from "../src/server/character-workflow.js";
import {
  readWorkbenchTarget,
  resolveWorkbenchTarget,
  type WorkbenchTarget,
} from "../src/shared/workbench-target.js";

test("打开资源、固定版本、远处节点、工作流步骤和任务：工具与 URL 解析一致且只读", () => {
  const dir = mkdtempSync(path.join(os.tmpdir(), "studio-deep-link-"));
  const core = new StudioCore(path.join(dir, "registry"));
  try {
    const project = core.projectManage({
      action: "create",
      name: "海岛",
      root: path.join(dir, "game"),
    });
    const run = createStandaloneCharacterRun(core, { projectId: project.id });
    const canvas = core.snapshot(project.id).canvases[0];
    const v1 = core.importFiles(project.id, [
      { name: "plan.md", bytes: Buffer.from("v1") },
    ]);
    const v2 = core.importFiles(
      project.id,
      [{ name: "plan.md", bytes: Buffer.from("v2") }],
      { assetId: v1.assetId },
    );
    const nodeId = "far-away-note";
    const patched = core.canvasPatch({
      projectId: project.id,
      canvasId: canvas.id,
      expectedRevision: canvas.graphRevision,
      target: "graph",
      ops: [
        {
          op: "add",
          node: {
            id: nodeId,
            type: "note",
            position: { x: 30000, y: -30000 },
            data: { title: "远处节点", text: "keep" },
          },
        },
      ],
    });
    const usageCanvas = core.canvasManage({
      action: "create",
      projectId: project.id,
      name: "固定资源的实际画布",
    });
    core.canvasPatch({
      projectId: project.id,
      canvasId: usageCanvas.id,
      expectedRevision: 0,
      target: "graph",
      ops: [
        {
          op: "add",
          node: {
            id: "fixed-version-usage",
            type: "asset",
            position: { x: 50, y: 50 },
            data: {
              title: "旧版资源引用",
              assetId: v1.assetId,
              revisionId: v1.id,
            },
          },
        },
      ],
    });
    const task = core.generationCreate({
      projectId: project.id,
      canvasId: canvas.id,
      expectedRevision: patched.graphRevision,
      kind: "image",
      name: "未启动任务",
      prompt: "test",
      inputNodeIds: [],
      position: { x: 0, y: 0 },
      idempotencyKey: "link-task",
    });
    const before = core.snapshot(project.id);
    const cases: WorkbenchTarget[] = [
      { assetId: v1.assetId },
      { revisionId: v1.id },
      { assetId: v1.assetId, revisionId: v1.id },
      { nodeId },
      { runId: run.id },
      { runId: run.id, nodeKey: run.dag![0].key },
      { runId: run.id, stepKey: run.steps[3].key },
      { taskId: task.id },
      { taskId: task.id, nodeId: task.nodeId },
      ...["overview", "assets", "settings", "runs"].map(
        (view) => ({ view }) as WorkbenchTarget,
      ),
    ];
    for (const input of cases) {
      const opened = dispatch(core, "studio_open", {
        projectId: project.id,
        ...input,
      }) as any;
      assert.equal(opened.navigationStatus, "requested");
      assert.ok(opened.navigationId);
      const parsed = readWorkbenchTarget(
        Object.fromEntries(new URL(opened.url).searchParams),
      );
      assert.deepEqual(resolveWorkbenchTarget(before, parsed), opened.target);
    }
    const located = dispatch(core, "studio_open", {
      projectId: project.id,
      revisionId: v1.id,
    }) as any;
    assert.equal(located.canvasId, usageCanvas.id);
    assert.equal(located.nodeId, "fixed-version-usage");
    assert.equal(
      (
        dispatch(core, "studio_open", {
          projectId: project.id,
          assetId: v1.assetId,
        }) as any
      ).revisionId,
      v2.id,
    );
    for (const [input, pattern] of [
      [{ nodeId: "missing" }, /节点不存在/],
      [{ assetId: "missing" }, /资源不存在/],
      [{ assetId: v1.assetId, revisionId: "missing" }, /版本不属于/],
      [{ runId: "missing" }, /运行不存在/],
      [{ nodeKey: "missing" }, /需要 runId/],
      [{ runId: run.id, nodeKey: "missing" }, /工作流节点不存在/],
      [{ runId: run.id, stepKey: "missing" }, /工作流阶段不存在/],
      [{ runId: run.id, canvasId: "wrong" }, /不匹配/],
      [{ assetId: v1.assetId, nodeId }, /不匹配/],
      [{ assetId: v1.assetId, view: "runs" }, /不匹配/],
      [{ taskId: "missing" }, /任务不存在/],
    ] as [WorkbenchTarget, RegExp][])
      assert.throws(
        () =>
          dispatch(core, "studio_open", { projectId: project.id, ...input }),
        pattern,
      );
    assert.deepEqual(
      core.snapshot(project.id),
      before,
      "打开目标不能执行制作或改写画布",
    );
  } finally {
    core.close();
    rmSync(dir, { recursive: true, force: true });
  }
});
