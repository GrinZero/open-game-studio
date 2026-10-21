import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync, readFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { StudioCore } from "../src/server/core.js";
import { dispatch } from "../src/server/operations.js";
import type { Canvas } from "../src/shared/types.js";

function setup() {
  const dir = mkdtempSync(path.join(os.tmpdir(), "studio-inputs-"));
  const home = path.join(dir, "registry");
  let core = new StudioCore(home);
  const p = core.projectManage({
    action: "create",
    name: "连线输入验收",
    root: path.join(dir, "game"),
  });
  let c: Canvas = core.snapshot(p.id).canvases[0];
  const doc = core.importFiles(p.id, [
    {
      name: "design.md",
      bytes: Buffer.from("# 角色设定\n蓝色披风，保持轮廓。"),
    },
  ]);
  const image = core.importFiles(p.id, [
    { name: "reference.png", bytes: readFileSync("samples/reference.png") },
  ]);
  const patch = (ops: unknown[], target = "graph") =>
    (c = core.canvasPatch({
      projectId: p.id,
      canvasId: c.id,
      target,
      expectedRevision: target === "graph" ? c.graphRevision : c.layoutRevision,
      ops,
    }));
  patch([
    {
      op: "add",
      node: {
        id: "doc",
        type: "asset",
        position: { x: 0, y: 0 },
        data: { title: "角色设定", assetId: doc.assetId, revisionId: doc.id },
      },
    },
    {
      op: "add",
      node: {
        id: "image",
        type: "asset",
        position: { x: 250, y: 0 },
        data: {
          title: "角色参考图",
          assetId: image.assetId,
          revisionId: image.id,
        },
      },
    },
    {
      op: "add",
      node: {
        id: "note",
        type: "note",
        position: { x: 0, y: 200 },
        data: { title: "造型要求", text: "保留兜帽和灯笼。" },
      },
    },
    {
      op: "connect",
      edge: {
        id: "doc-note",
        source: "doc",
        target: "note",
        kind: "reference",
      },
    },
    {
      op: "connect",
      edge: {
        id: "note-image",
        source: "note",
        target: "image",
        kind: "reference",
      },
    },
    { op: "instantiate", inputNodeIds: ["image"], position: { x: 500, y: 0 } },
  ]);
  const wf = c.nodes.find((n) => n.type === "workflow")!;
  const prepare = () =>
    core.prepare({
      projectId: p.id,
      canvasId: c.id,
      instanceId: wf.data.instanceId,
    });
  const start = (plan: ReturnType<typeof prepare>, key = "start-inputs-test") =>
    core.start({
      projectId: p.id,
      planId: plan.id,
      expectedRevision: plan.graphRevision,
      idempotencyKey: key,
    });
  return {
    p,
    doc,
    image,
    wf,
    patch,
    prepare,
    start,
    get c() {
      return c;
    },
    get core() {
      return core;
    },
    restart() {
      core.close();
      core = new StudioCore(home);
    },
    clean() {
      core.close();
      rmSync(dir, { recursive: true, force: true });
    },
  };
}

test("连线递归传递文档、便签和固定图片版本，跨路径去重并提供实际执行文件", () => {
  const x = setup();
  try {
    x.patch([
      {
        op: "connect",
        edge: {
          id: "doc-wf",
          source: "doc",
          target: x.wf.id,
          kind: "reference",
        },
      },
    ]);
    const plan = x.prepare();
    assert.deepEqual(plan.inputRevisionIds, [x.doc.id, x.image.id]);
    assert.deepEqual(
      plan.inputSnapshot.items.map((i) => i.title),
      ["角色设定", "造型要求", "角色参考图"],
    );
    assert.match(plan.inputSnapshot.prompt, /蓝色披风/);
    assert.match(plan.inputSnapshot.prompt, /保留兜帽和灯笼/);
    const context = x.core.context(x.p.id, x.c.id, ["image"]);
    assert.deepEqual(
      context.revisions.map((r) => r.id).sort(),
      [x.doc.id, x.image.id].sort(),
    );
    assert.ok(
      context.revisions.every((r) => path.isAbsolute(r.files[0].fileRef)),
    );
    const run = x.start(plan);
    const response = dispatch(x.core, "workflow_get", {
      projectId: x.p.id,
      runId: run.id,
    }) as any;
    assert.deepEqual(
      response.executionInputs.referenceFiles
        .map((f: any) => f.revisionId)
        .sort(),
      [x.doc.id, x.image.id].sort(),
    );
    assert.equal(response.executionInputs.prompt, plan.inputSnapshot.prompt);
  } finally {
    x.clean();
  }
});

test("断线与便签修改改变下一次输入，已启动运行、重跑和重启保留冻结快照", () => {
  const x = setup();
  try {
    const plan = x.prepare();
    const run = x.start(plan);
    const newer = x.core.importFiles(
      x.p.id,
      [{ name: "reference.png", bytes: readFileSync("samples/reference.png") }],
      { assetId: x.image.assetId },
    );
    assert.notEqual(newer.id, x.image.id);
    assert.deepEqual(x.prepare().inputRevisionIds, [x.doc.id, x.image.id]);
    x.patch([
      { op: "disconnect", id: "doc-note" },
      { op: "note", id: "note", text: "改成红色披风。" },
    ]);
    const next = x.prepare();
    assert.deepEqual(next.inputRevisionIds, [x.image.id]);
    assert.match(next.inputSnapshot.prompt, /红色披风/);
    assert.doesNotMatch(next.inputSnapshot.prompt, /蓝色披风/);
    assert.deepEqual(
      x.core.snapshot(x.p.id).runs[0].inputSnapshot,
      plan.inputSnapshot,
    );
    const rerun = x.core.rerun({
      projectId: x.p.id,
      runId: run.id,
      fromStep: "brief",
      expectedRevision: run.revision,
      inputRevisionIds: [newer.id],
      idempotencyKey: "rerun-inputs-test",
    });
    assert.deepEqual(rerun.inputSnapshot?.revisionIds, [newer.id]);
    assert.match(rerun.inputSnapshot!.prompt, /保留兜帽和灯笼/);
    assert.doesNotMatch(rerun.inputSnapshot!.prompt, /红色披风/);
    x.restart();
    assert.deepEqual(
      x.core.snapshot(x.p.id).runs.find((r) => r.id === run.id)?.inputSnapshot,
      plan.inputSnapshot,
    );
    assert.deepEqual(
      x.core.snapshot(x.p.id).runs.find((r) => r.id === rerun.id)
        ?.inputSnapshot,
      rerun.inputSnapshot,
    );
  } finally {
    x.clean();
  }
});

test("确认期间图修改拒绝过期计划；布局移动不影响输入或幂等启动", () => {
  const x = setup();
  try {
    const plan = x.prepare();
    x.patch([{ op: "note", id: "note", text: "新的要求" }]);
    assert.throws(() => x.start(plan), /版本冲突/);
    assert.equal(x.core.snapshot(x.p.id).runs.length, 0);
    const fresh = x.prepare();
    x.patch(
      [{ op: "move", id: "image", position: { x: 280, y: 10 } }],
      "layout",
    );
    const run = x.start(fresh);
    assert.equal(x.start(fresh).id, run.id);
    assert.deepEqual(run.inputSnapshot, fresh.inputSnapshot);
  } finally {
    x.clean();
  }
});

test("拦截循环、重复端点与未指定产出的工作流连接，事务不留下部分修改", () => {
  const x = setup();
  try {
    const before = x.c;
    for (const [source, target, pattern] of [
      ["image", "doc", /循环/],
      ["doc", "note", /已经连接/],
      [x.wf.id, "doc", /产出/],
      ["missing", "doc", /端点无效/],
    ] as const)
      assert.throws(
        () =>
          x.patch([
            { op: "note", id: "note", text: "不应保存" },
            {
              op: "connect",
              edge: { id: `bad-${source}`, source, target, kind: "reference" },
            },
          ]),
        pattern,
      );
    assert.deepEqual(x.core.snapshot(x.p.id).canvases[0], before);
    const edge = x.c.edges.find((e) => e.target === x.wf.id)!;
    x.patch([{ op: "disconnect", id: edge.id }]);
    assert.throws(() => x.prepare(), /没有输入/);
    x.patch([
      {
        op: "connect",
        edge: {
          id: "note-wf",
          source: "note",
          target: x.wf.id,
          kind: "reference",
        },
      },
    ]);
    x.patch([{ op: "disconnect", id: "doc-note" }]);
    const noteOnly = x.prepare();
    assert.deepEqual(noteOnly.inputRevisionIds, []);
    assert.match(x.start(noteOnly).inputSnapshot!.prompt, /保留兜帽/);
  } finally {
    x.clean();
  }
});

test("启动前自动跟随合法外部修改，损坏文件拒绝创建运行", () => {
  const x = setup();
  try {
    const plan = x.prepare();
    const filename = x.core.assetFile(x.p.id, x.doc.id, 0).file;
    writeFileSync(filename, "corrupted\0binary");
    assert.throws(() => x.start(plan), /文档包含二进制内容/);
    assert.throws(() => x.prepare(), /文档包含二进制内容/);
    assert.equal(x.core.snapshot(x.p.id).runs.length, 0);

    writeFileSync(filename, "# 设定更新\n自动同步最新文本内容");
    const run = x.start(plan);
    assert.match(run.inputSnapshot!.prompt, /自动同步最新文本内容/);
  } finally {
    x.clean();
  }
});

test("版本式创建自动连到已有节点，删除连线解除绑定；无节点的显式输入仍兼容", () => {
  const x = setup();
  try {
    x.patch([
      {
        op: "instantiate",
        inputRevisionIds: [x.image.id],
        position: { x: 800, y: 0 },
      },
    ]);
    const wf = x.c.nodes.filter((n) => n.type === "workflow").at(-1)!;
    const instance = x.c.instances.find((i) => i.id === wf.data.instanceId)!;
    assert.deepEqual(instance.inputRevisionIds, []);
    const edge = x.c.edges.find((e) => e.target === wf.id)!;
    assert.equal(edge.source, "image");
    x.patch([{ op: "disconnect", id: edge.id }]);
    assert.throws(
      () =>
        x.core.prepare({
          projectId: x.p.id,
          canvasId: x.c.id,
          instanceId: instance.id,
        }),
      /没有输入/,
    );
    const r = x.core.importFiles(x.p.id, [
      { name: "legacy.md", bytes: Buffer.from("旧版固定输入") },
    ]);
    x.patch([
      {
        op: "instantiate",
        inputRevisionIds: [r.id],
        position: { x: 1000, y: 0 },
      },
    ]);
    const legacy = x.c.instances.at(-1)!;
    assert.deepEqual(
      x.core.prepare({
        projectId: x.p.id,
        canvasId: x.c.id,
        instanceId: legacy.id,
      }).inputRevisionIds,
      [r.id],
    );
  } finally {
    x.clean();
  }
});
