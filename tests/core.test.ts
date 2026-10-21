import { test } from "node:test";
import assert from "node:assert/strict";
import {
  mkdtempSync,
  rmSync,
  readFileSync,
  writeFileSync,
  symlinkSync,
  mkdirSync,
  renameSync,
  existsSync,
  readdirSync,
  realpathSync,
} from "node:fs";
import path from "node:path";
import os from "node:os";
import { StudioCore, StudioError } from "../src/server/core.js";
function setup() {
  const dir = mkdtempSync(path.join(os.tmpdir(), "studio-test-"));
  const core = new StudioCore(path.join(dir, "config"));
  const p = core.projectManage({
    action: "create",
    name: "星铸者",
    root: path.join(dir, "game"),
    engine: "godot",
  });
  return {
    dir,
    core,
    p,
    c: core.snapshot(p.id).canvases[0],
    clean: () => {
      core.close();
      rmSync(dir, { recursive: true, force: true });
    },
  };
}
const doc = (
  name = "brief.md",
  text = "# 角色目标\nGodot, +Y up, 参考轮廓待审核。",
) => ({ name, bytes: Buffer.from(text) });

test("新建项目直接使用已有目录，保留文件且重复创建不重置项目", () => {
  const x = setup();
  const root = path.join(x.dir, "star-maker");
  mkdirSync(root);
  writeFileSync(path.join(root, "project.godot"), "existing game");
  try {
    const project = x.core.projectManage({
      action: "create",
      name: "starforger",
      root,
    });
    assert.equal(project.root, realpathSync(root));
    assert.equal(
      readFileSync(path.join(root, "project.godot"), "utf8"),
      "existing game",
    );
    assert.deepEqual(readdirSync(root).sort(), [
      ".game-studio",
      "project.godot",
    ]);
    const snapshot = x.core.snapshot(project.id);
    const repeated = x.core.projectManage({
      action: "create",
      name: "renamed",
      root,
    });
    assert.equal(repeated.id, project.id);
    assert.equal(repeated.name, project.name);
    assert.deepEqual(x.core.snapshot(project.id).canvases, snapshot.canvases);
    assert.equal(existsSync(path.join(root, "starforger")), false);
  } finally {
    x.clean();
  }
});

test("项目路径为文件时明确拒绝并保留内容", () => {
  const x = setup();
  const root = path.join(x.dir, "file");
  writeFileSync(root, "keep me");
  try {
    for (const action of ["create", "link"])
      assert.throws(
        () => x.core.projectManage({ action, name: "game", root }),
        /项目路径已被文件占用/,
      );
    assert.equal(readFileSync(root, "utf8"), "keep me");
  } finally {
    x.clean();
  }
});
function glb() {
  const raw = Buffer.from(
    JSON.stringify({
      asset: { version: "2.0" },
      scene: 0,
      scenes: [{ nodes: [] }],
    }),
  );
  const padding = (4 - (raw.length % 4)) % 4;
  const body = Buffer.concat([raw, Buffer.alloc(padding, 32)]);
  const header = Buffer.alloc(20);
  header.write("glTF");
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(body.length + 20, 8);
  header.writeUInt32LE(body.length, 12);
  header.writeUInt32LE(0x4e4f534a, 16);
  return { name: "candidate.glb", bytes: Buffer.concat([header, body]) };
}
const png = () => ({
  name: "reference.png",
  bytes: Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aM2cAAAAASUVORK5CYII=",
    "base64",
  ),
});
function instantiate(x: ReturnType<typeof setup>) {
  const r = x.core.importFiles(x.p.id, [png()]);
  const c = x.core.canvasPatch({
    projectId: x.p.id,
    canvasId: x.c.id,
    target: "graph",
    expectedRevision: 0,
    ops: [
      {
        op: "instantiate",
        inputRevisionIds: [r.id],
        position: { x: 300, y: 0 },
      },
    ],
  });
  const plan = x.core.prepare({
    projectId: x.p.id,
    canvasId: c.id,
    instanceId: c.instances[0].id,
  });
  const run = x.core.start({
    projectId: x.p.id,
    planId: plan.id,
    expectedRevision: c.graphRevision,
    idempotencyKey: "start-key-0001",
  });
  return { r, c, plan, run };
}
test("项目封面绑定图片文件与固定版本，重启、更新和重定位后保留，可移除", () => {
  const x = setup();
  try {
    const image = x.core.importFiles(x.p.id, [doc(), png()]);
    const cover = { revisionId: image.id, fileIndex: 1 };
    x.core.projectManage({ action: "update", projectId: x.p.id, cover });
    x.core.importFiles(x.p.id, [png()], { assetId: image.assetId });
    x.core.projectManage({
      action: "update",
      projectId: x.p.id,
      name: "新名称",
    });
    const resumed = new StudioCore(path.join(x.dir, "config"));
    try {
      assert.deepEqual(resumed.listProjects()[0].cover, cover);
      assert.deepEqual(resumed.snapshot(x.p.id).project.cover, cover);
      const root = path.join(x.dir, "moved");
      renameSync(x.p.root, root);
      resumed.projectManage({ action: "relocate", projectId: x.p.id, root });
      assert.deepEqual(resumed.listProjects()[0].cover, cover);
      assert.deepEqual(
        readFileSync(resumed.assetFile(x.p.id, image.id, 1).file),
        png().bytes,
      );
      resumed.projectManage({
        action: "update",
        projectId: x.p.id,
        cover: null,
      });
      assert.equal(resumed.listProjects()[0].cover, undefined);
      assert.equal(resumed.snapshot(x.p.id).assets.length, 1);
    } finally {
      resumed.close();
    }
  } finally {
    x.clean();
  }
});
test("项目封面拒绝非图片、外部项目版本、非法索引和篡改文件，原封面保持", () => {
  const x = setup();
  try {
    const image = x.core.importFiles(x.p.id, [png()]);
    const text = x.core.importFiles(x.p.id, [doc()]);
    const other = x.core.projectManage({
      action: "create",
      name: "其他项目",
      root: path.join(x.dir, "other"),
    });
    const foreign = x.core.importFiles(other.id, [png()]);
    const cover = { revisionId: image.id, fileIndex: 0 };
    x.core.projectManage({ action: "update", projectId: x.p.id, cover });
    for (const invalid of [
      { revisionId: text.id, fileIndex: 0 },
      { revisionId: foreign.id, fileIndex: 0 },
      { revisionId: image.id, fileIndex: 1 },
      { revisionId: image.id, fileIndex: -1 },
      { revisionId: image.id, fileIndex: 0.5 },
    ]) {
      assert.throws(() =>
        x.core.projectManage({
          action: "update",
          projectId: x.p.id,
          cover: invalid,
        }),
      );
      assert.deepEqual(
        x.core.listProjects().find((p) => p.id === x.p.id)?.cover,
        cover,
      );
    }
    writeFileSync(x.core.assetFile(x.p.id, image.id, 0).file, "tampered");
    assert.throws(() =>
      x.core.projectManage({ action: "update", projectId: x.p.id, cover }),
    );
  } finally {
    x.clean();
  }
});
test("跨画布引用共享不可变版本，删除节点不删除资源", () => {
  const x = setup();
  try {
    const rev = x.core.importFiles(x.p.id, [doc()]);
    const c2 = x.core.canvasManage({
      projectId: x.p.id,
      action: "create",
      name: "场景规划",
    });
    for (const [i, c] of [x.c, c2].entries())
      x.core.canvasPatch({
        projectId: x.p.id,
        canvasId: c.id,
        target: "graph",
        expectedRevision: 0,
        ops: [
          {
            op: "add",
            node: {
              id: `n_${i}`,
              type: "asset",
              position: { x: 0, y: 0 },
              data: {
                title: "设计说明",
                assetId: rev.assetId,
                revisionId: rev.id,
              },
            },
          },
        ],
      });
    x.core.canvasPatch({
      projectId: x.p.id,
      canvasId: x.c.id,
      target: "graph",
      expectedRevision: 1,
      ops: [{ op: "remove", id: "n_0" }],
    });
    const s = x.core.snapshot(x.p.id);
    assert.equal(s.assets.length, 1);
    assert.equal(s.canvases[1].nodes[0].data.revisionId, rev.id);
    assert.equal(
      readFileSync(x.core.assetFile(x.p.id, rev.id, 0).file).toString(),
      doc().bytes.toString(),
    );
  } finally {
    x.clean();
  }
});
test("graph 与 layout 的独立 CAS 保留并发 Agent 导入与 UI 移动", () => {
  const x = setup();
  try {
    const c = x.core.canvasPatch({
      projectId: x.p.id,
      canvasId: x.c.id,
      target: "graph",
      expectedRevision: 0,
      ops: [
        {
          op: "add",
          node: {
            id: "note_1",
            type: "note",
            position: { x: 0, y: 0 },
            data: { title: "便签", text: "约束" },
          },
        },
      ],
    });
    const core2 = new StudioCore(path.join(x.dir, "config"));
    try {
      core2.importFiles(x.p.id, [doc("texture-notes.md")]);
      core2.canvasPatch({
        projectId: x.p.id,
        canvasId: c.id,
        target: "graph",
        expectedRevision: 1,
        ops: [{ op: "note", id: "note_1", text: "新的约束" }],
      });
      x.core.canvasPatch({
        projectId: x.p.id,
        canvasId: c.id,
        target: "layout",
        expectedRevision: 0,
        ops: [{ op: "move", id: "note_1", position: { x: 150, y: 240 } }],
      });
      const final = x.core.snapshot(x.p.id);
      assert.equal(final.canvases[0].nodes[0].data.text, "新的约束");
      assert.deepEqual(final.canvases[0].nodes[0].position, { x: 150, y: 240 });
      assert.equal(final.assets.length, 1);
      assert.throws(
        () =>
          x.core.canvasPatch({
            projectId: x.p.id,
            canvasId: c.id,
            target: "graph",
            expectedRevision: 1,
            ops: [],
          }),
        (e) => e instanceof StudioError && e.status === 409,
      );
    } finally {
      core2.close();
    }
  } finally {
    x.clean();
  }
});
test("重复启动返回同一 Run，服务重启恢复固定输入且不重新生成", () => {
  const x = setup();
  try {
    const { plan, run, r, c } = instantiate(x);
    const again = x.core.start({
      projectId: x.p.id,
      planId: plan.id,
      expectedRevision: 0,
      idempotencyKey: "start-key-0001",
    });
    assert.equal(again.id, run.id);
    const resumed = new StudioCore(path.join(x.dir, "config"));
    try {
      const s = resumed.snapshot(x.p.id);
      assert.equal(s.runs.length, 1);
      assert.deepEqual(s.runs[0].inputRevisionIds, [r.id]);
      assert.equal(s.runs[0].steps[0].status, "awaiting_agent");
      resumed.canvasPatch({
        projectId: x.p.id,
        canvasId: c.id,
        target: "layout",
        expectedRevision: 0,
        ops: [{ op: "move", id: c.nodes[0].id, position: { x: 888, y: 0 } }],
      });
      assert.deepEqual(resumed.snapshot(x.p.id).runs[0].inputRevisionIds, [
        r.id,
      ]);
      assert.equal(s.runs[0].cost.estimate, null);
    } finally {
      resumed.close();
    }
  } finally {
    x.clean();
  }
});
test("预览不产生审核；审核需绑定当前候选；返修保存旧产物", () => {
  const x = setup();
  try {
    let { run } = instantiate(x);
    for (const [key, files] of [
      ["brief", [doc()]],
      ["reference", [png()]],
      ["candidate", [glb()]],
    ] as const) {
      const r = x.core.importFiles(x.p.id, [...files]);
      run = x.core.submit({
        projectId: x.p.id,
        runId: run.id,
        stepKey: key,
        revisionIds: [r.id],
        expectedRevision: run.revision,
      });
    }
    const candidate = run.steps[3].inputRevisionIds[0];
    x.core.assetFile(x.p.id, candidate, 0);
    assert.equal(x.core.snapshot(x.p.id).reviews.length, 0);
    assert.throws(
      () =>
        x.core.decide({
          projectId: x.p.id,
          runId: run.id,
          stepKey: "appearance",
          revisionId: candidate,
          decision: "accept",
          expectedRevision: 0,
        }),
      (e) => e instanceof StudioError && e.status === 409,
    );
    run = x.core.decide({
      projectId: x.p.id,
      runId: run.id,
      stepKey: "appearance",
      revisionId: candidate,
      decision: "request_changes",
      comment: "修复左肩结构",
      expectedRevision: run.revision,
    });
    assert.equal(run.steps[2].status, "awaiting_manual");
    assert.deepEqual(run.steps[2].outputRevisionIds, [candidate]);
    assert.equal(x.core.snapshot(x.p.id).reviews[0].comment, "修复左肩结构");
  } finally {
    x.clean();
  }
});
test("完整人工交接导出，版本 hash 保持，引擎保持 not_verified", () => {
  const x = setup();
  try {
    let { run } = instantiate(x);
    let rigRev = "";
    for (const [key, files] of [
      ["brief", [doc()]],
      ["reference", [png()]],
      ["candidate", [glb()]],
    ] as const) {
      const r = x.core.importFiles(x.p.id, [...files]);
      run = x.core.submit({
        projectId: x.p.id,
        runId: run.id,
        stepKey: key,
        revisionIds: [r.id],
        expectedRevision: run.revision,
      });
    }
    run = x.core.decide({
      projectId: x.p.id,
      runId: run.id,
      stepKey: "appearance",
      revisionId: run.steps[3].inputRevisionIds[0],
      decision: "accept",
      expectedRevision: run.revision,
    });
    for (const key of ["assembly", "rig"]) {
      const r = x.core.importFiles(x.p.id, [
        glb(),
        {
          name: "source.blend",
          bytes: Buffer.from(
            "BLENDER-v300synthetic-test-fixture-not-editability-proof",
          ),
        },
      ]);
      run = x.core.submit({
        projectId: x.p.id,
        runId: run.id,
        stepKey: key,
        revisionIds: [r.id],
        expectedRevision: run.revision,
      });
      rigRev = r.id;
    }
    run = x.core.decide({
      projectId: x.p.id,
      runId: run.id,
      stepKey: "deformation",
      revisionId: rigRev,
      decision: "accept",
      comment: "测试记录：人工审核契约，不代表实际模型质量",
      expectedRevision: run.revision,
    });
    const bundle = x.core.exportBundle({
      projectId: x.p.id,
      runId: run.id,
      revisionIds: [rigRev],
      expectedRevision: run.revision,
      idempotencyKey: "export-key-001",
    });
    assert.equal(bundle.validation.engine, "not_verified");
    assert.equal(x.core.snapshot(x.p.id).runs[0].status, "succeeded");
    assert.equal(
      JSON.parse(readFileSync(path.join(bundle.path, "manifest.json"), "utf8"))
        .assets[0].id,
      rigRev,
    );
    const repeated = x.core.exportBundle({
      projectId: x.p.id,
      runId: run.id,
      revisionIds: [rigRev],
      expectedRevision: run.revision,
      idempotencyKey: "export-key-001",
    });
    assert.equal(repeated.id, bundle.id);
  } finally {
    x.clean();
  }
});
test("局部重跑只使当前阶段与下游过期，前置成果与历史不变", () => {
  const x = setup();
  try {
    let { run } = instantiate(x);
    const brief = x.core.importFiles(x.p.id, [doc()]);
    run = x.core.submit({
      projectId: x.p.id,
      runId: run.id,
      stepKey: "brief",
      revisionIds: [brief.id],
      expectedRevision: 0,
    });
    const ref = x.core.importFiles(x.p.id, [png()]);
    run = x.core.submit({
      projectId: x.p.id,
      runId: run.id,
      stepKey: "reference",
      revisionIds: [ref.id],
      expectedRevision: 1,
    });
    const fresh = x.core.rerun({
      projectId: x.p.id,
      runId: run.id,
      fromStep: "candidate",
      expectedRevision: 2,
      idempotencyKey: "rerun-key-001",
    });
    assert.equal(fresh.steps[0].stale, false);
    assert.deepEqual(fresh.steps[0].outputRevisionIds, [brief.id]);
    assert.equal(fresh.steps[2].status, "awaiting_manual");
    assert.equal(fresh.steps[3].stale, true);
    const s = x.core.snapshot(x.p.id);
    assert.equal(s.runs.length, 2);
    assert.equal(s.runs[0].status, "cancelled");
    assert.deepEqual(s.runs[0].steps[1].outputRevisionIds, [ref.id]);
    const model = x.core.importFiles(x.p.id, [glb()]);
    const continued = x.core.submit({
      projectId: x.p.id,
      runId: fresh.id,
      stepKey: "candidate",
      revisionIds: [model.id],
      expectedRevision: fresh.revision,
    });
    const reviewed = x.core.decide({
      projectId: x.p.id,
      runId: fresh.id,
      stepKey: "appearance",
      revisionId: model.id,
      decision: "accept",
      expectedRevision: continued.revision,
    });
    assert.equal(reviewed.steps[3].stale, false);
    assert.equal(reviewed.steps[4].status, "awaiting_manual");
  } finally {
    x.clean();
  }
});
test("无效 GLB、路径穿越与符号链接逃逸被拒绝", () => {
  const x = setup();
  try {
    assert.throws(
      () =>
        x.core.importFiles(x.p.id, [
          { name: "bad.glb", bytes: Buffer.from("not a glb") },
        ]),
      /GLB/,
    );
    writeFileSync(path.join(x.dir, "outside.md"), "外部文件");
    symlinkSync(
      path.join(x.dir, "outside.md"),
      path.join(x.p.root, "escape.md"),
    );
    assert.throws(
      () => x.core.importPaths(x.p.id, ["escape.md"]),
      (e) => e instanceof StudioError && e.status === 403,
    );
    assert.throws(
      () => x.core.importPaths(x.p.id, ["../outside.md"]),
      (e) => e instanceof StudioError && e.status === 403,
    );
    assert.equal(x.core.snapshot(x.p.id).assets.length, 0);
  } finally {
    x.clean();
  }
});
test("外部合法修改自动刷新哈希并允许导出，损坏文件拒绝导出", () => {
  const x = setup();
  try {
    const r = x.core.importFiles(x.p.id, [doc()]);
    const file = x.core.assetFile(x.p.id, r.id, 0).file;
    writeFileSync(file, "corrupted\0binary");
    assert.throws(
      () =>
        x.core.exportBundle({
          projectId: x.p.id,
          revisionIds: [r.id],
          idempotencyKey: "export-tamper-key",
        }),
      /文档包含二进制内容/,
    );
    assert.equal(x.core.snapshot(x.p.id).exports.length, 0);

    writeFileSync(file, "# 外部编辑器更新后的文档\n保持零复制同步");
    const bundle = x.core.exportBundle({
      projectId: x.p.id,
      revisionIds: [r.id],
      idempotencyKey: "export-updated-key",
    });
    assert.equal(bundle.validation.technical, "passed");
    assert.equal(
      x.core.snapshot(x.p.id).assets[0].revisions[0].files[0].sizeBytes,
      Buffer.byteLength("# 外部编辑器更新后的文档\n保持零复制同步"),
    );
  } finally {
    x.clean();
  }
});
test("移动目录重新登记时身份和资源版本不变", () => {
  const x = setup();
  try {
    const r = x.core.importFiles(x.p.id, [doc()]);
    const next = path.join(x.dir, "moved");
    renameSync(x.p.root, next);
    assert.equal(x.core.listProjects()[0].available, false);
    const p = x.core.projectManage({
      action: "relocate",
      projectId: x.p.id,
      root: next,
    });
    assert.equal(p.id, x.p.id);
    assert.equal(x.core.snapshot(p.id).assets[0].revisions[0].id, r.id);
  } finally {
    x.clean();
  }
});
test("项目内文件直接路径引用零复制，上传相同文件自动去重，随仓库克隆自动恢复", () => {
  const x = setup();
  try {
    const spritesDir = path.join(x.p.root, "sprites");
    mkdirSync(spritesDir, { recursive: true });
    const heroPath = path.join(spritesDir, "hero.png");
    writeFileSync(heroPath, png().bytes);

    const rev1 = x.core.importPaths(x.p.id, ["sprites/hero.png"], {
      name: "主角立绘",
    });
    assert.equal(rev1.files[0].relativePath, "sprites/hero.png");
    assert.equal(existsSync(path.join(x.p.root, ".game-studio", "assets")), false);

    const rev2 = x.core.importFiles(
      x.p.id,
      [{ name: "hero.png", bytes: png().bytes }],
      { name: "重复上传立绘" },
    );
    assert.equal(rev2.files[0].relativePath, "sprites/hero.png");
    assert.equal(existsSync(path.join(x.p.root, ".game-studio", "assets")), false);

    x.core.projectManage({
      action: "update",
      projectId: x.p.id,
      name: "可移植仓库项目",
      cover: { revisionId: rev1.id, fileIndex: 0 },
    });

    const clonedRoot = path.join(x.dir, "cloned-repo");
    mkdirSync(path.join(clonedRoot, "sprites"), { recursive: true });
    mkdirSync(path.join(clonedRoot, ".game-studio"), { recursive: true });
    writeFileSync(
      path.join(clonedRoot, "sprites", "hero.png"),
      png().bytes,
    );
    writeFileSync(
      path.join(clonedRoot, ".game-studio", "state.json"),
      readFileSync(path.join(x.p.root, ".game-studio", "state.json")),
    );

    const peerCore = new StudioCore(path.join(x.dir, "peer-config"));
    try {
      peerCore.discoverWorkspaces([clonedRoot]);
      const projects = peerCore.listProjects();
      assert.equal(projects.length, 1);
      assert.equal(projects[0].id, x.p.id);
      assert.equal(projects[0].name, "可移植仓库项目");
      assert.deepEqual(projects[0].cover, { revisionId: rev1.id, fileIndex: 0 });
      const snap = peerCore.snapshot(x.p.id);
      assert.equal(snap.assets.length, 2);
      assert.equal(
        peerCore.assetFile(x.p.id, rev1.id, 0).file,
        path.join(realpathSync(clonedRoot), "sprites", "hero.png"),
      );
    } finally {
      peerCore.close();
    }
  } finally {
    x.clean();
  }
});
