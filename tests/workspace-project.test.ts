import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, symlinkSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { StudioCore } from "../src/server/core.js";
import { dispatch } from "../src/server/operations.js";
import { workspaceProject } from "../src/server/workspace-project.js";

test("工作区匹配最近的登记目录，排除前缀碰撞、归档、失效与多项目歧义", () => {
  const dir = mkdtempSync(path.join(os.tmpdir(), "studio-workspace-"));
  const core = new StudioCore(path.join(dir, "registry"));
  try {
    const parent = core.projectManage({
      action: "create",
      name: "主项目",
      root: path.join(dir, "game"),
    });
    const nested = core.projectManage({
      action: "create",
      name: "子项目",
      root: path.join(parent.root, "child"),
    });
    const sub = path.join(nested.root, "assets");
    mkdirSync(sub);
    const alias = path.join(dir, "alias");
    symlinkSync(nested.root, alias);
    mkdirSync(path.join(dir, "game-other"));
    const projects = core.listProjects();
    assert.equal(workspaceProject(projects, [sub]), nested.id);
    assert.equal(
      workspaceProject(projects, [pathToFileURL(alias).href]),
      nested.id,
    );
    assert.equal(
      workspaceProject(projects, [path.join(dir, "game-other")]),
      undefined,
    );
    assert.equal(
      workspaceProject(projects, [parent.root, nested.root]),
      undefined,
    );
    assert.equal(workspaceProject(projects, [nested.root, sub]), nested.id);
    assert.equal(
      workspaceProject(
        projects.map((p) => ({ ...p, archived: true })),
        [sub],
      ),
      undefined,
    );
    assert.equal(
      workspaceProject(
        projects.map((p) => ({ ...p, available: false })),
        [sub],
      ),
      undefined,
    );
    assert.equal(
      workspaceProject(projects, [
        "relative",
        "file://remote/game",
        path.join(dir, "missing"),
      ]),
      undefined,
    );
    const opened = dispatch(core, "studio_open", { workspaceRoot: sub }) as any;
    assert.equal(opened.project.project.id, nested.id);
    assert.equal(
      (
        dispatch(core, "studio_open", {
          projectId: parent.id,
          workspaceRoot: sub,
        }) as any
      ).project.project.id,
      parent.id,
    );
    assert.equal((dispatch(core, "studio_open", {}) as any).project, undefined);
  } finally {
    core.close();
    rmSync(dir, { recursive: true, force: true });
  }
});

test("studio_open 返回指定画布深链接并拒绝无效、跨项目和归档画布", () => {
  const dir = mkdtempSync(path.join(os.tmpdir(), "studio-canvas-launch-"));
  const core = new StudioCore(path.join(dir, "registry"));
  try {
    const project = core.projectManage({
      action: "create",
      name: "海岛",
      root: path.join(dir, "game"),
    });
    const canvas = core.canvasManage({
      action: "create",
      projectId: project.id,
      name: "海岛主城资源总览",
    });
    const opened = dispatch(core, "studio_open", {
      projectId: project.id,
      canvasId: canvas.id,
    }) as any;
    assert.equal(opened.canvasId, canvas.id);
    assert.equal(new URL(opened.url).searchParams.get("canvasId"), canvas.id);
    assert.equal(new URL(opened.url).searchParams.get("projectId"), project.id);
    assert.equal(
      (
        dispatch(core, "studio_open", {
          workspaceRoot: project.root,
          canvasId: canvas.id,
        }) as any
      ).canvasId,
      canvas.id,
    );
    assert.throws(
      () => dispatch(core, "studio_open", { canvasId: canvas.id }),
      /需要明确项目/,
    );
    assert.throws(
      () =>
        dispatch(core, "studio_open", {
          projectId: project.id,
          canvasId: "missing",
        }),
      /画布不存在/,
    );
    const other = core.projectManage({
      action: "create",
      name: "其他项目",
      root: path.join(dir, "other"),
    });
    assert.throws(
      () =>
        dispatch(core, "studio_open", {
          projectId: other.id,
          canvasId: canvas.id,
        }),
      /画布不存在/,
    );
    core.canvasManage({
      action: "update",
      projectId: project.id,
      canvasId: canvas.id,
      expectedRevision: canvas.graphRevision,
      archived: true,
    });
    assert.throws(
      () =>
        dispatch(core, "studio_open", {
          projectId: project.id,
          canvasId: canvas.id,
        }),
      /已归档/,
    );
  } finally {
    core.close();
    rmSync(dir, { recursive: true, force: true });
  }
});
