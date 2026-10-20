import { verifyNodeSizing } from "./node-sizing-smoke.mjs";
import { verifyCanvasTools } from "./canvas-tools-smoke.mjs";
import { verifyResourceComposer } from "./resource-composer-smoke.mjs";
import { verifyCanvasDrop } from "./canvas-drop-smoke.mjs";
import { verifyLongNote } from "./long-note-smoke.mjs";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  existsSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import os from "node:os";
import assert from "node:assert/strict";
import { verifyProjectCover } from "./cover-smoke.mjs";
import { verifyResponsiveWorkspace } from "./responsive-smoke.mjs";
import { verifyCanvasContextMenu } from "./context-menu-smoke.mjs";
import { verifyCanvasInputs } from "./input-smoke.mjs";
import { verifyModelPreview } from "./model-preview-smoke.mjs";
const { chromium } = await import(
  process.env.PLAYWRIGHT_MODULE || "playwright"
);
const base = process.env.STUDIO_TEST_URL || "http://127.0.0.1:4311";
const output = path.resolve("var/qa");
mkdirSync(output, { recursive: true });
const projectDir = path.join(
  mkdtempSync(path.join(os.tmpdir(), "studio-browser-")),
  "浏览器验收项目",
);
mkdirSync(projectDir);
writeFileSync(path.join(projectDir, "existing.txt"), "preserve existing files");
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("response", async (response) => {
  if (response.url().includes("/api/") && response.status() >= 400)
    console.error(
      "HTTP error:",
      response.status(),
      response.url(),
      await response.text(),
    );
});
try {
  await page.goto(base);
  await page.waitForLoadState("networkidle");
  await page.evaluate(() => {
    window.__versionReloadProbe = "old-document";
  });
  await page.getByRole("button", { name: /重新加载工作台/ }).press("Enter");
  await page.waitForURL(/studioReload=/);
  await page.getByRole("button", { name: /重新加载工作台/ }).waitFor();
  assert.equal(
    await page.evaluate(() => window.__versionReloadProbe),
    undefined,
  );
  await page.screenshot({
    path: path.join(output, "browser-version-reload.png"),
  });
  console.log(
    "PASS: version button performs a full HTTP document reload with a fresh URL.",
  );
  await page.getByRole("button", { name: "新建项目", exact: true }).click();
  await page.getByLabel("项目名称", { exact: true }).fill("浏览器验收项目");
  const directory = page.getByLabel("项目目录", { exact: true });
  await directory.fill(projectDir);
  let picks = 0;
  await page.route("**/api/directories/pick", async (route) => {
    assert.equal(route.request().postDataJSON().mode, "create");
    picks++;
    await route.fulfill({
      status: picks === 2 ? 400 : 200,
      contentType: "application/json",
      body: JSON.stringify(
        picks === 1
          ? { path: null }
          : picks === 2
            ? { error: "选择窗口暂时不可用" }
            : { path: projectDir },
      ),
    });
  });
  await page.getByRole("button", { name: "选择文件夹", exact: true }).click();
  await page.getByRole("button", { name: "选择文件夹", exact: true }).waitFor();
  assert.equal(await directory.inputValue(), projectDir);
  await page.getByRole("button", { name: "选择文件夹", exact: true }).click();
  await page
    .getByRole("alert")
    .filter({ hasText: "选择窗口暂时不可用" })
    .waitFor();
  await page.getByRole("button", { name: "选择文件夹", exact: true }).click();
  await page.getByRole("button", { name: "选择文件夹", exact: true }).waitFor();
  const subdirectory = page.getByRole("checkbox", {
    name: "按项目名称创建子文件夹",
  });
  const preview = page.getByLabel("最终项目路径", { exact: true });
  assert.equal(await subdirectory.isChecked(), false);
  assert.equal(await directory.inputValue(), projectDir);
  await page.getByLabel("项目名称", { exact: true }).fill("自动更新目录");
  assert.equal(await directory.inputValue(), projectDir);
  assert.equal(await preview.textContent(), projectDir);
  await subdirectory.check();
  assert.equal(
    await page.getByLabel("父目录", { exact: true }).inputValue(),
    projectDir,
  );
  assert.equal(
    await preview.textContent(),
    path.join(projectDir, "自动更新目录"),
  );
  await page.getByLabel("项目名称", { exact: true }).fill("../星铸者");
  assert.equal(await preview.textContent(), path.join(projectDir, "-星铸者"));
  await subdirectory.uncheck();
  assert.equal(await directory.inputValue(), projectDir);
  assert.equal(await preview.textContent(), projectDir);
  await subdirectory.check();
  assert.equal(await preview.textContent(), path.join(projectDir, "-星铸者"));
  await subdirectory.uncheck();
  const manualDirectory = path.join(path.dirname(projectDir), "手动目录");
  await directory.fill(manualDirectory);
  await page.getByLabel("项目名称", { exact: true }).fill("浏览器验收项目");
  assert.equal(await directory.inputValue(), manualDirectory);
  await page.getByRole("button", { name: "选择文件夹", exact: true }).click();
  await page.getByRole("button", { name: "选择文件夹", exact: true }).waitFor();
  assert.equal(await directory.inputValue(), projectDir);
  await page.screenshot({
    path: path.join(output, "project-folder-picker.png"),
  });
  await page.getByRole("button", { name: "创建项目", exact: true }).click();
  await page.getByText("这是你的创作画布").waitFor();
  await verifyResponsiveWorkspace({ page, prefix: "browser" });
  const response = await page.request.get(`${base}/api/projects`);
  const projects = await response.json();
  const project = projects.find((p) => p.root === realpathSync(projectDir));
  assert.ok(project);
  assert.equal(
    readFileSync(path.join(projectDir, "existing.txt"), "utf8"),
    "preserve existing files",
  );
  assert.equal(existsSync(path.join(projectDir, "浏览器验收项目")), false);
  // Upload goes through the browser form and the same Core operation as MCP.
  await page
    .locator("input[type=file]")
    .setInputFiles(path.resolve("samples/reference.png"));
  await page
    .getByRole("dialog")
    .getByLabel("资源名称", { exact: true })
    .fill("巡林者参考");
  await page
    .getByRole("button", { name: "导入并登记版本", exact: true })
    .click();
  await page
    .locator(".asset-dock")
    .getByRole("button", { name: "展开全部", exact: true })
    .click();
  await page.getByRole("button", { name: "巡林者参考 图片 · v1" }).click();
  await page.locator(".asset-preview img").waitFor();
  assert.equal(
    await page
      .locator(".asset-preview img")
      .evaluate((img) => img.complete && img.naturalWidth > 0),
    true,
  );
  await page
    .getByRole("button", { name: "添加到当前画布", exact: true })
    .click();
  await page.locator(".asset-node").first().waitFor();
  await page.getByRole("button", { name: "便签", exact: true }).click();
  await page
    .getByRole("dialog")
    .getByLabel("内容", { exact: true })
    .fill("保留兜帽与灯笼。\n浏览器实际输入。");
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "保存", exact: true })
    .click();
  await page.locator(".note-node").first().waitFor();
  await page.locator(".asset-node").first().waitFor({ state: "visible" });
  await page.getByRole("button", { name: "fit view" }).click();
  await page.locator(".asset-node").first().click();
  await page.getByRole("button", { name: "制作角色", exact: true }).click();
  await page.getByRole("button", { name: "添加制作分组", exact: true }).click();
  await page.locator(".workflow-node").first().waitFor();
  // Zoom to fit the material and newly expanded workflow, then persist the view.
  await page.getByRole("button", { name: "fit view" }).click();
  await page.waitForTimeout(500);
  await page.locator(".workflow-node").first().click();
  await verifyCanvasTools({
    page,
    prefix: "browser",
    readSnapshot: async () =>
      (await page.request.get(`${base}/api/projects/${project.id}`)).json(),
  });
  await verifyCanvasInputs({
    page,
    prefix: "browser",
    readSnapshot: async () =>
      (await page.request.get(`${base}/api/projects/${project.id}`)).json(),
  });
  await verifyCanvasContextMenu({
    page,
    prefix: "browser",
    readSnapshot: async () =>
      (await page.request.get(`${base}/api/projects/${project.id}`)).json(),
  });
  await verifyLongNote({
    page,
    prefix: "browser",
    readSnapshot: async () =>
      (await page.request.get(`${base}/api/projects/${project.id}`)).json(),
  });
  await verifyNodeSizing({
    page,
    prefix: "browser",
    readSnapshot: async () =>
      (await page.request.get(`${base}/api/projects/${project.id}`)).json(),
  });
  await page.getByText("等待 Codex", { exact: true }).first().waitFor();
  await page.screenshot({
    path: path.join(output, "browser-canvas.png"),
    fullPage: true,
  });
  let s = await (
    await page.request.get(`${base}/api/projects/${project.id}`)
  ).json();
  assert.equal(s.runs.length, 1);
  assert.equal(s.runs[0].steps[0].status, "awaiting_agent");
  assert.equal(s.assets.length, 1);
  assert.equal(s.canvases[0].instances.length, 1);
  // Verify manual/Agent continuation through server tools and refresh in the browser.
  const importResponse = await page.request.post(
    `${base}/api/projects/${project.id}/import`,
    {
      multipart: {
        name: "角色 Brief",
        files: {
          name: "brief.md",
          mimeType: "text/plain",
          buffer: readFileSync("samples/brief.md"),
        },
      },
    },
  );
  assert.ok(importResponse.ok());
  const brief = await importResponse.json();
  const submit = await page.request.post(
    `${base}/api/tools/workflow_submit_artifact`,
    {
      data: {
        projectId: project.id,
        runId: s.runs[0].id,
        stepKey: "brief",
        revisionIds: [brief.id],
        expectedRevision: s.runs[0].revision,
      },
    },
  );
  assert.ok(submit.ok());
  await page.reload();
  await page.waitForLoadState("networkidle");
  await page
    .getByRole("button")
    .filter({ hasText: "浏览器验收项目" })
    .first()
    .click();
  await page.locator(".workflow-node").first().click();
  await page.locator(".next-step h3").filter({ hasText: "制作参考" }).waitFor();
  s = await (
    await page.request.get(`${base}/api/projects/${project.id}`)
  ).json();
  assert.equal(s.runs[0].steps[1].status, "awaiting_manual");
  // Drag is a layout mutation and must not affect pinned inputs or graph revision.
  const before = s.canvases[0];
  const node = page.locator(".note-node").first();
  const bounds = await node.boundingBox();
  assert.ok(bounds);
  await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + 30);
  await page.mouse.down();
  await page.mouse.move(bounds.x + bounds.width / 2 + 80, bounds.y + 70, {
    steps: 12,
  });
  await page.mouse.up();
  await page.waitForTimeout(600);
  s = await (
    await page.request.get(`${base}/api/projects/${project.id}`)
  ).json();
  assert.equal(s.canvases[0].graphRevision, before.graphRevision);
  assert.ok(s.canvases[0].layoutRevision > before.layoutRevision);
  assert.deepEqual(s.runs[0].inputRevisionIds, [
    before.nodes.find((n) => n.type === "asset").data.revisionId,
  ]);
  await page.getByRole("button", { name: "新建画布", exact: true }).click();
  await page.getByRole("dialog").getByLabel("画布名称").fill("场景规划");
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "保存", exact: true })
    .click();
  await page.getByText("这是你的创作画布").waitFor();
  await page.getByRole("button", { name: "巡林者参考 图片 · v2" }).click();
  await page.getByRole("button", { name: "版本", exact: true }).click();
  await page
    .locator(".version-list")
    .getByRole("button", { name: /^版本 1/ })
    .click();
  await page.getByRole("button", { name: "预览", exact: true }).click();
  await page
    .getByRole("button", { name: "添加到当前画布", exact: true })
    .click();
  s = await (
    await page.request.get(`${base}/api/projects/${project.id}`)
  ).json();
  assert.equal(s.assets.length, 2);
  assert.equal(s.canvases.length, 2);
  assert.equal(
    s.canvases[1].nodes[0].data.revisionId,
    s.canvases[0].nodes.find((n) => n.type === "asset").data.revisionId,
  );
  // Browser security and malformed CAS calls.
  const cross = await page.request.post(`${base}/api/tools/studio_open`, {
    headers: { Origin: "https://example.com" },
    data: {},
  });
  assert.equal(cross.status(), 403);
  const conflict = await page.request.post(`${base}/api/tools/canvas_patch`, {
    data: {
      projectId: project.id,
      canvasId: s.canvases[0].id,
      target: "graph",
      expectedRevision: 0,
      ops: [],
    },
  });
  assert.equal(conflict.status(), 409);
  await verifyModelPreview({
    page,
    prefix: "browser",
    readSnapshot: async () =>
      (await page.request.get(`${base}/api/projects/${project.id}`)).json(),
  });
  await verifyResourceComposer({
    page,
    prefix: "browser",
    readSnapshot: async () =>
      (await page.request.get(`${base}/api/projects/${project.id}`)).json(),
  });
  await verifyCanvasDrop({
    page,
    prefix: "browser",
    readSnapshot: async () =>
      (await page.request.get(`${base}/api/projects/${project.id}`)).json(),
  });
  await verifyProjectCover({
    page,
    project,
    readSnapshot: async () =>
      (await page.request.get(`${base}/api/projects/${project.id}`)).json(),
  });
  await page.screenshot({
    path: path.join(output, "project-cover-home.png"),
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(250);
  await page.screenshot({
    path: path.join(output, "browser-mobile.png"),
    fullPage: true,
  });
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth,
    ),
    false,
  );
  assert.deepEqual(errors, []);
  console.log(
    "Browser smoke passed: UI creation/import, fixed references, workflow start, Agent continuation, refresh recovery, drag persistence, multi-canvas reuse, Origin rejection, CAS conflict, real GLB loading/rotation, mobile width.",
  );
} catch (e) {
  await page.screenshot({
    path: path.join(output, "browser-failure.png"),
    fullPage: true,
  });
  console.error("PAGE ERRORS", errors);
  console.error(await page.locator("body").innerText());
  throw e;
} finally {
  await browser.close();
}
