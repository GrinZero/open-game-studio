import { verifyDeepLinks } from "./deep-link-smoke.mjs";
import { verifyCharacterSource } from "./character-source-smoke.mjs";
import { verifyNodeSizing } from "./node-sizing-smoke.mjs";
import { verifyCanvasTools } from "./canvas-tools-smoke.mjs";
import { verifyResourceComposer } from "./resource-composer-smoke.mjs";
import { verifyCanvasDrop } from "./canvas-drop-smoke.mjs";
import { verifyLongNote } from "./long-note-smoke.mjs";
import { createServer } from "node:http";
import { mkdirSync, mkdtempSync, realpathSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { once } from "node:events";
import { build } from "esbuild";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { chromium } from "playwright";
import { verifyProjectCover } from "./cover-smoke.mjs";
import { verifyResponsiveWorkspace } from "./responsive-smoke.mjs";
import { verifyResponsiveHome } from "./home-responsive-smoke.mjs";
import { verifyCanvasContextMenu } from "./context-menu-smoke.mjs";
import { verifyCanvasInputs } from "./input-smoke.mjs";
import { verifyModelPreview } from "./model-preview-smoke.mjs";

const dir = mkdtempSync(path.join(os.tmpdir(), "studio-app-"));
const existingDirectory = path.join(dir, "已有游戏");
mkdirSync(existingDirectory);
const probe = createServer();
await new Promise((resolve) => probe.listen(0, "127.0.0.1", resolve));
const binaryPort = probe.address().port;
await new Promise((resolve) => probe.close(resolve));
const binaryOrigin = `http://127.0.0.1:${binaryPort}`;
const serviceEnv = {
  ...process.env,
  STUDIO_HOME: path.join(dir, "registry"),
  PORT: String(binaryPort),
};
const binaryService = spawn(
  process.execPath,
  ["node_modules/tsx/dist/cli.mjs", "src/server/http.ts", "--production"],
  { env: serviceEnv, stdio: "pipe" },
);
await new Promise((resolve, reject) => {
  const timer = setTimeout(
    () => reject(new Error("Binary HTTP service timed out")),
    15000,
  );
  binaryService.stdout.on("data", (data) => {
    if (String(data).includes("Open Game Studio →")) {
      clearTimeout(timer);
      resolve();
    }
  });
  binaryService.once("exit", () => {
    clearTimeout(timer);
    reject(new Error("Binary HTTP service exited"));
  });
});
const client = new Client({ name: "studio-sandbox-check", version: "1.0.0" });
await client.connect(
  new StdioClientTransport({
    command: process.execPath,
    args: ["node_modules/tsx/dist/cli.mjs", "src/server/mcp.ts"],
    env: serviceEnv,
    stderr: "pipe",
  }),
);
const created = await client.callTool({
  name: "project_manage",
  arguments: {
    action: "create",
    name: "内嵌验收",
    root: path.join(dir, "game"),
  },
});
const project = created.structuredContent.result;
const resource = await client.readResource({
  uri: "ui://game-studio/workbench",
});
const host = await build({
  stdin: {
    contents: `
import { AppBridge, PostMessageTransport } from '@modelcontextprotocol/ext-apps/app-bridge';
const iframe = document.querySelector('iframe');
const initialMode = new URLSearchParams(location.search).get('mode');
const bridge = new AppBridge(null, {name:'test-host',version:'1.0.0'}, {serverTools:{},serverResources:{},downloadFile:{},updateModelContext:{text:{}},message:{text:{}}}, {hostContext:{theme:'dark',...(initialMode ? {displayMode:initialMode} : {})}});
window.__openedLinks = [];
bridge.onopenlink = async args => { window.__openedLinks.push(args.url); return {}; };
window.__referenceUpdates = [];
window.__modelMessages = [];
bridge.onmessage = async args => {
  if (window.__rejectModelMessage) return {isError:true};
  window.__modelMessages.push(args);
  return {};
};
bridge.onupdatemodelcontext = async args => {
  if (window.__rejectReferenceUpdate) throw new Error('引用更新被宿主拒绝');
  window.__referenceUpdates.push(args);
  return {};
};
window.__setStudioHostContext = context => bridge.setHostContext(context);
window.__sendStudioLaunch = async (args, result) => { await bridge.sendToolInput({arguments:args}); await bridge.sendToolResult({content:[],structuredContent:{result}}); };
bridge.oncalltool = async args => (await fetch('/tool', {method:'POST',body:JSON.stringify(args)})).json();
bridge.onreadresource = async args => { const result = await (await fetch('/read-resource', {method:'POST',body:JSON.stringify(args)})).json(); if (result.error) throw new Error(result.error); return result; };
bridge.ondownloadfile = async args => { await fetch('/download', {method:'POST',body:JSON.stringify(args)}); return {}; };
bridge.oninitialized = () => { document.body.dataset.connected = 'true'; const projectId = new URLSearchParams(location.search).get('projectId'); if (projectId) bridge.sendToolInput({arguments:Object.fromEntries(new URLSearchParams(location.search))}); };
await bridge.connect(new PostMessageTransport(iframe.contentWindow, iframe.contentWindow));
iframe.srcdoc = await (await fetch('/resource')).text();
`,
    resolveDir: process.cwd(),
  },
  bundle: true,
  write: false,
  format: "esm",
  minify: true,
});
let reloadResourceReads = 0;
let reloadResourceFailure = false;
let reloadResourceHtml;
let downloaded;
let directoryPicks = 0;
let bridgeFileReads = 0;
let httpFileReads = 0;
const csp = `default-src 'none'; script-src 'unsafe-inline' blob:; style-src 'unsafe-inline'; img-src blob: data:; font-src data:; connect-src blob: ${binaryOrigin}; media-src blob:; worker-src blob:`;
const server = createServer(async (req, res) => {
  try {
    if (req.url === "/resource") {
      res.setHeader("Content-Type", "text/html");
      res.end(
        resource.contents[0].text.replace(
          "<head>",
          `<head><meta http-equiv="Content-Security-Policy" content="${csp}">`,
        ),
      );
    } else if (req.url === "/read-resource") {
      let body = "";
      for await (const bytes of req) body += bytes;
      const args = JSON.parse(body);
      assert.equal(args.uri, "ui://game-studio/workbench");
      reloadResourceReads++;
      res.setHeader("Content-Type", "application/json");
      if (reloadResourceFailure)
        res.end(JSON.stringify({ error: "构建读取失败" }));
      else {
        const latest = await client.readResource(args);
        if (reloadResourceHtml) latest.contents[0].text = reloadResourceHtml;
        res.end(JSON.stringify(latest));
      }
    } else if (req.url === "/tool" || req.url === "/download") {
      let body = "";
      for await (const bytes of req) body += bytes;
      const args = JSON.parse(body);
      res.setHeader("Content-Type", "application/json");
      if (req.url === "/download") {
        downloaded = args;
        res.end("{}");
      } else {
        assert.equal(args.name, "studio_ui_request");
        if (/\/files\/[\w-]+\/\d+$/.test(args.arguments.path))
          bridgeFileReads++;
        if (args.arguments.path === "/api/directories/pick") {
          assert.equal(args.arguments.method, "POST");
          directoryPicks++;
          const selectedDirectory =
            args.arguments.body.mode === "link" ? existingDirectory : dir;
          res.end(
            JSON.stringify({
              content: [],
              structuredContent: {
                result: {
                  path: directoryPicks === 1 ? null : selectedDirectory,
                },
              },
            }),
          );
        } else res.end(JSON.stringify(await client.callTool(args)));
      }
    } else {
      res.setHeader("Content-Type", "text/html");
      res.end(
        `<html><body style="margin:0"><iframe title="test-app" sandbox="allow-scripts allow-downloads" style="width:100%;height:100vh;border:0"></iframe><script type="module">${host.outputFiles[0].text.replaceAll("</script", "<\\/script")}</script></body></html>`,
      );
    }
  } catch (error) {
    res.statusCode = 500;
    res.end(JSON.stringify({ error: String(error) }));
  }
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
const errors = [];
await page.addInitScript(() => {
  window.__revokedAssetUrls = [];
  const revoke = URL.revokeObjectURL.bind(URL);
  URL.revokeObjectURL = (url) => {
    window.__revokedAssetUrls.push(url);
    revoke(url);
  };
});
page.on("request", (request) => {
  if (request.url().startsWith(binaryOrigin) && request.method() === "GET")
    httpFileReads++;
});
if (process.env.STUDIO_TEST_ASSET_BRIDGE_ONLY)
  await page.route(`${binaryOrigin}/**`, (route) => route.abort());
page.on("pageerror", (error) => errors.push(error.message));
page.on("console", (message) => {
  if (message.type() === "error") console.error(message.text());
});
try {
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.waitForFunction(() => document.body.dataset.connected === "true");
  const ui = page.frameLocator("iframe");
  if (process.env.STUDIO_TEST_CHARACTER_SOURCE_ONLY) {
    mkdirSync("var/qa", { recursive: true });
    const api = async (route, body) => {
      const response = await fetch(
        `${binaryOrigin}${route}`,
        body
          ? {
              method: "POST",
              ...(body instanceof FormData
                ? {}
                : { headers: { "Content-Type": "application/json" } }),
              body: body instanceof FormData ? body : JSON.stringify(body),
            }
          : undefined,
      );
      const result = await response.json();
      assert.equal(response.ok, true, JSON.stringify(result));
      return result;
    };
    await page.goto(`${binaryOrigin}/?projectId=${project.id}`);
    await verifyCharacterSource({
      page,
      api,
      projectId: project.id,
      prefix: "http",
    });
    const embeddedProject = await api("/api/tools/project_manage", {
      action: "create",
      name: "角色原画内嵌验收",
      root: path.join(dir, "character-embedded"),
    });
    await page.goto(
      `http://127.0.0.1:${server.address().port}/?projectId=${embeddedProject.id}`,
    );
    await verifyCharacterSource({
      page,
      ui,
      api,
      projectId: embeddedProject.id,
      prefix: "mcp",
    });
    assert.deepEqual(errors, []);
    console.log(
      "PASS: HTTP/MCP character creation requires explicit image choice, previews images, uploads, cancels and replaces source revisions.",
    );
  } else if (process.env.STUDIO_TEST_LAUNCH_ONLY) {
    mkdirSync("var/qa", { recursive: true });
    await ui.locator(".project-card").waitFor();
    await verifyResponsiveHome({ page, ui, prefix: "mcp" });
    const opened = await client.callTool({
      name: "studio_open",
      arguments: { workspaceRoot: project.root },
    });
    await page.evaluate(
      (result) => window.__sendStudioLaunch({}, result),
      opened.structuredContent.result,
    );
    await ui.locator(".workspace").waitFor();
    await ui
      .getByRole("button", { name: `查看${project.name}项目概览`, exact: true })
      .waitFor();
    const target = (
      await client.callTool({
        name: "canvas_manage",
        arguments: {
          action: "create",
          projectId: project.id,
          name: "海岛主城资源总览",
        },
      })
    ).structuredContent.result;
    const targetOpen = (
      await client.callTool({
        name: "studio_open",
        arguments: {
          projectId: project.id,
          canvasId: target.id,
        },
      })
    ).structuredContent.result;
    // An existing app must react to a same-project canvas change, including
    // when the host sends only a completed tool result.
    await page.evaluate(
      (result) => window.__sendStudioLaunch({}, result),
      targetOpen,
    );
    await ui
      .locator(
        `select[aria-label="切换画布"] option[value="${target.id}"]:checked`,
      )
      .waitFor({ state: "attached" });
    assert.equal(
      await ui.getByRole("combobox", { name: "切换画布" }).inputValue(),
      target.id,
    );
    await page.goto(
      `http://127.0.0.1:${server.address().port}/?projectId=${project.id}&canvasId=${target.id}`,
    );
    await ui.locator(".workspace").waitFor();
    assert.equal(
      await ui.getByRole("combobox", { name: "切换画布" }).inputValue(),
      target.id,
    );
    await page.goto(
      targetOpen.url.replace(/http:\/\/127\.0\.0\.1:\d+/, binaryOrigin),
    );
    await page.locator(".workspace").waitFor();
    assert.equal(
      await page.getByRole("combobox", { name: "切换画布" }).inputValue(),
      target.id,
    );
    await page.goto(
      `http://127.0.0.1:${server.address().port}/?projectId=${project.id}`,
    );
    await ui.locator(".workspace").waitFor();
    await verifyDeepLinks({
      page,
      ui,
      client,
      project,
      hostOrigin: `http://127.0.0.1:${server.address().port}`,
      binaryOrigin,
    });
    await page.goto(
      `http://127.0.0.1:${server.address().port}/?projectId=${project.id}`,
    );
    await ui.locator(".workspace").waitFor();
    await ui
      .getByRole("button", { name: `查看${project.name}项目概览`, exact: true })
      .waitFor();
    await page.goto(`${binaryOrigin}/?projectId=${project.id}`);
    await page.locator(".workspace").waitFor();
    await page
      .getByRole("button", { name: `查看${project.name}项目概览`, exact: true })
      .waitFor();
    await page.goto(binaryOrigin);
    await page.locator(".project-card").waitFor();
    await verifyResponsiveHome({ page, prefix: "http" });
    assert.deepEqual(errors, []);
    console.log(
      "PASS: HTTP/MCP responsive project index, early tool input, late tool result, workspace selection and browser project link.",
    );
  } else if (process.env.STUDIO_TEST_DROP_ONLY) {
    await ui
      .locator(".project-card-open")
      .filter({ hasText: project.name })
      .click();
    await verifyCanvasDrop({
      page,
      ui,
      prefix: "mcp-drop",
      readSnapshot: async () =>
        (
          await client.callTool({
            name: "project_open",
            arguments: { projectId: project.id },
          })
        ).structuredContent.result,
    });
    assert.deepEqual(errors, []);
  } else {
    await ui
      .getByRole("button")
      .filter({ hasText: "内嵌验收" })
      .first()
      .waitFor();
    assert.equal(
      await ui.locator("html").getAttribute("data-mcp-display-mode"),
      "inline",
      "missing host display mode defaults to a bounded embed",
    );
    // A host may keep the original srcdoc cached even after the runtime updates.
    // Read a newer bundle and navigate the iframe itself; repeated refreshes must
    // reinitialize the MCP bridge and keep the project registry available.
    const originalMarker =
      resource.contents[0].text.match(/v0\.1\.7-\d{14}/)[0];
    const newerMarker = "v0.1.7-20990101010101";
    const versionButton = () =>
      ui.getByRole("button", { name: /重新加载工作台/ });
    reloadResourceFailure = true;
    await versionButton().click();
    await ui.getByRole("alert").filter({ hasText: "重新加载失败" }).waitFor();
    assert.ok(await versionButton().isEnabled());
    await ui.getByRole("button", { name: "关闭刷新提示", exact: true }).click();
    reloadResourceFailure = false;
    reloadResourceHtml = resource.contents[0].text.replaceAll(
      originalMarker.slice(-14),
      newerMarker.slice(-14),
    );
    await versionButton().click();
    await ui
      .getByRole("button", {
        name: `重新加载工作台 · ${newerMarker}`,
        exact: true,
      })
      .waitFor();
    assert.ok(
      (await ui.locator("html").evaluate(() => location.href)).startsWith(
        "blob:",
      ),
    );
    assert.ok(
      (await page.locator("iframe").getAttribute("srcdoc")).includes(
        originalMarker,
      ),
    );
    await ui
      .getByRole("button")
      .filter({ hasText: "内嵌验收" })
      .first()
      .waitFor();
    reloadResourceHtml = undefined;
    await versionButton().press("Enter");
    await ui
      .getByRole("button", {
        name: `重新加载工作台 · ${originalMarker}`,
        exact: true,
      })
      .waitFor();
    await ui
      .getByRole("button")
      .filter({ hasText: "内嵌验收" })
      .first()
      .waitFor();
    assert.equal(reloadResourceReads, 3);
    mkdirSync("var/qa", { recursive: true });
    await page.screenshot({ path: "var/qa/mcp-version-reload.png" });
    console.log(
      "PASS: version button loads the latest MCP document despite cached srcdoc, reconnects repeatedly, and preserves the page after read failure.",
    );
    async function verifyDisplayModes() {
      for (const mode of ["inline", "fullscreen", "inline", "fullscreen"]) {
        await page.evaluate(
          (mode) => window.__setStudioHostContext({ displayMode: mode }),
          mode,
        );
        await ui.locator(`html[data-mcp-display-mode="${mode}"]`).waitFor();
        for (const height of [480, 720, 960, 1200]) {
          await page.setViewportSize({ width: 1440, height });
          await page.waitForTimeout(100);
          const expectedHeight =
            mode === "fullscreen" ? height : Math.min(height, 720);
          assert.equal(
            await ui
              .locator(".app-shell")
              .evaluate((element) => element.getBoundingClientRect().height),
            expectedHeight,
            `${mode} shell height at ${height}`,
          );
          if (await ui.locator(".workspace").count()) {
            assert.equal(
              await ui
                .locator(".workspace")
                .evaluate((element) => element.getBoundingClientRect().height),
              expectedHeight,
              `${mode} workspace height at ${height}`,
            );
          }
        }
        // Partial context changes must preserve the current display mode.
        await page.evaluate(() =>
          window.__setStudioHostContext({ theme: "dark" }),
        );
        assert.equal(
          await ui.locator("html").getAttribute("data-mcp-display-mode"),
          mode,
        );
      }
    }
    assert.equal(
      await ui
        .locator(".app-shell")
        .evaluate((element) => element.getBoundingClientRect().height),
      720,
      "unspecified initial mode stays below 720px",
    );
    await verifyDisplayModes();
    for (const mode of ["inline", "fullscreen"]) {
      await page.goto(`http://127.0.0.1:${server.address().port}?mode=${mode}`);
      await ui
        .getByRole("button")
        .filter({ hasText: "内嵌验收" })
        .first()
        .waitFor();
      assert.equal(
        await ui.locator("html").getAttribute("data-mcp-display-mode"),
        mode,
        `initial host mode ${mode}`,
      );
      assert.equal(
        await ui
          .locator(".app-shell")
          .evaluate((element) => element.getBoundingClientRect().height),
        mode === "fullscreen" ? 1200 : 720,
        `initial ${mode} shell height`,
      );
    }
    await page.setViewportSize({ width: 1440, height: 960 });
    await ui
      .getByRole("button")
      .filter({ hasText: "内嵌验收" })
      .first()
      .click();
    await ui.getByText("这是你的创作画布").waitFor();
    await verifyDisplayModes();
    console.log(
      "PASS: inline/missing mode caps at 720px; fullscreen fills host; home/workspace resize and mode transitions preserve the correct height.",
    );
    await page.setViewportSize({ width: 1440, height: 960 });
    mkdirSync("var/qa", { recursive: true });
    await verifyResponsiveWorkspace({ page, ui, prefix: "mcp" });
    await ui
      .locator("input[type=file]")
      .setInputFiles(path.resolve("samples/reference.png"));
    await ui
      .getByRole("dialog")
      .getByLabel("资源名称", { exact: true })
      .fill("内嵌参考图");
    await ui
      .getByRole("button", { name: "导入并登记版本", exact: true })
      .click();
    await ui
      .locator(".asset-dock")
      .getByRole("button", { name: "展开全部", exact: true })
      .click();
    await ui.getByRole("button", { name: "内嵌参考图 图片 · v1" }).click();
    await ui.locator(".asset-preview img").waitFor();
    await ui.locator(".asset-preview img").evaluate((img) => img.decode());
    await ui
      .getByRole("button", { name: "添加到切换画布", exact: true })
      .click();
    await ui.locator(".asset-node img").evaluate((img) => img.decode());
    await ui.getByRole("button", { name: "便签", exact: true }).click();
    await ui
      .getByRole("dialog")
      .getByLabel("内容", { exact: true })
      .fill("沙箱内保存，重新加载接续。");
    await ui
      .getByRole("dialog")
      .getByRole("button", { name: "保存", exact: true })
      .click();
    await ui.locator(".note-node").waitFor();
    const snapshot = (
      await client.callTool({
        name: "project_open",
        arguments: { projectId: project.id },
      })
    ).structuredContent.result;
    assert.equal(snapshot.assets.length, 1);
    assert.equal(snapshot.canvases[0].nodes.length, 2);
    await ui.locator(".asset-node").click();
    await ui.getByText("文件与完整性哈希", { exact: true }).click();
    await ui.getByRole("link", { name: "reference.png", exact: true }).click();
    for (let i = 0; i < 20 && !downloaded; i++) await page.waitForTimeout(100);
    assert.ok(downloaded.contents[0].resource.blob);
    await verifyModelPreview({
      page,
      ui,
      prefix: "mcp",
      readSnapshot: async () =>
        (
          await client.callTool({
            name: "project_open",
            arguments: { projectId: project.id },
          })
        ).structuredContent.result,
    });
    await ui.locator(".asset-node").filter({ hasText: "内嵌参考图" }).click();
    await ui.getByRole("button", { name: "制作角色", exact: true }).click();
    await ui.getByRole("button", { name: "添加制作分组", exact: true }).click();
    await ui.locator(".workflow-node").first().waitFor();
    await verifyCanvasInputs({
      page,
      ui,
      prefix: "mcp",
      readSnapshot: async () =>
        (
          await client.callTool({
            name: "project_open",
            arguments: { projectId: project.id },
          })
        ).structuredContent.result,
    });
    await verifyCanvasContextMenu({
      page,
      ui,
      prefix: "mcp",
      readDownload: () => downloaded,
      readSnapshot: async () =>
        (
          await client.callTool({
            name: "project_open",
            arguments: { projectId: project.id },
          })
        ).structuredContent.result,
    });
    await verifyCanvasTools({
      page,
      ui,
      prefix: "mcp",
      readSnapshot: async () =>
        (
          await client.callTool({
            name: "project_open",
            arguments: { projectId: project.id },
          })
        ).structuredContent.result,
    });
    await verifyLongNote({
      page,
      ui,
      prefix: "mcp",
      readSnapshot: async () =>
        (
          await client.callTool({
            name: "project_open",
            arguments: { projectId: project.id },
          })
        ).structuredContent.result,
    });
    await verifyNodeSizing({
      page,
      ui,
      prefix: "mcp",
      readSnapshot: async () =>
        (
          await client.callTool({
            name: "project_open",
            arguments: { projectId: project.id },
          })
        ).structuredContent.result,
    });
    await ui.getByRole("button", { name: "全部项目", exact: true }).click();
    await ui.getByRole("button", { name: "新建项目", exact: true }).click();
    await ui.getByLabel("项目名称", { exact: true }).fill("内嵌新建");
    await ui.getByRole("button", { name: "选择文件夹", exact: true }).click();
    await ui.getByRole("button", { name: "选择文件夹", exact: true }).waitFor();
    assert.equal(
      await ui.getByLabel("项目目录", { exact: true }).inputValue(),
      "",
    );
    await ui.getByRole("button", { name: "选择文件夹", exact: true }).click();
    await ui.getByRole("button", { name: "选择文件夹", exact: true }).waitFor();
    assert.equal(
      await ui.getByLabel("项目目录", { exact: true }).inputValue(),
      dir,
    );
    const subdirectory = ui.getByRole("checkbox", {
      name: "按项目名称创建子文件夹",
    });
    assert.equal(await subdirectory.isChecked(), false);
    assert.equal(
      await ui.getByLabel("最终项目路径", { exact: true }).textContent(),
      dir,
    );
    await subdirectory.check();
    assert.equal(
      await ui.getByLabel("父目录", { exact: true }).inputValue(),
      dir,
    );
    assert.equal(
      await ui.getByLabel("最终项目路径", { exact: true }).textContent(),
      path.join(dir, "内嵌新建"),
    );
    await ui.getByRole("button", { name: "创建项目", exact: true }).click();
    await ui.getByText("这是你的创作画布").waitFor();
    const projects = (
      await client.callTool({ name: "studio_open", arguments: {} })
    ).structuredContent.result.projects;
    assert.equal(projects.length, 2);
    assert.equal(
      projects.find((p) => p.name === "内嵌新建").root,
      realpathSync(path.join(dir, "内嵌新建")),
    );
    assert.equal(errors.length, 0, errors.join("\n"));
    await ui.getByRole("button", { name: "全部项目", exact: true }).click();
    await ui.getByRole("button", { name: "关联已有项目", exact: true }).click();
    await ui.getByLabel("项目名称", { exact: true }).fill("已有游戏");
    await ui.getByRole("button", { name: "选择文件夹", exact: true }).click();
    await ui.getByRole("button", { name: "选择文件夹", exact: true }).waitFor();
    assert.equal(
      await ui.getByLabel("项目目录", { exact: true }).inputValue(),
      existingDirectory,
    );
    await ui.getByRole("button", { name: "关联项目", exact: true }).click();
    await ui.getByText("这是你的创作画布").waitFor();
    const linked = (
      await client.callTool({ name: "studio_open", arguments: {} })
    ).structuredContent.result.projects;
    assert.equal(linked.length, 3);
    assert.equal(
      linked.find((p) => p.name === "已有游戏").root,
      realpathSync(existingDirectory),
    );
    await ui.getByRole("button", { name: "全部项目", exact: true }).click();
    await ui
      .locator(".project-card-open")
      .filter({ hasText: project.name })
      .click();
    await verifyProjectCover({
      page,
      ui,
      project,
      readSnapshot: async () =>
        (
          await client.callTool({
            name: "project_open",
            arguments: { projectId: project.id },
          })
        ).structuredContent.result,
    });
    assert.deepEqual(errors, []);
    assert.ok(
      httpFileReads > 0,
      "the embedded app attempted binary HTTP reads",
    );
    if (process.env.STUDIO_TEST_ASSET_BRIDGE_ONLY)
      assert.ok(
        bridgeFileReads > 0,
        "blocked HTTP fell back to the MCP bridge",
      );
    else
      assert.equal(
        bridgeFileReads,
        0,
        "binary HTTP did not transfer any base64 file chunks",
      );
    await verifyResourceComposer({
      page,
      ui,
      prefix: "mcp",
      readSnapshot: async () =>
        (
          await client.callTool({
            name: "project_open",
            arguments: { projectId: project.id },
          })
        ).structuredContent.result,
    });
    await verifyCanvasDrop({
      page,
      ui,
      prefix: "mcp",
      readSnapshot: async () =>
        (
          await client.callTool({
            name: "project_open",
            arguments: { projectId: project.id },
          })
        ).structuredContent.result,
    });
    mkdirSync("var/qa", { recursive: true });
    await page.screenshot({ path: "var/qa/mcp-embedded-workbench.png" });
    console.log(
      `PASS: opaque sandbox, models/images, shared Blob lifetime, host download; ${httpFileReads} binary HTTP reads and ${bridgeFileReads} bridge file chunks.`,
    );
    await page.route("**/tool", async (route) => {
      const args = route.request().postDataJSON();
      if (args.arguments?.path !== "/api/tools/studio_open")
        return route.continue();
      const response = await route.fetch();
      const data = await response.json();
      delete data.structuredContent.result.build;
      await route.fulfill({ response, json: data });
    });
    await page.reload();
    await ui.getByText("工作台已升级，需要重新连接", { exact: true }).waitFor();
    assert.equal(
      await ui.getByRole("button", { name: "添加资源", exact: true }).count(),
      0,
    );
    await page.screenshot({ path: "var/qa/mcp-stale-service.png" });
    console.log(
      "PASS: an older MCP connection shows a reconnect screen instead of unsupported resource actions.",
    );
  }
} catch (error) {
  console.error("UI errors:", errors);
  console.error(
    "Saved state:",
    JSON.stringify(
      (
        await client.callTool({
          name: "project_open",
          arguments: { projectId: project.id },
        })
      ).structuredContent.result.canvases,
    ),
  );
  mkdirSync("var/qa", { recursive: true });
  await page.screenshot({ path: "var/qa/mcp-ui-failure.png" });
  throw error;
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
  await client.close();
  binaryService.kill("SIGTERM");
  await once(binaryService, "exit");
  rmSync(dir, { recursive: true, force: true });
}
