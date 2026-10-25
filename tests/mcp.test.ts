import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import path from "node:path";
import os from "node:os";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { ListRootsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { pathToFileURL } from "node:url";
import { DatabaseSync } from "node:sqlite";
test("真实 stdio MCP 握手、工具发现与 HTTP 同领域项目状态", async () => {
  const dir = mkdtempSync(path.join(os.tmpdir(), "studio-mcp-"));
  const client = new Client(
    { name: "studio-test", version: "1.0.0" },
    { capabilities: { roots: {} } },
  );
  let roots: { uri: string }[] = [];
  client.setRequestHandler(ListRootsRequestSchema, async () => ({ roots }));
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: ["./node_modules/tsx/dist/cli.mjs", "src/server/mcp.ts"],
    env: {
      ...Object.fromEntries(
        Object.entries(process.env).filter(
          (entry): entry is [string, string] => entry[1] !== undefined,
        ),
      ),
      STUDIO_HOME: path.join(dir, "registry"),
      CODEX_HOME: dir,
    },
    stderr: "pipe",
  });
  try {
    await client.connect(transport);
    const list = await client.listTools();
    for (const name of [
      "document_save",
      "generation_create",
      "generation_start",
      "generation_get",
      "generation_adopt",
      "generation_complete",
    ])
      assert.ok(list.tools.some((t) => t.name === name));
    assert.equal(list.tools.length, 27);
    assert.ok(list.tools.some((t) => t.name === "workflow_decide"));
    const openTool = list.tools.find((t) => t.name === "studio_open")!;
    for (const field of [
      "view",
      "canvasId",
      "nodeId",
      "assetId",
      "revisionId",
      "runId",
      "nodeKey",
      "stepKey",
      "taskId",
    ]) {
      assert.ok(
        openTool.inputSchema.properties?.[field],
        `studio_open exposes ${field}`,
      );
    }
    assert.equal(openTool.icons?.[0].mimeType, "image/svg+xml");
    assert.equal(
      Buffer.from(openTool.icons![0].src.split(",")[1], "base64").toString(),
      readFileSync("assets/icon.svg", "utf8"),
    );
    assert.deepEqual((openTool._meta?.["openai/ui"] as any).entrypoints, [
      { type: "global" },
      { type: "thread" },
    ]);
    const resource = await client.readResource({
      uri: "ui://game-studio/workbench",
    });
    const view = resource.contents[0];
    assert.equal(view.mimeType, "text/html;profile=mcp-app");
    assert.ok("text" in view && view.text.includes('<div id="root"></div>'));
    assert.deepEqual((view._meta?.ui as any).csp.resourceDomains, [
      "blob:",
      "data:",
    ]);
    assert.ok(
      (view._meta?.ui as any).csp.connectDomains.includes(
        `http://127.0.0.1:${Number(process.env.PORT || 4310)}`,
      ),
    );
    assert.deepEqual(
      (list.tools.find((t) => t.name === "studio_ui_request")!._meta?.ui as any)
        .visibility,
      ["app"],
    );
    const opened = await client.callTool({
      name: "studio_open",
      arguments: {},
    });
    assert.equal(opened.isError, undefined);
    assert.deepEqual((opened.structuredContent as any).result.projects, []);
    const result = await client.callTool({
      name: "project_manage",
      arguments: {
        action: "create",
        name: "MCP 项目",
        root: path.join(dir, "game"),
      },
    });
    const p = (result.structuredContent as any).result;
    assert.ok(p.id);
    const threadId = "01900000-0000-7000-8000-000000000001";
    const state = new DatabaseSync(path.join(dir, "state_5.sqlite"));
    state.exec("CREATE TABLE threads(id TEXT PRIMARY KEY, cwd TEXT NOT NULL)");
    state.prepare("INSERT INTO threads VALUES (?, ?)").run(threadId, p.root);
    state.close();
    const nativeOpen = await client.callTool({
      name: "studio_open",
      arguments: {},
      _meta: { threadId },
    });
    assert.equal(
      (nativeOpen.structuredContent as any).result.project.project.id,
      p.id,
    );
    const nativeBridge = await client.callTool({
      name: "studio_ui_request",
      arguments: { path: "/api/tools/studio_open", method: "POST", body: {} },
      _meta: { threadId },
    });
    assert.equal(
      (nativeBridge.structuredContent as any).result.project.project.id,
      p.id,
    );
    roots = [{ uri: pathToFileURL(p.root).href }];
    const automatic = await client.callTool({
      name: "studio_open",
      arguments: {},
    });
    assert.equal(
      (automatic.structuredContent as any).result.project.project.id,
      p.id,
    );
    assert.ok(
      (automatic.structuredContent as any).result.url.includes(
        `projectId=${p.id}`,
      ),
    );
    const automaticBridge = await client.callTool({
      name: "studio_ui_request",
      arguments: { path: "/api/tools/studio_open", method: "POST", body: {} },
    });
    assert.equal(
      (automaticBridge.structuredContent as any).result.project.project.id,
      p.id,
    );
    const snapshot = await client.callTool({
      name: "project_open",
      arguments: { projectId: p.id },
    });
    assert.equal((snapshot.structuredContent as any).result.canvases.length, 1);
    const bridge = async (args: Record<string, unknown>) => {
      const response = await client.callTool({
        name: "studio_ui_request",
        arguments: args,
      });
      assert.ok(!response.isError, JSON.stringify(response));
      return (response.structuredContent as any).result;
    };
    const fileBytes = Buffer.from("角色参考\n".repeat(70000));
    const imported = await bridge({
      path: `/api/projects/${p.id}/import`,
      method: "POST",
      body: {
        files: [
          { name: "reference.txt", base64: fileBytes.toString("base64") },
        ],
        name: "桥接资料",
      },
    });
    const route = `/api/projects/${p.id}/files/${imported.id}/0`;
    const first = await bridge({ path: route, method: "GET", offset: 0 });
    assert.equal(first.next, 512 * 1024);
    assert.equal(first.eof, false);
    const second = await bridge({
      path: route,
      method: "GET",
      offset: first.next,
    });
    assert.equal(second.eof, true);
    assert.deepEqual(
      Buffer.concat([
        Buffer.from(first.base64, "base64"),
        Buffer.from(second.base64, "base64"),
      ]),
      fileBytes,
    );
    const forbidden = await client.callTool({
      name: "studio_ui_request",
      arguments: { path: "/api/../../etc/passwd", method: "GET" },
    });
    assert.equal(forbidden.isError, true);
    const invalidPicker = await client.callTool({
      name: "studio_ui_request",
      arguments: {
        path: "/api/directories/pick",
        method: "POST",
        body: { mode: "invalid" },
      },
    });
    assert.equal(invalidPicker.isError, true);
    const invalid = await client.callTool({
      name: "project_open",
      arguments: { projectId: "missing" },
    });
    assert.equal(invalid.isError, true);
  } finally {
    await client.close();
    rmSync(dir, { recursive: true, force: true });
  }
});
