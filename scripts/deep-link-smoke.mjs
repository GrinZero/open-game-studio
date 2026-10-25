import assert from "node:assert/strict";

export async function verifyDeepLinks({
  page,
  ui,
  client,
  project,
  hostOrigin,
  binaryOrigin,
}) {
  const invoke = async (name, args) => {
    const response = await client.callTool({ name, arguments: args });
    assert.ok(!response.isError, JSON.stringify(response));
    return response.structuredContent.result;
  };
  const bridge = async (path, body) =>
    invoke("studio_ui_request", { path, method: "POST", body });
  const snapshot = () => invoke("project_open", { projectId: project.id });
  await page.goto(`${hostOrigin}/?projectId=${project.id}&view=overview`);
  await ui.locator(".workspace").waitFor();
  const canvas = (await snapshot()).canvases[0];
  const file = (text) => ({
    name: "海岛定位验收.md",
    base64: Buffer.from(text).toString("base64"),
  });
  const v1 = await bridge(`/api/projects/${project.id}/import`, {
    files: [file("旧版固定资源")],
    name: "海岛定位验收",
  });
  await bridge(`/api/projects/${project.id}/import`, {
    files: [file("新版资源")],
    assetId: v1.assetId,
  });
  await invoke("canvas_patch", {
    projectId: project.id,
    canvasId: canvas.id,
    expectedRevision: canvas.graphRevision,
    target: "graph",
    ops: [
      {
        op: "add",
        node: {
          id: "far-link-note",
          type: "note",
          position: { x: 25000, y: -15000 },
          data: { title: "远处定位节点", text: "不修改内容" },
        },
      },
    ],
  });
  const run = await bridge(
    `/api/projects/${project.id}/character-workflow/create`,
    { characterName: "定位验收角色" },
  );
  const run2 = await bridge(
    `/api/projects/${project.id}/character-workflow/create`,
    { characterName: "定位验收角色二" },
  );
  const task = await invoke("generation_create", {
    projectId: project.id,
    canvasId: canvas.id,
    expectedRevision: (await snapshot()).canvases[0].graphRevision,
    kind: "image",
    name: "定位验收任务",
    prompt: "只创建草稿",
    inputNodeIds: [],
    position: { x: 400, y: 0 },
    idempotencyKey: "deep-link-task",
  });
  const open = async (args) =>
    invoke("studio_open", { projectId: project.id, ...args });
  const nodeKey = run.dag.find((n) => n.stageGroup === "rig").key;
  const otherNodeKey = run.dag.find((n) => n.stageGroup === "texture").key;
  const stepKey = run.steps[3].key;
  const cases = [
    {
      args: { assetId: v1.assetId, revisionId: v1.id },
      verify: async (surface) => {
        await surface
          .locator(".inspector-resource-name")
          .getByRole("heading", { name: "海岛定位验收", exact: true })
          .waitFor();
        assert.match(
          await surface.locator(".inspector-resource-name").innerText(),
          /v1/,
        );
      },
    },
    {
      args: { nodeId: "far-link-note" },
      verify: async (surface) => {
        await surface
          .locator('.react-flow__node[data-id="far-link-note"].selected')
          .waitFor();
        const bounds = await surface
          .locator('.react-flow__node[data-id="far-link-note"]')
          .evaluate((el) => {
            const node = el.getBoundingClientRect();
            const pane = el.closest(".react-flow").getBoundingClientRect();
            return {
              nx: node.x,
              ny: node.y,
              nr: node.right,
              nb: node.bottom,
              px: pane.x,
              py: pane.y,
              pr: pane.right,
              pb: pane.bottom,
            };
          });
        assert.ok(
          bounds.nx >= bounds.px &&
            bounds.nr <= bounds.pr &&
            bounds.ny >= bounds.py &&
            bounds.nb <= bounds.pb,
          JSON.stringify(bounds),
        );
      },
    },
    {
      args: { runId: run.id, nodeKey },
      verify: async (surface) => {
        await surface
          .locator(`[data-workflow-node="${nodeKey}"][aria-current="step"]`)
          .waitFor();
        assert.equal(
          await surface
            .getByRole("combobox", { name: "切换角色工作流运行" })
            .inputValue(),
          run.id,
        );
      },
    },
    {
      args: { runId: run2.id, nodeKey: otherNodeKey },
      verify: async (surface) => {
        await surface
          .locator(
            `[data-workflow-node="${otherNodeKey}"][aria-current="step"]`,
          )
          .waitFor();
        assert.equal(
          await surface
            .getByRole("combobox", { name: "切换角色工作流运行" })
            .inputValue(),
          run2.id,
        );
      },
    },
    {
      args: { runId: run.id, stepKey },
      verify: async (surface) => {
        await surface
          .locator(
            `[data-workflow-step="${stepKey}"] button[aria-expanded="true"]`,
          )
          .waitFor();
      },
    },
    {
      args: { taskId: task.id },
      verify: async (surface) => {
        await surface
          .getByRole("dialog")
          .getByRole("heading", { name: "定位验收任务", exact: true })
          .waitFor();
      },
    },
  ];
  for (const item of cases) {
    console.log("CHECK deep link:", JSON.stringify(item.args));
    const opened = await open(item.args);
    // Existing MCP app: a late result on the same project must update the view.
    await page.evaluate(
      (result) => window.__sendStudioLaunch({}, result),
      opened,
    );
    await item.verify(ui);
    // Close a task dialog before using the header refresh control.
    if (item.args.taskId)
      await ui.getByRole("dialog").locator('button[aria-label="关闭"]').click();
    // Reload the embedded bundle itself; the host still holds the original HTML.
    await ui.getByRole("button", { name: /重新加载工作台 ·/ }).click();
    await item.verify(ui);
    // Initial tool input before the app renders.
    const search = new URL(opened.url).search;
    await page.goto(`${hostOrigin}/${search}`);
    await item.verify(ui);
    // The exact same link must land on the same target through HTTP.
    await page.goto(`${binaryOrigin}/${search}`);
    await item.verify(page);
    await page.goto(`${hostOrigin}/?projectId=${project.id}`);
    await ui.locator(".workspace").waitFor();
  }
  const viewport = page.viewportSize();
  for (const width of [748, 390]) {
    await page.setViewportSize({ width, height: 900 });
    const narrowNode = await open({ nodeId: "far-link-note" });
    const search = new URL(narrowNode.url).search;
    for (const [origin, surface] of [
      [binaryOrigin, page],
      [hostOrigin, ui],
    ]) {
      await page.goto(`${origin}/${search}`);
      await cases[1].verify(surface);
      assert.equal(
        await surface.locator(".inspector").isVisible(),
        false,
        "窄窗口的节点定位不能被详情栏遮挡",
      );
    }
  }
  await page.setViewportSize(viewport);
  // Repeated requests for the same target must refocus after manual navigation.
  const repeated = await open({ runId: run.id, nodeKey });
  await page.evaluate(
    (result) => window.__sendStudioLaunch({}, result),
    repeated,
  );
  await cases[2].verify(ui);
  await ui.getByRole("button", { name: "资源", exact: true }).click();
  const repeatedAgain = await open({ runId: run.id, nodeKey });
  await page.evaluate(
    (result) => window.__sendStudioLaunch({}, result),
    repeatedAgain,
  );
  await cases[2].verify(ui);
  // Invalid URL must show a concrete error and no unrelated canvas.
  await page.goto(
    `${binaryOrigin}/?projectId=${project.id}&nodeId=missing-node`,
  );
  await page.getByRole("alert").filter({ hasText: "画布节点不存在" }).waitFor();
  assert.equal(await page.locator(".workspace").count(), 0);
  const after = await snapshot();
  assert.equal(
    after.canvases[0].layoutRevision,
    canvas.layoutRevision,
    "自动定位不得改写保存的视口",
  );
  assert.equal(
    after.tasks.find((t) => t.id === task.id).status,
    "draft",
    "打开任务不得启动生成",
  );
  await page.goto(`${hostOrigin}/?projectId=${project.id}`);
  await ui.locator(".workspace").waitFor();
  // An old connected service must offer the same exact target in its browser fallback.
  await page.route("**/tool", async (route) => {
    const args = route.request().postDataJSON();
    if (args.arguments?.path !== "/api/tools/studio_open")
      return route.continue();
    const response = await route.fetch();
    const data = await response.json();
    delete data.structuredContent.result.build;
    await route.fulfill({ response, json: data });
  });
  const fallback = await open({ runId: run.id, nodeKey });
  await page.goto(`${hostOrigin}/${new URL(fallback.url).search}`);
  await ui.getByRole("button", { name: "打开新版工作台", exact: true }).click();
  await page.waitForFunction(() => window.__openedLinks.length > 0);
  const openedLink = await page.evaluate(() => window.__openedLinks.at(-1));
  assert.equal(new URL(openedLink).searchParams.get("runId"), run.id);
  assert.equal(new URL(openedLink).searchParams.get("nodeKey"), nodeKey);
  await page.unroute("**/tool");
  await page.goto(`${hostOrigin}/?projectId=${project.id}`);
  await ui.locator(".workspace").waitFor();
  console.log(
    "PASS: HTTP/MCP asset version, offscreen node focus, DAG steps across runs, macro steps, tasks, embedded reload, repeat navigation and invalid links.",
  );
}
