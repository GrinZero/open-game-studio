import assert from "node:assert/strict";

export async function verifyCharacterSource({
  page,
  ui = page,
  api,
  projectId,
  prefix,
}) {
  const png = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aM2cAAAAASUVORK5CYII=",
    "base64",
  );
  const importImage = async (name) => {
    const form = new FormData();
    form.append("files", new Blob([png], { type: "image/png" }), name);
    return api(`/api/projects/${projectId}/import`, form);
  };
  const first = await importImage("map-reference.png");
  const chosen = await importImage("character-source.png");
  await page.reload();
  await ui.getByRole("button", { name: "角色工作流", exact: true }).click();
  await ui
    .locator(".char-empty-hero")
    .getByRole("button", { name: "新建角色", exact: true })
    .click();
  const drawer = ui
    .locator(".char-drawer-card")
    .filter({ hasText: "选择原画，创建角色" });
  const create = drawer.getByRole("button", { name: "创建角色", exact: true });
  assert.equal(await create.isDisabled(), true);
  assert.equal(
    await drawer.getByLabel("角色原画", { exact: true }).inputValue(),
    "",
  );
  let snap = await api(`/api/projects/${projectId}`);
  assert.equal(snap.runs.length, 0);
  await drawer.getByLabel("角色原画", { exact: true }).selectOption(chosen.id);
  await drawer.getByAltText("已选角色原画预览").waitFor();
  await drawer
    .getByLabel("角色名称", { exact: true })
    .fill(`${prefix}原画验收`);
  await create.click();
  await ui.getByRole("button", { name: "更换原画", exact: true }).waitFor();
  snap = await api(`/api/projects/${projectId}`);
  assert.equal(snap.runs.length, 1);
  assert.equal(snap.runs[0].dagState.views.original.revisionId, chosen.id);
  assert.ok(!snap.runs[0].inputRevisionIds.includes(first.id));
  await ui.getByRole("button", { name: "更换原画", exact: true }).click();
  const picker = ui.getByRole("dialog", { name: "更换原画", exact: true });
  await picker.getByRole("button", { name: /选用 map-reference/ }).click();
  await picker.getByRole("button", { name: "取消", exact: true }).click();
  snap = await api(`/api/projects/${projectId}`);
  assert.equal(snap.runs[0].dagState.views.original.revisionId, chosen.id);
  await ui.getByRole("button", { name: "更换原画", exact: true }).click();
  await picker.getByRole("tab", { name: "与 Codex 生成", exact: true }).click();
  await picker
    .getByLabel("原画生成要求", { exact: true })
    .fill("保留发型，生成新的角色立绘");
  await picker.getByRole("button", { name: /参考 character-source/ }).click();
  const detail = picker.locator("textarea[aria-label='角色制作会话详情']");
  const sourceContext = await detail.inputValue();
  assert.ok(sourceContext.includes(chosen.id));
  assert.ok(!sourceContext.includes(first.id));
  assert.ok(sourceContext.includes('"slotKey":"original"'));
  assert.ok(sourceContext.includes("保留发型，生成新的角色立绘"));
  if (prefix === "mcp") {
    await picker
      .getByRole("button", { name: "发送到当前对话并制作", exact: true })
      .click();
    await picker
      .getByRole("status")
      .filter({ hasText: "已发送到当前 Codex 对话" })
      .waitFor();
    const messages = await page.evaluate(() => window.__modelMessages);
    assert.equal(messages.at(-1).role, "user");
    assert.ok(messages.at(-1).content[0].text.includes(chosen.id));
    assert.ok(messages.at(-1).content[0].text.includes('"slotKey":"original"'));
    assert.ok(messages.at(-1).content[0].text.includes("开始真实制作"));
  } else {
    assert.equal(
      await picker
        .getByRole("button", { name: "发送到当前对话并制作", exact: true })
        .count(),
      0,
    );
    assert.equal(
      await picker
        .getByRole("button", { name: "复制给 Codex", exact: true })
        .count(),
      0,
    );
    let opened;
    await page.route(
      `**/api/projects/${projectId}/character-workflow/open-codex-session`,
      async (route) => {
        opened = route.request().postDataJSON();
        await route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({ status: "open_requested" }),
        });
      },
    );
    await picker
      .getByRole("button", { name: "打开 Codex 会话", exact: true })
      .click();
    await picker
      .getByRole("status")
      .filter({ hasText: "已请求打开 Codex 新对话" })
      .waitFor();
    assert.equal(opened.slotKey, "original");
    assert.ok(opened.referenceRevisionIds.includes(chosen.id));
    assert.ok(!opened.referenceRevisionIds.includes(first.id));
  }
  await page.screenshot({
    path: `var/qa/${prefix}-character-generation-dialog.png`,
  });
  const previousViewport = page.viewportSize();
  await page.setViewportSize({ width: 360, height: 800 });
  assert.ok(
    await picker.evaluate(
      (element) => element.getBoundingClientRect().width <= 360,
    ),
  );
  assert.equal(
    await picker.getByRole("button", { name: "取消", exact: true }).isVisible(),
    true,
  );
  await page.screenshot({
    path: `var/qa/${prefix}-character-dialog-mobile.png`,
  });
  await page.setViewportSize(previousViewport);

  await picker.getByRole("tab", { name: "选择已有原画", exact: true }).click();
  await picker.getByLabel("或上传原画", { exact: true }).setInputFiles({
    name: "replacement-source.png",
    mimeType: "image/png",
    buffer: png,
  });
  await picker
    .getByRole("button", { name: "采用这张原画", exact: true })
    .waitFor({ state: "visible" });
  // Wait for upload selection, not just for the dialog's existing apply button.
  await picker
    .getByLabel("角色原画", { exact: true })
    .locator("option")
    .filter({ hasText: "replacement-source" })
    .waitFor({ state: "attached" });
  const adoptedRevisionId = await picker
    .getByLabel("角色原画", { exact: true })
    .inputValue();
  // Upload finishes with the new revision selected and the apply action enabled.
  await picker
    .getByRole("button", { name: "采用这张原画", exact: true })
    .click();
  await ui.getByRole("button", { name: "更换原画", exact: true }).waitFor();
  snap = await api(`/api/projects/${projectId}`);
  const uploaded =
    snap.assets.find((asset) =>
      asset.revisions.some((rev) =>
        rev.files.some(
          (file) => file.originalName === "replacement-source.png",
        ),
      ),
    ) || snap.assets.find((asset) => asset.name.includes("replacement-source"));
  assert.ok(uploaded);
  assert.equal(adoptedRevisionId, uploaded.latestRevisionId);
  assert.equal(
    snap.runs[0].dagState.views.original.revisionId,
    uploaded.latestRevisionId,
  );
  assert.ok(!snap.runs[0].inputRevisionIds.includes(chosen.id));
  await ui.getByRole("button", { name: "新建角色", exact: true }).click();
  assert.equal(
    await drawer.getByLabel("角色原画", { exact: true }).inputValue(),
    "",
  );
  assert.equal(await create.isDisabled(), true);
  await drawer.getByLabel("或上传原画", { exact: true }).setInputFiles({
    name: "creation-source.png",
    mimeType: "image/png",
    buffer: png,
  });
  await create.click();
  await ui.getByRole("button", { name: "更换原画", exact: true }).waitFor();
  snap = await api(`/api/projects/${projectId}`);
  assert.equal(snap.runs.length, 2);
  assert.ok(
    snap.runs.some(
      (run) =>
        run.dagState.views.original.revisionId !== uploaded.latestRevisionId &&
        run.dagState.views.original.revisionId !== chosen.id,
    ),
  );
  const next = ui.getByRole("region", { name: "现在该做什么", exact: true });
  await next.waitFor();
  assert.ok((await next.innerText()).includes("头部四视图"));
  await next.getByRole("button", { name: /与 Codex 制作头部四视图/ }).click();
  const taskDialog = ui.getByRole("dialog", {
    name: "与 Codex 制作 · 头部四视图生成",
    exact: true,
  });
  await taskDialog.waitFor();
  const taskContext = await taskDialog
    .getByLabel("角色制作会话详情", { exact: true })
    .inputValue();
  assert.ok(taskContext.includes("head_multiview"));
  assert.ok(taskContext.includes("workflow_submit_artifact"));
  await taskDialog
    .getByRole("button", { name: "返回工作台", exact: true })
    .click();
  const stage = ui.getByRole("region", { name: "本阶段操作指引", exact: true });
  assert.ok(
    (await stage.innerText()).includes(
      "先完成：头部四视图生成、身体四视图生成",
    ),
  );
  assert.equal(
    await ui
      .getByRole("button", { name: "确认多视图并进入 3D 候选", exact: true })
      .isDisabled(),
    true,
  );
  await ui.getByRole("tab", { name: /2. 头\/身/ }).click();
  assert.equal(
    await ui
      .getByRole("button", { name: "采用此头部候选", exact: true })
      .first()
      .isDisabled(),
    true,
  );
  assert.ok(
    (await ui.locator(".char-candidates-grid").first().innerText()).includes(
      "等待前置步骤",
    ),
  );
  await ui.getByRole("tab", { name: /1. 多视图/ }).click();
  const activeRunId = await ui
    .getByRole("combobox", { name: "切换角色工作流运行", exact: true })
    .inputValue();
  snap = await api(`/api/projects/${projectId}`);
  const activeRun = snap.runs.find((run) => run.id === activeRunId);
  await api(`/api/projects/${projectId}/character-workflow/edit-view`, {
    runId: activeRunId,
    slotKey: "original",
    revisionId: first.id,
    expectedRevision: activeRun.revision,
  });
  await ui
    .getByRole("region", { name: "现在该做什么", exact: true })
    .getByText("map-reference", { exact: true })
    .waitFor();
  await ui
    .getByLabel("协作同步状态", { exact: true })
    .filter({ hasText: "自动同步" })
    .waitFor();
  await page.screenshot({
    path: `var/qa/${prefix}-character-source.png`,
    fullPage: true,
  });
}
