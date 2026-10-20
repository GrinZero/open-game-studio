import assert from "node:assert/strict";
import { expect } from "@playwright/test";

export async function verifyCanvasTools({
  page,
  ui = page,
  readSnapshot,
  prefix,
}) {
  const toolbar = ui.getByRole("toolbar", { name: "画布工具", exact: true });
  const select = toolbar.getByRole("button", { name: "选择", exact: true });
  const hand = toolbar.getByRole("button", { name: "抓手", exact: true });
  await expect(select).toHaveAttribute("aria-pressed", "true");
  await toolbar.getByRole("button", { name: "适应画布" }).click();
  await page.waitForTimeout(350);
  const original = await readSnapshot();
  const canvas = original.canvases.find((c) =>
    c.nodes.some((n) => n.type === "note"),
  );
  const saved = async () =>
    (await readSnapshot()).canvases.find((c) => c.id === canvas.id);
  const positions = (c) =>
    c.nodes.map(({ id, position }) => ({ id, position }));
  const transform = () =>
    ui.locator(".react-flow__viewport").evaluate((el) => {
      const m = new DOMMatrix(getComputedStyle(el).transform);
      return { x: m.e, y: m.f, zoom: m.a };
    });
  const drag = async (point, dx = 55, dy = 25) => {
    await page.mouse.move(point.x, point.y);
    await page.mouse.down();
    await page.mouse.move(point.x + dx, point.y + dy, { steps: 10 });
    await page.mouse.up();
  };
  const center = async (locator) => {
    const b = await locator.boundingBox();
    assert.ok(b);
    return { x: b.x + b.width / 2, y: b.y + b.height / 2 };
  };
  const note = canvas.nodes.find((n) => n.type === "note");
  const card = ui.locator(`.react-flow__node[data-id="${note.id}"]`);
  await card.locator(".node-heading").click();
  await hand.click();
  await expect(hand).toHaveAttribute("aria-pressed", "true");
  // Text previews carry nodrag/nowheel; the hand must pan through them as well as images.
  for (const surface of [
    card.locator(".note-body"),
    ui.locator(".asset-node .asset-thumb").first(),
  ]) {
    if (!(await surface.count())) continue;
    const before = await transform();
    const point = await center(surface);
    await drag(point);
    await expect
      .poll(async () => (await transform()).x)
      .toBeCloseTo(before.x + 55, 0);
    await expect
      .poll(async () => (await saved()).viewport.x)
      .toBeCloseTo(before.x + 55, 0);
  }
  const point = await center(card.locator(".note-body"));
  const beforeWheel = await transform();
  await page.mouse.move(point.x, point.y);
  await page.mouse.wheel(0, 60);
  await expect
    .poll(async () => Math.abs((await transform()).y - beforeWheel.y))
    .toBeGreaterThan(10);
  assert.deepEqual(
    positions(await saved()),
    positions(canvas),
    "hand gestures must not move nodes",
  );
  assert.equal((await saved()).graphRevision, canvas.graphRevision);
  await expect(card).toHaveClass(/selected/);
  await page.screenshot({ path: `var/qa/${prefix}-canvas-hand.png` });
  await select.click();
  await expect(select).toHaveAttribute("aria-pressed", "true");
  const beforeSelect = await saved();
  await drag(await center(card.locator(".node-heading")), 35, 20);
  await expect
    .poll(
      async () =>
        (await saved()).nodes.find((n) => n.id === note.id).position.x,
    )
    .toBeGreaterThan(note.position.x + 5);
  await ui.getByRole("button", { name: "撤销布局", exact: true }).click();
  await expect
    .poll(async () => positions(await saved()))
    .toEqual(positions(beforeSelect));
  await toolbar.getByRole("button", { name: "适应画布" }).click();
  console.log(
    `PASS: ${prefix} hand pans and scrolls over cards without changing nodes; selection dragging and undo still work.`,
  );
}
