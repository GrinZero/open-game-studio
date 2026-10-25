import { test } from "node:test";
import assert from "node:assert/strict";

test("窗口内启动目标：同项目切换、重复消息去重、重复打开与跨项目不串目标", async () => {
  Object.defineProperty(globalThis, "location", {
    configurable: true,
    value: { search: "?projectId=island&canvasId=overview" },
  });
  const {
    launchRequest,
    launchProject,
    launchCanvas,
    receiveLaunch,
    subscribeLaunch,
  } = await import("../src/ui/workbench-launch.js");
  try {
    assert.equal(launchProject(), "island");
    assert.equal(launchCanvas("island"), "overview");
    assert.equal(launchCanvas("other"), undefined);
    let notices = 0;
    const unsubscribe = subscribeLaunch(() => notices++);
    const target = {
      projectId: "island",
      runId: "run1",
      nodeKey: "rig",
      view: "character",
    };
    receiveLaunch({ target, navigationId: "open1" });
    assert.equal(launchRequest().nodeKey, "rig");
    assert.equal(notices, 1);
    receiveLaunch({ target, navigationId: "open1" });
    assert.equal(notices, 1);
    receiveLaunch({ target, navigationId: "open2" });
    assert.equal(notices, 2, "再次打开同一目标必须重新定位");
    receiveLaunch({ projectId: "island", assetId: "asset1", revisionId: "v1" });
    assert.equal(launchRequest().runId, undefined);
    assert.equal(launchRequest().revisionId, "v1");
    receiveLaunch({ project: { project: { id: "other" } } });
    assert.deepEqual(launchRequest(), { projectId: "other" });
    assert.equal(notices, 4);
    unsubscribe();
  } finally {
    Reflect.deleteProperty(globalThis, "location");
  }
});
