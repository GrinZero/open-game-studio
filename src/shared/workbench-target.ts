import type { ProjectSnapshot } from "./types.js";

export const workbenchViews = [
  "overview",
  "assets",
  "canvas",
  "character",
  "runs",
  "settings",
] as const;
export type WorkbenchView = (typeof workbenchViews)[number];
export const targetFields = [
  "projectId",
  "view",
  "canvasId",
  "nodeId",
  "assetId",
  "revisionId",
  "runId",
  "nodeKey",
  "stepKey",
  "taskId",
] as const;
export interface WorkbenchTarget {
  projectId?: string;
  view?: WorkbenchView;
  canvasId?: string;
  nodeId?: string;
  assetId?: string;
  revisionId?: string;
  runId?: string;
  nodeKey?: string;
  stepKey?: string;
  taskId?: string;
}
export function readWorkbenchTarget(value: unknown): WorkbenchTarget {
  const target: Record<string, string> = {};
  if (value && typeof value === "object") {
    for (const field of targetFields) {
      const entry = (value as Record<string, unknown>)[field];
      if (typeof entry === "string") target[field] = entry;
    }
  }
  return target as WorkbenchTarget;
}
export function targetQuery(target: WorkbenchTarget) {
  const query = new URLSearchParams();
  for (const field of targetFields)
    if (target[field]) query.set(field, target[field]!);
  return query;
}

// Shared by the tool and browser so invalid URLs cannot silently open another item.
export function resolveWorkbenchTarget(
  snapshot: ProjectSnapshot,
  input: WorkbenchTarget,
): WorkbenchTarget {
  const t = { ...readWorkbenchTarget(input), projectId: snapshot.project.id };
  const fail = (message: string): never => {
    throw new Error(message);
  };
  for (const field of targetFields) {
    if (
      input[field] !== undefined &&
      (!input[field] || input[field]!.length > 100)
    )
      fail(`打开参数无效：${field}`);
  }
  if (input.projectId && input.projectId !== snapshot.project.id)
    fail("打开目标不属于此项目");
  if (snapshot.project.archived) fail("项目已归档");
  if (t.view && !workbenchViews.includes(t.view)) fail("页面不存在");
  if ((t.nodeKey || t.stepKey) && !t.runId) fail("打开工作流步骤需要 runId");
  if (t.nodeKey && t.stepKey) fail("请指定一种工作流步骤");
  if (t.runId && (t.assetId || t.revisionId || t.taskId))
    fail("工作流与资源目标不能混用");
  const run = t.runId ? snapshot.runs.find((r) => r.id === t.runId) : undefined;
  if (t.runId && !run) fail("工作流运行不存在");
  const task = t.taskId
    ? snapshot.tasks?.find((task) => task.id === t.taskId)
    : undefined;
  if (t.taskId && !task) fail("资源制作任务不存在");
  for (const owner of [run, task]) {
    if (!owner) continue;
    if (t.canvasId && t.canvasId !== owner.canvasId) fail("目标与画布不匹配");
    t.canvasId = owner.canvasId;
  }
  if (t.nodeId && !t.canvasId) {
    const matches = snapshot.canvases.filter(
      (c) => !c.archived && c.nodes.some((n) => n.id === t.nodeId),
    );
    if (matches.length !== 1)
      fail(matches.length ? "节点存在多处，请指定 canvasId" : "画布节点不存在");
    t.canvasId = matches[0].id;
  }
  const canvas = t.canvasId
    ? snapshot.canvases.find((c) => c.id === t.canvasId && !c.archived)
    : undefined;
  if (t.canvasId && !canvas) fail("画布不存在或已归档");
  const node = t.nodeId
    ? canvas?.nodes.find((n) => n.id === t.nodeId)
    : undefined;
  if (t.nodeId && !node) fail("画布节点不存在");
  if (node && run && node.data.instanceId !== run.instanceId)
    fail("节点与工作流不匹配");
  if (node && task && node.data.taskId !== task.id)
    fail("节点与资源制作任务不匹配");
  if (t.revisionId && !t.assetId && !node) {
    t.assetId = snapshot.assets.find((a) =>
      a.revisions.some((r) => r.id === t.revisionId),
    )?.id;
    if (!t.assetId) fail("资源版本不存在");
  }
  if (node && (t.assetId || t.revisionId)) {
    if (t.assetId && node.data.assetId !== t.assetId)
      fail("资源与画布节点不匹配");
    if (t.revisionId && node.data.revisionId !== t.revisionId)
      fail("资源版本与画布节点不匹配");
    t.assetId = node.data.assetId;
  }
  const asset = t.assetId
    ? snapshot.assets.find((a) => a.id === t.assetId && !a.archived)
    : undefined;
  if (t.assetId && !asset) fail("资源不存在或已归档");
  if (asset) {
    t.revisionId ||= asset.latestRevisionId;
    if (!asset.revisions.some((r) => r.id === t.revisionId))
      fail("资源版本不属于指定资源");
  }
  // Prefer the exact saved usage of a resource when it has one unambiguous
  // location. Never substitute another revision merely to find a canvas.
  if (asset && !t.canvasId) {
    const usages = snapshot.canvases
      .filter((c) => !c.archived)
      .flatMap((c) =>
        c.nodes
          .filter(
            (n) =>
              n.data.assetId === asset.id && n.data.revisionId === t.revisionId,
          )
          .map((n) => ({ canvasId: c.id, nodeId: n.id })),
      );
    const canvases = new Set(usages.map((usage) => usage.canvasId));
    if (canvases.size === 1) t.canvasId = usages[0].canvasId;
    if (usages.length === 1) t.nodeId = usages[0].nodeId;
  }
  if (t.nodeKey && !run?.dag?.some((n) => n.key === t.nodeKey))
    fail("工作流节点不存在");
  if (t.stepKey && !run?.steps.some((s) => s.key === t.stepKey))
    fail("工作流阶段不存在");
  const requiredView = t.nodeKey
    ? "character"
    : t.stepKey || t.nodeId || t.assetId
      ? "canvas"
      : undefined;
  if (requiredView && t.view && t.view !== requiredView)
    fail("页面与打开目标不匹配");
  t.view ||= requiredView || (run ? "character" : task ? "runs" : "canvas");
  if (run && !["canvas", "character"].includes(t.view))
    fail("工作流需要打开画布或角色工作流页面");
  if (task && !["canvas", "runs"].includes(t.view))
    fail("资源制作任务需要打开画布或运行记录");
  if (t.view === "canvas" && !t.canvasId)
    t.canvasId = snapshot.canvases.find((c) => !c.archived)?.id;
  if (t.view === "canvas" && !t.canvasId) fail("项目缺少可用画布");
  return t;
}
