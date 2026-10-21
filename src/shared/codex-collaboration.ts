import { formatCodexContext, type CodexContext } from "./codex-context";

export function characterWorkbenchUrl(
  projectId: string,
  runId: string,
  nodeKey: string,
  port = 4310,
) {
  const url = new URL(`http://127.0.0.1:${port}/`);
  url.search = new URLSearchParams({
    projectId,
    view: "character",
    runId,
    nodeKey,
  }).toString();
  return url.href;
}

export function characterProductionPrompt(context: CodexContext) {
  return formatCodexContext(
    context,
    `请按以下角色目标和用户需求开始真实制作，并将进度、真实产物回填到右侧工作台的指定位置。用户希望在对话中继续调整，右侧自动更新。只操作这个项目和指定角色，不修改工作台插件源码；遵守原画、前置步骤和审核关卡。生成完成前不要用占位图片或空模型标记成功。`,
  );
}

export function characterSessionInput(context: CodexContext) {
  const target = context.characterTarget;
  if (!target) throw new Error("缺少角色制作目标");
  return {
    runId: target.runId,
    nodeKey: target.nodeKey,
    slotKey: target.slotKey,
    expectedRevision: target.runRevision,
    referenceRevisionIds: context.inputSnapshot.revisionIds,
    prompt: target.request,
  };
}

export function codexNewSessionUrl(
  workspace: string,
  prompt: string,
  browserUrl: string,
) {
  const url = new URL("codex://new");
  url.search = new URLSearchParams({
    path: workspace,
    prompt,
    browserUrl,
  }).toString();
  return url.href;
}
