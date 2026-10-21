import type { ProjectSnapshot, WorkflowRun } from "./types";
import type { MultiviewSlotKey, CharacterStageGroup } from "./character-dag";
import type { CodexContext } from "./codex-context";

export function viewNodeKey(slotKey: MultiviewSlotKey) {
  return slotKey === "original"
    ? "source_illustration"
    : slotKey.startsWith("head_")
      ? "head_multiview"
      : slotKey.startsWith("body_")
        ? "body_multiview"
        : "balance_three_views";
}

function nodeViews(run: WorkflowRun, nodeKey: string) {
  return Object.values(run.dagState?.views || {}).filter(
    (slot) => viewNodeKey(slot.key) === nodeKey,
  );
}

export function nodeGuidance(run: WorkflowRun, key: string) {
  const node = run.dag?.find((item) => item.key === key);
  const blockedBy = (node?.dependsOn || [])
    .map((dependency) => run.dag?.find((item) => item.key === dependency))
    .filter((item) => item && (item.status !== "succeeded" || item.stale));
  const complete = node?.status === "succeeded" && !node.stale;
  const running = node?.status === "running";
  const ready = !!node && !running && !blockedBy.length;
  let reason = complete
    ? "已完成，可以查看结果；需要调整时可在 Codex 会话中继续描述。"
    : running
      ? `正在制作${node?.lease?.message ? `：${node.lease.message}` : "，进度与结果会自动同步。"}`
      : blockedBy.length
        ? `先完成：${blockedBy.map((item) => item!.title).join("、")}。`
        : node?.mode === "review"
          ? "结果已齐备，请检查本阶段的图片或模型，再确认通过。"
          : key === "source_illustration"
            ? "选择已有原画，或打开 Codex 会话生成一张。"
            : "输入已齐备，可以在 Codex 对话中制作，进度与产物会自动同步。";
  const views = nodeViews(run, key);
  const filled = views.filter((slot) => slot.revisionId).length;
  if (!complete && ready && key !== "source_illustration" && filled)
    reason = `已回填 ${filled}/${views.length} 张视图。${filled === views.length ? "请让 Codex 核对并提交整组产物，再进入下一步。" : "继续制作剩余视图，整组完成后提交。"}`;
  return { node, blockedBy, complete, running, ready, reason };
}

export const CHARACTER_STAGE_GUIDANCE: Record<
  CharacterStageGroup,
  { task: string; after: string }
> = {
  reference: {
    task: "先确认原画，再制作头部四视图和身体四视图；两组可以并行。随后制作全身三视图。",
    after:
      "检查人物一致性、正侧背方向及眼线、肩线、腰臀线对齐，通过多视图审核后进入 3D 候选。",
  },
  candidate: {
    task: "多视图审核通过后，分别制作 4 个头部和 4 个身体模型候选。各选一个，点击采用。",
    after: "检查头身搭配与造型，确认选型后进入部件拆解和贴图。",
  },
  texture: {
    task: "依次完成网格拆解、部件分类和 UV 引导图，再制作五组贴图；五组可并行。",
    after: "检查五组贴图和摇摆部件归类，完成 MToon 总装后校准头身比例。",
  },
  proportions: {
    task: "对照全身三视图调整头、发、身的位置和缩放，设置身高，并核对关节标记。",
    after: "确认比例和关节位置后，通过此阶段审核，进入骨骼蒙皮。",
  },
  rig: {
    task: "依次完成骨骼映射、蒙皮权重传递和 90 帧物理检查。",
    after: "预览动作，检查关节变形和穿模；确认后通过动作审核。",
  },
  delivery: {
    task: "动作审核通过后，填写作者、授权等元数据，再制作 VRM 成品包。",
    after: "预览真实 VRM，确认可播放和物理效果后下载交付。",
  },
};

export function characterCodexContext(
  s: ProjectSnapshot,
  run: WorkflowRun,
  nodeKey: string,
  options: {
    slotKey?: MultiviewSlotKey;
    referenceRevisionIds?: string[];
    prompt?: string;
  } = {},
): CodexContext {
  const canvas = s.canvases.find((item) => item.id === run.canvasId);
  const workflowNode = canvas?.nodes.find(
    (item) => item.data.instanceId === run.instanceId,
  );
  const node = run.dag?.find((item) => item.key === nodeKey);
  const title = options.slotKey
    ? run.dagState?.views[options.slotKey].label || nodeKey
    : node?.title || nodeKey;
  const ids = [
    ...new Set(
      options.referenceRevisionIds ?? [
        ...(node?.inputRevisionIds || []),
        ...nodeViews(run, nodeKey).flatMap((slot) =>
          slot.revisionId ? [slot.revisionId] : [],
        ),
        ...(options.slotKey && run.dagState?.views[options.slotKey].revisionId
          ? [run.dagState.views[options.slotKey].revisionId!]
          : []),
      ],
    ),
  ];
  const revisions = ids.map((id) => {
    const asset = s.assets.find((item) =>
      item.revisions.some((revision) => revision.id === id),
    );
    const rev = asset?.revisions.find((revision) => revision.id === id);
    if (!asset || !rev) throw new Error("参考版本已不可用，请刷新后重新选择。");
    return { asset, rev };
  });
  return {
    project: s.project,
    canvas: {
      id: run.canvasId,
      name: canvas?.name || "角色制作",
      graphRevision: canvas?.graphRevision || run.graphRevision,
    },
    selection: workflowNode
      ? [
          {
            ...workflowNode.data,
            nodeId: workflowNode.id,
            title: `${run.dagState?.characterName || "角色"} / ${title}`,
          },
        ]
      : [],
    characterTarget: {
      runId: run.id,
      runRevision: run.revision,
      nodeKey,
      slotKey: options.slotKey,
      title,
      request: [
        options.slotKey && options.slotKey !== "original"
          ? `只制作「${title}」这个槽位对应的一张图片；不要将整组拼图填入单张视图。`
          : undefined,
        node?.description,
        ...(options.referenceRevisionIds === undefined
          ? nodeViews(run, nodeKey)
              .filter((slot) => slot.revisionId)
              .map((slot) => `已有视图：${slot.label} · ${slot.revisionId}`)
          : []),
        run.dagState?.stylePrompt,
        options.prompt?.trim(),
      ]
        .filter(Boolean)
        .join("\n"),
    },
    inputSnapshot: {
      graphRevision: run.graphRevision,
      targetNodeIds: workflowNode ? [workflowNode.id] : [],
      revisionIds: ids,
      prompt: options.prompt || "",
      items: revisions.map(({ asset, rev }) => ({
        type: "asset",
        role: "reference",
        title: asset.name,
        assetId: asset.id,
        revisionId: rev.id,
        version: rev.number,
        kind: asset.kind,
        files: rev.files,
      })),
    },
    revisions: revisions.map(({ asset, rev }) => ({
      id: rev.id,
      assetId: asset.id,
      files: rev.files.map((file) => ({
        name: file.name,
        fileRef: `${s.project.root.replace(/\/$/, "")}/${file.relativePath}`,
      })),
    })),
    runs: [run],
  };
}
