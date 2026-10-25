import { randomUUID } from "node:crypto";
import {
  resolveWorkbenchTarget,
  targetQuery,
  workbenchViews,
} from "../shared/workbench-target.js";
import { studioBuild } from "../shared/build.js";
import { z } from "zod";
import { StudioCore, StudioError } from "./core.js";
import { workspaceProject } from "./workspace-project.js";
import {
  generationCapabilities,
  startGeneration,
  controlGeneration,
} from "./generation.js";
import {
  CHARACTER_DAG_TEMPLATE,
  type CharacterDagNode,
} from "../shared/character-dag.js";
import {
  claimDagNode,
  reportDagNodeProgress,
  submitDagNodeOutput,
  advanceDagNode,
  autoAdvanceToNextGate,
  selectDagCandidate,
  importCharakitArchive,
  listClaimableDagNodes,
} from "./character-workflow.js";
const id = z.string().min(1).max(100);
export const tools = {
  studio_open: {
    description:
      "打开 Open Game Studio 工作台。自动匹配当前会话工作区；调用时请传当前会话的绝对工作目录 workspaceRoot，或显式 projectId；可传 view、canvasId、nodeId、assetId/revisionId、runId + nodeKey（DAG）或 stepKey（阶段）、taskId 精确定位。返回 navigationStatus=requested 仅表示已发出打开目标，须验证实际界面。",
    schema: z.object({
      projectId: id.optional(),
      canvasId: id.optional(),
      view: z.enum(workbenchViews).optional(),
      nodeId: id.optional(),
      assetId: id.optional(),
      revisionId: id.optional(),
      runId: id.optional(),
      nodeKey: id.optional(),
      stepKey: id.optional(),
      taskId: id.optional(),
      workspaceRoot: z.string().max(4096).optional(),
      workspaceRoots: z.array(z.string().max(4096)).max(32).optional(),
    }),
  },
  project_manage: {
    description:
      "创建、关联、更新或重定位本地项目。封面使用项目内图片的固定版本与文件索引，null 移除封面。移除登记不会删除外部文件。",
    schema: z.object({
      action: z.enum(["create", "link", "update", "relocate"]),
      projectId: id.optional(),
      name: z.string().optional(),
      root: z.string().optional(),
      engine: z.enum(["godot", "unity", "unreal", "other"]).optional(),
      collection: z.string().optional(),
      archived: z.boolean().optional(),
      cover: z
        .object({
          revisionId: id,
          fileIndex: z.number().int().nonnegative(),
        })
        .nullable()
        .optional(),
    }),
  },
  project_open: {
    description: "读取显式指定项目的资源、画布、运行、审核与导出快照。",
    schema: z.object({ projectId: id }),
  },
  asset_search: {
    description: "检索项目资源，分页返回摘要，latest 不代表工作流已采用版本。",
    schema: z.object({
      projectId: id,
      query: z.string().default(""),
      kind: z.string().optional(),
      cursor: z.number().int().min(0).default(0),
      limit: z.number().int().min(1).max(100).default(30),
    }),
  },
  document_save: {
    description:
      "创建或编辑 Markdown 文档，保存不可变版本并更新指定画布节点，其他画布保留原版本。",
    schema: z.object({
      projectId: id,
      canvasId: id,
      nodeId: id.optional(),
      expectedRevision: z.number().int(),
      assetId: id.optional(),
      expectedLatestRevisionId: id.optional(),
      name: z.string().min(1).max(120),
      text: z.string().max(300000),
      position: z.object({ x: z.number(), y: z.number() }).optional(),
    }),
  },
  generation_capabilities: {
    description: "查询工作台资源制作的本机 Codex 执行入口。",
    schema: z.object({}),
  },
  generation_create: {
    description:
      "创建图片、视频、模型、骨架动画或音频任务，冻结参考版本和参数并添加画布节点。尚不提交生成。",
    schema: z.object({
      projectId: id,
      canvasId: id,
      expectedRevision: z.number().int(),
      kind: z.enum(["image", "video", "model", "animation", "audio"]),
      name: z.string().min(1).max(120),
      prompt: z.string().min(1).max(30000),
      inputNodeIds: z.array(id).default([]),
      idempotencyKey: id,
      position: z.object({ x: z.number(), y: z.number() }),
      settings: z
        .object({
          aspectRatio: z.enum(["1:1", "16:9", "9:16", "4:3"]).optional(),
          duration: z.number().int().min(1).max(60).optional(),
          animation: z.string().max(100).optional(),
          transparent: z.boolean().optional(),
        })
        .default({}),
    }),
  },
  generation_get: {
    description: "读取持久资源任务、冻结输入与实际参考路径。",
    schema: z.object({ projectId: id, taskId: id }),
  },
  generation_start: {
    description:
      "一次性启动草稿任务，通过本机 Codex 和已配置生成工具制作。重启和重复点击不重复提交。",
    schema: z.object({
      projectId: id,
      taskId: id,
      expectedRevision: z.number().int(),
    }),
  },
  generation_control: {
    description: "停止本地制作，保留任务和产物；远端已提交工作需按原 ID 查询。",
    schema: z.object({
      projectId: id,
      taskId: id,
      expectedRevision: z.number().int(),
      action: z.enum(["cancel", "interrupt"]),
    }),
  },
  generation_adopt: {
    description:
      "采用已导入的固定资源版本作为真实任务结果，用于中断恢复或人工制作。验证类型与完整性，动画验证骨架与动作。",
    schema: z.object({
      projectId: id,
      taskId: id,
      expectedRevision: z.number().int(),
      revisionId: id,
    }),
  },
  generation_complete: {
    description:
      "将真实制作结果验证入库并回填原画布节点。动画必须有骨架和动画片段。",
    schema: z.object({
      projectId: id,
      taskId: id,
      expectedRevision: z.number().int(),
      fileRefs: z.array(z.string()).min(1).max(12),
      summary: z.string().max(30000).default(""),
    }),
  },
  asset_import: {
    description:
      "从已登记项目范围导入文件，或为指定资源创建不可变新版本。外部文件先通过 UI 上传或复制进项目。",
    schema: z.object({
      projectId: id,
      fileRefs: z.array(z.string()).min(1).max(12),
      assetId: id.optional(),
      name: z.string().optional(),
      parentRevisionIds: z.array(id).optional(),
      runId: id.optional(),
    }),
  },
  asset_manage: {
    description: "更新资源名称、标签或回收状态；不删除版本和引用。",
    schema: z.object({
      projectId: id,
      assetId: id,
      expectedRevision: z.number().int(),
      name: z.string().optional(),
      tags: z.array(z.string()).optional(),
      archived: z.boolean().optional(),
    }),
  },
  asset_inspect: {
    description: "读取资源的全部版本和跨画布使用位置。",
    schema: z.object({ projectId: id, assetId: id }),
  },
  canvas_open: {
    description: "读取画布语义、布局、视口和独立版本号。",
    schema: z.object({ projectId: id, canvasId: id }),
  },
  canvas_manage: {
    description:
      "创建、复制、改名或归档画布；复制保留资源引用，不复制可执行制作历史。",
    schema: z.object({
      projectId: id,
      action: z.enum(["create", "update", "duplicate"]),
      canvasId: id.optional(),
      name: z.string().optional(),
      archived: z.boolean().optional(),
      expectedRevision: z.number().int().optional(),
    }),
  },
  canvas_patch: {
    description:
      "提交 CAS 画布变更。reference 连线绑定上游固定版本与便签输入；instantiate 支持 inputNodeIds 或 inputRevisionIds。graph 支持 add/remove/note/connect/disconnect/instantiate；note 可更新 title 或 text，省略字段保留原值；layout 支持 move/resize/viewport；resize 使用 id、width（160–4000）、height（120–4000），宽高同时为 null 恢复默认。不执行生成。",
    schema: z.object({
      projectId: id,
      canvasId: id,
      target: z.enum(["graph", "layout"]),
      expectedRevision: z.number().int(),
      ops: z.array(z.record(z.unknown())),
    }),
  },
  workflow_prepare: {
    description:
      "解析制作分组的上游连线，冻结资产版本、文档和便签输入，返回 inputSnapshot、执行方式、能力、费用与 graphRevision。使用人工导入路线。",
    schema: z.object({ projectId: id, canvasId: id, instanceId: id }),
  },
  workflow_start: {
    description: "幂等启动已准备的角色制作计划，返回持久运行与下一阶段。",
    schema: z.object({
      projectId: id,
      planId: id,
      expectedRevision: z.number().int(),
      idempotencyKey: z.string(),
    }),
  },
  workflow_get: {
    description: "读取运行快照、待处理步骤及审核历史。",
    schema: z.object({ projectId: id, runId: id }),
  },
  workflow_decide: {
    description:
      "采用审核输入中明确指定的版本，或记录具体返修意见。Agent 只能在用户已有授权的范围内设置 actor=delegated_agent。",
    schema: z.object({
      projectId: id,
      runId: id,
      stepKey: z.enum(["appearance", "deformation"]),
      revisionId: id,
      expectedRevision: z.number().int(),
      decision: z.enum(["accept", "request_changes"]),
      comment: z.string().default(""),
      actor: z.enum(["user", "delegated_agent"]).default("user"),
    }),
  },
  workflow_submit_artifact: {
    description:
      "将已导入的不可变资源版本提交给人工或 Agent 阶段。检查所需文件类型，Blend 可编辑性需实际制作工具验证。",
    schema: z.object({
      projectId: id,
      runId: id,
      stepKey: id,
      revisionIds: z.array(id),
      expectedRevision: z.number().int(),
    }),
  },
  job_control: {
    description:
      "取消本地运行、从明确阶段建立局部重跑，或执行角色工作流 DAG 的认领(claim)/进度(progress)/推进(advance)/候选切换(select_candidate)/导入(import_charakit)。",
    schema: z.object({
      projectId: id,
      runId: id,
      action: z.enum([
        "cancel",
        "rerun",
        "claim_dag_node",
        "progress_dag_node",
        "advance_dag_node",
        "auto_advance_dag",
        "select_candidate",
        "import_charakit",
      ]),
      expectedRevision: z.number().int().optional(),
      fromStep: id.optional(),
      inputRevisionIds: z.array(id).optional(),
      idempotencyKey: z.string().optional(),
      nodeKey: z.string().optional(),
      agentId: z.string().optional(),
      agentName: z.string().optional(),
      progress: z.number().optional(),
      message: z.string().optional(),
      source: z.enum(["head", "body"]).optional(),
      candidateId: z.string().optional(),
      archivePath: z.string().optional(),
    }),
  },
  export_bundle: {
    description:
      "导出固定资源版本和 manifest；校验文件完整性，引擎验收保持 not_verified。运行导出必须满足当前变形审核。",
    schema: z.object({
      projectId: id,
      revisionIds: z.array(id),
      runId: id.optional(),
      expectedRevision: z.number().int().optional(),
      idempotencyKey: z.string(),
    }),
  },
  selection_context: {
    description:
      "获取明确选择集的项目、画布、资源固定版本及制作上下文；不自动发起 Agent 任务。",
    schema: z.object({ projectId: id, canvasId: id, nodeIds: z.array(id) }),
  },
} as const;
export type ToolName = keyof typeof tools;
export function dispatch(
  core: StudioCore,
  name: string,
  args: unknown,
): unknown {
  if (!Object.hasOwn(tools, name)) throw new StudioError(404, "工具不存在");
  const a = tools[name as ToolName].schema.parse(args) as Record<string, any>;
  switch (name) {
    case "studio_open": {
      const roots = a.workspaceRoot
        ? [a.workspaceRoot]
        : a.workspaceRoots || [];
      if (roots.length && !a.projectId) {
        core.discoverWorkspaces(roots);
      }
      const projects = core.listProjects();
      const projectId = a.projectId || workspaceProject(projects, roots);
      const project = projectId ? core.snapshot(projectId) : undefined;
      const hasTarget = [
        "canvasId",
        "nodeId",
        "assetId",
        "revisionId",
        "runId",
        "nodeKey",
        "stepKey",
        "taskId",
        "view",
      ].some((key) => a[key]);
      if (hasTarget && !project)
        throw new StudioError(400, "打开目标需要明确项目或匹配的工作区");
      let target;
      try {
        target = project ? resolveWorkbenchTarget(project, a) : undefined;
      } catch (error) {
        throw new StudioError(
          400,
          error instanceof Error ? error.message : String(error),
        );
      }
      const query = targetQuery(target || {});
      return {
        schemaVersion: 1,
        build: studioBuild,
        workspaceRoot: a.workspaceRoot,
        projects,
        project,
        ...target,
        target,
        navigationId: randomUUID(),
        navigationStatus: "requested",
        url: `http://127.0.0.1:${process.env.PORT || 4310}${query.size ? `/?${query}` : ""}`,
        capabilities: {
          webUI: true,
          mcpTools: true,
          embeddedApp: "registered",
          canvasInputs: true,
          provider: false,
          engine: "not_verified",
        },
      };
    }
    case "project_manage":
      return core.projectManage(a);
    case "project_open":
      return core.snapshot(a.projectId);
    case "asset_search": {
      const list = core
        .snapshot(a.projectId)
        .assets.filter(
          (x) =>
            !x.archived &&
            (!a.kind || x.kind === a.kind) &&
            `${x.name} ${x.tags.join(" ")}`
              .toLowerCase()
              .includes(a.query.toLowerCase()),
        );
      return {
        assets: list
          .slice(a.cursor, a.cursor + a.limit)
          .map(({ revisions, ...x }) => ({
            ...x,
            versionCount: revisions.length,
          })),
        cursor: a.cursor + a.limit < list.length ? a.cursor + a.limit : null,
        total: list.length,
      };
    }
    case "document_save":
      return core.documentSave(a);
    case "generation_capabilities":
      return generationCapabilities();
    case "generation_create":
      return core.generationCreate(a);
    case "generation_get":
      return core.generationGet(a.projectId, a.taskId);
    case "generation_start":
      return startGeneration(core, a as any);
    case "generation_control":
      return controlGeneration(core, a as any);
    case "generation_adopt":
      return core.generationAdopt(a as any);
    case "generation_complete":
      return core.generationComplete(
        a.projectId,
        a.taskId,
        a.expectedRevision,
        a.fileRefs,
        a.summary,
      );
    case "asset_import":
      return core.importPaths(a.projectId, a.fileRefs, a);
    case "asset_manage":
      return core.assetManage(a);
    case "asset_inspect": {
      const s = core.snapshot(a.projectId);
      const asset = s.assets.find((x) => x.id === a.assetId);
      if (!asset) throw new StudioError(404, "资源不存在");
      return {
        asset,
        references: s.canvases.flatMap((c) =>
          c.nodes
            .filter((n) => n.data.assetId === a.assetId)
            .map((n) => ({
              canvasId: c.id,
              canvasName: c.name,
              nodeId: n.id,
              revisionId: n.data.revisionId,
            })),
        ),
      };
    }
    case "canvas_open": {
      const c = core
        .snapshot(a.projectId)
        .canvases.find((c) => c.id === a.canvasId);
      if (!c) throw new StudioError(404, "画布不存在");
      return c;
    }
    case "canvas_manage":
      return core.canvasManage(a);
    case "canvas_patch":
      return core.canvasPatch(a);
    case "workflow_prepare":
      return core.prepare(a);
    case "workflow_start":
      return core.start(a);
    case "workflow_get": {
      const s = core.snapshot(a.projectId);
      const r = s.runs.find((r) => r.id === a.runId);
      if (!r) throw new StudioError(404, "运行不存在");
      return {
        run: r,
        executionInputs: core.runInputs(a.projectId, r),
        reviews: s.reviews.filter((d) => d.runId === r.id),
        nextAction: r.steps.find((s) => s.status.startsWith("awaiting_")),
        claimableDagNodes: listClaimableDagNodes(core, {
          projectId: a.projectId,
          runId: r.id,
        }).nodes.map((x: { node: CharacterDagNode }) => x.node),
      };
    }
    case "workflow_decide":
      return core.decide(a);
    case "workflow_submit_artifact": {
      const isDagNodeKey = CHARACTER_DAG_TEMPLATE.some(
        (n) => n.key === a.stepKey,
      );
      if (isDagNodeKey) {
        return submitDagNodeOutput(core, {
          projectId: a.projectId,
          runId: a.runId,
          nodeKey: a.stepKey,
          revisionIds: a.revisionIds,
          expectedRevision: a.expectedRevision,
        });
      }
      return core.submit(a);
    }
    case "job_control": {
      if (a.action === "cancel")
        return core.cancel(
          a.projectId,
          a.runId,
          z.number().int().parse(a.expectedRevision),
        );
      if (a.action === "rerun") return core.rerun(a);
      if (a.action === "claim_dag_node") return claimDagNode(core, a);
      if (a.action === "progress_dag_node")
        return reportDagNodeProgress(core, a);
      if (a.action === "advance_dag_node") return advanceDagNode(core, a);
      if (a.action === "auto_advance_dag")
        return autoAdvanceToNextGate(core, a);
      if (a.action === "select_candidate") return selectDagCandidate(core, a);
      if (a.action === "import_charakit") return importCharakitArchive(core, a);
      return core.rerun(a);
    }
    case "export_bundle":
      return core.exportBundle(a);
    case "selection_context":
      return core.context(a.projectId, a.canvasId, a.nodeIds);
  }
}
