import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { randomUUID, createHash } from "node:crypto";
import { mkdirSync, realpathSync, writeFileSync } from "node:fs";
import path from "node:path";
import { z } from "zod";
import { StudioCore, StudioError } from "./core.js";
import {
  CHARACTER_DAG_TEMPLATE,
  MULTIVIEW_SLOT_META,
  type MultiviewSlotKey,
} from "../shared/character-dag.js";
import {
  characterCodexContext,
  viewNodeKey,
} from "../shared/character-workflow-guidance.js";
import {
  characterWorkbenchUrl,
  characterProductionPrompt,
  codexNewSessionUrl,
} from "../shared/codex-collaboration.js";

export function projectChangeVersion(core: StudioCore, projectId: string) {
  const { db, project } = core.db(projectId);
  try {
    const digest = createHash("sha256").update(project.updatedAt);
    digest.update(
      String(
        db.prepare("SELECT COALESCE(MAX(seq), 0) AS seq FROM events").get()!
          .seq,
      ),
    );
    // Some existing edits increment entity revisions without appending an event.
    for (const table of ["assets", "canvases", "runs", "tasks"]) {
      digest.update(
        JSON.stringify(
          db
            .prepare(
              `SELECT id, json_extract(data, '$.revision') AS revision, json_extract(data, '$.graphRevision') AS graphRevision, json_extract(data, '$.layoutRevision') AS layoutRevision FROM ${table} ORDER BY id`,
            )
            .all(),
        ),
      );
    }
    return { version: digest.digest("hex") };
  } finally {
    db.close();
  }
}

export function prepareCharacterSession(
  core: StudioCore,
  projectId: string,
  input: unknown,
  port = Number(process.env.PORT || 4310),
) {
  const args = z
    .object({
      runId: z.string().min(1).max(100),
      nodeKey: z.string().min(1).max(100),
      slotKey: z.string().optional(),
      expectedRevision: z.number().int().nonnegative(),
      referenceRevisionIds: z.array(z.string().min(1).max(100)).max(100),
      prompt: z.string().max(20000),
    })
    .parse(input);
  const snapshot = core.snapshot(projectId);
  const run = snapshot.runs.find((item) => item.id === args.runId);
  if (!run) throw new StudioError(404, "角色制作记录不存在");
  if (run.revision !== args.expectedRevision)
    throw new StudioError(409, "角色进度已更新，请核对当前目标后重新打开会话");
  if (!CHARACTER_DAG_TEMPLATE.some((node) => node.key === args.nodeKey))
    throw new StudioError(400, "角色制作步骤不存在");
  if (
    args.slotKey &&
    (!MULTIVIEW_SLOT_META.some((slot) => slot.key === args.slotKey) ||
      viewNodeKey(args.slotKey as MultiviewSlotKey) !== args.nodeKey)
  )
    throw new StudioError(400, "视图槽位与制作步骤不匹配");
  const context = characterCodexContext(snapshot, run, args.nodeKey, {
    slotKey: args.slotKey as MultiviewSlotKey | undefined,
    referenceRevisionIds: args.referenceRevisionIds,
    prompt: args.prompt,
  });
  for (const revision of context.revisions)
    revision.files.forEach((_file, index) =>
      core.assetFile(projectId, revision.id, index),
    );
  const root = realpathSync(snapshot.project.root);
  const directory = path.join(root, ".game-studio", "codex-collaboration");
  mkdirSync(directory, { recursive: true });
  const relative = path.relative(root, realpathSync(directory));
  if (relative.startsWith("..") || path.isAbsolute(relative))
    throw new StudioError(403, "角色协作目录位于项目之外");
  const contextFile = path.join(directory, `${randomUUID()}.md`);
  const workbenchUrl = characterWorkbenchUrl(
    projectId,
    run.id,
    args.nodeKey,
    port,
  );
  writeFileSync(
    contextFile,
    `${characterProductionPrompt(context)}\n\n右侧工作台：${workbenchUrl}\n`,
    { flag: "wx" },
  );
  const prompt = `请读取角色制作任务文件：${contextFile}\n按照其中指定的角色、目标槽位、固定参考版本和制作要求开始工作，产物与进度回填右侧工作台。之后我会在这个对话中继续调整，右侧会自动同步。\n工作台：${workbenchUrl}`;
  return {
    contextFile,
    workbenchUrl,
    url: codexNewSessionUrl(root, prompt, workbenchUrl),
  };
}

const execute = promisify(execFile);
export async function openCharacterSession(
  core: StudioCore,
  projectId: string,
  input: unknown,
  open: (url: string) => Promise<void> = async (url) => {
    if (process.platform !== "darwin")
      throw new StudioError(503, "当前系统尚未配置 Codex 桌面会话入口");
    try {
      await execute("/usr/bin/open", ["-b", "com.openai.codex", url], {
        timeout: 10000,
        maxBuffer: 4096,
      });
    } catch {
      throw new StudioError(
        503,
        "无法打开 Codex 桌面应用，请确认应用已安装后重试",
      );
    }
  },
) {
  const session = prepareCharacterSession(core, projectId, input);
  await open(session.url);
  return { status: "open_requested", ...session };
}
