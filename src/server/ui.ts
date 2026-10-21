import { openSync, readSync, closeSync, statSync } from "node:fs";
import { z } from "zod";
import { StudioCore, StudioError } from "./core.js";
import { dispatch } from "./operations.js";
import {
  discoverLocalCharakuruProjects,
  dispatchCharacterWorkflowRoute,
  listClaimableDagNodes,
} from "./character-workflow.js";
import { pickDirectory, pickFiles } from "./directories.js";
import { assetHttpUrl } from "./asset-http.js";
import {
  openCharacterSession,
  projectChangeVersion,
} from "./codex-collaboration.js";

export const uiRequestSchema = {
  path: z.string().max(400),
  method: z.enum(["GET", "POST"]),
  body: z.unknown().optional(),
  offset: z.number().int().nonnegative().optional(),
};
export async function uiRequest(
  core: StudioCore,
  args: {
    path: string;
    method: "GET" | "POST";
    body?: unknown;
    offset?: number;
  },
) {
  const { path: route, method, body } = args;
  if (method === "POST" && route === "/api/directories/pick")
    return pickDirectory(body);
  if (method === "POST" && route === "/api/files/pick") return pickFiles(body);
  if (method === "GET" && route === "/api/projects") return core.listProjects();
  const version = route.match(/^\/api\/projects\/([\w-]+)\/change-version$/);
  if (method === "GET" && version)
    return projectChangeVersion(core, version[1]);
  const session = route.match(
    /^\/api\/projects\/([\w-]+)\/character-workflow\/open-codex-session$/,
  );
  if (method === "POST" && session)
    return openCharacterSession(core, session[1], body);
  if (method === "GET" && route === "/api/charakuru/local-projects")
    return { projects: discoverLocalCharakuruProjects() };
  const charClaimable = route.match(
    /^\/api\/projects\/([\w-]+)\/character-workflow\/claimable$/,
  );
  if (method === "GET" && charClaimable)
    return listClaimableDagNodes(core, { projectId: charClaimable[1] });
  const charAction = route.match(
    /^\/api\/projects\/([\w-]+)\/character-workflow\/([\w-]+)$/,
  );
  if (method === "POST" && charAction)
    return dispatchCharacterWorkflowRoute(
      core,
      charAction[1],
      charAction[2],
      body,
    );
  const project = route.match(/^\/api\/projects\/([\w-]+)$/);
  if (method === "GET" && project) return core.snapshot(project[1]);
  const workspaceFiles = route.match(
    /^\/api\/projects\/([\w-]+)\/workspace-files$/,
  );
  if (method === "GET" && workspaceFiles)
    return { files: core.listWorkspaceFiles(workspaceFiles[1]) };
  const importPathsRoute = route.match(
    /^\/api\/projects\/([\w-]+)\/import-paths$/,
  );
  if (method === "POST" && importPathsRoute) {
    const data = z
      .object({
        fileRefs: z.array(z.string().min(1).max(4000)).min(1).max(12),
        name: z.string().optional(),
        assetId: z.string().optional(),
        sourceRef: z.string().optional(),
      })
      .parse(body);
    return core.importPaths(importPathsRoute[1], data.fileRefs, {
      name: data.name,
      assetId: data.assetId,
      sourceRef: data.sourceRef,
      allowExternalCopy: true,
    });
  }
  const operation = route.match(/^\/api\/tools\/([\w_]+)$/);
  if (method === "POST" && operation) return dispatch(core, operation[1], body);
  const upload = route.match(/^\/api\/projects\/([\w-]+)\/import$/);
  if (method === "POST" && upload) {
    const data = z
      .object({
        files: z
          .array(z.object({ name: z.string(), base64: z.string() }))
          .min(1)
          .max(12),
        name: z.string().optional(),
        assetId: z.string().optional(),
      })
      .parse(body);
    const maxBytes = 200 * 1024 * 1024;
    if (
      data.files.reduce((sum, f) => sum + f.base64.length, 0) >
      Math.ceil(maxBytes / 3) * 4 + 48
    )
      throw new StudioError(400, "文件集超过 200 MB");
    if (
      data.files.some(
        (f) =>
          !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(
            f.base64,
          ),
      )
    )
      throw new StudioError(400, "文件编码无效");
    return core.importFiles(
      upload[1],
      data.files.map((f) => ({
        name: f.name,
        bytes: Buffer.from(f.base64, "base64"),
      })),
      {
        name: data.name,
        assetId: data.assetId,
      },
    );
  }
  const media = route.match(
    /^\/api\/projects\/([\w-]+)\/files\/([\w-]+)\/(\d+)(\/http)?$/,
  );
  if (method === "GET" && media) {
    const { file, metadata } = core.assetFile(
      media[1],
      media[2],
      Number(media[3]),
    );
    const total = statSync(file).size;
    if (media[4])
      return {
        url: assetHttpUrl(core.configDir, route.slice(0, -5)),
        mime: metadata.mime,
        name: metadata.name,
        total,
      };
    const offset = args.offset || 0;
    if (offset > total) throw new StudioError(400, "文件偏移无效");
    const bytes = Buffer.alloc(Math.min(512 * 1024, total - offset));
    const fd = openSync(file, "r");
    let count;
    try {
      count = readSync(fd, bytes, 0, bytes.length, offset);
    } finally {
      closeSync(fd);
    }
    return {
      base64: bytes.subarray(0, count).toString("base64"),
      mime: metadata.mime,
      name: metadata.name,
      total,
      next: offset + count,
      eof: offset + count === total,
    };
  }
  throw new StudioError(400, "不允许调用此工作台接口");
}
