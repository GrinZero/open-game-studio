import { studioBuild } from "../shared/build.js";
import { reconcileGenerations } from "./generation.js";
import express from "express";
import multer from "multer";
import { existsSync } from "node:fs";
import path from "node:path";
import { z } from "zod";
import { StudioCore, StudioError } from "./core.js";
import { dispatch } from "./operations.js";
import {
  discoverLocalCharakuruProjects,
  dispatchCharacterWorkflowRoute,
  listClaimableDagNodes,
} from "./character-workflow.js";
import { pickDirectory, pickFiles } from "./directories.js";
import { fileURLToPath } from "node:url";
import { verifyAssetHttpUrl } from "./asset-http.js";
import {
  openCharacterSession,
  projectChangeVersion,
} from "./codex-collaboration.js";
const app = express();
const core = new StudioCore();
reconcileGenerations(core);
const generationRecovery = setInterval(() => {
  try {
    reconcileGenerations(core);
  } catch (e) {
    console.error("generation recovery", e);
  }
}, 10000);
generationRecovery.unref();
const port = Number(process.env.PORT || 4310);
const repo = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);
app.disable("x-powered-by");
app.use((req, res, next) => {
  const allowed = ["127.0.0.1", "localhost", "[::1]"];
  const host = req.headers.host?.replace(/:\d+$/, "");
  if (!host || !allowed.includes(host))
    return res.status(403).json({ error: "只接受本地工作台请求" });
  // Only a short-lived, file-specific capability permits cross-origin reads.
  // Project APIs and writes retain their existing same-origin restriction.
  if (req.query.signature !== undefined || req.query.expires !== undefined) {
    if (
      !["GET", "HEAD", "OPTIONS"].includes(req.method) ||
      !verifyAssetHttpUrl(
        core.configDir,
        req.path,
        req.query.expires,
        req.query.signature,
      )
    )
      return res.status(403).json({ error: "文件链接无效或已过期" });
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Cache-Control", "private, no-store");
    if (req.method === "OPTIONS") {
      res.setHeader("Access-Control-Allow-Methods", "GET, HEAD");
      res.setHeader("Access-Control-Allow-Private-Network", "true");
      return res.sendStatus(204);
    }
  } else if (req.headers.origin) {
    try {
      const u = new URL(req.headers.origin);
      if (!allowed.includes(u.hostname) || u.port !== String(port))
        return res.status(403).json({ error: "跨来源请求被拒绝" });
    } catch {
      return res.status(403).json({ error: "来源无效" });
    }
  }
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("Referrer-Policy", "no-referrer");
  next();
});
app.use("/api", express.json({ limit: "2mb" }));
app.get("/api/health", (_req, res) =>
  res.json({
    status: "ok",
    schemaVersion: 1,
    build: studioBuild,
    node: process.version,
    capabilities: {
      provider: false,
      embeddedApp: "registered",
      canvasInputs: true,
      assetHttp: "signed-binary",
      engine: "not_verified",
    },
  }),
);
app.get("/api/projects", (_req, res) => res.json(core.listProjects()));
app.get("/api/projects/:id/change-version", (req, res) =>
  res.json(projectChangeVersion(core, req.params.id)),
);
app.get("/api/projects/:id/changes", (req, res) => {
  let version = projectChangeVersion(core, req.params.id).version;
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();
  const send = () => res.write(`data: ${JSON.stringify({ version })}\n\n`);
  send();
  let ticks = 0;
  const timer = setInterval(() => {
    try {
      const next = projectChangeVersion(core, req.params.id).version;
      if (next !== version) {
        version = next;
        send();
      } else if (++ticks % 30 === 0) res.write(": keepalive\n\n");
    } catch {
      res.end();
    }
  }, 500);
  timer.unref();
  res.on("close", () => clearInterval(timer));
});
app.post(
  "/api/projects/:id/character-workflow/open-codex-session",
  async (req, res) =>
    res.json(await openCharacterSession(core, req.params.id, req.body)),
);
app.get("/api/charakuru/local-projects", (_req, res) =>
  res.json({ projects: discoverLocalCharakuruProjects() }),
);
app.get("/api/projects/:id/character-workflow/claimable", (req, res) =>
  res.json(
    listClaimableDagNodes(core, {
      projectId: req.params.id,
      runId: typeof req.query.runId === "string" ? req.query.runId : undefined,
    }),
  ),
);
app.post("/api/projects/:id/character-workflow/:action", (req, res) =>
  res.json(
    dispatchCharacterWorkflowRoute(
      core,
      req.params.id,
      req.params.action,
      req.body,
    ),
  ),
);
app.post("/api/directories/pick", async (req, res) =>
  res.json(await pickDirectory(req.body)),
);
app.post("/api/files/pick", async (req, res) =>
  res.json(await pickFiles(req.body)),
);
app.get("/api/projects/:id", (req, res) =>
  res.json(core.snapshot(req.params.id)),
);
app.get("/api/projects/:id/workspace-files", (req, res) =>
  res.json({ files: core.listWorkspaceFiles(req.params.id) }),
);
app.post("/api/projects/:id/import-paths", (req, res) => {
  const body = z
    .object({
      fileRefs: z.array(z.string().min(1).max(4000)).min(1).max(12),
      name: z.string().optional(),
      assetId: z.string().optional(),
      sourceRef: z.string().optional(),
    })
    .parse(req.body);
  res.json(
    core.importPaths(req.params.id, body.fileRefs, {
      name: body.name,
      assetId: body.assetId,
      sourceRef: body.sourceRef,
      allowExternalCopy: true,
    }),
  );
});
app.post("/api/tools/:tool", (req, res) =>
  res.json(dispatch(core, req.params.tool, req.body)),
);
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 200 * 1024 * 1024,
    files: 12,
    fields: 8,
    fieldSize: 12000,
  },
});
app.post("/api/projects/:id/import", upload.array("files", 12), (req, res) => {
  const files = req.files as Express.Multer.File[];
  const opts = z
    .object({
      name: z.string().optional(),
      assetId: z.string().optional(),
      sourceRef: z.string().optional(),
    })
    .parse(req.body);
  res.json(
    core.importFiles(
      req.params.id as string,
      files.map((f) => ({ name: f.originalname, bytes: f.buffer })),
      opts,
    ),
  );
});
app.get("/api/projects/:id/files/:revisionId/:index", (req, res) => {
  const index = z.coerce.number().int().nonnegative().parse(req.params.index);
  const { file, metadata } = core.assetFile(
    req.params.id as string,
    req.params.revisionId as string,
    index,
  );
  res.setHeader("Content-Type", metadata.mime);
  res.setHeader("Content-Security-Policy", "default-src 'none'; sandbox");
  if (
    req.query.download ||
    metadata.mime === "text/plain" ||
    metadata.mime === "application/octet-stream"
  )
    res.attachment(metadata.name);
  res.setHeader("Content-Type", metadata.mime);
  res.sendFile(file, { dotfiles: "allow" });
});
if (
  process.env.NODE_ENV === "production" ||
  process.argv.includes("--production")
) {
  app.use(express.static(path.join(repo, "dist/ui")));
  app.get("/{*rest}", (_req, res) =>
    res.sendFile(path.join(repo, "dist/ui/index.html")),
  );
} else {
  const { createServer } = await import("vite");
  const vite = await createServer({
    root: repo,
    server: { middlewareMode: true },
    appType: "spa",
  });
  app.use(vite.middlewares);
}
app.use(
  (
    error: any,
    _req: express.Request,
    res: express.Response,
    _next: express.NextFunction,
  ) => {
    const status =
      error instanceof StudioError
        ? error.status
        : error instanceof z.ZodError || error instanceof multer.MulterError
          ? 400
          : 500;
    if (status === 500) console.error(error);
    res
      .type("application/json")
      .status(status)
      .json({
        error:
          status === 500 ? "本地服务操作失败；详情见服务日志" : error.message,
      });
  },
);
const server = app.listen(port, "127.0.0.1", () =>
  console.log(`Open Game Studio → http://127.0.0.1:${port}`),
);
const stop = () => {
  clearInterval(generationRecovery);
  server.close(() => {
    core.close();
    process.exit(0);
  });
};
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
