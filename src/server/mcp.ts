import { reconcileGenerations } from "./generation.js";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import type {
  CallToolResult,
  JSONRPCMessage,
} from "@modelcontextprotocol/sdk/types.js";
import type { ZodRawShape } from "zod";
import {
  registerAppResource,
  registerAppTool,
  RESOURCE_MIME_TYPE,
} from "@modelcontextprotocol/ext-apps/server";
import { OpenAIExtensions } from "@openai/mcp-extensions/server";
import { readFileSync } from "node:fs";
import { StudioCore } from "./core.js";
import { dispatch, tools } from "./operations.js";
import { uiRequest, uiRequestSchema } from "./ui.js";
import { assetHttpOrigin } from "./asset-http.js";
import { codexWorkspace } from "./codex-workspace.js";
const core = new StudioCore();
reconcileGenerations(core);
const server = new McpServer({ name: "game-studio", version: "0.1.9" });
new OpenAIExtensions(server);
const appUri = "ui://game-studio/workbench";
async function openingArgs(
  args: Record<string, unknown>,
  meta?: Record<string, unknown>,
) {
  if (args.projectId || args.workspaceRoot || args.workspaceRoots) return args;
  const workspaceRoot = codexWorkspace(meta);
  if (workspaceRoot) return { ...args, workspaceRoot };
  if (!server.server.getClientCapabilities()?.roots) return args;
  try {
    const { roots } = await server.server.listRoots({}, { timeout: 2000 });
    return {
      ...args,
      workspaceRoots: roots.map((root) => root.uri).slice(0, 32),
    };
  } catch {
    return args;
  }
}
const csp = {
  connectDomains: ["blob:", assetHttpOrigin()],
  resourceDomains: ["blob:", "data:"],
};
const resourceMeta = {
  ui: { csp, prefersBorder: false },
  "openai/ui": {
    preferredDisplayMode: "fullscreen",
    availableDisplayModes: ["fullscreen", "inline"],
  },
};
registerAppResource(
  server,
  "game-studio-workbench",
  appUri,
  {
    mimeType: RESOURCE_MIME_TYPE,
    _meta: resourceMeta,
  },
  async () => ({
    contents: [
      {
        uri: appUri,
        mimeType: RESOURCE_MIME_TYPE,
        text: readFileSync(
          new URL("../../dist/mcp/index.html", import.meta.url),
          "utf8",
        ),
        _meta: resourceMeta,
      },
    ],
  }),
);
const icon =
  "data:image/svg+xml;base64," +
  readFileSync(new URL("../../assets/icon.svg", import.meta.url)).toString(
    "base64",
  );
for (const [name, tool] of Object.entries(tools)) {
  const inputSchema: ZodRawShape = tool.schema.shape;
  const config = { description: tool.description, inputSchema };
  const handler = async (
    args: Record<string, unknown>,
    extra: { _meta?: Record<string, unknown> },
  ): Promise<CallToolResult> => {
    try {
      const result = dispatch(
        core,
        name,
        name === "studio_open" ? await openingArgs(args, extra._meta) : args,
      );
      return {
        content: [{ type: "text", text: JSON.stringify(result) }],
        structuredContent: { schemaVersion: 1, result },
      };
    } catch (error) {
      return {
        isError: true,
        content: [
          {
            type: "text",
            text: error instanceof Error ? error.message : String(error),
          },
        ],
      };
    }
  };
  if (name === "studio_open") {
    const appConfig = {
      ...config,
      title: "Open Game Studio",
      annotations: {
        readOnlyHint: true,
        destructiveHint: false,
        openWorldHint: false,
      },
      icons: [{ src: icon, mimeType: "image/svg+xml" }],
      _meta: {
        ui: { resourceUri: appUri },
        "openai/ui": { entrypoints: [{ type: "global" }, { type: "thread" }] },
      },
    };
    registerAppTool(server, name, appConfig, handler);
  } else server.registerTool(name, config, handler);
}
registerAppTool(
  server,
  "studio_ui_request",
  {
    description: "工作台内部数据与文件桥接，仅供界面调用。",
    inputSchema: uiRequestSchema,
    _meta: { ui: { visibility: ["app"] } },
  },
  async (args, extra) => {
    try {
      return {
        content: [{ type: "text", text: "工作台请求完成" }],
        structuredContent: {
          result: await uiRequest(
            core,
            args.path === "/api/tools/studio_open"
              ? {
                  ...args,
                  body: await openingArgs(
                    (args.body || {}) as Record<string, unknown>,
                    extra._meta,
                  ),
                }
              : args,
          ),
        },
      };
    } catch (error) {
      return {
        isError: true,
        content: [
          {
            type: "text",
            text: error instanceof Error ? error.message : String(error),
          },
        ],
      };
    }
  },
);
// SDK 1.32's high-level tool registry omits icons from tools/list. Restore the
// standard protocol field at the transport boundary without replacing discovery.
class StudioTransport extends StdioServerTransport {
  override send(message: JSONRPCMessage) {
    if ("result" in message && Array.isArray(message.result.tools)) {
      message = {
        ...message,
        result: {
          ...message.result,
          tools: message.result.tools.map((tool) =>
            tool.name === "studio_open"
              ? { ...tool, icons: [{ src: icon, mimeType: "image/svg+xml" }] }
              : tool,
          ),
        },
      };
    }
    return super.send(message);
  }
}
await server.connect(new StudioTransport());
process.on("SIGTERM", () => {
  core.close();
  process.exit(0);
});
