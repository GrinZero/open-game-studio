import { launchRequest } from "./workbench-launch";
import { useEffect, useState } from "react";
import type { App } from "@modelcontextprotocol/ext-apps";
import { AssetFileCache, type AssetFile } from "./asset-file-cache";
import { CodexReferenceStore } from "./codex-references";

let appTransport: App | undefined;
export function installAppTransport(app: App) {
  appTransport = app;
}
export function isEmbeddedApp() {
  return !!appTransport;
}
export function supportsCodexReferences() {
  const capability = appTransport?.getHostCapabilities()?.updateModelContext;
  return !!capability?.text;
}
export function supportsCodexMessages() {
  return !!appTransport?.getHostCapabilities()?.message?.text;
}
export async function sendCodexMessage(text: string) {
  if (!supportsCodexMessages())
    throw new Error("当前窗口不支持发送到对话，请使用“打开 Codex 会话”。");
  const result = await appTransport!.sendMessage(
    { role: "user", content: [{ type: "text", text }] },
    { timeout: 15000 },
  );
  if (result.isError)
    throw new Error("Codex 未接受制作消息，请重试或打开独立会话。");
}
export async function updateCodexReferences(text: string) {
  if (!supportsCodexReferences())
    throw new Error("当前窗口无法直接添加对话引用，请复制引用后粘贴到 Codex。");
  await appTransport!.updateModelContext(
    { content: [{ type: "text", text }] },
    { timeout: 15000 },
  );
}
export const codexReferences = new CodexReferenceStore(updateCodexReferences);
export async function reloadWorkbench() {
  if (!appTransport) {
    // A fresh navigation bypasses cached HTML; assets have content-hashed URLs.
    const url = new URL(window.location.href);
    url.searchParams.set("studioReload", crypto.randomUUID());
    window.location.replace(url.href);
    return;
  }
  // Reloading srcdoc repeats the host's cached HTML. Read the current bundle
  // from the live MCP server, then navigate this iframe to a new document.
  const resource = await appTransport.readServerResource(
    { uri: "ui://game-studio/workbench" },
    { timeout: 15000 },
  );
  const shell = resource.contents.find(
    (item) => item.uri === "ui://game-studio/workbench" && "text" in item,
  );
  if (
    !shell ||
    !("text" in shell) ||
    !shell.text.includes('<div id="root"></div>')
  )
    throw new Error("服务未返回工作台页面，请重试。");
  const url = URL.createObjectURL(
    new Blob(
      [
        shell.text.replace(
          "</head>",
          `<script type="application/json" id="studio-launch">${JSON.stringify(launchRequest()).replaceAll("<", "\\u003c")}</script></head>`,
        ),
      ],
      { type: "text/html" },
    ),
  );
  // Do not tear down the current bridge before navigation succeeds. A failed
  // resource read leaves the existing workbench fully usable.
  window.location.replace(url);
}
async function appRequest<T>(
  args: Record<string, unknown>,
  signal?: AbortSignal,
): Promise<T> {
  const result = await appTransport!.callServerTool(
    { name: "studio_ui_request", arguments: args },
    {
      signal,
      ...(args.path === "/api/directories/pick" ? { timeout: 135000 } : {}),
    },
  );
  if (result.isError)
    throw new Error(
      result.content
        .filter((c) => c.type === "text")
        .map((c) => c.text)
        .join("\n"),
    );
  return (result.structuredContent as { result: T }).result;
}
function base64(bytes: Uint8Array) {
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += 32768)
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 32768));
  return btoa(binary);
}
export async function request<T>(url: string, init?: RequestInit): Promise<T> {
  if (appTransport) {
    let body: unknown;
    if (init?.body instanceof FormData) {
      const form = init.body;
      const files = form.getAll("files") as File[];
      if (files.reduce((size, f) => size + f.size, 0) > 200 * 1024 * 1024)
        throw new Error("文件集超过 200 MB");
      body = {
        files: await Promise.all(
          files.map(async (f) => ({
            name: f.name,
            base64: base64(new Uint8Array(await f.arrayBuffer())),
          })),
        ),
        name: form.get("name") || undefined,
        assetId: form.get("assetId") || undefined,
      };
    } else if (typeof init?.body === "string") body = JSON.parse(init.body);
    return appRequest<T>(
      { path: url, method: init?.method || "GET", body },
      init?.signal || undefined,
    );
  }
  const r = await fetch(url, init);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || `请求失败 ${r.status}`);
  return data;
}
export function call<T = any>(name: string, args: unknown) {
  return request<T>(`/api/tools/${name}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(args),
  });
}
export const fileUrl = (projectId: string, revisionId: string, index = 0) =>
  `/api/projects/${projectId}/files/${revisionId}/${index}`;

const maxFileBytes = 200 * 1024 * 1024;
async function readFile(
  route: string,
  signal: AbortSignal,
): Promise<AssetFile> {
  try {
    const file = await appRequest<{
      url: string;
      mime: string;
      name: string;
      total: number;
    }>({ path: `${route}/http`, method: "GET" }, signal);
    signal.throwIfAborted();
    if (file.total > maxFileBytes) throw new Error("文件超过 200 MB");
    const response = await fetch(file.url, {
      credentials: "omit",
      referrerPolicy: "no-referrer",
      signal: AbortSignal.any([signal, AbortSignal.timeout(30000)]),
    });
    if (!response.ok) throw new Error(`文件请求失败 ${response.status}`);
    const blob = await response.blob();
    if (blob.size !== file.total || blob.type.split(";")[0] !== file.mime)
      throw new Error("文件传输无效");
    return { blob, name: file.name };
  } catch (error) {
    signal.throwIfAborted();
    // Older hosts may block loopback HTTP or still have the previous CSP.
    // Keep the bridge usable there without retaining abandoned transfers.
    return readBridgeFile(route, signal);
  }
}
async function readBridgeFile(
  route: string,
  signal: AbortSignal,
): Promise<AssetFile> {
  const chunks: BlobPart[] = [];
  let offset = 0,
    mime = "application/octet-stream",
    name = "asset";
  while (true) {
    signal.throwIfAborted();
    const chunk = await appRequest<{
      base64: string;
      mime: string;
      name: string;
      total: number;
      next: number;
      eof: boolean;
    }>(
      {
        path: route,
        method: "GET",
        offset,
      },
      signal,
    );
    if (chunk.total > maxFileBytes || (!chunk.eof && chunk.next <= offset))
      throw new Error("文件传输无效");
    chunks.push(Uint8Array.from(atob(chunk.base64), (c) => c.charCodeAt(0)));
    mime = chunk.mime;
    name = chunk.name;
    offset = chunk.next;
    if (chunk.eof) break;
  }
  const blob = new Blob(chunks, { type: mime });
  return { blob, name };
}
const fileCache = new AssetFileCache(readFile);
export function useAssetFileUrl(
  projectId: string,
  revisionId?: string,
  index = 0,
) {
  const route = revisionId ? fileUrl(projectId, revisionId, index) : "";
  const [loaded, setLoaded] = useState<{
    route: string;
    url?: string;
    error?: string;
  }>();
  useEffect(() => {
    setLoaded(undefined);
    if (!appTransport || !route) return;
    let active = true;
    const lease = fileCache.acquire(route);
    lease.promise
      .then(({ url }) => {
        if (active) setLoaded({ route, url });
      })
      .catch((error) => {
        if (active) setLoaded({ route, error: String(error) });
      });
    return () => {
      active = false;
      lease.release();
    };
  }, [route]);
  return appTransport
    ? loaded?.route === route
      ? loaded
      : { route }
    : { route, url: route || undefined };
}
export async function downloadAssetFile(route: string) {
  const lease = fileCache.acquire(route);
  try {
    const file = await lease.promise;
    await appTransport!.downloadFile({
      contents: [
        {
          type: "resource",
          resource: {
            uri: `file:///${encodeURIComponent(file.name)}`,
            mimeType: file.blob.type,
            blob: base64(new Uint8Array(await file.blob.arrayBuffer())),
          },
        },
      ],
    });
  } finally {
    lease.release();
  }
}
