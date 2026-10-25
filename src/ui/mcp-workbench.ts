import { targetQuery } from "../shared/workbench-target";
import { App } from "@modelcontextprotocol/ext-apps";
import { installAppTransport, request } from "./api";
import { studioBuild } from "../shared/build";
import { launchRequest, receiveLaunch } from "./workbench-launch";
import "./style.css";
document.documentElement.dataset.mcpApp = "true";
// Hosts may omit the initial mode. Keep that case bounded like a chat embed.
document.documentElement.dataset.mcpDisplayMode = "inline";
const app = new App(
  { name: "Open Game Studio", version: "0.1.14" },
  { availableDisplayModes: ["fullscreen", "inline"] },
  { autoResize: false },
);
app.ontoolinput = ({ arguments: args }) => receiveLaunch(args);
app.ontoolresult = (result) =>
  receiveLaunch(
    (result.structuredContent as { result?: unknown } | undefined)?.result,
  );
function syncDisplayMode() {
  document.documentElement.dataset.mcpDisplayMode =
    app.getHostContext()?.displayMode === "fullscreen"
      ? "fullscreen"
      : "inline";
}
app.addEventListener("hostcontextchanged", syncDisplayMode);
await app.connect();
syncDisplayMode();
installAppTransport(app);
const service = await request<{ build?: string }>("/api/tools/studio_open", {
  method: "POST",
  body: JSON.stringify(launchRequest()),
}).catch(() => undefined);
if (service) receiveLaunch(service);
if (service && service.build !== studioBuild) {
  const fallbackUrl = `http://127.0.0.1:4310/?${targetQuery(launchRequest())}`;
  const root = document.getElementById("root")!;
  const panel = document.createElement("div");
  panel.className = "workbench-reconnect";
  const title = document.createElement("h2");
  title.textContent = "工作台已升级，需要重新连接";
  const detail = document.createElement("p");
  detail.textContent =
    "当前插件仍连接旧版服务。请重新连接 Open Game Studio 插件后打开工作台，也可以先使用浏览器中的新版。项目和资源均已保留。";
  const button = document.createElement("button");
  button.className = "button primary";
  button.textContent = "打开新版工作台";
  button.onclick = async () => {
    try {
      await app.openLink({ url: fallbackUrl });
    } catch {
      detail.textContent = `请打开 ${fallbackUrl} 使用新版工作台，或重新连接 Open Game Studio 插件。`;
    }
  };
  panel.append(title, detail, button);
  root.append(panel);
} else {
  await import("./main");
}
