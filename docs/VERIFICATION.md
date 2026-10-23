# 首轮验证记录

## 2026-10-12：英文主页与五语 README

- 英文作为 `README.md` 主页，新增简中、繁中、日文与韩文版本。五版共用章节顺序、稳定导航锚点、表格、安装命令和文档链接。原创 SVG 横幅及隔离示例环境的真实截图保存在 `docs/images/`，附来源与 SHA256；README 翻译不代表 UI 已多语言化。
- GitHub 原标签页已刷新确认英文主页、原创横幅与五语导航，实际点击试用锚点成功。实页发现外部徽章加载失败后改为本仓库 `docs/images/badges.svg`，所有主页展示图均本地保存。
- 开发、构建与热更新操作移至 `CONTRIBUTING.md`，安装配置与画布操作分别移至 `docs/SETUP.md`、`docs/WORKBENCH.md`。核对代码后修正为 26 个领域工具、1 个界面桥接工具、10 个技能，并更新工具契约中的旧入口说明。没有更改已有领域测试断言。
- 新增 `npm run check:docs` 与文档 CI，验证本地链接、资产、锚点、语言结构/命令一致及工具/技能数量。文档检查、TypeScript/两套生产构建、distribution 检查和 41 项测试通过。
- 已核实 LaunchAgent 与安装后的 MCP 配置指向缓存外 runtime。安装和同步脚本携带全部语言 README、CONTRIBUTING、文档图片与许可声明。实际运行目录文件哈希匹配；HTTP 4310 返回相同构建，新 stdio MCP 连接发现 27 个工具并读取相同内嵌 HTML。3 个项目登记前后一致，无服务端逻辑更新，无需重启服务。
- CUA 清单仅暴露 GitHub 浏览器标签，用户原主窗口工作台未暴露；尝试重新调用 `studio_open` 只代表当前聊天入口请求，不证明原主窗口刷新。无法确认用户原工作台已显示本次构建标识，需重新打开该工作台。本次不将新连接或验证页面作为原主窗口热更新证据。

## 2026-10-12：工作台左上角品牌图标

- Brand 组件的 Boxes 图标替换为共享 `assets/icon.svg`，以 22px 显示在原品牌容器中。HTTP 与 MCP 打包均内嵌 SVG data URL，MCP 无需外部图标请求。
- TypeScript 与两套构建通过；实际 HTTP 4310 页面和已安装 MCP 新连接的沙箱页面均显示新图标，检查图片解码成功、SVG 内容与共享资产一致，品牌区域无旧 Boxes 图标。截图为 `var/qa/http-brand-icon.png` 与 `var/qa/mcp-brand-icon.png`。
- 核实 LaunchAgent 与已安装 MCP 配置后，同步 UI 源码、SVG 类型声明、MCP 构建脚本、两套构建和本文档至实际 runtime。备份为 `hot-update-backups/2026-10-11T17-58-28.773Z-brand-icon`。无服务端变更，无需重启 HTTP 服务。
- 当前可操作的浏览器与 MCP App 标签清单均未暴露用户原主窗口工作台；原生 UI 禁止访问 Codex 主窗口。原页面尚未刷新或验证，需重新打开工作台；新沙箱页面验收不代表原主窗口已更新。

## 2026-10-12：按展示模式限制工作台高度

- 根据用户补充，恢复聊天内嵌的 720px 上限。只有宿主明确声明 `displayMode=fullscreen` 时填满视口；`inline` 或未提供模式时保持限高，继续关闭自动尺寸上报，避免聊天插件被撑得过高。
- 初始化读取宿主上下文，并监听展示模式变更；主题等局部上下文通知读取 SDK 合并后的上下文，不会误改当前展示模式。
- TypeScript、HTTP / MCP 生产构建及完整 opaque-origin MCP UI 回归通过。首页与画布覆盖 480、720、960、1200px 高度、未提供初始模式、明确 inline / fullscreen 初始模式以及反复切换；聊天模式始终不超过 720px，全屏模式始终填满宿主。图片 / 模型、资源制作、下载和重连回归也通过，19 次二进制 HTTP 读取，0 个桥接文件块。
- 已核实实际运行配置，同步 CSS、MCP 入口与 HTTP / MCP 构建及回归脚本至缓存外 runtime。真实 HTTP 页面四种高度均填满；安装配置建立的只读 MCP 新连接返回 HTML 与仓库构建 SHA256 一致。无服务端修改，无需重启服务。备份：`hot-update-backups/2026-10-11T17-53-35.591Z-display-mode-height`。
- 已重新调用 studio_open，但当前聊天未暴露可操作的 MCP App 标签。原主窗口仍无法通过原生 UI 工具访问，不能确认其已刷新；需用户重新打开工作台，宿主保留旧资源时需重新连接插件。此修正替代之前统一移除高度上限的方案。

## 2026-10-12：侧边栏图标

- 将立方体图标替换为浅蓝色「游戏手柄＋创作星芒」SVG，检查深浅背景与 24 / 32px 显示；预览保存在 `assets/preview/game-studio-icon.png`。
- 插件与 MCP 入口共用 `assets/icon.svg`，安装器将 assets 同步到缓存外 runtime。当前 SDK 1.32 的高层工具注册会忽略 icons，stdio transport 在 tools/list 响应中补齐 studio_open 的标准 icons 字段。
- TypeScript、HTTP / MCP 构建与真实 stdio 回归通过；回归直接验证客户端收到的图标与 SVG 文件一致。
- 已核实实际 LaunchAgent 与已安装 MCP 配置，同步 runtime、插件源目录及插件缓存的 SVG、MCP 源码和两套构建。备份为 `hot-update-backups/2026-10-11T17-51-37.085Z-sidebar-icon`。
- 原生 UI 工具禁止访问 Codex 主窗口，无法刷新或验证用户实际侧边栏。既有 MCP 连接仍需重新连接以加载新服务代码；新连接验收不代表既有连接或主窗口已更新。未重启无关服务，未修改项目登记或资产数据。

## 2026-10-12：内嵌工作台底部留白

- 移除 MCP 内嵌 `.app-shell` 的 720px 高度上限，沿 html / body / root 的高度链填满宿主可用空间，保留面板内部滚动。
- TypeScript 与 HTTP / MCP 生产构建通过。增加首页和画布在 480、720、960、1200px 宿主高度下的回归断言，直接比较工作台底边与宿主视口底边，避免仅检查画布填满已被截短的父容器。
- 完整 opaque-origin MCP UI 回归通过，包括响应式面板、图片与模型预览、Blob 生命周期、下载、资源制作与旧服务重连提示；19 次二进制 HTTP 读取，0 个 Base64 文件块。
- 已核实实际 HTTP LaunchAgent 与 MCP 配置指向缓存外 runtime，并同步 CSS、两套构建。HTTP 4310 的实际 HTML 与仓库构建一致；已安装 MCP 的只读新连接返回的内嵌 HTML SHA256 与仓库构建一致。真实 HTTP 项目页面四种高度全部填满视口；新 MCP 沙箱验证页面也通过高度与宽度响应式检查。
- 无服务端变更，无需重启服务。备份为 `hot-update-backups/2026-10-11T17-49-10.486Z-viewport-height`。原生 UI 工具禁止访问 Codex 主窗口，当前聊天也未暴露主窗口工作台标签，因此尚未刷新或确认用户原主窗口生效；需用户重新打开该工作台，新验证页面不代表原主窗口已更新。

## 2026-10-12：Open Game Studio 命名更新

- 工作台品牌、首页页脚、HTTP / MCP 页面标题、插件显示名、默认提示、技能描述与 README 统一为 Open Game Studio；保留插件 ID、包名、MCP URI、服务标签和数据目录，兼容已有安装。
- TypeScript、两套生产构建和 41 项测试通过。已核实 LaunchAgent 与已安装 MCP 配置使用缓存外 runtime；同步源码、HTTP / MCP 构建、文档与已安装插件元数据，仅重启 HTTP LaunchAgent。更新前副本保存在 `hot-update-backups/2026-10-11T17-47-32-172Z-open-game-studio`。
- HTTP 实际响应标题与脚本含新名称；安装缓存的真实 MCP 配置建立只读新连接，工具标题为 Open Game Studio，内嵌 HTML 与仓库构建 SHA256 一致。3 个项目登记与更新前一致。
- 重新调用 studio_open 后，本聊天 MCP App 的 DOM 已实际显示 Open Game Studio。此为聊天侧栏的验证；原主窗口未暴露为可操作页面，原生 UI 工具拒绝访问 Codex，因此无法确认原主窗口已刷新。同一宿主再次调用 studio_open 后仍缓存旧页脚 CODEX STUDIO · ALPHA，宿主标签仍为 Game Studio；磁盘及 HTTP 构建已含新页脚。需要重新连接插件以清除宿主缓存，再重新打开工作台。

## 2026-10-11：v0.1.7 Blob 生命周期与二进制 HTTP 传输

- MCP 内嵌预览优先获取单文件、5 分钟有效的签名链接，通过本机 HTTP 一次读取二进制 Blob。CSP 新增配置端口的 loopback 连接来源；仅签名文件的只读请求允许跨来源，其他 API 和写入保留原同源限制。宿主阻止 HTTP 时保留 MCP 分块回退。
- 传输和 Blob URL 按使用者计数共享。最后一个预览或下载释放后取消未完成传输、回收 URL、移除缓存；避免切换详情时释放仍被画布引用的模型。覆盖 StrictMode、重复释放、失败重试与迟到结果的测试。
- 生产构建、TypeScript 与 33 项测试通过。opaque-origin MCP UI 验收正常通道为 16 次二进制 HTTP 读取、0 个 Base64 文件块；故意阻断 HTTP 后为 16 个桥接文件块，模型、图片、下载与共享 URL 生命周期均通过。HTTP 浏览器整体验收亦通过。
- 源码、HTTP / MCP 构建与版本 0.1.7 同步到实际 runtime，最小范围重启 HTTP LaunchAgent，健康检查返回 `assetHttp=signed-binary`。未重装依赖；3 个项目登记保留。备份：`hot-update-backups/20261011-004736-asset-http`。
- 从安装缓存 0.1.7 的实际 MCP 配置启动新连接，确认服务版本、嵌入 HTML SHA256 和 CSP；通过签名 HTTP 读取「星铸者」真实模型 12,223,964 字节，SHA256 与登记版本一致。此为新连接的只读验收，不代表其他已有 MCP 连接自动重载。
- 主窗口未暴露为当前聊天的浏览器标签，UI 自动化禁止操作 Codex 原生窗口。用户已按提示重新打开工作台，并明确确认「已重新打开，预览正常」；主窗口刷新证据来自用户确认，非新验证页截图。

## 2026-10-11：画布右键菜单（热更新）

- 参考 OhMyGame 官方 `asset-canvas-workspace.tsx` 的空白画布 / 节点菜单结构，提供空白处添加便签、定点导入、资源栏、粘贴、布局撤销 / 重做、全选与适应视图；节点查看、便签编辑、主文件下载、上下文复制、多选副本、制作分组及移除引用；连线提供断开输入。
- 会话内节点剪贴板可用于同一项目的不同画布。复制固定版本、文字、相对位置和所选节点内部连线，重映射节点 / 连线 ID；拒绝制作分组复制和跨项目粘贴。移除引用保留资源，制作分组保留运行历史。布局撤销仍仅撤销位置修改。
- TypeScript / HTTP 与 MCP 生产构建、33 项领域与 MCP 测试通过。HTTP 与 opaque-origin MCP 浏览器回归验证右键创建 / 编辑、复制 / 粘贴、多选副本、移除引用、真实下载、断线 / 重连、Escape 关闭菜单、定点导入新版本并保留既有固定版本引用；截图为 `var/qa/browser-context-selection.png`、`var/qa/mcp-context-selection.png`。
- 已核实 LaunchAgent 仍运行缓存外 runtime 的 HTTP 服务，4310 端口与原登记库不变。仅同步 UI 源码、HTTP / MCP 构建和说明文档；资源由服务读取磁盘，未重装依赖或重启无关服务。更新前备份为 `hot-update-backups/20261011-003537-context-menu`。
- 两套安装产物 SHA256 与本次右键构建一致；从安装缓存实际 MCP 配置启动的只读验证读到相同内嵌 HTML。用户原来的 IAB 标签页已经刷新并恢复到「星铸者 → 主角」，显示当时的 `index-DVBsgx9W.js` 和真实节点右键菜单，截图为 `var/qa/installed-context-menu.jpg`。7 个节点与素材保持，仅恢复总览视口。本次没有把新测试页面当成主窗口热更新证据。
- 验收过程中同仓库另一个任务更新了资源缓存、HTTP 文件传输与模型 Blob 释放测试。保留这些源码修改；右键最终验收使用实际运行目录的隔离副本，避免混用缓存服务接口。新缓存生命周期的模型断言属于并行任务。最终构建已合并已发布的 v0.1.7 缓存传输，HTTP 回归与 33 项测试通过；隔离的 MCP 右键专项通过。最后修复编辑弹窗受视口保存禁用的时序问题，最终 UI 产物为 `index-DRpq-rhe.js`，已同步 HTTP 与 MCP。再次刷新时原 IAB 标签不再属于当前会话（列表为空），因此不能确认原窗口已加载这最后一处修复，已请求重新打开工作台。 最后整套 MCP 回归在模型预览的滚动画布视口断言（`model-preview-smoke.mjs:160`）失败，未将其记为完整通过；该模型回归与右键专项结果分别记录。

## 2026-10-11：v0.1.6 连线制作输入

- 连线递归收集固定资产版本、文档和便签，创建分组自动连线；准备计划与启动校验实际文件和哈希，确认后保存文字、版本与文件快照。断线影响后续计划，历史运行与局部重跑保留冻结上下文。
- 25 项领域 / MCP 测试和 TypeScript、生产构建通过。HTTP Chromium 与 MCP opaque 沙箱操作验证真实接线、输入清单、文字确认、启动和冻结快照；截图为 `var/qa/browser-input-plan.png`、`var/qa/browser-input-frozen.png`、`var/qa/mcp-input-plan.png`、`var/qa/mcp-input-frozen.png`。
- 模型预览回归在创建 / 启动工作流前通过。在同一 opaque 沙箱先启动工作流再首次加载模型的组合测试中，拖动相机角度断言曾失败；尚未确认该序列在原生宿主中的表现，不将它记为已通过。
- 已核实本机 LaunchAgent 的运行目录、4310 端口及 `.game-studio` 登记库。HTTP 服务加载新代码，健康检查 `canvasInputs=true`，3 个项目登记保留。源码、HTTP / MCP 构建与文档同步到缓存外运行目录，旧运行副本备份在 `hot-update-backups/20261011-000410`。
- 插件版本已登记为 0.1.6，没有重新安装运行依赖。使用安装缓存的实际 `.mcp.json` 做只读握手，确认 MCP 服务 0.1.6、`canvasInputs=true` 和内嵌 HTML 的 SHA256 与本次构建一致；这次独立握手不代表已有主窗口连接恢复。
- 原生 Codex 主窗口访问被 UI 自动化工具拒绝。旧 MCP 连接关闭后宿主返回 `Transport closed`，需重新发现插件 / 打开工作台；新验证页面不证明用户主窗口已刷新。

## 2026-10-10：v0.1.5 画布 3D 预览

- GLB 画布卡片直接渲染固定版本模型，自动旋转和播放动画；详情保留旋转、缩放与多动画选择。共用查看器显示加载进度、文件错误及 WebGL 中断，切换版本清除旧状态。屏幕外节点卸载查看器并停止模型读取。
- 切换选中项或更新素材时保留 React Flow 的 measured 尺寸，修复尺寸丢失后卡片偶尔不可见的问题。
- MCP 自包含构建处理 Three 可选 Meshopt WASM 初始化的拒绝：无 WASM 权限的宿主仍可渲染普通 GLB；需要该解码器的压缩模型保持加载失败反馈。未放宽宿主 CSP。内嵌响应式验收按宿主实际剩余高度验证，不假设宿主超过 720px。
- TypeScript、生产构建及全部 19 项领域/MCP 测试通过。真实 Chromium 浏览器和 MCP opaque 沙箱验证 GLB 渲染、动画播放/切换、详情旋转/缩放、从模型区域拖动与滚动、v2 导入后画布继续引用 v1、加载失败与切换恢复，无未处理页面错误。截图：`var/qa/browser-canvas-model-preview.png`、`var/qa/mcp-canvas-model-preview.png`。
- 已安装并启用 v0.1.5，HTTP 后台服务加载新运行时，HTTP 与 MCP 构建均已同步。通过新开的隐藏验证页实际观察「星铸者 / 主角」7 个模型均已渲染，截图为 `var/qa/installed-canvas-model-preview.png`；只切换验证页的画布和面板，未修改用户项目内容。用户当前 Codex 主窗口未暴露为 MCP App tab，Computer Use 禁止操作 Codex 应用，无法刷新旧主窗口；需用户重新打开 Game Studio。新验证页不能作为旧主窗口已热刷新的证据。

## 2026-10-10：v0.1.4 项目目录选择优化

- 新建项目默认直接使用所选文件夹，改项目名称不会改目录；可勾选「按项目名称创建子文件夹」，此时输入标为「父目录」，最终项目路径独立展示。切换选项不会重复追加目录名，手动填写路径与选择器采用相同规则。
- `project_manage create` 支持已有文件夹并保留原文件，未创建的目录仍会创建；已登记的目录返回原项目，文件占用路径明确拒绝。已有项目未移动或修改。
- 全部 19 项自动测试及生产构建通过。浏览器 Chromium 冒烟验证默认使用已有目录、保留文件、取消、错误重试、改名、子目录开关、手动路径与最终路径；MCP App 沙箱冒烟验证可选子目录的实际创建和已有项目关联。选择器结果由测试模拟。
- 已安装 v0.1.4，本机后台健康检查通过。安装后生产表单验证 `star-maker` 默认保持原路径，勾选后预览为 `star-maker/starforger`，未提交或改动用户项目。截图为 `var/qa/installed-project-folder-picker.png` 和 `var/qa/installed-project-subfolder.png`。已连接的 MCP 进程需重新加载后获取新版界面。

## 2026-10-10：v0.1.2 插件版本更新

- 项目文件夹选择作为 v0.1.2 发布，同步更新包清单、锁文件、插件清单及 MCP 服务/内嵌 UI 版本。此前只更新运行时，插件版本仍为 v0.1.1，宿主可能继续使用旧版缓存。
- 生产构建与重新安装完成，`codex plugin list` 确认 v0.1.2 已安装且启用，缓存位于 `game-studio/0.1.2`。使用此缓存的实际 MCP 配置完成新 stdio 握手，服务返回 v0.1.2、19 个工具，内嵌资源包含新版版本与文件夹选择 UI；后台 HTTP 健康检查正常。

## 2026-10-10：项目文件夹选择

- 新建项目支持选择保存位置，自动生成同名子目录；名称变化同步目录，手动修改后保持用户填写的路径。关联已有项目直接选择目录。取消保留表单，错误可重试，选择过程中禁止重复提交。
- 浏览器 HTTP 与 Codex 内嵌 MCP 桥接共用 macOS 原生 NSOpenPanel。调用异步执行，不阻塞工作台服务；路径通过 argv 传递，脚本不拼接用户输入。其他平台保留手动路径。
- 15 项自动测试通过，包括目录命名边界、特殊字符、取消、重复打开、失败恢复与 stdio MCP 桥接校验。生产构建通过。
- 浏览器与隔离 MCP App 的真实 Chromium 冒烟通过。选择器结果在这两项自动测试中模拟，覆盖创建与关联、取消、错误提示、路径自动更新与手动覆盖；截图为 `var/qa/project-folder-picker.png`。
- 将生产选择器的同一 JXA 脚本打包为临时 QA app，通过 Computer Use 实际观察 NSOpenPanel，并验证确认返回当前仓库目录、取消返回空选择。生产 osascript 启动已执行；实际 Codex 宿主窗口仍不可由 Computer Use 操作。
- 已通过 `install:local` 更新本机运行时与插件，后台 HTTP 健康检查正常；通过 Codex 内置浏览器实际观察安装后的生产表单，截图为 `var/qa/installed-project-folder-picker.png`。已连接的 MCP 进程需重启 Codex 加载更新。

## 2026-10-10：v0.1.1 侧边栏入口修复

- v0.1.0 只有领域工具和浏览器 URL，缺少全局界面入口，因此不会自动出现在侧边栏「…」菜单。v0.1.1 为 `studio_open` 增加 `openai/ui.entrypoints` 的 global/thread 入口、图标与 `ui://game-studio/workbench` MCP App 资源。
- 使用官方 MCP Apps / OpenAI Extensions SDK。内嵌界面是自包含的原有 React 工作台，通过仅供 app 使用的 `studio_ui_request` 读取项目、保存修改、上传文件及分块读取媒体。它不需要放宽 HTTP Origin 或直接访问 localhost。
- 真实 Chromium 和官方 AppBridge 在无 same-origin、无 HTTP 网络、无表单提交权限的隔离沙箱内通过：握手、项目读取与新建、PNG 导入与预览、资源和便签保存，以及宿主下载。测试使用临时登记库，不改动用户示例；截图为 `var/qa/mcp-embedded-workbench.png`。
- 严格类型检查、生产构建和全部 10 个测试通过；真实 stdio 验证 18 个领域工具与 1 个仅供 app 调用的桥接工具、入口 metadata、资源读取、分块文件及路径拒绝。
- Codex 原生应用不允许由 Computer Use 操作，因此没有声称实际观察到侧边栏菜单。需要宿主重新发现新版本入口；用户若仍未看到，应退出并重新打开 Codex。实际原生菜单观察与消息/文件入口仍待验证。

## 2026-10-10：本机 Codex 安装体验

- 再次通过全部 10 个测试及 TypeScript / Vite 生产构建。
- `npm run install:local` 通过当前 Codex CLI 的本地市场入口安装 `game-studio@game-studio-local` v0.1.0；插件列表实际返回 `installed: true` 和 `enabled: true`。
- 插件缓存内 `.mcp.json` 的真实 stdio 服务完成握手，发现 18 个工具；`studio_open` 与后台 HTTP 返回同一份项目列表。
- 运行时与依赖复制到独立安装目录；用户级 LaunchAgent 实际处于运行状态，HTTP 健康检查返回 `ok`。
- 真实 Chromium 打开安装后的生产 UI，显示「星铸者 · 示例」的资源与制作流程，无页面脚本错误；截图位于 `var/qa/installed-workbench.png`。
- 浏览器面板打开请求已发送。当前聊天的工具热加载和 MCP Apps 内嵌未验证；新聊天应重新发现安装的技能和 MCP 工具。模型 Provider 与引擎 worker 仍未接入。

## 2026-10-09：基础功能

2026-10-09，本地 macOS / Node 24。以下记录是实际执行结果。

- TypeScript 严格类型检查通过。
- Vite 生产构建通过；主包约 439 KB，GLB 查看器约 1.07 MB，后者仅在打开模型时加载。查看器仍有大分块提示，尚未完成性能验收。
- 10 个 Node 自动测试通过，包括真实 stdio MCP 握手及 18 个工具发现。
- 真实 Chromium 浏览器流程通过：创建项目、上传图片、读取实际图片、添加引用、便签、模板展开、启动持久 Run、提交 Brief、刷新接续、拖动布局保存、跨画布共享引用、Origin 拒绝和 CAS 冲突。
- 自包含 GLB 测试网格实际加载；拖动后相机角度发生变化。该网格只验证查看器，不是角色制作成果。
- 1440×960 与 390×844 截图已查看。窄屏没有文档横向溢出，详情作为右侧抽屉呈现。
- 生产依赖的 `npm audit --omit=dev` 返回 0 个已知漏洞。上传组件已升级到 Multer 2.4.0，并重新通过真实上传流程。
- plugin.json 与 mcp.json 通过对应 Agent Plugins 1.0 JSON Schema 校验。此项仅验证包结构，不表示已安装或通过实际 Codex 宿主内嵌测试。

领域测试验证不可变文件 hash、版本 CAS、重启后的持久状态、幂等启动/导出、候选审核、反馈、局部重跑和源目录边界。用于导出契约测试的 Blend 是明确标注的合成文件头夹具，不能作为可编辑性证据。没有执行模型服务生成、真实 Blend 打开、骨架变形或 Godot 导入。

依据的主要技术文档：[React Flow](https://reactflow.dev/learn)、[MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)、[OpenAI 插件包装](https://developers.openai.com/plugins/build/plugins)与 [Agent Plugins MCP schema](https://agent-plugins.org/schemas/1.0.0/mcp.schema.json)。

## 2026-10-11 统一资源制作验收

- `tests/resources.test.ts`：文档版本与跨画布固定引用、保存冲突、任务幂等、参考冻结、结果原子回填、重启、产物类型和隔离 worker 执行。测试 worker 使用明确的测试执行器，不提交付费模型请求。
- `scripts/resource-composer-smoke.mjs`：HTTP 与 opaque-origin MCP 工作台中验证全部添加入口、Markdown 编辑和预览、HTML 不执行、文档新版本、资源库旧版本引用、五类制作表单、持久草稿。
- 真实模型生成服务的认证、计费提交、最终美术质量需在具体任务中验证，不能由本地执行器测试推断通过。
- 发布需要分别验证运行目录同步、HTTP 服务加载、MCP 服务加载及用户实际主窗口刷新。验证页面不能替代主窗口。
# 2026-10-11：资源用途分类

- 画布资源栏按角色、装备、植被、建筑等用途分组，只显示非空分类。默认折叠，显示数量，支持展开／收起全部，并按项目记住展开状态。搜索名称、标签或分类时自动展开匹配资源。
- 现有资源按名称和标签推断用途；装备关键字优先于主角前缀。星铸者现有 13 个资源分为角色 1、装备 6、植被 6。详情可手动指定分类或恢复自动分类，资源页支持用途筛选。画布侧栏不受资源页格式和用途筛选影响。
- 手动分类通过现有 asset_manage CAS 接口保存为 `分类:<用途>` 标签，保留其他标签，不改资源文件、版本、节点和引用；因此无需更改或重启 HTTP／MCP 服务。
- `npm run build` 和 41 项测试通过。`node scripts/category-smoke.mjs` 使用隔离登记库，HTTP 与真实 stdio MCP/AppBridge 内嵌界面均通过分组、折叠、搜索、手动分类持久化、刷新、版本不变、用途筛选及画布筛选隔离验收。截图 `var/qa/categories-http.png`、`var/qa/categories-mcp.png`。
- 已核实 LaunchAgent 和安装缓存 MCP 配置指向缓存外 runtime，4310 端口和 `.game-studio` 登记库保留。仅同步修改过的 UI／共享分类源码、HTTP／MCP 产物及本文档；无服务端变更，未重启服务或重装插件。备份 `hot-update-backups/20261011-191159-asset-categories`。安装缓存的真实 stdio MCP 读取到与源码一致的内嵌 HTML，并成功校验原模型文件。新开隐藏验证页打开「星铸者 → 林谷遗迹」，显示角色 1、装备 6、植被 6，DOM 版本为 `usage-v1`；前后快照确认资产和所有画布内容完全一致。
- 原主窗口未暴露为当前会话的 MCP／IAB 标签，Computer Use 禁止操作 Codex 应用。重新打开工作台的 app 请求返回 queued，仍无法确认原主窗口已刷新。新验证页已生效不能代表旧主窗口已热更新；用户需重新打开工作台。

## 2026-10-23：v0.1.8 发布验证

- 汇总 Monster 骨骼分类、姿态特征与训练、动作先验、复考接口、Colab 配置，以及逐帧多视角轮廓审查和姿态桥接；Monster 离线包版本为 0.2.1。
- `npm run build`、`npm run check:distribution`、70 项工作台测试、25 项 Monster CPU 检查、8 项逐帧审查测试与文档检查通过。逐帧测试在临时 Python 环境中运行，包含 NumPy、Pillow、OpenCV 和 SciPy；未执行 GPU 训练或真实 Blender 动画质量验收。
- 已核实本地 LaunchAgent、插件源与 Codex 安装缓存均使用稳定 runtime；插件登记为 0.1.8 并启用。HTTP 服务重载，新安装缓存的 MCP 握手返回 0.1.8，内嵌 HTML 与构建文件一致。项目登记保留。
- 已在本聊天侧栏观察到新版工作台。原生主窗口受 Computer Use 访问限制，无法证明此前打开的主窗口已刷新；已有 MCP 连接也不能以新连接握手代替验证。
