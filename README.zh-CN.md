<p align="center">
  <img src="docs/images/banner.svg" alt="Open Game Studio — Your local game production workbench for Codex" width="100%" />
</p>

<p align="center">
<a href="README.md">English</a> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.zh-TW.md">繁體中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.ko.md">한국어</a>
</p>

<h1 align="center">Open Game Studio</h1>

<p align="center"><strong>在 Codex 里，把游戏创作整理成一个本地工作台。</strong></p>

<p align="center">
  <img src="docs/images/badges.svg" alt="Alpha · Node.js 24+ · Codex MCP Apps · Local SQLite · MIT License" width="100%" />
</p>

参考图、文档、媒体和 3D 资源放到同一张画布上，版本、制作决策与导出记录都能追溯到本机项目文件。

> **Alpha · macOS 安装器 · MIT 许可证。** 本地整理可独立使用；AI 制作需要自行配置工具与账号。README 提供多语言版本，工作台界面目前仍使用中文。

[试用](#try-it) · [安装](#install) · [功能](#features) · [预览](#gallery) · [文档](#documentation) · [边界](#limits)

---

<a id="try-it"></a>

## 先试一句话

安装后，在新的 Codex 聊天里粘贴：

```text
打开 Open Game Studio 工作台。
在我选择的文件夹里创建「林谷遗迹」项目。
把角色目标文档和参考图放到画布上，
连接为角色制作分组的输入。
开始制作前，先展示冻结的输入清单。
```

工作台负责整理与查看，对话负责表达制作意图。连线提供输入，启动制作是单独的操作。

还没有自己的文件组织习惯？可以阅读可选的[项目文件指南](skills/studio-project/references/project-file-guide.md)，然后说：`在我的游戏项目中采用 Open Game Studio 文件规范的标准方案。` 可以只选部分规则；已有项目保留原布局，迁移需另行明确提出。

<a id="install"></a>

## 安装与打开

需要 **macOS**、**Node.js 24+**（含 npm），以及提供 `codex plugin` 命令的 Codex CLI。内嵌工作台还需要宿主支持 MCP Apps。

执行一行命令安装（或手动克隆仓库运行安装器）：

```sh
curl -fsSL https://raw.githubusercontent.com/GrinZero/codex-game-studio/main/scripts/bootstrap.sh | bash
```

```sh
git clone https://github.com/GrinZero/codex-game-studio.git
cd codex-game-studio
npm ci
npm run install:local
```

安装器构建两套工作台、复制独立运行时、安装 `game-studio@game-studio-local`，并通过用户级 LaunchAgent 启动浏览器工作台。目前使用本地源码安装，不提供 npm 包或远程插件市场安装方式。

1. 开启 **新的 Codex 聊天**，输入：`打开 Open Game Studio 工作台。`
2. 宿主显示已登记入口时，也可在侧边栏 **…** 菜单中选择 **Open Game Studio**。入口未出现时重启 Codex。
3. 备用浏览器入口为 [http://127.0.0.1:4310](http://127.0.0.1:4310)。在仓库目录运行 `npm run demo`，即可导入原创的 **星铸者 · 示例**。

示例将插画与目标文档导入两张画布，不执行生成、审核或引擎验收。

更新、卸载、自定义路径和直接 MCP 接入见 [安装配置](docs/SETUP.md)。

<a id="features"></a>

## 可以做什么

| 能力 | 实际用途 |
| --- | --- |
| 本地项目 | 新建或关联文件夹、设置封面，保留已有文件的组织方式。 |
| 资源版本 | 按文件集导入，记录 SHA256 与不可变历史；新增版本不会替换画布已引用版本。 |
| 自由画布 | 摆放资源与便签、连接输入、跨画布复用版本，并撤销或重做布局操作。 |
| 文档与媒体 | Markdown 编辑保存为新版本；预览图片、音视频和自包含 GLB，控制模型动画。 |
| 资源制作 | 准备图片、视频、模型、骨架动画或音频任务，冻结参考后调用本机 Codex 与已配置工具。 |
| 角色流程 | 按目标 → 参考 → 候选 → 外形审核 → 组装 → 绑定 → 变形审核 → 导出推进，保存决策并局部重跑。 |
| 可追溯交付 | 按明确版本导出，附文件清单与哈希；引擎验收单独验证。 |

具体操作见 [工作台指南](docs/WORKBENCH.md)。

<a id="gallery"></a>

## 工作台预览

截图来自隔离的本地示例环境，使用仓库原创素材，展示当前中文界面，不代表已完成的生产资产。

**项目库 · 深色主题**

![项目库 · 深色主题](docs/images/projects.png)

**连线画布 · 深色主题**

![连线画布 · 深色主题](docs/images/canvas.png)

[Image provenance](docs/images/README.md)

<a id="documentation"></a>

## 工具与文档

**26 个领域 MCP 工具 + 1 个界面专用桥接工具 · 11 个工作流技能。** HTTP 与 MCP 内嵌工作台共用项目核心与资源历史。

| 入口 | 内容 |
| --- | --- |
| [安装、数据路径与 MCP 配置](docs/SETUP.md) | 安装、更新、卸载与直接接入。 |
| [工作台指南](docs/WORKBENCH.md) | 画布编辑、版本选择与制作交接。 |
| [工具契约](docs/TOOL_CONTRACT.md) | 完整工具索引、输入 schema 与状态规则。 |
| [验证记录](docs/VERIFICATION.md) | 已执行检查与仍需实测的能力。 |
| [实现状态](docs/IMPLEMENTATION.md) | 架构、未完成阶段与后续计划。 |
| [Monster 模块](packages/monster/README.md) | [Open in Colab](https://colab.research.google.com/github/GrinZero/codex-game-studio/blob/main/packages/monster/notebooks/colab.ipynb) 一键启动、Worker 源码与自动配对本机客户端。 |

<a id="limits"></a>

## 当前边界

- **仍处于 Alpha：** 本地项目、版本管理和制作流程已实现；内置 Provider API、密钥管理、完整费用面板、游戏目录自动扫描、数据库备份和 Blender/Godot 验证 worker 仍未完成。

- **制作工具自行配置：** 本机 Codex 执行器不捆绑模型服务；可用能力、登录、费用与提交确认取决于具体工具。缺工具或需要确认时明确记录失败原因，不生成占位资产。停止本地任务不会取消已提交的远端工作。

- **预览不等于验收：** GLB 结构检查与文件哈希不能证明网格质量、Blend 可编辑性、骨架变形或引擎兼容性。导出清单的引擎验证保持 `not_verified`。

- **固定版本保持不变：** 新资源版本不会替换画布或审核已采用版本。暂不导入 SVG 或依赖外部文件的 glTF；撤销/重做目前仅覆盖布局。

- **宿主表现有差异：** MCP App 入口已登记并通过沙箱测试，原生 Codex 中的显示、缓存与刷新行为取决于宿主。

<a id="contributing"></a>

## 参与开发

开发命令、检查、构建标识与运行目录同步见 [CONTRIBUTING.md](CONTRIBUTING.md)。报告问题时附界面构建标识、宿主、复现步骤与预期结果；移除凭证和私有项目文件。

<a id="license"></a>

## 许可与来源

本项目采用 [MIT 许可证](LICENSE)。依赖许可与原创示例来源见 [第三方声明](licenses/THIRD_PARTY.md)。

README 展示方式参考 [Tripo Studio for Codex](https://github.com/GrinZero/tripo-studio-plugin)；这里的横幅与示例截图来自本项目。
