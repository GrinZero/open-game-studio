# Monster 许可范围与发布要求

Copyright (c) 2026 GrinZero.

SPDX-License-Identifier: AGPL-3.0-only

## 本模块采用的许可

从 Monster 0.2.0 起，本项目持有版权的 Monster 代码、文档、Notebook、部署脚本、客户端、协议及技能以 **GNU Affero General Public License，版本 3 only** 提供。完整原文见 [LICENSE](LICENSE)；不附加“禁止商用”条款，也不自动许可未来版本的 AGPL。

该范围包括 `packages/monster/` 中的项目原创部分，以及构建导出的 `skills/monster-workflow/` 技能副本。其余工作台代码继续使用仓库默认 MIT。第三方许可文本本身及第三方内容按各自条款提供；本说明不重新许可第三方内容。

最初两份 MonsterStudio Notebook 的作者已确认：其中自写的流水线与算法代码由其持有版权。依赖、复制的第三方实现及模型权重仍须按第三方条款处理，不能仅凭 Notebook 归属把它们重新许可。

## 商业使用与源码义务

AGPL 允许商用、收费分发和收费托管服务，不要求合规商业用户向本项目付费。其约束在于使用者须保留下游获得源码、修改和再分发的权利。

- 分发受许可覆盖的软件或修改版时，保留版权及许可声明，按第 4–6 条等适用条款交付对应源码；二进制交付不能替代源码义务。
- 修改后的程序通过网络与用户交互时，按第 13 条，在界面提供明显入口，让这些用户免费取得所运行版本的对应源码。只链接到随时变化的上游 `main` 分支，不能证明交付了修改版源码。
- 对应源码包括构建、安装和运行该受覆盖程序所需的源码、接口定义及相关控制脚本。部署者应提供与其实际修改匹配的源码归档和必要依赖来源；本仓库生成的部署归档不能替代另行修改或组合后的源码交付。
- 不应把独立服务、独立客户端、整个网站或所有部署基础设施都概括为必须 AGPL 开源；是否构成组合程序或衍生作品取决于实际集成方式，不能仅凭使用 HTTP 就保证互相独立。
- 生成的 GLB、视频、训练输入和用户项目不会仅因运行 Monster 自动变为 AGPL；仍应核查其中包含的代码、素材、权重和模型输出条款。

这些说明概括使用意图，不修改 [AGPL 正文](LICENSE)。按原样部署、仅作私人修改、分发修改版和提供网络交互服务的义务并不完全相同。

## 既有 MIT 授权

改许可不追溯撤回已经授予的 MIT 权利。历史 MIT 版本及其中已获授权的代码仍可按原条款使用或分叉；AGPL 不能阻止他人继续使用那些版本。为保留历史声明，随包附上 [原 MIT 文本](licenses/Monster-prior-MIT.txt)。

本修订及后续贡献采用上述 AGPL 范围；不能声称把历史上已获 MIT 授权的代码变为只能按 AGPL 使用。今后引入其他作者的代码时，应确认其许可或获得改许可授权，保留适用声明。

## 上游边界

核心依赖及模型见 [第三方清单](THIRD_PARTY.md)。本模块的 AGPL 不授予 SkinTokens、Blender、ComfyUI、H3 等上游的额外权利。

- MIT、BSD 与 Apache 的代码可在满足各自声明和其他条件后纳入适用 AGPL 的作品，原许可和版权声明仍需保留。
- ComfyUI 与 VideoHelperSuite 使用 GPL v3；Blender 的核心使用 GPL-2.0-or-later，所用 bpy 5.2 渲染与 4.5 SkinTokens 内部路径须考虑 GPL v3 选项及 GPL/AGPL 第 13 条的组合规则。上游 GPL 部分不因此改成 AGPL。`studio_bpy_worker.py`、`pose_visibility.py` 及部分评测代码直接调用 Blender API，不能宣称它们与 Blender 一定相互独立。
- H3 的权重社区许可包含地域、商业门槛、展示及用途义务；它不是本模块的 AGPL。当前部署脚本单独下载并通过 ComfyUI 服务使用权重，没有把权重装入源码归档。此结构并非对任何再打包、组合分发或商业部署的全面合规保证。

## 发布前与部署时

运行 `npm run monster:check` 和 `npm run monster:package`，确认源码归档含 `LICENSE`、本说明、`THIRD_PARTY.md` 及 `licenses/` 中的保留文本。独立分发导出的技能时也保留其 `LICENSE.txt`、`PRIOR-MIT.txt` 和 `LICENSING.md`。

提供网络服务的部署者负责履行适用的 AGPL 源码提供义务，并核查模型许可；仓库中的一份许可文件不会自动建立在线源码下载入口，也不会替部署者接受上游协议。若同时分发第三方仓库、模型或二进制，应额外提供这些组件要求的版权声明、许可、NOTICE、对应源码及修改说明。

核查依据：[GNU AGPL v3](https://www.gnu.org/licenses/agpl-3.0.en.html)、[GNU 许可 FAQ](https://www.gnu.org/licenses/gpl-faq.en.html)、[Blender 许可 FAQ](https://www.blender.org/support/faq/)。本次核查覆盖核心直接上游；未对 Colab 中全部传递依赖、CUDA 组件、FFmpeg 构建或用户素材完成穷尽审计。
