# Monster

[![Open in Colab](../../docs/images/open-in-colab.svg)](https://colab.research.google.com/github/GrinZero/codex-game-studio/blob/main/packages/monster/notebooks/colab.ipynb)

Monster 是 Open Game Studio monorepo 中的角色绑定、姿态训练与动作制作模块。Python Worker、客户端、协议、Colab 入口和部署脚本集中在本目录维护。本机工作台继续使用仓库根目录的 `src/`，本轮不搬动其启动路径。

## 目录与运行边界

| 目录           | 用途                                                    | 运行位置       |
| -------------- | ------------------------------------------------------- | -------------- |
| `worker/`      | SkinTokens 候选绑定、Blender、DINOv2、MiniMax H3 流水线 | Colab G4       |
| `integration/` | 有认证、版本握手及文件下载的 Python 客户端与 CLI        | 本机后端或终端 |
| `contracts/`   | 实际 Gradio API 的参数顺序、输出与协议版本              | 两端           |
| `notebooks/`   | 配置、安装指定部署包、准备环境和启动                    | Colab          |
| `scripts/`     | 检查、打包、安装、Colab 生命周期                        | 按入口区分     |
| `skills/`      | [Monster 工作流技能](skills/monster-workflow/SKILL.md)与 SOP/审计参考 | 本机插件技能目录 |
| `tests/`       | 源码迁移、协议、安装和客户端的 CPU 测试                 | 本机与 CI      |

## 在 Colab 启动（推荐：一键打开）

[![Open in Colab](../../docs/images/open-in-colab.svg)](https://colab.research.google.com/github/GrinZero/codex-game-studio/blob/main/packages/monster/notebooks/colab.ipynb)

1. 直接点击上方 **[Open in Colab](https://colab.research.google.com/github/GrinZero/codex-game-studio/blob/main/packages/monster/notebooks/colab.ipynb)** 徽章打开仓库中的 [Notebook](notebooks/colab.ipynb)，选择 G4 GPU。
2. 依次运行 Cell 1（默认直接从 GitHub 拉取最新代码并配置，默认使用 `H3_PROFILE="full"` 满血权重与 `1344×768` 双视图分辨率，可选开启 `ENABLE_2K_RESOLUTION=True` 切换至 `2560×1440` 2K 全链路模式）、2（SkinTokens）、3（ComfyUI/H3）、4（启动 Worker）。首次安装和模型下载仍需等待；之后复用既有 Drive 缓存。启动后可执行 Cell 5 查看实时任务进度（当前阶段、轮次、批次、loss、GPU 状态与预计剩余时间，自动定时刷新）。
3. Worker 写入私有的 `MonsterStudio/runtime/endpoint_registry.json`（含 Gradio 地址、用户名、密码及协议版本），并在控制台输出一行 `connect` 配对命令。

如果需要测试尚未推送到 GitHub 的本地修改，也可运行 `npm run monster:package`，把生成的 `packages/monster/dist/monster-<version>.tar.gz` 上传到 Google Drive 的 `MyDrive/MonsterStudio/releases/`，并在 Cell 1 填写 `RELEASE_ARCHIVE`（或打开同次生成的 `.ipynb`）。

代码从 `/content/monster-releases/` 加载，临时计算文件仍在 `/content/monster-studio/`，持久数据仍在 `MyDrive/MonsterStudio/`。旧 `src/` 备份不会被覆盖，也不会被当作新的代码来源。Cell 1/4 在已有任务忙碌时拒绝重载；空闲时关闭旧 Gradio App 再加载新版本。安装器不改动模型缓存、训练结果或项目文件。

### SkinTokens 依赖缓存

`bootstrap-skintokens.sh` 通过 `uv_cache_session.py` 管理缓存生命周期。Drive 在 `SkinTokens/cache/uv_snapshots/<兼容性键>/` 保存不可变归档与校验记录；启动时校验归档并恢复到 VM 本地的 `/content/skintokens-service/cache_local/uv`，所有 `uv` 安装与锁操作都在本地执行。键包含 Python/CUDA/PyTorch、uv 版本及上游固定提交。

安装进程退出后才打包新增缓存、上传并实际读回校验；缓存未改变时跳过回存。安装中断不会发布正在写入的归档，损坏的新快照会回退到旧的有效快照。已有本地缓存不会被旧快照覆盖。模型权重与编译好的 FlashAttention wheel 继续使用 Drive 中既有的持久目录；不自动清理历史归档，也不把缓存或凭据打进发布包。此方式沿用 [uv 官方跨运行恢复与保存缓存的做法](https://docs.astral.sh/uv/guides/integration/github/#caching)，并避开 DriveFS 的原子锁限制。

通过分段恢复取得的 PyTorch wheel 可保存在 `SkinTokens/cache/wheels/py311-torch291-cu130/`。启动脚本优先将它复制到 VM 并校验固定版本的官方 SHA-256，再交给 uv 安装；缺失或损坏时使用原固定上游索引。这样不会因本地文件与索引 URL 的缓存键不同而再次下载大文件。

## 在本机连接与调用（零配置环境与自动发现）

本机不安装 CUDA、torch 或 bpy。首次执行 `npm run monster:call` 时，[`run-python.mjs`](scripts/run-python.mjs) 会自动在 `packages/monster/.venv` 创建虚拟环境并安装 `gradio_client`，无需手动建环境或设置 `MONSTER_PYTHON`。

**配对方式（二选一）：**
- **方式 A（已安装 Google Drive 桌面版，零操作）**：客户端会自动发现 `~/Library/CloudStorage/GoogleDrive-*/MyDrive/MonsterStudio/runtime/endpoint_registry.json`，无需传 `--registry`。
- **方式 B（未安装网盘桌面版，一行命令配对）**：复制 Colab Cell 4 输出的配对命令在本机运行一次（会自动验证心跳并保存到 Git 忽略的 `var/monster/endpoint_registry.json`）：
  ```sh
  npm run monster:call -- connect '{"url":"https://xxxx.gradio.live","username":"monster","token":"..."}'
  ```

配对或自动发现后，即可直接调用：

```sh
npm run monster:call -- status
npm run monster:call -- rig creature_01 /absolute/path/input.glb
npm run monster:call -- train creature_01 --samples 5000 --epochs 15
npm run monster:call -- animate creature_01 'idle:loop walk:loop'
npm run monster:call -- shutdown
```

如需指定其他登记文件或输出目录，仍可显式传入 `--registry /absolute/path/endpoint_registry.json --output /absolute/path/results`。

## 检查与离线打包

在仓库根目录运行（Node.js 24+、Python 3.11.8+）：

```sh
npm ci
npm run monster:check
npm run monster:package
```

这两个命令仅使用 Python 标准库，不安装模型、调用 GPU 或提交生成任务。产物位于 `packages/monster/dist/`：

- `monster-<version>.tar.gz`：部署代码、依赖清单、AGPL 许可、许可范围及第三方声明。
- 同名 `.ipynb`：已填写该部署包路径的 Colab 入口。

版本号来自 [package.json](package.json)。只打包明确允许的源文件，不包括模型权重、项目、运行凭据、缓存或训练结果。CI 会检查并上传部署产物，不会自动部署或启动 GPU。

认证沿用原 Worker 的 Gradio 用户名/密码（登记文件里的 `token` 是密码，不是 Bearer 头）。客户端先读取真实心跳检查 `protocol_version`，再按 [协议](contracts/protocol.json) 调用；文件输入通过 Gradio 上传，ZIP 输出由客户端下载到指定目录。登记文件和密码不提交到仓库，CLI 结果不打印密码。

客户端不自动重发结果未知的任务。连接与 `status` 会触发一次心跳并重置 600 秒空闲计时，不提供持续保活轮询。客户端的动作生成默认不自动释放 GPU，便于先下载结果，完成后可显式调用 `shutdown`；原 Worker API 的自动释放参数和看门狗行为保留。Python 调用方也可导入 `integration/monster_client.py` 中的 `MonsterClient`。

## 当前接入范围

本轮完成代码归属、版本化部署、实际 Worker API 客户端和安装/热更新目录支持。工作台现有 `generation_start` 仍通过本机 Codex 执行；尚未增加 Monster 连接面板、任务调度或自动资源入库。CLI 下载后的产物可通过工作台现有导入功能登记。后续接入应复用已有项目、任务和资源版本，不能新建第二套项目数据库。

协议版本独立于算法 `FEATURE_VERSION` 和工作台 `studioBuild`；本轮协议版本为 1，包版本为 0.2.0。单纯本地构建不会把云端 Worker 更新到同一版本。

## 来源与验证边界

来源是用户提供的 [MonsterStudio Notebook](https://drive.google.com/file/d/1Zs1rJSnQC9fa7G76Qdj6ALmArBuom179/view) 与优化版 [Colab Notebook](https://colab.research.google.com/drive/1hmk2iXWntDqXbujEE1i3_9hCcsPLvcuR)（实际构建标识为 `v2.3-20261020004015`，原文件名含 `v2_2_fixed`），已合入热图峰值加权、5×5 峰值概率集中度置信度、2D Hampel 尖峰剔除、5 点 Savitzky–Golay 坐标/四元数滤波及受限局部欧拉多子骨刚性杠杆臂 IK 求解器；不把原 Notebook 的重复 Base64 源码再纳入维护。

六个 Python 算法脚本和 SkinTokens bootstrap 按原源码提取，给 Worker 心跳和登记文件加上包/协议/发布标识。ComfyUI 安装单元格拆成脚本，改用模块路径、环境参数与独立依赖清单。SkinTokens、FlashAttention、ComfyUI、VideoHelperSuite 和 H3 的原固定提交保留。依赖及模型仍适用各自许可，见 [第三方声明](THIRD_PARTY.md)。

CPU 检查覆盖可解析性、Notebook 入口、实际 API 与契约一致、归档打包、路径越界拦截、版本安装、忙碌重载阻止及客户端调用。它不证明 GPU 安装成功、候选绑定/训练/视频生成质量或 GLB/Blend 的真实工具验收；这些需要在实际 G4 上另行运行。原流程中定义但未注册的 `reexam_only` 不列为可调用 API；原动作 API 的 `bake_glb_on_cloud` 参数尚未控制流水线分支，客户端固定使用现有云端烘焙行为。

## 开源许可

Monster 的项目代码、文档和技能采用 **[GNU AGPL v3 only](LICENSE)**（`AGPL-3.0-only`），版权署名为 `Copyright (c) 2026 GrinZero`。许可范围、旧版 MIT 权利及发布要求见 [许可说明](LICENSING.md)。仓库其他部分的默认 MIT 许可不覆盖本模块或导出的 `monster-workflow` 技能。

AGPL 允许商业使用、收费分发和收费服务。分发时须履行对应源码等义务；修改版通过网络供用户交互时，须按第 13 条向这些用户免费提供该版本的对应源码。这不意味着任何 SaaS、独立客户端或生成资产都自动适用 AGPL，也不要求把独立服务的全部源码公开。

第三方代码、依赖和模型保留自己的许可。特别是 **MiniMax H3 使用带地域、商业门槛和可接受用途限制的社区许可**，不能因为 Monster 改为 AGPL 就视为已获所有商业部署授权。核心上游版本、证据及保留的许可文本见 [第三方声明](THIRD_PARTY.md)。部署归档附带这些说明与许可文本；它不包含模型权重或第三方仓库检出。
