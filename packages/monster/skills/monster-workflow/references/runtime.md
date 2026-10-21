# 当前实现与操作入口

## 规范来源与迁移适配

[SOP 快照](workflow-and-audit.md) 来自用户指定的 `star-maker/docs/开发管理/MonsterStudio完整流程与用户审计规范.md`。该快照随技能分发；不要求目标机器存在原 star-maker 仓库。规范中的质量要求和用户关卡保持原意。

SOP §7.1 的旧 Cell 1 向 `MonsterStudio/src` 同步代码，已改为直接从 GitHub 拉取 `packages/monster`（或解包可选的本地部署包）至 `/content/monster-releases/`，不覆盖 Drive 的原 `src` 备份。Notebook 仍按 Cell 1 → 2 → 3 → 4 准备和启动；Cell 1 默认启用 `H3_PROFILE="full"`（BF16/FP16 满血权重）与默认 `1344×768` 分辨率，并提供 `ENABLE_2K_RESOLUTION`（`MONSTER_ENABLE_2K=1`，切换为 `2560×1440` 双视图拼合＝单面板 `1280×1440`）特性开关；Cell 2 使用独立 SkinTokens/bpy 环境（`studio_bpy_worker.py` 默认启用 Cycles 硬件 GPU 光追＋`scene.cycles.denoising_use_gpu=True` GPU OpenImageDenoise＋`use_persistent_data=True` 常驻显存＋`/dev/shm` 内存盘暂存＋表面可见性静态拓扑缓存），Cell 3 准备 ComfyUI/H3 及 2K 两阶段潜空间超分与分块精修节点/权重（`Comfyui_Minimax_h3_latent_Upscaler` + `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`；`dinov2_pose_trainer.py` 默认启用冻结主干融合 SDPA 注意力、CUDA TF32 / `cudnn.benchmark`、`persistent_workers=True` 异步 DataLoader 与批量 `torch.inference_mode()` 推理），Worker 监听 7865、写入 Drive 登记文件并在控制台输出一行 `connect` 配对命令。

## 本机开发与云端部署

操作前定位包含 `packages/monster/package.json` 的源码仓库或已安装运行目录，核实 npm workspace；不要在任意游戏项目或只有技能的插件缓存目录执行这些命令。

```sh
npm run monster:check
npm run monster:package
```

日常可直接通过 GitHub 的 **Open in Colab** 打开 `packages/monster/notebooks/colab.ipynb` 启动云端 Worker，无需手动打包或上传 `.tar.gz`。如需测试本地未提交改动，再使用 `packages/monster/dist/` 下的打包归档。同步本机插件运行目录不会切换已有 Colab 实例。任务忙碌时不可切换代码版本。

持久数据沿用 `MyDrive/MonsterStudio/monsters/<monster_id>/`，环境缓存沿用 `SkinTokens/` 和 `MonsterStudio/cache/h3_models/`。模型持久传输遵循 U1/U2；只改脚本、关节范围、seed、提示词不重传模型，新增考试视频单独持久化。

## 实际 API 与边界

先检查所部署版本的 `contracts/protocol.json`、Worker 源码与实际 endpoint，不根据文件里存在某个函数就假定它已注册。迁移基线的协议版本为 1，提供 `heartbeat`、`rig_candidates`、`train_monster_probe`、`generate_actions_batch`、`shutdown`。

`integration/monster_client.py` 提供 Python 客户端和 `npm run monster:call`（首次调用自动准备 `packages/monster/.venv`，且默认自动发现本机 Google Drive 挂载的 `endpoint_registry.json` 或 `var/monster/endpoint_registry.json`）。认证是 Gradio 用户名/密码；登记文件的 `token` 作为密码使用，不是 Bearer Token。登记文件是私密运行数据，不加入 Git 或审阅材料。客户端在真实心跳上核对协议版本，不能只看可能过期的登记文件。状态查询会重置旧看门狗计时，不能据此假定自动释放已禁用。

本机可执行一次配对或状态查询：

```sh
npm run monster:call -- status
```

当前客户端的 `rig` 和自定义 Blend 输入会使用 Gradio 临时文件上传，**不符合 SOP 的 U1/U2 持久传输要求**。正式流程须先将模型上传 Drive，并使用已验证的云端挂载路径调用方式或补齐持久路径适配；不要把现有 CLI 当作合规生产捷径。

## 迁移基线尚缺的保障

这是基于 `v2.3-20261020004015` 提取的运行代码事实，执行时须按实际版本重新核查：

| 要求 | 迁移基线事实 | 执行约束 |
| --- | --- | --- |
| 视频反馈期间保留 GPU | `ColabLifecycleGuard` 仍启用 600 秒自动释放；动作 API 默认 `auto_shutdown_after=True`，在 `finally` 释放 | 进入此阶段前实际禁用冲突的自动释放并验证；客户端传 `False` 只覆盖该次批次参数，不会禁用看门狗。不得用持续心跳伪装成已修复。尚未具备保障时保留已完成材料并报告缺口，不启动不符合约定的视频迭代 |
| G1/G2/G3 后端门禁 | API 尚无完整用户决定/版本依赖硬门禁，部分调用会直接进入合成和训练 | Agent 按用户记录约束自己的调度；不得宣称后端已拦截，也不提前调用绕过关卡。代码层保障另行实现、测试 |
| 四候选完整工程与 G1 材料 | 现有绑定 API 不承诺四套恢复装饰后的 Blend、同动作视频和 16 张骨骼图 | 逐项制作并核查审阅包；自动 audit 分数和候选 ZIP 不替代 G1 |
| G3 已有视频考试 | 原训练流程包含合成的 96 帧技术考试；`api_reexam_only` 存在但未注册 | 准备独立已有视频与实际 GLB 回读重渲染链；不能把合成回归或不存在的远程端点当作 G3 |
| 同步失败终止、烘焙分支与当批清单 | SOP §7.1 列为必改项；`bake_glb_on_cloud` 尚未控制分支，批次打包会遍历已有目录 | 生产前核查/补齐并验证相应逻辑；不能仅凭 API 成功或 ZIP 存在宣称达标 |
| 工作台 Monster 调度与入库 | 现有 `generation_start` 仍是本机 Codex 路线，无专用 Monster 面板/自动入库 | 使用实际工具完成授权工作，产物通过现有资源导入登记；不虚构工具、按钮或状态 |

编写技能不修改这些实现，也不授权为读取规范而启动 GPU、上传私有资产或关闭其他会话的实例。若用户任务需要补齐缺失能力，先完成该范围内的实现与验证；技术准备可以继续，正式阶段不能跨越尚未满足的用户关卡。
