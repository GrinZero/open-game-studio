# 当前实现与操作入口

## 规范来源与迁移适配

[SOP 快照](workflow-and-audit.md) 来自用户指定的 `star-maker/docs/开发管理/MonsterStudio完整流程与用户审计规范.md`。该快照随技能分发；不要求目标机器存在原 star-maker 仓库。规范中的质量要求和用户关卡保持原意。

SOP §7.1 的旧 Cell 1 向 `MonsterStudio/src` 同步代码，已改为直接从 GitHub 拉取 `packages/monster`（或解包可选的本地部署包）至 `/content/monster-releases/`，不覆盖 Drive 的原 `src` 备份。Notebook 仍按 Cell 1 → 2 → 3 → 4 准备和启动，并在 Cell 5 提供了实时任务进度监控面板（显示当前阶段、轮次、批次、loss、GPU 状态与预计剩余时间，自动定时刷新）；Cell 1 默认启用 `H3_PROFILE="full"`（BF16/FP16 满血权重）、默认双视图 `[Front | Side]` 与 `1344×768` 分辨率、默认启用 `ENABLE_MOCAP_PRIORS=True`（`MONSTER_ENABLE_MOCAP_PRIORS=1`，启用置信度自适应 1-Euro 滤波、躯干刚性簇 ICP 根姿态、球窝/铰链/扭转关节分类先验、接触感知脚部钉扎及父子骨向量结构损失），并提供 `ENABLE_2K_RESOLUTION`（`MONSTER_ENABLE_2K=1`，切换为 `2560×1440` 双视图拼合＝单面板 `1280×1440`）、`ENABLE_3VIEW_STUDIO`（`MONSTER_ENABLE_3VIEW=1`，三视图 AB 测试总开关：切换为单画布正视居中三联画 `[Left | Front | Right]`，768P 下 `1344×768`＝单面板 `448×768 × 3`，2K 下 `2688×1440`＝单面板 `896×1440 × 3`）以及 `ENABLE_DENSE_SURFACE=False`（默认关闭 `MONSTER_ENABLE_DENSE_SURFACE=0`；显式设为 `True` 时开启稠密表面坐标＋不确定度门控＋Stage 6 Warm-Start 可微 LBS 叶骨轴向扭转精修）特性开关；Cell 2 使用独立 SkinTokens/bpy 环境（`studio_bpy_worker.py` 默认启用 Cycles 硬件 GPU 光追＋`scene.cycles.denoising_use_gpu=True` GPU OpenImageDenoise＋`use_persistent_data=True` 常驻显存＋`/dev/shm` 内存盘暂存＋表面可见性静态拓扑缓存），Cell 3 准备 ComfyUI/H3 及 2K 两阶段潜空间超分与分块精修节点/权重（`Comfyui_Minimax_h3_latent_Upscaler` + `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`；`dinov2_pose_trainer.py` 默认启用冻结主干融合 SDPA 注意力、CUDA TF32 / `cudnn.benchmark`、`persistent_workers=True` 异步 DataLoader 与批量 `torch.inference_mode()` 推理），Worker 监听 7865、写入 Drive 登记文件并在控制台输出一行 `connect` 配对命令。

## 本机固化 SOP 执行器与云端部署

操作前定位包含 `packages/monster/package.json` 的源码仓库或已安装运行目录，核实 npm workspace；不要在任意游戏项目或只有技能的插件缓存目录执行这些命令。

```sh
npm run monster:check
npm run monster:package
npm run monster:sop -- --help
```

全流程 SOP 的本地执行层已固化在 [monster_sop.py](../scripts/monster_sop.py)，包含 8 个标准子命令，各子命令与能力如下；脚本复用与按需补充原则见 [Monster 工作流](../SKILL.md)：

| 子命令 | 固化脚本 | 覆盖 SOP 阶段与核心职责 |
| --- | --- | --- |
| `gate` | [sop_gate.py](../scripts/sop_gate.py) | 维护 `reviews.json` + `stage_state.json`，计算 SHA-256 指纹，校验 `<KIND>_<SIDE>_<GROUP>_<INDEX>` 规范骨骼命名门禁，执行 `G1`/`G2`/`G3`/`final_batch` 下游硬门禁断言与 10 类变更分级失效推导 |
| `preflight` | [sop_preflight.py](../scripts/sop_preflight.py) | 阶段一工作区与 Git 红线守卫（阻断未获 `final_batch` 批准前写入 `art_source/` 或受管清单）、7 项模型体检、装饰拆离备份 `decorations_manifest.json`、预绑定骨骼规范命名门禁与 U1 Google Drive 持久同步 |
| `g1-pack` | [sop_g1_pack.py](../scripts/sop_g1_pack.py) | 阶段二 4 候选装饰恢复、强制 `<KIND>_<SIDE>_<GROUP>_<INDEX>` 骨骼规范命名重命名与核验表、16 张标准八面体（Octahedral）骨骼叠加图、4 类怪物形态同动作对照视频与 `g1_review.html` 生成 |
| `joint-limits` | [sop_joint_limits.py](../scripts/sop_joint_limits.py) | G1 通过后几何轴向世界空间语义探测＋LBS 蒙皮拉伸安全上限测算（`probe`），以及 Agent 专属定制 `scratch/work/<id>/joint_limits.json`（`apply`/`show`，Schema `monster-joint-limits-v1`），定义动作无关的完整生理运动包络与全局 `ROOT` 旋转/平移范围（修改仅失效 G2/G3，保留 G1） |
| `g2-pack` | [sop_g2_pack.py](../scripts/sop_g2_pack.py) | 阶段三自动校验 G1 硬门禁、加载 `joint_limits.json` 并按锁定原生分辨率与双/三视图排布执行 5 模态分层边界探索随机采样生成 `pose_001`～`pose_050`、2D 标签叠加图、单屏 50 缩略图总览 `g2_50_overview.jpg`、双侧边界触达率审计、异常重抽日志 `rejection_redraw_log.json` 与交互式 `g2_review.html` |
| `g3-exam` | [sop_g3_exam.py](../scripts/sop_g3_exam.py) | 阶段四自动校验 G1+G2 硬门禁、U2 增量持久化同步（比对 SHA-256 跳过未改动 `.blend`）、严格区分 `3d_gt` / `2d_only` / `unlabeled_video` 三类边界（绝不编造 MPJPE/SO(3)）、生成并排对照视频与 `g3_review.html` |
| `deliver` | [sop_delivery.py](../scripts/sop_delivery.py) | 阶段五自动校验 G3 硬门禁、3 次重试极线同步失败硬终止阻断、当批独立清单 `batch_manifest.json` 过滤、本地 GLB 回读与 `:loop` 首尾接缝跳变检测（`compute_loop_seam_metrics`）及 `delivery_review.html` |
| `colab-guard` | [sop_colab_guard.py](../scripts/sop_colab_guard.py) | 阶段六跨会话租约表 `active_sessions.json`、阶段前置释放校验（仅允许在 `stage1_candidates_fetched` 或 `final_batch_approved` 释放，视频审阅/迭代期间硬阻断释放）与安全下线核验 |

### 本地预检与真实证据

`preflight` 优先使用可导入 `bpy` 的 Python；本机也会发现已安装的 Blender。可用 `MONSTER_BPY_PYTHON` 或 `MONSTER_BLENDER` 指定路径。没有可用后端、Worker 报错或产物缺失时必须失败，不生成占位图或有效 U1 输入；旧成功报告在每次执行前立即失效。

预检保存完整装饰备份、保留分件装配位置的 GLB、导出重导入检查、几何统计和真实 `1344×768` 正/侧视图。几何统计会在副本中合并 glTF UV/法线缝处的重合顶点，不修改源模型；边界、接触面和装饰语义仍需实际检查。不会推测固定的肢体遮挡率。

自动检查通过后，首次执行仍返回 `status=awaiting_preflight_review`、`passed=false`（CLI 退出码 2）。执行者查看报告、正侧视图及必要的备份工程后，记录实际姿态、朝向、遮挡、装饰与开口判断；此记录是预检检查证据，不替代 G1/G2/G3 用户关卡，也不能绕过自动检查发现的几何错误：

```sh
npm run monster:sop -- preflight --monster-id <id> --input <source.glb> --project-root <project>
npm run monster:sop -- preflight --monster-id <id> --project-root <project> --review-visual --review-note '<实际观察结论>'
```

`--review-visual` 只登记已查看的当前证据，不重新运行 Blender；可在该命令添加 `--sync-u1 --drive-root <MonsterStudio目录>` 完成持久同步。U1 校验源模型、输入 GLB、检查图、装饰清单和完整备份的 SHA-256；任何证据改变、遗失或旧版 `passed=true` 报告均要求重新预检。

`g1-pack` 同样复用 `_worker_command` 调用本机 Blender 或 `bpy` 环境执行 `--mode prepare_g1_candidates`，要求 4 套真实候选 GLB（`candidate_0..3.glb`）与 `ffmpeg` 均可用：在 Blender 中完成分件保持、装饰恢复（`rigid` 绑骨 / `flex` 独立蒙皮子网格）、全网格 `<KIND>_<SIDE>_<GROUP>_<INDEX>` 规范重命名、未压缩 `.blend` 保存（`compress=False`，兼容 Blender 5.x 与工作台回读）、16 张真实 3D 网格＋3D 八面体骨骼叠加图（`BLENDER_WORKBENCH`）、按形态模板驱动的真实变形拉伸指标审计（`follow_through_issues.json`）以及 `front`/`side`/`three_quarter` 三视角 2×2 四候选同步对比视频编码（校验 MP4 `ftyp`、`moov` 与非空 `mdat` 盒子），并写入 `g1_report.json` 记录全部 SHA-256 证据，绝不回退模板火柴人图或 24 字节占位 MP4。

日常可直接通过 GitHub 的 **Open in Colab** 打开 `packages/monster/notebooks/colab.ipynb` 启动云端 Worker，无需手动打包或上传 `.tar.gz`。如需测试本地未提交改动，再使用 `packages/monster/dist/` 下的打包归档。同步本机插件运行目录不会切换已有 Colab 实例。任务忙碌时不可切换代码版本。

持久数据沿用 `MyDrive/MonsterStudio/monsters/<monster_id>/`，环境缓存沿用 `SkinTokens/` 和 `MonsterStudio/cache/h3_models/`。模型持久传输遵循 U1/U2；只改脚本、关节范围、seed、提示词不重传模型，新增考试视频单独持久化。

SkinTokens 的 `uv` 活动缓存使用 VM 本地目录；`scripts/uv_cache_session.py` 在启动时从 Drive 的 `SkinTokens/cache/uv_snapshots/<兼容性键>/` 恢复已核验归档，等待安装进程退出后再回存新增缓存并读回核验。已有本地缓存不覆盖，缓存未改变不重复上传，归档按不可变批次保留且不自动清理；FlashAttention wheel 与模型权重继续复用既有 Drive 缓存。仅修改缓存脚本不切换正在忙碌的 Worker，也不产生新的模型审计通过状态。

## 实际 API 与边界

先检查所部署版本的 `contracts/protocol.json`、Worker 源码与实际 endpoint。当前协议版本为 1，提供 `heartbeat`、`rig_candidates`、`train_monster_probe`、`reexam_only`、`generate_actions_batch`、`shutdown` 共 6 个注册端点。

`integration/monster_client.py` 提供 Python 客户端和 `npm run monster:call`（首次调用自动准备 `packages/monster/.venv`，且默认自动发现本机 Google Drive 挂载的 `endpoint_registry.json` 或 `var/monster/endpoint_registry.json`）。认证是 Gradio 用户名/密码；登记文件的 `token` 作为密码使用，不是 Bearer Token。登记文件是私密运行数据，不加入 Git 或审阅材料。客户端在真实心跳上核对协议版本，不能只看可能过期的登记文件。

本机可执行状态查询或零重训快速复考：

```sh
npm run monster:call -- status
npm run monster:call -- reexam --monster_id <id>
```

`rig_candidates` 支持不传临时文件、直接从 Google Drive U1 持久目录（`MyDrive/MonsterStudio/monsters/<id>/u1/rig_input_variant.glb`）读取输入并归档全部 4 套 `candidate_0..3.glb` 至 Drive。

## 已固化的代码级保障与运行核查表

执行时仍须核对远端实际运行的 Worker 版本是否已同步最新代码：

| 要求 | 已固化代码保障 | 运行时核查约束 |
| --- | --- | --- |
| 视频反馈期间保留 GPU | `ColabLifecycleGuard` 默认 `idle_timeout_sec=0`（禁用 600s 空闲自毁）；`api_generate_actions_batch` 默认 `auto_shutdown_after=False`；本地 `sop_colab_guard.py` 维护跨会话租约与阶段释放前置条件 | 若远端仍运行旧版 Worker，须重新运行 Cell 1 + Cell 4 更新服务端，或通过 `heartbeat` 确认远端配置 |
| G1/G2/G3 与骨骼命名硬门禁 | 本地 `sop_gate.py`、`sop_preflight.py`、`sop_g1_pack.py`、`sop_g2_pack.py`、`sop_g3_exam.py`、`sop_delivery.py` 在入口自动执行 `assert_gate` 与 `validate_canonical_rig_bone_names` 硬拦截 | 严禁绕过 `monster_sop.py` 直接调用底层渲染或烘焙函数跳过用户关卡 |
| 四候选完整工程与 G1 材料 | `api_rig_candidates` 自动归档 `candidate_0..3.glb` 至 Drive；`sop_g1_pack.py` 自动完成装饰恢复、规范重命名、16 张八面体骨骼图、同动作对照视频与 `g1_review.html` | 自动 audit 分数和候选 ZIP 不替代用户 G1 审批 |
| G3 已有视频考试与快速复考 | `api_reexam_only` 已在协议与 Gradio 端点正式注册；`api_train_monster_probe` 覆写前自动备份 `backups/dinov2_head_<ts>.pt`；`sop_g3_exam.py` 严格按三类边界生成报告 | 严禁编造无真值视频的 MPJPE / SO(3) 数值 |
| 同步失败硬终止、烘焙分支与当批清单 | `pipeline_runner.py` 在 3 次重试极线同步均失败时抛出 `RuntimeError` 硬终止，禁止将失步视频送入 `bake_ik`；`bake_glb_on_cloud` 控制云端烘焙分支；ZIP 仅打包当批动作并写入 `batch_manifest.json` | 生产交付时运行 `monster_sop.py deliver` 完成本地 GLB 回读与 `:loop` 首尾接缝核验 |
| 工作台 Monster 调度与入库 | 在 `final_batch` 获用户明确批准前，`sop_preflight.py` 拦截任何写入 `art_source/` 或受管清单的操作 | 最终验收通过后再导入正式项目资源目录 |
