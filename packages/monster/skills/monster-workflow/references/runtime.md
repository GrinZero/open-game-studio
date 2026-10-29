# 当前实现与操作入口

## 角色目录契约

[角色独立工作目录规范 v1](character-workspace.md) 定义每个角色的制作根、入口索引、版本依赖和本地/Drive 映射。当前 SOP 脚本使用 `scratch/work/<monster_id>/`、`scratch/checks/<monster_id>/` 和 `.game-studio/monster/<monster_id>/reviews.json`，详见该规范中的路径映射表。

## 云端 Notebook 与运行架构

[完整 SOP](workflow-and-audit.md) 与 [v3 算法与运行参数](video2rig-v3.md) 随本技能分发。Colab Notebook（`packages/monster/notebooks/colab.ipynb`）按 Cell 1 → 2 → 3 → 4 准备和启动，并在 Cell 5 提供实时任务进度监控面板（显示当前阶段、轮次、批次、loss、GPU 状态与预计剩余时间，自动定时刷新）：

- **Cell 1**：挂载 Google Drive，从 GitHub 拉取 `packages/monster`（或解包指定的发布归档）至 `/content/monster-releases/`；默认启用 `H3_PROFILE="full"`（BF16/FP16 满血权重）、双视图 `[Front | Side]` 与 `1344×768` 分辨率，支持配置 `ENABLE_2K_RESOLUTION`（`MONSTER_ENABLE_2K=1`，切换为 `2560×1440` 双视图或 `2688×1440` 三视图）、`ENABLE_3VIEW_STUDIO`（`MONSTER_ENABLE_3VIEW=1`，切换为单画布正视居中三联画 `[Left | Front | Right]`）及训练微调模式 `TRAINING_MODE`。
- **Cell 2**：分别准备 SkinTokens Python 3.11 / 内部 `bpy==4.5.15` 环境与独立渲染 `/content/monster-bpy/venv` Python 3.13 / `bpy==5.2.3` 环境（`studio_bpy_worker.py` 默认启用 EEVEE 8 spp＋最多 16 副本动画条带批量渲染；可选 Cycles 硬件 GPU 光追＋GPU OIDN＋常驻显存，详见 [EEVEE 动画条带渲染](video2rig-v3.md#eevee-动画条带渲染)）。
- **Cell 3**：准备 ComfyUI（`127.0.0.1:8193`）、MiniMax H3、2K 两阶段潜空间超分与分块精修节点/权重（`Comfyui_Minimax_h3_latent_Upscaler` + `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`）及 DINOv2 权重缓存（`dinov2_pose_trainer.py` 默认启用融合 SDPA 注意力、CUDA TF32 / `cudnn.benchmark`、`persistent_workers=True` 异步 DataLoader 与批量 `torch.inference_mode()` 推理）。
- **Cell 4**：启动 Worker 监听 `7865` 端口，写入 `MonsterStudio/runtime/endpoint_registry.json` 并在控制台输出一行 `connect` 配对命令。

## 本机固化 SOP 执行器与云端部署

候选动态展示执行 [候选视频审阅规范 v1](candidate-video-review.md)，其中列明 CLI、HTML 构建器、证据校验及上游 Worker 的入口边界。`g1-pack` 默认输出 12 段独立视频作为首要入口；旧 HTML 调用自动从同步源提取，缺材料时明确失败。历史报告兼容不代表旧页面已迁移。

操作前定位包含 `packages/monster/package.json` 的源码仓库或已安装运行目录，核实 npm workspace；不要在任意游戏项目或只有技能的插件缓存目录执行这些命令。

```sh
npm run monster:check
npm run monster:package
npm run monster:sop -- --help
```

全流程 SOP 的本地执行层已固化在 [monster_sop.py](../scripts/monster_sop.py)，包含 9 个标准子命令，各子命令与职责如下；脚本复用与按需补充原则见 [Monster 工作流](../SKILL.md)：

| 子命令 | 固化脚本 | 覆盖 SOP 阶段与核心职责 |
| --- | --- | --- |
| `gate` | [sop_gate.py](../scripts/sop_gate.py) | 维护 `reviews.json` + `stage_state.json`，计算 SHA-256 指纹，校验解剖学规范骨骼命名门禁（`monster-bone-taxonomy-v3`），执行 `G1`/`G2`/`G3`/`final_batch` 下游硬门禁断言与变更分级失效推导 |
| `preflight` | [sop_preflight.py](../scripts/sop_preflight.py) | 阶段一工作区与 Git 红线守卫（阻断未获 `final_batch` 批准前写入 `art_source/` 或受管清单）、7 项模型体检、自动网格无损清理与 `1.2% H` 双层体素水密壳 `shell.npz` / `watertight_shell.glb` 生成、装饰拆离备份 `decoration_manifest.json`、预绑定骨骼规范命名门禁与 U1 Google Drive 持久同步 |
| `g1-pack` | [sop_g1_pack.py](../scripts/sop_g1_pack.py) | 阶段二 4 候选骨架树几何精修（`prepare_candidate`）、完整装饰恢复、强制解剖学骨骼规范命名重命名与核验表、16 张标准八面体（Octahedral）骨骼叠加图、4 类怪物形态同动作对照视频与 `g1_review.html` 生成 |
| `joint-limits` | [sop_joint_limits.py](../scripts/sop_joint_limits.py) | G1 通过后几何轴向角色（`bend`/`twist`/`side`、铰链平面 `hinge.axis`/`straight`、下颌单向铰链）与 LBS 蒙皮拉伸安全上限探测＋通用三层几何/拓扑关节协调（`intra_bone_coupling` + `inter_bone_couplings` + `body_part_volumes`）测算（`probe`），以及 Agent 专属定制 `scratch/work/<id>/joint_limits.json`（`apply`/`show`，Schema `monster-joint-limits-v1`） |
| `g2-pack` | [sop_g2_pack.py](../scripts/sop_g2_pack.py) | 阶段三自动校验 G1 硬门禁、加载 `joint_limits.json` 并通过 `RigModel` 世界根与保存起始姿态路径执行 与训练一致的 video2rig 起始姿态采样生成 `pose_001`～`pose_050`、包含表皮关键点的 2D 标签叠加图、单屏 50 缩略图总览 `g2_50_overview.jpg`、双侧边界触达率实测（仅报告）、异常重抽日志 `rejection_redraw_log.json` 与交互式 `g2_review.html`（锁定 `g2_review_sha256`） |
| `g3-exam` | [sop_g3_exam.py](../scripts/sop_g3_exam.py) | 阶段四自动校验 G1+G2 硬门禁、U2 增量持久化同步（比对 SHA-256 跳过未改动 `.blend`）、真实 Blender 烘焙小数关键帧 GLB 与回读重渲染、严格区分 `3d_gt` / `2d_only` / `unlabeled_video` 三类边界（绝不编造 MPJPE/SO(3)）、生成同视角/3/4/轮廓对比视频与 `g3_review.html`（锁定 `g3_review_sha256`） |
| `deliver` | [sop_delivery.py](../scripts/sop_delivery.py) | 阶段五自动校验 G3 硬门禁、3 次重试极线同步失败硬终止阻断、当批独立清单 `batch_manifest.json` 过滤、本地真实 Blender GLB 回读逐帧核对与 `:loop` 实际 Action 小数起止帧首尾接缝检测（`compute_loop_seam_metrics`）及 `delivery_review.html` |
| `colab-guard` | [sop_colab_guard.py](../scripts/sop_colab_guard.py) | 阶段六跨会话租约表 `active_sessions.json`、阶段前置释放校验（仅允许在 `stage1_candidates_fetched` 或 `final_batch_approved` 释放，视频审阅/迭代期间硬阻断释放）与安全下线核验 |
| `formal-gate` | [sop_formal_gate.py](../scripts/sop_formal_gate.py) | 形式化数学门禁（$\mathcal{P}_1$ $\mathrm{SO}(3)$/限角盒/骨长与偏移守恒、$\mathcal{P}_2$ 非中性/随机/边界姿态像素重建与解析雅可比完备性、$\mathcal{P}_3$ 折叠不变量、$\mathcal{P}_4$ 实测网格表面接触收据） |

### 本地预检与真实证据

`preflight` 优先使用可导入 `bpy` 的 Python；本机也会发现已安装的 Blender。可用 `MONSTER_BPY_PYTHON` 或 `MONSTER_BLENDER` 指定路径。默认优先查找独立 `bpy==5.2.3` 环境（Colab `/content/monster-bpy/venv/bin/python`；桌面本地集成目录 `blender-python-5.2.3/bin/python`）。没有可用后端、Worker 报错或产物缺失时必须失败，不生成占位图或有效 U1 输入；历史报告在每次重新执行前自动刷新。

预检保存完整装饰备份、保留分件装配位置的 GLB（自动无损清理孤立点、悬空边、零面积退化面与重复面）、`1.2% H` 双层体素水密壳 `shell.npz`、Marching Cubes 重建 3D 水密外壳 `watertight_shell.glb`、导出重导入检查、几何统计、真实 `1344×768` 原模型正/侧视图 `preflight_front_side.png`、真实 `1344×768` 水密壳正/侧视图 `preflight_watertight_shell.png` 以及并排审阅页 `preflight_review.html`。几何统计在探测副本中合并 glTF UV/法线缝处的重合顶点，不修改源模型；边界、接触面、水密壳包裹与装饰语义仍需实际检查。

自动检查通过后，首次执行返回 `status=awaiting_preflight_review`、`passed=false`（CLI 退出码 2）。**本步骤为 Human-in-the-Loop 用户确认关卡**：Agent 必须先向用户展示 `preflight_front_side.png`（原模型正侧视图）与 `preflight_watertight_shell.png`（水密壳正侧视图）并明确提醒当前预检与水密壳结果 OK，**停下等待用户确认**；严禁在用户看到结果并确认前自行调用 `--review-visual` 或继续执行 U1 上传。待用户明确确认后，传入 `--user-quote '<用户确认原话>'` 登记通过并同步 U1：

```sh
npm run monster:sop -- preflight --monster-id <id> --input <source.glb> --project-root <project>
# 向用户展示 preflight_front_side.png 与 preflight_watertight_shell.png 并等待用户确认后执行：
npm run monster:sop -- preflight --monster-id <id> --project-root <project> --review-visual --review-note '<实际观察结论>' --user-quote '<用户确认原话>' --sync-u1 --drive-root <MonsterStudio目录>
```

`--review-visual` 要求必须提供 `--user-quote`（用户确认原话），只登记已确认的当前证据，不重新运行 Blender；可在该命令添加 `--sync-u1 --drive-root <MonsterStudio目录>` 完成持久同步。U1 校验源模型、输入 GLB、`shell.npz`、`watertight_shell.glb`、原模型检查图、水密壳检查图、装饰清单和完整备份的 SHA-256；任何证据改变或遗失均要求重新预检。

`g1-pack` 同样复用 `_worker_command` 调用本机 Blender 或 `bpy` 环境执行 `--mode prepare_g1_candidates`，要求 4 套真实候选 GLB（`candidate_0..3.glb`）与 `ffmpeg` 均可用：在 Blender 中完成骨架树几何精修（`prepare_candidate`）、分件保持、装饰恢复（`rigid` 绑骨 / `flex` 独立蒙皮子网格）、全网格解剖学规范重命名（`monster-bone-taxonomy-v3`）、未压缩 `.blend` 保存（`compress=False`）、16 张真实 3D 网格＋3D 八面体骨骼叠加图（`BLENDER_WORKBENCH`）、按形态模板驱动的真实变形拉伸指标审计（`follow_through_issues.json`）以及 `front`/`side`/`three_quarter` 三视角 2×2 四候选同步对比视频编码（校验 MP4 `ftyp`、`moov` 与非空 `mdat` 盒子），并写入 `g1_report.json` 记录全部 SHA-256 证据，绝不回退模板火柴人图或 24 字节占位 MP4。

日常可直接通过 GitHub 的 **Open in Colab** 打开 `packages/monster/notebooks/colab.ipynb` 启动云端 Worker，无需手动打包或上传 `.tar.gz`。如需测试本地未提交改动，再使用 `packages/monster/dist/` 下的打包归档。同步本机插件运行目录不会切换正在运行的 Colab 实例；任务忙碌时不可切换代码版本。

持久数据位于 `MyDrive/MonsterStudio/monsters/<monster_id>/`，环境缓存位于 `SkinTokens/` 和 `MonsterStudio/cache/h3_models/`。模型持久传输遵循 U1/U2；只改配置、关节范围、seed、提示词不重传模型，新增考试视频单独持久化。

SkinTokens 的 `uv` 活动缓存使用 VM 本地目录；`scripts/uv_cache_session.py` 在启动时从 Drive 的 `SkinTokens/cache/uv_snapshots/<兼容性键>/` 恢复已核验归档，等待安装进程退出后再回存新增缓存并读回核验。FlashAttention wheel 与模型权重复用既有 Drive 缓存。

## 实际 API 与边界

先检查所部署版本的 `contracts/protocol.json`、Worker 源码与实际 endpoint。当前协议版本为 1，提供 `heartbeat`、`rig_candidates`、`train_monster_probe`、`reexam_only`、`generate_actions_batch`、`shutdown` 共 6 个注册端点。

`integration/monster_client.py` 提供 Python 客户端和 `npm run monster:call`（首次调用自动准备 `packages/monster/.venv`，默认自动发现本机 Google Drive 挂载的 `endpoint_registry.json` 或 `var/monster/endpoint_registry.json`）。认证是 Gradio 用户名/密码；登记文件的 `token` 作为密码使用。登记文件是私密运行数据，不加入 Git 或审阅材料。客户端在真实心跳上核对协议版本。

本机可执行状态查询或零重训快速复考：

```sh
npm run monster:call -- status
npm run monster:call -- reexam --monster_id <id>
```

`rig_candidates` 支持直接从 Google Drive U1 持久目录（`MyDrive/MonsterStudio/monsters/<id>/u1/rig_input_variant.glb`）读取输入并归档全部 4 套 `candidate_0..3.glb` 至 Drive。

## 已固化的代码级保障与运行核查表

执行时仍须核对远端实际运行的 Worker 版本是否已同步最新代码：

| 要求 | 已固化代码保障 | 运行时核查约束 |
| --- | --- | --- |
| 视频反馈期间保留 GPU | `ColabLifecycleGuard` 默认 `idle_timeout_sec=0`（禁用 600s 空闲自毁）；`api_generate_actions_batch` 默认 `auto_shutdown_after=False`；本地 `sop_colab_guard.py` 维护跨会话租约与阶段释放前置条件 | 若远端仍需更新代码，须在空闲时重新运行 Cell 1 + Cell 4，并通过 `heartbeat` 确认远端配置 |
| G1/G2/G3 与骨骼命名硬门禁 | 本地 `sop_gate.py`、`sop_preflight.py`、`sop_g1_pack.py`、`sop_g2_pack.py`、`sop_g3_exam.py`、`sop_delivery.py` 在入口自动执行 `assert_gate` 与 `validate_canonical_rig_bone_names` 硬拦截 | 严禁绕过 `monster_sop.py` 直接调用底层渲染或烘焙函数跳过用户关卡 |
| 四候选完整工程与 G1 材料 | `api_rig_candidates` 自动归档 `candidate_0..3.glb` 至 Drive；`sop_g1_pack.py` 自动完成骨架树精修、装饰恢复、规范重命名、16 张八面体骨骼图、同动作对照视频与 `g1_review.html` | 自动 audit 分数和候选 ZIP 不替代用户 G1 审批 |
| G3 已有视频考试与快速复考 | `api_reexam_only` 已在协议与 Gradio 端点正式注册；`api_train_monster_probe` 覆写前自动备份 `backups/dinov2_head_<ts>.pt`；`sop_g3_exam.py` 严格按三类边界生成报告并锁定 `g3_review_sha256` | 严禁编造无真值视频的 MPJPE / SO(3) 数值 |
| 同步失败硬终止、烘焙分支与当批清单 | `pipeline_runner.py` 在 3 次重试极线同步均失败时抛出 `RuntimeError` 硬终止，禁止将失步视频送入烘焙；ZIP 仅打包当批动作并写入 `batch_manifest.json` | 生产交付时运行 `monster_sop.py deliver` 完成本地 GLB 回读与 `:loop` 首尾接缝核验 |
| 工作台 Monster 调度与入库 | 在 `final_batch` 获用户明确批准前，`sop_preflight.py` 拦截任何写入 `art_source/` 或受管清单的操作 | 最终验收通过后再导入正式项目资源目录 |

## 主干微调与冻结对照

训练默认采用 `partial` 微调模式：默认视觉主干为 `MONSTER_BACKBONE=dinov2-small`（`facebook/dinov2-small` / `dinov2_vits14_reg`，$d=384$，12 层，patch 14；可选切换 `dinov2_vitb14_reg` $d=768$）；前 4 个 block、patch/位置/register embedding 保持冻结（`FROZEN=4`），解冻最后 8 个 block 与输出 LayerNorm。前 1 轮仅热身训练检测头，之后主干学习率 `1e-4`（检测头的 0.1 倍）、检测头学习率 `1e-3`，余弦衰减；梯度裁剪 `1.0`。合成训练使用 `896×896` 方形面板、p95×1.12 统一移动窗口裁剪 `448×448` 输入与 `224×224` 手爪局部裁剪输入，推理默认采用 `MONSTER_DECODE_MODE=parabola` 二次抛物线亚像素峰值解码，并支持三模型独立角色训练（`reader_role="base"` 基础关键点+$K$通道深度 `reader.pt`/`dinov2_head.pt`、`reader_role="crops"` 手爪局部放大器 `reader-crops.pt` 默认关闭 `MONSTER_READER_CROPS=0`、`reader_role="dense"` 按角色总高 $H$ 归一化的密集场检测器 `reader-dense.pt` 默认关闭 `MONSTER_READER_DENSE=0`）。默认 micro-batch `2`、累积 `16` 次，完整累积窗口等效 batch `32`；主干训练 block 使用 activation checkpointing 与融合 SDPA 降低显存开销。

Notebook Cell 1 提供 `TRAINING_MODE`（`partial` / `frozen`）、`UNFREEZE_LAST_BLOCKS`、`BACKBONE_LR`。高级参数通过 `MONSTER_TRAIN_<字段大写>` 环境变量，或训练器 `--training_config '<JSON>'` 配置；字段见 `worker/pose_finetuning.py` 的 `TrainingConfig`。Worker 的训练子进程自动继承这些环境参数，Gradio 协议不变。`frozen` 对照采用相同 seed、姿态划分、增强、micro-batch 与检测头配置；若要比较 4 / 8 个解冻 block，只改变 `unfreeze_last_blocks`。

按 seed=42 固定留出 5% **姿态**，同一姿态全部视图归于同一侧；验证禁用训练增强，不能用 G3 视频选择 checkpoint。保存验证损失最优的有效微调 epoch，同时保存完整主干、检测头、`reader_window`、`temporal_context_contract`、配置、姿态划分与所选 epoch；原权重备份不删除。`*.training.json` 记录完整训练过程。`--init_checkpoint /absolute/path/old.pt` 可从现有同角色、同契约检测器初始化继续训练（核对骨名、起始姿态契约、标定和检测头，避免混用角色权重）。

在已满足同版本 G1/G2 的情况下，仅修改主干训练方式可复用既有合成集，不必重抽 G2。用独立输出路径保存冻结／微调实验，复用同一考试视频、分辨率、检测监督与求解器；新权重必须重新提交 G3，原 G3 不自动迁移。留出损失下降不等于 G3 动画已达标，合掌、交叉肢体、遮挡与手指旋转仍须逐项检查。

先经 SOP 核实 G1/G2 后，在 GPU Worker 同一运行环境执行对照（以下路径为占位示例）：

```sh
python worker/dinov2_pose_trainer.py --mode train --synth_dir /data/synth --meta_path /data/studio_rig_meta.json --ckpt_path /data/experiments/frozen.pt --epochs 15 --training_config '{"mode":"frozen"}'
python worker/dinov2_pose_trainer.py --mode train --synth_dir /data/synth --meta_path /data/studio_rig_meta.json --ckpt_path /data/experiments/partial4.pt --epochs 15 --training_config '{"mode":"partial","unfreeze_last_blocks":4}'
```

本机插件同步不等于云端 Worker 更新。没有活跃实例时只交付代码、Notebook 与部署包，不宣称正式训练完成或 G3 效果提升。

## 输入朝向与 Stage 1 蒙皮架构

预检→U1→Stage 1→G1 统一以用户确认的正视侧视朝向与输入网格为唯一依据，网格清理绝不自动旋转。Stage 1 采用 `SkinTokens + skeleton.analyse + 默认焊接后 Bone Heat (--skinning_mode bone_heat) + 可选启用来自 SkinTokens 的智能蒙皮权重折叠 (--skinning_mode skintokens)` 架构，支持通过 `--skinning_mode` 选择蒙皮算法。
