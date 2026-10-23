---
name: monster-workflow
description: 将静态怪物模型推进到绑定、姿态训练和动作交付，按 MonsterStudio 的 G1/G2/G3 用户审计关卡执行；适用于完整制作、阶段接续和返修。
---

# Monster 工作流

以 [MonsterStudio 完整流程与用户审计规范](references/workflow-and-audit.md) 为制作依据。先读 [当前实现与操作入口](references/runtime.md)，确认正在使用的版本、能力与固化脚本入口。本技能已将全流程 SOP 执行层固化为统一 CLI 调度器 [monster_sop.py](scripts/monster_sop.py) 及 7 个标准子模块（[sop_gate.py](scripts/sop_gate.py)、[sop_preflight.py](scripts/sop_preflight.py)、[sop_g1_pack.py](scripts/sop_g1_pack.py)、[sop_g2_pack.py](scripts/sop_g2_pack.py)、[sop_g3_exam.py](scripts/sop_g3_exam.py)、[sop_delivery.py](scripts/sop_delivery.py)、[sop_colab_guard.py](scripts/sop_colab_guard.py)）；**优先复用现有脚本，尽量减少重复编写 Python / Blender / FFmpeg / JSON 处理代码；现有脚本不能满足需求时，可按需补充或编写临时代码**。本技能不替代普通项目整理，也不因编写文档或检查代码而启动 GPU。

## 定位与恢复

确定游戏项目、`monster_id`、当前输入版本与用户此次范围。使用工作台时先读取对应项目与资源版本；没有工作台项目时沿用用户指定的文件夹，不自动登记其他项目。核对现有产物、阶段记录、用户反馈、云端 endpoint 和原提交记录，从第一个未满足前置条件的阶段继续。结果未知时查询原任务，不重新提交。

读 [审计记录与版本失效规则](references/reviews.md)。通过固化账本脚本初始化或查询当前关卡状态、记录用户明确审批或执行变更分级失效推导：

```sh
# 1. 查看当前怪物审计账本与版本指纹（自动初始化）
npm run monster:sop -- gate --monster-id <id> --project-root . --drive-root "$MONSTER_DRIVE_ROOT" status

# 2. 硬门禁断言（下游执行前自动调用，未获用户明确 approved 或版本哈希失配时直接拦截）
npm run monster:sop -- gate --monster-id <id> --project-root . --drive-root "$MONSTER_DRIVE_ROOT" assert-gate --require G1

# 3. 发生返修或输入变更时，按 SOP 10 类变更类型自动推导失效关卡与需重跑阶段
npm run monster:sop -- gate --monster-id <id> --project-root . --drive-root "$MONSTER_DRIVE_ROOT" request-changes \
  --gate G2 --change-type sampler_script_or_seed_change --quote "<用户原话>"
```

保留 G1/G2/G3 与最终批次验收的明确用户决定及所绑定的版本；现有同版本的有效授权继续使用。源文档、自动评分、Agent 建议和等待超时不能充当用户通过。通用角色技能允许的委托审核，不替代这条 Monster 流程的用户关卡。

## 按阶段执行

只读取和执行本次任务需要的 SOP 章节；不要把整条生产流程套到局部检查任务上。各阶段优先使用 `monster_sop.py`（或 `npm run monster:sop --`）对应子命令：

1. **预检与 U1（SOP §2，含工作区红线守卫与强制骨骼规范命名，调用 [sop_preflight.py](scripts/sop_preflight.py)）**：
   ```sh
   npm run monster:sop -- preflight \
     --monster-id <id> \
     --input <path/to/source.glb> \
     --project-root . \
     --drive-root "$MONSTER_DRIVE_ROOT" \
     --sync-u1
   ```
   保留原工程，在 `scratch/work/` 下从副本准备精简主体输入（`rig_input_variant`）。脚本自动执行**工作区与 Git 卫生红线守卫**（在 `final_batch` 获用户明确确认前，严禁将任何中间产物放入 `art_source/`、禁止在 `art_source/` 下嵌套建立阶段/版本子目录、严禁加入项目 Git / LFS / DVC 或改动 `asset_sources.json` / `source_dependencies.json` / `runtime_manifest.json` 等受管清单）。自动完成 7 项模型体检（姿态朝向、实际几何原点归零、装饰拓扑独立性拆离并备份 `decoration_manifest.json`、网格质量、正侧视遮挡、多套动画冲突、骨骼命名门禁）。**强制骨骼命名预检（必做硬门禁）：** 凡输入模型已带骨架或在生成骨架后进入下游前，**必须在预检中将全部变形骨骼强制规范命名为 `<KIND>_<SIDE>_<GROUP>_<INDEX>` 格式**（10 类 `KIND`：`ROOT`/`RIGID`/`SPINE`/`HEAD`/`JAW`/`LIMB`/`WING`/`FLEX`/`TAIL`/`DIGIT`；`SIDE`：`C`/`L`/`R`），同步更新网格 `vertex_groups`、父子引用、动画 `fcurves` 与 `studio_rig_meta.json`，并通过 `validate_canonical_rig_bone_names` 校验零匿名骨（`valid=True`）；**严禁携带 `bone_0..N`、`Bone.001` 等未规范命名的骨骼跳过预检进入后续流程**。脚本自动产出正/侧视实模检查图、`preflight_report.json` 与 `u1_manifest.json`，并同步至 Google Drive 持久目录（U1），不能用 Gradio 临时上传代替 U1。

2. **四候选、强制骨骼规范命名、装饰恢复与 G1（SOP §3，调用 [sop_g1_pack.py](scripts/sop_g1_pack.py)）**：
   ```sh
   npm run monster:sop -- g1-pack \
     --monster-id <id> \
     --archetype humanoid \
     --project-root . \
     --drive-root "$MONSTER_DRIVE_ROOT"
   ```
   Colab 运行 SkinTokens 生成 4 套候选骨架，保存 `.blend`、4 套 `candidate_0..3.glb` 与元数据至 Drive 并取回本地后，通过 `npm run monster:sop -- colab-guard --monster-id <id> safe-shutdown --reason stage1_candidates_fetched` 释放本任务独占实例。`g1-pack` 脚本在本地将完整装饰恢复至 4 套候选模型（刚性挂件绑骨骼父级，柔性分件配蒙皮权重并保持独立分件，不焊接合并至主体），**强制调用 `canonicalize_rig_bone_names_in_blender` 将全部变形骨骼重命名为 `<KIND>_<SIDE>_<GROUP>_<INDEX>` 规范格式并通过 `validate_canonical_rig_bone_names` 硬门禁校验**。按怪物形态（`winged` 带翼类 / `humanoid` 人形类 / `quadruped` 四足或多足兽 / `aberrant` 异形怪物）在相同时间线、帧率、相位、相机和灯光下输出正/侧/3⁄4 视角同步对照视频（`g1_candidates_sync_*.mp4`），以及每套正/侧/背/3⁄4 的材质预览＋标准八面体（Octahedral）变形骨骼透显叠加图（共 16 张）、骨骼规范命名核验表、随动问题清单与交互式审阅页 `g1_review.html`，并将关卡状态置为 `awaiting_rig_review`。用户明确选定后执行 `npm run monster:sop -- gate --monster-id <id> approve --gate G1 --candidate-id <0..3> --rig-meta <meta_path> --quote "<用户原话>"`。未完成骨骼规范命名核验或未获 G1 通过，严禁运行正式 50 姿势采样。

3. **50 个真实随机姿势与 G2（SOP §4，调用 [sop_g2_pack.py](scripts/sop_g2_pack.py)）**：
   ```sh
   npm run monster:sop -- g2-pack \
     --monster-id <id> \
     --seed 42 \
     --num-poses 50 \
     --project-root . \
     --drive-root "$MONSTER_DRIVE_ROOT"
   ```
   脚本启动前自动断言 G1 已获用户明确通过且 `selected_candidate_id` 与模型指纹完全匹配。仅对 G1 选定的完整绑定模型运行正式合成采样器，**直接按当前锁定的目标视频生成分辨率与视角模式渲染原图**（默认双视图 `[Front | Side(Left)]`：768P 拼合 `1344×768`＝单视图 `672×768`，2K 拼合 `2560×1440`＝单视图 `1280×1440`；若开启 `ENABLE_3VIEW_STUDIO=True` / `MONSTER_ENABLE_3VIEW=1` 三视图 AB 测试特性开关，则采用单画布正视居中三联画 `[Left(+X→-X) | Front(-Y→+Y) | Right(-X→+X)]`：768P 拼合 `1344×768`＝单视图 `448×768 × 3`，2K 拼合 `2688×1440`＝单视图 `896×1440 × 3`），生成 `pose_001`～`pose_050`（含正交多视图原图、`pose_001_overlay.png`～`pose_050_overlay.png` 2D 标签叠加图、单屏 50 缩略图总览 `g2_50_overview.jpg`、关节随机范围与异常重抽日志 `rejection_redraw_log.json` 及 `g2_review.html`）。固定 `seed` 并记录真实抽样与异常重抽对应关系，严禁反复换 seed 挑图或手工摆拍代替。等待用户认可随机策略并执行 `npm run monster:sop -- gate --monster-id <id> approve --gate G2 --seed 42 --quote "<用户原话>"`，不能直接进入大规模合成或训练。

4. **U2、合成训练与 G3（SOP §5，调用 [sop_g3_exam.py](scripts/sop_g3_exam.py)）**：
   ```sh
   npm run monster:sop -- g3-exam \
     --monster-id <id> \
     --project-root . \
     --drive-root "$MONSTER_DRIVE_ROOT" \
     --exam-video <path/to/exam_video.mp4>
   ```
   脚本启动前自动断言 G1 与 G2 均已获用户明确通过且指纹一致，并自动执行 **U2 增量持久化同步**（比对本地与 Drive SHA-256，仅在 `.blend`/GLB 或 `studio_rig_meta.json` 发生实质改动时上传，避免重复上传未改动的模型文件）。**训练图与静止首帧基准图（双视图 `idle_split_1344x768.png` / `idle_split_2560x1440.png`，或三视图 `idle_3view_1344x768.png` / `idle_3view_2688x1440.png`）必须使用与目标视频生成完全一致的原生渲染分辨率与视角排布**。**Blender 摄影棚渲染全链路开启硬件 GPU 加速**（Cycles `OPTIX`/`CUDA`/`METAL` + 禁用 CPU 混合分块 + `scene.cycles.denoising_use_gpu = True` + `use_persistent_data = True`）。按约定合成训练集（基线默认 **5,000 姿态**，采用四模态分层解耦采样器并执行 `audit_pose_dataset_coverage` 校验指间解耦有效秩、反相比例与分位数覆盖率）并冻结 DINOv2 主干训练专属热图头与辅助头，Worker 自动备份旧权重至 `backups/dinov2_head_<timestamp>.pt` 后保存 `dinov2_head.pt`。使用独立于训练集、且具备相同分辨率与相机标定的已有视频考试（或注册端点 `reexam_only` 零重训快速复考），导出实际 GLB 并重新导入 Blender 按原相机同分辨率重渲染。`g3-exam` 脚本严格按三类边界生成报告（传入 `--gt-3d-json` 时为 `3d_gt` 输出 `all_bones`/`primary_bones`/`by_kind`；传入 `--gt-2d-json` 时为 `2d_only` 仅输出 2D 重投影误差；默认无标注视频为 `unlabeled_video` 仅输出跨视图同步、置信度自一致性与骨长守恒，**绝不编造 MPJPE / SO(3) 数值**），产出原视频与烘焙 GLB 重渲染并排对照视频 `g3_comparison_sync.mp4` 及 `g3_review.html`，由用户做出 G3 判定并执行 `npm run monster:sop -- gate --monster-id <id> approve --gate G3 --checkpoint <ckpt_path> --quote "<用户原话>"`。

5. **生产动作与逐片交付（SOP §6，调用 [sop_delivery.py](scripts/sop_delivery.py)）**：
   ```sh
   npm run monster:sop -- deliver \
     --monster-id <id> \
     --actions idle:loop walk:loop run:loop slash:once \
     --project-root . \
     --drive-root "$MONSTER_DRIVE_ROOT"
   ```
   脚本启动前自动断言当前 `dinov2_head.pt` 具备有效 G3 通过记录。提交动作规格（124 帧、24 FPS、8 步 Turbo，默认 `H3_PROFILE="full"`），生成视频分辨率与视角数与训练标定严格一致。抽帧预测双视图骨骼端点并校验竖直同步，默认最多换 seed 重试 3 次；**重试耗尽仍未通过同步检查时，云端 `pipeline_runner.py` 与本地 `sop_delivery.py` 均直接硬终止并阻断失败动作，禁止继续执行后续 IK 烘焙或输出不合格 GLB**。通过三层通用＋Kind 专属运动学重建生成三维轨迹并烘焙导出 `<action_name>_animated.glb`。回存 Drive（打包清单 `batch_manifest.json` 仅列本批产物，绝不混入历史遗留文件），`deliver` 脚本在本地自动核验 GLB 骨架、蒙皮、贴图与 `:loop` 动作首尾接缝跳变（`compute_loop_seam_metrics`），生成逐片验收页 `delivery_review.html` 提交用户验收。原文件和历史结果不覆盖。

到达用户关卡时，先调用对应 `monster_sop.py` 子命令完成本次授权范围内可准备的全部审阅材料，再给出材料、固定版本、具体待决定事项，以及所依据的本技能/SOP 条款。不要要求用户审批尚未制作的材料。只缺其中一个决定时，不重复询问已经明确且仍有效的决定。

## Colab 生命周期

执行云端工作或释放实例前读 SOP §7，并始终通过固化守卫脚本 [sop_colab_guard.py](scripts/sop_colab_guard.py) 管理实例生命周期：

```sh
# 1. 注册或续租当前会话在实例上的占用登记
npm run monster:sop -- colab-guard --monster-id <id> --session-id <session_id> acquire-lease --stage stage5_video_iteration

# 2. 尝试安全释放实例（自动校验阶段前置条件 + 跨会话租约 + 远端忙闲状态）
npm run monster:sop -- colab-guard --monster-id <id> --session-id <session_id> safe-shutdown --reason final_batch_approved
```

- 启动前盘点当前账号现有运行时和各任务 `--config` 登记，优先跨任务、跨会话复用；工作目录与日志隔离，GPU 任务排队执行。
- 四候选已回存 Drive、核验且取回本地后，调用 `colab-guard --monster-id <id> safe-shutdown --reason stage1_candidates_fetched` 释放本任务独占实例；随后保持本地恢复装饰和审阅，不空挂独占 GPU。
- 视频生成、用户查看、等待反馈及调参重生成全过程（如 `stage5_video_iteration`、`awaiting_exam_review`），`colab-guard` 硬阻断任何释放请求，持续保留并复用实例。单片完成、代码检查通过或等待回复期间均不关闭实例，也不安排额外定时关闭。
- 本批清单内全部视频的当前版本获用户明确满意并记录 `final_batch = approved` 后，调用 `colab-guard --monster-id <id> safe-shutdown --reason final_batch_approved` 完成跨会话安全检查与实例释放。
- 每次关闭前 `colab-guard` 自动执行跨会话安全检查：核对目标实例 endpoint、`active_sessions.json` 中其他本地 Codex 会话租约、远端 Worker `busy` 状态与 GPU 队列；发现其他会话使用或排队依赖时自动拒绝释放并记录保留原因。

## 返修与结束

按 SOP §4.2 / §5.2 和审计记录依赖，使用 `npm run monster:sop -- gate --monster-id <id> request-changes --gate <G1|G2|G3|final_batch> --change-type <type> --quote "<用户原话>"` 自动确定分级回退范围（详见 [审计记录与版本失效规则](references/reviews.md)）：仅采样策略问题（`sampler_script_or_seed_change`）改脚本重抽 50 个提交 G2（保留原 G1）；选定模型局部缺陷（`local_model_or_bone_naming_change`）在本地修复后补交受影响的 G1 材料确认并重做 G2；自动骨架结构缺陷（`skeleton_topology_defect`）退回四候选生成 → 恢复装饰 → G1 → G2；输入模型基础缺陷（`u1_input_defect`）退回阶段一预检与 U1 → 四候选 → G1 → G2；G3 失败按视频/标定/IK（`exam_video_or_ik_fix` / `mocap_priors_change`，修后直接调用 `reexam_only` 重考 G3）、姿态泛化不足（`pose_generalization_insufficient`，退回 G2 重采样训练）或骨架缺陷（`skeleton_topology_defect`，退回 G1）分级回退；**仅切换或调参运动学求解先验（`MONSTER_ENABLE_MOCAP_PRIORS`）不改变数据集签名且兼容已有 `dinov2_head.pt`，可直接调用 `npm run monster:call -- reexam --monster_id <id>` 用已有视频重考 G3 而无需重抽数据或重训模型；但若从纯关键点模式新开启稠密表面精修（`dense_surface_enabled_0_to_1`），或升级/更改了目标视频最高分辨率档位或视角模式（`resolution_or_view_mode_change`），则必须按新配置重新合成训练标签并重训 `dinov2_head.pt`、重考 G3**。

返回真实项目与资源版本、产物位置、锁定分辨率、已执行检查、当前关卡及 GPU 保留/释放状态。工作台导入采用不可变资源版本；导入不代表用户审计通过。未执行的 Blender 回读、用户验收、GPU 或引擎验证明确标为未验证。不要因新增技能而宣称后端门禁、Colab 迁移或生产流程已完成。
