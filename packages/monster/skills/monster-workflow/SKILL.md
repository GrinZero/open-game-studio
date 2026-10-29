---
name: monster-workflow
description: 将静态怪物模型推进到绑定、姿态训练和动作交付，按 MonsterStudio 的 G1/G2/G3 用户审计关卡执行；适用于完整制作、阶段接续和返修。
---

# Monster 工作流

以 [MonsterStudio 完整流程与用户审计规范](references/workflow-and-audit.md) 为制作依据。先读 [当前实现与操作入口](references/runtime.md) 与 [v3 算法与运行参数](references/video2rig-v3.md)，确认当前运行配置与固化脚本入口。本技能已将全流程 SOP 执行层固化为统一 CLI 调度器 [monster_sop.py](scripts/monster_sop.py) 及 9 个标准子模块（[sop_gate.py](scripts/sop_gate.py)、[sop_preflight.py](scripts/sop_preflight.py)、[sop_g1_pack.py](scripts/sop_g1_pack.py)、[sop_joint_limits.py](scripts/sop_joint_limits.py)、[sop_g2_pack.py](scripts/sop_g2_pack.py)、[sop_g3_exam.py](scripts/sop_g3_exam.py)、[sop_delivery.py](scripts/sop_delivery.py)、[sop_colab_guard.py](scripts/sop_colab_guard.py)、[sop_formal_gate.py](scripts/sop_formal_gate.py)）；**优先复用现有脚本，尽量减少重复编写 Python / Blender / FFmpeg / JSON 处理代码；现有脚本不能满足需求时，可按需补充或编写临时代码**。本技能不替代普通项目整理，也不因编写文档或检查代码而启动 GPU。

## 完整模型的压缩规范

模型的几何、UV、材质与贴图完成后，默认在插件或全局技能目录加载 `glb-compress/SKILL.md` 及其 `references/compression-policy.md`。先按项目与面数生成压缩建议，再在专用临时目录对照原版与 WebP 1K/2K 的实际画面；Godot 项目始终关闭 Draco。压缩作为交付派生版本，保留未压缩源，不替换 G1/G2/G3 固定输入、训练源或既有审核记录；骨架与动作完成后的贴图压缩还应核对代表姿态。必要细节在 2K 仍受损时保留更高分辨率或原版。

## 每步完成后的下一步提示

每个实际执行的制作步骤或可独立验收的子步骤完成并核验后，立即向用户给出 1–2 句简易提示，再继续后续操作；同一轮连续完成多步也要逐步提示，不能只在阶段结束或最终回复时汇总。提示使用「已完成：<本步结果及可查看的产物/检查依据>。下一步：<紧接着的一项具体动作>。」的形式，用通俗语言说明即可，无需复述整条 SOP。

- 以实际产物和检查结果确认本步完成；任务已提交、仍在运行或结果未知时，报告真实状态，下一步写查询或核验原任务。
- 下一步依据当前角色、固定版本与有效关卡决定，包含返修、恢复和局部任务的实际分支；需要用户操作时，简要说明要查看什么、决定什么。
- 审阅材料就绪与用户审批通过分别提示。**阶段一预检与水密壳确认（Preflight Human-in-the-Loop）**、G1、G2、G3 与最终批次验收均为必须由人看到结果并确认后才能进入下一步的关卡：
  - 例如预检完成后：「已完成：Blender 预检与自动水密壳生成已通过（准备副本清理了 X 个退化面，生成 3D 水密壳 Y 体素，已展示原模型正侧视图 `preflight_front_side.png` 与水密壳正侧视图 `preflight_watertight_shell.png`）。下一步：请查看并确认预检与水密壳效果无误；确认后我将执行 U1 持久上传并生成四套候选骨架。」随后**必须停下等待用户回复**，严禁未获用户确认即自行调用 `--review-visual` 或继续执行 U1 上传与四候选生成。
  - 例如 G1 就绪后：「已完成：G1 审阅包已生成并核验（附审阅页）。下一步：请查看四套候选并选定一套；G1 通过后再进行关节限制探测。」
  - 仅在已获授权范围内且不属于上述用户确认关卡的子步骤，提示后直接继续。
- 检查失败或遇到阻碍时，说明未完成项及紧接着的修复动作，修复并核验后再提示完成。本次请求范围全部完成时，明确「本次范围已完成」；有后续建议则标为可选，没有则说明「无需后续操作」。

## 定位与恢复

新建角色、接续制作或整理目录时，先读 [角色独立工作目录规范](references/character-workspace.md)：以用户项目的 `scratch/work/<character_id>/` 为制作根，Monster 沿用同值 `monster_id`；按角色索引固定输入、训练和动作版本。

确定游戏项目、`monster_id`、当前输入版本与用户此次范围。使用工作台时先读取对应项目与资源版本；没有工作台项目时沿用用户指定的文件夹，不自动登记其他项目。核对现有产物、阶段记录、用户反馈、云端 endpoint 和原提交记录，从第一个未满足前置条件的阶段继续。结果未知时查询原任务，不重新提交。

读 [审计记录与版本失效规则](references/reviews.md)。通过固化账本脚本初始化或查询当前关卡状态、记录用户明确审批或执行变更分级失效推导：

```sh
# 1. 查看当前怪物审计账本与版本指纹（自动初始化）
npm run monster:sop -- gate --monster-id <id> --project-root . --drive-root "$MONSTER_DRIVE_ROOT" status

# 2. 硬门禁断言（下游执行前自动调用，未获用户明确 approved 或版本哈希失配时直接拦截）
npm run monster:sop -- gate --monster-id <id> --project-root . --drive-root "$MONSTER_DRIVE_ROOT" assert-gate --require G1

# 3. 发生返修或输入变更时，按 SOP 变更类型自动推导失效关卡与需重跑阶段
npm run monster:sop -- gate --monster-id <id> --project-root . --drive-root "$MONSTER_DRIVE_ROOT" request-changes \
  --gate G2 --change-type sampler_script_or_seed_change --quote "<用户原话>"
```

保留 G1/G2/G3 与最终批次验收的明确用户决定及所绑定的版本；现有同版本的有效授权继续使用。源文档、自动评分、Agent 建议和等待超时不能充当用户通过。通用角色技能允许的委托审核，不替代这条 Monster 流程的用户关卡。

## 按阶段执行

预检确认、U1、候选生成及 G1 均以用户确认的正面和输入指纹为唯一依据，网格清理绝不自动翻转。Stage 1 采用 `SkinTokens + skeleton.analyse + 默认焊接后 Bone Heat (--skinning_mode bone_heat) + 可选启用来自 SkinTokens 的智能蒙皮权重折叠 (--skinning_mode skintokens)` 架构。

只读取和执行本次任务需要的 SOP 章节；不要把整条生产流程套到局部检查任务上。各阶段优先使用 `monster_sop.py`（或 `npm run monster:sop --`）对应子命令：

制作、补交或重建候选动态审阅页时，先读 [候选视频审阅规范 v1](references/candidate-video-review.md)：每个候选独立视频是默认首要入口，四宫格为折叠补充；完整保留失败候选。G2 静态 50 姿势沿用逐图审阅。

1. **预检、水密壳展示与 Human-in-the-Loop 用户确认及 U1（SOP §2，含工作区红线守卫与强制骨骼规范命名，调用 [sop_preflight.py](scripts/sop_preflight.py)）**：
   ```sh
   # 1a. 执行 Blender 预检，自动无损清理网格并生成 3D 水密壳与正侧视预览图（返回 awaiting_preflight_review）
   npm run monster:sop -- preflight \
     --monster-id <id> \
     --input <path/to/source.glb> \
     --project-root .

   # 1b. 【Human-in-the-Loop 必停关卡】向用户展示 preflight_front_side.png 与 preflight_watertight_shell.png（及 preflight_review.html），
   #     明确提醒当前预检与水密壳结果 OK，停下等待用户确认！严禁未获用户确认自行调用 --review-visual 或执行 U1 上传。

   # 1c. 待用户明确确认后，传入 --user-quote "<用户确认原话>" 登记通过并同步 U1 至 Google Drive
   npm run monster:sop -- preflight \
     --monster-id <id> \
     --project-root . \
     --drive-root "$MONSTER_DRIVE_ROOT" \
     --review-visual \
     --review-note "<姿态/遮挡/装饰/水密壳核验结论>" \
     --user-quote "<用户确认原话>" \
     --sync-u1
   ```
   保留原工程，在 `scratch/work/` 下从副本准备精简主体输入（`rig_input_variant`）。脚本自动执行**工作区与 Git 卫生红线守卫**（在 `final_batch` 获用户明确确认前，严禁将任何中间产物放入 `art_source/`、禁止在 `art_source/` 下嵌套建立阶段/版本子目录、严禁加入项目 Git / LFS / DVC 或改动 `asset_sources.json` / `source_dependencies.json` / `runtime_manifest.json` 等受管清单）。自动完成 7 项模型体检（姿态朝向、实际几何原点归零、装饰拓扑独立性拆离并备份 `decoration_manifest.json`、网格质量自动无损清理与自动水密壳生成、正侧视遮挡、多套动画冲突、骨骼命名门禁）。**自动网格无损清理、自动水密壳（`shell.npz` + `watertight_shell.glb`）与可视化确认：** `sop_preflight_check` 在保存完整源备份 `decorations_backup.blend` 后，自动在准备副本上通过 `_sanitize_preflight_meshes` 清理不影响 UV/外观的孤立点、悬空线边、退化零面积面与重复面（严禁手工编写破坏 UV 的全局焊接或拓扑缝合脚本），自动调用 `export_shell` 生成持久化的 `1.2% H` 双层体素水密壳 `scratch/work/<id>/shell.npz`（含 `thin` 表皮与 `solid` 闭运算 + 6 连通洪水填充实心水密体），并通过 Marching Cubes 重建导出 3D 水密外壳模型 `scratch/work/<id>/watertight_shell.glb`、渲染水密壳正侧视图 `scratch/checks/<id>/preflight_watertight_shell.png` 及并排审阅页 `scratch/checks/<id>/preflight_review.html`。对于含叶片、披风、领口等正常开放边界或 AI 生成的多层接触非流形分件模型，只要自动生成的水密壳有效（`watertight_ready=True`），自动检查即就绪（`automatic_ready=True`、`watertight_shell_fallback_active=True`），下游骨架精修与 G1 蒙皮（`heat_bind_updated_skeleton` → `solid_copy` $100^3$ Marching Cubes 水密外壳 Bone Heat + `cKDTree` 最近表面权重回传）自动接管，无需手工补洞或改网格拓扑。**Human-in-the-Loop 验证（必做）：** 预检执行完毕后，**必须向用户展示原模型正侧视图 `preflight_front_side.png` 与水密壳正侧视图 `preflight_watertight_shell.png`，明确提醒当前预检步骤与水密壳状态 OK，并等待用户看到结果确认后，方可携带 `--user-quote` 执行 `--review-visual --sync-u1` 进入下一步**。**强制骨骼命名预检（必做硬门禁）：** 凡输入模型已带骨架或在生成骨架后进入下游前，**必须在预检中将全部变形骨骼强制规范命名为解剖学语义名**（Schema `monster-bone-taxonomy-v3`：中轴 `Root`、`Hips`、`Spine1..N`、`Neck1..N`、`Head`、`Jaw`、`Tail1..N` 不加 `.C` 后缀；成对肢体使用 `.L` / `.R` 后缀如 `Shoulder.L`、`Arm.L`、`LowerArm.L`、`Hand.L`、`Thigh.R`、`Shin.R`、`Foot.R`、`Toes.R`；指趾 `Finger1_1.L`、`Toe1_1.R`；附属 `Flex1_1.L`、`Ear1_1.L`、`Rigid1`、`Weapon.R`、`Prop_*`；涵盖 `root`/`rigid`/`spine`/`head`/`jaw`/`limb`/`wing`/`flex`/`tail`/`digit` 10 类运动角色），同步更新网格 `vertex_groups`、父子引用、动画 `fcurves` 与 `studio_rig_meta.json`，并通过 `validate_canonical_rig_bone_names` 校验零匿名骨（`valid=True`）；**严禁携带 `bone_0..N`、`Bone.001` 等未规范命名的骨骼跳过预检进入后续流程**。脚本自动产出 `preflight_front_side.png`、`preflight_watertight_shell.png`、`watertight_shell.glb`、`shell.npz`、`preflight_review.html`、`preflight_report.json` 与 `u1_manifest.json`，并在获用户确认后同步至 Google Drive 持久目录（U1），不能用 Gradio 临时上传代替 U1。

2. **四候选生成、骨架树几何精修、强制骨骼规范命名、装饰恢复与 G1（SOP §3，调用 [sop_g1_pack.py](scripts/sop_g1_pack.py)）**：
   ```sh
   npm run monster:sop -- g1-pack \
     --monster-id <id> \
     --archetype humanoid \
     --project-root . \
     --drive-root "$MONSTER_DRIVE_ROOT"
   ```
   Colab 运行 SkinTokens 生成 4 套候选骨架，保存 `.blend`、4 套 `candidate_0..3.glb` 与元数据至 Drive 并取回本地后，通过 `npm run monster:sop -- colab-guard --monster-id <id> safe-shutdown --reason stage1_candidates_fetched` 释放本任务独占实例。`g1-pack` 脚本在本地执行 `prepare_candidate`（`1% H` 体素壳外骨裁剪与尖端二分回缩、默认保留完整 Finger/Toe 指趾树 `MONSTER_FOLD_DIGITS=0`、连续薄末段自动切分刚性 `Weapon` 子骨、成对子树对称化、保留非零起始姿态 `monster-start-pose-v1` 及 $100^3$ 水密外壳 `solid_copy` Bone Heat 蒙皮重绑），将完整装饰恢复至 4 套候选模型（刚性挂件绑骨骼父级，柔性分件配蒙皮权重并保持独立分件，不焊接合并至主体），**强制调用 `canonicalize_rig_bone_names_in_blender` 将全部变形骨骼重命名为解剖学语义名（`Root`、`Hips`、`Spine1`、`Arm.L`、`Thigh.R`、`Finger1_1.L` 等，标记 `reader_contract="monster-reader-v3"`）并通过 `validate_canonical_rig_bone_names` 硬门禁校验**。按怪物形态（`winged` 带翼类 / `humanoid` 人形类 / `quadruped` 四足或多足兽 / `aberrant` 异形怪物）在相同时间线、帧率、相位、相机和灯光下输出正/侧/3⁄4 视角同步对照视频（`g1_candidates_sync_*.mp4`），以及每套正/侧/背/3⁄4 的材质预览＋标准八面体（Octahedral）变形骨骼透显叠加图（共 16 张）、骨骼规范命名核验表、随动问题清单与交互式审阅页 `g1_review.html`，并将关卡状态置为 `awaiting_rig_review`。用户明确选定后执行 `npm run monster:sop -- gate --monster-id <id> approve --gate G1 --candidate-id <0..3> --rig-meta <meta_path> --quote "<用户原话>"`。未完成骨骼规范命名核验或未获 G1 通过，严禁运行正式 50 姿势采样。

3. **G1 通过后关节限制几何/LBS 探测与 Agent 专属定制（SOP §4.0，调用 [sop_joint_limits.py](scripts/sop_joint_limits.py)） + 50 个边界探索随机姿势与 G2（SOP §4.1–§4.2，调用 [sop_g2_pack.py](scripts/sop_g2_pack.py)）**：
   ```sh
   # 3a. G1 通过后，对选定骨架执行几何轴向语义与 LBS 网格安全上限探测，生成 scratch/work/<id>/joint_limits.json
   npm run monster:sop -- joint-limits --monster-id <id> --project-root . --drive-root "$MONSTER_DRIVE_ROOT" probe --archetype humanoid

   # 3b. Agent 结合怪物形态（动作无关的完整生理运动包络 + 全局根轴角 SO(3) 全向旋转/平移）按需微调各骨限制（保持 G1 通过状态）
   npm run monster:sop -- joint-limits --monster-id <id> --project-root . --drive-root "$MONSTER_DRIVE_ROOT" apply \
     --set-bone "Neck1=-35:40,-30:30,-60:60" \
     --notes "按怪兽颈部与胸腔装甲间隙定制动作无关完整运动包络"

   # 3b-verify. 形式化验证硬门禁（joint-limits 与 g2-pack 已自动内嵌执行，也可独立调用校验骨架/轨迹 P1-P4 不变量）
   npm run monster:sop -- formal-gate rig --monster-id <id> --project-root . --drive-root "$MONSTER_DRIVE_ROOT"

   # 3c. 基于 joint_limits.json 运行 video2rig 起始姿态采样生成 50 个小样与 G2 审阅包
   npm run monster:sop -- g2-pack \
     --monster-id <id> \
     --seed 42 \
     --num-poses 50 \
     --project-root . \
     --drive-root "$MONSTER_DRIVE_ROOT"
   ```
   脚本启动前自动断言 G1 已获用户明确通过且 `selected_candidate_id` 与模型指纹完全匹配。先通过 `joint-limits` 产出或定制独立版本化配置 `scratch/work/<id>/joint_limits.json`（Schema `monster-joint-limits-v1`：在保存起始姿态 `p0` 中测定每骨 `bend`/`twist`/`side` 轴角角色、肘膝铰链平面 `hinge.axis`/`straight`、下颌 `Jaw` 单向张嘴范围 `[-0.05, +0.70]` rad 与无观测叶骨非零起始扭转锁定 `frozen_start_channels_rad`；世界根使用轴角 $\mathrm{SO}(3)$ 全向范围支持 360° 转身与 `±75°` 俯仰，定义怪兽动作无关的完整生理安全包络）。随后通过 `RigModel` 世界根与保存起始姿态路径（保留不参与拟合的组织链与固定骨）对 G1 选定的完整绑定模型运行 **video2rig 起始姿态采样器＋全网格硬部件相交/拉伸闸门**（video2rig 起始姿态采样：G2 与训练共用分布；30% 概率将幅度缩至 0.4，非铰链轴围绕保存的 p0 高斯采样，肘膝主轴在合法范围内均匀采样。保留小概率全身转向、俯仰、接触与位移；双侧极限覆盖率仅报告，不作为通过阈值。保留越限、非有限值、意外冻结、固定比例联动与网格安全检查。），按当前锁定的目标视角模式与分辨率由真实 Blender 渲染输出 `pose_001`～`pose_050`（含多视图原图、逐视角无损裁切图 `pose_XXX_<view>.png`、包含表皮关键点的 `pose_001_overlay.png`～`pose_050_overlay.png` 2D 标签叠加图、单屏 50 缩略图总览 `g2_50_overview.jpg`、`joint_limits.json` 范围总表、逐样本 `boundary_joints` 边界触达标注、异常重抽日志 `rejection_redraw_log.json`，以及基于专用审阅模板 [assets/g2_review.html](assets/g2_review.html) 构建的交互式审阅工作台 `g2_review.html`，锁定总签名 `g2_review_sha256`）。固定 `seed` 并记录真实抽样与异常重抽对应关系，严禁反复换 seed 挑图或手工摆拍代替。等待用户认可姿势分布与变形质量并执行 `npm run monster:sop -- gate --monster-id <id> approve --gate G2 --seed 42 --quote "<用户原话>"`，不能直接进入大规模合成或训练。

4. **U2、标定合成训练与 G3（SOP §5，调用 [sop_g3_exam.py](scripts/sop_g3_exam.py)）**：
   ```sh
   npm run monster:sop -- g3-exam \
     --monster-id <id> \
     --project-root . \
     --drive-root "$MONSTER_DRIVE_ROOT" \
     --exam-video <path/to/exam_video.mp4>
   ```
   脚本启动前自动断言 G1 与 G2 均已获用户明确通过且指纹一致，并自动执行 **U2 增量持久化同步**（比对本地与 Drive SHA-256，仅在 `.blend`/GLB 或 `studio_rig_meta.json` 发生实质改动时上传）。**静止首帧基准图（双视图 `idle_split_1344x768.png` / `idle_split_2560x1440.png`，或三视图 `idle_3view_1344x768.png` / `idle_3view_2688x1440.png`）与目标视频生成严格保持一致的原生分辨率与视角排布；合成训练集使用独立的 `896×896`/视角方形面板与 `training_calibration`**。**Blender 摄影棚默认 EEVEE 8 spp GPU 渲染，Stage 2 正交视图采用最多 16 副本动画条带批量渲染（按最大宽度拆批，896px 面板每批 12 副本，前一半清晰、后一半 0.5 快门运动模糊）**，配置见 [渲染入口](references/video2rig-v3.md#eevee-动画条带渲染)。按标定采样契约（`monster-calibrated-sampling-v2`，默认 **5,000 姿态**，包含表皮关键点 `.tip`/`.sidea1..2`/`.sideb1..2`/`.held_tip`/`.held_mid`、`Prop_*` `.end1/.end2/.mid`、BVH + CPU Z-buffer 混合可见性及 `labels.npz`/`visible.npz`/`poses.npz`）合成数据，通过 `ReaderDataset` 统计整批训练图关键点跨度 p95×1.12 统一窗口边长（`monster-reader-training-p95-window-v1`）同步裁剪至 `448×448` Reader 输入，并按 [主干微调配置](references/runtime.md#主干微调与冻结对照) 训练 `Video2RigReaderNet`（默认 `MONSTER_BACKBONE=dinov2-small` $d=384$，冻结 embedding 与前 4 层，热身检测头 1 轮后解冻后 8 层及 LayerNorm；接入 `t-1/t+1` 双向上下文防抄袭损坏 `corrupt_context` 与 13 副本挥击残影 `smear_limbs_capsules_gpu`；按姿态留出 5% 验证集保存验证最优权重）。使用独立于训练集、具备相同分辨率与相机标定的已有视频运行 `infer_v3_frames`（首帧阴影剥离与轮廓对齐 → 粗读定位窗口中心 + 448 窗口 8x TTA 抛物线亚像素峰值解码 → 左右消歧与读数预处理 → 标定多视角极线同步检查 → 短遮挡桥接 → 解析雅可比混合 Cauchy 2D 重投影求解 `fit_clip_2d_reprojection` → 8 步 `clean_motion` 后处理 + `surface_contact.py` 实测表面接触守卫 + `at_limit` 卡限诊断），由真实 Blender 按 `monster-video-timing-v1` 小数帧率烘焙导出 GLB、重新导入核对世界矩阵并渲染同视角、`50°/18°` 3/4 视角与轮廓叠加对比视频 `g3_comparison_sync.mp4`。`g3-exam` 脚本严格按 `3d_gt` / `2d_only` / `unlabeled_video` 三类边界生成报告（**绝不编造 MPJPE / SO(3) 数值**），锁定 `g3_review_sha256` 签名并产出 `g3_review.html`，由用户做出 G3 判定并执行 `npm run monster:sop -- gate --monster-id <id> approve --gate G3 --checkpoint <ckpt_path> --quote "<用户原话>"`。

5. **生产动作与逐片交付（SOP §6，调用 [sop_delivery.py](scripts/sop_delivery.py)）**：
   ```sh
   npm run monster:sop -- deliver \
     --monster-id <id> \
     --actions idle:loop walk:loop run:loop slash:once \
     --project-root . \
     --drive-root "$MONSTER_DRIVE_ROOT"
   ```
   脚本启动前自动断言当前 checkpoint 具备有效 G3 通过记录。提交动作规格（124 帧、24 FPS、8 步 Turbo，默认 `H3_PROFILE="full"`，支持 `MONSTER_SLOW_MOTION_SCALE=0.5` 半速生成与 `MONSTER_TIME_SCALE=2` 倍速恢复），生成视频分辨率与视角数与训练标定严格一致。抽帧运行 Reader 并在短遮挡桥接前校验标定多视角极线同步，默认最多换 seed 重试 3 次；**重试耗尽仍未通过同步检查时，云端 `pipeline_runner.py` 与本地 `sop_delivery.py` 均直接硬终止并阻断失败动作，禁止继续执行后续求解烘焙或输出不合格 GLB**。通过 2D 重投影求解、8 步 `clean_motion` 与实测表面接触守卫重建三维轨迹（保存 `motion_structure`）并由真实 Blender 烘焙导出 `<action_name>_animated.glb`。回存 Drive（打包清单 `batch_manifest.json` 仅列本批产物，绝不混入历史遗留文件），`deliver` 脚本在本地自动回读 GLB 逐帧核对骨骼、世界旋转、蒙皮、贴图、轨迹形式化不变量（调用 [sop_formal_gate.py](scripts/sop_formal_gate.py) 校验 $\mathcal{P}_1$ $\mathrm{SO}(3)$/限角盒/骨长与偏移守恒及 $\mathcal{P}_4$ 表面接地质量）与 `:loop` 动作实际 Action 小数起止帧首尾接缝跳变（`compute_loop_seam_metrics`），生成逐片验收页 `delivery_review.html` 提交用户验收。原文件和历史结果不覆盖。

到达用户关卡时，先调用对应 `monster_sop.py` 子命令完成本次授权范围内可准备的全部审阅材料，再给出材料、固定版本、具体待决定事项，以及所依据的本技能/SOP 条款。不要要求用户审批尚未制作的材料。只缺其中一个决定时，不重复询问已经明确且仍有效的决定。

## 云端训练持久化（必做）

使用 Colab 或其他临时云端运行时训练前，必须挂载并核验 Google Drive、持久卷或对象存储等跨实例保留的存储，按角色和训练版本固定持久目录，并确认可写、剩余容量足够及写入后可读回。持久存储未就绪时，先完成配置与核验再启动训练。

- 将预训练模型权重与下载缓存、训练检查点（最新与验证最优）、恢复所需的优化器/学习率调度器/混合精度状态、epoch/step 与随机数状态，以及训练配置、代码版本、数据集清单/签名、骨架元数据和日志保存到持久目录。仅有推理权重不能视为可续训检查点；记录实际支持的恢复状态与缺失项。
- 云端本地磁盘可用作加速缓存，但关键文件必须持续回存持久存储：每个 epoch 或预先记录的固定 step 间隔保存最新检查点，验证最优时另存最优版本；阶段完成、正常退出或主动释放实例前再次同步。采用临时文件写入后原子替换或带校验的版本化上传，保留上一份有效检查点，核验文件大小与校验和后再登记成功。
- 启动或重连时先检查持久缓存与已有检查点，核对角色、训练版本、配置和数据集签名后复用缓存或恢复训练；签名改变时创建新版本并说明原因。记录持久路径、最近成功同步时间、checkpoint 对应 epoch/step 与恢复命令；在新进程中验证检查点可加载、恢复状态可读。缓存、检查点或恢复信息仅存在于 `/content`、`/tmp`、容器磁盘或临时下载链接时，不能报告持久化完成。

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

按 SOP §4.2 / §5.3 和审计记录依赖，使用 `npm run monster:sop -- gate --monster-id <id> request-changes --gate <G1|G2|G3|final_batch> --change-type <type> --quote "<用户原话>"` 自动确定分级回退范围（详见 [审计记录与版本失效规则](references/reviews.md)）：
- 仅关节限制或采样策略问题（`sampler_script_or_seed_change`）：更新 `joint_limits.json` 或采样参数，重抽 50 个姿势提交 G2（保留原 G1）。
- 选定模型局部缺陷（`local_model_or_bone_naming_change`）：在本地修复网格、骨骼命名/轴向、蒙皮权重或装饰绑定后，补交受影响的 G1 材料确认并重做 G2。
- 自动骨架结构缺陷（`skeleton_topology_defect`）：退回四候选生成 → 骨架精修与恢复装饰 → G1 → G2。
- 输入模型基础缺陷（`u1_input_defect`）：退回阶段一预检与 U1 → 四候选 → G1 → G2。
- G3 失败：按考试视频/标定/重投影求解与后处理参数问题（`exam_video_or_ik_fix`，修后直接调用 `reexam_only` 用已有视频重考 G3 而无需重抽数据或重训模型）、姿态泛化不足（`pose_generalization_insufficient`，退回 G2 重采样训练）或骨架缺陷（`skeleton_topology_defect`，退回 G1）分级回退；若更改了目标视频分辨率档位、视角排布或相机焦距（`resolution_or_view_mode_change`），或变更了标签契约/开启密集表面探针，则必须按新配置重新合成训练标签并训练 Reader、重考 G3。

返回真实项目与资源版本、产物位置、锁定分辨率、已执行检查、当前关卡及 GPU 保留/释放状态。工作台导入采用不可变资源版本；导入不代表用户审计通过。未执行的 Blender 回读、用户验收、GPU 或引擎验证明确标为未验证。
