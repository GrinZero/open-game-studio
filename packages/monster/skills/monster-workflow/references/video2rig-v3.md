# v3 Reader 算法与运行参数

本文档定义当前标准管线（`monster-bone-taxonomy-v3` 骨架契约、`monster-reader-v3` 探针契约、`v3-anatomical-reader-4` 特征契约）的运行参数、骨架与表皮点构造、EEVEE 条带渲染、训练增强、2D 重投影求解器及 `clean_motion` 后处理规范。

---

## 1. 核心运行环境变量

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MONSTER_CAMERA_LENS` | `0` | `0` 为正交相机；正数为透视相机毫米焦距（如 `50`，Blender CLI 同受 `--lens 50` 控制）。`Camera.scale` 统一表示垂直画幅世界高度 |
| `MONSTER_STUDIO_LAYOUT` | `horizontal` | `horizontal` 为横向多面板排布；`stacked` 为侧视在上、俯视在下（共享横向像素方向，与三视图互斥） |
| `MONSTER_ENABLE_3VIEW` | `0` | `0` 为双视图 `[Front \| Side]`；`1` 为单画布正视居中三联画 `[Left \| Front \| Right]` |
| `MONSTER_ENABLE_2K` | `0` | `0` 为 768P（`1344×768`）；`1` 为 2K 1440P（双视图 `2560×1440`，三视图 `2688×1440`） |
| `MONSTER_AUTO_FACING` | 兼容变量 | 不再授权清理阶段自动旋转。正面由预检明确调整并经用户确认锁定，几何清理绝不改动朝向；几何估计仅为诊断建议 |
| `MONSTER_SKINNING_MODE` | `bone_heat` | Stage 1 蒙皮算法：默认 `bone_heat`（焊接后 Bone Heat 蒙皮），可选 `skintokens`（SkinTokens 智能蒙皮权重折叠） |
| `MONSTER_FOLD_DIGITS` | `0` | 默认保留完整手指/脚趾树（`Finger*`/`Toe*`）；显式设为 `1` 时按分支最大弧长 `< 0.22H` 折叠短指趾并合并权重 |
| `MONSTER_VISIBILITY_MODE` | `hybrid` | 同时计算并保存 BVH 射线 `visibility_ratio` 与 CPU Z-buffer 布尔可见性（可选 `zbuffer` \| `bvh`） |
| `MONSTER_EXPORT_NPZ_LABELS` | `1` | 在 `keypoints_gt.json` 旁同步导出 `labels.npz`、`visible.npz`、`poses.npz` |
| `MONSTER_BACKBONE` | `dinov2-small` | 视觉主干：默认 `dinov2-small`（`facebook/dinov2-small` / `dinov2_vits14_reg`，$d=384$，12 层，patch 14）；可选 `dinov2_vitb14_reg`（$d=768$） |
| `MONSTER_DECODE_MODE` | `parabola` | 推理热图解码器：默认 `parabola` 三点二次抛物线亚像素峰值解码 + 实际 sigmoid 峰值置信度（GPU 训练损失保持可微 DSNT） |
| `MONSTER_READER_CROPS` | `0` | 可选开启独立手爪局部放大器 `reader-crops.pt`（`224×224` 输入，局部裁剪置信度 `cq > c` 时替换手爪/足端关键点） |
| `MONSTER_READER_DENSE` | `0` | 可选开启独立密集表面场检测器 `reader-dense.pt`（骨骼局部坐标按角色总高 $H$ 归一化） |
| `MONSTER_NO_CUT_TRACKS` | `1` | 开启 Viterbi 高置信平滑轨迹防误切否决 |
| `MONSTER_TWOWAY_JUMP` | `1` | 开启前后双向一致常速跳变过滤 |
| `MONSTER_PANEL_DRIFT` | `1` | 开启双正交共享轴 `> 5% H` 中线漂移校正 |
| `MONSTER_CLEAN_MOTION` | `1` | 启用完整 8 步 `clean_motion` 运动后处理管线 |
| `MONSTER_SURFACE_CONTACT` | `1` | 启用 `surface_contact.py` 实测网格鞋底接触与防穿模守卫 |
| `MONSTER_VIDEO_FPS` | `24` | 视频采样时间基准（支持小数帧率，Blender 以 `fps/fps_base` 表示） |
| `MONSTER_SLOW_MOTION_SCALE` | `1` | 设为 `0.5` 时提示生成半速动作，并默认以 `MONSTER_TIME_SCALE=2` 倍速映射烘焙（显式 `MONSTER_TIME_SCALE` 优先） |

---

## 2. 骨架树几何精修、关节轴向与表皮关键点

### 2.1 骨架树精修与权重重绑（`prepare_candidate`）

- **Stage 1 蒙皮架构**：整体候选生成与绑定采用 `SkinTokens + skeleton.analyse + 默认焊接后 Bone Heat (--skinning_mode bone_heat) + 可选启用来自 SkinTokens 的智能蒙皮权重折叠 (--skinning_mode skintokens)` 架构。
- **双层体素壳与壳外骨修剪**：几何精修使用 `1% H` 体素，预检与阶段二均独立导出 `1.2% H` 双层体素水密壳 `shell.npz`（并校验 `watertight_ready=True`）。壳外比例通过随长度/体素调整的偶数中点采样测量，严格超过 50% 才删除；尖端回缩采用二分定位，最多缩至原长 30%。
- **指趾保留与刚性武器切分**：默认 `fold_digits=False` 保留完整指趾树。若显式折叠，则按每条 digit 分支最大弧长 `< 0.22H` 判定，删除表记录各骨最终祖先并将非 digit 附件重挂到保留掌骨，完整短指扇折叠后掌骨延至各远端均值，Blender 自动合并被删组权重并归一化。连续薄末段长度严格 `> 0.10H`、厚度不超过 `min(1.5 体素, 0.5 * 近端掌半径)` 时，在薄段开始后 `0.03H` 保留拳头并切分刚性 `Weapon` 子骨。
- **脚底定位与成对树对称修整**：脚底按最近脚段分配低处网格点，以朝前投影 97% 分位尾部的均值定位脚尖。成对子树在体积 IoU `>= 0.80` 时镜像平均；其余在链长比 `>= 0.75` 且对应骨段不过度悬殊时等长，递归平移后代以保持关节连接。
- **非零起始姿态与水密外壳 Bone Heat 兜底**：重绑时保留非零起始姿态（`monster-start-pose-v1`，保存完整 `bones`、`matrix_local`、`pose_basis`、rest/start 网格）及所有网格实际表面；首帧校准使用 start 网格。`heat_bind_updated_skeleton` 在整体或任意 `>= 2%` 角色顶点的主分件覆盖率 `< 50%` 时自动回退到 $100^3$ Marching Cubes 水密外壳 `solid_copy` Bone Heat + `cKDTree` 最近表面权重回传；`< 2%` 角色顶点的独立小碎片（不含独立 `Prop`）按最近骨段 $1/d^4$ 分配至多四影响，并清理远侧串权重。

### 2.2 几何关节轴与非零起始姿态（`axis_roles`）

- **轴角色与铰链平面**：在恢复后的起始姿态 `p0` 中测量每骨 `rx/ry/rz` 的 `bend`/`twist`/`side` 角色；`flatness` 使用四个局部体积截面测定。铰链关节使用静息父子骨弯曲平面选择局部 X 或 Z 轴（近乎笔直时使用起始姿态轴角色），明确记录 `hinge.axis` 与 `axis_name`；固定其余 Euler 分量后，用正弦/余弦解析求与父骨最接近的 `straight`。
- **有向铰链范围与 LBS 二分探测**：起始弯曲严格超过 `12°` 时沿实际继续折叠方向确定正反铰链；其余按角色正面（膝向后、肘向前）。范围为有向 `straight - 0.05` 到 `straight + 2.7` rad 与用户显式范围及 LBS 安全范围的交集；首次 `15°` 失败时在 `0°` 与失败角之间做 8 次二分细分。
- **下颌单向铰链与叶骨起始扭转锁定**：`Jaw` 使用实体上下唇之间前方开放的空腔定位，由 `configure_jaw_hinge` 配置向下张嘴单向铰链范围 `[-0.05, +0.70]` rad（其余轴 `[-0.08, 0.08]`）。无观察依据的叶骨扭转若在保存的 `p0` 中非零，通过 `frozen_start_channels_rad` 锁定在该角度，配置合并、重探测与采样均保持该锁定值。

### 2.3 表皮关键点与自定义相机（`v3-anatomical-reader-4`）

- **表皮关键点槽位与镜像定向（`surface_keypoints`）**：通过 `skin.bone_names` 映射实际蒙皮索引（胶囊半径使用相同映射），叶骨依据当前父子关系动态判定。全部原始头/尾 `2N` 槽位排列在前，附加表皮点跟在其后：
  - 对严格 `w > 0.3` 且至少具 4 个支持顶点的骨段，生成 98% 轴向 `.tip` 以及由 `(outward, facing, up)` 确定性定向横截面特征向量提取的 1%/99% 主次轴侧点 `.sidea1/2`、`.sideb1/2`（保证 `.L`/`.R` 成对骨骼解剖侧严格镜像一致，用于约束手掌/叶骨自身轴向 roll 与父骨扭转）。
  - 手部蒙皮延伸 `> 2.5 L_bone` 时自动生成 `.held_tip` 与 `.held_mid` 胶囊/地面守卫点。
  - 独立道具 `Prop_*` 生成 `.end1`、`.end2`、`.mid` 及用于完整旋转观测的离轴点（训练期支持左到右排序，求解期支持按观测动态交换 `end1/end2`）。
- **相机导出与标定（`studio_camera.Camera`）**：支持 `studio_bpy_worker.py --mode export_cameras --blend_path <scene.blend> --out_report <cameras.json>` 导出外参、投影、焦距和镜头偏移，支持非方形横/竖画幅与 `AUTO`/`HORIZONTAL`/`VERTICAL` sensor fit（`Camera.scale` 统一表示垂直画幅世界高度）。

---

## EEVEE 动画条带渲染

渲染环境采用独立 Python 3.13 / NumPy 2 / `bpy==5.2.3`（Colab `/content/monster-bpy/venv`；桌面可通过 `MONSTER_BLENDER_DEPS` 指定同 ABI 依赖目录，直接依赖固定于 `worker/blender-requirements.txt` 与 `scripts/bootstrap-blender.sh`），与 SkinTokens 内部的 Python 3.11 / `bpy==4.5.15` 隔离。

摄影棚默认 EEVEE 8 spp，Stage 2 正交视图默认启用批量条带渲染（`MONSTER_RENDER_STRIP=1`）：每个副本保留独立骨架、Action、变形修改器、道具和灯光/阴影隔离快照，沿相机右方向排列，奇偶副本沿视线深度错开 2.5m 并保留半面板间隔，输出仍为每姿态一张拼图及原坐标标签。清晰样本（前半批）与 0.5 快门运动模糊样本（后半批，中心帧为帧 1）分批渲染，渲染收据记入 `keypoints_gt.json.render_pipeline`（含实际批量、批次数、渲染调用和耗时）。

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| `MONSTER_RENDER_ENGINE` | `EEVEE` | 可选 `CYCLES`（启用硬件 GPU 光追 + `denoising_use_gpu=True` + `use_persistent_data=True`）；实际引擎记入标定 |
| `MONSTER_EEVEE_SAMPLES` | `8` | EEVEE 渲染采样数 |
| `MONSTER_RENDER_STRIP` | `1` | `1` 启用正交条带批量渲染；`0` 退回逐样本渲染（透视视图与 G2 边界审阅默认逐样本渲染） |
| `MONSTER_STRIP_COPIES` | `16` | 目标最大批量，范围 1–16 |
| `MONSTER_STRIP_MAX_WIDTH` | `16384` | 含间隔的最大条带像素宽度；896px 面板含半面板间隔时自动按每批 12 份拆批（5,000 姿态共 418 批、双视图 836 次渲染） |

---

## 3. 标定采样、移动窗口与三模型训练

### 3.1 标定合成采样（`monster-calibrated-sampling-v2`）

- G2 与训练使用 `video2rig-start-pose-v1`：围绕保存起始姿势采样，30% 概率缩幅到 0.4；非铰链轴高斯采样，铰链主轴均匀采样。范围覆盖率仅作诊断。详细公式与安全检查见 [SOP §4.1](workflow-and-audit.md#41-与-video2rig-一致的起始姿态采样)。
- 合成训练使用独立的 `896×896`/视角方形面板与 `training_calibration`（视频首帧与推理保持锁定的原生分辨率）。
- 接触目标来自实际蒙皮后的躯干三角面；画幅采样使用所有标定相机 `[3%, 97%]` 的共同可行区域。
- 按帧抽取 360° 全向 Yaw、`±75°` Pitch、漫游、跳跃、持握、张嘴、STRIKE、LUNGE 事件，在每个样本的 `v3_sampling` 中记录目标、误差和参数，并由同一骨架微扰姿态投影生成前序上下文（`monster-context-pose-v1`）。

### 3.2 p95 移动窗口裁剪（`monster-reader-training-p95-window-v1`）

- `ReaderDataset` 统计整批训练图片中画幅内关键点（每图至少 3 个点）横/纵跨度的 p95 再乘 `1.12`，四舍五入并限制在面板短边内，作为整批统一的窗口边长；窗口中心跟随各图关键点，将 `896×896` RGB、2D 关键点、时序上下文与稠密场同步变换到 `448×448`（完整表皮包围盒仍保存在标签中供诊断）。
- 训练 checkpoint 保存 `reader_window`（含统一边长和训练面板尺寸）。推理时按 `实际面板高度 / 训练面板高度` 换算同一边长：全序列粗读仅定位窗口中心，失读中心沿时间插值后平滑，严禁将逐帧点范围抖动变成窗口缩放。

### 3.3 `Video2RigReaderNet`、三探针角色与防抄袭/残影增强

1. **网络结构与微调**：默认 `MONSTER_BACKBONE=dinov2-small`（$d=384$，`FROZEN=4` 冻结 `embeddings` 与前 4 层 block，热身检测头 1 轮后以主干 `1e-4`、检测头 `1e-3` 微调后 8 层 block + 末端 `norm`），配备 `STRIDE=4`、`HEAT=112`、`CH=96` 轻量双路融合解码器。
2. **三模型探针角色**：
   - **基础检测器 `reader.pt` / `dinov2_head.pt`（`reader_role="base"`）**：关键点热图 + $K$ 通道逐关键点深度头 `dhead` `(B, K, H, W)`，通过 `sample_maps(dmap, coords)` 双线性采样 `point_depth` `(B, K)`，由 `sight_depth_targets` 按 `(X - mid_body) @ cam_fwd / H` 对画幅内全部关键点施加 SmoothL1 监督。
   - **手爪局部放大器 `reader-crops.pt`（`reader_role="crops"`）**：`end_groups` 提取侧向非松散肢体链末段及指/趾后代，按最末两节骨长之和 `1.6 * reach` 夹紧 `[72, 320]` px 裁剪缩放至 `224×224`；推理期 `read_crops` 仅在局部裁剪置信度 `cq > c` 时替换肢端关键点。
   - **密集表面场检测器 `reader-dense.pt`（`reader_role="dense"`）**：`rasterize_dense_fields` 默认 `local_norm="height"` 将骨骼局部坐标 `(pts - head) @ R` 按角色总高 $H$ 归一化；损失权重采用背景 `1.0`、前景骨骼 `3.0`、`part_weight=5.0`、`local_weight=20.0`。
3. **双向上下文防抄袭增强（`corrupt_context`）与挥击残影（`smear_limbs_capsules_gpu`）**：
   - 采用 `t-1 / t+1` 双向邻帧上下文（首尾帧夹紧，`temporal_context_contract="monster-reader-bidirectional-context-v2"`），训练期对上下文施加 `1.5 px` 高斯抖动、5% 远距离离群点（`±0.2 * SIZE`）、16% 单点随机丢弃、25% 整条对称肢体左右互换（`twins`）、`U(0.3, 1.0)` 置信度随机缩放与 20% 全帧上下文清空。
   - 以 35% 概率在训练图抽取 1～2 条可见肢体线段，沿法线方向（`atan2 + pi/2 ± 0.6 rad`）用 `torch.roll` 平移 13 份软边缘 2D 胶囊体副本做预乘 Alpha 混合。

---

## 4. 推理、重投影求解、后处理与质检

### 4.1 全序列阅读与左右消歧预处理（`preprocess_readings`）

1. **首帧对齐与时间基准**：首帧剥离地面阴影并在完整三角形集合上对齐相机轮廓（输出优化前后 IoU 与双向 Chamfer；透视优化 6 外参，正交投影保持输入标定并报告 `unobservable_parameters=["tz"]`）。前景均值 DIS 光流仅在周期性抽帧证据成立时调整内容节奏，映射为 `monster-video-timing-v1`（含 `fps`、`slow_motion_scale`、`time_scale` 与已缩放的 scene-frame 时间）。
2. **8x TTA 抛物线解码**：在 `448×448` 移动窗口上运行 8x 逆变换平均热图，通过 `parabola` 三点二次抛物线亚像素峰值解码提取坐标，并使用实际 sigmoid 峰值作为置信度。
3. **几何消歧与降权过滤**：按面板顺序保留神经网络 `view_id`，在几何消歧阶段将主视图（`front` 或 `top`）置为主观测，依次执行 `fix_twins_viterbi`（首帧交换惩罚 + `no_cut_tracks`）、`jump_pass_constant_velocity`（常速插值检验 + 父骨连接点距离保护 + `twoway`）、`panel_drift`（正交共享轴 `> 5% H` 中线漂移校正）、`downweight_overlapping_twins`（`twin_same` 并拢站立保护）、`downweight_flipped_limbs`（`flipped` 24 帧内短暂翻转 `×0.1` 降权）与 `downweight_contradicting_far_limb`（`pick_panels` 3 帧窗口扩展 + 远侧肢根相机深度支持 + `zero_contradicted` 跨视角矛盾清零）。

### 4.2 解析雅可比混合 Cauchy 2D 重投影求解（`fit_clip_2d_reprojection`）

- **极线同步门禁**：在短遮挡桥接前直接测量真实读数的跨视角重投影一致性（`geometric_consistency`）。
- **混合 Cauchy 核与约束项**：
  - 对 2D 重投影残差与单侧遮挡视线深度残差（`w_depth=0.2`、`depth_where="unseen"`）启用 `robust_mode="scipy_hybrid"`（`robust_cauchy_loss` + `x_scale="jac"`；前后三帧置信度 `>= 0.6` 且二阶像素差分 `< 2%` 投影身高的稳定轨迹，其 Cauchy 尺度从 6px 放宽至 12px）。
  - 着地脚掌使用骨骼局部横向单位向量施加 `flat_feet=120`（腾空和竖直蹄不施加）；接触集合在每帧求解前固定，优先采用当前可用像素读数，失读时用前帧深度。
  - 胶囊体与地面防穿模约束按 `3000/H` 处理，中心线上沿可运动的确定性方向逃逸。
  - 角色特定静息先验 `w_prior=2.0`（`RigModel.prior_weights`：`twist=25.0`、铰链非弯曲轴 `60.0`、`spine=15.0`、`clavicle=40.0`、`toe=30.0`、`loose=6.0`、主弯曲轴 `2.0`、根旋转单视角 `2.0`/多视角 `0.2`、根平移 `0.0`）。`reset` 仅恢复局部关节起始值，保留当前世界根位移与转向。
  - 诊断记录每个起点代价、逃逸前像素误差、稳定读数数量及当前着地脚掌。

### 4.3 8 步 `clean_motion` 后处理、表面接触守卫与快速/正式质检

- **8 步 `clean_motion`**：按映射后的真实秒数依序执行 `smooth_turn` → `despike_reading_backed`（仅修改有毛刺角通道与 `> 0.03H` 根位移毛刺，高置信投影验证，不平均正常根位移）→ `lowpass_motion_adaptive`（4/5/8Hz 自适应 Butterworth 低通）→ `floor_bend_and_smooth_lift`（静止着地段与 60 遍高斯单调抬升包络）→ `recover_root_translation_from_steps`（原地支撑、55% 移动份额、0.92 方向一致性、`0.08H/s` 速度、`steps_steady=0.5` 步幅离散度门控，2Hz 低通积分）→ `lock_feet`（双端点钉死 IK、1 帧间隙闭合与 `lock_blend=3` 淡入淡出）→ `simulate_damped_spring_chains`（每帧 4 个约束子步、4Hz/0.55 阻尼解析状态转移、`spring_soft=0.4` 根到尖频率渐减，保持骨长并反解局部限角）→ `keep_out`（终态胶囊防穿模推出）。
- **表面接触守卫与形式化门禁**：`surface_contact.py` 保留冻结局部旋转与父系偏移，仅调整实际拟合通道与授权根高度并输出接触收据；推理轨迹保存 `motion_structure`（`motion_roots`/`fitted_bones`），由 `sop_formal_gate.py` 使用实际 FK/像素求解器校验 $\mathcal{P}_1$–$\mathcal{P}_4$（中性与非中性/随机/边界姿态像素重建、解析雅可比、失读保持、$\mathrm{SO}(3)$/限角盒/骨长守恒与表皮接地收据）。
- **CPU/CUDA 软光栅化与真实 Blender 回读**：`worker/software_rasterizer.py`（`rasterize_triangles_torch` 面积分桶 Z-buffer + 三点 Lambertian 光照 + `render_fast_review_frames`）提供 `[原视频 | 拟合渲染 | 50/50 叠加 + 红绿轮廓线 + IoU | 3/4 环绕视角]` 快速四联屏质检；正式 G3 与生产交付始终由真实 Blender 烘焙小数关键帧 GLB（每骨每帧提供有效世界矩阵并保留空 `Root` 下 `Hips` 位移）、重新导入核对世界矩阵并编码真实审查视频。
