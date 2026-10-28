# v3 Reader 算法与运行参数

新候选由云端 SkinTokens 的同一 4 候选入口生成，在 G1 准备时进行几何精修和 v3 命名。默认保留 Finger/Toe；只有显式 `MONSTER_FOLD_DIGITS=1` 才折叠。`monster-bone-taxonomy-v3` 与 `monster-reader-v3` 分别表示骨架和训练/权重契约；旧 v2 评测与旧 checkpoint 保持原路径，不覆盖原文件和审批。

- `MONSTER_CAMERA_LENS=0`：默认正交；正数为毫米焦距，例如 `50`。Blender CLI 同样接受 `--lens 50`。
- `MONSTER_STUDIO_LAYOUT=horizontal`：默认现有横向多面板；`stacked` 使用侧视在上、俯视在下，两者共享横向像素方向。三视图与 stacked 互斥。
- `studio_bpy_worker.py --mode export_cameras --blend_path <scene.blend> --out_report <cameras.json>`：导出场景相机的外参、投影、焦距和偏移。`studio_camera.Camera` 为 Worker/FK/推理共用模型。
- `MONSTER_RENDER_STRIP=1`：默认每批最多 16 个正交姿态横排渲染；奇偶副本沿视线错开 2.5m。透视或堆叠使用逐面板渲染。`0` 可关闭条带以排查渲染差异。
- `MONSTER_AUTO_FACING=1`：SkinTokens 输入清洗副本按网格对称、头/脚尖前伸估计朝向；显式 yaw 覆盖自动估计。G1 仍需审核朝向与骨架，不能把启发式结果当作通过。
- `MONSTER_SLOW_MOTION_SCALE=0.5`：提示生成半速动作，并默认以 `MONSTER_TIME_SCALE=2` 的时间映射烘焙。默认均为 `1`；显式 `MONSTER_TIME_SCALE` 优先。禁特效约束始终追加。
- `MONSTER_VIDEO_FPS=24`：视频采样时间基准。DIS 光流以局部趋势检测周期性丢帧，按内容帧率映射为非整数时间；真实加速不会按速度重写时间，静止和重复图像保留编码时间间隔；GLB 保留小数关键帧。

视频首帧/推理保持锁定的原生分辨率。v3 合成训练使用独立的 896 方形面板与 `training_calibration`；窗口输入 448，局部手爪精修输入 224，共享网络权重。旧权重的输入、辅助头与骨名不兼容，必须重新生成标签并训练。默认冻结 DINOv2 前 4 层和 embedding，热身头 1 轮后微调后 8 层；默认主干 `1e-4`、头 `1e-3`，支持现有 `MONSTER_TRAIN_*` 配置和冻结对照。

渲染依赖由 `worker/blender-requirements.txt` 与 `scripts/bootstrap-blender.sh` 安装：新版 NumPy、SciPy、Pillow、scikit-image、threadpoolctl、OpenCV；训练额外使用现有 PyTorch/DINOv2。桌面 Blender 可通过 `MONSTER_BLENDER_DEPS` 指定与其 Python ABI 相同的依赖目录；SOP 从同 ABI 的调用解释器传递 site-packages，无需改 Blender 安装。不能把 Python 3.13 的二进制依赖交给 Python 3.11 的 Blender。

v3 推理：首帧阴影/相机对齐 → 全序列粗读 → 平滑窗口、前后邻帧上下文与 8x 逆变换热图平均 → 手爪精修/骨骼局部密集场解码 → 左右消歧 → 真实读数同步门禁 → 短遮挡桥接 → 解析雅可比 Cauchy 像素拟合 → 读数验证去尖峰、踏步根位移、弹簧链 → 卡限诊断。同步门禁在桥接前测量；跨机器逐比特一致不作承诺。

G3 必须提供真实预测 JSON 和独立视频，SOP 调用真实 Blender 烘焙 GLB、重新导入并输出同视角、50°/18° 的 3/4 视角和轮廓叠加片。轮廓 IoU 是背景分割外观指标；无 3D 真值时，3D 误差留空。代码/模板更新不会自动更新正在运行的 Colab Worker，不能在忙碌实例上替换代码；沿现有生命周期守卫在安全时切换归档版本，并重新完成对应角色门禁。

0.3.1 的数据缓存/训练契约新增 `monster-start-pose-v1` 与 `monster-context-pose-v1`，特征版本为 `v3-anatomical-reader-3`。标签必须保存完整 bones、matrix_local、pose_basis、rest/start 网格；前序上下文由同一骨架微扰姿态投影生成。旧的早期 v3 标签须重新生成；缺少起始姿态契约的 checkpoint 必须重训。v2 只读路径保持原分流。

几何精修用 1% H 体素，1.2% H 双壳独立导出（阶段一预检 `sop_preflight.py` 与阶段二 `prepare_candidate` 均自动持久化 `shell.npz` 并验证 `watertight_ready`；预检自动无损清理孤立点、悬空边、零面积退化面与重复面，对带开放边界或非流形多层薄片的网格自动启用水密壳兜底，并且在 `heat_bind_updated_skeleton` 中当整体或任意 `>=2%` 角色顶点的主分件覆盖率 `<50%` 时自动回退到 $100^3$ Marching Cubes 水密外壳 `solid_copy` Bone Heat + `cKDTree` 权重回传）。重绑时保留非零起始姿态及所有网格的实际表面，首帧校准使用 start 网格。flatness 使用四个局部体积截面；Jaw 使用实体上下唇之间前方开放的空腔。

这些接口的集成不表示原七阶段 Plan 全部验收，剩余项目见仓库 `docs/specs/monster_video2rig_completion_audit.md`。

0.3.3 的合成数据新增 `monster-calibrated-sampling-v1`：接触目标来自实际蒙皮后的躯干三角面；画幅采样使用所有标定相机 `[3%,97%]` 的共同可行区域。Yaw/Pitch/漫游/跳跃/持握/张嘴/STRIKE/LUNGE 按帧抽取事件，并在每个样本的 `v3_sampling` 中记录目标、误差和参数。早期 v3 标签缺少该契约时须重建，不能复用为新采样验收证据。

堆叠与三视图按原面板顺序保留神经网络 view ID；几何消歧阶段再将 front（堆叠为 top）放为主观测，逐一处理其他侧视。v3 GLB 烘焙要求每骨每帧提供有效世界矩阵，并保留空 Root 下 Hips 的位移；验证片之外还要比较实际导入GLB的小数帧世界矩阵。


## EEVEE 动画条带渲染

2026-10-27：Colab Cell 2 固定渲染 `bpy==5.2.3`（独立 Python 3.13 / NumPy 2 环境 `/content/monster-bpy/venv`）；SkinTokens 内部维持 Python 3.11 / NumPy 1.26.4，使用该 ABI 的最新 LTS `bpy==4.5.15`，EEVEE 多副本灯光/阴影隔离要求 Blender ≥4.3；旧 4.2 环境会明确阻止条带模式，可临时设置 `MONSTER_RENDER_STRIP=0`。升级需在无活动任务时运行 Cell 1 → Cell 2 → Cell 4；本地代码同步不代表云端升级完成。摄影棚默认 EEVEE 8 spp，Stage 2 正交视图启用批量渲染。副本保留独立骨架、Action、变形修改器、道具和灯光快照；输出仍为每姿态一张拼图及原坐标标签。沿相机右方向排列，奇偶深度错开 2.5m，增加半面板间隔。支持双/三视图及上下堆叠；透视视图与 G2 边界审阅逐样本渲染。清晰和模糊样本分批，条带快门固定 0.5，标签为中心帧 1。渲染失败会恢复场景并清理本次副本。

| 环境变量 | 默认 | 用途 |
| --- | --- | --- |
| `MONSTER_RENDER_ENGINE` | `EEVEE` | 可选 `CYCLES`，保留 GPU OIDN；实际引擎记入标定 |
| `MONSTER_EEVEE_SAMPLES` | `8` | EEVEE 渲染采样数 |
| `MONSTER_RENDER_STRIP` | `1` | `0` 退回逐样本渲染 |
| `MONSTER_STRIP_COPIES` | `16` | 目标最大批量，范围 1–16 |
| `MONSTER_STRIP_MAX_WIDTH` | `16384` | 含间隔的最大条带像素宽度；应按 GPU 上限设置 |

896px 面板含半面板间隔时默认每批最多 12 份；5,000 个姿态且前后各半清晰/模糊时，共 418 批、双视图 836 次渲染（逐样本为 10,000 次）。原生 2K 大面板会进一步减小批量。副本数减少的是场景提交开销，像素与几何工作量仍存在，不能直接承诺 16 倍吞吐。

v3 原图保持896×896/视角。ReaderDataset统计整批训练图片中在画幅内的关键点范围（每图至少三个点），取最大横/纵范围的p95再乘1.12，四舍五入并限制在面板短边内；不是取某一帧点坐标的p95。所有图片使用同一窗口边长，中心跟随各图关键点；RGB、关键点、上下文与稠密场同步变换到448×448。完整表皮包围盒仍保存在标签中作诊断，不再作为Reader训练边长。窗口可越界并填充，出画幅的内容无法恢复。有效分辨率收益依实际窗口而定，不保证固定1.8倍。

训练checkpoint必须保存reader_window（monster-reader-training-p95-window-v1），包含边长和训练面板尺寸。推理按实际面板高度/训练面板高度换算同一边长，粗读只定位中心，失读中心沿时间插值后平滑；不能把每帧范围抖动变成缩放。缺少此窗口契约的早期v3权重须重训，旧文件保留；具有当前起始姿态、上下文和采样契约的全画幅标签可直接重新用于训练，无需为窗口统计重复渲染。v2分流保持。

收据位于 `keypoints_gt.json.render_pipeline`，含实际批量、批次数、渲染调用和耗时。引擎、采样与条带配置写入数据集签名；重新生成数据集与训练权重时须沿用有效 G1/G2 授权及版本规则。此次插件开发没有重新生成任何角色训练集或修改关卡审批。Colab 正在运行的进程不会因本地同步而改变；部署新代码后按现有生命周期在无活动任务时重载 Worker。

0.3.4 的轨迹保存 `monster-video-timing-v1`：`fps`、`slow_motion_scale`、`time_scale` 与已缩放的 scene-frame 时间。`MONSTER_VIDEO_FPS` 支持小数帧率；Blender用 `fps/fps_base` 表示，GLB导出/导入必须使用同一标定。慢动作默认按倒数倍速恢复，显式 `MONSTER_TIME_SCALE` 优先；配置必须有限、FPS/倍速为正，慢动作比例位于 `(0,1]`。审查片以源视频实际帧率乘回放倍速编码，SOP保留真实渲染结果，不再覆盖成24 FPS。

后处理使用 cadence 和倍速映射后的真实秒数。去尖峰仅修改有毛刺的角通道，并用受影响关键点的高置信投影判断；不平均根位移。踏步恢复要求原地支撑、55%移动份额、0.92方向一致性、0.08H/s速度，按真实秒数作2Hz低通积分；已有根运动保持，独立道具保持，持握道具随身体平移。弹簧在每帧间隔内作4个约束子步，使用4Hz/.55阻尼解析状态转移，保持骨长并反解局部限角；静止转向姿态保留不可观测扭转。

## 0.3.5 求解器接触与标定

着地脚掌使用骨骼局部横向单位向量施加 `flat_feet=120`，不依赖可见的 `.sidea` 标签；腾空和竖直蹄不施加此项。接触集合在每帧求解前固定，优先采用当前可用像素读数，失读时用前帧深度。胶囊和地面约束按 `3000/H` 处理，中心线上用可运动的确定性方向逃逸。诊断保存每个起点代价、逃逸前像素误差、稳定读数数量及当前着地脚掌。

稳定轨迹要求前/中/后三帧置信度至少 `.6`，二阶像素差分小于该相机投影身高的 `2%`；其Cauchy尺度从6px变为12px。reset只恢复局部关节起始值，保留当前世界根位移与转向。

远侧肢体降权还需当前成对肢根的相机深度支持；近侧肢体与躯干不因低置信而被一并降权。透视像素身高使用标定焦距/起始身体深度，独立道具不参与身体中心。

首帧轮廓对齐保留完整三角形集合，输出优化前后IoU与双向Chamfer。透视优化6个外参；正交投影无法从像素识别光轴平移，保持该输入标定并明确报告 `unobservable_parameters=["tz"]`，不会声称恢复该自由度。

## 0.3.6 几何关节轴与非零起始姿态

`axis_roles` 在恢复后的 p0 中测量每骨 `rx/ry/rz` 的 bend/twist/side 角色。铰链使用静息父子骨的弯曲平面选择局部 X 或 Z 轴；近乎笔直时使用起始姿态的轴角色。`hinge.axis` 与 `axis_name` 明确记录通道，早期没有该字段的铰链仍按 X 读取。固定其余Euler分量后，用正弦/余弦解析求与父骨最接近的 `straight`，不再量化为半度。

起始弯曲严格超过12°时沿实际继续折叠的方向确定正反铰链；其余按角色正面，膝向后、肘向前。范围为有向 `straight-.05` 到 `straight+2.7` 弧度，并保留用户显式范围与LBS安全范围的交集。X/Z轴交换时，安全探针使用相应弯曲轴，正反方向均实测。首次15°已失败时从0°与失败角之间细分8次，不把失败的15°记录成安全值；交集为空直接失败。

无观察依据的叶骨扭转若在保存的p0中非零，配置通过 `frozen_start_channels_rad` 锁在该角度，保留精度；不会归零或增加求解自由度。`allow_leaf_twist` 此时仅允许保留这个非零固定值，实际范围仍为相同上下限。配置合并和重新探测保留该锁定，采样也使用同一值。旧模型若在这些条件下生成过错误的初始配置，应重建关节配置与对应标签。

## 0.3.7 骨架树精修与权重

`fold_digits=False` 保留完整指趾树。显式折叠时按每条digit分支的最大弧长判断 `<.22H`，而不是按单节长度；精确.22H及长分支保持。删除表记录各骨最终祖先，非digit附件重挂到保留掌骨；完整短指扇折叠后，掌骨到各远端均值。Blender合并被删组的权重，随后清理/归一化。

壳外比例通过随长度/体素调整的偶数中点采样测量，严格超过50%才删除；尖端回缩用二分定位，最多缩至原长30%。武器须具备连续薄末段，长度严格>.10H、厚度不超过1.5体素和近端掌部半径一半两者的较小值；在薄段开始后.03H保留拳头，再生成刚性Weapon子骨。短爪尖、过厚或中断后不足长的段不切。

脚底按最近脚段分配低处网格点，以朝前投影97%分位尾部的均值定位脚尖，避免左右近距脚互相取点。骨架对称修整处理完整成对树：体积IoU>=.80镜像平均；其余在链长比>=.75且对应骨段不过度悬殊时只等长，递归平移后代以保持关节连接。多出不成对附件的树保持原样。

独立碎片的2%阈值使用整个角色的顶点总数，独立Prop网格不进入分母；较小片按最近骨段1/d⁴分配至多四影响。恰好2%不进入碎片重分配；远侧串权重清理和刚性Weapon/Prop规则仍独立执行。

## 0.3.8 表皮点与自定义相机

表皮关键点用 `skin.bone_names` 映射实际蒙皮索引，关键点槽位顺序与蒙皮顺序可不同；胶囊半径使用相同映射。叶骨依据当前父关系判定，不信任可能过期的children缓存。严格w>.3的静息蒙皮顶点支持98%轴向.tip和横截面1%/99%主次轴侧点；不足4个支持点时不生成虚假标记。全部原始头/尾2N槽位在前，附加点跟在其后，Prop保留.end1/.end2/.mid及用于完整旋转的离轴点。

双关节像素反例验证：手掌自身轴向roll不会移动头尾端点，必须用侧点恢复；父骨扭转还由非共线的下游点约束。两者按各自有限范围共同求解，不能用端点误差为零证明扭转正确。

自定义相机支持非方形横/竖画幅与AUTO/HORIZONTAL/VERTICAL sensor fit。Camera.scale统一表示垂直画幅世界高度；从Blender正交相机导出时按实际fit轴换算。AUTO虽然随画幅选择fit轴，其物理传感器尺寸仍使用sensor_width，仅显式VERTICAL使用sensor_height。镜头偏移按实际fit像素尺度换算。

特征版本更新为 `v3-anatomical-reader-4`，数据签名随之变化以防旧的错误表皮点缓存继续复用。已有checkpoint按其保存的extra_points/槽位推理，不能把旧训练证据当作新标签语义的验收。

## 0.3.9 双路径架构与完整 video2rig 后处理收敛

从无骨骼模型 → 有骨骼模型 → 有动作模型的全链路（保留云端 VAST-AI SkinTokens 四候选与 MiniMax H3 视频合成）收敛为两条可通过环境变量精确切换的路径：

1. **路径一（默认主路径 `MONSTER_PIPELINE_MODE=video2rig`）**：完整落地 `video2rig` 的骨架精修、表皮关键点与标签契约、CPU Z-buffer + BVH 可见性、标定采样、DINOv2 Reader、2D 重投影求解器与 8 步 `clean_motion` 后处理，同时融合我方优势（默认保留完整手指/脚趾 `MONSTER_FOLD_DIGITS=0`、LBS 边拉伸安全关节角探测、实测三角面接触守卫 `surface_contact.py`、首尾帧闭环锁与 G1–G3 形式化门禁）。
2. **路径二（Feature Flag 旧版兼容路径 `MONSTER_PIPELINE_MODE=legacy`）**：将原有 `monster-bone-taxonomy-v2` 命名、基础 `2N` 头尾标签、`DualPanelMonsterDataset` 与 3D 融合 + 分类型 FK 求解器收敛在显式开关之后。

| 环境变量 | `video2rig` 默认 | `legacy` 默认 | 说明 |
| --- | --- | --- | --- |
| `MONSTER_PIPELINE_MODE` | `video2rig` | `legacy` | 顶层双路径总开关（`video2rig` \| `legacy`） |
| `MONSTER_RIG_CONTRACT` | `v3` | `v2` | G1 骨架后处理与契约（`v3` 执行 `prepare_candidate`；`v2` 保持旧分类） |
| `MONSTER_FOLD_DIGITS` | `0` | `0` | 默认保留完整手指/脚趾；显式 `1` 按 `<0.22H` 分支弧长折叠短指趾 |
| `MONSTER_KEYPOINT_MODE` | `video2rig` | `legacy` | `video2rig` 启用 `.tip/.sidea/.sideb/.end1/.end2/.mid`；`legacy` 仅 `2N` 头尾 |
| `MONSTER_VISIBILITY_MODE` | `hybrid` | `bvh` | `hybrid` 同时保存 BVH `visibility_ratio` 与 CPU Z-buffer 布尔可见性；可选 `zbuffer` \| `bvh` |
| `MONSTER_EXPORT_NPZ_LABELS` | `1` | `0` | 在 `keypoints_gt.json` 旁同步导出 `labels.npz`、`visible.npz`、`poses.npz` |
| `MONSTER_DECODE_MODE` | `parabola` | `dsnt` | 热图峰值解码器：`parabola`（`video2rig` 三点抛物线亚像素峰值，推理默认；训练损失保持可微 DSNT）或 `dsnt`（软 argmax） |
| `MONSTER_SOLVER_MODE` | `reprojection_v3` | `legacy_3d_kind` | Stage 4/6 动作求解器：`reprojection_v3`（`infer_v3_frames` 默认开启 `w_prior=2.0`、`w_depth=0.2`、`depth_where="unseen"`、`robust_mode="scipy_hybrid"`）或 `legacy_3d_kind` |
| `MONSTER_CLEAN_MOTION` | `1` | `0` | 启用完整 8 步 `clean_motion` 后处理管线 |
| `MONSTER_SURFACE_CONTACT` | `1` | `0` | 启用 `surface_contact.py` 实测网格鞋底接触与防穿模守卫 |
| `MONSTER_NO_CUT_TRACKS` | `1` | `0` | 默认开启 `no_cut_tracks` 高置信平滑轨迹防误切否决 |
| `MONSTER_TWOWAY_JUMP` | `1` | `0` | 默认开启 `twoway` 正反向一致常速跳变过滤 |
| `MONSTER_PANEL_DRIFT` | `1` | `0` | 默认开启 `panel_drift` 双正交共享轴 `>5% H` 中线漂移校正 |
| `MONSTER_BACKBONE` | `dinov2-small` | `dinov2_vitb14_reg` | 视觉主干选择：`dinov2-small`（`facebook/dinov2-small` / `dinov2_vits14_reg`，$d=384$，`video2rig` 默认）或 `dinov2_vitb14_reg`（$d=768$，`legacy` 默认） |
| `MONSTER_READER_CROPS` | `0` | `0` | 可选开启独立手爪局部放大器 `reader-crops.pt`（默认关闭） |
| `MONSTER_READER_DENSE` | `0` | `0` | 可选开启独立密集表面场检测器 `reader-dense.pt`（默认关闭） |

本批次补齐的 `video2rig` 关键细项：
- **表皮点镜像一致方向与 Jaw 单向铰链**：`surface_keypoints` 使用 `(outward, facing, up)` 确定性定向横截面特征向量，使 `.L`/`.R` 成对骨骼的 `.sidea1/2` 与 `.sideb1/2` 解剖侧严格镜像一致；`configure_jaw_hinge` 为 `Jaw` 配置向下张嘴单向铰链范围 `[-0.05, +0.70]` rad（其余轴 `[-0.08, 0.08]`）。
- **持握武器内部守卫与对称端交换**：手部蒙皮延伸 `>2.5 L_bone` 时自动生成 `.held_tip` 与 `.held_mid` 胶囊/地面守卫点；`Prop_*` 双端标签支持训练期左到右排序与求解期按观测动态交换 `end1/end2`，并加入 `prop_place` 多起点候选与模型自身跨肢体胶囊遮挡降权。
- **读数预处理与完整 8 步 `clean_motion`**：补齐 `no_cut_tracks`、`twoway` 双向常速跳变过滤与 `panel_drift`（正交共享轴 `>5% H` 中线漂移校正）；`clean_motion` 依序执行 `smooth_turn`、`despike_reading_backed`（含 `>0.03H` 根位移毛刺与单肢快动作保护）、`lowpass_motion_adaptive`（4/5/8Hz 速度自适应 Butterworth 低通）、`floor_bend_and_smooth_lift`（静止着地段与 60 遍高斯单调抬升包络）、`recover_root_translation_from_steps`（含 `steps_steady=0.5` 步幅离散度门控）、`lock_feet`（双端点钉死 IK、1 帧间隙闭合与 `lock_blend=3` 线性淡入淡出）、`simulate_damped_spring_chains`（含 `spring_soft=0.4` 根到尖频率渐减）与 `keep_out`（终态胶囊防穿模推出）。


## G2 与交付契约补齐

G2边界采样也走RigModel的世界根/保存起始姿态路径，保留不参与拟合的组织链与固定骨；边界角/位移仍来自锁定seed和joint_limits.json。审查记录直接使用实际Blender输出的joints/reader_points/training_calibration，包含透视与上下堆叠；失败时不生成替代示意图。覆盖审计在实际拟合通道上进行，世界根角度单独记录，原始局部角标签保持不变。

新推理轨迹保存motion_structure（motion_roots/fitted_bones），供G3与交付检查冻结basis、世界根位移、父子偏移、SO(3)、骨长与局部限角。v3数学门禁用实际FK/像素求解器检查中性固定点、解析J与失读保持；v2数学探针保留原路，不能代替v3证据。只有轨迹提供实际表皮接触收据时，才在对应报告标记已检查P4。


## 0.3.17 审查修复

- cadence仅在周期性抽帧证据成立时调整内容节奏，低光流不代表零时间；停顿、低幅动作和重复编码图像保留时间间隔。前景光流取均值，避免静止躯干掩盖手脚局部运动。
- v3热图置信度使用实际sigmoid峰值；8x TTA取逆变换平均热图的峰值，不以空间softmax的相对集中度提高置信度。v2原有解码语义保留。
- 表面接触守卫保留输入的完整冻结局部旋转与父坐标系偏移，仅调整实际拟合通道和授权根高度；无接触时不以跨帧先验平滑姿态。
- 新训练按上游离线Reader使用t-1/t+1双向上下文，首尾索引夹到首/末帧；checkpoint写入`temporal_context_contract=monster-reader-bidirectional-context-v2`。缺少该字段的旧权重仍按t-1/t-2因果上下文及旧边界填充推理，未知契约拒绝。旧标签可复用后重新训练，新双向训练不会默默继承旧因果初始化权重。
- v3 P2必须实际验证非中性、随机和边界姿态的像素重建，不能仅靠起始姿态固定点证明可恢复性。回执明确记录采样覆盖、不可观测旋转和被守卫排除的姿态，不宣称穷尽全部姿态。


## 0.3.18 video2rig 深度对齐：dinov2-small、三模型拆分、防抄袭/残影、左右消歧、先验/Cauchy 求解与软光栅化

1. **`dinov2-small` 主干模式与 `Video2RigReaderNet`**：
   - 支持通过 `MONSTER_BACKBONE=dinov2-small`（或训练/模型参数 `backbone_name="dinov2-small"`）切换至 `facebook/dinov2-small` / `dinov2_vits14_reg`（$d=384$，12 层，patch 14）。
   - `configure_backbone` 与 `Video2RigReaderNet` 同时兼容 `torch.hub`（`blocks` + `norm`）与 HuggingFace `transformers`（`encoder.layer` + `layernorm`），默认遵循 `FROZEN=4`（冻结 `embeddings` 与前 4 层 block，微调后 8 层 block + 末端 `norm`），并提供 `STRIDE=4`、`HEAT=112`、`CH=96` 的轻量级双路融合解码器。
2. **三模型独立训练/推理（`reader.pt` / `reader-crops.pt` / `reader-dense.pt`）、$K$ 通道深度头与按总高 $H$ 归一化的局部场**：
   - **基础检测器 `reader.pt`（`reader_role="base"`）**：关键点热图 + $K$ 通道逐关键点深度头 `dhead` `(B, K, H, W)`，通过 `sample_maps(dmap, coords)` 在每个关键点预测 2D 坐标处双线性采样得到 `point_depth` `(B, K)`，由 `sight_depth_targets` 按角色高度归一化视线深度 `(X - mid_body) @ cam_fwd / H` 对所有画幅内关键点（含被遮挡点）施加 SmoothL1 监督；`load_probe_weights` 自动将旧版 1 通道 `dhead` 扩展为 $K$ 通道。
   - **手爪局部放大器 `reader-crops.pt`（`reader_role="crops"`，默认关闭 `MONSTER_READER_CROPS=0`）**：`end_groups` 提取每条侧向非松散肢体链末段及其指/趾后代，`crop_sides` 按最末两节骨长之和 `1.6 * reach` 计算裁剪边长并夹紧至 `[72, 320]` px；推理期 `read_crops` 仅在局部裁剪置信度 `cq > c` 时覆盖手爪/足端关键点。
   - **密集表面场检测器 `reader-dense.pt`（`reader_role="dense"`，默认关闭 `MONSTER_READER_DENSE=0`）**：`rasterize_dense_fields` 默认 `local_norm="height"`，将骨骼局部坐标 `(pts - head) @ R` 按角色总高 $H$ 归一化（消除短骨/指节数值放大）；`joints_from_dense_fields` 支持 `"height"`、`"bone_length"` 与旋转不变 RMS `"auto"` 尺度识别；`dense_reader_losses` 采用背景权重 `1.0`、前景骨骼权重 `3.0`、`part_weight=5.0`、`local_weight=20.0`。
3. **时序上下文防抄袭增强 (`corrupt_context`) 与胶囊体挥击残影 (`smear_limbs_capsules_gpu`)**：
   - `corrupt_context` 对 `t-1 / t+1` 双向邻帧上下文施加 `1.5 px` 高斯抖动、5% 远距离离群点（`±0.2 * SIZE`）、16% 单点随机丢弃、25% 整条对称肢体左右互换（`twins`）、`U(0.3, 1.0)` 置信度随机缩放与 20% 全帧上下文清空。
   - `smear_limbs_capsules_gpu` 以 35% 概率在每张训练图抽取 1～2 条可见肢体线段，生成软边缘 2D 胶囊体掩膜并沿法线方向（`atan2 + pi/2 ± 0.6 rad`）用 `torch.roll` 平移 13 份副本做预乘 Alpha 混合。
4. **左右消歧与多视角阅读预处理细项**：
   - `fix_twins_viterbi`：首帧 `t=0` 交换态施加 `switch_cost` 初始惩罚；`no_cut_tracks=True` 时对内部平滑且左右间距充足的交换区间整段取消翻转。
   - `jump_pass_constant_velocity`：逐关键点常速插值检验 + `_twin_parent_links` 父骨骼连接点距离保护（父点间距 `>= 20 px` 时要求子点交换同样缩短到父点的距离）+ `twoway` 前后双向一致判定。
   - `downweight_overlapping_twins`（`twin_same`）：检查同肢体链相邻关键点 `i-1, i+1` 的左右间距，当 `0 < max(apart) <= 3 * near`（双腿真实并拢站立）时不降权。
   - `downweight_flipped_limbs`（`flipped`）：上端关节静止（`< 2% H`）而肢体方向突然翻转 `> 60°` 并在 `<= 24` 帧内恢复（`< 30°`）时，末端权重乘以 `0.1`。
   - `downweight_contradicting_far_limb`（`pick_panels`）支持 `midline_ratio=2.0` 与 3 帧窗口扩展 `window_expand=True`；`_apply_body_picture_weights` 支持 `zero_contradicted=True` 将落在其他身体部位上且跨视角矛盾的远端观测清零。
5. **求解器角色先验 (`RigModel.prior_weights`) 与混合 Cauchy 损失 (`robust_cauchy_loss`)**：
   - `RigModel.prior_weights(one_view=...)` 按关节自由度角色赋予静息先验权重：`twist=25.0`、铰链非弯曲轴 `60.0`、`spine=15.0`、`clavicle=40.0`、`toe=30.0`、`loose=6.0`、主弯曲轴 `2.0`、根旋转 `2.0`（单视角）/ `0.2`（多视角）、根平移 `0.0`。
   - `fit_clip_2d_reprojection` 支持 `w_prior`、逐关键点视线深度残差 `depth`（`w_depth=0.2`）与 `robust_mode="scipy_hybrid"`（通过 `robust_cauchy_loss(c_scale, n_robust)` 对前 `n_robust` 个 2D/深度数据残差启用 SciPy 原生三行 `[rho, rho', rho'']` Cauchy 核 + `x_scale="jac"`，而对地面/胶囊体防穿模/平足/先验/平滑约束保持严格二次 L2）。
6. **CPU/PyTorch 软光栅化器与快速四联屏质检 (`software_rasterizer.py`)**：
   - `rasterize_triangles_torch`（按面积分桶 `<64` / `<1024` / 大三角形 + `Z * P + p` 排序的 CPU/CUDA 通用 Z-buffer 光栅化器）、`vertex_normals` 与 `render_mesh_software`（三点 Lambertian 光照 + 顶点色/UV 贴图双线性采样）。
   - `render_fast_review_frames` 免 Blender 快速生成 `[原视频 | 拟合渲染 | 50/50 叠加 + 红绿轮廓线 + IoU | 3/4 环绕视角]` 四联屏对照帧序列、剪影 IoU 统计与 MP4 视频。
