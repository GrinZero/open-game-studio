> 当前版本为0.3.17。本页保留历史部署记录；0.3.16后续复核发现的四个问题与上下文语义修复，以[0.3.17交接](monster_video2rig_0317_handoff.md)为准。测试通过不表示穷尽全部姿态或真实角色质量。

# Monster v3 Video2Rig 集成验证

本页保留0.3.0历史证据。当前0.3.16已完成七阶段代码机制、本地逐项验收与实际运行目录热更新；252项零跳过及最终证据见[0.3.16交接](monster_video2rig_0316_handoff.md)和[完成度复核](monster_video2rig_completion_audit.md)。历史93项结果不代替后续细项验收。

日期：2026-10-26（Asia/Taipei）。Monster 部署包版本 `0.3.0`；接口协议继续为 `1`，骨架 schema 为 `monster-bone-taxonomy-v3`，新 Reader 契约为 `monster-reader-v3`。

源码入口与七阶段对应关系见 [实施计划](monster_video2rig_integration_plan.md#落地入口与兼容边界2026-10-26)。新机制通过候选准备、合成、训练、checkpoint 分流、视频推理和 G3 实际回读入口调用；旧 checkpoint 继续使用原推理路径。云端仍保留 SkinTokens 4 候选，默认不折叠指/趾。

## 已执行证据

| 检查 | 实际结果 |
| --- | --- |
| `MONSTER_TEST_BLENDER=/Applications/Blender.app/Contents/MacOS/Blender npm run monster:check` | 93 项测试全部通过、零跳过；语法、Notebook、API 契约及形式化门禁通过 |
| 再运行默认 `npm run monster:check` | 93 项通过，其中 4 项按配置跳过真实 Blender；默认路径仍保持 CPU 可运行 |
| `npm run monster:formal-gate -- self-test` | P1–P4 通过；新增 v3 FK/相机数值验证通过，FK 雅可比最大差分误差约 `1.58e-10` |
| `npm run monster:eval -- run fantasy_creature_2k_v2 --tier fast` | 历史 2K 评测完成，9 个 golden-frame 探针全部 PASS |
| `npm test` | 73 项通过，零失败/跳过 |
| 实际 Blender v3 夹具 | 正交/透视投影与 Blender 原生投影一致；100 格外壳 Bone Heat、完整 16 副本条带、896 面板及局部密集标签生成通过 |
| 实际标签 → CPU 训练 → 重载 → 推理 | 真实渲染标签进入生产训练函数，保存/重载所有新头及首帧参考成功；使用 TinyBackbone 的两轮 CPU 冒烟测试通过 |
| 实际 GLB 回读 | 小数关键帧 `1.5` / `2.5` 经烘焙、导出和重新导入，生成同视角、3/4（50°/18°）与轮廓叠加片；报告标记 `actual_glb_readback=true` |

本机 Blender 为 `5.3.0 Beta`、Python 3.13。Colab 的 bpy 4.2/Python 3.11 依赖已写入 bootstrap；本次没有启动或重启云端 GPU，也未验证真实 DINOv2/角色训练精度、显存峰值或云端吞吐。TinyBackbone 和几何夹具只验证接口、梯度、坐标变换与产物真实性，不能代替真实角色的 G1/G2/G3。

原始日志、GLB 回读报告与示例片保存在本机 `var/monster/verification/video2rig_v3_20261026/`（忽略目录）。2K 历史基准产物在 `var/monster/evals/runs/20261026_144824_fantasy_creature_2k_v2_fast_taxonomy_kind_v2/`。可复现的 Blender 夹具为 `packages/monster/tests/fixture_v3_blender.py`。

## 兼容与验收边界

- 新骨名、局部轴、蒙皮、附加点、部件/局部坐标头改变了训练契约；不能把旧权重改一个 schema 字符串后复用。使用已发布的迁移失效规则重新准备固定版本材料及训练。
- 未移动用户角色目录、修改生产 GLB、迁移审核账本或代替用户批准任何关卡。仓库原有未提交改动保留。
- 单线程 BLAS 在固定环境下降低分支差异；跨机器、依赖版本不承诺逐比特一致。
- 轮廓 IoU 是背景分割视频与实际渲染 alpha 的外观一致性指标，不是三维真值误差。缺少可比较投影点时，2D 误差也留空。
- v3 G3 缺少真实预测、独立视频或 Blender 时直接失败；原有 legacy G3 的占位回退不会被 v3 使用。

## 本机热更新

已执行 `npm run monster:package`、`npm run build`、`npm run check:distribution` 和 `node scripts/sync-runtime.mjs`。核实已安装 MCP 配置与 LaunchAgent 均指向 `/Users/bugyaluwang/.local/share/codex-integrations/game-studio-local/runtime`，HTTP 端口为 4310。

- 构建标识：`v0.1.14-20261026231225`，Asia/Taipei 构建时间戳。
- 实际 HTTP 提供的 JS 含此标识；已安装配置建立的真实 stdio MCP 连接返回相同标识、27 个工具，内嵌 HTML SHA-256 与构建完全一致。
- 在实际运行目录启动 Python 导入，确认模块根位于 runtime，包版本 `0.3.0`、骨架 v3 与 Reader v3 契约已加载。常驻 Node 服务逻辑未修改，Python 工作模块按子进程调用加载，未重启无关服务。
- 项目登记 API 在验证前后内容/哈希完全一致。备份：`backups/resources-1793027587098`（位于本机集成目录）。
- 检查时 Chrome 标签和 AGY 主窗口均没有打开 Game Studio 工作台；AGY 主窗口是原计划会话。随后在 Chrome 主窗口新打开工作台，点击版本按钮完整刷新，肉眼可见 `v0.1.14-20261026231225`。这是新打开并保留的主窗口页，不声称已刷新一个不存在的旧工作台窗口；聊天内嵌面板没有被当作主窗口证明。
- 版本截图：`var/monster/verification/video2rig_v3_20261026/main-workbench.png`；运行核验：同目录 `runtime-verification.json`。

源码和技能同步不等于远端 Colab Worker 已切换版本；本次没有操作云端实例。生产角色须在安全空闲时部署 `monster-0.3.0.tar.gz`，按新契约重新生成标签、训练及完成门禁。

## 0.3.1 反例修复批次（2026-10-27 Asia/Taipei）

全 Plan 保持进行中；本批次补齐的具体机制和剩余项见完成度复核，不以测试数量声称全部需求完成。

- 非零起始姿态：实际 Blender 覆盖根位移、多个局部旋转、骨 roll、分离网格、重绑前后表面、只读导出、首帧和 p0 FK。缓存/标签保存 start/rest 与 `monster-start-pose-v1`。
- 训练上下文：同一完整骨架生成两帧前序微扰姿态，投影和裁剪后输入；新 `monster-context-pose-v1`，特征版本 `v3-anatomical-reader-3`。缺契约的早期 v3 标签须重建。
- 修正独立 1% H 几何体素、真实体积镜像 IoU、四种朝向、四个局部 flatness 截面和真正前开嘴部空腔。
- 修正 Colab/Notebook 默认后8层/1e-4，增加12层结构梯度验证；修正按角色 H 的残影和实际脚点 RGB 接触阴影。
- 修正跨视图粗读驱动的逐帧密集场深度中心、透视手爪裁剪和空 Root→Hips 冗余自由度；补胶囊骨中点和离轴道具 Kabsch。
- cadence 只由周期性丢帧证据调整时间；真实加速和单次挥击不再按流速重定时。

`MONSTER_TEST_BLENDER=... npm run monster:check`：**114 项全部通过、零跳过**，包含真实 Blender、标签→CPU训练→重载和 P1–P4；`npm test`：**73 项全部通过**。历史2K fast：**9个 golden-frame 全部 PASS**，产物 `var/monster/evals/runs/20261026_160337_fantasy_creature_2k_v2_fast_taxonomy_kind_v2/`。

已重新打包 `monster-0.3.1.tar.gz`（69个发布文件）、build 和 distribution check，并热同步到已核实的实际 runtime。

- 当前构建标识：`v0.1.14-20261027000323`；HTTP JS 与真实已安装配置的 MCP HTML 一致，27工具；MCP HTML SHA-256 `302b45fd67d2a59a9c1ca7f91bce796ccd5661817023f622dbb4812b9ad32b19`。
- runtime 的23个 Python Worker 文件逐一哈希相同；从 runtime 启动解释器导入，路径、0.3.1、Reader/start/context/feature 契约均核实。Python作业按新子进程加载；本批次未修改常驻Node逻辑。
- 项目登记前后哈希仍为 `3517552ab1f6836d1fe979c2b263f156f0faa5236624f2a415323b731ba55392`；保留2个项目。备份 `backups/resources-1793030700651`。
- 上批保留的主窗口标签已被关闭，当前浏览器清单无工作台页；重新在 Chrome 主窗口打开工作台并点击版本完整刷新。实际可见新标识，保留标签 `331867554`。这是重新打开并刷新后的主窗口，未以聊天内嵌页或已关闭旧页作为刷新证明。
- 日志、前后登记快照、运行验证和 `main-workbench.png` 位于 `var/monster/verification/video2rig_v3_031/`（忽略目录）。未操作云端实例，也未用代码检查代替角色审核。

## 0.3.2 几何与初次绑定批次（2026-10-27 Asia/Taipei）

继续原 Plan，未缩减范围。缺腿/多足路径改为按共享躯干分叉与实体厚度提取自由肢段，移植到最近躯干骨；后髋改按膝盖、当地脊柱及体积半径定位；长 Hips/Spine 后段与弯曲多节尾按弧长重切、最多10节并保留附件父链。缺权或新增变形骨时，实际原网格先运行 ARMATURE_AUTO，实际获权率低于50%才进入外壳 Heat；已有完整输入权重可保留。

- 新增共享分叉、后髋精确坐标/前肢不变、弯曲多节尾/附件、无既有尾的长 Hips 反例。
- 实际 Blender 再次验证缺权原网格的直接 Bone Heat，非零起始表面误差仍在 `3e-6` 内。
- 包含真实 Blender 的 `monster:check`：**118项通过，零跳过**；语法、Notebook、API契约、CPU训练重载与P1–P4一并通过。本批次未改Node源码，前批次73项Node测试及9个历史golden-frame证据继续适用。
- 重新打包 `monster-0.3.2.tar.gz`、build、distribution和runtime同步已完成。当前标识 **`v0.1.14-20261027001425`**。
- 已安装配置的真实 MCP/HTTP 标识一致，27工具；23个Worker文件逐一哈希相同，runtime实际导入0.3.2及新几何/绑定函数成功。MCP HTML SHA-256：`8034f44ce43791278b436b6eb890d3f274dcbcec83fa5dda38e050c7068c388b`。
- 同一Chrome主窗口标签 `331867554` 已刷新并可见新版本，未另开验证页替代；截图和日志 `var/monster/verification/video2rig_v3_032/`。登记哈希仍为 `3517552ab1f6836d1fe979c2b263f156f0faa5236624f2a415323b731ba55392`。备份 `backups/resources-1793031321876`。

仍有尚未完成的机制/专项验收，见完成度复核。新几何单测和新版本号不证明完整七阶段已完成；生产角色与云端实例均未代操作。

## 0.3.3 标定采样、堆叠与刚体回读批次（2026-10-27 Asia/Taipei）

全 Plan 仍在推进；以下是本批次的具体证据，不把代码入口或通过数量当作全部需求验收。

- 新 `pose_sampling_v3.py`：实际 posed trunk 三角面上的面积采样和接触 IK、标定相机共同画幅多边形、逐帧动作事件。50k次统计验证18%合拢/17%体表接触/25%全向/20%俯仰/50%漫游/30%跳/50%张嘴/75%持握/25%STRIKE/30%LUNGE事件；真实速度幅度仍是独立待验项。
- 生产合成入口接收真实蒙皮与实际相机；记录每帧 `v3_sampling` 的目标、接触误差、根参数、持握关系与事件。漫游检查包含正交、透视、镜头偏移、三视图和堆叠；道具不参与角色体表/中心计算。新标签契约 `monster-calibrated-sampling-v1`，旧v3缓存需重建。
- 横向与side/top堆叠均实际执行896标签渲染、共享水平像素列检查、生产缓存验证、CPU训练/重载；堆叠实际进入推理、GLB同视角/3/4/轮廓回读。TinyBackbone只验证消费协议，不证明真实DINOv2识别精度。
- 网络view ID保持训练面板顺序；几何阶段以front/top为主观测，逐一消歧其他侧视。Oracle反例验证原始堆叠面板读取、网络ID、侧视左右交换修复及最终命名字段；另测重叠.06H边界和跳变时置信度随读数转移。
- 实际Blender自定义透视相机导出JSON，再由生产Camera解析消费，投影一致。
- 新增独立道具网格的全旋转/位移GLB导出导入，逐项比较0/1.5/2.5帧的实际世界矩阵。空Root下Hips的真实采样、非零p0与GLB大角度/位移也实际比较矩阵。v3烘焙要求所有骨骼每帧具有有效世界姿态。
- 回归揭露并修复了独立Prop骨主导角色Bone Heat的错误：角色Heat暂时排除Prop骨，Prop网格不进入角色外壳；恢复use_deform后，道具仅绑定自身1.0，角色清理Prop串权重。
- 非有限UV/置信度按缺失处理；无观测帧不触发五转身起点，保留有限姿态，诊断支持严格JSON序列化。

最终 `MONSTER_TEST_BLENDER=... npm run monster:check`：**129项全部通过、零跳过**，含真实Blender、CPU训练重载及P1–P4。`npm test`：**73项全部通过**。历史2K fast：**9个golden-frame全部PASS**，产物 `var/monster/evals/runs/20261026_164318_fantasy_creature_2k_v2_fast_taxonomy_kind_v2/`。

重新打包 `monster-0.3.3.tar.gz`（70个发布文件）、build、distribution和运行同步均完成。

- 当前标识：**`v0.1.14-20261027004427`**；HTTP和真实已安装配置建立的MCP资源一致，27工具。MCP HTML SHA-256 `dab948f9bcaaed1367ae6778eeeb0468f922e2858f0e2cd704fbbcc29d0f118a`。
- runtime中24个Worker文件逐一哈希相同；从实际runtime导入0.3.3、新采样契约和视图消歧入口成功。常驻Node逻辑未修改，Python调用加载新子进程；未操作云端实例。
- 同一Chrome主窗口标签 `331867554` 已刷新，截图可见新版本；没有用新验证页替代。
- 项目登记前后保持2个项目，哈希 `3517552ab1f6836d1fe979c2b263f156f0faa5236624f2a415323b731ba55392`。备份 `backups/resources-1793033125590`。日志、登记快照、运行验证及 `main-workbench.png` 在 `var/monster/verification/video2rig_v3_033/`。

剩余项继续按完成度复核执行；未代替用户审批角色，也未将旧v2基准当作v3识别质量证明。

## 0.3.4 时间标定与物理后处理批次（2026-10-27 Asia/Taipei）

全 Plan 继续实施，剩余机制和专项证据仍以完成度复核为准。本批次修复以下实际差异：

- 新 `video_timing.py` 统一 FPS、慢动作比例与显式回放倍速；轨迹保存 `monster-video-timing-v1`，早期v3轨迹也可从cadence.scene_fps恢复时钟。生成提示和H3编码使用同一配置；堆叠提示与默认尺寸保持side/top布局。
- Blender烘焙与导入均设置FPS/fps_base，避免重新导入后恢复24 FPS；G3帧及形式化GLB回读保留小数时间。原生测试验证30 FPS、半速生成/2倍回放、GLB accessor中的实际秒数、0/.75/1.25帧矩阵和ffprobe读取的60 FPS审查片。
- G3 SOP直接保留真实渲染的已标定视频；测试把真实Blender产物送过SOP协议入口，确认视频字节不被24 FPS重新编码覆盖。测试只替换临时账本/关卡存取与重复渲染调用，未审批用户角色。v3缺少真实GLB/比较帧/视频时直接失败。
- 去尖峰改逐骨角通道、逐受影响关键点验证；真实单肢快动作保留、错误毛刺移除、6°边界不动，根平移不被一起平均。
- 踏步恢复按最终回放的物理秒数处理；验证55%移动份额、.92方向一致性、.08H/s速度两侧和2Hz频率响应，覆盖真实铰接支撑腿、已有根移动、持握/独立道具、非均匀及重复帧。
- 弹簧使用4Hz/.55阻尼解析状态转移，每帧4个约束子步、父锚点插值、骨长投影与局部限角。修复共线根锚点任意SVD扭转，以及局部反解重估根方向导致静止转向尾巴偏移的问题。测试核验频率/衰减、静止扭转、倍速/重复帧、嵌套尾/耳/柔性链、骨长和限角。

当前工作树包含已保留的其他编辑及Blender运行时/条带测试；完整 `MONSTER_TEST_BLENDER=... npm run monster:check`：**154项全部通过、零跳过**，语法、Notebook、API和P1–P4通过。`npm test`：**73项通过**。历史2K fast：**9个golden-frame全部PASS**，产物 `var/monster/evals/runs/20261026_171619_fantasy_creature_2k_v2_fast_taxonomy_kind_v2/`。代码文件修改时间均早于该轮通过检查；未覆盖并行编辑。

重新打包 `monster-0.3.4.tar.gz`（76个发布文件）、build、distribution与运行同步完成。最终核验：

- 当前构建 **`v0.1.14-20261027011913`**，HTTP与真实已安装配置建立的MCP资源一致、27工具；MCP HTML SHA-256 `7b0b89579d961d10f2605cc1afce0d0a5ff7a05b78a4e9978e9af0d07f287185`。
- runtime的27个Python Worker文件逐一哈希相同，实际runtime启动导入0.3.4及新时间契约成功；30 FPS/半速示例实际映射为 `[0,.75,1.25]`。常驻Node逻辑未修改，Python任务按新子进程加载；未操作远端云实例。
- 旧主窗口标签331867554已关闭，Chrome清单无其他工作台页；在主窗口重新打开331867572并点版本完整刷新，可见新标识。此为重新打开并刷新的主窗口，不声称刷新了已关闭的旧页，也未用内嵌验证页替代。
- 项目登记仍为2个项目，前后哈希 `3517552ab1f6836d1fe979c2b263f156f0faa5236624f2a415323b731ba55392`。备份 `backups/resources-1793035160411`。日志、登记快照、运行验证和 `main-workbench.png` 位于 `var/monster/verification/video2rig_v3_034/`。

本批次完成时间与物理机制的相应验收，未把它替代其余Plan算法、形态覆盖和SOP契约验收。

## 0.3.5 求解器、接触与相机校准批次（2026-10-27 Asia/Taipei）

全Plan保持进行中，本批次继续补齐完成度复核中的生产差异与专项证据。

- `flat_feet=120` 改为独立于Reader侧点的单位横向向量，具备全链路解析雅可比；按当前像素/前帧深度固定每帧接触集合。无表面槽位、趾部失读的落地反例中，roll从.7rad回到<.001rad；前后腾空仍保留.7rad，像素中位误差<.001。正立蹄不被当作平掌。
- 稳定轨迹改按每台标定相机的投影身高2%与三帧置信度>=.6判定；其Cauchy尺度6→12px。可信平滑面板面对抖动矛盾读数时，根位置误差从均值>.2降到最大<.04；双倍分辨率判定相同。35帧完全遮挡保持已有有限姿态，复现读数后重新捕获。
- 180°转向夹具实际形成基础prev/read的局部陷阱：中位像素误差>8，全部五个转身起点运行，胜出目标代价低于基础的0.1%，最终<.001px。reset恢复局部关节并保留主体世界运动；报告增加每起点代价、逃逸前误差、稳定点数和着地脚掌。
- 躯干胶囊和地面改按3000/H施加约束；胶囊中心线的距离不可微时，选取实际可运动的确定性逃逸方向。完整像素拟合覆盖长肢两端在外、中点穿过正中心的退化场景，约束点残余深度<.01%H；地下高置信读数也不能拉低实体。保留原数值雅可比回归。
- 远侧降权加入当前成对肢根的相机深度依据，避免把近侧肢体和躯干的低置信读数一起削弱；验证转身后近远身份反转、.12H及置信边界、主视角不可确认时保留。透视身高使用焦距与身体深度，独立Prop不进入身体中心。
- 已知非对称3D网格的首帧相机偏移，经实际光栅、双向Chamfer+IoU优化恢复：正交/透视IoU从<.70升到>.99，顶点平均投影误差<1px。正交tz是不可观测自由度，保持输入并报告；已对齐相机无漂移，空轮廓失败。5016面测试确认不再因超过5000面而退化成稀疏顶点轮廓。
- 卡限诊断覆盖25%/26%帧、2°包含边界、铰链排除；用户审核指纹消费仍在后续验收范围。

完整 `MONSTER_TEST_BLENDER=... npm run monster:check`：**169项全部通过、零跳过**，含真实Blender、CPU训练/重载、Notebook/API和P1–P4。`npm test`：**73项通过**。历史2K fast：**9个golden-frame全部PASS**，产物 `var/monster/evals/runs/20261026_173752_fantasy_creature_2k_v2_fast_taxonomy_kind_v2/`。ruff关键检查与diff空白检查通过。

已打包 `monster-0.3.5.tar.gz`（76个发布文件）、build、distribution并同步实际runtime。

- 当前构建 **`v0.1.14-20261027013938`**；真实已安装配置建立的MCP资源与HTTP一致，27工具，MCP HTML SHA-256 `1507f6b9bd41d3ac8426568afe4c69647d29dd26907525f8440d07c076447b3e`。
- runtime的27个Python Worker逐一哈希相同；从实际runtime导入0.3.5、方向雅可比与标定身高方法成功。Node常驻服务代码本轮未改；后续Python任务使用新子进程加载。未操作远端云实例。
- 本次Chrome主窗口的两个既有工作台标签331867569和331867572均点击版本完整刷新，可见新构建；未另开验证页替代旧页。主窗口截图、另一页AX证据、日志和运行验证在 `var/monster/verification/video2rig_v3_035/`。
- 项目登记仍为2个项目，前后哈希 `3517552ab1f6836d1fe979c2b263f156f0faa5236624f2a415323b731ba55392`。备份 `backups/resources-1793036398549`。

剩余项继续按完成度复核执行；本批次不代表七阶段全部验收，也不替代真实角色的训练质量和用户关卡审批。

## 0.3.6 几何铰链与LBS安全域批次（2026-10-27 Asia/Taipei）

原Plan继续实施。本批次补齐P3的几何轴角色和单向铰链，并修复因此发现的p0配置回读差异。

- 新 `axis_roles` 在最终恢复的p0中测量每骨bend/twist/side；共享解析世界关节轴与方向雅可比。铰链优先按静息弯曲平面选择X/Z，近直时按轴角色，固定其余Euler分量解析搜索straight；不再固定X或按半度网格量化。
- 正/反腿×X/Z滚转×四世界朝向的组合覆盖、非零p0、两侧直肘、12°两侧与精确边界、显式空交集全部直接验证。>12°沿当前继续折叠方向，其余按角色正面膝向后/肘向前；完整2.75rad有向区间与显式范围相交。
- 探针按实际铰链轴扩展搜索域并测量LBS，而不是在X轴套用旧默认。首次15°拉伸已失败时，从0与失败角之间细分8次、向安全侧取整；短边错权重网格实测安全角<5°，角度再加.2°即失败，绝不把失败的15°声称为安全。配置生成/合并保留几何与实测安全域的交集及精度。
- 真实Blender夹具增加带roll的UpperArm/LowerArm、局部旋转和独立网格混合权重。准备候选后以实际Blender父子向量核对铰链起始弯曲；关节配置生成和读回后，再逐骨比较p0世界位置/矩阵，保留原先非零Hips位移、重绑表面、只读导出及GLB回读。
- 该原生反例揭露配置把未观察叶骨扭转归零的问题：现在 `frozen_start_channels_rad` 保留已有角度且上下限相等，不新增优化自由度；配置合并、重新探测、实际合成标签均保持相同值。辅助几何测量从保存的世界p0读取，不依赖蒙皮质量尚未确定的组织根通道分配。

完整 `MONSTER_TEST_BLENDER=... npm run monster:check`：**178项全部通过、零跳过**，含真实Blender、训练/重载、Notebook/API与P1–P4；`npm test`：**73项通过**；历史2K fast：**9个golden-frame全部PASS**。历史产物 `var/monster/evals/runs/20261026_181016_fantasy_creature_2k_v2_fast_taxonomy_kind_v2/`。ruff关键检查和diff空白检查通过。

已打包 `monster-0.3.6.tar.gz`（76个发布文件）、build、distribution并同步实际runtime。

- 构建 **`v0.1.14-20261027021201`**；真实已安装MCP资源与HTTP一致，27工具，MCP HTML SHA-256 `c54ea49ff3b5055816acf6cffc1166678a305613e38a7149b22e36fb531a2aaa`。
- runtime的27个Python Worker逐一哈希相同；从实际runtime导入0.3.6并执行rolled-Z铰链及非零叶扭转锁定样例成功，证据 `worker-import.json`。本批未修改常驻Node逻辑，后续Python子进程加载新源码；远端Colab实例未操作。
- 此前两个工作台页均已关闭，Chrome清单只剩一个ChatGPT页；已在主窗口重新打开331867591并点版本完整刷新，实际可见新构建。不是声称刷新了已关闭的旧页，也未以聊天内嵌页替代主窗口。截图及日志在 `var/monster/verification/video2rig_v3_036/`。
- 项目登记前后保持2个项目，哈希 `3517552ab1f6836d1fe979c2b263f156f0faa5236624f2a415323b731ba55392`；备份 `backups/resources-1793038362993`。

仍有其余Plan机制和门禁专项未完成，以完成度复核继续推进；本批的原生夹具和单测不代替真实角色审批或云端训练识别质量验收。

## 0.3.7 骨架树清理与权重边界批次（2026-10-27 Asia/Taipei）

原Plan继续实施，本批次完成指趾完整树、壳外比例与尖端、武器薄末段、头部残枝、鞋底/对称连接和碎片权重的专项修正。

- 默认保留完整指趾树；显式折叠按整分支最大弧长严格<.22H，而非逐骨节。真实Blender候选保留长指，删除短指/趾及其顶点组，.3+.7权重合并至掌骨1.0，附件重新挂接。
- 壳外比例使用自适应偶数中点样本，严格>50%才删除；回缩尖端二分寻找体素边界，保留至少30%原长。连续薄末段严格>.10H且满足两个厚度上限才截武器，避免短爪误判；实际闭合网格生成Weapon刚性子骨且刀身远端权重1.0。
- 头骨头尾归中，镜像配对核对两端；短单侧head残枝剔除，成对耳/长角/flex耳保留。鞋底97%表面按最近脚段归属；完整左右树在实际体积IoU>=.80时镜像平均，非对称链长比>=.75时保留方向拉齐长度并平移后代保持连接。
- 碎片<2%按整个角色全部网格统计，独立Prop不进入分母或角色候选。实际Blender验证精确2%保持、低于2%按实测骨段距离1/d⁴归一化、远侧串权重移除、独立道具自身1.0。

完整 `MONSTER_TEST_BLENDER=... npm run monster:check`：**188项通过、零跳过**，含原生Blender、CPU训练重载、Notebook/API与P1–P4。`npm test`：**73项通过**；历史2K fast：**9个golden-frame全部PASS**，产物 `var/monster/evals/runs/20261026_183403_fantasy_creature_2k_v2_fast_taxonomy_kind_v2/`。

已打包 `monster-0.3.7.tar.gz`、build、distribution并同步实际runtime。

- 构建 **`v0.1.14-20261027023722`**；已安装配置建立的真实MCP资源与HTTP一致，27工具，MCP HTML SHA-256 `f1de21d08ed2ec3dccc4c3eb950b95127db65bb02ca72d41d9dd579dfe314b20`。runtime的27个Python Worker逐一哈希相同；Python任务按子进程加载新源码，常驻Node逻辑未修改。
- Chrome扩展连接已不可用且此前标签已关闭。本次通过原生Chrome主窗口重新打开HTTP工作台，实际可见新标识，截图保存于 `var/monster/verification/video2rig_v3_037/main-workbench.png`。此为实际主浏览器，不以聊天内嵌验证页代替。
- 项目登记仍为2个项目，前后哈希 `3517552ab1f6836d1fe979c2b263f156f0faa5236624f2a415323b731ba55392`；备份 `backups/resources-1793040694349`。测试日志、登记快照与运行验证保存在同一证据目录。

剩余机制及专项验收以完成度复核为准，本批次不证明整个七阶段完成。

## 0.3.8 表皮点与非方形相机批次（2026-10-27 Asia/Taipei）

原Plan未完成。本批发布范围是表皮标记、骨名索引与相机换算修正；发布之后出现的同目录并发迁移是单独的未发布工作树，不能混用测试结果。

- 表皮点与躯干胶囊按skin.bone_names映射实际权重索引；请求槽位可重排。叶骨按当前parent关系判定，修复缺失/陈旧children缓存造成的虚假.tip。严格w>.3、四点最小支持、98%轴向叶尖、主次轴1/99%与Prop端点/中点均有直接断言。2N头尾槽位保持在前。
- 原生Blender夹具使用非零p0、roll、世界旋转和两个实际蒙皮网格，测得20个原始端点、38个附加点；FK和生产Reader标签与实际骨架矩阵一致。横/竖画幅×三种sensor fit×正交/透视共12种含shift配置，经Camera导出JSON再解析，与world_to_camera_view逐点比较，最大误差`5.824e-5px`。
- 修正AUTO/HORIZONTAL正交相机的垂直世界尺度；AUTO竖图透视仍使用sensor_width，只有显式VERTICAL使用sensor_height。实际runtime解释器已加载并执行新函数，记录`worker-import.json`。
- 两关节像素求解反例证明：头尾不能观察手掌自身roll，加入侧点后同时恢复父骨扭转与末端roll，参数误差<1e-4rad、像素误差<.001。
- 特征版本`v3-anatomical-reader-4`改变数据签名，避免复用错误表皮点缓存。已有权重仍使用其保存的槽位，不把旧训练结果当作新标签验收。

发布源码初次完整检查192项通过。随后同目录并发改写求解器、读数与训练管线，导致源码/发布包/一次同步内容不一致。没有覆盖工作树，已将runtime中混入的一个求解器文件保存备份后校正为固定的0.3.8发布包内容。再对该发布内容建立独立目录、运行同一完整check：**192项通过、零跳过**，语法、Notebook/API、原生Blender、CPU训练重载和P1–P4通过；日志`monster-v3-038-check-release.log`。发布的全部76个文件、27个Worker与该检查快照/runtime逐一哈希相同。Node **73项通过**、历史2K fast **9个golden-frame PASS**；历史产物`var/monster/evals/runs/20261026_190559_fantasy_creature_2k_v2_fast_taxonomy_kind_v2/`。

- 发布包SHA-256：`a6597e6bd84135c78c1a0079995b9e92155d23f39f5481811f1761e1e4192047`。逐文件哈希见`published-files-sha256.json`与`published-worker-sha256.json`。
- 已完成package、build、check:distribution和实际运行更新。构建**`v0.1.14-20261027030844`**，HTTP/MCP一致，27工具；MCP HTML SHA-256 `22d4dcf001e781725441ebd4dc0b721fbc5cdde4f112ac5ff29b09bed2f7d9ee`。
- 实际Chrome主窗口点击版本完整刷新后可见该标识，保留主窗口截图与AX；没有另开验证页替代。项目登记仍2个，哈希`3517552ab1f6836d1fe979c2b263f156f0faa5236624f2a415323b731ba55392`。常驻Node代码未修改，后续Python子进程加载发布内容；远端Colab未操作。同步备份`backups/resources-1793041795388`。

并发写入已只读定位至Antigravity的 **Human Differences Beyond Skeletal Structure** 会话`5434b36e-423d-4882-8eb0-f68562755905`，UI显示Working、Cancel及相同文件的连续Edited记录；未中断或向该会话发送消息。它与用户最初提供的73e5d078计划会话不同。活动证据`agy-activity.txt`。

后续工作树两次复跑分别出现地面回归、远侧置信度.5边界回归；本聊天已为前者添加仅低置信遮挡降权的修正与针对性通过测试，但这些后续内容**尚未发布**，不能宣称最新工作树全量通过或全部热更新。当前至少7个Worker和1个测试文件与发布快照不同，详见`pending-source-differences.json`及逐文件源码/patch快照。为避免继续交叉覆盖，停止对这些冲突源码写入和整体同步，等待明确唯一写入者；原Plan goal保持未完成。全部证据在`var/monster/verification/video2rig_v3_038/`，交接见`monster_video2rig_038_handoff.md`。
