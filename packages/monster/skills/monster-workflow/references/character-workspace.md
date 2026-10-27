# 角色独立工作目录规范 v1

状态：2026-10-26 确定的目录与数据归属规范。适用于 Open Game Studio 项目中的角色制作；Monster 流程沿用 `monster_id` 作为 `character_id`。本次制定规范及技能入口；现有 SOP/Worker 的路径解析、初始化、迁移和同步尚未全面实现 v1，实际兼容入口见 §8。文中的新增 JSON 是接口约定，尚无对应的自动校验器或新 CLI。

## 1. 一个角色，一个工作根目录

角色的本地制作根目录固定为 `<project_root>/scratch/work/<character_id>/`。`project_root` 是用户游戏项目根目录，不能使用插件源码、插件运行目录或终端当前目录代替。没有工作台登记的项目沿用用户指定的项目文件夹，不自动登记。

- `character_id` 是稳定身份，不用显示名、动作名或某一次训练的名称充当身份。新角色使用 `chr_<UUID的32位小写十六进制>`，历史 `monster_id` 原样保留。显示名可改，身份及根目录保持不变。
- 一个角色可以有多次制作 Run、多个模型/骨架版本、摄影棚配置、数据集、训练和动作。服装、装饰或武器改变时建立相关输入/绑定版本；出现新人物身份时才新建角色。
- 同一个项目内角色 ID 唯一；云端旧路径按 `monster_id` 隔离，跨项目遇到历史同名 ID 必须先核对来源及指纹，禁止直接复用。跨项目复用通过显式导入固定资源版本完成，记录来源项目与来源角色。
- 文件夹只在对应阶段需要时建立。初始化只建立入口文件与已知目录，尚未产出的数据不得用空文件或假指纹占位。
- 角色工作区属于用户项目数据，不放入 `packages/monster/`、插件缓存或安装 runtime。项目、任务、审核和资源登记继续使用现有 `.game-studio/studio.db`，不另建角色项目数据库。

## 2. 标准目录树与职责

以下树中的 `<…_id>` 是不可变版本/任务 ID；目录内的固定文件名便于工具读取，父级 ID 防止历史覆盖。

```text
<project_root>/
├── scratch/work/<character_id>/
│   ├── character.json                 # 身份、项目关联、当前固定输入及兼容路径索引
│   ├── source/<source_id>/            # 原始输入快照、来源和依赖；保存后只读
│   │   └── source_manifest.json
│   ├── rig/<rig_id>/                  # 一次候选批次、修整、完整装饰及选定绑定
│   │   ├── rig_manifest.json
│   │   ├── candidates/               # candidate_0..3 及各自依赖
│   │   ├── selected/                 # 完整模型、贴图、studio_rig_meta.json
│   │   └── limits/<limits_id>/joint_limits.json
│   ├── studio/<studio_id>/            # 摄影棚、视角排布、原生分辨率及首帧
│   │   ├── studio_manifest.json
│   │   ├── calibration.json
│   │   └── start/                    # idle_split_* 或 idle_3view_*，保持既有文件名
│   ├── training/
│   │   ├── datasets/<dataset_id>/     # 姿态、渲染图、标签、可见度、可选稠密标签
│   │   │   └── dataset_manifest.json
│   │   └── runs/<training_id>/        # 一次训练的配置、权重、指标、日志和代码版本
│   │       ├── training_manifest.json
│   │       └── checkpoints/dinov2_head.pt
│   ├── clips/<action_id>/<clip_id>/   # 每次视频生成或外部视频导入的原始证据
│   │   ├── clip_manifest.json
│   │   ├── prompt.txt
│   │   ├── request.json              # 脱敏生成参数和已提交任务 ID
│   │   └── input.mp4
│   ├── actions/<action_id>/<solve_id>/ # 同一视频的一次读取/重建/烘焙尝试
│   │   ├── solve_manifest.json
│   │   ├── observations/             # 二维点、置信度、深度、可见度等原始预测
│   │   ├── reconstruction/           # 三维轨迹、求解配置、接触事件及诊断
│   │   └── output/                   # 烘焙 blend/GLB、回读验证与对照视频
│   ├── exams/<exam_id>/              # G3/独立评测的输入引用、GT边界和结果
│   │   └── exam_manifest.json
│   ├── reviews/<gate>/<review_id>/   # 固定版本的预检、G1/G2/G3/逐片审阅证据
│   │   ├── review_manifest.json
│   │   └── index.html
│   ├── deliveries/<batch_id>/        # 本批固定动作版本和 batch_manifest.json
│   ├── cache/<dependency_signature>/ # 可重建的解码帧、临时渲染/计算缓存
│   └── logs/<job_id>/                # 阶段任务日志、进度和恢复记录
├── scratch/checks/<character_id>/   # 现有 SOP 审阅产物；兼容期保留原位
└── .game-studio/
    ├── monster/<character_id>/reviews.json  # 现有 Monster 审核决定账本
    ├── studio.db                    # 现有项目/资源/Run/审核登记
    ├── assets/                      # 工作台管理的不可变导入版本
    └── exports/                     # 工作台管理的固定版本导出
```

路径按内容职责划分，不再另建 `stage1/phase2/最终版/最终版2/` 等平行树。`rig_id` 包含一次完整候选批次；G1 选择在该批次内固定候选。模型或蒙皮实质变化产生新 `rig_id`；只改变关节范围产生新 `limits_id`，保留绑定版本。`action_id` 使用稳定的小写动作键，如 `walk`、`slash_01`；循环/单次、帧率和动作意图放入配置。版本/任务 ID 使用类别前缀加 UUID，路径只含 ASCII 小写字母、数字、下划线和连字符。

`source/` 保存实际消费输入的快照及来源。工作台已有不可变资源可按固定 `project_id + asset_id + revision_id + 文件索引 + SHA-256` 引用，离线交接再收集依赖。外部路径不能被当作随时存在的隐含依赖。贴图和装饰跟随所属输入/绑定版本，不建立可被任意修改的全角色公共贴图目录。

`clips/` 的原视频是制作证据，不是缓存。G3 可引用这里同一固定视频，无需再复制。`exams/` 与生产动作隔离，保留测试用途及训练/验证独立性；训练图不能冒充独立考试视频。

## 3. character.json 是入口索引

入口只保存身份、引用和当前选择；几何、骨骼属性、关节范围和相机参数分别由既有专业文件管理。`character.json` 不复制另一份骨骼配置，也不保存一套平行审核状态。

最小示例（ID 仅示意，实际创建时生成）：

```json
{
  "schema": "game-studio-character-workspace-v1",
  "character_id": "chr_0123456789abcdef0123456789abcdef",
  "monster_id": "chr_0123456789abcdef0123456789abcdef",
  "display_name": "棘甲猎手",
  "project_ref": null,
  "layout": "v1",
  "revision": 1,
  "created_at": "2026-10-26T20:00:00+08:00",
  "active": {
    "source_id": null,
    "rig_id": null,
    "limits_id": null,
    "studio_id": null,
    "dataset_id": null,
    "training_id": null
  },
  "action_selections": {},
  "compatibility_paths": {}
}
```

- `project_ref` 为已存在工作台项目的 `{ "project_id": "实际ID" }`；未登记项目为 `null`。`monster_id` 仅 Monster 路线填写，值与 `character_id` 相同；其他路线省略。通用角色 DAG 的 `run_id` 写入各任务 manifest，不能从当前显示名推测角色归属，现有 Run 无角色映射时需显式关联。
- `active` 只表示当前选择，不能证明 G1/G2/G3 通过。初始化的未知项为 `null`；`action_selections` 按动作固定 `clip_id`、`solve_id`。每次执行先将这些引用解析为固定路径、大小与 SHA-256，写入本次 manifest，再开始计算。入口每次成功更新递增 `revision`，写入时校验预期修订号。
- v1 内部文件引用统一使用 `workspace://rig/<rig_id>/…` 等角色根相对逻辑 URI；`project://` 相对游戏项目根；`drive://` 相对本次选定的 `MONSTER_DRIVE_ROOT`；`external://<ref_id>` 必须在对应 manifest 中以位置、指纹及恢复方式登记。它们是设计中的逻辑引用，现有 CLI 尚不会自动解析。
- 从 URI 解码后校验归属与真实路径；拒绝 `..`、绝对路径和逃出声明根的符号链接。历史绝对路径作为 `compatibility_paths` 的机器绑定记录保留，不能伪装成可移植引用。
- 配置使用无 BOM UTF-8、JSON 数字及布尔值；时间包含时区，展示与文件中约定的日期使用 Asia/Taipei。目录版本身份不依赖时间戳，时间只用于记录。

## 4. 版本清单、依赖与恢复

各阶段 manifest 至少包含 `schema`、`character_id`、所属项目引用、阶段 ID、可选 `run_id/job_id`、父版本引用、实际输入/输出文件清单、参数快照、代码版本、创建时间及 `status`。文件记录含逻辑 URI、字节数、SHA-256；只有实际存在并核验的文件才写入输出。外部生成器额外保存 provider、模型标识、任务 ID、seed（服务未返回则 `null`）和脱敏原请求；接收未知的任务恢复原 ID，不用同名文件推断成功或重付费提交。

规范状态为 `preparing → running → completed|failed|cancelled`；中断且结果待查询时使用 `unknown`。用户审核状态仍由现有审核系统提供，不能把 `completed` 写成 `approved`。这个状态约定用于新增 manifest，不要求改写旧文件的状态字段。

每次新任务先新建 ID 和输出目录，将输入及配置锁定。进行中的进度可更新；`completed` 后该版本的输入、配置、产物和清单冻结。重算、返修、不同 seed 或不同参数建立新 ID，并用 `parent_id`/`supersedes` 关联历史。文件写入先落同文件系统临时路径，核验成功后原子发布，最后更新入口选择；取消、失败、未知结果均保留证据且不切换当前有效选择。

依赖链必须能还原为：

```text
source → rig + limits + studio → dataset → training/checkpoint
studio/start + prompt/generator → clip
rig + limits + studio + checkpoint + clip + solver/config → solve
固定 solve + 实际回读证据 → review → delivery → 工作台固定资源版本/导出
```

训练清单记录实际 backbone、冻结/微调方案、输入尺度、热图/辅助头、feature flag、数据集签名、训练参数及依赖环境。求解清单记录观测缓存签名、求解器版本、先验、接触事件、烘焙设置；不得只用一个可覆盖的 `latest.pt` 或动作名绑定依赖。

缓存签名覆盖实际消费的模型/蒙皮/装饰、相机/排布/分辨率、标签定义、数据/权重、预处理、代码与相关参数。仅凭路径相同或文件存在不能复用。可分层复用解码、观测和求解缓存；更换求解器不必因此重生成视频。具体关卡失效范围由 [审核与回退规范](reviews.md) 决定，新增目录不能改变其含义。

并行写入按角色的配置更新和版本选择设置单写者/锁，耗时任务使用各自 job 输出目录。写入前核对入口版本，入口变动时报告冲突并合并不相交修改，禁止后写者抹掉其他动作选择或审核记录。命名不同不等于已经实现了并发锁。

## 5. 审阅、项目登记与正式交付

`reviews/<gate>/<review_id>/` 存放固定证据包：HTML、图、视频、报告及其哈希。`gate` 包括 `preflight`、`G1`、`G2`、`G3`、`final_batch`，使用现有门禁名称。兼容期旧 `g1_review.html` 等文件名保持；新目录可用 `index.html`，引用必须使用实际存在的入口。

- 用户决定和审批历史由现有 `.game-studio/monster/<id>/reviews.json` 及既有工作台审核记录各按其流程管理；两种流程之间没有自动审批互通。角色入口只引用，不建立第二个可写 `approved` 账本。
- Drive 的 `stage_state.json` 保持 Worker 阶段字段；其中 `sop_ledger` 沿用现有同步约定，不视为另一个独立审批源。发现本地/云端冲突时保留两份和版本来源，先核对再继续。
- 审阅路径或文件布局变化也可能使旧绝对路径/指纹引用失效；迁移时验证引用解析，不因内容看似相同而复制“通过”。
- Monster 全批最终验收前，中间产物留在角色工作区及既有 `scratch/checks`；遵守现有 `art_source/`、受管清单和 Git/LFS/DVC 红线。
- 最终入库调用工作台已有导入入口，得到 `.game-studio/assets/` 下的不可变 revision，登记 `project_id/asset_id/revision_id` 及输入来源。受管 `art_source/` 与引擎目录按项目已有规则发布，不把角色过程目录塞进去，也不直接修改工作台 DB。
- `deliveries/<batch_id>/batch_manifest.json` 只列本批固定动作版本和验证证据，记录最终导入/导出引用。入库本身不替代最终用户验收，也不证明引擎验收。

## 6. 本地、Drive 与 Colab 的映射

本地角色目录是制作入口；Drive 是经校验的持久归档；Colab 是可丢失的计算副本。保持相同 `character_id` 和版本 ID，不以执行机器创建新角色。

| 位置 | 路径及约束 |
| --- | --- |
| 本地 v1 工作根 | `<project>/scratch/work/<character_id>/` |
| 现有 Drive | `<MONSTER_DRIVE_ROOT>/monsters/<monster_id>/`，保留 `u1/`、当前模型/权重及 `stage_state.json` 的兼容入口 |
| 新版 Drive 内容归档（待同步器支持） | 上述角色根下 `workspace/`，内部对应 v1 相对目录，入口 `workspace/character.json` |
| 现有 Colab | 沿用 Worker 的 `WORK_ROOT/work/<monster_id>/…` 及当前阶段目录；新 job 按清单显式映射输入输出 |
| 共享运行环境 | 模型基础权重、Python/CUDA 环境、ComfyUI/H3、SkinTokens 缓存仍在已有共享位置，以版本/哈希引用 |

v1 不把公共模型、虚拟环境或 GPU 租约复制进每个角色。Worker endpoint、认证密码和 API key 留在既有私密运行配置，任务清单只引用连接 ID，日志/导出包脱敏。GPU 生命周期继续执行 `colab-guard`，目录整理不启动、重启或释放云端实例。

传输遵守 U1/U2：模型变更才上传模型；仅配置、提示词、seed 或求解器变化时按实际增量同步。上传用任务独立暂存位置，按清单核验字节数及 SHA-256，校验完成后发布清单并记录真实 Drive 文件 ID。DriveFS 上不假定跨目录重命名具有本地文件系统同等原子性；仅完成清单可被下游消费，未完成对象可恢复续传。

同步按文件归属单向发布版本/归档结果；多端更新 `character.json` 先核对修订号和输入指纹。禁止整个角色目录做无条件双向覆盖，避免旧 checkpoint、旧选择或旧阶段状态覆盖新结果。

## 7. 保留、清理与可移植交接

原始输入、首帧/相机、原视频、训练标签与数据集清单、checkpoint、固定动作、审核证据及交付清单均属持久材料。训练渲染图仅在种子、实际代码、完整配置和环境可复现且已有校验归档时，才可按明确清理策略移出本地；默认保留。

只有 `cache/` 内经 manifest 标明可重建、且未被审核证据或当前任务唯一引用的文件可清理。日志按任务保留，历史失败和旧版本不能用删除表达失效。清理列出影响与恢复来源，使用可恢复方式，不触碰原工程和正式资源。

交接以 `character.json` 为入口，沿当前固定版本及指定历史版本收集依赖，附审核账本的只读快照、缺失/外部依赖列表、全部文件指纹和恢复方法。恢复后逐一校验；权重、模型、相机或视频缺失时明确报告。仅复制一个含外部引用的文件夹不宣称完全自包含。新机器解析机器绑定路径并更新位置映射，资产 ID 和内容哈希保持不变。

## 8. 当前兼容路径与实施顺序

当前脚本仍读取下表中的位置。执行现有 SOP 时沿用这些路径；可新增入口索引记录其归属，但不能先搬到新树再期待旧脚本自动发现。

| 现有实际位置 | v1 对应职责 |
| --- | --- |
| `scratch/work/<id>/rig_input_variant.glb`、`decoration_manifest.json`、`decorations_backup.blend` | 输入预检/装饰快照，归属 `source/<source_id>` 或相应 rig 准备版本 |
| `scratch/work/<id>/candidates/`、`g1_candidates/` | `rig/<rig_id>/candidates` 及选定完整绑定 |
| `scratch/work/<id>/joint_limits.json` | `rig/<rig_id>/limits/<limits_id>/joint_limits.json` |
| `scratch/checks/<id>/preflight_report.json`、`g1_review/`、`g2_review/`、`g3_review/`、`delivery_review/` | 各 gate 的固定审阅包 |
| `.game-studio/monster/<id>/reviews.json` | 既有审核权威位置，保留不搬 |
| Drive `monsters/<id>/rigged_pruned.blend`、`studio_rig_meta.json`、`dinov2_head.pt`、`animations/` | 已核验版本的 Worker 兼容入口，不能当作不可变历史档案 |
| Colab `WORK_ROOT/work/<id>/actions/<action>/` | 对应 `clip_id/solve_id` 的临时计算目录 |
| 仓库 `var/monster/runs/<run_id>/` 及已有大型评测归档 | 保留现存位置并登记角色、run 及来源；插件源码目录不再新增用户角色生产档案 |
| `.game-studio/generated/`、`.game-studio/character-dag/` 和通用角色 DAG Run | 沿用任务管理位置；明确关联角色和输入/输出 revision，不复制另一套 Run 状态 |

实施按三阶段推进，各阶段完成证据独立报告：

1. **规范接入（本次）**：发布本规范，加入技能、项目说明及运行目录；不修改已运行任务和历史文件位置。
2. **路径适配（后续实现）**：增加统一角色路径解析器和入口/manifest 校验，SOP、Worker、客户端都通过它解析；同时兼容旧布局。生成 ID、冲突检查、签名缓存、传输与单写锁完成前，不能宣称 v1 已自动执行。
3. **逐角色迁移（单独任务）**：先盘点本地/Drive/正在运行任务及引用，制作 dry-run 映射和备份；任务结束后复制、核验、回读审核与下游路径，再切换兼容入口。保留旧目录及回滚映射；用真实任务验证成功后才讨论清理。既有用户批准只在原依赖指纹一致且可追溯时保留。

路径适配验收覆盖：两个角色隔离、同角色两次训练/两个动作/同视频两次求解共存、入口切换不改历史、失败任务不污染当前选择、配置改变时缓存失效、未知生成任务按原 ID 恢复、并发更新冲突、旧 SOP 路径兼容、移动项目后的依赖解析、Drive 中断恢复、审核状态保持及实际 GLB 回读。新路径支持与云端部署须分别验证。
