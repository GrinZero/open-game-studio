# 初始操作契约

HTTP 为 `POST /api/tools/<tool>`，JSON 参数。stdio MCP 注册相同 26 个领域工具，另有 1 个 app 专用桥接工具，输入 schema 由 `src/server/operations.ts` 提供。所有项目内变更显式携带 projectId；浏览器上传是 `/api/projects/<projectId>/import` multipart。

项目表单通过 `POST /api/directories/pick` 请求本机文件夹选择，参数 `{ "mode": "create" | "link", "initialPath"?: "/absolute/path" }`。返回 `{ "path": "/selected/directory" }`，取消返回 `{ "path": null }`；只选择位置，不登记或创建项目。新建时 UI 默认直接使用所选目录；只有勾选「按项目名称创建子文件夹」时才追加项目文件夹名。表单显示最终路径，并将其作为 `root` 调用 `project_manage`。`create` 支持已有文件夹，保留已有文件；尚未存在的目录会创建。已经登记的目录返回原项目，不重置画布或项目名称。此接口同样由 app 专用 `studio_ui_request` 桥接，不增加模型可调用的领域工具。

项目封面使用 `project_manage` 更新：`{ "action": "update", "projectId": "p_...", "cover": { "revisionId": "rev_...", "fileIndex": 0 } }`。`fileIndex` 是该版本文件集中的零起始索引；必须引用本项目内的图片文件，保存时校验文件存在与 SHA256。传 `"cover": null` 移除封面，不传则保留当前封面。封面固定到所选版本，不随资源最新版本更新。上传新封面复用资源导入，再设置返回的图片版本；内嵌 UI 复用同一 MCP 桥接与 blob 图片预览。

## 完整工具索引

参数 schema 的事实源为 `src/server/operations.ts`；界面桥接为 `src/server/ui.ts`。

| 类别 | 工具 |
| --- | --- |
| 工作台与项目 | `studio_open`, `project_manage`, `project_open` |
| 资源 | `asset_search`, `asset_import`, `asset_manage`, `asset_inspect` |
| 文档 | `document_save` |
| 资源制作 | `generation_capabilities`, `generation_create`, `generation_get`, `generation_start`, `generation_control`, `generation_adopt`, `generation_complete` |
| 画布与上下文 | `canvas_open`, `canvas_manage`, `canvas_patch`, `selection_context` |
| 角色流程 | `workflow_prepare`, `workflow_start`, `workflow_get`, `workflow_decide`, `workflow_submit_artifact`, `job_control` |
| 导出 | `export_bundle` |
| 界面专用 | `studio_ui_request` |

## 画布

创建便签或资源引用：

```json
{
  "projectId": "p_...",
  "canvasId": "c_...",
  "target": "graph",
  "expectedRevision": 0,
  "ops": [
    {
      "op": "add",
      "node": {
        "id": "n_unique",
        "type": "note",
        "position": { "x": 80, "y": 80 },
        "data": { "title": "制作便签", "text": "保留轮廓" }
      }
    }
  ]
}
```

资源节点 type=asset，data 内包含匹配 assetId 与 revisionId。workflow 类型不通过 add 创建，而通过 instantiate：

```json
{
  "projectId": "p_...",
  "canvasId": "c_...",
  "target": "graph",
  "expectedRevision": 1,
  "ops": [
    {
      "op": "instantiate",
      "inputRevisionIds": ["rev_..."],
      "position": { "x": 400, "y": 80 }
    }
  ]
}
```

布局独立提交：

```json
{
  "projectId": "p_...",
  "canvasId": "c_...",
  "target": "layout",
  "expectedRevision": 0,
  "ops": [
    { "op": "move", "id": "n_unique", "position": { "x": 100, "y": 220 } }
  ]
}
```

其他 graph 操作：remove 的 id；note 的 id 与 title/text（至少一个），title 去除首尾空白后为 1–120 字，省略字段保留原值；connect 的 edge={id,source,target,kind:"reference"}；disconnect 的 id。reference 连线表示 source 向 target 提供制作输入，递归收集上游固定版本和便签；文档与便签作为文字输入，其他资产作为参考。禁止自连、循环、重复端点和制作分组作为 source（需选择明确产出版本）。连线不自动启动生成。布局 viewport 使用 viewport={x,y,zoom}。冲突返回 HTTP 409，调用方重新读取并合并意图。

## 制作

workflow_prepare 返回 planId（返回对象 id）、固定图版本、输入与 manual_import 路线。workflow_start 使用该计划、expectedRevision 与 idempotencyKey，返回持久 Run。相同幂等键只返回同一个 Run；在不同计划上复用键会被拒绝。

workflow_submit_artifact 通过 runId / stepKey / revisionIds / expectedRevision 提交已导入产物。brief 需要文档，reference 需要图片，candidate 需要 GLB，assembly/rig 每个版本都需要同一文件集的 GLB 与 Blend。此检查不证明 Blend 可编辑性。

workflow_decide 在 appearance 或 deformation 阶段绑定 revisionId、当前 expectedRevision、accept/request_changes 和 comment。actor 为 user 或 delegated_agent。API 不替 Agent 判断用户授权范围，包内 Skills 要求明确已有授权。反馈的相机、部位和时间点目前保存在 comment；独立 evidenceRefs 结构需下一版扩展。

job_control 当前操作的是本地 Run，cancel 保留成果，rerun 从指定产出阶段建立新 Run。传 fromStep / expectedRevision / idempotencyKey，可选 inputRevisionIds。这个工具不宣称已存在 Provider Job。

export_bundle 通过固定 revisionIds 和幂等键生成独立目录。带 runId 时还需 expectedRevision，且版本必须严格匹配该 Run 的已审核交接输入。manifest 记录文件清单、hash、审核与验证范围；engine 始终 not_verified。

## 读取与能力

studio_open 返回项目、标准化的 `target`、深链接 URL 与真实能力状态。传 `projectId` 或当前会话 `workspaceRoot` 定位项目；可传 `view`（overview/assets/canvas/character/runs/settings）、`canvasId`、`nodeId`、`assetId` / `revisionId`、`runId` + `nodeKey`（DAG 步骤）或 `stepKey`（制作阶段）、`taskId`。节点、运行和任务可推导所属画布；固定版本可推导资源。冲突、缺失、跨项目或归档目标明确报错，不回退到无关对象。资源打开详情与固定版本，节点选中并居中，DAG 步骤切换运行和阶段，制作阶段展开并滚动，任务打开详情且不启动制作。HTTP URL、MCP 首次输入、后续结果和内嵌页面刷新使用同一目标解析；每次打开返回独立 `navigationId`，允许重新定位同一目标。`navigationStatus=requested` 仅证明目标已解析并发出，不证明宿主的原窗口已经显示；Agent 必须检查实际页面后才能报告“已打开”。`canvas_open`、`project_open` 仍只读取数据。工作区匹配已登记项目的最近根目录，忽略归档或不可用项目；多个工作区有歧义时保留项目列表，不使用插件安装目录或全局最近项目。

`studio_open` 已绑定 `ui://game-studio/workbench` MCP App 资源及 global/thread 入口。Codex 直接打开入口或界面初始化时，优先按调用 `_meta.threadId` 只读查询本地 `CODEX_HOME/state_*.sqlite` 中该会话的 `cwd`；只查准确会话，不读取聊天正文、不改数据库、不以最近聊天或 MCP 进程环境会话代替调用身份。宿主不提供原生会话 ID 或本地状态不可用时回退到标准 MCP roots；显式 `projectId` / `workspaceRoot` 优先。`studio_ui_request` 仅供界面调用，负责项目数据、导入与媒体桥接。原生宿主是否显示入口仍取决于宿主支持，不能从 manifest 存在推断当前宿主表现。

## 连线输入与冻结（2026-10-10）

instantiate 支持 inputNodeIds（资产 / 便签）和 inputRevisionIds。画布中存在对应版本节点时创建输入连线；没有对应节点的版本保留为显式固定输入。旧制作分组的显式输入保持兼容，不随删除新连线移除。

workflow_prepare 根据制作分组的入边递归解析输入，最多 40 项、20 个资产版本，按固定 revisionId 去重。inputSnapshot 含 graphRevision / targetNodeIds / revisionIds / items / prompt。items 包含标题、节点、类型、文字用途、资产版本、文件和便签快照。文档内容从实际固定版本文件读取，每个文本文件提供前 30,000 字符，截断时 textTruncated=true，完整文件可通过 fileRef 读取。准备和启动均校验文件存在、项目边界与 SHA256。没有输入时拒绝准备。

workflow_start 必须使用准备时的 graphRevision；便签、连线等图修改使旧计划失效，布局移动不会。运行持久化 inputSnapshot；后续断线、便签改动、新版本导入不会改变已冻结输入。workflow_get 的 executionInputs 返回 projectRoot / prompt / notes / referenceFiles / stepInputFiles，每个文件含固定 revisionId、版本、哈希和实际 fileRef。后续阶段可继续读取原始制作上下文；阶段成果输入仍按既有审核流程推进，不以参考连线绕过审核。

局部重跑保持原快照；从 brief 阶段显式替换 inputRevisionIds 时，替换资产输入并保留原冻结便签，不重新读取当前画布便签。

## 资源创建与制作（2026-10-11）

`document_save` 创建或编辑 Markdown 文档，参数为 projectId/canvasId/expectedRevision/name/text/position。编辑时还要传 nodeId/assetId/expectedLatestRevisionId。资产新版本与指定节点的更新在同一事务提交；其他节点保持旧固定版本。冲突不覆盖文档或画布。

`generation_create` 支持 image/video/model/animation/audio，要求 name/prompt/position/idempotencyKey 和当前画布 expectedRevision，可指定 inputNodeIds 与 settings（aspectRatio、duration、animation、transparent）。创建草稿任务及 generation 节点，冻结上游版本、文档和文字，并校验实际文件哈希。动画必须包含模型输入。相同幂等键只能复用同一组输入；未完成 generation 节点不能作为上游或复制。

`generation_start` 一次性领取 draft，通过本机 Codex 执行制作。执行器路径来自 STUDIO_CODEX_BIN，默认 /opt/homebrew/bin/codex；使用现有 Codex 登录和已配置工具。任务设置是制作要求，具体支持范围、登录、参数与付费确认由制作工具检查。工具缺失或需要用户操作时记录失败原因，不伪造素材，不自动确认付费提交。执行目录为项目 .game-studio/generated/taskId，保存 worker.log、输出 JSON 与工具返回的真实提交记录。重复点击、断线或重启不重复启动远端生成。

`generation_get` 返回任务与实际冻结参考路径。`generation_complete` 验证真实文件的类型后，在同一事务入库并将原 generation 节点转换成固定版本 asset，保留输入连线。动画 GLB 须包含 skins 和 animations；此结构校验不代表变形质量通过。若任务节点被移除，产物仍保留在资源库。

`generation_adopt` 将已导入固定版本回填到 draft/failed/interrupted/running 任务，用于人工制作或原远端任务的恢复；动画同样验证骨架和片段。`generation_control` 停止本地执行并保留历史，不能撤销已提交的远端工作。服务会恢复已写入的执行结果，失去执行器的任务标记中断；无自动重提。

此路线是 Codex 工具编排，不是内置 BYOK Provider API。服务没有承诺每种生成工具已安装、已登录或有免费额度。仅安装 Codex 二进制不证明所有模型服务可用。

节点尺寸属于 layout：`{ "op": "resize", "id": "node_id", "width": 480, "height": 320 }`。宽度为 160–4000，高度为 120–4000 的有限数字；同时传入 `width: null, height: null` 恢复默认大小。缩放左侧或顶部时，在同一事务中追加 move 更新位置。resize 只增加 layoutRevision，不使 graph 计划失效。add 的 node 可带 width/height，复制节点与画布保留已保存尺寸；旧节点省略尺寸时保持默认显示。
