import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  Handle,
  Position,
  MarkerType,
  type Node,
  type Edge,
  type NodeProps,
} from "@xyflow/react";
import {
  Workflow,
  Play,
  Check,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Download,
  Sparkles,
  Layers,
  Box,
  Image as ImageIcon,
  Sliders,
  ShieldCheck,
  GitBranch,
  Terminal,
  Copy,
  FolderOpen,
  ChevronRight,
  Cpu,
  Eye,
  Ruler,
  Bone,
  PackageCheck,
} from "lucide-react";
import type {
  Asset,
  AssetRevision,
  ProjectSnapshot,
  WorkflowRun,
} from "../shared/types";
import {
  CHARACTER_DAG_TEMPLATE,
  MULTIVIEW_SLOT_META,
  TEXTURE_GROUP_IDS,
  TEXTURE_GROUP_LABELS,
  type CharacterDagNode,
  type CharacterStageGroup,
  type MultiviewSlotKey,
  type TextureGroupId,
} from "../shared/character-dag";
import {
  call,
  downloadAssetFile,
  fileUrl,
  isEmbeddedApp,
  request,
  useAssetFileUrl,
} from "./api";
import { ModelPreview } from "./ModelPreview";
import { characterLaunch } from "./workbench-launch";
import {
  CharacterSourceDialog,
  CharacterTaskDialog,
} from "./CharacterCreationDialog";
import {
  CHARACTER_STAGE_GUIDANCE,
  nodeGuidance,
  viewNodeKey,
} from "./character-workflow-guidance";
import { VrmPreview, type VrmViewerSettings } from "./VrmPreview";

function triggerDownload(projectId: string, revisionId: string, index = 0) {
  const route = fileUrl(projectId, revisionId, index);
  if (isEmbeddedApp()) {
    downloadAssetFile(route).catch(() => {});
  } else {
    window.open(`${route}?download=1`, "_blank", "noopener");
  }
}

function RevisionModelViewer({
  projectId,
  revisionId,
  name,
}: {
  projectId: string;
  revisionId: string;
  name: string;
}) {
  const { url } = useAssetFileUrl(projectId, revisionId, 0);
  if (!url) {
    return (
      <div className="char-slot-placeholder">
        <span>加载 3D 模型中…</span>
      </div>
    );
  }
  return <ModelPreview src={url} name={name} interactive detailed />;
}

function RevisionVrmViewer({
  projectId,
  revisionId,
  name,
  settings,
  onSettingsChange,
}: {
  projectId: string;
  revisionId: string;
  name: string;
  settings: VrmViewerSettings;
  onSettingsChange: (change: Partial<VrmViewerSettings>) => void;
}) {
  const { url, error } = useAssetFileUrl(projectId, revisionId, 0);
  if (!url)
    return (
      <div className="char-slot-placeholder" role={error ? "alert" : "status"}>
        {error || "加载 VRM 文件中…"}
      </div>
    );
  return (
    <VrmPreview
      key={url}
      src={url}
      name={name}
      initialSettings={settings}
      onSettingsChange={onSettingsChange}
    />
  );
}

interface LocalCharakuruProjectInfo {
  id: string;
  name: string;
  archivePath: string;
  projectDir: string;
  hasVrm: boolean;
  hasBlend: boolean;
  updatedAt: string;
}

const STAGE_TABS: Array<{
  key: CharacterStageGroup;
  title: string;
  subtitle: string;
  gateKey?: string;
}> = [
  {
    key: "reference",
    title: "1. 多视图与平衡参考",
    subtitle: "8向部位图 + 全身三视图 + 水平线对齐",
    gateKey: "images_gate",
  },
  {
    key: "candidate",
    title: "2. 头/身 3D 候选四选一",
    subtitle: "头×4 + 身×4 对比选型与非破坏分支",
    gateKey: "candidates_gate",
  },
  {
    key: "texture",
    title: "3. 部件拆解与五组重贴图",
    subtitle: "连通域拆解 + 5组4K贴图 + 独立摇摆组",
  },
  {
    key: "proportions",
    title: "4. 头身比例与关节 Marker",
    subtitle: "三视图叠加 + 身高标尺 + 61点关节校准",
    gateKey: "proportions_gate",
  },
  {
    key: "rig",
    title: "5. 骨架蒙皮与 90 帧物理",
    subtitle: "Humanoid映射 + 素体权重 + SpringBone检验",
    gateKey: "motion_gate",
  },
  {
    key: "delivery",
    title: "6. VRM 1.0 动作预览与交付",
    subtitle: "VRMA动作 + 物理开关 + 元数据打包",
  },
];

const DAG_NODE_POSITIONS: Record<string, { x: number; y: number }> = {
  source_illustration: { x: 20, y: 150 },
  head_multiview: { x: 250, y: 70 },
  body_multiview: { x: 250, y: 230 },
  balance_three_views: { x: 480, y: 150 },
  images_gate: { x: 700, y: 150 },
  head_candidates: { x: 930, y: 70 },
  body_candidates: { x: 930, y: 230 },
  candidates_gate: { x: 1160, y: 150 },
  mesh_segmentation: { x: 1390, y: 150 },
  part_classification: { x: 1620, y: 150 },
  uv_guide_prep: { x: 1850, y: 150 },
  retexture_hair: { x: 2090, y: 10 },
  retexture_head: { x: 2090, y: 85 },
  retexture_upper_body: { x: 2090, y: 160 },
  retexture_lower_body: { x: 2090, y: 235 },
  retexture_accessories: { x: 2090, y: 310 },
  mtoon_assembly: { x: 2340, y: 150 },
  proportions_gate: { x: 2570, y: 150 },
  humanoid_rig: { x: 2800, y: 150 },
  weight_transfer: { x: 3030, y: 150 },
  spring_verify: { x: 3260, y: 150 },
  motion_gate: { x: 3490, y: 150 },
  vrm_export: { x: 3720, y: 150 },
};

const statusLabelMap: Record<string, string> = {
  pending: "等待前置",
  awaiting_agent: "可认领 (Agent)",
  awaiting_manual: "待执行",
  awaiting_review: "待审核把关",
  running: "执行中 (Lease)",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
};

function RevisionImageThumb({
  projectId,
  revisionId,
  alt,
  showGuides,
  emptyState,
}: {
  projectId: string;
  revisionId?: string;
  alt: string;
  showGuides?: boolean;
  emptyState?: ReactNode;
}) {
  const { url, error } = useAssetFileUrl(projectId, revisionId, 0);
  if (!revisionId) {
    return (
      <div className="char-slot-placeholder">
        <ImageIcon size={20} />
        <span>{emptyState || "等待制作，请查看本阶段的操作指引。"}</span>
      </div>
    );
  }
  return (
    <div className="char-slot-img-wrap">
      {url && !error ? (
        <img src={url} alt={alt} loading="lazy" />
      ) : (
        <div className="char-slot-placeholder">
          <ImageIcon size={20} />
          <span>加载中…</span>
        </div>
      )}
      {showGuides && (
        <div className="char-alignment-guides" aria-hidden="true">
          <span className="guide-line eye" title="眼线基准 (Eye Line)" />
          <span className="guide-line shoulder" title="肩线基准 (Shoulder)" />
          <span className="guide-line waist" title="腰臀线基准 (Waist/Hip)" />
          <span className="guide-line knee" title="膝/裙摆线基准 (Knee/Hem)" />
        </div>
      )}
    </div>
  );
}

function CharacterSourcePicker({
  projectId,
  assets,
  value,
  disabled,
  onChange,
  onUpload,
}: {
  projectId: string;
  assets: Asset[];
  value: string;
  disabled: boolean;
  onChange: (revisionId: string) => void;
  onUpload: (file: File) => Promise<void>;
}) {
  return (
    <div className="char-source-picker">
      <label className="field">
        角色原画
        <select
          aria-label="角色原画"
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
        >
          <option value="">请选择角色原画</option>
          {assets.map((asset) => (
            <optgroup key={asset.id} label={asset.name}>
              {asset.revisions.map((rev) => (
                <option key={rev.id} value={rev.id}>
                  {asset.name} · v{rev.number}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </label>
      <label className="field">
        或上传原画
        <input
          aria-label="或上传原画"
          type="file"
          accept="image/png,image/jpeg,image/webp"
          disabled={disabled}
          onChange={(e) => {
            const file = e.target.files?.[0];
            e.target.value = "";
            if (file) void onUpload(file);
          }}
        />
      </label>
      {value && (
        <RevisionImageThumb
          projectId={projectId}
          revisionId={value}
          alt="已选角色原画预览"
        />
      )}
    </div>
  );
}

function DagFlowNodeCard({ data }: NodeProps) {
  const node = data.dagNode as CharacterDagNode;
  const active = Boolean(data.selectedNode);
  const isGate = node.mode === "review";
  const statusClass = node.stale
    ? "stale"
    : node.status === "succeeded"
      ? "succeeded"
      : node.status === "running"
        ? "running"
        : node.status.startsWith("awaiting_")
          ? "ready"
          : "pending";
  return (
    <div
      className={`char-dag-node ${statusClass} ${isGate ? "gate" : ""} ${active ? "selected" : ""}`}
    >
      <Handle type="target" position={Position.Left} />
      <div className="char-dag-node-head">
        <span className="char-dag-mode-tag">
          {isGate
            ? "GATE 审核"
            : node.mode === "agent"
              ? "AGENT"
              : node.mode === "local"
                ? "LOCAL"
                : "MANUAL"}
        </span>
        <span className={`char-dag-status-dot ${statusClass}`} />
      </div>
      <div className="char-dag-node-title">{node.title}</div>
      <div className="char-dag-node-meta">
        <span>
          {node.stale
            ? "需重跑 (Stale)"
            : statusLabelMap[node.status] || node.status}
        </span>
        {node.outputRevisionIds.length > 0 && (
          <span className="mono">×{node.outputRevisionIds.length}</span>
        )}
      </div>
      {node.lease && (
        <div className="char-dag-lease-bar">
          <div
            className="char-dag-lease-fill"
            style={{
              width: `${Math.min(100, Math.max(8, node.lease.progress))}%`,
            }}
          />
          <small>
            {node.lease.agentName} ({node.lease.progress}%)
          </small>
        </div>
      )}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const dagNodeTypes = {
  charDag: DagFlowNodeCard,
};

export function CharacterWorkflowPage({
  s,
  activeRunId,
  onSelectRun,
  refresh,
  notify,
}: {
  s: ProjectSnapshot;
  activeRunId?: string;
  onSelectRun: (runId: string) => void;
  refresh: () => Promise<void>;
  notify: (msg: string) => void;
}) {
  const runs = useMemo(() => [...s.runs].reverse(), [s.runs]);
  const run: WorkflowRun | undefined = useMemo(
    () =>
      runs.find((r) => r.id === activeRunId) ||
      runs.find((r) => r.status === "active") ||
      runs[0],
    [runs, activeRunId],
  );

  const launchedNode = run?.dag?.find(
    (node) => node.key === characterLaunch(s.project.id)?.nodeKey,
  );
  const [activeStage, setActiveStage] = useState<CharacterStageGroup>(
    launchedNode?.stageGroup || "reference",
  );
  const [selectedNodeKey, setSelectedNodeKey] = useState<string>(
    launchedNode?.key || "head_multiview",
  );
  const [busy, setBusy] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [localCharakuru, setLocalCharakuru] = useState<
    LocalCharakuruProjectInfo[]
  >([]);
  const [showAgentProtocol, setShowAgentProtocol] = useState(false);
  const [showFlow, setShowFlow] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newCharName, setNewCharName] = useState("新二次元角色");
  const [newCharPrompt, setNewCharPrompt] = useState(
    "日系二次元 3D 角色，MToon 赛璐璐材质，VRM 1.0 标准 55 骨骼 + SpringBone 物理摇摆",
  );
  const [newCharImageRevId, setNewCharImageRevId] = useState("");
  const [sourcePickerRunId, setSourcePickerRunId] = useState<string>();
  const [replacementSourceRevId, setReplacementSourceRevId] = useState("");
  const [codexTarget, setCodexTarget] = useState<{
    runId: string;
    nodeKey: string;
    slotKey?: MultiviewSlotKey;
  }>();
  const [customCharakitPath, setCustomCharakitPath] = useState("");
  const [gateComment, setGateComment] = useState("");
  const [slotEditPrompts, setSlotEditPrompts] = useState<
    Record<string, string>
  >({});
  const [componentFilter, setComponentFilter] = useState<string>("all");

  useEffect(() => {
    request<{ projects: LocalCharakuruProjectInfo[] }>(
      "/api/charakuru/local-projects",
    )
      .then((res) => setLocalCharakuru(res.projects || []))
      .catch(() => setLocalCharakuru([]));
  }, []);

  const dagNodes = run?.dag || [];
  const state = run?.dagState;
  const selectedDagNode =
    dagNodes.find((n) => n.key === selectedNodeKey) || dagNodes[0];

  const flowNodes: Node[] = useMemo(
    () =>
      dagNodes.map((n) => ({
        id: n.key,
        type: "charDag",
        position: DAG_NODE_POSITIONS[n.key] || { x: 0, y: 0 },
        data: {
          dagNode: n,
          selectedNode: n.key === selectedDagNode?.key,
        },
        draggable: false,
      })),
    [dagNodes, selectedDagNode?.key],
  );

  const flowEdges: Edge[] = useMemo(
    () =>
      dagNodes.flatMap((n) =>
        n.dependsOn.map((dep) => ({
          id: `${dep}->${n.key}`,
          source: dep,
          target: n.key,
          animated: n.status === "running" || n.status.startsWith("awaiting_"),
          markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
          style: {
            stroke:
              n.status === "succeeded" && !n.stale
                ? "var(--success, #22c55e)"
                : n.status.startsWith("awaiting_")
                  ? "var(--accent, #3b82f6)"
                  : "var(--border-strong, #52525b)",
            strokeWidth: 1.75,
          },
        })),
      ),
    [dagNodes],
  );

  const imageAssets = useMemo(
    () => s.assets.filter((a) => !a.archived && a.kind === "image"),
    [s.assets],
  );

  const revLookup = useMemo(() => {
    const map = new Map<string, { asset: Asset; rev: AssetRevision }>();
    for (const a of s.assets) {
      for (const r of a.revisions) map.set(r.id, { asset: a, rev: r });
    }
    return map;
  }, [s.assets]);

  function openSourceDialog() {
    if (!run) return;
    setReplacementSourceRevId(state?.views.original.revisionId || "");
    setErrorMsg("");
    setSourcePickerRunId(run.id);
  }

  function focusNode(key: string) {
    const node = dagNodes.find((item) => item.key === key);
    if (!node) return;
    setActiveStage(node.stageGroup);
    setSelectedNodeKey(key);
    document
      .getElementById("character-stage-guide")
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function openNodeTask(key: string, slotKey?: MultiviewSlotKey) {
    if (!run) return;
    const guidance = nodeGuidance(run, key);
    if (guidance.blockedBy.length) {
      focusNode(guidance.blockedBy[0]!.key);
      return;
    }
    if (guidance.running) {
      focusNode(key);
      return;
    }
    if (key === "source_illustration") {
      openSourceDialog();
      return;
    }
    setSelectedNodeKey(key);
    setActiveStage(guidance.node!.stageGroup);
    setCodexTarget({ runId: run.id, nodeKey: key, slotKey });
  }

  function emptyNode(key: string, slotKey?: MultiviewSlotKey) {
    if (!run) return null;
    const guidance = nodeGuidance(run, key);
    return (
      <span className="character-empty-guidance">
        <strong>
          {guidance.running
            ? "正在制作"
            : guidance.blockedBy.length
              ? "等待前置步骤"
              : "等待制作"}
        </strong>
        <span>{guidance.reason}</span>
        <button
          type="button"
          className="button small"
          disabled={busy || guidance.running}
          onClick={() => openNodeTask(key, slotKey)}
        >
          {guidance.blockedBy.length
            ? "前往前置步骤"
            : key === "source_illustration"
              ? "选择或生成原画"
              : "与 Codex 制作"}
        </button>
      </span>
    );
  }

  async function invokeAction<T = WorkflowRun>(
    action: string,
    body: Record<string, unknown>,
    successMsg?: string,
  ): Promise<T | undefined> {
    if (action === "advance-node") {
      openNodeTask(String(body.nodeKey));
      return undefined;
    }
    if (action === "edit-view" && !body.revisionId) {
      const slotKey = body.slotKey as MultiviewSlotKey;
      openNodeTask(viewNodeKey(slotKey), slotKey);
      return undefined;
    }
    setBusy(true);
    setErrorMsg("");
    try {
      const res = await request<T>(
        `/api/projects/${s.project.id}/character-workflow/${action}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      );
      await refresh();
      if (successMsg) notify(successMsg);
      return res;
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
      return undefined;
    } finally {
      setBusy(false);
    }
  }

  async function handleImportCharakit(archivePath?: string, name?: string) {
    const imported = await invokeAction<WorkflowRun>(
      "import-charakit",
      {
        archivePath: archivePath || customCharakitPath || undefined,
        characterName: name,
      },
      "已从 Charakuru (.charakit) 完整迁移角色工作流与资产",
    );
    if (imported?.id) {
      onSelectRun(imported.id);
    }
  }

  async function handleCreateRun() {
    if (!newCharImageRevId) {
      setErrorMsg("请先选择或上传角色原画");
      return;
    }
    const created = await invokeAction<WorkflowRun>(
      "create",
      {
        characterName: newCharName.trim() || "新角色",
        stylePrompt: newCharPrompt,
        sourceImageRevisionId: newCharImageRevId || undefined,
      },
      `已创建角色：${newCharName}`,
    );
    if (created?.id) {
      onSelectRun(created.id);
      setShowCreateModal(false);
      setNewCharImageRevId("");
    }
  }

  async function uploadSourceImage(file: File, select: (id: string) => void) {
    setBusy(true);
    setErrorMsg("");
    try {
      const form = new FormData();
      form.append("files", file);
      const revision = await request<AssetRevision>(
        `/api/projects/${s.project.id}/import`,
        { method: "POST", body: form },
      );
      await refresh();
      select(revision.id);
    } catch (error) {
      setErrorMsg(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  function handleAutoAdvance() {
    if (!run) return;
    const next = dagNodes.find((node) => {
      const guidance = nodeGuidance(run, node.key);
      return guidance.ready && !guidance.complete;
    });
    if (next) {
      if (next.mode === "review") focusNode(next.key);
      else openNodeTask(next.key);
    }
  }

  async function handleSelectCandidate(
    source: "head" | "body",
    candidateId: string,
  ) {
    if (!run) return;
    const res = await invokeAction<{
      run: WorkflowRun;
      branched: boolean;
      parentRunId?: string;
    }>("select-candidate", {
      runId: run.id,
      source,
      candidateId,
    });
    if (res) {
      if (res.branched) {
        onSelectRun(res.run.id);
        notify(
          `已派生非破坏性新分支 (${res.run.id.slice(0, 8)}) 采用 ${candidateId}，原分支贴图完整保留`,
        );
      } else {
        notify(
          `已选定${source === "head" ? "头部" : "身体"}候选：${candidateId}`,
        );
      }
    }
  }

  function handleSyncToCodex() {
    if (selectedDagNode) openNodeTask(selectedDagNode.key);
  }

  async function handleExportVrmBundle() {
    if (!run) return;
    setBusy(true);
    setErrorMsg("");
    try {
      const vrmRevId = state?.vrmDelivery.vrmRevisionId;
      if (!vrmRevId) {
        openNodeTask("vrm_export");
        return;
      }
      if (vrmRevId) {
        await call("export_bundle", {
          projectId: s.project.id,
          revisionIds: [vrmRevId],
          idempotencyKey: `vrm-export-${run.id}-${run.revision}`,
        });
      }
      await refresh();
      notify("已生成 VRM 1.0 成品包并写入项目交付记录 (.game-studio/exports)");
    } catch (e) {
      setErrorMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const completedDagCount = dagNodes.filter(
    (n) => n.status === "succeeded" && !n.stale,
  ).length;

  return (
    <div className="content-page char-workflow-page">
      {/* Top Header */}
      <div className="char-page-header">
        <div className="char-page-title-block">
          <div className="char-title-row">
            <Workflow size={20} />
            <h1>角色制作</h1>
            <span className="char-badge accent">
              {completedDagCount} / {dagNodes.length || 23} 步骤已完成
            </span>
            {run?.branchFrom && (
              <button
                type="button"
                className="char-badge branch"
                title="点击切回父分支运行"
                onClick={() => onSelectRun(run.branchFrom!.runId)}
              >
                <GitBranch size={12} />
                分支自 {run.branchFrom.runId.slice(0, 8)} (切换{" "}
                {run.branchFrom.candidateId})
              </button>
            )}
            {state?.importedCharakit && (
              <span className="char-badge green">
                已迁移 Charakuru 归档 ({state.importedCharakit.fileCount} 文件)
              </span>
            )}
          </div>
          <p className="char-subtitle">
            从单张角色立绘到可商用 VRM 1.0 成品：支持头/身并行多视图、4×2
            候选分支选型、168+ 部件自动分群、5 路并行 4K 贴图、61 点关节 Marker
            与 90 帧 SpringBone 物理验证。
          </p>
        </div>

        <div className="char-header-actions">
          {runs.length > 0 && (
            <select
              aria-label="切换角色工作流运行"
              className="char-run-select"
              value={run?.id || ""}
              onChange={(e) => onSelectRun(e.target.value)}
            >
              {runs.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.dagState?.characterName || "角色制作"} · {r.id.slice(0, 8)}
                  {r.branchFrom ? ` [分支:${r.branchFrom.candidateId}]` : ""} (
                  {
                    (r.dag || []).filter(
                      (n) => n.status === "succeeded" && !n.stale,
                    ).length
                  }
                  /{(r.dag || []).length} 节点)
                </option>
              ))}
            </select>
          )}
          <button
            type="button"
            className="button small"
            onClick={() => setShowCreateModal(!showCreateModal)}
          >
            <Sparkles size={14} />
            新建角色
          </button>
          {localCharakuru.length > 0 && (
            <button
              type="button"
              className="button small"
              disabled={busy}
              onClick={() =>
                handleImportCharakit(
                  localCharakuru[0].archivePath,
                  "青绿衣装少女 (Charakuru 迁移)",
                )
              }
            >
              <FolderOpen size={14} />
              一键导入本机 Charakuru 角色
            </button>
          )}
          <button
            type="button"
            className="button small"
            onClick={() => setShowAgentProtocol(!showAgentProtocol)}
          >
            <Terminal size={14} />多 Agent 接入协议
          </button>
          <button
            type="button"
            className="button small"
            onClick={handleSyncToCodex}
          >
            <Copy size={14} />在 Codex 中处理当前步骤
          </button>
          {run && (
            <button
              type="button"
              className="button small primary"
              disabled={busy}
              onClick={handleAutoAdvance}
            >
              <Play size={14} />
              查看下一步
            </button>
          )}
        </div>
      </div>

      {errorMsg && (
        <div className="char-alert error" role="alert">
          <AlertTriangle size={15} />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Create / Import Modal Drawer */}
      {showCreateModal && (
        <div className="char-drawer-card">
          <div className="char-drawer-head">
            <h3>新建角色 / 导入 .charakit 归档</h3>
            <button
              type="button"
              className="button small"
              onClick={() => setShowCreateModal(false)}
            >
              收起
            </button>
          </div>
          <div className="char-create-grid">
            <div className="char-create-col">
              <h4>选择原画，创建角色</h4>
              <label className="field">
                角色名称
                <input
                  value={newCharName}
                  onChange={(e) => setNewCharName(e.target.value)}
                  placeholder="例如：星铸巡林者"
                />
              </label>
              <CharacterSourcePicker
                projectId={s.project.id}
                assets={imageAssets}
                value={newCharImageRevId}
                disabled={busy}
                onChange={setNewCharImageRevId}
                onUpload={(file) =>
                  uploadSourceImage(file, setNewCharImageRevId)
                }
              />
              <label className="field">
                风格与规格要求
                <textarea
                  rows={2}
                  value={newCharPrompt}
                  onChange={(e) => setNewCharPrompt(e.target.value)}
                />
              </label>
              <button
                type="button"
                className="button small primary"
                disabled={busy || !newCharImageRevId}
                onClick={handleCreateRun}
              >
                <Play size={14} />
                创建角色
              </button>
            </div>
            <div className="char-create-col">
              <h4>从 Charakuru (.charakit) 归档迁移</h4>
              {localCharakuru.map((item) => (
                <div key={item.id} className="char-local-project-item">
                  <div>
                    <strong>{item.name}</strong>
                    <code className="mono">{item.archivePath}</code>
                    <small>
                      {item.hasVrm ? "含导出 avatar.vrm · " : ""}
                      {item.hasBlend ? "含绑定 avatar.blend" : ""}
                    </small>
                  </div>
                  <button
                    type="button"
                    className="button small primary"
                    disabled={busy}
                    onClick={() =>
                      handleImportCharakit(
                        item.archivePath,
                        "青绿衣装少女 (Charakuru)",
                      )
                    }
                  >
                    导入此项目
                  </button>
                </div>
              ))}
              <label className="field">
                或指定任意本地 .charakit 归档绝对路径
                <div style={{ display: "flex", gap: "8px" }}>
                  <input
                    value={customCharakitPath}
                    onChange={(e) => setCustomCharakitPath(e.target.value)}
                    placeholder="/path/to/project.charakit"
                  />
                  <button
                    type="button"
                    className="button small"
                    disabled={busy || !customCharakitPath.trim()}
                    onClick={() => handleImportCharakit(customCharakitPath)}
                  >
                    导入归档
                  </button>
                </div>
              </label>
            </div>
          </div>
        </div>
      )}

      {/* Multi-Agent Protocol Drawer */}
      {showAgentProtocol && (
        <div className="char-drawer-card protocol">
          <div className="char-drawer-head">
            <div>
              <h3>
                多 Agent 接入与并行任务认领协议 (External Agent DAG Protocol)
              </h3>
              <p className="muted">
                无需手动复制 SKILL：Codex、Claude Code、Blender MCP Worker 或
                Tripo Worker 可直接通过 MCP 工具或 HTTP
                接口并发认领就绪节点、续期租约并回填不可变版本。
              </p>
            </div>
            <button
              type="button"
              className="button small"
              onClick={() => setShowAgentProtocol(false)}
            >
              关闭面板
            </button>
          </div>
          <div className="char-protocol-grid">
            <div className="char-protocol-box">
              <strong>1. 查询可并行认领节点 (GET)</strong>
              <pre className="mono">{`GET /api/projects/${s.project.id}/character-workflow/claimable${run ? `?runId=${run.id}` : ""}`}</pre>
              <small>
                返回所有处于 awaiting_agent / stale
                的并行节点、冻结输入文件绝对路径及技能契约。
              </small>
            </div>
            <div className="char-protocol-box">
              <strong>2. 认领节点并获取租约 (POST / MCP)</strong>
              <pre className="mono">{`POST /api/projects/${s.project.id}/character-workflow/claim
{
  "runId": "${run?.id || "run_xxx"}",
  "nodeKey": "${selectedDagNode?.key || "head_multiview"}",
  "agentId": "tripo-worker-01",
  "agentName": "Tripo 3D Agent",
  "leaseSeconds": 300
}`}</pre>
              <small>
                或调用 MCP 工具：
                <code>
                  job_control(action="claim_dag_node", nodeKey="
                  {selectedDagNode?.key || "head_multiview"}")
                </code>
              </small>
            </div>
            <div className="char-protocol-box">
              <strong>3. 提交节点产物并解锁下游 (POST / MCP)</strong>
              <pre className="mono">{`POST /api/projects/${s.project.id}/character-workflow/submit
{
  "runId": "${run?.id || "run_xxx"}",
  "nodeKey": "${selectedDagNode?.key || "head_multiview"}",
  "fileRefs": ["/abs/path/to/output.png"],
  "note": "Agent 完成制作"
}`}</pre>
              <small>
                或调用 MCP 工具：
                <code>
                  workflow_submit_artifact(stepKey="
                  {selectedDagNode?.key || "head_multiview"}",
                  revisionIds=[...])
                </code>
              </small>
            </div>
          </div>
        </div>
      )}

      {!run || !state ? (
        <div className="char-empty-hero">
          <Workflow size={40} />
          <h2>创建你的第一个 3D 角色</h2>
          <p>
            支持直接从单张立绘启动 23 节点全流程制作，或一键导入本机 Charakuru
            角色工程（含 8 向部位图、头/身 3D 候选、168
            部件拆解、贴图引导图与导出 VRM）。
          </p>
          <div
            style={{ display: "flex", gap: "12px", justifyContent: "center" }}
          >
            <button
              type="button"
              className="button primary"
              disabled={busy}
              onClick={() => setShowCreateModal(true)}
            >
              <Sparkles size={15} />
              新建角色
            </button>
            {localCharakuru.length > 0 && (
              <button
                type="button"
                className="button"
                disabled={busy}
                onClick={() =>
                  handleImportCharakit(
                    localCharakuru[0].archivePath,
                    "青绿衣装少女 (Charakuru 迁移)",
                  )
                }
              >
                <FolderOpen size={15} />
                一键导入本机 Charakuru 角色 ({localCharakuru[0].id.slice(0, 8)})
              </button>
            )}
          </div>
        </div>
      ) : (
        <>
          <section className="character-next-actions" aria-label="现在该做什么">
            <div>
              <span className="character-guide-kicker">现在该做什么</span>
              <h2>
                {state.views.original.revisionId
                  ? "先确认原画，再完成下面的制作"
                  : "先准备角色原画"}
              </h2>
              <p>
                左侧对话提出要求，制作结果会自动更新到这里。 原画 → 多视图 → 3D
                候选 → 拆解贴图 → 比例与绑定 → 动作检查 → 交付
              </p>
              {state.views.original.revisionId && (
                <p>
                  当前原画：
                  <strong>
                    {revLookup.get(state.views.original.revisionId)?.asset
                      .name || "已选图片"}
                  </strong>
                  。请确认它就是要制作的角色。
                </p>
              )}
              <button
                type="button"
                className="button small"
                onClick={openSourceDialog}
              >
                查看或更换原画
              </button>
            </div>
            <div className="character-next-list">
              {dagNodes
                .filter((node) => {
                  const guidance = nodeGuidance(run, node.key);
                  return guidance.ready && !guidance.complete;
                })
                .slice(0, 3)
                .map((node) => (
                  <div key={node.key}>
                    <strong>{node.title}</strong>
                    <p>{nodeGuidance(run, node.key).reason}</p>
                    <button
                      type="button"
                      className="button small primary"
                      disabled={busy}
                      onClick={() =>
                        node.mode === "review"
                          ? focusNode(node.key)
                          : openNodeTask(node.key)
                      }
                    >
                      {node.mode === "review"
                        ? "去检查并审核"
                        : node.key === "source_illustration"
                          ? "选择或生成原画"
                          : `与 Codex 制作${node.title.replace("生成", "")}`}
                    </button>
                  </div>
                ))}
              {!dagNodes.some((node) => {
                const guidance = nodeGuidance(run, node.key);
                return guidance.ready && !guidance.complete;
              }) && (
                <p>
                  {run.status === "succeeded"
                    ? "制作已完成，可以前往交付页预览和下载。"
                    : "当前任务正在制作，进度与结果会自动更新。"}
                </p>
              )}
              <p className="panel-hint">
                与左侧 Codex 对话协作，制作进度和产物会自动同步到这里。
              </p>
            </div>
          </section>
          <button
            type="button"
            className="button small character-flow-toggle"
            aria-expanded={showFlow}
            onClick={() => setShowFlow(!showFlow)}
          >
            {showFlow ? "收起流程图与步骤详情" : "查看流程图与步骤详情"}
          </button>
          {showFlow && (
            <>
              {/* Interactive DAG Graph + Node Inspector */}
              <div className="char-dag-board-wrap">
                <div className="char-dag-flow-canvas">
                  <div className="char-dag-board-toolbar">
                    <span>
                      <strong>DAG 拓扑图</strong> ·
                      点击任意节点可定位对应工作台或单独指派 Agent
                    </span>
                    <div className="char-dag-legend">
                      <span>
                        <i className="dot succeeded" /> 已完成
                      </span>
                      <span>
                        <i className="dot ready" /> 就绪可认领
                      </span>
                      <span>
                        <i className="dot running" /> Agent 执行中
                      </span>
                      <span>
                        <i className="dot gate" /> 审核关卡 (Gate)
                      </span>
                      <span>
                        <i className="dot stale" /> 上游变更需重跑
                      </span>
                    </div>
                  </div>
                  <ReactFlow
                    nodes={flowNodes}
                    edges={flowEdges}
                    nodeTypes={dagNodeTypes}
                    fitView
                    minZoom={0.25}
                    maxZoom={1.5}
                    style={{ background: "var(--bg-surface)" }}
                    proOptions={{ hideAttribution: true }}
                    onNodeClick={(_e, node) => {
                      setSelectedNodeKey(node.id);
                      const found = dagNodes.find((d) => d.key === node.id);
                      if (found) setActiveStage(found.stageGroup);
                    }}
                  >
                    <Background gap={20} size={1} />
                    <Controls showInteractive={false} />
                  </ReactFlow>
                </div>

                {selectedDagNode && (
                  <div className="char-dag-node-inspector">
                    <div className="char-node-insp-head">
                      <div>
                        <span className="mono muted">
                          {selectedDagNode.key}
                        </span>
                        <h3>{selectedDagNode.title}</h3>
                      </div>
                      <span
                        className={`char-badge ${selectedDagNode.status === "succeeded" && !selectedDagNode.stale ? "green" : "accent"}`}
                      >
                        {selectedDagNode.stale
                          ? "上游已变更 (Stale)"
                          : statusLabelMap[selectedDagNode.status] ||
                            selectedDagNode.status}
                      </span>
                    </div>
                    <p className="char-node-desc">
                      {selectedDagNode.description}
                    </p>
                    <div className="char-node-kv">
                      <div>
                        <small>前置依赖节点</small>
                        <code className="mono">
                          {selectedDagNode.dependsOn.length
                            ? selectedDagNode.dependsOn.join(", ")
                            : "无 (起始节点)"}
                        </code>
                      </div>
                      <div>
                        <small>绑定技能契约</small>
                        <code className="mono">
                          {selectedDagNode.agentSkill || "人工确认关卡"}
                        </code>
                      </div>
                      <div>
                        <small>输入/输出不可变版本</small>
                        <code className="mono">
                          输入 {selectedDagNode.inputRevisionIds.length} · 产出{" "}
                          {selectedDagNode.outputRevisionIds.length}
                        </code>
                      </div>
                    </div>
                    {selectedDagNode.lease && (
                      <div className="char-node-lease-box">
                        <strong>
                          当前租约：{selectedDagNode.lease.agentName} (
                          {selectedDagNode.lease.progress}%)
                        </strong>
                        <p>{selectedDagNode.lease.message}</p>
                      </div>
                    )}
                    <p className="character-node-guidance">
                      {nodeGuidance(run, selectedDagNode.key).reason}
                    </p>
                    <div className="char-node-actions">
                      {selectedDagNode.mode !== "review" ? (
                        <button
                          type="button"
                          className="button small primary"
                          disabled={
                            busy || selectedDagNode.status === "running"
                          }
                          onClick={() => openNodeTask(selectedDagNode.key)}
                        >
                          <Cpu size={13} />
                          {nodeGuidance(run, selectedDagNode.key).blockedBy
                            .length
                            ? "前往前置步骤"
                            : selectedDagNode.key === "source_illustration"
                              ? "选择或生成原画"
                              : "与 Codex 制作此步骤"}
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="button small primary"
                          disabled={
                            busy ||
                            !nodeGuidance(run, selectedDagNode.key).ready ||
                            nodeGuidance(run, selectedDagNode.key).complete
                          }
                          onClick={() =>
                            invokeAction(
                              "decide-gate",
                              {
                                runId: run.id,
                                nodeKey: selectedDagNode.key,
                                decision: "accept",
                                comment:
                                  gateComment ||
                                  `审核通过：${selectedDagNode.title}`,
                              },
                              `已通过审核关卡：${selectedDagNode.title}`,
                            )
                          }
                        >
                          <CheckCircle2 size={13} />
                          通过此审核关卡
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
          {/* 6 Stage Navigation Tabs */}
          <div className="char-stage-tabs" role="tablist">
            {STAGE_TABS.map((tab) => {
              const stageNodes = dagNodes.filter(
                (n) => n.stageGroup === tab.key,
              );
              const doneCount = stageNodes.filter(
                (n) => n.status === "succeeded" && !n.stale,
              ).length;
              return (
                <button
                  key={tab.key}
                  type="button"
                  role="tab"
                  aria-selected={activeStage === tab.key}
                  className={`char-stage-tab ${activeStage === tab.key ? "active" : ""}`}
                  onClick={() => setActiveStage(tab.key)}
                >
                  <div className="char-stage-tab-top">
                    <strong>{tab.title}</strong>
                    <span className="mono">
                      {doneCount}/{stageNodes.length}
                    </span>
                  </div>
                  <small>{tab.subtitle}</small>
                </button>
              );
            })}
          </div>

          <section
            className="character-stage-guide"
            id="character-stage-guide"
            aria-label="本阶段操作指引"
          >
            <h2>{STAGE_TABS.find((tab) => tab.key === activeStage)?.title}</h2>
            <p>
              <strong>你需要做：</strong>
              {CHARACTER_STAGE_GUIDANCE[activeStage].task}
            </p>
            <p>
              <strong>完成后：</strong>
              {CHARACTER_STAGE_GUIDANCE[activeStage].after}
            </p>
            <ol className="character-stage-checklist">
              {dagNodes
                .filter((node) => node.stageGroup === activeStage)
                .map((node) => {
                  const guidance = nodeGuidance(run, node.key);
                  return (
                    <li
                      key={node.key}
                      className={selectedNodeKey === node.key ? "current" : ""}
                    >
                      <span
                        className={`character-step-status ${guidance.complete ? "complete" : ""}`}
                      >
                        {guidance.complete ? (
                          <CheckCircle2 size={17} />
                        ) : guidance.running ? (
                          <RefreshCw size={17} />
                        ) : (
                          <span>
                            {node.mode === "review" ? "审核" : "制作"}
                          </span>
                        )}
                      </span>
                      <div>
                        <strong>{node.title}</strong>
                        <p>{guidance.reason}</p>
                      </div>
                      <button
                        type="button"
                        className="button small"
                        disabled={busy || guidance.running}
                        onClick={() =>
                          node.mode === "review" && guidance.ready
                            ? focusNode(node.key)
                            : openNodeTask(node.key)
                        }
                      >
                        {guidance.complete
                          ? "查看 / 调整"
                          : guidance.running
                            ? "制作中"
                            : guidance.blockedBy.length
                              ? "前往前置步骤"
                              : node.mode === "review"
                                ? "查看审核内容"
                                : node.key === "source_illustration"
                                  ? "选择或生成"
                                  : "与 Codex 制作"}
                      </button>
                    </li>
                  );
                })}
            </ol>
          </section>
          {/* STAGE 1: Reference & 8-View + 3 Balance Views */}
          {activeStage === "reference" && (
            <div className="char-workbench-panel">
              <div className="char-workbench-head">
                <div>
                  <h2>阶段 1 · 8向正交部位图与全身平衡三视图工作台</h2>
                  <p>
                    头部四视图（仅含头与发）与身体四视图（颈部以下去头
                    A-Pose）并行生成，配合水平对齐参考线校验眼线、肩线、腰臀线与裙摆高度。
                  </p>
                </div>
                <div className="char-workbench-controls">
                  <label className="char-toggle">
                    <input
                      type="checkbox"
                      checked={state.placement.showHorizontalGuides}
                      onChange={(e) =>
                        invokeAction("update-placement", {
                          runId: run.id,
                          showHorizontalGuides: e.target.checked,
                        })
                      }
                    />
                    显示水平对齐参考线 (Alignment Guides)
                  </label>
                  <button
                    type="button"
                    className="button small"
                    disabled={
                      busy || !nodeGuidance(run, "head_multiview").ready
                    }
                    onClick={() =>
                      invokeAction(
                        "advance-node",
                        { runId: run.id, nodeKey: "head_multiview" },
                        "已生成头部四视图",
                      )
                    }
                  >
                    与 Codex 制作头部四视图
                  </button>
                  <button
                    type="button"
                    className="button small"
                    disabled={
                      busy || !nodeGuidance(run, "body_multiview").ready
                    }
                    onClick={() =>
                      invokeAction(
                        "advance-node",
                        { runId: run.id, nodeKey: "body_multiview" },
                        "已生成身体四视图",
                      )
                    }
                  >
                    与 Codex 制作身体四视图
                  </button>
                  <button
                    type="button"
                    className="button small primary"
                    disabled={
                      busy ||
                      !nodeGuidance(run, "images_gate").ready ||
                      nodeGuidance(run, "images_gate").complete
                    }
                    onClick={() =>
                      invokeAction(
                        "decide-gate",
                        {
                          runId: run.id,
                          nodeKey: "images_gate",
                          decision: "accept",
                          comment:
                            "8向部位视图与全身平衡三视图水平线对齐确认通过",
                        },
                        "已确认多视图审核关卡，解锁头/身 3D 候选生成",
                      )
                    }
                  >
                    <Check size={14} />
                    确认多视图并进入 3D 候选
                  </button>
                </div>
              </div>

              <div className="char-views-section">
                <h3>原画立绘 & 全身平衡参考三视图 (Balance 3-Views)</h3>
                <div className="char-views-grid four">
                  {(
                    [
                      "original",
                      "balance_front",
                      "balance_left",
                      "balance_back",
                    ] as MultiviewSlotKey[]
                  ).map((slotKey) => {
                    const slot = state.views[slotKey];
                    return (
                      <div key={slotKey} className="char-view-card">
                        <div className="char-view-card-head">
                          <strong>{slot.label}</strong>
                          <span className="mono">{slot.resolution}</span>
                        </div>
                        <RevisionImageThumb
                          projectId={s.project.id}
                          revisionId={slot.revisionId}
                          emptyState={emptyNode(viewNodeKey(slotKey), slotKey)}
                          alt={slot.label}
                          showGuides={state.placement.showHorizontalGuides}
                        />
                        {slotKey === "original" ? (
                          <div className="char-source-actions">
                            <button
                              type="button"
                              className="button small"
                              disabled={busy}
                              onClick={openSourceDialog}
                            >
                              {slot.revisionId ? "更换原画" : "选择或生成原画"}
                            </button>
                          </div>
                        ) : slot.revisionId ? (
                          <div className="char-view-card-foot">
                            <input
                              value={slotEditPrompts[slotKey] || ""}
                              onChange={(e) =>
                                setSlotEditPrompts({
                                  ...slotEditPrompts,
                                  [slotKey]: e.target.value,
                                })
                              }
                              placeholder="单图微调提示词 (如：修正袖口纹样)…"
                            />
                            <button
                              type="button"
                              className="button small"
                              disabled={busy}
                              onClick={() =>
                                invokeAction(
                                  "edit-view",
                                  {
                                    runId: run.id,
                                    slotKey,
                                    editPrompt:
                                      slotEditPrompts[slotKey] ||
                                      "局部重绘微调",
                                  },
                                  `已更新视图槽位：${slot.label}`,
                                )
                              }
                            >
                              与 Codex 调整此视图
                            </button>
                          </div>
                        ) : null}
                      </div>
                    );
                  })}
                </div>

                <h3>
                  头部正交四视图 (Head 4-Views · 眼线/耳线/发际线水平对齐)
                </h3>
                <div className="char-views-grid four">
                  {(
                    [
                      "head_front",
                      "head_left",
                      "head_right",
                      "head_back",
                    ] as MultiviewSlotKey[]
                  ).map((slotKey) => {
                    const slot = state.views[slotKey];
                    return (
                      <div key={slotKey} className="char-view-card">
                        <div className="char-view-card-head">
                          <strong>{slot.label}</strong>
                          <span className="mono">{slot.resolution}</span>
                        </div>
                        <RevisionImageThumb
                          projectId={s.project.id}
                          revisionId={slot.revisionId}
                          emptyState={emptyNode(viewNodeKey(slotKey), slotKey)}
                          alt={slot.label}
                          showGuides={state.placement.showHorizontalGuides}
                        />
                        {slot.revisionId && (
                          <div className="char-view-card-foot">
                            <input
                              value={slotEditPrompts[slotKey] || ""}
                              onChange={(e) =>
                                setSlotEditPrompts({
                                  ...slotEditPrompts,
                                  [slotKey]: e.target.value,
                                })
                              }
                              placeholder="单张头部视图微调指令…"
                            />
                            <button
                              type="button"
                              className="button small"
                              disabled={busy}
                              onClick={() =>
                                invokeAction(
                                  "edit-view",
                                  {
                                    runId: run.id,
                                    slotKey,
                                    editPrompt:
                                      slotEditPrompts[slotKey] ||
                                      "头部视图微调",
                                  },
                                  `已更新头部视图：${slot.label}`,
                                )
                              }
                            >
                              与 Codex 调整此视角
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                <h3>
                  身体去头 A-Pose 正交四视图 (Body 4-Views ·
                  肩线/腰臀线/裙摆水平对齐)
                </h3>
                <div className="char-views-grid four">
                  {(
                    [
                      "body_front",
                      "body_left",
                      "body_right",
                      "body_back",
                    ] as MultiviewSlotKey[]
                  ).map((slotKey) => {
                    const slot = state.views[slotKey];
                    return (
                      <div key={slotKey} className="char-view-card">
                        <div className="char-view-card-head">
                          <strong>{slot.label}</strong>
                          <span className="mono">{slot.resolution}</span>
                        </div>
                        <RevisionImageThumb
                          projectId={s.project.id}
                          revisionId={slot.revisionId}
                          emptyState={emptyNode(viewNodeKey(slotKey), slotKey)}
                          alt={slot.label}
                          showGuides={state.placement.showHorizontalGuides}
                        />
                        {slot.revisionId && (
                          <div className="char-view-card-foot">
                            <input
                              value={slotEditPrompts[slotKey] || ""}
                              onChange={(e) =>
                                setSlotEditPrompts({
                                  ...slotEditPrompts,
                                  [slotKey]: e.target.value,
                                })
                              }
                              placeholder="单张身体视图微调指令…"
                            />
                            <button
                              type="button"
                              className="button small"
                              disabled={busy}
                              onClick={() =>
                                invokeAction(
                                  "edit-view",
                                  {
                                    runId: run.id,
                                    slotKey,
                                    editPrompt:
                                      slotEditPrompts[slotKey] ||
                                      "身体视图微调",
                                  },
                                  `已更新身体视图：${slot.label}`,
                                )
                              }
                            >
                              与 Codex 调整此视角
                            </button>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {/* STAGE 2: Head ×4 & Body ×4 Candidates */}
          {activeStage === "candidate" && (
            <div className="char-workbench-panel">
              <div className="char-workbench-head">
                <div>
                  <h2>
                    阶段 2 · 头部 ×4 与身体 ×4 3D 候选对比与非破坏分支选型
                  </h2>
                  <p>
                    浏览候选不会意外篡改选择；点击「采用此候选」显式锁定版本。若下游已完成重贴图，切换候选将自动派生新分支
                    Run，原分支贴图不丢失。
                  </p>
                </div>
                <div className="char-workbench-controls">
                  <button
                    type="button"
                    className="button small"
                    disabled={
                      busy || !nodeGuidance(run, "head_candidates").ready
                    }
                    onClick={() =>
                      invokeAction(
                        "advance-node",
                        { runId: run.id, nodeKey: "head_candidates" },
                        "已生成 4 个头部 3D 候选",
                      )
                    }
                  >
                    与 Codex 制作头部 4 候选
                  </button>
                  <button
                    type="button"
                    className="button small"
                    disabled={
                      busy || !nodeGuidance(run, "body_candidates").ready
                    }
                    onClick={() =>
                      invokeAction(
                        "advance-node",
                        { runId: run.id, nodeKey: "body_candidates" },
                        "已生成 4 个身体 3D 候选",
                      )
                    }
                  >
                    与 Codex 制作身体 4 候选
                  </button>
                  <button
                    type="button"
                    className="button small primary"
                    disabled={
                      busy ||
                      !nodeGuidance(run, "candidates_gate").ready ||
                      nodeGuidance(run, "candidates_gate").complete ||
                      !state.selectedHeadCandidateId ||
                      !state.selectedBodyCandidateId
                    }
                    onClick={() =>
                      invokeAction(
                        "decide-gate",
                        {
                          runId: run.id,
                          nodeKey: "candidates_gate",
                          decision: "accept",
                          comment: `选定头部 ${state.selectedHeadCandidateId} + 身体 ${state.selectedBodyCandidateId}`,
                        },
                        "已确认头/身 3D 候选，进入网格拆解与 5 组重贴图",
                      )
                    }
                  >
                    <Check size={14} />
                    锁定选型并进入拆解贴图
                  </button>
                </div>
              </div>

              <h3>
                头部 3D 候选 (Head Candidates 1–4) · 当前采用：
                {state.selectedHeadCandidateId || "未选择"}
              </h3>
              <div className="char-candidates-grid">
                {state.headCandidates.map((cand) => {
                  const isSelected = state.selectedHeadCandidateId === cand.id;
                  return (
                    <div
                      key={cand.id}
                      className={`char-candidate-card ${isSelected ? "selected" : ""}`}
                    >
                      <div className="char-view-card-head">
                        <strong>{cand.label}</strong>
                        <span
                          className={`char-badge ${isSelected ? "green" : ""}`}
                        >
                          {isSelected ? "★ 当前采用" : cand.status}
                        </span>
                      </div>
                      <div className="char-candidate-preview">
                        {cand.revisionId ? (
                          <RevisionModelViewer
                            projectId={s.project.id}
                            revisionId={cand.revisionId}
                            name={cand.label}
                          />
                        ) : (
                          <RevisionImageThumb
                            projectId={s.project.id}
                            alt={cand.label}
                            emptyState={emptyNode("head_candidates")}
                          />
                        )}
                        {cand.revisionId && (
                          <div className="char-candidate-overlay-stats">
                            <span>
                              顶点:{" "}
                              {cand.vertices?.toLocaleString() || "未统计"}
                            </span>
                            <span>
                              面数: {cand.faces?.toLocaleString() || "未统计"}
                            </span>
                          </div>
                        )}
                      </div>
                      <p className="char-cand-summary">
                        {cand.summary ||
                          cand.topologyGrade ||
                          "完成头部四候选制作后，在此比较并采用一个。"}
                      </p>
                      <div className="char-cand-actions">
                        <button
                          type="button"
                          className={`button small ${isSelected ? "primary" : ""}`}
                          disabled={busy || isSelected || !cand.revisionId}
                          onClick={() => handleSelectCandidate("head", cand.id)}
                        >
                          {isSelected ? "已采用此头部" : "采用此头部候选"}
                        </button>
                        {cand.revisionId && (
                          <button
                            type="button"
                            className="button small"
                            onClick={() =>
                              triggerDownload(s.project.id, cand.revisionId!, 0)
                            }
                          >
                            <Download size={13} />
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              <h3>
                身体 3D 候选 (Body Candidates 1–4) · 当前采用：
                {state.selectedBodyCandidateId || "未选择"}
              </h3>
              <div className="char-candidates-grid">
                {state.bodyCandidates.map((cand) => {
                  const isSelected = state.selectedBodyCandidateId === cand.id;
                  return (
                    <div
                      key={cand.id}
                      className={`char-candidate-card ${isSelected ? "selected" : ""}`}
                    >
                      <div className="char-view-card-head">
                        <strong>{cand.label}</strong>
                        <span
                          className={`char-badge ${isSelected ? "green" : ""}`}
                        >
                          {isSelected ? "★ 当前采用" : cand.status}
                        </span>
                      </div>
                      <div className="char-candidate-preview">
                        {cand.revisionId ? (
                          <RevisionModelViewer
                            projectId={s.project.id}
                            revisionId={cand.revisionId}
                            name={cand.label}
                          />
                        ) : (
                          <RevisionImageThumb
                            projectId={s.project.id}
                            alt={cand.label}
                            emptyState={emptyNode("body_candidates")}
                          />
                        )}
                        {cand.revisionId && (
                          <div className="char-candidate-overlay-stats">
                            <span>
                              顶点:{" "}
                              {cand.vertices?.toLocaleString() || "未统计"}
                            </span>
                            <span>
                              面数: {cand.faces?.toLocaleString() || "未统计"}
                            </span>
                          </div>
                        )}
                      </div>
                      <p className="char-cand-summary">
                        {cand.summary ||
                          cand.topologyGrade ||
                          "完成身体四候选制作后，在此比较并采用一个。"}
                      </p>
                      <div className="char-cand-actions">
                        <button
                          type="button"
                          className={`button small ${isSelected ? "primary" : ""}`}
                          disabled={busy || isSelected || !cand.revisionId}
                          onClick={() => handleSelectCandidate("body", cand.id)}
                        >
                          {isSelected ? "已采用此身体" : "采用此身体候选"}
                        </button>
                        {cand.revisionId && (
                          <button
                            type="button"
                            className="button small"
                            onClick={() =>
                              triggerDownload(s.project.id, cand.revisionId!, 0)
                            }
                          >
                            <Download size={13} />
                          </button>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* STAGE 3: Mesh Segmentation, 5 Texture Groups & Independent Spring Groups */}
          {activeStage === "texture" && (
            <div className="char-workbench-panel">
              <div className="char-workbench-head">
                <div>
                  <h2>阶段 3 · 网格部件拆解、五分类 4K 重贴图与独立摇摆组</h2>
                  <p>
                    已拆解 {state.segmentation.components.length}{" "}
                    个连通网格部件（共{" "}
                    {state.segmentation.totalVertices.toLocaleString()} 顶点 /{" "}
                    {state.segmentation.totalFaces.toLocaleString()} 面），归入
                    5 大标准贴图组并行重贴图，同时将物理摇摆件拆入独立
                    SpringGroups。
                  </p>
                </div>
                <div className="char-workbench-controls">
                  <button
                    type="button"
                    className="button small"
                    disabled={
                      busy || !nodeGuidance(run, "mesh_segmentation").ready
                    }
                    onClick={() =>
                      invokeAction(
                        "advance-node",
                        { runId: run.id, nodeKey: "mesh_segmentation" },
                        "已完成网格部件连通域拆解",
                      )
                    }
                  >
                    1. 部件拆解
                  </button>
                  <button
                    type="button"
                    className="button small"
                    disabled={busy || !nodeGuidance(run, "uv_guide_prep").ready}
                    onClick={() =>
                      invokeAction(
                        "advance-node",
                        { runId: run.id, nodeKey: "uv_guide_prep" },
                        "已生成 5 组 UV 色块引导图",
                      )
                    }
                  >
                    2. 烘培 UV 引导图
                  </button>
                  <button
                    type="button"
                    className="button small primary"
                    disabled={
                      busy || !nodeGuidance(run, "mtoon_assembly").ready
                    }
                    onClick={() =>
                      invokeAction(
                        "advance-node",
                        { runId: run.id, nodeKey: "mtoon_assembly" },
                        "已完成 MToon 卡通材质转换与总装合并",
                      )
                    }
                  >
                    <Layers size={14} />
                    3. MToon 材质总装合并
                  </button>
                </div>
              </div>

              <h3>五分类并行 4K 贴图组 (5 Canonical Texture Groups)</h3>
              <div className="char-texture-groups-grid">
                {TEXTURE_GROUP_IDS.map((gid) => {
                  const tg = state.textureGroups[gid];
                  return (
                    <div key={gid} className="char-tg-card">
                      <div className="char-view-card-head">
                        <strong>{tg.label}</strong>
                        <span
                          className={`char-badge ${tg.status === "retextured" ? "green" : "accent"}`}
                        >
                          {tg.status === "retextured"
                            ? "4K 已回传"
                            : tg.status === "uv_ready"
                              ? "UV 已就绪"
                              : "待展开"}
                        </span>
                      </div>
                      <div className="char-tg-thumbs">
                        <div>
                          <small>UV 引导图 (Guide)</small>
                          <RevisionImageThumb
                            projectId={s.project.id}
                            revisionId={tg.uvGuideRevisionId}
                            alt={`${tg.label} UV Guide`}
                            emptyState={emptyNode("uv_guide_prep")}
                          />
                        </div>
                        <div>
                          <small>参考图 / 4K 贴图</small>
                          <RevisionImageThumb
                            projectId={s.project.id}
                            revisionId={tg.textureImageRevisionId}
                            alt={`${tg.label} Texture`}
                            emptyState={emptyNode(`retexture_${gid}`)}
                          />
                        </div>
                      </div>
                      <div className="char-tg-meta">
                        <span>包含部件：{tg.componentNames.length} 个</span>
                        <span>分辨率：{tg.resolution}</span>
                        <span>
                          MToon 阴影色：<code>{tg.shadeColor}</code>
                        </span>
                      </div>
                      <button
                        type="button"
                        className="button small full"
                        disabled={
                          busy || !nodeGuidance(run, `retexture_${gid}`).ready
                        }
                        onClick={() =>
                          invokeAction(
                            "advance-node",
                            {
                              runId: run.id,
                              nodeKey: `retexture_${gid}`,
                            },
                            `已完成 ${tg.label} 4K 重贴图`,
                          )
                        }
                      >
                        单独执行 {gid} 4K 重贴图
                      </button>
                    </div>
                  );
                })}
              </div>

              <div className="char-split-two">
                <div className="char-subpanel">
                  <div className="char-subpanel-head">
                    <h3>
                      部件分群清单 ({state.segmentation.components.length} 部件)
                    </h3>
                    <select
                      value={componentFilter}
                      onChange={(e) => setComponentFilter(e.target.value)}
                    >
                      <option value="all">全部贴图组</option>
                      {TEXTURE_GROUP_IDS.map((gid) => (
                        <option key={gid} value={gid}>
                          {TEXTURE_GROUP_LABELS[gid]}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="char-component-table-wrap">
                    <table className="char-table">
                      <thead>
                        <tr>
                          <th>部件 ID / 名称</th>
                          <th>来源</th>
                          <th>顶点 / 面</th>
                          <th>所属贴图组 (5分类)</th>
                          <th>物理摇摆组</th>
                        </tr>
                      </thead>
                      <tbody>
                        {state.segmentation.components
                          .filter(
                            (c) =>
                              componentFilter === "all" ||
                              c.textureGroup === componentFilter,
                          )
                          .slice(0, 40)
                          .map((comp) => (
                            <tr key={comp.id}>
                              <td className="mono">{comp.name}</td>
                              <td>
                                {comp.source === "head" ? "头部" : "身体"}
                              </td>
                              <td className="mono">
                                {comp.vertices} / {comp.faces}
                              </td>
                              <td>
                                <select
                                  value={comp.textureGroup}
                                  onChange={(e) =>
                                    invokeAction("update-classification", {
                                      runId: run.id,
                                      componentId: comp.id,
                                      textureGroup: e.target.value,
                                    })
                                  }
                                >
                                  {TEXTURE_GROUP_IDS.map((gid) => (
                                    <option key={gid} value={gid}>
                                      {gid}
                                    </option>
                                  ))}
                                </select>
                              </td>
                              <td>
                                <select
                                  value={comp.springGroupId || ""}
                                  onChange={(e) =>
                                    invokeAction("update-classification", {
                                      runId: run.id,
                                      componentId: comp.id,
                                      springGroupId: e.target.value || null,
                                    })
                                  }
                                >
                                  <option value="">固定 (不摇摆)</option>
                                  {state.springGroups.map((sg) => (
                                    <option key={sg.id} value={sg.id}>
                                      {sg.name}
                                    </option>
                                  ))}
                                </select>
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="char-subpanel">
                  <h3>独立物理摇摆组 (SpringGroups · 与贴图组解耦)</h3>
                  <div className="char-spring-groups-list">
                    {state.springGroups.map((sg) => (
                      <div key={sg.id} className="char-spring-card">
                        <div className="char-view-card-head">
                          <strong>{sg.name}</strong>
                          <span className="char-badge">
                            {sg.componentNames.length} 个摇摆部件
                          </span>
                        </div>
                        <div className="char-spring-sliders">
                          <label>
                            刚度 (Stiffness: {sg.stiffness})
                            <input
                              type="range"
                              min="0.1"
                              max="2.5"
                              step="0.05"
                              value={sg.stiffness}
                              onChange={(e) =>
                                invokeAction("update-classification", {
                                  runId: run.id,
                                  springGroupUpdate: {
                                    id: sg.id,
                                    stiffness: Number(e.target.value),
                                  },
                                })
                              }
                            />
                          </label>
                          <label>
                            重力 (Gravity: {sg.gravityPower})
                            <input
                              type="range"
                              min="0"
                              max="1"
                              step="0.02"
                              value={sg.gravityPower}
                              onChange={(e) =>
                                invokeAction("update-classification", {
                                  runId: run.id,
                                  springGroupUpdate: {
                                    id: sg.id,
                                    gravityPower: Number(e.target.value),
                                  },
                                })
                              }
                            />
                          </label>
                          <label>
                            阻尼 (Drag: {sg.dragForce})
                            <input
                              type="range"
                              min="0.05"
                              max="1"
                              step="0.05"
                              value={sg.dragForce}
                              onChange={(e) =>
                                invokeAction("update-classification", {
                                  runId: run.id,
                                  springGroupUpdate: {
                                    id: sg.id,
                                    dragForce: Number(e.target.value),
                                  },
                                })
                              }
                            />
                          </label>
                        </div>
                        <small className="muted">
                          碰撞体组：{sg.colliderNames.join(", ")}
                        </small>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* STAGE 4: Proportions, Height Ruler & 61 Joint Markers */}
          {activeStage === "proportions" && (
            <div className="char-workbench-panel">
              <div className="char-workbench-head">
                <div>
                  <h2>
                    阶段 4 · 头身比例对齐、米制身高标尺与 61 点关节 Marker
                  </h2>
                  <p>
                    对照半透明全身平衡三视图与米制身高标尺调节头/发/身位移与缩放，并校核
                    21 个身体关节 + 40 个手指关节 Marker。
                  </p>
                </div>
                <div className="char-workbench-controls">
                  <button
                    type="button"
                    className={`button small ${state.placement.rigMode === "marker" ? "primary" : ""}`}
                    onClick={() =>
                      invokeAction("update-placement", {
                        runId: run.id,
                        rigMode: "marker",
                      })
                    }
                  >
                    61点 Marker 引导绑定模式
                  </button>
                  <button
                    type="button"
                    className={`button small ${state.placement.rigMode === "tripo" ? "primary" : ""}`}
                    onClick={() =>
                      invokeAction("update-placement", {
                        runId: run.id,
                        rigMode: "tripo",
                      })
                    }
                  >
                    Tripo 自动绑定模式
                  </button>
                  <button
                    type="button"
                    className="button small primary"
                    disabled={
                      busy ||
                      !nodeGuidance(run, "proportions_gate").ready ||
                      nodeGuidance(run, "proportions_gate").complete
                    }
                    onClick={() =>
                      invokeAction(
                        "decide-gate",
                        {
                          runId: run.id,
                          nodeKey: "proportions_gate",
                          decision: "accept",
                          comment: `确认总身高 ${state.placement.targetHeightMeters}m 及 ${state.placement.rigMode} 关节对位`,
                        },
                        "已确认头身比例与关节 Marker，进入骨骼蒙皮与物理验证",
                      )
                    }
                  >
                    <Check size={14} />
                    确认比例与 Marker 并进入绑定
                  </button>
                </div>
              </div>

              <div className="char-split-two">
                <div className="char-subpanel">
                  <h3>头/发/身空间对位与米制身高标尺</h3>
                  <div className="char-proportion-stage">
                    {state.placement.showHeightRuler && (
                      <div className="char-height-ruler">
                        <span>
                          {state.placement.targetHeightMeters.toFixed(2)}m
                          (头顶)
                        </span>
                        <span>
                          {(state.placement.targetHeightMeters * 0.82).toFixed(
                            2,
                          )}
                          m (肩颈)
                        </span>
                        <span>
                          {(state.placement.targetHeightMeters * 0.56).toFixed(
                            2,
                          )}
                          m (骨盆)
                        </span>
                        <span>
                          {(state.placement.targetHeightMeters * 0.28).toFixed(
                            2,
                          )}
                          m (膝盖)
                        </span>
                        <span>0.00m (地面)</span>
                      </div>
                    )}
                    <div className="char-proportion-preview">
                      <RevisionImageThumb
                        projectId={s.project.id}
                        revisionId={state.views.balance_front.revisionId}
                        emptyState={emptyNode(
                          "balance_three_views",
                          "balance_front",
                        )}
                        alt="全身平衡对位参考"
                        showGuides={true}
                      />
                    </div>
                  </div>

                  <div className="char-transform-controls">
                    <div className="char-transform-row">
                      <label>
                        目标总身高 (米)
                        <input
                          type="number"
                          step="0.01"
                          min="0.8"
                          max="2.5"
                          value={state.placement.targetHeightMeters}
                          onChange={(e) =>
                            invokeAction("update-placement", {
                              runId: run.id,
                              targetHeightMeters: Number(e.target.value),
                            })
                          }
                        />
                      </label>
                      <label className="char-toggle">
                        <input
                          type="checkbox"
                          checked={state.placement.linkHairWithHead}
                          onChange={(e) =>
                            invokeAction("update-placement", {
                              runId: run.id,
                              linkHairWithHead: e.target.checked,
                            })
                          }
                        />
                        头发跟随头部同步缩放/位移
                      </label>
                    </div>

                    {(["head", "hair", "body"] as const).map((partKey) => {
                      const pt = state.placement.parts[partKey];
                      return (
                        <div key={partKey} className="char-part-slider-box">
                          <strong>
                            {partKey === "head"
                              ? "头部 (Head)"
                              : partKey === "hair"
                                ? "头发 (Hair)"
                                : "身体 (Body)"}{" "}
                            · 高度 Y={pt.y}m · 缩放={pt.scale}x
                          </strong>
                          <div className="char-part-inputs">
                            <label>
                              高度 Y
                              <input
                                type="number"
                                step="0.01"
                                value={pt.y}
                                onChange={(e) =>
                                  invokeAction("update-placement", {
                                    runId: run.id,
                                    parts: {
                                      [partKey]: {
                                        ...pt,
                                        y: Number(e.target.value),
                                      },
                                    },
                                  })
                                }
                              />
                            </label>
                            <label>
                              缩放 Scale
                              <input
                                type="number"
                                step="0.05"
                                value={pt.scale}
                                onChange={(e) =>
                                  invokeAction("update-placement", {
                                    runId: run.id,
                                    parts: {
                                      [partKey]: {
                                        ...pt,
                                        scale: Number(e.target.value),
                                      },
                                    },
                                  })
                                }
                              />
                            </label>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                <div className="char-subpanel">
                  <h3>
                    关节 Marker 坐标校核 (21 身体关节 +{" "}
                    {state.placement.fingerMarkers.length} 手部关节)
                  </h3>
                  <div className="char-component-table-wrap">
                    <table className="char-table">
                      <thead>
                        <tr>
                          <th>关节部位</th>
                          <th>X (左右)</th>
                          <th>Y (高度)</th>
                          <th>Z (前后)</th>
                          <th>置信度</th>
                        </tr>
                      </thead>
                      <tbody>
                        {state.placement.bodyMarkers.map((m) => (
                          <tr key={m.key}>
                            <td>
                              <strong>{m.label}</strong>
                            </td>
                            <td className="mono">{m.x.toFixed(3)}</td>
                            <td className="mono">{m.y.toFixed(3)}</td>
                            <td className="mono">{m.z.toFixed(3)}</td>
                            <td>
                              <span className="char-badge green">
                                {Math.round(m.confidence * 100)}%
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* STAGE 5: Humanoid Rig, Weight Transfer & 90-Frame Spring Verification */}
          {activeStage === "rig" && (
            <div className="char-workbench-panel">
              <div className="char-workbench-head">
                <div>
                  <h2>
                    阶段 5 · Humanoid 骨骼映射、素体权重传递与 90 帧物理自动检验
                  </h2>
                  <p>
                    构建 {state.rigging.humanoidBoneCount} 根 VRM 1.0
                    标准骨骼，从标准素体 (
                    {state.rigging.weightTransfer.donorMesh})
                    传递平滑蒙皮权重，并运行 90 帧物理开关对比模拟。
                  </p>
                </div>
                <div className="char-workbench-controls">
                  <button
                    type="button"
                    className="button small"
                    disabled={busy || !nodeGuidance(run, "humanoid_rig").ready}
                    onClick={() =>
                      invokeAction(
                        "advance-node",
                        { runId: run.id, nodeKey: "humanoid_rig" },
                        "已生成 55 根 Humanoid 标准骨骼映射",
                      )
                    }
                  >
                    1. 生成 Humanoid 骨架
                  </button>
                  <button
                    type="button"
                    className="button small"
                    disabled={
                      busy || !nodeGuidance(run, "weight_transfer").ready
                    }
                    onClick={() =>
                      invokeAction(
                        "advance-node",
                        { runId: run.id, nodeKey: "weight_transfer" },
                        "已完成素体蒙皮权重传递与接缝平滑",
                      )
                    }
                  >
                    2. 素体权重传递
                  </button>
                  <button
                    type="button"
                    className="button small"
                    disabled={busy || !nodeGuidance(run, "spring_verify").ready}
                    onClick={() =>
                      invokeAction(
                        "advance-node",
                        { runId: run.id, nodeKey: "spring_verify" },
                        "已完成 90 帧 SpringBone 物理开关对比检验",
                      )
                    }
                  >
                    3. 运行 90 帧物理检验
                  </button>
                  <button
                    type="button"
                    className="button small primary"
                    disabled={
                      busy ||
                      !nodeGuidance(run, "motion_gate").ready ||
                      nodeGuidance(run, "motion_gate").complete
                    }
                    onClick={() =>
                      invokeAction(
                        "decide-gate",
                        {
                          runId: run.id,
                          nodeKey: "motion_gate",
                          decision: "accept",
                          comment: "变形与 90 帧物理防穿模审核通过",
                        },
                        "已通过动作与物理审核，解锁 VRM 1.0 成品交付",
                      )
                    }
                  >
                    <Check size={14} />
                    通过变形与物理审核
                  </button>
                </div>
              </div>

              <div className="char-split-two">
                <div className="char-subpanel">
                  <h3>90 帧 SpringBone 物理 ON/OFF 自动对比检验报告</h3>
                  <table className="char-table">
                    <thead>
                      <tr>
                        <th>摇摆物理链</th>
                        <th>骨骼节数</th>
                        <th>峰值位移 (m)</th>
                        <th>收敛帧</th>
                        <th>穿模碰撞</th>
                        <th>检验结论</th>
                      </tr>
                    </thead>
                    <tbody>
                      {state.rigging.springVerification.chains.map((ch) => (
                        <tr key={ch.groupId}>
                          <td>
                            <strong>{ch.name}</strong>
                          </td>
                          <td className="mono">{ch.boneCount} 节</td>
                          <td className="mono">
                            {ch.peakDisplacementMeters} m
                          </td>
                          <td className="mono">第 {ch.settledFrame} 帧</td>
                          <td className="mono">{ch.colliderHits} 次穿模</td>
                          <td>
                            <span
                              className={`char-badge ${ch.passed ? "green" : "accent"}`}
                            >
                              {ch.passed ? "✓ 90帧检验通过" : "待运行模拟"}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="char-subpanel">
                  <h3>VRM 1.0 Humanoid 核心骨骼映射 & 素体权重传递</h3>
                  <p className="muted">
                    供体素体：
                    <code>{state.rigging.weightTransfer.donorMesh}</code> ·
                    接缝平滑：
                    {state.rigging.weightTransfer.smoothedSeams.join(", ")}
                  </p>
                  <div className="char-component-table-wrap">
                    <table className="char-table">
                      <thead>
                        <tr>
                          <th>VRM Humanoid 骨骼槽位</th>
                          <th>绑定源骨骼 (Blender Armature)</th>
                          <th>状态</th>
                        </tr>
                      </thead>
                      <tbody>
                        {state.rigging.mappedBones.map((b) => (
                          <tr key={b.humanoidBone}>
                            <td className="mono">{b.humanoidBone}</td>
                            <td className="mono">{b.sourceBone}</td>
                            <td>
                              <span className="char-badge green">已映射</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* STAGE 6: VRM 1.0 Motion Preview & Metadata Delivery */}
          {activeStage === "delivery" && (
            <div className="char-workbench-panel">
              <div className="char-workbench-head">
                <div>
                  <h2>阶段 6 · VRM 1.0 动作预览、商业授权元数据与成果交接</h2>
                  <p>
                    在线测试导出成品（内置人形动作、自定义
                    VRMA、物理开关、骨骼与线框切换），写入 VRM 1.0
                    元数据并打包不可变交付包。
                  </p>
                </div>
                <div className="char-workbench-controls">
                  {state.vrmDelivery.vrmRevisionId && (
                    <button
                      type="button"
                      className="button small"
                      onClick={() =>
                        triggerDownload(
                          s.project.id,
                          state.vrmDelivery.vrmRevisionId!,
                          0,
                        )
                      }
                    >
                      <Download size={14} />
                      下载 avatar.vrm
                    </button>
                  )}
                  {state.vrmDelivery.blendRevisionId && (
                    <button
                      type="button"
                      className="button small"
                      onClick={() =>
                        triggerDownload(
                          s.project.id,
                          state.vrmDelivery.blendRevisionId!,
                          0,
                        )
                      }
                    >
                      <Download size={14} />
                      下载可编辑 avatar.blend
                    </button>
                  )}
                  <button
                    type="button"
                    className="button small primary"
                    disabled={busy}
                    onClick={handleExportVrmBundle}
                  >
                    <PackageCheck size={14} />
                    导出 VRM 1.0 成品交付包
                  </button>
                </div>
              </div>

              <div className="char-split-two char-delivery-layout">
                <div className="char-subpanel">
                  <h3>3D VRM 实时动作与物理预览台</h3>
                  {state.vrmDelivery.vrmRevisionId ? (
                    <RevisionVrmViewer
                      projectId={s.project.id}
                      revisionId={state.vrmDelivery.vrmRevisionId}
                      name={`${state.characterName}.vrm`}
                      settings={state.vrmDelivery}
                      onSettingsChange={(viewer) =>
                        invokeAction("update-vrm-meta", {
                          runId: run.id,
                          viewer,
                        })
                      }
                    />
                  ) : (
                    <div className="char-vrm-stage char-slot-placeholder">
                      {emptyNode("vrm_export")}
                    </div>
                  )}
                </div>

                <div className="char-subpanel">
                  <h3>VRM 1.0 标准元数据与商业授权配置 (VRMC_vrm)</h3>
                  <div className="char-vrm-meta-form">
                    <label className="field">
                      VRM 模型名称 (Avatar Name)
                      <input
                        value={state.vrmDelivery.meta.name}
                        onChange={(e) =>
                          invokeAction("update-vrm-meta", {
                            runId: run.id,
                            meta: { name: e.target.value },
                          })
                        }
                      />
                    </label>
                    <label className="field">
                      版本号 (Version)
                      <input
                        value={state.vrmDelivery.meta.version}
                        onChange={(e) =>
                          invokeAction("update-vrm-meta", {
                            runId: run.id,
                            meta: { version: e.target.value },
                          })
                        }
                      />
                    </label>
                    <label className="field">
                      作者署名 (Authors)
                      <input
                        value={state.vrmDelivery.meta.authors.join(", ")}
                        onChange={(e) =>
                          invokeAction("update-vrm-meta", {
                            runId: run.id,
                            meta: {
                              authors: e.target.value
                                .split(",")
                                .map((x) => x.trim())
                                .filter(Boolean),
                            },
                          })
                        }
                      />
                    </label>
                    <label className="field">
                      商业使用许可 (Commercial Usage)
                      <select
                        value={state.vrmDelivery.meta.commercialUsage}
                        onChange={(e) =>
                          invokeAction("update-vrm-meta", {
                            runId: run.id,
                            meta: { commercialUsage: e.target.value },
                          })
                        }
                      >
                        <option value="personalNonProfit">
                          仅限个人非商业使用 (personalNonProfit)
                        </option>
                        <option value="personalProfit">
                          允许个人商业使用 (personalProfit)
                        </option>
                        <option value="corporation">
                          允许企业与商业项目使用 (corporation)
                        </option>
                      </select>
                    </label>
                    <label className="field">
                      二次修改权限 (Modification)
                      <select
                        value={state.vrmDelivery.meta.modify}
                        onChange={(e) =>
                          invokeAction("update-vrm-meta", {
                            runId: run.id,
                            meta: { modify: e.target.value },
                          })
                        }
                      >
                        <option value="prohibited">
                          禁止修改 (prohibited)
                        </option>
                        <option value="allowModification">
                          允许修改 (allowModification)
                        </option>
                        <option value="allowModificationRedistribution">
                          允许修改并再分发 (allowModificationRedistribution)
                        </option>
                      </select>
                    </label>
                    <label className="field">
                      版权声明 (Copyright)
                      <input
                        value={state.vrmDelivery.meta.copyrightInformation}
                        onChange={(e) =>
                          invokeAction("update-vrm-meta", {
                            runId: run.id,
                            meta: { copyrightInformation: e.target.value },
                          })
                        }
                      />
                    </label>
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
      {run && sourcePickerRunId === run.id && (
        <CharacterSourceDialog
          key={run.id}
          snapshot={s}
          run={run}
          selectedRevisionId={replacementSourceRevId}
          onSelect={setReplacementSourceRevId}
          onUpload={(file) =>
            uploadSourceImage(file, setReplacementSourceRevId)
          }
          busy={busy}
          error={errorMsg}
          notify={notify}
          onClose={() => setSourcePickerRunId(undefined)}
          onAdopt={async () => {
            const updated = await invokeAction(
              "edit-view",
              {
                runId: run.id,
                slotKey: "original",
                revisionId: replacementSourceRevId,
                expectedRevision: run.revision,
              },
              "已更换角色原画，下游制作需重新生成和审核",
            );
            if (updated) setSourcePickerRunId(undefined);
          }}
        />
      )}
      {run && codexTarget?.runId === run.id && (
        <CharacterTaskDialog
          key={`${run.id}:${codexTarget.nodeKey}:${codexTarget.slotKey || "group"}`}
          snapshot={s}
          run={run}
          nodeKey={codexTarget.nodeKey}
          slotKey={codexTarget.slotKey}
          initialPrompt={
            codexTarget.slotKey ? slotEditPrompts[codexTarget.slotKey] : ""
          }
          onClose={() => setCodexTarget(undefined)}
          notify={notify}
        />
      )}
    </div>
  );
}
