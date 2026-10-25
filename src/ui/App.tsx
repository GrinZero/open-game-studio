import {
  resolveWorkbenchTarget,
  type WorkbenchTarget,
  type WorkbenchView,
} from "../shared/workbench-target";
import {
  ResourceMenu,
  DocumentEditor,
  ResourceLibrary,
  GenerationComposer,
  GenerationNode,
  GenerationDetails,
  CanvasDocumentPreview,
  CanvasMediaPreview,
  Markdown,
  resourceChoices,
  generationLabels,
  taskLabels,
  type AddResourceKind,
} from "./ResourceComposer";
import studioIcon from "../../assets/icon.svg";
import { CodexHandoff } from "./CodexHandoff";
import { CodexReferenceBar } from "./CodexReferences";
import { VersionBadge } from "./VersionBadge";
import { CharacterWorkflowPage } from "./CharacterWorkflowPage";
import {
  launchProject,
  launchRequest,
  subscribeLaunch,
} from "./workbench-launch";
import { ProjectLiveSync } from "./ProjectLiveSync";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  MarkerType,
  Handle,
  NodeResizer,
  Position,
  applyNodeChanges,
  applyEdgeChanges,
  type NodeProps,
  type Node,
  type Edge,
  type ReactFlowInstance,
  useNodesInitialized,
  useReactFlow,
} from "@xyflow/react";
import {
  Boxes,
  Plus,
  ArrowUpRight,
  FolderOpen,
  Search,
  LayoutGrid,
  Image as ImageIcon,
  Workflow,
  Settings,
  ChevronRight,
  ChevronDown,
  ArrowLeft,
  X,
  Upload,
  FileText,
  Box,
  StickyNote,
  MousePointer2,
  Hand,
  Check,
  Play,
  MoreHorizontal,
  Copy,
  Download,
  Undo2,
  Redo2,
  Layers,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  AlertTriangle,
  RefreshCw,
  Archive,
  RotateCcw,
  CheckCircle2,
  Square,
  Pencil,
  Grip,
  Link2,
  Folder,
  Loader2,
  Sun,
  Moon,
  Trash2,
  Clipboard,
  Maximize,
  Unplug,
} from "lucide-react";
import type {
  Asset,
  AssetRevision,
  Canvas,
  CanvasInputSnapshot,
  Project,
  ProjectSnapshot,
  StudioNode,
  WorkflowRun,
  WorkflowPlan,
} from "../shared/types";
import {
  canvasConnectionError,
  resolveCanvasInputs,
} from "../shared/canvas-inputs";
import {
  call,
  fileUrl,
  request,
  useAssetFileUrl,
  isEmbeddedApp,
  downloadAssetFile,
} from "./api";
import { useTheme } from "./theme";
import { projectPath } from "../shared/project-location";
import { ProjectCoverImage, ProjectCoverModal } from "./ProjectCover";
import { ModelPreview } from "./ModelPreview";
import { AssetPreviewSwitcher } from "./AssetPreviewSwitcher";
import type { PreviewOption } from "./asset-preview-options";
import { AssetGroups, AssetCategoryEditor } from "./AssetCategories";
import { assetCategories, assetCategory } from "../shared/asset-categories";
import { CanvasContextMenu, type CanvasMenuAction } from "./CanvasContextMenu";
import {
  acceptsCanvasDrop,
  captureCanvasImageDrop,
  resolveCanvasImages,
  type CanvasImageDrop,
} from "./canvas-drop";
import {
  canvasClipboard,
  copyCanvasSelection,
  pasteCanvasSelection,
} from "./canvas-clipboard";

const uuid = (p = "n") =>
  `${p}_${crypto.randomUUID?.() || Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) => byte.toString(16).padStart(2, "0")).join("")}`;
const stamp = (s: string) =>
  new Date(s).toLocaleDateString("zh-CN", { month: "short", day: "numeric" });

const engineName: Record<string, string> = {
  godot: "Godot",
  unity: "Unity",
  unreal: "Unreal",
  other: "通用",
};

const kindName: Record<string, string> = {
  image: "图片",
  model: "3D 模型",
  document: "文档",
  source: "源文件",
  audio: "音频",
  video: "视频",
};

const statusName: Record<string, string> = {
  pending: "等待前置",
  awaiting_agent: "等待 Codex",
  awaiting_manual: "等待提交",
  awaiting_review: "待审核",
  succeeded: "已完成",
  cancelled: "已取消",
  active: "进行中",
};

function freePosition(
  canvas: Canvas | undefined,
  preferred: { x: number; y: number },
  width = 240,
  height = 260,
) {
  let result = { ...preferred };
  for (let attempt = 0; attempt < 100; attempt++) {
    const collision = canvas?.nodes.some((n) => {
      const w = n.type === "workflow" ? 280 : n.type === "note" ? 220 : 230;
      const h = n.type === "workflow" ? 420 : n.type === "note" ? 210 : 250;
      return (
        result.x < n.position.x + w + 24 &&
        result.x + width + 24 > n.position.x &&
        result.y < n.position.y + h + 24 &&
        result.y + height + 24 > n.position.y
      );
    });
    if (!collision) return result;
    result = {
      x: preferred.x + ((attempt % 3) + 1) * (width + 36),
      y: preferred.y + Math.floor(attempt / 3) * (height + 36),
    };
  }
  return result;
}

function IconButton({
  children,
  label,
  onClick,
  disabled = false,
  expanded,
  controls,
}: {
  children: ReactNode;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  expanded?: boolean;
  controls?: string;
}) {
  return (
    <button
      type="button"
      className="icon-button"
      title={label}
      aria-label={label}
      aria-expanded={expanded}
      aria-controls={controls}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}

// Follow the available iframe width, while allowing a manual override until
// the layout crosses the breakpoint again. No storage is needed in the sandbox.
function useResponsivePanel(query: string, initialOpen?: boolean) {
  const [open, setOpen] = useState(
    () => initialOpen ?? !window.matchMedia(query).matches,
  );
  useEffect(() => {
    const media = window.matchMedia(query);
    const update = () => setOpen(!media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, [query]);
  return [open, setOpen] as const;
}

function Badge({
  children,
  tone = "muted",
}: {
  children: ReactNode;
  tone?: string;
}) {
  return <span className={`badge ${tone}`}>{children}</span>;
}

function Modal({
  title,
  children,
  onClose,
  className,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  className?: string;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    ref.current?.showModal();
  }, []);
  return (
    <dialog
      ref={ref}
      className={className}
      aria-label={title}
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="modal-head">
        <h2>{title}</h2>
        <IconButton label="关闭" onClick={onClose}>
          <X size={16} />
        </IconButton>
      </div>
      {children}
    </dialog>
  );
}

export function App() {
  const { resolvedTheme, toggleTheme } = useTheme();
  const [projects, setProjects] = useState<Project[]>([]);
  const [snapshot, setSnapshot] = useState<ProjectSnapshot>();
  const [projectId, setProjectId] = useState<string | undefined>(launchProject);
  const [launchEpoch, setLaunchEpoch] = useState(0);
  const [loadedLaunch, setLoadedLaunch] = useState<{
    target: WorkbenchTarget;
    epoch: number;
  }>();
  const activeLaunch = useRef(0);
  const activeProject = useRef(projectId);
  activeProject.current = projectId;
  const [homeNavigation, setHomeNavigation] = useResponsivePanel(
    "(max-width: 1100px)",
  );
  const homeToggleRef = useRef<HTMLButtonElement>(null);
  const homeCloseRef = useRef<HTMLButtonElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [toast, setToast] = useState("");
  const [newProject, setNewProject] = useState<"create" | "link" | null>(null);
  const [coverProject, setCoverProject] = useState<Project>();
  const [group, setGroup] = useState("全部项目");
  const [query, setQuery] = useState("");
  const [archived, setArchived] = useState(false);

  useEffect(
    () =>
      subscribeLaunch(() => {
        setProjectId(launchProject());
        activeLaunch.current += 1;
        setLaunchEpoch(activeLaunch.current);
      }),
    [],
  );
  useEffect(() => {
    if (
      !homeNavigation ||
      projectId ||
      !window.matchMedia("(max-width: 1100px)").matches
    )
      return;
    homeCloseRef.current?.focus();
    function dismiss(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setHomeNavigation(false);
        homeToggleRef.current?.focus();
      }
    }
    window.addEventListener("keydown", dismiss);
    return () => window.removeEventListener("keydown", dismiss);
  }, [homeNavigation, projectId, setHomeNavigation]);
  function closeHomeNavigation() {
    if (window.matchMedia("(max-width: 1100px)").matches)
      setHomeNavigation(false);
  }

  const notify = useCallback((s: string) => {
    setToast(s);
  }, []);

  useEffect(() => {
    if (toast) {
      const timer = setTimeout(() => setToast(""), 3500);
      return () => clearTimeout(timer);
    }
  }, [toast]);

  const refreshProjects = useCallback(async () => {
    setProjects(await request<Project[]>("/api/projects"));
  }, []);

  const refresh = useCallback(async () => {
    if (projectId) {
      const epoch = activeLaunch.current;
      const next = await request<ProjectSnapshot>(`/api/projects/${projectId}`);
      if (activeProject.current === projectId && epoch === activeLaunch.current)
        setSnapshot(next);
    }
    await refreshProjects();
  }, [projectId, refreshProjects]);

  useEffect(() => {
    refreshProjects()
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [refreshProjects]);

  useEffect(() => {
    if (!projectId) {
      setSnapshot(undefined);
      return;
    }
    let disposed = false;
    setLoading(true);
    request<ProjectSnapshot>(`/api/projects/${projectId}`)
      .then((s) => {
        if (disposed) return;
        const intent = launchRequest();
        const target = resolveWorkbenchTarget(
          s,
          intent.projectId === projectId ? intent : { projectId },
        );
        setLoadedLaunch({ target, epoch: launchEpoch });
        setSnapshot(s);
        setError("");
      })
      .catch((e) => {
        if (!disposed) setError(e.message);
      })
      .finally(() => {
        if (!disposed) setLoading(false);
      });
    return () => {
      disposed = true;
    };
  }, [projectId, launchEpoch]);

  async function perform(task: () => Promise<unknown>) {
    try {
      await task();
      await refresh();
    } catch (e) {
      setError((e as Error).message);
      throw e;
    }
  }

  const groups = [
    ...new Set(projects.map((p) => p.collection).filter(Boolean)),
  ];
  const filtered = projects.filter(
    (p) =>
      p.archived === archived &&
      (group === "全部项目" || p.collection === group) &&
      p.name.toLowerCase().includes(query.toLowerCase()),
  );

  return (
    <div
      className={`app-shell${homeNavigation ? "" : " home-navigation-collapsed"}`}
    >
      <CodexReferenceBar />
      {snapshot &&
      projectId &&
      snapshot.project.id === projectId &&
      loadedLaunch ? (
        <Workspace
          key={`${projectId}:${loadedLaunch.epoch}`}
          launch={loadedLaunch.target}
          snapshot={snapshot}
          refresh={refresh}
          onBack={() => {
            setProjectId(undefined);
            refreshProjects();
          }}
          perform={perform}
          notify={notify}
          resolvedTheme={resolvedTheme}
          toggleTheme={toggleTheme}
        />
      ) : (
        <>
          {homeNavigation && (
            <button
              className="home-backdrop"
              aria-label="关闭项目列表导航"
              tabIndex={-1}
              onClick={() => {
                setHomeNavigation(false);
                homeToggleRef.current?.focus();
              }}
            />
          )}
          <aside
            className="home-sidebar"
            id="home-navigation"
            hidden={!homeNavigation}
            aria-label="项目列表导航"
          >
            <div className="workspace-brand">
              <Brand />
              <button
                ref={homeCloseRef}
                className="icon-button"
                aria-label="收起项目列表导航"
                onClick={() => {
                  setHomeNavigation(false);
                  homeToggleRef.current?.focus();
                }}
              >
                <PanelLeftClose size={16} />
              </button>
            </div>
            <button
              className={
                !archived && group === "全部项目"
                  ? "side-item active"
                  : "side-item"
              }
              onClick={() => {
                setGroup("全部项目");
                setArchived(false);
                closeHomeNavigation();
              }}
            >
              <LayoutGrid size={15} />
              全部项目<span>{projects.filter((p) => !p.archived).length}</span>
            </button>
            <div className="side-heading">项目集</div>
            {groups.map((g) => (
              <button
                className={
                  !archived && group === g ? "side-item active" : "side-item"
                }
                key={g}
                onClick={() => {
                  setGroup(g);
                  setArchived(false);
                  closeHomeNavigation();
                }}
              >
                <Folder size={15} />
                {g}
              </button>
            ))}
            <button
              className={archived ? "side-item active" : "side-item"}
              onClick={() => {
                setArchived(true);
                setGroup("全部项目");
                closeHomeNavigation();
              }}
            >
              <Archive size={15} />
              已归档
            </button>
            <div className="sidebar-bottom">
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <div className="local-status">
                  <i />
                  本地工作台
                </div>
                <button
                  className="icon-button"
                  title="切换主题"
                  aria-label="切换主题"
                  onClick={toggleTheme}
                >
                  {resolvedTheme === "dark" ? (
                    <Sun size={14} />
                  ) : (
                    <Moon size={14} />
                  )}
                </button>
              </div>
              <span className="mono muted" style={{ fontSize: "10px" }}>
                OPEN GAME STUDIO · ALPHA
              </span>
            </div>
          </aside>
          <main className="home-main">
            <header className="home-top">
              <div className="home-location">
                <button
                  ref={homeToggleRef}
                  className="icon-button"
                  aria-label={
                    homeNavigation ? "收起项目列表导航" : "展开项目列表导航"
                  }
                  aria-expanded={homeNavigation}
                  aria-controls="home-navigation"
                  onClick={() => setHomeNavigation(!homeNavigation)}
                >
                  {homeNavigation ? (
                    <PanelLeftClose size={16} />
                  ) : (
                    <PanelLeftOpen size={16} />
                  )}
                </button>
                <div className="breadcrumbs">
                  <span>工作台</span>
                  <ChevronRight size={13} />
                  <strong>{archived ? "已归档项目" : group}</strong>
                </div>
                <VersionBadge />
              </div>
              <div className="home-actions">
                <label className="search compact">
                  <Search size={13} />
                  <input
                    placeholder="搜索项目…"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                </label>
                <button
                  className="button small"
                  onClick={() => setNewProject("link")}
                >
                  <FolderOpen size={14} />
                  关联已有项目
                </button>
                <button
                  className="button small primary"
                  onClick={() => setNewProject("create")}
                >
                  <Plus size={14} />
                  新建项目
                </button>
              </div>
            </header>
            <section className="projects-section">
              <div className="section-head">
                <h2>
                  {archived ? "已归档项目" : group}
                  <span>({filtered.length})</span>
                </h2>
              </div>
              {loading ? (
                <div className="empty-state">
                  <Loader2 className="spin" size={20} />
                  <p>正在读取工作台…</p>
                </div>
              ) : (
                <div className="project-grid">
                  {filtered.map((p) => (
                    <article className="project-card" key={p.id}>
                      <button
                        className="project-card-open"
                        onClick={() =>
                          p.available
                            ? setProjectId(p.id)
                            : setError(
                                "项目目录丢失。可在关联已有项目后恢复原路径，或使用 MCP project_manage 的 relocate 操作。",
                              )
                        }
                      >
                        <div className="project-card-cover">
                          <ProjectCoverImage
                            projectId={p.id}
                            projectName={p.name}
                            cover={p.available ? p.cover : undefined}
                            alt={`${p.name}封面`}
                          />
                        </div>
                        <div className="project-card-info">
                          <div className="project-card-header">
                            <div
                              style={{
                                display: "flex",
                                gap: "6px",
                                alignItems: "center",
                              }}
                            >
                              <Badge tone="blue">{engineName[p.engine]}</Badge>
                              {p.collection && <Badge>{p.collection}</Badge>}
                              {!p.available && (
                                <Badge tone="amber">目录丢失</Badge>
                              )}
                            </div>
                            <span
                              className="muted mono"
                              style={{ fontSize: "11px" }}
                            >
                              {stamp(p.updatedAt)}
                            </span>
                          </div>
                          <div className="project-card-title">
                            <h3>{p.name}</h3>
                            <ArrowUpRight size={16} className="card-arrow" />
                          </div>
                          <div
                            className="project-card-path mono"
                            title={p.root}
                          >
                            {p.root}
                          </div>
                        </div>
                      </button>
                      <button
                        className="button small project-card-cover-action"
                        aria-label={`${p.name}：${p.cover ? "更换封面" : "设置封面"}`}
                        disabled={!p.available}
                        onClick={() => setCoverProject(p)}
                      >
                        <ImageIcon size={13} />
                        {p.cover ? "更换封面" : "设置封面"}
                      </button>
                    </article>
                  ))}
                  {!archived && (
                    <button
                      className="new-project-card"
                      onClick={() => setNewProject("create")}
                    >
                      <Plus size={20} />
                      <strong>开始一个新项目</strong>
                    </button>
                  )}
                </div>
              )}
            </section>
          </main>
        </>
      )}
      {newProject && (
        <ProjectModal
          mode={newProject}
          onClose={() => setNewProject(null)}
          onCreate={async (data) => {
            try {
              const p = await call<Project>("project_manage", data);
              await refreshProjects();
              setNewProject(null);
              setProjectId(p.id);
              notify("项目已创建并打开");
            } catch (e) {
              setError((e as Error).message);
            }
          }}
        />
      )}
      {coverProject && (
        <ProjectCoverModal
          project={coverProject}
          onClose={() => setCoverProject(undefined)}
          onSaved={async () => {
            await refreshProjects();
            notify("项目封面已保存");
          }}
        />
      )}
      {error && (
        <div className="error-toast" role="alert">
          <AlertTriangle size={15} />
          <div>{error}</div>
          <button onClick={() => setError("")} aria-label="关闭错误">
            <X size={14} />
          </button>
        </div>
      )}
      {toast && (
        <div className="toast" role="status">
          <CheckCircle2 size={15} />
          {toast}
        </div>
      )}
    </div>
  );
}

function Brand() {
  return (
    <div className="brand">
      <div className="brand-mark">
        <img src={studioIcon} width={22} height={22} alt="" />
      </div>
      <div className="brand-info">
        <span className="brand-name">Open Game Studio</span>
        <span className="brand-tag mono">FOR CODEX</span>
      </div>
    </div>
  );
}

function ProjectModal({
  mode,
  onCreate,
  onClose,
}: {
  mode: "create" | "link";
  onCreate: (a: unknown) => Promise<void>;
  onClose: () => void;
}) {
  const [name, setName] = useState("");
  const [root, setRoot] = useState("");
  const [createSubdirectory, setCreateSubdirectory] = useState(false);
  const [picking, setPicking] = useState(false);
  const [pickerError, setPickerError] = useState("");
  const [engine, setEngine] = useState("godot");
  const [collection, setCollection] = useState("");
  const [busy, setBusy] = useState(false);
  const finalRoot =
    mode === "create" && createSubdirectory && root
      ? projectPath(root, name)
      : root;
  const chooseDirectory = async () => {
    setPickerError("");
    setPicking(true);
    try {
      const result = await request<{ path: string | null }>(
        "/api/directories/pick",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            mode,
            initialPath: root || undefined,
          }),
        },
      );
      if (result.path) {
        setRoot(result.path);
      }
    } catch (error) {
      setPickerError((error as Error).message);
    } finally {
      setPicking(false);
    }
  };
  const create = async () => {
    setBusy(true);
    try {
      await onCreate({
        action: mode,
        name,
        root: finalRoot,
        engine,
        collection,
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      title={mode === "create" ? "新建项目" : "关联已有项目"}
      onClose={busy || picking ? () => {} : onClose}
    >
      <form
        onSubmit={async (e) => {
          e.preventDefault();
          if (busy || picking) return;
          await create();
        }}
      >
        <label className="field">
          项目名称
          <input
            autoFocus
            required
            value={name}
            maxLength={120}
            disabled={busy || picking}
            onChange={(e) => setName(e.target.value)}
            placeholder="例如：星铸者"
          />
        </label>
        <div className="field">
          <label htmlFor="project-directory">
            {mode === "create" && createSubdirectory ? "父目录" : "项目目录"}
          </label>
          <div className="directory-input">
            <input
              id="project-directory"
              aria-describedby="project-directory-help"
              required
              value={root}
              maxLength={4096}
              disabled={busy || picking}
              onChange={(e) => setRoot(e.target.value)}
              placeholder={
                mode === "create"
                  ? createSubdirectory
                    ? "选择用于创建子文件夹的父目录"
                    : "选择或填写项目文件夹"
                  : "选择已有项目文件夹"
              }
            />
            <button
              type="button"
              className="button"
              disabled={busy || picking}
              onClick={chooseDirectory}
            >
              {picking ? (
                <Loader2 size={14} className="spin" />
              ) : (
                <FolderOpen size={14} />
              )}
              {picking ? "选择中…" : "选择文件夹"}
            </button>
          </div>
          <small id="project-directory-help">
            {mode === "create"
              ? createSubdirectory
                ? "将在父目录内按项目名称创建子文件夹。"
                : "直接使用此文件夹；目录不存在时会创建，已有文件保留。"
              : "选择已有游戏目录进行关联，不移动或重排原有文件。"}
          </small>
          {mode === "create" && (
            <label className="directory-option">
              <input
                type="checkbox"
                checked={createSubdirectory}
                disabled={busy || picking}
                onChange={(e) => setCreateSubdirectory(e.target.checked)}
              />
              按项目名称创建子文件夹
            </label>
          )}
          {root && (
            <div className="directory-preview">
              <span>最终项目路径</span>
              <output className="mono" aria-label="最终项目路径">
                {finalRoot}
              </output>
            </div>
          )}
          {pickerError && (
            <p className="directory-error" role="alert">
              {pickerError}
            </p>
          )}
        </div>
        <div className="form-row">
          <label className="field">
            目标引擎
            <select value={engine} onChange={(e) => setEngine(e.target.value)}>
              {Object.entries(engineName).map(([v, n]) => (
                <option key={v} value={v}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            项目集（可选）
            <input
              value={collection}
              onChange={(e) => setCollection(e.target.value)}
              placeholder="例如：主线制作"
            />
          </label>
        </div>
        <div className="modal-footer">
          <button
            type="button"
            className="button small"
            disabled={busy || picking}
            onClick={onClose}
          >
            取消
          </button>
          <button
            className="button small primary"
            disabled={busy || picking || !name.trim() || !root.trim()}
            type={isEmbeddedApp() ? "button" : "submit"}
            onClick={isEmbeddedApp() ? create : undefined}
          >
            {busy ? <Loader2 className="spin" size={14} /> : <Plus size={14} />}{" "}
            {mode === "create" ? "创建项目" : "关联项目"}
          </button>
        </div>
      </form>
    </Modal>
  );
}

type View = WorkbenchView;

function Workspace({
  launch,
  snapshot: s,
  refresh,
  onBack,
  perform,
  notify,
  resolvedTheme,
  toggleTheme,
}: {
  launch: WorkbenchTarget;
  snapshot: ProjectSnapshot;
  refresh: () => Promise<void>;
  onBack: () => void;
  perform: (f: () => Promise<unknown>) => Promise<void>;
  notify: (s: string) => void;
  resolvedTheme: "dark" | "light";
  toggleTheme: () => void;
}) {
  const [view, setView] = useState<View>(launch.view || "canvas");
  const [canvasId, setCanvasId] = useState(
    launch.canvasId || s.canvases.find((c) => !c.archived)?.id || "",
  );
  const [selected, setSelected] = useState<string[]>(
    launch.nodeId ? [launch.nodeId] : [],
  );
  const activeCanvasId = useRef(canvasId);
  activeCanvasId.current = canvasId;
  const [assetId, setAssetId] = useState<string | undefined>(launch.assetId);
  const [runId, setRunId] = useState<string | undefined>(launch.runId);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const [categoryFilter, setCategoryFilter] = useState("all");
  const [sidebar, setSidebar] = useResponsivePanel("(max-width: 1100px)");
  const launchInspector = Boolean(
    launch.assetId ||
      (launch.nodeId && !window.matchMedia("(max-width: 1200px)").matches) ||
      launch.stepKey ||
      (launch.runId && launch.view === "canvas"),
  );
  const [assetDock, setAssetDock] = useResponsivePanel(
    "(max-width: 800px)",
    launchInspector && window.matchMedia("(max-width: 800px)").matches
      ? false
      : undefined,
  );
  const [inspector, setInspector] = useResponsivePanel(
    "(max-width: 1200px)",
    launchInspector ? true : undefined,
  );
  const [modal, setModal] = useState<"canvas" | "note" | "template" | null>(
    null,
  );
  const [resourceMenu, setResourceMenu] = useState<{ x: number; y: number }>();
  const [resourceAction, setResourceAction] = useState<{
    kind: AddResourceKind;
    at: { x: number; y: number };
    node?: StudioNode;
    inputs: string[];
  }>();
  const [taskDetailsId, setTaskDetailsId] = useState<string | undefined>(
    launch.taskId,
  );
  const [renameNote, setRenameNote] = useState<StudioNode>();
  const [noteAction, setNoteAction] = useState<{
    position?: { x: number; y: number };
    node?: StudioNode;
    preview?: boolean;
  }>({});
  const [handoff, setHandoff] = useState<{
    canvasId: string;
    nodeIds: string[];
  }>();
  const [previewRequest, setPreviewRequest] = useState(0);
  const [importPosition, setImportPosition] = useState<{
    x: number;
    y: number;
  }>();
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(false);
  const [pendingFiles, setPendingFiles] = useState<File[]>([]);
  const [pathModalOpen, setPathModalOpen] = useState(false);
  const [pendingPaths, setPendingPaths] = useState<string[]>([]);
  const [workspaceFiles, setWorkspaceFiles] = useState<
    { relativePath: string; name: string; sizeBytes: number }[]
  >([]);
  const [pathQuery, setPathQuery] = useState("");
  const [manualPath, setManualPath] = useState("");
  const [pickingFiles, setPickingFiles] = useState(false);
  const [importName, setImportName] = useState("");
  const [importTarget, setImportTarget] = useState("");
  const [busy, setBusy] = useState(false);
  const imageImportPending = useRef(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const sidebarToggleRef = useRef<HTMLButtonElement>(null);
  const sidebarCloseRef = useRef<HTMLButtonElement>(null);
  const assetDockOpenRef = useRef<HTMLButtonElement>(null);
  const latest = useRef(s);
  latest.current = s;

  useEffect(() => {
    if (!sidebar || !window.matchMedia("(max-width: 1100px)").matches) return;
    sidebarCloseRef.current?.focus();
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !document.querySelector("dialog[open]")) {
        setSidebar(false);
        sidebarToggleRef.current?.focus();
      }
    };
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [sidebar, setSidebar]);

  function navigate(next: View) {
    setView(next);
    if (window.matchMedia("(max-width: 1100px)").matches) setSidebar(false);
  }

  const c =
    s.canvases.find((c) => c.id === canvasId) ||
    s.canvases.find((c) => !c.archived);

  const assets = s.assets.filter(
    (a) =>
      !a.archived &&
      `${a.name} ${a.tags.join(" ")} ${assetCategory(a)}`
        .toLowerCase()
        .includes(query.toLowerCase()) &&
      (view !== "assets" || filter === "all" || a.kind === filter) &&
      (view !== "assets" ||
        categoryFilter === "all" ||
        assetCategory(a) === categoryFilter),
  );

  const selectedNode = c?.nodes.find((n) => selected.includes(n.id));
  const selectedAsset = s.assets.find(
    (a) => a.id === (assetId || selectedNode?.data.assetId),
  );
  const run =
    s.runs.find((r) => r.id === runId) ||
    [...s.runs]
      .reverse()
      .find((r) => r.instanceId === selectedNode?.data.instanceId);
  const pending = s.runs.filter((r) => r.status === "active");

  const mutate = useCallback(
    async (name: string, args: unknown) => {
      setSaving(true);
      setSaveError(false);
      try {
        const v = await call(name, args);
        if (name === "job_control" && (args as any).action === "rerun")
          setRunId(v.id);
        await refresh();
        return v;
      } catch (e) {
        setSaveError(true);
        await perform(() => Promise.reject(e));
      } finally {
        setSaving(false);
      }
    },
    [refresh, perform],
  );

  const patch = useCallback(
    async (target: "graph" | "layout", ops: unknown[]) => {
      const cv = latest.current.canvases.find((x) => x.id === canvasId);
      if (!cv) return;
      return mutate("canvas_patch", {
        projectId: s.project.id,
        canvasId: cv.id,
        target,
        expectedRevision:
          target === "graph" ? cv.graphRevision : cv.layoutRevision,
        ops,
      });
    },
    [canvasId, mutate, s.project.id],
  );

  const addAsset = async (
    a: Asset,
    r: AssetRevision = a.revisions[0],
    dropPosition?: { x: number; y: number },
  ) => {
    if (!c) return;
    const n: StudioNode = {
      id: uuid(),
      type: "asset",
      position: dropPosition || freePosition(c, { x: 80, y: 60 }),
      data: { title: a.name, assetId: a.id, revisionId: r.id },
    };
    await patch("graph", [{ op: "add", node: n }]);
    setView("canvas");
    setSelected([n.id]);
    setAssetId(undefined);
    notify("已添加资源引用");
  };

  const openPathImportModal = (
    position?: { x: number; y: number },
    targetAssetId = "",
  ) => {
    setImportPosition(position);
    setImportTarget(targetAssetId);
    setPendingPaths([]);
    setImportName("");
    setPathQuery("");
    setManualPath("");
    setPathModalOpen(true);
    request<{
      files: { relativePath: string; name: string; sizeBytes: number }[];
    }>(`/api/projects/${s.project.id}/workspace-files`)
      .then((res) => setWorkspaceFiles(res.files || []))
      .catch(() => setWorkspaceFiles([]));
  };

  const chooseResource = (
    kind: AddResourceKind,
    at = freePosition(c, {
      x: c ? (180 - c.viewport.x) / c.viewport.zoom : 100,
      y: c ? (100 - c.viewport.y) / c.viewport.zoom : 100,
    }),
    node?: StudioNode,
  ) => {
    if (kind === "text") {
      setNoteAction({ position: at });
      setModal("note");
      return;
    }
    if (kind === "path") {
      openPathImportModal(at, "");
      return;
    }
    if (kind === "upload" || kind === "source") {
      setImportPosition(at);
      setImportTarget("");
      inputRef.current?.click();
      return;
    }
    setResourceAction({
      kind,
      at,
      node,
      inputs: selected.filter((id) =>
        c?.nodes.some(
          (n) => n.id === id && (n.type === "asset" || n.type === "note"),
        ),
      ),
    });
  };

  async function pickLocalFiles() {
    setPickingFiles(true);
    try {
      const res = await request<{
        cancelled: boolean;
        paths: string[] | null;
      }>("/api/files/pick", {
        method: "POST",
        body: JSON.stringify({ initialPath: s.project.root }),
      });
      if (!res.cancelled && res.paths?.length) {
        const merged = Array.from(new Set([...pendingPaths, ...res.paths]));
        setPendingPaths(merged);
        if (!importName.trim() && merged.length >= 1) {
          const base = merged[0].split(/[\\/]/).pop() || "";
          setImportName(base.replace(/\.[^.]+$/, ""));
        }
      }
    } catch {
    } finally {
      setPickingFiles(false);
    }
  }

  async function submitPathImport() {
    if (!pendingPaths.length) return;
    setBusy(true);
    try {
      await perform(async () => {
        const revision = await request<AssetRevision>(
          `/api/projects/${s.project.id}/import-paths`,
          {
            method: "POST",
            body: JSON.stringify({
              fileRefs: pendingPaths,
              name: importName || undefined,
              assetId: importTarget || undefined,
            }),
          },
        );
        if (importPosition) {
          const fallbackTitle = (
            pendingPaths[0].split(/[\\/]/).pop() || ""
          ).replace(/\.[^.]+$/, "");
          const node: StudioNode = {
            id: uuid(),
            type: "asset",
            position: importPosition,
            data: {
              title: importName || fallbackTitle || "资源",
              assetId: revision.assetId,
              revisionId: revision.id,
            },
          };
          await patch("graph", [{ op: "add", node }]);
          chooseNodes([node.id]);
        }
      });
      setPathModalOpen(false);
      setPendingPaths([]);
      setImportName("");
      setImportTarget("");
      setImportPosition(undefined);
      notify("已引用路径并登记版本");
    } catch {
    } finally {
      setBusy(false);
    }
  }

  async function uploadFiles() {
    if (!pendingFiles.length) return;
    setBusy(true);
    try {
      const fd = new FormData();
      pendingFiles.forEach((f) => fd.append("files", f));
      if (importName) fd.append("name", importName);
      if (importTarget) fd.append("assetId", importTarget);
      await perform(async () => {
        const revision = await request<AssetRevision>(
          `/api/projects/${s.project.id}/import`,
          { method: "POST", body: fd },
        );
        if (importPosition) {
          const node: StudioNode = {
            id: uuid(),
            type: "asset",
            position: importPosition,
            data: {
              title: importName || pendingFiles[0].name.replace(/\.[^.]+$/, ""),
              assetId: revision.assetId,
              revisionId: revision.id,
            },
          };
          await patch("graph", [{ op: "add", node }]);
          chooseNodes([node.id]);
        }
      });
      setPendingFiles([]);
      setImportName("");
      setImportTarget("");
      setImportPosition(undefined);
      notify("已导入并登记版本");
    } catch {
    } finally {
      setBusy(false);
    }
  }

  async function dropImages(
    drop: CanvasImageDrop,
    position: { x: number; y: number },
  ) {
    if (!c) return;
    // Capture this board before reading images; changing boards during a slow
    // import must never attach the result to the new board.
    const targetCanvasId = c.id;
    if (imageImportPending.current) {
      notify("正在导入，请稍后再拖入图片");
      return;
    }
    imageImportPending.current = true;
    const nodes: StudioNode[] = [];
    setBusy(true);
    try {
      await perform(async () => {
        const files = await resolveCanvasImages(drop);
        try {
          for (const [index, file] of files.entries()) {
            const body = new FormData();
            body.append("files", file);
            body.append("name", file.name.replace(/\.[^.]+$/, ""));
            const revision = await request<AssetRevision>(
              `/api/projects/${s.project.id}/import`,
              { method: "POST", body },
            );
            nodes.push({
              id: uuid(),
              type: "asset",
              position: {
                x: position.x + (index % 4) * 280,
                y: position.y + Math.floor(index / 4) * 300,
              },
              data: {
                title: file.name.replace(/\.[^.]+$/, ""),
                assetId: revision.assetId,
                revisionId: revision.id,
              },
            });
          }
          const board = latest.current.canvases.find(
            (board) => board.id === targetCanvasId,
          );
          if (!board) throw new Error("目标画布已不可用");
          // Use the captured board explicitly, and let failures reach the drop
          // error message instead of treating a rejected patch as a success.
          await call("canvas_patch", {
            projectId: s.project.id,
            canvasId: targetCanvasId,
            target: "graph",
            expectedRevision: board.graphRevision,
            ops: nodes.map((node) => ({ op: "add", node })),
          });
        } catch (error) {
          await refresh();
          if (nodes.length)
            throw new Error(
              `${(error as Error).message}；已导入的图片保留在项目资源中，可重新添加到画布。`,
            );
          throw error;
        }
      });
      if (activeCanvasId.current === targetCanvasId)
        chooseNodes(nodes.map((node) => node.id));
      notify(`已添加 ${nodes.length} 个图片节点`);
    } finally {
      imageImportPending.current = false;
      setBusy(false);
    }
  }

  function chooseNodes(ids: string[]) {
    setSelected(ids);
    setAssetId(undefined);
    setRunId(undefined);
  }

  const nav: Array<[View, ReactNode, string]> = [
    ["overview", <LayoutGrid size={15} />, "概览"],
    ["assets", <ImageIcon size={15} />, "资源"],
    ["canvas", <Layers size={15} />, "画布"],
    ["character", <Workflow size={15} />, "角色工作流"],
    ["runs", <Workflow size={15} />, "运行记录"],
    ["settings", <Settings size={15} />, "项目设置"],
  ];

  return (
    <div className="workspace">
      {sidebar && (
        <button
          className="sidebar-backdrop"
          aria-label="关闭项目导航"
          tabIndex={-1}
          onClick={() => {
            setSidebar(false);
            sidebarToggleRef.current?.focus();
          }}
        />
      )}
      <aside
        id="workspace-navigation"
        className="workspace-sidebar"
        hidden={!sidebar}
        aria-label="项目导航"
      >
        <div className="workspace-brand">
          <Brand />
          <button
            ref={sidebarCloseRef}
            className="icon-button sidebar-close"
            aria-label="收起项目导航"
            title="收起项目导航"
            onClick={() => {
              setSidebar(false);
              sidebarToggleRef.current?.focus();
            }}
          >
            <PanelLeftClose size={16} />
          </button>
        </div>
        <button className="back-link" onClick={onBack}>
          <ArrowLeft size={13} />
          全部项目
        </button>
        <div className="project-switch">
          <span className="project-avatar">{s.project.name.slice(0, 1)}</span>
          <div>
            <strong>{s.project.name}</strong>
            <small>{engineName[s.project.engine]}</small>
          </div>
        </div>
        <nav>
          {nav.map(([v, i, n]) => (
            <button
              key={v}
              className={view === v ? "side-item active" : "side-item"}
              onClick={() => navigate(v)}
            >
              {i}
              {n}
              {v === "runs" && pending.length > 0 && (
                <span className="count-dot">{pending.length}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="side-heading">
          画布
          <IconButton label="新建画布" onClick={() => setModal("canvas")}>
            <Plus size={13} />
          </IconButton>
        </div>
        {s.canvases
          .filter((c) => !c.archived)
          .map((ca) => (
            <button
              key={ca.id}
              className={
                view === "canvas" && c?.id === ca.id
                  ? "canvas-link selected"
                  : "canvas-link"
              }
              onClick={() => {
                setCanvasId(ca.id);
                navigate("canvas");
                chooseNodes([]);
              }}
            >
              <span className="canvas-square" />
              {ca.name}
              <small>{ca.nodes.length}</small>
            </button>
          ))}
        <div className="sidebar-bottom">
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <div className="local-status">
              <i />
              Core 已连接
            </div>
            <button
              className="icon-button"
              title="切换主题"
              aria-label="切换主题"
              onClick={toggleTheme}
            >
              {resolvedTheme === "dark" ? (
                <Sun size={14} />
              ) : (
                <Moon size={14} />
              )}
            </button>
          </div>
          <p
            className="mono muted"
            title={s.project.root}
            style={{ fontSize: "10px" }}
          >
            {s.project.root}
          </p>
        </div>
      </aside>
      <main className="workspace-main">
        <header className="workspace-top">
          <div className="workspace-location">
            <button
              ref={sidebarToggleRef}
              className="icon-button"
              aria-label={sidebar ? "收起项目导航" : "展开项目导航"}
              title={sidebar ? "收起项目导航" : "展开项目导航"}
              aria-expanded={sidebar}
              aria-controls="workspace-navigation"
              onClick={() => setSidebar(!sidebar)}
            >
              {sidebar ? (
                <PanelLeftClose size={16} />
              ) : (
                <PanelLeftOpen size={16} />
              )}
            </button>
            <div className="breadcrumbs">
              <button
                type="button"
                className="breadcrumb-project"
                title="查看项目概览"
                aria-label={`查看${s.project.name}项目概览`}
                onClick={() => navigate("overview")}
              >
                {s.project.name}
              </button>
              <ChevronRight size={13} />
              {view === "canvas" && c ? (
                <div className="breadcrumb-canvas">
                  <select
                    aria-label="切换画布"
                    title="切换画布"
                    value={c.id}
                    onChange={(event) => {
                      setCanvasId(event.target.value);
                      navigate("canvas");
                      chooseNodes([]);
                    }}
                  >
                    {s.canvases
                      .filter((canvas) => !canvas.archived)
                      .map((canvas) => (
                        <option key={canvas.id} value={canvas.id}>
                          {canvas.name}
                        </option>
                      ))}
                  </select>
                  <ChevronDown size={13} aria-hidden="true" />
                </div>
              ) : (
                <strong>{nav.find(([v]) => v === view)?.[2]}</strong>
              )}
            </div>
            <VersionBadge />
          </div>
          <div className="top-actions">
            <span className={`save-status ${saveError ? "warning" : ""}`}>
              {saving ? (
                <Loader2 className="spin" size={13} />
              ) : saveError ? (
                <AlertTriangle size={13} />
              ) : (
                <Check size={13} />
              )}{" "}
              {saving ? "正在保存" : saveError ? "保存失败" : "已保存到本地"}
            </span>
            <ProjectLiveSync
              projectId={s.project.id}
              refresh={refresh}
              paused={saving || busy}
            />
            {view === "canvas" && (
              <IconButton
                label={assetDock ? "收起资源栏" : "展开资源栏"}
                expanded={assetDock}
                controls="workspace-assets"
                onClick={() => {
                  setAssetDock(!assetDock);
                  if (
                    !assetDock &&
                    window.matchMedia("(max-width: 800px)").matches
                  )
                    setInspector(false);
                }}
              >
                <ImageIcon size={16} />
              </IconButton>
            )}
            {view === "canvas" && (
              <IconButton
                label={inspector ? "收起面板" : "展开面板"}
                expanded={inspector}
                onClick={() => {
                  setInspector(!inspector);
                  if (
                    !inspector &&
                    window.matchMedia("(max-width: 800px)").matches
                  )
                    setAssetDock(false);
                }}
              >
                {inspector ? (
                  <PanelRightClose size={16} />
                ) : (
                  <PanelRightOpen size={16} />
                )}
              </IconButton>
            )}
          </div>
        </header>
        {view === "canvas" && c ? (
          <>
            <div className="canvas-header">
              <div className="canvas-title-group">
                {!assetDock && (
                  <button
                    ref={assetDockOpenRef}
                    type="button"
                    className="button small"
                    title="展开项目资源"
                    aria-label="展开项目资源"
                    aria-expanded={false}
                    aria-controls="workspace-assets"
                    onClick={() => {
                      setAssetDock(true);
                      if (window.matchMedia("(max-width: 800px)").matches)
                        setInspector(false);
                    }}
                  >
                    <PanelLeftOpen size={14} />
                    项目资源
                  </button>
                )}
                <h2>{c.name}</h2>
                <Badge>{c.nodes.length} 个节点</Badge>
              </div>
              <div className="canvas-actions">
                {selected.length > 0 && (
                  <button
                    className="button small"
                    onClick={() =>
                      setHandoff({ canvasId: c.id, nodeIds: [...selected] })
                    }
                  >
                    <Link2 size={14} />
                    添加引用 · {selected.length}
                  </button>
                )}
                <button
                  className="button small primary"
                  aria-label="添加资源"
                  aria-expanded={!!resourceMenu}
                  onClick={(e) => {
                    const rect = e.currentTarget.getBoundingClientRect();
                    setResourceMenu({ x: rect.left, y: rect.bottom + 8 });
                  }}
                >
                  <Plus size={14} />
                  添加资源
                </button>
                <button
                  className="button small"
                  onClick={() => {
                    setNoteAction({});
                    setModal("note");
                  }}
                >
                  <StickyNote size={14} />
                  便签
                </button>
                <button
                  className="button small"
                  onClick={() => {
                    setImportPosition(undefined);
                    inputRef.current?.click();
                  }}
                >
                  <Upload size={14} />
                  导入素材
                </button>
                <button
                  className="button small primary"
                  disabled={!selectedNode?.data.revisionId || saving}
                  onClick={() => setModal("template")}
                >
                  <Workflow size={14} />
                  制作角色
                </button>
              </div>
            </div>
            <div className="canvas-body">
              <div
                id="workspace-assets"
                className="asset-dock"
                hidden={!assetDock}
              >
                <div className="dock-title">
                  <strong>项目资源</strong>
                  <div className="dock-title-actions">
                    <span>{s.assets.filter((a) => !a.archived).length}</span>
                    <IconButton
                      label="收起项目资源"
                      expanded={true}
                      controls="workspace-assets"
                      onClick={() => {
                        setAssetDock(false);
                        requestAnimationFrame(() =>
                          assetDockOpenRef.current?.focus(),
                        );
                      }}
                    >
                      <PanelLeftClose size={15} />
                    </IconButton>
                  </div>
                </div>
                <label className="search compact">
                  <Search size={13} />
                  <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="搜索资源…"
                  />
                </label>
                <AssetGroups
                  key={s.project.id}
                  assets={assets}
                  projectId={s.project.id}
                  searching={!!query.trim()}
                  renderAsset={(a) => (
                    <button
                      className="dock-asset"
                      key={a.id}
                      title="点击查看 · 拖入画布添加引用"
                      draggable
                      onDragStart={(e) =>
                        e.dataTransfer.setData("application/studio-asset", a.id)
                      }
                      onClick={() => {
                        setAssetId(a.id);
                        setRunId(undefined);
                        setSelected([]);
                        setInspector(true);
                        if (window.matchMedia("(max-width: 800px)").matches)
                          setAssetDock(false);
                      }}
                    >
                      <AssetThumb asset={a} projectId={s.project.id} />
                      <span>
                        <strong>{a.name}</strong>
                        <small>
                          {kindName[a.kind]} · v{a.revisions[0]?.number}
                        </small>
                      </span>
                      <Grip size={13} style={{ color: "var(--text-muted)" }} />
                    </button>
                  )}
                />
                <button
                  className="dock-import"
                  onClick={() => {
                    setImportPosition(undefined);
                    inputRef.current?.click();
                  }}
                >
                  <Plus size={14} />
                  导入资源
                </button>
              </div>
              <CanvasBoard
                key={c.id}
                canvas={c}
                snapshot={s}
                selected={selected}
                focusNodeId={launch.nodeId}
                onSelect={chooseNodes}
                patch={patch}
                saving={saving || busy}
                addAsset={addAsset}
                onImageDrop={dropImages}
                resolvedTheme={resolvedTheme}
                notify={notify}
                onNote={(position, node, preview) => {
                  setNoteAction({ position, node, preview });
                  setModal("note");
                }}
                onRenameNote={setRenameNote}
                onImport={(position) => {
                  setImportPosition(position);
                  setImportTarget("");
                  inputRef.current?.click();
                }}
                onResources={() => setAssetDock(true)}
                onResource={chooseResource}
                onTask={(id) => setTaskDetailsId(id)}
                onView={() => {
                  setPreviewRequest((v) => v + 1);
                  setInspector(true);
                  if (window.matchMedia("(max-width:800px)").matches)
                    setAssetDock(false);
                }}
                onContext={(nodeIds) =>
                  setHandoff({ canvasId: c.id, nodeIds: [...nodeIds] })
                }
                onWorkflow={() => setModal("template")}
              />
              {inspector && (
                <Inspector
                  key={previewRequest}
                  snapshot={s}
                  asset={selectedAsset}
                  fixedRevisionId={
                    selectedNode?.data.revisionId ||
                    (assetId === launch.assetId ? launch.revisionId : undefined)
                  }
                  launchStepKey={
                    run?.id === launch.runId ? launch.stepKey : undefined
                  }
                  run={run}
                  node={selectedNode}
                  addAsset={addAsset}
                  mutate={mutate}
                  patch={patch}
                  notify={notify}
                  selected={selected}
                  onContext={() =>
                    setHandoff({ canvasId: c.id, nodeIds: [...selected] })
                  }
                  canvas={c}
                  onPreviewNote={() => {
                    setNoteAction({ node: selectedNode, preview: true });
                    setModal("note");
                  }}
                  onImportVersion={(id) => {
                    setImportPosition(undefined);
                    setImportTarget(id);
                    inputRef.current?.click();
                  }}
                  onImportPathVersion={(id) =>
                    openPathImportModal(undefined, id)
                  }
                  onOpenCharacterWorkflow={(targetRunId) => {
                    if (targetRunId) setRunId(targetRunId);
                    navigate("character");
                  }}
                />
              )}
            </div>
          </>
        ) : null}
        {view === "character" && (
          <CharacterWorkflowPage
            s={s}
            launchNodeKey={runId === launch.runId ? launch.nodeKey : undefined}
            activeRunId={runId}
            onSelectRun={(id) => setRunId(id)}
            refresh={refresh}
            notify={notify}
          />
        )}
        {view === "assets" && (
          <div className="content-page">
            <div className="page-title">
              <div>
                <h1>项目资源</h1>
                <p>集中查看与管理所有不可变素材及版本记录。</p>
              </div>
              <div style={{ display: "flex", gap: "8px" }}>
                <button
                  className="button small"
                  onClick={() => openPathImportModal(undefined, "")}
                >
                  <FolderOpen size={14} />
                  引用本地路径
                </button>
                <button
                  className="button small primary"
                  onClick={() => {
                    setImportPosition(undefined);
                    inputRef.current?.click();
                  }}
                >
                  <Upload size={14} />
                  导入资源
                </button>
              </div>
            </div>
            <div className="resource-toolbar">
              <label className="resource-category-filter">
                用途
                <select
                  aria-label="筛选用途分类"
                  value={categoryFilter}
                  onChange={(e) => setCategoryFilter(e.target.value)}
                >
                  <option value="all">全部分类</option>
                  {assetCategories
                    .filter((category) =>
                      s.assets.some(
                        (a) => !a.archived && assetCategory(a) === category,
                      ),
                    )
                    .map((category) => (
                      <option key={category} value={category}>
                        {category}
                      </option>
                    ))}
                </select>
              </label>
              <div className="filter-tabs">
                {[
                  ["all", "全部"],
                  ["image", "图片"],
                  ["model", "3D 模型"],
                  ["document", "文档"],
                  ["source", "源文件"],
                  ["video", "视频"],
                  ["audio", "音频"],
                ].map(([v, n]) => (
                  <button
                    key={v}
                    className={filter === v ? "selected" : ""}
                    onClick={() => setFilter(v)}
                  >
                    {n}
                  </button>
                ))}
              </div>
              <label className="search compact">
                <Search size={13} />
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="搜索名称、标签或分类…"
                />
              </label>
            </div>
            <div className="asset-grid">
              {assets.map((a) => (
                <button
                  className="asset-tile"
                  key={a.id}
                  onClick={() => {
                    setAssetId(a.id);
                    setSelected([]);
                    setView("canvas");
                    setInspector(true);
                  }}
                >
                  <AssetThumb asset={a} projectId={s.project.id} large />
                  <div>
                    <h3>{a.name}</h3>
                    <span>
                      {kindName[a.kind]}
                      <Badge>v{a.revisions[0]?.number}</Badge>
                    </span>
                  </div>
                </button>
              ))}
            </div>
            {!assets.length && (
              <div className="empty-state">
                <ImageIcon size={28} />
                <p>暂无资源</p>
                <button
                  className="button small"
                  onClick={() => {
                    setImportPosition(undefined);
                    inputRef.current?.click();
                  }}
                >
                  导入素材
                </button>
              </div>
            )}
          </div>
        )}
        {view === "runs" && (
          <div className="content-page">
            <div className="page-title">
              <div>
                <h1>运行记录</h1>
                <p>制作流程执行轨迹、前置输入与审核记录。</p>
              </div>
              <Badge tone="blue">{pending.length} 进行中</Badge>
            </div>
            {!!s.tasks?.length && (
              <>
                <h2 className="section-title">资源制作</h2>
                {[...s.tasks].reverse().map((t) => (
                  <button
                    className="run-row"
                    key={t.id}
                    onClick={() => setTaskDetailsId(t.id)}
                  >
                    <div className="run-icon">
                      <Box size={18} />
                    </div>
                    <div>
                      <h3>{t.name}</h3>
                      <p>
                        {generationLabels[t.kind]} · {stamp(t.createdAt)}
                      </p>
                    </div>
                    <Badge tone={t.status === "succeeded" ? "green" : "amber"}>
                      {taskLabels[t.status]}
                    </Badge>
                    <ChevronRight size={15} />
                  </button>
                ))}
              </>
            )}
            {!s.runs.length && !s.tasks?.length ? (
              <div className="empty-state">
                <Workflow size={28} />
                <p>暂无运行记录，在画布中选中素材点击「制作角色」开始。</p>
              </div>
            ) : (
              s.runs
                .slice()
                .reverse()
                .map((r) => (
                  <button
                    className="run-row"
                    key={r.id}
                    onClick={() => {
                      setRunId(r.id);
                      setAssetId(undefined);
                      setSelected([]);
                      setCanvasId(r.canvasId);
                      navigate("character");
                    }}
                  >
                    <div className="run-icon">
                      <Workflow size={18} />
                    </div>
                    <div>
                      <h3>
                        角色制作
                        {r.parentRunId && <Badge tone="blue">局部重跑</Badge>}
                      </h3>
                      <p>
                        {r.steps.find((s) => s.status.startsWith("awaiting"))
                          ?.title || statusName[r.status]}{" "}
                        · {stamp(r.createdAt)}
                      </p>
                    </div>
                    <div className="run-progress">
                      {r.steps.map((st) => (
                        <span
                          key={st.key}
                          className={
                            st.status === "succeeded"
                              ? "complete"
                              : st.status.startsWith("awaiting")
                                ? "current"
                                : ""
                          }
                        />
                      ))}
                    </div>
                    <Badge tone={r.status === "succeeded" ? "green" : "amber"}>
                      {statusName[r.status]}
                    </Badge>
                    <ChevronRight
                      size={15}
                      style={{ color: "var(--text-muted)" }}
                    />
                  </button>
                ))
            )}
          </div>
        )}
        {view === "overview" && (
          <div className="content-page">
            <div className="page-title">
              <div>
                <h1>{s.project.name}</h1>
                <p>游戏工作台状态与制作产物摘要。</p>
              </div>
              <Badge tone="blue">{engineName[s.project.engine]}</Badge>
            </div>
            <div className="overview-stats">
              {[
                [
                  s.assets.filter((a) => !a.archived).length,
                  "项目资源",
                  <ImageIcon size={18} />,
                ],
                [
                  s.canvases.filter((c) => !c.archived).length,
                  "制作画布",
                  <Layers size={18} />,
                ],
                [pending.length, "进行中流程", <Workflow size={18} />],
                [s.exports.length, "成果导出", <Boxes size={18} />],
              ].map(([v, n, i]) => (
                <div key={String(n)}>
                  {i}
                  <strong>{v}</strong>
                  <span>{n}</span>
                </div>
              ))}
            </div>
            <h2 className="section-title">制作画布</h2>
            <div className="overview-boards">
              {s.canvases
                .filter((c) => !c.archived)
                .map((c) => (
                  <button
                    className="board-card"
                    key={c.id}
                    onClick={() => {
                      setCanvasId(c.id);
                      setView("canvas");
                    }}
                  >
                    <Layers size={20} style={{ color: "var(--text-muted)" }} />
                    <h3>{c.name}</h3>
                    <p>
                      {c.nodes.length} 个节点 · {c.instances.length} 个制作分组
                    </p>
                  </button>
                ))}
            </div>
            <h2 className="section-title">成果交接记录</h2>
            {!s.exports.length && (
              <p className="muted" style={{ fontSize: "12px" }}>
                暂无导出记录。流程完成后在此处查看导出的固定版本与 manifest。
              </p>
            )}
            {s.exports.map((e) => (
              <div className="export-row" key={e.id}>
                <Download size={16} />
                <div>
                  <strong>导出包 · {stamp(e.createdAt)}</strong>
                  <code className="mono">{e.path}</code>
                </div>
                <Badge tone="green">文件完整性校验通过</Badge>
              </div>
            ))}
          </div>
        )}
        {view === "settings" && (
          <ProjectSettings
            project={s.project}
            refresh={refresh}
            mutate={mutate}
            onBack={onBack}
            notify={notify}
            canvases={s.canvases}
          />
        )}
      </main>
      <input
        className="hidden"
        ref={inputRef}
        type="file"
        multiple
        accept=".png,.jpg,.jpeg,.webp,.gif,.glb,.blend,.md,.txt,.json,.mp4,.webm,.mp3,.wav,.ogg,.m4a,.ts,.tsx,.js,.jsx,.css,.html,.gd,.py,.shader,.glsl,.cs,.toml,.yaml,.yml"
        onChange={(e) => {
          const files = Array.from(e.target.files || []);
          if (files.length) {
            setPendingFiles(files);
            setImportName(
              files.length === 1 ? files[0].name.replace(/\.[^.]+$/, "") : "",
            );
          }
          e.target.value = "";
        }}
      />
      {pendingFiles.length > 0 && (
        <Modal
          title="导入资源"
          onClose={() => {
            setPendingFiles([]);
            setImportTarget("");
            setImportPosition(undefined);
          }}
        >
          <div className="modal-body" style={{ padding: 0 }}>
            <p className="modal-description">
              选择的文件将登记为不可变版本文件集。
            </p>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "6px",
                marginBottom: "14px",
              }}
            >
              {pendingFiles.map((f) => (
                <div
                  key={f.name}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    padding: "6px 10px",
                    background: "var(--bg-surface)",
                    borderRadius: "6px",
                    fontSize: "12px",
                  }}
                >
                  <FileText size={14} />
                  <span
                    style={{
                      flex: 1,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {f.name}
                  </span>
                  <span className="mono muted">
                    {(f.size / 1024).toFixed(0)} KB
                  </span>
                </div>
              ))}
            </div>
            <label className="field">
              资源名称
              <input
                required
                value={importName}
                onChange={(e) => setImportName(e.target.value)}
                placeholder="例如：巡林者 · 角色模型"
              />
            </label>
            <label className="field">
              版本归属
              <select
                value={importTarget}
                onChange={(e) => setImportTarget(e.target.value)}
              >
                <option value="">新建独立资源</option>
                {s.assets
                  .filter((a) => !a.archived)
                  .map((a) => (
                    <option key={a.id} value={a.id}>
                      {a.name} · 添加新版本
                    </option>
                  ))}
              </select>
            </label>
            <div className="modal-footer">
              <button
                className="button small"
                onClick={() => {
                  const pos = importPosition;
                  const target = importTarget;
                  setPendingFiles([]);
                  openPathImportModal(pos, target);
                }}
              >
                <FolderOpen size={14} />
                切换为本地路径引用
              </button>
              <button
                className="button small"
                onClick={() => {
                  setPendingFiles([]);
                  setImportTarget("");
                  setImportPosition(undefined);
                }}
              >
                取消
              </button>
              <button
                className="button small primary"
                disabled={busy || !importName.trim()}
                onClick={uploadFiles}
              >
                {busy ? (
                  <Loader2 size={14} className="spin" />
                ) : (
                  <Upload size={14} />
                )}
                导入并登记版本
              </button>
            </div>
          </div>
        </Modal>
      )}
      {pathModalOpen && (
        <Modal
          title="引用本地文件路径"
          onClose={() => {
            setPathModalOpen(false);
            setPendingPaths([]);
            setImportTarget("");
            setImportPosition(undefined);
          }}
        >
          <div className="modal-body" style={{ padding: 0 }}>
            <p className="modal-description">
              直接引用项目目录内的文件路径（零复制）；若选择项目外文件则自动归档到{" "}
              <code className="mono">.game-studio/assets</code>{" "}
              并按哈希去重，随仓库上传后其他成员拉取即可正常显示。
            </p>
            <div
              style={{
                display: "flex",
                gap: "8px",
                marginBottom: "10px",
                alignItems: "center",
              }}
            >
              <button
                type="button"
                className="button small"
                disabled={pickingFiles}
                onClick={pickLocalFiles}
              >
                {pickingFiles ? (
                  <Loader2 size={14} className="spin" />
                ) : (
                  <FolderOpen size={14} />
                )}
                浏览本机文件…
              </button>
              <input
                value={manualPath}
                onChange={(e) => setManualPath(e.target.value)}
                placeholder="输入项目内相对路径（如 sprites/hero.png）或绝对路径"
                aria-label="文件路径"
                style={{ flex: 1 }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && manualPath.trim()) {
                    e.preventDefault();
                    const p = manualPath.trim();
                    const next = Array.from(new Set([...pendingPaths, p]));
                    setPendingPaths(next);
                    setManualPath("");
                    if (!importName.trim()) {
                      const base = p.split(/[\\/]/).pop() || "";
                      setImportName(base.replace(/\.[^.]+$/, ""));
                    }
                  }
                }}
              />
              <button
                type="button"
                className="button small"
                disabled={!manualPath.trim()}
                onClick={() => {
                  const p = manualPath.trim();
                  if (!p) return;
                  const next = Array.from(new Set([...pendingPaths, p]));
                  setPendingPaths(next);
                  setManualPath("");
                  if (!importName.trim()) {
                    const base = p.split(/[\\/]/).pop() || "";
                    setImportName(base.replace(/\.[^.]+$/, ""));
                  }
                }}
              >
                添加路径
              </button>
            </div>
            {workspaceFiles.length > 0 && (
              <div style={{ marginBottom: "12px" }}>
                <label className="field" style={{ marginBottom: "6px" }}>
                  项目目录内已有文件（点击直接勾选引用）
                  <input
                    value={pathQuery}
                    onChange={(e) => setPathQuery(e.target.value)}
                    placeholder="筛选项目内文件…"
                  />
                </label>
                <div
                  style={{
                    maxHeight: "150px",
                    overflowY: "auto",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: "6px",
                    padding: "4px",
                    display: "flex",
                    flexDirection: "column",
                    gap: "4px",
                  }}
                >
                  {workspaceFiles
                    .filter((f) =>
                      `${f.relativePath} ${f.name}`
                        .toLowerCase()
                        .includes(pathQuery.toLowerCase()),
                    )
                    .slice(0, 80)
                    .map((f) => {
                      const checked = pendingPaths.includes(f.relativePath);
                      return (
                        <button
                          type="button"
                          key={f.relativePath}
                          onClick={() => {
                            const next = checked
                              ? pendingPaths.filter((p) => p !== f.relativePath)
                              : [...pendingPaths, f.relativePath];
                            setPendingPaths(next);
                            if (!importName.trim() && next.length === 1) {
                              const base = next[0].split(/[\\/]/).pop() || "";
                              setImportName(base.replace(/\.[^.]+$/, ""));
                            }
                          }}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "8px",
                            padding: "5px 8px",
                            background: checked
                              ? "var(--bg-elevated)"
                              : "transparent",
                            border: "none",
                            borderRadius: "4px",
                            textAlign: "left",
                            cursor: "pointer",
                            fontSize: "12px",
                            color: "var(--text-primary)",
                          }}
                        >
                          <Check
                            size={13}
                            style={{ opacity: checked ? 1 : 0.2 }}
                          />
                          <code
                            className="mono"
                            style={{
                              flex: 1,
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                            }}
                          >
                            {f.relativePath}
                          </code>
                          <span className="mono muted">
                            {Math.max(1, Math.round(f.sizeBytes / 1024))} KB
                          </span>
                        </button>
                      );
                    })}
                </div>
              </div>
            )}
            {pendingPaths.length > 0 && (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "6px",
                  marginBottom: "14px",
                }}
              >
                {pendingPaths.map((p) => (
                  <div
                    key={p}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      padding: "6px 10px",
                      background: "var(--bg-surface)",
                      borderRadius: "6px",
                      fontSize: "12px",
                    }}
                  >
                    <FileText size={14} />
                    <code
                      className="mono"
                      style={{
                        flex: 1,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                      }}
                    >
                      {p}
                    </code>
                    <button
                      type="button"
                      className="icon-button"
                      aria-label={`移除 ${p}`}
                      onClick={() =>
                        setPendingPaths(pendingPaths.filter((x) => x !== p))
                      }
                    >
                      <X size={13} />
                    </button>
                  </div>
                ))}
              </div>
            )}
            <label className="field">
              资源名称
              <input
                required
                value={importName}
                onChange={(e) => setImportName(e.target.value)}
                placeholder="例如：巡林者 · 角色模型"
              />
            </label>
            <label className="field">
              版本归属
              <select
                value={importTarget}
                onChange={(e) => setImportTarget(e.target.value)}
              >
                <option value="">新建独立资源</option>
                {s.assets
                  .filter((a) => !a.archived)
                  .map((a) => (
                    <option key={a.id} value={a.id}>
                      {a.name} · 添加新版本
                    </option>
                  ))}
              </select>
            </label>
            <div className="modal-footer">
              <button
                className="button small"
                onClick={() => {
                  setPathModalOpen(false);
                  setPendingPaths([]);
                  setImportTarget("");
                  setImportPosition(undefined);
                }}
              >
                取消
              </button>
              <button
                className="button small primary"
                disabled={busy || !importName.trim() || !pendingPaths.length}
                onClick={submitPathImport}
              >
                {busy ? (
                  <Loader2 size={14} className="spin" />
                ) : (
                  <FolderOpen size={14} />
                )}
                引用并登记版本
              </button>
            </div>
          </div>
        </Modal>
      )}
      {handoff && (
        <CodexHandoff
          projectId={s.project.id}
          canvasId={handoff.canvasId}
          nodeIds={handoff.nodeIds}
          notify={notify}
          onClose={() => setHandoff(undefined)}
        />
      )}
      {modal === "canvas" && (
        <TextModal
          title="新建画布"
          label="画布名称"
          placeholder="例如：角色探索"
          onClose={() => setModal(null)}
          onSave={async (name) => {
            const c = await mutate("canvas_manage", {
              projectId: s.project.id,
              action: "create",
              name,
            });
            if (c) {
              setCanvasId(c.id);
              setView("canvas");
              setModal(null);
            }
          }}
        />
      )}
      {renameNote && (
        <TextModal
          title="重命名便签"
          label="便签名称"
          maxLength={120}
          placeholder="输入便签名称（最多 120 字）"
          initialValue={renameNote.data.title}
          onClose={() => setRenameNote(undefined)}
          onSave={async (title) => {
            const saved = await patch("graph", [
              { op: "note", id: renameNote.id, title: title.trim() },
            ]);
            if (!saved) throw new Error("Note was not renamed");
            setRenameNote(undefined);
          }}
        />
      )}
      {modal === "note" && (
        <NoteModal
          key={noteAction.node?.id || "new-note"}
          initialValue={noteAction.node?.data.text}
          initialTitle={noteAction.node?.data.title}
          existing={!!noteAction.node}
          preview={noteAction.preview}
          onClose={() => setModal(null)}
          onSave={async (text, title) => {
            if (noteAction.node) {
              const saved = await patch("graph", [
                { op: "note", id: noteAction.node.id, text, title },
              ]);
              if (!saved) throw new Error("Note was not saved");
              setModal(null);
              return;
            }
            const n: StudioNode = {
              id: uuid(),
              type: "note",
              position:
                noteAction.position ||
                freePosition(c, { x: 100, y: 80 }, 220, 210),
              data: { title, text },
            };
            const saved = await patch("graph", [{ op: "add", node: n }]);
            if (!saved) throw new Error("Note was not saved");
            setModal(null);
            chooseNodes([n.id]);
          }}
        />
      )}
      {modal === "template" &&
        selectedNode &&
        selected.every(
          (id) => c?.nodes.find((n) => n.id === id)?.type !== "workflow",
        ) && (
          <Modal title="展开角色制作" onClose={() => setModal(null)}>
            <div className="modal-body" style={{ padding: 0 }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "10px",
                  marginBottom: "12px",
                }}
              >
                <Workflow size={20} />
                <div>
                  <strong style={{ fontSize: "13px" }}>角色制作 · v1.0</strong>
                  <p className="muted" style={{ fontSize: "11px" }}>
                    固定 8 阶段流水线模板
                  </p>
                </div>
              </div>
              <div
                style={{
                  padding: "8px 12px",
                  background: "var(--bg-surface)",
                  borderRadius: "6px",
                  marginBottom: "12px",
                  fontSize: "12px",
                }}
              >
                <span className="muted" style={{ marginRight: "8px" }}>
                  固定输入
                </span>
                {c?.nodes
                  .filter((n) => selected.includes(n.id))
                  .map((n) => (
                    <div key={n.id}>
                      <strong>{n.data.title}</strong>
                      <code className="mono muted" style={{ marginLeft: 8 }}>
                        {n.data.revisionId?.slice(0, 12) || "便签内容"}
                      </code>
                    </div>
                  ))}
              </div>
              <div className="template-stages">
                {[
                  "目标",
                  "参考",
                  "候选",
                  "审核",
                  "组装",
                  "绑定",
                  "复核",
                  "交接",
                ].map((x, i) => (
                  <span key={x}>
                    <small className="mono">0{i + 1}</small>
                    {x}
                  </span>
                ))}
              </div>
              <div className="modal-footer">
                <button className="button small" onClick={() => setModal(null)}>
                  取消
                </button>
                <button
                  className="button small primary"
                  disabled={saving}
                  onClick={async () => {
                    try {
                      await patch("graph", [
                        {
                          op: "instantiate",
                          inputNodeIds: selected,
                          position: freePosition(
                            c,
                            {
                              x: selectedNode.position.x + 320,
                              y: selectedNode.position.y,
                            },
                            280,
                            420,
                          ),
                        },
                      ]);
                      setModal(null);
                      chooseNodes([]);
                      notify("已添加制作分组");
                    } catch {}
                  }}
                >
                  <Plus size={14} />
                  添加制作分组
                </button>
              </div>
            </div>
          </Modal>
        )}
      {resourceMenu && (
        <ResourceMenu
          position={resourceMenu}
          onClose={() => setResourceMenu(undefined)}
          onChoose={(kind) => chooseResource(kind)}
        />
      )}
      {resourceAction &&
        c &&
        (resourceAction.kind === "document" ? (
          <DocumentEditor
            snapshot={s}
            canvas={c}
            node={resourceAction.node}
            at={resourceAction.at}
            onClose={() => setResourceAction(undefined)}
            onSaved={refresh}
          />
        ) : ["library", "existing-document"].includes(resourceAction.kind) ? (
          <ResourceLibrary
            snapshot={s}
            documentsOnly={resourceAction.kind === "existing-document"}
            onClose={() => setResourceAction(undefined)}
            onChoose={(a, r) => addAsset(a, r, resourceAction.at)}
          />
        ) : (
          <GenerationComposer
            snapshot={s}
            canvas={c}
            kind={
              resourceAction.kind as import("../shared/types").GenerationKind
            }
            inputNodeIds={resourceAction.inputs}
            at={resourceAction.at}
            onClose={() => setResourceAction(undefined)}
            onSaved={refresh}
          />
        ))}
      {taskDetailsId && s.tasks?.find((t) => t.id === taskDetailsId) && (
        <GenerationDetails
          task={s.tasks.find((t) => t.id === taskDetailsId)!}
          projectId={s.project.id}
          assets={s.assets}
          onClose={() => setTaskDetailsId(undefined)}
          onRefresh={refresh}
        />
      )}
    </div>
  );
}

function NoteModal({
  initialValue = "",
  initialTitle = "制作便签",
  existing,
  preview = false,
  onClose,
  onSave,
}: {
  initialValue?: string;
  initialTitle?: string;
  existing: boolean;
  preview?: boolean;
  onClose: () => void;
  onSave: (value: string, title: string) => Promise<void>;
}) {
  const [value, setValue] = useState(initialValue);
  const [title, setTitle] = useState(initialTitle);
  const [mode, setMode] = useState(preview ? "preview" : "edit");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const dirty = value !== initialValue || title !== initialTitle;
  const save = async () => {
    setBusy(true);
    setError("");
    try {
      await onSave(value, title.trim());
    } catch {
      setError("保存失败，请重试。内容仍保留在编辑区。");
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal
      title={
        existing ? (mode === "preview" ? "便签全文" : "编辑便签") : "添加便签"
      }
      className="note-dialog"
      onClose={onClose}
    >
      <div className="note-dialog-toolbar">
        <div
          className="note-mode-switch"
          role="group"
          aria-label="便签显示模式"
        >
          <button
            className="button small"
            aria-pressed={mode === "preview"}
            onClick={() => setMode("preview")}
          >
            预览
          </button>
          <button
            className="button small"
            aria-pressed={mode === "edit"}
            onClick={() => setMode("edit")}
          >
            编辑
          </button>
        </div>
        <span className="note-text-count">
          {Array.from(value).length.toLocaleString()} 字
          {dirty ? " · 未保存" : ""}
        </span>
        <button
          className="button small"
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(value);
              setCopied(true);
              setError("");
            } catch {
              setError("复制失败，请选择全文后复制。");
            }
          }}
        >
          <Copy size={13} />
          {copied ? "已复制" : "复制全文"}
        </button>
      </div>
      <label className="field note-title-field">
        便签名称
        <input
          aria-label="便签名称"
          maxLength={120}
          value={title}
          readOnly={mode === "preview"}
          onChange={(e) => setTitle(e.target.value)}
        />
      </label>
      <div className="note-dialog-content">
        {mode === "preview" ? (
          <div
            className="note-fulltext"
            tabIndex={0}
            role="region"
            aria-label="便签全文内容"
          >
            {value || "暂无内容"}
          </div>
        ) : (
          <textarea
            className="note-editor"
            aria-label="内容"
            autoFocus
            maxLength={30000}
            value={value}
            placeholder="记录造型设计、设定约束或提示…"
            onChange={(e) => {
              setValue(e.target.value);
              setCopied(false);
            }}
          />
        )}
      </div>
      {error && (
        <p className="note-dialog-error" role="alert">
          {error}
        </p>
      )}
      <div className="modal-footer note-dialog-footer">
        <span className="muted">
          {mode === "preview" ? "保留原文与换行" : "支持多行提示词"}
        </span>
        <button className="button small" onClick={onClose}>
          {dirty ? "取消" : "关闭"}
        </button>
        {(mode === "edit" || dirty) && (
          <button
            className="button small primary"
            disabled={busy || !value.trim() || !title.trim()}
            onClick={save}
          >
            {busy ? "保存中…" : "保存"}
          </button>
        )}
      </div>
    </Modal>
  );
}

function TextModal({
  title,
  label,
  placeholder,
  multiline,
  initialValue = "",
  maxLength,
  onClose,
  onSave,
}: {
  title: string;
  label: string;
  placeholder: string;
  multiline?: boolean;
  initialValue?: string;
  maxLength?: number;
  onClose: () => void;
  onSave: (v: string) => Promise<void>;
}) {
  const [value, setValue] = useState(initialValue);
  const [busy, setBusy] = useState(false);
  const save = async () => {
    setBusy(true);
    try {
      await onSave(value);
    } catch {
    } finally {
      setBusy(false);
    }
  };
  return (
    <Modal title={title} onClose={onClose}>
      <form
        onSubmit={async (e) => {
          e.preventDefault();
          await save();
        }}
      >
        <label className="field">
          {label}
          {multiline ? (
            <textarea
              aria-label={label}
              autoFocus
              required
              rows={4}
              value={value}
              placeholder={placeholder}
              onChange={(e) => setValue(e.target.value)}
            />
          ) : (
            <input
              aria-label={label}
              autoFocus
              required
              maxLength={maxLength}
              value={value}
              placeholder={placeholder}
              onChange={(e) => setValue(e.target.value)}
            />
          )}
        </label>
        <div className="modal-footer">
          <button type="button" className="button small" onClick={onClose}>
            取消
          </button>
          <button
            className="button small primary"
            disabled={busy || !value.trim()}
            type={isEmbeddedApp() ? "button" : "submit"}
            onClick={isEmbeddedApp() ? save : undefined}
          >
            保存
          </button>
        </div>
      </form>
    </Modal>
  );
}

function AssetThumb({
  asset,
  projectId,
  revision,
  large = false,
}: {
  asset: Asset;
  projectId: string;
  revision?: AssetRevision;
  large?: boolean;
}) {
  const r = revision || asset.revisions[0];
  const f = r?.files.find((f) => f.mime.startsWith("image/"));
  const { url } = useAssetFileUrl(
    projectId,
    f ? r.id : undefined,
    f ? r.files.indexOf(f) : 0,
  );
  return (
    <div className={`asset-thumb ${large ? "large" : ""}`}>
      {f && url ? (
        <img loading="lazy" src={url} alt={asset.name} />
      ) : asset.kind === "model" ? (
        <Box size={large ? 36 : 16} strokeWidth={1.5} />
      ) : asset.kind === "document" ? (
        <FileText size={large ? 32 : 16} strokeWidth={1.5} />
      ) : (
        <Boxes size={large ? 32 : 16} strokeWidth={1.5} />
      )}
    </div>
  );
}

function CanvasModelPreview({
  projectId,
  asset,
  revision,
}: {
  projectId: string;
  asset: Asset;
  revision: AssetRevision;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  const index = revision.files.findIndex((f) => f.mime === "model/gltf-binary");
  useEffect(() => {
    const element = container.current;
    if (!element) return;
    const observer = new IntersectionObserver(
      ([entry]) => setVisible(entry.isIntersecting),
      { rootMargin: "160px" },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  // Do not transfer offscreen GLBs through the MCP bridge or keep their viewers mounted.
  const { url, error } = useAssetFileUrl(
    projectId,
    visible && index >= 0 ? revision.id : undefined,
    index,
  );
  return (
    <div ref={container} className="asset-thumb canvas-model-preview">
      {index < 0 || error ? (
        <div className="model-preview-state is-error" role="alert">
          <AlertTriangle size={16} />
          <span>{error ? "无法读取模型文件" : "此版本没有 GLB 文件"}</span>
        </div>
      ) : url ? (
        <ModelPreview src={url} name={asset.name} />
      ) : (
        <div className="model-preview-state" role="status">
          <Loader2 className="spin" size={16} />
          <span>等待预览模型…</span>
        </div>
      )}
    </div>
  );
}

function AssetNode({ data, selected }: NodeProps) {
  const d = data as any;
  return (
    <div className={`studio-node asset-node ${selected ? "is-selected" : ""}`}>
      <Handle type="target" position={Position.Left} />
      <div className="node-heading">
        <span>{kindName[d.asset?.kind] || "资源"}</span>
        <MoreHorizontal size={14} />
      </div>
      {d.asset && d.revision ? (
        <AssetPreviewSwitcher asset={d.asset} revision={d.revision}>
          {(preview) =>
            preview?.kind === "model" ? (
              <CanvasModelPreview
                asset={d.asset}
                projectId={d.projectId}
                revision={d.revision}
              />
            ) : preview?.kind === "document" ? (
              <CanvasDocumentPreview
                projectId={d.projectId}
                revision={d.revision}
              />
            ) : preview && ["video", "audio"].includes(preview.kind) ? (
              <CanvasMediaPreview
                projectId={d.projectId}
                asset={d.asset}
                revision={d.revision}
                index={preview.index}
              />
            ) : (
              <AssetThumb
                asset={d.asset}
                projectId={d.projectId}
                revision={d.revision}
                large
              />
            )
          }
        </AssetPreviewSwitcher>
      ) : (
        <div className="missing-preview">
          <AlertTriangle size={16} />
          资源不可用
        </div>
      )}
      <div className="node-caption">
        <strong>{d.title}</strong>
        <span className="mono">
          v{d.revision?.number || "?"}
          <span className="node-dot" />
          {d.asset?.kind === "model"
            ? "GLB"
            : d.revision?.files[0]?.name.split(".").pop()?.toUpperCase()}
        </span>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

function NoteNode({ data, selected }: NodeProps) {
  const text = String(data.text || "");
  return (
    <div className={`studio-node note-node ${selected ? "is-selected" : ""}`}>
      <Handle type="target" position={Position.Left} />
      <div className="node-heading">
        <span
          className="note-node-title"
          title={String(data.title || "制作便签")}
        >
          <StickyNote size={13} />
          <span>{String(data.title || "制作便签")}</span>
        </span>
        <Pencil size={12} style={{ color: "var(--text-muted)" }} />
      </div>
      <div className="note-body nodrag nowheel">
        {(data.customSize ? text : text.slice(0, 600)) || "暂无内容"}
      </div>
      <div className="note-footer">
        <span>{Array.from(text).length.toLocaleString()} 字</span>
        <button
          className="note-read-button nodrag nopan"
          onClick={(e) => {
            e.stopPropagation();
            (data.onRead as (() => void) | undefined)?.();
          }}
        >
          查看全文 <ArrowUpRight size={12} />
        </button>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

function WorkflowNode({ data, selected }: NodeProps) {
  const run = data.run as WorkflowRun | undefined;
  const step = run?.steps.find((s) => s.status.startsWith("awaiting"));
  return (
    <div
      className={`studio-node workflow-node ${selected ? "is-selected" : ""}`}
    >
      <Handle type="target" position={Position.Left} />
      <div className="workflow-node-head">
        <Workflow size={16} />
        <div>
          <strong>角色制作</strong>
          <small className="mono">8 阶段流水线</small>
        </div>
        <Badge tone="blue">{run ? statusName[run.status] : "未开始"}</Badge>
      </div>
      <div className="workflow-node-steps">
        {(
          run?.steps ||
          [
            "角色目标",
            "制作参考",
            "模型候选",
            "候选审核",
            "整理与组装",
            "绑定与变形",
            "变形审核",
            "成果交接",
          ].map((title, i) => ({ key: String(i), title, status: "pending" }))
        ).map((s, i) => (
          <div
            key={s.key}
            className={
              s.status === "succeeded"
                ? "done"
                : s.status.startsWith("awaiting")
                  ? "current"
                  : ""
            }
          >
            <span>
              {s.status === "succeeded" ? (
                <Check size={10} />
              ) : (
                String(i + 1).padStart(2, "0")
              )}
            </span>
            {s.title}
            {s.status.startsWith("awaiting") && (
              <small>{statusName[s.status]}</small>
            )}
          </div>
        ))}
      </div>
      <div className="workflow-node-footer">
        <div>
          <span className="status-dot" />
          {step
            ? `${step.title} · ${statusName[step.status]}`
            : run?.status === "succeeded"
              ? "已完成"
              : "点击展开制作"}
        </div>
        <ArrowUpRight size={13} />
      </div>
    </div>
  );
}

function ResizableNode(props: NodeProps) {
  const Content = nodeContents[props.type as keyof typeof nodeContents];
  return (
    <>
      <NodeResizer
        isVisible={props.selected && !props.data.saving}
        minWidth={160}
        minHeight={120}
        maxWidth={4000}
        maxHeight={4000}
        color="var(--accent-blue)"
        onResizeStart={() => (props.data.onResizeStart as () => void)()}
        onResizeEnd={(_event, size) =>
          (
            props.data.onResizeEnd as (
              id: string,
              size: { x: number; y: number; width: number; height: number },
            ) => void
          )(props.id, size)
        }
      />
      <div
        className={`resizable-node ${props.data.customSize ? "has-custom-size" : ""}`}
      >
        <Content {...props} />
      </div>
    </>
  );
}

const nodeContents = {
  asset: AssetNode,
  note: NoteNode,
  workflow: WorkflowNode,
  generation: GenerationNode,
};
const nodeTypes = {
  asset: ResizableNode,
  note: ResizableNode,
  workflow: ResizableNode,
  generation: ResizableNode,
};
const layoutSnapshot = (canvas: Canvas) =>
  canvas.nodes.map(({ id, position, width, height }) => ({
    id,
    position: { ...position },
    width,
    height,
  }));

function CanvasLaunchFocus({
  nodeId,
  focusing,
}: {
  nodeId?: string;
  focusing: { current: boolean };
}) {
  const initialized = useNodesInitialized();
  const flow = useReactFlow();
  const focused = useRef(false);
  useEffect(() => {
    if (!nodeId || !initialized || focused.current) return;
    focused.current = true;
    focusing.current = true;
    void flow.fitView({
      nodes: [{ id: nodeId }],
      padding: 0.4,
      minZoom: 0.2,
      maxZoom: 1,
      duration: 0,
    });
  }, [nodeId, initialized, flow, focusing]);
  return null;
}

function CanvasBoard({
  canvas: c,
  snapshot: s,
  selected,
  focusNodeId,
  onSelect,
  patch,
  saving,
  addAsset,
  onImageDrop,
  resolvedTheme,
  notify,
  onNote,
  onRenameNote,
  onImport,
  onResources,
  onView,
  onWorkflow,
  onResource,
  onTask,
  onContext,
}: {
  canvas: Canvas;
  snapshot: ProjectSnapshot;
  selected: string[];
  focusNodeId?: string;
  onSelect: (ids: string[]) => void;
  patch: (target: "graph" | "layout", ops: unknown[]) => Promise<any>;
  saving: boolean;
  addAsset: (
    a: Asset,
    r?: AssetRevision,
    position?: { x: number; y: number },
  ) => Promise<void>;
  onImageDrop: (
    drop: CanvasImageDrop,
    position: { x: number; y: number },
  ) => Promise<void>;
  resolvedTheme: "dark" | "light";
  notify: (message: string) => void;
  onNote: (
    position?: { x: number; y: number },
    node?: StudioNode,
    preview?: boolean,
  ) => void;
  onRenameNote: (node: StudioNode) => void;
  onImport: (position: { x: number; y: number }) => void;
  onResources: () => void;
  onView: () => void;
  onWorkflow: () => void;
  onResource: (
    kind: AddResourceKind,
    at?: { x: number; y: number },
    node?: StudioNode,
  ) => void;
  onTask: (id: string) => void;
  onContext: (nodeIds: string[]) => void;
}) {
  const [sizeEditor, setSizeEditor] = useState<{
    id: string;
    width: string;
    height: string;
  }>();
  const [sizeError, setSizeError] = useState("");
  const [draggingImage, setDraggingImage] = useState(false);
  const [importingImage, setImportingImage] = useState(false);
  const imageDropPending = useRef(false);
  const [menu, setMenu] = useState<{
    kind: "pane" | "node" | "edge";
    screen: { x: number; y: number };
    position: { x: number; y: number };
    ids: string[];
    edgeId?: string;
  }>();
  const closeMenu = () => setMenu(undefined);
  function openMenu(
    event: { preventDefault: () => void; clientX: number; clientY: number },
    kind: "pane" | "node" | "edge",
    id?: string,
  ) {
    event.preventDefault();
    const ids =
      kind === "node" && id ? (selected.includes(id) ? selected : [id]) : [];
    if (kind === "node") onSelect(ids);
    setMenu({
      kind,
      screen: { x: event.clientX, y: event.clientY },
      position: flow.current?.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      }) || { x: 0, y: 0 },
      ids,
      edgeId: kind === "edge" ? id : undefined,
    });
  }
  const [history, setHistory] = useState<
    Array<ReturnType<typeof layoutSnapshot>>
  >([]);
  const [redo, setRedo] = useState<typeof history>([]);
  const before = useRef<ReturnType<typeof layoutSnapshot>>([]);
  const flow = useRef<ReactFlowInstance<Node, Edge> | null>(null);
  const dragging = useRef(false);
  const launchFocusing = useRef(false);
  const [zoom, setZoom] = useState(c.viewport.zoom);
  const [canvasTool, setCanvasTool] = useState<"select" | "pan">("select");
  const panning = canvasTool === "pan";

  const beginLayout = useCallback(() => {
    setMenu(undefined);
    dragging.current = true;
    before.current = layoutSnapshot(c);
  }, [c]);

  const finishResize = useCallback(
    async (
      id: string,
      size: { x: number; y: number; width: number; height: number },
    ) => {
      try {
        await patch("layout", [
          { op: "resize", id, width: size.width, height: size.height },
          { op: "move", id, position: { x: size.x, y: size.y } },
        ]);
        setHistory((h) => [...h.slice(-29), before.current]);
        setRedo([]);
      } catch {
        setNodes((current) =>
          current.map((node) => {
            const saved = c.nodes.find((n) => n.id === node.id);
            return saved
              ? {
                  ...node,
                  position: saved.position,
                  data: { ...node.data, customSize: saved.width !== undefined },
                  width: saved.width,
                  height: saved.height,
                  style:
                    saved.width && saved.height
                      ? { width: saved.width, height: saved.height }
                      : undefined,
                }
              : node;
          }),
        );
      } finally {
        dragging.current = false;
      }
    },
    [c, patch],
  );

  const enrich = useCallback(
    () =>
      c.nodes.map((n) => {
        const a = s.assets.find((a) => a.id === n.data.assetId);
        return {
          ...n,
          selected: selected.includes(n.id),
          style:
            n.width && n.height
              ? { width: n.width, height: n.height }
              : undefined,
          data: {
            ...n.data,
            saving,
            customSize: n.width !== undefined,
            onResizeStart: beginLayout,
            onResizeEnd: finishResize,
            asset: a,
            revision: a?.revisions.find((r) => r.id === n.data.revisionId),
            projectId: s.project.id,
            onRead: () => onNote(n.position, n, true),
            task: s.tasks?.find((t) => t.id === n.data.taskId),
            run: s.runs
              .filter((r) => r.instanceId === n.data.instanceId)
              .at(-1),
          },
        } as Node;
      }),
    [
      c,
      s.assets,
      s.runs,
      s.tasks,
      s.project.id,
      selected,
      onNote,
      saving,
      beginLayout,
      finishResize,
    ],
  );

  const [nodes, setNodes] = useState<Node[]>(enrich);
  const [edges, setEdges] = useState<Edge[]>(
    c.edges.map((e) => ({
      ...e,
      label: "输入参考",
      markerEnd: { type: MarkerType.ArrowClosed },
      style: {
        stroke: resolvedTheme === "dark" ? "#52525b" : "#a1a1aa",
        strokeDasharray: "4 4",
      },
      labelStyle: {
        fill: resolvedTheme === "dark" ? "#a1a1aa" : "#71717a",
        fontSize: 10,
      },
    })),
  );
  useEffect(() => {
    if (!dragging.current)
      setNodes((current) =>
        enrich().map((node) => ({
          ...node,
          // Selection and asset updates must retain React Flow's measurements.
          // Dropping them hides unchanged nodes until ResizeObserver fires again.
          measured: current.find((previous) => previous.id === node.id)
            ?.measured,
        })),
      );
  }, [enrich]);

  const nodeCount = useRef(c.nodes.length);
  useEffect(() => {
    const added = c.nodes.length > nodeCount.current;
    nodeCount.current = c.nodes.length;
    if (added) {
      const timer = setTimeout(
        () =>
          flow.current?.fitView({ padding: 0.2, duration: 180, maxZoom: 1 }),
        100,
      );
      return () => clearTimeout(timer);
    }
  }, [c.nodes.length]);

  useEffect(
    () =>
      setEdges(
        c.edges.map((e) => ({
          ...e,
          label: "输入参考",
          markerEnd: { type: MarkerType.ArrowClosed },
          style: {
            stroke: resolvedTheme === "dark" ? "#52525b" : "#a1a1aa",
            strokeDasharray: "4 4",
          },
          labelStyle: {
            fill: resolvedTheme === "dark" ? "#a1a1aa" : "#71717a",
            fontSize: 10,
          },
        })),
      ),
    [c.edges, resolvedTheme],
  );

  async function saveSize(reset = false) {
    if (!sizeEditor) return;
    const width = Number(sizeEditor.width),
      height = Number(sizeEditor.height);
    if (
      !reset &&
      (!Number.isFinite(width) ||
        !Number.isFinite(height) ||
        width < 160 ||
        height < 120 ||
        width > 4000 ||
        height > 4000)
    ) {
      setSizeError("宽度需为 160–4000，高度需为 120–4000。");
      return;
    }
    const previous = layoutSnapshot(c);
    try {
      await patch("layout", [
        {
          op: "resize",
          id: sizeEditor.id,
          width: reset ? null : width,
          height: reset ? null : height,
        },
      ]);
      setHistory((h) => [...h.slice(-29), previous]);
      setRedo([]);
      setSizeEditor(undefined);
    } catch (error) {
      setSizeError(String(error));
    }
  }

  async function undo(forward = false) {
    const stack = forward ? redo : history;
    const to = stack.at(-1);
    if (!to) return;
    try {
      const current = layoutSnapshot(c);
      await patch(
        "layout",
        to
          .filter((n) => c.nodes.some((c) => c.id === n.id))
          .flatMap((n) => [
            { op: "move", id: n.id, position: n.position },
            {
              op: "resize",
              id: n.id,
              width: n.width ?? null,
              height: n.height ?? null,
            },
          ]),
      );
      if (forward) {
        setRedo(redo.slice(0, -1));
        setHistory([...history, current]);
      } else {
        setHistory(history.slice(0, -1));
        setRedo([...redo, current]);
      }
    } catch {}
  }

  const menuNodes = c.nodes.filter((n) => menu?.ids.includes(n.id));
  const canCopy =
    menuNodes.length > 0 &&
    menuNodes.every((n) => n.type !== "workflow" && n.type !== "generation");
  const single = menuNodes.length === 1 ? menuNodes[0] : undefined;
  async function paste(value = canvasClipboard, position = menu?.position) {
    if (!value || !position) return;
    const result = pasteCanvasSelection(value, s.project.id, position, () =>
      uuid(),
    );
    await patch("graph", result.ops);
    onSelect(result.nodes.map((n) => n.id));
    notify("已粘贴节点与选中节点之间的连线");
  }
  const [addMenu, setAddMenu] = useState<{
    screen: { x: number; y: number };
    position: { x: number; y: number };
  }>();
  const menuGroups: CanvasMenuAction[][] = !menu
    ? []
    : menu.kind === "pane"
      ? [
          [
            {
              label: "添加资源…",
              icon: <Plus size={15} />,
              disabled: saving,
              run: () => {
                setAddMenu({ screen: menu.screen, position: menu.position });
              },
            },
            {
              label: "新建文档",
              icon: <FileText size={15} />,
              disabled: saving,
              run: () => onResource("document", menu.position),
            },
            {
              label: "引用已有文档",
              icon: <FileText size={15} />,
              disabled: saving,
              run: () => onResource("existing-document", menu.position),
            },
            {
              label: "添加便签",
              icon: <StickyNote size={15} />,
              disabled: saving,
              run: () => onNote(menu.position),
            },
            {
              label: "导入素材到此处",
              icon: <Upload size={15} />,
              disabled: saving,
              run: () => onImport(menu.position),
            },
            {
              label: "打开项目资源栏",
              icon: <FolderOpen size={15} />,
              run: onResources,
            },
          ],
          [
            {
              label: "粘贴节点",
              icon: <Clipboard size={15} />,
              disabled:
                saving ||
                !canvasClipboard ||
                canvasClipboard.projectId !== s.project.id,
              run: () => paste(),
            },
            {
              label: "撤销布局",
              icon: <Undo2 size={15} />,
              disabled: saving || !history.length,
              run: () => undo(),
            },
            {
              label: "重做布局",
              icon: <Redo2 size={15} />,
              disabled: saving || !redo.length,
              run: () => undo(true),
            },
          ],
          [
            {
              label: "全选节点",
              icon: <MousePointer2 size={15} />,
              disabled: !c.nodes.length,
              run: () => onSelect(c.nodes.map((n) => n.id)),
            },
            {
              label: "适应全部节点",
              icon: <Maximize size={15} />,
              run: () => flow.current?.fitView({ padding: 0.2, maxZoom: 1 }),
            },
          ],
        ]
      : menu.kind === "edge"
        ? [
            [
              {
                label: "断开输入连线",
                icon: <Unplug size={15} />,
                disabled: saving,
                run: () =>
                  patch("graph", [{ op: "disconnect", id: menu.edgeId }]),
              },
            ],
          ]
        : [
            [
              ...(single
                ? [
                    {
                      label:
                        single.type === "note"
                          ? "编辑便签"
                          : single.type === "workflow"
                            ? "查看制作流程"
                            : "查看素材",
                      icon: <Pencil size={15} />,
                      run: () =>
                        single.type === "note"
                          ? onNote(undefined, single)
                          : onView(),
                    },
                    ...(single.type === "note"
                      ? [
                          {
                            label: "重命名便签",
                            icon: <Pencil size={15} />,
                            disabled: saving,
                            run: () => onRenameNote(single),
                          },
                        ]
                      : []),
                    {
                      label: "设置节点大小",
                      icon: <Maximize size={15} />,
                      disabled: saving,
                      run: () => {
                        const live = flow.current?.getNode(single.id);
                        setSizeError("");
                        setSizeEditor({
                          id: single.id,
                          width: String(
                            Math.round(
                              live?.measured?.width || single.width || 220,
                            ),
                          ),
                          height: String(
                            Math.round(
                              live?.measured?.height || single.height || 200,
                            ),
                          ),
                        });
                      },
                    },
                  ]
                : []),
              ...(single?.data.revisionId
                ? [
                    {
                      label: "下载素材主文件",
                      icon: <Download size={15} />,
                      run: async () => {
                        try {
                          const route = fileUrl(
                            s.project.id,
                            single.data.revisionId!,
                          );
                          if (isEmbeddedApp()) await downloadAssetFile(route);
                          else {
                            const link = document.createElement("a");
                            link.href = `${route}?download=1`;
                            link.download = "";
                            document.body.append(link);
                            link.click();
                            link.remove();
                          }
                        } catch {
                          notify("下载失败，请重试");
                        }
                      },
                    },
                  ]
                : []),
              {
                label: "以所选节点制作角色",
                icon: <Workflow size={15} />,
                disabled: saving || !canCopy || menuNodes.length > 20,
                run: onWorkflow,
              },
              {
                label: "添加到对话引用…",
                icon: <Link2 size={15} />,
                run: () => onContext(menu.ids),
              },
            ],
            [
              {
                label: "复制节点",
                icon: <Copy size={15} />,
                disabled: !canCopy,
                run: () => {
                  copyCanvasSelection(c, menu.ids, s.project.id);
                  notify("已复制，可在本项目画布右键粘贴");
                },
              },
              {
                label: "创建副本",
                icon: <Plus size={15} />,
                disabled: saving || !canCopy,
                run: () =>
                  paste(copyCanvasSelection(c, menu.ids, s.project.id), {
                    x: Math.min(...menuNodes.map((n) => n.position.x)) + 40,
                    y: Math.min(...menuNodes.map((n) => n.position.y)) + 40,
                  }),
              },
            ],
            [
              {
                label: "从画布移除",
                icon: <Trash2 size={15} />,
                danger: true,
                disabled: saving || !canCopy,
                hint: canCopy ? "保留素材" : "保留运行记录",
                run: async () => {
                  await patch(
                    "graph",
                    menu.ids.map((id) => ({ op: "remove", id })),
                  );
                  onSelect([]);
                },
              },
            ],
          ];
  return (
    <div
      className={`canvas-area${panning ? " is-panning" : ""}`}
      tabIndex={-1}
      onDragOver={(e) => {
        if (!acceptsCanvasDrop(e.dataTransfer)) return;
        e.preventDefault();
        e.stopPropagation();
        e.dataTransfer.dropEffect =
          imageDropPending.current || saving ? "none" : "copy";
        setDraggingImage(
          !e.dataTransfer.types.includes("application/studio-asset"),
        );
      }}
      onDragLeave={(e) => {
        if (
          !e.currentTarget.contains(e.relatedTarget as globalThis.Node | null)
        )
          setDraggingImage(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        e.stopPropagation();
        setDraggingImage(false);
        if (imageDropPending.current || saving) {
          notify("正在保存，请稍后再拖入图片");
          return;
        }
        const position = flow.current?.screenToFlowPosition({
          x: e.clientX,
          y: e.clientY,
        });
        if (!position) return;
        const a = s.assets.find(
          (a) => a.id === e.dataTransfer.getData("application/studio-asset"),
        );
        if (a) {
          addAsset(a, undefined, position).catch(() => {});
          return;
        }
        if (e.dataTransfer.types.includes("application/studio-asset")) return;
        const drop = captureCanvasImageDrop(e.dataTransfer);
        imageDropPending.current = true;
        setImportingImage(true);
        onImageDrop(drop, position)
          .catch(() => {})
          .finally(() => {
            imageDropPending.current = false;
            setImportingImage(false);
          });
      }}
    >
      {(draggingImage || importingImage) && (
        <div className="canvas-drop-hint" role="status">
          {importingImage ? (
            <Loader2 size={22} className="spin" />
          ) : (
            <ImageIcon size={22} />
          )}
          <strong>
            {importingImage ? "正在导入图片…" : "松开以添加图片节点"}
          </strong>
          {!importingImage && <span>图片会保存到当前项目</span>}
        </div>
      )}
      <div className="canvas-floating-tools">
        <IconButton
          label="撤销布局"
          disabled={!history.length || saving}
          onClick={() => undo()}
        >
          <Undo2 size={14} />
        </IconButton>
        <IconButton
          label="重做布局"
          disabled={!redo.length || saving}
          onClick={() => undo(true)}
        >
          <Redo2 size={14} />
        </IconButton>
      </div>
      <div className="canvas-add-control" role="toolbar" aria-label="画布工具">
        <button
          type="button"
          className="icon-button"
          aria-label="画布添加资源"
          title="添加资源"
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            setAddMenu({
              screen: { x: rect.left, y: rect.top - 590 },
              position: freePosition(c, {
                x: (160 - c.viewport.x) / c.viewport.zoom,
                y: (100 - c.viewport.y) / c.viewport.zoom,
              }),
            });
          }}
        >
          <Plus size={20} />
        </button>
        <button
          type="button"
          className="icon-button"
          aria-label="选择"
          title="选择 · 拖动卡片或框选"
          aria-pressed={!panning}
          onClick={() => {
            closeMenu();
            setCanvasTool("select");
            onSelect([]);
          }}
        >
          <MousePointer2 size={20} />
        </button>
        <button
          type="button"
          className="icon-button"
          aria-label="抓手"
          title="抓手 · 在卡片或空白处拖动画布"
          aria-pressed={panning}
          onClick={() => {
            closeMenu();
            setCanvasTool("pan");
          }}
        >
          <Hand size={20} />
        </button>
        <button
          type="button"
          className="icon-button"
          aria-label="适应画布"
          title="适应画布"
          onClick={() =>
            flow.current?.fitView({ padding: 0.2, duration: 180, maxZoom: 1 })
          }
        >
          <Maximize size={20} />
        </button>
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onInit={(instance) => {
          flow.current = instance;
        }}
        onNodesChange={(changes) =>
          setNodes((n) =>
            applyNodeChanges(changes, n).map((node) =>
              changes.some(
                (change) =>
                  change.type === "dimensions" &&
                  change.id === node.id &&
                  change.setAttributes,
              )
                ? { ...node, data: { ...node.data, customSize: true } }
                : node,
            ),
          )
        }
        onEdgesChange={(changes) =>
          setEdges((e) => applyEdgeChanges(changes, e))
        }
        onNodeClick={(e, n) =>
          onSelect(
            e.shiftKey
              ? selected.includes(n.id)
                ? selected.filter((id) => id !== n.id)
                : [...selected, n.id]
              : [n.id],
          )
        }
        onNodeDoubleClick={(_e, n) => {
          const node = c.nodes.find((x) => x.id === n.id);
          if (node?.type === "generation" && node.data.taskId)
            onTask(node.data.taskId);
          else if (
            node?.type === "asset" &&
            s.assets.find((a) => a.id === node.data.assetId)?.kind ===
              "document"
          )
            onResource("document", node.position, node);
          else if (node?.type === "note") onNote(node.position, node, true);
          else onView();
        }}
        onPaneClick={() => {
          closeMenu();
          if (!panning) onSelect([]);
        }}
        onPaneContextMenu={(e) => openMenu(e, "pane")}
        onNodeContextMenu={(e, n) => openMenu(e, "node", n.id)}
        onEdgeContextMenu={(e, edge) => openMenu(e, "edge", edge.id)}
        onSelectionContextMenu={(e) => {
          const id = selected[0];
          if (id) openMenu(e, "node", id);
        }}
        onMoveStart={(event) => {
          if (event) closeMenu();
        }}
        onSelectionEnd={() =>
          onSelect(
            flow.current
              ?.getNodes()
              .filter((n) => n.selected)
              .map((n) => n.id) || [],
          )
        }
        onNodeDragStart={beginLayout}
        onNodeDragStop={async (_e, _n, moved) => {
          try {
            await patch(
              "layout",
              moved.map((n) => ({
                op: "move",
                id: n.id,
                position: n.position,
              })),
            );
            setHistory((h) => [...h.slice(-29), before.current]);
            setRedo([]);
          } catch {
            setNodes(enrich());
          } finally {
            dragging.current = false;
          }
        }}
        onConnect={(connection) => {
          if (connection.source && connection.target)
            patch("graph", [
              {
                op: "connect",
                edge: {
                  id: uuid("e"),
                  source: connection.source,
                  target: connection.target,
                  kind: "reference",
                },
              },
            ]).catch(() => {});
        }}
        isValidConnection={(connection) =>
          !!connection.source &&
          !!connection.target &&
          !canvasConnectionError(c, connection.source, connection.target)
        }
        onBeforeDelete={async ({ nodes, edges }) => {
          try {
            await patch("graph", [
              ...nodes.map((n) => ({ op: "remove", id: n.id })),
              ...edges.map((e) => ({ op: "disconnect", id: e.id })),
            ]);
            return true;
          } catch {
            return false;
          }
        }}
        onMove={(_e, v) => setZoom(v.zoom)}
        onMoveEnd={(event, v) => {
          if (launchFocusing.current) {
            launchFocusing.current = false;
            if (!event) return;
          }
          if (
            Math.abs(v.x - c.viewport.x) > 1 ||
            Math.abs(v.y - c.viewport.y) > 1 ||
            Math.abs(v.zoom - c.viewport.zoom) > 0.01
          )
            patch("layout", [{ op: "viewport", viewport: v }]).catch(() => {});
        }}
        defaultViewport={c.viewport}
        minZoom={0.2}
        maxZoom={2}
        panOnScroll
        selectionOnDrag={!panning}
        panOnDrag={panning ? [0, 1] : [1]}
        nodesDraggable={!panning}
        nodesConnectable={!panning}
        elementsSelectable={!panning}
        selectionKeyCode={panning ? null : "Shift"}
        zoomOnDoubleClick={!panning}
        deleteKeyCode={
          menu || sizeEditor || panning ? null : ["Backspace", "Delete"]
        }
        colorMode={resolvedTheme === "light" ? "light" : "dark"}
      >
        <CanvasLaunchFocus nodeId={focusNodeId} focusing={launchFocusing} />
        <Background
          color={resolvedTheme === "dark" ? "#27272a" : "#e4e4e7"}
          gap={20}
          size={1}
        />
        <MiniMap
          nodeColor={(n) =>
            n.type === "note"
              ? "#fbbf24"
              : n.type === "workflow"
                ? "#38bdf8"
                : "#71717a"
          }
          maskColor={
            resolvedTheme === "dark"
              ? "rgba(18, 18, 20, 0.75)"
              : "rgba(248, 249, 250, 0.8)"
          }
        />
        <Controls showInteractive={false} />
      </ReactFlow>
      {sizeEditor && (
        <Modal
          title="设置节点大小"
          onClose={() => !saving && setSizeEditor(undefined)}
        >
          <div
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                event.target instanceof HTMLInputElement
              ) {
                event.preventDefault();
                if (!saving) void saveSize();
              }
            }}
          >
            <div className="modal-body node-size-fields">
              <label>
                宽度
                <input
                  aria-label="节点宽度"
                  type="number"
                  min="160"
                  max="4000"
                  step="any"
                  required
                  value={sizeEditor.width}
                  onChange={(e) =>
                    setSizeEditor({ ...sizeEditor, width: e.target.value })
                  }
                />
              </label>
              <label>
                高度
                <input
                  aria-label="节点高度"
                  type="number"
                  min="120"
                  max="4000"
                  step="any"
                  required
                  value={sizeEditor.height}
                  onChange={(e) =>
                    setSizeEditor({ ...sizeEditor, height: e.target.value })
                  }
                />
              </label>
              <p className="panel-hint">
                选中节点后，也可拖动边框或四角调整大小。
              </p>
              {sizeError && (
                <p className="resource-error" role="alert">
                  {sizeError}
                </p>
              )}
            </div>
            <div className="modal-footer">
              <button
                type="button"
                className="button small"
                disabled={saving}
                onClick={() => saveSize(true)}
              >
                恢复默认大小
              </button>
              <button
                type="button"
                className="button small"
                disabled={saving}
                onClick={() => setSizeEditor(undefined)}
              >
                取消
              </button>
              <button
                type="button"
                className="button primary small"
                disabled={saving}
                onClick={() => saveSize()}
              >
                保存大小
              </button>
            </div>
          </div>
        </Modal>
      )}
      {menu && (
        <CanvasContextMenu
          position={menu.screen}
          label={
            menu.kind === "pane"
              ? "画布操作"
              : menu.kind === "edge"
                ? "连线操作"
                : menu.ids.length > 1
                  ? `已选 ${menu.ids.length} 个节点`
                  : "节点操作"
          }
          groups={
            menu.kind === "node"
              ? [
                  ...menuGroups,
                  resourceChoices
                    .filter((choice) =>
                      [
                        "image",
                        "video",
                        "model",
                        "animation",
                        "audio",
                      ].includes(choice.kind),
                    )
                    .map((choice) => ({
                      label: `制作${choice.label}`,
                      icon: <choice.icon size={15} />,
                      disabled:
                        saving ||
                        !menu.ids.some((id) =>
                          c.nodes.some(
                            (n) =>
                              n.id === id && ["asset", "note"].includes(n.type),
                          ),
                        ),
                      run: () =>
                        onResource(
                          choice.kind,
                          freePosition(c, {
                            x: menu.position.x + 300,
                            y: menu.position.y,
                          }),
                        ),
                    })),
                  ...(menu.ids.length === 1 &&
                  c.nodes.find((n) => n.id === menu.ids[0])?.type === "asset" &&
                  s.assets.find(
                    (a) =>
                      a.id ===
                      c.nodes.find((n) => n.id === menu.ids[0])?.data.assetId,
                  )?.kind === "document"
                    ? [
                        [
                          {
                            label: "编辑文档",
                            icon: <FileText size={15} />,
                            run: () => {
                              const n = c.nodes.find(
                                (n) => n.id === menu.ids[0],
                              )!;
                              onResource("document", n.position, n);
                            },
                          },
                        ],
                      ]
                    : []),
                  ...(menu.ids.length === 1 &&
                  c.nodes.find((n) => n.id === menu.ids[0])?.data.taskId
                    ? [
                        [
                          {
                            label: "查看制作任务",
                            run: () =>
                              onTask(
                                c.nodes.find((n) => n.id === menu.ids[0])!.data
                                  .taskId!,
                              ),
                          },
                        ],
                      ]
                    : []),
                ]
              : menuGroups
          }
          onClose={closeMenu}
        />
      )}
      {addMenu && (
        <ResourceMenu
          position={addMenu.screen}
          onClose={() => setAddMenu(undefined)}
          onChoose={(kind) => onResource(kind, addMenu.position)}
        />
      )}
      {!c.nodes.length && (
        <div className="canvas-empty">
          <Layers size={28} />
          <h3>这是你的创作画布</h3>
          <p>
            从资源栏拖入素材，或点击上方「便签」。
            <br />
            选中素材后点击「制作角色」开始生产。
          </p>
        </div>
      )}
      <div className="canvas-caption">
        <span>
          {panning
            ? "抓手模式 · 拖动卡片或空白处移动画布"
            : "连线将上游素材与便签作为制作输入"}
        </span>
        <span className="mono">{Math.round(zoom * 100)}%</span>
      </div>
    </div>
  );
}

function Inspector({
  snapshot: s,
  asset,
  node,
  run,
  fixedRevisionId,
  launchStepKey,
  addAsset,
  mutate,
  patch,
  notify,
  selected,
  canvas,
  onImportVersion,
  onImportPathVersion,
  onPreviewNote,
  onContext,
  onOpenCharacterWorkflow,
}: {
  snapshot: ProjectSnapshot;
  asset?: Asset;
  node?: StudioNode;
  run?: WorkflowRun;
  fixedRevisionId?: string;
  launchStepKey?: string;
  addAsset: (
    a: Asset,
    r?: AssetRevision,
    position?: { x: number; y: number },
  ) => Promise<void>;
  mutate: (name: string, a: unknown) => Promise<any>;
  patch: (target: "graph" | "layout", ops: unknown[]) => Promise<any>;
  notify: (s: string) => void;
  selected: string[];
  canvas: Canvas;
  onImportVersion: (id: string) => void;
  onImportPathVersion: (id: string) => void;
  onPreviewNote: () => void;
  onContext: () => void;
  onOpenCharacterWorkflow?: (runId?: string) => void;
}) {
  const [tab, setTab] = useState("preview");
  const [previewRevisionId, setPreviewRevisionId] = useState(
    fixedRevisionId || asset?.revisions[0]?.id || "",
  );
  const [note, setNote] = useState(node?.data.text || "");
  const [noteTitle, setNoteTitle] = useState(node?.data.title || "制作便签");

  useEffect(() => {
    setPreviewRevisionId(fixedRevisionId || asset?.revisions[0]?.id || "");
    setTab("preview");
  }, [asset?.id, fixedRevisionId]);

  useEffect(() => setNote(node?.data.text || ""), [node?.id, node?.data.text]);
  useEffect(
    () => setNoteTitle(node?.data.title || "制作便签"),
    [node?.id, node?.data.title],
  );

  const r =
    asset?.revisions.find((r) => r.id === previewRevisionId) ||
    asset?.revisions[0];

  return (
    <aside className="inspector">
      <div className="inspector-title">
        <strong>
          {run || node?.type === "workflow"
            ? "制作流程"
            : asset
              ? "资源详情"
              : node?.type === "note"
                ? "制作便签"
                : "画布详情"}
        </strong>
      </div>
      {run || node?.type === "workflow" ? (
        <WorkflowInspector
          snapshot={s}
          run={run}
          launchStepKey={launchStepKey}
          instanceId={node?.data.instanceId}
          canvas={canvas}
          mutate={mutate}
          notify={notify}
          onOpenCharacterWorkflow={onOpenCharacterWorkflow}
        />
      ) : asset && r ? (
        <>
          <div className="inspector-resource-name">
            <h3>{asset.name}</h3>
            <span>
              {kindName[asset.kind]}
              <Badge>v{r.number}</Badge>
            </span>
          </div>
          <AssetCategoryEditor
            asset={asset}
            projectId={s.project.id}
            mutate={mutate}
            notify={notify}
          />
          {node && canvas.edges.some((e) => e.target === node.id) && (
            <NodeInputSummary
              canvas={canvas}
              nodeId={node.id}
              assets={s.assets}
            />
          )}
          <div className="inspector-tabs">
            {[
              ["preview", "预览"],
              ["versions", "版本"],
              ["usage", "使用位置"],
            ].map(([v, n]) => (
              <button
                key={v}
                className={tab === v ? "active" : ""}
                onClick={() => setTab(v)}
              >
                {n}
              </button>
            ))}
          </div>
          {tab === "preview" ? (
            <>
              <AssetPreview
                projectId={s.project.id}
                asset={asset}
                revision={r}
              />
              <div className="detail-list">
                <div>
                  <span>来源</span>
                  <strong>{r.provenance.sourceRef}</strong>
                </div>
                <div>
                  <span>文件数</span>
                  <strong>
                    {r.files.length} 个 (
                    {(
                      r.files.reduce((v, f) => v + f.sizeBytes, 0) / 1024
                    ).toFixed(0)}{" "}
                    KB)
                  </strong>
                </div>
                <div>
                  <span>导入时间</span>
                  <strong>{stamp(r.createdAt)}</strong>
                </div>
              </div>
              <div className="inspector-action-stack">
                {asset.kind === "image" &&
                  r.files.some((f) => f.mime.startsWith("image/")) && (
                    <button
                      className="button small"
                      onClick={() =>
                        mutate("project_manage", {
                          action: "update",
                          projectId: s.project.id,
                          cover: {
                            revisionId: r.id,
                            fileIndex: r.files.findIndex((f) =>
                              f.mime.startsWith("image/"),
                            ),
                          },
                        })
                          .then(() => notify("已设为项目封面"))
                          .catch(() => {})
                      }
                    >
                      <ImageIcon size={13} />
                      设为项目封面
                    </button>
                  )}
                <button
                  className="button small"
                  onClick={() => addAsset(asset, r).catch(() => {})}
                >
                  <Plus size={13} />
                  添加到当前画布
                </button>
                <button
                  className="button small"
                  onClick={() => onImportVersion(asset.id)}
                >
                  <Upload size={13} />
                  导入新版本
                </button>
                <button
                  className="button small"
                  onClick={() => onImportPathVersion(asset.id)}
                >
                  <FolderOpen size={13} />
                  引用路径更新版本
                </button>
                {selected.length > 0 && (
                  <button className="button small primary" onClick={onContext}>
                    <Link2 size={13} />
                    添加到对话引用…
                  </button>
                )}
              </div>
              <details className="file-details">
                <summary>文件与完整性哈希</summary>
                {r.files.map((f, i) => (
                  <div key={f.name}>
                    <AssetFileLink
                      projectId={s.project.id}
                      revisionId={r.id}
                      index={i}
                      name={f.name}
                    />
                    <code className="mono muted" title={f.relativePath}>
                      路径 {f.relativePath}
                    </code>
                    <code className="mono muted" title={f.sha256}>
                      SHA256 {f.sha256.slice(0, 16)}…
                    </code>
                  </div>
                ))}
              </details>
            </>
          ) : tab === "versions" ? (
            <div className="version-list">
              {asset.revisions.map((rev) => (
                <button
                  key={rev.id}
                  className={rev.id === r.id ? "selected" : ""}
                  onClick={() => setPreviewRevisionId(rev.id)}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                    }}
                  >
                    <strong>版本 {rev.number}</strong>
                    {rev.id === fixedRevisionId && (
                      <Badge tone="blue">画布引用</Badge>
                    )}
                  </div>
                  <small className="muted">
                    {stamp(rev.createdAt)} · {rev.files.length} 个文件
                  </small>
                </button>
              ))}
            </div>
          ) : (
            <div className="usage-list">
              {s.canvases.flatMap((c) =>
                c.nodes
                  .filter((n) => n.data.assetId === asset.id)
                  .map((n) => (
                    <div
                      key={n.id}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                        padding: "8px",
                        background: "var(--bg-surface)",
                        borderRadius: "6px",
                      }}
                    >
                      <Layers size={14} />
                      <div>
                        <strong>{c.name}</strong>
                        <small className="muted" style={{ marginLeft: "6px" }}>
                          引用 v
                          {
                            asset.revisions.find(
                              (r) => r.id === n.data.revisionId,
                            )?.number
                          }
                        </small>
                      </div>
                    </div>
                  )),
              )}
            </div>
          )}
        </>
      ) : node?.type === "note" ? (
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          <label className="field" style={{ margin: 0 }}>
            便签名称
            <input
              aria-label="便签名称"
              maxLength={120}
              value={noteTitle}
              onChange={(e) => setNoteTitle(e.target.value)}
            />
          </label>
          <button className="button small" onClick={onPreviewNote}>
            <Maximize size={13} />
            查看全文
          </button>
          {canvas.edges.some((e) => e.target === node.id) && (
            <NodeInputSummary
              canvas={canvas}
              nodeId={node.id}
              assets={s.assets}
            />
          )}
          <label className="field" style={{ margin: 0 }}>
            便签内容
            <textarea
              rows={8}
              value={note}
              onChange={(e) => setNote(e.target.value)}
            />
          </label>
          <button
            className="button small primary"
            disabled={!noteTitle.trim()}
            onClick={() =>
              patch("graph", [
                {
                  op: "note",
                  id: node.id,
                  text: note,
                  title: noteTitle.trim(),
                },
              ]).catch(() => {})
            }
          >
            保存便签
          </button>
          <button
            className="button small danger"
            onClick={() =>
              patch("graph", [{ op: "remove", id: node.id }]).catch(() => {})
            }
          >
            移除节点
          </button>
          <button className="button small" onClick={onContext}>
            <Link2 size={13} />
            添加到对话引用…
          </button>
        </div>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "14px" }}>
          <div className="detail-list">
            <div>
              <span>资源节点</span>
              <strong>
                {canvas.nodes.filter((n) => n.type === "asset").length}
              </strong>
            </div>
            <div>
              <span>制作分组</span>
              <strong>{canvas.instances.length}</strong>
            </div>
            <div>
              <span>参考连接</span>
              <strong>{canvas.edges.length}</strong>
            </div>
          </div>
          <div
            style={{
              padding: "12px",
              background: "var(--bg-surface)",
              borderRadius: "8px",
              display: "flex",
              flexDirection: "column",
              gap: "6px",
              fontSize: "11px",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span className="muted">框选节点</span>
              <kbd>拖动空白</kbd>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span className="muted">平移画布</span>
              <kbd>滚动 / 中键</kbd>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span className="muted">缩放画布</span>
              <kbd>双指捏合</kbd>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
              <span className="muted">删除节点</span>
              <kbd>Delete</kbd>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}

function AssetFileLink({
  projectId,
  revisionId,
  index,
  name,
}: {
  projectId: string;
  revisionId: string;
  index: number;
  name: string;
}) {
  const [error, setError] = useState("");
  return (
    <>
      <a
        href={`${fileUrl(projectId, revisionId, index)}?download=1`}
        style={{ display: "flex", alignItems: "center", gap: "6px" }}
        onClick={(event) => {
          if (!isEmbeddedApp()) return;
          event.preventDefault();
          downloadAssetFile(fileUrl(projectId, revisionId, index)).catch(
            (error) => setError(String(error)),
          );
        }}
      >
        <Download size={12} />
        {name}
      </a>
      {error && <span role="alert">{error}</span>}
    </>
  );
}

function AssetPreview({
  projectId,
  asset,
  revision: r,
}: {
  projectId: string;
  asset: Asset;
  revision: AssetRevision;
}) {
  return (
    <AssetPreviewSwitcher asset={asset} revision={r}>
      {(preview) => (
        <AssetFilePreview
          key={`${r.id}:${preview?.index ?? 0}`}
          projectId={projectId}
          asset={asset}
          revision={r}
          preview={preview}
        />
      )}
    </AssetPreviewSwitcher>
  );
}

function AssetFilePreview({
  projectId,
  asset,
  revision: r,
  preview,
}: {
  projectId: string;
  asset: Asset;
  revision: AssetRevision;
  preview?: PreviewOption;
}) {
  const [text, setText] = useState("");
  const [bad, setBad] = useState(false);
  const i = preview?.index ?? 0;
  const kind = preview?.kind ?? asset.kind;
  const f = r.files[i];
  const { url, error } = useAssetFileUrl(projectId, r.id, i);

  useEffect(() => {
    setBad(false);
    setText("");
    if (f.mime === "text/plain" && url)
      fetch(url)
        .then((x) => x.text())
        .then((t) => setText(t.slice(0, 15000)))
        .catch(() => setBad(true));
  }, [url, f.mime]);

  return (
    <div className="asset-preview">
      {bad || error ? (
        <div
          style={{
            textAlign: "center",
            color: "var(--accent-red)",
            padding: "16px",
          }}
        >
          <AlertTriangle size={20} />
          <p style={{ marginTop: "6px", fontSize: "11px" }}>无法读取文件内容</p>
        </div>
      ) : !url ? (
        <span>正在读取资源…</span>
      ) : kind === "image" ? (
        <img src={url} alt={asset.name} onError={() => setBad(true)} />
      ) : kind === "model" ? (
        <ModelPreview src={url} name={asset.name} />
      ) : kind === "document" ? (
        <Markdown text={text || "正在读取文档…"} />
      ) : kind === "video" ? (
        <video
          controls
          playsInline
          preload="metadata"
          src={url}
          aria-label={asset.name}
          onError={() => setBad(true)}
        />
      ) : kind === "audio" ? (
        <audio
          controls
          preload="metadata"
          src={url}
          aria-label={asset.name}
          onError={() => setBad(true)}
        />
      ) : (
        <div
          style={{
            textAlign: "center",
            color: "var(--text-muted)",
            padding: "16px",
          }}
        >
          <Boxes size={32} />
          <p style={{ marginTop: "6px", fontSize: "11px" }}>源文件资产</p>
        </div>
      )}
    </div>
  );
}

function previewInputs(
  canvas: Canvas,
  nodeId: string | undefined,
  assets: Asset[],
) {
  try {
    return {
      snapshot: nodeId
        ? resolveCanvasInputs(canvas, [nodeId], assets)
        : undefined,
      error: "",
    };
  } catch (error) {
    return {
      snapshot: undefined,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function NodeInputSummary({
  canvas,
  nodeId,
  assets,
}: {
  canvas: Canvas;
  nodeId: string;
  assets: Asset[];
}) {
  const inputs = previewInputs(canvas, nodeId, assets);
  return inputs.error ? (
    <p className="input-error" role="alert">
      {inputs.error}
    </p>
  ) : inputs.snapshot ? (
    <InputSummary snapshot={inputs.snapshot} title="节点制作上下文" />
  ) : null;
}

function InputSummary({
  snapshot,
  title,
}: {
  snapshot: CanvasInputSnapshot;
  title: string;
}) {
  return (
    <section className="input-summary" aria-label={title}>
      <div className="input-summary-heading">
        <strong>{title}</strong>
        <small>{snapshot.items.length} 项</small>
      </div>
      {!snapshot.items.length && (
        <p className="panel-hint">暂无输入，请连接素材或便签。</p>
      )}
      {snapshot.items.map((item, index) => (
        <div
          className="input-summary-item"
          key={`${item.nodeId || item.revisionId}-${index}`}
        >
          <div className="input-summary-label">
            {item.type === "note" ? (
              <StickyNote size={13} />
            ) : item.role === "prompt" ? (
              <FileText size={13} />
            ) : (
              <Boxes size={13} />
            )}
            <strong>{item.title}</strong>
            <Badge>{item.role === "prompt" ? "文字输入" : "参考素材"}</Badge>
          </div>
          {item.revisionId && (
            <small className="mono" title={item.revisionId}>
              固定 v{item.version} · {item.revisionId.slice(0, 14)}
            </small>
          )}
          {item.files?.map((f) => (
            <small key={f.relativePath}>{f.name}</small>
          ))}
          {item.text && (
            <details>
              <summary>查看文字内容</summary>
              <pre>{item.text}</pre>
            </details>
          )}
          {item.textTruncated && (
            <p className="panel-hint">
              文档仅显示前 30,000 字符，执行时可读取完整版本文件。
            </p>
          )}
        </div>
      ))}
    </section>
  );
}

function WorkflowInspector({
  snapshot: s,
  launchStepKey,
  run,
  instanceId,
  canvas,
  mutate,
  notify,
  onOpenCharacterWorkflow,
}: {
  snapshot: ProjectSnapshot;
  launchStepKey?: string;
  run?: WorkflowRun;
  instanceId?: string;
  canvas: Canvas;
  mutate: (n: string, a: unknown) => Promise<any>;
  notify: (s: string) => void;
  onOpenCharacterWorkflow?: (runId?: string) => void;
}) {
  const [chosen, setChosen] = useState<string[]>([]);
  const [comment, setComment] = useState("");
  const [busy, setBusy] = useState(false);
  const [expanded, setExpanded] = useState<string | null>(
    launchStepKey || null,
  );
  const [previewId, setPreviewId] = useState<string>();
  const [rerunStep, setRerunStep] = useState("");
  const [rerunInput, setRerunInput] = useState("");
  const [plan, setPlan] = useState<WorkflowPlan>();
  const [startKey, setStartKey] = useState("");
  const [actionError, setActionError] = useState("");
  const workflowNode = canvas.nodes.find(
    (n) => n.type === "workflow" && n.data.instanceId === instanceId,
  );
  const liveInputs = previewInputs(canvas, workflowNode?.id, s.assets);

  useEffect(() => {
    setPlan(undefined);
    setActionError("");
  }, [instanceId, canvas.id]);

  const launchStepRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    launchStepRef.current?.scrollIntoView({ block: "center" });
  }, [launchStepKey]);
  const step = run?.steps.find((st) => st.status.startsWith("awaiting"));

  useEffect(() => {
    setChosen([]);
    setComment("");
    setPreviewId(undefined);
  }, [run?.id, step?.key, run?.revision]);

  const revisions = s.assets
    .filter((a) => !a.archived)
    .flatMap((a) => a.revisions.map((r) => ({ a, r })));

  const input = s.assets
    .flatMap((a) => a.revisions.map((r) => ({ a, r })))
    .find(({ r }) => r.id === previewId);

  const candidates = revisions.filter(({ r }) =>
    step?.mode === "review"
      ? step.inputRevisionIds.includes(r.id)
      : step?.key === "brief"
        ? r.files.some((f) => f.mime === "text/plain")
        : step?.key === "reference"
          ? r.files.some((f) => f.mime.startsWith("image/"))
          : r.files.some((f) => f.mime === "model/gltf-binary"),
  );

  async function doAction(f: () => Promise<unknown>) {
    setBusy(true);
    setActionError("");
    try {
      await f();
    } catch (error) {
      setActionError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function start() {
    const prepared = await call<WorkflowPlan>("workflow_prepare", {
      projectId: s.project.id,
      canvasId: canvas.id,
      instanceId,
    });
    setPlan(prepared);
    setStartKey(uuid("op"));
  }
  async function confirmStart() {
    if (!plan) return;
    const started = await mutate("workflow_start", {
      projectId: s.project.id,
      planId: plan.id,
      expectedRevision: plan.graphRevision,
      idempotencyKey: startKey,
    });
    if (started) setPlan(undefined);
  }

  return (
    <div className="workflow-inspector">
      <div className="workflow-panel-heading">
        <Workflow size={18} />
        <div>
          <h3>角色制作</h3>
          <small>8 阶段流水线 · 23 节点 DAG</small>
        </div>
      </div>
      {onOpenCharacterWorkflow && (
        <button
          type="button"
          className="button small full"
          onClick={() => onOpenCharacterWorkflow(run?.id)}
        >
          <ArrowUpRight size={13} />
          打开角色工作流 (DAG) 独立页面
        </button>
      )}
      {actionError && (
        <p className="input-error" role="alert">
          {actionError}
        </p>
      )}
      {!run ? (
        <>
          <p className="panel-hint">
            连接素材或便签到分组左侧。开始前确认输入，制作将使用这些固定版本和文字。
          </p>
          {liveInputs.error ? (
            <p className="input-error" role="alert">
              {liveInputs.error}
            </p>
          ) : (
            liveInputs.snapshot && (
              <InputSummary
                snapshot={liveInputs.snapshot}
                title="本次制作输入"
              />
            )
          )}
          <button
            className="button primary full"
            disabled={
              busy || !!liveInputs.error || !liveInputs.snapshot?.items.length
            }
            onClick={() => doAction(start)}
          >
            <Play size={14} />
            开始制作
          </button>
        </>
      ) : (
        <>
          {run.inputSnapshot ? (
            <InputSummary
              snapshot={run.inputSnapshot}
              title="已冻结的制作输入"
            />
          ) : (
            <p className="panel-hint">
              此运行使用旧版固定输入，可通过 workflow_get 读取资产文件。
            </p>
          )}
          {run.inputSnapshot &&
            run.inputSnapshot.graphRevision !== canvas.graphRevision && (
              <p className="panel-hint">
                画布已调整；当前运行继续使用上方冻结的输入。
              </p>
            )}
          <div className="workflow-panel-progress">
            <strong>
              {
                run.steps.filter((x) => x.status === "succeeded" && !x.stale)
                  .length
              }{" "}
              / 8 阶段完成
            </strong>
            <Badge tone={run.status === "succeeded" ? "green" : "amber"}>
              {statusName[run.status]}
            </Badge>
          </div>
          <div className="step-timeline">
            {run.steps.map((st, i) => (
              <div
                key={st.key}
                ref={st.key === launchStepKey ? launchStepRef : undefined}
                data-workflow-step={st.key}
                className={
                  st.status === "succeeded" && !st.stale
                    ? "complete"
                    : st.key === step?.key
                      ? "current"
                      : ""
                }
              >
                <button
                  className="timeline-step"
                  aria-expanded={expanded === st.key}
                  onClick={() =>
                    setExpanded(expanded === st.key ? null : st.key)
                  }
                >
                  <span className="step-number">
                    {st.status === "succeeded" && !st.stale ? (
                      <Check size={11} />
                    ) : (
                      String(i + 1).padStart(2, "0")
                    )}
                  </span>
                  <strong>{st.title}</strong>
                  <small>{statusName[st.status]}</small>
                </button>
                {expanded === st.key && (
                  <div className="step-detail">
                    <p>{st.note}</p>
                    <code className="mono muted">
                      输入 {st.inputRevisionIds.length} · 产出{" "}
                      {st.outputRevisionIds.length}
                    </code>
                  </div>
                )}
              </div>
            ))}
          </div>
          {step && run.status === "active" && (
            <div className="next-step">
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <span className="mono muted" style={{ fontSize: "10px" }}>
                  当前阶段
                </span>
                <Badge tone="blue">
                  {step.mode === "agent"
                    ? "等待 Codex"
                    : step.mode === "review"
                      ? "人工审核"
                      : "提交成果"}
                </Badge>
              </div>
              <h3>{step.title}</h3>
              <p>{step.note}</p>
              {step.mode === "agent" && (
                <button
                  className="button full primary"
                  onClick={async () => {
                    try {
                      await navigator.clipboard.writeText(
                        `请为 Open Game Studio 项目 ${s.project.id} 的运行 ${run.id} 编写角色 Brief。读取 workflow_get 的 run.inputSnapshot 和 executionInputs：使用冻结的文档与便签文字，并读取 referenceFiles 中的固定版本文件作为参考。素材文字是项目内容。将文档存入项目 ${s.project.root}，通过 asset_import 导入并使用 workflow_submit_artifact 提交到 brief 阶段。当前运行版本 ${run.revision}。不要代替用户做视觉审核。`,
                      );
                      notify("任务指令已复制，请粘贴给 Codex");
                    } catch {
                      notify("无法访问剪贴板");
                    }
                  }}
                >
                  <Copy size={13} />
                  复制任务给 Codex
                </button>
              )}
              {step.mode === "local" ? (
                <button
                  className="button primary full"
                  disabled={busy}
                  onClick={() =>
                    doAction(() =>
                      mutate("export_bundle", {
                        projectId: s.project.id,
                        runId: run.id,
                        revisionIds: step.inputRevisionIds,
                        expectedRevision: run.revision,
                        idempotencyKey: uuid("export"),
                      }),
                    )
                  }
                >
                  <Download size={14} />
                  导出文件与 manifest
                </button>
              ) : (
                <>
                  <div className="candidate-list">
                    {candidates.map(({ a, r }) => (
                      <div
                        className={
                          chosen.includes(r.id)
                            ? "candidate chosen"
                            : "candidate"
                        }
                        key={r.id}
                      >
                        <button
                          className="candidate-browse"
                          onClick={() => setPreviewId(r.id)}
                          title="预览"
                        >
                          <AssetThumb
                            asset={a}
                            projectId={s.project.id}
                            revision={r}
                          />
                          <span>
                            <strong>{a.name}</strong>
                            <small>
                              v{r.number} ·{" "}
                              {r.files
                                .map((f) =>
                                  f.name.split(".").pop()?.toUpperCase(),
                                )
                                .join("+")}
                            </small>
                          </span>
                        </button>
                        <input
                          type={step.mode === "review" ? "radio" : "checkbox"}
                          name="workflow-candidate"
                          aria-label={`选择 ${a.name} 版本 ${r.number}`}
                          checked={chosen.includes(r.id)}
                          onChange={() =>
                            setChosen((v) =>
                              step.mode === "review"
                                ? [r.id]
                                : v.includes(r.id)
                                  ? v.filter((x) => x !== r.id)
                                  : [...v, r.id],
                            )
                          }
                        />
                      </div>
                    ))}
                  </div>
                  {!candidates.length && (
                    <p className="dock-empty" style={{ padding: "8px" }}>
                      请先在资源库导入本阶段产物文件。
                    </p>
                  )}
                  {step.mode === "review" ? (
                    <>
                      <label className="field" style={{ margin: "8px 0" }}>
                        {step.key === "deformation"
                          ? "测试证据"
                          : "审核意见 / 返修说明"}
                        <textarea
                          rows={2}
                          value={comment}
                          onChange={(e) => setComment(e.target.value)}
                          placeholder="填写审核批注…"
                        />
                      </label>
                      <button
                        className="button primary full"
                        disabled={
                          !chosen.length ||
                          busy ||
                          (step.key === "deformation" && !comment.trim())
                        }
                        onClick={() =>
                          doAction(() =>
                            mutate("workflow_decide", {
                              projectId: s.project.id,
                              runId: run.id,
                              stepKey: step.key,
                              revisionId: chosen[0],
                              decision: "accept",
                              comment,
                              expectedRevision: run.revision,
                            }),
                          )
                        }
                      >
                        <Check size={14} />
                        采用此版本
                      </button>
                      <button
                        className="button full"
                        disabled={!chosen.length || !comment.trim() || busy}
                        onClick={() =>
                          doAction(() =>
                            mutate("workflow_decide", {
                              projectId: s.project.id,
                              runId: run.id,
                              stepKey: step.key,
                              revisionId: chosen[0],
                              decision: "request_changes",
                              comment,
                              expectedRevision: run.revision,
                            }),
                          )
                        }
                      >
                        <RotateCcw size={13} />
                        针对当前版本返修
                      </button>
                    </>
                  ) : (
                    <button
                      className="button primary full"
                      disabled={!chosen.length || busy}
                      onClick={() =>
                        doAction(() =>
                          mutate("workflow_submit_artifact", {
                            projectId: s.project.id,
                            runId: run.id,
                            stepKey: step.key,
                            revisionIds: chosen,
                            expectedRevision: run.revision,
                          }),
                        )
                      }
                    >
                      <Upload size={14} />
                      提交本阶段成果
                    </button>
                  )}
                </>
              )}
            </div>
          )}
          {run.status === "succeeded" && (
            <div
              style={{
                padding: "14px",
                background: "var(--accent-green-bg)",
                border: "1px solid var(--accent-green-border)",
                borderRadius: "8px",
                textAlign: "center",
              }}
            >
              <CheckCircle2
                size={20}
                style={{ color: "var(--accent-green)", margin: "0 auto 6px" }}
              />
              <strong style={{ fontSize: "13px", display: "block" }}>
                制作成果已交接
              </strong>
              <p
                className="muted"
                style={{ fontSize: "11px", marginTop: "4px" }}
              >
                导出记录已生成
              </p>
            </div>
          )}
          <details className="rerun-panel">
            <summary>局部重跑与操作</summary>
            <label className="field" style={{ marginTop: "8px" }}>
              重跑起始阶段
              <select
                value={rerunStep}
                onChange={(e) => setRerunStep(e.target.value)}
              >
                <option value="">选择阶段</option>
                {run.steps
                  .filter((x) => ["manual", "agent"].includes(x.mode))
                  .map((st) => (
                    <option key={st.key} value={st.key}>
                      {st.title}
                    </option>
                  ))}
              </select>
            </label>
            <label className="field">
              替换输入（可选）
              <select
                value={rerunInput}
                onChange={(e) => setRerunInput(e.target.value)}
              >
                <option value="">保持原输入</option>
                {revisions.map(({ a, r }) => (
                  <option key={r.id} value={r.id}>
                    {a.name} · v{r.number}
                  </option>
                ))}
              </select>
            </label>
            <button
              className="button small full"
              disabled={!rerunStep || busy}
              onClick={() =>
                doAction(() =>
                  mutate("job_control", {
                    projectId: s.project.id,
                    runId: run.id,
                    action: "rerun",
                    fromStep: rerunStep,
                    expectedRevision: run.revision,
                    ...(rerunInput ? { inputRevisionIds: [rerunInput] } : {}),
                    idempotencyKey: uuid("rerun"),
                  }),
                )
              }
            >
              <RefreshCw size={13} />
              执行局部重跑
            </button>
            {run.status === "active" && (
              <button
                className="button small full danger"
                style={{ marginTop: "6px" }}
                disabled={busy}
                onClick={() =>
                  doAction(() =>
                    mutate("job_control", {
                      projectId: s.project.id,
                      runId: run.id,
                      action: "cancel",
                      expectedRevision: run.revision,
                    }),
                  )
                }
              >
                <Square size={12} />
                停止运行
              </button>
            )}
          </details>
          <details className="review-history">
            <summary>
              审核记录 ({s.reviews.filter((r) => r.runId === run.id).length})
            </summary>
            {s.reviews
              .filter((r) => r.runId === run.id)
              .map((r) => (
                <div key={r.id}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                    }}
                  >
                    <strong>
                      {r.scope === "appearance" ? "候选外形" : "变形审核"}
                    </strong>
                    <Badge tone={r.decision === "accept" ? "green" : "amber"}>
                      {r.decision === "accept" ? "采用" : "返修"}
                    </Badge>
                  </div>
                  <p className="muted">{r.comment || "无补充批注"}</p>
                </div>
              ))}
          </details>
        </>
      )}
      {input && (
        <Modal
          title={`预览 · ${input.a.name} v${input.r.number}`}
          onClose={() => setPreviewId(undefined)}
        >
          <div className="modal-body" style={{ padding: 0 }}>
            <AssetPreview
              projectId={s.project.id}
              asset={input.a}
              revision={input.r}
            />
            <div className="modal-footer">
              <button
                className="button small"
                onClick={() => setPreviewId(undefined)}
              >
                关闭预览
              </button>
            </div>
          </div>
        </Modal>
      )}
      {plan && !run && (
        <Modal title="确认制作输入" onClose={() => !busy && setPlan(undefined)}>
          <div className="modal-body input-plan">
            <p className="panel-hint">
              本次使用以下固定版本、文档和便签。确认后冻结为运行输入，由 Codex
              与人工按阶段制作。
            </p>
            <InputSummary snapshot={plan.inputSnapshot} title="本次制作输入" />
            {actionError && (
              <p className="input-error" role="alert">
                {actionError}
              </p>
            )}
            {plan.graphRevision !== canvas.graphRevision && (
              <p className="input-error" role="alert">
                画布已变化，请重新读取输入再确认。
              </p>
            )}
            <div className="modal-footer">
              <button
                className="button small"
                disabled={busy}
                onClick={() => setPlan(undefined)}
              >
                返回调整
              </button>
              <button
                className="button small"
                disabled={busy}
                onClick={() => doAction(start)}
              >
                重新读取输入
              </button>
              <button
                className="button small primary"
                disabled={busy || plan.graphRevision !== canvas.graphRevision}
                onClick={() => doAction(confirmStart)}
              >
                {busy ? (
                  <Loader2 size={14} className="spin" />
                ) : (
                  <Play size={14} />
                )}
                确认并开始
              </button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
}

function ProjectSettings({
  project: p,
  refresh,
  mutate,
  onBack,
  notify,
  canvases,
}: {
  project: Project;
  refresh: () => Promise<void>;
  mutate: (n: string, a: unknown) => Promise<any>;
  onBack: () => void;
  notify: (s: string) => void;
  canvases: Canvas[];
}) {
  const [name, setName] = useState(p.name);
  const [collection, setCollection] = useState(p.collection);
  const [engine, setEngine] = useState(p.engine);
  const [editingCover, setEditingCover] = useState(false);
  const saveSettings = () =>
    mutate("project_manage", {
      action: "update",
      projectId: p.id,
      name,
      collection,
      engine,
    })
      .then(() => notify("项目设置已保存"))
      .catch(() => {});

  return (
    <div className="content-page settings-page">
      <div className="page-title">
        <div>
          <h1>项目设置</h1>
          <p>管理项目基础信息、画布与归档状态。</p>
        </div>
      </div>
      <section className="settings-cover">
        <h3>项目封面</h3>
        <div className="cover-preview">
          <ProjectCoverImage
            projectId={p.id}
            projectName={p.name}
            cover={p.cover}
            alt={`${p.name}封面`}
          />
        </div>
        <button className="button small" onClick={() => setEditingCover(true)}>
          <ImageIcon size={14} />
          {p.cover ? "更换封面" : "设置封面"}
        </button>
      </section>
      {editingCover && (
        <ProjectCoverModal
          project={p}
          onClose={() => setEditingCover(false)}
          onSaved={async () => {
            await refresh();
            notify("项目封面已保存");
          }}
        />
      )}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          saveSettings();
        }}
      >
        <label className="field">
          项目名称
          <input
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <div className="form-row">
          <label className="field">
            项目集
            <input
              value={collection}
              onChange={(e) => setCollection(e.target.value)}
              placeholder="所属项目集分类"
            />
          </label>
          <label className="field">
            目标引擎
            <select
              value={engine}
              onChange={(e) => setEngine(e.target.value as Project["engine"])}
            >
              {Object.entries(engineName).map(([v, n]) => (
                <option value={v} key={v}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        </div>
        <label className="field">
          项目绝对路径
          <input readOnly value={p.root} className="mono" />
        </label>
        <button
          className="button small primary"
          style={{ alignSelf: "flex-start" }}
          type={isEmbeddedApp() ? "button" : "submit"}
          onClick={isEmbeddedApp() ? saveSettings : undefined}
        >
          保存设置
        </button>
      </form>
      <div className="settings-capabilities">
        <h3>能力与验证</h3>
        <div>
          <span>本地资产存储与 CAS</span>
          <Badge tone="green">正常</Badge>
        </div>
        <div>
          <span>GLB 3D 模型渲染</span>
          <Badge tone="green">已支持</Badge>
        </div>
        <div>
          <span>生成 Provider</span>
          <Badge>待接入</Badge>
        </div>
      </div>
      <div className="settings-capabilities">
        <h3>画布管理</h3>
        {canvases.map((c) => (
          <div key={c.id}>
            <span>
              {c.name} {c.archived && <Badge>已归档</Badge>}
            </span>
            <button
              className="text-button"
              onClick={() =>
                mutate("canvas_manage", {
                  projectId: p.id,
                  canvasId: c.id,
                  action: "update",
                  archived: !c.archived,
                  expectedRevision: c.graphRevision,
                }).catch(() => {})
              }
            >
              {c.archived ? "恢复" : "归档"}
            </button>
          </div>
        ))}
      </div>
      <div className="archive-section">
        <h3>归档此项目</h3>
        <p>归档后项目不会在主列表展示，所有资产与历史记录将完整保留。</p>
        <button
          className="button small"
          style={{ alignSelf: "flex-start" }}
          onClick={() =>
            mutate("project_manage", {
              action: "update",
              projectId: p.id,
              archived: !p.archived,
            })
              .then(onBack)
              .catch(() => {})
          }
        >
          <Archive size={14} />
          {p.archived ? "恢复项目" : "归档项目"}
        </button>
      </div>
    </div>
  );
}
