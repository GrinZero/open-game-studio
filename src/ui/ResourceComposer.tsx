import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  FileText,
  FolderOpen,
  StickyNote,
  Image,
  Film,
  Box,
  PersonStanding,
  Upload,
  Music2,
  Code2,
  X,
  Loader2,
  Play,
  Check,
  AlertTriangle,
} from "lucide-react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type {
  Asset,
  AssetRevision,
  Canvas,
  GenerationKind,
  GenerationTask,
  ProjectSnapshot,
  StudioNode,
} from "../shared/types";
import { call, useAssetFileUrl } from "./api";

export type AddResourceKind =
  | "document"
  | "existing-document"
  | "text"
  | GenerationKind
  | "upload"
  | "path"
  | "library"
  | "source";
export const resourceChoices: {
  kind: AddResourceKind;
  label: string;
  detail: string;
  icon: typeof FileText;
  section: string;
}[] = [
  {
    kind: "document",
    label: "文档",
    detail: "编写 Markdown 制作说明",
    icon: FileText,
    section: "文档",
  },
  {
    kind: "existing-document",
    label: "已有文档",
    detail: "引用项目内已有文档",
    icon: FolderOpen,
    section: "文档",
  },
  {
    kind: "text",
    label: "文本",
    detail: "可复用的提示词与便签",
    icon: StickyNote,
    section: "资源",
  },
  {
    kind: "image",
    label: "图片",
    detail: "根据描述与参考生成图片",
    icon: Image,
    section: "资源",
  },
  {
    kind: "video",
    label: "视频",
    detail: "根据描述与参考生成视频",
    icon: Film,
    section: "资源",
  },
  {
    kind: "model",
    label: "3D 模型",
    detail: "从文字或图片制作模型",
    icon: Box,
    section: "资源",
  },
  {
    kind: "animation",
    label: "3D 动画",
    detail: "给角色绑定骨架并添加动作",
    icon: PersonStanding,
    section: "资源",
  },
  {
    kind: "audio",
    label: "音频",
    detail: "制作音乐、音效或语音",
    icon: Music2,
    section: "资源",
  },
  {
    kind: "source",
    label: "源文件",
    detail: "导入代码或可编辑制作文件",
    icon: Code2,
    section: "资源",
  },
  {
    kind: "path",
    label: "引用本地路径",
    detail: "直接选择项目或本地文件路径，项目内零复制",
    icon: FolderOpen,
    section: "导入与引用",
  },
  {
    kind: "upload",
    label: "上传",
    detail: "从本机导入素材",
    icon: Upload,
    section: "导入与引用",
  },
  {
    kind: "library",
    label: "从资源库添加",
    detail: "复用项目资源的固定版本",
    icon: FolderOpen,
    section: "导入与引用",
  },
];
export const generationLabels: Record<GenerationKind, string> = {
  image: "图片",
  video: "视频",
  model: "3D 模型",
  animation: "3D 动画",
  audio: "音频",
};
export const taskLabels: Record<GenerationTask["status"], string> = {
  draft: "待制作",
  running: "制作中",
  succeeded: "已完成",
  failed: "制作失败",
  interrupted: "已中断",
  cancelled: "已停止",
};
export function ResourceMenu({
  position,
  onChoose,
  onClose,
}: {
  position: { x: number; y: number };
  onChoose: (kind: AddResourceKind) => void;
  onClose: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.querySelector<HTMLButtonElement>("button")?.focus();
    const outside = (e: PointerEvent) => {
      if (!ref.current?.contains(e.target as Node)) onClose();
    };
    document.addEventListener("pointerdown", outside, true);
    return () => document.removeEventListener("pointerdown", outside, true);
  }, []);
  return createPortal(
    <div
      ref={ref}
      className="resource-add-menu"
      role="menu"
      aria-label="添加资源"
      style={{
        left: Math.max(8, Math.min(position.x, window.innerWidth - 308)),
        top: Math.max(8, Math.min(position.y, window.innerHeight - 610)),
      }}
      onKeyDown={(e) => {
        if (e.key === "Escape" || e.key === "Tab") {
          onClose();
          return;
        }
        if (["ArrowUp", "ArrowDown", "Home", "End"].includes(e.key)) {
          e.preventDefault();
          const buttons = Array.from(
            ref.current!.querySelectorAll<HTMLButtonElement>("button"),
          );
          const i = buttons.indexOf(
            document.activeElement as HTMLButtonElement,
          );
          buttons[
            e.key === "Home"
              ? 0
              : e.key === "End"
                ? buttons.length - 1
                : (i + (e.key === "ArrowDown" ? 1 : -1) + buttons.length) %
                  buttons.length
          ]?.focus();
        }
      }}
    >
      {resourceChoices.map((choice, i) => (
        <div key={choice.kind}>
          {(i === 0 || resourceChoices[i - 1].section !== choice.section) && (
            <div className="resource-menu-section">{choice.section}</div>
          )}
          <button
            role="menuitem"
            onClick={() => {
              onClose();
              onChoose(choice.kind);
            }}
          >
            <choice.icon size={19} />
            <span>
              <strong>{choice.label}</strong>
              <small>{choice.detail}</small>
            </span>
          </button>
        </div>
      ))}
    </div>,
    document.body,
  );
}
export function ResourceDialog({
  title,
  onClose,
  children,
  wide = false,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    ref.current?.showModal();
    return () => ref.current?.close();
  }, []);
  return (
    <dialog
      ref={ref}
      className={
        wide ? "resource-dialog resource-dialog-wide" : "resource-dialog"
      }
      aria-label={title}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
    >
      <div className="modal-head">
        <h2>{title}</h2>
        <button className="icon-button" aria-label="关闭" onClick={onClose}>
          <X size={18} />
        </button>
      </div>
      {children}
    </dialog>
  );
}
export function Markdown({ text }: { text: string }) {
  return (
    <div className="studio-markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{text}</ReactMarkdown>
    </div>
  );
}
export function DocumentEditor({
  snapshot: s,
  canvas,
  node,
  at,
  onClose,
  onSaved,
}: {
  snapshot: ProjectSnapshot;
  canvas: Canvas;
  node?: StudioNode;
  at: { x: number; y: number };
  onClose: () => void;
  onSaved: () => Promise<void>;
}) {
  const asset = s.assets.find((a) => a.id === node?.data.assetId);
  const r = asset?.revisions.find((r) => r.id === node?.data.revisionId);
  const { url, error: fileError } = useAssetFileUrl(s.project.id, r?.id);
  const [name, setName] = useState(node?.data.title || "制作说明"),
    [text, setText] = useState(node ? "" : "# 制作说明\n\n"),
    [mode, setMode] = useState("edit");
  const [busy, setBusy] = useState(false),
    [loading, setLoading] = useState(!!node),
    [error, setError] = useState("");
  const key = useRef(crypto.randomUUID());
  useEffect(() => {
    if (!url) return;
    const controller = new AbortController();
    fetch(url, { signal: controller.signal })
      .then((r) => {
        if (!r.ok) throw new Error("文档读取失败");
        return r.text();
      })
      .then((t) => {
        setText(t);
        setLoading(false);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(String(e));
      });
    return () => controller.abort();
  }, [url]);
  const save = async () => {
    setBusy(true);
    setError("");
    try {
      await call("document_save", {
        projectId: s.project.id,
        canvasId: canvas.id,
        expectedRevision: canvas.graphRevision,
        nodeId: node?.id || `n_${key.current}`,
        assetId: asset?.id,
        expectedLatestRevisionId: r?.id,
        name,
        text,
        position: at,
      });
      await onSaved();
      onClose();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <ResourceDialog
      title={node ? "编辑文档" : "新建文档"}
      onClose={onClose}
      wide
    >
      <div className="modal-body">
        <label className="field">
          文档名称
          <input
            aria-label="文档名称"
            maxLength={120}
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <div className="document-editor-toolbar">
          <span>Markdown{r ? ` · v${r.number}` : ""}</span>
          <div>
            <button
              className={`button small ${mode === "edit" ? "primary" : ""}`}
              onClick={() => setMode("edit")}
            >
              编辑
            </button>
            <button
              className={`button small ${mode === "preview" ? "primary" : ""}`}
              onClick={() => setMode("preview")}
            >
              预览
            </button>
          </div>
        </div>
        {mode === "edit" ? (
          <textarea
            className="document-editor-input"
            aria-label="文档内容"
            value={text}
            disabled={loading}
            onChange={(e) => setText(e.target.value)}
            maxLength={300000}
          />
        ) : (
          <div className="document-editor-preview">
            <Markdown text={text} />
          </div>
        )}
        {node && (
          <p className="panel-hint">
            保存为新版本，当前节点采用新版本；其他画布仍引用原版本。
          </p>
        )}
        {(error || fileError) && (
          <p role="alert" className="resource-error">
            {error || fileError}
          </p>
        )}
      </div>
      <div className="modal-footer">
        <button className="button small" onClick={onClose}>
          取消
        </button>
        <button
          className="button small primary"
          disabled={busy || loading || !name.trim()}
          onClick={save}
        >
          {busy && <Loader2 size={14} className="spin" />}保存文档
        </button>
      </div>
    </ResourceDialog>
  );
}
export function ResourceLibrary({
  snapshot,
  documentsOnly,
  onChoose,
  onClose,
}: {
  snapshot: ProjectSnapshot;
  documentsOnly: boolean;
  onChoose: (asset: Asset, revision: AssetRevision) => Promise<void>;
  onClose: () => void;
}) {
  const [query, setQuery] = useState(""),
    [kind, setKind] = useState("all"),
    [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const assets = snapshot.assets.filter(
    (a) =>
      !a.archived &&
      (!documentsOnly || a.kind === "document") &&
      (kind === "all" || a.kind === kind) &&
      `${a.name} ${a.tags.join(" ")}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  return (
    <ResourceDialog
      title={documentsOnly ? "引用已有文档" : "从资源库添加"}
      onClose={onClose}
      wide
    >
      <div className="modal-body">
        <div className="resource-library-filters">
          <input
            aria-label="搜索资源库"
            autoFocus
            placeholder="搜索名称或标签…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          {!documentsOnly && (
            <select
              aria-label="资源类型"
              value={kind}
              onChange={(e) => setKind(e.target.value)}
            >
              <option value="all">全部类型</option>
              {[
                ["document", "文档"],
                ["image", "图片"],
                ["video", "视频"],
                ["model", "3D 模型"],
                ["audio", "音频"],
                ["source", "源文件"],
              ].map(([v, label]) => (
                <option value={v} key={v}>
                  {label}
                </option>
              ))}
            </select>
          )}
        </div>
        <div className="resource-library-list">
          {assets.map((a) => (
            <LibraryRow
              key={a.id}
              asset={a}
              busy={busy}
              onChoose={async (r) => {
                setBusy(true);
                setError("");
                try {
                  await onChoose(a, r);
                  onClose();
                } catch (e) {
                  setError(String(e));
                } finally {
                  setBusy(false);
                }
              }}
            />
          ))}
          {!assets.length && (
            <p className="dock-empty">
              {query
                ? "没有匹配的资源。"
                : documentsOnly
                  ? "项目内还没有文档，先新建文档或上传 Markdown 文件。"
                  : "资源库为空，先上传素材或创建资源。"}
            </p>
          )}
        </div>
        {error && (
          <p role="alert" className="resource-error">
            {error}
          </p>
        )}
        <p className="panel-hint">
          添加的是固定版本引用，同一资源可在多张画布复用。
        </p>
      </div>
    </ResourceDialog>
  );
}
function LibraryRow({
  asset,
  busy,
  onChoose,
}: {
  asset: Asset;
  busy: boolean;
  onChoose: (r: AssetRevision) => void;
}) {
  const [revisionId, setRevisionId] = useState(asset.latestRevisionId);
  return (
    <div className="resource-library-row">
      <FileText size={20} />
      <span>
        <strong>{asset.name}</strong>
        <small>
          {asset.kind} · {asset.revisions.length} 个版本
        </small>
      </span>
      <select
        aria-label={`${asset.name}版本`}
        value={revisionId}
        onChange={(e) => setRevisionId(e.target.value)}
      >
        {asset.revisions.map((r) => (
          <option key={r.id} value={r.id}>
            v{r.number}
          </option>
        ))}
      </select>
      <button
        disabled={busy}
        className="button small"
        onClick={() =>
          onChoose(asset.revisions.find((r) => r.id === revisionId)!)
        }
      >
        添加引用
      </button>
    </div>
  );
}
export function GenerationComposer({
  snapshot: s,
  canvas,
  kind,
  inputNodeIds,
  at,
  onClose,
  onSaved,
}: {
  snapshot: ProjectSnapshot;
  canvas: Canvas;
  kind: GenerationKind;
  inputNodeIds: string[];
  at: { x: number; y: number };
  onClose: () => void;
  onSaved: () => Promise<void>;
}) {
  const [name, setName] = useState(`新${generationLabels[kind]}`),
    [prompt, setPrompt] = useState(""),
    [refs, setRefs] = useState(inputNodeIds);
  const [ratio, setRatio] = useState("1:1"),
    [duration, setDuration] = useState(5),
    [animation, setAnimation] = useState("idle, walk, run"),
    [transparent, setTransparent] = useState(false);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [available, setAvailable] = useState<boolean>();
  const idempotency = useRef(crypto.randomUUID()),
    createdTask = useRef<GenerationTask | undefined>(undefined);
  useEffect(() => {
    call("generation_capabilities", {})
      .then((c) => setAvailable(c.available))
      .catch((e) => {
        setAvailable(false);
        setError(String(e));
      });
  }, []);
  const eligible = canvas.nodes.filter(
    (n) => n.type === "asset" || n.type === "note",
  );
  const hasModel = refs.some(
    (id) =>
      s.assets.find(
        (a) => a.id === canvas.nodes.find((n) => n.id === id)?.data.assetId,
      )?.kind === "model",
  );
  const create = async (start: boolean) => {
    setBusy(true);
    setError("");
    try {
      const t =
        createdTask.current ||
        (await call<GenerationTask>("generation_create", {
          projectId: s.project.id,
          canvasId: canvas.id,
          expectedRevision: canvas.graphRevision,
          name,
          kind,
          prompt,
          inputNodeIds: refs,
          position: at,
          idempotencyKey: idempotency.current,
          settings: {
            ...(kind === "image" || kind === "video"
              ? { aspectRatio: ratio }
              : {}),
            ...(kind === "video" || kind === "audio" ? { duration } : {}),
            ...(kind === "animation" ? { animation } : {}),
            ...(kind === "image" ? { transparent } : {}),
          },
        }));
      createdTask.current = t;
      if (start)
        await call("generation_start", {
          projectId: s.project.id,
          taskId: t.id,
          expectedRevision: t.revision,
        });
      await onSaved();
      onClose();
    } catch (e) {
      setError(String(e));
      await onSaved();
    } finally {
      setBusy(false);
    }
  };
  return (
    <ResourceDialog title={`制作${generationLabels[kind]}`} onClose={onClose}>
      <div className="modal-body">
        <label className="field">
          资源名称
          <input
            aria-label="资源名称"
            autoFocus
            value={name}
            maxLength={120}
            disabled={!!createdTask.current}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <label className="field">
          制作描述
          <textarea
            aria-label="制作描述"
            rows={5}
            placeholder={
              kind === "animation"
                ? "描述骨架类型、动作、是否原地播放…"
                : "描述用途、风格、内容与质量要求…"
            }
            value={prompt}
            maxLength={30000}
            disabled={!!createdTask.current}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </label>
        <div className="resource-settings">
          {(kind === "image" || kind === "video") && (
            <label className="field">
              画幅
              <select
                aria-label="画幅"
                value={ratio}
                onChange={(e) => setRatio(e.target.value)}
              >
                {["1:1", "16:9", "9:16", "4:3"].map((r) => (
                  <option key={r}>{r}</option>
                ))}
              </select>
            </label>
          )}
          {(kind === "video" || kind === "audio") && (
            <label className="field">
              时长（秒）
              <input
                aria-label="时长（秒）"
                type="number"
                min={1}
                max={60}
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value))}
              />
            </label>
          )}
          {kind === "animation" && (
            <label className="field">
              动作
              <input
                aria-label="动作"
                value={animation}
                onChange={(e) => setAnimation(e.target.value)}
              />
            </label>
          )}
          {kind === "image" && (
            <label className="checkbox-field">
              <input
                type="checkbox"
                checked={transparent}
                onChange={(e) => setTransparent(e.target.checked)}
              />
              透明背景
            </label>
          )}
        </div>
        <div className="resource-references">
          <strong>参考素材与文字</strong>
          {eligible.map((n) => (
            <label key={n.id}>
              <input
                type="checkbox"
                checked={refs.includes(n.id)}
                disabled={!!createdTask.current}
                onChange={(e) =>
                  setRefs((r) =>
                    e.target.checked
                      ? [...r, n.id]
                      : r.filter((id) => id !== n.id),
                  )
                }
              />
              <span>{n.data.title}</span>
              <small>
                {n.type === "note"
                  ? "文本"
                  : `固定版本 ${s.assets.find((a) => a.id === n.data.assetId)?.revisions.find((r) => r.id === n.data.revisionId)?.number || "?"}`}
              </small>
            </label>
          ))}
          {!eligible.length && (
            <p className="panel-hint">
              可先添加参考图、文档或文本，也可以直接根据描述制作。
            </p>
          )}
        </div>
        {kind === "animation" && !hasModel && (
          <p role="alert" className="resource-error">
            选择一个 3D 模型作为动画输入。
          </p>
        )}
        <p className="panel-hint">
          通过本机 Codex
          和已配置工具制作。提示词、设置与参考版本会保存；模型服务的登录、费用与可用能力由对应工具检查。
        </p>
        {available === false && (
          <p role="alert" className="resource-error">
            本机 Codex 暂不可用，可以保存任务后继续。
          </p>
        )}
        {error && (
          <p role="alert" className="resource-error">
            {error}
          </p>
        )}
      </div>
      <div className="modal-footer">
        <button
          className="button small"
          disabled={
            busy ||
            !name.trim() ||
            !prompt.trim() ||
            (kind === "animation" && !hasModel)
          }
          onClick={() => create(false)}
        >
          保存任务
        </button>
        <button
          className="button small primary"
          disabled={
            busy ||
            available !== true ||
            !name.trim() ||
            !prompt.trim() ||
            (kind === "animation" && !hasModel)
          }
          onClick={() => create(true)}
        >
          {busy ? <Loader2 size={14} className="spin" /> : <Play size={14} />}
          开始制作
        </button>
      </div>
    </ResourceDialog>
  );
}
export function GenerationNode({ data, selected }: NodeProps) {
  const d = data as unknown as { title: string; task?: GenerationTask };
  const t = d.task;
  return (
    <div
      className={`studio-node generation-node ${selected ? "is-selected" : ""}`}
    >
      <Handle type="target" position={Position.Left} />
      <div className="node-heading">
        <span>{t ? generationLabels[t.kind] : "资源制作"}</span>
        {t?.status === "running" ? (
          <Loader2 size={15} className="spin" />
        ) : t?.status === "failed" ? (
          <AlertTriangle size={15} />
        ) : (
          <Box size={15} />
        )}
      </div>
      <strong>{d.title}</strong>
      <p>{t?.prompt}</p>
      <div className="generation-node-status">
        {t ? taskLabels[t.status] : "读取任务…"}
      </div>
      {t?.error && <small>{t.error}</small>}
    </div>
  );
}
export function GenerationDetails({
  task,
  projectId,
  assets,
  onClose,
  onRefresh,
}: {
  task: GenerationTask;
  projectId: string;
  assets: Asset[];
  onClose: () => void;
  onRefresh: () => Promise<void>;
}) {
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  const action = async (name: string, extra: object = {}) => {
    setBusy(true);
    setError("");
    try {
      await call(name, {
        projectId,
        taskId: task.id,
        expectedRevision: task.revision,
        ...extra,
      });
      await onRefresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };
  const [adoptId, setAdoptId] = useState("");
  const compatible = assets.filter(
    (a) =>
      !a.archived &&
      a.kind === (task.kind === "animation" ? "model" : task.kind),
  );
  return (
    <ResourceDialog title={task.name} onClose={onClose}>
      <div className="modal-body">
        <div className="task-status-line">
          <span>{generationLabels[task.kind]}</span>
          <strong>{taskLabels[task.status]}</strong>
        </div>
        <Markdown text={task.prompt} />
        <p className="panel-hint">{task.summary}</p>
        <h3>固定输入</h3>
        {task.inputSnapshot.items.map((i, n) => (
          <p key={n}>
            {i.title}
            {i.version ? ` · v${i.version}` : " · 文本"}
          </p>
        ))}
        {!task.inputSnapshot.items.length && (
          <p className="panel-hint">根据文字描述制作。</p>
        )}
        <pre className="task-settings">
          {JSON.stringify(task.settings, null, 2)}
        </pre>
        {(task.error || error) && (
          <p role="alert" className="resource-error">
            {error || task.error}
          </p>
        )}
        {(task.status === "failed" || task.status === "interrupted") && (
          <p className="panel-hint">
            查看项目 .game-studio/generated/{task.id}{" "}
            内的任务记录。已提交任务应按原 ID
            查询，可将完成文件上传到资源库，不会自动重提。
          </p>
        )}
        {["draft", "failed", "interrupted"].includes(task.status) && (
          <div className="resource-adopt">
            <label className="field">
              采用已导入产物
              <select
                aria-label="采用已导入产物"
                value={adoptId}
                onChange={(e) => setAdoptId(e.target.value)}
              >
                <option value="">选择固定资源版本…</option>
                {compatible.flatMap((a) =>
                  a.revisions.map((r) => (
                    <option key={r.id} value={r.id}>
                      {a.name} · v{r.number}
                    </option>
                  )),
                )}
              </select>
            </label>
            <button
              className="button small"
              disabled={busy || !adoptId}
              onClick={() =>
                action("generation_adopt", { revisionId: adoptId })
              }
            >
              <Check size={14} />
              回填任务结果
            </button>
          </div>
        )}
      </div>
      <div className="modal-footer">
        {task.status === "draft" && (
          <button
            className="button small primary"
            disabled={busy}
            onClick={() => action("generation_start")}
          >
            <Play size={14} />
            开始制作
          </button>
        )}
        {["draft", "running"].includes(task.status) && (
          <button
            className="button small"
            disabled={busy}
            onClick={() => action("generation_control", { action: "cancel" })}
          >
            停止本地制作
          </button>
        )}
        <button className="button small" onClick={onClose}>
          关闭
        </button>
      </div>
    </ResourceDialog>
  );
}
export function CanvasDocumentPreview({
  projectId,
  revision,
}: {
  projectId: string;
  revision: AssetRevision;
}) {
  const { url, error } = useAssetFileUrl(projectId, revision.id);
  const [text, setText] = useState(""),
    [failed, setFailed] = useState(false);
  useEffect(() => {
    setText("");
    setFailed(false);
    if (!url) return;
    const ac = new AbortController();
    fetch(url, { signal: ac.signal })
      .then((r) => {
        if (!r.ok) throw new Error();
        return r.text();
      })
      .then((t) => setText(t.slice(0, 4000)))
      .catch(() => {
        if (!ac.signal.aborted) setFailed(true);
      });
    return () => ac.abort();
  }, [url]);
  return (
    <div className="canvas-document-preview nodrag nowheel">
      {error || failed ? (
        "无法读取文档"
      ) : text ? (
        <Markdown text={text} />
      ) : (
        "正在读取文档…"
      )}
    </div>
  );
}
export function CanvasMediaPreview({
  projectId,
  asset,
  revision,
  index,
}: {
  projectId: string;
  asset: Asset;
  revision: AssetRevision;
  index: number;
}) {
  const { url, error } = useAssetFileUrl(projectId, revision.id, index);
  const [failedUrl, setFailedUrl] = useState<string>();
  return (
    <div
      className="canvas-media-preview nodrag nowheel nopan"
      onPointerDown={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
      onMouseDown={(e) => e.stopPropagation()}
    >
      {error || (url && failedUrl === url) ? (
        <span role="alert">无法播放媒体，请检查文件格式或编码</span>
      ) : !url ? (
        <Loader2 className="spin" size={20} />
      ) : revision.files[index]?.mime.startsWith("video/") ? (
        <video
          key={url}
          controls
          playsInline
          preload="metadata"
          src={url}
          aria-label={asset.name}
          onError={() => setFailedUrl(url)}
        />
      ) : (
        <>
          <Music2 size={32} />
          <audio
            key={url}
            onError={() => setFailedUrl(url)}
            controls
            preload="metadata"
            src={url}
            aria-label={asset.name}
          />
        </>
      )}
    </div>
  );
}
