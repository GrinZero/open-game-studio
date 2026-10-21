import { useState } from "react";
import { createPortal } from "react-dom";
import { Check, Image as ImageIcon, Link2, Search } from "lucide-react";
import type { Asset, ProjectSnapshot, WorkflowRun } from "../shared/types";
import type { MultiviewSlotKey } from "../shared/character-dag";
import { ResourceDialog } from "./ResourceComposer";
import {
  supportsCodexMessages,
  sendCodexMessage,
  request,
  useAssetFileUrl,
} from "./api";
import { formatCodexContext, type CodexContext } from "./codex-context";
import { characterCodexContext } from "./character-workflow-guidance";
import {
  characterProductionPrompt,
  characterSessionInput,
} from "../shared/codex-collaboration";

function ImagePreview({
  projectId,
  revisionId,
  name,
}: {
  projectId: string;
  revisionId?: string;
  name: string;
}) {
  const { url, error } = useAssetFileUrl(projectId, revisionId, 0);
  return (
    <div className="character-library-image">
      {url && !error ? (
        <img src={url} alt={name} loading="lazy" />
      ) : (
        <ImageIcon size={22} aria-label={error || "图片加载中"} />
      )}
    </div>
  );
}

export function CharacterCodexActions({
  context,
  notify,
  actionContainer,
}: {
  context: CodexContext;
  actionContainer?: HTMLElement | null;
  notify: (message: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState("");
  const direct = supportsCodexMessages();
  const text = formatCodexContext(context);
  async function start(mode: "current" | "new") {
    setBusy(true);
    setError("");
    setResult("");
    try {
      if (mode === "current") {
        await sendCodexMessage(characterProductionPrompt(context));
        const message =
          "已发送到当前 Codex 对话。可以在左侧继续调整，右侧会自动同步进度和结果。";
        setResult(message);
        notify(message);
      } else {
        await request(
          `/api/projects/${context.project.id}/character-workflow/open-codex-session`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(characterSessionInput(context)),
          },
        );
        const message =
          "已请求打开 Codex 新对话，并带入角色任务与右侧工作台。发送预填需求后即可开始，结果会自动同步。";
        setResult(message);
        notify(message);
      }
    } catch (error) {
      setError(
        error instanceof Error ? error.message : "无法连接 Codex，请重试。",
      );
    } finally {
      setBusy(false);
    }
  }
  const buttons = (
    <div className="character-dialog-buttons">
      {direct && (
        <button
          className="button primary"
          disabled={busy}
          onClick={() => start("current")}
        >
          <Link2 size={15} />
          发送到当前对话并制作
        </button>
      )}
      <button
        className={`button ${direct ? "" : "primary"}`}
        disabled={busy}
        onClick={() => start("new")}
      >
        <Link2 size={15} />
        打开 Codex 会话
      </button>
    </div>
  );
  return (
    <div className="character-codex-actions">
      <p className="panel-hint">
        {direct
          ? "直接把要求发送到左侧 Codex 对话开始制作，右侧自动同步。也可以为这个位置打开独立会话。"
          : "点击“打开 Codex 会话”，自动带入当前角色、目标位置和参考图片，并在右侧打开工作台。发送需求后，即可边对话边查看更新。"}
      </p>
      {actionContainer ? createPortal(buttons, actionContainer) : buttons}
      {result && (
        <p role="status" className="character-dialog-success">
          <Check size={15} />
          {result}
        </p>
      )}
      {error && (
        <p role="alert" className="resource-error">
          {error}
        </p>
      )}
      <details className="character-context-preview">
        <summary>查看会话目标与参考</summary>
        <textarea aria-label="角色制作会话详情" readOnly value={text} />
      </details>
    </div>
  );
}

export function CharacterTaskDialog({
  snapshot,
  run,
  nodeKey,
  slotKey,
  onClose,
  notify,
  initialPrompt = "",
}: {
  snapshot: ProjectSnapshot;
  run: WorkflowRun;
  nodeKey: string;
  slotKey?: MultiviewSlotKey;
  initialPrompt?: string;
  onClose: () => void;
  notify: (message: string) => void;
}) {
  const [prompt, setPrompt] = useState(initialPrompt);
  const [actionContainer, setActionContainer] = useState<HTMLDivElement | null>(
    null,
  );
  const context = characterCodexContext(snapshot, run, nodeKey, {
    slotKey,
    prompt,
  });
  return (
    <ResourceDialog
      title={`与 Codex 制作 · ${context.characterTarget!.title}`}
      onClose={onClose}
      wide
    >
      <div className="modal-body character-task-dialog">
        <p>
          生成结果将回到「{run.dagState?.characterName} /{" "}
          {context.characterTarget!.title}」。
        </p>
        <p className="panel-hint">
          {run.dag?.find((item) => item.key === nodeKey)?.description}
        </p>
        <div className="character-input-summary">
          <strong>使用的输入</strong>
          {context.inputSnapshot.items.length ? (
            context.inputSnapshot.items.map((item) => (
              <span key={item.revisionId}>
                {item.title} · v{item.version}
              </span>
            ))
          ) : (
            <span>暂无参考素材，可在需求里描述角色。</span>
          )}
        </div>
        <label className="field">
          补充制作要求
          <textarea
            aria-label="补充制作要求"
            rows={3}
            value={prompt}
            placeholder="描述要保留的特征、需要调整的细节…"
            onChange={(event) => setPrompt(event.target.value)}
          />
        </label>
        <CharacterCodexActions
          context={context}
          notify={notify}
          actionContainer={actionContainer}
        />
        <p className="panel-hint">
          制作进度与真实产物会自动更新到工作台；一组产物全部齐备后，再进入审核。
        </p>
      </div>
      <div className="modal-footer">
        <button className="button" onClick={onClose}>
          返回工作台
        </button>
        <div ref={setActionContainer} />
      </div>
    </ResourceDialog>
  );
}

export function CharacterSourceDialog({
  snapshot,
  run,
  selectedRevisionId,
  onSelect,
  onUpload,
  onAdopt,
  onClose,
  busy,
  error,
  notify,
}: {
  snapshot: ProjectSnapshot;
  run: WorkflowRun;
  selectedRevisionId: string;
  onSelect: (id: string) => void;
  onUpload: (file: File) => Promise<void>;
  onAdopt: () => Promise<void>;
  onClose: () => void;
  busy: boolean;
  error: string;
  notify: (message: string) => void;
}) {
  const [tab, setTab] = useState<"existing" | "codex">("existing");
  const [actionContainer, setActionContainer] = useState<HTMLDivElement | null>(
    null,
  );
  const [query, setQuery] = useState("");
  const [limit, setLimit] = useState(24);
  const [references, setReferences] = useState<string[]>([]);
  const [prompt, setPrompt] = useState(
    "生成一张全身角色原画，保持角色特征清楚，适合后续制作正交多视图和 3D 模型。",
  );
  const assets = snapshot.assets.filter(
    (asset) => !asset.archived && asset.kind === "image",
  );
  const filtered = assets.filter((asset) =>
    `${asset.name} ${asset.tags.join(" ")}`
      .toLowerCase()
      .includes(query.toLowerCase()),
  );
  const selectedAsset = assets.find((asset) =>
    asset.revisions.some((revision) => revision.id === selectedRevisionId),
  );
  const generationContext = characterCodexContext(
    snapshot,
    run,
    "source_illustration",
    { slotKey: "original", referenceRevisionIds: references, prompt },
  );
  function choose(asset: Asset) {
    if (tab === "existing") onSelect(asset.latestRevisionId);
    else
      setReferences((current) =>
        current.includes(asset.latestRevisionId)
          ? current.filter((id) => id !== asset.latestRevisionId)
          : [...current, asset.latestRevisionId],
      );
  }
  return (
    <ResourceDialog
      title="更换原画"
      onClose={() => {
        if (!busy) onClose();
      }}
      wide
    >
      <div className="modal-body character-source-dialog">
        <p>
          为「{run.dagState?.characterName}」选择原画，或在 Codex 会话中生成并自动更新到这里。
        </p>
        <div
          className="character-source-tabs"
          role="tablist"
          aria-label="原画来源"
        >
          <button
            role="tab"
            aria-selected={tab === "existing"}
            onClick={() => setTab("existing")}
          >
            选择已有原画
          </button>
          <button
            role="tab"
            aria-selected={tab === "codex"}
            onClick={() => setTab("codex")}
          >
            与 Codex 生成
          </button>
        </div>
        {tab === "codex" && (
          <div className="character-generate-brief">
            <label className="field">
              你想生成什么
              <textarea
                aria-label="原画生成要求"
                rows={3}
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
              />
            </label>
            <strong>选择参考图片（可选、多选）</strong>
            <p className="panel-hint">
              {references.length
                ? `已选择 ${references.length} 张固定版本参考。取消选择即可移除。`
                : "不选参考则按文字生成；也可以从下方选择已有图片，基于它们生成新的原画。"}
            </p>
            {!!references.length && (
              <button
                className="button small"
                onClick={() => setReferences([])}
              >
                清空参考
              </button>
            )}
          </div>
        )}
        <label className="character-library-search">
          <Search size={16} />
          <input
            aria-label="搜索已有原画"
            placeholder="搜索图片名称或标签…"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setLimit(24);
            }}
          />
        </label>
        <div
          className="character-image-library"
          aria-label={tab === "existing" ? "已有原画" : "生成参考图片"}
        >
          {filtered.slice(0, limit).map((asset) => {
            const selected =
              tab === "existing"
                ? selectedAsset?.id === asset.id
                : references.includes(asset.latestRevisionId);
            return (
              <button
                key={asset.id}
                type="button"
                className={`character-image-option ${selected ? "selected" : ""}`}
                aria-label={`${tab === "existing" ? "选用" : "参考"} ${asset.name}`}
                aria-pressed={selected}
                disabled={busy}
                onClick={() => choose(asset)}
              >
                <ImagePreview
                  projectId={snapshot.project.id}
                  revisionId={asset.latestRevisionId}
                  name={asset.name}
                />
                <span>{asset.name}</span>
                <small>
                  {selected
                    ? "已选择"
                    : `v${asset.revisions.find((rev) => rev.id === asset.latestRevisionId)?.number || 1}`}
                </small>
              </button>
            );
          })}
        </div>
        {!filtered.length && (
          <p className="panel-hint">
            {query
              ? "没有匹配的图片。试试其他名称，或上传图片。"
              : "项目还没有图片。可以上传，或切换到“与 Codex 生成”。"}
          </p>
        )}
        {filtered.length > limit && (
          <button
            className="button small"
            onClick={() => setLimit((value) => value + 24)}
          >
            显示更多图片（还有 {filtered.length - limit} 张）
          </button>
        )}
        {tab === "existing" ? (
          <div className="character-source-selection">
            <div>
              <label className="field">
                或上传原画
                <input
                  aria-label="或上传原画"
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  disabled={busy}
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    event.target.value = "";
                    if (file) void onUpload(file);
                  }}
                />
              </label>
              {selectedAsset && (
                <label className="field">
                  采用的原画版本
                  <select
                    aria-label="角色原画"
                    value={selectedRevisionId}
                    disabled={busy}
                    onChange={(event) => onSelect(event.target.value)}
                  >
                    {selectedAsset.revisions.map((revision) => (
                      <option key={revision.id} value={revision.id}>
                        {selectedAsset.name} · v{revision.number}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              <p className="panel-hint">
                更换后，下游制作结果需重新生成和审核；已有资产会保留。
              </p>
            </div>
            {selectedAsset && (
              <ImagePreview
                projectId={snapshot.project.id}
                revisionId={selectedRevisionId}
                name="已选角色原画预览"
              />
            )}
          </div>
        ) : (
          <CharacterCodexActions
            context={generationContext}
            notify={notify}
            actionContainer={actionContainer}
          />
        )}
        {error && (
          <p role="alert" className="resource-error">
            {error}
          </p>
        )}
      </div>
      <div className="modal-footer">
        <button className="button" disabled={busy} onClick={onClose}>
          取消
        </button>
        {tab === "existing" && (
          <button
            className="button primary"
            disabled={
              busy ||
              !selectedRevisionId ||
              selectedRevisionId === run.dagState?.views.original.revisionId
            }
            onClick={onAdopt}
          >
            采用这张原画
          </button>
        )}
        <div ref={setActionContainer} />
      </div>
    </ResourceDialog>
  );
}
