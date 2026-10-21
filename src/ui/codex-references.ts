import { formatCodexContext, type CodexContext } from "./codex-context";

export interface CodexReference {
  key: string;
  context: CodexContext;
}

/** The host replaces context on each update, so always send the full set.
 * Commit only after acknowledgement, and serialize additions/removals. */
export class CodexReferenceStore {
  private references: readonly CodexReference[] = [];
  private listeners = new Set<() => void>();
  private pending: Promise<void> = Promise.resolve();

  constructor(private publish: (text: string) => Promise<void>) {}

  getSnapshot = () => this.references;
  subscribe = (listener: () => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  private change(
    transform: (
      current: readonly CodexReference[],
    ) => readonly CodexReference[],
  ) {
    const operation = this.pending.then(async () => {
      const next = transform(this.references);
      await this.publish(
        next.length
          ? next
              .map((reference) => formatCodexContext(reference.context))
              .join("\n\n")
          : "Open Game Studio：用户已移除全部工作台引用。后续消息不再默认引用先前所选素材。",
      );
      this.references = next;
      this.listeners.forEach((listener) => listener());
    });
    this.pending = operation.catch(() => {});
    return operation;
  }

  add(context: CodexContext) {
    const frozen = structuredClone(context);
    const key = JSON.stringify([
      frozen.project.id,
      frozen.canvas.id,
      frozen.selection.map((item) => item.nodeId).sort(),
      ...(frozen.characterTarget
        ? [
            frozen.characterTarget.runId,
            frozen.characterTarget.nodeKey,
            frozen.characterTarget.slotKey,
          ]
        : []),
    ]);
    return this.change((current) => [
      ...current.filter((item) => item.key !== key),
      { key, context: frozen },
    ]);
  }
  remove(key: string) {
    return this.change((current) => current.filter((item) => item.key !== key));
  }
  clear() {
    return this.change(() => []);
  }
}
