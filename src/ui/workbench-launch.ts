import {
  readWorkbenchTarget,
  targetFields,
  type WorkbenchTarget,
} from "../shared/workbench-target";

// Per-window launch intent. Never use a global last-opened project.
function initialTarget(): WorkbenchTarget {
  const saved =
    typeof document !== "undefined"
      ? document.getElementById("studio-launch")?.textContent
      : undefined;
  if (saved) {
    try {
      return readWorkbenchTarget(JSON.parse(saved));
    } catch {
      /* fall back to URL */
    }
  }
  return readWorkbenchTarget(
    Object.fromEntries(new URLSearchParams(location.search)),
  );
}
let target = initialTarget();
let navigationId: string | undefined;
const listeners = new Set<() => void>();
export function launchRequest() {
  return target;
}
export function launchProject() {
  return target.projectId;
}
export function launchCanvas(project: string) {
  return project === target.projectId ? target.canvasId : undefined;
}
export function characterLaunch(project: string) {
  return project === target.projectId &&
    (target.view === "character" || (target.runId && !target.stepKey))
    ? target
    : undefined;
}
export function subscribeLaunch(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}
export function receiveLaunch(value: unknown) {
  if (!value || typeof value !== "object") return;
  const data = value as {
    target?: unknown;
    project?: { project?: { id?: string } };
    projectId?: string;
    navigationId?: string;
    [key: string]: unknown;
  };
  const next = readWorkbenchTarget(data.target || data);
  next.projectId ||= data.project?.project?.id;
  if (!next.projectId) return;
  // Hosts replay project-only context when a reloaded iframe connects. Keep
  // the more specific target restored from that iframe's saved launch data.
  if (
    next.projectId === target.projectId &&
    Object.keys(next).length === 1 &&
    !data.navigationId &&
    !data.target
  )
    return;
  const same = targetFields.every((field) => next[field] === target[field]);
  if (same && (!data.navigationId || data.navigationId === navigationId))
    return;
  target = next;
  navigationId = data.navigationId;
  for (const listener of listeners) listener();
}
