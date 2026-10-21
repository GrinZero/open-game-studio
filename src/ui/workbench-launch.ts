// Launch context belongs to this app instance, not a global last-used project.
let projectId =
  new URLSearchParams(location.search).get("projectId") || undefined;
const listeners = new Set<() => void>();
export function launchProject() {
  return projectId;
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
    projectId?: unknown;
    project?: { project?: { id?: unknown } };
  };
  const next = data.projectId || data.project?.project?.id;
  if (typeof next !== "string" || !next || next === projectId) return;
  projectId = next;
  for (const listener of listeners) listener();
}

export function characterLaunch(project: string) {
  const query = new URLSearchParams(location.search);
  if (query.get("projectId") !== project || query.get("view") !== "character")
    return undefined;
  return {
    runId: query.get("runId") || undefined,
    nodeKey: query.get("nodeKey") || undefined,
  };
}
