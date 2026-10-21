import { useEffect, useRef, useState } from "react";
import { isEmbeddedApp, request } from "./api";

export function ProjectLiveSync({
  projectId,
  refresh,
  paused,
}: {
  projectId: string;
  refresh: () => Promise<void>;
  paused: boolean;
}) {
  const refreshRef = useRef(refresh);
  const pausedRef = useRef(paused);
  refreshRef.current = refresh;
  pausedRef.current = paused;
  const [status, setStatus] = useState<"connecting" | "live" | "reconnecting">(
    "connecting",
  );
  useEffect(() => {
    let closed = false;
    let loading = false;
    let checking = false;
    let pending = false;
    let version: string | undefined;
    const abort = new AbortController();
    async function flush() {
      if (closed || loading || !pending || pausedRef.current) return;
      loading = true;
      pending = false;
      try {
        await refreshRef.current();
        if (!closed) setStatus("live");
      } catch {
        if (!closed) {
          pending = true;
          setStatus("reconnecting");
        }
      } finally {
        loading = false;
      }
    }
    let events: EventSource | undefined;
    if (!isEmbeddedApp()) {
      events = new EventSource(`/api/projects/${projectId}/changes`);
      events.onmessage = () => {
        if (!closed) {
          pending = true;
          void flush();
        }
      };
      events.onerror = () => {
        if (!closed) setStatus("reconnecting");
      };
    }
    async function tick() {
      if (isEmbeddedApp() && !checking && !closed) {
        checking = true;
        try {
          const next = await request<{ version: string }>(
            `/api/projects/${projectId}/change-version`,
            { signal: abort.signal },
          );
          if (closed) return;
          if (version !== next.version) {
            version = next.version;
            pending = true;
          }
          setStatus("live");
        } catch {
          if (!closed) setStatus("reconnecting");
        } finally {
          checking = false;
        }
      }
      await flush();
    }
    void tick();
    const timer = setInterval(() => {
      void tick();
    }, 1000);
    return () => {
      closed = true;
      abort.abort();
      clearInterval(timer);
      events?.close();
    };
  }, [projectId]);
  return (
    <span
      className={`project-live-sync ${status}`}
      role="status"
      aria-label="协作同步状态"
    >
      <i aria-hidden="true" />
      {status === "live"
        ? "对话协作 · 自动同步"
        : status === "reconnecting"
          ? "协作同步重连中…"
          : "正在连接协作同步…"}
    </span>
  );
}
