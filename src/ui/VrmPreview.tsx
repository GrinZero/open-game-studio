import { useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Bone,
  Grid2X2,
  Loader2,
  Pause,
  Play,
  RotateCcw,
  Upload,
} from "lucide-react";
import type { StageSettings, StageStatus, VrmStage } from "./vrm-stage";
import { TEST_DURATION, TEST_MOTIONS } from "./vrm-test-motion";

export type VrmViewerSettings = {
  motionClip: string;
  physicsEnabled: boolean;
  showBones: boolean;
  wireframe: boolean;
};
export function VrmPreview({
  src,
  name,
  initialSettings,
  onSettingsChange,
}: {
  src: string;
  name: string;
  initialSettings: VrmViewerSettings;
  onSettingsChange: (change: Partial<VrmViewerSettings>) => void;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<VrmStage | null>(null);
  const [motion, setMotion] = useState(
    initialSettings.motionClip in TEST_MOTIONS
      ? initialSettings.motionClip
      : "walk_cycle_90f.vrma",
  );
  const [physics, setPhysics] = useState(initialSettings.physicsEnabled);
  const [bones, setBones] = useState(initialSettings.showBones);
  const [wireframe, setWireframe] = useState(initialSettings.wireframe);
  const [playing, setPlaying] = useState(true);
  const [time, setTime] = useState(0);
  const [customName, setCustomName] = useState("");
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState("");
  const [status, setStatus] = useState<StageStatus>({
    phase: "loading",
    message: "",
    springJoints: 0,
    duration: TEST_DURATION,
  });
  const current = useRef({
    settings: { motion, physics, bones, wireframe } as StageSettings,
    playing,
  });
  useEffect(() => {
    current.current = {
      settings: { motion, physics, bones, wireframe },
      playing,
    };
    stageRef.current?.setSettings(current.current.settings);
    stageRef.current?.setPlaying(playing);
  }, [motion, physics, bones, wireframe, playing]);
  useEffect(() => {
    let active = true;
    let stage: VrmStage | undefined;
    import("./vrm-stage")
      .then(({ createVrmStage }) => {
        if (!active || !hostRef.current) return;
        stage = createVrmStage(
          hostRef.current,
          src,
          (status) => {
            if (active) setStatus(status);
          },
          (time) => {
            if (active) setTime(time);
          },
        );
        stage.setSettings(current.current.settings);
        stage.setPlaying(current.current.playing);
        stageRef.current = stage;
      })
      .catch((error) => {
        if (active)
          setStatus({
            phase: "error",
            message: `无法启动 VRM 预览：${error instanceof Error ? error.message : String(error)}`,
            duration: TEST_DURATION,
            springJoints: 0,
          });
      });
    return () => {
      active = false;
      stageRef.current = null;
      stage?.dispose();
    };
  }, [src]);
  const loaded = status.phase === "ready";
  const frames = Math.max(1, Math.round(status.duration * 30));
  const frame = Math.min(frames, Math.floor(time * 30) + 1);
  return (
    <section className="vrm-preview" aria-label={`${name} 动作预览`}>
      <div className="vrm-preview-toolbar">
        <label className="vrm-motion-field">
          <span>测试动作</span>
          <select
            aria-label="测试动作"
            value={motion}
            disabled={importing}
            onChange={(event) => {
              const value = event.target.value;
              setMotion(value);
              setCustomName("");
              setImportError("");
              setPlaying(true);
              stageRef.current?.seek(0);
              onSettingsChange({ motionClip: value });
            }}
          >
            {Object.entries(TEST_MOTIONS).map(([value, label]) => (
              <option value={value} key={value}>
                {label} · 内置
              </option>
            ))}
            {customName && <option value="__custom">{customName}</option>}
          </select>
        </label>
        <label
          className={`button small vrm-import ${!loaded || importing ? "is-disabled" : ""}`}
        >
          <Upload size={14} />
          {importing ? "加载动作中…" : "载入 VRMA"}
          <input
            type="file"
            accept=".vrma"
            aria-label="载入自定义 VRMA"
            disabled={!loaded || importing}
            onChange={async (event) => {
              const file = event.target.files?.[0];
              event.target.value = "";
              if (!file || !stageRef.current) return;
              setImporting(true);
              setImportError("");
              try {
                if (await stageRef.current.loadVrma(file)) {
                  setCustomName(file.name);
                  setMotion("__custom");
                  setPlaying(true);
                }
              } catch {
                setImportError(
                  "无法加载 VRMA 动作，请检查文件格式与人形动作数据。",
                );
              } finally {
                setImporting(false);
              }
            }}
          />
        </label>
      </div>
      <div className="vrm-preview-viewport">
        <div ref={hostRef} className="vrm-render-host" />
        {loaded && (
          <>
            <span className="vrm-playback-badge" role="status">
              <i className={playing ? "is-playing" : ""} />
              {playing ? "正在播放" : "已暂停"} ·{" "}
              {motion === "__custom"
                ? customName
                : TEST_MOTIONS[motion as keyof typeof TEST_MOTIONS]}
            </span>
            <span className="vrm-viewport-hint">
              拖动旋转 · 滚轮缩放 · 右键平移
            </span>
          </>
        )}
        {!loaded && (
          <div
            className={`vrm-preview-message ${status.phase === "error" ? "is-error" : ""}`}
            role={status.phase === "error" ? "alert" : "status"}
          >
            {status.phase === "error" ? (
              <AlertTriangle size={20} />
            ) : (
              <Loader2 size={20} className="spin" />
            )}
            <span>{status.message || "正在加载 VRM 与人形骨架…"}</span>
          </div>
        )}
      </div>
      <div className="vrm-playback-controls">
        <button
          className="button small"
          disabled={!loaded}
          aria-label={playing ? "暂停测试" : "播放测试"}
          onClick={() => setPlaying(!playing)}
        >
          {playing ? <Pause size={14} /> : <Play size={14} />}
          {playing ? "暂停" : "播放"}
        </button>
        <input
          type="range"
          aria-label="动作进度"
          min="0"
          max={frames - 1}
          step="1"
          value={Math.min(frames - 1, frame - 1)}
          disabled={!loaded}
          onChange={(event) => {
            setPlaying(false);
            stageRef.current?.seek(Number(event.target.value) / 30);
          }}
        />
        <output aria-label="当前帧">
          {String(frame).padStart(2, "0")} / {frames} 帧
        </output>
        <button
          className="button small"
          disabled={!loaded}
          aria-label="重播测试"
          title="从头播放"
          onClick={() => {
            stageRef.current?.seek(0);
            setPlaying(true);
          }}
        >
          <RotateCcw size={14} />
        </button>
      </div>
      <div className="vrm-view-controls">
        <button
          className={`button small ${physics ? "primary" : ""}`}
          disabled={!loaded || !status.springJoints}
          aria-pressed={physics && !!status.springJoints}
          onClick={() => {
            setPhysics(!physics);
            onSettingsChange({ physicsEnabled: !physics });
          }}
        >
          物理：
          {!loaded
            ? "加载中"
            : !status.springJoints
              ? "无 SpringBone"
              : physics
                ? "开"
                : "关"}
        </button>
        <button
          className="button small"
          disabled={!loaded}
          aria-pressed={bones}
          onClick={() => {
            setBones(!bones);
            onSettingsChange({ showBones: !bones });
          }}
        >
          <Bone size={14} />
          骨骼
        </button>
        <button
          className="button small"
          disabled={!loaded}
          aria-pressed={wireframe}
          onClick={() => {
            setWireframe(!wireframe);
            onSettingsChange({ wireframe: !wireframe });
          }}
        >
          <Grid2X2 size={14} />
          线框
        </button>
        <button
          className="button small vrm-reset-view"
          disabled={!loaded}
          onClick={() => stageRef.current?.resetView()}
        >
          重置视角
        </button>
      </div>
      <p className="vrm-preview-note">
        {motion === "__custom"
          ? "自定义 VRMA · 仅用于本次预览"
          : "内置人形动作 · 90 帧 / 30 fps"}
        {loaded && ` · ${status.springJoints} 个物理关节`}
      </p>
      {importError && (
        <p className="vrm-import-error" role="alert">
          {importError}
        </p>
      )}
    </section>
  );
}
