import type {
  CharacterDagNode,
  CharacterWorkflowState,
} from "./character-dag.js";
export type Engine = "godot" | "unity" | "unreal" | "other";
export interface ProjectCover {
  revisionId: string;
  fileIndex: number;
}
export interface Project {
  id: string;
  name: string;
  root: string;
  engine: Engine;
  collection: string;
  archived: boolean;
  createdAt: string;
  updatedAt: string;
  available: boolean;
  cover?: ProjectCover;
}
export type AssetKind =
  | "image"
  | "model"
  | "document"
  | "audio"
  | "video"
  | "source";
export interface AssetFile {
  name: string;
  relativePath: string;
  sha256: string;
  sizeBytes: number;
  role: "source" | "runtime" | "preview";
  mime: string;
}
export interface AssetRevision {
  id: string;
  assetId: string;
  projectId: string;
  number: number;
  files: AssetFile[];
  parentRevisionIds: string[];
  createdByRunId?: string;
  provenance: { sourceRef: string };
  createdAt: string;
}
export interface Asset {
  id: string;
  name: string;
  kind: AssetKind;
  tags: string[];
  archived: boolean;
  revision: number;
  latestRevisionId: string;
  createdAt: string;
  revisions: AssetRevision[];
}
export interface StudioNode {
  width?: number;
  height?: number;
  id: string;
  type: "asset" | "note" | "workflow" | "generation";
  position: { x: number; y: number };
  data: {
    title: string;
    assetId?: string;
    revisionId?: string;
    text?: string;
    instanceId?: string;
    taskId?: string;
  };
}
export interface StudioEdge {
  id: string;
  source: string;
  target: string;
  kind: "reference";
}
export interface WorkflowInstance {
  id: string;
  templateVersion: string;
  name: string;
  inputRevisionIds: string[];
}
export interface CanvasInput {
  nodeId?: string;
  title: string;
  type: "asset" | "note";
  role: "prompt" | "reference";
  assetId?: string;
  revisionId?: string;
  version?: number;
  kind?: AssetKind;
  text?: string;
  textTruncated?: boolean;
  files?: AssetFile[];
}
export interface CanvasInputSnapshot {
  graphRevision: number;
  targetNodeIds: string[];
  revisionIds: string[];
  items: CanvasInput[];
  prompt: string;
}
export interface WorkflowPlan {
  id: string;
  graphRevision: number;
  inputRevisionIds: string[];
  inputSnapshot: CanvasInputSnapshot;
  executionMode: string;
}
export interface Canvas {
  id: string;
  name: string;
  archived: boolean;
  graphRevision: number;
  layoutRevision: number;
  nodes: StudioNode[];
  edges: StudioEdge[];
  viewport: { x: number; y: number; zoom: number };
  instances: WorkflowInstance[];
  updatedAt: string;
}
export type StepStatus =
  | "pending"
  | "awaiting_agent"
  | "awaiting_manual"
  | "awaiting_review"
  | "succeeded"
  | "cancelled";
export interface WorkflowStep {
  key: string;
  title: string;
  mode: "agent" | "manual" | "review" | "local";
  status: StepStatus;
  inputRevisionIds: string[];
  outputRevisionIds: string[];
  stale: boolean;
  note: string;
}
export interface WorkflowRun {
  id: string;
  projectId: string;
  canvasId: string;
  instanceId: string;
  templateVersion: string;
  graphRevision: number;
  inputRevisionIds: string[];
  inputSnapshot?: CanvasInputSnapshot;
  steps: WorkflowStep[];
  dag?: CharacterDagNode[];
  dagState?: CharacterWorkflowState;
  branchFrom?: {
    runId: string;
    source: "head" | "body";
    candidateId: string;
    createdAt: string;
  };
  status: "active" | "succeeded" | "cancelled";
  revision: number;
  createdAt: string;
  updatedAt: string;
  parentRunId?: string;
  cost: { status: "not_applicable"; estimate: null; reason: string };
}
export interface ReviewDecision {
  id: string;
  runId: string;
  stepKey: string;
  inputRevisionIds: string[];
  outputRevisionIds: string[];
  scope: "appearance" | "deformation" | "final";
  decision: "accept" | "request_changes";
  actor: "user" | "delegated_agent";
  comment: string;
  createdAt: string;
}
export interface ExportBundle {
  id: string;
  runId?: string;
  revisionIds: string[];
  path: string;
  createdAt: string;
  validation: {
    technical: "passed";
    engine: "not_verified";
    visual: "passed" | "not_verified";
  };
}
export interface ProjectSnapshot {
  project: Project;
  assets: Asset[];
  canvases: Canvas[];
  tasks?: GenerationTask[];
  runs: WorkflowRun[];
  reviews: ReviewDecision[];
  exports: ExportBundle[];
}
export const phaseLabels = [
  "角色目标",
  "制作参考",
  "模型候选",
  "候选审核",
  "整理与组装",
  "绑定与变形",
  "变形审核",
  "成果交接",
];

export type GenerationKind =
  | "image"
  | "video"
  | "model"
  | "animation"
  | "audio";
export interface GenerationTask {
  id: string;
  projectId: string;
  canvasId: string;
  nodeId: string;
  kind: GenerationKind;
  name: string;
  prompt: string;
  settings: {
    aspectRatio?: string;
    duration?: number;
    animation?: string;
    transparent?: boolean;
  };
  inputSnapshot: CanvasInputSnapshot;
  status:
    | "draft"
    | "running"
    | "succeeded"
    | "failed"
    | "interrupted"
    | "cancelled";
  revision: number;
  idempotencyKey: string;
  outputRevisionId?: string;
  error?: string;
  summary?: string;
  workerPid?: number;
  createdAt: string;
  updatedAt: string;
}
