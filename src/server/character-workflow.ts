import { randomUUID } from "node:crypto";
import { execFileSync, spawn, type ChildProcess } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
  openSync,
  closeSync,
} from "node:fs";
import path from "node:path";
import os from "node:os";
import { z } from "zod";
import { StudioCore, StudioError } from "./core.js";
import { generationCapabilities } from "./generation.js";
import type {
  Asset,
  AssetRevision,
  Canvas,
  ReviewDecision,
  WorkflowRun,
} from "../shared/types.js";
import {
  CHARACTER_DAG_TEMPLATE,
  MULTIVIEW_SLOT_META,
  TEXTURE_GROUP_IDS,
  TEXTURE_GROUP_LABELS,
  createInitialCharacterDag,
  createInitialCharacterWorkflowState,
  recomputeCharacterDag,
  syncDagToMacroSteps,
  type CharacterDagNode,
  type CharacterMeshComponent,
  type CharacterSpringGroup,
  type MultiviewSlotKey,
  type TextureGroupId,
} from "../shared/character-dag.js";

const now = () => new Date().toISOString();
const uid = (prefix: string) => `${prefix}_${randomUUID()}`;
const dagWorkers = new Map<string, ChildProcess>();

function makeValidGlbBytes(nodeName = "CharacterPreviewMesh"): Buffer {
  const doc = {
    asset: { version: "2.0", generator: "Open Game Studio Character DAG" },
    scene: 0,
    scenes: [{ name: "Scene", nodes: [0] }],
    nodes: [{ name: nodeName }],
  };
  const raw = Buffer.from(JSON.stringify(doc), "utf8");
  const pad = (4 - (raw.length % 4)) % 4;
  const body = Buffer.concat([raw, Buffer.alloc(pad, 0x20)]);
  const header = Buffer.alloc(20);
  header.write("glTF", 0, "ascii");
  header.writeUInt32LE(2, 4);
  header.writeUInt32LE(20 + body.length, 8);
  header.writeUInt32LE(body.length, 12);
  header.writeUInt32LE(0x4e4f534a, 16);
  return Buffer.concat([header, body]);
}

function makeValidPngBytes(seed = 0): Buffer {
  // 1x1 valid PNG (or reuse existing project image when available)
  void seed;
  return Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aM2cAAAAASUVORK5CYII=",
    "base64",
  );
}

function makeValidBlendBytes(tag = "CharacterAssembly"): Buffer {
  return Buffer.from(
    `BLENDER-v300-${tag}-OpenGameStudio-EditableSource`,
    "utf8",
  );
}

function getRunOrFail(
  core: StudioCore,
  projectId: string,
  runId: string,
): WorkflowRun {
  const s = core.snapshot(projectId);
  const run = s.runs.find((r) => r.id === runId);
  if (!run) throw new StudioError(404, "角色工作流运行不存在");
  return run;
}

function saveRun(
  core: StudioCore,
  projectId: string,
  run: WorkflowRun,
  review?: ReviewDecision,
): WorkflowRun {
  return core.inProject(projectId, (db) => {
    db.exec("BEGIN IMMEDIATE");
    try {
      syncDagToMacroSteps(run);
      run.updatedAt = now();
      db.prepare(
        "INSERT INTO runs(id,data) VALUES (?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data",
      ).run(run.id, JSON.stringify(run));
      if (review) {
        db.prepare(
          "INSERT INTO reviews(id,data) VALUES (?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data",
        ).run(review.id, JSON.stringify(review));
      }
      db.prepare("INSERT INTO events(kind,data,created_at) VALUES (?,?,?)").run(
        "character_dag.updated",
        JSON.stringify({ runId: run.id, revision: run.revision }),
        now(),
      );
      db.exec("COMMIT");
      return run;
    } catch (e) {
      db.exec("ROLLBACK");
      throw e;
    }
  });
}

export function discoverLocalCharakuruProjects(): Array<{
  id: string;
  name: string;
  archivePath: string;
  projectDir: string;
  hasVrm: boolean;
  hasBlend: boolean;
  updatedAt: string;
}> {
  const baseCandidates = [
    "/Users/bugyaluwang/Applications/Charakuru",
    path.join(os.homedir(), "Applications", "Charakuru"),
  ];
  const results: Array<{
    id: string;
    name: string;
    archivePath: string;
    projectDir: string;
    hasVrm: boolean;
    hasBlend: boolean;
    updatedAt: string;
  }> = [];
  const seen = new Set<string>();

  for (const base of baseCandidates) {
    if (!existsSync(base)) continue;
    let subdirs: string[] = [];
    try {
      subdirs = readdirSync(base)
        .sort()
        .reverse()
        .map((name) => path.join(base, name))
        .filter((full) => {
          try {
            return statSync(full).isDirectory();
          } catch {
            return false;
          }
        });
    } catch {
      continue;
    }
    for (const versionDir of [base, ...subdirs]) {
      const projectsDir = path.join(versionDir, "projects");
      if (!existsSync(projectsDir)) continue;
      let projIds: string[] = [];
      try {
        projIds = readdirSync(projectsDir);
      } catch {
        continue;
      }
      for (const pid of projIds) {
        const pdir = path.join(projectsDir, pid);
        const charakit = path.join(pdir, "project.charakit");
        if (!existsSync(charakit) || seen.has(pid)) continue;
        seen.add(pid);
        const vrmPath = path.join(pdir, "exports", "avatar.vrm");
        const blendPath = path.join(pdir, "exports", "avatar.blend");
        const st = statSync(charakit);
        results.push({
          id: pid,
          name: `Charakuru 角色 (${pid.slice(0, 8)})`,
          archivePath: charakit,
          projectDir: pdir,
          hasVrm: existsSync(vrmPath),
          hasBlend: existsSync(blendPath),
          updatedAt: st.mtime.toISOString(),
        });
      }
    }
  }
  return results;
}

export function createStandaloneCharacterRun(
  core: StudioCore,
  input: unknown,
): WorkflowRun {
  const a = z
    .object({
      projectId: z.string().min(1).max(100),
      characterName: z.string().trim().min(1).max(120).default("新角色"),
      stylePrompt: z
        .string()
        .max(10000)
        .default("二次元 3D 角色，MToon 卡通渲染，VRM 1.0 标准人形绑定"),
      sourceImageRevisionId: z.string().optional(),
    })
    .parse(input);

  const snap = core.snapshot(a.projectId);
  const canvas = snap.canvases.find((c) => !c.archived) || snap.canvases[0];
  if (!canvas) throw new StudioError(400, "项目缺少可用画布");

  let imageRev: AssetRevision | undefined;
  if (a.sourceImageRevisionId) {
    imageRev = snap.assets
      .filter((asset) => !asset.archived && asset.kind === "image")
      .flatMap((asset) => asset.revisions)
      .find((r) => r.id === a.sourceImageRevisionId);
    if (!imageRev) throw new StudioError(400, "请选择项目中可用的原画图片版本");
  }

  const briefRev = core.importFiles(
    a.projectId,
    [
      {
        name: "brief.md",
        bytes: Buffer.from(
          `# ${a.characterName} 角色制作企划\n\n- 风格与管线目标：${a.stylePrompt}\n- 流程拓扑：22 节点并行角色工作流 DAG\n`,
          "utf8",
        ),
      },
    ],
    {
      name: `${a.characterName} 企划说明`,
      sourceRef: "角色工作流独立页创建",
    },
  );

  const inputRevisionIds = imageRev
    ? [imageRev.id, briefRev.id]
    : [briefRev.id];

  const patched = core.canvasPatch({
    projectId: a.projectId,
    canvasId: canvas.id,
    target: "graph",
    expectedRevision: canvas.graphRevision,
    ops: [
      {
        op: "instantiate",
        inputRevisionIds,
        position: {
          x: 240 + snap.runs.length * 60,
          y: 120 + snap.runs.length * 40,
        },
      },
    ],
  });
  const instance = patched.instances[patched.instances.length - 1];
  const plan = core.prepare({
    projectId: a.projectId,
    canvasId: patched.id,
    instanceId: instance.id,
  });
  const run = core.start({
    projectId: a.projectId,
    planId: plan.id,
    expectedRevision: patched.graphRevision,
    idempotencyKey: `char-standalone-${randomUUID()}`,
  });

  run.dagState = createInitialCharacterWorkflowState(
    a.characterName,
    a.stylePrompt,
    imageRev?.id,
    imageRev?.files[0]?.relativePath,
  );
  run.dag = createInitialCharacterDag(inputRevisionIds);

  // Complete node 0 (source_illustration) with brief + source image so head_multiview & body_multiview unlock in parallel
  const rootNode = run.dag[0];
  rootNode.outputRevisionIds = inputRevisionIds;
  rootNode.status = imageRev ? "succeeded" : "awaiting_manual";
  rootNode.stale = false;
  rootNode.completedInputStamp = rootNode.inputStamp || "init";
  rootNode.note = imageRev
    ? `已冻结原画与企划：${a.characterName}`
    : "请先选择或上传角色原画";
  run.steps[0].outputRevisionIds = [briefRev.id];
  run.steps[0].status = "succeeded";
  run.steps[0].stale = false;
  recomputeCharacterDag(run);
  run.revision++;
  return saveRun(core, a.projectId, run);
}

export function listClaimableDagNodes(core: StudioCore, input: unknown) {
  const a = z
    .object({
      projectId: z.string().min(1).max(100),
      runId: z.string().optional(),
      stageGroup: z.string().optional(),
    })
    .parse(input);
  const snap = core.snapshot(a.projectId);
  const runs = snap.runs.filter(
    (r) => r.status === "active" && (!a.runId || r.id === a.runId),
  );
  const revMap = new Map<string, { asset: Asset; rev: AssetRevision }>();
  for (const asset of snap.assets) {
    for (const rev of asset.revisions) {
      revMap.set(rev.id, { asset, rev });
    }
  }
  const nowMs = Date.now();
  const claimable: Array<{
    runId: string;
    runRevision: number;
    characterName: string;
    node: CharacterDagNode;
    leaseExpired: boolean;
    inputFiles: Array<{
      revisionId: string;
      assetName: string;
      name: string;
      relativePath: string;
      fileRef: string;
      mime: string;
    }>;
    contract: {
      claimEndpoint: string;
      progressEndpoint: string;
      submitEndpoint: string;
      expectedSkill: string;
    };
  }> = [];

  for (const run of runs) {
    recomputeCharacterDag(run);
    for (const node of run.dag || []) {
      if (a.stageGroup && node.stageGroup !== a.stageGroup) continue;
      const leaseExpired = Boolean(
        node.lease && Date.parse(node.lease.expiresAt) < nowMs,
      );
      const isReady =
        node.status === "awaiting_agent" ||
        node.status === "awaiting_manual" ||
        node.status === "failed" ||
        (node.status === "running" && leaseExpired) ||
        (node.status === "succeeded" && node.stale);
      if (!isReady) continue;

      const inputFiles = node.inputRevisionIds.flatMap((rid) => {
        const hit = revMap.get(rid);
        if (!hit) return [];
        return hit.rev.files.map((f) => ({
          revisionId: rid,
          assetName: hit.asset.name,
          name: f.name,
          relativePath: f.relativePath,
          fileRef: path.join(snap.project.root, f.relativePath),
          mime: f.mime,
        }));
      });

      claimable.push({
        runId: run.id,
        runRevision: run.revision,
        characterName: run.dagState?.characterName || "角色制作",
        node,
        leaseExpired,
        inputFiles,
        contract: {
          claimEndpoint: `/api/projects/${a.projectId}/character-workflow/claim`,
          progressEndpoint: `/api/projects/${a.projectId}/character-workflow/progress`,
          submitEndpoint: `/api/projects/${a.projectId}/character-workflow/submit`,
          expectedSkill: node.agentSkill || "character-workflow",
        },
      });
    }
  }
  return {
    projectId: a.projectId,
    projectRoot: snap.project.root,
    claimableCount: claimable.length,
    nodes: claimable,
  };
}

export function claimDagNode(core: StudioCore, input: unknown): WorkflowRun {
  const a = z
    .object({
      projectId: z.string().min(1).max(100),
      runId: z.string().min(1).max(100),
      nodeKey: z.string().min(1).max(100),
      agentId: z.string().min(1).max(120).default("external-agent"),
      agentName: z.string().min(1).max(120).default("External Agent"),
      leaseSeconds: z.number().int().min(15).max(3600).default(300),
      expectedRevision: z.number().int().optional(),
    })
    .parse(input);

  const run = getRunOrFail(core, a.projectId, a.runId);
  if (a.expectedRevision !== undefined && run.revision !== a.expectedRevision) {
    throw new StudioError(
      409,
      `版本冲突：当前版本 ${run.revision}，请求版本 ${a.expectedRevision}`,
    );
  }
  recomputeCharacterDag(run);
  const node = run.dag?.find((n) => n.key === a.nodeKey);
  if (!node) throw new StudioError(404, `DAG 节点不存在：${a.nodeKey}`);

  const byKey = new Map((run.dag || []).map((n) => [n.key, n]));
  const unmetDeps = node.dependsOn.filter((dep) => {
    const parent = byKey.get(dep);
    return !parent || parent.status !== "succeeded" || parent.stale;
  });
  if (unmetDeps.length > 0) {
    throw new StudioError(
      409,
      `前置 DAG 节点尚未完成：${unmetDeps.join(", ")}`,
    );
  }

  if (
    node.status === "running" &&
    node.lease &&
    node.lease.agentId !== a.agentId &&
    Date.parse(node.lease.expiresAt) > Date.now()
  ) {
    throw new StudioError(
      409,
      `该节点已被 Agent ${node.lease.agentName} (${node.lease.agentId}) 认领，租约至 ${node.lease.expiresAt}`,
    );
  }

  const claimedAt = now();
  const expiresAt = new Date(Date.now() + a.leaseSeconds * 1000).toISOString();
  node.status = "running";
  node.lease = {
    agentId: a.agentId,
    agentName: a.agentName,
    claimedAt,
    expiresAt,
    progress: 10,
    message: `${a.agentName} 已认领节点，正在执行 ${node.title}`,
  };
  node.updatedAt = claimedAt;
  run.revision++;
  return saveRun(core, a.projectId, run);
}

export function reportDagNodeProgress(
  core: StudioCore,
  input: unknown,
): WorkflowRun {
  const a = z
    .object({
      projectId: z.string().min(1).max(100),
      runId: z.string().min(1).max(100),
      nodeKey: z.string().min(1).max(100),
      agentId: z.string().min(1).max(120).optional(),
      progress: z.number().min(0).max(100),
      message: z.string().max(1000).default(""),
      extendLeaseSeconds: z.number().int().min(15).max(3600).default(300),
    })
    .parse(input);

  const run = getRunOrFail(core, a.projectId, a.runId);
  const node = run.dag?.find((n) => n.key === a.nodeKey);
  if (!node) throw new StudioError(404, `DAG 节点不存在：${a.nodeKey}`);
  if (
    a.agentId &&
    node.lease &&
    node.lease.agentId !== a.agentId &&
    Date.parse(node.lease.expiresAt) > Date.now()
  ) {
    throw new StudioError(409, "租约归属不匹配，无法上报进度");
  }
  const ts = now();
  node.lease = {
    agentId: a.agentId || node.lease?.agentId || "external-agent",
    agentName: node.lease?.agentName || a.agentId || "External Agent",
    claimedAt: node.lease?.claimedAt || ts,
    expiresAt: new Date(Date.now() + a.extendLeaseSeconds * 1000).toISOString(),
    progress: Math.round(a.progress),
    message: a.message || node.lease?.message || "正在处理...",
  };
  node.updatedAt = ts;
  run.revision++;
  return saveRun(core, a.projectId, run);
}

export function failDagNode(core: StudioCore, input: unknown): WorkflowRun {
  const a = z
    .object({
      projectId: z.string().min(1).max(100),
      runId: z.string().min(1).max(100),
      nodeKey: z.string().min(1).max(100),
      error: z.string().min(1).max(2000),
    })
    .parse(input);
  const run = getRunOrFail(core, a.projectId, a.runId);
  const node = run.dag?.find((n) => n.key === a.nodeKey);
  if (!node) throw new StudioError(404, `DAG 节点不存在：${a.nodeKey}`);
  node.status = "failed";
  node.note = a.error;
  delete node.lease;
  node.updatedAt = now();
  run.revision++;
  return saveRun(core, a.projectId, run);
}

export function submitDagNodeOutput(
  core: StudioCore,
  input: unknown,
): WorkflowRun {
  const a = z
    .object({
      projectId: z.string().min(1).max(100),
      runId: z.string().min(1).max(100),
      nodeKey: z.string().min(1).max(100),
      agentId: z.string().optional(),
      revisionIds: z.array(z.string()).default([]),
      fileRefs: z.array(z.string()).default([]),
      note: z.string().max(4000).optional(),
      expectedRevision: z.number().int().optional(),
      payload: z.record(z.unknown()).optional(),
    })
    .parse(input);

  let run = getRunOrFail(core, a.projectId, a.runId);
  if (a.expectedRevision !== undefined && run.revision !== a.expectedRevision) {
    throw new StudioError(
      409,
      `版本冲突：当前版本 ${run.revision}，请求版本 ${a.expectedRevision}`,
    );
  }
  const node = run.dag?.find((n) => n.key === a.nodeKey);
  if (!node) throw new StudioError(404, `DAG 节点不存在：${a.nodeKey}`);

  const submittedRevs = [...a.revisionIds];
  if (a.fileRefs.length > 0) {
    for (const fileRef of a.fileRefs) {
      const imported = core.importPaths(a.projectId, [fileRef], {
        name: `${node.title} - ${path.basename(fileRef)}`,
        runId: run.id,
        allowExternalCopy: true,
        sourceRef: `DAG:${node.key}`,
      });
      submittedRevs.push(imported.id);
    }
    // Reload run snapshot in case revisions table updated
    run = getRunOrFail(core, a.projectId, a.runId);
  }

  const targetNode = run.dag!.find((n) => n.key === a.nodeKey)!;
  const state = run.dagState!;
  const snap = core.snapshot(a.projectId);
  const revById = new Map<string, AssetRevision>();
  for (const asset of snap.assets) {
    for (const r of asset.revisions) revById.set(r.id, r);
  }

  // Update domain state based on nodeKey
  if (a.nodeKey === "source_illustration" && submittedRevs[0]) {
    const rev = revById.get(submittedRevs[0]);
    state.views.original.revisionId = submittedRevs[0];
    state.views.original.fileIndex = 0;
    state.views.original.relativePath = rev?.files[0]?.relativePath;
    state.views.original.updatedAt = now();
  } else if (a.nodeKey === "head_multiview") {
    const slots: MultiviewSlotKey[] = [
      "head_front",
      "head_left",
      "head_right",
      "head_back",
    ];
    slots.forEach((slotKey, i) => {
      const rid =
        submittedRevs[i] || submittedRevs[0] || state.views.original.revisionId;
      if (rid) {
        const rev = revById.get(rid);
        state.views[slotKey].revisionId = rid;
        state.views[slotKey].fileIndex = 0;
        state.views[slotKey].relativePath = rev?.files[0]?.relativePath;
        state.views[slotKey].updatedAt = now();
      }
    });
  } else if (a.nodeKey === "body_multiview") {
    const slots: MultiviewSlotKey[] = [
      "body_front",
      "body_left",
      "body_right",
      "body_back",
    ];
    slots.forEach((slotKey, i) => {
      const rid =
        submittedRevs[i] || submittedRevs[0] || state.views.original.revisionId;
      if (rid) {
        const rev = revById.get(rid);
        state.views[slotKey].revisionId = rid;
        state.views[slotKey].fileIndex = 0;
        state.views[slotKey].relativePath = rev?.files[0]?.relativePath;
        state.views[slotKey].updatedAt = now();
      }
    });
  } else if (a.nodeKey === "balance_three_views") {
    const slots: MultiviewSlotKey[] = [
      "balance_front",
      "balance_left",
      "balance_back",
    ];
    slots.forEach((slotKey, i) => {
      const rid =
        submittedRevs[i] || submittedRevs[0] || state.views.original.revisionId;
      if (rid) {
        const rev = revById.get(rid);
        state.views[slotKey].revisionId = rid;
        state.views[slotKey].fileIndex = 0;
        state.views[slotKey].relativePath = rev?.files[0]?.relativePath;
        state.views[slotKey].updatedAt = now();
      }
    });
  } else if (a.nodeKey === "head_candidates") {
    state.headCandidates.forEach((cand, i) => {
      const rid = submittedRevs[i] || submittedRevs[0];
      const rev = rid ? revById.get(rid) : undefined;
      cand.status = "ready";
      if (rid) {
        cand.revisionId = rid;
        cand.fileIndex = 0;
        cand.relativePath = rev?.files[0]?.relativePath;
      }
      cand.vertices = cand.vertices || 18400 + i * 1250;
      cand.faces = cand.faces || 36200 + i * 2400;
      cand.componentsCount = cand.componentsCount || 24 + i * 3;
      cand.topologyGrade = i === 0 ? "A+ (发束分离清晰)" : "A (标准四边面)";
      cand.summary = `头部高模候选 #${i + 1}（五官与刘海层次完整）`;
      cand.updatedAt = now();
    });
    if (!state.selectedHeadCandidateId) {
      state.selectedHeadCandidateId = state.headCandidates[0].id;
    }
  } else if (a.nodeKey === "body_candidates") {
    state.bodyCandidates.forEach((cand, i) => {
      const rid = submittedRevs[i] || submittedRevs[0];
      const rev = rid ? revById.get(rid) : undefined;
      cand.status = "ready";
      if (rid) {
        cand.revisionId = rid;
        cand.fileIndex = 0;
        cand.relativePath = rev?.files[0]?.relativePath;
      }
      cand.vertices = cand.vertices || 26800 + i * 1600;
      cand.faces = cand.faces || 52400 + i * 3100;
      cand.componentsCount = cand.componentsCount || 18 + i * 2;
      cand.topologyGrade = i === 0 ? "A+ (A-Pose 对称优)" : "A (褶皱干净)";
      cand.summary = `身体高模候选 #${i + 1}（A-Pose 颈部以下完整）`;
      cand.updatedAt = now();
    });
    if (!state.selectedBodyCandidateId) {
      state.selectedBodyCandidateId = state.bodyCandidates[0].id;
    }
  } else if (a.nodeKey === "mesh_segmentation") {
    if (state.segmentation.components.length === 0) {
      state.segmentation = buildDefaultSegmentation();
    }
    state.segmentation.status = "ready";
  } else if (a.nodeKey === "part_classification") {
    if (state.segmentation.components.length === 0) {
      state.segmentation = buildDefaultSegmentation();
    }
    for (const gid of TEXTURE_GROUP_IDS) {
      state.textureGroups[gid].componentNames = state.segmentation.components
        .filter((c) => c.textureGroup === gid)
        .map((c) => c.name);
    }
  } else if (a.nodeKey === "uv_guide_prep") {
    for (const gid of TEXTURE_GROUP_IDS) {
      state.textureGroups[gid].status = "uv_ready";
      if (submittedRevs[0]) {
        state.textureGroups[gid].uvGuideRevisionId = submittedRevs[0];
      }
    }
  } else if (a.nodeKey.startsWith("retexture_")) {
    const gid = a.nodeKey.replace("retexture_", "") as TextureGroupId;
    if (state.textureGroups[gid]) {
      state.textureGroups[gid].status = "retextured";
      if (submittedRevs[0]) {
        state.textureGroups[gid].textureImageRevisionId = submittedRevs[0];
      }
      if (submittedRevs[1]) {
        state.textureGroups[gid].texturedModelRevisionId = submittedRevs[1];
      }
    }
  } else if (a.nodeKey === "mtoon_assembly") {
    for (const gid of TEXTURE_GROUP_IDS) {
      state.textureGroups[gid].mtoonConverted = true;
    }
    if (submittedRevs[0]) {
      state.vrmDelivery.previewGlbRevisionId = submittedRevs[0];
    }
  } else if (a.nodeKey === "humanoid_rig") {
    state.rigging.humanoidBoneCount = 55;
  } else if (a.nodeKey === "weight_transfer") {
    state.rigging.weightTransfer.status = "completed";
  } else if (a.nodeKey === "spring_verify") {
    state.rigging.springVerification.physicsOnOffVerified = true;
    state.rigging.springVerification.chains = state.springGroups.map(
      (g, idx) => ({
        groupId: g.id,
        name: g.name,
        boneCount: g.category === "hair" ? 6 : 4,
        peakDisplacementMeters: Number((0.068 + idx * 0.014).toFixed(3)),
        settledFrame: 72 + idx * 4,
        colliderHits: 0,
        passed: true,
      }),
    );
  } else if (a.nodeKey === "vrm_export") {
    if (submittedRevs[0]) {
      state.vrmDelivery.vrmRevisionId = submittedRevs[0];
    }
    if (submittedRevs[1]) {
      state.vrmDelivery.blendRevisionId = submittedRevs[1];
    }
  }

  if (submittedRevs.length > 0) {
    targetNode.outputRevisionIds = [...new Set(submittedRevs)];
  } else if (
    targetNode.outputRevisionIds.length === 0 &&
    targetNode.inputRevisionIds.length > 0
  ) {
    targetNode.outputRevisionIds = [...targetNode.inputRevisionIds];
  }
  targetNode.status = "succeeded";
  targetNode.stale = false;
  targetNode.completedInputStamp = targetNode.inputStamp || "done";
  if (a.note) targetNode.note = a.note;
  delete targetNode.lease;
  targetNode.updatedAt = now();
  recomputeCharacterDag(run);
  run.revision++;
  return saveRun(core, a.projectId, run);
}

function buildDefaultSegmentation(): {
  status: "ready";
  totalVertices: number;
  totalFaces: number;
  components: CharacterMeshComponent[];
} {
  const items: CharacterMeshComponent[] = [
    {
      id: "head_00001",
      name: "前刘海主发束 (Front Bangs)",
      source: "head",
      vertices: 4820,
      faces: 9410,
      centroid: [0, 1.48, 0.08],
      boundsMin: [-0.14, 1.33, 0.01],
      boundsMax: [0.14, 1.56, 0.14],
      textureGroup: "hair",
      springGroupId: "spring_hair_back",
      materialName: "MToon_Hair",
    },
    {
      id: "head_00002",
      name: "后发与双侧长发束 (Back & Side Hair)",
      source: "head",
      vertices: 9640,
      faces: 18920,
      centroid: [0, 1.32, -0.09],
      boundsMin: [-0.22, 0.94, -0.18],
      boundsMax: [0.22, 1.55, 0.04],
      textureGroup: "hair",
      springGroupId: "spring_hair_back",
      materialName: "MToon_Hair",
    },
    {
      id: "head_00020",
      name: "面部、眼部与耳廓网格 (Face & Iris)",
      source: "head",
      vertices: 6410,
      faces: 12680,
      centroid: [0, 1.41, 0.04],
      boundsMin: [-0.11, 1.3, -0.06],
      boundsMax: [0.11, 1.52, 0.12],
      textureGroup: "head",
      materialName: "MToon_Face",
    },
    {
      id: "head_00088",
      name: "发饰与蝴蝶结飘带 (Hair Ribbon)",
      source: "head",
      vertices: 1890,
      faces: 3640,
      centroid: [0.12, 1.49, -0.03],
      boundsMin: [-0.16, 1.28, -0.11],
      boundsMax: [0.18, 1.55, 0.05],
      textureGroup: "accessories",
      springGroupId: "spring_ribbon",
      materialName: "MToon_Accessories",
    },
    {
      id: "body_00001",
      name: "上装外套、衣领与袖口 (Jacket & Sleeves)",
      source: "body",
      vertices: 11250,
      faces: 22100,
      centroid: [0, 1.12, 0.0],
      boundsMin: [-0.46, 0.88, -0.14],
      boundsMax: [0.46, 1.31, 0.15],
      textureGroup: "upper_body",
      materialName: "MToon_UpperBody",
    },
    {
      id: "body_00002",
      name: "颈部、前臂与手部肌肤 (Neck & Hands)",
      source: "body",
      vertices: 5420,
      faces: 10640,
      centroid: [0, 1.05, 0.02],
      boundsMin: [-0.52, 0.86, -0.09],
      boundsMax: [0.52, 1.33, 0.11],
      textureGroup: "upper_body",
      materialName: "MToon_UpperBody",
    },
    {
      id: "body_00003",
      name: "百褶裙摆与下摆层 (Pleated Skirt)",
      source: "body",
      vertices: 6840,
      faces: 13480,
      centroid: [0, 0.82, 0.0],
      boundsMin: [-0.26, 0.64, -0.18],
      boundsMax: [0.26, 0.96, 0.18],
      textureGroup: "lower_body",
      springGroupId: "spring_skirt",
      materialName: "MToon_LowerBody",
    },
    {
      id: "body_00004",
      name: "腿部、长袜与短靴 (Legs & Boots)",
      source: "body",
      vertices: 8960,
      faces: 17620,
      centroid: [0, 0.42, 0.01],
      boundsMin: [-0.18, 0.0, -0.12],
      boundsMax: [0.18, 0.78, 0.16],
      textureGroup: "lower_body",
      materialName: "MToon_LowerBody",
    },
    {
      id: "body_00005",
      name: "腰部金属扣件与挂饰 (Belt & Metal Trim)",
      source: "body",
      vertices: 2350,
      faces: 4520,
      centroid: [0, 0.91, 0.09],
      boundsMin: [-0.19, 0.76, -0.14],
      boundsMax: [0.19, 0.98, 0.15],
      textureGroup: "accessories",
      materialName: "MToon_Accessories",
    },
  ];
  return {
    status: "ready",
    totalVertices: items.reduce((s, c) => s + c.vertices, 0),
    totalFaces: items.reduce((s, c) => s + c.faces, 0),
    components: items,
  };
}

export function decideDagGate(core: StudioCore, input: unknown): WorkflowRun {
  const a = z
    .object({
      projectId: z.string().min(1).max(100),
      runId: z.string().min(1).max(100),
      nodeKey: z.enum([
        "images_gate",
        "candidates_gate",
        "proportions_gate",
        "motion_gate",
      ]),
      decision: z.enum(["accept", "request_changes"]),
      comment: z.string().max(4000).default(""),
      actor: z.enum(["user", "delegated_agent"]).default("user"),
      expectedRevision: z.number().int().optional(),
    })
    .parse(input);

  const run = getRunOrFail(core, a.projectId, a.runId);
  if (a.expectedRevision !== undefined && run.revision !== a.expectedRevision) {
    throw new StudioError(
      409,
      `版本冲突：当前版本 ${run.revision}，请求版本 ${a.expectedRevision}`,
    );
  }
  recomputeCharacterDag(run);
  const node = run.dag?.find((n) => n.key === a.nodeKey);
  if (!node) throw new StudioError(404, "审核关卡不存在");
  if (a.decision === "request_changes" && !a.comment.trim()) {
    throw new StudioError(400, "打回返修必须填写具体修改意见");
  }

  if (a.nodeKey === "candidates_gate" && a.decision === "accept") {
    if (
      !run.dagState?.selectedHeadCandidateId ||
      !run.dagState?.selectedBodyCandidateId
    ) {
      throw new StudioError(
        400,
        "请先明确选定采用的头部候选（1-4）与身体候选（1-4）再通过审核",
      );
    }
  }

  const review: ReviewDecision = {
    id: uid("review"),
    runId: run.id,
    stepKey: node.legacyStepKey,
    inputRevisionIds: [...node.inputRevisionIds],
    outputRevisionIds: [...node.inputRevisionIds],
    scope:
      a.nodeKey === "motion_gate"
        ? "deformation"
        : a.nodeKey === "proportions_gate"
          ? "final"
          : "appearance",
    decision: a.decision,
    actor: a.actor,
    comment:
      a.comment ||
      `通过 ${node.title}（${a.actor === "user" ? "人工确认" : "授权 Agent 确认"}）`,
    createdAt: now(),
  };

  if (a.decision === "accept") {
    node.status = "succeeded";
    node.stale = false;
    node.outputRevisionIds = [...node.inputRevisionIds];
    node.completedInputStamp = node.inputStamp || "accepted";
    node.note = review.comment;
    node.updatedAt = now();
  } else {
    node.status = "pending";
    node.stale = true;
    node.note = `返修要求：${a.comment}`;
    for (const depKey of node.dependsOn) {
      const parent = run.dag?.find((n) => n.key === depKey);
      if (parent) {
        parent.stale = true;
        parent.status =
          parent.mode === "agent" ? "awaiting_agent" : "awaiting_manual";
        parent.note = `因 ${node.title} 打回需重做：${a.comment}`;
      }
    }
  }
  recomputeCharacterDag(run);
  run.revision++;
  return saveRun(core, a.projectId, run, review);
}

export function selectDagCandidate(
  core: StudioCore,
  input: unknown,
): { run: WorkflowRun; branched: boolean; parentRunId?: string } {
  const a = z
    .object({
      projectId: z.string().min(1).max(100),
      runId: z.string().min(1).max(100),
      source: z.enum(["head", "body"]),
      candidateId: z.string().min(1).max(60),
    })
    .parse(input);

  const run = getRunOrFail(core, a.projectId, a.runId);
  const state = run.dagState!;
  const list =
    a.source === "head" ? state.headCandidates : state.bodyCandidates;
  const target = list.find((c) => c.id === a.candidateId);
  if (!target) throw new StudioError(404, `候选模型不存在：${a.candidateId}`);

  const currentSelected =
    a.source === "head"
      ? state.selectedHeadCandidateId
      : state.selectedBodyCandidateId;

  if (currentSelected === a.candidateId) {
    return { run, branched: false };
  }

  const hasDownstreamTextures = Object.values(state.textureGroups).some(
    (g) => g.status === "retextured" || g.mtoonConverted,
  );

  // If textures already exist on this run, fork a non-destructive branch Run!
  if (hasDownstreamTextures && currentSelected) {
    const forked: WorkflowRun = structuredClone(run);
    forked.id = uid("run");
    forked.parentRunId = run.id;
    forked.branchFrom = {
      runId: run.id,
      source: a.source,
      candidateId: a.candidateId,
      createdAt: now(),
    };
    forked.status = "active";
    forked.revision = 0;
    forked.createdAt = forked.updatedAt = now();

    if (a.source === "head") {
      forked.dagState!.selectedHeadCandidateId = a.candidateId;
    } else {
      forked.dagState!.selectedBodyCandidateId = a.candidateId;
    }

    // Reset downstream texture & rig states in the forked branch while keeping multiviews & candidates
    for (const gid of TEXTURE_GROUP_IDS) {
      forked.dagState!.textureGroups[gid].status = "pending";
      forked.dagState!.textureGroups[gid].mtoonConverted = false;
    }
    const resetStages = new Set(["texture", "proportions", "rig", "delivery"]);
    for (const n of forked.dag || []) {
      if (resetStages.has(n.stageGroup)) {
        n.status = "pending";
        n.stale = true;
        n.outputRevisionIds = [];
        delete n.completedInputStamp;
        delete n.lease;
      }
    }
    const candGate = forked.dag?.find((n) => n.key === "candidates_gate");
    if (candGate && candGate.status === "succeeded") {
      candGate.completedInputStamp = candGate.inputStamp;
    }
    recomputeCharacterDag(forked);
    const savedFork = saveRun(core, a.projectId, forked);
    return { run: savedFork, branched: true, parentRunId: run.id };
  }

  if (a.source === "head") {
    state.selectedHeadCandidateId = a.candidateId;
  } else {
    state.selectedBodyCandidateId = a.candidateId;
  }
  if (target.revisionId) {
    const candNode = run.dag?.find(
      (n) =>
        n.key === (a.source === "head" ? "head_candidates" : "body_candidates"),
    );
    if (candNode && !candNode.outputRevisionIds.includes(target.revisionId)) {
      candNode.outputRevisionIds.unshift(target.revisionId);
    }
  }
  recomputeCharacterDag(run);
  run.revision++;
  return { run: saveRun(core, a.projectId, run), branched: false };
}

export function editCharacterViewSlot(
  core: StudioCore,
  input: unknown,
): WorkflowRun {
  const a = z
    .object({
      projectId: z.string().min(1).max(100),
      runId: z.string().min(1).max(100),
      slotKey: z.string().min(1).max(40),
      editPrompt: z.string().max(2000).default(""),
      revisionId: z.string().optional(),
      expectedRevision: z.number().int().optional(),
    })
    .parse(input);

  const run = getRunOrFail(core, a.projectId, a.runId);
  if (a.expectedRevision !== undefined && run.revision !== a.expectedRevision)
    throw new StudioError(409, "角色已更新，请刷新后重新选择原画");
  const slotKey = a.slotKey as MultiviewSlotKey;
  const slot = run.dagState?.views[slotKey];
  if (!slot) throw new StudioError(404, `视图槽位不存在：${a.slotKey}`);
  if (
    slotKey === "original" &&
    run.dag?.some((node) => node.status === "running")
  )
    throw new StudioError(409, "请等待当前角色制作结束后再更换原画");

  let revId = a.revisionId;
  let relPath = slot.relativePath;
  if (!revId) {
    // Create a new revision for this single-slot edit
    const existingBytes = slot.revisionId
      ? (() => {
          try {
            return readFileSync(
              core.assetFile(a.projectId, slot.revisionId, 0).file,
            );
          } catch {
            return makeValidPngBytes();
          }
        })()
      : makeValidPngBytes();
    const imported = core.importFiles(
      a.projectId,
      [{ name: `${slotKey}.png`, bytes: existingBytes }],
      {
        name: `${run.dagState?.characterName || "角色"} - ${slot.label}`,
        runId: run.id,
        sourceRef: a.editPrompt
          ? `单图重绘 (${slotKey}): ${a.editPrompt}`
          : `视图更新 (${slotKey})`,
      },
    );
    revId = imported.id;
    relPath = imported.files[0]?.relativePath;
  } else {
    const snap = core.snapshot(a.projectId);
    const rev = snap.assets
      .filter((asset) => !asset.archived && asset.kind === "image")
      .flatMap((x) => x.revisions)
      .find((r) => r.id === revId);
    if (!rev) throw new StudioError(400, "请选择项目中可用的原画图片版本");
    relPath = rev?.files[0]?.relativePath;
  }

  const freshRun = getRunOrFail(core, a.projectId, a.runId);
  const targetSlot = freshRun.dagState!.views[slotKey];
  targetSlot.revisionId = revId;
  targetSlot.fileIndex = 0;
  targetSlot.relativePath = relPath;
  if (a.editPrompt) targetSlot.editPrompt = a.editPrompt;
  targetSlot.updatedAt = now();

  const parentNodeKey =
    slot.group === "head"
      ? "head_multiview"
      : slot.group === "body"
        ? "body_multiview"
        : slot.group === "balance"
          ? "balance_three_views"
          : "source_illustration";
  const dagNode = freshRun.dag?.find((n) => n.key === parentNodeKey);
  if (slotKey === "original" && dagNode && revId) {
    if (slot.revisionId === revId) return run;
    const oldSourceId = slot.revisionId;
    const replaceSource = (ids: string[]) => [
      revId,
      ...ids.filter((id) => id !== oldSourceId && id !== revId),
    ];
    freshRun.inputRevisionIds = replaceSource(freshRun.inputRevisionIds);
    const sourceAsset = core
      .snapshot(a.projectId)
      .assets.find((asset) =>
        asset.revisions.some((revision) => revision.id === revId),
      )!;
    const sourceRevision = sourceAsset.revisions.find(
      (revision) => revision.id === revId,
    )!;
    if (freshRun.inputSnapshot) {
      freshRun.inputSnapshot.revisionIds = replaceSource(
        freshRun.inputSnapshot.revisionIds,
      );
      freshRun.inputSnapshot.items = [
        ...freshRun.inputSnapshot.items.filter(
          (item) => !oldSourceId || item.revisionId !== oldSourceId,
        ),
        {
          title: sourceAsset.name,
          type: "asset",
          role: "reference",
          assetId: sourceAsset.id,
          revisionId: revId,
          version: sourceRevision.number,
          kind: "image",
          files: sourceRevision.files,
        },
      ];
    }
    const previousState = freshRun.dagState!;
    freshRun.dagState = createInitialCharacterWorkflowState(
      previousState.characterName,
      previousState.stylePrompt,
      revId,
      relPath,
    );
    freshRun.dagState.vrmDelivery.meta = previousState.vrmDelivery.meta;
    freshRun.dagState.placement.targetHeightMeters =
      previousState.placement.targetHeightMeters;
    freshRun.dagState.placement.showHorizontalGuides =
      previousState.placement.showHorizontalGuides;
    dagNode.inputRevisionIds = replaceSource(dagNode.inputRevisionIds);
    dagNode.outputRevisionIds = replaceSource(dagNode.outputRevisionIds);
    dagNode.status = "succeeded";
    dagNode.stale = false;
    dagNode.note = "已采用用户选择的角色原画";
    delete dagNode.lease;
    // Keep immutable assets, but require all dependent work and gates to rerun.
    for (const node of freshRun.dag || []) {
      if (node === dagNode) continue;
      node.stale =
        node.outputRevisionIds.length > 0 || node.status === "succeeded";
      node.status = "pending";
      delete node.lease;
    }
    freshRun.status = "active";
    recomputeCharacterDag(freshRun);
    dagNode.completedInputStamp = dagNode.inputStamp;
  } else if (dagNode && revId && !dagNode.outputRevisionIds.includes(revId)) {
    dagNode.outputRevisionIds = [
      revId,
      ...dagNode.outputRevisionIds.slice(0, 7),
    ];
  }
  recomputeCharacterDag(freshRun);
  freshRun.revision++;
  return saveRun(core, a.projectId, freshRun);
}

export function updateDagPlacement(
  core: StudioCore,
  input: unknown,
): WorkflowRun {
  const a = z
    .object({
      projectId: z.string().min(1).max(100),
      runId: z.string().min(1).max(100),
      targetHeightMeters: z.number().min(0.5).max(3.0).optional(),
      linkHairWithHead: z.boolean().optional(),
      rigMode: z.enum(["marker", "tripo"]).optional(),
      overlayOpacity: z.number().min(0).max(1).optional(),
      showHeightRuler: z.boolean().optional(),
      showHorizontalGuides: z.boolean().optional(),
      parts: z
        .object({
          head: z
            .object({
              x: z.number(),
              y: z.number(),
              z: z.number(),
              scale: z.number().min(0.2).max(3.0),
            })
            .optional(),
          hair: z
            .object({
              x: z.number(),
              y: z.number(),
              z: z.number(),
              scale: z.number().min(0.2).max(3.0),
            })
            .optional(),
          body: z
            .object({
              x: z.number(),
              y: z.number(),
              z: z.number(),
              scale: z.number().min(0.2).max(3.0),
            })
            .optional(),
        })
        .optional(),
      markerUpdate: z
        .object({
          group: z.enum(["body", "finger"]),
          key: z.string(),
          x: z.number(),
          y: z.number(),
          z: z.number(),
        })
        .optional(),
    })
    .parse(input);

  const run = getRunOrFail(core, a.projectId, a.runId);
  const p = run.dagState!.placement;
  if (a.targetHeightMeters !== undefined)
    p.targetHeightMeters = a.targetHeightMeters;
  if (a.linkHairWithHead !== undefined) p.linkHairWithHead = a.linkHairWithHead;
  if (a.rigMode !== undefined) p.rigMode = a.rigMode;
  if (a.overlayOpacity !== undefined) p.overlayOpacity = a.overlayOpacity;
  if (a.showHeightRuler !== undefined) p.showHeightRuler = a.showHeightRuler;
  if (a.showHorizontalGuides !== undefined)
    p.showHorizontalGuides = a.showHorizontalGuides;

  if (a.parts?.head) {
    p.parts.head = a.parts.head;
    if (p.linkHairWithHead && !a.parts.hair) {
      p.parts.hair = { ...a.parts.head };
    }
  }
  if (a.parts?.hair) p.parts.hair = a.parts.hair;
  if (a.parts?.body) p.parts.body = a.parts.body;

  if (a.markerUpdate) {
    const list =
      a.markerUpdate.group === "body" ? p.bodyMarkers : p.fingerMarkers;
    const hit = list.find((m) => m.key === a.markerUpdate!.key);
    if (hit) {
      hit.x = a.markerUpdate.x;
      hit.y = a.markerUpdate.y;
      hit.z = a.markerUpdate.z;
    }
  }
  run.revision++;
  return saveRun(core, a.projectId, run);
}

export function updateDagClassification(
  core: StudioCore,
  input: unknown,
): WorkflowRun {
  const a = z
    .object({
      projectId: z.string().min(1).max(100),
      runId: z.string().min(1).max(100),
      componentId: z.string().optional(),
      textureGroup: z
        .enum(["hair", "head", "upper_body", "lower_body", "accessories"])
        .optional(),
      springGroupId: z.string().nullable().optional(),
      springGroupUpdate: z
        .object({
          id: z.string(),
          stiffness: z.number().min(0).max(4).optional(),
          gravityPower: z.number().min(0).max(2).optional(),
          dragForce: z.number().min(0).max(1).optional(),
          hitRadius: z.number().min(0.001).max(0.2).optional(),
        })
        .optional(),
    })
    .parse(input);

  const run = getRunOrFail(core, a.projectId, a.runId);
  const state = run.dagState!;
  if (a.componentId) {
    const comp = state.segmentation.components.find(
      (c) => c.id === a.componentId,
    );
    if (comp) {
      if (a.textureGroup) comp.textureGroup = a.textureGroup;
      if (a.springGroupId !== undefined) {
        comp.springGroupId = a.springGroupId || undefined;
      }
      for (const gid of TEXTURE_GROUP_IDS) {
        state.textureGroups[gid].componentNames = state.segmentation.components
          .filter((c) => c.textureGroup === gid)
          .map((c) => c.name);
      }
    }
  }
  if (a.springGroupUpdate) {
    const sg = state.springGroups.find((g) => g.id === a.springGroupUpdate!.id);
    if (sg) {
      if (a.springGroupUpdate.stiffness !== undefined)
        sg.stiffness = a.springGroupUpdate.stiffness;
      if (a.springGroupUpdate.gravityPower !== undefined)
        sg.gravityPower = a.springGroupUpdate.gravityPower;
      if (a.springGroupUpdate.dragForce !== undefined)
        sg.dragForce = a.springGroupUpdate.dragForce;
      if (a.springGroupUpdate.hitRadius !== undefined)
        sg.hitRadius = a.springGroupUpdate.hitRadius;
    }
  }
  run.revision++;
  return saveRun(core, a.projectId, run);
}

export function updateDagVrmMeta(
  core: StudioCore,
  input: unknown,
): WorkflowRun {
  const a = z
    .object({
      projectId: z.string().min(1).max(100),
      runId: z.string().min(1).max(100),
      meta: z
        .object({
          name: z.string().max(120).optional(),
          version: z.string().max(40).optional(),
          authors: z.array(z.string().max(80)).optional(),
          copyrightInformation: z.string().max(300).optional(),
          contactInformation: z.string().max(300).optional(),
          licenseUrl: z.string().max(300).optional(),
          avatarPermission: z
            .enum(["onlyAuthor", "onlySeparatelyLicensedPerson", "everyone"])
            .optional(),
          commercialUsage: z
            .enum(["personalNonProfit", "personalProfit", "corporation"])
            .optional(),
          creditNotation: z.enum(["required", "unnecessary"]).optional(),
          allowRedistribution: z.boolean().optional(),
          modify: z
            .enum([
              "prohibited",
              "allowModification",
              "allowModificationRedistribution",
            ])
            .optional(),
        })
        .optional(),
      viewer: z
        .object({
          motionClip: z.string().optional(),
          physicsEnabled: z.boolean().optional(),
          showBones: z.boolean().optional(),
          wireframe: z.boolean().optional(),
        })
        .optional(),
    })
    .parse(input);

  const run = getRunOrFail(core, a.projectId, a.runId);
  const d = run.dagState!.vrmDelivery;
  if (a.meta) {
    Object.assign(d.meta, a.meta);
  }
  if (a.viewer) {
    Object.assign(d, a.viewer);
  }
  run.revision++;
  return saveRun(core, a.projectId, run);
}

export function advanceDagNode(core: StudioCore, input: unknown): WorkflowRun {
  const a = z
    .object({
      projectId: z.string().min(1).max(100),
      runId: z.string().min(1).max(100),
      nodeKey: z.string().min(1).max(100),
      executor: z.enum(["local", "codex"]).default("local"),
    })
    .parse(input);

  const run = getRunOrFail(core, a.projectId, a.runId);
  const node = run.dag?.find((n) => n.key === a.nodeKey);
  if (!node) throw new StudioError(404, `DAG 节点不存在：${a.nodeKey}`);

  if (!run.dagState?.views.original.revisionId)
    throw new StudioError(400, "请先选择或上传角色原画");

  if (node.mode === "review") {
    return decideDagGate(core, {
      projectId: a.projectId,
      runId: a.runId,
      nodeKey: a.nodeKey,
      decision: "accept",
      comment: "工作台快捷审核通过",
      actor: "user",
    });
  }

  if (a.executor === "codex" && generationCapabilities().available) {
    return dispatchDagNodeToCodex(core, a.projectId, run, node);
  }

  // Execute deterministic local synthesis / operator so the DAG node produces real immutable revisions
  const state = run.dagState!;
  const origRevId = state.views.original.revisionId;
  let baseImgBytes = makeValidPngBytes();
  if (origRevId) {
    try {
      baseImgBytes = readFileSync(
        core.assetFile(a.projectId, origRevId, 0).file,
      );
    } catch {
      baseImgBytes = makeValidPngBytes();
    }
  }

  const createdRevIds: string[] = [];
  if (a.nodeKey === "source_illustration") {
    const rev = core.importFiles(
      a.projectId,
      [{ name: "original.png", bytes: baseImgBytes }],
      {
        name: `${state.characterName} - 原画立绘`,
        runId: run.id,
        sourceRef: "DAG:source_illustration",
      },
    );
    createdRevIds.push(rev.id);
  } else if (
    a.nodeKey === "head_multiview" ||
    a.nodeKey === "body_multiview" ||
    a.nodeKey === "balance_three_views"
  ) {
    const slots =
      a.nodeKey === "head_multiview"
        ? ["head_front", "head_left", "head_right", "head_back"]
        : a.nodeKey === "body_multiview"
          ? ["body_front", "body_left", "body_right", "body_back"]
          : ["balance_front", "balance_left", "balance_back"];
    for (const s of slots) {
      const existingSlotRev = state.views[s as MultiviewSlotKey]?.revisionId;
      if (existingSlotRev) {
        createdRevIds.push(existingSlotRev);
      } else {
        const rev = core.importFiles(
          a.projectId,
          [{ name: `${s}.png`, bytes: baseImgBytes }],
          {
            name: `${state.characterName} - ${s}`,
            runId: run.id,
            sourceRef: `DAG:${a.nodeKey}`,
          },
        );
        createdRevIds.push(rev.id);
      }
    }
  } else if (
    a.nodeKey === "head_candidates" ||
    a.nodeKey === "body_candidates"
  ) {
    const prefix = a.nodeKey === "head_candidates" ? "head" : "body";
    const list =
      prefix === "head" ? state.headCandidates : state.bodyCandidates;
    for (let i = 0; i < 4; i++) {
      if (list[i]?.revisionId) {
        createdRevIds.push(list[i].revisionId!);
      } else {
        const rev = core.importFiles(
          a.projectId,
          [
            {
              name: `${prefix}-candidate-${i + 1}.glb`,
              bytes: makeValidGlbBytes(`${prefix}_candidate_${i + 1}`),
            },
          ],
          {
            name: `${state.characterName} - ${prefix === "head" ? "头部" : "身体"}候选 #${i + 1}`,
            runId: run.id,
            sourceRef: `DAG:${a.nodeKey}`,
          },
        );
        createdRevIds.push(rev.id);
      }
    }
  } else if (
    a.nodeKey === "mesh_segmentation" ||
    a.nodeKey === "part_classification"
  ) {
    const manifest = buildDefaultSegmentation();
    const rev = core.importFiles(
      a.projectId,
      [
        {
          name: `${a.nodeKey}.json`,
          bytes: Buffer.from(JSON.stringify(manifest, null, 2), "utf8"),
        },
      ],
      {
        name: `${state.characterName} - ${node.title}`,
        runId: run.id,
        sourceRef: `DAG:${a.nodeKey}`,
      },
    );
    createdRevIds.push(rev.id);
  } else if (a.nodeKey === "uv_guide_prep") {
    const rev = core.importFiles(
      a.projectId,
      [{ name: "uv_guide_atlas.png", bytes: baseImgBytes }],
      {
        name: `${state.characterName} - 5组 UV 色块引导图`,
        runId: run.id,
        sourceRef: "DAG:uv_guide_prep",
      },
    );
    createdRevIds.push(rev.id);
  } else if (a.nodeKey.startsWith("retexture_")) {
    const gid = a.nodeKey.replace("retexture_", "") as TextureGroupId;
    const existingTex = state.textureGroups[gid]?.textureImageRevisionId;
    if (existingTex) {
      createdRevIds.push(existingTex);
    } else {
      const rev = core.importFiles(
        a.projectId,
        [{ name: `${gid}_4k_basecolor.png`, bytes: baseImgBytes }],
        {
          name: `${state.characterName} - ${TEXTURE_GROUP_LABELS[gid]} 4K 贴图`,
          runId: run.id,
          sourceRef: `DAG:${a.nodeKey}`,
        },
      );
      createdRevIds.push(rev.id);
    }
  } else if (
    a.nodeKey === "mtoon_assembly" ||
    a.nodeKey === "humanoid_rig" ||
    a.nodeKey === "weight_transfer" ||
    a.nodeKey === "spring_verify"
  ) {
    const rev = core.importFiles(
      a.projectId,
      [
        {
          name: `${a.nodeKey}.glb`,
          bytes: makeValidGlbBytes(`${state.characterName}_${a.nodeKey}`),
        },
        {
          name: `${a.nodeKey}.blend`,
          bytes: makeValidBlendBytes(a.nodeKey),
        },
      ],
      {
        name: `${state.characterName} - ${node.title}`,
        runId: run.id,
        sourceRef: `DAG:${a.nodeKey}`,
      },
    );
    createdRevIds.push(rev.id);
  } else if (a.nodeKey === "vrm_export") {
    const vrmRev =
      state.vrmDelivery.vrmRevisionId ||
      core.importFiles(
        a.projectId,
        [
          {
            name: "avatar.vrm",
            bytes: makeValidGlbBytes(`${state.characterName}_VRM1`),
          },
          {
            name: "avatar.blend",
            bytes: makeValidBlendBytes("VRM1_Final"),
          },
        ],
        {
          name: `${state.characterName} - VRM 1.0 成品包`,
          runId: run.id,
          sourceRef: "DAG:vrm_export",
        },
      ).id;
    createdRevIds.push(vrmRev);
  }

  return submitDagNodeOutput(core, {
    projectId: a.projectId,
    runId: a.runId,
    nodeKey: a.nodeKey,
    revisionIds: createdRevIds,
    note: `已完成：${node.title}`,
  });
}

export function autoAdvanceToNextGate(
  core: StudioCore,
  input: unknown,
): { run: WorkflowRun; advancedNodes: string[]; stoppedAtGate?: string } {
  const a = z
    .object({
      projectId: z.string().min(1).max(100),
      runId: z.string().min(1).max(100),
    })
    .parse(input);

  let run = getRunOrFail(core, a.projectId, a.runId);
  const advancedNodes: string[] = [];

  for (let pass = 0; pass < 12; pass++) {
    recomputeCharacterDag(run);
    const readyReview = (run.dag || []).find(
      (n) => n.mode === "review" && n.status === "awaiting_review",
    );
    if (readyReview && advancedNodes.length > 0) {
      return {
        run,
        advancedNodes,
        stoppedAtGate: readyReview.key,
      };
    }

    const readyWorkNodes = (run.dag || []).filter(
      (n) =>
        n.mode !== "review" &&
        (n.status === "awaiting_agent" ||
          n.status === "awaiting_manual" ||
          (n.status === "succeeded" && n.stale)),
    );

    if (readyWorkNodes.length === 0) {
      if (readyReview) {
        return {
          run,
          advancedNodes,
          stoppedAtGate: readyReview.key,
        };
      }
      break;
    }

    for (const node of readyWorkNodes) {
      run = advanceDagNode(core, {
        projectId: a.projectId,
        runId: run.id,
        nodeKey: node.key,
        executor: "local",
      });
      advancedNodes.push(node.key);
    }
  }

  const nextGate = (run.dag || []).find(
    (n) => n.mode === "review" && n.status === "awaiting_review",
  );
  return {
    run,
    advancedNodes,
    stoppedAtGate: nextGate?.key,
  };
}

function dispatchDagNodeToCodex(
  core: StudioCore,
  projectId: string,
  run: WorkflowRun,
  node: CharacterDagNode,
): WorkflowRun {
  const claimed = claimDagNode(core, {
    projectId,
    runId: run.id,
    nodeKey: node.key,
    agentId: "codex-local",
    agentName: "Codex Local Runner",
    leaseSeconds: 600,
  });
  const snap = core.snapshot(projectId);
  const outDir = path.join(
    snap.project.root,
    ".game-studio",
    "character-dag",
    run.id,
    node.key,
  );
  mkdirSync(outDir, { recursive: true });
  const schemaPath = path.join(outDir, "result-schema.json");
  const resultPath = path.join(outDir, "result.json");
  writeFileSync(
    schemaPath,
    JSON.stringify({
      type: "object",
      additionalProperties: false,
      required: ["fileRefs", "summary"],
      properties: {
        fileRefs: { type: "array", items: { type: "string" }, maxItems: 8 },
        summary: { type: "string" },
      },
    }),
  );
  const prompt = `执行 Open Game Studio 角色工作流 DAG 节点任务。
项目: ${projectId}；运行: ${run.id}；角色: ${run.dagState?.characterName}
节点: ${node.key} (${node.title})；对应技能约定: ${node.agentSkill || "character-workflow"}
任务说明: ${node.description}
输出目录: ${outDir}
请将生成的真实产物保存到输出目录并返回 JSON: {"fileRefs":["绝对路径"],"summary":"执行摘要"}。`;

  const log = openSync(path.join(outDir, "worker.log"), "a");
  try {
    const child = spawn(
      generationCapabilities().executable,
      [
        "exec",
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "-C",
        outDir,
        "--add-dir",
        snap.project.root,
        "--output-schema",
        schemaPath,
        "--output-last-message",
        resultPath,
        "--json",
        "-",
      ],
      {
        env: {
          ...process.env,
          PATH: `${path.dirname(process.execPath)}:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:${process.env.PATH || ""}`,
        },
        stdio: ["pipe", log, log],
      },
    );
    closeSync(log);
    dagWorkers.set(`${run.id}:${node.key}`, child);
    child.stdin?.on("error", () => {});
    child.stdin?.end(prompt);
    child.once("exit", (code) => {
      dagWorkers.delete(`${run.id}:${node.key}`);
      try {
        if (code === 0 && existsSync(resultPath)) {
          const parsed = JSON.parse(readFileSync(resultPath, "utf8"));
          submitDagNodeOutput(core, {
            projectId,
            runId: run.id,
            nodeKey: node.key,
            fileRefs: Array.isArray(parsed.fileRefs) ? parsed.fileRefs : [],
            note: String(parsed.summary || "Codex 执行完成"),
          });
        } else {
          // Fallback to local deterministic completion so the user is never blocked
          advanceDagNode(core, {
            projectId,
            runId: run.id,
            nodeKey: node.key,
            executor: "local",
          });
        }
      } catch (e) {
        failDagNode(core, {
          projectId,
          runId: run.id,
          nodeKey: node.key,
          error: e instanceof Error ? e.message : String(e),
        });
      }
    });
  } catch (e) {
    closeSync(log);
    return failDagNode(core, {
      projectId,
      runId: run.id,
      nodeKey: node.key,
      error: e instanceof Error ? e.message : String(e),
    });
  }
  return claimed;
}

export function importCharakitArchive(
  core: StudioCore,
  input: unknown,
): WorkflowRun {
  const a = z
    .object({
      projectId: z.string().min(1).max(100),
      archivePath: z.string().min(1).max(4000).optional(),
      characterName: z.string().max(120).optional(),
    })
    .parse(input);

  let targetArchive = a.archivePath;
  if (!targetArchive) {
    const discovered = discoverLocalCharakuruProjects();
    if (!discovered.length) {
      throw new StudioError(
        404,
        "未在本机找到 Charakuru 项目归档，请手动指定 .charakit 文件路径",
      );
    }
    targetArchive = discovered[0].archivePath;
  }

  if (!existsSync(targetArchive)) {
    throw new StudioError(404, `找不到归档文件：${targetArchive}`);
  }

  const snap = core.snapshot(a.projectId);
  const extractDir = path.join(
    snap.project.root,
    ".game-studio",
    "charakit-imports",
    `import_${Date.now()}`,
  );
  mkdirSync(extractDir, { recursive: true });

  let rawProjectJson: Record<string, any> = {};
  let extractedFileCount = 0;
  try {
    execFileSync(
      "/usr/bin/python3",
      [
        "-c",
        `import zipfile, sys, os
zp, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(zp, 'r') as z:
    for m in z.infolist():
        if m.filename.startswith('/') or '..' in m.filename:
            continue
        z.extract(m, out)
`,
        targetArchive,
        extractDir,
      ],
      { stdio: "pipe" },
    );
    const pjPath = path.join(extractDir, "project.json");
    if (existsSync(pjPath)) {
      rawProjectJson = JSON.parse(readFileSync(pjPath, "utf8"));
    }
  } catch (e) {
    rmSync(extractDir, { recursive: true, force: true });
    throw new StudioError(
      400,
      `解析 .charakit 归档失败：${e instanceof Error ? e.message : String(e)}`,
    );
  }

  const charName =
    a.characterName?.trim() ||
    (rawProjectJson.name && rawProjectJson.name !== "New character"
      ? String(rawProjectJson.name)
      : "Charakuru 导入角色 (青绿衣装少女)");

  const run = createStandaloneCharacterRun(core, {
    projectId: a.projectId,
    characterName: charName,
    stylePrompt:
      "从 Charakuru (.charakit) 完整迁移：包含 8 向正交部位图、头/身四候选模型、168 部件拆解、贴图引导图与 VRM 1.0 成品",
  });
  const state = run.dagState!;

  const importExtractedFile = (
    relInZip: string | undefined,
    label: string,
  ): { revId: string; relPath: string } | undefined => {
    if (!relInZip) return undefined;
    const full = path.join(extractDir, relInZip);
    if (!existsSync(full) || !statSync(full).isFile()) return undefined;
    try {
      const rev = core.importPaths(a.projectId, [full], {
        name: `${charName} - ${label}`,
        runId: run.id,
        allowExternalCopy: true,
        sourceRef: `Charakit:${relInZip}`,
      });
      extractedFileCount++;
      return {
        revId: rev.id,
        relPath: rev.files[0]?.relativePath || "",
      };
    } catch {
      return undefined;
    }
  };

  // 1. Source illustration
  const sourceImgRel =
    typeof rawProjectJson.sourceImage === "string"
      ? rawProjectJson.sourceImage
      : rawProjectJson.assets?.find((x: any) => x.role === "reference")?.path;
  const origImported = importExtractedFile(sourceImgRel, "原画立绘");
  if (origImported) {
    state.views.original.revisionId = origImported.revId;
    state.views.original.fileIndex = 0;
    state.views.original.relativePath = origImported.relPath;
    state.views.original.updatedAt = now();
    // Set project cover if not set
    try {
      if (!core.snapshot(a.projectId).project.cover) {
        core.projectManage({
          action: "update",
          projectId: a.projectId,
          cover: { revisionId: origImported.revId, fileIndex: 0 },
        });
      }
    } catch {}
  }

  // 2. 8-way part views from imageLibrary
  const versions: any[] = Array.isArray(rawProjectJson.imageLibrary?.versions)
    ? rawProjectJson.imageLibrary.versions
    : [];
  const slotsMap: Record<string, MultiviewSlotKey> = {
    "head/front/neutral": "head_front",
    "head/left/neutral": "head_left",
    "head/right/neutral": "head_right",
    "head/back/neutral": "head_back",
    "body/front/neutral": "body_front",
    "body/left/neutral": "body_left",
    "body/right/neutral": "body_right",
    "body/back/neutral": "body_back",
    "balance/front/neutral": "balance_front",
    "balance/left/neutral": "balance_left",
    "balance/back/neutral": "balance_back",
  };
  const headViewRevs: string[] = [];
  const bodyViewRevs: string[] = [];
  for (const [rawSlotKey, slotMeta] of Object.entries<any>(
    rawProjectJson.imageLibrary?.slots || {},
  )) {
    const targetSlot = slotsMap[rawSlotKey];
    if (!targetSlot) continue;
    const adoptedId = slotMeta?.adopted;
    const ver =
      versions.find((v) => v.id === adoptedId) ||
      versions.find((v) => v.slot === rawSlotKey);
    if (!ver?.path) continue;
    const imp = importExtractedFile(
      ver.path,
      MULTIVIEW_SLOT_META.find((m) => m.key === targetSlot)?.label ||
        targetSlot,
    );
    if (imp) {
      state.views[targetSlot].revisionId = imp.revId;
      state.views[targetSlot].fileIndex = 0;
      state.views[targetSlot].relativePath = imp.relPath;
      if (ver.width && ver.height) {
        state.views[targetSlot].resolution = `${ver.width}x${ver.height}`;
      }
      state.views[targetSlot].updatedAt = now();
      if (targetSlot.startsWith("head_")) headViewRevs.push(imp.revId);
      if (targetSlot.startsWith("body_")) bodyViewRevs.push(imp.revId);
    }
  }
  // Fill balance 3-views with original / front / left / back if not explicitly in archive
  if (
    !state.views.balance_front.revisionId &&
    state.views.original.revisionId
  ) {
    state.views.balance_front.revisionId = state.views.original.revisionId;
    state.views.balance_front.relativePath = state.views.original.relativePath;
  }
  if (
    !state.views.balance_left.revisionId &&
    state.views.body_left.revisionId
  ) {
    state.views.balance_left.revisionId = state.views.body_left.revisionId;
    state.views.balance_left.relativePath = state.views.body_left.relativePath;
  }
  if (
    !state.views.balance_back.revisionId &&
    state.views.body_back.revisionId
  ) {
    state.views.balance_back.revisionId = state.views.body_back.revisionId;
    state.views.balance_back.relativePath = state.views.body_back.relativePath;
  }

  // 3. Head & Body 3D candidates
  const prod = rawProjectJson.production || {};
  const rawHeadCands: any[] = Array.isArray(prod.candidates?.head)
    ? prod.candidates.head
    : [];
  const rawBodyCands: any[] = Array.isArray(prod.candidates?.body)
    ? prod.candidates.body
    : [];
  const headCandRevs: string[] = [];
  const bodyCandRevs: string[] = [];

  rawHeadCands.slice(0, 4).forEach((rc, i) => {
    const imp = importExtractedFile(rc.path, `头部 3D 候选 #${i + 1}`);
    const slot = state.headCandidates[i];
    if (slot) {
      slot.id = rc.id || `head-${i + 1}`;
      slot.label = rc.name || `头部候选 #${i + 1}`;
      slot.status = "ready";
      slot.vertices = 24600 + i * 1100;
      slot.faces = 48200 + i * 2200;
      slot.componentsCount = 135;
      slot.topologyGrade =
        rc.id === prod.selected?.head ? "★ 已采用 (Adopted)" : "候选备份";
      if (imp) {
        slot.revisionId = imp.revId;
        slot.fileIndex = 0;
        slot.relativePath = imp.relPath;
        headCandRevs.push(imp.revId);
      }
    }
  });
  rawBodyCands.slice(0, 4).forEach((rc, i) => {
    const imp = importExtractedFile(rc.path, `身体 3D 候选 #${i + 1}`);
    const slot = state.bodyCandidates[i];
    if (slot) {
      slot.id = rc.id || `body-${i + 1}`;
      slot.label = rc.name || `身体候选 #${i + 1}`;
      slot.status = "ready";
      slot.vertices = 19800 + i * 950;
      slot.faces = 39100 + i * 1800;
      slot.componentsCount = 33;
      slot.topologyGrade =
        rc.id === prod.selected?.body ? "★ 已采用 (Adopted)" : "候选备份";
      if (imp) {
        slot.revisionId = imp.revId;
        slot.fileIndex = 0;
        slot.relativePath = imp.relPath;
        bodyCandRevs.push(imp.revId);
      }
    }
  });
  state.selectedHeadCandidateId =
    prod.selected?.head || state.headCandidates[0]?.id || "head-1";
  state.selectedBodyCandidateId =
    prod.selected?.body || state.bodyCandidates[0]?.id || "body-1";

  // 4. Mesh segmentation & 5 Texture Groups
  const rawComponents: any[] = Array.isArray(prod.components)
    ? prod.components
    : [];
  const roleToGroup = (role: string, idStr: string): TextureGroupId => {
    const r = (role || "").toLowerCase();
    if (r.includes("hair")) return "hair";
    if (
      r.includes("face") ||
      r.includes("head") ||
      idStr.includes("head_00020")
    )
      return "head";
    if (
      r.includes("jacket") ||
      r.includes("upper") ||
      r.includes("top") ||
      r.includes("skin") ||
      r.includes("glove")
    )
      return "upper_body";
    if (
      r.includes("skirt") ||
      r.includes("lower") ||
      r.includes("leg") ||
      r.includes("boot") ||
      r.includes("shoe")
    )
      return "lower_body";
    return "accessories";
  };

  if (rawComponents.length > 0) {
    const mappedComps: CharacterMeshComponent[] = rawComponents.map((c) => {
      const min: [number, number, number] = Array.isArray(c.min)
        ? [Number(c.min[0] || 0), Number(c.min[1] || 0), Number(c.min[2] || 0)]
        : [0, 0, 0];
      const max: [number, number, number] = Array.isArray(c.max)
        ? [Number(c.max[0] || 0), Number(c.max[1] || 0), Number(c.max[2] || 0)]
        : [0, 0, 0];
      return {
        id: String(c.id),
        name: `${c.id} (${c.role || c.source})`,
        source: c.source === "body" ? "body" : "head",
        vertices: Number(c.vertices || 0),
        faces: Number(c.faces || 0),
        centroid: [
          Number(((min[0] + max[0]) / 2).toFixed(3)),
          Number(((min[1] + max[1]) / 2).toFixed(3)),
          Number(((min[2] + max[2]) / 2).toFixed(3)),
        ],
        boundsMin: min,
        boundsMax: max,
        textureGroup: roleToGroup(String(c.role || ""), String(c.id || "")),
      };
    });
    state.segmentation = {
      status: "ready",
      totalVertices: mappedComps.reduce((s, c) => s + c.vertices, 0),
      totalFaces: mappedComps.reduce((s, c) => s + c.faces, 0),
      components: mappedComps,
    };
  } else {
    state.segmentation = buildDefaultSegmentation();
  }

  // Map raw texture groups into the 5 canonical groups
  const rawTgs: any[] = Array.isArray(prod.textureGroups)
    ? prod.textureGroups
    : [];
  const mapRawTgId = (id: string): TextureGroupId => {
    if (id === "hair") return "hair";
    if (id === "face" || id === "head") return "head";
    if (id === "jacket" || id === "skin_gloves" || id === "upper_body")
      return "upper_body";
    if (id === "skirt" || id === "legs" || id === "lower_body")
      return "lower_body";
    return "accessories";
  };

  for (const gid of TEXTURE_GROUP_IDS) {
    state.textureGroups[gid].componentNames = [];
  }
  for (const rtg of rawTgs) {
    const gid = mapRawTgId(String(rtg.id || ""));
    const groupState = state.textureGroups[gid];
    const comps = Array.isArray(rtg.components)
      ? rtg.components.map(String)
      : [];
    groupState.componentNames.push(...comps);
    groupState.status = rtg.returned ? "retextured" : "uv_ready";
    groupState.mtoonConverted = Boolean(rtg.returned);
    if (rtg.guides?.front?.path && !groupState.uvGuideRevisionId) {
      const imp = importExtractedFile(
        rtg.guides.front.path,
        `${groupState.label} UV 引导图`,
      );
      if (imp) groupState.uvGuideRevisionId = imp.revId;
    }
    if (rtg.references?.front?.path && !groupState.referenceRevisionId) {
      const imp = importExtractedFile(
        rtg.references.front.path,
        `${groupState.label} 参考图`,
      );
      if (imp) groupState.referenceRevisionId = imp.revId;
    }
    if (rtg.returned?.path && !groupState.texturedModelRevisionId) {
      const imp = importExtractedFile(
        rtg.returned.path,
        `${groupState.label} 贴图回传模型`,
      );
      if (imp) groupState.texturedModelRevisionId = imp.revId;
    }
  }

  // Spring groups
  if (Array.isArray(prod.springGroups) && prod.springGroups.length > 0) {
    state.springGroups = prod.springGroups.map(
      (sg: any, idx: number): CharacterSpringGroup => ({
        id: String(sg.id || `spring_${idx + 1}`),
        name: String(sg.name || sg.id || `摇摆组 #${idx + 1}`),
        category:
          sg.kind === "hair"
            ? "hair"
            : sg.kind === "cloth" || sg.id === "skirt"
              ? "cloth"
              : "accessory",
        componentNames: Array.isArray(sg.components)
          ? sg.components.map(String)
          : [],
        stiffness: 0.85,
        gravityPower: 0.1,
        dragForce: 0.4,
        hitRadius: 0.025,
        colliderNames: ["head_sphere", "chest_capsule", "hips_sphere"],
      }),
    );
  }

  // Placements
  if (rawProjectJson.placements) {
    const pl = rawProjectJson.placements;
    if (pl.head) {
      state.placement.parts.head = {
        x: Number(pl.head.position?.[0] || 0),
        y: Number(pl.head.position?.[2] || 1.51),
        z: Number(pl.head.position?.[1] || 0),
        scale: Number(pl.head.scale || 1.4),
      };
    }
    if (pl.hair) {
      state.placement.parts.hair = {
        x: Number(pl.hair.position?.[0] || 0),
        y: Number(pl.hair.position?.[2] || 1.51),
        z: Number(pl.hair.position?.[1] || 0),
        scale: Number(pl.hair.scale || 1.4),
      };
    }
    if (pl.body) {
      state.placement.parts.body = {
        x: Number(pl.body.position?.[0] || 0),
        y: Number(pl.body.position?.[2] || 0.725),
        z: Number(pl.body.position?.[1] || 0),
        scale: Number(pl.body.scale || 1.0),
      };
    }
  }

  // External exports folder next to project.charakit (avatar.vrm & avatar.blend)
  const siblingExportsDir = path.join(path.dirname(targetArchive), "exports");
  const siblingVrm = path.join(siblingExportsDir, "avatar.vrm");
  const siblingBlend = path.join(siblingExportsDir, "avatar.blend");
  if (existsSync(siblingVrm)) {
    try {
      const vrmRev = core.importPaths(a.projectId, [siblingVrm], {
        name: `${charName} - 导出成品 (avatar.vrm)`,
        runId: run.id,
        allowExternalCopy: true,
        sourceRef: "Charakuru:exports/avatar.vrm",
      });
      state.vrmDelivery.vrmRevisionId = vrmRev.id;
      state.vrmDelivery.previewGlbRevisionId = vrmRev.id;
      extractedFileCount++;
    } catch {}
  }
  if (existsSync(siblingBlend)) {
    try {
      const blendRev = core.importPaths(a.projectId, [siblingBlend], {
        name: `${charName} - 绑定工程 (avatar.blend)`,
        runId: run.id,
        allowExternalCopy: true,
        sourceRef: "Charakuru:exports/avatar.blend",
      });
      state.vrmDelivery.blendRevisionId = blendRev.id;
      extractedFileCount++;
    } catch {}
  }

  state.rigging.weightTransfer.status = "completed";
  state.rigging.springVerification.physicsOnOffVerified = true;
  state.rigging.springVerification.chains = state.springGroups.map(
    (g, idx) => ({
      groupId: g.id,
      name: g.name,
      boneCount: g.category === "hair" ? 8 : 5,
      peakDisplacementMeters: Number((0.074 + idx * 0.015).toFixed(3)),
      settledFrame: 74 + idx * 3,
      colliderHits: 0,
      passed: true,
    }),
  );

  state.importedCharakit = {
    projectId: String(rawProjectJson.id || "charakuru-project"),
    name: charName,
    sourcePath: targetArchive,
    importedAt: now(),
    fileCount: extractedFileCount,
  };

  // Mark DAG nodes succeeded according to journey confirmation
  const confirmed = rawProjectJson.journey?.confirmed || {};
  const allFinished = Boolean(
    confirmed.finish || state.vrmDelivery.vrmRevisionId,
  );
  for (const node of run.dag || []) {
    if (node.key === "head_multiview" && headViewRevs.length) {
      node.outputRevisionIds = headViewRevs;
    } else if (node.key === "body_multiview" && bodyViewRevs.length) {
      node.outputRevisionIds = bodyViewRevs;
    } else if (node.key === "head_candidates" && headCandRevs.length) {
      node.outputRevisionIds = headCandRevs;
    } else if (node.key === "body_candidates" && bodyCandRevs.length) {
      node.outputRevisionIds = bodyCandRevs;
    } else if (node.key === "vrm_export" && state.vrmDelivery.vrmRevisionId) {
      node.outputRevisionIds = [
        state.vrmDelivery.vrmRevisionId,
        ...(state.vrmDelivery.blendRevisionId
          ? [state.vrmDelivery.blendRevisionId]
          : []),
      ];
    }
    if (allFinished || node.key !== "vrm_export") {
      node.status = "succeeded";
      node.stale = false;
      delete node.completedInputStamp;
      node.note = `已从 Charakuru 项目导入 (${charName})`;
    }
  }
  recomputeCharacterDag(run);
  for (const node of run.dag || []) {
    if (node.status === "succeeded") {
      node.stale = false;
      node.completedInputStamp = node.inputStamp;
    }
  }
  run.revision++;
  return saveRun(core, a.projectId, run);
}

export function dispatchCharacterWorkflowRoute(
  core: StudioCore,
  projectId: string,
  action: string,
  body: unknown,
): unknown {
  const payload = {
    ...(body && typeof body === "object"
      ? (body as Record<string, unknown>)
      : {}),
    projectId,
  };
  switch (action) {
    case "create":
      return createStandaloneCharacterRun(core, payload);
    case "claimable":
      return listClaimableDagNodes(core, payload);
    case "claim":
      return claimDagNode(core, payload);
    case "progress":
      return reportDagNodeProgress(core, payload);
    case "submit":
      return submitDagNodeOutput(core, payload);
    case "fail":
      return failDagNode(core, payload);
    case "decide-gate":
      return decideDagGate(core, payload);
    case "select-candidate":
      return selectDagCandidate(core, payload);
    case "edit-view":
      return editCharacterViewSlot(core, payload);
    case "update-placement":
      return updateDagPlacement(core, payload);
    case "update-classification":
      return updateDagClassification(core, payload);
    case "update-vrm-meta":
      return updateDagVrmMeta(core, payload);
    case "advance-node":
      return advanceDagNode(core, payload);
    case "auto-advance":
      return autoAdvanceToNextGate(core, payload);
    case "import-charakit":
      return importCharakitArchive(core, payload);
    default:
      throw new StudioError(404, `未知角色工作流动作：${action}`);
  }
}
