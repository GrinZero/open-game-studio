import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { execFileSync } from "node:child_process";
import path from "node:path";
import os from "node:os";
import { StudioCore, StudioError } from "../src/server/core.js";
import { dispatch } from "../src/server/operations.js";
import {
  createStandaloneCharacterRun,
  listClaimableDagNodes,
  claimDagNode,
  reportDagNodeProgress,
  submitDagNodeOutput,
  decideDagGate,
  selectDagCandidate,
  autoAdvanceToNextGate,
  importCharakitArchive,
} from "../src/server/character-workflow.js";

function setup() {
  const dir = mkdtempSync(path.join(os.tmpdir(), "studio-char-dag-"));
  const core = new StudioCore(path.join(dir, "config"));
  const p = core.projectManage({
    action: "create",
    name: "二次元角色工坊",
    root: path.join(dir, "game"),
    engine: "unity",
  });
  return {
    dir,
    core,
    p,
    clean: () => {
      core.close();
      rmSync(dir, { recursive: true, force: true });
    },
  };
}

test("角色工作流 DAG 支持并行多视图认领、租约互斥、自动推进、候选非破坏分支与完整交付", () => {
  const x = setup();
  try {
    const run1 = createStandaloneCharacterRun(x.core, {
      projectId: x.p.id,
      characterName: "星穹巡林者",
      stylePrompt: "MToon 赛璐璐，1.58m 标准身高",
    });
    assert.ok(run1.dag && run1.dag.length === 23);
    assert.equal(run1.dag[0].key, "source_illustration");
    assert.equal(run1.dag[0].status, "succeeded");

    // Both head_multiview and body_multiview should be claimable in parallel
    const claimable1 = listClaimableDagNodes(x.core, {
      projectId: x.p.id,
      runId: run1.id,
    });
    const keys1 = claimable1.nodes.map((n) => n.node.key);
    assert.deepEqual(keys1.sort(), ["body_multiview", "head_multiview"]);

    // Agent A claims head_multiview
    const claimed = claimDagNode(x.core, {
      projectId: x.p.id,
      runId: run1.id,
      nodeKey: "head_multiview",
      agentId: "agent-alpha",
      agentName: "Alpha Image Worker",
      leaseSeconds: 120,
    });
    const headNode = claimed.dag!.find((n) => n.key === "head_multiview")!;
    assert.equal(headNode.status, "running");
    assert.equal(headNode.lease?.agentId, "agent-alpha");

    // Agent B attempting to claim head_multiview before lease expiry is rejected with 409
    assert.throws(
      () =>
        claimDagNode(x.core, {
          projectId: x.p.id,
          runId: run1.id,
          nodeKey: "head_multiview",
          agentId: "agent-beta",
          agentName: "Beta Worker",
        }),
      (e) => e instanceof StudioError && e.status === 409,
    );

    // Agent A reports progress
    const progressed = reportDagNodeProgress(x.core, {
      projectId: x.p.id,
      runId: run1.id,
      nodeKey: "head_multiview",
      agentId: "agent-alpha",
      progress: 65,
      message: "已生成前/左/右三张正交头部视图",
    });
    assert.equal(
      progressed.dag!.find((n) => n.key === "head_multiview")!.lease?.progress,
      65,
    );

    // Auto-advance completes ready work nodes and stops at Gate 1 (images_gate)
    const step1 = autoAdvanceToNextGate(x.core, {
      projectId: x.p.id,
      runId: run1.id,
    });
    // head_multiview was running, let's submit it too
    submitDagNodeOutput(x.core, {
      projectId: x.p.id,
      runId: run1.id,
      nodeKey: "head_multiview",
      note: "头部四视图完成",
    });
    const toGate1 = autoAdvanceToNextGate(x.core, {
      projectId: x.p.id,
      runId: run1.id,
    });
    assert.equal(toGate1.stoppedAtGate, "images_gate");

    // Pass Gate 1 -> unlocks head_candidates & body_candidates in parallel
    decideDagGate(x.core, {
      projectId: x.p.id,
      runId: run1.id,
      nodeKey: "images_gate",
      decision: "accept",
      comment: "8向视图对齐通过",
    });
    const toGate2 = autoAdvanceToNextGate(x.core, {
      projectId: x.p.id,
      runId: run1.id,
    });
    assert.equal(toGate2.stoppedAtGate, "candidates_gate");

    // Pass Gate 2 -> unlocks segmentation, classification, uv_guide_prep, and 5 parallel retexture nodes!
    decideDagGate(x.core, {
      projectId: x.p.id,
      runId: run1.id,
      nodeKey: "candidates_gate",
      decision: "accept",
      comment: "采用 head-1 与 body-1",
    });
    const toGate3 = autoAdvanceToNextGate(x.core, {
      projectId: x.p.id,
      runId: run1.id,
    });
    assert.equal(toGate3.stoppedAtGate, "proportions_gate");
    assert.equal(
      toGate3.run.dagState?.textureGroups.hair.status,
      "retextured",
    );

    // Switching candidate after textures exist automatically forks a non-destructive branch Run!
    const forkResult = selectDagCandidate(x.core, {
      projectId: x.p.id,
      runId: run1.id,
      source: "head",
      candidateId: "head-2",
    });
    assert.equal(forkResult.branched, true);
    assert.equal(forkResult.parentRunId, run1.id);
    assert.equal(forkResult.run.dagState?.selectedHeadCandidateId, "head-2");
    assert.equal(
      forkResult.run.dagState?.textureGroups.hair.status,
      "pending",
    );

    // Original run still preserves its retextured status!
    const origSnap = x.core
      .snapshot(x.p.id)
      .runs.find((r) => r.id === run1.id)!;
    assert.equal(origSnap.dagState?.selectedHeadCandidateId, "head-1");
    assert.equal(origSnap.dagState?.textureGroups.hair.status, "retextured");

    // Via MCP dispatch: job_control can also advance & finish the original run
    decideDagGate(x.core, {
      projectId: x.p.id,
      runId: run1.id,
      nodeKey: "proportions_gate",
      decision: "accept",
    });
    const toGate4 = dispatch(x.core, "job_control", {
      projectId: x.p.id,
      runId: run1.id,
      action: "auto_advance_dag",
    }) as { stoppedAtGate?: string };
    assert.equal(toGate4.stoppedAtGate, "motion_gate");
  } finally {
    x.clean();
  }
});

test("支持将 .charakit 归档完整迁移为 23 节点 DAG 运行", () => {
  const x = setup();
  try {
    const zipPath = path.join(x.dir, "sample.charakit");
    const pngBase64 =
      "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aM2cAAAAASUVORK5CYII=";
    execFileSync(
      "/usr/bin/python3",
      [
        "-c",
        `import zipfile, json, base64, sys
zp, b64 = sys.argv[1], sys.argv[2]
png = base64.b64decode(b64)
proj = {
  "id": "charakit-001",
  "name": "测试迁移少女",
  "sourceImage": "assets/orig.png",
  "imageLibrary": {
    "slots": {
      "head/front/neutral": {"adopted": "v1"},
      "body/front/neutral": {"adopted": "v2"}
    },
    "versions": [
      {"id": "v1", "slot": "head/front/neutral", "path": "images/hf.png", "width": 1024, "height": 1024},
      {"id": "v2", "slot": "body/front/neutral", "path": "images/bf.png", "width": 1024, "height": 1024}
    ]
  },
  "production": {
    "selected": {"head": "head_1", "body": "body_1"},
    "components": [
      {"id": "head_001", "source": "head", "role": "hair", "vertices": 1200, "faces": 2400, "min": [-0.1, 1.2, -0.1], "max": [0.1, 1.5, 0.1]}
    ],
    "textureGroups": [
      {"id": "hair", "name": "髪", "components": ["head_001"], "returned": {"path": "images/hf.png"}}
    ],
    "springGroups": [
      {"id": "hair", "name": "髪", "kind": "hair", "components": ["head_001"]}
    ]
  },
  "journey": {"confirmed": {"images": "ok", "candidates": "ok", "materials": "ok", "rig": "ok", "finish": "ok"}}
}
with zipfile.ZipFile(zp, "w") as z:
    z.writestr("project.json", json.dumps(proj))
    z.writestr("assets/orig.png", png)
    z.writestr("images/hf.png", png)
    z.writestr("images/bf.png", png)
`,
        zipPath,
        pngBase64,
      ],
      { stdio: "pipe" },
    );

    const run = importCharakitArchive(x.core, {
      projectId: x.p.id,
      archivePath: zipPath,
    });
    assert.equal(run.dagState?.characterName, "测试迁移少女");
    assert.ok(run.dagState?.views.original.revisionId);
    assert.ok(run.dagState?.views.head_front.revisionId);
    assert.equal(run.dagState?.segmentation.components.length, 1);
    assert.equal(run.dagState?.textureGroups.hair.status, "retextured");
  } finally {
    x.clean();
  }
});
