import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { execFileSync } from "node:child_process";
import path from "node:path";
import os from "node:os";
import {
  nodeGuidance,
  characterCodexContext,
} from "../src/ui/character-workflow-guidance";
import { formatCodexContext } from "../src/ui/codex-context";
import { CodexReferenceStore } from "../src/ui/codex-references";
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
  editCharacterViewSlot,
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
    const source = importSource(x.core, x.p.id, "巡林者.png");
    const run1 = createStandaloneCharacterRun(x.core, {
      projectId: x.p.id,
      characterName: "星穹巡林者",
      stylePrompt: "MToon 赛璐璐，1.58m 标准身高",
      sourceImageRevisionId: source.id,
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
    assert.equal(toGate3.run.dagState?.textureGroups.hair.status, "retextured");

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
    assert.equal(forkResult.run.dagState?.textureGroups.hair.status, "pending");

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

function importSource(core: StudioCore, projectId: string, name: string) {
  return core.importFiles(projectId, [
    {
      name,
      bytes: Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aM2cAAAAASUVORK5CYII=",
        "base64",
      ),
    },
  ]);
}

test("角色操作指引按前置、制作中和审核状态给出可执行的下一步", () => {
  const x = setup();
  try {
    const source = importSource(x.core, x.p.id, "角色.png");
    let run = createStandaloneCharacterRun(x.core, {
      projectId: x.p.id,
      sourceImageRevisionId: source.id,
    });
    assert.equal(nodeGuidance(run, "head_multiview").ready, true);
    assert.equal(nodeGuidance(run, "balance_three_views").ready, false);
    assert.match(
      nodeGuidance(run, "balance_three_views").reason,
      /头部四视图生成、身体四视图生成/,
    );
    assert.equal(nodeGuidance(run, "images_gate").ready, false);
    run = editCharacterViewSlot(x.core, {
      projectId: x.p.id,
      runId: run.id,
      slotKey: "head_front",
      revisionId: source.id,
    });
    assert.match(nodeGuidance(run, "head_multiview").reason, /已回填 1\/4/);
    run = claimDagNode(x.core, {
      projectId: x.p.id,
      runId: run.id,
      nodeKey: "head_multiview",
      agentId: "test",
    });
    assert.equal(nodeGuidance(run, "head_multiview").running, true);
    assert.equal(nodeGuidance(run, "head_multiview").ready, false);
    run = autoAdvanceToNextGate(x.core, {
      projectId: x.p.id,
      runId: run.id,
    }).run;
    assert.equal(nodeGuidance(run, "images_gate").ready, false);
  } finally {
    x.clean();
  }
});

test("Codex 角色引用保留具体槽位、选定参考和回填位置，不同槽位能同时引用", async () => {
  const x = setup();
  try {
    const source = importSource(x.core, x.p.id, "原画.png");
    const other = importSource(x.core, x.p.id, "地图.png");
    const run = createStandaloneCharacterRun(x.core, {
      projectId: x.p.id,
      sourceImageRevisionId: source.id,
    });
    const snapshot = x.core.snapshot(x.p.id);
    const context = characterCodexContext(
      snapshot,
      run,
      "source_illustration",
      {
        slotKey: "original",
        referenceRevisionIds: [source.id],
        prompt: "保持发型，换一套衣服",
      },
    );
    const text = formatCodexContext(context);
    assert.match(text, /保持发型，换一套衣服/);
    assert.match(text, /"slotKey":"original"/);
    assert.match(text, /expectedRevision/);
    assert.ok(text.includes(path.join(x.p.root, source.files[0].relativePath)));
    assert.ok(!text.includes(other.id));
    assert.equal(context.characterTarget?.runId, run.id);
    const scratch = characterCodexContext(
      snapshot,
      run,
      "source_illustration",
      { slotKey: "original", referenceRevisionIds: [] },
    );
    assert.deepEqual(scratch.inputSnapshot.revisionIds, []);
    const store = new CodexReferenceStore(async () => {});
    for (const slotKey of ["head_front", "head_left"] as const)
      await store.add(
        characterCodexContext(snapshot, run, "head_multiview", { slotKey }),
      );
    assert.equal(store.getSnapshot().length, 2);
  } finally {
    x.clean();
  }
});

test("角色原画必须显式选择，不能自动采用项目首图或无效版本", () => {
  const x = setup();
  try {
    importSource(x.core, x.p.id, "地图截图.png");
    const source = importSource(x.core, x.p.id, "角色立绘.png");
    const draft = createStandaloneCharacterRun(x.core, { projectId: x.p.id });
    assert.equal(draft.dagState?.views.original.revisionId, undefined);
    assert.equal(draft.dag?.[0].status, "awaiting_manual");
    assert.equal(
      draft.dag?.find((n) => n.key === "head_multiview")?.status,
      "pending",
    );
    assert.throws(
      () =>
        autoAdvanceToNextGate(x.core, { projectId: x.p.id, runId: draft.id }),
      /请先选择或上传角色原画/,
    );
    const before = x.core.snapshot(x.p.id);
    for (const revisionId of [
      "missing",
      before.assets.find((a) => a.kind !== "image")!.latestRevisionId,
    ]) {
      assert.throws(
        () =>
          createStandaloneCharacterRun(x.core, {
            projectId: x.p.id,
            sourceImageRevisionId: revisionId,
          }),
        /可用的原画图片版本/,
      );
    }
    assert.equal(x.core.snapshot(x.p.id).runs.length, before.runs.length);
    const run = createStandaloneCharacterRun(x.core, {
      projectId: x.p.id,
      sourceImageRevisionId: source.id,
    });
    assert.equal(run.dagState?.views.original.revisionId, source.id);
    assert.ok(run.inputRevisionIds.includes(source.id));
  } finally {
    x.clean();
  }
});

test("更换原画固定新版本，保留旧资产并使下游制作和审核重新执行", () => {
  const x = setup();
  try {
    const oldSource = importSource(x.core, x.p.id, "旧原画.png");
    const source = importSource(x.core, x.p.id, "新原画.png");
    let run = createStandaloneCharacterRun(x.core, {
      projectId: x.p.id,
      sourceImageRevisionId: oldSource.id,
    });
    const initialRevision = run.revision;
    run = autoAdvanceToNextGate(x.core, {
      projectId: x.p.id,
      runId: run.id,
    }).run;
    const outputs = run.dag!.find(
      (n) => n.key === "head_multiview",
    )!.outputRevisionIds;
    assert.throws(
      () =>
        editCharacterViewSlot(x.core, {
          projectId: x.p.id,
          runId: run.id,
          slotKey: "original",
          revisionId: source.id,
          expectedRevision: initialRevision,
        }),
      /角色已更新/,
    );
    assert.throws(
      () =>
        editCharacterViewSlot(x.core, {
          projectId: x.p.id,
          runId: run.id,
          slotKey: "original",
          revisionId: "missing",
        }),
      /可用的原画图片版本/,
    );
    run = editCharacterViewSlot(x.core, {
      projectId: x.p.id,
      runId: run.id,
      slotKey: "original",
      revisionId: source.id,
      expectedRevision: run.revision,
    });
    assert.equal(run.dagState?.views.original.revisionId, source.id);
    assert.equal(
      run.dagState?.views.original.relativePath,
      source.files[0].relativePath,
    );
    assert.ok(run.inputRevisionIds.includes(source.id));
    assert.ok(!run.inputRevisionIds.includes(oldSource.id));
    assert.ok(!run.dag![0].outputRevisionIds.includes(oldSource.id));
    const head = run.dag!.find((n) => n.key === "head_multiview")!;
    assert.equal(head.status, "awaiting_agent");
    assert.equal(head.stale, true);
    assert.equal(
      run.dag!.find((n) => n.key === "images_gate")!.status,
      "pending",
    );
    assert.deepEqual(head.outputRevisionIds, outputs);
    for (const id of [oldSource.id, ...outputs])
      assert.ok(x.core.assetFile(x.p.id, id, 0).file);
    assert.equal(run.dagState?.views.head_front.revisionId, undefined);
    assert.ok(run.inputSnapshot?.revisionIds.includes(source.id));
    assert.ok(!run.inputSnapshot?.revisionIds.includes(oldSource.id));
    run = autoAdvanceToNextGate(x.core, {
      projectId: x.p.id,
      runId: run.id,
    }).run;
    assert.ok(!outputs.includes(run.dagState!.views.head_front.revisionId!));
    assert.equal(
      run.dag!.find((n) => n.key === "head_multiview")!.stale,
      false,
    );
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
