import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, mkdir, writeFile, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { Document } from "@gltf-transform/core";
import sharp from "sharp";
import { getIO, parseGLB } from "./engine.mjs";
import { projectContext, recommendCompression } from "./policy.mjs";
const run = promisify(execFile);
const scripts = path.dirname(fileURLToPath(import.meta.url));
const low = {
  bytes: 1000000,
  triangles: 996,
  skins: 0,
  animations: 0,
  morphTargets: 0,
  textures: [{ bytes: 990000, size: [8192, 8192] }],
};
test("recommendation keeps low-poly topology, offers 1K/2K and retains unselected state", () => {
  const p = recommendCompression(low, { engine: "godot" });
  assert.equal(p.selected, null);
  assert.equal(p.preferredTrial, "ktx2");
  assert.equal(p.candidates[0].id, "ktx2");
  assert.equal(p.candidates[0].options.resizeMax, 0);
  assert.equal(p.review.humanApprovalRequired, true);
  assert.equal(p.review.decision, null);
  assert.ok(p.candidates.some((c) => c.id === "webp1k"));
  assert.ok(p.candidates.some((c) => c.id === "webp4k"));
  assert.ok(
    p.candidates.every(
      (c) =>
        c.options.geometry === "off" &&
        c.options.simplify === "off" &&
        c.options.weld === "off",
    ),
  );
  assert.ok(
    p.candidates.every(
      (c) =>
        c.options.keepNames &&
        !c.options.stripAnimations &&
        !c.options.stripTangents,
    ),
  );
});
test("size share, animation and source resolution modify recommendations", () => {
  const small = recommendCompression(
    { ...low, textures: [{ bytes: 990000, size: [512, 512] }] },
    { engine: "godot" },
  );
  assert.ok(!small.candidates.some((c) => c.id === "webp1k"));
  const high = recommendCompression(
    {
      ...low,
      triangles: 372308,
      textures: [{ bytes: 147000, size: [1024, 1024] }],
    },
    { engine: "godot" },
  );
  assert.equal(high.analysis.triangleBand, "high");
  assert.ok(high.candidates.some((c) => c.id === "geometry90"));
  assert.match(high.reasons.join(" "), /limited ceiling/);
  for (const field of ["skins", "animations", "morphTargets"]) {
    const p = recommendCompression(
      { ...low, triangles: 100000, [field]: 1 },
      { engine: "godot" },
    );
    assert.ok(!p.candidates.some((c) => c.id === "geometry90"));
  }
  const empty = recommendCompression(
    { ...low, textures: [] },
    { engine: "generic" },
  );
  assert.equal(empty.preferredTrial, null);
});
test("recommendation ranks actual payloads and defaults to a KTX2 trial pending human choice", () => {
  const plan = recommendCompression(
    {
      ...low,
      textures: [
        {
          name: "color",
          bytes: 10000,
          size: [2048, 2048],
          mimeType: "image/jpeg",
          slots: ["baseColorTexture"],
        },
        {
          name: "roughness-metallic",
          bytes: 900000,
          size: [4096, 4096],
          mimeType: "image/png",
          slots: ["metallicRoughnessTexture"],
        },
      ],
    },
    { engine: "godot" },
  );
  assert.equal(plan.textureAssessment[0].index, 1);
  assert.match(
    plan.textureAssessment[0].suggestedChecks.join(" "),
    /lossless re-encoding/,
  );
  assert.equal(plan.preferredTrial, "ktx2");
  assert.equal(plan.selected, null);
  assert.match(plan.nextStep, /wait_for_user_visual_comparison/);
});
test("Godot repo discovery and CLI enforcement work outside project cwd", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "glb-policy-"));
  try {
    const root = path.join(dir, "project"),
      assets = path.join(root, "art/models"),
      out = path.join(dir, "out");
    await mkdir(assets, { recursive: true });
    await mkdir(path.join(root, "game"));
    await mkdir(out);
    await writeFile(path.join(root, "game/project.godot"), "config_version=5");
    const input = path.join(assets, "model.glb");
    const d = new Document(),
      b = d.createBuffer();
    const p = d.createPrimitive().setAttribute(
      "POSITION",
      d
        .createAccessor()
        .setBuffer(b)
        .setType("VEC3")
        .setArray(new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0])),
    );
    const t = d
      .createTexture()
      .setMimeType("image/png")
      .setImage(
        await sharp({
          create: { width: 32, height: 32, channels: 3, background: "red" },
        })
          .png()
          .toBuffer(),
      );
    p.setMaterial(d.createMaterial().setBaseColorTexture(t));
    d.createScene().addChild(
      d.createNode().setMesh(d.createMesh().addPrimitive(p)),
    );
    await writeFile(input, await (await getIO()).writeBinary(d));
    assert.equal(projectContext({ input, cwd: dir }).engine, "godot");
    assert.equal(
      projectContext({ project: path.join(root, "game"), cwd: dir }).engine,
      "godot",
    );
    assert.throws(
      () => projectContext({ project: root, engine: "generic" }),
      /Godot/,
    );
    const call = (script, args) =>
      run(process.execPath, [path.join(scripts, script), ...args], {
        cwd: dir,
      });
    const output = path.join(out, "auto.glb");
    const report = JSON.parse(
      (await call("compress.mjs", [input, output])).stdout,
    );
    assert.equal(report.projectPolicy.engine, "godot");
    assert.equal(report.effective.geometry, "off");
    assert.equal(report.effective.textures, "ktx2");
    assert.equal(report.review.status, "awaiting_user_comparison");
    assert.equal(report.review.decision, null);
    assert.ok(
      !(parseGLB(await readFile(output)).extensionsRequired ?? []).includes(
        "KHR_draco_mesh_compression",
      ),
    );
    await assert.rejects(
      () =>
        call("compress.mjs", [
          input,
          path.join(out, "forbidden.glb"),
          "--geometry",
          "on",
        ]),
      /forbids Draco/,
    );
    await writeFile(
      path.join(dir, "on.json"),
      JSON.stringify({ geometry: "on" }),
    );
    await assert.rejects(
      () =>
        call("compress.mjs", [
          input,
          path.join(out, "config.glb"),
          "--config",
          path.join(dir, "on.json"),
        ]),
      /forbids Draco/,
    );
    const planDir = path.join(out, "plan");
    const plan = JSON.parse(
      (
        await call("recommend.mjs", [
          input,
          "--project",
          root,
          "--plan-dir",
          planDir,
        ])
      ).stdout,
    );
    assert.equal(plan.context.engine, "godot");
    assert.equal(
      JSON.parse(await readFile(path.join(planDir, "webp2k.json"), "utf8"))
        .geometry,
      "off",
    );
    const matrixDir = path.join(out, "matrix");
    await call("compare.mjs", [input, matrixDir, "--project", root]);
    const matrix = JSON.parse(
      await readFile(path.join(matrixDir, "comparison.json"), "utf8"),
    );
    assert.ok(!matrix.some((r) => r.name === "draco"));
    assert.ok(matrix.every((r) => r.effective.geometry === "off"));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
