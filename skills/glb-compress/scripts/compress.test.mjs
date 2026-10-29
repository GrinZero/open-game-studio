import test from "node:test";
import assert from "node:assert/strict";
import { Document } from "@gltf-transform/core";
import sharp from "sharp";
import {
  mkdtemp,
  readFile,
  writeFile,
  symlink,
  link,
  rm,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { getIO, compressBytes, parseGLB } from "./engine.mjs";
import { resolveOptions } from "./options.mjs";
import { weldWithTolerance } from "./weld.mjs";
const run = promisify(execFile);
const noChange = { preset: "none", textures: "off", resizeMax: 0 };
async function fixture({ alpha = false, morph = false, skin = false } = {}) {
  const d = new Document(),
    b = d.createBuffer("buffer");
  const attr = (name, type, array) =>
    d.createAccessor(name).setType(type).setArray(array).setBuffer(b);
  const positions = [],
    uv = [],
    normal = [],
    tangent = [],
    indices = [];
  const n = 12;
  for (let y = 0; y <= n; y++)
    for (let x = 0; x <= n; x++) {
      positions.push(x / n, y / n, 0);
      uv.push(x / n, y / n);
      normal.push(0, 0, 1);
      tangent.push(1, 0, 0, 1);
    }
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      const i = y * (n + 1) + x;
      indices.push(i, i + 1, i + n + 1, i + 1, i + n + 2, i + n + 1);
    }
  const p = d
    .createPrimitive()
    .setAttribute(
      "POSITION",
      attr("positions", "VEC3", new Float32Array(positions)),
    )
    .setAttribute("NORMAL", attr("normals", "VEC3", new Float32Array(normal)))
    .setAttribute("TEXCOORD_0", attr("UV", "VEC2", new Float32Array(uv)))
    .setAttribute(
      "TANGENT",
      attr("tangents", "VEC4", new Float32Array(tangent)),
    )
    .setIndices(attr("indices", "SCALAR", new Uint16Array(indices)));
  const texture = d
    .createTexture("color")
    .setMimeType("image/png")
    .setImage(
      await sharp({
        create: {
          width: 64,
          height: 32,
          channels: alpha ? 4 : 3,
          background: alpha
            ? { r: 40, g: 100, b: 200, alpha: 0.4 }
            : { r: 40, g: 100, b: 200 },
        },
      })
        .png()
        .toBuffer(),
    );
  const mat = d.createMaterial("paint").setBaseColorTexture(texture);
  if (alpha) mat.setAlphaMode("BLEND");
  p.setMaterial(mat);
  if (morph) {
    p.addTarget(
      d
        .createPrimitiveTarget()
        .setAttribute(
          "POSITION",
          attr("delta", "VEC3", new Float32Array(positions.length)),
        ),
    );
  }
  const mesh = d.createMesh("grid").addPrimitive(p),
    node = d.createNode("model").setMesh(mesh);
  const scene = d.createScene("scene").addChild(node);
  d.getRoot().setDefaultScene(scene);
  if (skin) {
    const joint = d.createNode("joint");
    scene.addChild(joint);
    node.setSkin(d.createSkin("rig").addJoint(joint));
    p.setAttribute(
      "JOINTS_0",
      attr("joints", "VEC4", new Uint16Array((positions.length / 3) * 4)),
    );
    const w = new Float32Array((positions.length / 3) * 4);
    for (let i = 0; i < w.length; i += 4) w[i] = 1;
    p.setAttribute("WEIGHTS_0", attr("weights", "VEC4", w));
  }
  const sampler = d
    .createAnimationSampler("sampler")
    .setInput(attr("times", "SCALAR", new Float32Array([0, 1])))
    .setOutput(
      attr("translation", "VEC3", new Float32Array([0, 0, 0, 0, 1, 0])),
    );
  d.createAnimation("move")
    .addSampler(sampler)
    .addChannel(
      d
        .createAnimationChannel("channel")
        .setTargetNode(node)
        .setTargetPath("translation")
        .setSampler(sampler),
    );
  return { d, bytes: await (await getIO()).writeBinary(d) };
}
const fixturePromise = fixture();
const read = async (bytes) => (await getIO()).readBinary(bytes);
test("defaults, website labels, strict options and precision conflicts", () => {
  assert.equal(resolveOptions().effective.resizeMax, 1024);
  assert.equal(resolveOptions({ preset: "none" }).effective.geometry, "off");
  assert.equal(
    resolveOptions({ simplify: "light" }).effective.simplifyRatio,
    0.9,
  );
  assert.equal(
    resolveOptions({ weld: "aggressive" }).effective.weldTolerance,
    0.001,
  );
  assert.throws(() => resolveOptions({ resizeMax: -1 }));
  assert.throws(() => resolveOptions({ typo: true }));
  assert.throws(() => resolveOptions({ keepNames: "false" }));
  assert.throws(() =>
    resolveOptions({ simplify: true, preservePrecision: true }),
  );
});
test("all off preserves texture, positions, names and animations", async () => {
  const { bytes, d } = await fixturePromise,
    { output, report } = await compressBytes(bytes, noChange),
    out = await read(output);
  assert.deepEqual(
    Buffer.from(out.getRoot().listTextures()[0].getImage()),
    Buffer.from(d.getRoot().listTextures()[0].getImage()),
  );
  assert.deepEqual(
    out
      .getRoot()
      .listMeshes()[0]
      .listPrimitives()[0]
      .getAttribute("POSITION")
      .getArray(),
    d
      .getRoot()
      .listMeshes()[0]
      .listPrimitives()[0]
      .getAttribute("POSITION")
      .getArray(),
  );
  assert.equal(report.after.animations, 1);
  assert.equal(out.getRoot().listNodes()[0].getName(), "model");
});
test("default Draco + WebP roundtrip and aspect-ratio resize", async () => {
  const { bytes } = await fixturePromise,
    { report } = await compressBytes(bytes, { resizeMax: 16 });
  assert.deepEqual(report.after.textures[0].size, [16, 8]);
  assert.ok(
    report.after.extensionsRequired.includes("KHR_draco_mesh_compression"),
  );
  assert.ok(report.after.extensionsRequired.includes("EXT_texture_webp"));
  assert.equal(report.before.triangles, report.after.triangles);
  assert.equal(report.after.animations, 1);
});
test("resize without reformat, no upscaling, JPEG conversion", async () => {
  const { bytes } = await fixturePromise;
  const a = await compressBytes(bytes, { ...noChange, resizeMax: 16 });
  assert.equal(a.report.after.textures[0].mimeType, "image/png");
  assert.deepEqual(a.report.after.textures[0].size, [16, 8]);
  const b = await compressBytes(bytes, { ...noChange, resizeMax: 2048 });
  assert.deepEqual(b.report.after.textures[0].size, [64, 32]);
  const c = await compressBytes(bytes, { ...noChange, textures: "jpeg" });
  assert.equal(c.report.after.textures[0].mimeType, "image/jpeg");
});
test("JPEG keeps alpha as PNG with explicit warning", async () => {
  const { bytes } = await fixture({ alpha: true }),
    { report, output } = await compressBytes(bytes, {
      ...noChange,
      textures: "jpeg",
    });
  assert.equal(report.after.textures[0].mimeType, "image/png");
  assert.match(report.warnings.join(" "), /alpha/);
  const doc = await read(output);
  assert.equal(
    (await sharp(doc.getRoot().listTextures()[0].getImage()).metadata())
      .hasAlpha,
    true,
  );
});
test("strip data and remove names changes actual payload", async () => {
  const { bytes } = await fixturePromise,
    { output, report } = await compressBytes(bytes, {
      ...noChange,
      stripAnimations: true,
      stripTangents: true,
      keepNames: false,
    });
  const d = await read(output);
  assert.equal(report.after.animations, 0);
  assert.equal(d.getRoot().listNodes()[0].getName(), "");
  assert.equal(
    d.getRoot().listMeshes()[0].listPrimitives()[0].getAttribute("TANGENT"),
    null,
  );
  assert.ok(output.length < bytes.length);
  assert.ok(
    !d
      .getRoot()
      .listAccessors()
      .some((a) => a.getName() === "times"),
  );
});
test("simplification reduces triangles and reports actual retention", async () => {
  const { bytes } = await fixturePromise,
    { report } = await compressBytes(bytes, {
      ...noChange,
      simplify: "aggressive",
    });
  assert.ok(report.after.triangles < report.before.triangles);
  assert.equal(
    report.actualTriangleRetainRatio,
    report.after.triangles / report.before.triangles,
  );
});
test("weld tolerance merges nearby vertices but protects UV, normal, joints and morph seams", () => {
  for (const seam of ["none", "uv", "normal", "joints", "morph"]) {
    const d = new Document(),
      b = d.createBuffer();
    const attr = (type, array) =>
      d
        .createAccessor()
        .setType(type)
        .setArray(new Float32Array(array))
        .setBuffer(b);
    const p = d
      .createPrimitive()
      .setMode(0)
      .setAttribute(
        "POSITION",
        attr("VEC3", [0, 0, 0, 0.00005, 0, 0, 1, 0, 0]),
      );
    if (seam === "uv")
      p.setAttribute("TEXCOORD_0", attr("VEC2", [0, 0, 1, 0, 0, 0]));
    if (seam === "normal")
      p.setAttribute("NORMAL", attr("VEC3", [0, 1, 0, 0, 0, 1, 0, 1, 0]));
    if (seam === "joints")
      p.setAttribute(
        "JOINTS_0",
        d
          .createAccessor()
          .setType("VEC4")
          .setArray(new Uint16Array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]))
          .setBuffer(b),
      );
    if (seam === "morph")
      p.addTarget(
        d
          .createPrimitiveTarget()
          .setAttribute("POSITION", attr("VEC3", [0, 0, 0, 0, 1, 0, 0, 0, 0])),
      );
    d.createMesh().addPrimitive(p);
    weldWithTolerance(d, 0.0001);
    assert.equal(
      p.getAttribute("POSITION").getCount(),
      seam === "none" ? 2 : 3,
      seam,
    );
  }
});
test("skin, morph and animation data survive compression and simplification", async () => {
  for (const type of ["skin", "morph"]) {
    const { bytes } = await fixture({ [type]: true });
    const { output, report } = await compressBytes(bytes),
      d = await read(output);
    if (type === "skin") assert.equal(report.after.skins, 1);
    else
      assert.equal(
        d.getRoot().listMeshes()[0].listPrimitives()[0].listTargets().length,
        1,
      );
    assert.equal(report.after.animations, 1);
    const reduced = await compressBytes(bytes, {
      ...noChange,
      simplify: "aggressive",
    });
    const reducedDoc = await read(reduced.output);
    const primitive = reducedDoc.getRoot().listMeshes()[0].listPrimitives()[0];
    const count = primitive.getAttribute("POSITION").getCount();
    assert.ok(reduced.report.after.triangles < reduced.report.before.triangles);
    assert.equal(reduced.report.after.animations, 1);
    if (type === "skin") {
      assert.equal(reduced.report.after.skins, 1);
      assert.equal(primitive.getAttribute("JOINTS_0").getCount(), count);
      assert.equal(primitive.getAttribute("WEIGHTS_0").getCount(), count);
    } else
      assert.equal(
        primitive.listTargets()[0].getAttribute("POSITION").getCount(),
        count,
      );
    assert.match(reduced.report.warnings.join(" "), /animated deformation/);
  }
});
test("precision mode keeps exact geometry and decodes incoming Draco when switched off", async () => {
  const { bytes, d } = await fixturePromise,
    { output, report } = await compressBytes(bytes, {
      preservePrecision: true,
    });
  assert.ok(
    !report.after.extensionsRequired.includes("KHR_draco_mesh_compression"),
  );
  assert.deepEqual(
    (await read(output))
      .getRoot()
      .listMeshes()[0]
      .listPrimitives()[0]
      .getAttribute("POSITION")
      .getArray(),
    d
      .getRoot()
      .listMeshes()[0]
      .listPrimitives()[0]
      .getAttribute("POSITION")
      .getArray(),
  );
  const compressed = await compressBytes(bytes);
  const decoded = await compressBytes(compressed.output, noChange);
  assert.ok(
    !decoded.report.after.extensionsRequired.includes(
      "KHR_draco_mesh_compression",
    ),
  );
});
test("WebP mipmaps requests sampler filter and reports runtime generation", async () => {
  const { bytes } = await fixturePromise,
    { output, report } = await compressBytes(bytes, {
      geometry: "off",
      mipmaps: true,
    });
  assert.match(report.warnings.join(" "), /runtime/);
  const d = await read(output);
  assert.equal(
    d.getRoot().listMaterials()[0].getBaseColorTextureInfo().getMinFilter(),
    9987,
  );
});
test("invalid container and unknown extension fail instead of silently dropping data", async () => {
  assert.throws(() => parseGLB(Buffer.from("invalid")), /glTF/);
  const { bytes } = await fixturePromise;
  const old = Buffer.from(bytes),
    json = parseGLB(old);
  json.extensionsUsed = ["VENDOR_unknown"];
  let text = JSON.stringify(json);
  text += " ".repeat((4 - (text.length % 4)) % 4);
  const bin = old.subarray(20 + old.readUInt32LE(12));
  const out = Buffer.alloc(20 + text.length + bin.length);
  old.copy(out, 0, 0, 20);
  out.writeUInt32LE(out.length, 8);
  out.writeUInt32LE(text.length, 12);
  out.write(text, 20);
  bin.copy(out, 20 + text.length);
  await assert.rejects(() => compressBytes(out), /Unsupported extension/);
});
test("KTX2 rejects unavailable encoder instead of silently falling back", async () => {
  const { bytes } = await fixturePromise;
  await assert.rejects(
    () =>
      compressBytes(
        bytes,
        { textures: "ktx2" },
        { toktx: "/nonexistent/toktx" },
      ),
    /requires local/,
  );
});
test(
  "KTX2 encodes actual BasisLZ mip levels",
  { skip: !process.env.TOKTX && "Set TOKTX to run real KTX2 encoding" },
  async () => {
    const { bytes } = await fixturePromise;
    const { output, report } = await compressBytes(
      bytes,
      { ...noChange, textures: "ktx2", mipmaps: true },
      { toktx: process.env.TOKTX },
    );
    assert.equal(report.after.textures[0].mimeType, "image/ktx2");
    assert.equal(report.after.textures[0].levels, 7);
    assert.ok(report.after.extensionsRequired.includes("KHR_texture_basisu"));
    const texture = Buffer.from(
      (await read(output)).getRoot().listTextures()[0].getImage(),
    );
    assert.equal(texture.readUInt32LE(12), 0, "Universal compressed format");
    assert.equal(texture.readUInt32LE(44), 1, "BasisLZ supercompression");
    const retained = await compressBytes(output, noChange);
    assert.equal(retained.report.after.textures[0].levels, 7);
  },
);
test("all encoder presets and weld levels produce readable GLB", async () => {
  const { bytes } = await fixturePromise;
  for (const preset of ["balanced", "max", "fast", "none"]) {
    const { report } = await compressBytes(bytes, { preset, weld: "light" });
    assert.equal(report.after.triangles, report.before.triangles);
    assert.equal(
      report.after.extensionsRequired.includes("KHR_draco_mesh_compression"),
      preset !== "none",
    );
  }
});
test("CLI supports config, report, inspect, rejects overwrites and all source aliases", async () => {
  const dir = await mkdtemp(path.join(tmpdir(), "glb-cli-test-")),
    cli = fileURLToPath(new URL("./compress.mjs", import.meta.url));
  try {
    const input = path.join(dir, "input.glb"),
      output = path.join(dir, "out.glb"),
      config = path.join(dir, "config.json");
    const { bytes } = await fixturePromise;
    await writeFile(input, bytes);
    await writeFile(config, JSON.stringify(noChange));
    const call = (...args) => run(process.execPath, [cli, ...args]);
    const r = await call(
      input,
      output,
      "--config",
      config,
      "--resizeMax",
      "16",
    );
    const report = JSON.parse(r.stdout);
    assert.equal(report.after.textures[0].size[0], 16);
    assert.deepEqual(
      JSON.parse(await readFile(`${output}.report.json`, "utf8")),
      report,
    );
    assert.equal(
      JSON.parse((await call("--inspect", output)).stdout).triangles,
      288,
    );
    await assert.rejects(() => call(input, output), /already exists/);
    await assert.rejects(
      () => call(input, input, "--force"),
      /different files/,
    );
    const sym = path.join(dir, "sym.glb"),
      hard = path.join(dir, "hard.glb");
    await symlink(input, sym);
    await link(input, hard);
    for (const alias of [sym, hard])
      await assert.rejects(
        () => call(input, alias, "--force"),
        /different files/,
      );
    await assert.rejects(
      () =>
        call(input, path.join(dir, "new.glb"), "--report", input, "--force"),
      /different files/,
    );
    assert.deepEqual(await readFile(input), Buffer.from(bytes));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
