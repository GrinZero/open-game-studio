import { NodeIO, Logger } from "@gltf-transform/core";
import {
  ALL_EXTENSIONS,
  KHRDracoMeshCompression,
  EXTTextureWebP,
  KHRTextureBasisu,
} from "@gltf-transform/extensions";
import {
  simplify,
  getTextureColorSpace,
  listTextureSlots,
} from "@gltf-transform/functions";
import draco from "draco3dgltf";
import { MeshoptDecoder, MeshoptSimplifier } from "meshoptimizer";
import sharp from "sharp";
import { readFile, writeFile, mkdtemp, rm } from "node:fs/promises";
import { tmpdir, homedir } from "node:os";
import { existsSync } from "node:fs";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { resolveOptions } from "./options.mjs";
import { weldWithTolerance } from "./weld.mjs";
const run = promisify(execFile);
let ioPromise;
export function getIO() {
  return (ioPromise ??= (async () =>
    new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
      "draco3d.decoder": await draco.createDecoderModule(),
      "draco3d.encoder": await draco.createEncoderModule(),
      "meshopt.decoder": MeshoptDecoder,
    }))());
}
export function parseGLB(bytes) {
  const b = Buffer.from(bytes);
  if (
    b.length < 20 ||
    b.toString("ascii", 0, 4) !== "glTF" ||
    b.readUInt32LE(4) !== 2 ||
    b.readUInt32LE(8) !== b.length ||
    b.readUInt32LE(16) !== 0x4e4f534a
  )
    throw Error("Expected a complete glTF 2.0 binary (.glb)");
  const length = b.readUInt32LE(12);
  if (20 + length > b.length) throw Error("Invalid GLB JSON chunk length");
  const json = JSON.parse(b.toString("utf8", 20, 20 + length));
  const supported = new Set(ALL_EXTENSIONS.map((e) => e.EXTENSION_NAME));
  for (const ext of new Set([
    ...(json.extensionsUsed ?? []),
    ...(json.extensionsRequired ?? []),
  ])) {
    if (!supported.has(ext))
      throw Error(
        `Unsupported extension ${ext}; refusing to silently discard it`,
      );
  }
  for (const item of [...(json.buffers ?? []), ...(json.images ?? [])]) {
    if (item.uri && !item.uri.startsWith("data:"))
      throw Error(
        "Only self-contained GLB is supported; external resources are not fetched",
      );
  }
  return json;
}
export async function stats(document, bytes) {
  const r = document.getRoot();
  let triangles = 0,
    vertices = 0,
    primitives = 0;
  for (const m of r.listMeshes())
    for (const p of m.listPrimitives()) {
      primitives++;
      const v = p.getAttribute("POSITION")?.getCount() ?? 0,
        n = p.getIndices()?.getCount() ?? v;
      vertices += v;
      if (p.getMode() === 4) triangles += n / 3;
      else if (p.getMode() === 5 || p.getMode() === 6)
        triangles += Math.max(0, n - 2);
    }
  const textures = [];
  for (const t of r.listTextures()) {
    const image = t.getImage();
    let size = t.getSize(),
      levels = 1;
    if (t.getMimeType() === "image/ktx2" && image?.length >= 44) {
      const b = Buffer.from(image);
      size = [b.readUInt32LE(20), b.readUInt32LE(24)];
      levels = b.readUInt32LE(40);
    } else if (image && !size) {
      const m = await sharp(image).metadata();
      size = [m.width, m.height];
    }
    textures.push({
      slots: listTextureSlots(t),
      name: t.getName(),
      mimeType: t.getMimeType(),
      size,
      levels,
      bytes: image?.length ?? 0,
    });
  }
  return {
    bytes,
    triangles,
    vertices,
    primitives,
    meshes: r.listMeshes().length,
    morphTargets: r
      .listMeshes()
      .reduce(
        (sum, m) =>
          sum +
          m.listPrimitives().reduce((n, p) => n + p.listTargets().length, 0),
        0,
      ),
    nodes: r.listNodes().length,
    skins: r.listSkins().length,
    animations: r.listAnimations().length,
    textures,
    extensionsRequired: r.listExtensionsRequired().map((e) => e.extensionName),
  };
}
function disposeOrphanAccessors(document) {
  for (const a of document.getRoot().listAccessors())
    if (a.listParents().every((p) => p.propertyType === "Root")) a.dispose();
}
async function processTextures(document, o, warnings, toktx) {
  if (o.textures === "ktx2") {
    try {
      await run(toktx, ["--version"]);
    } catch {
      throw Error(
        "KTX2 requires local KTX-Software toktx. Install it or pass --toktx /absolute/path/to/toktx; no fallback output was written.",
      );
    }
  }
  const temp = await mkdtemp(path.join(tmpdir(), "glb-compress-"));
  try {
    for (const [i, t] of document.getRoot().listTextures().entries()) {
      const image = t.getImage();
      if (!image) continue;
      const oldMime = t.getMimeType();
      if (oldMime === "image/ktx2") {
        if (o.textures === "off" && !o.resizeMax && !o.mipmaps) continue;
        throw Error(
          "Existing KTX2 texture cannot be resized/re-encoded by this tool. Use textures=off, resizeMax=0, mipmaps=false to preserve it.",
        );
      }
      const meta = await sharp(image).metadata();
      const resizing =
        o.resizeMax > 0 && Math.max(meta.width, meta.height) > o.resizeMax;
      if (o.textures === "off" && !resizing) continue;
      let pipeline = sharp(image);
      if (resizing)
        pipeline = pipeline.resize({
          width: o.resizeMax,
          height: o.resizeMax,
          fit: "inside",
          withoutEnlargement: true,
        });
      let format =
        o.textures === "off"
          ? { "image/png": "png", "image/jpeg": "jpeg", "image/webp": "webp" }[
              oldMime
            ]
          : o.textures;
      if (!format) throw Error(`Cannot resize ${oldMime}`);
      if (format === "jpeg" && meta.hasAlpha) {
        format = "png";
        warnings.push(
          `Texture ${i}: kept PNG instead of JPEG to preserve alpha.`,
        );
      }
      let result;
      if (format === "ktx2") {
        const slots = listTextureSlots(t);
        const color = getTextureColorSpace(t) === "srgb";
        if (color && slots.some((s) => !/color|emissive|diffuse/i.test(s)))
          throw Error(
            `Texture ${i} is shared by color and data slots; split it before KTX2 encoding`,
          );
        const png = path.join(temp, `${i}.png`),
          out = path.join(temp, `${i}.ktx2`);
        // PNG chunk defaults must not reinterpret normal/ORM data as sRGB.
        await pipeline.png().toFile(png);
        const pngMeta = await sharp(png).metadata();
        if (pngMeta.width % 4 || pngMeta.height % 4)
          throw Error(
            `Texture ${i}: KHR_texture_basisu requires base dimensions divisible by 4; got ${pngMeta.width}x${pngMeta.height}`,
          );
        const args = [
          "--t2",
          "--2d",
          "--upper_left_maps_to_s0t0",
          "--encode",
          "etc1s",
          "--qlevel",
          String(o.encoding.ktxQuality),
          "--clevel",
          o.preset === "fast" ? "1" : "2",
          "--assign_oetf",
          color ? "srgb" : "linear",
          "--assign_primaries",
          color ? "bt709" : "none",
        ];
        if (o.mipmaps) args.push("--genmipmap");
        args.push(out, png);
        const env = { ...process.env };
        delete env.TOKTX_OPTIONS;
        try {
          const encoded = await run(toktx, args, {
            maxBuffer: 4 * 1024 * 1024,
            env,
          });
          if (encoded.stderr.trim())
            warnings.push(`Texture ${i}: ${encoded.stderr.trim()}`);
        } catch (e) {
          throw Error(`toktx failed: ${e.stderr || e.message}`);
        }
        result = await readFile(out);
        // Some toktx releases ignore unknown encoder names with exit code 0.
        // Verify BasisLZ payload, not merely a KTX2 container or MIME label.
        if (
          result.length < 80 ||
          result.readUInt32LE(12) !== 0 ||
          result.readUInt32LE(44) !== 1
        )
          throw Error("toktx did not produce ETC1S/BasisLZ compressed KTX2");
      } else if (format === "webp")
        result = await pipeline
          .webp({ quality: o.encoding.quality, effort: o.encoding.effort })
          .toBuffer();
      else if (format === "jpeg")
        result = await pipeline
          .jpeg({ quality: o.encoding.quality, chromaSubsampling: "4:4:4" })
          .toBuffer();
      else result = await pipeline.png().toBuffer();
      t.setImage(result).setMimeType(`image/${format}`).setURI("");
    }
  } finally {
    await rm(temp, { recursive: true, force: true });
  }
  // Rebuild texture extension declarations from actual resulting payloads.
  for (const E of [EXTTextureWebP, KHRTextureBasisu]) {
    const mime = E === EXTTextureWebP ? "image/webp" : "image/ktx2";
    const used = document
      .getRoot()
      .listTextures()
      .some((t) => t.getMimeType() === mime);
    if (used) document.createExtension(E).setRequired(true);
    else
      for (const e of document.getRoot().listExtensionsUsed())
        if (e.extensionName === E.EXTENSION_NAME) e.dispose();
  }
  if (o.mipmaps)
    for (const t of document.getRoot().listTextures()) {
      // TextureInfo belongs to materials/extensions, rather than the image itself.
      for (const edge of document.getGraph().listParentEdges(t)) {
        const parent = edge.getParent(),
          getter = `get${edge.getName()[0].toUpperCase()}${edge.getName().slice(1)}Info`;
        if (typeof parent[getter] === "function")
          parent[getter]()?.setMinFilter(9987);
      }
    }
}
export async function compressBytes(
  bytes,
  input = {},
  {
    toktx = process.env.TOKTX ||
      (existsSync(path.join(homedir(), ".local/share/glb-compress/bin/toktx"))
        ? path.join(homedir(), ".local/share/glb-compress/bin/toktx")
        : "toktx"),
  } = {},
) {
  parseGLB(bytes);
  const { requested, effective: o, warnings } = resolveOptions(input);
  const io = await getIO(),
    document = await io.readBinary(new Uint8Array(bytes));
  document.setLogger(new Logger(Logger.Verbosity.SILENT));
  const before = await stats(document, bytes.length);
  // Input compression is decoded by NodeIO. Re-enable only requested codecs.
  for (const e of document.getRoot().listExtensionsUsed())
    if (
      ["KHR_draco_mesh_compression", "EXT_meshopt_compression"].includes(
        e.extensionName,
      )
    )
      e.dispose();
  if (o.stripAnimations)
    for (const a of document.getRoot().listAnimations()) {
      for (const c of a.listChannels()) c.dispose();
      for (const s of a.listSamplers()) s.dispose();
      a.dispose();
    }
  if (o.stripTangents)
    for (const m of document.getRoot().listMeshes())
      for (const p of m.listPrimitives()) {
        p.setAttribute("TANGENT", null);
        for (const t of p.listTargets()) t.setAttribute("TANGENT", null);
      }
  if (o.weld) weldWithTolerance(document, o.weldTolerance);
  if (o.simplify) {
    // meshoptimizer's position-only simplifier cannot bound deformation errors.
    if (
      document.getRoot().listSkins().length ||
      document
        .getRoot()
        .listMeshes()
        .some((m) =>
          m
            .listPrimitives()
            .some((p) => p.listTargets().length || p.getAttribute("JOINTS_0")),
        )
    )
      warnings.push(
        "Skinned/morph simplification uses rest-pose positions and remaps weights/morph attributes. The error bound does not measure animated deformation; review poses in the target renderer.",
      );
    await document.transform(
      simplify({
        simplifier: MeshoptSimplifier,
        ratio: o.simplifyRatio,
        error: o.simplifyError,
      }),
    );
  }
  await processTextures(document, o, warnings, toktx);
  if (!o.keepNames)
    for (const p of document
      .getGraph()
      .listEdges()
      .map((e) => e.getChild()))
      if (typeof p.setName === "function") p.setName("");
  disposeOrphanAccessors(document);
  if (o.geometry === "on") {
    const e = o.encoding;
    document
      .createExtension(KHRDracoMeshCompression)
      .setRequired(true)
      .setEncoderOptions({
        method: KHRDracoMeshCompression.EncoderMethod.EDGEBREAKER,
        encodeSpeed: e.encodeSpeed,
        decodeSpeed: e.decodeSpeed,
        quantizationBits: {
          POSITION: e.position,
          NORMAL: e.normal,
          TEX_COORD: e.texcoord,
          COLOR: e.color,
          GENERIC: e.generic,
        },
      });
  }
  // Compaction may create accessors without a buffer. Consolidate into the GLB.
  const buffer = document.getRoot().listBuffers()[0] ?? document.createBuffer();
  for (const a of document.getRoot().listAccessors()) a.setBuffer(buffer);
  for (const b of document.getRoot().listBuffers())
    if (b !== buffer) b.dispose();
  const output = await io.writeBinary(document);
  parseGLB(output);
  // Read back compressed output, so counts reflect Draco's actual vertex stream.
  const after = await stats(await io.readBinary(output), output.length);
  const retained = before.triangles ? after.triangles / before.triangles : null;
  if (o.simplify && retained !== null && retained > o.simplifyRatio + 0.005)
    warnings.push(
      `Simplification retained ${(retained * 100).toFixed(2)}%; target ${(o.simplifyRatio * 100).toFixed(0)}% was limited by topology/error.`,
    );
  if (output.length >= bytes.length)
    warnings.push(
      "Output is not smaller than input. Keep the original when size is the only objective.",
    );
  return {
    output,
    report: {
      tool: "glb-compress",
      version: "1.0.0",
      requested,
      effective: o,
      before,
      after,
      savedBytes: bytes.length - output.length,
      savedPercent: (1 - output.length / bytes.length) * 100,
      actualTriangleRetainRatio: retained,
      ...(o.textures === "ktx2" && after.textures.length
        ? {
            review: {
              status: "awaiting_user_comparison",
              humanApprovalRequired: true,
              decision: null,
              selected: null,
              requiredEvidence: [
                "Actual candidate loaded in target renderer",
                "Original/candidate same-camera normal and closest-distance views",
                "PBR under different light directions and alpha edges where present",
                "Bytes, dimensions, encoding and observed differences",
              ],
              adoption:
                "Wait for the user to choose this exact candidate; do not import, replace production assets or mark it accepted automatically.",
            },
          }
        : {}),
      warnings,
    },
  };
}
