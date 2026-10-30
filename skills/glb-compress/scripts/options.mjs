export const defaults = Object.freeze({
  preset: "balanced",
  geometry: "on",
  textures: "ktx2",
  resizeMax: 0,
  simplify: "off",
  simplifyRatio: 0.75,
  simplifyError: 0.0001,
  weld: "off",
  weldTolerance: 0.0001,
  keepNames: true,
  stripAnimations: false,
  stripTangents: false,
  mipmaps: false,
  preservePrecision: false,
});
export const presets = Object.freeze({
  balanced: {
    quality: 80,
    effort: 4,
    encodeSpeed: 5,
    decodeSpeed: 5,
    position: 14,
    normal: 10,
    texcoord: 12,
    color: 8,
    generic: 12,
    ktxQuality: 128,
  },
  max: {
    quality: 70,
    effort: 6,
    encodeSpeed: 0,
    decodeSpeed: 0,
    position: 12,
    normal: 8,
    texcoord: 10,
    color: 8,
    generic: 10,
    ktxQuality: 96,
  },
  fast: {
    quality: 80,
    effort: 0,
    encodeSpeed: 10,
    decodeSpeed: 10,
    position: 14,
    normal: 10,
    texcoord: 12,
    color: 8,
    generic: 12,
    ktxQuality: 128,
  },
  none: {
    quality: 80,
    effort: 4,
    encodeSpeed: 5,
    decodeSpeed: 5,
    position: 14,
    normal: 10,
    texcoord: 12,
    color: 8,
    generic: 12,
    ktxQuality: 128,
  },
});
export const schema = {
  defaults,
  enums: {
    preset: Object.keys(presets),
    geometry: ["on", "off"],
    textures: ["webp", "jpeg", "off", "ktx2"],
    simplify: ["off", "light", "balanced", "aggressive"],
    weld: ["off", "light", "standard", "aggressive"],
  },
  websiteResizeChoices: [0, 512, 1024, 2048],
  simplifyRatios: { light: 0.9, balanced: 0.75, aggressive: 0.6 },
  weldTolerances: { light: 0.00001, standard: 0.0001, aggressive: 0.001 },
  localPresetEncoding: presets,
};
export function resolveOptions(input = {}) {
  for (const k of Object.keys(input))
    if (!Object.hasOwn(defaults, k))
      throw Error(`Unknown compression option: ${k}`);
  const o = { ...defaults, ...input };
  for (const [k, v] of Object.entries(defaults)) {
    if (typeof v === "boolean" && typeof o[k] !== "boolean")
      throw Error(`${k} must be true or false`);
  }
  for (const [k, values] of Object.entries(schema.enums)) {
    if ((k === "simplify" || k === "weld") && typeof o[k] === "boolean")
      continue;
    if (!values.includes(o[k]))
      throw Error(`${k} must be one of ${values.join(", ")}`);
  }
  for (const k of [
    "resizeMax",
    "simplifyRatio",
    "simplifyError",
    "weldTolerance",
  ]) {
    if (typeof o[k] !== "number" || !Number.isFinite(o[k]))
      throw Error(`${k} must be a finite number`);
  }
  if (!Number.isInteger(o.resizeMax) || o.resizeMax < 0 || o.resizeMax > 32768)
    throw Error("resizeMax must be an integer from 0 to 32768 (0 = no limit)");
  if (o.simplifyRatio <= 0 || o.simplifyRatio > 1)
    throw Error("simplifyRatio must be > 0 and <= 1");
  if (o.simplifyError < 0 || o.simplifyError > 1)
    throw Error("simplifyError must be between 0 and 1");
  if (o.weldTolerance <= 0 || o.weldTolerance > 1)
    throw Error("weldTolerance must be > 0 and <= 1 model unit");
  const warnings = [];
  const effective = { ...o, encoding: { ...presets[o.preset] } };
  effective.simplify = o.simplify !== "off" && o.simplify !== false;
  effective.weld = o.weld !== "off" && o.weld !== false;
  effective.simplifyRatio =
    schema.simplifyRatios[o.simplify] ?? o.simplifyRatio;
  effective.weldTolerance = schema.weldTolerances[o.weld] ?? o.weldTolerance;
  if (o.preset === "none") effective.geometry = "off";
  if (o.preservePrecision) {
    if (effective.simplify || effective.weld)
      throw Error(
        "preservePrecision conflicts with simplify or tolerance weld; disable them explicitly",
      );
    if (effective.geometry === "on")
      warnings.push(
        "preservePrecision disables Draco to preserve geometry values; texture processing is independent.",
      );
    effective.geometry = "off";
  }
  if (o.mipmaps && o.textures !== "ktx2")
    warnings.push(
      "PNG/JPEG/WebP cannot embed mip chains; mipmaps=true requests runtime trilinear filtering only. KTX2 embeds the levels.",
    );
  return { requested: o, effective, warnings };
}
