import { existsSync, statSync, realpathSync } from "node:fs";
import path from "node:path";

// Project detection accepts both engine roots and repositories containing game/.
export function projectContext({
  project,
  input,
  cwd = process.cwd(),
  engine = "auto",
} = {}) {
  if (!["auto", "godot", "generic"].includes(engine))
    throw Error("engine must be auto, godot or generic");
  if (project && (!existsSync(project) || !statSync(project).isDirectory()))
    throw Error("project must be an existing directory");
  const starts = project
    ? [project]
    : [input ? path.dirname(path.resolve(input)) : null, cwd].filter(Boolean);
  for (const start of starts) {
    let current = realpathSync(start);
    while (true) {
      for (const candidate of [
        path.join(current, "project.godot"),
        path.join(current, "game/project.godot"),
      ]) {
        if (existsSync(candidate)) {
          if (engine === "generic")
            throw Error(
              "Detected a Godot project; engine=generic cannot bypass its no-Draco rule",
            );
          return {
            engine: "godot",
            projectRoot: path.dirname(candidate),
            detectedBy: candidate,
          };
        }
      }
      const parent = path.dirname(current);
      if (parent === current) break;
      current = parent;
    }
  }
  return {
    engine: engine === "godot" ? "godot" : "generic",
    projectRoot: project ? path.resolve(project) : null,
    detectedBy:
      engine === "auto" ? "no-project.godot-found" : "explicit-engine",
  };
}
export function enforceProjectPolicy(options, context) {
  if (context.engine !== "godot") return { ...options };
  if (options.geometry === "on")
    throw Error("Godot compression policy forbids Draco: use --geometry off");
  return { ...options, geometry: "off" };
}

export function recommendCompression(
  stats,
  context,
  { usage = "standard" } = {},
) {
  if (!["standard", "close", "distant"].includes(usage))
    throw Error("usage must be standard, close or distant");
  const textureBytes = stats.textures.reduce((sum, t) => sum + t.bytes, 0);
  const share = stats.bytes ? textureBytes / stats.bytes : 0;
  const textureAssessment = stats.textures
    .map((t, index) => ({
      index,
      name: t.name ?? "",
      slots: t.slots ?? [],
      mimeType: t.mimeType ?? null,
      size: t.size ?? null,
      bytes: t.bytes,
      containerShare: stats.bytes ? t.bytes / stats.bytes : 0,
      suggestedChecks: [
        ...(["image/png", "image/webp"].includes(t.mimeType)
          ? [
              "Measure original-size lossless re-encoding; verify decoded pixels and data interpretation before claiming lossless savings.",
            ]
          : []),
        "Choose encoding/resolution from actual size savings and material appearance; retain original when benefit is insufficient.",
      ],
    }))
    .sort((a, b) => b.bytes - a.bytes || a.index - b.index);
  const band =
    stats.triangles < 10000
      ? "low"
      : stats.triangles < 100000
        ? "medium"
        : "high";
  const maxSize = Math.max(
    0,
    ...stats.textures.flatMap((t) => t.size ?? [0, 0]),
  );
  const animated =
    stats.skins > 0 || stats.animations > 0 || (stats.morphTargets ?? 0) > 0;
  const baseline = {
    geometry: "off",
    textures: "off",
    resizeMax: 0,
    simplify: "off",
    weld: "off",
    keepNames: true,
    stripAnimations: false,
    stripTangents: false,
  };
  const reasons = [
    `${stats.triangles} triangles (${band} triage band); textures ${(share * 100).toFixed(1)}% of container bytes.`,
    "Inspect the ranked per-texture assessment first. No texture type is automatically excluded or selected; bulk codec profiles are exploratory candidates, not a delivery recommendation.",
  ];
  const candidates = [];
  if (stats.textures.length) {
    candidates.push({
      id: "ktx2",
      options: { ...baseline, textures: "ktx2", resizeMax: 0 },
      role: "default KTX2 candidate at source resolution; user visual comparison required before adoption",
    });
    candidates.push({
      id: "webp2k",
      options: { ...baseline, textures: "webp", resizeMax: 2048 },
      role: "exploratory whole-file candidate; assess per-texture suitability before execution",
    });
    if (maxSize > 1024)
      candidates.push({
        id: "webp1k",
        options: { ...baseline, textures: "webp", resizeMax: 1024 },
        role: "smaller candidate; compare at maximum intended screen size",
      });
    else
      reasons.push(
        "Source textures are already <=1K; 1K and 2K caps would not resize them. Compare encoding-only against original.",
      );
    candidates.push({
      id: "webp-original-size",
      options: { ...baseline, textures: "webp", resizeMax: 0 },
      role: "fallback if resolution reduction visibly loses detail",
    });
    if (maxSize <= 2048) candidates.pop();
    if (maxSize > 4096)
      candidates.push({
        id: "webp4k",
        options: { ...baseline, textures: "webp", resizeMax: 4096 },
        role: "detail fallback when 2K still blurs fine patterns",
      });
  }
  if (band === "low")
    reasons.push(
      "Keep low-poly topology; prioritize texture encoding/resolution. No automatic simplification or tolerance welding.",
    );
  if (share < 0.5)
    reasons.push(
      "Most bytes are outside images; texture optimization alone has a limited ceiling. Inspect geometry/animation data before expecting large savings.",
    );
  if (band !== "low" && !animated) {
    candidates.push({
      id: "geometry90",
      options: { ...baseline, simplify: "light" },
      role: "optional independent geometry trial; not an automatic delivery choice",
    });
  }
  if (animated)
    reasons.push(
      "Preserve skins, animation, morphs, tangents and names. Geometry reduction requires a separate deformation comparison; do not make it a default optimization.",
    );
  if (context.engine === "godot")
    reasons.push(
      "Godot: Draco is disabled for every candidate, including high-poly models.",
    );
  reasons.push(
    usage === "close"
      ? "Close-up use: compare at the closest intended distance; retain needed resolution and material detail."
      : usage === "distant"
        ? "Distant use: lower resolutions may help; inspect alpha edges and the closest allowed camera distance."
        : "Choose encoding and resolution only after inspecting actual payloads and measuring savings and same-camera visual changes.",
  );
  return {
    policyVersion: "1.2.0",
    context,
    usage,
    analysis: {
      triangles: stats.triangles,
      triangleBand: band,
      textureBytes,
      textureShare: share,
      nonTextureBytes: stats.bytes - textureBytes,
      maxTextureDimension: maxSize,
      animated,
    },
    status: "candidates_require_visual_comparison",
    nextStep: stats.textures.length
      ? "generate_ktx2_candidate_then_wait_for_user_visual_comparison"
      : "inspect_actual_texture_payloads_and_measure_relevant_candidates",
    preferredTrial: stats.textures.length ? "ktx2" : null,
    selected: null,
    textureAssessment,
    original: {
      id: "original",
      role: "reference and valid final choice when compression worsens appearance or saves little",
    },
    candidates,
    reasons,
    review: {
      humanApprovalRequired: stats.textures.length > 0,
      status: stats.textures.length
        ? "awaiting_comparison_evidence_and_user_choice"
        : "not_applicable",
      decision: null,
      acceptance:
        "Only the user's explicit choice of the current candidate authorizes adoption; a new output invalidates previous approval.",
      views: [
        "normal gameplay distance",
        "closest intended distance",
        "alpha edges and normal/roughness detail",
        "representative animation poses when present",
      ],
      select:
        "Smallest candidate with no meaningful loss at intended viewing conditions; if none passes, retain original.",
      requiredEvidence: [
        "actual size and decoded counts",
        "same-camera original/candidate renders",
        "target renderer and screen occupancy",
        "observed detail/color/alpha/silhouette differences",
      ],
    },
  };
}
