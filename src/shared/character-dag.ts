import type { StepStatus, WorkflowRun, WorkflowStep } from "./types.js";

export type CharacterStageGroup =
  | "reference"
  | "candidate"
  | "texture"
  | "proportions"
  | "rig"
  | "delivery";

export type TextureGroupId =
  | "hair"
  | "head"
  | "upper_body"
  | "lower_body"
  | "accessories";

export type MultiviewSlotKey =
  | "original"
  | "head_front"
  | "head_left"
  | "head_right"
  | "head_back"
  | "body_front"
  | "body_left"
  | "body_right"
  | "body_back"
  | "balance_front"
  | "balance_left"
  | "balance_back";

export interface CharacterAgentLease {
  agentId: string;
  agentName: string;
  claimedAt: string;
  expiresAt: string;
  progress: number;
  message: string;
}

export interface CharacterDagNode {
  key: string;
  title: string;
  stageGroup: CharacterStageGroup;
  mode: "agent" | "manual" | "review" | "local";
  status: StepStatus | "running" | "failed";
  dependsOn: string[];
  parallelGroup?: string;
  legacyStepKey: string;
  agentSkill?: string;
  description: string;
  inputRevisionIds: string[];
  outputRevisionIds: string[];
  inputStamp: string;
  completedInputStamp?: string;
  stale: boolean;
  note: string;
  lease?: CharacterAgentLease;
  metrics?: Record<string, string | number | boolean>;
  updatedAt: string;
}

export interface CharacterViewSlot {
  key: MultiviewSlotKey;
  label: string;
  group: "source" | "head" | "body" | "balance";
  angle: "original" | "front" | "left" | "right" | "back";
  revisionId?: string;
  fileIndex?: number;
  relativePath?: string;
  resolution?: string;
  alignmentScore?: number;
  editPrompt?: string;
  updatedAt?: string;
}

export interface CharacterCandidateSlot {
  id: string;
  index: number;
  source: "head" | "body";
  label: string;
  status: "pending" | "ready" | "failed";
  revisionId?: string;
  fileIndex?: number;
  relativePath?: string;
  previewRevisionId?: string;
  vertices?: number;
  faces?: number;
  componentsCount?: number;
  topologyGrade?: string;
  summary?: string;
  updatedAt?: string;
}

export interface CharacterMeshComponent {
  id: string;
  name: string;
  source: "head" | "body";
  vertices: number;
  faces: number;
  centroid: [number, number, number];
  boundsMin: [number, number, number];
  boundsMax: [number, number, number];
  textureGroup: TextureGroupId;
  springGroupId?: string;
  materialName?: string;
}

export interface CharacterTextureGroupState {
  id: TextureGroupId;
  label: string;
  source: "head" | "body" | "mixed";
  componentNames: string[];
  status: "pending" | "uv_ready" | "retextured";
  uvGuideRevisionId?: string;
  referenceRevisionId?: string;
  sourceModelRevisionId?: string;
  texturedModelRevisionId?: string;
  textureImageRevisionId?: string;
  resolution: string;
  mtoonConverted: boolean;
  shadeColor: string;
  outlineWidth: number;
  notes: string;
}

export interface CharacterSpringGroup {
  id: string;
  name: string;
  category: "hair" | "cloth" | "ribbon" | "accessory";
  componentNames: string[];
  stiffness: number;
  gravityPower: number;
  dragForce: number;
  hitRadius: number;
  colliderNames: string[];
}

export interface PartTransform {
  x: number;
  y: number;
  z: number;
  scale: number;
}

export interface JointMarkerPoint {
  key: string;
  label: string;
  group: "body" | "left_hand" | "right_hand";
  x: number;
  y: number;
  z: number;
  confidence: number;
}

export interface CharacterPlacementState {
  targetHeightMeters: number;
  linkHairWithHead: boolean;
  rigMode: "marker" | "tripo";
  overlayOpacity: number;
  showHeightRuler: boolean;
  showHorizontalGuides: boolean;
  parts: {
    head: PartTransform;
    hair: PartTransform;
    body: PartTransform;
  };
  bodyMarkers: JointMarkerPoint[];
  fingerMarkers: JointMarkerPoint[];
}

export interface CharacterRiggingState {
  humanoidBoneCount: number;
  mappedBones: Array<{
    humanoidBone: string;
    sourceBone: string;
    status: "mapped" | "optional_missing";
  }>;
  weightTransfer: {
    donorMesh: string;
    method: "nearest_surface_smoothed";
    smoothedSeams: string[];
    maxInfluences: number;
    status: "pending" | "completed";
  };
  springVerification: {
    framesSimulated: number;
    physicsOnOffVerified: boolean;
    chains: Array<{
      groupId: string;
      name: string;
      boneCount: number;
      peakDisplacementMeters: number;
      settledFrame: number;
      colliderHits: number;
      passed: boolean;
    }>;
  };
}

export interface CharacterVrmMeta {
  name: string;
  version: string;
  authors: string[];
  copyrightInformation: string;
  contactInformation: string;
  references: string[];
  thirdPartyLicenses: string;
  thumbnailRevisionId?: string;
  licenseUrl: string;
  avatarPermission: "onlyAuthor" | "onlySeparatelyLicensedPerson" | "everyone";
  commercialUsage: "personalNonProfit" | "personalProfit" | "corporation";
  creditNotation: "required" | "unnecessary";
  allowRedistribution: boolean;
  modify:
    | "prohibited"
    | "allowModification"
    | "allowModificationRedistribution";
}

export interface CharacterVrmDeliveryState {
  meta: CharacterVrmMeta;
  vrmRevisionId?: string;
  blendRevisionId?: string;
  previewGlbRevisionId?: string;
  motionClip: string;
  physicsEnabled: boolean;
  showBones: boolean;
  wireframe: boolean;
}

export interface CharacterWorkflowState {
  characterName: string;
  stylePrompt: string;
  views: Record<MultiviewSlotKey, CharacterViewSlot>;
  headCandidates: CharacterCandidateSlot[];
  bodyCandidates: CharacterCandidateSlot[];
  selectedHeadCandidateId: string | null;
  selectedBodyCandidateId: string | null;
  segmentation: {
    status: "idle" | "ready";
    totalVertices: number;
    totalFaces: number;
    components: CharacterMeshComponent[];
  };
  textureGroups: Record<TextureGroupId, CharacterTextureGroupState>;
  springGroups: CharacterSpringGroup[];
  placement: CharacterPlacementState;
  rigging: CharacterRiggingState;
  vrmDelivery: CharacterVrmDeliveryState;
  importedCharakit?: {
    projectId: string;
    name: string;
    sourcePath: string;
    importedAt: string;
    fileCount: number;
  };
}

export const TEXTURE_GROUP_IDS: TextureGroupId[] = [
  "hair",
  "head",
  "upper_body",
  "lower_body",
  "accessories",
];

export const TEXTURE_GROUP_LABELS: Record<TextureGroupId, string> = {
  hair: "头发 (Hair)",
  head: "头部与面容 (Head)",
  upper_body: "上装与躯干 (Upper Body)",
  lower_body: "下装与腿足 (Lower Body)",
  accessories: "配饰与道具 (Accessories)",
};

export const MULTIVIEW_SLOT_META: Array<{
  key: MultiviewSlotKey;
  label: string;
  group: CharacterViewSlot["group"];
  angle: CharacterViewSlot["angle"];
}> = [
  { key: "original", label: "原画立绘", group: "source", angle: "original" },
  { key: "head_front", label: "头部正面", group: "head", angle: "front" },
  { key: "head_left", label: "头部左侧", group: "head", angle: "left" },
  { key: "head_right", label: "头部右侧", group: "head", angle: "right" },
  { key: "head_back", label: "头部背面", group: "head", angle: "back" },
  { key: "body_front", label: "身体正面", group: "body", angle: "front" },
  { key: "body_left", label: "身体左侧", group: "body", angle: "left" },
  { key: "body_right", label: "身体右侧", group: "body", angle: "right" },
  { key: "body_back", label: "身体背面", group: "body", angle: "back" },
  { key: "balance_front", label: "全身平衡正面", group: "balance", angle: "front" },
  { key: "balance_left", label: "全身平衡侧面", group: "balance", angle: "left" },
  { key: "balance_back", label: "全身平衡背面", group: "balance", angle: "back" },
];

export const DEFAULT_BODY_MARKERS: JointMarkerPoint[] = [
  { key: "head", label: "头顶 (Head)", group: "body", x: 0.0, y: 1.46, z: 0.0, confidence: 0.99 },
  { key: "neck", label: "颈部 (Neck)", group: "body", x: 0.0, y: 1.31, z: 0.0, confidence: 0.99 },
  { key: "chest", label: "上胸 (Chest)", group: "body", x: 0.0, y: 1.18, z: 0.0, confidence: 0.98 },
  { key: "spine", label: "脊椎 (Spine)", group: "body", x: 0.0, y: 1.03, z: 0.0, confidence: 0.98 },
  { key: "hips", label: "骨盆 (Hips)", group: "body", x: 0.0, y: 0.89, z: 0.0, confidence: 0.99 },
  { key: "leftShoulder", label: "左肩 (L.Shoulder)", group: "body", x: 0.09, y: 1.28, z: 0.0, confidence: 0.97 },
  { key: "leftUpperArm", label: "左上臂 (L.UpperArm)", group: "body", x: 0.16, y: 1.25, z: 0.0, confidence: 0.97 },
  { key: "leftLowerArm", label: "左前臂 (L.LowerArm)", group: "body", x: 0.31, y: 1.09, z: 0.0, confidence: 0.96 },
  { key: "leftHand", label: "左腕 (L.Hand)", group: "body", x: 0.45, y: 0.93, z: 0.02, confidence: 0.95 },
  { key: "rightShoulder", label: "右肩 (R.Shoulder)", group: "body", x: -0.09, y: 1.28, z: 0.0, confidence: 0.97 },
  { key: "rightUpperArm", label: "右上臂 (R.UpperArm)", group: "body", x: -0.16, y: 1.25, z: 0.0, confidence: 0.97 },
  { key: "rightLowerArm", label: "右前臂 (R.LowerArm)", group: "body", x: -0.31, y: 1.09, z: 0.0, confidence: 0.96 },
  { key: "rightHand", label: "右腕 (R.Hand)", group: "body", x: -0.45, y: 0.93, z: 0.02, confidence: 0.95 },
  { key: "leftUpperLeg", label: "左大腿 (L.UpperLeg)", group: "body", x: 0.09, y: 0.85, z: 0.0, confidence: 0.98 },
  { key: "leftLowerLeg", label: "左膝 (L.LowerLeg)", group: "body", x: 0.09, y: 0.48, z: 0.01, confidence: 0.98 },
  { key: "leftFoot", label: "左踝 (L.Foot)", group: "body", x: 0.09, y: 0.09, z: -0.01, confidence: 0.97 },
  { key: "leftToes", label: "左趾 (L.Toes)", group: "body", x: 0.09, y: 0.02, z: 0.09, confidence: 0.95 },
  { key: "rightUpperLeg", label: "右大腿 (R.UpperLeg)", group: "body", x: -0.09, y: 0.85, z: 0.0, confidence: 0.98 },
  { key: "rightLowerLeg", label: "右膝 (R.LowerLeg)", group: "body", x: -0.09, y: 0.48, z: 0.01, confidence: 0.98 },
  { key: "rightFoot", label: "右踝 (R.Foot)", group: "body", x: -0.09, y: 0.09, z: -0.01, confidence: 0.97 },
  { key: "rightToes", label: "右趾 (R.Toes)", group: "body", x: -0.09, y: 0.02, z: 0.09, confidence: 0.95 },
];

export function createDefaultFingerMarkers(): JointMarkerPoint[] {
  const fingers = [
    { key: "thumb", label: "拇指", zOffset: 0.04 },
    { key: "index", label: "食指", zOffset: 0.02 },
    { key: "middle", label: "中指", zOffset: 0.0 },
    { key: "ring", label: "无名指", zOffset: -0.02 },
    { key: "little", label: "小指", zOffset: -0.04 },
  ];
  const joints = ["metacarpal", "proximal", "intermediate", "distal"];
  const points: JointMarkerPoint[] = [];
  for (const side of ["left", "right"] as const) {
    const sign = side === "left" ? 1 : -1;
    for (const f of fingers) {
      for (let i = 0; i < joints.length; i++) {
        points.push({
          key: `${side}_${f.key}_${joints[i]}`,
          label: `${side === "left" ? "左" : "右"}${f.label} J${i + 1}`,
          group: side === "left" ? "left_hand" : "right_hand",
          x: Number((sign * (0.47 + i * 0.018)).toFixed(4)),
          y: Number((0.91 - i * 0.012).toFixed(4)),
          z: Number((0.02 + f.zOffset).toFixed(4)),
          confidence: 0.94,
        });
      }
    }
  }
  return points;
}

export const CHARACTER_DAG_TEMPLATE: Array<{
  key: string;
  title: string;
  stageGroup: CharacterStageGroup;
  mode: CharacterDagNode["mode"];
  dependsOn: string[];
  parallelGroup?: string;
  legacyStepKey: string;
  agentSkill?: string;
  description: string;
}> = [
  {
    key: "source_illustration",
    title: "原画与角色企划",
    stageGroup: "reference",
    mode: "agent",
    dependsOn: [],
    legacyStepKey: "brief",
    agentSkill: "character-workflow",
    description: "冻结角色原画立绘、设定说明与目标头身比，生成统一设计基准。",
  },
  {
    key: "head_multiview",
    title: "头部四视图生成",
    stageGroup: "reference",
    mode: "agent",
    dependsOn: ["source_illustration"],
    parallelGroup: "multiview",
    legacyStepKey: "reference",
    agentSkill: "charakuru-multi-view-images",
    description: "生成前/左/右/后 4 张仅含头部与头发的正交白底图（眼线/耳线严格水平对齐）。",
  },
  {
    key: "body_multiview",
    title: "身体四视图生成",
    stageGroup: "reference",
    mode: "agent",
    dependsOn: ["source_illustration"],
    parallelGroup: "multiview",
    legacyStepKey: "reference",
    agentSkill: "charakuru-multi-view-images",
    description: "生成前/左/右/后 4 张颈部以下去头 A-Pose 正交白底图（肩/腰/臀/裙摆水平对齐）。",
  },
  {
    key: "balance_three_views",
    title: "全身平衡参考三视图",
    stageGroup: "reference",
    mode: "agent",
    dependsOn: ["head_multiview", "body_multiview"],
    legacyStepKey: "reference",
    agentSkill: "charakuru-multi-view-images",
    description: "合成全身前/左/后 3 张头身比例平衡图，供后续 3D 对位与身高标尺对齐。",
  },
  {
    key: "images_gate",
    title: "多视图与对齐审核",
    stageGroup: "reference",
    mode: "review",
    dependsOn: ["balance_three_views"],
    legacyStepKey: "reference",
    description: "使用水平对齐参考线审核 8 张部位视图与 3 张全身平衡图；支持单张重绘或替换。",
  },
  {
    key: "head_candidates",
    title: "头部 3D 候选生成 ×4",
    stageGroup: "candidate",
    mode: "agent",
    dependsOn: ["images_gate"],
    parallelGroup: "candidates_3d",
    legacyStepKey: "candidate",
    agentSkill: "charakuru-tripo-3d-candidates",
    description: "基于头部四视图并行生成 4 个高模头部 3D 候选（head-1 ~ head-4）。",
  },
  {
    key: "body_candidates",
    title: "身体 3D 候选生成 ×4",
    stageGroup: "candidate",
    mode: "agent",
    dependsOn: ["images_gate"],
    parallelGroup: "candidates_3d",
    legacyStepKey: "candidate",
    agentSkill: "charakuru-tripo-3d-candidates",
    description: "基于身体四视图并行生成 4 个高模 A-Pose 身体 3D 候选（body-1 ~ body-4）。",
  },
  {
    key: "candidates_gate",
    title: "头/身候选四选一审核",
    stageGroup: "candidate",
    mode: "review",
    dependsOn: ["head_candidates", "body_candidates"],
    legacyStepKey: "appearance",
    description: "显式选定采用哪一个头部候选与身体候选；贴图后切换候选会自动派生非破坏性新分支。",
  },
  {
    key: "mesh_segmentation",
    title: "网格部件拆解与连通域分析",
    stageGroup: "texture",
    mode: "local",
    dependsOn: ["candidates_gate"],
    legacyStepKey: "assembly",
    agentSkill: "charakuru-3d-part-segmentation",
    description: "对选定的头部与身体网格执行部件边界拆解，统计每个连通部件的顶点数、面数与包围盒。",
  },
  {
    key: "part_classification",
    title: "五分类贴图组与摇摆物归类",
    stageGroup: "texture",
    mode: "agent",
    dependsOn: ["mesh_segmentation"],
    legacyStepKey: "assembly",
    agentSkill: "charakuru-part-classification",
    description: "将全部网格部件归入 5 个标准贴图组（hair/head/upper_body/lower_body/accessories）并提取独立物理摇摆组。",
  },
  {
    key: "uv_guide_prep",
    title: "UV 展开与色块引导图烘培",
    stageGroup: "texture",
    mode: "local",
    dependsOn: ["part_classification"],
    legacyStepKey: "assembly",
    agentSkill: "charakuru-blender-texture-prep",
    description: "按 5 大贴图组整理无重叠单材质 UV，导出色块引导图（Guide）与参考拼图。",
  },
  {
    key: "retexture_hair",
    title: "头发重贴图 (4K)",
    stageGroup: "texture",
    mode: "agent",
    dependsOn: ["uv_guide_prep"],
    parallelGroup: "retexture",
    legacyStepKey: "assembly",
    agentSkill: "charakuru-tripo-retexture",
    description: "基于头发 UV 引导图与头部多视图生成 4K 头发贴图及高光层次。",
  },
  {
    key: "retexture_head",
    title: "面部与头部重贴图 (4K)",
    stageGroup: "texture",
    mode: "agent",
    dependsOn: ["uv_guide_prep"],
    parallelGroup: "retexture",
    legacyStepKey: "assembly",
    agentSkill: "charakuru-tripo-retexture",
    description: "生成五官、虹膜高光与肤色过渡清晰的 4K 头部贴图。",
  },
  {
    key: "retexture_upper_body",
    title: "上装与躯干重贴图 (4K)",
    stageGroup: "texture",
    mode: "agent",
    dependsOn: ["uv_guide_prep"],
    parallelGroup: "retexture",
    legacyStepKey: "assembly",
    agentSkill: "charakuru-tripo-retexture",
    description: "生成领口、袖口、胸前纹样与布料接缝清晰的 4K 上装贴图。",
  },
  {
    key: "retexture_lower_body",
    title: "下装与鞋袜重贴图 (4K)",
    stageGroup: "texture",
    mode: "agent",
    dependsOn: ["uv_guide_prep"],
    parallelGroup: "retexture",
    legacyStepKey: "assembly",
    agentSkill: "charakuru-tripo-retexture",
    description: "生成裙褶、裤装、袜口与鞋靴细节一致的 4K 下装贴图。",
  },
  {
    key: "retexture_accessories",
    title: "配饰与道具重贴图 (4K)",
    stageGroup: "texture",
    mode: "agent",
    dependsOn: ["uv_guide_prep"],
    parallelGroup: "retexture",
    legacyStepKey: "assembly",
    agentSkill: "charakuru-tripo-retexture",
    description: "生成发饰、蝴蝶结、金属扣件与随身道具的 4K 配饰贴图。",
  },
  {
    key: "mtoon_assembly",
    title: "MToon 卡通材质与总装合并",
    stageGroup: "texture",
    mode: "local",
    dependsOn: [
      "retexture_hair",
      "retexture_head",
      "retexture_upper_body",
      "retexture_lower_body",
      "retexture_accessories",
    ],
    legacyStepKey: "assembly",
    description: "将 5 组 4K 贴图装配为二次元 MToon 材质（含阴影色与描边），输出可编辑总装模型。",
  },
  {
    key: "proportions_gate",
    title: "头身比例对齐与关节 Marker 审核",
    stageGroup: "proportions",
    mode: "review",
    dependsOn: ["mtoon_assembly"],
    legacyStepKey: "assembly",
    description: "对照半透明全身平衡三视图与米制身高标尺调节头/发/身位移缩放，并校核 21 体关节 + 40 手指 Marker。",
  },
  {
    key: "humanoid_rig",
    title: "Humanoid 骨骼生成与映射",
    stageGroup: "rig",
    mode: "agent",
    dependsOn: ["proportions_gate"],
    legacyStepKey: "rig",
    agentSkill: "charakuru-blender-rig-vrm",
    description: "依据关节 Marker 或 Tripo 自动骨骼构建 55 根 VRM 1.0 Humanoid 标准骨架映射。",
  },
  {
    key: "weight_transfer",
    title: "素体蒙皮权重传递与接缝平滑",
    stageGroup: "rig",
    mode: "local",
    dependsOn: ["humanoid_rig"],
    legacyStepKey: "rig",
    agentSkill: "charakuru-blender-rig-vrm",
    description: "从标准人形素体传递肩/肘/髋/膝平滑蒙皮权重，消除跨部件破面与顶点撕裂。",
  },
  {
    key: "spring_verify",
    title: "SpringBone 物理链与 90 帧自动检验",
    stageGroup: "rig",
    mode: "local",
    dependsOn: ["weight_transfer"],
    legacyStepKey: "rig",
    agentSkill: "charakuru-blender-rig-vrm",
    description: "为独立摇摆组绑定 SpringBone 骨骼链与防穿模碰撞体，运行 90 帧物理开关对比测试。",
  },
  {
    key: "motion_gate",
    title: "动作预览与物理防穿模审核",
    stageGroup: "rig",
    mode: "review",
    dependsOn: ["spring_verify"],
    legacyStepKey: "deformation",
    description: "加载步行与自定义 VRMA 动作，在线切换物理 ON/OFF、骨骼显示与线框模式，确认変形与摇摆自然。",
  },
  {
    key: "vrm_export",
    title: "VRM 1.0 元数据写入与交付归档",
    stageGroup: "delivery",
    mode: "local",
    dependsOn: ["motion_gate"],
    legacyStepKey: "delivery",
    description: "写入 VRM 1.0 作者、商业授权与缩略图元数据，打包导出 .vrm、.blend 与不可变交付清单。",
  },
];

export function createInitialCharacterWorkflowState(
  characterName = "新角色",
  stylePrompt = "",
  initialImageRevisionId?: string,
  initialImagePath?: string,
): CharacterWorkflowState {
  const views = {} as Record<MultiviewSlotKey, CharacterViewSlot>;
  for (const meta of MULTIVIEW_SLOT_META) {
    views[meta.key] = {
      key: meta.key,
      label: meta.label,
      group: meta.group,
      angle: meta.angle,
      resolution: meta.key === "original" ? "2048x2048" : "2048x2048 (1:1 白底)",
      alignmentScore: meta.key === "original" ? 1 : 0.98,
    };
  }
  if (initialImageRevisionId) {
    views.original.revisionId = initialImageRevisionId;
    views.original.fileIndex = 0;
    views.original.relativePath = initialImagePath;
    views.original.updatedAt = new Date().toISOString();
  }

  const headCandidates: CharacterCandidateSlot[] = [1, 2, 3, 4].map((index) => ({
    id: `head-${index}`,
    index,
    source: "head",
    label: `头部候选 #${index}`,
    status: "pending",
  }));
  const bodyCandidates: CharacterCandidateSlot[] = [1, 2, 3, 4].map((index) => ({
    id: `body-${index}`,
    index,
    source: "body",
    label: `身体候选 #${index}`,
    status: "pending",
  }));

  const textureGroups = {} as Record<TextureGroupId, CharacterTextureGroupState>;
  const defaultShades: Record<TextureGroupId, string> = {
    hair: "#5a4d7a",
    head: "#f2cbbd",
    upper_body: "#4c5b88",
    lower_body: "#38415e",
    accessories: "#d4af37",
  };
  for (const id of TEXTURE_GROUP_IDS) {
    textureGroups[id] = {
      id,
      label: TEXTURE_GROUP_LABELS[id],
      source: id === "hair" || id === "head" ? "head" : id === "accessories" ? "mixed" : "body",
      componentNames: [],
      status: "pending",
      resolution: "4096x4096",
      mtoonConverted: false,
      shadeColor: defaultShades[id],
      outlineWidth: id === "head" ? 0.0008 : 0.0014,
      notes: "",
    };
  }

  const springGroups: CharacterSpringGroup[] = [
    {
      id: "spring_hair_back",
      name: "后发与长发束 (Hair Back)",
      category: "hair",
      componentNames: ["hair_back_main", "hair_side_l", "hair_side_r"],
      stiffness: 0.85,
      gravityPower: 0.08,
      dragForce: 0.42,
      hitRadius: 0.025,
      colliderNames: ["head_sphere", "neck_capsule", "upper_chest_sphere"],
    },
    {
      id: "spring_ribbon",
      name: "蝴蝶结与发带 (Ribbon)",
      category: "ribbon",
      componentNames: ["ribbon_tail_l", "ribbon_tail_r"],
      stiffness: 0.65,
      gravityPower: 0.12,
      dragForce: 0.35,
      hitRadius: 0.018,
      colliderNames: ["head_sphere", "shoulder_l_sphere", "shoulder_r_sphere"],
    },
    {
      id: "spring_skirt",
      name: "裙摆与外套下摆 (Skirt Hem)",
      category: "cloth",
      componentNames: ["skirt_pleats_front", "skirt_pleats_back"],
      stiffness: 1.15,
      gravityPower: 0.15,
      dragForce: 0.5,
      hitRadius: 0.032,
      colliderNames: ["hips_sphere", "upper_leg_l_capsule", "upper_leg_r_capsule"],
    },
  ];

  const standardHumanoidBones = [
    ["hips", "J_Bip_C_Hips"],
    ["spine", "J_Bip_C_Spine"],
    ["chest", "J_Bip_C_Chest"],
    ["upperChest", "J_Bip_C_UpperChest"],
    ["neck", "J_Bip_C_Neck"],
    ["head", "J_Bip_C_Head"],
    ["leftShoulder", "J_Bip_L_Shoulder"],
    ["leftUpperArm", "J_Bip_L_UpperArm"],
    ["leftLowerArm", "J_Bip_L_LowerArm"],
    ["leftHand", "J_Bip_L_Hand"],
    ["rightShoulder", "J_Bip_R_Shoulder"],
    ["rightUpperArm", "J_Bip_R_UpperArm"],
    ["rightLowerArm", "J_Bip_R_LowerArm"],
    ["rightHand", "J_Bip_R_Hand"],
    ["leftUpperLeg", "J_Bip_L_UpperLeg"],
    ["leftLowerLeg", "J_Bip_L_LowerLeg"],
    ["leftFoot", "J_Bip_L_Foot"],
    ["leftToes", "J_Bip_L_ToeBase"],
    ["rightUpperLeg", "J_Bip_R_UpperLeg"],
    ["rightLowerLeg", "J_Bip_R_LowerLeg"],
    ["rightFoot", "J_Bip_R_Foot"],
    ["rightToes", "J_Bip_R_ToeBase"],
  ] as const;

  return {
    characterName,
    stylePrompt,
    views,
    headCandidates,
    bodyCandidates,
    selectedHeadCandidateId: null,
    selectedBodyCandidateId: null,
    segmentation: {
      status: "idle",
      totalVertices: 0,
      totalFaces: 0,
      components: [],
    },
    textureGroups,
    springGroups,
    placement: {
      targetHeightMeters: 1.58,
      linkHairWithHead: true,
      rigMode: "marker",
      overlayOpacity: 0.42,
      showHeightRuler: true,
      showHorizontalGuides: true,
      parts: {
        head: { x: 0, y: 1.32, z: 0, scale: 1.0 },
        hair: { x: 0, y: 1.32, z: 0, scale: 1.0 },
        body: { x: 0, y: 0, z: 0, scale: 1.0 },
      },
      bodyMarkers: structuredClone(DEFAULT_BODY_MARKERS),
      fingerMarkers: createDefaultFingerMarkers(),
    },
    rigging: {
      humanoidBoneCount: 55,
      mappedBones: standardHumanoidBones.map(([humanoidBone, sourceBone]) => ({
        humanoidBone,
        sourceBone,
        status: "mapped",
      })),
      weightTransfer: {
        donorMesh: "VRM1_Standard_Female_BaseBody",
        method: "nearest_surface_smoothed",
        smoothedSeams: ["neck_head_seam", "shoulder_sleeve_l", "shoulder_sleeve_r", "waist_skirt_seam"],
        maxInfluences: 4,
        status: "pending",
      },
      springVerification: {
        framesSimulated: 90,
        physicsOnOffVerified: false,
        chains: springGroups.map((g) => ({
          groupId: g.id,
          name: g.name,
          boneCount: g.category === "hair" ? 6 : 4,
          peakDisplacementMeters: 0,
          settledFrame: 0,
          colliderHits: 0,
          passed: false,
        })),
      },
    },
    vrmDelivery: {
      meta: {
        name: characterName,
        version: "1.0.0",
        authors: ["Open Game Studio Creator"],
        copyrightInformation: `© ${new Date().getFullYear()} Open Game Studio`,
        contactInformation: "",
        references: ["Charakuru Workflow DAG"],
        thirdPartyLicenses: "MToon / VRM 1.0 Specification",
        licenseUrl: "https://vrm.dev/licenses/1.0/",
        avatarPermission: "everyone",
        commercialUsage: "personalProfit",
        creditNotation: "unnecessary",
        allowRedistribution: false,
        modify: "allowModification",
      },
      motionClip: "walk_cycle_90f.vrma",
      physicsEnabled: true,
      showBones: false,
      wireframe: false,
    },
  };
}

export function createInitialCharacterDag(
  inputRevisionIds: string[],
): CharacterDagNode[] {
  const ts = new Date().toISOString();
  const rootStamp = inputRevisionIds.slice().sort().join(",") || "init";
  return CHARACTER_DAG_TEMPLATE.map((tmpl, idx) => ({
    key: tmpl.key,
    title: tmpl.title,
    stageGroup: tmpl.stageGroup,
    mode: tmpl.mode,
    status: idx === 0 ? "awaiting_agent" : "pending",
    dependsOn: [...tmpl.dependsOn],
    parallelGroup: tmpl.parallelGroup,
    legacyStepKey: tmpl.legacyStepKey,
    agentSkill: tmpl.agentSkill,
    description: tmpl.description,
    inputRevisionIds: idx === 0 ? [...inputRevisionIds] : [],
    outputRevisionIds: [],
    inputStamp: idx === 0 ? rootStamp : "",
    stale: false,
    note: tmpl.description,
    updatedAt: ts,
  }));
}

export function recomputeCharacterDag(run: WorkflowRun): void {
  if (!run.dag || !run.dag.length) return;
  const byKey = new Map<string, CharacterDagNode>();
  for (const node of run.dag) byKey.set(node.key, node);

  for (const node of run.dag) {
    if (node.dependsOn.length === 0) {
      node.inputStamp =
        (node.inputRevisionIds.length
          ? node.inputRevisionIds
          : run.inputRevisionIds
        )
          .slice()
          .sort()
          .join(",") || "init";
      continue;
    }
    const parentDeps = node.dependsOn
      .map((depKey) => byKey.get(depKey))
      .filter((x): x is CharacterDagNode => Boolean(x));
    const parentOutputs = [
      ...new Set(parentDeps.flatMap((d) => d.outputRevisionIds)),
    ];
    if (parentOutputs.length > 0) {
      node.inputRevisionIds = parentOutputs;
    }
    const stampParts = parentDeps.map(
      (d) => `${d.key}:${d.status}:${d.outputRevisionIds.join("+")}`,
    );
    if (node.key === "mesh_segmentation" && run.dagState) {
      stampParts.push(
        `sel:${run.dagState.selectedHeadCandidateId || ""}:${run.dagState.selectedBodyCandidateId || ""}`,
      );
    }
    node.inputStamp = stampParts.join("|");

    const anyParentStale = parentDeps.some((d) => d.stale);
    const allParentsSucceeded = parentDeps.every(
      (d) => d.status === "succeeded" && !d.stale,
    );

    if (node.status === "succeeded" && !node.stale && !node.completedInputStamp) {
      node.completedInputStamp = node.inputStamp;
    }

    if (
      node.status === "succeeded" &&
      (anyParentStale ||
        (node.completedInputStamp &&
          node.completedInputStamp !== node.inputStamp))
    ) {
      node.stale = true;
    }

    if (allParentsSucceeded) {
      if (node.status === "pending" || (node.status === "succeeded" && node.stale)) {
        if (node.status === "pending") {
          node.status =
            node.mode === "review"
              ? "awaiting_review"
              : node.mode === "agent"
                ? "awaiting_agent"
                : "awaiting_manual";
        }
      }
    } else if (
      node.status.startsWith("awaiting_") &&
      !parentDeps.every((d) => d.status === "succeeded")
    ) {
      node.status = "pending";
    }
  }
}

export function syncMacroStepsToDag(run: WorkflowRun): void {
  if (!run.dag || !run.dag.length) return;
  const stepMap = new Map<string, WorkflowStep>();
  for (const s of run.steps) stepMap.set(s.key, s);

  // If a macro step was marked succeeded via legacy submit/decide, mark corresponding DAG nodes succeeded if not already
  for (const step of run.steps) {
    const dagNodes = run.dag.filter((n) => n.legacyStepKey === step.key);
    if (!dagNodes.length) continue;
    if (step.status === "succeeded" && !step.stale) {
      for (const n of dagNodes) {
        if (n.status !== "succeeded") {
          n.status = "succeeded";
          n.stale = false;
          if (!n.outputRevisionIds.length && step.outputRevisionIds.length) {
            n.outputRevisionIds = [...step.outputRevisionIds];
          }
          n.completedInputStamp = n.inputStamp || "synced";
        }
      }
    } else if (step.stale) {
      for (const n of dagNodes) {
        n.stale = true;
      }
    }
  }
  recomputeCharacterDag(run);
}

export function syncDagToMacroSteps(run: WorkflowRun): void {
  if (!run.dag || !run.dag.length) return;
  recomputeCharacterDag(run);
  for (let i = 0; i < run.steps.length; i++) {
    const step = run.steps[i];
    const groupNodes = run.dag.filter((n) => n.legacyStepKey === step.key);
    if (!groupNodes.length) continue;
    const allSucceeded = groupNodes.every(
      (n) => n.status === "succeeded" && !n.stale,
    );
    const anyStale = groupNodes.some((n) => n.stale);
    const outputs = [
      ...new Set(groupNodes.flatMap((n) => n.outputRevisionIds)),
    ];
    if (outputs.length) {
      step.outputRevisionIds = outputs;
    }
    const prevStepsDone = run.steps
      .slice(0, i)
      .every((s) => s.status === "succeeded" && !s.stale);
    if (allSucceeded) {
      step.status = "succeeded";
      step.stale = false;
      const nextStep = run.steps[i + 1];
      if (nextStep && !nextStep.inputRevisionIds.length && step.outputRevisionIds.length) {
        nextStep.inputRevisionIds = [...step.outputRevisionIds];
      }
    } else if (prevStepsDone) {
      step.status =
        step.mode === "review"
          ? "awaiting_review"
          : step.mode === "agent"
            ? "awaiting_agent"
            : "awaiting_manual";
      step.stale = anyStale;
    } else {
      step.status = "pending";
      step.stale = anyStale;
    }
  }
  if (run.steps.every((s) => s.status === "succeeded" && !s.stale)) {
    run.status = "succeeded";
  }
}
