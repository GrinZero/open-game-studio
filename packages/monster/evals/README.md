# Monster Studio Video-to-Animation Evaluation Suite (`evals/`)

This directory houses the standardized evaluation (`eval`) system for **Monster Studio's Video-to-Animation Reconstruction** pipeline (`2560×1440` 2K dual-orthographic split-screen video → 3D skeletal animation & GLB).

## Design Principles

1. **2K Benchmark Only (`fantasy_creature_2k_v2`)**:
   Evaluates native 2K dual-orthographic studio input (`1280×1440` per panel, `2560×1440` composite, 24 FPS, 288 frames, 77 bones across 7 anatomical kinds).
2. **Latest Solver Only (`taxonomy_kind_v2`)**:
   Tests the 3-layer general + kind-specific kinematic solver in [`worker/bone_taxonomy.py`](../worker/bone_taxonomy.py):
   - **Layer 1**: Hybrid (Name + Constraints + 3D Topology/Geometry) Bone Classifier (`root`, `rigid`, `spine`, `head`, `limb`, `flex`, `digit`, etc.)
   - **Layer 2**: Bounded short-gap (`<= 3` frames) Hampel 2D spike cleaning, dual-ortho epipolar V-consistency gating, 5-tap quadratic Savitzky-Golay smoothing, and 8-anchor IRLS Kabsch root fitting
   - **Layer 3**: Kind-specific FK solvers (Gimbal-free Swing-Twist on limbs, disconnected child-head pivot fusion on forearms, palm-fan fusion on hands/feet, sagittal-prior bounded Euler on spine/head/flex, coupled curl damping on digits)
3. **Zero Git Bloat + Google Drive Archive**:
   Only lightweight JSON manifests, schemas, golden-frame probes, and baseline metrics live in Git (`packages/monster/evals/`). Large binary fixtures (`rig.blend`, `dinov2_head.pt`, `exam_input_2k_v0001.mp4`, `exam_gt.json.gz`, `observations.npz`) are packed into a single deterministic archive (`var/monster/evals/archives/fantasy_creature_2k_v2.tar.gz`) that can be uploaded to Google Drive and pulled via `dataset pull`.
4. **Excluded from Colab Release Bundle**:
   `packages/monster/scripts/package-release.py` uses an explicit `SOURCE_DIRS = ("worker", "scripts", "integration", "notebooks", "docs")` allowlist, so `evals/` never bloats the Colab release tarball.

## Directory Layout

```text
packages/monster/evals/
├── cli.py                                  # Unified CLI entrypoint (npm run monster:eval -- ...)
├── schemas/
│   ├── benchmark_manifest.schema.json      # Schema for manifest.json
│   └── eval_report.schema.json             # Schema for eval_report.json
├── benchmarks/
│   └── fantasy_creature_2k_v2/
│       ├── manifest.json                   # SHA-256 digests, calibration, Drive bundle info
│       ├── golden_frames.json              # Key frame regression probes (Frame 0, 17, 60, 131, 211, 287)
│       └── baseline_latest.json            # Locked baseline metrics for taxonomy_kind_v2
├── datasets/
│   ├── drive_bundle.py                     # Pack, pull (local or Google Drive), and extract tar.gz
│   └── loader.py                           # SHA-256 verification and fixture loader
├── solvers/
│   └── latest_solver.py                    # Stage 1-4 execution adapter for taxonomy_kind_v2
├── metrics/
│   └── evaluator.py                        # 5-stage attribution, stratified kinematics, temporal & golden probes
├── runners/
│   ├── tier1_fast_solver.py                # Tier 1 (`fast`): CPU ~1.5s from cached observations.npz
│   ├── tier2_perception_infer.py           # Tier 2 (`infer`): MPS/CUDA DINOv2 2K inference + Tier 1
│   └── tier3_full_blender.py               # Tier 3 (`full`): Headless Blender GLB roundtrip (< 1e-4)
└── reporters/
    ├── markdown_reporter.py                # Markdown tables + baseline diff
    └── html_review_builder.py              # Self-contained HTML review dashboard
```

## Latest archived run

The 2026-10-21 retraining run is available as [`fantasy_creature_2k_v2`](benchmarks/fantasy_creature_2k_v2/README.md), including new weights, perception caches, cloud exam results and reproducibility receipts. G3 remains pending. v2 is now the default. The obsolete v1 fixture and baseline have been removed at the user’s request.

## Quick Commands

```bash
# 1. Verify local 2K fixture integrity against manifest.json SHA-256 digests
npm run monster:eval -- dataset verify fantasy_creature_2k_v2

# 2. Pull/extract fixture from local archive or Google Drive
npm run monster:eval -- dataset pull fantasy_creature_2k_v2

# 3. Run Tier 1 Fast Kinematic & Attribution Evaluation (~1.5s on CPU)
npm run monster:eval -- run fantasy_creature_2k_v2 --tier fast

# 4. Diagnose a specific frame across Stages 1 -> 2 -> 3 -> 4 (e.g., Frame 17 root flip guard)
npm run monster:eval -- diagnose fantasy_creature_2k_v2 --frame 17

# 5. Run Tier 2 Native 2K DINOv2 Inference + Solver (MPS/CUDA)
npm run monster:eval -- run fantasy_creature_2k_v2 --tier infer

# 6. Run Tier 3 End-to-End Headless Blender GLB Export & 5-Phase Skinning Roundtrip
npm run monster:eval -- run fantasy_creature_2k_v2 --tier full

# 7. Audit the 5,000-Sample 4-Regime Decoupled Pose Sampler on a Benchmark Rig (<1s on CPU)
npm run monster:eval -- audit-sampling fantasy_creature_2k_v2 --samples 5000
```
