"""Native G2 test renderer: real pose/audit/label pipeline with Workbench pixels."""

from pathlib import Path
import argparse, sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "worker"))
sys.path.append(
    str(
        root
        / ".venv/lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
)
import studio_bpy_worker as worker

parser = argparse.ArgumentParser()
for name in ("mode", "blend_path", "meta_path", "limits_path", "out_dir"):
    parser.add_argument("--" + name)
for name in ("num_samples", "seed"):
    parser.add_argument("--" + name, type=int)
a = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def workbench(scene, samples=None):
    scene.render.engine = "BLENDER_WORKBENCH"
    return {"engine": "BLENDER_WORKBENCH", "test_fixture": True}


worker.configure_blender_gpu_rendering = workbench
worker.generate_stage2_synthetic_dataset(
    a.blend_path,
    a.meta_path,
    a.out_dir,
    num_samples=a.num_samples,
    seed=a.seed,
    limits_path=a.limits_path,
    boundary_exploration=True,
)
print("ACTUAL_G2_BOUNDARY_LABELS")
