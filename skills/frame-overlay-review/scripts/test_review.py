"""Behavioral checks: geometry, paired coverage, input contracts, provenance, and single-frame multi-view Agent-in-the-loop pose refinement."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pose_bridge import parse_and_apply_pose_tweaks, render_pose_lbs_canvas, resolve_views_for_canvas
from review import metrics, review, review_frame


class ReviewTests(unittest.TestCase):
    def test_identical_shifted_and_empty_masks(self):
        original = np.zeros((80, 100), bool)
        original[20:60, 20:60] = True
        same = metrics(original, original)
        self.assertEqual(same["iou"], 1)
        self.assertEqual(same["boundary_mean_px"], 0)
        self.assertTrue(same["outline_passed"])
        self.assertEqual(same["centroid_shift_px"], [0.0, 0.0])
        self.assertEqual(same["error_clusters"], [])

        shifted = np.roll(original, 10, axis=1)
        comparison = metrics(original, shifted)
        self.assertAlmostEqual(comparison["iou"], 1200 / 2000)
        self.assertGreater(comparison["boundary_mean_px"], 0)
        self.assertFalse(comparison["outline_passed"])
        # Candidate is shifted +10px in X relative to reference, so pull vector points -X (-10px)
        self.assertAlmostEqual(comparison["centroid_shift_px"][0], 10.0, places=1)
        self.assertLess(comparison["mean_contour_pull_px"][0], -3.5)
        kinds = {c["kind"] for c in comparison["error_clusters"]}
        self.assertIn("protruding", kinds)
        self.assertIn("missing", kinds)
        for cluster in comparison["error_clusters"]:
            self.assertEqual(cluster["correction_vector_px"][0], -10)

        empty = metrics(np.zeros_like(original), np.zeros_like(original))
        self.assertFalse(empty["valid"])
        self.assertIsNone(empty["iou"])
        one_empty = metrics(original, np.zeros_like(original))
        self.assertFalse(one_empty["valid"])

    def fixture(self, root):
        sources = {}
        for stage in ("reference", "candidate", "baseline"):
            folder = root / stage
            folder.mkdir()
            for frame in range(3):
                arr = np.zeros((80, 100, 3), dtype=np.uint8)
                arr[20:60, 20:60] = 255
                Image.fromarray(arr).save(folder / f"frame_{frame:04d}.png")
            sources[stage] = {"directory": stage}
        config = {"version": 1, "action": "fixture", "fps": 24, "frame_count": 3,
                  "sources": sources, "views": [{"name": "front", "box": [0, 0, 100, 80]}],
                  "alignment": {"status": "verified", "evidence": "Identical synthetic canvas and timestamps"}}
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(config))
        return manifest, config

    def test_partial_coverage_uses_paired_frames_and_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _ = self.fixture(root)
            (root / "candidate/frame_0001.png").unlink()
            report = review(manifest, root / "out")
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["coverage"]["available_frames"], [0, 2])
            self.assertEqual(report["summary"][0]["paired_frames"], [0, 2])
            self.assertEqual(report["summary"][0]["count"], 2)
            self.assertEqual(report["missing"][0]["frame"], 1)
            self.assertEqual(len(report["provenance"][0]["sha256"]), 64)
            self.assertTrue((root / "out/index.html").is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                review(manifest, root / "out")

    def test_size_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, _ = self.fixture(root)
            Image.new("RGB", (99, 80)).save(root / "candidate/frame_0000.png")
            with self.assertRaisesRegex(ValueError, "no resizing"):
                review(manifest, root / "out")

    def test_preview_and_unverified_alignment_are_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, config = self.fixture(root)
            self.assertEqual(review(manifest, root / "full")["status"], "complete")
            self.assertEqual(review(manifest, root / "preview", "1")["status"], "incomplete")
            config["alignment"] = {"status": "unverified", "evidence": ""}
            manifest.write_text(json.dumps(config))
            self.assertEqual(review(manifest, root / "unverified")["status"], "incomplete")

    def test_masks_alpha_and_invalid_roi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, config = self.fixture(root)
            mask_dir = root / "masks"
            mask_dir.mkdir()
            for frame in range(3):
                # Uniform source photographs: only the explicit masks locate the object.
                for stage in config["sources"]:
                    Image.new("RGB", (100, 80), (30, 30, 30)).save(root / stage / f"frame_{frame:04d}.png")
                mask = np.zeros((80, 100), np.uint8)
                mask[20:60, 20:60] = 255
                Image.fromarray(mask).save(mask_dir / f"frame_{frame:04d}.png")
            for source in config["sources"].values():
                source["mask_directory"] = "masks"
            manifest.write_text(json.dumps(config))
            self.assertEqual(review(manifest, root / "explicit")["summary"][0]["candidate"]["iou"], 1)
            config["views"][0]["regions"] = [{"name": "background", "box": [0, 0, 10, 10]}]
            manifest.write_text(json.dumps(config))
            report = review(manifest, root / "roi")
            self.assertEqual(report["status"], "incomplete")
            self.assertTrue(report["invalid"])
            for source in config["sources"].values():
                source.pop("mask_directory")
            config["segmentation"] = {"mode": "alpha", "threshold": 127}
            manifest.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "alpha channel"):
                review(manifest, root / "no-alpha")

    def test_view_and_roi_bounds_are_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, config = self.fixture(root)
            config["views"].append(copy.deepcopy(config["views"][0]))
            manifest.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "unique"):
                review(manifest, root / "duplicate")
            config["views"].pop()
            config["views"][0]["box"] = [0, 0, 101, 80]
            manifest.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "outside"):
                review(manifest, root / "bounds")

    def test_single_frame_multiview_auto_split_and_panel_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 2-view composite image (400x200 => front: 0..200, side: 200..400)
            ref_arr = np.zeros((200, 400, 3), dtype=np.uint8)
            ref_arr[50:150, 60:140] = 220   # front silhouette
            ref_arr[50:150, 260:340] = 220  # side silhouette
            cand_arr = np.zeros((200, 400, 3), dtype=np.uint8)
            cand_arr[50:150, 60:140] = 220  # front matches
            cand_arr[50:150, 276:356] = 220  # side shifted +16px right

            ref_path = root / "ref_2view.png"
            cand_path = root / "cand_2view.png"
            Image.fromarray(ref_arr).save(ref_path)
            Image.fromarray(cand_arr).save(cand_path)

            out_dir = root / "frame_review"
            report = review_frame(
                output=out_dir,
                reference=ref_path,
                candidate=cand_path,
                layout="auto",
                rois=["side:upper:0,0,200,120"],
            )
            self.assertEqual(report["status"], "complete")
            self.assertFalse(report["outline_passed"])
            self.assertEqual([v["name"] for v in report["config"]["views"]], ["front", "side"])
            self.assertTrue((out_dir / "panel_compare.png").is_file())
            self.assertTrue((out_dir / "report.json").is_file())
            self.assertTrue((out_dir / "index.html").is_file())

            front_row = next(r for r in report["rows"] if r["view"] == "front" and r["region"] == "whole")
            side_row = next(r for r in report["rows"] if r["view"] == "side" and r["region"] == "whole")
            self.assertTrue(front_row["candidate"]["outline_passed"])
            self.assertFalse(side_row["candidate"]["outline_passed"])
            self.assertGreater(len(side_row["candidate"]["error_clusters"]), 0)

            # Verify --overwrite allows re-running into the same directory in an Agent loop
            report2 = review_frame(
                output=out_dir,
                reference=ref_path,
                candidate=ref_path,
                baseline=cand_path,
                layout="2view",
                overwrite=True,
            )
            self.assertTrue(report2["outline_passed"])
            side_summary = next(s for s in report2["summary"] if s["view"] == "side")
            self.assertGreater(side_summary["iou_gain"], 0.1)

    def test_pose_fk_tweak_lbs_render_and_auto_nudge_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Construct a 2-bone synthetic rig (Root_Body + Arm_R) with a 3D mesh box around each bone
            rig_meta = {
                "calibration": {
                    "center": [0.0, 0.0, 0.5],
                    "ortho_scale": 1.4,
                    "panel_width": 240,
                    "panel_height": 240,
                    "view_order": ["front", "side"],
                },
                "bones": {
                    "Root_Body": {
                        "parent": None,
                        "depth": 0,
                        "kind": "root",
                        "fixed_local": True,
                        "rest_head_world": [0.0, 0.0, 0.25],
                        "rest_tail_world": [0.0, 0.0, 0.65],
                        "rest_rot_mat_world": [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
                    },
                    "Arm_R": {
                        "parent": "Root_Body",
                        "depth": 1,
                        "kind": "limb",
                        "fixed_local": False,
                        "rest_head_world": [0.08, 0.0, 0.55],
                        "rest_tail_world": [0.46, 0.0, 0.55],
                        "rest_rot_mat_world": [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                    },
                },
            }

            def make_box_mesh(x0, x1, y0, y1, z0, z1, bone_idx):
                verts = np.array([
                    [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                    [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
                ], dtype=np.float32)
                tris = np.array([
                    [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                    [0, 3, 7], [0, 7, 4], [1, 2, 6], [1, 6, 5],
                ], dtype=np.int32)
                idx = np.zeros((8, 4), dtype=np.int32)
                idx[:, 0] = bone_idx
                w = np.zeros((8, 4), dtype=np.float32)
                w[:, 0] = 1.0
                return verts, tris, idx, w

            # Sorted bone names: ["Arm_R", "Root_Body"] -> Arm_R is index 0, Root_Body is index 1
            v_body, t_body, i_body, w_body = make_box_mesh(-0.10, 0.10, -0.09, 0.09, 0.20, 0.68, bone_idx=1)
            v_arm, t_arm, i_arm, w_arm = make_box_mesh(0.08, 0.48, -0.06, 0.06, 0.49, 0.61, bone_idx=0)
            skin_npz = root / "skinning.npz"
            np.savez_compressed(
                skin_npz,
                rest_verts=np.concatenate([v_body, v_arm], axis=0),
                triangles=np.concatenate([t_body, t_arm + 8], axis=0),
                skin_indices=np.concatenate([i_body, i_arm], axis=0),
                skin_weights=np.concatenate([w_body, w_arm], axis=0),
            )

            rig_meta_path = root / "studio_rig_meta.json"
            rig_meta_path.write_text(json.dumps(rig_meta, indent=2), encoding="utf-8")

            views = resolve_views_for_canvas(480, 240, layout="2view")
            # Target reference pose: Arm_R is at rest (0 deg)
            ref_bones, _ = parse_and_apply_pose_tweaks(rig_meta, {"frame": 0, "bones": {}}, frame_idx=0)
            with np.load(skin_npz) as sdata:
                skin_dict = {k: sdata[k] for k in sdata.files}
            ref_canvas = render_pose_lbs_canvas(rig_meta, ref_bones, views, (480, 240), skin_dict)
            ref_path = root / "ref_frame.png"
            ref_canvas.convert("RGB").save(ref_path)

            # Initial candidate pose: Arm_R is rotated +18 deg in front view (tilted down)
            bad_bones, _ = parse_and_apply_pose_tweaks(
                rig_meta, {"frame": 0, "bones": {}}, frame_idx=0, view_tweaks=["front:Arm_R:+18"]
            )
            pose_path = root / "pred_pose.json"
            pose_path.write_text(json.dumps({"frames": [{"frame": 0, "bones": bad_bones}]}, indent=2), encoding="utf-8")

            # Step 1: Run diagnostic without tweaks -> should detect Arm_R error and recommend negative front rotation
            diag_dir = root / "step1_diag"
            rep1 = review_frame(
                output=diag_dir,
                reference=ref_path,
                rig_meta_path=rig_meta_path,
                pose_json_path=pose_path,
                rig_skinning_path=skin_npz,
                frame=0,
            )
            self.assertGreater(len(rep1["top_bone_suggestions"]), 0)
            top_sug = rep1["top_bone_suggestions"][0]
            self.assertEqual(top_sug["bone"], "Arm_R")
            self.assertLess(top_sug["suggested_view_rot_deg"], -3.0)

            # Step 2: Run CLI with --auto-nudge-iters 3 and --out-pose-json -> should improve IoU and write refined pose
            refined_pose_path = root / "refined_pose.json"
            step2_dir = root / "step2_refined"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "review.py"),
                "frame",
                "--reference", str(ref_path),
                "--rig-meta", str(rig_meta_path),
                "--pose-json", str(pose_path),
                "--rig-skinning", str(skin_npz),
                "--frame", "0",
                "--auto-nudge-iters", "3",
                "--out-pose-json", str(refined_pose_path),
                "--output", str(step2_dir),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            cli_out = json.loads(proc.stdout.strip())
            self.assertGreater(cli_out["iou_gain"], 0.05)
            self.assertTrue(refined_pose_path.is_file())
            self.assertTrue((step2_dir / "panel_compare.png").is_file())


if __name__ == "__main__":
    unittest.main()
