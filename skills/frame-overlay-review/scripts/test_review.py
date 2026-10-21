"""Behavioral checks: geometry, paired coverage, input contracts and provenance."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from review import metrics, review


class ReviewTests(unittest.TestCase):
    def test_identical_shifted_and_empty_masks(self):
        original = np.zeros((80, 100), bool)
        original[20:60, 20:60] = True
        same = metrics(original, original)
        self.assertEqual(same["iou"], 1)
        self.assertEqual(same["boundary_mean_px"], 0)
        shifted = np.roll(original, 10, axis=1)
        comparison = metrics(original, shifted)
        self.assertAlmostEqual(comparison["iou"], 1200 / 2000)
        self.assertGreater(comparison["boundary_mean_px"], 0)
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


if __name__ == "__main__":
    unittest.main()
