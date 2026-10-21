#!/usr/bin/env python3
"""Compare synchronized frame sequences without image registration. See --help."""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(1)  # Predictable memory/runtime on full-resolution frame sequences.
VERSION = "1.0.0"
DEFAULT_MASK = {"mode": "border", "threshold": 45, "min_component": 36, "border": 20}
SCORES = ("iou", "dice", "boundary_mean_px", "boundary_p95_px", "boundary_within_tolerance")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def box_ok(box, width, height):
    return (isinstance(box, list) and len(box) == 4
            and all(type(n) is int for n in box)
            and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def clean_mask(mask, minimum):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.zeros(count, dtype=np.uint8)
    keep[np.where(stats[:, cv2.CC_STAT_AREA] >= minimum)] = 1
    keep[0] = 0
    return keep[labels].astype(bool)


def silhouette(image, settings, explicit=None):
    if explicit is not None:
        mask = np.asarray(explicit.convert("L")) > settings.get("mask_threshold", 127)
    elif settings["mode"] == "alpha":
        if "A" not in image.getbands():
            raise ValueError("alpha segmentation requires an alpha channel")
        mask = np.asarray(image.getchannel("A")) > settings["threshold"]
    elif settings["mode"] == "border":
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        border = min(settings.get("border", 20), image.width, image.height)
        background = settings.get("background")
        if background is None:
            background = np.median(np.concatenate([
                arr[:border].reshape(-1, 3), arr[-border:].reshape(-1, 3),
                arr[:, :border].reshape(-1, 3), arr[:, -border:].reshape(-1, 3),
            ]), axis=0)
        mask = np.linalg.norm(arr - np.array(background), axis=2) > settings["threshold"]
    else:
        raise ValueError("segmentation mode must be border or alpha")
    return clean_mask(mask, settings["min_component"])


def edge(mask):
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask & ~eroded.astype(bool)


def metrics(reference, candidate, tolerance=8):
    result = {"valid": bool(reference.any() and candidate.any()),
              "reference_pixels": int(reference.sum()), "render_pixels": int(candidate.sum())}
    if not result["valid"]:
        return {**result, "reason": "empty silhouette", **{k: None for k in SCORES}}
    intersection = int((reference & candidate).sum())
    re, ce = edge(reference), edge(candidate)
    dr = cv2.distanceTransform((~re).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dc = cv2.distanceTransform((~ce).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    distances = np.concatenate([dr[ce], dc[re]])
    return {**result, "iou": intersection / int((reference | candidate).sum()),
            "dice": 2 * intersection / int(reference.sum() + candidate.sum()),
            "boundary_mean_px": float(distances.mean()),
            "boundary_p95_px": float(np.percentile(distances, 95)),
            "boundary_within_tolerance": float((distances <= tolerance).mean())}


def selected_frames(spec, count):
    if spec == "all":
        return list(range(count))
    values = set()
    for item in spec.split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", item):
            raise ValueError("--frames expects all or e.g. 0,24,48-60")
        bounds = [int(n) for n in item.split("-")]
        if len(bounds) == 2 and bounds[0] > bounds[1]:
            raise ValueError("frame range is reversed")
        values.update(range(bounds[0], bounds[-1] + 1))
    if not values or min(values) < 0 or max(values) >= count:
        raise ValueError("selected frame is outside frame_count")
    return sorted(values)


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("version") != 1 or not config.get("action"):
        raise ValueError("manifest needs version:1 and action")
    if type(config.get("frame_count")) is not int or config["frame_count"] < 1:
        raise ValueError("frame_count must be a positive integer")
    if not isinstance(config.get("fps"), (int, float)) or not math.isfinite(config["fps"]) or config["fps"] <= 0:
        raise ValueError("fps must be positive and finite")
    sources = config.get("sources", {})
    if not {"reference", "candidate"} <= sources.keys() or not sources.keys() <= {"reference", "candidate", "baseline"}:
        raise ValueError("sources must contain reference/candidate and optional baseline")
    names = set()
    if not config.get("views"):
        raise ValueError("at least one view is required")
    for view in config["views"]:
        name = view.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError("view names must be unique simple identifiers")
        names.add(name)
        if not box_ok(view.get("box"), 1000000, 1000000):
            raise ValueError("invalid view rectangle")
        width, height = view["box"][2] - view["box"][0], view["box"][3] - view["box"][1]
        regions = {"whole"}
        for region in view.get("regions", []):
            name = region.get("name", "")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in regions or not box_ok(region.get("box"), width, height):
                raise ValueError("invalid/duplicate region or rectangle outside view")
            regions.add(name)
    alignment = config.setdefault("alignment", {"status": "unverified", "evidence": ""})
    if alignment.get("status") not in ("verified", "unverified"):
        raise ValueError("alignment.status must be verified or unverified")
    if alignment["status"] == "verified" and not alignment.get("evidence", "").strip():
        raise ValueError("verified alignment requires evidence")
    tolerance = config.setdefault("boundary_tolerance_px", 8)
    if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("boundary_tolerance_px must be finite and nonnegative")
    for source in sources.values():
        settings = {**DEFAULT_MASK, **config.get("segmentation", {}), **source.get("segmentation", {})}
        if settings["mode"] not in ("border", "alpha"):
            raise ValueError("segmentation mode must be border or alpha")
        for field in ("min_component", "border"):
            if type(settings[field]) is not int or settings[field] < 1:
                raise ValueError(f"{field} must be a positive integer")
        for field in ("threshold", "mask_threshold"):
            value = settings.get(field, 127)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and nonnegative")
        if "background" in settings and (not isinstance(settings["background"], list) or len(settings["background"]) != 3 or
                any(not isinstance(n, (int, float)) or not 0 <= n <= 255 for n in settings["background"])):
            raise ValueError("background must be three RGB values in 0..255")
        source["segmentation"] = settings
        for key in ("directory", "mask_directory"):
            if key == "directory" or key in source:
                source[key] = str((path.parent / source[key]).resolve())
    return config


def source_path(source, frame, mask=False):
    directory = Path(source["mask_directory" if mask else "directory"])
    pattern = source.get("mask_pattern", source.get("pattern", "frame_{frame:04d}.png")) if mask else source.get("pattern", "frame_{frame:04d}.png")
    relative = Path(pattern.format(frame=frame))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("frame pattern must stay inside its directory")
    return directory / relative


def summarize(rows, stages):
    # All stages use exactly the same valid frame set for each view/region.
    groups = {}
    for row in rows:
        key = (row["view"], row["region"])
        groups.setdefault(key, [])
        if all(row[stage]["valid"] for stage in stages):
            groups[key].append(row)
    summary = []
    for (view, region), group in groups.items():
        item = {"view": view, "region": region, "paired_frames": [r["frame"] for r in group], "count": len(group)}
        for stage in stages:
            item[stage] = {key: float(np.mean([r[stage][key] for r in group])) if group else None for key in SCORES}
        item["iou_gain"] = item["candidate"]["iou"] - item["baseline"]["iou"] if group and "baseline" in stages else None
        summary.append(item)
    return summary


def review(manifest, output, frames="all"):
    config = load_config(manifest)
    chosen = selected_frames(frames, config["frame_count"])
    if output.exists():
        raise ValueError("output already exists; choose a new review directory")
    # Prevent outputs inside immutable inputs, including a directory not created yet.
    for source in config["sources"].values():
        for key in ("directory", "mask_directory"):
            if key in source and output.is_relative_to(Path(source[key])):
                raise ValueError("output must be outside source directories")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    rows, assets, provenance, missing, invalid = [], [], [], [], []
    stages = ["candidate"] + (["baseline"] if "baseline" in config["sources"] else [])
    canvas_size = None
    for frame in chosen:
        paths = {name: source_path(source, frame) for name, source in config["sources"].items()}
        mask_paths = {name: source_path(source, frame, True) for name, source in config["sources"].items() if "mask_directory" in source}
        absent = [{"frame": frame, "source": name, "kind": kind, "path": str(path)}
                  for kind, mapping in (("image", paths), ("mask", mask_paths))
                  for name, path in mapping.items() if not path.is_file()]
        missing.extend(absent)
        if absent:
            continue
        images, masks = {}, {}
        for name, path in paths.items():
            with Image.open(path) as image:
                images[name] = image.copy()
            if canvas_size is None:
                canvas_size = images[name].size
            if images[name].size != canvas_size:
                raise ValueError(f"frame {frame} {name}: canvas {images[name].size} differs from {canvas_size}; no resizing allowed")
            info = {"frame": frame, "source": name, "path": str(path), "sha256": sha(path)}
            if name in mask_paths:
                with Image.open(mask_paths[name]) as mask:
                    masks[name] = mask.copy()
                if masks[name].size != canvas_size:
                    raise ValueError(f"frame {frame} {name}: mask dimensions differ")
                info.update(mask_path=str(mask_paths[name]), mask_sha256=sha(mask_paths[name]))
            provenance.append(info)
        for view in config["views"]:
            if not box_ok(view["box"], *canvas_size):
                raise ValueError(f"view {view['name']} is outside the image canvas")
            crops = {name: image.crop(view["box"]) for name, image in images.items()}
            silhouettes = {name: silhouette(image, config["sources"][name]["segmentation"],
                                           masks[name].crop(view["box"]) if name in masks else None)
                           for name, image in crops.items()}
            stem = f"{frame:06d}_{view['name']}"
            pictures = {}
            for name, image in crops.items():
                pictures[name] = f"images/{stem}_{name}.jpg"
                image.convert("RGB").save(output / pictures[name], quality=90)
            ref_mask, cand_mask = silhouettes["reference"], silhouettes["candidate"]
            re, ce = edge(ref_mask), edge(cand_mask)
            blend = np.array(Image.blend(crops["reference"].convert("RGB"), crops["candidate"].convert("RGB"), .5))
            blend[re], blend[ce], blend[re & ce] = (30, 240, 255), (255, 55, 210), (255, 255, 255)
            difference = np.zeros((*ref_mask.shape, 3), dtype=np.uint8)
            difference[ref_mask], difference[cand_mask], difference[ref_mask & cand_mask] = (30, 240, 255), (255, 55, 210), (235, 235, 235)
            for name, array in (("overlay", blend), ("mask", difference)):
                pictures[name] = f"images/{stem}_{name}.png"
                Image.fromarray(array).save(output / pictures[name])
            assets.append({"frame": frame, "view": view["name"], "images": pictures})
            regions = [{"name": "whole", "box": [0, 0, crops["reference"].width, crops["reference"].height]}, *view.get("regions", [])]
            for region in regions:
                left, top, right, bottom = region["box"]
                reference = ref_mask[top:bottom, left:right]
                row = {"frame": frame, "seconds": frame / config["fps"], "view": view["name"], "region": region["name"], "box": region["box"]}
                for stage in stages:
                    row[stage] = metrics(reference, silhouettes[stage][top:bottom, left:right], config["boundary_tolerance_px"])
                    if not row[stage]["valid"]:
                        invalid.append({"frame": frame, "view": view["name"], "region": region["name"], "stage": stage, "reason": row[stage]["reason"]})
                row["iou_gain"] = row["candidate"]["iou"] - row["baseline"]["iou"] if "baseline" in stages and all(row[s]["valid"] for s in stages) else None
                rows.append(row)
    valid_frames = sorted({r["frame"] for r in rows if all(r[s]["valid"] for s in stages)})
    complete = len(chosen) == config["frame_count"] and not missing and not invalid and bool(rows)
    verified = config["alignment"]["status"] == "verified"
    report = {"tool": "frame-overlay-review", "tool_version": VERSION,
              "created_at": datetime.now(timezone.utc).isoformat(), "manifest_sha256": sha(manifest),
              "config": config, "alignment": config["alignment"],
              "coordinate_contract": "Original pixels; fixed shared view/ROI boxes; no registration, translation, scaling or retiming.",
              "metric_scope": "2D silhouette agreement; not 3D or animation quality acceptance.",
              "status": "complete" if complete and verified else "incomplete",
              "coverage": {"total_frames": config["frame_count"], "selected_frames": chosen,
                           "available_frames": sorted({r['frame'] for r in rows}), "frames_with_valid_regions": valid_frames,
                           "full_sequence": len(chosen) == config["frame_count"], "complete": complete},
              "missing": missing, "invalid": invalid, "summary": summarize(rows, stages), "rows": rows,
              "worst_frames": sorted([r for r in rows if r["candidate"]["valid"]], key=lambda r: r["candidate"]["iou"])[:20],
              "regressed_frames": sorted([r for r in rows if r["iou_gain"] is not None and r["iou_gain"] < 0], key=lambda r: r["iou_gain"]),
              "provenance": provenance, "assets": assets}
    dump(output / "report.json", report)
    with (output / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["frame", "seconds", "view", "region", "iou_gain", *[f"{s}_{k}" for s in stages for k in ("valid", *SCORES)]]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row[k] for k in columns[:5]}, **{f"{s}_{k}": row[s][k] for s in stages for k in ("valid", *SCORES)}})
    template = (Path(__file__).resolve().parent.parent / "assets/report.html").read_text(encoding="utf-8")
    embedded = json.dumps(report, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    (output / "index.html").write_text(template.replace("__REPORT_DATA__", embedded), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create an unverified comparison manifest")
    for field in ("reference", "candidate", "action", "output"):
        init.add_argument("--" + field, required=True)
    init.add_argument("--baseline")
    init.add_argument("--fps", required=True, type=float)
    init.add_argument("--frame-count", required=True, type=int)
    init.add_argument("--pattern", default="frame_{frame:04d}.png")
    init.add_argument("--view", action="append", required=True, help="NAME:left,top,right,bottom; repeat for multiple views")
    run = sub.add_parser("review", help="Write an immutable comparison report")
    run.add_argument("manifest", type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--frames", default="all")
    args = parser.parse_args()
    try:
        if args.command == "init":
            output = Path(args.output).resolve()
            if output.exists():
                raise ValueError("manifest already exists")
            views = []
            for value in args.view:
                name, rectangle = value.split(":", 1)
                views.append({"name": name, "box": [int(n) for n in rectangle.split(",")]})
            config = {"version": 1, "action": args.action, "fps": args.fps, "frame_count": args.frame_count,
                      "sources": {name: {"directory": str(Path(getattr(args, name)).resolve()), "pattern": args.pattern}
                                  for name in ("reference", "candidate", "baseline") if getattr(args, name)},
                      "views": views, "segmentation": DEFAULT_MASK,
                      "alignment": {"status": "unverified", "evidence": ""}}
            output.parent.mkdir(parents=True, exist_ok=True)
            dump(output, config)
            print(output)
            return 0
        report = review(args.manifest.resolve(), args.output.resolve(), args.frames)
        print(json.dumps({"status": report["status"], "available_frames": len(report["coverage"]["available_frames"]),
                          "total_frames": report["coverage"]["total_frames"], "missing": len(report["missing"]),
                          "invalid": len(report["invalid"]), "report": str(args.output.resolve() / "index.html")}, ensure_ascii=False))
        return 0 if report["status"] == "complete" else 2
    except (ValueError, KeyError, TypeError, OSError, cv2.error) as error:
        print(f"frame-overlay-review: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
