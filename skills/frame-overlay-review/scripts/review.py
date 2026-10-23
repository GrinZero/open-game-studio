#!/usr/bin/env python3
"""Compare synchronized frame sequences or single multi-view frames without image registration. See --help."""
import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pose_bridge import (
    attribute_clusters_to_bones,
    build_view_diagnostic_card,
    compose_multiview_diagnostic_board,
    load_or_extract_rig_skinning,
    parse_and_apply_pose_tweaks,
    project_bones_to_views,
    render_pose_blender_canvas,
    render_pose_lbs_canvas,
    resolve_views_for_canvas,
    write_updated_pose_json,
)

cv2.setNumThreads(1)  # Predictable memory/runtime on full-resolution frame sequences.
VERSION = "2.0.0"
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


def metrics(
    reference,
    candidate,
    tolerance=8,
    bones_2d_view=None,
    frame_bones=None,
    rig_meta=None,
    view_name="front",
):
    result = {
        "valid": bool(reference.any() and candidate.any()),
        "reference_pixels": int(reference.sum()),
        "render_pixels": int(candidate.sum()),
    }
    if not result["valid"]:
        return {
            **result,
            "reason": "empty silhouette",
            **{k: None for k in SCORES},
            "outline_passed": False,
            "centroid_shift_px": None,
            "bbox_reference": None,
            "bbox_candidate": None,
            "bbox_delta_px": None,
            "mean_contour_pull_px": None,
            "error_clusters": [],
            "bone_suggestions": [],
        }

    intersection = int((reference & candidate).sum())
    union = int((reference | candidate).sum())
    iou = intersection / union
    dice = 2 * intersection / int(reference.sum() + candidate.sum())

    re, ce = edge(reference), edge(candidate)
    dr, labels_r = cv2.distanceTransformWithLabels(
        (~re).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL
    )
    dc, labels_c = cv2.distanceTransformWithLabels(
        (~ce).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE, labelType=cv2.DIST_LABEL_PIXEL
    )
    distances = np.concatenate([dr[ce], dc[re]])
    boundary_mean_px = float(distances.mean())
    boundary_p95_px = float(np.percentile(distances, 95))
    boundary_within_tolerance = float((distances <= tolerance).mean())
    outline_passed = bool(
        iou >= 0.85
        and boundary_within_tolerance >= 0.90
        and boundary_p95_px <= max(float(tolerance) * 1.5, 4.0)
    )

    ref_ys, ref_xs = np.nonzero(reference)
    cand_ys, cand_xs = np.nonzero(candidate)
    centroid_shift_px = [
        round(float(cand_xs.mean() - ref_xs.mean()), 2),
        round(float(cand_ys.mean() - ref_ys.mean()), 2),
    ]
    bbox_reference = [int(ref_xs.min()), int(ref_ys.min()), int(ref_xs.max()) + 1, int(ref_ys.max()) + 1]
    bbox_candidate = [int(cand_xs.min()), int(cand_ys.min()), int(cand_xs.max()) + 1, int(cand_ys.max()) + 1]
    bbox_delta_px = [bbox_candidate[i] - bbox_reference[i] for i in range(4)]

    # Build O(1) pixel-label lookup tables for exact contour pull vectors
    ry, rx = np.nonzero(re)
    max_lr = int(labels_r.max())
    lut_rx = np.zeros(max_lr + 1, dtype=np.int32)
    lut_ry = np.zeros(max_lr + 1, dtype=np.int32)
    lut_rx[labels_r[ry, rx]] = rx
    lut_ry[labels_r[ry, rx]] = ry

    cy, cx = np.nonzero(ce)
    max_lc = int(labels_c.max())
    lut_cx = np.zeros(max_lc + 1, dtype=np.int32)
    lut_cy = np.zeros(max_lc + 1, dtype=np.int32)
    lut_cx[labels_c[cy, cx]] = cx
    lut_cy[labels_c[cy, cx]] = cy

    pull_x = float(np.mean(lut_rx[labels_r[cy, cx]] - cx))
    pull_y = float(np.mean(lut_ry[labels_r[cy, cx]] - cy))
    mean_contour_pull_px = [round(pull_x, 2), round(pull_y, 2)]

    # Extract connected error clusters with directional correction vectors (anchor on candidate -> target on reference)
    error_clusters = []
    for kind, err_mask in (("protruding", candidate & ~reference), ("missing", reference & ~candidate)):
        if not err_mask.any():
            continue
        num_cc, cc_labels, cc_stats, cc_centroids = cv2.connectedComponentsWithStats(err_mask.astype(np.uint8), 8)
        for cc_id in range(1, num_cc):
            area = int(cc_stats[cc_id, cv2.CC_STAT_AREA])
            if area < 12:
                continue
            comp_ys, comp_xs = np.nonzero(cc_labels == cc_id)
            if kind == "protruding":
                d_vals = dr[comp_ys, comp_xs]
                max_dist = float(d_vals.max())
                mean_dist = float(d_vals.mean())
                if max_dist < 1.5:
                    continue
                on_edge = ce[comp_ys, comp_xs]
                if on_edge.any():
                    edge_idx = np.nonzero(on_edge)[0]
                    best_local = int(edge_idx[int(np.argmax(d_vals[edge_idx]))])
                else:
                    best_local = int(np.argmax(d_vals))
                ax, ay = int(comp_xs[best_local]), int(comp_ys[best_local])
                lbl = int(labels_r[ay, ax])
                tx, ty = int(lut_rx[lbl]), int(lut_ry[lbl])
            else:
                d_vals = dc[comp_ys, comp_xs]
                max_dist = float(d_vals.max())
                mean_dist = float(d_vals.mean())
                if max_dist < 1.5:
                    continue
                on_edge = re[comp_ys, comp_xs]
                if on_edge.any():
                    edge_idx = np.nonzero(on_edge)[0]
                    best_local = int(edge_idx[int(np.argmax(d_vals[edge_idx]))])
                else:
                    best_local = int(np.argmax(d_vals))
                tx, ty = int(comp_xs[best_local]), int(comp_ys[best_local])
                lbl = int(labels_c[ty, tx])
                ax, ay = int(lut_cx[lbl]), int(lut_cy[lbl])

            left = int(cc_stats[cc_id, cv2.CC_STAT_LEFT])
            top = int(cc_stats[cc_id, cv2.CC_STAT_TOP])
            width = int(cc_stats[cc_id, cv2.CC_STAT_WIDTH])
            height = int(cc_stats[cc_id, cv2.CC_STAT_HEIGHT])
            error_clusters.append({
                "kind": kind,
                "box": [left, top, left + width, top + height],
                "area_px": area,
                "centroid_px": [round(float(cc_centroids[cc_id, 0]), 1), round(float(cc_centroids[cc_id, 1]), 1)],
                "mean_distance_px": round(mean_dist, 2),
                "max_distance_px": round(max_dist, 2),
                "anchor_contour_px": [ax, ay],
                "target_contour_px": [tx, ty],
                "correction_vector_px": [tx - ax, ty - ay],
            })

    error_clusters.sort(key=lambda c: c["area_px"] * c["max_distance_px"], reverse=True)
    error_clusters = error_clusters[:8]
    bone_suggestions = (
        attribute_clusters_to_bones(error_clusters, bones_2d_view, frame_bones, rig_meta, view_name)
        if bones_2d_view
        else []
    )

    return {
        **result,
        "iou": iou,
        "dice": dice,
        "boundary_mean_px": boundary_mean_px,
        "boundary_p95_px": boundary_p95_px,
        "boundary_within_tolerance": boundary_within_tolerance,
        "outline_passed": outline_passed,
        "centroid_shift_px": centroid_shift_px,
        "bbox_reference": bbox_reference,
        "bbox_candidate": bbox_candidate,
        "bbox_delta_px": bbox_delta_px,
        "mean_contour_pull_px": mean_contour_pull_px,
        "error_clusters": error_clusters,
        "bone_suggestions": bone_suggestions,
    }


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


def _write_csv_and_html(output: Path, report: dict, stages: list[str]) -> None:
    dump(output / "report.json", report)
    with (output / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["frame", "seconds", "view", "region", "iou_gain", *[f"{s}_{k}" for s in stages for k in ("valid", *SCORES)]]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({**{k: row[k] for k in columns[:5]}, **{f"{s}_{k}": row[s][k] for s in stages for k in ("valid", *SCORES)}})
    template = (Path(__file__).resolve().parent.parent / "assets/report.html").read_text(encoding="utf-8")
    embedded = json.dumps(report, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    (output / "index.html").write_text(template.replace("__REPORT_DATA__", embedded), encoding="utf-8")


def review(manifest, output, frames="all", overwrite=False):
    config = load_config(manifest)
    chosen = selected_frames(frames, config["frame_count"])
    if output.exists() and not overwrite:
        raise ValueError("output already exists; choose a new review directory")
    # Prevent outputs inside immutable inputs, including a directory not created yet.
    for source in config["sources"].values():
        for key in ("directory", "mask_directory"):
            if key in source and output.is_relative_to(Path(source[key])):
                raise ValueError("output must be outside source directories")
    if output.exists() and overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
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
        frame_cards = []
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
            regions = [{"name": "whole", "box": [0, 0, crops["reference"].width, crops["reference"].height]}, *view.get("regions", [])]
            whole_cand_metrics = None
            for region in regions:
                left, top, right, bottom = region["box"]
                reference = ref_mask[top:bottom, left:right]
                row = {"frame": frame, "seconds": frame / config["fps"], "view": view["name"], "region": region["name"], "box": region["box"]}
                for stage in stages:
                    row[stage] = metrics(
                        reference,
                        silhouettes[stage][top:bottom, left:right],
                        config["boundary_tolerance_px"],
                        view_name=view["name"],
                    )
                    if not row[stage]["valid"]:
                        invalid.append({"frame": frame, "view": view["name"], "region": region["name"], "stage": stage, "reason": row[stage]["reason"]})
                if region["name"] == "whole":
                    whole_cand_metrics = row["candidate"]
                row["iou_gain"] = row["candidate"]["iou"] - row["baseline"]["iou"] if "baseline" in stages and all(row[s]["valid"] for s in stages) else None
                rows.append(row)
            if whole_cand_metrics is not None:
                card = build_view_diagnostic_card(
                    frame_idx=frame,
                    view_name=view["name"],
                    ref_crop=crops["reference"],
                    cand_crop=crops["candidate"],
                    compare_rgb=blend,
                    view_metrics=whole_cand_metrics,
                )
                card_rel = f"images/{stem}_panel.png"
                card.save(output / card_rel)
                pictures["panel"] = card_rel
                frame_cards.append(card)
            assets.append({"frame": frame, "view": view["name"], "images": pictures})
        if frame_cards and len(chosen) == 1:
            compose_multiview_diagnostic_board(frame_cards, output / "panel_compare.png")
    valid_frames = sorted({r["frame"] for r in rows if all(r[s]["valid"] for s in stages)})
    complete = len(chosen) == config["frame_count"] and not missing and not invalid and bool(rows)
    verified = config["alignment"]["status"] == "verified"
    whole_rows = [r for r in rows if r["region"] == "whole" and r["candidate"]["valid"]]
    outline_passed = bool(whole_rows) and all(r["candidate"].get("outline_passed", False) for r in whole_rows)
    report = {"tool": "frame-overlay-review", "tool_version": VERSION,
              "created_at": datetime.now(timezone.utc).isoformat(), "manifest_sha256": sha(manifest),
              "config": config, "alignment": config["alignment"],
              "coordinate_contract": "Original pixels; fixed shared view/ROI boxes; no registration, translation, scaling or retiming.",
              "metric_scope": "2D silhouette agreement; not 3D or animation quality acceptance.",
              "status": "complete" if complete and verified else "incomplete",
              "outline_passed": outline_passed,
              "panel_compare": "panel_compare.png" if (output / "panel_compare.png").is_file() else None,
              "coverage": {"total_frames": config["frame_count"], "selected_frames": chosen,
                           "available_frames": sorted({r['frame'] for r in rows}), "frames_with_valid_regions": valid_frames,
                           "full_sequence": len(chosen) == config["frame_count"], "complete": complete},
              "missing": missing, "invalid": invalid, "summary": summarize(rows, stages), "rows": rows,
              "worst_frames": sorted([r for r in rows if r["candidate"]["valid"]], key=lambda r: r["candidate"]["iou"])[:20],
              "regressed_frames": sorted([r for r in rows if r["iou_gain"] is not None and r["iou_gain"] < 0], key=lambda r: r["iou_gain"]),
              "provenance": provenance, "assets": assets}
    _write_csv_and_html(output, report, stages)
    return report


def _resolve_single_or_dir_image(path_spec: str | Path, frame: int, pattern: str) -> Path:
    p = Path(path_spec).resolve()
    if p.is_dir():
        rel = Path(pattern.format(frame=frame))
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("frame pattern must stay inside its directory")
        p = p / rel
    if not p.is_file():
        raise FileNotFoundError(f"Image file not found: {p}")
    return p


def _parse_view_image_map(specs: list[str] | dict[str, str | Path] | None, frame: int, pattern: str) -> dict[str, Path]:
    if not specs:
        return {}
    out: dict[str, Path] = {}
    if isinstance(specs, dict):
        for k, v in specs.items():
            out[str(k).strip()] = _resolve_single_or_dir_image(v, frame, pattern)
    else:
        for item in specs:
            if "=" in item:
                v_name, p_str = item.split("=", 1)
            elif ":" in item:
                v_name, p_str = item.split(":", 1)
            else:
                raise ValueError(f"Invalid per-view image spec {item!r}; expected VIEW=PATH")
            out[v_name.strip()] = _resolve_single_or_dir_image(p_str.strip(), frame, pattern)
    return out


def _stitch_view_images(view_paths: dict[str, Path], view_order: list[str] | None = None) -> tuple[Image.Image, list[dict]]:
    names = view_order if view_order else list(view_paths.keys())
    loaded = []
    for v_name in names:
        if v_name not in view_paths:
            raise ValueError(f"Missing view {v_name!r} in per-view image inputs")
        with Image.open(view_paths[v_name]) as im:
            loaded.append((v_name, im.copy()))
    total_w = sum(im.width for _, im in loaded)
    max_h = max(im.height for _, im in loaded)
    has_alpha = any("A" in im.getbands() for _, im in loaded)
    mode = "RGBA" if has_alpha else "RGB"
    comp = Image.new(mode, (total_w, max_h))
    views = []
    x_off = 0
    for v_name, im in loaded:
        comp.paste(im.convert(mode), (x_off, 0))
        views.append({"name": v_name, "box": [x_off, 0, x_off + im.width, im.height], "regions": []})
        x_off += im.width
    return comp, views


def review_frame(
    output: Path,
    reference: str | Path | None = None,
    candidate: str | Path | None = None,
    baseline: str | Path | None = None,
    ref_views: list[str] | dict[str, str | Path] | None = None,
    cand_views: list[str] | dict[str, str | Path] | None = None,
    base_views: list[str] | dict[str, str | Path] | None = None,
    ref_mask: str | Path | None = None,
    cand_mask: str | Path | None = None,
    base_mask: str | Path | None = None,
    frame: int = 0,
    fps: float = 24.0,
    action: str = "single_frame_review",
    layout: str = "auto",
    views: list[str] | None = None,
    rois: list[str] | None = None,
    segmentation_mode: str = "border",
    threshold: float = 45.0,
    min_component: int = 36,
    border: int = 20,
    background: list[float] | None = None,
    tolerance: float = 8.0,
    rig_meta_path: str | Path | None = None,
    pose_json_path: str | Path | None = None,
    rig_skinning_path: str | Path | None = None,
    blend_path: str | Path | None = None,
    render_mode: str = "auto",
    tweaks: list[str] | None = None,
    set_eulers: list[str] | None = None,
    view_tweaks: list[str] | None = None,
    root_offset: str | list[float] | None = None,
    auto_nudge_iters: int = 0,
    out_pose_json: str | Path | None = None,
    pattern: str = "frame_{frame:04d}.png",
    overwrite: bool = False,
) -> dict:
    """Single-frame multi-view contour alignment diagnostic & Agent-in-the-loop pose refinement."""
    output = Path(output).resolve()
    if output.exists() and not overwrite:
        raise ValueError("output already exists; choose a new review directory or pass --overwrite")

    # 1. Load reference image(s)
    provenance = []
    ref_view_map = _parse_view_image_map(ref_views, frame, pattern)
    stitched_views = None
    if ref_view_map:
        ref_image, stitched_views = _stitch_view_images(ref_view_map)
        for v_name, p in ref_view_map.items():
            provenance.append({"frame": frame, "source": f"reference:{v_name}", "path": str(p), "sha256": sha(p)})
    elif reference is not None:
        ref_path = _resolve_single_or_dir_image(reference, frame, pattern)
        if output.is_relative_to(ref_path.parent) and ref_path.parent != output.parent:
            raise ValueError("output must be outside source directories")
        with Image.open(ref_path) as im:
            ref_image = im.copy()
        provenance.append({"frame": frame, "source": "reference", "path": str(ref_path), "sha256": sha(ref_path)})
    else:
        raise ValueError("frame review requires --reference or --ref-view")

    canvas_size = ref_image.size
    rig_meta = json.loads(Path(rig_meta_path).read_text(encoding="utf-8")) if rig_meta_path else None
    pose_data = json.loads(Path(pose_json_path).read_text(encoding="utf-8")) if pose_json_path else None

    # 2. Resolve multi-view studio layout on the reference canvas
    if stitched_views and not views:
        resolved_views = resolve_views_for_canvas(
            canvas_size[0],
            canvas_size[1],
            explicit_views=[f"{v['name']}:{v['box'][0]},{v['box'][1]},{v['box'][2]},{v['box'][3]}" for v in stitched_views],
            explicit_rois=rois,
        )
    else:
        calib = (rig_meta or {}).get("calibration")
        resolved_views = resolve_views_for_canvas(
            canvas_size[0],
            canvas_size[1],
            layout=layout,
            explicit_views=views,
            explicit_rois=rois,
            calib=calib,
        )

    names = set()
    for v in resolved_views:
        v_name = v.get("name", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", v_name) or v_name in names:
            raise ValueError("view names must be unique simple identifiers")
        names.add(v_name)
        if not box_ok(v.get("box"), *canvas_size):
            raise ValueError(f"view {v_name} box {v.get('box')} is outside canvas {canvas_size}")
        vw, vh = v["box"][2] - v["box"][0], v["box"][3] - v["box"][1]
        r_names = {"whole"}
        for r in v.get("regions", []):
            rn = r.get("name", "")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", rn) or rn in r_names or not box_ok(r.get("box"), vw, vh):
                raise ValueError("invalid/duplicate region or rectangle outside view")
            r_names.add(rn)

    seg_settings = {
        **DEFAULT_MASK,
        "mode": segmentation_mode,
        "threshold": float(threshold),
        "min_component": int(min_component),
        "border": int(border),
    }
    if background is not None:
        seg_settings["background"] = [float(x) for x in background]

    # 3. Load explicit baseline/candidate images if provided
    base_image = None
    base_view_map = _parse_view_image_map(base_views, frame, pattern)
    if base_view_map:
        base_image, _ = _stitch_view_images(base_view_map, [v["name"] for v in resolved_views])
        for v_name, p in base_view_map.items():
            provenance.append({"frame": frame, "source": f"baseline:{v_name}", "path": str(p), "sha256": sha(p)})
    elif baseline is not None:
        base_path = _resolve_single_or_dir_image(baseline, frame, pattern)
        with Image.open(base_path) as im:
            base_image = im.copy()
        provenance.append({"frame": frame, "source": "baseline", "path": str(base_path), "sha256": sha(base_path)})

    cand_image = None
    cand_view_map = _parse_view_image_map(cand_views, frame, pattern)
    if cand_view_map:
        cand_image, _ = _stitch_view_images(cand_view_map, [v["name"] for v in resolved_views])
        for v_name, p in cand_view_map.items():
            provenance.append({"frame": frame, "source": f"candidate:{v_name}", "path": str(p), "sha256": sha(p)})
    elif candidate is not None:
        cand_path = _resolve_single_or_dir_image(candidate, frame, pattern)
        with Image.open(cand_path) as im:
            cand_image = im.copy()
        provenance.append({"frame": frame, "source": "candidate", "path": str(cand_path), "sha256": sha(cand_path)})

    explicit_masks = {}
    for stage_name, m_spec in (("reference", ref_mask), ("candidate", cand_mask), ("baseline", base_mask)):
        if m_spec is not None:
            mp = _resolve_single_or_dir_image(m_spec, frame, pattern)
            with Image.open(mp) as mim:
                explicit_masks[stage_name] = mim.copy()
            if explicit_masks[stage_name].size != canvas_size:
                raise ValueError(f"{stage_name} mask dimensions {explicit_masks[stage_name].size} differ from {canvas_size}")

    # 4. Handle 3D Rig FK Pose Tweaking & Single-Frame Rendering
    frame_bones = None
    bones_2d_by_view = {}
    pose_refinement_info = None
    has_manual_tweaks = bool(tweaks or set_eulers or view_tweaks or root_offset)
    r_mode = (render_mode or "auto").strip().lower()
    if r_mode not in ("auto", "lbs", "blender", "none"):
        raise ValueError("--render-mode must be auto, lbs, blender, or none")

    def _render_from_bones(b_state: dict) -> Image.Image:
        if r_mode == "blender":
            if not blend_path:
                raise ValueError("--render-mode blender requires --blend <path.blend>")
            return render_pose_blender_canvas(Path(blend_path), rig_meta, b_state, resolved_views, canvas_size)
        skin = load_or_extract_rig_skinning(
            rig_meta,
            rig_skinning_path=Path(rig_skinning_path) if rig_skinning_path else None,
            blend_path=Path(blend_path) if blend_path else None,
        )
        return render_pose_lbs_canvas(rig_meta, b_state, resolved_views, canvas_size, skin)

    if rig_meta is not None:
        should_render = r_mode != "none" and (
            cand_image is None
            or has_manual_tweaks
            or int(auto_nudge_iters) > 0
            or r_mode in ("lbs", "blender")
        )
        # If tweaks or auto-nudge are requested and no baseline was explicitly passed, capture the pre-tweak baseline
        if should_render and (has_manual_tweaks or int(auto_nudge_iters) > 0) and base_image is None:
            if cand_image is not None:
                base_image = cand_image.copy()
                if "A" in base_image.getbands() and "baseline" not in explicit_masks:
                    explicit_masks["baseline"] = base_image.getchannel("A")
            else:
                untweaked_bones, _ = parse_and_apply_pose_tweaks(rig_meta, pose_data, frame_idx=frame)
                base_image = _render_from_bones(untweaked_bones)
                if "A" in base_image.getbands() and "baseline" not in explicit_masks:
                    explicit_masks["baseline"] = base_image.getchannel("A")

        frame_bones, tweak_meta = parse_and_apply_pose_tweaks(
            rig_meta,
            pose_data,
            frame_idx=frame,
            tweaks=tweaks,
            set_eulers=set_eulers,
            view_tweaks=view_tweaks,
            root_offset=root_offset,
        )
        calib = rig_meta.get("calibration", {})
        bones_2d_by_view = project_bones_to_views(frame_bones, rig_meta.get("bones", {}), calib, resolved_views)

        if should_render:
            cand_image = _render_from_bones(frame_bones)
            if "A" in cand_image.getbands() and cand_mask is None:
                explicit_masks["candidate"] = cand_image.getchannel("A")

        # Optional closed-loop automatic contour nudge iterations
        auto_history = []
        accum_view_tweaks = list(view_tweaks or [])
        if should_render and int(auto_nudge_iters) > 0:
            ref_sils = {
                v["name"]: silhouette(
                    ref_image.crop(v["box"]),
                    seg_settings,
                    explicit_masks["reference"].crop(v["box"]) if "reference" in explicit_masks else None,
                )
                for v in resolved_views
            }
            for it in range(int(auto_nudge_iters)):
                cand_sils = {
                    v["name"]: silhouette(
                        cand_image.crop(v["box"]),
                        seg_settings,
                        explicit_masks["candidate"].crop(v["box"]) if "candidate" in explicit_masks else None,
                    )
                    for v in resolved_views
                }
                iter_ious = []
                iter_suggestions = []
                all_passed = True
                for v in resolved_views:
                    vn = v["name"]
                    m_cur = metrics(
                        ref_sils[vn],
                        cand_sils[vn],
                        tolerance=tolerance,
                        bones_2d_view=bones_2d_by_view.get(vn),
                        frame_bones=frame_bones,
                        rig_meta=rig_meta,
                        view_name=vn,
                    )
                    if m_cur["valid"]:
                        iter_ious.append(m_cur["iou"])
                        if not m_cur["outline_passed"]:
                            all_passed = False
                        for s_item in m_cur.get("bone_suggestions", []):
                            iter_suggestions.append(s_item)
                mean_iou_before = float(np.mean(iter_ious)) if iter_ious else 0.0
                if all_passed or not iter_suggestions:
                    break
                iter_suggestions.sort(key=lambda s: s["total_error_area_px"] * abs(s["suggested_view_rot_deg"]), reverse=True)
                chosen_step_tweaks = []
                used_bones = set()
                for s_item in iter_suggestions:
                    b_n = s_item["bone"]
                    if b_n in used_bones:
                        continue
                    step_deg = round(0.65 * float(s_item["suggested_view_rot_deg"]), 2)
                    if abs(step_deg) < 0.25:
                        continue
                    chosen_step_tweaks.append(f"{s_item['view']}:{b_n}:{step_deg:+.2f}")
                    used_bones.add(b_n)
                    if len(chosen_step_tweaks) >= 2:
                        break
                if not chosen_step_tweaks:
                    break

                trial_view_tweaks = accum_view_tweaks + chosen_step_tweaks
                trial_bones, _ = parse_and_apply_pose_tweaks(
                    rig_meta,
                    pose_data,
                    frame_idx=frame,
                    tweaks=tweaks,
                    set_eulers=set_eulers,
                    view_tweaks=trial_view_tweaks,
                    root_offset=root_offset,
                )
                trial_cand = _render_from_bones(trial_bones)
                trial_mask = trial_cand.getchannel("A") if ("A" in trial_cand.getbands() and cand_mask is None) else explicit_masks.get("candidate")
                trial_ious = []
                for v in resolved_views:
                    vn = v["name"]
                    t_sil = silhouette(
                        trial_cand.crop(v["box"]),
                        seg_settings,
                        trial_mask.crop(v["box"]) if trial_mask is not None else None,
                    )
                    m_t = metrics(ref_sils[vn], t_sil, tolerance=tolerance, view_name=vn)
                    if m_t["valid"]:
                        trial_ious.append(m_t["iou"])
                mean_iou_after = float(np.mean(trial_ious)) if trial_ious else 0.0
                if mean_iou_after >= mean_iou_before - 1e-5:
                    accum_view_tweaks = trial_view_tweaks
                    frame_bones = trial_bones
                    cand_image = trial_cand
                    if trial_mask is not None:
                        explicit_masks["candidate"] = trial_mask
                    bones_2d_by_view = project_bones_to_views(frame_bones, rig_meta.get("bones", {}), calib, resolved_views)
                    auto_history.append({
                        "iter": it + 1,
                        "applied_view_tweaks": chosen_step_tweaks,
                        "mean_iou_before": round(mean_iou_before, 4),
                        "mean_iou_after": round(mean_iou_after, 4),
                    })
                else:
                    break

        if out_pose_json is not None:
            out_pose_p = Path(out_pose_json).resolve()
            write_updated_pose_json(pose_data, frame, frame_bones, out_pose_p)
        else:
            out_pose_p = None

        pose_refinement_info = {
            "frame": int(frame),
            "render_mode": r_mode if should_render else "none",
            "tweaks_applied": tweak_meta["tweaks_applied"],
            "auto_nudge_history": auto_history,
            "out_pose_json": str(out_pose_p) if out_pose_p else None,
        }

    if cand_image is None:
        raise ValueError("frame review requires --candidate, --cand-view, or (--rig-meta with skinning/blend)")

    if cand_image.size != canvas_size:
        raise ValueError(f"candidate canvas {cand_image.size} differs from reference {canvas_size}; no resizing allowed")
    if base_image is not None and base_image.size != canvas_size:
        raise ValueError(f"baseline canvas {base_image.size} differs from reference {canvas_size}; no resizing allowed")

    if output.exists() and overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)

    images = {"reference": ref_image, "candidate": cand_image}
    stages = ["candidate"]
    if base_image is not None:
        images["baseline"] = base_image
        stages.append("baseline")

    rows, assets, invalid, view_cards = [], [], [], []
    all_bone_suggestions = []

    for view in resolved_views:
        v_name = view["name"]
        crops = {name: img.crop(view["box"]) for name, img in images.items()}
        silhouettes = {
            name: silhouette(
                img,
                seg_settings,
                explicit_masks[name].crop(view["box"]) if name in explicit_masks else None,
            )
            for name, img in crops.items()
        }
        stem = f"{frame:06d}_{v_name}"
        pictures = {}
        for name, img in crops.items():
            pictures[name] = f"images/{stem}_{name}.jpg"
            img.convert("RGB").save(output / pictures[name], quality=90)

        ref_m, cand_m = silhouettes["reference"], silhouettes["candidate"]
        ref_edge, cand_edge = edge(ref_m), edge(cand_m)
        blend = np.array(Image.blend(crops["reference"].convert("RGB"), crops["candidate"].convert("RGB"), 0.5))
        blend[ref_edge], blend[cand_edge], blend[ref_edge & cand_edge] = (30, 240, 255), (255, 55, 210), (255, 255, 255)
        difference = np.zeros((*ref_m.shape, 3), dtype=np.uint8)
        difference[ref_m], difference[cand_m], difference[ref_m & cand_m] = (30, 240, 255), (255, 55, 210), (235, 235, 235)
        for name, array in (("overlay", blend), ("mask", difference)):
            pictures[name] = f"images/{stem}_{name}.png"
            Image.fromarray(array).save(output / pictures[name])

        regions = [{"name": "whole", "box": [0, 0, crops["reference"].width, crops["reference"].height]}, *view.get("regions", [])]
        whole_cand_metrics = None
        for region in regions:
            left, top, right, bottom = region["box"]
            ref_sub = ref_m[top:bottom, left:right]
            row = {"frame": frame, "seconds": frame / fps, "view": v_name, "region": region["name"], "box": region["box"]}
            for stage in stages:
                is_whole_cand = stage == "candidate" and region["name"] == "whole"
                row[stage] = metrics(
                    ref_sub,
                    silhouettes[stage][top:bottom, left:right],
                    tolerance=tolerance,
                    bones_2d_view=bones_2d_by_view.get(v_name) if is_whole_cand else None,
                    frame_bones=frame_bones if is_whole_cand else None,
                    rig_meta=rig_meta if is_whole_cand else None,
                    view_name=v_name,
                )
                if not row[stage]["valid"]:
                    invalid.append({"frame": frame, "view": v_name, "region": region["name"], "stage": stage, "reason": row[stage]["reason"]})
            if region["name"] == "whole":
                whole_cand_metrics = row["candidate"]
                for s_item in whole_cand_metrics.get("bone_suggestions", []):
                    all_bone_suggestions.append(s_item)
            row["iou_gain"] = (
                row["candidate"]["iou"] - row["baseline"]["iou"]
                if "baseline" in stages and all(row[s]["valid"] for s in stages)
                else None
            )
            rows.append(row)

        if whole_cand_metrics is not None:
            card = build_view_diagnostic_card(
                frame_idx=frame,
                view_name=v_name,
                ref_crop=crops["reference"],
                cand_crop=crops["candidate"],
                compare_rgb=blend,
                view_metrics=whole_cand_metrics,
                bones_2d_view=bones_2d_by_view.get(v_name),
            )
            card_rel = f"images/{stem}_panel.png"
            card.save(output / card_rel)
            pictures["panel"] = card_rel
            view_cards.append(card)

        assets.append({"frame": frame, "view": v_name, "images": pictures})

    panel_compare_rel = None
    if view_cards:
        compose_multiview_diagnostic_board(view_cards, output / "panel_compare.png")
        panel_compare_rel = "panel_compare.png"

    all_bone_suggestions.sort(key=lambda s: s["total_error_area_px"] * abs(s["suggested_view_rot_deg"]), reverse=True)
    if pose_refinement_info is not None:
        pose_refinement_info["recommended_next_tweak_view_args"] = [s["suggested_tweak_view_arg"] for s in all_bone_suggestions[:5]]
        pose_refinement_info["recommended_next_tweak_args"] = [s["suggested_tweak_arg"] for s in all_bone_suggestions[:5]]

    valid_frames = sorted({r["frame"] for r in rows if all(r[s]["valid"] for s in stages)})
    complete = not invalid and bool(rows)
    whole_rows = [r for r in rows if r["region"] == "whole" and r["candidate"]["valid"]]
    outline_passed = bool(whole_rows) and all(r["candidate"].get("outline_passed", False) for r in whole_rows)

    config_snapshot = {
        "version": 1,
        "action": action,
        "fps": fps,
        "frame_count": 1,
        "boundary_tolerance_px": tolerance,
        "segmentation": seg_settings,
        "views": resolved_views,
        "alignment": {"status": "verified", "evidence": "Single-frame multi-view direct overlay comparison"},
    }
    report = {
        "tool": "frame-overlay-review",
        "tool_version": VERSION,
        "mode": "single_frame",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config_snapshot,
        "alignment": config_snapshot["alignment"],
        "coordinate_contract": "Original pixels; fixed shared view/ROI boxes; no registration, translation, scaling or retiming.",
        "metric_scope": "2D silhouette agreement & directional contour pull vectors for Agent-in-the-loop pose refinement.",
        "status": "complete" if complete else "incomplete",
        "outline_passed": outline_passed,
        "panel_compare": panel_compare_rel,
        "pose_refinement": pose_refinement_info,
        "top_bone_suggestions": all_bone_suggestions[:8],
        "coverage": {
            "total_frames": 1,
            "selected_frames": [frame],
            "available_frames": sorted({r["frame"] for r in rows}),
            "frames_with_valid_regions": valid_frames,
            "full_sequence": True,
            "complete": complete,
        },
        "missing": [],
        "invalid": invalid,
        "summary": summarize(rows, stages),
        "rows": rows,
        "worst_frames": sorted([r for r in rows if r["candidate"]["valid"]], key=lambda r: r["candidate"]["iou"])[:20],
        "regressed_frames": sorted([r for r in rows if r["iou_gain"] is not None and r["iou_gain"] < 0], key=lambda r: r["iou_gain"]),
        "provenance": provenance,
        "assets": assets,
    }
    _write_csv_and_html(output, report, stages)
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

    run = sub.add_parser("review", help="Write a comparison report from a manifest")
    run.add_argument("manifest", type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--frames", default="all")
    run.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")

    frm = sub.add_parser("frame", help="Single-frame multi-view contour comparison, bone attribution & pose tweaking")
    frm.add_argument("--reference", help="Reference composite image file or frame directory")
    frm.add_argument("--ref-view", action="append", help="Per-view reference image VIEW=PATH; repeat for multiple views")
    frm.add_argument("--candidate", help="Candidate composite image file or frame directory (optional if rendering from 3D pose)")
    frm.add_argument("--cand-view", action="append", help="Per-view candidate image VIEW=PATH; repeat for multiple views")
    frm.add_argument("--baseline", help="Optional baseline composite image file or directory for IoU gain tracking")
    frm.add_argument("--base-view", action="append", help="Optional per-view baseline image VIEW=PATH")
    frm.add_argument("--ref-mask", help="Optional explicit reference silhouette mask")
    frm.add_argument("--cand-mask", help="Optional explicit candidate silhouette mask")
    frm.add_argument("--base-mask", help="Optional explicit baseline silhouette mask")
    frm.add_argument("--output", required=True, type=Path, help="Output directory for report.json, panel_compare.png, and index.html")
    frm.add_argument("--frame", type=int, default=0, help="Frame index (default: 0)")
    frm.add_argument("--fps", type=float, default=24.0, help="Sequence FPS for timestamp display (default: 24)")
    frm.add_argument("--action", default="single_frame_review", help="Action label for report header")
    frm.add_argument("--pattern", default="frame_{frame:04d}.png", help="Filename pattern when --reference/--candidate is a directory")
    frm.add_argument("--layout", default="auto", choices=("auto", "single", "2view", "3view"), help="Multi-view split preset")
    frm.add_argument("--view", action="append", help="Custom view rectangle NAME:left,top,right,bottom")
    frm.add_argument("--roi", action="append", help="Custom ROI rectangle inside a view VIEW:NAME:left,top,right,bottom")
    frm.add_argument("--segmentation-mode", default="border", choices=("border", "alpha"))
    frm.add_argument("--threshold", type=float, default=45.0)
    frm.add_argument("--min-component", type=int, default=36)
    frm.add_argument("--border", type=int, default=20)
    frm.add_argument("--tolerance", type=float, default=8.0, help="Boundary tolerance in pixels (default: 8)")
    frm.add_argument("--rig-meta", help="Path to studio_rig_meta.json for 2D bone projection, attribution & FK tweaking")
    frm.add_argument("--pose-json", help="Path to predicted_trajectory.json or single-frame pose JSON")
    frm.add_argument("--rig-skinning", help="Path to skinning .npz for fast (~15ms) LBS single-frame mesh rendering")
    frm.add_argument("--blend", help="Path to character .blend file for skinning extraction or Blender rendering")
    frm.add_argument("--render-mode", default="auto", choices=("auto", "lbs", "blender", "none"))
    frm.add_argument("--tweak", action="append", help="Add local Euler delta BONE:drx_deg,dry_deg,drz_deg")
    frm.add_argument("--set-euler", action="append", help="Set local Euler angles BONE:rx_deg,ry_deg,rz_deg")
    frm.add_argument("--tweak-view", action="append", help="Rotate bone in 2D view plane VIEW:BONE:deg")
    frm.add_argument("--tweak-root", help="Translate root bone dx,dy,dz in 3D world space")
    frm.add_argument("--auto-nudge-iters", type=int, default=0, help="Closed-loop automatic contour nudge iterations (default: 0)")
    frm.add_argument("--out-pose-json", help="Write updated pose/trajectory JSON to this path")
    frm.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")

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

        if args.command == "frame":
            report = review_frame(
                output=args.output,
                reference=args.reference,
                candidate=args.candidate,
                baseline=args.baseline,
                ref_views=args.ref_view,
                cand_views=args.cand_view,
                base_views=args.base_view,
                ref_mask=args.ref_mask,
                cand_mask=args.cand_mask,
                base_mask=args.base_mask,
                frame=args.frame,
                fps=args.fps,
                action=args.action,
                layout=args.layout,
                views=args.view,
                rois=args.roi,
                segmentation_mode=args.segmentation_mode,
                threshold=args.threshold,
                min_component=args.min_component,
                border=args.border,
                tolerance=args.tolerance,
                rig_meta_path=args.rig_meta,
                pose_json_path=args.pose_json,
                rig_skinning_path=args.rig_skinning,
                blend_path=args.blend,
                render_mode=args.render_mode,
                tweaks=args.tweak,
                set_eulers=args.set_euler,
                view_tweaks=args.tweak_view,
                root_offset=args.tweak_root,
                auto_nudge_iters=args.auto_nudge_iters,
                out_pose_json=args.out_pose_json,
                pattern=args.pattern,
                overwrite=args.overwrite,
            )
            whole_summaries = [s for s in report["summary"] if s["region"] == "whole" and s["candidate"]["iou"] is not None]
            mean_iou = round(float(np.mean([s["candidate"]["iou"] for s in whole_summaries])), 4) if whole_summaries else None
            mean_bpx = round(float(np.mean([s["candidate"]["boundary_mean_px"] for s in whole_summaries])), 2) if whole_summaries else None
            gains = [s["iou_gain"] for s in whole_summaries if s.get("iou_gain") is not None]
            mean_gain = round(float(np.mean(gains)), 4) if gains else None
            out_dir = args.output.resolve()
            summary_payload = {
                "status": report["status"],
                "frame": args.frame,
                "outline_passed": report["outline_passed"],
                "views": [v["name"] for v in report["config"]["views"]],
                "mean_iou": mean_iou,
                "mean_boundary_px": mean_bpx,
                "iou_gain": mean_gain,
                "top_bone_suggestions": [
                    {
                        "view": s["view"],
                        "bone": s["bone"],
                        "tweak_view": s["suggested_tweak_view_arg"],
                        "tweak_euler": s["suggested_tweak_arg"],
                        "error_area_px": s["total_error_area_px"],
                    }
                    for s in report.get("top_bone_suggestions", [])[:5]
                ],
                "panel_compare": str(out_dir / "panel_compare.png") if report.get("panel_compare") else None,
                "report_json": str(out_dir / "report.json"),
                "report": str(out_dir / "index.html"),
                "out_pose_json": (report.get("pose_refinement") or {}).get("out_pose_json"),
            }
            print(json.dumps(summary_payload, ensure_ascii=False))
            return 0 if report["status"] == "complete" else 2

        report = review(args.manifest.resolve(), args.output.resolve(), args.frames, overwrite=args.overwrite)
        print(json.dumps({"status": report["status"], "available_frames": len(report["coverage"]["available_frames"]),
                          "total_frames": report["coverage"]["total_frames"], "missing": len(report["missing"]),
                          "invalid": len(report["invalid"]), "report": str(args.output.resolve() / "index.html")}, ensure_ascii=False))
        return 0 if report["status"] == "complete" else 2
    except (ValueError, KeyError, TypeError, OSError, cv2.error) as error:
        print(f"frame-overlay-review: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
