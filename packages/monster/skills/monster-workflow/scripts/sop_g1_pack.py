#!/usr/bin/env python3
"""sop_g1_pack.py — Stage 2 4-Candidate Decoration Restoration, Canonical Bone Naming & G1 Review Pack (SOP §3).

Automates the entire G1 preparation workflow with real Headless Blender + FFmpeg evidence:
  1. Processes all 4 SkinTokens candidates (`candidate_0..3.glb`) via `studio_bpy_worker.py`:
     - Restores detached decorations (`rigid` -> Bone Parent; `flex` -> independent skinned sub-mesh)
     - Enforces configured canonical bone naming (V3 anatomical names by default) (`canonicalize_rig_bone_names_in_blender`)
     - Hard-gates on `validate_canonical_rig_bone_names` (`valid == True`, zero `bone_0..N` anonymous bones)
     - Saves uncompressed `candidate_{0..3}_full.blend` (`compress=False`) and `candidate_{0..3}_rig_meta.json`
  2. Generates the 16 Standard Octahedral Bone Views (SOP §3.2) in real Blender (`BLENDER_WORKBENCH`):
     - 4 candidates × 4 views (`front`, `side`, `back`, `three_quarter`) with 3D candidate mesh +
       3D Octahedral bone overlay (`cand_{0..3}_{view}_octahedral.png`, 16 images total, `640x640`)
  3. Drives all 4 candidates with the same SOP §3.3 Archetype Test Motion (`winged`, `humanoid`,
     `quadruped`, `aberrant`) on an identical timeline, phase, camera, and lighting in Blender:
     - Evaluates real deformed mesh displacement, edge stretch ratios, and unweighted vertices (`follow_through_issues.json`)
     - Renders synchronized `front`, `side`, and `three_quarter` 2x2 comparison frame sequences (`1280x720`)
     - Encodes real H.264/MP4 videos via `ffmpeg` and verifies `ftyp` + `moov` + non-empty `mdat` atoms
  4. Writes `g1_report.json` with SHA-256 evidence fingerprints, builds `g1_review.html`, and updates
     the SOP ledger to `awaiting_rig_review`. Never falls back to template stick figures or 24-byte MP4 stubs.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from contextlib import nullcontext
import uuid
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sop_gate import (  # noqa: E402
    MONSTER_ROOT,
    WORKER_DIR,
    derive_invalidations,
    load_ledger,
    mark_awaiting_review,
    now_taipei_iso,
    save_ledger,
    sha256_file,
)
from bone_taxonomy import (  # noqa: E402
    ARCHETYPE_SPECS,
    BONE_TAXONOMY_SCHEMA,
    G1_SYNC_VIEWS,
    G1_VIEWS,
    synthesize_archetype_pose_eulers,
    validate_canonical_rig_bone_names,
)
from sop_preflight import (  # noqa: E402
    _worker_command,
    _write_report,
    resolve_scratch_dirs,
)

G1_REVIEW_CONTRACT = "monster-candidate-video-review-v1"


def _validate_candidate_video_map(videos: dict) -> dict[int, dict[str, str]]:
    """Accept JSON string IDs, but never silently omit a candidate or view."""
    if not isinstance(videos, dict) or {str(cid) for cid in videos} != {str(i) for i in range(4)} or len(videos) != 4:
        raise RuntimeError("G1 独立视频必须包含候选 #0..3，不能回退四宫格主展示")
    result = {int(cid): views for cid, views in videos.items()}
    for cid, views in result.items():
        if not isinstance(views, dict) or set(views) != set(G1_SYNC_VIEWS) or not all(isinstance(v, str) and v for v in views.values()):
            raise RuntimeError(f"G1 候选 #{cid} 缺少正/侧/3⁄4 独立视频")
    if len({rel for views in result.values() for rel in views.values()}) != 12:
        raise RuntimeError("G1 候选和视角必须引用 12 段独立视频")
    return dict(sorted(result.items()))


def _find_ffmpeg_binary() -> str | None:
    """Finds an installed `ffmpeg` executable, or returns None if unavailable."""
    candidates = [
        os.environ.get("MONSTER_FFMPEG"),
        shutil.which("ffmpeg"),
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
    ]
    for cand in candidates:
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            return str(Path(cand))
    return None


def _validate_blend_file(blend_path: Path) -> None:
    """Verifies that `blend_path` is a real uncompressed Blender `.blend` file (not a stub or Zstd stream)."""
    if not blend_path.is_file():
        raise FileNotFoundError(f"G1 候选 Blender 文件不存在: {blend_path}")
    size = blend_path.stat().st_size
    if size < 1024:
        raise RuntimeError(f"G1 候选 Blender 文件过小 ({size} bytes)，拒绝占位文件: {blend_path}")
    with blend_path.open("rb") as f:
        header = f.read(32)
    if header.startswith(b"\x28\xb5\x2f\xfd") or header.startswith(b"\x1f\x8b"):
        raise RuntimeError(
            f"G1 候选 Blender 文件使用了压缩格式 ({blend_path.name})，工作台要求未压缩 .blend (compress=False)"
        )
    if not header.startswith(b"BLENDER"):
        raise RuntimeError(f"G1 候选文件不是合法未压缩 Blender 头: {blend_path}")
    if b"G1-CANDIDATE-" in header:
        raise RuntimeError(f"检测到伪造的占位 .blend 头，拒绝通过: {blend_path}")


def _validate_png_file(png_path: Path, expected_size: tuple[int, int] | None = None) -> tuple[int, int]:
    """Verifies that `png_path` is a real PNG image and optionally matches `expected_size` `(width, height)`."""
    if not png_path.is_file():
        raise FileNotFoundError(f"PNG 渲染图不存在: {png_path}")
    size = png_path.stat().st_size
    if size < 128:
        raise RuntimeError(f"PNG 渲染图过小 ({size} bytes): {png_path}")
    with png_path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise RuntimeError(f"非法的 PNG 文件头: {png_path}")
    width, height = struct.unpack(">II", header[16:24])
    if expected_size is not None and (width, height) != expected_size:
        raise RuntimeError(
            f"PNG 尺寸不匹配 ({png_path.name}): 实际 {width}x{height}，期望 {expected_size[0]}x{expected_size[1]}"
        )
    return int(width), int(height)


def _validate_mp4_file(mp4_path: Path) -> dict[str, Any]:
    """Parses ISO BMFF / MP4 top-level atoms and verifies `ftyp`, `moov`, and non-empty `mdat` payload."""
    if not mp4_path.is_file():
        raise FileNotFoundError(f"MP4 视频不存在: {mp4_path}")
    total_size = mp4_path.stat().st_size
    if total_size < 256:
        raise RuntimeError(
            f"MP4 视频文件过小 ({total_size} bytes <= 256 bytes)，疑似 24 字节 ftyp 占位文件: {mp4_path}"
        )

    boxes: list[str] = []
    mdat_payload_bytes = 0
    with mp4_path.open("rb") as f:
        offset = 0
        while offset + 8 <= total_size:
            f.seek(offset)
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            box_size, box_type_bytes = struct.unpack(">I4s", hdr)
            box_type = box_type_bytes.decode("latin1", errors="replace")
            header_len = 8
            if box_size == 1:
                ext = f.read(8)
                if len(ext) < 8:
                    break
                box_size = struct.unpack(">Q", ext)[0]
                header_len = 16
            elif box_size == 0:
                box_size = total_size - offset
            if box_size < header_len or offset + box_size > total_size:
                break
            boxes.append(box_type)
            if box_type == "mdat":
                mdat_payload_bytes += max(0, int(box_size - header_len))
            offset += box_size

    required_boxes = {"ftyp", "moov", "mdat"}
    missing = required_boxes.difference(boxes)
    if missing or mdat_payload_bytes <= 0:
        raise RuntimeError(
            f"MP4 视频结构不完整 ({mp4_path.name}): boxes={boxes}, missing={sorted(missing)}, "
            f"mdat_payload_bytes={mdat_payload_bytes}"
        )
    return {
        "path": str(mp4_path),
        "bytes": int(total_size),
        "boxes": boxes,
        "mdat_payload_bytes": int(mdat_payload_bytes),
    }


def _encode_sync_comparison_video(
    out_mp4: Path,
    frame_images: list[Path],
    fps: int = 24,
    candidate_view: str | None = None,
) -> Path:
    """Encodes a real MP4 video from `frame_images` via `ffmpeg` and validates MP4 box integrity.
    Never writes a 24-byte `ftyp` placeholder stub.
    """
    if not frame_images:
        raise RuntimeError(f"缺少同步对比序列帧，无法编码视频: {out_mp4.name}")
    for frame_path in frame_images:
        _validate_png_file(Path(frame_path))

    ffmpeg_bin = _find_ffmpeg_binary()
    if not ffmpeg_bin:
        raise RuntimeError("缺少 ffmpeg 运行时，无法编码真实 G1/G3 同步对比 MP4 视频")

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    # Keep identification inside the video when it is viewed outside the review page.
    if candidate_view is not None:
        from PIL import Image, ImageDraw, ImageFont

        with tempfile.TemporaryDirectory(prefix="monster_g1_labels_") as temp:
            labelled_frames = []
            for index, frame_path in enumerate(frame_images):
                with Image.open(frame_path) as source:
                    frame = source.convert("RGB")
                draw = ImageDraw.Draw(frame)
                font = ImageFont.load_default(size=20)
                for candidate_id in range(4):
                    x = (candidate_id % 2) * (frame.width // 2) + 12
                    y = (candidate_id // 2) * (frame.height // 2) + 12
                    label = f"Candidate {candidate_id} | {candidate_view} | Frame {index + 1:03d}/{len(frame_images):03d}"
                    bounds = draw.textbbox((x, y), label, font=font)
                    draw.rectangle((bounds[0]-6, bounds[1]-4, bounds[2]+6, bounds[3]+4), fill=(12, 18, 26))
                    draw.text((x, y), label, font=font, fill="white")
                target = Path(temp) / f"sync_frame_{index + 1:04d}.png"
                frame.save(target)
                labelled_frames.append(target)
            return _encode_sync_comparison_video(out_mp4, labelled_frames, fps=fps)
    pattern = str(Path(frame_images[0]).parent / "sync_frame_%04d.png")

    codec_attempts = [
        ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        ["-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    ]
    last_err: Exception | None = None
    for codec_args in codec_attempts:
        try:
            subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-framerate",
                    str(max(1, int(fps))),
                    "-i",
                    pattern,
                    *codec_args,
                    str(out_mp4),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
                timeout=60,
            )
            _validate_mp4_file(out_mp4)
            return out_mp4
        except Exception as exc:
            last_err = exc
            continue

    raise RuntimeError(f"ffmpeg 编码同步对比视频失败 ({out_mp4.name}): {last_err}")


# Alias kept for internal caller compatibility; strictly encodes real MP4 and never writes placeholders.
_encode_sync_comparison_video_or_placeholder = _encode_sync_comparison_video


def encode_g1_candidate_videos(sync_videos: dict[str, Path], out_dir: Path) -> dict[int, dict[str, str]]:
    """Separate each synchronized panel without changing its camera, timing or motion."""
    if set(sync_videos) != set(G1_SYNC_VIEWS):
        raise RuntimeError("G1 缺少正/侧/3⁄4 同步源视频，无法生成完整独立候选视频")
    ffmpeg = _find_ffmpeg_binary()
    if not ffmpeg:
        raise RuntimeError("缺少 ffmpeg，无法生成单候选视频")
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = {i: {} for i in range(4)}
    for view, source in sync_videos.items():
        if view not in G1_SYNC_VIEWS:
            raise ValueError(f"Unknown G1 view: {view}")
        _validate_mp4_file(source)
        for cid in range(4):
            name = f"candidate_{cid}_{view}.mp4"
            subprocess.run([
                ffmpeg, "-y", "-i", str(source), "-vf",
                f"crop=iw/2:ih/2:{'iw/2' if cid % 2 else '0'}:{'ih/2' if cid >= 2 else '0'}",
                "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(out_dir / name),
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=60)
            _validate_mp4_file(out_dir / name)
            videos[cid][view] = name
    return _validate_candidate_video_map(videos)


def validate_g1_orientation_evidence(report: dict[str, Any]) -> None:
    """Historical orientation receipt validator (no-op in simplified pipeline)."""
    return


def validate_g1_evidence(report: dict[str, Any]) -> None:
    """Validates that `g1_report.json` was produced by real Blender + FFmpeg and all artifacts match SHA-256."""
    if report.get("schema_version") != 2 or report.get("backend") != "blender":
        raise RuntimeError("G1 报告缺少真实 Blender 渲染与变形核验证据；必须重新运行 g1-pack")
    if not report.get("passed"):
        raise RuntimeError("G1 报告未通过真实性校验；必须重新运行 g1-pack")

    evidence = report.get("evidence", {})
    contract = report.get("presentation_contract")
    if contract is not None and contract != G1_REVIEW_CONTRACT:
        raise RuntimeError(f"不支持的 G1 候选视频审阅契约: {contract}")
    # Untagged historical reports remain readable. New reports cannot discard the
    # individual clips while retaining a valid presentation contract.
    if contract or "candidate_videos" in report or "candidate_videos" in evidence:
        videos = _validate_candidate_video_map(report.get("candidate_videos", {}))
        expected = {f"candidate_{cid}_{view}" for cid in videos for view in G1_SYNC_VIEWS}
        if set(evidence.get("candidate_videos", {})) != expected:
            raise RuntimeError("G1 独立候选视频证据必须完整覆盖 4 候选 × 3 视角")
        review_dir = Path(report["review_html"]).parent
        for cid, views in videos.items():
            for view, rel in views.items():
                item = evidence["candidate_videos"][f"candidate_{cid}_{view}"]
                if Path(item.get("path", "")).resolve() != (review_dir / rel).resolve():
                    raise RuntimeError("G1 独立候选视频引用与哈希证据不一致")
                if contract and (item.get("source_sync_sha256") != evidence.get("sync_videos", {}).get(view, {}).get("sha256")
                                 or item.get("crop_xywh") != [(cid % 2) * 640, (cid // 2) * 360, 640, 360]):
                    raise RuntimeError("G1 独立候选视频缺少一致的源视频哈希或裁切映射")
        if contract:
            timing = report.get("video_timing", {})
            counts = timing.get("frames_by_view", {})
            if (not isinstance(timing.get("fps"), (int, float)) or timing["fps"] <= 0
                    or set(counts) != set(G1_SYNC_VIEWS) or len(set(counts.values())) != 1
                    or not all(isinstance(n, int) and n >= 8 for n in counts.values())
                    or timing.get("transform") != "quadrant_crop_only_no_retiming"):
                raise RuntimeError("G1 独立候选视频缺少一致的帧率与同步帧数记录")
    required_groups = {
        "candidate_inputs",
        "candidate_blends",
        "candidate_metas",
        "octahedral_views",
        "sync_videos",
        "follow_through_issues",
        "review_html",
    }
    if not required_groups.issubset(evidence):
        raise RuntimeError("G1 证据清单不完整；必须重新运行 g1-pack")

    groups = ["candidate_inputs", "candidate_blends", "candidate_metas", "octahedral_views", "sync_videos"]
    if "candidate_videos" in evidence:
        groups.append("candidate_videos")
    for group_key in groups:
        items = evidence.get(group_key, {})
        if not isinstance(items, dict) or not items:
            raise RuntimeError(f"G1 证据组为空: {group_key}")
        for name, item in items.items():
            p = Path(item.get("path", ""))
            if not item.get("sha256") or sha256_file(p) != item["sha256"]:
                raise RuntimeError(f"G1 证据已改变或缺失 ({group_key}:{name})；必须重新运行 g1-pack")
            if group_key == "candidate_blends":
                _validate_blend_file(p)
            elif group_key == "octahedral_views":
                _validate_png_file(p, expected_size=(640, 640))
            elif group_key in ("sync_videos", "candidate_videos"):
                _validate_mp4_file(p)

    for single_key in ("follow_through_issues", "review_html"):
        item = evidence.get(single_key, {})
        p = Path(item.get("path", ""))
        if not item.get("sha256") or sha256_file(p) != item["sha256"]:
            raise RuntimeError(f"G1 证据已改变或缺失 ({single_key})；必须重新运行 g1-pack")
    if contract and f'content="{G1_REVIEW_CONTRACT}"' not in Path(report["review_html"]).read_text(encoding="utf-8"):
        raise RuntimeError("G1 审阅页面缺少独立候选视频契约标识")


def build_g1_review_html(
    monster_id: str,
    archetype: str,
    candidates_summary: list[dict[str, Any]],
    octahedral_rel_map: dict[int, dict[str, str]],
    sync_video_rels: dict[str, str],
    follow_through_issues: list[dict[str, Any]],
    out_html_path: Path,
    candidate_video_rels: dict[int, dict[str, str]] | None = None,
) -> Path:
    """Build individual-video-first review, upgrading old callers from real sync sources."""
    if candidate_video_rels is None:
        candidate_video_rels = encode_g1_candidate_videos(
            {view: out_html_path.parent / rel for view, rel in sync_video_rels.items()},
            out_html_path.parent,
        )
    candidate_video_rels = _validate_candidate_video_map(candidate_video_rels)
    for views in candidate_video_rels.values():
        for rel in views.values():
            _validate_mp4_file(out_html_path.parent / rel)
    spec = ARCHETYPE_SPECS[archetype]

    all_failed = bool(follow_through_issues) and all(x.get('status') == 'FAIL' for x in follow_through_issues)
    status_banner = ('G1 自动检查未通过 · 失败证据已保留，不能进入 G2' if all_failed
                     else 'G1 待人工审阅 · 自动检查不代替用户选择和批准')
    individual_cards = []
    view_labels = {'side': '侧面', 'front': '正面', 'three_quarter': '3/4 视角'}
    for cid, videos in candidate_video_rels.items():
        default = videos.get('side') or next(iter(videos.values()))
        options = ''.join(f'<option value="{html.escape(videos[v], quote=True)}"'
                          f'{" selected" if videos[v] == default else ""}>{view_labels[v]}</option>'
                          for v in ('side', 'front', 'three_quarter') if v in videos)
        issue = next((x for x in follow_through_issues if str(x['candidate_id']) == str(cid)), {})
        status = html.escape(str(issue.get('status', '未记录')))
        stretch = html.escape(str(issue.get('metrics', {}).get('max_edge_stretch_ratio', '未记录')))
        individual_cards.append(f'''<article class="candidate-video-card">
          <div class="card-header"><h3>候选 #{cid}</h3><span>自动检查：{status}</span></div>
          <video id="candidate-video-{cid}" controls loop muted playsinline preload="metadata"
            src="{html.escape(default, quote=True)}"></video>
          <div class="candidate-video-controls"><label>查看视角 <select aria-label="候选 #{cid} 视角"
            data-video-target="candidate-video-{cid}">{options}</select></label>
            <a data-video-link="candidate-video-{cid}" href="{html.escape(default, quote=True)}" target="_blank">单独打开视频</a></div>
          <p class="meta">最大边拉伸：{stretch} 倍 · 可独立暂停、拖动进度或全屏放大。</p>
        </article>''')
    cand_cards = []
    for c in candidates_summary:
        cid = int(c["candidate_id"])
        val = c["bone_naming_validation"]
        audit_rep = c.get("audit_report", {})
        views_html = "".join(
            f'<div class="view-cell"><div class="view-label">{v.upper()}</div>'
            f'<img src="{html.escape(octahedral_rel_map[cid][v])}" alt="cand {cid} {v}"/></div>'
            for v in G1_VIEWS
        )
        rename_rows = "".join(
            f"<tr><td><code>{html.escape(old_n)}</code></td><td><code>{html.escape(new_n)}</code></td>"
            f"<td><code>{html.escape(str(c['bones'][new_n].get('kind', '')).upper())}</code></td>"
            f"<td><code>{html.escape(str(c['bones'][new_n].get('sub_role', '')))}</code></td></tr>"
            for old_n, new_n in list(c.get("canonical_rename_map", {}).items())[:16]
            if new_n in c.get("bones", {})
        )
        cand_cards.append(f"""
        <section class="card">
          <div class="card-header">
            <h2>候选 #{cid} (<code>candidate_{cid}</code>)</h2>
            <span class="badge pass">规范命名 100% 合规 ({val['canonical_count']}/{val['total_bones']} 骨 · 零匿名骨)</span>
          </div>
          <p class="meta">
            几何评分：<code>{html.escape(str(audit_rep.get('score', 'N/A')))}</code> |
            穿出率：<code>{html.escape(str(audit_rep.get('outside_ratio', 'N/A')))}</code> |
            四象限覆盖：<code>{html.escape(str(audit_rep.get('quadrants_covered', 'N/A')))}</code> |
            已恢复装饰：{len(c.get('restored_decorations', []))} 项 |
            Blend: <code>{html.escape(Path(c.get('blend_file', '')).name)}</code>
          </p>
          <p class="meta">实际管线：<code>{html.escape(str(c.get('pipeline_feature_flags', {}).get('pipeline_mode', '未记录')))}</code> |
            骨架：<code>{html.escape(str(c.get('bone_taxonomy_schema', '未记录')))}</code> |
            Reader 契约：<code>{html.escape(str(c.get('reader_contract', '未记录')))}</code><br>
            G1 驱动：<code>{html.escape(str(c.get('g1_driver', {}).get('contract', '未记录')))}</code> |
            基准：<code>{html.escape(str(c.get('g1_driver', {}).get('reference_pose', '未记录')))}</code><br>
            世界运动根：<code>{html.escape(', '.join(c.get('g1_driver', {}).get('motion_roots', [])))}</code> |
            拟合骨 {len(c.get('g1_driver', {}).get('fitted_bones', []))}，冻结骨 {len(c.get('g1_driver', {}).get('frozen_bones', []))}。
            V3 使用实际 hinge.axis / axis_roles 与保存 p0；固定骨及未拟合骨保持保存局部基矩阵。</p>
          <div class="views-grid">{views_html}</div>
          <details style="margin-top:12px;">
            <summary>查看骨骼规范命名映射表 (<code>{html.escape(str(c.get('bone_taxonomy_schema', '未记录')))}</code>：Root / Spine1 / Arm.L / Shin.R)</summary>
            <table>
              <thead><tr><th>原骨骼名</th><th>规范骨骼名</th><th>KIND</th><th>Sub-Role</th></tr></thead>
              <tbody>{rename_rows}</tbody>
            </table>
          </details>
        </section>
        """)

    video_tabs = "".join(
        f'<div class="video-box"><h3>四候选同动作同步对照 · {v_name.upper()} 视角</h3>'
        f'<video controls loop muted playsinline style="width:100%;border-radius:8px;background:#000;">'
        f'<source src="{html.escape(v_rel)}" type="video/mp4"></video></div>'
        for v_name, v_rel in sync_video_rels.items()
    )

    issues_rows = "".join(
        f"<tr><td><code>candidate_{html.escape(str(item['candidate_id']))}</code></td>"
        f"<td>{html.escape(item['check'])}</td>"
        f"<td>{html.escape(item['status'])}</td>"
        f"<td>{html.escape(item['detail'])}</td></tr>"
        for item in follow_through_issues
    )

    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="monster-review-contract" content="{G1_REVIEW_CONTRACT}">
  <title>G1 骨架审计包 — {html.escape(monster_id)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0d1117; color: #e6edf3; margin: 0; padding: 24px; }}
    .container {{ max-width: 1360px; margin: 0 auto; }}
    .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
    .card-header {{ display: flex; justify-content: space-between; align-items: center; }}
    .badge {{ padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
    .pass {{ background: rgba(63,185,80,0.18); color: #3fb950; border: 1px solid rgba(63,185,80,0.4); }}
    .views-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 12px; }}
    .view-cell {{ background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 8px; text-align: center; }}
    .view-cell img {{ width: 100%; border-radius: 6px; }}
    .view-label {{ font-size: 12px; color: #8b949e; margin-bottom: 6px; font-weight: 600; }}
    .videos-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
    .review-status {{ color: #f0b778; padding: 12px 16px; border: 1px solid #6e5329; border-radius: 8px; }}
    .candidate-videos {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; }}
    .candidate-video-card {{ min-width: 0; background: #0d1117; border: 1px solid #30363d; padding: 16px; border-radius: 10px; }}
    .candidate-video-card h3 {{ margin: 0 0 12px; }}
    .candidate-video-card video {{ width: 100%; aspect-ratio: 16/9; border-radius: 6px; background: #000; }}
    .candidate-video-controls {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 12px; }}
    select {{ background: #21262d; color: #e6edf3; border: 1px solid #484f58; padding: 7px; border-radius: 5px; }}
    a {{ color: #79c0ff; }}
    @media (max-width: 900px) {{ .candidate-videos, .videos-grid {{ grid-template-columns: 1fr; }} .views-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #21262d; padding: 8px 10px; text-align: left; }}
    code {{ background: #21262d; padding: 2px 6px; border-radius: 4px; }}
    .meta {{ color: #8b949e; font-size: 13px; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>G1 · 候选视频审阅</h1>
    <p class="review-status">{status_banner}</p>
    <section class="card">
      <h2>1. 每个候选独立查看</h2>
      <p>每张卡片只播放一个候选。先看侧面，也可切换正面或 3/4 视角；使用视频全屏按钮放大。</p>
      <div class="candidate-videos">{"".join(individual_cards)}</div>
      <details><summary>补充：四候选同步对照</summary><div class="videos-grid">{video_tabs}</div></details>
    </section>
    <details class="card"><summary>检查范围、动作模板与同步条件</summary>
      <p class="meta">角色：<code>{html.escape(monster_id)}</code></p>
      <p>本次检查 G1 模板动作下的真实网格变形、起始姿态骨骼图与命名；未执行 G2 采样、Reader 训练或视频求解。报告 schema_version=2 是报告格式，DINOv2 是视觉主干名称。</p>
      <p>四套使用相同时间、相位、相机与灯光。补充四宫格排列：左上 #0，右上 #1，左下 #2，右下 #3。</p>
      <p class="meta">测试动作：{html.escape(spec['title'])} · {html.escape(spec['focus_checks'])}</p>
    </details>
    <details class="card"><summary>随动与网格拉伸检查清单</summary>
      <table>
        <thead><tr><th>候选 ID</th><th>检查项</th><th>状态</th><th>详情</th></tr></thead>
        <tbody>{issues_rows}</tbody>
      </table>
    </details>
    <details class="card"><summary>骨骼静态图、命名映射与驱动详情</summary>{"".join(cand_cards)}</details>
  </div>
  <script>
    document.querySelectorAll('[data-video-target]').forEach(select => {{
      select.addEventListener('change', () => {{
        const video = document.getElementById(select.dataset.videoTarget);
        const time = video.currentTime;
        const playing = !video.paused;
        video.onloadedmetadata = () => {{
          video.currentTime = Math.min(time, Math.max(0, video.duration - 0.001));
          if (playing) video.play().catch(() => {{}});
        }};
        video.src = select.value;
        document.querySelector('[data-video-link="' + video.id + '"]').href = select.value;
      }});
    }});
  </script>
</body>
</html>
"""
    out_html_path.parent.mkdir(parents=True, exist_ok=True)
    out_html_path.write_text(doc, encoding="utf-8")
    return out_html_path


def build_g1_review_pack(
    monster_id: str,
    *,
    archetype: str = "quadruped",
    candidates_dir: Path | str | None = None,
    project_root: Path | str | None = None,
    drive_root: Path | str | None = None,
    duration_sec: float = 6.0,
    fps: int = 24,
    skinning_mode: str = "bone_heat",
) -> dict[str, Any]:
    """End-to-end deterministic G1 builder: requires 4 real candidate GLBs, runs headless Blender
    (`studio_bpy_worker.py --mode prepare_g1_candidates`) to restore decorations, canonicalize bones,
    render 16 3D Octahedral bone views, evaluate real mesh deformation across the archetype test motion,
    and encode 12 individual candidate MP4s plus 3 supplementary synchronized 2x2 videos.
    """
    if archetype not in ARCHETYPE_SPECS:
        raise ValueError(f"Invalid archetype {archetype!r}; choose from {tuple(ARCHETYPE_SPECS)}")

    proj = Path(project_root or os.environ.get("MONSTER_PROJECT_ROOT") or Path.cwd()).resolve()
    dirs = resolve_scratch_dirs(proj, monster_id)
    work_dir = dirs["work_dir"]
    rig_id = 'rig_' + uuid.uuid4().hex
    run_dir = work_dir / 'rig' / rig_id
    stage_root = run_dir / 'g1_diagnostics'
    stage_root.mkdir(parents=True, exist_ok=False)
    checks_dir = dirs["checks_dir"] / "g1_review" / rig_id
    checks_dir.mkdir(parents=True, exist_ok=True)
    report_path = checks_dir / "g1_report.json"
    latest_report_path = dirs["checks_dir"] / "g1_review" / "g1_report.json"

    # Invalidate previous G1 approval/waiting state before running anything that can fail.
    ledger = load_ledger(monster_id, project_root=proj, drive_root=drive_root)
    derive_invalidations(ledger, change_type="skeleton_topology_defect")
    ledger["gates"]["G1"]["status"] = "not_ready"
    ledger["gates"]["G1"]["user_decision"] = "pending"
    save_ledger(ledger, project_root=proj, drive_root=drive_root)

    report: dict[str, Any] = {
        "schema_version": 2,
        "monster_id": monster_id,
        "archetype": archetype,
        "checked_at": now_taipei_iso(),
        "passed": False,
        "status": "running",
        "rig_id": rig_id,
        "diagnostic_directory": str(stage_root),
        "evidence": {},
    }
    _write_report(report_path, report)
    _write_report(latest_report_path, report)

    try:
        cand_src_dir = Path(candidates_dir).resolve() if candidates_dir else (work_dir / "candidates")
        cand_glbs = [cand_src_dir / f"candidate_{cid}.glb" for cid in range(4)]
        missing_cands = [str(p) for p in cand_glbs if not p.is_file()]
        if missing_cands:
            raise FileNotFoundError(
                f"G1 缺少 4 个真实候选绑定 GLB (candidate_0..3.glb): {missing_cands}"
            )

        decor_manifest_path = work_dir / "decoration_manifest.json"
        with nullcontext(str(stage_root)) as temp_dir:
            stage_dir = Path(temp_dir)
            worker_args = [
                "--mode",
                "prepare_g1_candidates",
                "--candidates",
                *[str(p) for p in cand_glbs],
                "--out_dir",
                str(stage_dir),
                "--archetype",
                archetype,
                "--skinning_mode",
                skinning_mode,
            ]
            if decor_manifest_path.is_file():
                worker_args.extend(["--decor_manifest", str(decor_manifest_path)])

            command = _worker_command(worker_args)
            if not command:
                raise RuntimeError(
                    "缺少 bpy/Blender 运行时，无法生成真实 G1 四候选骨架渲染、装饰恢复与同步对比视频"
                )

            subprocess.run(command, check=True, timeout=1800)

            summary_path = stage_dir / "bpy_g1_summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(f"Blender G1 worker 未生成 bpy_g1_summary.json: {summary_path}")
            bpy_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if bpy_summary.get("schema_version") != 2 or bpy_summary.get("render_engine") != "BLENDER_WORKBENCH":
                raise RuntimeError("Blender G1 worker 摘要缺少 schema_version=2 或 BLENDER_WORKBENCH 证据")

            eff_fps = int(bpy_summary.get("fps") or fps or 24)
            raw_ft_issues = bpy_summary.get("follow_through_issues", {})

            g1_cand_dir = run_dir / "candidates"
            g1_cand_dir.mkdir(parents=True, exist_ok=True)

            candidates_summary: list[dict[str, Any]] = []
            octahedral_rel_map: dict[int, dict[str, str]] = {}
            follow_through_issues: list[dict[str, Any]] = []
            evidence_inputs: dict[str, dict[str, str]] = {}
            evidence_blends: dict[str, dict[str, str]] = {}
            evidence_metas: dict[str, dict[str, str]] = {}
            evidence_views: dict[str, dict[str, str]] = {}
            evidence_videos: dict[str, dict[str, str]] = {}

            for cid in range(4):
                cand_key = f"candidate_{cid}"
                src_glb = cand_glbs[cid]
                evidence_inputs[cand_key] = {"path": str(src_glb), "sha256": sha256_file(src_glb)}

                staged_meta = stage_dir / f"candidate_{cid}_rig_meta.json"
                if not staged_meta.is_file():
                    staged_meta = stage_dir / cand_key / "studio_rig_meta.json"
                if not staged_meta.is_file():
                    raise FileNotFoundError(f"缺少候选骨骼元数据: {staged_meta}")

                staged_blend = stage_dir / f"candidate_{cid}_full.blend"
                _validate_blend_file(staged_blend)

                rig_meta = json.loads(staged_meta.read_text(encoding="utf-8"))
                val = validate_canonical_rig_bone_names(rig_meta.get("bones", {}))
                if not val["valid"]:
                    raise RuntimeError(
                        f"[SOP §3.1 硬门禁阻断] {cand_key} 含有未规范命名的骨骼: {val['non_canonical_bones']}"
                    )
                rig_meta["candidate_id"] = str(cid)
                rig_meta["bone_naming_validation"] = val

                dest_blend = g1_cand_dir / f"candidate_{cid}_full.blend"
                dest_meta = g1_cand_dir / f"candidate_{cid}_rig_meta.json"
                dest_sub_dir = g1_cand_dir / cand_key
                dest_sub_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(staged_blend, dest_blend)
                for auxiliary in ("shell.npz", "skeleton_postprocess.json"):
                    source_aux = stage_dir / cand_key / auxiliary
                    if source_aux.is_file():
                        shutil.copy2(source_aux, dest_sub_dir / auxiliary)
                if (dest_sub_dir / "shell.npz").is_file():
                    rig_meta["shell_npz"] = str(dest_sub_dir / "shell.npz")
                    sp = rig_meta.get("skeleton_postprocess")
                    if isinstance(sp, dict) and isinstance(sp.get("watertight_shell"), dict):
                        sp["watertight_shell"]["path"] = str(dest_sub_dir / "shell.npz")
                        if (dest_sub_dir / "skeleton_postprocess.json").is_file():
                            (dest_sub_dir / "skeleton_postprocess.json").write_text(
                                json.dumps(sp, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8",
                            )

                staged_full_glb = stage_dir / f"candidate_{cid}_full.glb"
                if staged_full_glb.is_file():
                    shutil.copy2(staged_full_glb, g1_cand_dir / f"candidate_{cid}_full.glb")

                rig_meta["blend_file"] = str(dest_blend)
                rig_meta["meta_file"] = str(dest_meta)
                meta_text = json.dumps(rig_meta, indent=2, ensure_ascii=False) + "\n"
                dest_meta.write_text(meta_text, encoding="utf-8")
                (dest_sub_dir / "studio_rig_meta.json").write_text(meta_text, encoding="utf-8")

                evidence_blends[cand_key] = {"path": str(dest_blend), "sha256": sha256_file(dest_blend)}
                evidence_metas[cand_key] = {"path": str(dest_meta), "sha256": sha256_file(dest_meta)}
                candidates_summary.append(rig_meta)

                octahedral_rel_map[cid] = {}
                for view in G1_VIEWS:
                    img_name = f"cand_{cid}_{view}_octahedral.png"
                    staged_img = stage_dir / img_name
                    _validate_png_file(staged_img, expected_size=(640, 640))
                    dest_img = checks_dir / img_name
                    shutil.copy2(staged_img, dest_img)
                    octahedral_rel_map[cid][view] = img_name
                    evidence_views[f"{cand_key}_{view}"] = {"path": str(dest_img), "sha256": sha256_file(dest_img)}

                ft_info = raw_ft_issues.get(cand_key)
                if not isinstance(ft_info, dict) or "max_vertex_displacement_norm" not in ft_info:
                    raise RuntimeError(f"缺少 {cand_key} 的真实网格变形与随动核验指标")

                ft_status = str(ft_info.get("status", "PASS"))
                detail_str = (
                    f"规范骨骼 {val['canonical_count']}/{val['total_bones']} 根；"
                    f"最大顶点位移比={ft_info.get('max_vertex_displacement_norm')}，"
                    f"最大边拉伸比={ft_info.get('max_edge_stretch_ratio')}，"
                    f"异常拉伸边占比={ft_info.get('max_bad_edge_ratio')}，"
                    f"未赋权顶点={ft_info.get('unweighted_vertices')}/{ft_info.get('total_vertices')}，"
                    f"几何评分={ft_info.get('geometric_score')} (穿出率={ft_info.get('outside_ratio')})"
                )
                if ft_info.get("issues"):
                    detail_str += f" | 发现项: {', '.join(str(x) for x in ft_info['issues'])}"

                follow_through_issues.append(
                    {
                        "candidate_id": cid,
                        "check": f"{ARCHETYPE_SPECS[archetype]['title']} 真实网格变形与装饰随动核验",
                        "status": ft_status,
                        "detail": detail_str,
                        "metrics": ft_info,
                    }
                )

            all_failed = all(item["status"] == "FAIL" for item in follow_through_issues)

            sync_video_rels: dict[str, str] = {}
            video_box_reports: dict[str, dict[str, Any]] = {}
            for v_name in G1_SYNC_VIEWS:
                frame_dir = stage_dir / "sync_frames" / v_name
                frame_images = sorted(frame_dir.glob("sync_frame_*.png"))
                if len(frame_images) < 8:
                    raise RuntimeError(
                        f"G1 同步对比视角 {v_name} 渲染帧数不足 ({len(frame_images)} < 8)"
                    )
                for f_img in frame_images:
                    _validate_png_file(f_img, expected_size=(1280, 720))

                mp4_name = f"g1_candidates_sync_{v_name}.mp4"
                staged_mp4 = stage_dir / mp4_name
                _encode_sync_comparison_video(staged_mp4, frame_images, fps=eff_fps, candidate_view=v_name)
                box_rep = _validate_mp4_file(staged_mp4)

                dest_mp4 = checks_dir / mp4_name
                shutil.copy2(staged_mp4, dest_mp4)
                sync_video_rels[v_name] = mp4_name
                video_box_reports[v_name] = box_rep
                evidence_videos[v_name] = {"path": str(dest_mp4), "sha256": sha256_file(dest_mp4)}

            candidate_video_rels = encode_g1_candidate_videos(
                {v: checks_dir / name for v, name in sync_video_rels.items()}, checks_dir
            )
            candidate_video_evidence = {
                f"candidate_{cid}_{view}": {"path": str(checks_dir / name),
                                          "sha256": sha256_file(checks_dir / name),
                                          "source_sync_sha256": evidence_videos[view]["sha256"],
                                          "crop_xywh": [(cid % 2) * 640, (cid // 2) * 360, 640, 360]}
                for cid, views in candidate_video_rels.items() for view, name in views.items()
            }

            issues_path = checks_dir / "follow_through_issues.json"
            issues_path.write_text(
                json.dumps(follow_through_issues, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            html_path = checks_dir / "g1_review.html"
            build_g1_review_html(
                monster_id=monster_id,
                archetype=archetype,
                candidates_summary=candidates_summary,
                octahedral_rel_map=octahedral_rel_map,
                sync_video_rels=sync_video_rels,
                follow_through_issues=follow_through_issues,
                out_html_path=html_path,
                candidate_video_rels=candidate_video_rels,
            )

            report.update(
                {
                    "backend": "blender",
                    "render_engine": bpy_summary.get("render_engine", "BLENDER_WORKBENCH"),
                    "passed": not all_failed,
                    "status": "failed" if all_failed else "awaiting_rig_review",
                    "candidate_checks": follow_through_issues,
                    "error": "G1 四个候选全部未通过真实网格变形核验；失败图、视频、指标已保留" if all_failed else None,
                    "review_html": str(html_path),
                    "follow_through_issues_json": str(issues_path),
                    "octahedral_views_rendered": 16,
                    "sync_videos": {k: str(checks_dir / v) for k, v in sync_video_rels.items()},
                    "candidate_videos": candidate_video_rels,
                    "presentation_contract": G1_REVIEW_CONTRACT,
                    "video_timing": {"fps": eff_fps, "frames_by_view": {
                        view: len(list((stage_dir / "sync_frames" / view).glob("sync_frame_*.png")))
                        for view in G1_SYNC_VIEWS}, "transform": "quadrant_crop_only_no_retiming"},
                    "video_validation": video_box_reports,
                    "evidence": {
                        "candidate_inputs": evidence_inputs,
                        "candidate_blends": evidence_blends,
                        "candidate_metas": evidence_metas,
                        "octahedral_views": evidence_views,
                        "sync_videos": evidence_videos,
                        "candidate_videos": candidate_video_evidence,
                        "follow_through_issues": {
                            "path": str(issues_path),
                            "sha256": sha256_file(issues_path),
                        },
                        "review_html": {
                            "path": str(html_path),
                            "sha256": sha256_file(html_path),
                        },
                    },
                }
            )
            _write_report(report_path, report)
            _write_report(latest_report_path, report)
            _write_report(run_dir / 'rig_manifest.json', {
                'schema': 'monster-rig-batch-v1', 'character_id': monster_id,
                'rig_id': rig_id, 'status': 'failed' if all_failed else 'completed',
                'g1_report': str(report_path), 'review_html': str(html_path),
                'candidate_inputs': evidence_inputs, 'created_at': report['checked_at'],
            })

    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        _write_report(report_path, report)
        _write_report(latest_report_path, report)
        raise

    if all_failed:
        ledger = load_ledger(monster_id, project_root=proj, drive_root=drive_root)
        ledger['gates']['G1'].update(status='not_ready', user_decision='pending',
            failure_report=str(report_path), diagnostic_review=str(html_path))
        save_ledger(ledger, project_root=proj, drive_root=drive_root)
        raise RuntimeError(report['error'])

    # Register awaiting_rig_review in SOP ledger only after all real Blender + FFmpeg evidence is verified
    ledger = load_ledger(monster_id, project_root=proj, drive_root=drive_root)
    mark_awaiting_review(
        ledger,
        "G1",
        review_pack_path=str(html_path),
        materials={
            "archetype": archetype,
            "g1_report_json": str(report_path),
            "presentation_contract": G1_REVIEW_CONTRACT,
            "rig_meta_path": candidates_summary[0]["meta_file"],
            "candidate_metas": [c["meta_file"] for c in candidates_summary],
            "candidate_blends": [c["blend_file"] for c in candidates_summary],
            "octahedral_views_count": 16,
            "sync_videos": {k: str(checks_dir / v) for k, v in sync_video_rels.items()},
            "candidate_videos": {str(cid): {v: str(checks_dir / name) for v, name in views.items()}
                                 for cid, views in candidate_video_rels.items()},
            "follow_through_issues_json": str(issues_path),
        },
    )
    save_ledger(ledger, project_root=proj, drive_root=drive_root)

    return {
        "monster_id": monster_id,
        "archetype": archetype,
        "status": "awaiting_rig_review",
        "g1_report_json": str(report_path),
        "presentation_contract": G1_REVIEW_CONTRACT,
        "review_html": str(html_path),
        "octahedral_views_rendered": 16,
        "sync_videos": {k: str(checks_dir / v) for k, v in sync_video_rels.items()},
        "candidate_videos": {str(cid): {v: str(checks_dir / name) for v, name in views.items()}
                             for cid, views in candidate_video_rels.items()},
        "candidates": [
            {
                "candidate_id": c["candidate_id"],
                "blend_file": c["blend_file"],
                "meta_file": c["meta_file"],
                "bone_naming_validation": c["bone_naming_validation"],
            }
            for c in candidates_summary
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monster SOP Stage 2 G1 Review Pack Builder")
    parser.add_argument("--monster-id", required=True)
    parser.add_argument("--archetype", default="quadruped", choices=tuple(ARCHETYPE_SPECS))
    parser.add_argument("--candidates-dir", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--drive-root", default=None)
    parser.add_argument("--duration-sec", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--skinning_mode", choices=["bone_heat", "skintokens"], default="bone_heat")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    res = build_g1_review_pack(
        args.monster_id,
        archetype=args.archetype,
        candidates_dir=args.candidates_dir,
        project_root=args.project_root,
        drive_root=args.drive_root,
        duration_sec=args.duration_sec,
        fps=args.fps,
        skinning_mode=args.skinning_mode,
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
