"""Production orchestration for the v3 Reader. Legacy checkpoints keep their existing path."""

from dataclasses import replace
import os
from pathlib import Path
import numpy as np
from PIL import Image
from studio_camera import cameras_from_calibration, triangulate
from reprojection_solver import RigModel, fit_clip_2d_reprojection
from reading_pipeline import (
    preprocess_readings,
    clean_motion,
    despike_reading_backed,
    recover_root_translation_from_steps,
    simulate_damped_spring_chains,
    at_limit,
    learn_frame0_shadow_table,
    strip_shadow,
    align_frame0_camera_chamfer,
    detect_video_cadence_optical_flow,
)
from pose_features import (
    clean_motion_enabled,
    no_cut_tracks_enabled,
    panel_drift_enabled,
    pipeline_feature_flags,
    reader_crops_enabled,
    reader_dense_enabled,
    surface_contact_enabled,
    twoway_jump_enabled,
)
from software_rasterizer import render_fast_review_frames, render_mesh_software


def geometric_consistency(cameras, uv, confidence, height):
    errors = []
    v_pairs = []
    for t in range(len(uv)):
        X = triangulate(cameras, uv[t], confidence[t])
        good = np.isfinite(X).all(-1) & ((confidence[t] > 0.3).sum(0) >= 2)
        if len(cameras) >= 2:
            both = (
                (confidence[t, 0] > 0.3)
                & (confidence[t, 1] > 0.3)
                & np.isfinite(uv[t, 0, :, 1])
                & np.isfinite(uv[t, 1, :, 1])
            )
            if both.any():
                v_pairs.append(
                    np.column_stack([uv[t, 0, both, 1], uv[t, 1, both, 1]])
                )
        for v, c in enumerate(cameras):
            valid = good & (confidence[t, v] > 0.3)
            if valid.any():
                errors.extend(
                    np.linalg.norm(
                        c.project(X[valid]) - uv[t, v, valid], axis=-1
                    ).tolist()
                )
    mean = float(np.mean(errors)) if errors else None
    v_corr = 0.0
    if v_pairs:
        vp = np.vstack(v_pairs)
        if len(vp) >= 2 and np.std(vp[:, 0]) > 1e-6 and np.std(vp[:, 1]) > 1e-6:
            c_val = float(np.corrcoef(vp[:, 0], vp[:, 1])[0, 1])
            v_corr = c_val if np.isfinite(c_val) else 0.0
        elif len(vp) >= 1 and mean is not None and mean < 18 * cameras[0].height / 768:
            v_corr = 1.0
    return {
        "gate_passed": bool(mean is not None and mean < 18 * cameras[0].height / 768),
        "mean_epipolar_diff_px": mean,
        "vertical_trajectory_corr": round(float(v_corr), 4),
        "normalized_epipolar_error": None if mean is None else mean / max(height, 1),
        "measured_observations": len(errors),
        "gate_method": "calibrated_multiview_reprojection",
    }


def multiview_dense_centers(rig, cameras, uv, confidence, skin):
    """Use coarse multi-view poses for the same posed-mesh mean used by dense labels."""
    from pose_features import lbs_deform_vertices_numpy

    previous = rig.start_params()
    centers = []
    for t in range(len(uv)):
        X = triangulate(cameras, uv[t], confidence[t])
        previous = rig.read_start(X, previous)
        _, heads, rotations = rig.fk(previous)
        vertices = lbs_deform_vertices_numpy(
            np.asarray(skin["rest_verts"]),
            np.asarray(skin["skin_indices"]),
            np.asarray(skin["skin_weights"]),
            np.asarray(skin["rest_heads"]),
            np.asarray(skin["rest_R"]),
            np.array([heads[n] for n in rig.names]),
            np.array([rotations[n] for n in rig.names]),
        )
        centers.append(vertices.mean(0))
    return np.asarray(centers)


def observation_view_permutation(calibration, cameras):
    """Primary view shows left/right separation; remaining panels are side observations.

    Neural view IDs and image layout stay in their trained order. Only the geometric
    reading pipeline is reordered, so three-view and side/top layouts use the same
    confidence/disambiguation semantics as front/side.
    """
    names = list(calibration["view_order"])
    if "front" in names:
        primary = names.index("front")
    elif "top" in names:
        primary = names.index("top")
    else:
        primary = int(np.argmax([np.linalg.norm(c.R[:2, 0]) for c in cameras]))
    return [primary] + [i for i in range(len(names)) if i != primary]


def infer_v3_frames(
    checkpoint,
    frame_paths,
    custom_backbone=None,
    *,
    crops_checkpoint=None,
    dense_checkpoint=None,
):
    import torch
    from dinov2_pose_trainer import (
        MonsterDINOv2PoseProbe,
        resolve_checkpoint_backbone_name,
    )
    from pose_finetuning import load_probe_weights
    from reader_v3 import (
        infer_reader_sequence,
        resolve_temporal_context_contract,
        scaled_window_side,
    )

    if not frame_paths:
        raise ValueError("V3 video inference requires at least one frame")
    from video_timing import VideoTiming

    timing = VideoTiming.from_environment()
    from rig_pose import START_POSE_CONTRACT

    if checkpoint.get("start_pose_contract") != START_POSE_CONTRACT:
        raise ValueError(
            "This v3 checkpoint predates the saved starting-pose contract; regenerate labels and retrain"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calib = checkpoint["calibration"]
    names = checkpoint["bone_names"]
    meta = checkpoint["bones_meta"]
    rig = RigModel(
        meta, names, checkpoint.get("rig_skinning"), checkpoint["extra_points"]
    )
    cameras_by_name = cameras_from_calibration(calib)
    order = calib["view_order"]
    cameras = [cameras_by_name[v] for v in order]
    W, H = int(calib["panel_width"]), int(calib["panel_height"])
    window = checkpoint.get("reader_window")
    scaled_window_side(window, W, H)
    temporal_context_contract = resolve_temporal_context_contract(checkpoint)
    stacked = calib.get("layout") == "stacked"
    images = []
    panels = [[] for _ in order]
    expected = (W, H * len(order)) if stacked else (W * len(order), H)
    for path in frame_paths:
        with Image.open(path) as raw:
            image = np.asarray(raw.convert("RGB"))
        if (image.shape[1], image.shape[0]) != expected:
            raise ValueError(
                f"Frame {Path(path).name} does not match calibrated {expected}"
            )
        images.append(image)
        for i in range(len(order)):
            panels[i].append(
                image[i * H : (i + 1) * H, :]
                if stacked
                else image[:, i * W : (i + 1) * W]
            )
    cadence = detect_video_cadence_optical_flow(images, fps=timing.fps)
    physical_times = (
        np.asarray(timing.map_source_frames(cadence["frame_times"])) / timing.fps
    )
    calibration = []
    skin = checkpoint.get("rig_skinning")
    if skin is not None:
        import cv2

        from rig_pose import start_pose_vertices

        start = start_pose_vertices(skin, meta)
        triangles = np.asarray(skin["triangles"])
        reference = checkpoint.get("frame0_reference")
        for v, c in enumerate(cameras):
            mask = np.zeros((H, W), np.uint8)
            proj = c.project(start)
            for face in triangles:
                cv2.fillConvexPoly(mask, np.rint(proj[face]).astype(np.int32), 1)
            first = panels[v][0]
            border = np.concatenate(
                [
                    first[:5].reshape(-1, 3),
                    first[-5:].reshape(-1, 3),
                    first[:, :5].reshape(-1, 3),
                    first[:, -5:].reshape(-1, 3),
                ]
            )
            bg = np.median(border, axis=0)
            target = np.linalg.norm(first.astype(float) - bg, axis=-1) > 25
            render = (
                reference[v]
                if reference is not None
                else np.where(mask[:, :, None], first, bg)
            ).astype(np.uint8)
            table = learn_frame0_shadow_table(render, first, mask)
            clean = strip_shadow(first, table)
            target &= np.linalg.norm(clean.astype(float) - bg, axis=-1) > 25
            cameras[v], diag = align_frame0_camera_chamfer(
                c, start, target, triangles=triangles, maxiter=20
            )
            panels[v] = [strip_shadow(im, table) for im in panels[v]]
            calibration.append(diag)
    base_backbone = resolve_checkpoint_backbone_name(checkpoint)
    model = MonsterDINOv2PoseProbe(
        num_keypoints=len(rig.point_names),
        custom_backbone=custom_backbone,
        backbone_name=base_backbone,
        enable_depth_visibility=True,
        enable_dense_surface=True,
        num_views=len(order),
        reader_v3=True,
        num_bones=len(names),
    ).to(device)
    load_probe_weights(model, checkpoint)
    model.eval()
    crops_ckpt = crops_checkpoint or (checkpoint.get("crops_checkpoint") if reader_crops_enabled() else None)
    dense_ckpt = dense_checkpoint or (checkpoint.get("dense_checkpoint") if reader_dense_enabled() else None)
    crops_model = None
    if crops_ckpt is not None:
        crops_model = MonsterDINOv2PoseProbe(
            num_keypoints=len(rig.point_names),
            custom_backbone=custom_backbone,
            backbone_name=resolve_checkpoint_backbone_name(crops_ckpt, default=base_backbone),
            enable_depth_visibility=True,
            enable_dense_surface=False,
            num_views=len(order),
            reader_v3=True,
            num_bones=len(names),
        ).to(device)
        load_probe_weights(crops_model, crops_ckpt)
        crops_model.eval()
    dense_model = None
    if dense_ckpt is not None:
        dense_model = MonsterDINOv2PoseProbe(
            num_keypoints=len(rig.point_names),
            custom_backbone=custom_backbone,
            backbone_name=resolve_checkpoint_backbone_name(dense_ckpt, default=base_backbone),
            enable_depth_visibility=True,
            enable_dense_surface=True,
            num_views=len(order),
            reader_v3=True,
            num_bones=len(names),
        ).to(device)
        load_probe_weights(dense_model, dense_ckpt)
        dense_model.eval()
    predictions = []
    weights = []
    reader_diagnostics = []
    first_passes = [
        infer_reader_sequence(model, panels[v], v, device, coarse_only=True)
        for v in range(len(order))
    ]
    centers = None
    if skin is not None:
        centers = multiview_dense_centers(
            rig,
            cameras,
            np.stack([p[0] for p in first_passes], axis=1),
            np.stack([p[1] for p in first_passes], axis=1),
            skin,
        )
    for v in range(len(order)):
        uv, conf, diag = infer_reader_sequence(
            model,
            panels[v],
            v,
            device,
            rig,
            cameras[v],
            first_pass=first_passes[v],
            center_depths=cameras[v].depth(centers) if centers is not None else None,
            window=window,
            temporal_context_contract=temporal_context_contract,
            crop_model=crops_model,
            dense_model=dense_model,
        )
        predictions.append(uv)
        weights.append(conf)
        reader_diagnostics.append(diag)
    uv = np.stack(predictions, axis=1)
    conf = np.stack(weights, axis=1)
    depth_list = [
        np.asarray(
            d.get("point_depth", np.zeros((len(uv), len(rig.point_names)), dtype=float)),
            dtype=float,
        )
        for d in reader_diagnostics
    ]
    depth_obs = np.stack(depth_list, axis=1) if depth_list else None
    readable = np.isfinite(uv).all(-1) & np.isfinite(conf)
    conf = np.where(readable, np.clip(conf, 0.0, 1.0), 0.0)
    uv = np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)
    permutation = observation_view_permutation(calib, cameras)
    uv = uv[:, permutation]
    conf = conf[:, permutation]
    if depth_obs is not None:
        depth_obs = np.nan_to_num(depth_obs[:, permutation], nan=0.0, posinf=0.0, neginf=0.0)
    cameras = [cameras[i] for i in permutation]
    order = [order[i] for i in permutation]
    from reading_pipeline import fix_all_twins_viterbi, jump_pass_constant_velocity, panel_drift

    use_no_cut = no_cut_tracks_enabled()
    use_twoway = twoway_jump_enabled()
    use_drift = panel_drift_enabled()
    gate_uv, gate_conf, _ = fix_all_twins_viterbi(
        rig, cameras, uv, conf, no_cut_tracks=use_no_cut
    )
    gate_uv, gate_conf = jump_pass_constant_velocity(
        rig, gate_uv, gate_conf, twoway=use_twoway
    )
    if use_drift:
        gate_uv, _ = panel_drift(rig, cameras, gate_uv, gate_conf)
    # Synchronization uses real readings before any low-confidence bridge interpolation.
    gate = geometric_consistency(cameras, gate_uv, gate_conf, H)
    uv, conf, cleaning = preprocess_readings(
        rig,
        cameras,
        uv,
        conf,
        no_cut_tracks=use_no_cut,
        twoway=use_twoway,
        correct_panel_drift=use_drift,
    )
    if not gate["gate_passed"]:
        return {
            **gate,
            "frames": [],
            "bones_meta": meta,
            "calibration": calib,
            "reader_contract": "monster-reader-v3",
        }
    params, solver = fit_clip_2d_reprojection(
        rig,
        cameras,
        uv,
        conf,
        w_prior=2.0,
        depth=depth_obs,
        w_depth=0.2,
        depth_where="unseen",
        robust_mode="scipy_hybrid",
    )
    fps = timing.fps
    if clean_motion_enabled():
        params, clean_report = clean_motion(
            rig,
            cameras,
            params,
            uv,
            conf,
            fps=fps,
            frame_times=physical_times,
            skin=skin,
        )
        steps = clean_report["root_from_steps"]
    else:
        params = despike_reading_backed(rig, params, cameras, uv, conf)
        params, steps = recover_root_translation_from_steps(
            rig, params, fps, frame_times=physical_times
        )
        params = simulate_damped_spring_chains(
            rig, params, fps, frame_times=physical_times
        )
        clean_report = {
            "steps_applied": [
                "despike_reading_backed",
                "recover_root_translation_from_steps",
                "simulate_damped_spring_chains",
            ],
            "root_from_steps": steps,
            "locked_feet": [],
            "keep_out_repairs": 0,
        }
    frames = rig.frames(params, conf)
    surface_diag = {}
    if surface_contact_enabled() and frames:
        actor_order = [n for n in rig.order if not str(n).startswith("Prop_")]
        body_root = next(
            (n for n in actor_order if n in rig.motion_roots),
            actor_order[0] if actor_order else None,
        )
        surface_cfg = None
        if body_root is not None:
            surface_cfg = meta.get(body_root, {}).get("surface_contact")
            if not isinstance(surface_cfg, dict):
                for b_cand in actor_order:
                    if isinstance(meta.get(b_cand, {}).get("surface_contact"), dict):
                        surface_cfg = meta[b_cand]["surface_contact"]
                        break
        if not isinstance(surface_cfg, dict) and isinstance(checkpoint.get("joint_limits_profile"), dict):
            surface_cfg = checkpoint["joint_limits_profile"].get("surface_contact")
        if (not isinstance(surface_cfg, dict) or not surface_cfg.get("support_sets")) and skin is not None and body_root is not None:
            rest_v = np.asarray(skin["rest_verts"], dtype=float)
            s_idx = np.asarray(skin["skin_indices"], dtype=int)
            s_w = np.asarray(skin["skin_weights"], dtype=float)
            s_names = {str(n): i for i, n in enumerate(skin["bone_names"])}
            support_sets = []
            for b_name in actor_order:
                if not b_name.startswith(("Foot", "FrontPaw", "HindPaw", "Tarsus")) or b_name not in s_names:
                    continue
                bi = s_names[b_name]
                dom = np.any((s_idx == bi) & (s_w > 0.3), axis=1)
                b_pts = rest_v[dom]
                near_floor = b_pts[b_pts[:, 2] <= rig.floor + 0.08 * rig.height]
                if len(near_floor) >= 1:
                    lowest = near_floor[np.argsort(near_floor[:, 2])[:8]]
                    local_s = (lowest - rig.h[b_name]) @ rig.R[b_name]
                    support_sets.append(
                        {
                            "bone": b_name,
                            "samples_local": local_s.tolist(),
                            "adjust_controls": True,
                        }
                    )
            if support_sets:
                surface_cfg = {
                    "enabled": True,
                    "floor_z": float(rig.floor),
                    "support_sets": support_sets,
                }
        if isinstance(surface_cfg, dict) and surface_cfg.get("enabled", True) and surface_cfg.get("support_sets") and body_root is not None:
            from surface_contact import apply_surface_floor_constraints

            eff_limits = {
                b: np.asarray(meta[b].get("limits_rad", [[-np.pi, np.pi]] * 3), dtype=float).tolist()
                for b in actor_order
            }
            surface_diag = apply_surface_floor_constraints(
                frames,
                actor_order,
                body_root,
                meta,
                {b: rig.h[b] for b in actor_order},
                {b: rig.t[b] for b in actor_order},
                {b: rig.R[b] for b in actor_order},
                eff_limits,
                {
                    **surface_cfg,
                    "fitted_bones": [
                        name
                        for name, entry in rig.slices.items()
                        if isinstance(entry, slice) or len(entry[0])
                    ],
                },
            )
    calib_center = list(calib.get("center", [0.0, 0.0, 0.5 * rig.height]))
    calib_depth_scale = float(calib.get("depth_scale", max(rig.height, 1e-3)))
    cx, cy = float(calib_center[0]), float(calib_center[1])
    for t, frame in enumerate(frames):
        for i, name in enumerate(names):
            bone = frame["bones"][name]
            hx, hy, _ = bone["head_3d"]
            tx, ty, _ = bone["tail_3d"]
            bone["depth_front_relative"] = [
                float((hy - cy) / calib_depth_scale),
                float((ty - cy) / calib_depth_scale),
            ]
            bone["depth_side_relative"] = [
                float((cx - hx) / calib_depth_scale),
                float((cx - tx) / calib_depth_scale),
            ]
            if "right" in order:
                bone["depth_right_relative"] = [
                    float((hx - cx) / calib_depth_scale),
                    float((tx - cx) / calib_depth_scale),
                ]
            for v, (view, camera) in enumerate(zip(order, cameras)):
                alias = "side" if view == "left" else view
                for endpoint, offset in (("head", 0), ("tail", 1)):
                    bone[f"uv_{alias}_{endpoint}"] = camera.project(
                        np.asarray(bone[endpoint + "_3d"])
                    ).tolist()
                    bone[f"reader_uv_{alias}_{endpoint}"] = uv[
                        t, v, 2 * i + offset
                    ].tolist()
    time_scale = timing.time_scale
    # Preserve coded holds and repeated images. Only explicit identical timestamps
    # supplied by a timing source share an instant; keep their latest reading.
    unique = {}
    for frame, time in zip(frames, timing.map_source_frames(cadence["frame_times"])):
        frame["frame"] = time
        unique[frame["frame"]] = frame
    frames = list(unique.values())
    result_calib = {
        "center": calib_center,
        "depth_scale": calib_depth_scale,
        **calib,
        "cameras": {name: cam.as_dict() for name, cam in zip(order, cameras)},
    }
    return {
        **gate,
        "bone_taxonomy_schema": "monster-bone-taxonomy-v3",
        "reader_contract": "monster-reader-v3",
        "pipeline_flags": pipeline_feature_flags(),
        "depth_visibility_enabled": True,
        "dense_surface_enabled": True,
        "bones_meta": meta,
        "calibration": result_calib,
        "motion_structure": {
            "motion_roots": sorted(rig.motion_roots),
            "fitted_bones": sorted(n for n, entry in rig.slices.items() if isinstance(entry, slice) or len(entry[0])),
        },
        "frames": frames,
        "taxonomy_diagnostics": {
            **solver,
            **surface_diag,
            "reading_preprocess": cleaning,
            "root_from_steps": steps,
            "clean_motion": clean_report,
            "at_limit": at_limit(rig, params),
        },
        "reader_diagnostics": reader_diagnostics,
        "frame0_calibration": calibration,
        "cadence": cadence,
        "time_scale": time_scale,
        "fps": timing.fps,
        "timing": timing.as_dict(),
        "joint_limits_profile": checkpoint.get("joint_limits_profile"),
    }
