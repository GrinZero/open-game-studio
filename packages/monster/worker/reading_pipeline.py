"""Observation disambiguation, first-frame calibration and reading-backed postprocessing."""

from dataclasses import replace
import math
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from studio_camera import triangulate
from reprojection_solver import exp_so3

NO_EFFECTS = (
    "No smoke, dust, sparks, glowing trails, blade light, particles, motion graphics, "
    "camera motion, cuts, text or visual effects. Keep the same body proportions, "
    "limb identities, synchronized views and fixed calibrated cameras throughout."
)


def _twin_groups(rig):
    name_to_point = {n: i for i, n in enumerate(rig.point_names)}
    roots = []
    for n in rig.names:
        if not n.endswith(".L") or n[:-2] + ".R" not in rig.meta:
            continue
        p = rig.parents[n]
        if p is None or not p.endswith(".L"):
            roots.append(n)
    groups = []
    for root in roots:
        owned = set()
        stack = [root]
        while stack:
            n = stack.pop()
            owned.add(n)
            stack.extend(a for a in rig.names if rig.parents[a] == n)
        left = []
        right = []
        for i, (name, bone) in enumerate(zip(rig.point_names, rig.point_bones)):
            twin = name.replace(".L", ".R")
            if bone in owned and twin in name_to_point:
                left.append(i)
                right.append(name_to_point[twin])
        if left:
            groups.append((np.array(left), np.array(right)))
    return groups


def fix_twins_viterbi(
    rig,
    cameras,
    uv,
    conf,
    side_index=1,
    switch_cost=6.0,
    lean=1.0,
    no_cut_tracks=False,
):
    uv = np.asarray(uv).copy()
    conf = np.asarray(conf).copy()
    T = len(uv)
    if not T or len(cameras) < 2:
        return uv, conf, []
    height_px = rig.height_px(cameras[side_index])
    swaps = []
    for left, right in _twin_groups(rig):
        ids = np.r_[left, right]
        flip = np.r_[right, left]
        cost = np.zeros((T, 2))
        for t in range(T):
            errors = []
            for state in (0, 1):
                obs = (
                    uv[t, :, ids].transpose(1, 0, 2).copy()
                )  # advanced indexing places K before V
                weights = conf[t, :, ids].T.copy()
                if state:
                    obs[side_index] = uv[t, side_index, flip]
                    weights[side_index] = conf[t, side_index, flip]
                X = triangulate(cameras, obs, weights)
                error = 0.0
                for v, c in enumerate(cameras):
                    good = np.isfinite(X).all(-1) & (weights[v] > 0.3)
                    if good.any():
                        error += np.sum(
                            np.linalg.norm(c.project(X[good]) - obs[v, good], axis=-1)
                            * weights[v, good]
                        )
                errors.append(error)
            keep, swap = errors
            emission = 0.0
            if swap < 0.6 * keep and keep - swap > 0.04 * height_px * len(ids):
                emission = 1.0
            elif keep < 0.6 * swap and swap - keep > 0.04 * height_px * len(ids):
                emission = -1.0
            if no_cut_tracks and t > 0 and emission > 0:
                # Veto a twin swap that breaks a smooth, confident side-view track (> 15% H jump).
                step_keep = np.linalg.norm(
                    uv[t, side_index, ids] - uv[t - 1, side_index, ids], axis=-1
                ).mean()
                step_swap = np.linalg.norm(
                    uv[t, side_index, flip] - uv[t - 1, side_index, ids], axis=-1
                ).mean()
                if (
                    step_keep < 0.05 * height_px
                    and step_swap > 0.15 * height_px
                    and np.mean(conf[t, side_index, ids]) > 0.7
                ):
                    emission = 0.0
            cost[t] = [
                max(0, emission),
                max(0, -emission) + (lean if emission == 0 else 0),
            ]
        dp = (
            cost[0] + np.array([0.0, float(switch_cost)])
            if T >= 2
            else cost[0].copy()
        )
        paths = np.zeros((T, 2), int)
        for t in range(1, T):
            matrix = dp[:, None] + np.array([[0, switch_cost], [switch_cost, 0]])
            paths[t] = matrix.argmin(0)
            dp = cost[t] + matrix.min(0)
        states = np.zeros(T, int)
        states[-1] = dp.argmin()
        for t in range(T - 1, 0, -1):
            states[t - 1] = paths[t, states[t]]
        if no_cut_tracks and np.any(states == 1):
            # Cancel any swap stretch [a, b) that cuts a smooth, well-separated side-view track (video2rig/reader/fit.py:side_path).
            starts = np.flatnonzero(np.diff(np.r_[0, states, 0]) == 1)
            ends = np.flatnonzero(np.diff(np.r_[0, states, 0]) == -1)
            for a, b in zip(starts, ends):
                cut = False
                for edge_t, prev_t in ((a, a - 1), (b, b - 1)):
                    if 0 <= prev_t and edge_t < T:
                        sep = float(
                            np.linalg.norm(
                                uv[edge_t, side_index, left]
                                - uv[edge_t, side_index, right],
                                axis=-1,
                            ).mean()
                        )
                        step_keep = float(
                            np.linalg.norm(
                                uv[edge_t, side_index, ids]
                                - uv[prev_t, side_index, ids],
                                axis=-1,
                            ).mean()
                        )
                        step_swap = float(
                            np.linalg.norm(
                                uv[edge_t, side_index, flip]
                                - uv[prev_t, side_index, ids],
                                axis=-1,
                            ).mean()
                        )
                        if (
                            sep > 0.08 * height_px
                            and step_keep < 0.05 * height_px
                            and step_swap > 0.12 * height_px
                        ):
                            cut = True
                            break
                if cut:
                    states[a:b] = 0
        for t in np.flatnonzero(states):
            uv[t, side_index, ids] = uv[t, side_index, flip]
            conf[t, side_index, ids] = conf[t, side_index, flip]
        swaps.append(
            {
                "left": rig.point_names[int(left[0])],
                "swapped_frames": np.flatnonzero(states).tolist(),
            }
        )
    return uv, conf, swaps


def _twin_parent_links(rig, left, right):
    """Build parent-keypoint link pairs `(child_idx_in_group, parent_idx_in_group)` along a twin limb group."""
    point_to_group_idx = {int(k): i for i, k in enumerate(left)}
    links = {}
    for i, k in enumerate(left):
        name = rig.point_names[int(k)]
        bone = rig.point_bones[int(k)]
        if name.endswith(".tail"):
            head_k = 2 * rig.index[bone]
            if head_k in point_to_group_idx and point_to_group_idx[head_k] != i:
                links[i] = point_to_group_idx[head_k]
        else:
            parent_bone = rig.parents.get(bone)
            if parent_bone is not None and parent_bone in rig.index:
                parent_tail = 2 * rig.index[parent_bone] + 1
                if parent_tail in point_to_group_idx and point_to_group_idx[parent_tail] != i:
                    links[i] = point_to_group_idx[parent_tail]
    return links


def _jump_pass_one_way(
    rig,
    uv,
    conf,
    *,
    twin_link: bool = True,
    twin_link_apart: float = 20.0,
    per_keypoint: bool = True,
):
    uv = uv.copy()
    conf = conf.copy()
    groups = _twin_groups(rig)
    swapped_mask = np.zeros((len(uv), uv.shape[1], len(groups)), dtype=bool)
    swapped_kp_mask = np.zeros((len(uv), uv.shape[1], uv.shape[2]), dtype=bool)
    for g_idx, (left, right) in enumerate(groups):
        ids = np.r_[left, right]
        flip = np.r_[right, left]
        links = _twin_parent_links(rig, left, right) if twin_link else {}
        for t in range(2, len(uv) - 1):
            for v in range(uv.shape[1]):
                expected = 2 * uv[t - 1, v, ids] - uv[t - 2, v, ids]
                keep = np.linalg.norm(uv[t, v, ids] - expected, axis=-1).mean()
                swap = np.linalg.norm(uv[t, v, flip] - expected, axis=-1).mean()
                next_error = np.linalg.norm(
                    uv[t + 1, v, ids] - uv[t, v, flip], axis=-1
                ).mean()
                if swap < 0.5 * keep and next_error < keep:
                    uv[t, v, ids] = uv[t, v, flip]
                    conf[t, v, ids] = conf[t, v, flip]
                    swapped_mask[t, v, g_idx] = True
                    swapped_kp_mask[t, v, left] = True
                    continue
                if not per_keypoint:
                    continue
                # Per-keypoint constant-velocity test with parent-chain proximity guard (video2rig/reader/fit.py:jump_pass)
                for i, (lk, rk) in enumerate(zip(left, right)):
                    if conf[t, v, lk] <= 0.2 or conf[t, v, rk] <= 0.2:
                        continue
                    if conf[t - 1, v, lk] <= 0.2 or conf[t - 1, v, rk] <= 0.2:
                        continue
                    if conf[t - 2, v, lk] <= 0.2 or conf[t - 2, v, rk] <= 0.2:
                        continue
                    sep = float(np.linalg.norm(uv[t, v, lk] - uv[t, v, rk]))
                    if sep < 20.0:
                        continue
                    pair = np.array([lk, rk])
                    pair_flip = np.array([rk, lk])
                    exp_pair = 2 * uv[t - 1, v, pair] - uv[t - 2, v, pair]
                    k_err = float(np.linalg.norm(uv[t, v, pair] - exp_pair, axis=-1).sum())
                    s_err = float(np.linalg.norm(uv[t, v, pair_flip] - exp_pair, axis=-1).sum())
                    if not (s_err < 0.3 * k_err and (k_err - s_err) >= 40.0):
                        continue
                    if i in links:
                        p_i = links[i]
                        pl, pr = int(left[p_i]), int(right[p_i])
                        p_sep = float(np.linalg.norm(uv[t, v, pl] - uv[t, v, pr]))
                        if p_sep >= twin_link_apart:
                            d_keep = float(
                                np.linalg.norm(uv[t, v, lk] - uv[t, v, pl])
                                + np.linalg.norm(uv[t, v, rk] - uv[t, v, pr])
                            )
                            d_swap = float(
                                np.linalg.norm(uv[t, v, rk] - uv[t, v, pl])
                                + np.linalg.norm(uv[t, v, lk] - uv[t, v, pr])
                            )
                            if d_swap > d_keep:
                                continue
                    uv[t, v, pair] = uv[t, v, pair_flip]
                    conf[t, v, pair] = conf[t, v, pair_flip]
                    swapped_kp_mask[t, v, lk] = True
    return uv, conf, swapped_mask, swapped_kp_mask


def jump_pass_constant_velocity(
    rig,
    uv,
    conf,
    twoway=False,
    *,
    twin_link: bool = True,
    twin_link_apart: float = 20.0,
    per_keypoint: bool = True,
):
    if not twoway:
        out_uv, out_conf, _, _ = _jump_pass_one_way(
            rig,
            uv,
            conf,
            twin_link=twin_link,
            twin_link_apart=twin_link_apart,
            per_keypoint=per_keypoint,
        )
        return out_uv, out_conf
    _, _, fwd_mask, fwd_kp = _jump_pass_one_way(
        rig,
        uv,
        conf,
        twin_link=twin_link,
        twin_link_apart=twin_link_apart,
        per_keypoint=per_keypoint,
    )
    _, _, bwd_mask_rev, bwd_kp_rev = _jump_pass_one_way(
        rig,
        uv[::-1],
        conf[::-1],
        twin_link=twin_link,
        twin_link_apart=twin_link_apart,
        per_keypoint=per_keypoint,
    )
    agree_group = fwd_mask & bwd_mask_rev[::-1]
    agree_kp = fwd_kp & bwd_kp_rev[::-1]
    out_uv = uv.copy()
    out_conf = conf.copy()
    for g_idx, (left, right) in enumerate(_twin_groups(rig)):
        ids = np.r_[left, right]
        flip = np.r_[right, left]
        for t, v in np.argwhere(agree_group[:, :, g_idx]):
            out_uv[t, v, ids] = out_uv[t, v, flip]
            out_conf[t, v, ids] = out_conf[t, v, flip]
        for lk, rk in zip(left, right):
            for t, v in np.argwhere(agree_kp[:, :, lk] & ~agree_group[:, :, g_idx]):
                pair = np.array([lk, rk])
                pair_flip = np.array([rk, lk])
                out_uv[t, v, pair] = out_uv[t, v, pair_flip]
                out_conf[t, v, pair] = out_conf[t, v, pair_flip]
    return out_uv, out_conf


def fix_all_twins_viterbi(rig, cameras, uv, conf, no_cut_tracks=False):
    changes = []
    for side in range(1, len(cameras)):
        uv, conf, report = fix_twins_viterbi(
            rig, cameras, uv, conf, side_index=side, no_cut_tracks=no_cut_tracks
        )
        changes.extend([{**item, "side_index": side} for item in report])
    return uv, conf, changes


def downweight_overlapping_twins(rig, uv_or_cameras, conf_or_uv, height_px_or_conf):
    """Downweight overlapping left/right twin keypoints in secondary views (`twin_same`),
    unless adjacent keypoints along the limb chain are also close (`0 < max(apart) <= 3 * near`),
    which indicates both limbs are genuinely standing/posed together (video2rig/reader/fit.py:391-419).
    """
    if isinstance(uv_or_cameras, (list, tuple)) and uv_or_cameras and hasattr(uv_or_cameras[0], "project"):
        cameras = uv_or_cameras
        uv = np.asarray(conf_or_uv, dtype=float)
        conf = np.asarray(height_px_or_conf, dtype=float).copy()
        height_px = [rig.height_px(c) for c in cameras]
    else:
        uv = np.asarray(uv_or_cameras, dtype=float)
        conf = np.asarray(conf_or_uv, dtype=float).copy()
        height_px = height_px_or_conf
    heights = np.broadcast_to(height_px, (uv.shape[1],))
    for left, right in _twin_groups(rig):
        n_pts = len(left)
        for v in range(1, uv.shape[1]):
            near_thresh = 0.06 * heights[v]
            dist = np.linalg.norm(uv[:, v, left] - uv[:, v, right], axis=-1)
            near = dist < near_thresh
            for i, (l, r) in enumerate(zip(left, right)):
                adj = [j for j in (i - 1, i + 1) if 0 <= j < n_pts]
                for t in np.flatnonzero(near[:, i]):
                    if adj:
                        apart = [
                            float(dist[t, j])
                            for j in adj
                            if np.isfinite(dist[t, j])
                        ]
                        if apart and 0.0 < max(apart) <= 3.0 * near_thresh:
                            continue
                    conf[t, v, l] *= 0.4
                    conf[t, v, r] *= 0.4
    return conf


def downweight_flipped_limbs(
    rig,
    cameras,
    uv,
    conf,
    *,
    flip_deg: float = 60.0,
    flip_back_deg: float = 30.0,
    bridge_max: int = 24,
    flip_weight: float = 0.1,
    stay_frac: float = 0.02,
):
    """Downweight temporary limb-segment direction flips (`> 60 deg`) under a still upper joint (`< 2% H`)
    that flip back within `< 30 deg` in `<= 24` frames (video2rig/reader/fit.py:flipped).
    """
    conf_out = np.asarray(conf, dtype=float).copy()
    uv_arr = np.asarray(uv, dtype=float)
    T = len(uv_arr)
    if T < 3:
        return conf_out
    cos_flip = math.cos(math.radians(flip_deg))
    cos_back = math.cos(math.radians(flip_back_deg))
    heights = [rig.height_px(c) for c in cameras]

    for i, name in enumerate(rig.names):
        kind = rig.meta.get(name, {}).get("kind", "")
        if kind in ("root", "spine", "head", "tail", "ear", "loose", "rigid") or name in getattr(
            rig, "loose", ()
        ):
            continue
        kh, kt = 2 * i, 2 * i + 1
        if kt >= uv_arr.shape[2]:
            continue
        for v in range(uv_arr.shape[1]):
            stay_px = stay_frac * heights[min(v, len(heights) - 1)]
            vec = uv_arr[:, v, kt] - uv_arr[:, v, kh]
            vlen = np.linalg.norm(vec, axis=-1)
            valid = (
                (conf_out[:, v, kh] > 0.3)
                & (conf_out[:, v, kt] > 0.3)
                & np.isfinite(vlen)
                & (vlen > 2.0)
            )
            unit = np.zeros_like(vec)
            unit[valid] = vec[valid] / vlen[valid, None]
            t = 1
            while t < T - 1:
                if not (valid[t - 1] and valid[t]):
                    t += 1
                    continue
                head_step = float(
                    np.linalg.norm(uv_arr[t, v, kh] - uv_arr[t - 1, v, kh])
                )
                dot_enter = float(np.dot(unit[t - 1], unit[t]))
                if head_step <= stay_px and dot_enter < cos_flip:
                    # Search for return within bridge_max frames
                    end_limit = min(T, t + bridge_max + 1)
                    back_t = None
                    for b in range(t + 1, end_limit):
                        if not valid[b]:
                            continue
                        if float(np.dot(unit[t - 1], unit[b])) > cos_back:
                            back_t = b
                            break
                    if back_t is not None:
                        conf_out[t:back_t, v, kt] *= flip_weight
                        t = back_t
                        continue
                t += 1
    return conf_out


def bridge_unsure_stretches(uv, conf, max_gap=24):
    uv = uv.copy()
    conf = conf.copy()
    for v in range(uv.shape[1]):
        for k in range(uv.shape[2]):
            good = (conf[:, v, k] >= 0.4) & np.isfinite(uv[:, v, k]).all(-1)
            starts = np.flatnonzero(
                np.diff(np.r_[False, ~good, False].astype(int)) == 1
            )
            ends = np.flatnonzero(np.diff(np.r_[False, ~good, False].astype(int)) == -1)
            for a, b in zip(starts, ends):
                if a == 0 or b == len(uv) or b - a > max_gap:
                    continue
                alpha = np.arange(1, b - a + 1) / (b - a + 1)
                uv[a:b, v, k] = (1 - alpha[:, None]) * uv[a - 1, v, k] + alpha[
                    :, None
                ] * uv[b, v, k]
                conf[a:b, v, k] = np.maximum(conf[a:b, v, k], 0.25)
    return uv, conf


def panel_drift(rig, cameras, uv, conf):
    """Correct median shared-axis drift (> 5% H) between orthographic panels on midline bones."""
    uv_out = np.asarray(uv, float).copy()
    if len(cameras) < 2 or not len(uv_out):
        return uv_out, []
    midline = [
        k
        for k, owner in enumerate(rig.point_bones)
        if rig.meta.get(owner, {}).get("kind") in ("root", "spine", "head")
        and not owner.endswith((".L", ".R"))
    ]
    if not midline:
        return uv_out, []
    drifts = []
    c0 = cameras[0]
    if c0.type != "ortho":
        return uv_out, []
    h0 = rig.height_px(c0)
    for v in range(1, len(cameras)):
        cv = cameras[v]
        if cv.type != "ortho" or abs(rig.height_px(cv) - h0) > 1e-2 * h0:
            continue
        # Check which image axis (0=u or 1=v) shares the same 3D world direction between c0 and cv.
        for ax0 in (0, 1):
            dir0 = c0.R[ax0]
            for axv in (0, 1):
                dirv = cv.R[axv]
                if abs(float(dir0 @ dirv) - 1.0) > 1e-3:
                    continue
                valid = (
                    (conf[:, 0, midline] >= 0.6)
                    & (conf[:, v, midline] >= 0.6)
                    & np.isfinite(uv_out[:, 0, midline, ax0])
                    & np.isfinite(uv_out[:, v, midline, axv])
                )
                if not np.any(valid):
                    continue
                # Compare against rest-pose offset on that shared axis
                X0 = rig.fk(rig.start_params())[0][midline]
                expected_diff = cv.project(X0)[:, axv] - c0.project(X0)[:, ax0]
                measured_diff = uv_out[:, v, midline, axv] - uv_out[:, 0, midline, ax0] - expected_diff[None, :]
                median_drift = float(np.median(measured_diff[valid]))
                if abs(median_drift) > 0.05 * h0:
                    uv_out[:, v, :, axv] -= median_drift
                    drifts.append(
                        {
                            "view": int(v),
                            "axis": int(axv),
                            "drift_px": median_drift,
                            "drift_H": median_drift / max(h0, 1e-6),
                        }
                    )
    return uv_out, drifts


def downweight_contradicting_far_limb(
    rig,
    cameras,
    uv,
    conf,
    height_px=None,
    *,
    midline_ratio=None,
    window_expand: bool = False,
):
    conf = conf.copy()
    if len(cameras) < 2:
        return conf
    if height_px is None:
        height_px = [rig.height_px(c) for c in cameras]
    heights = np.broadcast_to(height_px, (len(cameras),))
    rest = rig.fk(rig.start_params())[0]
    previous = rest
    groups = _twin_groups(rig)
    midline = [
        k
        for k, owner in enumerate(rig.point_bones)
        if rig.meta.get(owner, {}).get("kind") in ("root", "spine", "head")
    ]
    T = len(uv)
    bad_mask = np.zeros(conf.shape, dtype=bool)
    for t in range(T):
        X = triangulate(cameras, uv[t], conf[t], previous)
        far = np.zeros(conf.shape[1:], bool)
        for left, right in groups:
            root = rig.point_bones[left[0]]
            while rig.parents[root] is not None and rig.parents[root].endswith(".L"):
                root = rig.parents[root]
            lroot = 2 * rig.index[root]
            rroot = 2 * rig.index[root[:-2] + ".R"]
            a, b = X[lroot], X[rroot]
            distance = np.linalg.norm(a - b)
            if distance < 1e-10:
                continue
            for v, camera in enumerate(cameras):
                difference = float(camera.depth(a) - camera.depth(b))
                if difference > 0.5 * distance:
                    far[v, left] = True
                elif difference < -0.5 * distance:
                    far[v, right] = True
        for v in range(1, len(cameras)):
            predicted = cameras[v].project(X)
            error = np.linalg.norm(predicted - uv[t, v], axis=-1)
            thresh = 0.12 * heights[v] + 1e-10
            if midline_ratio is not None and midline and np.isfinite(error[midline]).all():
                thresh = max(thresh, float(midline_ratio) * float(np.median(error[midline])))
            bad = (
                far[v]
                & ~far[0]
                & (conf[t, 0] > 0.7)
                & (conf[t, v] < 0.5)
                & (error > thresh)
            )
            bad_mask[t, v] = bad
        previous = X
    if window_expand and T >= 3:
        bad_mask = (
            bad_mask
            | np.r_[bad_mask[:1], bad_mask[:-1]]
            | np.r_[bad_mask[1:], bad_mask[-1:]]
        )
    conf[bad_mask] *= 0.1
    return conf


pick_panels = downweight_contradicting_far_limb


def preprocess_readings(
    rig,
    cameras,
    uv,
    conf,
    *,
    no_cut_tracks=None,
    twoway=None,
    correct_panel_drift=None,
    check_flipped_limbs=True,
):
    from pose_features import (
        no_cut_tracks_enabled,
        panel_drift_enabled,
        twoway_jump_enabled,
    )

    if no_cut_tracks is None:
        no_cut_tracks = no_cut_tracks_enabled()
    if twoway is None:
        twoway = twoway_jump_enabled()
    if correct_panel_drift is None:
        correct_panel_drift = panel_drift_enabled()
    uv, conf, swaps = fix_all_twins_viterbi(
        rig, cameras, uv, conf, no_cut_tracks=no_cut_tracks
    )
    uv, conf = jump_pass_constant_velocity(rig, uv, conf, twoway=twoway)
    drifts = []
    if correct_panel_drift:
        uv, drifts = panel_drift(rig, cameras, uv, conf)
    H = np.array([rig.height_px(camera) for camera in cameras])
    conf = downweight_overlapping_twins(rig, uv, conf, H)
    if check_flipped_limbs:
        conf = downweight_flipped_limbs(rig, cameras, uv, conf)
    uv, conf = bridge_unsure_stretches(uv, conf)
    conf = downweight_contradicting_far_limb(
        rig, cameras, uv, conf, H, midline_ratio=2.0, window_expand=True
    )
    return uv, conf, {"twins": swaps, "bridge_max_frames": 24, "panel_drifts": drifts}


def despike_reading_backed(rig, params, cameras, uv, confidence, despike_limbs=True):
    """Filter individual angular (and root translation) spikes only when their affected readings support it."""
    out = params.copy()
    for t in range(1, len(params) - 1):
        average = (params[t - 1] + params[t + 1]) / 2
        entries = list(rig.slices.values()) if despike_limbs else [
            s for s in rig.slices.values() if isinstance(s, slice)
        ]
        for entry in entries:
            if isinstance(entry, slice):
                rot_idx = np.arange(entry.start, entry.start + 3)
                trans_idx = np.arange(entry.start + 3, entry.stop)
                spike = np.r_[
                    rot_idx[abs(params[t, rot_idx] - average[rot_idx]) > math.radians(6)],
                    trans_idx[abs(params[t, trans_idx] - average[trans_idx]) > 0.03 * rig.height],
                ]
            else:
                indices = entry[0]
                spike = indices[
                    abs(params[t, indices] - average[indices]) > math.radians(6)
                ]
            if not len(spike):
                continue
            original = out[t].copy()
            proposal = original.copy()
            proposal[spike] = average[spike]
            proposal = np.clip(proposal, rig.lower, rig.upper)
            X = rig.fk(original)[0]
            smooth = rig.fk(proposal)[0]
            affected = np.linalg.norm(X - smooth, axis=-1) > 1e-5 * rig.height
            trusted = (
                (confidence[t] > 0.3 if not isinstance(entry, slice) else confidence[t] > 0.5)
                & affected[None]
                & np.isfinite(uv[t]).all(-1)
            )

            def error(points):
                residual = np.stack(
                    [
                        np.linalg.norm(camera.project(points) - uv[t, v], axis=-1)
                        for v, camera in enumerate(cameras)
                    ]
                )
                return float(np.median(residual[trusted])) if trusted.any() else 0.0

            if error(smooth) - error(X) <= 1.0:
                out[t] = proposal
    return out


def _physical_frame_times(count, fps, frame_times):
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Postprocessing FPS must be finite and positive")
    times = (
        np.arange(count, dtype=float) / fps
        if frame_times is None
        else np.asarray(frame_times, float)
    )
    if (
        times.shape != (count,)
        or not np.isfinite(times).all()
        or (np.diff(times) < 0).any()
    ):
        raise ValueError("Postprocessing needs ordered finite times in seconds")
    return times


def _lowpass_in_seconds(values, times, fps, cutoff=2.0):
    """Resample physical time before filtering; repeated coded frames add no time."""
    from scipy.signal import butter, sosfiltfilt

    values = np.asarray(values, float)
    keep = np.r_[np.diff(times) > 1e-10, True]
    unique = times[keep]
    samples = values[keep]
    if len(unique) < 4 or unique[-1] - unique[0] < 1e-10:
        return values.copy()
    count = max(4, int(min(4 * len(unique), (unique[-1] - unique[0]) * fps)) + 1)
    uniform = np.linspace(unique[0], unique[-1], count)
    sample_fps = (count - 1) / (uniform[-1] - uniform[0])
    interpolated = np.column_stack(
        [np.interp(uniform, unique, samples[:, i]) for i in range(values.shape[1])]
    )
    sos = butter(2, min(cutoff, sample_fps * 0.4), fs=sample_fps, output="sos")
    filtered = sosfiltfilt(sos, interpolated, axis=0, padlen=min(count - 1, 9))
    return np.column_stack(
        [np.interp(times, uniform, filtered[:, i]) for i in range(values.shape[1])]
    )


def _adaptive_cutoff_hz(
    series_1d,
    times,
    fps,
    is_angle=True,
    height=1.0,
    lowpass_hz=8.0,
    lowpass_slow_hz=4.0,
    lowpass_burst_hz=5.0,
):
    dt = np.diff(times)
    advancing = dt > 1e-10
    if not np.any(advancing):
        return lowpass_slow_hz
    diffs = np.abs(np.diff(series_1d, axis=0)[advancing])
    speeds = diffs / dt[advancing, None] if series_1d.ndim == 2 else diffs / dt[advancing]
    deg_per_s = np.degrees(speeds) if is_angle else (speeds / max(height, 1e-6)) * 180.0
    med = float(np.median(deg_per_s))
    p90 = float(np.percentile(deg_per_s, 90))
    if med < 150.0 and p90 < 450.0:
        col = series_1d if series_1d.ndim == 2 else series_1d[:, None]
        ref_3hz = _lowpass_in_seconds(col, times, fps, cutoff=3.0)
        ref_diffs = np.abs(np.diff(ref_3hz, axis=0)[advancing]) / dt[advancing, None]
        ref_deg = np.degrees(ref_diffs) if is_angle else (ref_diffs / max(height, 1e-6)) * 180.0
        if float(np.max(ref_deg)) > 450.0:
            return lowpass_burst_hz
        return lowpass_slow_hz
    return lowpass_hz


def smooth_turn(
    rig,
    params,
    fps=24.0,
    frame_times=None,
    turn_spike_deg=6.0,
    lowpass_hz=8.0,
    lowpass_slow_hz=4.0,
    lowpass_burst_hz=5.0,
):
    """Smooth root rotation via sign-continuous quaternions, 1-frame spike removal, and adaptive low-pass."""
    from scipy.spatial.transform import Rotation
    from reprojection_solver import unwrap_rotvecs

    out = np.asarray(params, float).copy()
    if len(out) < 4:
        return out
    times = _physical_frame_times(len(out), fps, frame_times)
    for n, sl in rig.slices.items():
        if not isinstance(sl, slice):
            continue
        rotvecs = out[:, sl.start : sl.start + 3]
        quats = Rotation.from_rotvec(rotvecs).as_quat()  # [x, y, z, w]
        for t in range(1, len(quats)):
            if float(np.dot(quats[t], quats[t - 1])) < 0.0:
                quats[t] = -quats[t]
        for t in range(1, len(quats) - 1):
            mid = quats[t - 1] + quats[t + 1]
            norm_mid = float(np.linalg.norm(mid))
            if norm_mid < 1e-8:
                continue
            mid = mid / norm_mid
            ang_deg = math.degrees(2.0 * math.acos(float(np.clip(abs(np.dot(quats[t], mid)), -1.0, 1.0))))
            span_deg = math.degrees(
                2.0 * math.acos(float(np.clip(abs(np.dot(quats[t - 1], quats[t + 1])), -1.0, 1.0)))
            )
            if ang_deg > turn_spike_deg and ang_deg > 1.5 * span_deg:
                quats[t] = mid if float(np.dot(mid, quats[t - 1])) >= 0.0 else -mid
        cutoff = _adaptive_cutoff_hz(
            unwrap_rotvecs(Rotation.from_quat(quats).as_rotvec()),
            times,
            fps,
            is_angle=True,
            lowpass_hz=lowpass_hz,
            lowpass_slow_hz=lowpass_slow_hz,
            lowpass_burst_hz=lowpass_burst_hz,
        )
        filtered_q = _lowpass_in_seconds(quats, times, fps, cutoff=cutoff)
        norms = np.linalg.norm(filtered_q, axis=-1, keepdims=True)
        filtered_q = filtered_q / np.maximum(norms, 1e-12)
        out[:, sl.start : sl.start + 3] = unwrap_rotvecs(
            Rotation.from_quat(filtered_q).as_rotvec()
        )
    return out


def lowpass_motion_adaptive(
    rig,
    params,
    fps=24.0,
    frame_times=None,
    lowpass_hz=8.0,
    lowpass_slow_hz=4.0,
    lowpass_burst_hz=5.0,
):
    """Velocity-adaptive 2nd-order zero-phase Butterworth low-pass filter (4 / 5 / 8 Hz)."""
    out = np.asarray(params, float).copy()
    if len(out) < 4:
        return out
    times = _physical_frame_times(len(out), fps, frame_times)
    for n, entry in rig.slices.items():
        if isinstance(entry, slice):
            trans_idx = np.arange(entry.start + 3, entry.stop)
            for col in trans_idx:
                cutoff = _adaptive_cutoff_hz(
                    out[:, col],
                    times,
                    fps,
                    is_angle=False,
                    height=rig.height,
                    lowpass_hz=lowpass_hz,
                    lowpass_slow_hz=lowpass_slow_hz,
                    lowpass_burst_hz=lowpass_burst_hz,
                )
                out[:, col] = _lowpass_in_seconds(
                    out[:, col : col + 1], times, fps, cutoff=cutoff
                )[:, 0]
        else:
            indices, _ = entry
            for col in indices:
                cutoff = _adaptive_cutoff_hz(
                    out[:, col],
                    times,
                    fps,
                    is_angle=True,
                    lowpass_hz=lowpass_hz,
                    lowpass_slow_hz=lowpass_slow_hz,
                    lowpass_burst_hz=lowpass_burst_hz,
                )
                out[:, col] = _lowpass_in_seconds(
                    out[:, col : col + 1], times, fps, cutoff=cutoff
                )[:, 0]
    return np.clip(out, rig.lower, rig.upper)


def floor_bend_and_smooth_lift(rig, params, cameras=None, floor_bend="still"):
    """Bend penetrating limbs toward the floor while preserving camera rays, then apply 60-pass smooth root-Z lift."""
    out = np.asarray(params, float).copy()
    if not len(out):
        return out
    actor_mask = np.array(
        [not str(b).startswith("Prop_") for b in rig.point_bones], dtype=bool
    )
    if floor_bend == "still":
        for t in range(len(out)):
            X, J, _, _ = rig.fk(out[t], jac=True)
            below = np.flatnonzero(actor_mask & (X[:, 2] < rig.floor - 0.002 * rig.height))
            if not len(below):
                continue
            limb_cols = []
            for k in below:
                owner = rig.point_bones[int(k)]
                curr = owner
                while curr is not None and curr not in rig.motion_roots:
                    entry = rig.slices.get(curr)
                    if entry is not None and not isinstance(entry, slice):
                        limb_cols.extend(entry[0].tolist())
                    curr = rig.parents.get(curr)
            limb_cols = sorted(set(limb_cols))
            if limb_cols:
                dz = (rig.floor - X[below, 2])
                Jz = J[below, 2][:, limb_cols]
                if cameras:
                    proj_rows = []
                    for cam in cameras:
                        _, P = cam.project(X, jac=True)
                        proj_rows.append((P @ J[:, :, limb_cols]).reshape(-1, len(limb_cols)) * (0.05 * rig.height / max(rig.height_px(cam), 1.0)))
                    A = np.vstack([Jz * 10.0, *proj_rows, np.eye(len(limb_cols)) * 0.2])
                    b = np.r_[dz * 10.0, np.zeros(A.shape[0] - len(dz))]
                else:
                    A = np.vstack([Jz * 10.0, np.eye(len(limb_cols)) * 0.2])
                    b = np.r_[dz * 10.0, np.zeros(len(limb_cols))]
                step, *_ = np.linalg.lstsq(A, b, rcond=None)
                out[t, limb_cols] = np.clip(
                    out[t, limb_cols] + step, rig.lower[limb_cols], rig.upper[limb_cols]
                )
    roots = [
        s for n, s in rig.slices.items() if isinstance(s, slice) and not n.startswith("Prop_")
    ]
    if roots and np.any(actor_mask):
        need = np.array(
            [max(0.0, float(rig.floor - rig.fk(p)[0][actor_mask, 2].min())) for p in out],
            dtype=float,
        )
        if np.any(need > 0):
            lift = need.copy()
            kernel = np.array([0.25, 0.5, 0.25], dtype=float)
            for _ in range(60):
                lift = np.maximum(
                    np.convolve(np.pad(lift, 1, mode="edge"), kernel, mode="valid"),
                    need,
                )
            for root_slice in roots:
                out[:, root_slice.start + 5] += lift
    return out


def recover_root_translation_from_steps(
    rig, params, fps=24.0, frame_times=None, steps_steady=0.5
):
    times = _physical_frame_times(len(params), fps, frame_times)
    if len(params) < 4:
        return params, {"applied": False, "reason": "short_clip"}
    ids = [
        2 * rig.index[n] + 1
        for n in rig.names
        if n.startswith(("Foot", "FrontPaw", "HindPaw", "Tarsus"))
    ]
    roots = [
        (n, s)
        for n, s in rig.slices.items()
        if isinstance(s, slice) and not n.startswith("Prop_")
    ]
    if not ids or not roots:
        return params, {"applied": False, "reason": "no_feet_or_body_root"}
    root, root_slice = roots[0]
    points = np.array([rig.fk(p)[0] for p in params])
    feet = points[:, ids]
    dt = np.diff(times)
    advancing = dt > 1e-10
    velocity = np.divide(
        np.diff(feet[:, :, :2], axis=0),
        dt[:, None, None],
        out=np.zeros((len(dt), len(ids), 2)),
        where=advancing[:, None, None],
    )
    contact = (feet[:-1, :, 2] < rig.floor + 0.05 * rig.height) & (
        feet[1:, :, 2] < rig.floor + 0.05 * rig.height
    )
    v = np.zeros((len(dt), 2))
    for t in range(len(v)):
        if contact[t].any():
            v[t] = np.median(velocity[t, contact[t]], axis=0)
    speeds = np.linalg.norm(v, axis=-1)
    moving = (speeds > 0.08 * rig.height) & advancing
    share = float(moving.sum() / max(1, advancing.sum()))
    report = {
        "applied": False,
        "moving_share": share,
        "frequency_hz": 2.0,
        "body_root": root,
    }
    if share < 0.55:
        return params, {**report, "reason": "insufficient_backward_steps"}
    directions = v[moving] / np.linalg.norm(v[moving], axis=-1, keepdims=True)
    consistency = float(np.linalg.norm(directions.mean(0)))
    report["direction_consistency"] = consistency
    if consistency < 0.92:
        return params, {**report, "reason": "inconsistent_steps"}
    pace_spread = float(np.std(speeds[moving]) / max(np.mean(speeds[moving]), 1e-10))
    report["pace_spread"] = pace_spread
    if steps_steady is not None and pace_spread > float(steps_steady):
        return params, {**report, "reason": "unsteady_step_pace"}
    location = params[:, root_slice.start + 3 : root_slice.start + 5]
    filtered_root = _lowpass_in_seconds(location, times, fps)
    root_velocity = np.divide(
        np.diff(filtered_root, axis=0),
        dt[:, None],
        out=np.zeros_like(v),
        where=advancing[:, None],
    )
    root_speed = float(np.median(np.linalg.norm(root_velocity[moving], axis=-1)))
    report["root_speed_H_per_s"] = root_speed / rig.height
    if root_speed > 0.08 * rig.height:
        return params, {**report, "reason": "existing_root_motion"}
    midtimes = (times[:-1] + times[1:]) / 2
    smooth = np.zeros_like(v)
    if advancing.any():
        smooth[advancing] = _lowpass_in_seconds(-v[advancing], midtimes[advancing], fps)
    shift = np.vstack([np.zeros(2), np.cumsum(smooth * dt[:, None], axis=0)])
    out = params.copy()
    out[:, root_slice.start + 3 : root_slice.start + 5] += shift
    hands = [
        2 * rig.index[n] + 1
        for n in rig.names
        if n.startswith(("Hand", "FrontPaw", "WingHand"))
    ]
    translated = []
    for n, s in rig.slices.items():
        if not isinstance(s, slice) or not n.startswith("Prop_") or not hands:
            continue
        distance = np.linalg.norm(
            points[:, 2 * rig.index[n], None] - points[:, hands], axis=-1
        ).min(-1)
        if np.mean(distance < 0.08 * rig.height) >= 0.55:
            out[:, s.start + 3 : s.start + 5] += shift
            translated.append(n)
    return out, {
        **report,
        "applied": True,
        "displacement": shift[-1].tolist(),
        "translated_props": translated,
    }


def lock_feet(
    rig,
    params,
    cameras=None,
    uv=None,
    confidence=None,
    fps=24.0,
    frame_times=None,
    lock_blend=3,
):
    """Pin both foot endpoints (head & tail) during stance stretches (>=3 frames) with linear fade-in/out."""
    out = np.asarray(params, float).copy()
    T = len(out)
    if T < 3:
        return out, []
    times = _physical_frame_times(T, fps, frame_times)
    dt = np.maximum(np.diff(times, prepend=times[0] - 1.0 / max(fps, 1.0)), 1e-6)
    X_seq = np.array([rig.fk(p)[0] for p in out])
    X_rest = rig.fk(rig.start_params())[0]
    foot_bones = [
        n
        for n in rig.names
        if n.startswith(("Foot.", "FrontPaw.", "HindPaw.", "Tarsus"))
    ]
    locked_reports = []
    for fname in foot_bones:
        fi = 2 * rig.index[fname]
        ends = [fi, fi + 1]
        rest_z = X_rest[ends, 2]
        near_floor = np.all(
            np.abs(X_seq[:, ends, 2] - rest_z[None, :]) <= 0.03 * rig.height, axis=-1
        ) | np.all(X_seq[:, ends, 2] <= rig.floor + 0.035 * rig.height, axis=-1)
        vel_3d = np.linalg.norm(np.diff(X_seq[:, ends], axis=0, prepend=X_seq[:1, ends]), axis=-1) / dt[:, None]
        slow_3d = np.all(vel_3d < 0.20 * rig.height, axis=-1)
        slow_2d = np.ones(T, dtype=bool)
        if cameras is not None and uv is not None and confidence is not None:
            for v, cam in enumerate(cameras):
                h_px = max(rig.height_px(cam), 1.0)
                vis = np.all(confidence[:, v, ends] > 0.4, axis=-1)
                vel_uv = (
                    np.linalg.norm(
                        np.diff(uv[:, v, ends], axis=0, prepend=uv[:1, v, ends]), axis=-1
                    )
                    / dt[:, None]
                )
                slow_2d &= (~vis) | np.all(vel_uv < 0.20 * h_px, axis=-1)
        stance = near_floor & slow_3d & slow_2d
        # Close 1-frame gaps inside a stance stretch
        for t in range(1, T - 1):
            if not stance[t] and stance[t - 1] and stance[t + 1]:
                stance[t] = True
        starts = np.flatnonzero(np.diff(np.r_[False, stance, False].astype(int)) == 1)
        stops = np.flatnonzero(np.diff(np.r_[False, stance, False].astype(int)) == -1)
        chain_cols = []
        curr = fname
        while curr is not None and curr not in rig.motion_roots:
            entry = rig.slices.get(curr)
            if entry is not None and not isinstance(entry, slice):
                chain_cols.extend(entry[0].tolist())
            curr = rig.parents.get(curr)
        chain_cols = sorted(set(chain_cols))
        if not chain_cols:
            continue
        for a, b in zip(starts, stops):
            if b - a < 3:
                continue
            pin_target = np.median(X_seq[a:b, ends], axis=0)
            b_start = max(0, a - int(lock_blend))
            b_end = min(T, b + int(lock_blend))
            for t in range(b_start, b_end):
                if t < a:
                    w = (t - b_start + 1) / float(a - b_start + 1)
                elif t >= b:
                    w = (b_end - t) / float(b_end - b + 1)
                else:
                    w = 1.0
                goal = (1.0 - w) * X_seq[t, ends] + w * pin_target
                for _ in range(3):
                    Xt, Jt, _, _ = rig.fk(out[t], jac=True)
                    err = (goal - Xt[ends]).ravel()
                    if float(np.linalg.norm(err)) < 1e-5 * rig.height:
                        break
                    Jc = Jt[ends][:, :, chain_cols].reshape(-1, len(chain_cols))
                    A = np.vstack([Jc, np.eye(len(chain_cols)) * 0.05])
                    rhs = np.r_[err, np.zeros(len(chain_cols))]
                    step, *_ = np.linalg.lstsq(A, rhs, rcond=None)
                    out[t, chain_cols] = np.clip(
                        out[t, chain_cols] + step,
                        rig.lower[chain_cols],
                        rig.upper[chain_cols],
                    )
            locked_reports.append({"foot": fname, "start": int(a), "end": int(b)})
    return out, locked_reports


def simulate_damped_spring_chains(
    rig,
    params,
    fps=24.0,
    frequency=4.0,
    damping=0.55,
    frame_times=None,
    spring_soft=0.0,
):
    """World-space constrained springs, four substeps per physical frame interval."""
    from scipy.linalg import expm

    times = _physical_frame_times(len(params), fps, frame_times)
    if (
        not np.isfinite(frequency)
        or frequency <= 0
        or not np.isfinite(damping)
        or damping < 0
    ):
        raise ValueError("Spring frequency must be positive and damping nonnegative")
    out = params.copy()
    tips = {}
    velocity = {}
    previous_heads = {}
    previous_targets = {}
    chain = [
        n
        for n in rig.order
        if n.startswith(("Tail", "Ear", "Flex"))
        and n in rig.slices
        and not isinstance(rig.slices[n], slice)
    ]
    chain_depth = {}
    chain_len = {}
    for n in chain:
        ancestors = []
        curr = n
        while curr in chain:
            ancestors.append(curr)
            curr = rig.parents.get(curr)
        chain_depth[n] = len(ancestors) - 1
        root_anc = ancestors[-1]
        chain_len[root_anc] = max(chain_len.get(root_anc, 1), len(ancestors))
    bone_omega = {}
    for n in chain:
        curr = n
        while rig.parents.get(curr) in chain:
            curr = rig.parents.get(curr)
        total_n = chain_len.get(curr, 1)
        idx_i = chain_depth[n]
        taper = 1.0 - float(spring_soft) * (idx_i / float(max(total_n - 1, 1)))
        bone_omega[n] = 2.0 * np.pi * frequency * max(0.2, taper)
    matrices = {}
    for t, p in enumerate(params):
        interval = 0.0 if t == 0 else times[t] - times[t - 1]
        original = rig.fk(p)[0]
        for name in chain:
            omega = bone_omega[name]
            transition = None
            if interval > 0:
                key = (float(interval), float(omega))
                if key not in matrices:
                    matrices[key] = expm(
                        np.array([[0.0, 1.0], [-omega * omega, -2 * damping * omega]])
                        * (interval / 4)
                    )
                transition = matrices[key]
            index = 2 * rig.index[name] + 1
            target = original[index]
            _, heads, _ = rig.fk(out[t])
            head = heads[name]
            length = np.linalg.norm(rig.t[name] - rig.h[name])
            if name not in tips:
                tips[name] = target.copy()
                velocity[name] = np.zeros(3)
                previous_heads[name] = head.copy()
                previous_targets[name] = target.copy()
            anchor_velocity = (
                (head - previous_heads[name]) / interval
                if interval > 0
                else np.zeros(3)
            )
            for substep in range(4):
                fraction = (substep + 1) / 4
                anchor = previous_heads[name] * (1 - fraction) + head * fraction
                goal = previous_targets[name] * (1 - fraction) + target * fraction
                if transition is not None:
                    displacement = tips[name] - goal
                    tips[name] = (
                        goal
                        + transition[0, 0] * displacement
                        + transition[0, 1] * velocity[name]
                    )
                    velocity[name] = (
                        transition[1, 0] * displacement
                        + transition[1, 1] * velocity[name]
                    )
                direction = tips[name] - anchor
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    direction = target - head
                    norm = np.linalg.norm(direction)
                direction = direction / max(norm, 1e-10)
                tips[name] = anchor + direction * length
                relative_velocity = velocity[name] - anchor_velocity
                velocity[name] -= direction * np.dot(relative_velocity, direction)
            out[t] = rig.align_bone_to_tip(name, tips[name], out[t])
            tips[name] = rig.fk(out[t])[0][index]
            previous_heads[name] = head.copy()
            previous_targets[name] = target.copy()
    return out


def keep_out(rig, params):
    """Post-cleanup pass pushing any penetrating free-limb or weapon guard points out of trunk capsules."""
    out = np.asarray(params, float).copy()
    if not getattr(rig, "capsules", ()):
        return out, 0
    free_cols = sorted(
        {
            int(c)
            for n, entry in rig.slices.items()
            if not isinstance(entry, slice)
            and rig.meta.get(n, {}).get("kind") not in ("root", "spine", "head")
            for c in entry[0]
        }
    )
    if not free_cols:
        return out, 0
    repaired_frames = 0
    for t in range(len(out)):
        for _ in range(4):
            X, J, _, _ = rig.fk(out[t], jac=True)
            res, G = rig.guard(X, J)
            active = np.flatnonzero(res > 1e-3)
            if not len(active):
                break
            Gc = G[active][:, free_cols]
            if not np.any(np.abs(Gc) > 1e-8):
                break
            A = np.vstack([Gc, np.eye(len(free_cols)) * 50.0])
            rhs = np.r_[-res[active], np.zeros(len(free_cols))]
            step, *_ = np.linalg.lstsq(A, rhs, rcond=None)
            out[t, free_cols] = np.clip(
                out[t, free_cols] + step,
                rig.lower[free_cols],
                rig.upper[free_cols],
            )
            repaired_frames += 1
    return out, repaired_frames


def clean_motion(
    rig,
    cameras,
    params,
    uv,
    confidence,
    fps=24.0,
    frame_times=None,
    skin=None,
    *,
    steps_steady=0.5,
    lock_blend=3,
    spring_soft=0.4,
    enable_lowpass=True,
    enable_lock_feet=True,
):
    """Full 8-step video2rig post-solve motion cleanup pipeline."""
    out = np.asarray(params, float).copy()
    if not len(out):
        return out, {"steps_applied": []}
    out = smooth_turn(rig, out, fps=fps, frame_times=frame_times)
    out = despike_reading_backed(rig, out, cameras, uv, confidence, despike_limbs=True)
    if enable_lowpass and len(out) >= 4:
        out = lowpass_motion_adaptive(rig, out, fps=fps, frame_times=frame_times)
    out = floor_bend_and_smooth_lift(rig, out, cameras=cameras, floor_bend="still")
    out, steps_report = recover_root_translation_from_steps(
        rig, out, fps=fps, frame_times=frame_times, steps_steady=steps_steady
    )
    locked_feet = []
    if enable_lock_feet and len(out) >= 3:
        out, locked_feet = lock_feet(
            rig,
            out,
            cameras=cameras,
            uv=uv,
            confidence=confidence,
            fps=fps,
            frame_times=frame_times,
            lock_blend=lock_blend,
        )
    out = simulate_damped_spring_chains(
        rig, out, fps=fps, frame_times=frame_times, spring_soft=spring_soft
    )
    out, keep_out_repairs = keep_out(rig, out)
    return out, {
        "steps_applied": [
            "smooth_turn",
            "despike_reading_backed",
            "lowpass_motion_adaptive",
            "floor_bend_and_smooth_lift",
            "recover_root_translation_from_steps",
            "lock_feet",
            "simulate_damped_spring_chains",
            "keep_out",
        ],
        "root_from_steps": steps_report,
        "locked_feet": locked_feet,
        "keep_out_repairs": keep_out_repairs,
    }


def at_limit(rig, params):
    warnings = []
    for n, s in rig.slices.items():
        if isinstance(s, slice) or rig.meta[n].get("hinge"):
            continue
        ix, _ = s
        if not len(ix):
            continue
        near = (
            np.minimum(params[:, ix] - rig.lower[ix], rig.upper[ix] - params[:, ix])
            <= np.deg2rad(2) + 1e-12
        )
        share = float(near.any(1).mean()) if len(params) else 0
        if share > 0.25:
            warnings.append({"bone": n, "share": share})
    return warnings


def learn_frame0_shadow_table(render_rgb, video_rgb, render_mask, bins=32):
    render = np.asarray(render_rgb)
    video = np.asarray(video_rgb)
    mask = np.asarray(render_mask, bool)
    if render.shape != video.shape or render.shape[:2] != mask.shape:
        raise ValueError("Frame-zero calibration shapes differ")
    # Color likelihoods in known background vs known creature, learned only from frame zero.
    key = np.clip(video.astype(int) * bins // 256, 0, bins - 1)
    keys = key[:, :, 0] * bins * bins + key[:, :, 1] * bins + key[:, :, 2]
    dark = (video.mean(-1) < render.mean(-1) * 0.92) & ~mask
    bg = np.bincount(keys[dark], minlength=bins**3).astype(float)
    fg = np.bincount(keys[mask], minlength=bins**3).astype(float)
    probability = bg / (bg + fg + 1)
    return {
        "bins": bins,
        "shadow_probability": probability,
        "background": np.median(render[~mask], axis=0)
        if (~mask).any()
        else np.array([255, 255, 255]),
    }


def strip_shadow(rgb, table):
    out = np.asarray(rgb).copy()
    bins = table["bins"]
    key = np.clip(out.astype(int) * bins // 256, 0, bins - 1)
    ids = key[:, :, 0] * bins * bins + key[:, :, 1] * bins + key[:, :, 2]
    out[table["shadow_probability"][ids] > 0.7] = table["background"]
    return out


def align_frame0_camera_chamfer(
    camera, points, target_mask, triangles=None, maxiter=80
):
    """Six camera extrinsic DOFs, bidirectional silhouette Chamfer plus rasterized IoU."""
    mask = np.asarray(target_mask, bool)
    H, W = mask.shape
    if (H, W) != (camera.height, camera.width) or not mask.any():
        raise ValueError("A nonempty calibrated silhouette is required")
    if max(H, W) > 256:
        import cv2

        ratio = 256 / max(H, W)
        width = max(1, round(W * ratio))
        height = max(1, round(H * ratio))
        small = replace(
            camera,
            width=width,
            height=height,
            focal_px=camera.focal_px * ratio,
            shift_px=tuple(np.asarray(camera.shift_px) * ratio),
        )
        fitted, report = align_frame0_camera_chamfer(
            small,
            points,
            cv2.resize(
                mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            ).astype(bool),
            triangles,
            maxiter,
        )
        return replace(
            fitted,
            width=W,
            height=H,
            focal_px=camera.focal_px,
            shift_px=camera.shift_px,
        ), report
    boundary = mask & ~ndimage.binary_erosion(mask)
    target = np.argwhere(boundary)[:, ::-1]
    if len(target) > 3000:
        target = target[:: max(1, len(target) // 3000)]
    tree = cKDTree(target)
    points = np.asarray(points)

    def make(q):
        E = exp_so3(q[:3])
        translation = np.zeros(3)
        translation[: len(q) - 3] = q[3:]
        t = E @ camera.t + translation
        if camera.type == "ortho":
            # Orthographic pixels contain no optical-depth translation information.
            # Keep that gauge at the supplied calibration instead of reporting recovery.
            t[2] = camera.t[2]
        return replace(camera, R=E @ camera.R, t=t)

    def measurement(q):
        c = make(q)
        uv = c.project(points)
        valid = c.depth(points) > 0 if c.type == "persp" else np.ones(len(points), bool)
        projected = uv[valid]
        if not len(projected):
            return 1e8, 0.0, float(max(H, W))
        canvas = np.zeros((H, W), bool)
        pix = np.rint(projected).astype(int)
        good = ((pix >= 0) & (pix < [W, H])).all(-1)
        pix = pix[good]
        canvas[pix[:, 1], pix[:, 0]] = True
        if triangles is not None:
            import cv2

            filled = np.zeros((H, W), np.uint8)
            for face in np.asarray(triangles, int):
                if valid[face].all():
                    cv2.fillConvexPoly(
                        filled,
                        np.rint(np.clip(uv[face], -1e6, 1e6)).astype(np.int32),
                        1,
                    )
            canvas = filled.astype(bool)
        else:
            canvas = ndimage.binary_fill_holes(
                ndimage.binary_closing(canvas, iterations=2)
            )
        edge = canvas & ~ndimage.binary_erosion(canvas)
        source = np.argwhere(edge)[:, ::-1]
        if not len(source):
            return 1e8, 0.0, float(max(H, W))
        d1 = tree.query(source)[0].mean()
        d2 = cKDTree(source).query(target)[0].mean()
        iou = np.count_nonzero(canvas & mask) / max(1, np.count_nonzero(canvas | mask))
        loss = (d1 + d2) / max(H, W) + 0.5 * (1 - iou) + 1e-6 * np.dot(q, q)
        return float(loss), float(iou), float((d1 + d2) / 2)

    zero = np.zeros(5 if camera.type == "ortho" else 6)
    initial, before_iou, before_chamfer = measurement(zero)
    result = minimize(
        lambda q: measurement(q)[0],
        zero,
        method="Powell",
        options={"maxiter": maxiter, "xtol": 1e-4},
    )
    accepted = result.fun < initial
    _, after_iou, after_chamfer = measurement(result.x if accepted else zero)
    return (make(result.x) if accepted else camera), {
        "before": initial,
        "after": min(initial, float(result.fun)),
        "converged": bool(result.success),
        "accepted": bool(accepted),
        "silhouette_iou_before": before_iou,
        "silhouette_iou_after": after_iou,
        "chamfer_px_before": before_chamfer,
        "chamfer_px_after": after_chamfer,
        "analysis_resolution": [W, H],
        "optimized_parameters": ["rx", "ry", "rz", "tx", "ty"]
        + (["tz"] if camera.type == "persp" else []),
        "unobservable_parameters": ["tz"] if camera.type == "ortho" else [],
        "rasterized_faces": len(triangles) if triangles is not None else 0,
    }


def cadence_from_flow_steps(speeds, fps=24.0):
    """Infer periodic missing content frames, without turning real acceleration into time.

    A local speed trend removes gradual acceleration only for cadence classification.
    Output intervals remain discrete content-frame intervals; arbitrary speed ratios are
    never integrated as timestamps. Low or zero motion is not evidence of a missing
    time interval: a held pose still occupies coded frames. Times use the caller's
    scene-frame units.
    """
    speeds = np.asarray(speeds, float)
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(speeds).all():
        raise ValueError("Cadence needs finite flow speeds and a positive FPS")
    trend = np.array(
        [np.median(speeds[max(0, i - 3) : i + 4]) for i in range(len(speeds))]
    )
    relative = speeds / np.maximum(trend, 1e-3)
    moving = trend > 0.05
    period, phase, content_fps = None, None, float(fps)
    scores = {}
    for candidate, rate in ((4, 30.0), (24, 25.0)):
        groups = []
        for remainder in range(candidate):
            selected = moving & (np.arange(len(speeds)) % candidate == remainder)
            groups.append(
                float(np.median(relative[selected])) if selected.sum() >= 3 else 1.0
            )
        scores[str(candidate)] = groups
        winner = int(np.argmax(groups))
        high = moving & (np.arange(len(speeds)) % candidate == winner)
        # Repeated evidence is necessary: a single real strike is not a dropped-frame cadence.
        reliable = high.sum() >= 3 and np.mean(relative[high] > 1.45) >= 0.75
        others = np.delete(groups, winner)
        if reliable and groups[winner] > 1.5 and np.max(others) < 1.2:
            period, phase, content_fps = candidate, winner, rate
            break
    advances = np.ones(len(speeds)) * (fps / content_fps)
    if period:
        advances[np.arange(len(speeds)) % period == phase] *= 2
    return {
        "frame_times": np.r_[0.0, np.cumsum(advances)].tolist(),
        "cadence": float(advances.mean()) if len(advances) else 1.0,
        "flow_speed": speeds.tolist(),
        "relative_flow": relative.tolist(),
        "period": period,
        "phase": phase,
        "content_fps": content_fps,
        "scene_fps": float(fps),
        "relative_by_period": scores,
    }


def detect_video_cadence_optical_flow(frames, fps=24.0, masks=None):
    import cv2

    images = [np.asarray(im) for im in frames]
    if len(images) < 2:
        return (
            cadence_from_flow_steps([], fps)
            if images
            else {"frame_times": [], "cadence": 1.0}
        )
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    speeds = []

    def gray(im):
        g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if im.ndim == 3 else im
        return cv2.resize(g, (320, 192))

    previous = gray(images[0])
    for index, im in enumerate(images[1:]):
        current = gray(im)
        field = flow.calc(previous, current, None)
        if masks is not None:
            subject = (
                cv2.resize(
                    (np.asarray(masks[index]) | np.asarray(masks[index + 1])).astype(
                        np.uint8
                    ),
                    (320, 192),
                )
                > 0
            )
        else:
            # Ignore the empty studio, whose zero flow can swamp small hands/feet.
            bg = np.median(np.r_[previous[0], previous[-1], current[0], current[-1]])
            subject = (abs(previous.astype(float) - bg) > 12) | (
                abs(current.astype(float) - bg) > 12
            )
        speed = np.linalg.norm(field, axis=-1)
        # A stationary trunk can occupy most subject pixels while a hand or foot
        # moves. Use the subject's mean displacement, as the upstream cadence
        # detector does, so that minority motion contributes to the cadence signal.
        speeds.append(float(speed[subject].mean()) if subject.any() else 0.0)
        previous = current
    return cadence_from_flow_steps(speeds, fps)
