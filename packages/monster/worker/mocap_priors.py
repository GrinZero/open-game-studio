#!/usr/bin/env python3
"""mocap_priors.py — XSXB-Mocap inspired multi-view calibration, robust motion priors,
contact-aware foot locking, windowed peak DSNT, and synthetic data physical gates.

All mechanisms in this module are controlled by the `MONSTER_ENABLE_MOCAP_PRIORS` feature
flag (enabled by default: `"1"`), and are orthogonal to `MONSTER_ENABLE_DENSE_SURFACE`.
No top-level `torch` or `bpy` imports so this module is usable across CPU evals, Blender
workers, and GPU training runtimes.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any, Callable

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

MOCAP_PRIORS_ENV = "MONSTER_ENABLE_MOCAP_PRIORS"
MOCAP_PRIORS_SCHEMA = "monster-mocap-priors-v1"


def mocap_priors_enabled() -> bool:
    """Returns True when `MONSTER_ENABLE_MOCAP_PRIORS` is enabled (default: `'1'`)."""
    value = os.environ.get(MOCAP_PRIORS_ENV, "1").strip().lower()
    if value not in {"0", "1", "true", "false", "on", "off"}:
        raise ValueError(f"Invalid {MOCAP_PRIORS_ENV}: {value!r}")
    return value in {"1", "true", "on"}


# ==============================================================================
# 1. Per-Frame View Scale & Vertical Drift Calibration (`la[t]`, `dy[t]`)
# ==============================================================================

def _build_temporal_diff_gramians(T: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns `(D1^T D1, D2^T D2)` first- and second-difference Gramian matrices of shape `(T, T)`."""
    eye = np.eye(T, dtype=np.float64)
    if T >= 2:
        d1 = eye[1:] - eye[:-1]
        g1 = d1.T @ d1
    else:
        g1 = np.zeros((T, T), dtype=np.float64)
    if T >= 3:
        d2 = eye[2:] - 2.0 * eye[1:-1] + eye[:-2]
        g2 = d2.T @ d2
    else:
        g2 = np.zeros((T, T), dtype=np.float64)
    return g1, g2


def _solve_panel_scale_and_dy_sequence(
    uv_src: np.ndarray,
    uv_ref: np.ndarray,
    conf_src: np.ndarray,
    conf_ref: np.ndarray,
    panel_w: float,
    panel_h: float,
    scale_2k: float,
    vis_src: np.ndarray | None = None,
    vis_ref: np.ndarray | None = None,
    max_zoom_log: float = 0.077,  # ~ln(1.08), +/-8% max zoom
    max_dy_ratio: float = 0.06,
    deadband_dy_px: float | None = None,
    deadband_la: float = 0.0035,
    irls_iters: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Fits smooth per-frame log-scale `la[t]` and vertical shift `dy[t]` mapping `uv_src` -> `uv_ref`
    vertical coordinates:
        v_src_cal[t, k] = exp(la[t]) * (v_src[t, k] - v0) + v0 + dy[t]
        u_src_cal[t, k] = exp(la[t]) * (u_src[t, k] - u0) + u0
    Uses confidence/visibility-weighted Huber IRLS + Whittaker temporal smoothing + an affine
    epipolar consensus gate so pose-dependent occlusion noise on already-calibrated cameras never
    triggers false camera drift.
    """
    src = np.asarray(uv_src, dtype=np.float64)
    ref = np.asarray(uv_ref, dtype=np.float64)
    c_s = np.asarray(conf_src, dtype=np.float64)
    c_r = np.asarray(conf_ref, dtype=np.float64)
    T, K, _ = src.shape

    u0 = 0.5 * float(panel_w)
    v0 = 0.5 * float(panel_h)
    ys = src[..., 1] - v0
    yr = ref[..., 1] - v0

    # Base confidence + visibility weights: exclude occluded keypoints whose discrepancy is pose error
    pair_conf = np.clip(c_s, 0.0, 1.0) * np.clip(c_r, 0.0, 1.0)
    valid_mask = (c_s >= 0.35) & (c_r >= 0.35)
    if vis_src is not None and vis_ref is not None:
        v_s = np.clip(np.asarray(vis_src, dtype=np.float64), 0.0, 1.0)
        v_r = np.clip(np.asarray(vis_ref, dtype=np.float64), 0.0, 1.0)
        pair_conf = pair_conf * v_s * v_r
        valid_mask = valid_mask & (v_s >= 0.55) & (v_r >= 0.55)
    if not np.any(valid_mask):
        valid_mask = (c_s >= 0.25) & (c_r >= 0.25)
    w_base = (pair_conf ** 2) * valid_mask.astype(np.float64)

    g1, g2 = _build_temporal_diff_gramians(T)
    h2 = float(panel_h) ** 2
    # Gentle zero-ridge (numerical stability only) + strong temporal smoothness (camera drift is low-frequency)
    reg_la = (0.015 * h2) * np.eye(T, dtype=np.float64) + (6.0 * h2) * g1 + (18.0 * h2) * g2
    reg_dy = 0.02 * np.eye(T, dtype=np.float64) + 25.0 * g1 + 60.0 * g2

    la = np.zeros(T, dtype=np.float64)
    dy = np.zeros(T, dtype=np.float64)
    huber_base = 12.0 * float(scale_2k)

    for it in range(max(1, int(irls_iters))):
        res = np.exp(la[:, None]) * ys + dy[:, None] - yr
        abs_res = np.abs(res)
        huber_delta = (3.5 * huber_base) if it == 0 else huber_base
        w_huber = np.where(abs_res > huber_delta, huber_delta / np.maximum(abs_res, 1e-9), 1.0)
        w = w_base * w_huber

        exp_la = np.exp(la)
        j_la = exp_la[:, None] * ys
        w_j2 = np.sum(w * (j_la ** 2), axis=1)
        w_j1 = np.sum(w * j_la, axis=1)
        w_1 = np.sum(w, axis=1)

        target_lin = yr - (exp_la[:, None] * ys - j_la * la[:, None])
        rhs_la = np.sum(w * j_la * target_lin, axis=1)
        rhs_dy = np.sum(w * target_lin, axis=1)

        H_ll = np.diag(w_j2) + reg_la
        H_ld = np.diag(w_j1)
        H_dd = np.diag(w_1) + reg_dy

        H = np.block([[H_ll, H_ld], [H_ld, H_dd]])
        rhs = np.concatenate([rhs_la, rhs_dy], axis=0)
        try:
            sol = np.linalg.solve(H, rhs)
            la = np.clip(sol[:T], -float(max_zoom_log), float(max_zoom_log))
            max_dy_px = float(max_dy_ratio) * float(panel_h)
            dy = np.clip(sol[T:], -max_dy_px, max_dy_px)
        except np.linalg.LinAlgError:
            break

    db_dy = float(deadband_dy_px) if deadband_dy_px is not None else (1.5 * float(scale_2k))
    db_la = float(deadband_la)

    rms_dy = float(np.sqrt(np.mean(dy ** 2)))
    rms_la = float(np.sqrt(np.mean(la ** 2)))
    if rms_dy <= db_dy and rms_la <= db_la:
        return src.copy(), np.zeros(T, dtype=np.float64), np.zeros(T, dtype=np.float64), False

    # Epipolar affine consensus gate: verify that (la, dy) genuinely explains a coherent panel-wide
    # camera shift/zoom rather than fitting to scattered pose-occlusion outliers.
    res_before = np.abs(ys - yr)
    res_after = np.abs(np.exp(la[:, None]) * ys + dy[:, None] - yr)
    eval_mask = valid_mask if np.any(valid_mask) else np.ones_like(valid_mask, dtype=bool)
    med_before = float(np.median(res_before[eval_mask]))
    med_after = float(np.median(res_after[eval_mask]))
    if med_after > 5.0 * float(scale_2k) or med_after > 0.60 * max(med_before, 1e-6):
        return src.copy(), np.zeros(T, dtype=np.float64), np.zeros(T, dtype=np.float64), False

    la_eff = la if rms_la > db_la else np.zeros_like(la)
    dy_eff = dy if rms_dy > db_dy else np.zeros_like(dy)
    scales = np.exp(la_eff)

    out = src.copy()
    out[..., 0] = scales[:, None] * (src[..., 0] - u0) + u0
    out[..., 1] = scales[:, None] * (src[..., 1] - v0) + v0 + dy_eff[:, None]
    return out, la_eff, dy_eff, True


def calibrate_view_scale_drift(
    uv_front: np.ndarray,
    uv_side: np.ndarray,
    conf_front: np.ndarray,
    conf_side: np.ndarray,
    calib: dict[str, Any],
    uv_right: np.ndarray | None = None,
    conf_right: np.ndarray | None = None,
    vis_front: np.ndarray | None = None,
    vis_side: np.ndarray | None = None,
    vis_right: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Estimates and compensates per-frame inter-panel scale breathing (`la[t]`) and vertical
    baseline drift (`dy[t]`) before epipolar V-consistency gating and 3D triangulation.

    Returns `(uv_front_calibrated, uv_right_calibrated, diagnostics)`.
    """
    uv_f = np.asarray(uv_front, dtype=np.float64)
    uv_r = np.asarray(uv_right, dtype=np.float64) if uv_right is not None else None
    if not mocap_priors_enabled() or uv_f.ndim != 3 or uv_f.shape[0] < 1:
        return uv_f.copy(), (uv_r.copy() if uv_r is not None else None), {
            "view_drift_calibrated": False,
            "max_abs_dy_px": 0.0,
            "mean_abs_dy_px": 0.0,
            "max_abs_scale_pct": 0.0,
        }

    panel_h = float(calib.get("panel_height", 1440.0))
    panel_w = float(calib.get("panel_width", panel_h * (672.0 / 768.0)))
    scale_2k = panel_h / 768.0

    uv_f_cal, la_f, dy_f, applied_f = _solve_panel_scale_and_dy_sequence(
        uv_src=uv_f,
        uv_ref=uv_side,
        conf_src=conf_front,
        conf_ref=conf_side,
        panel_w=panel_w,
        panel_h=panel_h,
        scale_2k=scale_2k,
        vis_src=vis_front,
        vis_ref=vis_side,
    )

    applied_r = False
    la_r = np.zeros_like(la_f)
    dy_r = np.zeros_like(dy_f)
    uv_r_cal = None
    if uv_r is not None and conf_right is not None:
        uv_r_cal, la_r, dy_r, applied_r = _solve_panel_scale_and_dy_sequence(
            uv_src=uv_r,
            uv_ref=uv_f_cal,
            conf_src=conf_right,
            conf_ref=conf_front,
            panel_w=panel_w,
            panel_h=panel_h,
            scale_2k=scale_2k,
            vis_src=vis_right,
            vis_ref=vis_front,
        )

    max_dy = max(float(np.max(np.abs(dy_f))), float(np.max(np.abs(dy_r)))) if len(dy_f) else 0.0
    mean_dy = float(np.mean(np.abs(dy_f))) if len(dy_f) else 0.0
    max_scale_pct = (math.exp(max(float(np.max(np.abs(la_f))), float(np.max(np.abs(la_r))))) - 1.0) * 100.0 if len(la_f) else 0.0

    diag = {
        "view_drift_calibrated": bool(applied_f or applied_r),
        "max_abs_dy_px": round(max_dy, 4),
        "mean_abs_dy_px": round(mean_dy, 4),
        "max_abs_scale_pct": round(max_scale_pct, 4),
    }
    return uv_f_cal, uv_r_cal, diag


# ==============================================================================
# 2. Local Windowed Peak Soft-Argmax (Multimodal Occlusion Peak Protection)
# ==============================================================================

def windowed_peak_dsnt_numpy(
    prob: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    uv_global: np.ndarray,
    window_radius: int = 3,
    min_peak_mass: float = 0.22,
    pull_deadband_px: float = 2.5,
) -> np.ndarray:
    """Refines global DSNT 2D coordinates `uv_global` using a `(2*R+1) x (2*R+1)` local window
    around each heatmap's discrete peak `(i*, j*)`.

    When a secondary multimodal peak (e.g. crossed left/right limbs or background shadow) pulls
    `uv_global` away from the primary peak by more than `pull_deadband_px` while the primary peak
    window holds at least `min_peak_mass` probability, blends smoothly toward the local windowed
    sub-pixel expectation so predictions never float in empty space between two peaks.
    """
    p = np.asarray(prob, dtype=np.float64)
    gx = np.asarray(grid_x, dtype=np.float64).reshape(-1)
    gy = np.asarray(grid_y, dtype=np.float64).reshape(-1)
    uv_g = np.asarray(uv_global, dtype=np.float64)

    H, W = p.shape[-2], p.shape[-1]
    lead_shape = p.shape[:-2]
    N = int(np.prod(lead_shape)) if lead_shape else 1
    p_flat = p.reshape(N, H, W)
    uv_flat = uv_g.reshape(N, 2).copy()

    flat_idx = np.argmax(p_flat.reshape(N, H * W), axis=-1)
    iy_peak = flat_idx // W
    ix_peak = flat_idx % W

    rows = np.arange(H, dtype=np.int64)[None, :, None]
    cols = np.arange(W, dtype=np.int64)[None, None, :]
    R = int(max(1, window_radius))
    win_mask = (np.abs(rows - iy_peak[:, None, None]) <= R) & (np.abs(cols - ix_peak[:, None, None]) <= R)

    p_win = p_flat * win_mask
    mass_win = np.sum(p_win, axis=(-2, -1))
    norm_win = p_win / np.maximum(mass_win[:, None, None], 1e-12)

    u_loc = np.sum(norm_win * gx[None, None, :], axis=(-2, -1))
    v_loc = np.sum(norm_win * gy[None, :, None], axis=(-2, -1))
    uv_loc = np.stack([u_loc, v_loc], axis=-1)

    pull_dist = np.linalg.norm(uv_loc - uv_flat, axis=-1)
    # Gate activates when primary peak has sufficient local mass AND global mean is pulled away
    mass_gate = np.clip((mass_win - float(min_peak_mass)) / max(1e-6, 0.55 - float(min_peak_mass)), 0.0, 1.0)
    dist_gate = np.clip((pull_dist - float(pull_deadband_px)) / max(1e-6, 6.0), 0.0, 1.0)
    gate = (mass_gate * dist_gate)[:, None]

    out = uv_flat + gate * (uv_loc - uv_flat)
    return out.reshape(uv_g.shape)


def windowed_peak_dsnt_torch(
    prob: Any,
    grid_x: Any,
    grid_y: Any,
    uv_global: Any,
    window_radius: int = 3,
    min_peak_mass: float = 0.22,
    pull_deadband_px: float = 2.5,
) -> Any:
    """PyTorch straight-through local windowed peak soft-argmax refinement.
    Preserves exact global DSNT gradients during backpropagation while eliminating multimodal
    secondary-peak pull in forward predictions and inference.
    """
    import torch

    B, K, H, W = prob.shape
    R = int(max(1, window_radius))
    with torch.no_grad():
        flat_idx = prob.view(B, K, H * W).argmax(dim=-1)
        iy_peak = (flat_idx // W).unsqueeze(-1).unsqueeze(-1)
        ix_peak = (flat_idx % W).unsqueeze(-1).unsqueeze(-1)
        rows = torch.arange(H, device=prob.device).view(1, 1, H, 1)
        cols = torch.arange(W, device=prob.device).view(1, 1, 1, W)
        win_mask = ((rows - iy_peak).abs() <= R) & ((cols - ix_peak).abs() <= R)
        p_win = prob * win_mask.to(prob.dtype)
        mass_win = p_win.sum(dim=(-2, -1))
        norm_win = p_win / mass_win.unsqueeze(-1).unsqueeze(-1).clamp_min(1e-12)
        u_loc = (norm_win * grid_x).sum(dim=(-2, -1))
        v_loc = (norm_win * grid_y).sum(dim=(-2, -1))
        uv_loc = torch.stack([u_loc, v_loc], dim=-1)
        pull_dist = torch.linalg.norm(uv_loc - uv_global, dim=-1)
        mass_gate = ((mass_win - float(min_peak_mass)) / max(1e-6, 0.55 - float(min_peak_mass))).clamp(0.0, 1.0)
        dist_gate = ((pull_dist - float(pull_deadband_px)) / 6.0).clamp(0.0, 1.0)
        gate = (mass_gate * dist_gate).unsqueeze(-1)
        delta_loc = gate * (uv_loc - uv_global)

    return uv_global + delta_loc


# ==============================================================================
# 3. IRLS Charbonnier Acceleration Smoother & Velocity-Gated De-Jitter Filter
# ==============================================================================

def whittaker_charbonnier_smooth_trajectory(
    y: np.ndarray,
    weights: np.ndarray | None = None,
    lam_accel: float = 4.0,
    lam_jerk: float = 10.0,
    irls_iters: int = 3,
    charbonnier_sigma: float = 3.6,
) -> np.ndarray:
    """Zero-phase confidence-weighted Whittaker-Henderson + Minimum-Jerk smoother with
    IRLS Charbonnier edge-preserving acceleration weighting (inspired by `XSXB-Mocap` `src/fit.py`).

    When instantaneous acceleration `||D_2 x_t||` is within normal motion/noise bounds
    (`<= charbonnier_sigma * MAD`), applies full quadratic acceleration + jerk smoothing.
    When a rapid strike, gallop turn, or sharp inflection produces an acceleration spike
    exceeding `charbonnier_sigma * MAD`, relaxes the local second/third-difference penalty via
    Charbonnier weight `1 / sqrt(1 + (||a_t|| / delta)^2)` so motion peaks are not rounded off.
    """
    arr = np.asarray(y, dtype=np.float64)
    T = int(arr.shape[0])
    if T < 7 or (lam_accel <= 0.0 and lam_jerk <= 0.0):
        return arr.copy()

    eye = np.eye(T, dtype=np.float64)
    d2 = eye[2:] - 2.0 * eye[1:-1] + eye[:-2]      # (T-2, T)
    d3 = eye[3:] - 3.0 * eye[2:-1] + 3.0 * eye[1:-2] - eye[:-3]  # (T-3, T)
    g2_base = d2.T @ d2
    g3_base = d3.T @ d3
    base_penalty = float(lam_accel) * g2_base + float(lam_jerk) * g3_base

    def _solve_2d_sequence(seq: np.ndarray, w_vec: np.ndarray) -> np.ndarray:
        W_diag = np.diag(w_vec)
        wy = w_vec[:, None] * seq
        x_quad = np.linalg.solve(W_diag + base_penalty, wy)
        if irls_iters <= 1:
            return x_quad

        # Estimate acceleration distribution on a lightly regularized pass so sharp peaks are not pre-flattened
        x_light = np.linalg.solve(W_diag + 0.12 * base_penalty, wy)
        a_light = np.linalg.norm(d2 @ x_light, axis=-1)
        med_a = float(np.median(a_light))
        mad_a = float(np.median(np.abs(a_light - med_a))) * 1.4826
        delta_a = max(1e-6, med_a + float(charbonnier_sigma) * max(mad_a, 0.25 * med_a))
        if float(np.max(a_light)) <= delta_a:
            # No sharp high-acceleration inflection peaks -> exact quadratic solution
            return x_quad

        x_curr = x_light
        delta_j = 1.6 * delta_a
        for _ in range(int(irls_iters) - 1):
            a_curr = np.linalg.norm(d2 @ x_curr, axis=-1)
            excess_a = np.maximum(0.0, a_curr - 0.65 * delta_a) / max(0.5 * delta_a, 1e-9)
            w_a = 1.0 / (1.0 + excess_a ** 2)

            j_curr = np.linalg.norm(d3 @ x_curr, axis=-1)
            excess_j = np.maximum(0.0, j_curr - 0.65 * delta_j) / max(0.5 * delta_j, 1e-9)
            w_j = 1.0 / (1.0 + excess_j ** 2)

            pen_irls = float(lam_accel) * (d2.T @ (w_a[:, None] * d2)) + float(lam_jerk) * (d3.T @ (w_j[:, None] * d3))
            x_curr = np.linalg.solve(W_diag + pen_irls, wy)
        return x_curr

    if arr.ndim == 2:
        w = np.ones(T, dtype=np.float64) if weights is None else np.clip(np.asarray(weights, dtype=np.float64).reshape(T), 0.05, 1.0)
        return _solve_2d_sequence(arr, w)

    if arr.ndim == 3:
        _, K, D = arr.shape
        out = np.empty_like(arr, dtype=np.float64)
        w_all = np.ones((T, K), dtype=np.float64) if weights is None else np.clip(np.asarray(weights, dtype=np.float64), 0.05, 1.0)
        for k in range(K):
            out[:, k, :] = _solve_2d_sequence(arr[:, k, :], w_all[:, k])
        return out

    return arr.copy()


def dejitter_velocity_gated(
    x: np.ndarray,
    v_fast: float | None = None,
    passes: int = 2,
    strength: float = 0.55,
) -> np.ndarray:
    """Velocity-gated micro-jitter filter (`XSXB-Mocap` `src/bl_apply.py` lines 65-115):
        lp(x)_t = 0.25 x_{t-1} + 0.50 x_t + 0.25 x_{t+1}
        x_out = x + strength * (lp(x) - x) / (1 + (v / v_fast)^2)
    Suppresses stationary micro-tremor when instantaneous velocity `v` is small while passing
    fast motion peaks (`v >> v_fast`) with zero attenuation.
    """
    arr = np.asarray(x, dtype=np.float64)
    T = int(arr.shape[0])
    if T < 5 or strength <= 0.0:
        return arr.copy()

    lp = arr.copy()
    for _ in range(max(1, int(passes))):
        nxt = lp.copy()
        nxt[1:-1] = 0.25 * lp[:-2] + 0.50 * lp[1:-1] + 0.25 * lp[2:]
        lp = nxt

    step_diff = np.linalg.norm(arr[1:] - arr[:-1], axis=-1, keepdims=True)
    speed_prev = np.concatenate([step_diff[:1], step_diff], axis=0)
    speed_next = np.concatenate([step_diff, step_diff[-1:]], axis=0)
    speed = np.maximum(speed_prev, speed_next)

    if v_fast is None or v_fast <= 0.0:
        # Adaptive velocity threshold from median speed
        med_speed = np.median(speed, axis=0, keepdims=True)
        v_thresh = np.maximum(1e-5, 1.35 * med_speed)
    else:
        v_thresh = float(v_fast)

    gate = float(strength) / (1.0 + (speed / v_thresh) ** 2)
    return arr + gate * (lp - arr)


# ==============================================================================
# 4. Floor Penetration Clamp & Contact-Aware Foot Anti-Skate Locking
# ==============================================================================

def _detect_contact_intervals(mask: np.ndarray, min_len: int = 3) -> list[tuple[int, int]]:
    """Returns contiguous `[start, end]` inclusive frame intervals of length `>= min_len` where `mask` is True."""
    intervals: list[tuple[int, int]] = []
    T = int(len(mask))
    i = 0
    while i < T:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < T and mask[j + 1]:
            j += 1
        if (j - i + 1) >= int(min_len):
            intervals.append((i, j))
        i = j + 1
    return intervals


def apply_floor_and_foot_contact_constraints(
    frames_out: list[dict[str, Any]],
    order: list[str],
    root: str,
    classified_bones: dict[str, dict[str, Any]],
    rest_H: dict[str, np.ndarray],
    rest_T: dict[str, np.ndarray],
    rest_R: dict[str, np.ndarray],
    effective_base_limits: dict[str, list[list[float]]],
    frame_macro_collapse: np.ndarray,
    clamp_local_delta_euler_fn: Callable[[np.ndarray, list[list[float]]], np.ndarray],
    minimum_swing_matrix_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict[str, Any]:
    """Enforces floor non-penetration (`z >= z_floor - eps`) and contact-aware foot anti-skate
    locking (`XSXB-Mocap` `src/fit.py` & `src/bl_apply.py`) while maintaining 100% exact forward
    kinematics (`tail = head + R_world[:, 1] * rest_len`) and zero joint-limit violations.
    """
    T = len(frames_out)
    if T == 0 or not mocap_priors_enabled():
        return {
            "floor_clamped_frames": 0,
            "foot_contact_locked_frames": 0,
            "max_floor_penetration_fixed": 0.0,
            "contact_bones": [],
        }

    surface_config = classified_bones[root].get("surface_contact")
    from pose_features import surface_contact_enabled

    if surface_contact_enabled() and isinstance(surface_config, dict) and surface_config.get("enabled", False):
        from surface_contact import apply_surface_floor_constraints
        return apply_surface_floor_constraints(
            frames_out, order, root, classified_bones, rest_H, rest_T, rest_R,
            effective_base_limits, surface_config,
        )

    all_rest_z = [float(rest_H[b][2]) for b in order] + [float(rest_T[b][2]) for b in order]
    z_floor = min(all_rest_z)
    z_top = max(all_rest_z)
    rig_h = max(1e-3, z_top - z_floor)

    # Identify ground-contact foot/paw candidate bones whose rest-pose tail lies in bottom 12% of rig height
    contact_bones: list[str] = []
    for b in order:
        info = classified_bones[b]
        kind = str(info.get("kind", "limb"))
        if kind not in ("limb", "digit"):
            continue
        if float(rest_T[b][2]) <= z_floor + 0.12 * rig_h:
            contact_bones.append(b)

    # 1. Detect stance contact intervals on contact_bones where foot is near floor and nearly stationary
    foot_locked_frames_set: set[int] = set()
    locked_tail_targets: dict[tuple[int, str], tuple[np.ndarray, float]] = {}

    if contact_bones and T >= 4:
        for b in contact_bones:
            tails_seq = np.array([frames_out[f]["bones"][b]["tail_3d"] for f in range(T)], dtype=np.float64)
            confs_seq = np.array([float(frames_out[f]["bones"][b].get("confidence", 0.5)) for f in range(T)], dtype=np.float64)
            vel = np.zeros(T, dtype=np.float64)
            vel[1:] = np.linalg.norm(tails_seq[1:] - tails_seq[:-1], axis=-1)
            vel[0] = vel[1] if T > 1 else 0.0

            near_floor = (tails_seq[:, 2] >= z_floor - 0.03 * rig_h) & (tails_seq[:, 2] <= z_floor + 0.045 * rig_h)
            low_speed = vel <= 0.012 * rig_h
            # Only lock when there is actual micro-skating near the ground and no macro collapse
            contact_mask = near_floor & low_speed & (frame_macro_collapse <= 0.15)
            intervals = _detect_contact_intervals(contact_mask, min_len=3)
            for s_idx, e_idx in intervals:
                w_seg = np.clip(confs_seq[s_idx : e_idx + 1], 0.1, 1.0)
                anchor_pos = np.sum(tails_seq[s_idx : e_idx + 1] * w_seg[:, None], axis=0) / float(np.sum(w_seg))
                anchor_pos[2] = max(float(anchor_pos[2]), z_floor)
                seg_len = e_idx - s_idx + 1
                for f in range(s_idx, e_idx + 1):
                    # Smooth ease-in/out weight at interval boundaries
                    dist_edge = min(f - s_idx, e_idx - f)
                    blend_w = 0.45 if (seg_len >= 4 and dist_edge == 0) else 0.75
                    locked_tail_targets[(f, b)] = (anchor_pos.copy(), blend_w)
                    foot_locked_frames_set.add(f)

    # 2. Apply contact foot anti-skate locking and floor non-penetration clamp per frame
    floor_clamped_frames = 0
    max_penetration_fixed = 0.0
    floor_tol = 0.015 * rig_h

    for f in range(T):
        bmap_f = frames_out[f]["bones"]
        m_col = float(frame_macro_collapse[f])

        # Check if any contact bone in frame f has a locked stance target
        frame_locked_bones = [b for b in contact_bones if (f, b) in locked_tail_targets]
        if frame_locked_bones:
            posed_R: dict[str, np.ndarray] = {}
            heads: dict[str, np.ndarray] = {}
            root_head = np.asarray(bmap_f[root]["head_3d"], dtype=np.float64)

            for n in order:
                info = classified_bones[n]
                p = info.get("parent")
                k = str(info.get("kind", "limb"))
                lims = copy.deepcopy(effective_base_limits[n])
                has_custom_lims = str(info.get("limits_source", "")) in ("agent_custom", "probe_initial")
                if m_col > 0.0 and not has_custom_lims:
                    if n == root:
                        pitch_exp = m_col * math.radians(28.0)
                        lims[0] = [min(float(lims[0][0]), -pitch_exp), max(float(lims[0][1]), pitch_exp)]
                    elif k in ("spine", "head") and info.get("side") == "C":
                        pitch_exp = m_col * math.radians(35.0)
                        lims[0] = [min(float(lims[0][0]), -pitch_exp), max(float(lims[0][1]), pitch_exp)]

                R_w_curr = np.asarray(bmap_f[n]["rot_mat_world"], dtype=np.float64)
                if p and p in posed_R:
                    init_R = posed_R[p] @ (rest_R[p].T @ rest_R[n])
                    head = heads[p] + posed_R[p] @ (rest_R[p].T @ (rest_H[n] - rest_H[p]))
                else:
                    init_R = rest_R[n]
                    head = root_head

                if k == "rigid" or bool(info.get("fixed_local", False)):
                    delta = np.eye(3, dtype=np.float64)
                else:
                    delta = init_R.T @ R_w_curr
                    if (f, n) in locked_tail_targets:
                        target_tail, blend_w = locked_tail_targets[(f, n)]
                        rest_len = float(info.get("rest_length", np.linalg.norm(rest_T[n] - rest_H[n])))
                        curr_tail = head + (init_R @ delta)[:, 1] * rest_len
                        blended_tail = (1.0 - blend_w) * curr_tail + blend_w * target_tail
                        v_des = blended_tail - head
                        v_norm = float(np.linalg.norm(v_des))
                        if v_norm > 1e-6:
                            R_sw = minimum_swing_matrix_fn((init_R @ delta)[:, 1], v_des / v_norm)
                            delta = init_R.T @ (R_sw @ (init_R @ delta))
                    delta = clamp_local_delta_euler_fn(delta, lims)

                R_world = init_R @ delta
                posed_R[n] = R_world
                heads[n] = head
                rest_len = float(info.get("rest_length", np.linalg.norm(rest_T[n] - rest_H[n])))
                tail = head + R_world[:, 1] * rest_len
                bmap_f[n]["head_3d"] = [round(float(x), 5) for x in head]
                bmap_f[n]["tail_3d"] = [round(float(x), 5) for x in tail]
                bmap_f[n]["rot_mat_world"] = [[round(float(x), 5) for x in row] for row in R_world]

        # Global floor non-penetration clamp on ground-contact limb/foot bones:
        # Pure vertical root translation preserves all local angles & limits, bounded by actual root drop
        # so local limb extension on a stationary torso cannot lift the entire rig above its observed height.
        if contact_bones:
            min_z_frame = min(
                min(float(bmap_f[n]["head_3d"][2]), float(bmap_f[n]["tail_3d"][2]))
                for n in contact_bones
            )
            penetration = (z_floor - floor_tol) - min_z_frame
            root_drop = max(0.0, float(rest_H[root][2]) - float(bmap_f[root]["head_3d"][2]))
            max_lift = root_drop if root_drop > 1e-4 else 0.0
            lift = min(float(penetration), max_lift)
            if lift > 1e-5:
                floor_clamped_frames += 1
                max_penetration_fixed = max(max_penetration_fixed, float(lift))
                for n in order:
                    h_n = bmap_f[n]["head_3d"]
                    t_n = bmap_f[n]["tail_3d"]
                    bmap_f[n]["head_3d"] = [h_n[0], h_n[1], round(float(h_n[2]) + lift, 5)]
                    bmap_f[n]["tail_3d"] = [t_n[0], t_n[1], round(float(t_n[2]) + lift, 5)]

    return {
        "floor_clamped_frames": int(floor_clamped_frames),
        "foot_contact_locked_frames": int(len(foot_locked_frames_set)),
        "max_floor_penetration_fixed": round(float(max_penetration_fixed), 5),
        "contact_bones": contact_bones,
    }


# ==============================================================================
# 5. Stage 2 Synthetic Data Priors: Cosine-Wave Chains, Edge Stretch Gate,
#    Fast-Limb Afterimage Ghost Compositing & LBS Skinning Consistency
# ==============================================================================

def compute_chain_wave_target_euler(
    chain_index: int,
    lims: list[list[float]],
    wave_state: tuple[float, float, float, float],
) -> list[float]:
    """Computes a smooth travelling cosine-wave target `[rx_wave, 0.0, rz_wave]` along a multi-bone
    `tail`, `flex`, or `wing` chain (`XSXB-Mocap` `src/bl_render.py` lines 163-167):
        q(i) = mid + amp * cos(phase + freq * i)
    """
    phase_x, amp_u_x, phase_z, amp_u_z = wave_state
    idx = float(max(0, int(chain_index)))
    mid_x = 0.5 * (float(lims[0][0]) + float(lims[0][1]))
    half_x = 0.5 * (float(lims[0][1]) - float(lims[0][0]))
    mid_z = 0.5 * (float(lims[2][0]) + float(lims[2][1]))
    half_z = 0.5 * (float(lims[2][1]) - float(lims[2][0]))

    rx_w = mid_x + float(amp_u_x) * half_x * math.cos(float(phase_x) + 0.70 * idx)
    rz_w = mid_z + float(amp_u_z) * half_z * math.cos(float(phase_z) + 0.65 * idx)
    return [
        float(np.clip(rx_w, float(lims[0][0]), float(lims[0][1]))),
        0.0,
        float(np.clip(rz_w, float(lims[2][0]), float(lims[2][1]))),
    ]


def extract_unique_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    """Extracts unique undirected mesh edges `(M, 2)` from triangle indices `(F, 3)`."""
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] < 3 or len(f) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    e01 = np.sort(f[:, [0, 1]], axis=1)
    e12 = np.sort(f[:, [1, 2]], axis=1)
    e20 = np.sort(f[:, [2, 0]], axis=1)
    all_edges = np.vstack([e01, e12, e20])
    return np.unique(all_edges, axis=0)


def compute_mesh_edge_stretch_metric(
    rest_vertices: np.ndarray,
    posed_vertices: np.ndarray,
    edges_or_faces: np.ndarray,
    max_stretch: float = 1.80,
    min_stretch: float = 0.45,
    max_bad_ratio: float = 0.02,
) -> dict[str, Any]:
    """Evaluates deformed mesh edge stretch ratios `r = L_posed / L_rest` (`XSXB-Mocap`
    `src/bl_render.py` lines 113-118) to detect extreme skinning tear/collapse artifacts.

    A pose passes if the fraction of edges with `(r > max_stretch) | (r < min_stretch)` is
    at most `max_bad_ratio` (default 2%).
    """
    v_rest = np.asarray(rest_vertices, dtype=np.float64)
    v_posed = np.asarray(posed_vertices, dtype=np.float64)
    ef = np.asarray(edges_or_faces, dtype=np.int64)
    if ef.ndim == 2 and ef.shape[1] == 3:
        edges = extract_unique_edges_from_faces(ef)
    elif ef.ndim == 2 and ef.shape[1] == 2:
        edges = ef
    else:
        edges = np.zeros((0, 2), dtype=np.int64)

    if len(edges) == 0 or len(v_rest) == 0 or v_rest.shape != v_posed.shape:
        return {
            "passed": True,
            "bad_edge_ratio": 0.0,
            "max_stretch_ratio": 1.0,
            "min_stretch_ratio": 1.0,
            "edge_count": 0,
        }

    d_rest = np.linalg.norm(v_rest[edges[:, 0]] - v_rest[edges[:, 1]], axis=-1)
    d_posed = np.linalg.norm(v_posed[edges[:, 0]] - v_posed[edges[:, 1]], axis=-1)
    valid = d_rest > 1e-5
    if not np.any(valid):
        return {
            "passed": True,
            "bad_edge_ratio": 0.0,
            "max_stretch_ratio": 1.0,
            "min_stretch_ratio": 1.0,
            "edge_count": int(len(edges)),
        }

    ratios = d_posed[valid] / d_rest[valid]
    bad_mask = (ratios > float(max_stretch)) | (ratios < float(min_stretch))
    bad_ratio = float(np.mean(bad_mask))
    return {
        "passed": bool(bad_ratio <= float(max_bad_ratio)),
        "bad_edge_ratio": round(bad_ratio, 5),
        "max_stretch_ratio": round(float(np.max(ratios)), 4),
        "min_stretch_ratio": round(float(np.min(ratios)), 4),
        "edge_count": int(np.sum(valid)),
    }


def bone_motion_blur_speed_factor(kind: str, sub_role: str = "", chain_index: int = 0) -> float:
    """Returns per-bone relative speed factor (`FAST` in `XSXB-Mocap` `src/bl_render.py` lines 124-129).
    Trunk/root bones move slowly (`0.12 - 0.25`) while distal limbs, wings, digits, and tails move
    rapidly (`1.1 - 2.2`), producing realistic limb afterimages in synthetic motion-blur composites.
    """
    k = kind.strip().lower()
    idx = max(0, int(chain_index))
    if k == "rigid":
        return 0.0
    if k in ("root", "spine"):
        return 0.15
    if k in ("head", "jaw"):
        return 0.35
    if k in ("limb", "digit", "wing"):
        base = 1.45 if sub_role in ("forearm_hinge", "lower_limb", "hand_hub") else 1.10
        return min(2.4, base + 0.28 * idx)
    if k in ("tail", "flex"):
        return min(2.2, 0.90 + 0.25 * idx)
    return 1.0


def sample_ghost_perturbation_eulers(
    base_eulers: dict[str, list[float]],
    classified_bones: dict[str, dict[str, Any]],
    effective_limits_fn: Callable[[dict[str, Any]], list[list[float]]],
    rng: Any,
    dt_scale: float = 0.14,
) -> dict[str, list[float]]:
    """Perturbs `base_eulers` proportionally to `bone_motion_blur_speed_factor` so fast limbs/tails
    trail behind the main pose while the torso remains sharp."""
    ghost_eulers: dict[str, list[float]] = {}
    for b_name, rot in base_eulers.items():
        info = classified_bones.get(b_name, {})
        kind = str(info.get("kind", "limb"))
        if kind == "rigid" or bool(info.get("fixed_local", False)):
            ghost_eulers[b_name] = [0.0, 0.0, 0.0]
            continue
        speed = bone_motion_blur_speed_factor(
            kind=kind,
            sub_role=str(info.get("sub_role", "")),
            chain_index=int(info.get("chain_index", 0)),
        )
        lims = effective_limits_fn(info)
        out_rot: list[float] = []
        for ax in range(3):
            lo, hi = float(lims[ax][0]), float(lims[ax][1])
            span = hi - lo
            if span <= 1e-6:
                out_rot.append(float(rot[ax]))
                continue
            delta = float(rng.uniform(-1.0, 1.0)) * float(dt_scale) * speed * (0.35 * span)
            out_rot.append(float(np.clip(float(rot[ax]) + delta, lo, hi)))
        if not bool(info.get("children")):
            out_rot[1] = 0.0
        ghost_eulers[b_name] = out_rot
    return ghost_eulers


def blend_limb_afterimage_ghost_rgba(
    main_rgba: np.ndarray,
    ghost_rgba_list: list[np.ndarray],
    ghost_opacity: float = 0.28,
) -> np.ndarray:
    """Composites 1-2 translucent ghost pose renders behind the opaque main pose render
    (`XSXB-Mocap` `src/bl_render.py` lines 252-265) to simulate AI video fast-limb afterimages
    and motion blur while keeping ground-truth labels locked to `main_rgba`.
    """
    main = np.asarray(main_rgba, dtype=np.float64)
    if not ghost_rgba_list or ghost_opacity <= 0.0:
        return np.clip(main, 0.0, 255.0).astype(np.uint8)

    scale = 255.0 if float(np.max(main)) > 1.5 else 1.0
    m_norm = np.clip(main / scale, 0.0, 1.0)
    m_rgb = m_norm[..., :3]
    m_a = m_norm[..., 3:4]

    n_g = len(ghost_rgba_list)
    g_rgb_acc = np.zeros_like(m_rgb)
    g_a_acc = np.zeros_like(m_a)
    weights = [float(ghost_opacity) / float(i + 1) for i in range(n_g)]
    for g_arr, w_g in zip(ghost_rgba_list, weights):
        g_norm = np.clip(np.asarray(g_arr, dtype=np.float64) / scale, 0.0, 1.0)
        g_rgb = g_norm[..., :3]
        g_a = g_norm[..., 3:4] * w_g
        g_rgb_acc = g_rgb_acc + g_rgb * g_a
        g_a_acc = g_a_acc + g_a

    g_a_clamped = np.clip(g_a_acc, 0.0, 0.65)
    g_rgb_mean = g_rgb_acc / np.maximum(g_a_acc, 1e-8)

    # Main foreground sits OPACITY-FIRST over the ghost trail so primary silhouette remains crisp
    out_a = m_a + (1.0 - m_a) * g_a_clamped
    out_rgb = (m_rgb * m_a + (1.0 - m_a) * g_rgb_mean * g_a_clamped) / np.maximum(out_a, 1e-8)
    out = np.concatenate([out_rgb, out_a], axis=-1)
    if scale > 1.5:
        return np.clip(np.round(out * 255.0), 0.0, 255.0).astype(np.uint8)
    return np.clip(out, 0.0, 1.0).astype(main_rgba.dtype)


def verify_blender_lbs_skin_consistency(
    rest_vertices: np.ndarray,
    blender_posed_vertices: np.ndarray,
    bone_weights: np.ndarray,
    bone_indices: np.ndarray,
    rest_heads: np.ndarray,
    rest_rots: np.ndarray,
    posed_heads: np.ndarray,
    posed_rots: np.ndarray,
    lbs_deform_fn: Callable[..., np.ndarray],
    tol_ratio: float = 0.015,
) -> dict[str, Any]:
    """Verifies that Blender's evaluated armature mesh deformation matches analytical Linear Blend
    Skinning (`lbs_deform_fn`) within `tol_ratio` of the rest-pose bounding-box diagonal
    (`XSXB-Mocap` `src/rigtorch.py` lines 142-193).
    """
    v_rest = np.asarray(rest_vertices, dtype=np.float64)
    v_bl = np.asarray(blender_posed_vertices, dtype=np.float64)
    if len(v_rest) == 0 or v_rest.shape != v_bl.shape:
        return {
            "consistent": True,
            "max_vertex_error": 0.0,
            "mean_vertex_error": 0.0,
            "relative_max_error": 0.0,
        }
    v_lbs = lbs_deform_fn(
        v_rest,
        bone_weights,
        bone_indices,
        rest_heads,
        rest_rots,
        posed_heads,
        posed_rots,
    )
    errs = np.linalg.norm(v_lbs - v_bl, axis=-1)
    max_err = float(np.max(errs))
    mean_err = float(np.mean(errs))
    bbox_diag = float(np.linalg.norm(np.max(v_rest, axis=0) - np.min(v_rest, axis=0)))
    rel_max = max_err / max(1e-6, bbox_diag)
    return {
        "consistent": bool(rel_max <= float(tol_ratio)),
        "max_vertex_error": round(max_err, 6),
        "mean_vertex_error": round(mean_err, 6),
        "relative_max_error": round(rel_max, 6),
    }
