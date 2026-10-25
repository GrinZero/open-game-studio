"""Bounded surface-contact fitting for rigs carrying measured support samples.

The optional per-rig samples come from evaluated rigid support meshes. Root
evidence is preserved while movable ancestors solve floor penetration together;
the rest-pose bone endpoint height is never substituted for the visible sole.
"""
from __future__ import annotations

import math
import copy
from typing import Any
import numpy as np


def assert_surface_contact_quality(trajectory: dict[str, Any], action: str = '') -> None:
    if trajectory.get('taxonomy_diagnostics', {}).get('contact_quality_passed') is False:
        raise RuntimeError(f"Action '{action}' passed view sync but measured surface contact is unresolved; "
                           "hard-stopping before IK bake.")


def _rotation(a):
    x, y, z = a
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    return np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                     [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx],
                     [-sy, cy*sx, cy*cx]])


def _euler(r):
    y = math.asin(float(np.clip(-r[2, 0], -1, 1)))
    if abs(math.cos(y)) > 1e-6:
        return np.array([math.atan2(r[2, 1], r[2, 2]), y, math.atan2(r[1, 0], r[0, 0])])
    return np.array([math.atan2(-r[1, 2], r[1, 1]), y, 0.0])


def apply_surface_floor_constraints(frames_out, order, root, classified_bones,
                                    rest_H, rest_T, rest_R, effective_base_limits,
                                    config: dict[str, Any]):
    groups = config.get('support_sets', [])
    if not groups:
        raise ValueError('Surface contact enabled without measured support sets')
    floor = float(config.get('floor_z', 0.0))
    height = max(1e-4, max(float(v[2]) for v in rest_T.values()) -
                 min(float(v[2]) for v in rest_H.values()))
    root_budget = max(0., min(.03, float(config.get('root_correction_fraction', .02))))
    tolerance = max(1e-6, float(config.get('penetration_tolerance_fraction', .002)) * height)
    groups = [(copy.deepcopy(g), str(g['bone']), np.asarray(g.get('samples_local', []), dtype=float)) for g in groups]
    controls = set()
    for g, bone, points in groups:
        influences = g.get('influences')
        if influences:
            if not g.get('vertex_count') or any(i['bone'] not in classified_bones for i in influences):
                raise ValueError(f'Invalid weighted surface set for {bone}')
            for influence in influences:
                influence['_indices'] = np.asarray(influence['indices'], dtype=int)
                influence['_local'] = np.asarray(influence['samples_local'], dtype=float)
                influence['_weights'] = np.asarray(influence['weights'], dtype=float)[:, None]
        elif bone not in classified_bones or points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError(f'Invalid surface support set for {bone}')
        if g.get('adjust_controls', True):
            for start in ([i['bone'] for i in influences] if influences else [bone]):
                cur = start
                while cur and cur != root:
                    info = classified_bones[cur]
                    if info.get('kind') != 'rigid' and not info.get('fixed_local', False):
                        controls.add(cur)
                    cur = info.get('parent')
    variables = [(b, axis) for b in order if b in controls for axis in range(3)
                 if effective_base_limits[b][axis][1] - effective_base_limits[b][axis][0] > 1e-6]
    base_lower = np.array([effective_base_limits[b][i][0] for b, i in variables] + [-root_budget])
    base_upper = np.array([effective_base_limits[b][i][1] for b, i in variables] + [root_budget])
    sorted_controls = sorted(controls)
    parents = {b: classified_bones[b].get('parent') for b in order}
    rr_rel = {}
    off_rel = {}
    vec_loc = {}
    for b in order:
        p = parents[b]
        if p and p in rest_R:
            rr_rel[b] = rest_R[p].T @ rest_R[b]
            off_rel[b] = rest_R[p].T @ (rest_H[b] - rest_H[p])
        vec_loc[b] = rest_R[b].T @ (rest_T[b] - rest_H[b])
    descendants = {}
    for b in order:
        desc = {b}
        for c in order:
            cur = c
            while cur:
                if cur in desc:
                    desc.add(c)
                    break
                cur = parents.get(cur)
        descendants[b] = desc
    for g, _, _ in groups:
        if g.get('influences'):
            for inf in g['influences']:
                inf['_weights_1d'] = inf['_weights'][:, 0]
    previous_pose = None
    anchors = {}
    previous_centers = {}
    diagnostics = []
    for index, frame in enumerate(frames_out):
        bones = frame['bones']
        root_head = np.array(bones[root]['head_3d'], dtype=float)
        lower, upper = base_lower.copy(), base_upper.copy()
        absolute = classified_bones[root].get('global_root', {}).get('max_translation_world_xyz')
        if isinstance(absolute, (list, tuple)) and len(absolute) == 3:
            displacement = float(root_head[2]-rest_H[root][2])
            lower[-1] = max(lower[-1], (-float(absolute[2])-displacement)/height)
            upper[-1] = min(upper[-1], (float(absolute[2])-displacement)/height)
            if lower[-1] > upper[-1]:
                raise ValueError('Observed root lies outside the configured envelope and contact correction budget')
        local_angles = {}
        original_heads = {}
        original_tails = {}
        for b in order:
            info = classified_bones[b]
            parent = parents[b]
            initial = (np.asarray(bones[parent]['rot_mat_world']) @ rr_rel[b]
                       if parent and parent in bones else rest_R[b])
            local_angles[b] = _euler(initial.T @ np.asarray(bones[b]['rot_mat_world']))
            original_heads[b] = np.asarray(bones[b]['head_3d'], dtype=float)
            original_tails[b] = np.asarray(bones[b]['tail_3d'], dtype=float)
            if info.get('fixed_local', False) or info.get('kind') == 'rigid':
                local_angles[b] = np.zeros(3)
        base_local_rots = {b: _rotation(local_angles[b]) for b in order}
        initial_x = np.array([local_angles[b][i] for b, i in variables] + [0.])
        initial_x = np.clip(initial_x, lower, upper)

        def fk_bones(x, base_state=None, dirty_bone=None):
            angles = {b: a.copy() for b, a in local_angles.items()}
            changed_bones = set()
            for value, (b, axis) in zip(x[:-1], variables):
                angles[b][axis] = value
                changed_bones.add(b)
            if base_state is not None and dirty_bone is not None:
                base_heads, base_rots, base_tails = base_state
                dirty_set = descendants[dirty_bone]
                heads = {b: base_heads[b] for b in order if b not in dirty_set}
                rotations = {b: base_rots[b] for b in order if b not in dirty_set}
                tails = {b: base_tails[b] for b in order if b not in dirty_set}
                iter_order = [b for b in order if b in dirty_set]
            else:
                heads, rotations, tails = {}, {}, {}
                iter_order = order
            for b in iter_order:
                parent = parents[b]
                if parent and parent in rotations:
                    rp = rotations[parent]
                    initial = rp @ rr_rel[b]
                    head = heads[parent] + rp @ off_rel[b]
                else:
                    initial = rest_R[b]
                    head = np.array([root_head[0], root_head[1], root_head[2] + x[-1]*height])
                r_loc = _rotation(angles[b]) if b in changed_bones else base_local_rots[b]
                rot = initial @ r_loc
                rotations[b] = rot
                heads[b] = head
                tails[b] = head + rot @ vec_loc[b]
            return heads, rotations, tails

        def fk(x, include_guards=False):
            heads, rotations, tails = fk_bones(x)
            surfaces = []
            for g, bone, points in groups:
                if not include_guards and not g.get('adjust_controls', True):
                    surfaces.append(None)
                elif g.get('influences'):
                    surface = np.zeros((int(g['vertex_count']), 3))
                    for influence in g['influences']:
                        b = influence['bone']
                        surface[influence['_indices']] += (influence['_local'] @ rotations[b].T + heads[b]) * influence['_weights']
                    surfaces.append(surface)
                else:
                    surfaces.append(points @ rotations[bone].T + heads[bone])
            return heads, rotations, tails, surfaces

        _, _, _, raw_surfaces = fk(initial_x)
        initial_penetration = max(0., floor-min(float(s[:, 2].min()) for s in raw_surfaces if s is not None))
        locks = []
        locked_gis = {}
        for gi, ((g, _, _), surface) in enumerate(zip(groups, raw_surfaces)):
            if surface is None:
                continue
            center = surface.mean(0)
            speed = float(np.linalg.norm(center-previous_centers.get(gi, center)))
            near = abs(float(surface[:, 2].min())-floor) <= .035*height
            stationary = speed <= .012*height
            if g.get('lock_xy', False) and near and stationary:
                if gi not in anchors:
                    sample = int(np.argmin(surface[:, 2]))
                    point = surface[sample].copy();point[2] = floor
                    anchors[gi] = (sample, point)
                locks.append((gi, *anchors[gi]))
                locked_gis.setdefault(gi, []).append(anchors[gi])
            else:
                anchors.pop(gi, None)
            previous_centers[gi] = center

        scale_ht = 1.5 / height
        scale_pen = 300. / height
        scale_lock = 3. / height

        def eval_residual_from_state(x, heads, rotations, tails):
            r = []
            for b in sorted_controls:
                r.extend((heads[b]-original_heads[b]) * scale_ht)
                r.extend((tails[b]-original_tails[b]) * scale_ht)
            locked_pts = {}
            for gi, (g, bone, points) in enumerate(groups):
                if not g.get('adjust_controls', True):
                    continue
                if gi in locked_gis:
                    if g.get('influences'):
                        surface = np.zeros((int(g['vertex_count']), 3))
                        for influence in g['influences']:
                            b = influence['bone']
                            surface[influence['_indices']] += (influence['_local'] @ rotations[b].T + heads[b]) * influence['_weights']
                    else:
                        surface = points @ rotations[bone].T + heads[bone]
                    min_z = float(surface[:, 2].min())
                    locked_pts[gi] = surface
                else:
                    if g.get('influences'):
                        z_vals = np.zeros(int(g['vertex_count']))
                        for influence in g['influences']:
                            b = influence['bone']
                            z_vals[influence['_indices']] += (influence['_local'] @ rotations[b][2] + heads[b][2]) * influence['_weights_1d']
                        min_z = float(z_vals.min())
                    else:
                        min_z = float((points @ rotations[bone][2]).min() + heads[bone][2])
                r.append(min(0., min_z - floor) * scale_pen)
            for gi, sample, anchor in locks:
                r.extend((locked_pts[gi][sample] - anchor) * scale_lock)
            r.extend((x[:-1]-initial_x[:-1])*.025)
            r.append(x[-1]*15.)
            if previous_pose is not None:
                r.extend((x[:-1]-previous_pose[:-1])*.015)
            return np.asarray(r)

        def residual(x):
            heads, rotations, tails = fk_bones(x)
            return eval_residual_from_state(x, heads, rotations, tails)

        max_iters = int(config.get('max_iterations', 24))
        n_vars = len(initial_x)
        eye_vars = np.eye(n_vars) * 2e-5

        def optimize_from(start_x):
            x_cur = start_x.copy()
            damping = .001
            for _ in range(max_iters):
                base_state = fk_bones(x_cur)
                r = eval_residual_from_state(x_cur, *base_state)
                cost = float(r @ r)
                cols = []
                for j in range(n_vars):
                    x_eps = x_cur + eye_vars[j]
                    if j < len(variables):
                        st_eps = fk_bones(x_eps, base_state=base_state, dirty_bone=variables[j][0])
                    else:
                        st_eps = fk_bones(x_eps)
                    r_eps = eval_residual_from_state(x_eps, *st_eps)
                    cols.append((r_eps - r) * 5e4)
                jac = np.column_stack(cols)
                gradient = jac.T @ r
                free = ~(((x_cur <= lower+1e-7) & (gradient > 0)) |
                         ((x_cur >= upper-1e-7) & (gradient < 0)))
                step = np.zeros(n_vars)
                if not np.any(free):
                    break
                jf = jac[:, free]
                step[free] = np.linalg.solve(jf.T @ jf + np.eye(int(free.sum()))*damping, -gradient[free])
                step = np.clip(step, -.15, .15)
                accepted = False
                for scale in (1., .5, .2, .05):
                    candidate = np.clip(x_cur+scale*step, lower, upper)
                    rr = residual(candidate)
                    if float(rr @ rr) < cost-1e-10:
                        x_cur = candidate;damping = max(1e-7, damping*.7);accepted = True;break
                if not accepted:
                    damping = min(1e5, damping*5.)
                if np.linalg.norm(step) < 1e-7:
                    break
            r_final = residual(x_cur)
            _, _, _, surfs = fk(x_cur)
            active_pen = max(0., floor-min((float(s[:, 2].min()) for s in surfs if s is not None), default=floor))
            return x_cur, float(r_final @ r_final), active_pen

        # Always optimize from the observed kinematic pose `initial_x` first so
        # valid crouching/kneeling postures are not discarded at iteration 0 just
        # because a distal toe/claw has minor initial penetration.
        x, best_cost, active_pen = optimize_from(initial_x.copy())
        needed_from_init = active_pen + min(1e-5 * height, tolerance * .01)
        if active_pen > tolerance and (x[-1] + needed_from_init / height > upper[-1] + 1e-10):
            # Endpoint penetration stalled beyond a hinge's local minimum and
            # exceeds the remaining root budget. Try deterministic fallback starts
            # (previous pose and interior seeds) scored after optimization.
            fallback_starts = []
            if previous_pose is not None:
                fallback_starts.append(np.clip(previous_pose, lower, upper))
            fallback_starts.append((lower+upper)*.5)
            for sign in (-1., 1.):
                seed = initial_x.copy()
                for bone in sorted_controls:
                    choices = [j for j, (b, _) in enumerate(variables) if b == bone]
                    if choices:
                        j = max(choices, key=lambda k: upper[k]-lower[k])
                        seed[j] = .65*(upper[j] if sign > 0 else lower[j])
                fallback_starts.append(seed)
            for seed in fallback_starts:
                seed = seed.copy()
                seed[-1] = 0.
                cand_x, cand_cost, cand_pen = optimize_from(seed)
                if (cand_pen <= tolerance and active_pen > tolerance) or (
                    cand_pen < active_pen - 1e-5
                ) or (abs(cand_pen - active_pen) <= 1e-5 and cand_cost < best_cost):
                    x, best_cost, active_pen = cand_x, cand_cost, cand_pen
        heads, rotations, tails, surfaces = fk(x, include_guards=True)
        penetration = max(0., floor-min(float(s[:, 2].min()) for s in surfaces))
        # The joint fit can stall just below the floor. Use only the remaining
        # original root budget for a rigid translation; never erase a crouch or
        # exceed the character's absolute root envelope. Rigid translation does
        # not alter part-to-part clearance or bone lengths.
        warm_pose = x.copy()
        fallback_lift = 0.
        if penetration > tolerance:
            needed = penetration + min(1e-5*height, tolerance*.01)
            if x[-1] + needed/height <= upper[-1] + 1e-10:
                fallback_lift = needed
                x[-1] += needed/height
                heads, rotations, tails, surfaces = fk(x, include_guards=True)
                penetration = max(0., floor-min(float(s[:, 2].min()) for s in surfaces))
        for b in order:
            bones[b]['head_3d'] = heads[b].tolist()
            bones[b]['tail_3d'] = tails[b].tolist()
            bones[b]['rot_mat_world'] = rotations[b].tolist()
        previous_pose = warm_pose
        diagnostics.append({'frame': index, 'root_z_correction': float(x[-1]*height),
                            'surface_penetration': penetration, 'bounded_fallback_lift': fallback_lift,
                            'penetration_fixed': max(0., initial_penetration-penetration),
                            'locked_support_count': len(locks)})
    unresolved = sum(d['surface_penetration'] > tolerance for d in diagnostics)
    return {'contact_method': 'bounded_measured_surface_ancestor_fit_v1',
            'contact_quality_passed': unresolved == 0,
            'surface_floor_unresolved_frames': unresolved,
            'surface_floor_max_penetration': max((d['surface_penetration'] for d in diagnostics), default=0.),
            'root_contact_max_correction': max((abs(d['root_z_correction']) for d in diagnostics), default=0.),
            'foot_contact_locked_frames': sum(d['locked_support_count'] > 0 for d in diagnostics),
            'floor_clamped_frames': sum(d['root_z_correction'] > 1e-5 for d in diagnostics),
            'max_floor_penetration_fixed': max((d['penetration_fixed'] for d in diagnostics), default=0.),
            'contact_bones': list(dict.fromkeys(b for _, b, _ in groups)),
            'contact_frames': diagnostics}
