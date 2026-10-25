"""Action-scoped measured-surface contact refinement with bounded temporal FK.

Events are supplied explicitly, separate from a character's joint envelope.
No detector or training state is modified. Numerical convergence is not a
physical acceptance result; diagnostics and an export gate are mandatory.
"""

import copy
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.sparse import lil_matrix


def rotations(a):
    x, y, z = np.moveaxis(a, -1, 0)
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    return np.stack(
        [
            cz * cy,
            cz * sy * sx - sz * cx,
            cz * sy * cx + sz * sx,
            sz * cy,
            sz * sy * sx + cz * cx,
            sz * sy * cx - cz * sx,
            -sy,
            cy * sx,
            cy * cx,
        ],
        axis=-1,
    ).reshape(a.shape[:-1] + (3, 3))


def solve_window(frames, meta, profile, event, euler_fn):
    original = copy.deepcopy(frames)
    order = sorted(meta, key=lambda n: meta[n]["depth"])
    N = len(frames)
    root = order[0]
    rh = {b: np.array(meta[b]["rest_head_world"]) for b in order}
    rt = {b: np.array(meta[b]["rest_tail_world"]) for b in order}
    rr = {b: np.array(meta[b]["rest_rot_mat_world"]) for b in order}
    H = max(v[2] for v in rt.values()) - min(v[2] for v in rh.values())
    angles = {}
    heads = {}
    rots = {}
    for b in order:
        p = meta[b].get("parent")
        r = np.array([f["bones"][b]["rot_mat_world"] for f in frames])
        initial = (
            np.array([f["bones"][p]["rot_mat_world"] for f in frames]) @ rr[p].T @ rr[b]
            if p
            else np.broadcast_to(rr[b], r.shape)
        )
        angles[b] = np.array([euler_fn(v) for v in np.swapaxes(initial, -1, -2) @ r])
        heads[b] = np.array([f["bones"][b]["head_3d"] for f in frames])
        rots[b] = r
    variables = [
        (b, j)
        for b in order
        if b in event["controls"] and not meta[b].get("fixed_local")
        for j in range(3)
        if meta[b]["limits_rad"][j][1] - meta[b]["limits_rad"][j][0] > 1e-6
    ]
    K = len(variables)
    x0 = np.array([[angles[b][f, j] for b, j in variables] for f in range(N)])
    lo = np.array([meta[b]["limits_rad"][j][0] for b, j in variables])
    hi = np.array([meta[b]["limits_rad"][j][1] for b, j in variables])
    x0 = np.clip(x0, lo, hi)
    groups = {}
    for g in profile["surface_contact"]["support_sets"]:
        if len(g.get("influences", [])) != 1:
            continue
        inf = g["influences"][0]
        v = np.array(inf["samples_local"])
        hull = ConvexHull(v)
        groups[g["id"]] = {
            "bone": inf["bone"],
            "points": v[hull.vertices],
            "mean": v.mean(0),
            "hull_normals": hull.equations[:, :3],
        }

    def fk(x):
        local = {b: a.copy() for b, a in angles.items()}
        for j, (b, axis) in enumerate(variables):
            local[b][:, axis] = x[:, j]
        hh = {}
        r = {}
        tt = {}
        for b in order:
            p = meta[b].get("parent")
            if p:
                initial = r[p] @ rr[p].T @ rr[b]
                hh[b] = hh[p] + np.einsum("fij,j->fi", r[p], rr[p].T @ (rh[b] - rh[p]))
            else:
                initial = rr[b]
                hh[b] = heads[b]
            r[b] = initial @ rotations(local[b])
            tt[b] = hh[b] + np.einsum("fij,j->fi", r[b], rr[b].T @ (rt[b] - rh[b]))
        surfaces = {
            name: np.einsum("fij,vj->fvi", r[g["bone"]], g["points"])
            + hh[g["bone"]][:, None, :]
            for name, g in groups.items()
        }
        return hh, r, tt, surfaces

    hbase, rbase, tbase, sbase = fk(x0)
    fids = np.array([f.get("frame", i) for i, f in enumerate(frames)])
    a, b, c, d = event["phase"]
    smooth = lambda t: np.clip(t, 0, 1) ** 2 * (3 - 2 * np.clip(t, 0, 1))
    w = smooth((fids - a) / (b - a)) * smooth((d - fids) / (d - c))
    surface_id = event["surface"]
    g = groups[surface_id]
    base_z = sbase[surface_id][:, :, 2].min(1)
    target_z = base_z * (1 - w) + event.get("floor_z", 0.0) * w
    obs = event.get("endpoint_targets", {})
    targets = []
    for bone, vals in obs.items():
        targets.append((bone, np.array(vals), float(event.get("endpoint_weight", 4.0))))
    planes = []
    for pair in event["separation_pairs"]:
        an, bn = pair["a"], pair["b"]
        A = sbase[an]
        B = sbase[bn]
        if "axis" in pair:
            axis = np.broadcast_to(np.array(pair["axis"], float), (N, 3)).copy()
            axis /= np.linalg.norm(axis, axis=1)[:, None]
            desired = np.full(N, pair.get("gap", 0.002))
        else:
            axes = []
            desired = []
            for f in range(N):
                ns = np.concatenate(
                    [
                        groups[an]["hull_normals"] @ rbase[groups[an]["bone"]][f].T,
                        groups[bn]["hull_normals"] @ rbase[groups[bn]["bone"]][f].T,
                    ]
                )
                ns = np.concatenate([ns, -ns])
                gap = (B[f] @ ns.T).min(0) - (A[f] @ ns.T).max(0)
                j = np.argmax(gap)
                axes.append(ns[j])
                desired.append(min(pair.get("gap", 0.002), max(0.0, gap[j])))
            axis = np.array(axes)
            desired = np.array(desired)
        planes.append((an, bn, axis, desired))
    axis_local = np.array(event["surface_axis_local"], dtype=float)
    axis_local /= np.linalg.norm(axis_local)
    axis_target = np.array(event["surface_axis_world"], dtype=float)
    axis_target /= np.linalg.norm(axis_target)
    center_target = np.array(event["center_xy"])
    center_tolerance = np.array(event.get("center_tolerance", [0.04, 0.04]))
    joint_delta_scale = np.ones((N, 1)) * 0.35
    joint_delta_scale[[0, -1]] = 100.0

    def local_residual(x):
        hh, r, tt, s = fk(x)
        res = []
        for bone, target, weight in targets:
            res.append((hh[bone] - target) * weight * w[:, None] / H)
        for name, v in s.items():
            res.append(
                np.minimum(v[:, :, 2].min(1) - event.get("floor_z", 0.0), 0)[:, None]
                * 300
                / H
            )
        res.append((s[surface_id][:, :, 2].min(1) - target_z)[:, None] * 120 / H)
        center = hh[g["bone"]] + np.einsum("fij,j->fi", r[g["bone"]], g["mean"])
        dist = center[:, :2] - center_target
        res.append(
            np.sign(dist)
            * np.maximum(np.abs(dist) - center_tolerance, 0)
            * event.get("center_weight", 12.0)
            * w[:, None]
            / H
        )
        axis = np.einsum("fij,j->fi", r[g["bone"]], axis_local)
        res.append(
            (axis - axis_target) * event.get("orientation_weight", 1.0) * w[:, None]
        )
        for an, bn, axis, desired in planes:
            gap = np.einsum("fvi,fi->fv", s[bn], axis).min(1) - np.einsum(
                "fvi,fi->fv", s[an], axis
            ).max(1)
            if an == surface_id:
                gap = np.maximum(gap, s[an][:, :, 2].min(1) - s[bn][:, :, 2].max(1))
            res.append(
                np.minimum(gap - desired, 0)[:, None]
                * event.get("collision_weight", 200.0)
                / H
            )
        res.append((x - x0) * joint_delta_scale)
        return np.concatenate(res, axis=1)

    L = local_residual(x0).shape[1]

    def residual(flat):
        x = flat.reshape(N, K)
        delta = x - x0
        return np.concatenate(
            [
                local_residual(x).ravel(),
                (np.diff(delta, axis=0) * event.get("velocity_weight", 2.0)).ravel(),
                (
                    np.diff(delta, n=2, axis=0) * event.get("acceleration_weight", 6.0)
                ).ravel(),
            ]
        )

    sparsity = lil_matrix((N * L + (N - 1) * K + (N - 2) * K, N * K), dtype=int)
    for f in range(N):
        sparsity[f * L : (f + 1) * L, f * K : (f + 1) * K] = 1
    for f in range(N - 1):
        sparsity[N * L + f * K : N * L + (f + 1) * K, f * K : (f + 2) * K] = 1
    for f in range(N - 2):
        sparsity[
            N * L + (N - 1) * K + f * K : N * L + (N - 1) * K + (f + 1) * K,
            f * K : (f + 3) * K,
        ] = 1
    fit = least_squares(
        residual,
        x0.ravel(),
        bounds=(np.tile(lo, N), np.tile(hi, N)),
        jac_sparsity=sparsity.tocsr(),
        max_nfev=event.get("max_nfev", 100),
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        verbose=0,
    )
    x = fit.x.reshape(N, K)
    x[[0, -1]] = x0[[0, -1]]
    hh, r, tt, s = fk(x)
    for f, frame in enumerate(frames):
        for bone in order:
            cur = bone
            affected = False
            while cur:
                if cur in event["controls"]:
                    affected = True
                    break
                cur = meta[cur].get("parent")
            if affected:
                frame["bones"][bone].update(
                    head_3d=hh[bone][f].tolist(),
                    tail_3d=tt[bone][f].tolist(),
                    rot_mat_world=r[bone][f].tolist(),
                )
        if f in (0, N - 1):
            frame["bones"] = copy.deepcopy(original[f]["bones"])
    diag = {
        "cost": fit.cost,
        "success": bool(fit.success),
        "message": fit.message,
        "evaluations": fit.nfev,
        "variables_per_frame": K,
        "frames": N,
        "hammer_min_z": s[surface_id][:, :, 2].min(1).tolist(),
        "target_z": target_z.tolist(),
        "phase_weight": w.tolist(),
        "plane_gaps": [],
    }
    for an, bn, axis, desired in planes:
        gap = np.einsum("fvi,fi->fv", s[bn], axis).min(1) - np.einsum(
            "fvi,fi->fv", s[an], axis
        ).max(1)
        if an == surface_id:
            gap = np.maximum(gap, s[an][:, :, 2].min(1) - s[bn][:, :, 2].max(1))
        diag["plane_gaps"].append(
            {
                "a": an,
                "b": bn,
                "min_gap": float(gap.min()),
                "contact_min_gap": float(gap[w > 0.999].min())
                if np.any(w > 0.999)
                else None,
                "target": desired.tolist(),
                "gaps": gap.tolist(),
            }
        )
    return frames, diag


def frames_fingerprint(frames):
    """Bind hand-authored events to the exact trajectory they were reviewed on."""
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(
            frames, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def refine_trajectory(trajectory, bones_meta, profile, config):
    """Return a refined copy; absent events leave the original object untouched.

    Unsupported surfaces and stale annotations fail explicitly. Inputs, the
    character's limits, and frames outside event windows are never rewritten.
    """
    if not config or not config.get("events"):
        return trajectory
    if config.get("schema") != "monster-motion-contact-v1":
        raise ValueError("Unsupported motion contact schema")
    frames = trajectory["frames"]
    if config.get("frame_count") != len(frames):
        raise ValueError("Contact event frame count does not match trajectory")
    if config.get("input_frames_sha256") != frames_fingerprint(frames):
        raise ValueError(
            "Stale contact annotations: input trajectory fingerprint differs"
        )
    from bone_taxonomy import apply_joint_limits_profile, matrix_to_euler_xyz_numpy

    meta = apply_joint_limits_profile(bones_meta, profile)
    result = copy.deepcopy(trajectory)
    reports = []
    used = set()
    height = max(float(x["rest_tail_world"][2]) for x in meta.values()) - min(
        float(x["rest_head_world"][2]) for x in meta.values()
    )
    height = max(height, 1e-4)
    groups = {g["id"]: g for g in profile["surface_contact"]["support_sets"]}
    for event in config["events"]:
        start, end = event["window"]
        if not (
            isinstance(start, int)
            and isinstance(end, int)
            and 0 <= start < end < len(frames)
            and end - start >= 3
        ):
            raise ValueError("Invalid contact event window")
        if used.intersection(range(start, end + 1)):
            raise ValueError("Overlapping contact windows must be solved as one event")
        used.update(range(start, end + 1))
        phase = event["phase"]
        if len(phase) != 4 or not (
            start <= phase[0] < phase[1] <= phase[2] < phase[3] <= end
        ):
            raise ValueError("Contact phases must be ordered inside the solve window")
        if any(frames[i].get("frame", i) != i for i in range(start, end + 1)):
            raise ValueError(
                "Contact events require contiguous zero-based frame indices"
            )
        if not event["controls"] or any(
            b not in meta or meta[b].get("parent") is None for b in event["controls"]
        ):
            raise ValueError("Contact controls must name existing non-root bones")
        if (
            len(event["center_xy"]) != 2
            or len(event.get("center_tolerance", [0.04, 0.04])) != 2
        ):
            raise ValueError("Contact center must have two components")
        if (
            np.linalg.norm(event["surface_axis_local"]) < 1e-8
            or np.linalg.norm(event["surface_axis_world"]) < 1e-8
        ):
            raise ValueError("Contact orientation axes must be nonzero")
        required = {event["surface"]}
        for pair in event["separation_pairs"]:
            required.update([pair["a"], pair["b"]])
            if "axis" in pair and np.linalg.norm(pair["axis"]) < 1e-8:
                raise ValueError("Separation axis must be nonzero")
        for name in required:
            group = groups.get(name)
            inf = group.get("influences", []) if group else []
            if len(inf) != 1 or not np.allclose(inf[0]["weights"], 1.0):
                raise ValueError(
                    f"Contact target/guard {name} requires a measured rigid surface"
                )
        for bone, points in event.get("endpoint_targets", {}).items():
            if (
                bone not in meta
                or np.asarray(points).shape != (end - start + 1, 3)
                or not np.isfinite(points).all()
            ):
                raise ValueError(
                    "Endpoint targets must match the window and contain finite positions"
                )
        repaired, report = solve_window(
            result["frames"][start : end + 1],
            meta,
            profile,
            event,
            matrix_to_euler_xyz_numpy,
        )
        result["frames"][start : end + 1] = repaired
        contact = np.asarray(report["phase_weight"]) > 0.999
        errors = np.abs(
            np.asarray(report["hammer_min_z"])[contact] - event.get("floor_z", 0.0)
        )
        gap_error = max(
            (
                max(0.0, -p["contact_min_gap"])
                for p in report["plane_gaps"]
                if p["contact_min_gap"] is not None
            ),
            default=0.0,
        )
        report["max_contact_height_error"] = float(errors.max()) if errors.size else 0.0
        report["contact_targets_passed"] = bool(
            report["max_contact_height_error"] <= 0.005 * height
            and gap_error <= 0.0005 * height
        )
        report["window"] = [start, end]
        reports.append(report)
    # Evaluate every weighted measured mesh, not just rigid proxies used by the solver.
    minima = []
    for frame in result["frames"]:
        lowest = float("inf")
        for group in groups.values():
            if group.get("influences"):
                points = np.zeros((group["vertex_count"], 3))
                for inf in group["influences"]:
                    bone = frame["bones"][inf["bone"]]
                    posed = (
                        np.asarray(inf["samples_local"])
                        @ np.asarray(bone["rot_mat_world"]).T
                        + bone["head_3d"]
                    )
                    points[np.asarray(inf["indices"], dtype=int)] += (
                        posed * np.asarray(inf["weights"])[:, None]
                    )
            else:
                bone = frame["bones"][group["bone"]]
                points = (
                    np.asarray(group["samples_local"])
                    @ np.asarray(bone["rot_mat_world"]).T
                    + bone["head_3d"]
                )
            lowest = min(lowest, float(points[:, 2].min()))
        minima.append(lowest)
    floor = profile["surface_contact"].get("floor_z", 0.0)
    tolerance = (
        profile["surface_contact"].get("penetration_tolerance_fraction", 0.002) * height
    )
    floor_ok = all(z >= floor - tolerance for z in minima)
    diagnostics = result.setdefault("taxonomy_diagnostics", {})
    diagnostics["motion_contact"] = {
        "schema": config["schema"],
        "events": reports,
        "all_surface_min_z": minima,
        "targets_passed": all(r["contact_targets_passed"] for r in reports),
        "floor_passed": floor_ok,
    }
    diagnostics["contact_quality_passed"] = floor_ok and all(
        r["contact_targets_passed"] for r in reports
    )
    return result


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Refine a frozen trajectory using reviewed action contact events."
    )
    for name in ("input", "meta", "limits", "events", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(Path(path).read_text())
    meta = read(args.meta)
    result = refine_trajectory(
        read(args.input), meta.get("bones", meta), read(args.limits), read(args.events)
    )
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False))
    print(
        json.dumps(
            {
                "output": args.output,
                "contact_quality_passed": result.get("taxonomy_diagnostics", {}).get(
                    "contact_quality_passed"
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
