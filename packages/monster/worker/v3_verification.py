"""Numerical proof probes for the v3 FK/projector contract, run by the formal self-test."""

import numpy as np
from reprojection_solver import RigModel
from studio_camera import cameras_from_calibration


def verify_v3_numerics(meta, seed=42, skin=None):
    rig = RigModel(meta, skin=skin)
    rng = np.random.default_rng(seed)
    p = np.clip(rng.uniform(-0.1, 0.1, rig.size), rig.lower, rig.upper)
    points, J, heads, rotations = rig.fk(p, True)
    epsilon = 1e-6
    finite = np.stack(
        [
            (
                rig.fk(p + np.eye(rig.size)[i] * epsilon)[0]
                - rig.fk(p - np.eye(rig.size)[i] * epsilon)[0]
            )
            / (2 * epsilon)
            for i in range(rig.size)
        ],
        axis=-1,
    )
    derivative_error = float(np.max(np.abs(finite - J)))
    length_error = max(
        abs(
            np.linalg.norm(points[2 * i + 1] - points[2 * i])
            - np.linalg.norm(rig.t[n] - rig.h[n])
        )
        for i, n in enumerate(rig.names)
    )
    so3_error = max(
        max(np.linalg.norm(R.T @ R - np.eye(3)), abs(np.linalg.det(R) - 1))
        for R in rotations.values()
    )
    camera_errors = []
    for c in cameras_from_calibration(
        {"center": [0, 0, 1], "ortho_scale": 3, "panel_width": 448, "panel_height": 448}
    ).values():
        uv, P = c.project(points, True)
        camera_errors.append(
            float(np.max(np.abs(c.ray_at_depth(uv, c.depth(points)) - points)))
        )
    passed = (
        derivative_error < 1e-6
        and length_error < 1e-10
        and so3_error < 1e-10
        and max(camera_errors) < 1e-10
    )
    report = {
        "passed": passed,
        "fk_jacobian_max_error": derivative_error,
        "bone_length_max_error": float(length_error),
        "so3_max_error": float(so3_error),
        "camera_inverse_max_error": max(camera_errors),
        "dofs": rig.size,
    }
    if not passed:
        raise RuntimeError("V3 formal numerical probes failed: " + str(report))
    return report
