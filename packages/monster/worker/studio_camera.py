"""Calibrated orthographic/pinhole cameras; pixels are panel-local, +v points down.

Extrinsics use computer-vision axes (+x right, +y down, +z forward).
Blender conversion is explicit, so custom cameras and generated studios share one model.
"""

from dataclasses import dataclass
import json
import numpy as np


@dataclass(frozen=True)
class Camera:
    R: np.ndarray
    t: np.ndarray
    width: int
    height: int
    type: str = "ortho"
    scale: float = 1.0
    focal_px: float = 1.0
    shift_px: tuple = (0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "R", np.asarray(self.R, dtype=float).reshape(3, 3))
        object.__setattr__(self, "t", np.asarray(self.t, dtype=float).reshape(3))
        if (
            self.type not in ("ortho", "persp")
            or min(self.width, self.height, self.scale, self.focal_px) <= 0
        ):
            raise ValueError("Invalid camera type, resolution, scale or focal length")
        if (
            not np.isfinite(self.R).all()
            or not np.isfinite(self.t).all()
            or not np.allclose(self.R @ self.R.T, np.eye(3), atol=1e-5)
            or np.linalg.det(self.R) < 0
        ):
            raise ValueError("Camera rotation must be in SO(3)")

    def project(self, X, jac=False):
        q = np.asarray(X, dtype=float) @ self.R.T + self.t
        if self.type == "persp":
            z = np.maximum(q[..., 2], 1e-7)
            uv = q[..., :2] * (self.focal_px / z[..., None])
            J = np.zeros(q.shape[:-1] + (2, 3))
            J[..., 0, 0] = J[..., 1, 1] = self.focal_px / z
            in_front = q[..., 2] > 1e-7
            J[..., :, 2] = np.where(
                in_front[..., None],
                -self.focal_px * q[..., :2] / z[..., None] ** 2,
                0.0,
            )
        else:
            s = self.height / self.scale
            uv = q[..., :2] * s
            J = np.broadcast_to(np.array([[s, 0, 0], [0, s, 0]]), q.shape[:-1] + (2, 3))
        uv = uv + np.array([self.width / 2, self.height / 2]) + self.shift_px
        return (uv, J @ self.R) if jac else uv

    def depth(self, X):
        return (np.asarray(X) @ self.R.T + self.t)[..., 2]

    def ray_at_depth(self, u, z):
        xy = np.asarray(u) - np.array([self.width / 2, self.height / 2]) - self.shift_px
        z = np.asarray(z)
        xy = xy * (
            z[..., None] / self.focal_px
            if self.type == "persp"
            else self.scale / self.height
        )
        q = np.concatenate([xy, np.broadcast_to(z, xy.shape[:-1])[..., None]], axis=-1)
        return (q - self.t) @ self.R

    def as_dict(self):
        return {
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "width": self.width,
            "height": self.height,
            "type": self.type,
            "scale": self.scale,
            "focal_px": self.focal_px,
            "shift_px": list(self.shift_px),
        }

    @classmethod
    def from_blender(cls, obj, width, height):
        M = np.asarray(obj.matrix_world, dtype=float)
        C = np.diag([1.0, -1.0, -1.0])
        R = C @ M[:3, :3].T
        data = obj.data
        # Blender VERTICAL sensor fit is used by generated studios. AUTO follows image aspect.
        vertical = data.sensor_fit == "VERTICAL" or (
            data.sensor_fit == "AUTO" and height > width
        )
        fit_pixels = height if vertical else width
        # AUTO selects its fit axis by aspect, but Blender still uses sensor_width
        # as the physical sensor size. Only explicit VERTICAL uses sensor_height.
        sensor = data.sensor_height if data.sensor_fit == "VERTICAL" else data.sensor_width
        focal = float(data.lens) * fit_pixels / float(sensor)
        return cls(
            R,
            -R @ M[:3, 3],
            int(width),
            int(height),
            "ortho" if data.type == "ORTHO" else "persp",
            float(data.ortho_scale) * height / fit_pixels,
            focal,
            (-float(data.shift_x) * fit_pixels, float(data.shift_y) * fit_pixels),
        )


def cameras_from_calibration(calib):
    if calib.get("cameras"):
        return {k: Camera(**v) for k, v in calib["cameras"].items()}
    center = np.asarray(calib["center"], dtype=float)
    W, H = int(calib["panel_width"]), int(calib["panel_height"])
    # Right-handed view bases, matching existing front/side pixel conventions.
    rotations = {
        "front": [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
        "side": [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
        "left": [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
        "right": [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
        "top": [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
    }
    out = {}
    for view in calib.get("view_order", ("front", "side")):
        R = np.asarray(rotations[view], dtype=float)
        t = -R @ center + [0, 0, calib["ortho_scale"] * 2.5]
        out[view] = Camera(R, t, W, H, scale=calib["ortho_scale"])
    return out


def triangulate(cameras, uv, confidence, previous=None):
    """Weighted ray intersection; one view uses previous camera depth, never invented disparity."""
    cameras = list(cameras)
    uv, confidence = np.asarray(uv), np.asarray(confidence)
    points = np.zeros((uv.shape[1], 3))
    for k in range(len(points)):
        valid = [
            v
            for v in range(len(cameras))
            if confidence[v, k] >= 0.2 and np.isfinite(uv[v, k]).all()
        ]
        if len(valid) < 2:
            if previous is not None:
                points[k] = previous[k]
                if valid:
                    c = cameras[valid[0]]
                    points[k] = c.ray_at_depth(uv[valid[0], k], c.depth(previous[k]))
            else:
                points[k] = np.nan
            continue
        A, b = [], []
        for v in valid:
            c, w = cameras[v], confidence[v, k]
            if c.type == "ortho":
                xy = (
                    (uv[v, k] - [c.width / 2, c.height / 2] - c.shift_px)
                    * c.scale
                    / c.height
                )
                a, rhs = c.R[:2], xy - c.t[:2]
            else:
                xy = (uv[v, k] - [c.width / 2, c.height / 2] - c.shift_px) / c.focal_px
                a = c.R[:2] - xy[:, None] * c.R[2]
                rhs = xy * c.t[2] - c.t[:2]
            A.extend(a * w)
            b.extend(rhs * w)
        points[k] = np.linalg.lstsq(A, b, rcond=None)[0]
    return points


def export_cameras(path, named_cameras, width, height):
    payload = {
        name: Camera.from_blender(obj, width, height).as_dict()
        for name, obj in named_cameras.items()
    }
    with open(path, "w") as stream:
        json.dump(
            {"schema": "monster-cameras-v1", "cameras": payload}, stream, indent=2
        )
    return payload
