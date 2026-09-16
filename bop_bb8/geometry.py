"""Geometry helpers for ordered 3D bounding-box corners and BOP projection.

Conventions (kept identical to BoxDreamer and the BOP format):

- Model / camera coordinates use OpenCV convention: ``+x`` right, ``+y`` down,
  ``+z`` forward.  Lengths are millimeters.
- ``cam_R_m2c`` (row-major 9 values) and ``cam_t_m2c`` (3 values) map model
  coordinates to camera coordinates: ``P_c = R_m2c @ P_m + t_m2c``.
- Corner order (channel ``i`` of the heatmap ↔ ``corners_3d[i]`` ↔
  ``keypoints_2d[i]``):
  0 ``(xmin,ymin,zmin)``, 1 ``(xmax,ymin,zmin)``, 2 ``(xmax,ymax,zmin)``,
  3 ``(xmin,ymax,zmin)``, 4 ``(xmin,ymin,zmax)``, 5 ``(xmax,ymin,zmax)``,
  6 ``(xmax,ymax,zmax)``, 7 ``(xmin,ymax,zmax)``.
- ``cam_K`` is row-major 9 values, reshaped to 3x3.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CORNER_SIGNS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ],
    dtype=np.float64,
)

PROJECTION_EPS = 1e-6


def corners_from_aabb(min_xyz: np.ndarray, max_xyz: np.ndarray) -> np.ndarray:
    """Return the 8 ordered AABB corners (mm) as a ``[8, 3]`` array."""
    min_xyz = np.asarray(min_xyz, dtype=np.float64)
    max_xyz = np.asarray(max_xyz, dtype=np.float64)
    return min_xyz[None, :] + CORNER_SIGNS * (max_xyz - min_xyz)[None, :]


@dataclass(frozen=True)
class BOPModel:
    """BOP object model described by its axis-aligned bounding box in mm.

    ``scale`` multiplies the model corners uniformly.  It defaults to ``1.0``:
    the model is used exactly as authored.  A diagnostic scale fitted from
    ``bbox_obj`` is *not* a confirmed metric parameter, so callers must pass any
    non-unit scale explicitly after the model/data correspondence is confirmed.
    """

    obj_id: int
    min_xyz_mm: np.ndarray
    max_xyz_mm: np.ndarray
    diameter_mm: float = float("nan")
    scale: float = 1.0

    @classmethod
    def from_models_info(cls, path: str | Path, obj_id: int = 1, scale: float = 1.0) -> "BOPModel":
        with open(path) as f:
            info = json.load(f)
        entry = info[str(obj_id)]
        return cls(
            obj_id=obj_id,
            min_xyz_mm=np.array([entry["min_x"], entry["min_y"], entry["min_z"]], dtype=np.float64),
            max_xyz_mm=np.array([entry["max_x"], entry["max_y"], entry["max_z"]], dtype=np.float64),
            diameter_mm=float(entry.get("diameter", float("nan"))),
            scale=float(scale),
        )

    def corners_mm(self) -> np.ndarray:
        return corners_from_aabb(self.min_xyz_mm, self.max_xyz_mm) * self.scale

    def size_mm(self) -> np.ndarray:
        return (self.max_xyz_mm - self.min_xyz_mm) * self.scale


def parse_rotation(values) -> np.ndarray:
    """Reshape row-major ``cam_R_m2c`` into a 3x3 matrix."""
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def parse_intrinsics(values) -> np.ndarray:
    """Reshape row-major ``cam_K`` into a 3x3 matrix."""
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def crop_and_scale_intrinsics(
    K: np.ndarray,
    crop: tuple[int, int, int, int],
    out_size: tuple[int, int],
) -> np.ndarray:
    """Update intrinsics for crop ``(x0, y0, wc, hc)`` then resize to ``(W, H)``.

    ``K_crop = [[sx, 0, -sx*x0], [0, sy, -sy*y0], [0, 0, 1]] @ K`` with
    ``sx = W / wc`` and ``sy = H / hc``.
    """
    x0, y0, wc, hc = crop
    out_w, out_h = out_size
    sx, sy = out_w / wc, out_h / hc
    M = np.array([[sx, 0.0, -sx * x0], [0.0, sy, -sy * y0], [0.0, 0.0, 1.0]])
    return M @ np.asarray(K, dtype=np.float64)


def project_corners(
    corners_mm: np.ndarray,
    K: np.ndarray,
    R_m2c: np.ndarray,
    t_m2c: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project model corners into the (uncropped) input image.

    Returns ``(keypoints_2d, valid, points_cam)``:

    - ``keypoints_2d``: ``[8, 2]`` pixel coordinates; ``NaN`` where ``Z <= eps``.
    - ``valid``: ``[8]`` bool, ``True`` only for ``Z > eps`` and a projection
      inside ``[0, W) x [0, H)``; invalid corners must not be snapped to the
      image border.
    - ``points_cam``: ``[8, 3]`` camera-frame points in mm.
    """
    corners_mm = np.asarray(corners_mm, dtype=np.float64)
    R_m2c = parse_rotation(R_m2c)
    t_m2c = np.asarray(t_m2c, dtype=np.float64).reshape(3)
    width, height = image_size

    points_cam = corners_mm @ R_m2c.T + t_m2c[None, :]
    z = points_cam[:, 2]
    keypoints_2d = np.full((len(corners_mm), 2), np.nan, dtype=np.float64)
    positive = z > PROJECTION_EPS
    if positive.any():
        q = (np.asarray(K, dtype=np.float64) @ points_cam[positive].T).T
        keypoints_2d[positive] = q[:, :2] / q[:, 2:3]
    valid = positive.copy()
    valid[positive] &= (
        (keypoints_2d[positive, 0] >= 0.0)
        & (keypoints_2d[positive, 0] < width)
        & (keypoints_2d[positive, 1] >= 0.0)
        & (keypoints_2d[positive, 1] < height)
    )
    return keypoints_2d, valid, points_cam


def scale_keypoints_to_heatmap(
    keypoints_2d: np.ndarray,
    input_size: tuple[int, int],
    heatmap_size: tuple[int, int],
) -> np.ndarray:
    """Map input-frame pixel coordinates onto the heatmap grid.

    Both sizes are ``(height, width)``, matching the ``[8, H, W]`` heatmap
    layout.  The mapping is a pure per-axis scaling; do not mix coordinates
    from the original image, the network input and the heatmap in one call.
    """
    in_h, in_w = input_size
    out_h, out_w = heatmap_size
    kps = np.asarray(keypoints_2d, dtype=np.float64).copy()
    kps[..., 0] *= out_w / in_w
    kps[..., 1] *= out_h / in_h
    return kps
