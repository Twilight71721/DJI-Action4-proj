"""Optional perspective cuboid fitting with explicitly supplied camera intrinsics."""
from __future__ import annotations

import cv2
import numpy as np


def validate_camera(K, distortion):
    K = np.asarray(K, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if (K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1]) or not np.allclose([K[0, 1], K[1, 0]], 0)):
        raise ValueError('Expected finite pinhole camera K with positive focal lengths and zero skew')
    if distortion.size not in (0, 4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
        raise ValueError('Expected OpenCV pinhole distortion coefficients; fisheye parameters are not interchangeable')
    return K, distortion


def fit_perspective_cuboid(corners, points, K, *, distortion=(), reprojection_threshold=5.,
                          min_inliers=6, iterations=1000):
    """RANSAC + inlier-only LM refinement; never substitute bbox edges for corners.

    All coordinates and the threshold refer to the same image frame as K.
    Accepted means geometric consistency under the supplied camera model,
    not correct corner identity, calibration, or pose ground truth.
    """
    corners = np.ascontiguousarray(corners, dtype=np.float64)
    points = np.ascontiguousarray(points, dtype=np.float64)
    K, distortion = validate_camera(K, distortion)
    if (corners.shape != (8, 3) or not np.isfinite(corners).all()
            or np.linalg.matrix_rank(corners-corners.mean(0)) != 3 or points.shape != (8, 2)):
        raise ValueError('Expected ordered model corners [8,3] and corresponding image points [8,2]')
    if not 6 <= min_inliers <= 8 or iterations < 1 or not np.isfinite(reprojection_threshold) or reprojection_threshold <= 0:
        raise ValueError('Expected 6..8 inliers, positive iterations and reprojection threshold')
    base = dict(accepted=False, points=None, projection='perspective', min_inliers=min_inliers,
                reprojection_threshold_px=float(reprojection_threshold), iterations=int(iterations),
                note='Geometric consistency under supplied K; not verified pose accuracy')
    def reject(status, **details):
        return dict(base, status=status, **details)
    finite = np.isfinite(points).all(1)
    ids = np.flatnonzero(finite)
    if len(ids) < min_inliers:
        return reject('insufficient_correspondences')
    if np.linalg.matrix_rank(corners[ids]-corners[ids].mean(0)) < 3:
        return reject('degenerate_3d_correspondences')
    singular = np.linalg.svd(points[ids]-points[ids].mean(0), compute_uv=False)
    if singular[0] < 1e-6 or singular[1] < singular[0]*1e-3:
        return reject('collapsed_or_nearly_collinear_observations')
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            corners[ids], points[ids], K, distortion, iterationsCount=iterations,
            reprojectionError=float(reprojection_threshold), confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        if not success or inliers is None or len(inliers) < min_inliers:
            return reject('ransac_insufficient_inliers', inlier_count=0 if inliers is None else len(inliers))
        selected = ids[inliers.ravel()]
        if np.linalg.matrix_rank(corners[selected]-corners[selected].mean(0)) < 3:
            return reject('degenerate_inlier_geometry')
        rvec, tvec = cv2.solvePnPRefineLM(corners[selected], points[selected], K, distortion, rvec, tvec)
        rotation, _ = cv2.Rodrigues(rvec)
        camera_points = corners @ rotation.T + tvec.reshape(1, 3)
        if not np.isfinite(camera_points).all() or (camera_points[:, 2] <= 0).any():
            return reject('invalid_or_behind_camera_pose')
        projected, _ = cv2.projectPoints(corners, rvec, tvec, K, distortion)
    except cv2.error:
        return reject('opencv_solver_failed')
    projected = projected.reshape(8, 2)
    if not np.isfinite(projected).all():
        return reject('nonfinite_projection')
    residual = np.linalg.norm(projected[ids]-points[ids], axis=1)
    supported = ids[residual <= reprojection_threshold]
    if len(supported) < min_inliers:
        return reject('refinement_insufficient_inliers', inlier_count=len(supported))
    if np.linalg.matrix_rank(corners[supported]-corners[supported].mean(0)) < 3:
        return reject('degenerate_refined_geometry')
    return dict(base, accepted=True, status='geometrically_consistent_unverified',
                points=projected.tolist(), rvec=rvec.ravel().tolist(), tvec=tvec.ravel().tolist(),
                rotation=rotation.tolist(), inlier_indices=supported.tolist(), inlier_count=len(supported),
                mean_residual_px=float(residual.mean()),
                inlier_mean_residual_px=float(residual[residual <= reprojection_threshold].mean()))
