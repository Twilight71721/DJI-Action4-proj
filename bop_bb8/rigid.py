"""Diagnostic cuboid fitting under scaled orthographic projection.

No focal length is assumed or recovered. The returned rotation and scale are
weak-perspective fit parameters, not a calibrated metric 6D pose. Optional
rectangle residuals are explicit priors, not additional corner observations.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def project_weak_perspective(corners, rotation, scale, center):
    corners = np.asarray(corners, dtype=np.float64)
    centered = corners - corners.mean(axis=0)
    return scale * (centered @ np.asarray(rotation).T)[:, :2] + center


def fit_rigid_cuboid(corners, points, *, box_xyxy=None, box_weight=2.0, starts=12):
    """Fit ordered 3D corners to ordered 2D points with optional bbox prior.

    Nonfinite observations are ignored, with at least four noncoplanar model
    correspondences required. A prior-dominated proposal remains explicitly
    marked; callers must not interpret ``optimizer_success`` as pose accuracy.
    Input pixel units should be consistent (the diagnostic uses 256px crops).
    """
    corners = np.asarray(corners, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if corners.shape != (8, 3) or points.shape != (8, 2):
        raise ValueError("Expected corners [8,3] and points [8,2]")
    if not np.isfinite(corners).all() or np.linalg.matrix_rank(corners-corners.mean(0)) != 3:
        raise ValueError("Model corners must be finite and span three dimensions")
    if starts < 1 or box_weight < 0:
        raise ValueError("starts must be positive and box_weight nonnegative")
    valid = np.isfinite(points).all(axis=1)
    if valid.sum() < 4 or np.linalg.matrix_rank(corners[valid]-corners[valid].mean(0)) < 3:
        return dict(status="insufficient_correspondences", points=None)
    observed = points[valid]
    extent = float(np.ptp(observed, axis=0).max())
    box = None
    if box_xyxy is not None:
        box = np.asarray(box_xyxy, dtype=np.float64)
        if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= box[:2]).any():
            raise ValueError("Expected a finite nonempty xyxy box")
        if box_weight == 0:
            box = None
    if extent < 1e-6 and box is None:
        return dict(status="collapsed_observations", points=None)

    unit = np.linalg.norm(np.ptp(corners, axis=0))
    model = (corners-corners.mean(0))/unit
    center = observed.mean(0)
    affine = np.linalg.lstsq(model[valid]-model[valid].mean(0), observed-center, rcond=None)[0].T
    u, _, vt = np.linalg.svd(affine, full_matrices=False)
    rows = u @ vt
    initial_rotation = np.vstack([rows, np.cross(rows[0], rows[1])])
    rotations = [initial_rotation]
    rotations.extend(Rotation.random(max(0, starts-1), random_state=20260923).as_matrix())
    reference_extent = float(max(extent, 1.0)) if box is None else float(max(box[2:]-box[:2]))
    # Broad numerical bounds do not fix the proposed box to a chosen size.
    scale_bounds = (np.log(1e-4), np.log(reference_extent*100))

    def project(parameters):
        rotated = model @ Rotation.from_rotvec(parameters[:3]).as_matrix().T
        return np.exp(parameters[3])*rotated[:, :2] + parameters[4:6]

    def residual(parameters):
        projected = project(parameters)
        error = (projected[valid]-observed).ravel()
        if box is not None and box_weight > 0:
            bounds = np.r_[projected.min(0), projected.max(0)]
            error = np.r_[error, box_weight*(bounds-box)]
        return error

    candidates = []
    for rotation in rotations:
        projected = (model @ rotation.T)[:, :2]
        if box is None:
            q = projected[valid]-projected[valid].mean(0)
            scale = max(1e-3, np.sum(q*(observed-center))/max(np.sum(q*q),1e-12))
            translation = center-scale*projected[valid].mean(0)
        else:
            width = np.ptp(projected,axis=0)
            scale = float(np.dot(width,box[2:]-box[:2])/max(np.dot(width,width),1e-12))
            translation = (box[:2]+box[2:])/2-scale*(projected.min(0)+projected.max(0))/2
        initial = np.r_[Rotation.from_matrix(rotation).as_rotvec(),np.clip(np.log(scale),*scale_bounds),translation]
        result = least_squares(residual,initial,bounds=(np.r_[[-np.inf]*3,scale_bounds[0],[-np.inf]*2],
                                                       np.r_[[np.inf]*3,scale_bounds[1],[np.inf]*2]),
                               loss="soft_l1",f_scale=max(1.,reference_extent*.03),max_nfev=180)
        candidates.append(result)
    result = min(candidates,key=lambda x:x.cost)
    projected = project(result.x)
    displacement = float(np.linalg.norm(projected[valid]-observed,axis=1).mean())
    normalized_shift = displacement/reference_extent
    if box is not None and extent < .25*reference_extent:
        status = "prior_dominated_collapsed_input"
    elif normalized_shift > .10:
        status = "large_correction_unverified"
    else:
        status = "geometric_fit_unverified"
    return dict(status=status,points=projected.tolist(),rotation=Rotation.from_rotvec(result.x[:3]).as_matrix().tolist(),
                scale_px_per_model_unit=float(np.exp(result.x[3])/unit),center=result.x[4:6].tolist(),
                mean_corner_correction_px=displacement,normalized_correction=normalized_shift,
                uses_box_prior=box is not None and box_weight>0,box_weight=float(box_weight),
                optimizer_success=bool(result.success),objective=float(result.cost),
                calibrated_pose=False,projection="scaled_orthographic")
