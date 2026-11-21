"""Select ordered heatmap peaks jointly using an uncalibrated cuboid prior."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

from .rigid import fit_rigid_cuboid


def peak_candidates(heatmaps, input_size=(256, 256), max_peaks=6, min_ratio=.05):
    """Return separated local peaks, refined within a 3x3 neighborhood.

    Heights are relative within a channel, not calibrated probabilities.
    input_size follows (height, width), matching the heatmap decoder API.
    """
    maps = np.asarray(heatmaps, dtype=np.float64)
    if maps.ndim != 3 or maps.shape[0] != 8 or not np.isfinite(maps).all() or (maps < 0).any():
        raise ValueError("Expected eight finite nonnegative heatmaps [8,H,W]")
    if max_peaks < 1 or not 0 < min_ratio <= 1:
        raise ValueError("Invalid peak extraction settings")
    h,w = maps.shape[1:]
    scale = np.array([input_size[1]/w,input_size[0]/h])
    results = []
    for channel in maps:
        peak = channel.max()
        if peak <= 0 or np.ptp(channel) <= peak*1e-6:
            results.append(dict(points=[],ratios=[]))
            continue
        mask = (channel == maximum_filter(channel,size=3,mode='constant')) & (channel >= peak*min_ratio)
        ys,xs = np.where(mask)
        order = np.argsort(-channel[ys,xs],kind='stable')
        points,ratios,chosen = [],[],[]
        for idx in order:
            y,x = int(ys[idx]),int(xs[idx])
            if any((x-a)**2+(y-b)**2 < 9 for a,b in chosen):
                continue
            y0,y1,x0,x1 = max(0,y-1),min(h,y+2),max(0,x-1),min(w,x+2)
            weights = channel[y0:y1,x0:x1]
            yy,xx = np.mgrid[y0:y1,x0:x1]
            point = np.array([(xx*weights).sum(),(yy*weights).sum()])/weights.sum()*scale
            points.append(point.tolist())
            ratios.append(float(channel[y,x]/peak))
            chosen.append((x,y))
            if len(points) >= max_peaks:
                break
        results.append(dict(points=points,ratios=ratios))
    return results


def decode_joint_cuboid(heatmaps, corners, *, input_size=(256,256), box_xyxy=None,
                        beam_width=256, max_peaks=6, geometric_sigma=8., refine_count=12):
    """Beam search over local peaks, then refine promising rigid hypotheses.

    Eight channel identities remain fixed. A detection box contributes only
    a broad minimum-span prior (35% of its long side), not a bbox edge fit.
    Rejected hypotheses are retained for diagnosis but must not be presented
    as supported detections. No ground truth or camera intrinsics are inputs.
    """
    if beam_width < 1 or refine_count < 1 or geometric_sigma <= 0:
        raise ValueError("Search sizes and geometric_sigma must be positive")
    corners = np.asarray(corners,dtype=np.float64)
    if corners.shape != (8,3) or not np.isfinite(corners).all() or np.linalg.matrix_rank(corners-corners.mean(0)) != 3:
        raise ValueError("Expected eight finite noncoplanar model corners")
    candidates = peak_candidates(heatmaps,input_size,max_peaks)
    if any(not c['points'] for c in candidates):
        return dict(status='insufficient_heatmap_evidence',accepted=False,candidates=candidates,
                    selected_points=None,rigid_points=None,calibrated_pose=False)
    locations=[np.array(c['points']) for c in candidates]
    penalties=[-2*np.log(np.array(c['ratios'])) for c in candidates]
    minimum_span=0.
    if box_xyxy is not None:
        box=np.asarray(box_xyxy,dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or (box[2:]<=box[:2]).any():
            raise ValueError("Expected a finite nonempty xyxy rectangle")
        minimum_span=.35*max(box[2:]-box[:2])
    model=(corners-corners.mean(0))/np.linalg.norm(np.ptp(corners,axis=0))
    order=np.array([0,1,3,4,2,5,7,6])
    beams=np.empty((1,0),dtype=int)
    likelihood=np.zeros(1)
    for n,channel in enumerate(order,1):
        count=len(locations[channel])
        beams=np.c_[np.repeat(beams,count,axis=0),np.tile(np.arange(count),len(beams))]
        likelihood=np.repeat(likelihood,count)+np.tile(penalties[channel],len(likelihood))
        score=likelihood.copy()
        if n>=4:
            observed=np.stack([locations[j][beams[:,k]] for k,j in enumerate(order[:n])],axis=1)
            design=np.c_[model[order[:n]],np.ones(n)]
            coefficients=np.einsum('ij,bjk->bik',np.linalg.pinv(design),observed)
            matrix=coefficients[:,:3,:].transpose(0,2,1)
            u,s,v=np.linalg.svd(matrix,full_matrices=False)
            rows=u@v
            size=s.mean(1)
            projected=np.einsum('ij,bkj->bik',model,rows)*size[:,None,None]
            translation=observed.mean(1)-projected[:,order[:n]].mean(1)
            projected+=translation[:,None,:]
            score+=np.square(projected[:,order[:n]]-observed).sum((1,2))/geometric_sigma**2
            span=np.ptp(projected,axis=1).max(1)
            score+=8*np.square(np.maximum(0,minimum_span-span)/geometric_sigma)
        keep=np.argsort(score,kind='stable')[:beam_width]
        beams,likelihood=beams[keep],likelihood[keep]

    hypotheses=[]
    for beam in beams[:refine_count]:
        assignment=np.empty(8,dtype=int)
        assignment[order]=beam
        for _ in range(3):
            points=np.array([locations[j][assignment[j]] for j in range(8)])
            fit=fit_rigid_cuboid(corners,points,starts=1)
            if fit['points'] is None:
                break
            projected=np.array(fit['points'])
            updated=np.array([np.argmin(np.square(locations[j]-projected[j]).sum(1)/geometric_sigma**2+penalties[j]) for j in range(8)])
            if np.array_equal(updated,assignment):
                break
            assignment=updated
        points=np.array([locations[j][assignment[j]] for j in range(8)])
        fit=fit_rigid_cuboid(corners,points,starts=1)
        if fit['points'] is None:
            continue
        projected=np.array(fit['points'])
        differences=np.linalg.norm(points-projected,axis=1)
        span=float(np.ptp(projected,axis=0).max())
        cost=float(np.square(differences/geometric_sigma).sum()+sum(penalties[j][assignment[j]] for j in range(8)))
        cost+=8*(max(0,minimum_span-span)/geometric_sigma)**2
        hypotheses.append((cost,assignment,points,projected,fit,differences,span))
    if not hypotheses:
        return dict(status='collapsed_candidates',accepted=False,candidates=candidates,
                    selected_points=None,rigid_points=None,calibrated_pose=False)
    cost,assignment,points,projected,fit,differences,span=min(hypotheses,key=lambda h:h[0])
    supported=int((differences<=geometric_sigma*1.5).sum())
    accepted=span>=minimum_span and supported>=6
    status='joint_hypothesis_unverified' if accepted else 'rejected_inconsistent_heatmaps'
    return dict(status=status,accepted=bool(accepted),selected_points=points.tolist(),rigid_points=projected.tolist(),
                selected_indices=assignment.tolist(),candidates=candidates,objective=cost,
                changed_channels=int((assignment!=0).sum()),supported_channels=supported,
                mean_peak_to_rigid_px=float(differences.mean()),minimum_span_prior_px=float(minimum_span),
                fitted_span_px=span,calibrated_pose=False,fit=fit,
                note='accepted is a heuristic consistency check, not calibrated correctness')
