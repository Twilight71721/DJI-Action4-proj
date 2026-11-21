"""Full-eight real supervision and metrics in a fixed reference crop."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RealCornerDataset(Dataset):
    def __init__(self, root, split):
        if split not in {'adapt', 'validation', 'test'}:
            raise ValueError(f'Unknown real split: {split}')
        root = Path(root)
        document = json.loads((root / 'annotations.json').read_text())
        rows = document['records']
        with np.load(root / 'full8_supervision.npz', allow_pickle=False) as archive:
            if archive['id'].tolist() != [r['id'] for r in rows]:
                raise ValueError('Annotation/archive identity mismatch')
            if archive['split'].tolist() != [r['split'] for r in rows]:
                raise ValueError('Annotation/archive split mismatch')
            indices = np.flatnonzero(archive['split'] == split)
            self.records = [rows[i] for i in indices]
            self.tensors = {k: torch.from_numpy(archive[k][indices].copy())
                            for k in ('image', 'heatmap', 'valid', 'keypoints_2d')}
        if not len(indices):
            raise ValueError(f'No instances in {split}')
        if not self.tensors['valid'].all() or self.tensors['valid'].shape[1] != 8:
            raise ValueError('Every real instance must supervise all eight corners')
        if not (self.tensors['heatmap'].amax(dim=(-2, -1)) > .999).all():
            raise ValueError('Empty real target heatmap')
        expected = np.array([r['keypoints_input'] for r in self.records])
        if not np.allclose(self.tensors['keypoints_2d'].numpy(), expected, atol=1e-4):
            raise ValueError('Real target coordinates changed')
        self.split = split

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.tensors.items()}


def original_to_reference(points, reference_crop, input_size=256):
    crop = np.asarray(reference_crop, dtype=float)
    return (np.asarray(points) - crop[:2]) * input_size / crop[2:]


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(wh))
    union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection)
    return intersection / max(union, 1e-12)


def match_boxes(targets, predictions, threshold=.5):
    """One-to-one IoU matching; a missed detector never falls back to a GT box."""
    candidates = sorted([(box_iou(a, b), i, j) for i, a in enumerate(targets)
                         for j, b in enumerate(predictions)], reverse=True)
    matches, used = {}, set()
    for overlap, i, j in candidates:
        if overlap >= threshold and i not in matches and j not in used:
            matches[i] = (j, overlap)
            used.add(j)
    return matches


def error_statistics(errors):
    """Missing predictions count as PCK failures; conditional mean is explicit."""
    values = np.asarray(errors, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    return dict(corners=len(values), predicted_corners=len(finite),
                mean_error_px=float(finite.mean()) if len(finite) else None,
                median_error_px=float(np.median(finite)) if len(finite) else None,
                p95_error_px=float(np.percentile(finite, 95)) if len(finite) else None,
                pck5=float(np.sum(values <= 5) / len(values)) if len(values) else None,
                pck10=float(np.sum(values <= 10) / len(values)) if len(values) else None)


def summarize_real(records, model_name, mode, split, matched_only=False):
    selected = [r for r in records if r['split'] == split and
                (not matched_only or r['yolo_box'] is not None)]
    errors, original_errors, visible, inferred = [], [], [], []
    detected = 0
    for r in selected:
        prediction = r['predictions'][model_name][mode]
        if prediction is None:
            err, orig = np.full(8, np.inf), np.full(8, np.inf)
        else:
            detected += 1
            err = np.array(prediction['errors_reference_px'])
            orig = np.array(prediction['errors_original_px'])
        basis = np.array(r['point_basis']) == 'visible_edge_reference'
        errors.extend(err)
        original_errors.extend(orig)
        visible.extend(err[basis])
        inferred.extend(err[~basis])
    return dict(instances=len(selected), predicted_instances=detected,
                all8=error_statistics(errors), visible_reference=error_statistics(visible),
                inferred=error_statistics(inferred), original_pixels=error_statistics(original_errors))
