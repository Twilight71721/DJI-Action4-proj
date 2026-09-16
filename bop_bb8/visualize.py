"""Draw a label sample's wireframe and numbered corners onto its input frame."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .dataset import BOPInstanceSample

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

CORNER_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
]


def draw_sample(
    sample: BOPInstanceSample,
    bbox_obj: tuple[float, float, float, float] | None = None,
    out_path: str | Path | None = None,
) -> np.ndarray:
    """Return an RGB overlay with edges, corner indices and optional ``bbox_obj``."""
    image = np.ascontiguousarray(sample.image[:, :, ::-1].copy())
    for a, b in EDGES:
        if sample.valid[a] and sample.valid[b]:
            pa = tuple(np.round(sample.keypoints_2d[a]).astype(int))
            pb = tuple(np.round(sample.keypoints_2d[b]).astype(int))
            cv2.line(image, pa, pb, CORNER_COLORS[a], 2, cv2.LINE_AA)
    for i in range(8):
        if not sample.valid[i]:
            continue
        p = tuple(np.round(sample.keypoints_2d[i]).astype(int))
        cv2.circle(image, p, 3, CORNER_COLORS[i], -1, cv2.LINE_AA)
        cv2.putText(image, str(i), (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, CORNER_COLORS[i], 2, cv2.LINE_AA)
    if bbox_obj is not None:
        x, y, w, h = (int(round(v)) for v in bbox_obj)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), image)
    return image[:, :, ::-1]
