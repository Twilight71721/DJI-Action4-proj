"""Tight projected-cuboid crops with texture-free letterboxing.

The recorded crop is a virtual window in original-image coordinates. Only
pixels inside the supplied object rectangle are copied into that window, so
existing crop/intrinsics mappings remain valid without adding scene context.
"""
from __future__ import annotations

import numpy as np

FILL_VALUE = 114
DEFAULT_BORDER = 4


def corner_bounds(points):
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (8, 2) or not np.isfinite(points).all():
        raise ValueError('A tight cuboid crop requires eight finite projected corners')
    return tuple(np.r_[points.min(axis=0), points.max(axis=0)].tolist())


def raster_bounds(box):
    box = np.asarray(box, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError(f'Invalid tight box: {box}')
    # Include the pixel containing each extremal corner, including integer maxima.
    return tuple(np.r_[np.floor(box[:2]), np.floor(box[2:]) + 1].astype(int))


def tight_crop_window(box, input_size=(256, 256), border=DEFAULT_BORDER):
    width, height = map(int, input_size)
    if not isinstance(border, (int, np.integer)) or border < 0 or 2*border >= min(width, height):
        raise ValueError('Invalid input-pixel border')
    left, top, right, bottom = raster_bounds(box)
    units = max((right-left)/(width-2*border), (bottom-top)/(height-2*border))
    crop_w, crop_h = max(2, int(np.ceil(width*units))), max(2, int(np.ceil(height*units)))
    x = int(np.floor((left+right-crop_w)/2))
    y = int(np.floor((top+bottom-crop_h)/2))
    return x, y, crop_w, crop_h


def rectangular_crop_window(box, image_size):
    """Clip a detected body rectangle to the source; never add RGB padding."""
    raster_bounds(box)
    width, height = image_size
    # Detection boxes use continuous xyxy with an exclusive upper endpoint.
    left, top = np.maximum(np.floor(box[:2]), 0).astype(int)
    right, bottom = np.minimum(np.ceil(box[2:]), [width, height]).astype(int)
    if right <= left or bottom <= top:
        raise ValueError('Body rectangle does not intersect the source image')
    return int(left), int(top), int(right-left), int(bottom-top)


def choose_rectangular_size(crop, sizes):
    """Pick the closest aspect bucket; every bucket is resized without padding."""
    sizes = tuple(tuple(map(int, size)) for size in sizes)
    if not sizes or any(len(size) != 2 or min(size) < 32 or any(v % 32 for v in size) for size in sizes):
        raise ValueError('Aspect buckets must be W,H pairs divisible by 32 and >= 32')
    ratio = crop[2] / crop[3]
    return min(sizes, key=lambda size: abs(np.log((size[0]/size[1]) / ratio)))


def extract_window(image, crop, content_box=None, fill=FILL_VALUE):
    """Copy an in-frame intersection; negative crop origins never wrap around."""
    x, y, width, height = map(int, crop)
    if width <= 0 or height <= 0:
        raise ValueError('Crop dimensions must be positive')
    canvas = np.full((height, width, *image.shape[2:]), fill, dtype=image.dtype)
    left, top = max(0, x), max(0, y)
    right, bottom = min(image.shape[1], x+width), min(image.shape[0], y+height)
    if content_box is not None:
        bx1, by1, bx2, by2 = raster_bounds(content_box)
        left, top, right, bottom = max(left, bx1), max(top, by1), min(right, bx2), min(bottom, by2)
    if right > left and bottom > top:
        canvas[top-y:bottom-y, left-x:right-x] = image[top:bottom, left:right]
    return canvas


def fill_outside_box(window, crop, box):
    """Apply letterboxing after augmentation, keeping random backgrounds inside the box."""
    x, y, width, height = map(int, crop)
    if window.shape[:2] != (height, width):
        raise ValueError('Window shape and crop differ')
    left, top, right, bottom = raster_bounds(box)
    local = (left-x, top-y, right-x-1, bottom-y-1)
    # extract_window expects a nondegenerate float box; use direct half-open slicing.
    result = np.full_like(window, FILL_VALUE)
    x1, y1 = max(0, local[0]), max(0, local[1])
    x2, y2 = min(width, local[2]+1), min(height, local[3]+1)
    if x2 > x1 and y2 > y1:
        result[y1:y2, x1:x2] = window[y1:y2, x1:x2]
    return result
