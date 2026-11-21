"""Minimal BOP reader producing per-instance 8-corner label samples.

The reader follows the BOP association rules:

- ``image_id`` indexes ``scene_gt.json`` / ``scene_camera.json`` /
  ``scene_gt_info.json``;
- the array position inside ``scene_gt[image_id]`` is ``instance_idx`` and also
  indexes the mask files ``<image_id:06d>_<instance_idx:06d>.png``.

All geometry is in millimeters; poses map model -> OpenCV camera coordinates.
Frames are the *input* frames actually fed to the network (after the optional
crop / resize); ``K_input`` matches those frames.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .geometry import (
    BOPModel,
    crop_and_scale_intrinsics,
    parse_intrinsics,
    parse_rotation,
    project_corners,
    scale_keypoints_to_heatmap,
)
from .heatmap import DEFAULT_VALUE_RANGE, make_bbox_heatmaps
from .cropping import (corner_bounds, tight_crop_window, extract_window,
                       rectangular_crop_window, choose_rectangular_size)


@dataclass
class BOPInstanceSample:
    """One object instance in one image, with supervision targets."""

    image: np.ndarray            # [H, W, 3] uint8 RGB, the network input frame
    corners_3d: np.ndarray       # [8, 3] float32, model frame, mm
    keypoints_2d: np.ndarray     # [8, 2] float32, input-frame pixels; NaN if Z <= 0
    valid: np.ndarray            # [8] bool, inside prediction canvas and Z > 0
    heatmaps: np.ndarray         # [8, Hh, Wh] float32 in heatmap_value_range
    K_input: np.ndarray          # [3, 3] float64 intrinsics of the input frame
    R_m2c: np.ndarray            # [3, 3] float64
    t_m2c: np.ndarray            # [3] float64, mm
    scene_id: str
    image_id: int
    instance_idx: int
    obj_id: int
    image_path: str
    visib_fract: float
    crop_xywh: tuple[int, int, int, int]
    content_bbox_xyxy: tuple[float, float, float, float] | None = None


class BOPInstanceDataset:
    def __init__(
        self,
        root: str | Path,
        model: BOPModel,
        heatmap_size: tuple[int, int] | None = None,
        input_size: tuple[int, int] | None = None,
        crop: tuple[int, int, int, int] | None = None,
        scenes: list[str] | None = None,
        min_visib_fract: float = 0.0,
        heatmap_value_range: tuple[float, float] = DEFAULT_VALUE_RANGE,
        pose_scale: float = 1.0,
        crop_mode: str = "full",
        crop_padding: float = 1.3,
        heatmap_kernel: str = "boxdreamer",
        heatmap_sigma: float = 2.0,
        crop_jitter_shift: float = 0.0,
        crop_jitter_padding: tuple[float, float] | None = None,
        crop_jitter_seed: int = 0,
        crop_border: int = 4,
        require_all_corners: bool = False,
        heatmap_margin: int = 0,
        rect_sizes: tuple[tuple[int, int], ...] | None = None,
    ) -> None:
        """``heatmap_size`` is ``(Hh, Wh)``; ``input_size`` is ``(W, H)``.

        ``crop`` is ``(x0, y0, wc, hc)`` in the original RGB image; when it is
        omitted the whole image is used.  ``input_size`` defaults to the crop
        size, so no resize happens unless requested.
        """
        if pose_scale <= 0:
            raise ValueError(f"pose_scale must be positive, got {pose_scale}")
        if crop_mode not in {"full", "bbox", "cuboid", "rect"}:
            raise ValueError(f"Unsupported crop_mode: {crop_mode!r}")
        if crop_padding < 1.0:
            raise ValueError(f"crop_padding must be >= 1, got {crop_padding}")
        if crop_mode == "cuboid" and crop_padding != 1.0:
            raise ValueError("cuboid crops require crop_padding=1.0; border is solid-color input pixels")
        if not isinstance(crop_border, (int, np.integer)) or crop_border < 0:
            raise ValueError("crop_border must be a nonnegative integer")
        self.root = Path(root)
        self.model = model
        self.pose_scale = float(pose_scale)
        self.crop_mode = crop_mode
        self.crop_padding = float(crop_padding)
        self.crop_border = int(crop_border)
        self.require_all_corners = bool(require_all_corners)
        if not isinstance(heatmap_margin, int) or heatmap_margin < 0:
            raise ValueError('heatmap_margin must be an integer >= 0')
        if rect_sizes is not None and crop_mode != 'rect':
            raise ValueError('rect_sizes requires crop_mode=rect')
        if crop_mode == 'rect' and (crop_padding != 1.0 or crop_border != 0):
            raise ValueError('rect crops require padding=1.0 and border=0; RGB must not be padded')
        self.heatmap_margin = heatmap_margin
        self.rect_sizes = tuple(tuple(size) for size in rect_sizes) if rect_sizes is not None else None
        if self.rect_sizes is not None:
            choose_rectangular_size((0, 0, 100, 100), self.rect_sizes)
        self._image_sizes = {}
        if not np.isfinite(crop_jitter_shift) or not 0 <= crop_jitter_shift <= 0.5:
            raise ValueError("crop_jitter_shift must be finite and in [0, 0.5]")
        if crop_jitter_padding is not None:
            if (len(crop_jitter_padding) != 2 or not np.isfinite(crop_jitter_padding).all()
                    or not 1 <= crop_jitter_padding[0] <= crop_jitter_padding[1]):
                raise ValueError("crop_jitter_padding must satisfy 1 <= min <= max")
        if (crop_jitter_shift or crop_jitter_padding is not None) and crop_mode != "bbox":
            raise ValueError("crop jitter requires crop_mode='bbox'")
        if crop_jitter_seed < 0:
            raise ValueError("crop_jitter_seed must be nonnegative")
        self.crop_jitter_shift = float(crop_jitter_shift)
        self.crop_jitter_padding = crop_jitter_padding
        self.crop_jitter_seed = int(crop_jitter_seed)
        self.epoch = 0
        self.heatmap_kernel = heatmap_kernel
        self.heatmap_sigma = float(heatmap_sigma)
        self.crop = crop
        if crop is None:
            probe = self._scene_paths(scenes)[0]
            with open(probe / "scene_camera.json") as f:
                cam0 = json.load(f)
            first = cam0[sorted(cam0, key=int)[0]]
            probe_size = (int(first["width"]), int(first["height"])) if "width" in first else None
            if probe_size is None:
                image_candidates = []
                for suffix in ("*.png", "*.jpg", "*.jpeg"):
                    image_candidates.extend((probe / "rgb").glob(suffix))
                if not image_candidates:
                    raise FileNotFoundError(f"no RGB images found under {probe / 'rgb'}")
                image_probe = sorted(image_candidates)[0]
                img = cv2.imread(str(image_probe))
                probe_size = (img.shape[1], img.shape[0])
            self.crop = (0, 0, probe_size[0], probe_size[1])
        x0, y0, wc, hc = self.crop
        self.input_size = (int(input_size[0]), int(input_size[1])) if input_size else (int(wc), int(hc))
        self.heatmap_size = tuple(heatmap_size) if heatmap_size else (self.input_size[1], self.input_size[0])
        self.heatmap_value_range = tuple(heatmap_value_range)
        self._scene_cache: dict[str, tuple[dict, dict, dict]] = {}
        self.instances = self._build_index(scenes, min_visib_fract)

    @staticmethod
    def _bbox_crop(
        bbox: list[float] | tuple[float, ...], image_size: tuple[int, int], padding: float
    ) -> tuple[int, int, int, int]:
        """Return an in-frame square crop around BOP ``bbox_obj``."""
        x, y, w, h = map(float, bbox)
        image_w, image_h = image_size
        side = max(2, int(np.ceil(max(w, h) * padding)))
        side = min(side, image_w, image_h)
        cx, cy = x + 0.5 * w, y + 0.5 * h
        x0 = int(round(cx - 0.5 * side))
        y0 = int(round(cy - 0.5 * side))
        x0 = min(max(x0, 0), image_w - side)
        y0 = min(max(y0, 0), image_h - side)
        return x0, y0, side, side

    def _scene_paths(self, scenes: list[str] | None) -> list[Path]:
        if scenes is not None:
            return [self.root / s for s in scenes]
        return sorted(
            p for p in self.root.iterdir()
            if p.is_dir() and p.name.isdigit() and (p / "scene_gt.json").exists()
        )

    def _load_scene(self, scene_id: str):
        if scene_id not in self._scene_cache:
            base = self.root / scene_id
            with open(base / "scene_gt.json") as f:
                gt = json.load(f)
            with open(base / "scene_camera.json") as f:
                cam = json.load(f)
            with open(base / "scene_gt_info.json") as f:
                info = json.load(f)
            if set(gt) != set(cam) or set(gt) != set(info):
                raise ValueError(f"{scene_id}: image id sets differ across scene_gt/scene_camera/scene_gt_info")
            self._scene_cache[scene_id] = (gt, cam, info)
        return self._scene_cache[scene_id]

    def _build_index(self, scenes: list[str] | None, min_visib_fract: float) -> list[tuple[str, int, int]]:
        index: list[tuple[str, int, int]] = []
        for scene_path in self._scene_paths(scenes):
            scene_id = scene_path.name
            gt, _, info = self._load_scene(scene_id)
            for image_id in sorted(gt, key=int):
                for instance_idx, inst in enumerate(gt[image_id]):
                    if int(inst["obj_id"]) != self.model.obj_id:
                        raise ValueError(
                            f"{scene_id} image {image_id} instance {instance_idx}: "
                            f"obj_id={inst['obj_id']} does not match model obj_id={self.model.obj_id}"
                        )
                    visib = float(info[image_id][instance_idx].get("visib_fract", 1.0))
                    if visib < min_visib_fract:
                        continue
                    index.append((scene_id, int(image_id), instance_idx))
        return index

    def __len__(self) -> int:
        return len(self.instances)

    def set_epoch(self, epoch: int) -> None:
        """Call before creating the epoch iterator; do not use persistent workers."""
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def get_input_size(self, index: int) -> tuple[int, int]:
        """Read only metadata/image headers for deterministic shape batching."""
        if self.rect_sizes is None:
            return self.input_size
        scene, frame, instance = self.instances[index]
        _, cameras, info = self._load_scene(scene)
        camera = cameras[str(frame)]
        key = (scene, frame)
        if key not in self._image_sizes:
            if 'width' in camera and 'height' in camera:
                self._image_sizes[key] = (int(camera['width']), int(camera['height']))
            else:
                folder = self.root / scene / 'rgb'
                path = next(p for suffix in ('.png', '.jpg', '.jpeg')
                            if (p := folder / f'{frame:06d}{suffix}').is_file())
                with Image.open(path) as image:
                    self._image_sizes[key] = image.size
        x, y, w, h = info[str(frame)][instance]['bbox_obj']
        crop = rectangular_crop_window((x, y, x+w, y+h), self._image_sizes[key])
        return choose_rectangular_size(crop, self.rect_sizes)

    def _jitter_crop(self, bbox, image_size, baseline, K, R, t, identity):
        if self.crop_jitter_shift == 0 and self.crop_jitter_padding is None:
            return baseline
        # A local generator leaves sampler/model randomness identical in A/B runs.
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.crop_jitter_seed, self.epoch, *map(int, identity)]))
        original, _, _ = project_corners(self.model.corners_mm(), K, R, t, image_size=image_size)
        _, protected, _ = project_corners(
            self.model.corners_mm(), crop_and_scale_intrinsics(K, baseline, self.input_size),
            R, t, image_size=self.input_size)
        padding_range = self.crop_jitter_padding or (self.crop_padding, self.crop_padding)
        for _ in range(8):
            x, y, w, h = map(float, bbox)
            dx, dy = rng.uniform(-self.crop_jitter_shift, self.crop_jitter_shift, 2)
            candidate = self._bbox_crop(
                (x + dx * w, y + dy * h, w, h), image_size, rng.uniform(*padding_range))
            cx, cy, cw, ch = candidate
            points = (original - [cx, cy]) * (np.array(self.input_size) / [cw, ch])
            inside = (np.isfinite(points).all(axis=1) & (points >= 0).all(axis=1)
                      & (points < self.input_size).all(axis=1))
            if inside[protected].all():
                return candidate
        # Do not remove previously supervised corners to obtain an easier loss.
        return baseline

    def load_instance(self, scene_id: str, image_id: int, instance_idx: int) -> BOPInstanceSample:
        gt, cam, info = self._load_scene(scene_id)
        entry = gt[str(image_id)][instance_idx]
        R_m2c = parse_rotation(entry["cam_R_m2c"])
        t_m2c = np.asarray(entry["cam_t_m2c"], dtype=np.float64).reshape(3) * self.pose_scale
        K = parse_intrinsics(cam[str(image_id)]["cam_K"])

        rgb_dir = self.root / scene_id / "rgb"
        candidates = [rgb_dir / f"{image_id:06d}{suffix}" for suffix in (".png", ".jpg", ".jpeg")]
        image_path = next((path for path in candidates if path.exists()), candidates[0])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        content_box = None
        input_size = self.input_size
        if self.crop_mode == "bbox":
            crop = self._bbox_crop(
                info[str(image_id)][instance_idx]["bbox_obj"],
                (image.shape[1], image.shape[0]),
                self.crop_padding,
            )
            crop = self._jitter_crop(
                info[str(image_id)][instance_idx]["bbox_obj"],
                (image.shape[1], image.shape[0]), crop, K, R_m2c, t_m2c,
                (scene_id, image_id, instance_idx))
        elif self.crop_mode == "cuboid":
            original, _, _ = project_corners(self.model.corners_mm(), K, R_m2c, t_m2c,
                                             image_size=(image.shape[1], image.shape[0]))
            content_box = corner_bounds(original)
            crop = tight_crop_window(content_box, self.input_size, self.crop_border)
        elif self.crop_mode == "rect":
            bx, by, bw, bh = info[str(image_id)][instance_idx]["bbox_obj"]
            crop = rectangular_crop_window((bx, by, bx+bw, by+bh), (image.shape[1], image.shape[0]))
            if self.rect_sizes is not None:
                input_size = choose_rectangular_size(crop, self.rect_sizes)
        else:
            crop = self.crop
        K_input = crop_and_scale_intrinsics(K, crop, input_size)
        x0, y0, wc, hc = crop
        image = (extract_window(image, crop, content_box) if content_box is not None
                 else image[y0:y0 + hc, x0:x0 + wc])
        if (image.shape[1], image.shape[0]) != input_size:
            interpolation = cv2.INTER_AREA if input_size[0] * input_size[1] < wc * hc else cv2.INTER_LINEAR
            image = cv2.resize(image, input_size, interpolation=interpolation)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        corners_3d = self.model.corners_mm()
        keypoints_2d, valid, _ = project_corners(corners_3d, K_input, R_m2c, t_m2c, image_size=input_size)
        input_hw = (input_size[1], input_size[0])
        core_heatmap_size = ((input_hw[0]//4, input_hw[1]//4)
                             if self.rect_sizes is not None else self.heatmap_size)
        heatmap_kps = scale_keypoints_to_heatmap(keypoints_2d, input_hw, core_heatmap_size)
        heatmap_kps += self.heatmap_margin
        heatmap_size = tuple(size + 2*self.heatmap_margin for size in core_heatmap_size)
        if self.heatmap_margin:
            valid = (np.isfinite(heatmap_kps).all(1) & (heatmap_kps >= 0).all(1)
                     & (heatmap_kps < np.array(heatmap_size[::-1])).all(1))
        if self.require_all_corners and not valid.all():
            raise ValueError(
                f'{scene_id}/{image_id}/{instance_idx}: corners {np.flatnonzero(~valid).tolist()} '
                'outside the prediction canvas or behind the camera; calibrate a fixed canvas '
                'on training data instead of silently dropping corner supervision')
        heatmaps = make_bbox_heatmaps(
            heatmap_kps,
            valid,
            heatmap_size,
            self.heatmap_value_range,
            kernel=self.heatmap_kernel,
            sigma=self.heatmap_sigma,
        )

        return BOPInstanceSample(
            image=image,
            corners_3d=corners_3d.astype(np.float32),
            keypoints_2d=keypoints_2d.astype(np.float32),
            valid=valid,
            heatmaps=heatmaps,
            K_input=K_input,
            R_m2c=R_m2c,
            t_m2c=t_m2c,
            scene_id=scene_id,
            image_id=int(image_id),
            instance_idx=int(instance_idx),
            obj_id=int(entry["obj_id"]),
            image_path=str(image_path),
            visib_fract=float(info[str(image_id)][instance_idx].get("visib_fract", 1.0)),
            crop_xywh=crop,
            content_bbox_xyxy=content_box,
        )

    def __getitem__(self, index: int) -> BOPInstanceSample:
        scene_id, image_id, instance_idx = self.instances[index]
        return self.load_instance(scene_id, image_id, instance_idx)
