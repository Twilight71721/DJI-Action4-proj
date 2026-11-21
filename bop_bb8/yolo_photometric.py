"""Image-only augmentation for YOLO detection after geometry and before tensor formatting."""
import cv2
import numpy as np
from ultralytics.data.augment import Format
from ultralytics.models.yolo.detect.train import DetectionTrainer

from .photometric import augment_rgb


class YOLOPhotometric:
    def __call__(self, labels):
        # Ultralytics seeds NumPy independently in data-loader workers.
        rng = np.random.default_rng(np.random.randint(0, 2**32, dtype=np.uint32))
        rgb = cv2.cvtColor(labels['img'], cv2.COLOR_BGR2RGB)
        rgb, _ = augment_rgb(rgb, rng, profile='balanced')
        result = dict(labels)
        result['img'] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return result


class PhotometricDetectionTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        if mode == 'train':
            if self.args.close_mosaic != 0:
                raise ValueError('Use close_mosaic=0 so Ultralytics does not rebuild and remove the photometric transform')
            if not isinstance(dataset.transforms.transforms[-1], Format):
                raise RuntimeError('Unexpected YOLO transform order: final transform must be Format')
            dataset.transforms.insert(-1, YOLOPhotometric())
            print('[photometric] balanced: 20% unchanged, 68% gentle, 12% moderate; training only', flush=True)
        return dataset
