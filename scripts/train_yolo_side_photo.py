"""Fine-tune a separate YOLO candidate with balanced photometry; preserve the baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bop_bb8.yolo_photometric import PhotometricDetectionTrainer, YOLOPhotometric

import torch
import ultralytics
from ultralytics import YOLO


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--prediction-limit", type=int, default=0,
                        help="Limit saved preview images per split; validation metrics still use all images")
    parser.add_argument('--smoke-only',action='store_true')
    args = parser.parse_args()
    if args.prediction_limit < 0:
        parser.error('--prediction-limit must be nonnegative')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "train").exists():
        raise FileExistsError("Training output exists; use a new output directory")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    torch.set_num_threads(4)
    started = time.time()
    state = {"state": "starting", "pid": os.getpid(), "epoch": 0, "epochs": args.epochs}
    status_path = output / "status.json"
    write_json(status_path, state)
    config = dict(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                  batch=args.batch, device=0, workers=2, seed=20260927,
                  optimizer="AdamW", lr0=args.lr0, lrf=0.01, cos_lr=True,
                  patience=args.patience, warmup_epochs=1, amp=True, cache=False,
                  mosaic=0.0, close_mosaic=0, mixup=0.0, degrees=3.0,
                  translate=0.04, scale=0.15, fliplr=0.0, flipud=0.0,
                  hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                  project=str(output), name="train", plots=True, save=True,
                  verbose=False)
    write_json(output / "run_config.json", dict(
        train=config, ultralytics=ultralytics.__version__, torch=torch.__version__,
        weights=str(args.weights.resolve()),
        weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        data_manifest_sha256=hashlib.sha256((args.data.parent / "manifest.json").read_bytes()).hexdigest(),
        saved_prediction_limit_per_split=args.prediction_limit,
        photometric_profile="balanced", active_model_unchanged=True, smoke_only=args.smoke_only,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/"bop_bb8/yolo_photometric.py",ROOT/"bop_bb8/photometric.py"]},
        evaluation="best.pt selected using validation only; test evaluated once after training"))

    def on_epoch_end(trainer):
        state.update(state="training", epoch=trainer.epoch + 1,
                     elapsed_seconds=time.time() - started, metrics=trainer.metrics,
                     peak_torch_vram_gb=torch.cuda.max_memory_allocated() / 1024**3)
        write_json(status_path, state)

    try:
        model = YOLO(str(args.weights.resolve()))
        model.add_callback("on_fit_epoch_end", on_epoch_end)
        def verify_training_transform(trainer):
            transforms=trainer.train_loader.dataset.transforms.transforms
            if not any(isinstance(t,YOLOPhotometric) for t in transforms):
                raise RuntimeError('Photometric transform missing from actual training loader')
            manifest=json.loads((args.data.parent/'manifest.json').read_text())
            expected=manifest.get('summary',{}).get('training_entries')
            if expected is not None and len(trainer.train_loader.dataset)!=expected:
                raise RuntimeError('Training sampler size differs from the prepared manifest')
            write_json(output/'augmentation_verified.json',dict(
                transforms=[type(t).__name__ for t in transforms],profile='balanced',training_only=True,
                actual_train_entries=len(trainer.train_loader.dataset)))
        model.add_callback('on_train_start',verify_training_transform)
        model.train(trainer=PhotometricDetectionTrainer,**config)
        if hashlib.sha256(args.weights.read_bytes()).hexdigest()!=json.loads((output/'run_config.json').read_text())['weights_sha256']:
            raise RuntimeError('Baseline weights changed unexpectedly')
        if args.smoke_only:
            state.update(state='smoke_completed',elapsed_seconds=time.time()-started)
            write_json(status_path,state)
            return
        best = output / "train/weights/best.pt"
        detector = YOLO(str(best))
        metrics = {}
        for split in ("val", "test"):
            state.update(state=f"evaluating_{split}", elapsed_seconds=time.time() - started)
            write_json(status_path, state)
            result = detector.val(data=str(args.data.resolve()), split=split,
                                  imgsz=args.imgsz, batch=args.batch, device=0, workers=2,
                                  project=str(output), name=f"{split}_metrics", plots=True,
                                  verbose=False)
            metrics[split] = {k: float(v) for k, v in result.results_dict.items()}
            images = args.data.resolve().parent / "images" / split
            preview_source = str(images)
            if args.prediction_limit:
                preview_source = [str(p) for p in sorted(images.iterdir())
                                  if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}][:args.prediction_limit]
            for _ in detector.predict(source=preview_source, imgsz=args.imgsz, conf=0.25,
                                      iou=0.7, device=0, stream=True, save=True,
                                      save_txt=True, save_conf=True, verbose=False,
                                      project=str(output), name=f"{split}_predictions"):
                pass
        write_json(output / "metrics.json", metrics)
        state.update(state="completed", elapsed_seconds=time.time() - started,
                     best_weights=str(best), evaluation=metrics)
        write_json(status_path, state)
    except Exception:
        state.update(state="failed", elapsed_seconds=time.time() - started,
                     error=traceback.format_exc())
        write_json(status_path, state)
        raise


if __name__ == "__main__":
    main()
