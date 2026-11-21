"""Bounded real adaptation with a matched-update synthetic-only control."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8 import BOPInstanceDataset, BOPModel
from bop_bb8.net import coordinate_loss, focal_loss
from bop_bb8.real_adaptation import RealCornerDataset
from bop_bb8.torch_data import IMAGENET_MEAN, IMAGENET_STD
from evaluate_cleanblur5000 import load_model
from train_heatmap import evaluate

KEYS = ('image', 'heatmap', 'valid', 'keypoints_2d')


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


class ReplayDataset(Dataset):
    def __init__(self, bank):
        self.bank = bank
        self.mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
        self.std = torch.tensor(IMAGENET_STD)[:, None, None]

    def __len__(self):
        return len(self.bank['image'])

    def __getitem__(self, index):
        result = {k: v[index] for k, v in self.bank.items()}
        result['image'] = (result['image'].float() / 255 - self.mean) / self.std
        return result


def prepare_replay(source, output, seed):
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    families = {j['scene']: j['family'] for j in manifest['jobs']}
    model_info = BOPModel.from_models_info(source / 'models/models_info.json', 1)
    rng = np.random.default_rng(seed)
    banks, selected = {}, {}
    for split, per_family in [('train', 20), ('val', 5)]:
        sets = {variant: BOPInstanceDataset(source / variant, model_info,
                scenes=manifest['splits'][split], input_size=(256, 256),
                heatmap_size=(64, 64), crop_mode='bbox', crop_padding=1.3,
                heatmap_kernel='gaussian', heatmap_sigma=2.) for variant in ('clean', 'blur')}
        if sets['clean'].instances != sets['blur'].instances:
            raise ValueError('Clean/blur instance identities differ')
        groups = defaultdict(list)
        for i, (scene, frame, instance) in enumerate(sets['clean'].instances):
            groups[families[scene]].append(i)
        indices = []
        for family in sorted(groups):
            if len(groups[family]) < per_family:
                raise ValueError(f'Not enough examples in {split} family {family}')
            indices.extend(rng.choice(groups[family], per_family, replace=False).tolist())
        items, identities = [], []
        for variant, base in sets.items():
            for i in indices:
                sample = base[i]
                if not np.isfinite(sample.keypoints_2d[sample.valid]).all():
                    raise ValueError('Invalid synthetic corner coordinates')
                items.append(dict(image=torch.from_numpy(sample.image.copy()).permute(2, 0, 1),
                                  heatmap=torch.from_numpy(sample.heatmaps.copy()).float(),
                                  valid=torch.from_numpy(sample.valid.copy()).bool(),
                                  keypoints_2d=torch.from_numpy(np.nan_to_num(sample.keypoints_2d)).float()))
                identities.append(dict(scene=sample.scene_id, frame=sample.image_id,
                                       instance=sample.instance_idx, family=families[sample.scene_id],
                                       variant=variant, source=sample.image_path))
        banks[split] = {k: torch.stack([x[k] for x in items]) for k in KEYS}
        selected[split] = identities
        print(f'[replay] {split}: {len(items)} crops, {len(groups)} families, paired clean/blur', flush=True)
    if {r['scene'] for r in selected['train']} & {r['scene'] for r in selected['val']}:
        raise ValueError('Synthetic scene leakage')
    torch.save(banks, output / 'synthetic_replay.pt')
    write_json(output / 'synthetic_replay.json', dict(
        source=str(source), source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        seed=seed, instances=selected,
        note='Stratified replay subset, not the full server continuous-pose dataset. 20 train and 5 validation base instances per family, both clean and blur. No synthetic test examples.'))
    return banks


def criterion(logits, targets, valid, points):
    logits = logits.float()
    return focal_loss(logits, targets, valid) + 10 * coordinate_loss(logits, points, valid, (256, 256))


def metrics(model, dataset):
    values = evaluate(model, DataLoader(dataset, batch_size=8, shuffle=False), torch.device('cuda'),
                      criterion, (64, 64), (256, 256), False)
    return dict(zip(('loss', 'mean_error_px', 'pck5'), values))


def collate(dataset, indices):
    rows = [dataset[int(i)] for i in indices]
    return {k: torch.stack([r[k] for r in rows]) for k in KEYS}


def train_arm(name, args, replay, real_train, real_val, state):
    output = args.output / name
    output.mkdir(exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, initial_meta = load_model(args.checkpoint, torch.device('cuda'))
    head = [p for n, p in model.named_parameters() if not n.startswith('backbone.')]
    optimizer = torch.optim.AdamW([dict(params=head, lr=1e-4),
                                  dict(params=model.backbone.parameters(), lr=1e-5)], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    config = dict(initial_meta['args'])
    config.update(real_annotations=str(args.real), synthetic_replay=str(args.output/'synthetic_replay.pt'),
                  adaptation_arm=name, epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
                  batch_size=8, lr=1e-4, backbone_lr_scale=.1, freeze_epochs=2,
                  batch_norm='frozen running statistics in both arms', amp_dtype='bfloat16',
                  seed=args.seed, initial_checkpoint=str(args.checkpoint),
                  initial_checkpoint_sha256=initial_meta['sha256'],
                  online_background_config=None, crop_jitter_shift=0., crop_jitter_padding=None)
    write_json(output/'config.json', config)
    train_syn, val_syn = ReplayDataset(replay['train']), ReplayDataset(replay['val'])
    initial = dict(real_adapt=metrics(model, real_train), real_validation=metrics(model, real_val),
                   synthetic_validation=metrics(model, val_syn))
    write_json(output/'baseline.json', initial)
    best_error, best_epoch = initial['real_validation']['mean_error_px'], 0
    checkpoint_args = dict(config, input_size=[256, 256], crop_mode='bbox', crop_padding=1.3)

    def save(path, epoch, scores):
        torch.save(dict(model=model.state_dict(), epoch=epoch-1, args=checkpoint_args,
                        metrics=scores, selection='Minimum full-eight real validation mean error; initial checkpoint is eligible.'), path)

    save(output/'best.pth', 0, initial)
    # Both arms see the same first six synthetic slots at each optimizer step.
    rng = np.random.default_rng(args.seed)
    synthetic_indices = rng.integers(0, len(train_syn), size=(args.epochs, args.steps_per_epoch, 8))
    real_rng = np.random.default_rng(args.seed+1)
    real_order = []
    while len(real_order) < args.epochs*args.steps_per_epoch*2:
        real_order.extend(real_rng.permutation(len(real_train)).tolist())
    history = []
    with (output/'metrics.csv').open('w', newline='') as f:
        fields = ['epoch', 'train_loss', 'real_adapt_error', 'real_val_error', 'real_val_pck5',
                  'synthetic_val_error', 'synthetic_val_pck5', 'seconds', 'lr', 'head_only']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs+1):
            started = time.monotonic()
            model.freeze_backbone(epoch <= 2)
            model.train()
            # A few repeated real examples must not rewrite BatchNorm population statistics.
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            total_loss = 0.
            for step in range(args.steps_per_epoch):
                indices = synthetic_indices[epoch-1, step]
                batch = collate(train_syn, indices[:6] if name == 'real_mix' else indices)
                if name == 'real_mix':
                    start = ((epoch-1)*args.steps_per_epoch+step)*2
                    real = collate(real_train, real_order[start:start+2])
                    batch = {k: torch.cat([batch[k], real[k]]) for k in KEYS}
                batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(batch['image'])
                loss = criterion(logits, batch['heatmap'], batch['valid'], batch['keypoints_2d'])
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite {name} loss at {epoch}/{step}')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                total_loss += loss.item()
                if step % 10 == 0:
                    state.update(state='training', arm=name, epoch=epoch, step=step+1,
                                 epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
                                 latest_loss=loss.detach().item(), gradient_norm=grad_norm.detach().item())
                    write_json(args.output/'status.json', state)
            scores = dict(real_adapt=metrics(model, real_train), real_validation=metrics(model, real_val),
                          synthetic_validation=metrics(model, val_syn))
            if scores['real_validation']['mean_error_px'] < best_error:
                best_error = scores['real_validation']['mean_error_px']
                best_epoch = epoch
                save(output/'best.pth', epoch, scores)
            entry = dict(epoch=epoch, train_loss=total_loss/args.steps_per_epoch,
                         real_adapt_error=scores['real_adapt']['mean_error_px'],
                         real_val_error=scores['real_validation']['mean_error_px'],
                         real_val_pck5=scores['real_validation']['pck5'],
                         synthetic_val_error=scores['synthetic_validation']['mean_error_px'],
                         synthetic_val_pck5=scores['synthetic_validation']['pck5'],
                         seconds=time.monotonic()-started, lr=optimizer.param_groups[0]['lr'], head_only=epoch<=2)
            history.append(entry)
            writer.writerow(entry)
            f.flush()
            state.update(epoch_completed=epoch, best_epoch=best_epoch, best_real_validation_error=best_error,
                         epoch_metrics=entry)
            write_json(args.output/'status.json', state)
            print(f'[{name}] {json.dumps(entry)} best_epoch={best_epoch}', flush=True)
            scheduler.step()
        save(output/'last.pth', args.epochs, scores)
    summary = dict(initial=initial, best_epoch=best_epoch, best_real_validation_error=best_error,
                   updates=args.epochs*args.steps_per_epoch, history=history,
                   real_training_ids=[r['id'] for r in real_train.records] if name=='real_mix' else [],
                   real_validation_ids=[r['id'] for r in real_val.records], test_used_during_training=False)
    write_json(output/'summary.json', summary)
    del model, optimizer
    torch.cuda.empty_cache()
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real', type=Path, required=True)
    p.add_argument('--synthetic', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--steps-per-epoch', type=int, default=50)
    p.add_argument('--seed', type=int, default=20260925)
    args = p.parse_args()
    if args.epochs < 1 or args.steps_per_epoch < 1:
        p.error('Epochs and steps must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This pilot requires CUDA with bfloat16 support')
    state = dict(state='preparing', pid=os.getpid(), started_at=time.time(), epochs=args.epochs)
    write_json(args.output/'status.json', state)
    try:
        real_train, real_val = RealCornerDataset(args.real, 'adapt'), RealCornerDataset(args.real, 'validation')
        train_groups = {r['time_group'] for r in real_train.records}
        val_groups = {r['time_group'] for r in real_val.records}
        if train_groups & val_groups:
            raise ValueError('Real split group leakage')
        replay = prepare_replay(args.synthetic.resolve(), args.output, args.seed)
        annotation = json.loads((args.real/'annotations.json').read_text())
        annotated_xyz = np.array([r['xyz'] for r in annotation['corner_definition']['corners']])
        model_info = BOPModel.from_models_info(args.synthetic/'models/models_info.json', 1)
        if not np.allclose(annotated_xyz, model_info.corners_mm(), atol=1e-4):
            raise ValueError('Real and synthetic 3D corner definitions differ')
        write_json(args.output/'protocol.json', dict(
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            annotation_sha256=hashlib.sha256((args.real/'annotations.json').read_bytes()).hexdigest(),
            real_train=[r['id'] for r in real_train.records], real_validation=[r['id'] for r in real_val.records],
            batch_composition=dict(synthetic_control=[8, 0], real_mix=[6, 2]),
            epochs=args.epochs, steps_per_epoch=args.steps_per_epoch, seed=args.seed,
            torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
            all_eight_real_corners_supervised=True, test_used_for_selection=False,
            note='Single-seed small same-video adaptation. Visible and inferred manual points both supervised. No artificial occlusion or crop perturbation. Synthetic replay is a balanced subset, not all server assets.'))
        summaries = {}
        for name in ('real_mix', 'synthetic_control'):
            summaries[name] = train_arm(name, args, replay, real_train, real_val, state)
        write_json(args.output/'summary.json', summaries)
        state.update(state='completed', elapsed_seconds=time.time()-state['started_at'])
        write_json(args.output/'status.json', state)
    except Exception:
        state.update(state='failed', error=traceback.format_exc())
        write_json(args.output/'status.json', state)
        raise


if __name__ == '__main__':
    main()
