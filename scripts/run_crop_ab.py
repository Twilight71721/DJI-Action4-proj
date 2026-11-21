"""Run an isolated paired continuation experiment against existing server assets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False))


def verify(args, config, full=False):
    data = args.dataset.resolve()
    if sha(data / 'manifest.json') != config['dataset_manifest_sha256']:
        raise ValueError('Wrong dataset manifest; do not re-split or substitute datasets')
    if sha(args.checkpoint) != config['checkpoint_sha256']:
        raise ValueError('Wrong starting checkpoint; use the returned 5000 paired best.pth')
    manifest = json.loads((data / 'manifest.json').read_text())
    split = manifest['splits']
    if any(set(split[a]) & set(split[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('Split overlap')
    expected = set(split['train'] + split['val']) | {f'{100000 + int(s):06d}' for s in split['train']}
    actual = {p.name for p in (data / 'trainval').iterdir() if (p / 'scene_gt.json').exists()}
    if actual != expected:
        raise ValueError('trainval contains unexpected or missing scenes; potential test/blur-val leakage')
    for scene in sorted(expected):
        variant, source = ('blur', f'{int(scene)-100000:06d}') if int(scene) >= 100000 else ('clean', scene)
        for name in ['scene_gt.json', 'scene_camera.json', 'scene_gt_info.json']:
            if sha(data / 'trainval' / scene / name) != sha(data / variant / source / name):
                raise ValueError(f'trainval metadata mismatch: {scene}/{name}')
    for frame in manifest['frames']:
        for variant in ['clean', 'blur']:
            image = data / frame[variant]
            if not image.is_file():
                raise FileNotFoundError(image)
            if full and sha(image) != frame[variant + '_sha256']:
                raise ValueError(f'Image checksum mismatch: {image}')
            if frame['split'] == 'train' or (frame['split'] == 'val' and variant == 'clean'):
                scene = frame['scene'] if variant == 'clean' else f"{100000+int(frame['scene']):06d}"
                linked = data / 'trainval' / scene / 'rgb' / image.name
                if not linked.is_file() or (not linked.samefile(image) and sha(linked) != sha(image)):
                    raise ValueError(f'trainval RGB mismatch: {linked}')
    contract = ROOT / 'configs/crop_ab_data_hashes.json'
    if not contract.exists():
        raise FileNotFoundError('Missing packaged data metadata contract')
    for rel, digest in json.loads(contract.read_text()).items():
        if sha(data / rel) != digest:
            raise ValueError(f'Dataset geometry metadata mismatch: {rel}')
    if not args.real_manifest.is_file():
        raise FileNotFoundError(args.real_manifest)
    for record in json.loads(args.real_manifest.read_text())['records']:
        if not (args.real_manifest.parent / record['image']).is_file():
            raise FileNotFoundError(record['image'])
    print(f'[verify] manifest, checkpoint, geometry, splits, RGB paths OK; full RGB hashes={full}', flush=True)
    return manifest


def command(args, config, manifest, arm, smoke):
    out = args.output / (('smoke_' if smoke else '') + arm)
    cmd = [sys.executable, '-u', str(ROOT / 'scripts/train_heatmap.py'),
           '--dataset-root', str(args.dataset / 'trainval'), '--models-info', str(args.dataset / 'models/models_info.json'),
           '--init-checkpoint', str(args.checkpoint), '--backbone', 'resnet50', '--fpn-dim', '128',
           '--input-size', '256', '256', '--crop-mode', 'bbox', '--crop-padding', '1.3',
           '--heatmap-kernel', 'gaussian', '--heatmap-sigma', '2', '--loss', 'composite',
           '--coordinate-loss-weight', '10', '--pose-scale', '1', '--model-scale', '1',
           '--focal-alpha', '2', '--focal-beta', '4', '--min-visib-fract', '0',
           '--epochs', str(1 if smoke else config['epochs']), '--batch-size', str(config['batch_size']),
           '--lr', str(config['lr']), '--backbone-lr-scale', str(config['backbone_lr_scale']),
           '--weight-decay', str(config['weight_decay']), '--freeze-epochs', '0', '--cosine-lr',
           '--seed', str(config['seed']), '--num-workers', str(args.workers), '--blur-probability', '0',
           '--device', 'cuda', '--amp', '--amp-dtype', 'bfloat16', '--strict-updates', '--out-dir', str(out),
           '--val-scenes', *manifest['splits']['val']]
    if arm == 'B':
        cmd += ['--crop-jitter-shift', str(config['crop_jitter_shift']),
                '--crop-jitter-padding', *map(str, config['crop_jitter_padding'])]
    if smoke:
        cmd += ['--limit', str(max(32, 2 * config['batch_size']))]
    return out, cmd


def train(args, config, manifest, smoke):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing CPU training')
    for arm in args.arms:
        out, cmd = command(args, config, manifest, arm, smoke)
        if out.exists():
            raise FileExistsError(f'{out}: preserve it and choose a new output, or run only the unfinished arm')
        out.mkdir(parents=True)
        dump(out / 'command.json', cmd)
        dump(out / 'provenance.json', dict(config=config, arm=arm, smoke=smoke,
             gpu=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
             dataset=str(args.dataset), checkpoint=str(args.checkpoint),
             source_sha256={str(p.relative_to(ROOT)): sha(p) for directory in ['bop_bb8', 'scripts']
                            for p in sorted((ROOT / directory).rglob('*.py'))}))
        start = time.time()
        dump(out / 'status.json', dict(state='running', started_at=start))
        print('[launch]', arm, 'smoke' if smoke else 'formal', out, flush=True)
        try:
            with (out / 'train.log').open('w') as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            rows = list(csv.DictReader((out / 'metrics.csv').open()))
            actual_config = json.loads((out / 'config.json').read_text())
            expected_samples = max(32, 2*config['batch_size']) if smoke else 16000
            if actual_config['train_samples'] != expected_samples or actual_config['eval_samples'] != (expected_samples if smoke else 1000):
                raise RuntimeError('Unexpected train/validation instance count')
            expected_epochs = 1 if smoke else config['epochs']
            if len(rows) != expected_epochs or any(not math.isfinite(float(r[k])) for r in rows
                                                 for k in ['train_loss', 'val_loss', 'val_err_px', 'val_pck5']):
                raise RuntimeError('Incomplete or nonfinite training metrics')
            for name in ['best.pth', 'last.pth']:
                if not (out / name).is_file():
                    raise FileNotFoundError(out / name)
            dump(out / 'status.json', dict(state='completed', epochs=len(rows), elapsed_seconds=time.time()-start,
                 updates=expected_epochs * math.ceil((max(32, 2*config['batch_size']) if smoke else 16000)/config['batch_size'])))
        except BaseException:
            dump(out / 'status.json', dict(state='failed_or_interrupted', elapsed_seconds=time.time()-start))
            raise
        print('[done]', arm, rows[-1], flush=True)


def evaluate(args, config):
    sources = []
    for arm in ['A', 'B']:
        status = json.loads((args.output / arm / 'status.json').read_text())
        saved = json.loads((args.output / arm / 'provenance.json').read_text())
        if status['state'] != 'completed' or saved['smoke'] or saved['config'] != config:
            raise ValueError(f'{arm} is incomplete or has a mismatched experiment configuration')
        sources.append(saved['source_sha256'])
    if sources[0] != sources[1]:
        raise ValueError('A/B used different source snapshots')
    subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/evaluate_crop_ab.py'),
        '--dataset', str(args.dataset), '--baseline', str(args.checkpoint),
        '--a', str(args.output/'A/last.pth'), '--b', str(args.output/'B/last.pth'),
        '--real-manifest', str(args.real_manifest), '--output', str(args.output/'review'),
        '--workers', str(args.workers)], check=True, cwd=ROOT)


def pack(args):
    if not (args.output/'review/summary.json').exists():
        raise FileNotFoundError('Complete evaluation before packaging')
    if json.loads((args.output/'review/summary.json').read_text())['incomplete_diagnostic']:
        raise ValueError('Do not package a limited diagnostic as formal results')
    archive = args.output.with_suffix('.tar.gz')
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, 'w:gz', compresslevel=3) as tar:
        for p in sorted(args.output.rglob('*')):
            if p.is_file() and not any(part.startswith('smoke_') for part in p.relative_to(args.output).parts):
                name = str(Path(args.output.name)/p.relative_to(args.output))
                if p.suffix == '.pth':
                    import torch
                    checkpoint = torch.load(p, map_location='cpu', weights_only=False)
                    reduced = {k: v for k, v in checkpoint.items() if k != 'optimizer'}
                    reduced['original_checkpoint_sha256'] = sha(p)
                    reduced['export_note'] = 'Optimizer omitted for inference transfer; full checkpoint retained on server'
                    buffer = io.BytesIO()
                    torch.save(reduced, buffer)
                    entry = tarfile.TarInfo(name)
                    entry.size = buffer.tell()
                    buffer.seek(0)
                    tar.addfile(entry, buffer)
                else:
                    tar.add(p, arcname=name, recursive=False)
    archive.with_name(archive.name+'.sha256').write_text(f'{sha(archive)}  {archive.name}\n')
    print('[return]', archive, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--real-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--action', choices=['check', 'smoke', 'train', 'evaluate', 'pack', 'all'], default='check')
    parser.add_argument('--arms', nargs='+', choices=['A', 'B'], default=['A', 'B'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    for name in ['dataset', 'checkpoint', 'real_manifest', 'output']:
        setattr(args, name, getattr(args, name).resolve())
    config = json.loads((ROOT/'configs/crop_ab_20260923.json').read_text())
    if args.batch_size is not None:
        if args.batch_size < 2:
            parser.error('batch-size must be >= 2')
        config['batch_size'] = args.batch_size
        config['budget'] = f"{config['epochs']} * ceil(16000/{args.batch_size}) updates per arm"
    if args.dry_run:
        manifest = json.loads((args.dataset/'manifest.json').read_text())
        for arm in args.arms:
            print(json.dumps(command(args, config, manifest, arm, args.action=='smoke')[1]))
        return
    if args.action == 'pack':
        pack(args)
        return
    manifest = verify(args, config, full=args.action in ['check', 'all'])
    if args.action in ['smoke', 'all']:
        train(args, config, manifest, smoke=True)
    if args.action in ['train', 'all']:
        for arm in args.arms:
            smoke = args.output / ('smoke_'+arm)
            if (json.loads((smoke/'status.json').read_text())['state'] != 'completed'
                    or json.loads((smoke/'provenance.json').read_text())['config'] != config):
                raise RuntimeError('Pass smoke training with the same configuration first')
        train(args, config, manifest, smoke=False)
    if args.action in ['evaluate', 'all']:
        evaluate(args, config)
    if args.action == 'all':
        pack(args)


if __name__ == '__main__':
    main()
