"""Run the rectangular single-RGB protocol on an existing prepared synthetic dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from audit_rectangular_protocol import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/single_rgb_rect_20260926.json')
    parser.add_argument('--backbone', choices=['resnet18', 'resnet50'])
    parser.add_argument('--action', choices=['check', 'smoke', 'train'], default='check')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--smoke-samples', type=int, default=16)
    args = parser.parse_args()
    recipe = json.loads(args.config.read_text())
    prepared = args.prepared.resolve()
    result = audit(prepared, recipe)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'geometry_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    for name, stats in result['stats'].items():
        if stats['behind_camera_or_nonfinite_instances'] or stats['all_eight_inside_fraction'] != 1.:
            raise ValueError(f'{name} geometry is not covered; inspect geometry_audit.json before training')
    if recipe['training']['heatmap_margin'] > 32:
        raise ValueError('Unexpectedly large output domain; inspect geometry instead of allocating huge heatmaps')
    if args.action == 'check':
        print(json.dumps({name:{k:s[k] for k in ['instances', 'eligible_instances', 'all_eight_inside_fraction']}
                          for name,s in result['stats'].items()}, indent=2))
        return
    config = dict(recipe['training'])
    if args.backbone:
        config['backbone'] = args.backbone
    if args.workers is not None:
        config['num_workers'] = args.workers
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.action == 'smoke':
        if args.smoke_samples < 1:
            raise ValueError('--smoke-samples must be positive')
        config.update(epochs=1, batch_size=args.batch_size or 4,
                      num_workers=0 if args.workers is None else args.workers,
                      limit=args.smoke_samples, freeze_epochs=0)
    folder = args.output/('smoke_training' if args.action == 'smoke' else 'training')
    folder.mkdir(exist_ok=False)
    command = [sys.executable, '-u', str(ROOT/'scripts/train_heatmap.py'),
        '--dataset-root', str(prepared/'trainval'), '--models-info', str(prepared/'models/models_info.json'),
        '--online-background-config', str(prepared/'online_config.json'), '--out-dir', str(folder),
        '--val-scenes', *result['val_scenes']]
    for key, value in config.items():
        flag = '--'+key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                command.append(flag)
        elif key == 'rect_sizes':
            command += [flag, *[f'{w}x{h}' for w,h in value]]
        elif isinstance(value, list):
            command += [flag, *map(str, value)]
        else:
            command += [flag, str(value)]
    (folder/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for base in ['bop_bb8', 'scripts'] for p in (ROOT/base).rglob('*.py')}
    (folder/'provenance.json').write_text(json.dumps(dict(recipe=recipe, actual_training=config,
        geometry_audit=result, source_sha256=hashes, real_corner_training=False,
        detector_retrained=False, rgb_padding=False, smoke_only=args.action=='smoke'), indent=2)+'\n')
    with (folder/'train.log').open('w') as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'Completed {args.action}: {folder}')


if __name__ == '__main__':
    main()
