"""Train on a frozen scene split, track progress, then compare held-out predictions."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data, out = args.dataset.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metrics.csv').exists() or (out / 'status.json').exists():
        raise FileExistsError('Use a fresh run directory')
    manifest = json.loads((data / 'manifest.json').read_text())
    command = [sys.executable, '-u', str(ROOT / 'scripts/train_heatmap.py'),
               '--dataset-root', str(data / 'trainval'), '--models-info', str(data / 'models/models_info.json'),
               '--backbone', 'resnet50', '--cache-data', '--cosine-lr', '--epochs', '100',
               '--batch-size', '16', '--num-workers', '0', '--freeze-epochs', '2',
               '--lr', '0.001', '--backbone-lr-scale', '0.1', '--weight-decay', '0.0001',
               '--seed', '20260921', '--out-dir', str(out), '--val-scenes', *manifest['splits']['val']]
    (out / 'command.json').write_text(json.dumps(command, indent=2))
    files = [ROOT / 'scripts/train_heatmap.py', *sorted((ROOT / 'bop_bb8').rglob('*.py')),
             ROOT / 'scripts/compare_resnet_generalization.py']
    provenance = dict(dataset_manifest_sha256=hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest(),
                      source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                      initialization='ImageNet pretrained ResNet50, fresh FPN; no old overfit checkpoint initialization',
                      selection='Minimum validation corner error only; test excluded from trainval directory')
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    started = time.time()
    state = dict(state='preparing_data', pid=os.getpid(), epoch=0, epochs=100, counts=manifest['counts'])

    def status(**updates):
        state.update(updates, elapsed_seconds=time.time() - started)
        temp = out / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(out / 'status.json')

    try:
        status()
        with (out / 'train.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            status(train_pid=process.pid)
            while process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    pass
                metrics = out / 'metrics.csv'
                if metrics.exists():
                    rows = list(csv.DictReader(metrics.open()))
                    rows = [r for r in rows if r.get('seconds') is not None]
                    if rows:
                        best = min(rows, key=lambda r: float(r['val_err_px']))
                        status(state='training', epoch=int(rows[-1]['epoch']), latest=rows[-1], best_validation=best)
            if process.returncode:
                raise RuntimeError(f'Training exited with code {process.returncode}; see train.log')
        status(state='evaluating')
        with (out / 'evaluation.log').open('w') as log:
            subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/compare_resnet_generalization.py'),
                            '--dataset', str(data), '--run', str(out)], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        status(state='completed', report=str(out / 'review/index.html'))
    except Exception:
        status(state='failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
