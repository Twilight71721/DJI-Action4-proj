"""One server training run with mixed random backgrounds and synchronized crop jitter."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from server_background_assets import sha,dump,verify,pack


def verify_augmentation(args,manifest):
    assets=args.background_assets
    backgrounds=json.loads((assets/'manifest.json').read_text())
    masks=json.loads((assets/'camera_masks/manifest.json').read_text())
    expected={f"{f['scene']}/{int(f['frame']):06d}/{i}" for f in manifest['frames'] if f['split']=='train' for i in range(2)}
    if set(masks['records'])!=expected or len(backgrounds['records'])<1000:
        raise ValueError('Wrong mask coverage or incomplete background pool')
    for record in backgrounds['records']:
        if sha(assets/record['path'])!=record['sha256']:
            raise ValueError('Background checksum mismatch: '+record['path'])
    for record in masks['records'].values():
        if sha(assets/'camera_masks'/record['path'])!=record['sha256']:
            raise ValueError('Mask checksum mismatch: '+record['path'])
    print('[augmentation]',backgrounds['counts'],'masks',len(expected),flush=True)
    return dict(background_manifest_sha256=sha(assets/'manifest.json'),mask_manifest_sha256=sha(assets/'camera_masks/manifest.json'))


def train(args,config,manifest,assets,smoke):
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with BF16 support is required')
    out=args.output/('smoke_training' if smoke else 'training')
    if out.exists():raise FileExistsError(f'{out}: keep previous results and use a new output directory')
    out.mkdir(parents=True)
    cmd=[sys.executable,'-u',str(ROOT/'scripts/train_heatmap.py'),
        '--dataset-root',str(args.dataset/'trainval'),'--models-info',str(args.dataset/'models/models_info.json'),
        '--init-checkpoint',str(args.checkpoint),'--backbone','resnet50','--fpn-dim','128',
        '--input-size','256','256','--crop-mode','bbox','--crop-padding','1.3',
        '--heatmap-kernel','gaussian','--heatmap-sigma','2','--loss','composite','--coordinate-loss-weight','10',
        '--epochs',str(1 if smoke else config['epochs']),'--batch-size',str(config['batch_size']),
        '--lr',str(config['lr']),'--backbone-lr-scale',str(config['backbone_lr_scale']),
        '--weight-decay',str(config['weight_decay']),'--freeze-epochs','0','--cosine-lr',
        '--seed',str(config['seed']),'--num-workers',str(args.workers),'--blur-probability','0',
        '--device','cuda','--amp','--amp-dtype',config['amp_dtype'],'--strict-updates',
        '--crop-jitter-shift',str(config['crop_jitter_shift']),'--crop-jitter-padding',*map(str,config['crop_jitter_padding']),
        '--background-assets',str(args.background_assets),'--background-probability',str(config['background_probability']),
        '--out-dir',str(out),'--val-scenes',*manifest['splits']['val']]
    if smoke:cmd+=['--limit',str(max(32,config['batch_size']*2))]
    dump(out/'command.json',cmd)
    dump(out/'provenance.json',dict(config=config,assets=assets,smoke=smoke,
        checkpoint=str(args.checkpoint),gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for directory in ['bop_bb8','scripts'] for p in (ROOT/directory).rglob('*.py')}))
    start=time.time();dump(out/'status.json',dict(state='running'))
    print('[start]',out,flush=True)
    try:
        with (out/'train.log').open('w') as log:
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        rows=list(csv.DictReader((out/'metrics.csv').open()))
        n=1 if smoke else config['epochs'];samples=max(32,config['batch_size']*2) if smoke else 16000
        actual=json.loads((out/'config.json').read_text())
        if len(rows)!=n or actual['train_samples']!=samples or actual['eval_samples']!=(samples if smoke else 1000):
            raise RuntimeError('Incomplete training or unexpected sample count')
        if any(not math.isfinite(float(r[k])) for r in rows for k in ['train_loss','val_loss','val_err_px','val_pck5']):
            raise RuntimeError('Nonfinite metrics')
        for name in ['best.pth','last.pth']:
            if not (out/name).exists():raise FileNotFoundError(out/name)
        dump(out/'status.json',dict(state='completed',epochs=n,updates=n*math.ceil(samples/config['batch_size']),elapsed_seconds=time.time()-start))
    except BaseException:
        dump(out/'status.json',dict(state='failed_or_interrupted',elapsed_seconds=time.time()-start));raise
    print('[done]',rows[-1],flush=True)


def evaluate(args,config,assets):
    folder=args.output/'training'
    saved=json.loads((folder/'provenance.json').read_text())
    if saved['config']!=config or saved['assets']!=assets or json.loads((folder/'status.json').read_text())['state']!='completed':
        raise ValueError('Training/config/augmentation is incomplete or inconsistent')
    subprocess.run([sys.executable,'-u',str(ROOT/'scripts/evaluate_background_model.py'),
        '--dataset',str(args.dataset),'--baseline',str(args.checkpoint),'--checkpoint',str(folder/'best.pth'),
        '--real-manifest',str(args.real_manifest),'--output',str(args.output/'review'),'--workers',str(args.workers)],check=True,cwd=ROOT)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['dataset','checkpoint','real-manifest','output']:p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--background-assets',type=Path,default=ROOT/'augmentation')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=None)
    p.add_argument('--action',choices=['check','smoke','train','evaluate','pack','all'],default='check')
    args=p.parse_args()
    for name in ['dataset','checkpoint','real_manifest','output','background_assets']:
        setattr(args,name,getattr(args,name).resolve())
    config=json.loads((ROOT/'configs/background_train_20260923.json').read_text())
    if args.batch_size is not None:
        if args.batch_size<2:p.error('batch-size must be at least 2')
        config['batch_size']=args.batch_size
    if args.action=='pack':pack(args);return
    manifest=verify(args,config,full=args.action in ['check','all'])
    assets=verify_augmentation(args,manifest)
    if args.action in ['smoke','all']:train(args,config,manifest,assets,True)
    if args.action in ['train','all']:
        smoke=args.output/'smoke_training'
        saved=json.loads((smoke/'provenance.json').read_text())
        if saved['config']!=config or saved['assets']!=assets or json.loads((smoke/'status.json').read_text())['state']!='completed':
            raise ValueError('Pass smoke check with matching configuration and assets first')
        train(args,config,manifest,assets,False)
    if args.action in ['evaluate','all']:evaluate(args,config,assets)
    if args.action=='all':pack(args)


if __name__=='__main__':main()
