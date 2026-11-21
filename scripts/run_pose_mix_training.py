"""Server-only preparation, smoke, one training run, evaluation and compact export."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from server_background_assets import dump,sha,verify,pack


def check(args, config, full=False):
    actual=sha(args.checkpoint)
    if actual not in config['allowed_checkpoint_sha256']:
        raise ValueError('Use the background_mixed_30ep best checkpoint; unexpected SHA256')
    verify_config=dict(config,checkpoint_sha256=actual)
    manifest=verify(args,verify_config,full=full)
    assets=json.loads((args.assets/'manifest.json').read_text())
    if len(assets['samples'])!=246 or any(r['usage']!='foreground_candidate_for_server_compositing' for r in assets['samples']):
        raise ValueError('Wrong foreground package')
    for path,expected in json.loads((args.assets/'SHA256SUMS.json').read_text()).items():
        if sha(args.assets/path)!=expected:raise ValueError('Changed foreground asset: '+path)
    if not (args.existing_backgrounds/'manifest.json').exists():raise FileNotFoundError(args.existing_backgrounds/'manifest.json')
    return manifest


def prepare(args, config):
    from prepare_pose_mix_backgrounds import prepare as backgrounds
    from prepare_pose_mix_data import prepare as composite
    args.workspace.mkdir(parents=True,exist_ok=True)
    backgrounds(args.existing_backgrounds,args.workspace/'backgrounds',config['web_backgrounds'])
    composite(args.assets,args.workspace/'backgrounds',args.dataset,args.workspace/'mixed',config['new_pairs'],config['seed'])


def prepared_contract(args,config):
    import numpy as np
    from bop_bb8 import BOPInstanceDataset,BOPModel
    mixed=args.workspace/'mixed';meta=json.loads((mixed/'composition_manifest.json').read_text())
    if meta['pairs']!=config['new_pairs'] or meta['seed']!=config['seed'] or meta['extra_occlusion'] or meta['crop_jitter']:
        raise ValueError('Prepared data/config mismatch')
    if meta['foreground_manifest_sha256']!=sha(args.assets/'manifest.json') or meta['base_manifest_sha256']!=sha(args.dataset/'manifest.json'):
        raise ValueError('Prepared data sources changed')
    if meta['background_manifest_sha256']!=sha(args.workspace/'backgrounds/manifest.json'):
        raise ValueError('Prepared background pool changed')
    model=BOPModel.from_models_info(mixed/'models/models_info.json')
    train_scenes=[s for s in meta['old_scene_ids'] if s not in meta['val_scenes']]+meta['new_scene_ids']
    train=BOPInstanceDataset(mixed/'trainval',model,scenes=train_scenes,crop_mode='bbox',crop_padding=1.3)
    val=BOPInstanceDataset(mixed/'trainval',model,scenes=meta['val_scenes'],crop_mode='bbox',crop_padding=1.3)
    if len(train)!=16000+2*config['new_pairs'] or len(val)!=1000:raise ValueError('Unexpected mixed train/validation instance counts')
    # Check new data separately, including both paired variants at the start/end
    # of each generated scene. Training smoke also uses the new foregrounds.
    new=BOPInstanceDataset(mixed/'new_train',model,crop_mode='bbox',crop_padding=1.3,heatmap_kernel='gaussian',heatmap_sigma=2.)
    for sid in meta['new_scene_ids']:
        ids=[i for i,(s,_,_) in enumerate(new.instances) if s==sid]
        for i in sorted(set(ids[:2]+ids[-2:])):
            sample=new[i]
            if not np.isfinite(sample.heatmaps).all():raise ValueError('Nonfinite new heatmaps')
    return dict(config=config,checkpoint_sha256=sha(args.checkpoint),
        composition_sha256=sha(mixed/'composition_manifest.json'),train_instances=len(train),val_instances=len(val))


def train(args,config,manifest,contract,smoke):
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA/BF16 required')
    folder=args.output/('smoke_training' if smoke else 'training');folder.mkdir(parents=True,exist_ok=False)
    root=args.workspace/'mixed/trainval';val_scenes=manifest['splits']['val']
    if smoke:
        root=folder/'trainval';root.mkdir(exist_ok=False)
        sid=json.loads((args.workspace/'mixed/composition_manifest.json').read_text())['new_scene_ids'][0]
        (root/sid).symlink_to((args.workspace/'mixed/new_train'/sid).resolve(),target_is_directory=True)
        val_scenes=val_scenes[:1]
        (root/val_scenes[0]).symlink_to((args.dataset/'trainval'/val_scenes[0]).resolve(),target_is_directory=True)
    cmd=[sys.executable,'-u',str(ROOT/'scripts/train_heatmap.py'),'--dataset-root',str(root),
        '--models-info',str(args.dataset/'models/models_info.json'),'--obj-id','1','--pose-scale','1.0','--init-checkpoint',str(args.checkpoint),
        '--backbone','resnet50','--fpn-dim','128','--input-size','256','256','--crop-mode','bbox','--crop-padding','1.3',
        '--heatmap-kernel','gaussian','--heatmap-sigma','2','--loss','composite','--coordinate-loss-weight','10',
        '--epochs',str(1 if smoke else config['epochs']),'--batch-size',str(config['batch_size']),
        '--lr',str(config['lr']),'--backbone-lr-scale','0.1','--weight-decay','0.0001',
        '--freeze-epochs','0','--cosine-lr','--seed',str(config['seed']),'--num-workers',str(args.workers),
        '--blur-probability','0','--crop-jitter-shift','0','--device','cuda','--amp','--amp-dtype','bfloat16',
        '--strict-updates','--out-dir',str(folder),'--val-scenes',*val_scenes]
    if smoke:cmd+=['--limit',str(2*config['batch_size'])]
    dump(folder/'command.json',cmd);dump(folder/'provenance.json',contract)
    start=time.time();dump(folder/'status.json',dict(state='running'))
    try:
        with (folder/'train.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        rows=list(csv.DictReader((folder/'metrics.csv').open()));actual=json.loads((folder/'config.json').read_text())
        if len(rows)!=(1 if smoke else config['epochs']):raise ValueError('Incomplete epochs')
        if not smoke and (actual['train_samples']!=contract['train_instances'] or actual['eval_samples']!=1000):
            raise ValueError('Unexpected actual training sample count')
        if any(not math.isfinite(float(r[k])) for r in rows for k in ['train_loss','val_loss','val_err_px','val_pck5']):raise ValueError('Nonfinite metric')
        if not all((folder/n).exists() for n in ['best.pth','last.pth']):raise FileNotFoundError('Missing checkpoints')
        dump(folder/'status.json',dict(state='completed',elapsed_seconds=time.time()-start,epochs=len(rows),train_samples=actual['train_samples']))
    except BaseException:
        dump(folder/'status.json',dict(state='failed_or_interrupted',elapsed_seconds=time.time()-start));raise
    print('[done]',folder,rows[-1],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['dataset','checkpoint','existing-backgrounds','workspace','output']:p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--assets',type=Path,default=ROOT/'pose_assets')
    p.add_argument('--real-manifest',type=Path,default=ROOT/'real/manifest.json')
    p.add_argument('--workers',type=int,default=8);p.add_argument('--batch-size',type=int)
    p.add_argument('--action',choices=['check','prepare','smoke','train','evaluate','pack','all'],default='check')
    args=p.parse_args()
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    if args.workspace==args.output or args.output in args.workspace.parents or args.workspace in args.output.parents:
        raise ValueError('Keep prepared data workspace and result output in separate directory trees')
    config=json.loads((ROOT/'configs/pose_mix_no_occlusion_20260924.json').read_text())
    if args.batch_size is not None:
        if args.batch_size<2:raise ValueError('batch-size must be >=2')
        config['batch_size']=args.batch_size
    if args.action=='pack':pack(args);return
    manifest=check(args,config,full=args.action in ['check','all'])
    if args.action=='check':print('[check] input assets passed; formal training has not run');return
    if args.action in ['prepare','all']:prepare(args,config)
    contract=prepared_contract(args,config)
    if args.action in ['smoke','all']:train(args,config,manifest,contract,True)
    if args.action in ['train','all']:
        folder=args.output/'smoke_training'
        if json.loads((folder/'status.json').read_text())['state']!='completed' or json.loads((folder/'provenance.json').read_text())!=contract:
            raise ValueError('Matching smoke run required')
        train(args,config,manifest,contract,False)
    if args.action in ['evaluate','all']:
        folder=args.output/'training'
        if json.loads((folder/'status.json').read_text())['state']!='completed' or json.loads((folder/'provenance.json').read_text())!=contract:raise ValueError('Matching completed training required')
        subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_background_model.py'),'--dataset',str(args.dataset),
            '--baseline',str(args.checkpoint),'--checkpoint',str(folder/'best.pth'),'--real-manifest',str(args.real_manifest),
            '--output',str(args.output/'review'),'--workers',str(args.workers),'--experiment','pose_mix'],check=True,cwd=ROOT)
    if args.action=='all':pack(args)


if __name__=='__main__':main()
