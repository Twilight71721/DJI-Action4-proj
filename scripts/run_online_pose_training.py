"""Server workflow for systematic views and epoch-varying independent backgrounds."""
import argparse,csv,json,math,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from server_background_assets import dump,sha,verify,pack
from prepare_online_pose_backgrounds import prepare


def main():
    p=argparse.ArgumentParser()
    for flag in ['dataset','checkpoint','backgrounds','old-background-assets','workspace','output']:p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--assets',type=Path,default=ROOT/'pose_assets');p.add_argument('--real-manifest',type=Path,default=ROOT/'real/manifest.json')
    p.add_argument('--config',type=Path,default=ROOT/'configs/online_pose_background_20260924.json')
    p.add_argument('--action',choices=['check','prepare','smoke','train','evaluate','pack','all'],default='check');p.add_argument('--workers',type=int,default=8);p.add_argument('--batch-size',type=int,default=32)
    a=p.parse_args()
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    if a.output==a.workspace or a.output in a.workspace.parents or a.workspace in a.output.parents:raise ValueError('Separate workspace and result directories')
    config=json.loads(a.config.read_text());config['batch_size']=a.batch_size
    if a.batch_size<2:raise ValueError('batch-size >=2 required')
    if a.action=='pack':
        if not (a.output/'pose_holdout/summary.json').exists():raise FileNotFoundError('Missing pose holdout evaluation')
        if json.loads((a.output/'pose_holdout/summary.json').read_text())['incomplete_diagnostic']:raise ValueError('Incomplete pose holdout evaluation')
        pack(a);return
    actual=sha(a.checkpoint)
    if actual not in config['allowed_checkpoint_sha256']:raise ValueError('Use a starting checkpoint allowed by the selected config')
    manifest=verify(a,dict(config,checkpoint_sha256=actual),full=a.action in ['check','all'])
    banks=json.loads((a.assets/'manifest.json').read_text())['samples']
    if len(banks)!=config.get('expected_foreground_assets',518):raise ValueError('Unexpected foreground count')
    for split,key in [('train','expected_train_pose_assets'),('val','expected_val_pose_assets')]:
        if key in config and sum(r.get('split','train')==split for r in banks)!=config[key]:raise ValueError('Unexpected '+split+' foreground count')
    for rel,digest in json.loads((a.assets/'SHA256SUMS.json').read_text()).items():
        if sha(a.assets/rel)!=digest:raise ValueError('Foreground checksum mismatch '+rel)
    bg=json.loads((a.backgrounds/'manifest.json').read_text())
    if len(bg['records'])<5000:raise ValueError('Reuse the server large background pool, at least 5000 images')
    for r in bg['records']:
        if sha(a.backgrounds/r['path'])!=r['sha256']:raise ValueError('Background checksum mismatch')
    masks=json.loads((a.old_background_assets/'camera_masks/manifest.json').read_text())
    for r in masks['records'].values():
        if sha(a.old_background_assets/'camera_masks'/r['path'])!=r['sha256']:raise ValueError('Old mask checksum mismatch')
    if a.action=='check':print('CHECK PASSED');return
    if a.action in ['prepare','all']:prepare(a.dataset,a.assets,a.backgrounds,a.old_background_assets,a.workspace,config['new_pairs'],config['seed'])
    prepared=json.loads((a.workspace/'online_config.json').read_text())
    if prepared['pairs']!=config['new_pairs'] or prepared['seed']!=config['seed']:raise ValueError('Prepared/config mismatch')
    contract=dict(config=config,checkpoint_sha256=actual,prepared_sha256=sha(a.workspace/'online_config.json'),
        foreground_sha256=sha(a.assets/'manifest.json'),background_sha256=sha(a.backgrounds/'manifest.json'))
    def train(smoke):
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA/BF16 required')
        folder=a.output/('smoke_training' if smoke else 'training');folder.mkdir(parents=True,exist_ok=False)
        cmd=[sys.executable,'-u',str(ROOT/'scripts/train_heatmap.py'),'--dataset-root',str(a.workspace/'trainval'),
            '--models-info',str(a.workspace/'models/models_info.json'),'--init-checkpoint',str(a.checkpoint),
            '--backbone','resnet50','--fpn-dim','128','--input-size','256','256',
            '--crop-mode',config.get('crop_mode','bbox'),'--crop-padding',str(config.get('crop_padding',1.3)),
            '--crop-border',str(config.get('crop_border',4)),
            '--heatmap-kernel','gaussian','--heatmap-sigma','2','--loss','composite','--coordinate-loss-weight','10',
            '--epochs',str(1 if smoke else config['epochs']),'--batch-size',str(a.batch_size),'--lr','0.0001','--backbone-lr-scale','0.1',
            '--weight-decay','0.0001','--freeze-epochs','0','--cosine-lr','--seed',str(config['seed']),
            '--num-workers',str(a.workers),'--device','cuda','--amp','--amp-dtype','bfloat16','--strict-updates',
            '--online-background-config',str(a.workspace/'online_config.json'),'--out-dir',str(folder),'--val-scenes',*prepared['val_scenes']]
        if smoke:cmd+=['--limit',str(a.batch_size*2)]
        dump(folder/'command.json',cmd);dump(folder/'provenance.json',dict(contract=contract,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,source_sha256={str(p.relative_to(ROOT)):sha(p) for d in ['bop_bb8','scripts'] for p in (ROOT/d).rglob('*.py')}))
        start=time.time();dump(folder/'status.json',dict(state='running'))
        try:
            with (folder/'train.log').open('w') as log:subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            rows=list(csv.DictReader((folder/'metrics.csv').open()));actual_config=json.loads((folder/'config.json').read_text())
            if len(rows)!=(1 if smoke else config['epochs']):raise RuntimeError('Incomplete training')
            if not smoke and (actual_config['train_samples']!=prepared['expected_train_instances'] or actual_config['eval_samples']!=prepared['expected_val_instances']):raise RuntimeError('Wrong instance counts')
            if any(not math.isfinite(float(r[k])) for r in rows for k in ['train_loss','val_loss','val_err_px','val_pck5']):raise RuntimeError('Nonfinite metrics')
            dump(folder/'status.json',dict(state='completed',epochs=len(rows),elapsed_seconds=time.time()-start))
        except BaseException:
            dump(folder/'status.json',dict(state='failed_or_interrupted'));raise
    if a.action in ['smoke','all']:train(True)
    if a.action in ['train','all']:
        smoke=a.output/'smoke_training'
        if json.loads((smoke/'status.json').read_text())['state']!='completed' or json.loads((smoke/'provenance.json').read_text())['contract']!=contract:raise ValueError('Matching completed smoke required')
        train(False)
    if a.action in ['evaluate','all']:
        folder=a.output/'training'
        if json.loads((folder/'status.json').read_text())['state']!='completed' or json.loads((folder/'provenance.json').read_text())['contract']!=contract:raise ValueError('Training incomplete or contract mismatch')
        if config.get('crop_mode','bbox')=='cuboid':
            subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_tight_cuboid.py'),'--dataset',str(a.dataset),
                '--prepared',str(a.workspace),'--baseline',str(a.checkpoint),'--checkpoint',str(folder/'best.pth'),
                '--output',str(a.output),'--workers',str(a.workers)],check=True)
        else:
            subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_background_model.py'),'--dataset',str(a.dataset),'--baseline',str(a.checkpoint),
                '--checkpoint',str(folder/'best.pth'),'--real-manifest',str(a.real_manifest),'--output',str(a.output/'review'),'--workers',str(a.workers),'--experiment','online_background'],check=True)
            subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_online_pose_holdout.py'),'--prepared',str(a.workspace),'--baseline',str(a.checkpoint),'--checkpoint',str(folder/'best.pth'),'--output',str(a.output/'pose_holdout')],check=True)
        dump(a.output/'composition_audit.json',prepared);dump(a.output/'background_manifest.json',bg)
        dump(a.output/'foreground_manifest.json',dict(samples=banks))
        if (a.assets/'sampling_audit.json').exists():dump(a.output/'pose_sampling_audit.json',json.loads((a.assets/'sampling_audit.json').read_text()))
    if a.action=='all':pack(a)


if __name__=='__main__':main()
