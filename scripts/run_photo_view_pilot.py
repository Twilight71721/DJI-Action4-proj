"""Matched small continuation: control, photometric augmentation, plus view balancing."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=512)
    parser.add_argument('--epochs',type=int,default=3)
    args=parser.parse_args()
    config=json.loads((ROOT/'configs/single_rgb_rect_20260926.json').read_text())['training']
    local=json.loads((ROOT/'outputs/single_rgb_rect_20260926/timing_r18/config.json').read_text())
    checkpoint=ROOT/'outputs/server_return_rect_r18_20260926/rect_r18_fixed_yolo_30ep/training/best.pth'
    config.update(dataset_root=local['dataset_root'],models_info=local['models_info'],
        online_background_config=local['online_background_config'],val_scenes=local['val_scenes'],
        init_checkpoint=str(checkpoint),epochs=args.epochs,limit=args.samples,batch_size=8,num_workers=4,
        freeze_epochs=0,lr=3e-5,deterministic_cudnn=True,seed=20260926)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'protocol.json').write_text(json.dumps(dict(configuration=config,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        limitation='Local subset continuation, not full server training; no real corner supervision.',
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for base in ['bop_bb8','scripts'] for p in (ROOT/base).rglob('*.py')}),indent=2))
    for name,photo,balance in [('control',False,False),('photo',True,False),('photo_view',True,True)]:
        folder=args.output/name;folder.mkdir()
        command=[sys.executable,'-u',str(ROOT/'scripts/train_heatmap.py')]
        for key,value in dict(config,photometric_augmentation=photo,view_balance=balance,out_dir=str(folder)).items():
            flag='--'+key.replace('_','-')
            if isinstance(value,bool):
                if value:command.append(flag)
            elif key=='rect_sizes':command += [flag,*[f'{w}x{h}' for w,h in value]]
            elif isinstance(value,list):command += [flag,*map(str,value)]
            else:command += [flag,str(value)]
        (folder/'command.json').write_text(json.dumps(command,indent=2))
        print('START '+name,flush=True)
        with (folder/'train.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE '+name,flush=True)


if __name__=='__main__':main()
