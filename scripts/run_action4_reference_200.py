"""Generate 20 video representatives x 10 variants with strict geometry checks."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--profiles',type=int,nargs='+',default=list(range(20)))
    ap.add_argument('--frames-per-profile',type=int,default=10)
    ap.add_argument('--width',type=int,default=1624);ap.add_argument('--height',type=int,default=1232)
    ap.add_argument('--samples',type=int,default=48)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();out=args.output.resolve()
    if (out/'INCOMPLETE_REJECTED.md').exists():
        raise RuntimeError('This batch was rejected during visual review; use a new output directory after correcting the defect')
    out.mkdir(parents=True,exist_ok=args.resume)
    (out/'models').mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    for name in ('obj_000001.glb','models_info.json','corners_3d.json'):
        target=out/'models'/name
        if not target.exists():shutil.copy2(ROOT/'dataset/action4-v6-domain-pilot-200/models'/name,target)
    config=dict(profiles=args.profiles,frames_per_profile=args.frames_per_profile,resolution=[args.width,args.height],samples=args.samples,
                calibration='representative-frame guided approximation, not calibrated pose recovery',reference_pixels_used=False)
    path=out/'generation_config.json'
    if path.exists() and json.loads(path.read_text())!=config:raise RuntimeError('Resume configuration differs')
    path.write_text(json.dumps(config,indent=2))
    for batch,profile in enumerate(args.profiles):
        work=out/'render_metadata'/f'batch_{batch:03d}'
        if (work/'metadata.json').exists():continue
        if work.exists():raise RuntimeError(f'Incomplete batch must be inspected, not overwritten: {work}')
        cmd=[sys.executable,str(ROOT/'scripts/render_action4_wrist_hccepose.py'),'--output',str(out),
             '--model',str(out/'models/obj_000001.glb'),'--batch',str(batch),'--reference-profile',str(profile),
             '--frames',str(args.frames_per_profile),'--seed',str(202609250+profile*137),
             '--width',str(args.width),'--height',str(args.height),'--samples',str(args.samples)]
        print('START',batch,'PROFILE',profile,flush=True)
        with (out/'logs'/f'batch_{batch:03d}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('DONE',batch,'FRAMES',(batch+1)*args.frames_per_profile,flush=True)

if __name__=='__main__':main()
