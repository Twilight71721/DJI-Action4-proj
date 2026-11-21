"""Resumable, validated 5000-frame queue; preserve failed attempts and originals."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import fcntl

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'dataset/action4-current-approved-200-20260921'
SCRIPTS=['render_action4_wrist_hccepose.py','action4_reference_profiles.py','action4_extended_checks.py',
         'action4_cable_geometry.py','action4_dynamic_scene.py','action4_tailored_denim.py','action4_wrist_assets.py',
         'blender_real_scene_preview.py','reconcile_action4_annotations.py','review_action4_wrist_dataset.py',
         'review_action4_reference_dataset.py','run_action4_reference_5000.py']


def atomic(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def record_sources(out,plan):
    version=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory=out/'generation_sources'/version;directory.mkdir(parents=True)
    hashes={}
    for name in SCRIPTS:
        source=ROOT/'scripts'/name;shutil.copy2(source,directory/name);hashes[name]=digest(source)
    plan.update(source_version=version,source_sha256=hashes)
    atomic(out/'generation_config.json',plan)


def initialize(out):
    out.mkdir(parents=True,exist_ok=False)
    for name in ('models','jobs','logs','completed','review','splits','render_metadata','train_pbr','batch_checks','.staging'):
        (out/name).mkdir()
    for name in ('obj_000001.glb','models_info.json','corners_3d.json'):
        shutil.copy2(SOURCE/'models'/name,out/'models'/name)
    tree=ast.parse((ROOT/'scripts/action4_reference_profiles.py').read_text())
    specs=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SPECS' for t in n.targets))
    profiles=[]
    for index,(ref,action,progress,rolls,y,bank) in enumerate(specs):
        raised=action in ('raised','raised_open','thumbs_up')
        profiles.append(dict(index=index,reference_frame=ref,action=action,cloth_action={'flat_press':'spread'}.get(action,action),
                             hand_action={'flat_press':'press'}.get(action,action),progress=progress,rolls=rolls,contact_y=y,bank=bank,
                             scene='raised' if raised else 'tabletop',reconstruction='Reference-guided estimated geometry; real camera calibration unavailable'))
    # Entire long button-handling segment is validation; the final raised
    # segment is test. Its neighbouring frames/augmentations never enter train.
    train={0,1,12,13,14,15,16,17};val=set(range(2,12));test={18,19}
    budgets={}
    for i in range(20):
        total=500 if i in train else 50 if i in val else 250
        budgets[i]={'reference':total*7//10,'nearby':total*2//10,'hard':total//10}
    jobs=[]
    # Spread the first 100 frames across all 20 accepted reference families.
    order=[2,13,0,18,14,15,12,17,1,4,19,3,5,6,7,8,9,10,11,16]
    for pos,i in enumerate(order):
        tier=('reference','nearby','hard')[pos%3]
        jobs.append(dict(profile=i,tier=tier,frames=5,pilot=True));budgets[i][tier]-=5
    for i in range(20):
        for tier,n in budgets[i].items():
            while n:
                size=min(20,n);jobs.append(dict(profile=i,tier=tier,frames=size,pilot=False));n-=size
    cumulative=0
    interpolations={2:3,5:6,6:7,7:8,8:9,9:10,10:11,15:16}
    for job_id,job in enumerate(jobs):
        i=job['profile'];spec=dict(profiles[i])
        spec['source_reference_frames']=[spec['reference_frame']]
        if job['tier']=='nearby' and i in interpolations:
            other=profiles[interpolations[i]]
            spec['source_reference_frames'].append(other['reference_frame'])
            for key in ('reference_frame','progress','contact_y','bank'):
                spec[key]=(spec[key]+other[key])/2
            spec['reference_frame']=int(spec['reference_frame'])
            spec['rolls']=[(a+b)/2 for a,b in zip(spec['rolls'],other['rolls'])]
            spec['reconstruction']='Approximate interpolation between same-action reference poses; midpoint real frame inspected, no calibrated 3D recovery'
        split='train' if i in train else 'val' if i in val else 'test'
        group='initial_and_lowering' if i in (0,1) else 'button_segment_300_2200' if i in val else 'final_raised_2800_2849' if i in test else 'folding_2350_2750'
        spec.update(split=split,source_segment=group)
        job.update(id=job_id,seed=2026092200+job_id*1009,split=split,source_segment=group,light_offset=cumulative%4,spec=spec)
        atomic(out/'jobs'/f'{job_id:04d}.json',spec);cumulative+=job['frames']
    assert cumulative==5000
    plan=dict(total=5000,resolution=[1624,1232],samples=48,jobs=jobs,pilot_frames=100,
              split_policy='Group holdout: entire button segment for validation, final raised segment for test; therefore deliberately different action distributions, not IID.',
              glb_sha256=digest(out/'models/obj_000001.glb'),reference_pixels_used=False)
    record_sources(out,plan)
    atomic(out/'status.json',dict(state='prepared',completed_frames=0,total=5000))
    (out/'README.md').write_text('''# Action4 5000 reference-guided synthetic frames

1624x1232, Cycles OPTIX, 48 samples, exact approved GLB and LCD/tag assets.
Quotas: 3500 reference / 1000 nearby / 500 difficult candidates; 1250 per light.
4000 train / 500 val / 500 test. Splits are source-segment holdouts, deliberately
non-IID: button-handling is validation and final raised hands are test. No source
segment or its interpolated variants crosses splits. Do not randomly split frames.
The first 100 frames cover all 20 reference families and must pass batch checks
before the remaining queue can run. VISUAL_APPROVED.json is required at that gate.
Failed attempts remain under .staging and never count as completed frames.
Progress is written to status.json and completed/*.json, without a polling service.
Every accepted batch has native-annotation backups and reconciliation/audit reports.
The dataset remains an estimated reconstruction, with kinematic cloth and fixed
per-device simulated LCD images within each short batch, not full cloth physics.
Original real-camera calibration is unavailable. This is not a blind real-video test.
''')
    return plan


def gallery(out,plan):
    from PIL import Image,ImageDraw
    completed=[json.loads(p.read_text()) for p in sorted((out/'completed').glob('*.json'))]
    total=sum(r['frames'] for r in completed)
    cards=[];thumbs=[]
    for record in completed:
        i=record['id'];job=plan['jobs'][i];scene=f'{i:06d}'
        image=out/'train_pbr'/scene/'rgb/000000.png'
        thumb=out/'review'/f'batch_{i:04d}.jpg'
        if not thumb.exists():
            with Image.open(image) as im:im.thumbnail((406,308));im.convert('RGB').save(thumb,quality=88)
        label=f"{i:03d} {job['spec']['action']} {job['tier']} {job['split']}"
        thumbs.append((thumb,label))
        cards.append(f'<a href="../train_pbr/{scene}/rgb/000000.png"><img loading="lazy" src="{thumb.name}"><span>{label} · {job["frames"]} frames</span></a>')
    (out/'review/index.html').write_text('<!doctype html><meta charset="utf-8"><title>Action4 5000</title><style>body{background:#172129;color:#eee;font:17px system-ui;margin:30px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}img{width:100%}a{color:#bde4ff}span{display:block}</style>'+f'<h1>Action4 · {total} / 5000 张已验证</h1><p>每卡为一批代表图；原图和 BOP 标注位于 train_pbr。<a href="../status.json">状态</a> · <a href="../README.md">说明与限制</a></p><div class="grid">'+''.join(cards)+'</div>')
    if thumbs:
        thumbs=thumbs[:20]
        sheet=Image.new('RGB',(4*406,((len(thumbs)+3)//4)*338),'#172129');draw=ImageDraw.Draw(sheet)
        for i,(path,label) in enumerate(thumbs):
            x=i%4*406;y=i//4*338
            with Image.open(path) as im:sheet.paste(im,(x,y))
            draw.text((x+5,y+311),label,fill='white')
        sheet.save(out/'review/pilot_overview.jpg',quality=92)
    for split in ('train','val','test'):
        names=[f"{r['id']:06d}" for r in completed if plan['jobs'][r['id']]['split']==split]
        (out/'splits'/f'{split}_scenes.txt').write_text('\n'.join(names)+'\n')
    return completed,total


def run(out,plan,max_jobs):
    lock=(out/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (out/'queue.pid').write_text(str(os.getpid()))
    start=time.time();done=0
    completed,total=gallery(out,plan)
    def status(state,**extra):
        prior=sum(x['elapsed_seconds'] for x in completed)
        atomic(out/'status.json',dict(state=state,pid=os.getpid(),completed_frames=total,total=5000,
              updated_utc=datetime.now(timezone.utc).isoformat(),estimated_remaining_hours=(5000-total)*prior/max(total,1)/3600 if total else None,**extra))
    try:
        for job in plan['jobs']:
            ident=job['id'];marker=out/'completed'/f'{ident:04d}.json'
            if marker.exists():continue
            if not job['pilot'] and not (out/'VISUAL_APPROVED.json').exists():
                status('pilot_complete_awaiting_visual_review');return
            for name,expected in plan['source_sha256'].items():
                if digest(ROOT/'scripts'/name)!=expected:raise RuntimeError(f'Source changed since snapshot: {name}')
            if shutil.disk_usage(out).free<20*1024**3:raise RuntimeError('Less than 20 GiB disk space remains')
            status('rendering',job=ident,profile=job['profile'],tier=job['tier'],phase='pilot' if job['pilot'] else 'bulk')
            attempt=0
            while (out/'.staging'/f'job_{ident:04d}_attempt_{attempt:02d}').exists():attempt+=1
            work=out/'.staging'/f'job_{ident:04d}_attempt_{attempt:02d}';work.mkdir()
            (work/'models').symlink_to(out/'models',target_is_directory=True)
            seed=job['seed']+attempt*1000003
            log_path=out/'logs'/f'job_{ident:04d}_attempt_{attempt:02d}.log'
            tick=time.time()
            with log_path.open('w') as log:
                def command(script,*arguments):
                    cmd=[sys.executable,str(ROOT/'scripts'/script),*map(str,arguments)]
                    log.write('COMMAND '+repr(cmd)+'\n');log.flush()
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
                command('render_action4_wrist_hccepose.py','--output',work,'--model',out/'models/obj_000001.glb',
                        '--batch',0,'--frames',job['frames'],'--seed',seed,'--reference-spec',out/'jobs'/f'{ident:04d}.json',
                        '--variation',job['tier'],'--light-offset',job['light_offset'],'--extended-checks',
                        '--width',1624,'--height',1232,'--samples',48)
                command('reconcile_action4_annotations.py',work)
                command('review_action4_wrist_dataset.py',work,'--expected',job['frames'])
                command('review_action4_reference_dataset.py',work)
            meta=json.loads((work/'render_metadata/batch_000/metadata.json').read_text())
            if meta['glb_sha256']!=plan['glb_sha256']:raise RuntimeError('GLB hash mismatch')
            if len(meta['frames'])!=job['frames']:raise RuntimeError('Incomplete batch')
            if any('extended_geometry_checks' not in f for f in meta['frames']):raise RuntimeError('Missing extended audits')
            target=out/'train_pbr'/f'{ident:06d}'
            if target.exists():raise RuntimeError('Uncommitted destination exists; manual inspection required')
            shutil.move(str(work/'train_pbr/000000'),target)
            shutil.move(str(work/'render_metadata/batch_000'),out/'render_metadata'/f'batch_{ident:03d}')
            shutil.move(str(work/'review'),out/'batch_checks'/f'{ident:04d}')
            record=dict(id=ident,frames=job['frames'],seed=seed,attempt=attempt,split=job['split'],elapsed_seconds=time.time()-tick,
                        source_version=plan['source_version'],audit_directory=f'batch_checks/{ident:04d}',native_backups=str(work/'native_annotations'))
            atomic(marker,record)
            completed,total=gallery(out,plan);status('batch_complete',job=ident)
            print(f'ACCEPTED job={ident} total={total}/5000 elapsed={record["elapsed_seconds"]:.1f}s',flush=True)
            done+=1
            if max_jobs and done>=max_jobs:status('paused_after_requested_jobs');return
        if total!=5000:raise RuntimeError(f'Expected 5000 accepted frames, got {total}')
        counts=Counter();lights=Counter();tiers=Counter();seen=set()
        for record in completed:
            job=plan['jobs'][record['id']];counts[job['split']]+=record['frames'];tiers[job['tier']]+=record['frames']
            meta=json.loads((out/'render_metadata'/f'batch_{record["id"]:03d}'/'metadata.json').read_text())
            lights.update(f['lighting']['profile'] for f in meta['frames'])
            for rgb in (out/'train_pbr'/f'{record["id"]:06d}'/'rgb').glob('*.png'):
                sha=digest(rgb)
                if sha in seen:raise RuntimeError('Duplicate rendered RGB')
                seen.add(sha)
        assert counts=={'train':4000,'val':500,'test':500}
        assert tiers=={'reference':3500,'nearby':1000,'hard':500}
        assert len(lights)==4 and all(v==1250 for v in lights.values())
        status('final_validation')
        with (out/'logs/final_validation.log').open('w') as log:
            for script,extra in [('review_action4_wrist_dataset.py',['--expected','5000']),('review_action4_reference_dataset.py',[])]:
                subprocess.run([sys.executable,str(ROOT/'scripts'/script),str(out),*extra],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        atomic(out/'review/completion_validation.json',dict(frames=total,unique_rgb=len(seen),splits=dict(counts),tiers=dict(tiers),lights=dict(lights),errors=[]))
        status('complete',split_counts=dict(counts),lighting_counts=dict(lights))
    except Exception as error:
        status('failed',error=str(error),job=job['id'] if 'job' in locals() else None)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--init',action='store_true')
    parser.add_argument('--record-sources',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--max-jobs',type=int,default=0)
    args=parser.parse_args();out=args.output.resolve()
    plan=initialize(out) if args.init else json.loads((out/'generation_config.json').read_text())
    if args.record_sources:record_sources(out,plan)
    if args.run:run(out,plan,args.max_jobs)


if __name__=='__main__':main()
