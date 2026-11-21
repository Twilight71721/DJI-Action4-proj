"""Extract camera-excluding real-video patches, then merge into one background pool."""
import argparse
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from server_background_assets import sha, dump

ROOT = Path(__file__).resolve().parents[1]


def extract(out, count=1024, patches=8):
    tracks = json.loads((ROOT/'outputs/full_video_pose_coverage_20260924/tracked_boxes.json').read_text())
    rows = [json.loads(s) for s in (ROOT/'outputs/video_pose_coverage_20260924/frames.jsonl').read_text().splitlines()]
    eligible = [f for f in range(len(rows)) if all(t[f]['detection'] is not None and t[f]['detection']['confidence'] >= .25 for t in tracks)]
    if len(eligible) < count:
        raise ValueError('Not enough confidently detected frames')
    selected = [eligible[i] for i in np.linspace(0,len(eligible)-1,count,dtype=int)]
    out.mkdir(parents=True,exist_ok=False);(out/'images').mkdir()
    rng = np.random.default_rng(20260927);records=[];tiles=[];seen=set()
    cv2.setNumThreads(1)
    for n,f in enumerate(selected):
        path=ROOT/f'outputs/video_reference_audit/head_left_rgb_raw/frames_full/frame_{f:06d}.jpg'
        image=cv2.imread(str(path));h,w=image.shape[:2];boxes=[]
        # Include both tracks and ALL detector proposals from neighboring frames.
        for j in range(max(0,f-3),min(len(rows),f+4)):
            boxes.extend(t[j]['box_xyxy'] for t in tracks)
            boxes.extend(r['box_xyxy'] for r in rows[j]['objects'])
        boxes=np.array(boxes,float)
        pad=.25*(boxes[:,2:]-boxes[:,:2])+24
        boxes[:,:2]-=pad;boxes[:,2:]+=pad
        digest=sha(path)
        for k in range(patches):
            for attempt in range(1000):
                side=int(rng.uniform(.12,.30)*min(h,w))
                x=int(rng.integers(w-side+1));y=int(rng.integers(h-side+1))
                overlap=(np.minimum(boxes[:,2],x+side)>np.maximum(boxes[:,0],x)) & (np.minimum(boxes[:,3],y+side)>np.maximum(boxes[:,1],y))
                if overlap.any():continue
                crop=cv2.resize(image[y:y+side,x:x+side],(384,384),interpolation=cv2.INTER_AREA)
                ok,encoded=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,90]);assert ok
                import hashlib
                image_sha=hashlib.sha256(encoded.tobytes()).hexdigest()
                if image_sha in seen:continue
                break
            else:raise RuntimeError(f'No safe unique patch: frame {f}, patch {k}')
            seen.add(image_sha);rel=f'images/real_dense_{f:06d}_{k:02d}.jpg';(out/rel).write_bytes(encoded.tobytes())
            records.append(dict(path=rel,sha256=image_sha,width=384,height=384,source='user video background',
                frame=path.name,source_sha256=digest,background_group='real_frame_'+digest,
                crop_xywh=[x,y,side,side],exclusion_method='both detected tracks + all proposals at +/-3 frames; 25% +24px dilation',
                exclusion_boxes=boxes.tolist(),modifications='background crop, resize384, JPEG90'))
            if n%32==0 and k==0:
                tiles.append(cv2.resize(crop,(128,128)))
        if (n+1)%128==0:print('REAL BACKGROUNDS',n+1,len(records),flush=True)
    dump(out/'manifest.json',dict(records=records,counts=dict(real_frames=len(selected),real_patches=len(records)),
        limitation='Automated conservative exclusion, not manually verified segmentation. Same previously inspected video; not an independent real test.'))
    cv2.imwrite(str(out/'preview.jpg'),np.vstack([np.hstack(tiles[i:i+8]) for i in range(0,len(tiles),8)]))
    print('EXTRACTED',len(records),flush=True)


def merge(old, extra, out):
    out.mkdir(parents=True,exist_ok=False);records=[];seen=set();counts={}
    for name,source in [('existing',old),('real_extension',extra)]:
        count=0
        for r in json.loads((source/'manifest.json').read_text())['records']:
            p=(source/r['path']).resolve()
            if sha(p)!=r['sha256']:raise ValueError('Source checksum mismatch: '+str(p))
            if r['sha256'] in seen:continue
            seen.add(r['sha256']);count+=1
            rel=f'images/{name}/{Path(r["path"]).name}'
            target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(p)
            group=r.get('background_group') or ('real_frame_'+r['source_sha256'] if r.get('source_sha256') else r['sha256'])
            records.append(dict(r,path=rel,background_group=group))
        counts[name]=count
    dump(out/'manifest.json',dict(records=records,counts=counts,sampling='Uniform single mixed pool; no source-specific training branches'))
    print('MERGED',counts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=['extract','merge'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--existing',type=Path);p.add_argument('--extra',type=Path)
    a=p.parse_args()
    if a.action=='extract':extract(a.output.resolve())
    else:
        if a.existing is None or a.extra is None:p.error('merge requires --existing and --extra')
        merge(a.existing.resolve(),a.extra.resolve(),a.output.resolve())
