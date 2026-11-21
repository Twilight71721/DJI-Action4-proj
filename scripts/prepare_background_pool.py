"""Build one mixed background pool from public Open Images photos and local video crops."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import urllib.request

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_photo(path, image):
    h,w=image.shape[:2]
    if max(h,w)>640:
        image=cv2.resize(image,(round(w*640/max(h,w)),round(h*640/max(h,w))),interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(path),image,[cv2.IMWRITE_JPEG_QUALITY,85]):
        raise IOError(path)
    return dict(path=str(path.name),sha256=sha(path),width=image.shape[1],height=image.shape[0])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'assets/background_randomization_20260923')
    p.add_argument('--web-count',type=int,default=1024)
    p.add_argument('--real-frames',type=int,default=256)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    images=out/'images';images.mkdir(exist_ok=True);meta=out/'source_metadata';meta.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    metadata_url='https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv'
    csvpath=meta/'validation-images-with-rotation.csv'
    if not csvpath.exists():
        with urllib.request.urlopen(metadata_url,timeout=60) as response:csvpath.write_bytes(response.read())
    rows=[r for r in csv.DictReader(csvpath.open()) if 'creativecommons.org/licenses/by/' in r['License']]
    random.Random(20260923).shuffle(rows)
    def fetch(r):
        path=images/f"web_{r['ImageID']}.jpg"
        url=f"https://open-images-dataset.s3.amazonaws.com/validation/{r['ImageID']}.jpg"
        try:
            if path.exists():
                image=cv2.imread(str(path))
                if image is None:raise ValueError('Unreadable cached background')
                saved=dict(path=path.name,sha256=sha(path),width=image.shape[1],height=image.shape[0])
            else:
                with urllib.request.urlopen(url,timeout=25) as response:raw=response.read()
                image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                if image is None or min(image.shape[:2])<128:raise ValueError('Small or invalid image')
                saved=save_photo(path,image)
            return dict(saved,source='Open Images / CVDF',url=url,original_url=r['OriginalURL'],
                        landing_url=r['OriginalLandingURL'],author=r['Author'],license=r['License'],
                        modifications='resized to max640 pixels, JPEG quality85')
        except Exception as error:
            return dict(error=type(error).__name__,image_id=r['ImageID'])
    web=[];failures=[];cursor=0
    with ThreadPoolExecutor(max_workers=12) as pool:
        while len(web)<args.web_count:
            count=min(128,args.web_count-len(web))
            batch=list(pool.map(fetch,rows[cursor:cursor+count]));cursor+=count
            web.extend(x for x in batch if 'error' not in x);failures.extend(x for x in batch if 'error' in x)
            print('downloaded',len(web),'failed',len(failures),flush=True)
    (meta/'web_records.json').write_text(json.dumps(web,indent=2,ensure_ascii=False))
    # Automatic camera detection is only a background-patch exclusion heuristic, not pose supervision.
    from ultralytics import YOLO
    model=YOLO(str(ROOT/'outputs/yolo11s_real48_mixed_20260922/train/weights/best.pt'))
    paths=sorted((ROOT/'outputs/video_reference_audit/head_left_rgb_raw/frames_full').glob('*.jpg'))
    selected=[paths[i] for i in np.linspace(0,len(paths)-1,args.real_frames,dtype=int)]
    rng=np.random.default_rng(20260924);real=[]
    for number,path in enumerate(selected):
        frame=cv2.imread(str(path));h,w=frame.shape[:2]
        prediction=model.predict(frame,imgsz=640,conf=.05,device=0,verbose=False)[0]
        boxes=prediction.boxes.xyxy.cpu().numpy().copy()
        for box in boxes:
            padx=.15*(box[2]-box[0])+16;pady=.15*(box[3]-box[1])+16
            box += [-padx,-pady,padx,pady]
        for patch in range(2):
            candidate=None
            for attempt in range(160):
                side=int(rng.uniform(.18,.38)*min(w,h))
                x=int(rng.integers(0,w-side+1));y=int(rng.integers(0,h-side+1))
                overlap=(np.minimum(boxes[:,2],x+side)>np.maximum(boxes[:,0],x)) & (np.minimum(boxes[:,3],y+side)>np.maximum(boxes[:,1],y))
                if overlap.any():continue
                crop=frame[y:y+side,x:x+side]
                if crop.std()<8:continue
                candidate=(crop,x,y,side);break
            if candidate is None:raise RuntimeError(f'No background patch found for {path}')
            crop,x,y,side=candidate;dst=images/f'real_{path.stem}_{patch}.jpg'
            record=save_photo(dst,crop)
            real.append(dict(record,source='user video, mixed without split filtering',frame=path.name,
                             crop_xywh=[x,y,side,side],excluded_detector_boxes=boxes.tolist(),
                             source_sha256=sha(path),modifications='background region crop, resize max640, JPEG85'))
        if (number+1)%32==0:print('real frames',number+1,flush=True)
    records=web+real
    for r in records:r['path']='images/'+r['path']
    manifest=dict(version=1,seed=20260923,counts=dict(web=len(web),real_frames=len(selected),real_patches=len(real),total=len(records)),
                  sampling='Uniform from one mixed pool; no source/split isolation',records=records,download_failures=failures)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    (out/'SOURCES.md').write_text('# Background sources\n\nOpen Images photographs were downloaded from the CVDF public mirror. '
        'Original author, source URL and CC BY license for each image are retained in manifest.json. Images are resized and JPEG-compressed.\n\n'
        'https://storage.googleapis.com/openimages/web/download_v7.html\n\nhttps://github.com/cvdfoundation/open-images-dataset\n\n'
        'Local video patches are user-provided material. Source frames are mixed at user request, including previously inspected video segments. '
        'Real-frame evaluation is descriptive, not an independent held-out test.\n')
    tiles=[]
    for idx in np.linspace(0,len(records)-1,96,dtype=int):
        image=cv2.imread(str(out/records[idx]['path']));tiles.append(cv2.resize(image,(128,96)))
    cv2.imwrite(str(out/'contact_sheet.jpg'),np.vstack([np.hstack(tiles[i:i+8]) for i in range(0,len(tiles),8)]))
    print(json.dumps(manifest['counts']),flush=True)


if __name__=='__main__':main()
