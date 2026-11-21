"""Download additional public backgrounds on the server, reusing its existing pool."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import random
import shutil
import urllib.request

import cv2
import numpy as np

from server_background_assets import dump, sha


def prepare(existing, out, web_count=5120):
    if web_count < 1:raise ValueError('web-count must be positive')
    out.mkdir(parents=True,exist_ok=True);(out/'images').mkdir(exist_ok=True)
    old=json.loads((existing/'manifest.json').read_text())
    real=[]
    for record in old['records']:
        source=existing/record['path'];target=out/'images'/source.name
        if not source.exists() or sha(source)!=record['sha256']:
            raise ValueError('Existing background missing or changed: '+str(source))
        if not target.exists():shutil.copy2(source,target)
        if source.name.startswith('real_'):
            real.append(dict(record,path='images/'+source.name))
    if not real:raise ValueError('No real video background patches in the existing pool')
    metadata=out/'validation-images-with-rotation.csv'
    if not metadata.exists():
        url='https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv'
        with urllib.request.urlopen(url,timeout=60) as response:raw=response.read()
        metadata.write_bytes(raw)
    rows=[r for r in csv.DictReader(metadata.open()) if 'creativecommons.org/licenses/by/' in r['License']]
    random.Random(20260923).shuffle(rows)
    def fetch(r):
        dest=out/'images'/f"web_{r['ImageID']}.jpg"
        url=f"https://open-images-dataset.s3.amazonaws.com/validation/{r['ImageID']}.jpg"
        for attempt in range(2):
            try:
                image=cv2.imread(str(dest)) if dest.exists() else None
                if image is None:
                    with urllib.request.urlopen(url,timeout=30) as response:raw=response.read()
                    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                    if image is None or min(image.shape[:2])<128:raise ValueError('Unreadable or too small')
                    h,w=image.shape[:2]
                    if max(h,w)>640:image=cv2.resize(image,(round(w*640/max(h,w)),round(h*640/max(h,w))),interpolation=cv2.INTER_AREA)
                    if not cv2.imwrite(str(dest),image,[cv2.IMWRITE_JPEG_QUALITY,85]):raise IOError(dest)
                return dict(path='images/'+dest.name,sha256=sha(dest),source='Open Images / CVDF',url=url,
                    original_url=r['OriginalURL'],landing_url=r['OriginalLandingURL'],author=r['Author'],license=r['License'],
                    modifications='resize max640, JPEG85',width=image.shape[1],height=image.shape[0])
            except Exception as error:
                last_error=type(error).__name__
        return dict(error=last_error,image_id=r['ImageID'])
    cv2.setNumThreads(1);web=[];failed=[];cursor=0
    with ThreadPoolExecutor(max_workers=8) as executor:
        while len(web)<web_count and cursor<len(rows):
            count=min(128,web_count-len(web),len(rows)-cursor)
            result=list(executor.map(fetch,rows[cursor:cursor+count]));cursor+=count
            web.extend(r for r in result if 'error' not in r);failed.extend(r for r in result if 'error' in r)
            dump(out/'download_progress.json',dict(web=len(web),required=web_count,failures=failed))
            print('[backgrounds]',len(web),'/',web_count,'real',len(real),flush=True)
    if len(web)!=web_count:raise RuntimeError('Insufficient downloaded backgrounds; inspect download_progress.json and rerun')
    records=web+real
    dump(out/'manifest.json',dict(counts=dict(web=len(web),real_patches=len(real),total=len(records)),records=records,
         extra_occlusion=False,sampling='one uniformly mixed background pool',download_failures=failed))
    (out/'SOURCES.md').write_text('# Background sources\n\nOpen Images / CVDF photos: per-image author, original URL and CC BY license in manifest.json.\n'
        'https://storage.googleapis.com/openimages/web/download_v7.html\n'
        'https://github.com/cvdfoundation/open-images-dataset\n\n'
        'Real patches reuse the user-provided video pool; camera exclusion was automatic, not pixelwise guaranteed. '
        'The same video is therefore a qualitative adaptation diagnostic, not a blind test.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--existing',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--web-count',type=int,default=5120);a=p.parse_args();prepare(a.existing.resolve(),a.output.resolve(),a.web_count)
