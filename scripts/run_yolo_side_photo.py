"""Run a separate detector candidate and fixed evaluations without activating its weights."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_yolo_action4 import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=20)
    args=parser.parse_args();run=args.run.resolve();dataset=args.dataset.resolve()
    rollback=json.loads((run/'rollback.json').read_text());backup=Path(rollback['backup'])
    state=dict(state='starting',started_at=datetime.datetime.now().astimezone().isoformat(),active_model_unchanged=True)
    def status(name):
        state.update(state=name,updated_at=datetime.datetime.now().astimezone().isoformat())
        write_json(run/'workflow_status.json',state)
    def verify():
        for name in ['backup','original']:
            assert hashlib.sha256(Path(rollback[name]).read_bytes()).hexdigest()==rollback['sha256']
        for r in json.loads((dataset/'frozen_holdouts.json').read_text()):
            assert hashlib.sha256((dataset/r['path']).read_bytes()).hexdigest()==r['sha256']
    def call(script,*arguments):
        subprocess.run([sys.executable,'-u',str(ROOT/'scripts'/script),*map(str,arguments)],cwd=ROOT,check=True)
    try:
        verify();status('training')
        call('train_yolo_side_photo.py','--data',dataset/'data.yaml','--weights',backup,'--output',run,
             '--epochs',args.epochs,'--batch',4,'--imgsz',960,'--lr0',.00005,'--patience',7)
        status('reporting')
        call('report_yolo_action4.py','--run',run,'--dataset',dataset)
        call('compare_yolo_real_holdout.py','--run',run,'--dataset',dataset,'--baseline',backup,
             '--baseline-label','保留的原YOLO：清晰模糊5000＋真实48',
             '--trained-label','侧面真实帧补充＋balanced成像增强微调')
        status('synthetic_regression_check')
        from ultralytics import YOLO
        metrics={}
        for name,path in [('baseline',backup),('candidate',run/'train/weights/best.pt')]:
            result=YOLO(str(path)).val(data=str(ROOT/'dataset/action4-yolo-cleanblur5000-20260925/data.yaml'),
                split='test',imgsz=960,batch=4,device=0,workers=2,plots=False,verbose=False,
                project=str(run/'synthetic_regression'),name=name)
            metrics[name]={k:float(v) for k,v in result.results_dict.items()}
        write_json(run/'synthetic_regression.json',dict(metrics=metrics,
            note='Historical synthetic test includes paired clean/blur. Previous detectors may have seen some scenes; regression diagnostic only.'))
        verify()
        comparison=json.loads((run/'comparison/comparison.json').read_text())
        a,b=comparison['totals']['baseline'],comparison['totals']['mixed']
        write_json(run/'candidate_decision.json',dict(active_model_unchanged=True,baseline=str(backup),
            candidate=str(run/'train/weights/best.pt'),real_test_map50_95_delta=b['ap']['metrics/mAP50-95(B)']-a['ap']['metrics/mAP50-95(B)'],
            real_test_fp_delta=b['fp']-a['fp'],real_test_fn_delta=b['fn']-a['fn'],
            synthetic_map50_95_delta=metrics['candidate']['metrics/mAP50-95(B)']-metrics['baseline']['metrics/mAP50-95(B)'],
            instruction='Candidate only; inspect fixed reports before replacing the detector used by ResNet. Original and backup remain intact.'))
        status('completed')
    except Exception:
        state['error']=traceback.format_exc();status('failed');raise


if __name__=='__main__':main()
