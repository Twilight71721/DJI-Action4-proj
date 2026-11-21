"""Export portable multiview foregrounds and the online background server workflow."""
import json,shutil,tarfile
from pathlib import Path
from server_background_assets import sha,dump
ROOT=Path(__file__).resolve().parents[1]
NAME='resnet_multiview_online_background_20260924'


def assets(destination,new_bank):
    destination.mkdir(parents=True,exist_ok=False);records=[]
    sources=[('legacy',ROOT/'exports/full_video_pose_assets_20260924'),('grid',new_bank)]
    shutil.copytree(sources[0][1]/'models',destination/'models')
    for prefix,source in sources:
        for r in json.loads((source/'manifest.json').read_text())['samples']:
            if r['usage']!='foreground_candidate_for_server_compositing':continue
            record=dict(r,split=r.get('split','train'))
            for field in ['rgba','mask']:
                name=f'{prefix}/{field}/{r["id"]}.png';path=destination/name;path.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source/r[field],path);record[field]=name
            # Old source annotations are provenance, not dependencies on local preview paths.
            records.append(record)
    dump(destination/'manifest.json',dict(samples=records,train_assets=sum(r['split']=='train' for r in records),val_assets=sum(r['split']=='val' for r in records)))
    dump(destination/'SHA256SUMS.json',{str(p.relative_to(destination)):sha(p) for p in destination.rglob('*') if p.is_file()})
    return records


def main():
    out=ROOT/'exports'/NAME;out.mkdir(parents=True,exist_ok=False)
    files=[p.relative_to(ROOT) for p in (ROOT/'bop_bb8').rglob('*.py')]
    files += [Path('scripts')/n for n in ['train_heatmap.py','prepare_online_pose_backgrounds.py','run_online_pose_training.py',
        'evaluate_online_pose_holdout.py','evaluate_background_model.py','evaluate_cleanblur5000.py',
        'evaluate_heatmap_checkpoint.py','infer_resnet_yolo_real.py','server_background_assets.py','prepare_camera_masks.py']]
    files += [Path('tests')/n for n in ['test_online_background.py','test_background_augmentation.py','test_net.py','test_bop_bb8.py']]
    files += [Path('configs')/n for n in ['online_pose_background_20260924.json','background_data_hashes.json']]+[Path('requirements.txt')]
    for rel in files:
        target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
    bank=ROOT/'outputs/multiview_online_background_20260924'
    assets(out/'pose_assets',bank)
    shutil.copytree(ROOT/'exports/resnet_pose_mix_no_occlusion_20260924/real',out/'real')
    shutil.copy2(ROOT/'docs/ONLINE_POSE_BACKGROUND_SERVER_20260924.md',out/'OPENCODE_SERVER_GUIDE.md')
    shutil.copy2(ROOT/'prompts/opencode_online_pose_background_20260924.txt',out/'OPENCODE_PROMPT.txt')
    checks=out/'local_verification';checks.mkdir()
    shutil.copytree(bank/'review',checks/'views')
    shutil.copytree(ROOT/'outputs/online_background_local_check_20260924/review',checks/'online_preview')
    shutil.copytree(ROOT/'outputs/online_background_local_check_20260924/review_grid',checks/'new_views_online_preview')
    shutil.copy2(bank/'split_audit.json',checks/'split_audit.json')
    gpu=ROOT/'outputs/online_background_local_check_20260924/smoke_gpu';(checks/'gpu_smoke').mkdir()
    for n in ['metrics.csv','config.json','baseline.json','train.log']:shutil.copy2(gpu/n,checks/'gpu_smoke'/n)
    shutil.copy2(ROOT/'outputs/online_background_local_check_20260924/evaluation_smoke/summary.json',checks/'limited_pose_evaluation.json')
    smoke=ROOT/'outputs/online_background_local_check_20260924/smoke'
    for n in ['metrics.csv','config.json','baseline.json','train.log']:shutil.copy2(smoke/n,checks/n)
    shutil.copy2(ROOT/'outputs/online_background_local_check_20260924/verification.json',checks/'verification.json')
    (out/'START_HERE.md').write_text('# 多角度＋在线随机背景\n\n阅读OPENCODE_SERVER_GUIDE.md，给OpenCode的指令在OPENCODE_PROMPT.txt。复用服务器背景、原数据和最新pose_mix权重。无新增遮挡，清晰模糊配对，所有训练样本随epoch换背景。本地未运行正式训练。\n')
    sums={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'SHA256SUMS').write_text(''.join(f'{v}  {k}\n' for k,v in sums.items()))
    archive=out.with_suffix('.tar.gz')
    if archive.exists():raise FileExistsError(archive)
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:tar.add(out,arcname=out.name)
    import hashlib
    with tarfile.open(archive) as tar:
        for rel,digest in sums.items():
            if hashlib.sha256(tar.extractfile(out.name+'/'+rel).read()).hexdigest()!=digest:raise ValueError('Archive checksum mismatch')
    archive.with_name(archive.name+'.sha256').write_text(sha(archive)+'  '+archive.name+'\n')
    print('ARCHIVE',archive,archive.stat().st_size,'VERIFIED',len(sums),flush=True)


if __name__=='__main__':main()
