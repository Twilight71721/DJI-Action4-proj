"""Package only the new pose assets and executable server workflow, not old data."""
import json
from pathlib import Path
import shutil
import tarfile

from server_background_assets import sha,dump

ROOT=Path(__file__).resolve().parents[1]
NAME='resnet_pose_mix_no_occlusion_20260924'


def main():
    out=ROOT/'exports'/NAME;out.mkdir(exist_ok=False)
    files=[p.relative_to(ROOT) for p in (ROOT/'bop_bb8').rglob('*.py')]
    files += [Path('scripts')/n for n in ['run_pose_mix_training.py','prepare_pose_mix_data.py',
        'prepare_pose_mix_backgrounds.py','compose_pose_backgrounds.py','server_background_assets.py',
        'train_heatmap.py','evaluate_background_model.py','evaluate_cleanblur5000.py',
        'evaluate_heatmap_checkpoint.py','infer_resnet_yolo_real.py']]
    files += [Path('tests')/n for n in ['test_pose_mix_data.py','test_net.py','test_bop_bb8.py','test_blur_augmentation.py']]
    files += [Path('configs/pose_mix_no_occlusion_20260924.json'),Path('configs/background_data_hashes.json'),Path('requirements.txt')]
    for rel in files:
        target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
    shutil.copytree(ROOT/'exports/full_video_pose_assets_20260924',out/'pose_assets')
    shutil.copytree(ROOT/'exports/resnet5000_cleanblur_4090/real',out/'real')
    shutil.copytree(ROOT/'exports/resnet5000_background_mixed_20260923/real/frame0',out/'real/frame0')
    shutil.copy2(ROOT/'docs/POSE_MIX_NO_OCCLUSION_SERVER_20260924.md',out/'OPENCODE_SERVER_GUIDE.md')
    shutil.copy2(ROOT/'prompts/opencode_pose_mix_no_occlusion_20260924.txt',out/'OPENCODE_PROMPT.txt')
    (out/'START_HERE.md').write_text('# 新姿态混合训练增量包\n\n'
        '先校验 SHA256SUMS，阅读 OPENCODE_SERVER_GUIDE.md；给 OpenCode 的提示词在 OPENCODE_PROMPT.txt。\n\n'
        '不添加额外遮挡，关闭裁剪扰动；清晰/模糊成对。包内有 246 张前景和真实诊断帧，'
        '旧5000训练数据、上一轮背景best权重、旧背景池需复用服务器已有文件，不能删掉。'
        '大背景下载与16000张新增图像合成在服务器进行，本包未启动正式训练。\n')
    checks=out/'local_verification';checks.mkdir()
    work=ROOT/'outputs/pose_mix_bundle_verification_20260924/mixed'
    shutil.copytree(work/'preview',checks/'preview')
    shutil.copy2(work/'composition_manifest.json',checks/'small_composition_manifest.json')
    smoke=ROOT/'outputs/pose_mix_bundle_smoke_20260924/smoke_training';(checks/'smoke').mkdir()
    if json.loads((smoke/'status.json').read_text())['state']!='completed':raise ValueError('Local smoke did not complete')
    for name in ['metrics.csv','status.json','config.json','command.json','train.log']:
        shutil.copy2(smoke/name,checks/'smoke'/name)
    evaluation=ROOT/'outputs/pose_mix_bundle_eval_smoke_20260924'
    shutil.copy2(evaluation/'summary.json',checks/'limited_evaluation_summary.json')
    dump(checks/'verification.json',dict(full_input_asset_hash_check_passed=True,paired_composites=4,
        projection_error_px=json.loads((work/'composition_manifest.json').read_text())['max_projection_error_px'],
        paired_labels_equal=True,no_extra_occlusion_mask_check_passed=True,new_unit_tests_passed=2,
        local_smoke_train_samples=4,local_smoke_epochs=1,local_bf16_smoke_passed=True,
        evaluation_cases_checked_per_model=7,evaluation_limit_per_case=2,
        warning='Pipeline checks only, not formal training or performance evidence. Server downloads not repeated locally.'))
    all_files=sorted(p for p in out.rglob('*') if p.is_file())
    sums={str(p.relative_to(out)):sha(p) for p in all_files}
    (out/'SHA256SUMS').write_text(''.join(f'{v}  {k}\n' for k,v in sums.items()))
    archive=out.with_suffix('.tar.gz')
    if archive.exists():raise FileExistsError(archive)
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:tar.add(out,arcname=NAME)
    with tarfile.open(archive) as tar:
        import hashlib
        for path,expected in sums.items():
            if hashlib.sha256(tar.extractfile(NAME+'/'+path).read()).hexdigest()!=expected:raise ValueError('Archive verification failed')
    digest=sha(archive);archive.with_name(archive.name+'.sha256').write_text(digest+'  '+archive.name+'\n')
    dump(archive.with_name(archive.name+'.verification.json'),dict(files_verified=len(sums),bytes=archive.stat().st_size,sha256=digest))
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files_verified=len(sums),sha256=digest)),flush=True)


if __name__=='__main__':main()
