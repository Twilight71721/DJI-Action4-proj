"""Build a small standalone code/real-frame bundle; reuse server dataset and checkpoint."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'resnet5000_crop_ab_delta_20260923'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT/'exports'/NAME
    out.mkdir(parents=True, exist_ok=False)
    files = [p.relative_to(ROOT) for p in (ROOT/'bop_bb8').rglob('*.py')
             if p.name not in ['rigid.py','joint_decode.py']]
    files += [Path('scripts')/name for name in [
        'train_heatmap.py','run_crop_ab.py','evaluate_crop_ab.py','preview_crop_jitter.py',
        'evaluate_cleanblur5000.py','evaluate_heatmap_checkpoint.py','infer_resnet_yolo_real.py']]
    files += [Path('tests')/name for name in [
        'test_crop_jitter.py','test_net.py','test_bop_bb8.py','test_blur_augmentation.py']]
    files += [Path('requirements.txt'), Path('configs/crop_ab_20260923.json'), Path('configs/crop_ab_data_hashes.json')]
    for relative in files:
        dst = out/relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, dst)
    shutil.copy2(ROOT/'docs/CROP_AB_SERVER_20260923.md',out/'OPENCODE_SERVER_GUIDE.md')
    shutil.copy2(ROOT/'prompts/opencode_crop_ab_20260923.txt',out/'OPENCODE_PROMPT.txt')
    (out/'START_HERE.md').write_text(
        '# 只需上传这个小包\n\n先阅读 OPENCODE_SERVER_GUIDE.md，给OpenCode的提示词在 OPENCODE_PROMPT.txt。\n'
        '复用已有5000清晰/模糊数据和5000模型best.pth，不覆盖旧项目；本包不包含大型训练数据或权重。\n'
        '正式训练尚未在服务器开始。本地仅做标签验证和短程管线检查。\n'
        '请先校验SHA256SUMS，定位已有资产，按文档check、preview，然后在tmux执行--action all。\n')
    frame_dir = out/'real/frame0'
    frame_dir.mkdir(parents=True)
    shutil.copy2(ROOT/'outputs/uploaded_frame_bbox_20260923/source.jpg',frame_dir/'frame_000000.jpg')
    (frame_dir/'manifest.json').write_text(json.dumps(dict(records=[
        dict(frame=0, method='manual_uploaded', instance=0, box=[1136,996,1676,1420], image='frame_000000.jpg'),
        dict(frame=0, method='manual_uploaded', instance=1, box=[2100,1012,2740,1428], image='frame_000000.jpg')],
        protocol='Qualitative diagnostics only; no reliable corner GT; excluded from training.'),indent=2))
    preview = ROOT/'outputs/crop_ab_preparation_20260923/preview'
    shutil.copytree(preview, out/'local_verification/preview')
    checks = dict(preview=json.loads((preview/'verification.json').read_text()),
        tests={'crop_jitter':5,'network':18,'geometry':20,'blur':5},
        local_smoke=dict(batch_size=4, epochs=1, train_instances=32, val_instances=32,
                         precision='bfloat16', optimizer_updates_per_arm=8, formal_training=False),
        fp16_trial='Initial strict FP16 smoke failed on GradScaler overflow/skip; both arms then passed with BF16.',
        server_smoke='Must repeat with actual 4090 batch32; local smoke does not prove 4090 memory capacity.')
    smoke=ROOT/'outputs/crop_ab_preparation_20260923/local_smoke_bf16'
    for arm in ['A','B']:
        src=smoke/('smoke_'+arm)
        checks['local_smoke'][arm]=dict(status=json.loads((src/'status.json').read_text()),
            baseline=json.loads((src/'baseline.json').read_text()))
        dst=out/'local_verification'/('smoke_'+arm)
        dst.mkdir()
        for file in ['metrics.csv','baseline.json','status.json','train.log','command.json','config.json']:
            shutil.copy2(src/file,dst/file)
    (out/'local_verification/BUILD_VERIFICATION.json').write_text(json.dumps(checks,indent=2))
    final_checks = ROOT/'outputs/crop_ab_preparation_20260923/FINAL_VERIFICATION.json'
    if final_checks.exists():
        shutil.copy2(final_checks, out/'local_verification/FINAL_VERIFICATION.json')
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in sorted(out.rglob('*')) if p.is_file()))
    archive=out.with_suffix('.tar.gz')
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(out.rglob('*')):
            if p.is_file():
                tar.add(p,arcname=str(Path(NAME)/p.relative_to(out)),recursive=False)
    archive.with_name(archive.name+'.sha256').write_text(f'{sha(archive)}  {archive.name}\n')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),indent=2))


if __name__=='__main__':
    main()
