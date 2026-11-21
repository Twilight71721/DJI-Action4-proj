"""Package one random-background server training run with small masks and mixed photo assets."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]
NAME='resnet5000_background_mixed_20260923'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=ROOT/'exports'/NAME;out.mkdir(parents=True,exist_ok=False)
    files=[p.relative_to(ROOT) for p in (ROOT/'bop_bb8').rglob('*.py') if p.name not in ['rigid.py','joint_decode.py']]
    files += [Path('scripts')/name for name in ['train_heatmap.py','run_background_training.py','server_background_assets.py',
        'evaluate_background_model.py','preview_background_training.py','prepare_camera_masks.py',
        'evaluate_cleanblur5000.py','evaluate_heatmap_checkpoint.py','infer_resnet_yolo_real.py']]
    files += [Path('tests')/name for name in ['test_background_augmentation.py','test_crop_jitter.py','test_net.py','test_bop_bb8.py','test_blur_augmentation.py']]
    files += [Path('requirements.txt'),Path('configs/background_train_20260923.json'),Path('configs/background_data_hashes.json')]
    for rel in files:
        (out/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,out/rel)
    assets=ROOT/'assets/background_randomization_20260923';dest=out/'augmentation';dest.mkdir()
    for folder in ['images','camera_masks']:shutil.copytree(assets/folder,dest/folder)
    for name in ['manifest.json','SOURCES.md','contact_sheet.jpg']:shutil.copy2(assets/name,dest/name)
    shutil.copy2(ROOT/'docs/BACKGROUND_SERVER_20260923.md',out/'OPENCODE_SERVER_GUIDE.md')
    shutil.copy2(ROOT/'prompts/opencode_background_mixed_20260923.txt',out/'OPENCODE_PROMPT.txt')
    (out/'START_HERE.md').write_text('# 单一随机背景训练方案\n\n本包替代之前尚未执行的A/B包，不启动A/B训练。'
        '先阅读OPENCODE_SERVER_GUIDE.md，给OpenCode的完整提示词在OPENCODE_PROMPT.txt。\n\n'
        '已包含1536张统一混合背景和8000个机身投影mask。复用服务器已有5000数据和模型，不重新上传大数据。'
        '本地仅做短程检查，正式服务器训练尚未启动。\n\n'
        '检查SHA256SUMS后，在tmux运行文档中的run_background_training.py --action all。'
        '80%背景替换、20%原图，清晰/模糊配对，裁剪扰动，单模型30轮。\n')
    shutil.copytree(ROOT/'exports/resnet5000_crop_ab_delta_20260923/real',out/'real')
    checks=out/'local_verification';checks.mkdir()
    preparation=ROOT/'outputs/background_training_preparation_20260923'
    shutil.copytree(preparation/'preview_final',checks/'preview')
    (checks/'verification.json').write_text(json.dumps(dict(
        preview=json.loads((preparation/'preview_final/verification.json').read_text()),unit_tests=52,
        background_counts=json.loads((assets/'manifest.json').read_text())['counts'],
        mask_policy=json.loads((assets/'camera_masks/manifest.json').read_text())['mask_policy'],
        training='local smoke only; formal run must execute on server'),indent=2))
    smoke=preparation/'local_smoke_final/smoke_training';(checks/'smoke').mkdir()
    for name in ['metrics.csv','baseline.json','status.json','config.json','command.json','train.log']:
        shutil.copy2(smoke/name,checks/'smoke'/name)
    final=preparation/'FINAL_VERIFICATION.json'
    if final.exists():shutil.copy2(final,checks/'FINAL_VERIFICATION.json')
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in sorted(out.rglob('*')) if p.is_file()))
    archive=out.with_suffix('.tar.gz')
    if archive.exists():raise FileExistsError(archive)
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for path in sorted(out.rglob('*')):
            if path.is_file():tar.add(path,arcname=str(Path(NAME)/path.relative_to(out)),recursive=False)
    archive.with_name(archive.name+'.sha256').write_text(f'{sha(archive)}  {archive.name}\n')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),indent=2))


if __name__=='__main__':main()
