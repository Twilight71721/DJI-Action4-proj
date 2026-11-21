"""Package tight-crop training code; reuse existing server RGB and foregrounds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from build_continuous_pose_bundle import pack
from server_background_assets import sha

ROOT = Path(__file__).resolve().parents[1]
NAME = 'resnet_tight_cuboid_fixed_yolo_20260925'


def stage(out):
    out.mkdir(parents=True, exist_ok=False)
    files = [p.relative_to(ROOT) for p in (ROOT/'bop_bb8').rglob('*.py')]
    files += [Path('scripts')/name for name in [
        'train_heatmap.py', 'run_online_pose_training.py', 'prepare_online_pose_backgrounds.py',
        'server_background_assets.py', 'prepare_camera_masks.py', 'prepare_tight_cuboid.py',
        'evaluate_tight_cuboid.py', 'evaluate_cleanblur5000.py', 'evaluate_heatmap_checkpoint.py',
        'evaluate_background_model.py', 'evaluate_online_pose_holdout.py',
        'infer_resnet_yolo_real.py', 'infer_camera_video.py',
        'predict_tight_cuboid_real.py']]
    files += [Path('tests')/name for name in [
        'test_tight_crops.py', 'test_online_background.py', 'test_background_augmentation.py',
        'test_bop_bb8.py', 'test_net.py', 'test_crop_jitter.py']]
    files += [Path('configs')/name for name in ['tight_cuboid_20260925.json', 'background_data_hashes.json']]
    files += [Path('requirements.txt'), Path('requirements-yolo.txt'),
              Path('docs/TIGHT_CUBOID_INPUT_20260925.md')]
    for rel in files:
        target = out/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, target)
    shutil.copy2(ROOT/'docs/TIGHT_CUBOID_SERVER_PACKAGE_20260925.md', out/'OPENCODE_SERVER_GUIDE.md')
    shutil.copy2(ROOT/'prompts/opencode_tight_cuboid_20260925.txt', out/'OPENCODE_PROMPT.txt')
    shutil.copytree(ROOT/'exports/resnet_pose_mix_no_occlusion_20260924/real', out/'real')
    (out/'weights').mkdir()
    source = ROOT/'outputs/yolo11s_cleanblur5000_real48_20260925/train/weights/best.pt'
    shutil.copy2(source, out/'weights/yolo11s_frozen.pt')
    provenance = dict(yolo_frozen_sha256=sha(source), yolo_training=False,
        yolo_role='Existing body-box detector used unchanged for inference; GT-cuboid/body-box crop mismatch is explicit',
        resnet_config='configs/tight_cuboid_20260925.json',
        no_glb_or_material_changes=True, resnet_real_corner_finetuning=False,
        server_dependencies=['base clean/blur dataset', '4352 continuous-pose RGBA assets',
                             'mixed background pool', 'legacy camera masks',
                             'prepared continuous_data and continuous_pose epoch27 best checkpoint'])
    (out/'BUNDLE_MANIFEST.json').write_text(json.dumps(provenance, indent=2)+'\n')
    checks = out/'local_verification'
    preview = checks/'tight_crops'
    preview.mkdir(parents=True)
    source = ROOT/'dataset/action4-tight-cuboid-20260925'
    for name in ['summary.json', 'verification.json', 'crop_index.jsonl', 'real_cuboid_reference.json']:
        shutil.copy2(source/name, preview/name)
    shutil.copytree(source/'review', preview/'review')
    page = preview/'review/index.html'
    page.write_text(page.read_text().replace('../yolo/data.yaml', '../../../OPENCODE_SERVER_GUIDE.md'))
    smoke = checks/'gpu_smoke'
    smoke.mkdir()
    source = ROOT/'outputs/tight_cuboid_smoke_20260925'
    for name in ['config.json', 'metrics.csv', 'baseline.json']:
        shutil.copy2(source/'training'/name, smoke/name)
    shutil.copy2(source/'train.log', smoke/'train.log')
    for name in ['review', 'pose_holdout']:
        shutil.copy2(source/'evaluation_checked'/name/'summary.json', smoke/f'{name}_limited_summary.json')
    (out/'START_HERE.md').write_text(
        '# 紧八角框训练增量包\n\n先完整阅读 [服务器操作说明](OPENCODE_SERVER_GUIDE.md)，'
        '给 OpenCode 的指令在 [OPENCODE_PROMPT.txt](OPENCODE_PROMPT.txt)。\n\n'
        '复用上一轮服务器资产，不重新传输大型图像、RGBA、背景或 ResNet 权重。'
        '本包代码独立，不要覆盖旧项目。只训练 ResNet 30 轮，再用固定 YOLO 预测真实帧并打包；不训练或修改 YOLO。'
        '正式训练尚未在本地执行；local_verification 仅含输入审校与少量流程检查。'
        '没有修改 GLB 或侧面文字。\n')
    print('STAGED', out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['stage', 'pack'], required=True)
    args = parser.parse_args()
    output = ROOT/'exports'/NAME
    stage(output) if args.action == 'stage' else pack(output)
