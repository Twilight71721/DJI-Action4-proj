"""Package the photometric-only server experiment without duplicating existing datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_rectangular_bundle import ROOT, stage as stage_rect, copy, sha, sums, pack

NAME = 'resnet18_photometric_20260926'
CONFIG = 'single_rgb_rect_photo_balanced_20260926.json'


def stage(output):
    stage_rect(output)
    copy(ROOT/'configs'/CONFIG,output/'configs'/CONFIG)
    for name in ['photometric_views','paired_background']:
        copy(ROOT/f'tests/test_{name}.py',output/f'tests/test_{name}.py')
    copy(ROOT/'docs/PHOTOMETRIC_SERVER_20260926.md',output/'OPENCODE_SERVER_GUIDE.md')
    copy(ROOT/'prompts/opencode_photometric_20260926.txt',output/'OPENCODE_PROMPT.txt')
    copy(ROOT/'docs/PHOTOMETRIC_VIEW_EXPERIMENT_20260926.md',output/'docs/LOCAL_PILOT_PROTOCOL.md')
    copy(ROOT/'docs/RECTANGULAR_SERVER_PACKAGE_20260926.md',output/'docs/PREVIOUS_RECTANGULAR_PROTOCOL.md')
    baseline=ROOT/'outputs/server_return_rect_r18_20260926/rect_r18_fixed_yolo_30ep'
    for rel in ['summary.json','training/config.json','training/metrics.csv','real/network/predictions.json']:
        copy(baseline/rel,output/'reference_baseline'/rel)
    for name in ['evaluation.json','ANALYSIS.md']:
        copy(ROOT/'outputs/photo_view_pilot_20260926'/name,output/'local_pilot'/name)
    manifest=json.loads((output/'BUNDLE_MANIFEST.json').read_text())
    manifest.update(name=NAME,version=2,experiment='photometric_balanced_only',
        recipe='configs/'+CONFIG,expected_result='rect_r18_photometric_30ep.tar.gz',
        augmentation=dict(profile='balanced',unchanged_probability=.2,gentle_probability=.68,
                          moderate_probability=.12,geometry_changes=False,online_each_epoch=True),
        view_balance=False,paired_background_consistency=False,
        reference_baseline_summary_sha256=sha(baseline/'summary.json'))
    (output/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'START_HERE.md').write_text('# 成像增强训练包\n\n先阅读 OPENCODE_SERVER_GUIDE.md。'
        '完整操作提示在 OPENCODE_PROMPT.txt。\n\n复用服务器完整数据，训练30轮，'
        '从包内 ImageNet 权重初始化。在线成像增强约20%不变、68%轻度、12%中等；'
        '固定YOLO、矩形输入、原始八点编号和验证划分。无需上传大批图片。\n'
        '\nlocal_aug_preview/ 是本轮增强预览；local_input_preview/ 是旧裁剪协议示意。'
        '服务器还会生成 input_preview/，记录实际训练数据的增强结果。\n')
    sums(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['stage','pack'],required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'exports'/NAME)
    args=parser.parse_args()
    (stage if args.action=='stage' else pack)(args.output.resolve())
