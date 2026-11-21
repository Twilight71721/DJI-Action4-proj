"""Keep the existing detector fixed and evaluate the tight-crop ResNet on real RGB."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['real-manifest', 'yolo', 'resnet', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--box-definition', choices=['body', 'cuboid'], default='body')
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    checkpoint = torch.load(args.resnet, map_location='cpu', weights_only=False)
    saved = checkpoint['args']
    rectangular = saved['crop_mode'] == 'rect'
    if saved['crop_mode'] not in {'cuboid', 'rect'} or saved['crop_padding'] != 1.0 or saved.get('adaptation_arm'):
        raise ValueError('Use a tight-crop checkpoint without real-corner adaptation')
    if rectangular and (saved.get('input_contract_version') != 2 or saved.get('crop_border') != 0
                        or args.box_definition != 'body'):
        raise ValueError('Rectangular protocol-v2 requires unpadded body detections')
    del checkpoint
    yolo_digest = hashlib.sha256(args.yolo.read_bytes()).hexdigest()
    bundle_manifest = ROOT/'BUNDLE_MANIFEST.json'
    if bundle_manifest.is_file():
        expected = json.loads(bundle_manifest.read_text()).get('yolo_frozen_sha256')
        if expected and yolo_digest != expected:
            raise ValueError('YOLO differs from the frozen checkpoint recorded in the package')
    detector = YOLO(str(args.yolo))
    if len(detector.names) != 1:
        raise ValueError('Use the existing single-class Action4 detector')
    if args.box_definition == 'cuboid' and list(detector.names.values()) != ['action4_cuboid']:
        raise ValueError('Do not relabel existing body boxes as cuboid boxes')
    inputs = []
    seen = set()
    # Only RGB paths are used. Stored manual/old YOLO boxes are not inference inputs.
    for r in json.loads(args.real_manifest.read_text())['records']:
        image_path = (args.real_manifest.parent/r['image']).resolve()
        if image_path not in seen:
            inputs.append((image_path, r.get('frame')))
            seen.add(image_path)
    if not inputs:
        raise ValueError('Empty real-frame manifest')
    if len({p.stem for p, _ in inputs}) != len(inputs):
        raise ValueError('Real source basenames must be unique for the review exporter')
    args.output.mkdir(parents=True, exist_ok=False)
    predictions = []
    for source, frame in inputs:
        rgb = cv2.imread(str(source))
        if rgb is None:
            raise FileNotFoundError(source)
        result = detector.predict(rgb, imgsz=960, conf=.25, iou=.7,
                                  device=0, half=False, verbose=False)[0]
        boxes = result.boxes
        predictions.append(dict(source=str(source), frame=frame,
            xyxy=boxes.xyxy.cpu().numpy().tolist(), confidence=boxes.conf.cpu().numpy().tolist()))
    summary = dict(input_frames=len(inputs), detections=sum(len(r['xyxy']) for r in predictions),
        frames_without_detection=sum(not r['xyxy'] for r in predictions),
        detector_box_definition=args.box_definition, detector_retrained=False,
        body_box_approximation=not rectangular and args.box_definition == 'body',
        crop_mode=saved['crop_mode'], rgb_padding=not rectangular,
        imgsz=960, confidence_threshold=.25, nms_iou=.7,
        yolo_sha256=yolo_digest,
        resnet_sha256=hashlib.sha256(args.resnet.read_bytes()).hexdigest(),
        ground_truth_boxes_used=False, ground_truth_corners_used=False,
        crop_caveat=('Training uses synthetic bbox_obj; inference uses frozen YOLO body boxes. '
            'RGB is a tight rectangle, with no gray padding. Output heatmap margin retains off-ROI corners.'
            if rectangular else 'Training uses GT cuboid bounds; unchanged body detections are approximate bounds.'),
        caveat='Real video was used for background/pose design; this is a qualitative diagnostic, not a blind real test.')
    path = args.output/'detections.json'
    path.write_text(json.dumps(dict(summary=summary, predictions=predictions), indent=2))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    if summary['detections']:
        command = [sys.executable, str(ROOT/'scripts/infer_resnet_yolo_real.py'),
            '--detections', str(path), '--checkpoint', str(args.resnet),
            '--box-definition', args.box_definition, '--output', str(args.output/'network')]
        if args.box_definition == 'body' and not rectangular:
            command.append('--allow-body-box-approximation')
        subprocess.run(command, check=True)
        detail = '<a href="network/index.html">查看检测框与 ResNet 八角点结果</a>'
    else:
        detail = '<p>所有真实帧均未检出目标。检测器失败，未运行角点网络。没有改用人工真值框补救。</p>'
    (args.output/'index.html').write_text('<meta charset="utf-8"><h1>真实帧端到端诊断</h1>'
        f'<p>{len(inputs)} 帧，{summary["detections"]} 个检测，{summary["frames_without_detection"]} 帧无检测。'
        '没有使用真实框或角点标签；所有检测候选保留。保留既有 YOLO。'
        + ('网络输入为机身紧裁剪长方形，无灰色补边；图外角点由扩展热图预测。' if rectangular else
           '训练的八角外接框与预测的机身框有定义差异，不能保证八角点均位于裁剪内。')
        + '真实视频已用于背景和姿态设计，不是独立盲测。</p>'+detail)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
