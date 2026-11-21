"""Run fresh YOLO and rectangular ResNet inference; reconnect fixed points for display."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import Counter

from infer_camera_video import ROOT, sha256, text, write_json
from diagnose_point_rewiring import reconnect
from infer_resnet_yolo_real import prepare_crop, to_original
from bop_bb8.net import ResNetFPN, decode_heatmaps, heatmap_to_input_points


from export_rect_points_video import compose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--yolo', type=Path, required=True)
    parser.add_argument('--ffmpeg', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ffmpeg = args.ffmpeg
    if not Path(ffmpeg).is_file():
        raise FileNotFoundError(ffmpeg)
    source = args.video
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError('Cannot open source video')
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    detector = YOLO(str(args.yolo))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    saved = checkpoint['args']
    assert saved['crop_mode'] == 'rect' and not saved.get('adaptation_arm')
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = ResNetFPN(backbone=saved['backbone'], fpn_dim=saved['fpn_dim'],
                      pretrained=False, heatmap_margin=saved['heatmap_margin']).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    video = args.output/'camera_pose_full_video.mp4'
    command = [ffmpeg, '-hide_banner', '-loglevel', 'warning', '-n',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s:v', '1920x1080', '-r', str(fps),
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-threads', '4', '-movflags', '+faststart', str(video)]
    write_json(args.output/'manifest.json', dict(source_video=str(source.resolve()), source_sha256=sha256(source),
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
        checkpoint_epoch_one_based=checkpoint['epoch']+1, model_args=saved, yolo=str(args.yolo.resolve()),
        yolo_sha256=sha256(args.yolo), fps=fps, total_frames=total, encoder_command=command,
        detector=dict(imgsz=960, conf=.25, iou=.7, half=False),
        pnp=False, temporal_smoothing=False, real_corner_finetuning=False,
        display_rewiring=True, point_coordinates_changed=False, semantic_correspondence_preserved=False,
        notes=['Fresh YOLO and ResNet inference on every frame; no cached detections.',
               'Tight rectangular RGB input; display zoom is not the network input.',
               'Rewiring changes edges only; original network point coordinates retained.',
               'YOLO trained on some frames of this video; this is visualization, not independent evaluation.']))
    capture = cv2.VideoCapture(str(source))
    started = time.perf_counter()
    processed, detections = 0, 0
    counts = Counter()
    snapshots = {0, 125, 500, 1483, 2349, total-1}
    try:
        with (args.output/'encoder.log').open('w') as log, (args.output/'predictions.jsonl').open('w') as output:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            try:
                with torch.inference_mode():
                    for frame_index in range(total):
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError(f'Failed to decode frame {processed}')
                        records = []
                        result = detector.predict(frame, imgsz=960, conf=.25, iou=.7,
                            device=0, half=False, verbose=False)[0]
                        boxes = result.boxes.xyxy.cpu().numpy()
                        scores = result.boxes.conf.cpu().numpy()
                        order = sorted(range(len(boxes)), key=lambda i: float(boxes[i, 0]+boxes[i, 2]))
                        for detection_index, i in enumerate(order):
                            old = dict(bbox_xyxy=boxes[i].tolist(), yolo_confidence=float(scores[i]),
                                       detection_index=detection_index)
                            roi, tensor, crop = prepare_crop(frame, old['bbox_xyxy'], tuple(saved['input_size']),
                                saved['crop_padding'], saved['crop_mode'], saved['crop_border'], saved['rect_sizes'])
                            size = (roi.shape[1], roi.shape[0])
                            logits = model(tensor[None].cuda()).float()
                            points = heatmap_to_input_points(decode_heatmaps(logits, saved['decoder'], saved['decode_radius']),
                                tuple(logits.shape[-2:]), size[::-1], margin=saved['heatmap_margin'])[0].cpu().numpy()
                            if not np.isfinite(points).all():
                                raise ValueError('Nonfinite predicted points')
                            records.append(dict(detection_index=old['detection_index'], bbox_xyxy=old['bbox_xyxy'],
                                yolo_confidence=old['yolo_confidence'], crop_xywh=list(crop), input_size=list(size),
                                corners_crop=points.tolist(), corners_original=to_original(points, crop, size).tolist()))
                        for record in records:
                            fit = reconnect(record['corners_original'])
                            np.testing.assert_array_equal(fit['points_original'], record['corners_original'])
                            record['display_edges'] = fit['edges']
                            record['rewiring'] = fit
                        counts[len(records)] += 1
                        canvas = compose(frame, records, processed, total, fps, rewired=True)
                        cv2.rectangle(canvas, (0, 35), (1407, 63), (31, 26, 21), -1)
                        text(canvas, 'New YOLO + photometric R18 | Rewired display; points unchanged | No PnP', (22, 52), .55)
                        encoder.stdin.write(canvas.tobytes())
                        output.write(json.dumps(dict(frame=processed, time_seconds=processed/fps, detections=records), allow_nan=False)+'\n')
                        if processed in snapshots:
                            cv2.imwrite(str(args.output/f'frame_{processed:06d}.jpg'), canvas)
                        processed += 1
                        detections += len(records)
                        if processed == 1 or processed % 100 == 0 or processed == total:
                            elapsed = time.perf_counter()-started
                            status = dict(state='running', processed_frames=processed, total_frames=total,
                                eta_seconds=round(elapsed/processed*(total-processed)), elapsed_seconds=round(elapsed, 1))
                            write_json(args.output/'status.json', status)
                            output.flush()
                            print(json.dumps(status), flush=True)
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError('ffmpeg encoding failed')
            finally:
                if encoder.poll() is None:
                    encoder.terminate()
                    encoder.wait()
        check = cv2.VideoCapture(str(video))
        count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = check.get(cv2.CAP_PROP_FPS)
        first_ok, _ = check.read()
        check.set(cv2.CAP_PROP_POS_FRAMES, count-1)
        last_ok, _ = check.read()
        check.release()
        assert count == total and first_ok and last_ok and abs(actual_fps-fps) < .001
        summary = dict(state='complete', processed_frames=processed, output_frames=count, every_frame_processed=True,
            total_detections=detections, frames_by_detection_count=dict(counts), fps=actual_fps, duration_seconds=count/actual_fps,
            bytes=video.stat().st_size, elapsed_seconds=time.perf_counter()-started, video_sha256=sha256(video))
        write_json(args.output/'summary.json', summary)
        write_json(args.output/'status.json', summary)
        (args.output/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>新YOLO＋成像增强ResNet18：逐帧视频</title>
<style>body{{background:#14212a;color:#eee;font:18px/1.7 sans-serif;margin:30px}}video{{width:100%;max-width:1600px}}a{{color:#8df}}</style>
<h1>新 YOLO＋成像增强 ResNet18</h1><p>全部 {count} 帧，{actual_fps} fps，{count/actual_fps:.2f} 秒。YOLO 使用侧面增强轮次最佳权重（第4轮），ResNet 使用成像增强最佳权重（第30轮）。</p>
<video controls preload="metadata" poster="frame_000000.jpg" src="camera_pose_full_video.mp4"></video>
<p><a href="camera_pose_full_video.mp4" download>下载 MP4（{video.stat().st_size/1e6:.1f} MB）</a> · <a href="manifest.json">处理设置</a> · <a href="summary.json">完整性检查</a> · <a href="predictions.jsonl">逐帧预测点与显示连线</a></p>
<p>青色为检测框，黄色为八个原始预测点，紫色为重新选择的长方体图连线。连线沿用之前的840种拓扑搜索，仅用于展示，不移动或补充预测点，也不保证物理角点对应正确。无PnP、无时序平滑、无真实角点微调。全部失败预测和漏检帧保留。</p>
<p>右侧是显示放大图；网络输入仍为紧矩形裁剪。D1/D2每帧按左右排序，不是跟踪ID。这段视频中部分帧用于YOLO训练，并曾用于背景和姿态设计，因此不是独立盲测。</p>''')
        print(json.dumps(summary), flush=True)
    except Exception as error:
        write_json(args.output/'status.json', dict(state='failed', processed_frames=processed, error=str(error)))
        raise
    finally:
        capture.release()


if __name__ == '__main__':
    main()
