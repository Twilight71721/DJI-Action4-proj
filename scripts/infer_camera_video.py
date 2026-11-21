"""Run existing YOLO and BB8 checkpoints on every video frame and export a review."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8.net import decode_topk, heatmap_to_input_points
from bop_bb8.visualize import EDGES
from evaluate_cleanblur5000 import load_model
from infer_resnet_yolo_real import prepare_crop, to_original


BBOX_COLOR = (255, 215, 20)
EDGE_COLOR = (225, 55, 225)
POINT_COLOR = (0, 240, 255)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def text(image, message, position, scale=.6, color=(240, 243, 247), thickness=1):
    cv2.putText(image, message, position, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def wire(image, points, thickness=2, numbered=True):
    if not np.isfinite(points).all():
        return
    points = np.rint(points).astype(int)
    for a, b in EDGES:
        cv2.line(image, tuple(points[a]), tuple(points[b]), EDGE_COLOR, thickness, cv2.LINE_AA)
    for index, point in enumerate(points):
        cv2.circle(image, tuple(point), 4, POINT_COLOR, -1, cv2.LINE_AA)
        if numbered:
            text(image, str(index), tuple(point + [5, -5]), .47, POINT_COLOR)


def compose(frame, records, rois, frame_index, frame_count, fps, epoch, tight=False):
    canvas = np.full((1080, 1920, 3), (31, 26, 21), np.uint8)
    height, width = frame.shape[:2]
    scale = min(1408 / width, 1008 / height)
    target = (round(width * scale), round(height * scale))
    ox, oy = (1408-target[0])//2, 64+(1008-target[1])//2
    full = cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
    for r in records:
        box = np.rint(np.array(r['bbox_xyxy']).reshape(2, 2) * scale).astype(int)
        cv2.rectangle(full, tuple(box[0]), tuple(box[1]), BBOX_COLOR, 2)
        text(full, f"D{r['detection_index'] + 1} {r['yolo_confidence']:.2f}",
             (int(box[0, 0]), max(17, int(box[0, 1])-7)), .48, BBOX_COLOR)
        if r['finite_corners']:
            wire(full, np.array(r['corners_original']) * scale, 1, False)
    canvas[oy:oy+target[1], ox:ox+target[0]] = full
    text(canvas, f'Frame {frame_index:04d} / {frame_count-1:04d}    '
         f'Time {frame_index/fps:06.2f}s    Detections {len(records)}', (22, 27), .65)
    crop_label = 'tight cuboid' if tight else 'continuous-pose'
    text(canvas, f'ResNet50-FPN: {crop_label} best epoch {epoch}; no real-corner fine-tuning',
         (22, 53), .56)
    text(canvas, 'BBOX', (1433, 27), .53, BBOX_COLOR)
    text(canvas, 'PREDICTED 8 CORNERS', (1540, 27), .53, EDGE_COLOR)
    text(canvas, 'Per-frame output; no temporal smoothing', (1433, 52), .48)
    for slot in range(2):
        y = 64 + slot*504
        cv2.rectangle(canvas, (1408, y), (1919, y+503), (48, 45, 39), -1)
        if slot >= len(records):
            text(canvas, 'No additional YOLO detection', (1440, y+46), .55)
            continue
        r = records[slot]
        text(canvas, f"D{slot+1}  YOLO confidence {r['yolo_confidence']:.3f}",
             (1430, y+27), .58)
        roi = cv2.resize(rois[slot], (448, 448), interpolation=cv2.INTER_LINEAR)
        crop = r['crop_xywh']
        b = ((np.array(r['bbox_xyxy']).reshape(2, 2)-np.array(crop[:2])) /
             np.array(crop[2:]) * 448)
        b = np.rint(b).astype(int)
        cv2.rectangle(roi, tuple(b[0]), tuple(b[1]), BBOX_COLOR, 1)
        if r['finite_corners']:
            wire(roi, np.array(r['corners_crop']) * (448 / 256), 2)
        else:
            text(roi, 'INVALID CORNER OUTPUT', (15, 28), .6, (0, 70, 255), 2)
        canvas[y+40:y+488, 1440:1888] = roi
    if not records:
        text(canvas, 'NO YOLO DETECTION ON THIS FRAME', (50, 130), .85, (0, 100, 255), 2)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--yolo', type=Path, required=True)
    parser.add_argument('--resnet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ffmpeg', default=shutil.which('ffmpeg'))
    parser.add_argument('--limit', type=int, default=0, help='Smoke test only; full export requires 0')
    parser.add_argument('--allow-body-box-approximation', action='store_true',
                        help='Keep the existing body detector with tight ResNet; record crop-definition mismatch')
    args = parser.parse_args()
    if not args.ffmpeg:
        parser.error('An ffmpeg executable is required')
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output.resolve()
    started = time.perf_counter()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f'Cannot open {args.video}')
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_size = [int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    if not np.isfinite(fps) or fps <= 0 or source_count <= 0:
        raise ValueError('Invalid source video metadata')
    expected = min(args.limit, source_count) if args.limit else source_count
    model, metadata = load_model(args.resnet.resolve(), torch.device('cuda'), allow_tight=True)
    crop_mode = metadata['args']['crop_mode']
    crop_padding = metadata['args']['crop_padding']
    crop_border = metadata['args'].get('crop_border', 4)
    if metadata['args'].get('adaptation_arm'):
        raise ValueError('This export is for a checkpoint before real-corner adaptation')
    from ultralytics import YOLO
    detector = YOLO(str(args.yolo))
    approximate_body_box = crop_mode == 'cuboid' and list(detector.names.values()) != ['action4_cuboid']
    if approximate_body_box and not args.allow_body_box_approximation:
        raise ValueError('Pass --allow-body-box-approximation to explicitly keep the existing body detector')
    video_path = output / 'camera_pose_full_video.mp4'
    command = [str(args.ffmpeg), '-hide_banner', '-loglevel', 'warning', '-n',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s:v', '1920x1080',
               '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
               '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-threads', '4', '-movflags', '+faststart', str(video_path)]
    manifest = dict(source_video=str(args.video.resolve()), source_sha256=sha256(args.video),
                    source_frames=source_count, source_size=source_size, fps=fps,
                    expected_processed_frames=expected, incomplete_smoke_test=bool(args.limit),
                    resnet=metadata, yolo=str(args.yolo.resolve()), yolo_sha256=sha256(args.yolo),
                    detector=dict(imgsz=960, conf=.25, nms_iou=.7, half=False),
                    inference=f'FP32, crop_mode={crop_mode}, padding={crop_padding}, border={crop_border}, input 256x256, stable weighted top9 from logits',
                    output_size=[1920, 1080], temporal_smoothing=False, tracking=False,
                    body_box_approximation=approximate_body_box,
                    notes=['ResNet checkpoint precedes real-corner supervision; inspect YOLO training provenance separately.',
                           'No filtering or correction of finite corner predictions. No ground-truth overlays.',
                           'Detections sorted left to right independently per frame, not persistent IDs.',
                           'All detections shown on full frame; up to two leftmost crops enlarged.',
                           'Video already used for background/pose design; not independent blind evaluation.',
                           'Source video contains no audio stream.'], encoder_command=command)
    write_json(output / 'manifest.json', manifest)
    counts, bad_points, processed = Counter(), 0, 0
    snapshot_frames = {0, 125, 500, 1350, 1483, 1549, 2150, 2349, expected-1}
    encoder = None
    try:
        with (output/'encoder.log').open('w') as encoder_log, (output/'predictions.jsonl').open('w') as predictions:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=encoder_log)
            with torch.inference_mode():
                for frame_index in range(expected):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f'Premature video decode failure at frame {frame_index}')
                    detections = detector.predict(frame, imgsz=960, conf=.25, iou=.7,
                                                  device=0, half=False, verbose=False)[0]
                    boxes = detections.boxes.xyxy.cpu().numpy()
                    confidences = detections.boxes.conf.cpu().numpy()
                    order = sorted(range(len(boxes)), key=lambda i: float(boxes[i, 0]+boxes[i, 2]))
                    tensors, rois, records = [], [], []
                    for detection_index, i in enumerate(order):
                        roi, tensor, crop = prepare_crop(frame, boxes[i], (256, 256), crop_padding,
                                                        crop_mode, crop_border)
                        rois.append(roi)
                        tensors.append(tensor)
                        records.append(dict(detection_index=detection_index, bbox_xyxy=boxes[i].tolist(),
                                            yolo_confidence=float(confidences[i]), crop_xywh=list(crop)))
                    for begin in range(0, len(tensors), 4):
                        logits = model(torch.stack(tensors[begin:begin+4]).cuda()).float()
                        points = heatmap_to_input_points(decode_topk(logits, from_logits=True),
                                                        tuple(logits.shape[-2:]), (256, 256)).cpu().numpy()
                        peaks = logits.sigmoid().amax(dim=(-2, -1)).cpu().numpy()
                        for offset, (p, peak) in enumerate(zip(points, peaks), begin):
                            finite = bool(np.isfinite(p).all())
                            bad_points += int(not finite)
                            r = records[offset]
                            r.update(finite_corners=finite,
                                     corners_crop=p.tolist() if finite else None,
                                     corners_original=to_original(p, r['crop_xywh'], (256, 256)).tolist() if finite else None,
                                     heatmap_peaks=peak.tolist() if np.isfinite(peak).all() else None)
                    canvas = compose(frame, records, rois, frame_index, expected, fps, metadata['epoch_one_based'],
                                     crop_mode == 'cuboid')
                    encoder.stdin.write(canvas.tobytes())
                    predictions.write(json.dumps(dict(frame=frame_index, time_seconds=frame_index/fps,
                                                      detections=records), allow_nan=False)+'\n')
                    if frame_index in snapshot_frames:
                        cv2.imwrite(str(output/f'frame_{frame_index:06d}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
                    counts[len(records)] += 1
                    processed = frame_index+1
                    if processed == 1 or processed % 100 == 0 or processed == expected:
                        predictions.flush()
                        elapsed = time.perf_counter()-started
                        progress = dict(state='running', processed_frames=processed, total_frames=expected,
                                        elapsed_seconds=round(elapsed, 2), frames_per_second=round(processed/elapsed, 2),
                                        eta_seconds=round((expected-processed)*elapsed/processed),
                                        frames_by_detection_count=dict(counts))
                        write_json(output/'status.json', progress)
                        print(json.dumps(progress), flush=True)
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('Video encoding failed; see encoder.log')
        capture.release()
        check = cv2.VideoCapture(str(video_path))
        saved_count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
        saved_fps = float(check.get(cv2.CAP_PROP_FPS))
        saved_size = [int(check.get(cv2.CAP_PROP_FRAME_WIDTH)), int(check.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        first_ok, _ = check.read()
        check.set(cv2.CAP_PROP_POS_FRAMES, saved_count-1)
        last_ok, _ = check.read()
        check.release()
        if saved_count != expected or not first_ok or not last_ok or abs(saved_fps-fps) > 1e-3:
            raise RuntimeError(f'Output integrity check failed: {saved_count}, {saved_fps}, {first_ok}, {last_ok}')
        summary = dict(state='complete', processed_frames=processed, source_frames=source_count,
                       every_frame_processed=processed == source_count, output_frames=saved_count,
                       fps=saved_fps, duration_seconds=saved_count/saved_fps, output_size=saved_size,
                       frames_by_detection_count=dict(sorted(counts.items())),
                       total_detections=sum(n*count for n, count in counts.items()),
                       nonfinite_corner_instances=bad_points, bytes=video_path.stat().st_size,
                       elapsed_seconds=time.perf_counter()-started, video_sha256=sha256(video_path),
                       resnet_checkpoint_sha256=metadata['sha256'], resnet_epoch=metadata['epoch_one_based'])
        write_json(output/'summary.json', summary)
        write_json(output/'status.json', summary)
        details = ''.join(f'<a href="frame_{i:06d}.jpg">第 {i} 帧</a> '
                          for i in sorted(snapshot_frames) if (output/f'frame_{i:06d}.jpg').exists())
        page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>整段视频：YOLO＋未使用真实八点微调的 ResNet</title>
<style>body{{max-width:1600px;margin:30px auto;padding:0 20px;background:#14212a;color:#edf3f7;font:17px/1.7 sans-serif}}video{{width:100%;background:#000}}a{{color:#87d8ff}}code{{overflow-wrap:anywhere}}</style>
<h1>整段视频逐帧预测</h1>
<p>ResNet：第 {metadata['epoch_one_based']} 轮最佳权重，裁剪模式 {crop_mode}，未使用真实八角点微调。YOLO 权重及训练来源见模型设置清单。使用旧机身框近似紧八角框：{approximate_body_box}；机身框可能截掉部分投影角点。</p>
<p>全部 {processed} 帧，{saved_fps:g} fps，{saved_count/saved_fps:.2f} 秒。逐帧独立推理，无时序平滑、插值补框或人工角点修正。</p>
<video controls preload="metadata" poster="frame_000000.jpg" src="camera_pose_full_video.mp4"></video>
<p><a href="camera_pose_full_video.mp4" download>下载 MP4（{video_path.stat().st_size/1024**2:.1f} MiB）</a> · <a href="predictions.jsonl">逐帧预测</a> · <a href="manifest.json">模型及处理设置</a> · <a href="summary.json">完整性统计</a></p>
<p>青色为 YOLO bbox，紫色连线与黄色 0–7 点为 ResNet 预测，无真值叠加。右侧放大显示按图像左右排序的前两个检测，每帧重新排序，D1/D2 不是跟踪 ID。YOLO 置信度不是角点正确概率。</p>
<p>每帧检测数量分布：{html.escape(str(dict(sorted(counts.items()))))}。所有有限角点预测均保留，包括失败的框；检测数量不能作为准确率。该视频曾用于背景和姿态设计，不是独立盲测。</p>
<p>静态检查：{details}</p></html>'''
        (output/'index.html').write_text(page)
        (output/'README.md').write_text(
            '逐帧视频导出。使用未进行真实八点微调的连续姿态最佳 ResNet，最新 YOLO 检测器。\n\n'
            '查看 index.html；发送 camera_pose_full_video.mp4。青色 bbox、紫色预测连线、黄色角点。\n\n'
            '无跳帧、无时序平滑、无人工修正、无真值叠加。源码脚本 scripts/infer_camera_video.py；'
            'manifest.json 保存权重 SHA256 和参数，predictions.jsonl 每帧一条。\n')
        print(json.dumps(summary), flush=True)
    except Exception as error:
        write_json(output/'status.json', dict(state='failed', processed_frames=processed,
                                              total_frames=expected, error=str(error)))
        raise
    finally:
        capture.release()
        if encoder is not None and encoder.poll() is None:
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            encoder.wait()


if __name__ == '__main__':
    main()
