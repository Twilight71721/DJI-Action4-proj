"""Render every frame with rectangular-model corners, reusing verified YOLO detections."""
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

from infer_camera_video import ROOT, sha256, text, write_json
from infer_resnet_yolo_real import prepare_crop, to_original
from bop_bb8.net import ResNetFPN, decode_heatmaps, heatmap_to_input_points


def annotate(image, records, origin, scale, numbered):
    for record in records:
        box = np.rint((np.array(record['bbox_xyxy']).reshape(2, 2)-origin)*scale).astype(int)
        cv2.rectangle(image, tuple(box[0]), tuple(box[1]), (255, 215, 20), 1)
        points = np.rint((np.array(record['corners_original'])-origin)*scale).astype(int)
        for a, b in record.get('display_edges', []):
            cv2.line(image, tuple(points[a]), tuple(points[b]), (225, 55, 225), 1, cv2.LINE_AA)
        for i, point in enumerate(points):
            cv2.circle(image, tuple(point), 4, (0, 240, 255), -1, cv2.LINE_AA)
            if numbered:
                text(image, str(i), tuple(point+[5, -5]), .5, (0, 240, 255))


def compose(frame, records, index, total, fps, rewired=False):
    canvas = np.full((1080, 1920, 3), (31, 26, 21), np.uint8)
    h, w = frame.shape[:2]
    scale = min(1408/w, 1008/h)
    tw, th = round(w*scale), round(h*scale)
    full = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
    annotate(full, records, np.zeros(2), np.array([tw/w, th/h]), False)
    x, y = (1408-tw)//2, 64+(1008-th)//2
    canvas[y:y+th, x:x+tw] = full
    text(canvas, f'Frame {index:04d}/{total-1:04d} | {index/fps:.2f}s | {len(records)} detections', (22, 26))
    label = ('ResNet18 | Rewired display | original points unchanged | no PnP' if rewired else
             'ResNet18 rectangular input | RAW points only | no PnP / smoothing')
    text(canvas, label, (22, 52), .55)
    text(canvas, 'YOLO bbox + predicted points 0-7', (1425, 28), .5)
    for slot, record in enumerate(records[:2]):
        top = 64+504*slot
        text(canvas, f'D{slot+1} YOLO {record["yolo_confidence"]:.3f} | display zoom', (1420, top+25), .5)
        # This expanded viewport is display-only; inference uses the tight rectangle.
        box = np.array(record['bbox_xyxy'])
        center = (box[:2]+box[2:])/2
        side = max(box[2:]-box[:2])*1.4
        origin = center-side/2
        zoom_scale = 448/side
        transform = np.array([[zoom_scale, 0, -origin[0]*zoom_scale],
                              [0, zoom_scale, -origin[1]*zoom_scale]])
        zoom = cv2.warpAffine(frame, transform, (448, 448))
        annotate(zoom, [record], origin, zoom_scale, True)
        canvas[top+40:top+488, 1440:1888] = zoom
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cache = json.loads((args.cache/'manifest.json').read_text())
    ffmpeg = shutil.which('ffmpeg') or cache['encoder_command'][0]
    if not Path(ffmpeg).is_file():
        raise FileNotFoundError(f'ffmpeg executable unavailable: {ffmpeg}')
    source = Path(cache['source_video'])
    if sha256(source) != cache['source_sha256'] or sha256(Path(cache['yolo'])) != cache['yolo_sha256']:
        raise ValueError('Source or YOLO checkpoint differs from cached detections')
    entries = [json.loads(line) for line in (args.cache/'predictions.jsonl').read_text().splitlines()]
    assert len(entries) == cache['source_frames']
    assert all(row['frame'] == i for i, row in enumerate(entries))
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
    video = args.output/'camera_points_full_video.mp4'
    fps, total = cache['fps'], len(entries)
    command = [ffmpeg, '-hide_banner', '-loglevel', 'warning', '-n',
               '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s:v', '1920x1080', '-r', str(fps),
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-threads', '4', '-movflags', '+faststart', str(video)]
    write_json(args.output/'manifest.json', dict(source_video=str(source), source_sha256=cache['source_sha256'],
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha256(args.checkpoint),
        checkpoint_epoch=checkpoint['epoch'], model_args=saved, yolo=cache['yolo'],
        yolo_sha256=cache['yolo_sha256'], detection_cache=str(args.cache.resolve()),
        detection_cache_sha256=sha256(args.cache/'predictions.jsonl'), fps=fps, total_frames=total,
        points_only=True, pnp=False, temporal_smoothing=False, real_corner_finetuning=False,
        notes=['Cached YOLO boxes only; all ResNet points recomputed with current rectangular model.',
               'Display zoom includes context; network input is tight rectangular RGB without padding.',
               'Predictions, not ground truth. Video previously used for background and pose design.']))
    capture = cv2.VideoCapture(str(source))
    started = time.perf_counter()
    processed, detections = 0, 0
    snapshots = {0, 125, 500, 1483, 2349, total-1}
    try:
        with (args.output/'encoder.log').open('w') as log, (args.output/'predictions.jsonl').open('w') as output:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            try:
                with torch.inference_mode():
                    for entry in entries:
                        ok, frame = capture.read()
                        if not ok:
                            raise RuntimeError(f'Failed to decode frame {processed}')
                        records = []
                        for old in entry['detections']:
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
                        canvas = compose(frame, records, processed, total, fps)
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
            total_detections=detections, fps=actual_fps, duration_seconds=count/actual_fps,
            bytes=video.stat().st_size, elapsed_seconds=time.perf_counter()-started, video_sha256=sha256(video))
        write_json(args.output/'summary.json', summary)
        write_json(args.output/'status.json', summary)
        (args.output/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>真实视频：仅预测点，不连线</title>
<style>body{{background:#14212a;color:#eee;font:18px/1.7 sans-serif;margin:30px}}video{{width:100%;max-width:1600px}}a{{color:#8df}}</style>
<h1>ResNet18：逐帧八点预测，不连线</h1><p>全部 {count} 帧，{actual_fps} fps，{count/actual_fps:.2f} 秒。
黄色为原始预测点，青色为固定 YOLO 检测框。无 PnP、无时序平滑、无真实八点微调。右侧为显示用放大视图；网络仍使用紧矩形输入。</p>
<video controls preload="metadata" poster="frame_000000.jpg" src="camera_points_full_video.mp4"></video>
<p><a href="camera_points_full_video.mp4" download>下载 MP4</a> · <a href="manifest.json">处理设置</a> · <a href="summary.json">完整性检查</a></p>
<p>这是预测展示，不是真值标注。全部失败预测与漏检帧均保留。D1/D2 每帧按图像左右排序，不是跟踪 ID。</p>''')
        print(json.dumps(summary), flush=True)
    except Exception as error:
        write_json(args.output/'status.json', dict(state='failed', processed_frames=processed, error=str(error)))
        raise
    finally:
        capture.release()


if __name__ == '__main__':
    main()
