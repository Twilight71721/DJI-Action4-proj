"""Reconnect archived per-frame points with a cube graph and render the full video."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from diagnose_point_rewiring import reconnect
from export_rect_points_video import compose
from infer_camera_video import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True, help='Directory of the points-only full video')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source_manifest = json.loads((args.predictions/'manifest.json').read_text())
    predictions = args.predictions/'predictions.jsonl'
    predictions_hash = sha256(predictions)
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    total, fps = source_manifest['total_frames'], source_manifest['fps']
    if len(rows) != total or any(row['frame'] != i for i, row in enumerate(rows)):
        raise ValueError('Prediction frames are incomplete or out of order')
    source = Path(source_manifest['source_video'])
    if sha256(source) != source_manifest['source_sha256']:
        raise ValueError('Source video hash differs')
    old_manifest = json.loads((Path(source_manifest['detection_cache'])/'manifest.json').read_text())
    ffmpeg = shutil.which('ffmpeg') or old_manifest['encoder_command'][0]
    if not Path(ffmpeg).is_file():
        raise FileNotFoundError(ffmpeg)
    args.output.mkdir(parents=True, exist_ok=False)
    video = args.output/'camera_rewired_full_video.mp4'
    command = [ffmpeg, '-hide_banner', '-loglevel', 'warning', '-n', '-f', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s:v', '1920x1080', '-r', str(fps), '-i', 'pipe:0', '-an',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
        '-threads', '4', '-movflags', '+faststart', str(video)]
    write_json(args.output/'manifest.json', dict(source_manifest=source_manifest,
        predictions_sha256=predictions_hash, method='840 cube edge sets ranked by affine cube projection residual',
        point_coordinates_changed=False, pnp=False, semantic_correspondence_preserved=False,
        temporal_smoothing=False, model_retrained=False, topology_is_per_frame=True, encoder_command=command))
    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(source))
    processed = changed = detections = close_instances = 0
    started = time.perf_counter()
    snapshots = {0, 125, 500, 1483, 2349, total-1}
    try:
        with (args.output/'encoder.log').open('w') as log, (args.output/'rewiring.jsonl').open('w') as record_file:
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            try:
                for row in rows:
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f'Video decode failed at frame {processed}')
                    for record in row['detections']:
                        fit = reconnect(record['corners_original'])
                        np.testing.assert_array_equal(fit['points_original'], record['corners_original'])
                        record['display_edges'] = fit['edges']
                        record['rewiring'] = fit
                        detections += 1
                        changed += int(fit['changed_edges'] > 0)
                        close_instances += int(bool(fit['near_coincident_pairs']))
                    canvas = compose(frame, row['detections'], processed, total, fps, rewired=True)
                    encoder.stdin.write(canvas.tobytes())
                    record_file.write(json.dumps(row, allow_nan=False)+'\n')
                    if processed in snapshots:
                        cv2.imwrite(str(args.output/f'frame_{processed:06d}.jpg'), canvas)
                    processed += 1
                    if processed == 1 or processed % 250 == 0 or processed == total:
                        elapsed = time.perf_counter()-started
                        status = dict(state='running', processed_frames=processed, total_frames=total,
                                      eta_seconds=round(elapsed/processed*(total-processed)))
                        write_json(args.output/'status.json', status)
                        print(json.dumps(status), flush=True)
                encoder.stdin.close()
                if encoder.wait() != 0:
                    raise RuntimeError('Encoding failed; inspect encoder.log')
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
        assert sha256(predictions) == predictions_hash
        summary = dict(state='complete', processed_frames=processed, output_frames=count, fps=actual_fps,
            duration_seconds=count/actual_fps, detections=detections, rewired_instances=changed,
            instances_with_near_coincident_points=close_instances, original_predictions_unchanged=True,
            bytes=video.stat().st_size, elapsed_seconds=time.perf_counter()-started, video_sha256=sha256(video))
        write_json(args.output/'summary.json', summary)
        write_json(args.output/'status.json', summary)
        (args.output/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>八点重新连成长方体结构：完整视频</title>
<style>body{{background:#14212a;color:#eef;font:18px/1.7 sans-serif;margin:30px}}video{{width:100%;max-width:1600px}}a{{color:#8df}}</style>
<h1>重新连接八个预测点：完整视频</h1><p>{count} 帧，{actual_fps} fps，{count/actual_fps:.2f} 秒。</p>
<video controls preload="metadata" poster="frame_000000.jpg" src="camera_rewired_full_video.mp4"></video>
<p><a href="camera_rewired_full_video.mp4" download>下载 MP4（{video.stat().st_size/1e6:.1f} MB）</a> · <a href="summary.json">完整性检查</a></p>
<p>黄色为原来的八个预测点，紫色为重新选择的12条边，青色为YOLO框。逐帧遍历840种长方体图连接方式，以仿射投影一致性排序。
没有移动预测点或补点；点号仍为原始网络编号，连接关系不再代表已验证的物理角点对应。未使用PnP、真值修正或时序平滑。</p>
<p>这是线框展示，不是新的位姿精度结果。聚集、错位的点仍可能形成退化框，逐帧选边也可能跳变。全部检测及漏检帧保留。</p>''')
        print(json.dumps(summary), flush=True)
    except Exception as error:
        write_json(args.output/'status.json', dict(state='failed', processed_frames=processed, error=str(error)))
        raise
    finally:
        capture.release()


if __name__ == '__main__':
    main()
