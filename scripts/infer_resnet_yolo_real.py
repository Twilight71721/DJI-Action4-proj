"""Apply a BB8 checkpoint to saved YOLO detections without changing either model."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8 import BOPInstanceDataset, BOPModel
from bop_bb8.net import ResNetFPN, decode_heatmaps, heatmap_to_input_points, heatmap_probabilities
from bop_bb8.torch_data import IMAGENET_MEAN, IMAGENET_STD
from bop_bb8.visualize import EDGES
from bop_bb8.cropping import (tight_crop_window, extract_window, rectangular_crop_window,
                              choose_rectangular_size)
from bop_bb8.pnp import fit_perspective_cuboid, validate_camera


def prepare_crop(image, box, input_size, padding, crop_mode='bbox', crop_border=4, rect_sizes=None):
    x1, y1, x2, y2 = box
    if not np.isfinite(box).all() or x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid detection: {box}")
    if crop_mode == 'cuboid':
        if padding != 1.0:
            raise ValueError('Tight inference requires padding=1.0')
        crop = tight_crop_window(box, input_size, crop_border)
    elif crop_mode == 'bbox':
        crop = BOPInstanceDataset._bbox_crop(
            (x1, y1, x2 - x1, y2 - y1), (image.shape[1], image.shape[0]), padding)
    elif crop_mode == 'rect':
        if padding != 1.0 or crop_border != 0:
            raise ValueError('Rectangular inference requires padding=1.0 and border=0')
        crop = rectangular_crop_window(box, (image.shape[1], image.shape[0]))
        if rect_sizes is not None:
            input_size = choose_rectangular_size(crop, rect_sizes)
    else:
        raise ValueError(f'Unsupported inference crop mode: {crop_mode}')
    x, y, w, h = crop
    roi = extract_window(image, crop, box) if crop_mode == 'cuboid' else image[y:y+h, x:x+w]
    interpolation = cv2.INTER_AREA if input_size[0] * input_size[1] < w * h else cv2.INTER_LINEAR
    roi = cv2.resize(roi, input_size, interpolation=interpolation)
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255
    tensor = (tensor - torch.tensor(IMAGENET_MEAN)[:, None, None]) / torch.tensor(IMAGENET_STD)[:, None, None]
    return roi, tensor, crop


def to_original(points, crop, input_size):
    x, y, w, h = crop
    return points * np.array([w / input_size[0], h / input_size[1]]) + np.array([x, y])


def draw_points(image, points, numbered=True, thickness=1):
    result = image.copy()
    points = np.rint(points).astype(int)
    for a, b in EDGES:
        cv2.line(result, tuple(points[a]), tuple(points[b]), (220, 40, 220), thickness, cv2.LINE_AA)
    for i, point in enumerate(points):
        cv2.circle(result, tuple(point), 3, (0, 240, 255), -1, cv2.LINE_AA)
        if numbered:
            cv2.putText(result, str(i), tuple(point + [4, -4]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 240, 255), 1, cv2.LINE_AA)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pnp-camera', type=Path, help='Explicit original-frame K, distortion and image_size JSON')
    parser.add_argument('--models-info', type=Path, help='BOP model metadata, required for optional PnP')
    parser.add_argument('--pnp-threshold', type=float, default=5., help='RANSAC tolerance at body long side 256 px')
    parser.add_argument("--box-definition", choices=["body", "cuboid"], default="body",
                        help="Meaning of saved detector targets, not a way to relabel old body boxes")
    parser.add_argument("--allow-body-box-approximation", action="store_true",
                        help="Keep the existing body detector; record the GT-cuboid/body-box input mismatch")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved = checkpoint["args"]
    camera_config, pnp_corners = None, None
    if args.pnp_camera is not None:
        if args.models_info is None:
            raise ValueError('--pnp-camera requires --models-info for ordered model corners')
        camera_config = json.loads(args.pnp_camera.read_text())
        K_pnp, distortion_pnp = validate_camera(camera_config['K'], camera_config['dist_coeffs'])
        image_size_pnp = camera_config['image_size']
        if (len(image_size_pnp) != 2 or any(not isinstance(v, int) or v <= 0 for v in image_size_pnp)
                or not camera_config.get('source')):
            raise ValueError('Camera JSON requires positive integer image_size [W,H] and an explicit source')
        pnp_corners = BOPModel.from_models_info(args.models_info, saved.get('obj_id', 1),
                                                saved.get('model_scale', 1.)).corners_mm()
        cv2.setRNGSeed(20260926)
    size = tuple(saved["input_size"])
    padding = float(saved["crop_padding"])
    if saved["crop_mode"] not in {"bbox", "cuboid", "rect"}:
        raise ValueError("Expected object-crop checkpoint")
    if saved['crop_mode'] == 'rect' and args.box_definition != 'body':
        raise ValueError('Rectangular checkpoints require body detections, not GT cuboid bounds')
    approximate_body_box = saved['crop_mode'] == 'cuboid' and args.box_definition == 'body'
    if approximate_body_box and not args.allow_body_box_approximation:
        raise ValueError('Pass --allow-body-box-approximation to explicitly evaluate the fixed body detector with tight cuboid training')
    margin = saved.get('heatmap_margin', 0)
    model = ResNetFPN(backbone=saved["backbone"], fpn_dim=saved["fpn_dim"], pretrained=False,
                      heatmap_margin=margin).to(args.device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    entries = json.loads(args.detections.read_text())["predictions"]
    output = args.output.resolve()
    output.mkdir(parents=True)
    (output / "crops").mkdir()
    (output / "frames").mkdir()
    (output / 'inputs').mkdir()
    if camera_config is not None:
        (output/'frames_pnp').mkdir()
    records, panels, frames, heatmaps, cards = [], [], [], [], []
    skipped, heatmap_names = [], []
    with torch.inference_mode():
        for entry in entries:
            path = Path(entry["source"])
            if not path.is_absolute():
                path = ROOT / path
            if not entry["xyxy"]:
                skipped.append(str(path))
                continue
            image = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(path)
            if camera_config is not None and list(image.shape[1::-1]) != image_size_pnp:
                raise ValueError('Camera calibration resolution differs from source frame; transform K explicitly')
            frame = image.copy()
            pnp_frame = image.copy() if camera_config is not None else None
            if len(entry["xyxy"]) != len(entry["confidence"]):
                raise ValueError("Detection/confidence count mismatch")
            for i, (box, confidence) in enumerate(zip(entry["xyxy"], entry["confidence"])):
                roi, tensor, crop = prepare_crop(image, box, size, padding,
                                                saved['crop_mode'], saved.get('crop_border', 4), saved.get('rect_sizes'))
                actual_size = (roi.shape[1], roi.shape[0])
                logits = model(tensor[None].to(args.device)).float()
                probs = heatmap_probabilities(logits, saved.get('loss', 'composite'))
                points = heatmap_to_input_points(
                    decode_heatmaps(logits, saved.get('decoder', 'topk'), saved.get('decode_radius', 3)),
                    tuple(probs.shape[-2:]), actual_size[::-1], margin=margin)[0].cpu().numpy()
                if not np.isfinite(points).all():
                    raise ValueError("Non-finite corner predictions")
                original = to_original(points, crop, actual_size)
                restored = (original - np.array(crop[:2])) / np.array([crop[2]/actual_size[0], crop[3]/actual_size[1]])
                assert np.allclose(restored, points, atol=1e-5)
                hm = probs[0].cpu().numpy()
                heatmaps.append(hm)
                stem = f"{path.stem}_det{i}"
                heatmap_names.append(stem)
                cv2.imwrite(str(output / f'inputs/{stem}.png'), roi)
                annotated = draw_points(roi, points)
                detector_roi = ((np.array(box).reshape(2, 2) - np.array(crop[:2])) *
                                np.array([actual_size[0]/crop[2], actual_size[1]/crop[3]]))
                cv2.rectangle(annotated, tuple(np.rint(detector_roi[0]).astype(int)),
                              tuple(np.rint(detector_roi[1]).astype(int)), (255, 180, 0), 1)
                interior = hm[:, margin:-margin, margin:-margin] if margin else hm
                activation = cv2.resize(interior.max(axis=0), actual_size)
                if saved.get('loss') == 'spatial_kl':
                    activation = activation / max(float(activation.max()), 1e-12)
                color = cv2.applyColorMap(np.uint8(np.clip(activation, 0, 1)*255), cv2.COLORMAP_TURBO)
                panel = np.hstack([roi, annotated, cv2.addWeighted(roi, .6, color, .4, 0)])
                panel = cv2.copyMakeBorder(panel, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
                cv2.putText(panel, f"{stem} YOLO={confidence:.3f} | raw / corners / max heatmap",
                            (5, 23), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(str(output / f"crops/{stem}.jpg"), panel)
                panels.append(cv2.resize(panel, (768, round(panel.shape[0]*768/panel.shape[1]))))
                frame = draw_points(frame, original, thickness=3)
                cv2.rectangle(frame, tuple(np.rint(box[:2]).astype(int)), tuple(np.rint(box[2:]).astype(int)),
                              (255, 180, 0), 3)
                records.append(dict(id=stem, source=str(path), detection_index=i, yolo_box_xyxy=box,
                                    yolo_confidence=confidence, crop_xywh=crop,
                                    input_size=list(actual_size), heatmap_margin=margin,
                                    corners_outside_rgb=((points < 0).any(1) | (points >= actual_size).any(1)).tolist(),
                                    corners_roi=points.tolist(), corners_original=original.tolist(),
                                    heatmap_peak=hm.max(axis=(1, 2)).tolist()))
                pnp_link = ''
                if camera_config is not None:
                    fit = fit_perspective_cuboid(pnp_corners, original, K_pnp, distortion=distortion_pnp,
                        reprojection_threshold=args.pnp_threshold * max(crop[2:])/256)
                    records[-1]['pnp'] = fit
                    if fit['accepted']:
                        pnp_frame = draw_points(pnp_frame, np.asarray(fit['points']), thickness=3)
                    status_text = f'PnP D{i}: {fit["status"]}'
                    cv2.putText(pnp_frame, status_text, (20, 40+40*i), cv2.FONT_HERSHEY_SIMPLEX,
                                .8, (0, 240, 255), 2, cv2.LINE_AA)
                    pnp_link = f' · <a href="frames_pnp/{path.stem}.jpg">PnP 重投影（独立显示）</a>'
                cards.append(f'<article><h3>{html.escape(stem)} · YOLO {confidence:.3f}</h3>'
                             f'<a href="crops/{stem}.jpg"><img src="crops/{stem}.jpg"></a>'
                             f'<a href="inputs/{stem}.png">实际网络 RGB 输入</a> · '
                             f'<a href="frames/{path.stem}.jpg">原图定位</a>{pnp_link}</article>')
            cv2.imwrite(str(output / f"frames/{path.stem}.jpg"), frame)
            if pnp_frame is not None:
                cv2.imwrite(str(output/f'frames_pnp/{path.stem}.jpg'), pnp_frame)
            thumb = cv2.resize(frame, (650, round(frame.shape[0] * 650/frame.shape[1])))
            frames.append(thumb)
    if not panels:
        raise ValueError("No detections to process")
    # All detections are included, including false positives; no manual filtering.
    cv2.imwrite(str(output / "all_crops.jpg"), np.vstack(panels))
    if len({hm.shape for hm in heatmaps}) == 1:
        np.savez_compressed(output / "heatmaps.npz", probabilities=np.stack(heatmaps))
        heatmap_layout = 'probabilities[N,8,H,W]'
    else:
        np.savez_compressed(output / 'heatmaps.npz', **dict(zip(heatmap_names, heatmaps)))
        heatmap_layout = 'one [8,H,W] array per detection id; variable aspect buckets'
    summary = dict(input_frames=len(entries), detected_frames=len(frames), detections=len(records),
                   skipped_no_detection=len(skipped), input_size=size, crop_padding=padding,
                   crop_mode=saved['crop_mode'], crop_border=saved.get('crop_border', 4),
                   detector_box_definition=args.box_definition,
                   body_box_approximation=approximate_body_box,
                   checkpoint=str(args.checkpoint.resolve()),
                   checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   detections_sha256=hashlib.sha256(args.detections.read_bytes()).hexdigest(),
                   decoder=saved.get('decoder', 'topk'), decode_radius=saved.get('decode_radius', 3),
                   heatmap_margin=margin, rect_sizes=saved.get('rect_sizes'), heatmap_layout=heatmap_layout,
                   heatmap_values='spatial softmax' if saved.get('loss') == 'spatial_kl' else 'sigmoid',
                   no_ground_truth=True, pnp_performed=camera_config is not None,
                   pnp_camera=camera_config,
                   pnp_threshold_body_long_side_256=args.pnp_threshold if camera_config is not None else None,
                   pnp_accepted=sum(r.get('pnp', {}).get('accepted', False) for r in records),
                   warning="Raw wireframes are not pose estimates. Optional PnP poses are unverified; heatmap peaks are not calibrated accuracy.")
    (output / "predictions.json").write_text(json.dumps(dict(summary=summary, instances=records,
                                                            skipped_frames=skipped), indent=2))
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>YOLO → 八角点网络 真实帧</title>'
        '<style>body{font:17px sans-serif;background:#14212a;color:#eee;max-width:1100px;margin:35px auto}'
        'img{max-width:100%}a{color:#8dd2ff}article{padding:15px;background:#263641;margin:18px 0}p{line-height:1.7}</style>'
        f'<h1>YOLO 检测框 → {saved["backbone"]}：真实帧诊断</h1>'
        f'<p>{len(entries)} 张输入帧中，{len(frames)} 张有检测框，共 {len(records)} 个候选；'
        f'{len(skipped)} 张无检测结果。所有候选均保留，包括误检。</p>'
        '<p>每组从左到右：原始 ROI、八角点预测、8 通道最大热图叠加。青色为 YOLO 框，紫色为预测连线，黄色为角点编号。'
        f'裁剪模式 {saved["crop_mode"]}，padding={padding}；ImageNet 归一化；解码 {saved.get("decoder", "topk")}。'
        f'实际长方形输入存放在 inputs/；输出热图外延 {margin} 格不属于 RGB 输入。'
        f'检测框定义：{args.box_definition}，未使用真值补框。框外角点请查看原图投影。</p>'
        '<p><strong>无真实角点标注，不能计算像素误差或 PCK。'
        + ('PnP 使用显式提供的内参，结果独立显示；通过几何一致性检查不等于姿态正确。'
           if camera_config is not None else '未提供拍摄相机内参，本次未运行 PnP。') +
        '主页面的原始框只是八点连线，不保证构成有效三维投影。误检区域也会输出八点；热图亮不代表预测正确。</strong></p>'
        '<p><a href="predictions.json">原图坐标与元数据</a> · <a href="heatmaps.npz">原始热图</a></p>'
        + ''.join(cards))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
