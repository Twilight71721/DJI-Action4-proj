"""Index tight crops without duplicating RGB data; export matching detector targets."""
import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8 import BOPInstanceDataset, BOPModel
from bop_bb8.cropping import corner_bounds, raster_bounds, tight_crop_window
from bop_bb8.geometry import parse_intrinsics, parse_rotation, project_corners, crop_and_scale_intrinsics
from infer_resnet_yolo_real import prepare_crop, draw_points


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))


def title(image, label):
    result = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    cv2.putText(result, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, .41, (245, 245, 245), 1, cv2.LINE_AA)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--real-cases', type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    base = args.dataset.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    review = out/'review'
    review.mkdir()
    manifest = json.loads((base/'manifest.json').read_text())
    model = BOPModel.from_models_info(base/'models/models_info.json')
    dataset = BOPInstanceDataset(base/'clean', model, input_size=(256, 256), heatmap_size=(64, 64),
                                 scenes=sorted({r['scene'] for r in manifest['frames']}),
                                 crop_mode='cuboid', crop_padding=1., crop_border=4,
                                 heatmap_kernel='gaussian')
    partitions = [set(manifest['splits'][split]) for split in ('train', 'val', 'test')]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Source scene split overlap')
    source_sizes, seen, cards = {}, set(), []
    ratios, counts, expanded, source_outside, max_projection_error = [], Counter(), 0, 0, 0.
    frame_annotations = []
    for split in ('train', 'val', 'test'):
        (out/f'yolo/images/{split}').mkdir(parents=True)
        (out/f'yolo/labels/{split}').mkdir(parents=True)
    with (out/'crop_index.jsonl').open('w') as index:
        for frame_number, frame in enumerate(manifest['frames']):
            scene, image_id, split = frame['scene'], int(frame['frame']), frame['split']
            if scene not in manifest['splits'][split]:
                raise ValueError('Source split and frame metadata differ')
            gt, cameras, info = dataset._load_scene(scene)
            camera = cameras[str(image_id)]
            if scene not in source_sizes:
                if 'width' in camera and 'height' in camera:
                    source_sizes[scene] = (int(camera['width']), int(camera['height']))
                else:
                    probe = cv2.imread(str(base/frame['clean']))
                    if probe is None:
                        raise FileNotFoundError(frame['clean'])
                    source_sizes[scene] = (probe.shape[1], probe.shape[0])
            size = source_sizes[scene]
            K = parse_intrinsics(camera['cam_K'])
            labels, objects = [], []
            for instance, entry in enumerate(gt[str(image_id)]):
                R, t = parse_rotation(entry['cam_R_m2c']), np.array(entry['cam_t_m2c'])
                points, source_valid, _ = project_corners(model.corners_mm(), K, R, t, size)
                box = corner_bounds(points)
                crop = tight_crop_window(box)
                mapped, valid, _ = project_corners(model.corners_mm(), crop_and_scale_intrinsics(K, crop, (256, 256)),
                                                  R, t, (256, 256))
                if not valid.all():
                    raise ValueError(f'Tight crop lost corners: {scene}/{image_id}/{instance}')
                independent, _ = cv2.projectPoints(model.corners_mm(), cv2.Rodrigues(R)[0], t.astype(float), K, None)
                expected = (independent.reshape(8, 2)-np.array(crop[:2])) * 256/np.array(crop[2:])
                max_projection_error = max(max_projection_error, float(np.max(np.abs(expected-mapped))))
                old_crop = BOPInstanceDataset._bbox_crop(info[str(image_id)][instance]['bbox_obj'], size, 1.3)
                x1, y1, x2, y2 = raster_bounds(box)
                area = max(0, min(size[0], x2)-max(0, x1))*max(0, min(size[1], y2)-max(0, y1))
                ratio = area / (old_crop[2]*old_crop[3])
                ratios.append(ratio)
                expanded += int(ratio > 1)
                source_outside += int(not source_valid.all())
                record = dict(scene=scene, frame=image_id, instance=instance, split=split, family=frame['family'],
                              clean=frame['clean'], blur=frame['blur'], source_size=list(size),
                              content_bbox_xyxy=list(box), crop_xywh=list(crop),
                              keypoints_original=points.tolist(), keypoints_input=mapped.tolist(),
                              K_input=crop_and_scale_intrinsics(K, crop, (256, 256)).tolist(),
                              all_eight_valid=True, source_valid=source_valid.tolist(),
                              retained_source_area_ratio_vs_old=ratio)
                index.write(json.dumps(record, allow_nan=False)+'\n')
                counts[split] += 1
                clipped = np.clip(np.array(box).reshape(2, 2), [0, 0], size)
                center, wh = clipped.mean(0)/size, (clipped[1]-clipped[0])/size
                if np.any(wh <= 0):
                    raise ValueError('Projected detector target is outside the source image')
                labels.append('0 ' + ' '.join(f'{v:.9f}' for v in [*center, *wh]))
                objects.append(dict(instance=instance, cuboid_bbox_xyxy=list(box),
                                    detector_bbox_clipped=clipped.reshape(-1).tolist(),
                                    all_corners_inside_source=bool(source_valid.all())))
                if split == 'train' and (frame['family'], instance) not in seen:
                    seen.add((frame['family'], instance))
                    original = cv2.imread(str(base/frame['clean']))
                    old, _, _ = prepare_crop(original,
                        np.r_[info[str(image_id)][instance]['bbox_obj'][:2],
                              np.array(info[str(image_id)][instance]['bbox_obj'][:2])+
                              info[str(image_id)][instance]['bbox_obj'][2:]], (256, 256), 1.3)
                    fresh, _, actual = prepare_crop(original, box, (256, 256), 1., 'cuboid')
                    if tuple(actual) != tuple(crop):
                        raise ValueError('Preview and training crop differ')
                    loaded = dataset.load_instance(scene, image_id, instance)
                    np.testing.assert_array_equal(cv2.cvtColor(fresh, cv2.COLOR_BGR2RGB), loaded.image)
                    if not (loaded.heatmaps.max(axis=(1, 2)) > .999).all():
                        raise ValueError('Empty target channel')
                    panes = [title(old, 'OLD: body bbox x1.3'), title(fresh, 'NEW: tight 8-corner bounds'),
                             title(draw_points(fresh, mapped), 'NEW: all 8 target corners')]
                    filename = f'synthetic_{scene}_{image_id:06d}_{instance}.jpg'
                    cv2.imwrite(str(review/filename), np.hstack(panes), [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cards.append(f'<article><h3>合成训练样本 · 姿态族 {frame["family"]} · {scene}/{image_id}/{instance}</h3>'
                                 f'<p>保留原图采样面积为旧裁剪的 {ratio:.1%}；其他位置为纯灰色补边。</p>'
                                 f'<img loading="lazy" src="{filename}"></article>')
            for variant in ('clean', 'blur'):
                source = base/frame[variant]
                if not source.is_file():
                    raise FileNotFoundError(source)
                name = f'{scene}_{image_id:06d}_{variant}'
                (out/f'yolo/images/{split}'/(name+source.suffix)).symlink_to(source)
                (out/f'yolo/labels/{split}'/(name+'.txt')).write_text('\n'.join(labels)+'\n')
            frame_annotations.append(dict(scene=scene, frame=image_id, split=split, objects=objects))
            if (frame_number+1) % 1000 == 0:
                print(f'Indexed {frame_number+1}/{len(manifest["frames"])} source frames', flush=True)
    real_rows, real_cards = [], []
    if args.real_cases:
        cases = json.loads(args.real_cases.read_text())['records']
        for r in cases:
            original = cv2.imread(r['source'])
            if original is None:
                raise FileNotFoundError(r['source'])
            corners = np.array(r['keypoints_original'])
            bound = corner_bounds(corners)
            ref, _, crop = prepare_crop(original, bound, (256, 256), 1., 'cuboid')
            target = (corners-np.array(crop[:2]))*256/np.array(crop[2:])
            old, _, _ = prepare_crop(original, r['box_xyxy'], (256, 256), 1.3)
            row = dict(id=r['id'], split=r['split'], box_xyxy=bound, crop_xywh=crop,
                       keypoints_input=target.tolist(), point_basis=r['point_basis'],
                       source=r['source'], annotation_quality='Approximate manual eight corners; not calibrated GT')
            panes = [title(old, 'OLD reference bbox x1.3'), title(draw_points(ref, target), 'TIGHT manual 8-corner bounds')]
            if r['yolo_box'] is not None:
                predicted_box = np.array(r['yolo_box'])
                detector_roi, _, detector_crop = prepare_crop(original, predicted_box, (256, 256), 1., 'cuboid')
                inside = ((corners >= predicted_box[:2]) & (corners <= predicted_box[2:])).all(1)
                row['corners_inside_old_yolo_box'] = int(inside.sum())
                panes.append(title(detector_roi, f'OLD YOLO tight: contains {inside.sum()}/8'))
            filename = r['id']+'.jpg'
            cv2.imwrite(str(review/filename), np.hstack(panes), [cv2.IMWRITE_JPEG_QUALITY, 95])
            real_cards.append(f'<article><h3>{r["id"]} · {r["split"]}</h3><p>中间为人工八点范围，仅供审校；右侧是旧 YOLO 框直接裁紧，会截掉部分参考角点。没有把参考真值用于实际预测。</p><img loading="lazy" src="{filename}"></article>')
            real_rows.append(row)
        dump(out/'real_cuboid_reference.json', dict(records=real_rows, used_for_training=False,
             note='Partial instance annotations per frame, not complete YOLO labels. Do not mark other cameras as background.'))
    dump(out/'yolo/frame_targets.json', dict(records=frame_annotations,
         note='Synthetic projected-cuboid boxes. Same source splits; no real corner adaptation included.'))
    (out/'yolo/data.yaml').write_text(f'path: {out/"yolo"}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: action4_cuboid\n')
    summary = dict(crop_mode='cuboid', crop_padding=1., crop_border=4, neutral_rgb=[114]*3,
                   source_dataset=str(base), source_manifest_sha256=hashlib.sha256((base/'manifest.json').read_bytes()).hexdigest(),
                   source_frames=len(manifest['frames']), base_instances=len(ratios),
                   variant_images=len(manifest['frames'])*2, variant_instances=len(ratios)*2,
                   base_instances_by_split=dict(counts), source_splits_preserved=True,
                   all_eight_corners_retained=True, max_projection_error_px=max_projection_error,
                   instances_with_corners_outside_source=source_outside,
                   median_retained_source_area_ratio=float(np.median(ratios)),
                   mean_retained_source_area_ratio=float(np.mean(ratios)),
                   instances_needing_more_source_area_than_old=expanded,
                   new_training_started=False,
                   note='Area measures original RGB sampling, not foreground/background segmentation. RGB is cropped at training time, not duplicated. Real cases are review-only. New detector targets need training before claiming matching inference boxes.')
    dump(out/'manifest.json', dict(summary=summary, splits=manifest['splits']))
    dump(out/'yolo/manifest.json', dict(summary=summary, splits=manifest['splits'],
         label_definition='Axis-aligned bounds of all eight projected cuboid corners, clipped to the source image',
         class_name='action4_cuboid', real_corner_supervision=False))
    dump(out/'summary.json', summary)
    intro = ('<p>训练输入改为三维八角点投影的最小外接矩形。保持长宽比，矩形外及四周 4 个输入像素边缘填 RGB(114,114,114)，不带入原图背景。矩形内部仍可能包含背景。</p>'
             '<p>清晰/模糊配对、原有分组、随机背景与连续姿态保留；仅改变输入裁剪与标签变换。这里展示输入和标签，尚未进行新一轮正式训练，不能据此判断精度提升。</p>'
             '<p>预测端采用同一变换，但框必须由检测器估计。现有 YOLO 的机身框与八角框定义不同，已导出同源合成图的八角外接框检测标签，未使用真实参考角点充当预测。</p>')
    (review/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ResNet 紧八角框输入</title>'
        '<style>body{max-width:1200px;margin:30px auto;padding:0 20px;background:#14212a;color:#eef3f6;font:17px/1.7 sans-serif}img{max-width:100%}article{padding:20px 0;border-top:1px solid #506776}a{color:#87d8ff}</style>'
        '<h1>ResNet：紧贴八角框的输入调整</h1>'+intro+
        f'<p>已索引 {len(ratios)} 个基础实例、{len(ratios)*2} 个清晰/模糊实例；原图采样面积的中位数为旧裁剪的 {np.median(ratios):.1%}。{expanded} 个实例所需区域反而增加，不能为减少面积截掉角点。</p>'
        '<p><a href="../summary.json">全量审计</a> · <a href="../crop_index.jsonl">裁剪与八点索引</a> · <a href="../yolo/data.yaml">对应检测数据配置</a></p>'
        '<h2>合成训练输入与监督</h2>'+''.join(cards)+'<h2>真实帧裁剪定义核对</h2>'+''.join(real_cards)+'</html>')
    (out/'README.md').write_text('训练时动态裁剪原有图像，避免重新传输图像。查看 review/index.html、summary.json 和 crop_index.jsonl。\n\n'
        '训练配置：--crop-mode cuboid --crop-padding 1.0 --crop-border 4。旧划分与清晰/模糊配对不变。\n\n'
        'yolo/ 为同一定义的合成八角外接框标签，images 为本机原图符号链接；搬到服务器后用准备脚本重新生成链接。'
        '真实参考只用于审校，没有把不完整的逐实例标注当作完整 YOLO 标注。\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
