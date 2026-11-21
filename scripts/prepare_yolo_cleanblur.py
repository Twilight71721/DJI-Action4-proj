"""Convert paired clean/blur BOP images while preserving their frozen scene splits."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from prepare_yolo_bop import clip_box, normalize_box


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assignments = {}
    for split, scenes in manifest['splits'].items():
        for scene in scenes:
            if scene in assignments:
                raise ValueError(f'Scene assigned twice: {scene}')
            assignments[scene] = split
    jobs = {job['scene']: job for job in manifest['jobs']}
    annotations, records, hashes, pairs = {}, [], {}, set()
    output.mkdir(parents=True)
    for index, frame in enumerate(manifest['frames']):
        scene, frame_id, split = frame['scene'], str(int(frame['frame'])), frame['split']
        if assignments[scene] != split or jobs[scene]['split'] != split:
            raise ValueError(f'Split mismatch: {scene}')
        pair = (scene, frame_id)
        if pair in pairs:
            raise ValueError(f'Duplicate pair: {pair}')
        pairs.add(pair)
        if scene not in annotations:
            clean = source / 'clean' / scene
            gt = json.loads((clean / 'scene_gt.json').read_text())
            info = json.loads((clean / 'scene_gt_info.json').read_text())
            if set(gt) != set(info):
                raise ValueError(f'GT/info IDs differ: {scene}')
            for name, expected in [('scene_gt.json', gt), ('scene_gt_info.json', info)]:
                if json.loads((source / 'blur' / scene / name).read_text()) != expected:
                    raise ValueError(f'Blur geometry changed: {scene}/{name}')
            annotations[scene] = gt, info
        gt, info = annotations[scene]
        with Image.open(source / frame['clean']) as im:
            width, height = im.size
        if len(gt[frame_id]) != len(info[frame_id]):
            raise ValueError(f'Instance count mismatch: {pair}')
        boxes, invisible = [], []
        for instance, (obj, ann) in enumerate(zip(gt[frame_id], info[frame_id])):
            if obj['obj_id'] != 1:
                raise ValueError('Expected Action4 obj_id=1')
            box = clip_box(ann['bbox_obj'], width, height)
            if box is None or ann['px_count_visib'] == 0:
                invisible.append(instance)
                continue
            boxes.append(dict(instance=instance, xyxy=box,
                              xywhn=normalize_box(box, width, height),
                              visib_fract=ann['visib_fract']))
        for variant in ('clean', 'blur'):
            rgb = source / frame[variant]
            with Image.open(rgb) as im:
                im.verify()
                if im.size != (width, height):
                    raise ValueError(f'Paired image sizes differ: {pair}')
            digest = hashlib.sha256(rgb.read_bytes()).hexdigest()
            if digest != frame[variant + '_sha256']:
                raise ValueError(f'Image content changed: {rgb}')
            if digest in hashes:
                raise ValueError(f'Duplicate image: {rgb}, {hashes[digest]}')
            hashes[digest] = str(rgb)
            stem = f'{variant}_{scene}_{int(frame_id):06d}'
            image = output / 'images' / split / (stem + rgb.suffix)
            label = output / 'labels' / split / (stem + '.txt')
            image.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            image.symlink_to(rgb)
            label.write_text(''.join('0 ' + ' '.join(f'{v:.9f}' for v in b['xywhn']) + '\n'
                                     for b in boxes))
            records.append(dict(scene=scene, frame=f'{int(frame_id):06d}', split=split,
                                group=f'render_job_{scene}', family=frame['family'],
                                variant=variant, source=str(rgb), sha256=digest,
                                width=width, height=height, boxes=boxes,
                                invisible_instances=invisible,
                                image=str(image.relative_to(output)),
                                label=str(label.relative_to(output))))
        if (index + 1) % 500 == 0:
            print(f'Checked {index + 1}/{len(manifest["frames"])} clean/blur pairs', flush=True)
    actual = Counter(f['split'] for f in manifest['frames'])
    if dict(actual) != manifest['counts']:
        raise ValueError('Source split counts do not match manifest')
    summary = dict(images=dict(Counter(r['split'] for r in records)),
                   instances={s: sum(len(r['boxes']) for r in records if r['split'] == s)
                              for s in ('train', 'val', 'test')},
                   base_images=len(pairs), unique_rgb=len(hashes),
                   source=str(source), source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   bbox_policy='bbox_obj, image-clipped; paired blur does not change coordinates',
                   split_policy=manifest['protocol'],
                   evaluation_note='合成原图和模糊图始终在同一集合；保持原有渲染批次分组。历史 YOLO 权重可能见过部分合成留出场景，不能将其视为新的盲测。')
    (output / 'manifest.json').write_text(json.dumps(dict(summary=summary, frames=records),
                                                   indent=2, ensure_ascii=False))
    (output / 'data.yaml').write_text(f'path: {json.dumps(str(output))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: action4\n')
    review = output / 'review'
    review.mkdir()
    training = [r for r in records if r['split'] == 'train' and r['variant'] == 'clean']
    chosen = [training[i * (len(training) - 1) // 7] for i in range(8)]
    sheet = Image.new('RGB', (1218, 8 * 330), '#18212c')
    for row, record in enumerate(chosen):
        clean = source / 'clean' / record['scene'] / 'rgb' / (record['frame'] + '.png')
        blur = source / 'blur' / record['scene'] / 'rgb' / (record['frame'] + '.png')
        for col, path in enumerate((clean, blur)):
            with Image.open(path) as im:
                tile = im.convert('RGB')
            draw = ImageDraw.Draw(tile)
            for box in record['boxes']:
                draw.rectangle(box['xyxy'], outline='#00ff70', width=4)
            tile.thumbnail((609, 306))
            sheet.paste(tile, (col * 609, row * 330 + 24))
            ImageDraw.Draw(sheet).text((col * 609 + 6, row * 330 + 5),
                                      f'{path.parts[-4]} | {record["scene"]}/{record["frame"]}', fill='white')
    sheet.save(review / 'pairs.jpg', quality=93)
    (review / 'index.html').write_text('<meta charset="utf-8"><h1>YOLO 清晰/模糊配对标注</h1><p>左清晰，右模糊；绿色为相机框。每对使用相同标注与集合划分。</p><img width="100%" src="pairs.jpg">')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
