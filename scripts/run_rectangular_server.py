"""Checked server workflow for the frozen-detector rectangular ResNet18 experiment."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audit_rectangular_protocol import audit
from bop_bb8 import BOPModel
from bop_bb8.online_background import OnlinePoseBackgroundDataset
from bop_bb8.visualize import EDGES
from bop_bb8.photometric import PhotometricDataset, augment_rgb
from bop_bb8.torch_data import HeatmapDataset, IMAGENET_MEAN, IMAGENET_STD


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Checksum mismatch: {path}')


def verify_bundle():
    for line in (ROOT/'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        require_hash(ROOT/name, digest)


def run(script, *arguments):
    command = [sys.executable, '-u', str(ROOT/'scripts'/script), *map(str, arguments)]
    print('[run]', ' '.join(command), flush=True)
    env = dict(os.environ, TORCH_HOME=str(ROOT/'weights/torch'))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def training_complete(folder, epochs, train_samples, val_samples, recipe=None):
    config = read(folder/'config.json')
    expected = dict(epochs=epochs, train_samples=train_samples, eval_samples=val_samples,
        input_contract_version=2, crop_mode='rect', crop_padding=1., crop_border=0,
        backbone='resnet18', fpn_dim=128, loss='spatial_kl', decoder='local',
        no_pretrained=False, init_checkpoint=None, device='cuda', amp_dtype='bfloat16')
    if recipe is not None:
        expected.update(photometric_augmentation=recipe.get('photometric_augmentation',False),
            photometric_profile=recipe.get('photometric_profile','pilot'),
            view_balance=recipe.get('view_balance',False),paired_background=False,
            consistency_weight=0.)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Training contract differs: {key}={config.get(key)!r}, expected {value!r}')
    with (folder/'metrics.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    if [int(r['epoch']) for r in rows] != list(range(1, epochs+1)):
        raise ValueError('Training did not complete the requested epoch count')
    if not all(np.isfinite(float(value)) for row in rows for value in row.values()):
        raise ValueError('Non-finite training metrics')
    for name in ['best', 'last']:
        checkpoint = torch.load(folder/f'{name}.pth', map_location='cpu', weights_only=False)
        if name == 'last' and checkpoint['epoch'] != epochs-1:
            raise ValueError('Last checkpoint does not match the final epoch')
        if not all(torch.isfinite(t).all() for t in checkpoint['model'].values()):
            raise ValueError('Non-finite model parameters')
    return rows


def preview_inputs(prepared, geometry, config, output):
    output.mkdir(exist_ok=False)
    model = BOPModel.from_models_info(prepared/'models/models_info.json')
    data = OnlinePoseBackgroundDataset(prepared/'trainval', model,
        scenes=geometry['train_scenes'], online_background_config=prepared/'online_config.json',
        input_size=(256,256), heatmap_size=(64,64), crop_mode='rect', crop_padding=1., crop_border=0,
        rect_sizes=config['rect_sizes'], heatmap_margin=config['heatmap_margin'],
        min_visib_fract=config['train_min_visib_fract'], require_all_corners=True,
        crop_jitter_seed=config['seed'], heatmap_kernel='gaussian', heatmap_sigma=2.)
    indices = set(np.linspace(0, len(data)-1, 32, dtype=int).tolist())
    # Add one example of every actual input aspect bucket.
    seen = set()
    for i in range(len(data)):
        size = data.get_input_size(i)
        if size not in seen:
            indices.add(i)
            seen.add(size)
    cards, records = [], []
    photo = config.get('photometric_augmentation', False)
    tensors = HeatmapDataset(data)
    augmented = PhotometricDataset(tensors,config['seed'],config.get('photometric_profile','pilot'))
    mean = torch.tensor(IMAGENET_MEAN)[:,None,None]
    std = torch.tensor(IMAGENET_STD)[:,None,None]
    for n, i in enumerate(sorted(indices)):
        data.set_epoch(0)
        first = data[i]
        data.set_epoch(1)
        second = data[i]
        for field in ['keypoints_2d', 'K_input', 'heatmaps', 'valid']:
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
        images = [cv2.cvtColor(s.image, cv2.COLOR_RGB2BGR) for s in [first, second]]
        photo_settings = []
        if photo:
            images = [images[0]]
            # Hold RGB/background fixed to isolate photometric changes in this preview.
            data.set_epoch(0)
            source = tensors[i]
            for epoch in [0,1,2]:
                augmented.set_epoch(epoch)
                sample = augmented[i]
                for key in source:
                    if key != 'image':
                        torch.testing.assert_close(torch.as_tensor(source[key]),torch.as_tensor(sample[key]),rtol=0,atol=0)
                rgb = (sample['image']*std+mean).permute(1,2,0).mul(255).round().clamp(0,255).byte().numpy()
                rng = np.random.default_rng(np.random.SeedSequence([config['seed'],epoch,i,930]))
                expected, metadata = augment_rgb(first.image,rng,config.get('photometric_profile','pilot'))
                np.testing.assert_array_equal(rgb,expected)
                photo_settings.append(metadata)
                images.append(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        labeled = images[0].copy()
        points = np.rint(first.keypoints_2d).astype(int)
        for a, b in EDGES:
            cv2.line(labeled, tuple(points[a]), tuple(points[b]), (40,230,40), 1, cv2.LINE_AA)
        name = f'{n:03d}.png'
        cv2.imwrite(str(output/name), np.hstack([*images, labeled]))
        records.append(dict(identity=data.instances[i], input_hw=list(first.image.shape[:2]),
            all_eight_valid=bool(first.valid.all()), background_changed=bool(np.any(first.image != second.image)),
            unchanged_geometry=True,photometric_settings=photo_settings))
        cards.append(f'<h3>{html.escape(str(data.instances[i]))}</h3><img src="{name}">')
    dump(output/'summary.json', dict(samples=records, input_only_rgb=True, rgb_padding=False))
    (output/'index.html').write_text('<meta charset="utf-8"><style>img{max-width:100%}</style>'
        '<h1>服务器实际训练输入</h1><p>'+('每行：原图、三轮成像增强、原图标签。预览固定背景以单独检查成像变化；正式训练背景照常逐轮随机。'
        if photo else '每行：epoch 0 RGB、epoch 1 RGB、标签叠加。')+
        '最后一列仅用于检查，不输入网络；超出 RGB 的角点仍保留在输出热图中。</p>'+''.join(cards))


def check_data(args, contract, output):
    previous = args.previous.resolve()
    prepared = previous/'work/continuous_data'
    require_hash(prepared/'online_config.json', contract['prepared_sha256'])
    require_hash(previous/'pose_assets/manifest.json', contract['foreground_sha256'])
    require_hash(previous/'work/mixed_backgrounds/manifest.json', contract['background_sha256'])
    online = read(prepared/'online_config.json')
    base = Path(online['base_dataset'])
    require_hash(base/'manifest.json', contract['dataset_manifest_sha256'])
    manifest = read(base/'manifest.json')
    for rel, digest in read(ROOT/'configs/background_data_hashes.json').items():
        require_hash(base/rel, digest)
    for frame in manifest['frames']:
        for variant in ['clean', 'blur']:
            require_hash(base/frame[variant], frame[variant+'_sha256'])
    masks_root = Path(online['old_background_assets'])
    masks = read(masks_root/'camera_masks/manifest.json')['records']
    for record in masks.values():
        require_hash(masks_root/'camera_masks'/record['path'], record['sha256'])
    backgrounds = read(previous/'work/mixed_backgrounds/manifest.json')['records']
    for record in backgrounds:
        require_hash(previous/'work/mixed_backgrounds'/record['path'], record['sha256'])
    required = set(online['train_backgrounds'] + online['heldout_backgrounds'])
    required.update(r['rgba'] for r in online['new_records'].values())
    for path in required:
        if not Path(path).is_file():
            raise FileNotFoundError(f'Missing external asset: {path}')
    for path in {r['rgba'] for r in online['new_records'].values()}:
        rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4 or not rgba[:,:,3].any():
            raise ValueError(f'Invalid foreground RGBA: {path}')
    recipe = read(args.config)
    geometry = audit(prepared, recipe)
    dump(output/'geometry_audit.json', geometry)
    for name, count in [('train',40000), ('val',3048)]:
        stats = geometry['stats'][name]
        if stats['instances'] != count:
            raise ValueError(f'{name}: expected {count} full-server instances, found {stats["instances"]}')
        if stats['behind_camera_or_nonfinite_instances'] or stats['all_eight_inside_fraction'] != 1.:
            raise ValueError(f'{name}: geometry not covered; do not weaken the audit or drop validation corners')
    if geometry['stats']['val']['eligible_instances'] != 3048:
        raise ValueError('Validation must retain all 3048 instances')
    for scene in geometry['train_scenes']+geometry['val_scenes']:
        root = prepared/'trainval'/scene
        for frame in read(root/'scene_gt.json'):
            if not any((root/'rgb'/f'{int(frame):06d}{suffix}').is_file() for suffix in ['.jpg','.png','.jpeg']):
                raise FileNotFoundError(f'Missing prepared RGB {scene}/{frame}')
    preview_inputs(prepared, geometry, recipe['training'], output/'input_preview')
    return dict(prepared=str(prepared), base_dataset=str(base), geometry=geometry,
        recipe=recipe, data_contract=contract, background_records=len(backgrounds),
        real_corner_training=False, detector_retrained=False, artificial_occlusion=False)


def create_report(output):
    recipe = read(output/'checked.json')['recipe']['training']
    photo = recipe.get('photometric_augmentation',False)
    reports = {name:read(output/'evaluation'/name/'summary.json')
               for name in ['validation', 'test_clean', 'test_blur']}
    rows = []
    for name, report in reports.items():
        raw, pnp = report['body_long_side_256'], report['pnp']
        rows.append(f'<tr><td>{name}</td><td>{report["instances"]}</td><td>{raw["mean_err_px"]:.3f}</td>'
            f'<td>{raw["pck5"]:.2%}</td><td>{pnp["pck5_all_corners_rejection_counts_as_failure"]:.2%}</td>'
            f'<td>{pnp["accepted_instances"]}/{pnp["total_instances"]}</td></tr>')
    dump(output/'summary.json', dict(experiment=output.name, evaluations=reports,training_recipe=recipe,
        real=read(output/'real/summary.json'), real_pnp=False, real_corner_training=False,
        note='Synthetic oracle body boxes; real frozen-YOLO development diagnostic. Compare original numbered corners, not visual rewiring.'))
    (output/'index.html').write_text('<meta charset="utf-8"><style>body{font:18px system-ui;margin:32px}'
        'td,th{padding:12px;border:1px solid #aaa}table{border-collapse:collapse}</style>'
        '<h1>ResNet18 长方形紧裁剪：服务器训练结果</h1>'+
        ('<p>本轮仅增加 balanced 成像增强，约20%原图、68%轻度、12%中等；只增强训练集。视角加权关闭。</p>' if photo else '')+
        '<p>误差按机身框长边归一到 256 像素。PnP 使用合成内参；拒绝项在总 PCK 中计为失败。'
        '原始八点与 PnP 分开报告。真实帧没有可信内参，保留原始八点。'
        '</p><table><tr><th>集合</th><th>实例</th><th>原始均误差</th><th>原始 PCK5</th>'
        '<th>PnP 全量 PCK5</th><th>PnP 接受</th></tr>'+''.join(rows)+'</table>'
        '<p><a href="real/index.html">真实帧：YOLO 框、实际输入、八角点与热图</a></p>'
        '<p><a href="input_preview/index.html">实际训练输入检查</a></p>'
        '<p>合成集真值/预测：<a href="evaluation/validation/index.html">验证集</a> · '
        '<a href="evaluation/test_clean/index.html">清晰测试</a> · '
        '<a href="evaluation/test_blur/index.html">模糊测试</a></p>'
        '<p><a href="training/metrics.csv">每轮指标</a> · <a href="summary.json">完整数值</a></p>'
        '<p>真实视频已用于背景和姿态设计，此处是诊断，不能当独立真实测试。'
        '请结合相同划分的清晰/模糊测试与完整验证结果判断改善，不只看框的外观。</p>')


def pack_results(output):
    if not (output/'summary.json').is_file():
        raise ValueError('Complete evaluation and report before packaging')
    archive = output.with_suffix('.tar.gz')
    if archive.exists():
        raise FileExistsError(archive)
    digests = {}
    with tarfile.open(archive, 'w:gz', compresslevel=3) as tar:
        for path in sorted(output.rglob('*')):
            rel = path.relative_to(output)
            if not path.is_file() or 'smoke_training' in rel.parts:
                continue
            if path.is_symlink():
                raise ValueError(f'Result must not contain external symlinks: {path}')
            if path.suffix == '.pth':
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                checkpoint.pop('optimizer', None)
                checkpoint['original_checkpoint_sha256'] = sha(path)
                checkpoint['export_note'] = 'Optimizer omitted; full original retained on server.'
                buffer = io.BytesIO()
                torch.save(checkpoint, buffer)
                payload = buffer.getvalue()
            else:
                payload = path.read_bytes()
            digests[str(rel)] = hashlib.sha256(payload).hexdigest()
            entry = tarfile.TarInfo(str(Path(output.name)/rel))
            entry.size = len(payload)
            entry.mode = 0o644
            tar.addfile(entry, io.BytesIO(payload))
        payload = ''.join(f'{digest}  {name}\n' for name,digest in digests.items()).encode()
        entry = tarfile.TarInfo(f'{output.name}/SHA256SUMS')
        entry.size = len(payload)
        tar.addfile(entry, io.BytesIO(payload))
    with tarfile.open(archive) as tar:
        for name,digest in digests.items():
            if hashlib.sha256(tar.extractfile(f'{output.name}/{name}').read()).hexdigest() != digest:
                raise ValueError(f'Result archive verification failed: {name}')
    archive.with_name(archive.name+'.sha256').write_text(f'{sha(archive)}  {archive.name}\n')
    print(f'[return] {archive} ({archive.stat().st_size/1024**2:.1f} MiB)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path,
        default=ROOT.parent/'resnet_continuous_pose_background_20260924')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/rect_r18_fixed_yolo_30ep')
    parser.add_argument('--stage', choices=['check','smoke','train','evaluate','pack','all'], default='all')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/single_rgb_rect_20260926.json')
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        raise ValueError('Invalid batch size or workers')
    verify_bundle()
    args.output = args.output.resolve()
    args.config = args.config.resolve()
    recipe = read(args.config)['training']
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    contract = read(ROOT/'configs/rectangular_server_data_20260926.json')
    signature = dict(bundle_sha256=sha(ROOT/'SHA256SUMS'), previous=str(args.previous.resolve()),
                     batch_size=args.batch_size, workers=args.workers,recipe_sha256=sha(args.config))
    stages = ['check','smoke','train','evaluate','pack'] if args.stage == 'all' else [args.stage]
    if args.stage != 'pack':
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('CUDA with BF16 is required; CPU fallback is forbidden')
        import torchvision
        if torch.__version__.split('+')[0] != '2.8.0' or torchvision.__version__.split('+')[0] != '0.23.0':
            raise RuntimeError('Use the existing verified torch 2.8.0 / torchvision 0.23.0 environment')
        print(f'[GPU] {torch.cuda.get_device_name(0)}', flush=True)
    for stage in stages:
        print(f'[stage] {stage}', flush=True)
        if stage == 'check':
            if (output/'checked.json').exists():
                raise FileExistsError('Check already exists; use individual stages to continue, or a new output for a new run')
            if shutil.disk_usage(output).free < 8*1024**3:
                raise RuntimeError('Need at least 8 GiB free for checkpoints and result export')
            report = check_data(args, contract, output)
            dump(output/'checked.json', dict(signature=signature, **report,
                gpu=torch.cuda.get_device_name(0), torch=torch.__version__))
            continue
        checked = read(output/'checked.json')
        if checked['signature'] != signature:
            raise ValueError('Check signature differs: recheck in a new output directory')
        prepared = Path(checked['prepared'])
        require_hash(prepared/'online_config.json', contract['prepared_sha256'])
        if stage == 'smoke':
            run('train_single_rgb.py', '--prepared', prepared, '--output', output, '--action','smoke',
                '--batch-size',args.batch_size,'--workers',args.workers,'--smoke-samples',64,'--config',args.config)
            training_complete(output/'smoke_training',1,64,64,recipe)
            dump(output/'smoke_complete.json',signature)
        elif stage == 'train':
            if read(output/'smoke_complete.json') != signature:
                raise ValueError('Matching smoke test required')
            run('train_single_rgb.py','--prepared',prepared,'--output',output,'--action','train',
                '--batch-size',args.batch_size,'--workers',args.workers,'--config',args.config)
            training_complete(output/'training',recipe['epochs'],checked['geometry']['stats']['train']['eligible_instances'],3048,recipe)
            dump(output/'training_complete.json',signature)
        elif stage in ['evaluate','pack']:
            if read(output/'training_complete.json') != signature:
                raise ValueError('Formal training completion marker missing or mismatched')
            training_complete(output/'training',recipe['epochs'],checked['geometry']['stats']['train']['eligible_instances'],3048,recipe)
            if stage == 'pack':
                pack_results(output)
                continue
            checkpoint = output/'training/best.pth'
            base = Path(checked['base_dataset'])
            test = read(base/'manifest.json')['splits']['test']
            for name, data, scenes, role in [
                ('validation',prepared/'trainval',checked['geometry']['val_scenes'],'validation'),
                ('test_clean',base/'clean',test,'test'),('test_blur',base/'blur',test,'test')]:
                dest = output/'evaluation'/name
                if dest.exists():
                    raise FileExistsError(f'{dest}: preserve partial results under another name before retrying')
                run('evaluate_single_rgb.py','--checkpoint',checkpoint,'--dataset-root',data,
                    '--models-info',prepared/'models/models_info.json','--output',dest,
                    '--scenes',*scenes,'--role',role,'--workers',args.workers,'--pnp')
                summary = read(dest/'summary.json')
                expected = 3048 if role == 'validation' else 1000
                if summary['instances'] != expected or summary['limited_diagnostic'] or not summary['all_finite']:
                    raise ValueError(f'Incomplete {name} evaluation')
            run('predict_tight_cuboid_real.py','--real-manifest',ROOT/'real/manifest.json',
                '--yolo',ROOT/'weights/yolo11s_frozen.pt','--resnet',checkpoint,'--output',output/'real')
            shutil.copytree(ROOT/'real',output/'real/source_frames')
            shutil.copyfile(prepared/'models/models_info.json',output/'models_info.json')
            shutil.copyfile(ROOT/'BUNDLE_MANIFEST.json',output/'BUNDLE_MANIFEST.json')
            create_report(output)


if __name__ == '__main__':
    main()
