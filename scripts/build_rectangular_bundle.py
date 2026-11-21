"""Stage a compact rectangular training update that reuses verified server assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
NAME = 'resnet18_rect_fixed_yolo_20260926'
SCRIPTS = ['train_heatmap','train_single_rgb','audit_rectangular_protocol','evaluate_single_rgb',
    'evaluate_heatmap_checkpoint','infer_resnet_yolo_real','predict_tight_cuboid_real',
    'run_rectangular_server','prepare_camera_masks']
TESTS = ['pnp','rectangular_protocol','net','online_background','background_augmentation',
         'bop_bb8','crop_jitter','blur_augmentation','rectangular_server']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def stage(out):
    out.mkdir(parents=True, exist_ok=False)
    for source in (ROOT/'bop_bb8').rglob('*.py'):
        copy(source,out/source.relative_to(ROOT))
    for name in SCRIPTS:
        copy(ROOT/f'scripts/{name}.py',out/f'scripts/{name}.py')
    for name in TESTS:
        copy(ROOT/f'tests/test_{name}.py',out/f'tests/test_{name}.py')
    for name in ['single_rgb_rect_20260926','rectangular_server_data_20260926','background_data_hashes']:
        copy(ROOT/f'configs/{name}.json',out/f'configs/{name}.json')
    for name in ['requirements.txt','requirements-yolo.txt']:
        copy(ROOT/name,out/name)
    for name in ['SINGLE_RGB_RECT_DESIGN_20260926','PNP_CUBOID_POSTPROCESS_20260926','BOXDREAMER_DESIGN_AUDIT_20260926']:
        copy(ROOT/f'docs/{name}.md',out/f'docs/{name}.md')
    copy(ROOT/'docs/RECTANGULAR_SERVER_PACKAGE_20260926.md',out/'OPENCODE_SERVER_GUIDE.md')
    copy(ROOT/'prompts/opencode_rectangular_20260926.txt',out/'OPENCODE_PROMPT.txt')
    yolo = ROOT/'outputs/yolo11s_cleanblur5000_real48_20260925/train/weights/best.pt'
    if sha(yolo) != '78c522b39c2f40dd7fb83447f7987e2cc2cae5f712bf7a6e6620233d66655dd6':
        raise ValueError('Unexpected fixed YOLO weights')
    copy(yolo,out/'weights/yolo11s_frozen.pt')
    imagenet = Path.home()/'.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
    if not sha(imagenet).startswith('f37072fd'):
        raise ValueError('Unexpected ImageNet weights')
    copy(imagenet,out/'weights/torch/hub/checkpoints/resnet18-f37072fd.pth')
    real = ROOT/'exports/resnet_tight_cuboid_fixed_yolo_20260925/real'
    sources = sorted(real.rglob('*.jpg'))
    sources += [ROOT/f'outputs/single_rgb_rect_20260926/preview/frame_{f:06d}.png' for f in [473,500]]
    records = []
    for source in sources:
        copy(source,out/'real/frames'/source.name)
        records.append(dict(image=f'frames/{source.name}',frame=int(source.stem.split('_')[-1]),sha256=sha(source)))
    if len(records) != 11 or len({r['frame'] for r in records}) != 11:
        raise ValueError('Real diagnostic frame count mismatch')
    (out/'real/manifest.json').write_text(json.dumps(dict(records=records,
        usage='RGB only, qualitative development diagnostic; never ResNet corner supervision'),indent=2)+'\n')
    for path in (ROOT/'outputs/single_rgb_rect_20260926/preview').glob('*'):
        if path.name.endswith(('_rect.png','_old.png','_gray.png')) or path.name=='index.html':
            copy(path,out/'local_input_preview'/path.name)
    copy(ROOT/'outputs/single_rgb_rect_20260926/preview/inputs.json',out/'local_input_preview/inputs.json')
    copy(ROOT/'outputs/single_rgb_rect_20260926/geometry_audit.json',out/'local_qa/pilot_geometry_audit.json')
    copy(ROOT/'outputs/single_rgb_rect_20260926/benchmark.json',out/'local_qa/pilot_benchmark.json')
    preview = out/'local_input_preview/index.html'
    preview.write_text(preview.read_text().replace('../geometry_audit.json','../local_qa/pilot_geometry_audit.json')
                       .replace('../benchmark.json','../local_qa/pilot_benchmark.json'))
    (out/'BUNDLE_MANIFEST.json').write_text(json.dumps(dict(version=1,name=NAME,
        yolo_frozen_sha256=sha(yolo),imagenet_resnet18_sha256=sha(imagenet),
        external_full_dataset_required=True,raw_train_instances=40000,validation_instances=3048,
        real_diagnostic_frames=11,formal_training_performed_locally=False,
        synthetic_model_weights_included=False,initialization='ImageNet ResNet18; newly initialized prediction head',
        real_pnp_default=False,real_corner_training=False),indent=2)+'\n')
    (out/'START_HERE.md').write_text('# 服务器训练包\n\n先读 OPENCODE_SERVER_GUIDE.md；可复制 '
        'OPENCODE_PROMPT.txt 给 OpenCode。\n\n本包复用服务器 4 万实例的既有连续姿态数据。'
        '不是完整数据备份，也不是训练结果。新网络：ResNet18、无 RGB 补边的机身长方形输入。'
        '保持 YOLO 固定；PnP 默认只用于已知内参的合成评估。\n')
    sums(out)


def files(out):
    return sorted(p for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                  and p.name != 'SHA256SUMS')


def sums(out):
    (out/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in files(out)))


def pack(out):
    if not (out/'LOCAL_PACKAGE_QA.json').is_file():
        raise ValueError('Record local package QA before producing the deliverable')
    sums(out)
    archive = out.with_suffix('.tar.gz')
    if archive.exists():
        raise FileExistsError(archive)
    selected = [*files(out),out/'SHA256SUMS']
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for path in selected:
            if path.is_symlink():
                raise ValueError(f'Cannot package an external symlink: {path}')
            tar.add(path,arcname=str(Path(out.name)/path.relative_to(out)),recursive=False)
    with tarfile.open(archive) as tar:
        for path in selected:
            payload = tar.extractfile(str(Path(out.name)/path.relative_to(out))).read()
            if hashlib.sha256(payload).hexdigest() != sha(path):
                raise ValueError(f'Archive mismatch: {path}')
    digest = sha(archive)
    archive.with_name(archive.name+'.sha256').write_text(f'{digest}  {archive.name}\n')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=digest,files=len(selected)),indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=['stage','pack'],required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'exports'/NAME)
    args = parser.parse_args()
    (stage if args.action == 'stage' else pack)(args.output.resolve())
