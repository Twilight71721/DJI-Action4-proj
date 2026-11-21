"""Shared asset checks and compact result export for the background training bundle."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False))


def verify(args, config, full=False):
    data = args.dataset.resolve()
    if sha(data / 'manifest.json') != config['dataset_manifest_sha256']:
        raise ValueError('Wrong dataset manifest; do not re-split or substitute datasets')
    if sha(args.checkpoint) != config['checkpoint_sha256']:
        raise ValueError('Wrong starting checkpoint; use the returned 5000 paired best.pth')
    manifest = json.loads((data / 'manifest.json').read_text())
    split = manifest['splits']
    if any(set(split[a]) & set(split[b]) for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('Split overlap')
    expected = set(split['train'] + split['val']) | {f'{100000 + int(s):06d}' for s in split['train']}
    actual = {p.name for p in (data / 'trainval').iterdir() if (p / 'scene_gt.json').exists()}
    if actual != expected:
        raise ValueError('trainval contains unexpected or missing scenes; potential test/blur-val leakage')
    for scene in sorted(expected):
        variant, source = ('blur', f'{int(scene)-100000:06d}') if int(scene) >= 100000 else ('clean', scene)
        for name in ['scene_gt.json', 'scene_camera.json', 'scene_gt_info.json']:
            if sha(data / 'trainval' / scene / name) != sha(data / variant / source / name):
                raise ValueError(f'trainval metadata mismatch: {scene}/{name}')
    for frame in manifest['frames']:
        for variant in ['clean', 'blur']:
            image = data / frame[variant]
            if not image.is_file():
                raise FileNotFoundError(image)
            if full and sha(image) != frame[variant + '_sha256']:
                raise ValueError(f'Image checksum mismatch: {image}')
            if frame['split'] == 'train' or (frame['split'] == 'val' and variant == 'clean'):
                scene = frame['scene'] if variant == 'clean' else f"{100000+int(frame['scene']):06d}"
                linked = data / 'trainval' / scene / 'rgb' / image.name
                if not linked.is_file() or (not linked.samefile(image) and sha(linked) != sha(image)):
                    raise ValueError(f'trainval RGB mismatch: {linked}')
    contract = ROOT / 'configs/background_data_hashes.json'
    if not contract.exists():
        raise FileNotFoundError('Missing packaged data metadata contract')
    for rel, digest in json.loads(contract.read_text()).items():
        if sha(data / rel) != digest:
            raise ValueError(f'Dataset geometry metadata mismatch: {rel}')
    if not args.real_manifest.is_file():
        raise FileNotFoundError(args.real_manifest)
    for record in json.loads(args.real_manifest.read_text())['records']:
        if not (args.real_manifest.parent / record['image']).is_file():
            raise FileNotFoundError(record['image'])
    print(f'[verify] manifest, checkpoint, geometry, splits, RGB paths OK; full RGB hashes={full}', flush=True)
    return manifest


def pack(args):
    if not (args.output/'review/summary.json').exists():
        raise FileNotFoundError('Complete evaluation before packaging')
    if json.loads((args.output/'review/summary.json').read_text())['incomplete_diagnostic']:
        raise ValueError('Do not package a limited diagnostic as formal results')
    archive = args.output.with_suffix('.tar.gz')
    if archive.exists():
        raise FileExistsError(archive)
    with tarfile.open(archive, 'w:gz', compresslevel=3) as tar:
        for p in sorted(args.output.rglob('*')):
            if p.is_file() and not any(part.startswith('smoke_') for part in p.relative_to(args.output).parts):
                name = str(Path(args.output.name)/p.relative_to(args.output))
                if p.suffix == '.pth':
                    import torch
                    checkpoint = torch.load(p, map_location='cpu', weights_only=False)
                    reduced = {k: v for k, v in checkpoint.items() if k != 'optimizer'}
                    reduced['original_checkpoint_sha256'] = sha(p)
                    reduced['export_note'] = 'Optimizer omitted for inference transfer; full checkpoint retained on server'
                    buffer = io.BytesIO()
                    torch.save(reduced, buffer)
                    entry = tarfile.TarInfo(name)
                    entry.size = buffer.tell()
                    buffer.seek(0)
                    tar.addfile(entry, buffer)
                else:
                    tar.add(p, arcname=name, recursive=False)
    archive.with_name(archive.name+'.sha256').write_text(f'{sha(archive)}  {archive.name}\n')
    print('[return]', archive, flush=True)
