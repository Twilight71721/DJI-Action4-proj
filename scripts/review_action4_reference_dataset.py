"""Summarize reference coverage and attachment audits without claiming calibration."""
import argparse
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'outputs/video_reference_audit/head_left_rgb_raw/frames_full'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    review = root / 'review'
    review.mkdir(exist_ok=True)
    references = review / 'references'
    references.mkdir(exist_ok=True)
    totals = Counter()
    minima = {'cuff_m': float('inf'), 'mount_and_plug_m': float('inf'),
              'cable_segment_surface_m': float('inf'), 'camera_sample_m': float('inf')}
    errors, coverage, rows, digest = [], [], [], None
    for path in sorted((root / 'render_metadata').glob('batch_*/metadata.json')):
        data = json.loads(path.read_text())
        batch = int(path.parent.name.split('_')[-1])
        spec = data['reference_profile']
        reference_frame = spec['reference_frame']
        real = REFERENCE / f'frame_{reference_frame:06d}.jpg'
        target = references / real.name
        with Image.open(real) as im:
            im.thumbnail((812, 616))
            im.save(target, quality=92)
        for frame in data['frames']:
            totals['frames'] += 1
            totals['light_' + frame['lighting']['profile']] += 1
            totals['rejected_pose_proposals'] += len(frame['rejected_geometry_proposals'])
            for item in frame['cuff_clearance']:
                totals['cuffs_checked'] += 1
                minima['cuff_m'] = min(minima['cuff_m'], item['min_sample_clearance_m'])
                if item['skin_intersection_triangle_pairs'] or item['min_sample_clearance_m'] < .0005:
                    errors.append(f'{batch}/{frame["frame"]}: cuff clearance')
            for item in frame['mount_clearance']:
                totals['mount_and_plug_parts_checked'] += 1
                minima['mount_and_plug_m'] = min(minima['mount_and_plug_m'], item['min_sample_clearance_m'])
                if item['skin_intersection_triangle_pairs'] or item['min_sample_clearance_m'] < .0005:
                    errors.append(f'{batch}/{frame["frame"]}: mount/plug clearance')
            for item in frame.get('bottom_mount_clearance', []):
                totals['bracket_body_checks'] += 1
                gap = item['bottom_contact_gap_m']
                if item['body_envelope_intersection_pairs'] or item['body_internal_vertices'] or (gap is not None and not 0 <= gap <= .0005):
                    errors.append(f'{batch}/{frame["frame"]}: bracket/body contact')
            for item in frame['cable_clearance']:
                totals['cables_checked'] += 1
                d = min(item['conservative_segment_clearance_m'].values())
                minima['cable_segment_surface_m'] = min(minima['cable_segment_surface_m'], d)
                if d < .001 or item['connector_gap_m'] > 1e-7:
                    errors.append(f'{batch}/{frame["frame"]}: cable clearance/connection')
            for item in frame['fixture_clearance']:
                minima['camera_sample_m'] = min(minima['camera_sample_m'], min(item['camera_obb_sample_min_clearance_m'].values()))
            for item in frame['handedness'] or []:
                # The renderer rejects wrong chirality before recording metadata.
                totals['tabletop_hand_orientation_checks'] += 1
        if digest is not None and digest != data['glb_sha256']:
            errors.append(f'{batch}: GLB identity differs')
        digest = data['glb_sha256']
        coverage.append(dict(batch=batch, reference_frame=reference_frame, action=spec['action'],
                             frames=len(data['frames']), reconstruction=spec['reconstruction']))
        cards = []
        for f in range(min(4, len(data['frames']))):
            src = f'../train_pbr/{batch:06d}/rgb/{f:06d}.png'
            light = data['frames'][f]['lighting']['profile']
            cards.append(f'<a href="{src}" target="_blank"><img loading="lazy" src="{src}"><span>{html.escape(light)}</span></a>')
        rows.append(f'<section><h2>真实第 {reference_frame} 帧 · {html.escape(spec["action"])} · {len(data["frames"])} 个变体</h2>'
                    f'<div class="comparison"><a href="{target.relative_to(review)}"><img loading="lazy" src="{target.relative_to(review)}"><span>真实参考（仅供对照）</span></a><div class="variants">{"".join(cards)}</div></div></section>')
    actual_digest = hashlib.sha256((root / 'models/obj_000001.glb').read_bytes()).hexdigest()
    if digest != actual_digest:
        errors.append('Delivered GLB hash differs from render metadata')
    report = dict(counts=dict(totals), minimum_clearances=minima, coverage=coverage,
                  glb_sha256=actual_digest, errors=errors, attachment_checks_passed=not errors,
                  audit_scope='Evaluated cuff and mount/plug parts vs own skin; bottom bracket vs camera body envelope; dense cable tube bound vs both arms; sampled camera OBB vs both arms. Not cloth contact or full-scene collision validation.',
                  pose_status='Qualitative representative-frame guided; no measured real intrinsics or per-frame fitted 2D landmark error.',
                  limitations=['Not photorealistically identical to real video.',
                               'Camera tilt and hand contact still differ from reference poses.',
                               'Cloth uses kinematic deformation, without a physical contact solver.',
                               'Each device screen is rendered from its virtual lens at batch frame 0 and held within that batch.',
                               'No training or held-out real-video generalization evaluation has been run.'])
    (review / 'attachment_validation.json').write_text(json.dumps(report, indent=2))
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>真实帧与腕部安装修正对照</title>
<style>body{margin:28px;background:#172027;color:#eee;font:16px system-ui;line-height:1.75}h1{font-size:28px}h2{font-size:20px}a{color:#9bd9ff;text-decoration:none}img{width:100%;display:block}span{display:block;padding:6px}.comparison{display:grid;grid-template-columns:1fr 1fr;gap:16px}.variants{display:grid;grid-template-columns:1fr 1fr;gap:10px}section{border-top:1px solid #46535b;padding:14px 0}.note{max-width:1100px;background:#25313b;padding:18px}@media(max-width:850px){.comparison{grid-template-columns:1fr}}</style>
<h1>真实帧与腕部安装修正对照</h1>
<div class="note"><p>__COUNT__ 张，__PROFILES__ 个视频代表帧。相机移至腕带内侧，腕带按最终皮肤截面生成且向外加厚，底座改为曲面；插头与线缆也纳入间距检查。</p>
<p>几何检查通过并不等于真实姿态已准确复原。本批仍有机身倾角、手指接触、衣物形状与真实帧的差异；缺少真实相机标定，也没有逐帧二维关键点误差验收。因此这是供检查的合成样本，不能宣称“几乎相同”或已证明泛化能力。</p>
<p>真实图片只出现在此对照页，没有复制到合成 RGB 或屏幕贴图。<a href="index.html">全部样本</a> · <a href="attachment_validation.json">腕带、支架、插头、线缆检查</a> · <a href="validation.json">BOP 标注检查</a></p></div>
'''
    page = page.replace('__COUNT__', str(totals['frames'])).replace('__PROFILES__', str(len(coverage)))
    (review / 'reference_comparison.html').write_text(page + '\n'.join(rows) + '</html>')
    print(json.dumps({'counts':dict(totals),'minimum_clearances':minima,'errors':errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
