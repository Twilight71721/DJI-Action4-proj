"""Review video-wide pose neighborhoods without treating temporal labels as GT."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from review_video_pose_foregrounds import main as audit_foregrounds
from review_pose_replacement_pilot import draw

# Full-frame inspection: the left body is hidden or too dark to establish its
# face orientation. Interpolated detector boxes here are not usable pose evidence.
UNRESOLVED = {(1175, 'tag18'), (1425, 'tag18'), (1450, 'tag18'), (2150, 'tag18')}
FAMILY_NAMES = dict(rear_screen='背屏为主', side_rear_large='侧面＋较多背屏',
                    side_rear_small='侧面＋少量背屏', side_edge_on='侧面为主',
                    side_front_small='侧面＋少量前面', side_front_large='侧面＋较多前面',
                    unknown='遮挡／无法确定')


def main(out):
    audit_foregrounds(out)
    plan = json.loads((out / 'plan.json').read_text())
    audit = json.loads((out / 'audit.json').read_text())
    manifest = json.loads((out / 'manifest.json').read_text())
    review = out / 'review'
    (review / 'poses').mkdir(exist_ok=True)
    (review / 'contexts').mkdir(exist_ok=True)
    templates = {}
    for sample in manifest['samples']:
        rgba = cv2.imread(str(out / sample['rgba']), cv2.IMREAD_UNCHANGED)
        alpha = rgba[:, :, 3:] / 255
        neutral = np.rint(rgba[:, :, :3] * alpha + 90 * (1 - alpha)).astype(np.uint8)
        png = f'poses/{sample["id"]}.jpg'
        cv2.imwrite(str(review / png), draw(neutral, np.array(sample['keypoints_2d'])))
        templates.setdefault(sample['template'], []).append(dict(
            id=sample['id'], image=png, delta=sample['delta_degrees'],
            method=sample['pose_proposal_method']))

    references = []
    for reference in plan['references']:
        r = dict(reference)
        frame, device = r['frame'], r['device']
        r['family_label'] = FAMILY_NAMES[r['family']]
        r['status'] = ('user_retained' if 'user_retained' in r['pose_method'] else
                       'marker_supported_estimate' if 'marker' in r['pose_method'] else
                       'transition_interpolated' if 'slerp' in r['pose_method'] else
                       'manual_candidate' if 'manual_refined' in r['pose_method'] else
                       'coarse_orientation_neighborhood')
        r['precise_alignment_verified'] = False
        if (frame, device) in UNRESOLVED:
            r.update(status='unresolved_occluded', proposed_template=r['template'],
                     template=None, family='unknown', family_label=FAMILY_NAMES['unknown'],
                     source_preview=f'review/contexts/{frame:06d}.jpg')
        if frame in [108, 110, 112, 113, 114, 116, 118, 120, 122]:
            r['note'] = '快速转动阶段；粗分组不能代替该帧精确姿态，优先参考标记支持的候选。'
        if frame == 2849:
            r['note'] = '末帧含边界截断；此处前景素材为完整物体，截断增强尚未合成。'
        if r['status'] == 'unresolved_occluded':
            r['note'] = '完整帧复核后仍无法确定朝向；不把插值框或邻帧类别当作已确认姿态。'
        references.append(r)
        path = review / 'contexts' / f'{frame:06d}.jpg'
        if not path.exists():
            source = ROOT / f'outputs/video_reference_audit/head_left_rgb_raw/frames_full/frame_{frame:06d}.jpg'
            im = cv2.imread(str(source))
            cv2.imwrite(str(path), cv2.resize(im, (812, 616)), [cv2.IMWRITE_JPEG_QUALITY, 88])

    by_ref = {(r['frame'], r['device']): r for r in references}
    coverage = [json.loads(s) for s in (out / 'frame_coverage.jsonl').read_text().splitlines()]
    assert len(coverage) == 5700
    assert len({(r['frame'], r['device']) for r in coverage}) == 5700
    assert set(r['frame'] for r in coverage) == set(range(2850))
    for row in coverage:
        ref = by_ref[(row['reference_frame'], row['device'])]
        row.update(template=ref['template'], family=ref['family'], reference_status=ref['status'])
        row['orientation_verified_this_frame'] = False
        if ref['template'] is None:
            row['status'] = 'unresolved_reference_not_pose_covered'
        elif row['box_status'].startswith('interpolated'):
            row['status'] = 'missing_detection_temporal_proposal_needs_review'
    (out / 'frame_coverage.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in coverage))
    with (out / 'frame_coverage.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
    (out / 'reference_review.json').write_text(json.dumps(references, indent=2))
    segments = []
    for start, end in [(0, 124)] + [(s, min(s + 249, 2849)) for s in range(125, 2850, 250)]:
        rr = [r for r in references if start <= r['frame'] <= end]
        segments.append(dict(start_frame=start, end_frame=end,
            representative_frames=sorted({r['frame'] for r in rr}),
            source_group='video_adaptation_all', split='train_only',
            review_status='assistant_sampled_frame_review_not_user_verified',
            reference_status_counts=dict(Counter(r['status'] for r in rr)),
            families=dict(Counter(r['family'] for r in rr))))
    (out / 'segment_summary.json').write_text(json.dumps(segments, indent=2))
    supported_templates = {r['template'] for r in references if r['template']}
    for sample in manifest['samples']:
        if sample['template'] not in supported_templates:
            sample['usage'] = 'superseded_transition_candidate_not_for_export'
        sample['assigned_reference_frames'] = [r['frame'] for r in references if r['template'] == sample['template']]
    usable = sum(s['usage'] == 'foreground_candidate_for_server_compositing' for s in manifest['samples'])
    summary = dict(source_frames=2850, frame_device_rows=5700,
        keyframe_times=len({r['frame'] for r in references}), reference_views=len(references),
        rendered_foregrounds=len(manifest['samples']), orientation_templates=len(templates),
        supported_orientation_templates=len(supported_templates), usable_foregrounds=usable,
        reference_status_counts=dict(Counter(r['status'] for r in references)),
        frame_status_counts=dict(Counter(r['status'] for r in coverage)),
        segment_count=len(segments), max_reference_distance_frames=12,
        full_time_range_indexed=True, full_exact_pose_coverage_verified=False,
        user_frame500_choice='left20_up2 retained; isolated rendering uses view recentering',
        geometry_audit=audit['geometry_and_mask_checks_passed'],
        max_native_bop_projection_error_px=audit['max_native_bop_projection_error_px'],
        status='video_wide_pose_neighborhood_bank_with_explicit_uncertainty')
    (out / 'coverage_summary.json').write_text(json.dumps(summary, indent=2))
    manifest.update(status=summary['status'], coverage_summary=summary,
        counts=dict(Counter(s['usage'] for s in manifest['samples'])),
        caveats=['Full timeline indexing is not calibrated per-frame pose reconstruction.',
                 'Most unmarked views are visible-face bins with approximate rotations.',
                 'Four occluded reference views remain unresolved; temporal matches are not GT.',
                 'Each synthetic image has its own verified K/R/t and native body mask.',
                 'All descendants are train-only and share video_adaptation_all.',
                 'Full-video real-camera replacement, occlusion and truncation synthesis are not done.'])
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    data = dict(references=references, templates=templates, names=FAMILY_NAMES, summary=summary)
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>全视频姿态覆盖核查</title>
<style>body{margin:28px auto;max-width:1260px;padding:0 20px;background:#14232c;color:#e4edf1;font:17px/1.7 system-ui}a{color:#8de0ff}section,.card{background:#223944;padding:18px;margin:18px 0;border-radius:8px}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center}select,button,input{font:inherit;padding:6px}input{flex:1}img{max-width:100%;background:#596268}.pair{display:grid;grid-template-columns:1fr 1fr;gap:14px}.warn{color:#ffd590}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #53616a;text-align:left;padding:8px}.status{font-weight:bold;color:#a8e2ff}@media(max-width:760px){.pair{grid-template-columns:1fr}}</style>
<h1>整段视频：姿态候选库与覆盖记录</h1>
<p>扫描范围 <b>0–2849 帧</b>，144 个参考时刻、两台相机共 288 个参考视图；生成 <b>__TEMPLATES__ 个姿态模板、__RENDERS__ 张透明前景</b>。每张前景带原生 mask、K/R/t 和 8 角点投影。</p>
<p>其中 6 张初版过渡候选已被新版本取代，保留记录但不导出；可用素材为 <b>__USABLE__ 张</b>。</p>
<p class="warn"><b>覆盖范围：</b>全时间轴已有索引，主要可见面和转动阶段已有候选。未标记的侧视图采用粗角度估计，四处遮挡参考未确认；这不是逐帧精确位姿重建，也不是整段真实视频已完成替换。绿色框是真正渲染出的合成标签，不能当作真实帧真值。</p>
<p>第 500 帧保留左转 20°、上抬 2°。本页把物体移到光轴中心便于建立前景库，保持观察方向；原位置的替换结果仍在 <a href="../../video_side_pose_user20up2_20260924/review/index.html">原检查页</a>。</p>
<div class="card"><div class="controls"><button id="prev">上一时刻</button><select id="frame"></select><button id="next">下一时刻</button><select id="device"><option value="tag18">左侧实体相机／tag18／接口面</option><option value="tag19">右侧实体相机／tag19／DJI 面</option></select></div>
<div class="controls"><input type="range" id="time" min="0" max="143" value="0"><select id="variant"><option value="0">基础角度</option><option value="1">邻近角度 1</option><option value="2">邻近角度 2</option></select></div>
<h2 id="heading"></h2><p id="status" class="status"></p><p id="note" class="warn"></p>
<div class="pair"><div>真实参考裁剪（仅供观察）<img id="real"></div><div id="posepanel">合成姿态及其 BB8 标签<img id="pose"></div></div>
<p id="details"></p><details><summary>查看这一时刻的完整真实帧</summary><img id="context"></details></div>
<section><h2>仍需明确的四处遮挡参考</h2><p>第 1175、1425、1450、2150 帧的左侧相机。这里优先保留“未知”，不强行给出角度。它们的时间邻域也在 CSV 中标记未解决。</p>
<p>检测漏检与真实遮挡不是同一回事。所有 138 条漏检插值记录均有单独状态；其余检测框仍是自动候选，不是人工标注。</p></section>
<section><h2>时间分段</h2><table><thead><tr><th>帧范围</th><th>参考时刻数</th><th>说明</th></tr></thead><tbody id="segments"></tbody></table></section>
<section><h2>如何用于下一次训练</h2><p>这些素材是前景姿态增量，不是重训练结果。服务器端可在 mask 外随机换背景，并混合清晰／模糊图；几何标签保持不变。所有来自本视频的派生样本属于同一 train-only 来源组，不拆到独立测试集。</p>
<p>透明前景本身不含手、线缆、遮挡和画面截断。此部分与之前已接受的真实帧替换小样是不同资产；尚未把整段视频逐帧替换。</p>
<p><a href="../coverage_summary.json">覆盖统计</a> · <a href="../frame_coverage.csv">逐帧记录 CSV</a> · <a href="../reference_review.json">参考视图及不确定性</a> · <a href="../audit.json">标签检查</a> · <a href="../manifest.json">素材清单</a></p></section>
<script>const data=__DATA__,segments=__SEGMENTS__;
const frames=[...new Set(data.references.map(r=>r.frame))];
const el=id=>document.getElementById(id);frames.forEach(f=>{let o=document.createElement('option');o.value=f;o.textContent=`第 ${f} 帧 · ${(f/29.97).toFixed(2)} 秒`;el('frame').appendChild(o)});
const labels={user_retained:'用户选择的角度（未标定）',marker_supported_estimate:'标记辅助估计（内参与安装位置未标定）',transition_interpolated:'转动过渡候选：相邻参考姿态插值，尚未逐帧对齐',manual_candidate:'前一轮人工调整候选，未确认精确重合',coarse_orientation_neighborhood:'可见面粗分类＋邻近角度，尚未逐帧对齐',unresolved_occluded:'遮挡／证据不足：不算已确认姿态'};
function show(){const frame=+el('frame').value,device=el('device').value;el('time').value=frames.indexOf(frame);const r=data.references.find(x=>x.frame===frame&&x.device===device);el('heading').textContent=`第 ${frame} 帧 · ${device} · ${r.family_label}`;el('status').textContent=labels[r.status];el('note').textContent=r.note||'';el('real').src='../'+r.source_preview;el('context').src=`contexts/${String(frame).padStart(6,'0')}.jpg`;el('posepanel').style.visibility=r.template?'visible':'hidden';if(r.template){const s=data.templates[r.template][+el('variant').value];el('pose').src=s.image;el('details').textContent=`模板 ${r.template}；局部旋转增量 ${JSON.stringify(s.delta)}°；真实框状态：${r.box_status}。物体大小、位置已居中统一，不表示已贴合原图。`}else{el('details').textContent='没有可确认的真实姿态；暂不展示强行匹配的模型。'}}
['frame','device','variant'].forEach(id=>el(id).onchange=show);el('time').oninput=()=>{el('frame').value=frames[+el('time').value];show()};el('prev').onclick=()=>{el('frame').selectedIndex=Math.max(0,el('frame').selectedIndex-1);show()};el('next').onclick=()=>{el('frame').selectedIndex=Math.min(frames.length-1,el('frame').selectedIndex+1);show()};
segments.forEach(s=>{let tr=document.createElement('tr');for(const txt of [`${s.start_frame}–${s.end_frame}`,s.representative_frames.length,s.reference_status_counts.unresolved_occluded?'含无法确定的遮挡参考':'粗视角参考＋姿态候选']){let td=document.createElement('td');td.textContent=txt;tr.appendChild(td)}el('segments').appendChild(tr)});show();</script></html>'''
    page = page.replace('__DATA__', json.dumps(data)).replace('__SEGMENTS__', json.dumps(segments))
    page = page.replace('__TEMPLATES__', str(len(templates))).replace('__RENDERS__', str(len(manifest['samples'])))
    page = page.replace('__USABLE__', str(usable))
    (review / 'index.html').write_text(page)
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args().output.resolve())
