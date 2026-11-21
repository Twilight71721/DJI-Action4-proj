"""Summarize bbox controls, real adaptation, and synthetic retention."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    before = json.loads((root/'bbox_baseline/results.json').read_text())
    result = json.loads((root/'comparison/results.json').read_text())
    training = json.loads((root/'training/summary.json').read_text())
    labels = dict(initial='原 ResNet', synthetic_control='仅合成继续训练', real_mix='合成＋真实八点微调')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    curve_fields = [('real_adapt_error', 'real_adapt', 'Real adaptation set (5 objects)'),
                    ('real_val_error', 'real_validation', 'Real validation (3 objects)'),
                    ('synthetic_val_error', 'synthetic_validation', 'Synthetic replay validation (200 crops)')]
    for name, arm in training.items():
        epochs = [0] + [h['epoch'] for h in arm['history']]
        for ax, (field, key, title) in zip(axes, curve_fields):
            values = [arm['initial'][key]['mean_error_px']] + [h[field] for h in arm['history']]
            ax.plot(epochs, values, marker='.', label=name)
            ax.set(title=title, xlabel='Epoch', ylabel='Mean corner error / 256px crop')
            ax.grid(alpha=.25)
    axes[0].legend()
    fig.savefig(root/'learning_curves.png', dpi=160)
    plt.close(fig)
    rows, md_rows = [], []
    for split in ('adapt', 'validation', 'test'):
        for mode in ('reference', 'yolo'):
            for name in labels:
                m = result['metrics'][name][split][mode]
                mean = m['all8']['mean_error_px']
                err = f'{mean:.2f}' if mean is not None else '无输出'
                visible = m['visible_reference']['mean_error_px']
                inferred = m['inferred']['mean_error_px']
                v = f'{visible:.2f}' if visible is not None else '—'
                q = f'{inferred:.2f}' if inferred is not None else '—'
                values = [split, mode, labels[name], err, f'{m["all8"]["pck5"]:.1%}', v, q]
                rows.append('<tr>'+''.join(f'<td>{html.escape(x)}</td>' for x in values)+'</tr>')
                md_rows.append('| '+' | '.join(values)+' |')
    retention = []
    for name, arm in training.items():
        epoch = arm['best_epoch']
        if epoch:
            chosen = arm['history'][epoch-1]
            train_error, syn_error = chosen['real_adapt_error'], chosen['synthetic_val_error']
        else:
            train_error = arm['initial']['real_adapt']['mean_error_px']
            syn_error = arm['initial']['synthetic_validation']['mean_error_px']
        retention.append(dict(arm=name, best_epoch=epoch, real_training_error=train_error,
                              synthetic_error_before=arm['initial']['synthetic_validation']['mean_error_px'],
                              synthetic_error_after=syn_error))
    base = before['metrics']['initial']['validation']
    val = result['metrics']['real_mix']['validation']['reference']['all8']
    ctl = result['metrics']['synthetic_control']['validation']['reference']['all8']
    initial = result['metrics']['initial']['validation']['reference']['all8']
    test = result['metrics']['real_mix']['test']['reference']['all8']
    yolo_val = result['metrics']['real_mix']['validation']['yolo']['all8']
    main_text = (f'第一项：9/9 个已标注相机被新版 YOLO 检出。固定原 ResNet，真实验证的参考框误差 '
                 f'{base["reference"]["all8"]["mean_error_px"]:.2f}px，YOLO 框 '
                 f'{base["yolo"]["all8"]["mean_error_px"]:.2f}px。此次换框未解决主要角点错误。')
    adaptation_text = (f'第二项：同一批 3 个验证实例，参考框八点平均误差：原模型 '
                       f'{initial["mean_error_px"]:.2f}px → 仅合成对照 {ctl["mean_error_px"]:.2f}px → '
                       f'合成＋真实微调 {val["mean_error_px"]:.2f}px。'
                       f'微调后唯一测试实例误差 {test["mean_error_px"]:.2f}px，PCK@5 {test["pck5"]:.1%}。')
    syn_table = ''.join(f'<tr><td>{labels[r["arm"]]}</td><td>{r["best_epoch"]}</td>'
                        f'<td>{r["real_training_error"]:.2f}</td><td>{r["synthetic_error_before"]:.2f}</td>'
                        f'<td>{r["synthetic_error_after"]:.2f}</td></tr>' for r in retention)
    selected = training['real_mix']['best_epoch']
    conclusion = ('在这个小批次上，加入真实监督的验证平均误差低于相同训练预算的仅合成对照。'
                  if val['mean_error_px'] < ctl['mean_error_px'] else
                  '在这个小批次上，加入真实监督的验证平均误差没有低于相同训练预算的仅合成对照。')
    remaining = (f'仍未达到可用定位精度：参考框下验证 PCK@5 为 {val["pck5"]:.1%}，'
                 f'测试 PCK@5 为 {test["pck5"]:.1%}。微调模型换成新版 YOLO 框后，验证平均误差为 '
                 f'{yolo_val["mean_error_px"]:.2f}px，而参考框为 {val["mean_error_px"]:.2f}px，'
                 '说明裁剪敏感性仍然存在。训练实例拟合良好，不能替代留出帧上的泛化。')
    caution = '全部八点参与训练和评估；72 个标注点中 44 个为隐藏或虚拟角点的人工推定，因此同时列出可见参考点与推定点误差。本实验只有 5 个训练、3 个验证、1 个测试实例，且来自已用于背景适配的视频，不能当作独立视频泛化结论。'
    links = ('<p><a href="bbox_baseline/index.html">第一项：固定权重 bbox 对照</a> · '
             '<a href="comparison/index.html">第二项：原模型／仅合成／真实微调逐图对比</a> · '
             '<a href="training/real_mix/best.pth">真实微调最佳权重</a> · '
             '<a href="training/protocol.json">冻结实验协议</a> · <a href="ANALYSIS.md">分析文档</a></p>')
    (root/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>ResNet 真实样本适配结果</title><style>body{max-width:1450px;margin:32px auto;padding:0 22px;background:#14212a;color:#eef3f6;font:17px/1.7 sans-serif}img{width:100%}table{border-collapse:collapse}td,th{border:1px solid #567;padding:8px}a{color:#83d5ff}.note{background:#293f4b;padding:20px;border-radius:8px}article{padding:20px 0}</style>'
        '<h1>ResNet：bbox 对照与真实八点适配</h1>' + links +
        f'<div class="note"><p>{main_text}</p><p>{adaptation_text}</p><b>{conclusion}</b><p><strong>{remaining}</strong></p></div>'
        f'<p>{caution}</p><p>每个训练分支固定 15 轮×50 次更新，batch=8。真实微调每批 6 个合成＋2 个真实；对照为 8 个合成。'
        '同初始化、相同优化器与更新次数，前两轮冻结主干，之后低学习率微调，BatchNorm 统计保持不变。'
        '使用 800 个分层合成训练裁剪与 200 个合成验证裁剪，清晰模糊配对；未加入额外遮挡或裁剪扰动。'
        f'按真实验证误差选择权重（初始权重也可选），真实微调最佳为第 {selected} 轮。测试标签未用于选权重。</p>'
        '<h2>训练与保留能力</h2><img src="learning_curves.png">'
        '<table><tr><th>模型</th><th>最佳轮数</th><th>真实训练误差</th><th>合成验证训练前</th><th>合成验证训练后</th></tr>'
        + syn_table + '</table><h2>真实帧指标</h2><p>误差统一到参考裁剪 256 像素坐标。PCK@5 表示距人工参考点不超过 5 像素。两种 bbox 比较使用完全相同的目标实例。</p>'
        '<table><tr><th>集合</th><th>bbox</th><th>模型</th><th>八点均值 px</th><th>PCK@5</th><th>可见参考点均值</th><th>推定点均值</th></tr>'
        + ''.join(rows) + '</table><h2>代表性逐图结果</h2>'
        + ''.join(f'<article><h3>{r["id"]} · {r["split"]}</h3><img loading="lazy" src="comparison/{r["id"]}.jpg"></article>' for r in result['records']) + '</html>')
    text = '# ResNet bbox 对照与真实监督适配\n\n' + main_text + '\n\n' + adaptation_text + '\n\n' + conclusion + '\n\n' + remaining + '\n\n' + caution
    text += '\n\n| 集合 | bbox | 模型 | 八点均值px | PCK@5 | 可见参考点误差 | 推定点误差 |\n|---|---|---|---:|---:|---:|---:|\n' + '\n'.join(md_rows)
    text += '\n\n合成保留能力：\n\n' + '\n'.join(f'- {r["arm"]} 最佳 epoch {r["best_epoch"]}，真实训练误差 {r["real_training_error"]:.2f}px，合成验证 {r["synthetic_error_before"]:.2f} → {r["synthetic_error_after"]:.2f}px。' for r in retention)
    text += '\n\n本轮为单随机种子、小规模混合适配诊断。合成 replay 不包括仅存于服务器的完整连续姿态前景库。固定参考 bbox 与新版 YOLO bbox 的坐标均回到原图后统一尺度。所有历史结果、权重和原始标注保留。\n'
    (root/'ANALYSIS.md').write_text(text)
    print(main_text, adaptation_text, conclusion, sep='\n')


if __name__ == '__main__':
    main()
