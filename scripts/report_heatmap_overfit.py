#!/usr/bin/env python3
"""Build an auditable training-set overfit report from saved evaluation results."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir
    out = run / "evaluation"
    summary = json.loads((out / "summary.json").read_text())
    config = json.loads((run / "config.json").read_text())
    if not config.get("overfit"):
        raise ValueError("This report requires an explicit overfit run")
    baseline = json.loads((run / "baseline.json").read_text())
    history = list(csv.DictReader((run / "metrics.csv").open()))
    rows = list(csv.DictReader((out / "per_sample_metrics.csv").open()))
    data = np.load(out / "predictions.npz")
    errors = np.linalg.norm(data["predictions"] - data["truths"], axis=-1)
    errors[~data["valid"]] = np.nan
    manifest = json.loads((args.dataset / "images/manifest.json").read_text())
    lookup = {entry["source"]: entry for entry in manifest}
    for row in rows:
        source = f"train_pbr/{row['scene_id']}/rgb/{int(row['image_id']):06d}.png"
        row.update(lookup[source])

    grouped = {}
    for field in ("lighting", "action"):
        grouped[field] = {}
        for label in sorted({row[field] for row in rows}):
            selection = [int(row["dataset_index"]) for row in rows if row[field] == label]
            values = errors[selection]
            values = values[np.isfinite(values)]
            grouped[field][label] = dict(instances=len(selection), mean_px=float(values.mean()),
                                        pck5=float(np.mean(values <= 5)))
    (out / "group_metrics.json").write_text(json.dumps(grouped, indent=2))

    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].semilogy(epochs, [float(row["train_loss"]) for row in history], label="Training updates")
    axes[0].semilogy(epochs, [float(row["val_loss"]) for row in history], label="Same-set eval mode")
    axes[0].set_ylabel("Focal + 10 x coordinate loss")
    axes[0].legend()
    axes[1].plot(epochs, [float(row["val_err_px"]) for row in history])
    axes[1].set_ylabel("Mean corner error (256px crop)")
    axes[1].set_yscale("log")
    axes[2].plot(epochs, [100 * float(row["val_pck5"]) for row in history])
    axes[2].set_ylabel("PCK@5px (%)")
    axes[2].set_ylim(0, 101)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    fig.suptitle("ResNet50-FPN: same 400 instances for training and evaluation (no held-out set)")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=150)
    plt.close(fig)

    m = summary["all"]
    raw = summary["original_image_pixels"]
    worst = sorted(rows, key=lambda row: float(row["mean_err_px"]), reverse=True)
    mean_errors = np.nanmean(errors, axis=1)
    perfect_instances = int(np.sum(np.all((errors <= 5) | ~data["valid"], axis=1)))
    text = f"""# ResNet50：200 张合成图过拟合测试

本次是训练集记忆能力与流程检查，全部 200 张图、400 个相机实例用于训练并在同一批实例上评估。没有独立验证集，不代表真实视频泛化性能。

- ImageNet V2 预训练 ResNet50 + FPN（128 通道），随机初始化热图头。
- 输入为标注框扩展 1.3 倍的 256×256 RGB 裁剪；输出 8×64×64 角点热图。没有训练检测器。
- 100 轮，batch 16，AMP；AdamW，head LR 0.001、backbone LR 0.0001，cosine decay，weight decay 0，前 2 轮冻结 backbone。无随机增强，随机种子 20260921。
- 损失为 focal + 10×归一化坐标 Smooth-L1；采用现有 top-9 加权解码，不在评估时替换解码器。
- 以训练集平均角点误差选择 best.pth，第 {summary['checkpoint_epoch_one_based']} 轮；另保留 last.pth。
- 旧训练脚本 CSV 中 val_* 列在本次运行中指同一训练集的 eval-mode 指标，不是验证集。

| 指标 | 结果 |
|---|---:|
| 初始化后、训练前平均角点误差 | {baseline['err_px']:.3f} px |
| 最佳模型平均角点误差 | {m['mean_err_px']:.3f} px |
| 中位数 / P95 / 最大单角点误差 | {m['median_err_px']:.3f} / {m['p95_err_px']:.3f} / {m['max_err_px']:.3f} px |
| PCK@2 / @5 / @10 | {m['pck2']:.2%} / {m['pck5']:.2%} / {m['pck10']:.2%} |
| 八角点均在 5px 以内的实例 | {perfect_instances} / 400 |
| 误差最大实例的八角点平均误差 | {mean_errors.max():.3f} px |
| 有效角点 / 排除角点 | {m['corner_count']} / {summary['excluded_corners']} |
| 映射回原图的平均角点误差 | {raw['mean_err_px']:.3f} px |

上表除最后一项外均以 256×256 裁剪坐标计量。PCK@5 表示误差不超过 5px 的角点比例。每个实例使用已知目标框，不能据此声称具备整图检测能力；也未测量 PnP 三维位姿精度。

![训练曲线](training_curves.png)

![随机及最差样本](prediction_contact_sheet.png)

绿色为标注，紫色为预测；同时展示随机样本和误差最大的样本。网页可浏览全部 400 个实例。
"""
    tables = []
    for field, groups in grouped.items():
        text += f"\n## 按 {field} 分组（同一训练集）\n\n| 分组 | 实例数 | 平均误差 | PCK@5 |\n|---|---:|---:|---:|\n"
        table = "<table><tr><th>分组</th><th>实例</th><th>平均误差</th><th>PCK@5</th></tr>"
        for label, metrics in groups.items():
            n, err, pck = metrics["instances"], metrics["mean_px"], metrics["pck5"]
            text += f"| {label} | {n} | {err:.3f} px | {pck:.2%} |\n"
            table += f"<tr><td>{html.escape(label)}</td><td>{n}</td><td>{err:.3f}px</td><td>{pck:.2%}</td></tr>"
        tables.append(table + "</table>")
    (out / "report_zh.md").write_text(text)
    cards = []
    for row in worst:
        index = int(row["dataset_index"])
        cards.append(f"<figure><img loading='lazy' src='all_overlays/{index:04d}.jpg'><figcaption>#{index} · {html.escape(row['action'])} · {html.escape(row['lighting'])}<br>平均 {float(row['mean_err_px']):.3f}px；最大 {float(row['max_err_px']):.3f}px</figcaption></figure>")
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ResNet50 过拟合测试</title>
<style>body{{background:#14212a;color:#e9eff2;font:17px system-ui;max-width:1450px;margin:40px auto;padding:20px}}a{{color:#8dd2ff}}p{{line-height:1.7}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:12px}}figure{{margin:0;background:#263641;padding:10px}}figure img{{width:100%}}.wide{{width:100%}}td,th{{padding:9px 20px;border-bottom:1px solid #52616a;text-align:left}}table{{display:inline-table;margin:15px;vertical-align:top}}.metric{{font-size:26px;color:#a5e8c1}}</style>
<h1>ResNet50 · 200 张合成图过拟合测试</h1>
<p>全部 200 张、400 个相机实例参与训练；以下为同一训练集的拟合效果。使用标注框裁剪成 256×256 输入，<b>不是独立验证，也不是整图检测测试</b>。</p>
<p class="metric">平均误差 {m['mean_err_px']:.3f}px · PCK@5 {m['pck5']:.2%} · 8 点均 ≤5px：{perfect_instances}/400</p>
<p>100 轮训练，最佳为第 {summary['checkpoint_epoch_one_based']} 轮。绿色为真实标注，紫色为预测。像素误差按裁剪图计量；原图平均误差为 {raw['mean_err_px']:.3f}px。无法从本实验判断真实视频泛化能力。</p>
<p><a href="report_zh.md">完整报告</a> · <a href="summary.json">评估指标</a> · <a href="../metrics.csv">逐轮日志</a> · <a href="../best.pth">最佳权重</a></p>
<img class="wide" src="training_curves.png"><h2>随机样本与最差样本</h2><img class="wide" src="prediction_contact_sheet.png">
<h2>各光照、动作的训练集效果</h2>{''.join(tables)}
<h2>全部 400 个实例，按误差从大到小排列</h2><div class="grid">{''.join(cards)}</div></html>"""
    (out / "index.html").write_text(page)
    print(out / "index.html")


if __name__ == "__main__":
    main()
