"""Compare old/new BB8 checkpoints on frozen synthetic splits and fixed real crops."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bop_bb8 import BOPInstanceDataset, BOPModel
from bop_bb8.net import ResNetFPN, decode_topk, heatmap_to_input_points
from bop_bb8.torch_data import HeatmapDataset
from evaluate_heatmap_checkpoint import draw_overlay, metric_summary, make_contact_sheet
from infer_resnet_yolo_real import prepare_crop, draw_points


def load(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    saved = checkpoint['args']
    assert saved['input_size'] == [256, 256] and saved['crop_mode'] == 'bbox'
    assert saved['crop_padding'] == 1.3
    model = ResNetFPN(backbone=saved['backbone'], fpn_dim=saved['fpn_dim'], pretrained=False).cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    return model


@torch.inference_mode()
def predict(model, images):
    logits = model(images.cuda()).float()
    return heatmap_to_input_points(decode_topk(logits, k=9, from_logits=True),
                                   tuple(logits.shape[-2:]), (256, 256)).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    data, run = args.dataset.resolve(), args.run.resolve()
    out = run / 'review'
    out.mkdir(exist_ok=False)
    manifest = json.loads((data / 'manifest.json').read_text())
    checkpoints = dict(old=ROOT / 'outputs/resnet50_overfit_approved200/best.pth', new=run / 'best.pth')
    info = BOPModel.from_models_info(data / 'models/models_info.json', 1, 1.0)
    bases = {s: BOPInstanceDataset(data / 'all', info, scenes=manifest['splits'][s],
                                 input_size=(256, 256), heatmap_size=(64, 64),
                                 crop_mode='bbox', crop_padding=1.3,
                                 heatmap_kernel='gaussian', heatmap_sigma=2.0)
             for s in ('val', 'test')}
    annotations = json.loads((ROOT / 'dataset/action4-yolo-real48-mixed-20260922/manifest.json').read_text())
    real = [r for r in annotations['frames'] if r['domain'] == 'real' and r['split'] == 'test']
    yolo = json.loads((ROOT / 'outputs/yolo11s_real48_mixed_20260922/comparison/comparison.json').read_text())
    yolo_lookup = {Path(r['image']).stem: r['mixed']['predictions'] for r in yolo['frames']}
    crops, real_records = [], []
    for r in real:
        image = cv2.imread(r['source'])
        if image is None:
            raise FileNotFoundError(r['source'])
        for method, boxes in [('manual', r['boxes']), ('yolo', yolo_lookup[Path(r['image']).stem])]:
            for i, b in enumerate(boxes):
                roi, tensor, crop = prepare_crop(image, b['xyxy'], (256, 256), 1.3)
                crops.append((roi, tensor))
                real_records.append(dict(frame=r['frame'], method=method, instance=i,
                                         box=b['xyxy'], crop=crop, source=r['source'], predictions={}))
    summaries, synthetic_predictions = {}, {}
    for name, checkpoint in checkpoints.items():
        model = load(checkpoint)
        summaries[name] = {}
        synthetic_predictions[name] = {}
        for split, base in bases.items():
            loader = DataLoader(HeatmapDataset(base), batch_size=16, num_workers=2, shuffle=False)
            predictions, targets, valid = [], [], []
            for batch in loader:
                predictions.append(predict(model, batch['image']))
                targets.append(batch['keypoints_2d'].numpy())
                valid.append(batch['valid'].numpy())
            predictions, targets, valid = map(np.concatenate, (predictions, targets, valid))
            errors = np.linalg.norm(predictions - targets, axis=-1)
            errors[~valid] = np.nan
            summaries[name][split] = dict(metric_summary(errors), instances=len(base),
                                          excluded_corners=int((~valid).sum()))
            synthetic_predictions[name][split] = predictions
            np.savez_compressed(out / f'{name}_{split}.npz', predictions=predictions,
                                targets=targets, valid=valid, errors=errors)
        for start in range(0, len(crops), 16):
            pred = predict(model, torch.stack([x[1] for x in crops[start:start + 16]]))
            for i, points in enumerate(pred, start):
                real_records[i]['predictions'][name] = points.tolist()
        del model
        torch.cuda.empty_cache()
    synthetic_cards = []
    for split, base in bases.items():
        panels = []
        for i in np.linspace(0, len(base) - 1, min(12, len(base)), dtype=int):
            sample = base[int(i)]
            pair = [draw_overlay(sample.image, sample.keypoints_2d, synthetic_predictions[n][split][i],
                                 sample.valid, f'{n} {split} {sample.scene_id}/{sample.image_id}/{sample.instance_idx}')
                    for n in checkpoints]
            panels.append(np.hstack(pair))
        cv2.imwrite(str(out / f'{split}.jpg'), make_contact_sheet(panels, columns=2))
        synthetic_cards.append(f'<h2>{split}：左旧模型 / 右新模型；绿色标注 / 紫色预测</h2><a href="{split}.jpg"><img src="{split}.jpg"></a>')
    real_cards = []
    for i, r in enumerate(real_records):
        roi = crops[i][0]
        panel = np.hstack([roi, *[draw_points(roi, np.array(r['predictions'][n])) for n in checkpoints]])
        filename = f'real_{r["frame"]:06d}_{r["method"]}_{r["instance"]}.jpg'
        cv2.imwrite(str(out / filename), panel)
        real_cards.append(f'<h3>{r["frame"]:06d} · {r["method"]} · instance {r["instance"]}</h3><img loading="lazy" src="{filename}">')
    protocol = 'Synthetic val/test use fixed GT crops; errors in 256x256 input pixels. Real fixed manual/YOLO crops: qualitative only, no 8-corner ground truth. Same video as synthesis references; not an independent blind real test.'
    (out / 'summary.json').write_text(json.dumps(dict(protocol=protocol, synthetic=summaries,
        checkpoints={n: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n, p in checkpoints.items()}), indent=2))
    (out / 'real_predictions.json').write_text(json.dumps(dict(protocol=protocol, records=real_records), indent=2))
    rows = []
    for name in checkpoints:
        for split, m in summaries[name].items():
            rows.append(f'<tr><td>{name}</td><td>{split}</td><td>{m["instances"]}</td><td>{m["mean_err_px"]:.2f}</td><td>{m["pck5"]:.1%}</td><td>{m["p90_err_px"]:.2f}</td></tr>')
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>ResNet50 2630 张泛化对比</title><style>body{max-width:1200px;margin:30px auto;font:17px sans-serif;background:#14212a;color:#eee}img{max-width:100%}td,th{padding:12px;border:1px solid #aaa}a{color:#9df}</style><h1>ResNet50-FPN：200 张过拟合模型 vs 2630 张正常训练</h1><p>固定 2120 张训练、500 张验证、10 张测试。每张两个相机实例。最佳模型只依据验证集误差选择。数据源版本、训练划分及正则化有变化，不是仅改变数量的严格消融。</p><p>合成误差单位是 256×256 裁剪内像素；PCK@5 是误差不超过 5 像素的有效角点比例，使用真值框裁剪。测试集仅 10 张。</p><table><tr><th>模型</th><th>集合</th><th>实例</th><th>平均角点误差</th><th>PCK@5</th><th>P90 误差</th></tr>' + ''.join(rows) + '</table>' + ''.join(synthetic_cards) + '<h2>真实帧：原裁剪 / 旧 ResNet / 新 ResNet</h2><p>两模型使用完全相同的输入框。manual 为视觉估计框，yolo 为固定的已训练检测器输出；不根据角点预测改框。真实帧无可靠 8 角点真值，因此不计算真实误差或准确率。这些帧来自参与合成设计的视频，不是独立盲测。</p>' + ''.join(real_cards))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
