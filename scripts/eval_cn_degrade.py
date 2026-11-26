"""How far the canteen head degrades as the phone-photo artefacts grow.

The 22-class head is trained on clean studio photographs and has only ever been
scored on clean inputs, so nothing says how fast it falls apart under canteen
conditions.  This sweeps degrade_canteen_image() from 0.0 (plain resize) to 1.0
(the proxy the western models were trained on) and reports top-1, top-5, mean
posterior and the fraction the scope gate would still accept.

    python scripts/eval_cn_degrade.py --per-class 30
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from foodcls.app import _transforms  # noqa: E402
from foodcls.data import degrade_canteen_image  # noqa: E402
from foodcls.models import build_model  # noqa: E402

DEFAULT_CKPT = ROOT / 'results/runs/cn241_22_pre/best.pt'
OUT = ROOT / 'results/tables/cn_degrade.json'
EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
STRENGTHS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
#: the deployed canteen scope threshold (app.CN_SCOPE_THRESHOLD_21) and gate views
SCOPE_TH = 0.4088
GATE_VIEWS = 3


def sample(root: Path, per_class: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    items = []
    for cls in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(p for p in cls.iterdir() if p.suffix.lower() in EXTS)
        if not files:
            continue
        take = min(per_class, len(files))
        idx = sorted(rng.choice(len(files), size=take, replace=False).tolist())
        items += [(files[i], cls.name) for i in idx]
    return items


@torch.no_grad()
def score(model, paths, labels, names, device, temperature: float, batch: int = 24):
    tf = _transforms()[:GATE_VIEWS]
    correct = top5 = 0
    probs_all = []
    rob = []
    for start in range(0, len(paths), batch):
        chunk = paths[start:start + batch]
        views = torch.stack([torch.stack([t(Image.open(p).convert('RGB')) for t in tf])
                             for p in chunk]).to(device)
        b, v, c, h, w = views.shape
        logits = model(views.view(b * v, c, h, w)).float().view(b, v, -1)
        fused = torch.softmax(logits.mean(1) / temperature, dim=1)
        per_view = torch.softmax(logits / temperature, dim=2)
        probs_all.append(fused.cpu().numpy())
        rob.append(per_view.max(2).values.min(1).values.cpu().numpy())
        y = torch.tensor(labels[start:start + len(chunk)], device=device)
        correct += int((fused.argmax(1) == y).sum())
        top5 += int((fused.topk(5, dim=1).indices == y[:, None]).any(1).sum())
    probs = np.concatenate(probs_all) if probs_all else np.zeros((0, len(names)))
    rob = np.concatenate(rob) if rob else np.zeros(0)
    n = len(paths)
    return {
        'n': n,
        'top1': round(correct / n, 4),
        'top5': round(top5 / n, 4),
        'mean_prob': round(float(probs.max(1).mean()), 4),
        'gate_accept': round(float((rob >= SCOPE_TH).mean()), 4),
        'robust_p05': round(float(np.quantile(rob, 0.05)), 4),
        'robust_mean': round(float(rob.mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(DEFAULT_CKPT))
    ap.add_argument('--split', default='data/splits/cn241_22/test')
    ap.add_argument('--per-class', type=int, default=30)
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    names = list(ckpt['class_names'])
    model = build_model(ckpt.get('arch', 'resnet18'), len(names))
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    temperature = float(ckpt.get('temperature', 1.0) or 1.0)
    print(f"{Path(args.ckpt).parent.name}: {len(names)} classes, T={temperature:.3f}", flush=True)

    items = sample(Path(args.split), args.per_class)
    labels = [names.index(t) for _, t in items]
    print(f'{len(items)} images, {len(set(t for _, t in items))} classes', flush=True)

    curves = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for strength in STRENGTHS:
            paths = []
            for i, (src, _) in enumerate(items):
                dst = tmp / f'{strength:.1f}' / f'{i:05d}.jpg'
                degrade_canteen_image(src, dst, seed=1000 + i * 31, strength=strength)
                paths.append(dst)
            row = score(model, paths, labels, names, device, temperature)
            row['strength'] = strength
            curves.append(row)
            print(f"  strength {strength:.1f}: top1 {row['top1']:.3f} top5 {row['top5']:.3f} "
                  f"meanP {row['mean_prob']:.3f} gate {row['gate_accept']:.3f} "
                  f"rob p05 {row['robust_p05']:.3f}", flush=True)

    base = curves[0]['top1']
    out = {'ckpt': str(Path(args.ckpt).relative_to(ROOT)).replace('\\', '/'),
           'split': args.split, 'per_class': args.per_class,
           'scope_threshold': SCOPE_TH, 'gate_views': GATE_VIEWS,
           'strengths': list(STRENGTHS),
           'curves': curves,
           'top1_drop_at_full': round(base - curves[-1]['top1'], 4),
           'gate_drop_at_full': round(curves[0]['gate_accept'] - curves[-1]['gate_accept'], 4)}
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"top1 {base:.3f} -> {curves[-1]['top1']:.3f} "
          f"(-{out['top1_drop_at_full']:.3f}) | gate "
          f"{curves[0]['gate_accept']:.3f} -> {curves[-1]['gate_accept']:.3f}", flush=True)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()