"""Per-photograph breakdown of the 116-class model on the real canteen shots.

wide_real.json only reports the aggregate (n=100, top1 0.89).  This keeps every
prediction so the eleven misses can be attributed: which dish, which class the
model picked instead, whether the pick was in the 20-class subset the tab
restricts to, and how confident it was.

    python scripts/eval_wide_real_detail.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.classes_wide import WIDE_CLASS_NAMES, WIDE_NUM_CLASSES  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from scripts.train_wide import tta_transforms  # noqa: E402

CKPT = ROOT / 'results/runs/resnet18_food101_cn116/best.pt'
OUT = ROOT / 'results/tables/wide_real_detail.json'
SRC = ROOT / 'data/splits/canteen_real'
EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


def items() -> list[tuple[Path, str]]:
    out = []
    for folder in sorted(SRC.iterdir()):
        if not folder.is_dir() or folder.name not in CLASS_NAMES:
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in EXTS:
                out.append((p, folder.name))
    return out


@torch.no_grad()
def main() -> None:
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(ckpt.get('arch', 'resnet18'), WIDE_NUM_CLASSES)
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    tf = tta_transforms()
    rows = items()
    print(f'{len(rows)} photographs under {SRC.name}', flush=True)

    offset = len(CLASS_NAMES)
    out = []
    for path, truth in rows:
        img = Image.open(path).convert('RGB')
        views = torch.stack([t(img) for t in tf]).to(device)
        logits = model(views).float().mean(0)
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        order = np.argsort(-probs)
        top = WIDE_CLASS_NAMES[int(order[0])]
        top_in_subset = top in CLASS_NAMES
        # The demo's single tab restricts the softmax to the 20 western labels, so
        # its prediction is the best of those 20 -- not the global argmax relabelled.
        sub_cols = [WIDE_CLASS_NAMES.index(c) for c in CLASS_NAMES]
        sub = probs[sub_cols]
        sub_top = CLASS_NAMES[int(np.argmax(sub))]
        rank = int(np.where(order == WIDE_CLASS_NAMES.index(truth))[0][0]) + 1
        sub_rank = (sub_cols.index(WIDE_CLASS_NAMES.index(truth)) + 1
                    if WIDE_CLASS_NAMES.index(truth) in sub_cols else 0)
        out.append({
            'file': str(path.relative_to(SRC)).replace('\\', '/'),
            'truth': truth,
            'truth_is_canteen_dish': truth in CLASS_NAMES,
            'top1': top,
            'top1_prob': round(float(probs[order[0]]), 4),
            'top1_in_20': top_in_subset,
            'truth_rank': rank,
            'truth_prob': round(float(probs[WIDE_CLASS_NAMES.index(truth)]), 4),
            'correct': top == truth,
            'correct_20': sub_top == truth,
            'top1_20': sub_top,
            'top1_20_prob': round(float(np.max(sub)), 4),
            'truth_rank_20': sub_rank,
            'top3': [WIDE_CLASS_NAMES[int(i)] for i in order[:3]],
            'top3_prob': [round(float(probs[i]), 4) for i in order[:3]],
        })

    n = len(out)
    hits = [r for r in out if r['correct']]
    hits20 = [r for r in out if r['correct_20']]
    by_class = {}
    for r in out:
        d = by_class.setdefault(r['truth'], {'n': 0, 'correct': 0, 'correct_20': 0, 'misses': []})
        d['n'] += 1
        if r['correct']:
            d['correct'] += 1
        if r['correct_20']:
            d['correct_20'] += 1
        if not r['correct_20']:
            d['misses'].append({'file': r['file'], 'top1_20': r['top1_20'],
                                'p20': r['top1_20_prob'], 'rank20': r['truth_rank_20'],
                                'top1_116': r['top1'], 'p116': r['top1_prob']})
    per_class = {k: {'n': v['n'], 'correct': v['correct'], 'correct_20': v['correct_20'],
                     'acc_20': round(v['correct_20'] / v['n'], 3), 'misses': v['misses']}
                 for k, v in sorted(by_class.items())}

    misses = [r for r in out if not r['correct']]
    leak = [r for r in misses if r['top1_in_20']]
    summary = {
        'n': n,
        'top1_full116': round(len(hits) / n, 4),
        'top1_restricted20': round(len(hits20) / n, 4),
        'misses': len(misses),
        'miss_top1_outside_20': len(misses) - len(leak),
        'miss_top1_inside_20': len(leak),
        'mean_prob_correct': round(float(np.mean([r['top1_prob'] for r in hits])), 4) if hits else 0.0,
        'mean_prob_miss': round(float(np.mean([r['top1_prob'] for r in misses])), 4) if misses else 0.0,
    }
    data = {'ckpt': str(CKPT.relative_to(ROOT)).replace('\\', '/'),
            'threshold_note': 'top1_in_20 = prediction is one of the 20 labels the single tab scores',
            'summary': summary, 'per_class': per_class,
            'misses': [{k: r[k] for k in ('file', 'truth', 'top1', 'top1_prob', 'top1_20',
                                         'top1_20_prob', 'truth_rank', 'truth_rank_20',
                                         'truth_prob', 'top3', 'top3_prob')}
                       for r in misses]}
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"full116 {summary['top1_full116']:.4f} | restricted20 {summary['top1_restricted20']:.4f} "
          f"| misses {summary['misses']} (inside20 {summary['miss_top1_inside_20']}, "
          f"outside20 {summary['miss_top1_outside_20']})", flush=True)
    for r in misses:
        print(f"  {r['file']:<32} truth={r['truth']:<20} top1={r['top1']:<20} "
              f"p={r['top1_prob']:.3f} truth_rank={r['truth_rank']}")
    print(f'wrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()