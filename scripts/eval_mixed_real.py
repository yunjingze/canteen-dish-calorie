"""What the mixed-plate head does on the eight hand-shot real trays.

The plate head is trained on composited trays and its headline number (plate-F1
54.5) comes from those composites.  data/splits/mixed_real holds eight real
photographs with human dish labels; three of them have an empty dish list on
purpose (a tray of food the 40-class list does not cover), so they test the
"everything else" side rather than the recall side.

    python scripts/eval_mixed_real.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from foodcls.mixed import detect_parts  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from foodcls.plate_classes import (PLATE_CLASS_NAMES, PLATE_KCAL_PER_100G,  # noqa: E402
                                   PLATE_PORTION_G, PLATE_ZH)
from scripts.train_wide import tta_transforms  # noqa: E402

CKPT = ROOT / 'results/runs/resnet18_multilabel_plate40/best.pt'
SRC = ROOT / 'data/splits/mixed_real'
OUT = ROOT / 'results/tables/mixed_real.json'


def read_rows(path: Path) -> list[dict]:
    with open(path, newline='', encoding='utf-8') as fh:
        return [r for r in csv.DictReader(fh) if r.get('filename')]


def labels_of(row: dict) -> list[str]:
    raw = (row.get('dishes') or '').replace(';', ',').replace(' ', ',')
    return [t for t in (s.strip() for s in raw.split(',')) if t]


@torch.no_grad()
def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    names = list(ckpt['class_names'])
    model = build_model(ckpt.get('arch', 'resnet18'), len(names))
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    thresholds = ckpt.get('class_thresholds') or {}
    tf = tta_transforms()
    rows = read_rows(SRC / 'labels.csv')
    print(f'{len(rows)} real trays, {len(names)} classes', flush=True)

    cases = []
    tp = fp = fn = 0
    for row in rows:
        path = SRC / row['filename']
        img = Image.open(path).convert('RGB')
        views = torch.stack([t(img) for t in tf]).to(device)
        probs = torch.softmax(model(views).float().mean(0), dim=0).cpu().numpy()
        truth = labels_of(row)
        parts = detect_parts(probs, class_thresholds=thresholds, class_names=names,
                             zh_table=PLATE_ZH, kcal_table=PLATE_KCAL_PER_100G,
                             portion_table=PLATE_PORTION_G)
        # TTA on the plate head is a horizontal flip pair in the deployed code,
        # but tta_transforms() centres all three framings; take the mean logit.
        got = [p['class'] for p in parts]
        hit = [c for c in truth if c in got]
        extra = [c for c in got if c not in truth]
        tp += len(set(truth) & set(got))
        fp += len(set(got) - set(truth))
        fn += len(set(truth) - set(got))
        cases.append({
            'file': row['filename'],
            'truth': truth,
            'truth_empty': not truth,
            'predicted': got,
            'probs': [{'class': p['class'], 'prob': round(float(p['prob']), 4)} for p in parts],
            'hit': hit,
            'extra': extra,
            'non_vocab_items': [s.strip() for s in (row.get('non_vocab_items') or '').split(';') if s.strip()],
            'kca': round(sum(float(p.get('kcal') or 0) for p in parts), 1),
        })
        shown = ', '.join(p['class'] + ' ' + format(p['prob'], '.2f') for p in parts)
        print(f"{row['filename']}: truth={truth} got={got} [{shown}]", flush=True)

    # The eight trays are a diagnostic set, not an evaluation split: three have
    # no in-vocabulary dish at all, so precision/recall over the five labelled
    # trays is reported separately from the three empties.
    labelled = [c for c in cases if not c['truth_empty']]
    empty = [c for c in cases if c['truth_empty']]
    tp_l = sum(len(set(c['truth']) & set(c['predicted'])) for c in labelled)
    fp_l = sum(len(set(c['predicted']) - set(c['truth'])) for c in labelled)
    fn_l = sum(len(set(c['truth']) - set(c['predicted'])) for c in labelled)
    prec = tp_l / (tp_l + fp_l) if tp_l + fp_l else 0.0
    rec = tp_l / (tp_l + fn_l) if tp_l + fn_l else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = {'ckpt': str(CKPT.relative_to(ROOT)).replace('\\', '/'),
           'n': len(cases),
           'labelled_trays': len(labelled),
           'empty_trays': len(empty),
           'labelled': {'tp': tp_l, 'fp': fp_l, 'fn': fn_l,
                        'precision': round(prec, 4), 'recall': round(rec, 4),
                        'f1': round(f1, 4)},
           'empty_trays_predicted_parts': [len(c['predicted']) for c in empty],
           'cases': cases}
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"labelled {len(labelled)} trays: P {prec:.3f} R {rec:.3f} F1 {f1:.3f} "
          f"| empty trays predicted parts {out['empty_trays_predicted_parts']}", flush=True)
    print(f'wrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()