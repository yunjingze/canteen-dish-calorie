"""Do the pairing prior's gains survive on plates it did not generate?

scripts/eval_mixed_plate.py rebuilds the val plates with the same prior flag as the
model being scored, so the baseline and the prior model are never compared on the
same trays.  This script crosses the two: both checkpoints are scored on the uniform
plate set and on the prior plate set.  Only the model changes across a row.

    python scripts/eval_pairing_prior.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.data import SplitDataset  # noqa: E402
from foodcls.mixed import build_eval_set  # noqa: E402
from foodcls.plate_classes import PLATE_CLASS_NAMES  # noqa: E402
from foodcls.utils import pick_device  # noqa: E402
from scripts.train_mixed import (  # noqa: E402
    component_metrics, make_plate_loader, predict_probs, real_canteen_samples,
    targets_for,
)

CN_ROOT = ROOT / 'data/splits/cn241_21'
CKPTS = {
    'plate40': ROOT / 'results/runs/resnet18_multilabel_plate40/best.pt',
    'plate40_prior': ROOT / 'results/runs/resnet18_multilabel_plate40_prior/best.pt',
}
OUT = ROOT / 'results/tables/pairing_prior_cross.json'
OFFSET = len(CLASS_NAMES)


def build_sets(prior: bool):
    cn_names = [c['slug'] for c in _cn_table()]
    val_src = SplitDataset(ROOT / 'data/splits/val', CLASS_NAMES, transform=None)
    cn_val = SplitDataset(CN_ROOT / 'val', cn_names, transform=None)
    west = build_eval_set(val_src.samples, 600, seed=1234, num_classes=OFFSET, prior=prior)
    cn = build_eval_set(cn_val.samples, 600, seed=4321, num_classes=len(cn_names),
                        label_map=[OFFSET + i for i in range(len(cn_names))], prior=prior,
                        prior_names=cn_names)
    real = real_canteen_samples(ROOT / 'data')
    present = sorted({lab for _, lab in real})
    remap = {g: i for i, g in enumerate(present)}
    real_local = [(p, remap[lab]) for p, lab in real]
    rp = build_eval_set(real_local, 80, seed=777, num_classes=len(present),
                        label_map=present, prior=prior,
                        prior_names=[cn_names[g] for g in present])
    return {'western': west, 'canteen': cn, 'real': rp}


def _cn_table():
    import json as _json
    rows = _json.loads((ROOT / 'results/tables/cn241_21_classes.json').read_text(encoding='utf-8'))['classes']
    return [r for r in rows if r['slug'] != 'other_cn']


def score(ckpt_path, plates, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    from foodcls.models import build_model
    model = build_model(ckpt.get('arch', 'resnet18'), len(PLATE_CLASS_NAMES))
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()
    probs = predict_probs(model, make_plate_loader(plates, workers=2, num_classes=len(PLATE_CLASS_NAMES)), device)
    targets = targets_for(plates, len(PLATE_CLASS_NAMES))
    thr = float(ckpt.get('threshold', 0.5))
    per_class = {str(k): float(v) for k, v in (ckpt.get('class_thresholds') or {}).items()}
    global_m = component_metrics(targets, probs, thr, class_names=PLATE_CLASS_NAMES)
    tuned = component_metrics(targets, probs, 0.2, class_thresholds=per_class,
                              class_names=PLATE_CLASS_NAMES)
    # a threshold sweep at a *shared* cut-off is the only way to compare two
    # checkpoints whose own thresholds were fitted on different plate sets
    sweep = {f'{t:.2f}': component_metrics(targets, probs, float(t), class_names=PLATE_CLASS_NAMES)
             for t in np.round(np.arange(0.20, 0.61, 0.05), 2)}
    del model
    torch.cuda.empty_cache()
    return {'global': global_m, 'tuned': tuned, 'sweep': sweep}


def main():
    device = pick_device()
    print('device', device, flush=True)
    manifest = {mode: build_sets(prior=(mode == 'prior')) for mode in ('uniform', 'prior')}
    out = {'plates': {mode: {k: len(v) for k, v in sets.items()} for mode, sets in manifest.items()},
           'results': {}}
    for ckpt_name, path in CKPTS.items():
        out['results'][ckpt_name] = {}
        for mode, sets in manifest.items():
            out['results'][ckpt_name][mode] = {}
            for split, plates in sets.items():
                res = score(path, plates, device)
                out['results'][ckpt_name][mode][split] = res
                g, t = res['global'], res['tuned']
                print(f"{ckpt_name:14} on {mode:8} {split:8} "
                      f"global F1 {g['f1']*100:5.2f} exact {g['exact_match']*100:5.2f} parts {g['parts_per_plate']:.2f} | "
                      f"tuned F1 {t['f1']*100:5.2f} exact {t['exact_match']*100:5.2f} parts {t['parts_per_plate']:.2f}",
                      flush=True)
    OUT.write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('wrote', OUT.relative_to(ROOT))


if __name__ == '__main__':
    main()