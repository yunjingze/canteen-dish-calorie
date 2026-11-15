"""Analysis half of the ensembling study -- runs on the cached npz only."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUNS = [
    "resnet18_last_stage_aug",
    "resnet18_last_stage_aug_f101n",
    "resnet18_last_stage_aug_f101n_nodedup",
    "resnet18_last_stage_aug_f101n_e24",
    "resnet18_last_stage_aug_seed7",
]
SHORT = {"resnet18_last_stage_aug": "E4", "resnet18_last_stage_aug_f101n": "E7",
         "resnet18_last_stage_aug_f101n_nodedup": "E8",
         "resnet18_last_stage_aug_f101n_e24": "E12",
         "resnet18_last_stage_aug_seed7": "E13"}
SPLITS = ["test", "canteen", "canteen_s7"]


def load(run, split):
    d = np.load(ROOT / "results" / f"_ens_{run}__{split}.npz")
    return d["logits"].astype(np.float64), d["labels"]


def temp_of(run):
    try:
        cal = json.loads((ROOT / "results/runs" / run / "calibration.json").read_text(encoding="utf-8"))
        return float(cal.get("temperature_fused", cal.get("temperature", 1.0)))
    except FileNotFoundError:
        return 1.0


def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def acc(probs, y):
    return float((probs.argmax(1) == y).mean())


def accuracy_of_view0(z, y):
    return acc(softmax(z[:, 0, :]), y)


def accuracy_fused(z, y):
    return acc(softmax(z.mean(1)), y)


def main():
    data = {}
    for run in RUNS:
        data[run] = {s: load(run, s) for s in SPLITS}
    report = {"per_model": {}, "ensembles": {}}
    # also fill canteen accuracy per model
    for run in RUNS:
        for s in SPLITS:
            z, y = data[run][s]
            report["per_model"].setdefault(SHORT[run], {})[s] = {
                "single": acc(softmax(z[:, 0, :]), y),
                "fused": acc(softmax(z.mean(1)), y),
            }
    print(f"{'model':6} " + " ".join(f"{s:>18}" for s in SPLITS))
    for run in RUNS:
        cells = []
        for s in SPLITS:
            r = report["per_model"][SHORT[run]][s]
            cells.append(f"{100*r['single']:.2f}/{100*r['fused']:.2f}")
        print(f"{SHORT[run]:6} " + " ".join(f"{c:>18}" for c in cells))

    subsets = [("E7", ["resnet18_last_stage_aug_f101n"]),
               ("E7+E8", ["resnet18_last_stage_aug_f101n", "resnet18_last_stage_aug_f101n_nodedup"]),
               ("E4+E7", ["resnet18_last_stage_aug", "resnet18_last_stage_aug_f101n"]),
               ("E7+E12+E13", ["resnet18_last_stage_aug_f101n",
                               "resnet18_last_stage_aug_f101n_e24",
                               "resnet18_last_stage_aug_seed7"]),
               ("E4+E7+E8", ["resnet18_last_stage_aug", "resnet18_last_stage_aug_f101n",
                             "resnet18_last_stage_aug_f101n_nodedup"]),
               ("all5", RUNS)]
    print()
    print(f"{'ensemble':12} {'variant':7} " + " ".join(f"{s:>8}" for s in SPLITS))
    for name, runs in subsets:
        for variant in ("prob", "logitT", "logit"):
            probs = []
            ys = {}
            for s in SPLITS:
                stacked = []
                for run in runs:
                    z, y = data[run][s]
                    if variant == "prob":
                        stacked.append(softmax(z.mean(1) / temp_of(run)))
                    elif variant == "logitT":
                        stacked.append(z.mean(1) / temp_of(run))
                    else:
                        stacked.append(z.mean(1))
                    ys[s] = y
                m = np.mean(stacked, axis=0)
                p = m if variant == "prob" else softmax(m)
                report["ensembles"].setdefault(name, {}).setdefault(variant, {})[s] = acc(p, ys[s])
        row = report["ensembles"][name]
        for variant in ("prob", "logitT", "logit"):
            cells = " ".join(f"{100*row[variant][s]:8.2f}" for s in SPLITS)
            print(f"{name:12} {variant:7} {cells}")
    out = ROOT / "results/tables/ensemble.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
