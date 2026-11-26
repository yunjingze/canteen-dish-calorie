"""Fit one temperature for the five-model logit average (the deployed ensemble).

Writes results/tables/ensemble_calibration.json.  The ensemble repeats the
deployment path -- six views, average the logits, one temperature afterwards --
so it needs its own scalar; the per-model temperatures cancel out of a logit
average and must not be applied before it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.data import SplitDataset  # noqa: E402
from foodcls.classes import CLASS_NAMES  # noqa: E402

RUNS = [
    "resnet18_last_stage_aug",
    "resnet18_last_stage_aug_f101n",
    "resnet18_last_stage_aug_f101n_nodedup",
    "resnet18_last_stage_aug_f101n_e24",
    "resnet18_last_stage_aug_seed7",
]
SPLITS = ["val", "test", "canteen", "canteen_s7"]
OUT = ROOT / "results/tables/ensemble_calibration.json"


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def ece(p, y, bins=15):
    conf, hit = p.max(1), (p.argmax(1) == y)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += float(m.mean()) * abs(float(hit[m].mean()) - float(conf[m].mean()))
    return float(total)


def fit_temperature(z, y):
    """Temperature minimising NLL on the fused logits."""
    best = (0.0, 1.0)
    for t in np.arange(0.30, 2.51, 0.01):
        p = softmax(z / t)
        nll = -float(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())
        if best[0] == 0.0 or nll < best[0]:
            best = (nll, float(t))
    return best[1]


def load(run, split):
    p = ROOT / "results" / f"_ens_{run}__{split}.npz"
    if not p.is_file():
        return None, None
    d = np.load(p)
    return d["logits"].astype(np.float64), d["labels"]


def main():
    fused = {}
    for split in SPLITS:
        stack, labels = [], None
        for run in RUNS:
            z, y = load(run, split)
            if z is None:
                print(f"skip {split}: {run} has no cache", flush=True)
                break
            stack.append(z.mean(1))
            labels = y
        else:
            fused[split] = (np.mean(stack, axis=0), labels)
    if "val" not in fused:
        raise SystemExit("no val cache; cannot fit a temperature")
    z_val, y_val = fused["val"]
    temperature = fit_temperature(z_val, y_val)
    report = {"runs": RUNS, "temperature": round(temperature, 4), "splits": {}}
    print(f"fitted temperature {temperature:.2f} on {len(y_val)} val images", flush=True)
    for split, (z, y) in fused.items():
        row = {}
        for tag, t in (("raw", 1.0), ("calibrated", temperature)):
            p = softmax(z / t)
            row[tag] = {"n": int(len(y)), "accuracy": round(float((p.argmax(1) == y).mean()), 4),
                        "ece": round(ece(p, y), 4)}
        report["splits"][split] = row
        print(f"  {split:11s} n {row['raw']['n']:5d} acc {200*row['raw']['accuracy']:.2f}"
              f"  ece {100*row['raw']['ece']:.2f} -> {100*row['calibrated']['ece']:.2f}", flush=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("wrote", OUT.relative_to(ROOT), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())