"""Sweep thresholds for the 40-class western+canteen plate model.

Rebuilds the same deterministic validation plates the trainer uses (600 western
+ 600 canteen), sweeps the detection threshold, fits per-class cut-offs, and
reports the exact-match / parts-per-plate numbers the README quotes.  Writes
results/tables/mixed_plate40.json and results/figures/mixed_plate40.png.

    python scripts/eval_mixed_plate.py --ckpt results/runs/resnet18_multilabel_plate40/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from foodcls.classes import CLASS_NAMES                      # noqa: E402
from foodcls.classes_cn import CN_CLASS_NAMES                # noqa: E402
from foodcls.data import SplitDataset                        # noqa: E402
from foodcls.mixed import build_eval_set, detect_parts        # noqa: E402
from foodcls.models import build_model                        # noqa: E402
from foodcls.plate_classes import (                           # noqa: E402
    PLATE_CLASS_NAMES,
    PLATE_KCAL_PER_100G,
    PLATE_PORTION_G,
    PLATE_ZH,
)
from foodcls.utils import pick_device                         # noqa: E402
from train_mixed import (                                     # noqa: E402
    PlateEvalSet,
    SampleList,
    component_metrics,
    per_class_thresholds,
    real_canteen_samples,
)
from foodcls.data import build_transforms                     # noqa: E402
from torch.utils.data import DataLoader                        # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "results/runs/resnet18_multilabel_plate40/best.pt"))
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--n-west", type=int, default=600)
    p.add_argument("--n-cn", type=int, default=600)
    p.add_argument("--lo", type=float, default=0.05)
    p.add_argument("--hi", type=float, default=0.90)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--out", default=str(ROOT / "results/tables/mixed_plate40.json"))
    p.add_argument("--figure", default=str(ROOT / "results/figures/mixed_plate40.png"))
    return p.parse_args(argv)


@torch.no_grad()
def probs_for(model, plates, device, batch_size=64, num_classes=None):
    loader = DataLoader(PlateEvalSet(plates, num_classes=num_classes), batch_size=batch_size,
                        shuffle=False, num_workers=2)
    out = []
    for x, _ in loader:
        x = x.to(device)
        n = x.shape[0]
        probs = torch.sigmoid(model(torch.cat([x, torch.flip(x, dims=[3])], dim=0)).float())
        probs = torch.maximum(probs[:n], probs[n:])
        out.append(probs.cpu().numpy())
    return np.concatenate(out, axis=0)


def targets_for(plates, num_classes):
    t = np.zeros((len(plates), num_classes), dtype=np.float32)
    for i, (_, labels) in enumerate(plates):
        t[i, labels] = 1.0
    return t


def main(argv=None):
    args = parse_args(argv)
    device = pick_device()
    ckpt = torch.load(Path(args.ckpt), map_location=device, weights_only=False)
    names = list(ckpt.get("class_names") or PLATE_CLASS_NAMES)
    if len(names) != len(PLATE_CLASS_NAMES):
        raise SystemExit(f"checkpoint has {len(names)} classes, expected {len(PLATE_CLASS_NAMES)}")
    model = build_model(ckpt["arch"], len(names), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    print(f"ckpt {Path(args.ckpt).name} epoch {ckpt.get('epoch')} classes {len(names)}", flush=True)

    root = Path(args.data_root)
    offset = len(CLASS_NAMES)
    west_src = SplitDataset(root / "splits/val", CLASS_NAMES, transform=None)
    cn_src = SplitDataset(root / "splits/cn241_21/val", CN_CLASS_NAMES, transform=None)
    west = build_eval_set(west_src.samples, args.n_west, seed=1234, num_classes=offset)
    cn = build_eval_set(
        cn_src.samples,
        args.n_cn,
        seed=4321,
        num_classes=len(CN_CLASS_NAMES),
        label_map=[offset + i for i in range(len(CN_CLASS_NAMES))],
    )

    wp = probs_for(model, west, device, num_classes=len(names))
    cp = probs_for(model, cn, device, num_classes=len(names))
    wt = targets_for(west, len(names))
    ct = targets_for(cn, len(names))

    grid = np.round(np.arange(args.lo, args.hi + 1e-9, args.step), 3)
    rows = []
    for thr in grid:
        rows.append((float(thr), component_metrics(wt, wp, thr, class_names=names),
                     component_metrics(ct, cp, thr, class_names=names)))
    best = max(rows, key=lambda r: (r[1]["f1"], r[1]["exact_match"], r[0]))

    print(f"{'thr':>5} {'wP':>6} {'wR':>6} {'wF1':>6} {'wExact':>7} {'cP':>6} {'cR':>6} {'cF1':>6} {'cExact':>7}")
    for thr, m, c in rows:
        mark = " <-" if thr == best[0] else ""
        print(f"{thr:5.2f} {m['precision']*100:6.2f} {m['recall']*100:6.2f} {m['f1']*100:6.2f} "
              f"{m['exact_match']*100:7.2f} {c['precision']*100:6.2f} {c['recall']*100:6.2f} "
              f"{c['f1']*100:6.2f} {c['exact_match']*100:7.2f}{mark}")

    thr, m, c = best
    # each dish only appears on one cuisine's plates, so its cut-off is fitted
    # there; the other cuisine's column is all zeros and would drag it to 0.10
    class_thr = per_class_thresholds(wt, wp, class_names=names)
    class_thr.update(per_class_thresholds(ct, cp, class_names=names[offset:]))
    m_t = component_metrics(wt, wp, thr, class_thresholds=class_thr, class_names=names)
    c_t = component_metrics(ct, cp, thr, class_thresholds=class_thr, class_names=names)
    print(f"\nper-class cut-offs ({len(class_thr)} classes, {min(class_thr.values()):.2f}-{max(class_thr.values()):.2f}): "
          f"wF1 {m_t['f1']*100:.2f} wExact {m_t['exact_match']*100:.2f} "
          f"cF1 {c_t['f1']*100:.2f} cExact {c_t['exact_match']*100:.2f}")

    table = {"western": {}, "canteen": {}, "per_class": {}}
    for tag, probs, tgts, plates in (("western", wp, wt, west), ("canteen", cp, ct, cn)):
        glob = component_metrics(tgts, probs, thr, class_names=names)
        tuned = component_metrics(tgts, probs, thr, class_thresholds=class_thr, class_names=names)
        hits = 0
        empties = 0
        for row, (_, labels) in zip(probs, plates):
            picked = {p["class"] for p in detect_parts(row, threshold=thr, class_thresholds=class_thr, class_names=names,
                                                        zh_table=PLATE_ZH, kcal_table=PLATE_KCAL_PER_100G,
                                                        portion_table=PLATE_PORTION_G)}
            hits += picked == {names[i] for i in labels}
            empties += not picked
        table[tag] = {
            "global": glob,
            "per_class": tuned,
            "exact_match_detect_parts": hits / len(plates),
            "empty_plates": empties / len(plates),
        }
        print(f"{tag:8s} global F1 {glob['f1']*100:.2f} P {glob['precision']*100:.2f} R {glob['recall']*100:.2f} "
              f"exact {glob['exact_match']*100:.2f} parts {glob['parts_per_plate']:.2f} | "
              f"per-class F1 {tuned['f1']*100:.2f} exact {tuned['exact_match']*100:.2f} | "
              f"detect_parts exact {100*hits/len(plates):.2f} empty {100*empties/len(plates):.1f}")

    # single-dish photos: does adding 20 canteen rows cost the western classes?
    singles = {}
    real = real_canteen_samples(root)
    cn_test = SplitDataset(root / "splits/cn241_21/test", CN_CLASS_NAMES, transform=None)
    for tag, samples, shift in (("western_real", real, 0), ("canteen_test", cn_test.samples, offset)):
        if not samples:
            continue
        ds = SampleList(samples, build_transforms(False, False))
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
        preds, truths = [], []
        with torch.no_grad():
            for x, y in loader:
                preds.append(model(x.to(device)).argmax(1).cpu().numpy())
                truths.append(y.numpy() + shift)
        preds, truths = np.concatenate(preds), np.concatenate(truths)
        top1 = float((preds == truths).mean())
        top_cn = float((preds >= offset).mean())
        singles[tag] = {"n": int(len(truths)), "top1": top1, "share_canteen_pred": top_cn}
        print(f"{tag:13s} n {len(truths):5d} top-1 {top1*100:6.2f}% (canteen class predicted {top_cn*100:.1f}%)")

    out = {
        "singles": singles,
        "checkpoint": str(Path(args.ckpt)),
        "epoch": ckpt.get("epoch"),
        "num_classes": len(names),
        "threshold": thr,
        "class_thresholds": class_thr,
        "checkpoint_threshold": ckpt.get("threshold"),
        "plates": {"western": len(west), "canteen": len(cn)},
        "sweep": [{"threshold": r[0], **{f"w_{k}": v for k, v in r[1].items()},
                   **{f"c_{k}": v for k, v in r[2].items()}} for r in rows],
        **table,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot([r[0] for r in rows], [r[1]["f1"] * 100 for r in rows], marker="o", label="western plates")
        ax.plot([r[0] for r in rows], [r[2]["f1"] * 100 for r in rows], marker="s", label="canteen plates")
        ax.axvline(thr, color="grey", ls="--", lw=1, label=f"chosen {thr:.2f}")
        ax.set_xlabel("detection threshold")
        ax.set_ylabel("component F1 (%)")
        ax.set_title("40-class plate model: threshold sweep")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=150)
        print(f"wrote {args.figure}", flush=True)
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"skip figure: {exc}", flush=True)


if __name__ == "__main__":
    main()
