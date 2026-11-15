"""Sweep the detection threshold of the multi-label plate model.

The trainer scans 0.20-0.70 once per epoch, but early epochs sit on the lower
bound, which hides where the optimum really is.  This script rebuilds the same
deterministic validation/canteen plates, sweeps a wider grid, prints the table,
and optionally writes the chosen threshold back into the checkpoint so the web
demo uses it.

    python scripts/eval_mixed.py --write
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
from foodcls.data import SplitDataset, build_transforms      # noqa: E402
from foodcls.mixed import build_eval_set, detect_parts       # noqa: E402
from foodcls.models import build_model                       # noqa: E402
from foodcls.utils import pick_device                        # noqa: E402
from train_mixed import (  # noqa: E402
    PlateEvalSet,
    component_metrics,
    evaluate_single,
    per_class_thresholds,
)
from torch.utils.data import DataLoader                       # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "results/runs/resnet18_multilabel_aug/best.pt"))
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--n-val", type=int, default=600)
    p.add_argument("--n-canteen", type=int, default=300)
    p.add_argument("--lo", type=float, default=0.05)
    p.add_argument("--hi", type=float, default=0.90)
    p.add_argument("--step", type=float, default=0.05)
    p.add_argument("--write", action="store_true", help="store the chosen threshold in the checkpoint")
    p.add_argument("--figure", default=str(ROOT / "results/figures/mixed_threshold.png"))
    return p.parse_args(argv)


@torch.no_grad()
def probs_for(model, plates, device, batch_size=64):
    loader = DataLoader(PlateEvalSet(plates), batch_size=batch_size, shuffle=False, num_workers=2)
    out = []
    for x, _ in loader:
        x = x.to(device)
        # flip-TTA, identical to what the web app does at inference time
        n = x.shape[0]
        probs = torch.sigmoid(model(torch.cat([x, torch.flip(x, dims=[3])], dim=0)).float())
        probs = torch.maximum(probs[:n], probs[n:])
        out.append(probs.cpu().numpy())
    return np.concatenate(out, axis=0)


def targets_for(plates):
    t = np.zeros((len(plates), len(CLASS_NAMES)), dtype=np.float32)
    for i, (_, labels) in enumerate(plates):
        t[i, labels] = 1.0
    return t


def main(argv=None):
    args = parse_args(argv)
    device = pick_device()
    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    root = Path(args.data_root)
    val_src = SplitDataset(root / "splits/val", CLASS_NAMES, transform=None)
    canteen_src = SplitDataset(root / "splits/canteen_clean", CLASS_NAMES, transform=None)
    val_plates = build_eval_set(val_src.samples, args.n_val, seed=1234)
    canteen_plates = build_eval_set(canteen_src.samples, args.n_canteen, seed=999)

    vp = probs_for(model, val_plates, device)
    cp = probs_for(model, canteen_plates, device)
    vt, ct = targets_for(val_plates), targets_for(canteen_plates)

    grid = np.round(np.arange(args.lo, args.hi + 1e-9, args.step), 3)
    rows = []
    for thr in grid:
        m = component_metrics(vt, vp, thr)
        c = component_metrics(ct, cp, thr)
        rows.append((float(thr), m, c))
    best = max(rows, key=lambda r: (r[1]["f1"], r[1]["exact_match"], r[0]))
    print(f"{'thr':>5} {'valP':>6} {'valR':>6} {'valF1':>6} {'valExact':>8} {'parts':>6} {'proxyF1':>7}")
    for thr, m, c in rows:
        mark = " <-" if thr == best[0] else ""
        print(f"{thr:5.2f} {m['precision']*100:6.2f} {m['recall']*100:6.2f} {m['f1']*100:6.2f} "
              f"{m['exact_match']*100:8.2f} {m['parts_per_plate']:6.2f} {c['f1']*100:7.2f}{mark}")

    thr, m, c = best
    print(f"\nglobal threshold {thr:.2f}: valF1 {m['f1']*100:.2f} valExact {m['exact_match']*100:.2f} "
          f"proxyF1 {c['f1']*100:.2f} parts/plate {m['parts_per_plate']:.2f}")

    class_thr = per_class_thresholds(vt, vp)
    m_t = component_metrics(vt, vp, thr, class_thresholds=class_thr)
    c_t = component_metrics(ct, cp, thr, class_thresholds=class_thr)
    print(f"per-class cut-offs: valF1 {m_t['f1']*100:.2f} valExact {m_t['exact_match']*100:.2f} "
          f"proxyF1 {c_t['f1']*100:.2f} parts/plate {m_t['parts_per_plate']:.2f}")

    # exact-match through detect_parts (top-3 cap, 10% floor)
    hits = 0
    for probs, (_, labels) in zip(vp, val_plates):
        picked = {p["class"] for p in detect_parts(probs, threshold=thr, class_thresholds=class_thr)}
        hits += picked == {CLASS_NAMES[i] for i in labels}
    print(f"detect_parts exact match: {hits}/{len(val_plates)} = {100*hits/len(val_plates):.1f}%")
    empty = sum(1 for probs in vp if not detect_parts(probs, threshold=thr, class_thresholds=class_thr))
    print(f"plates with no part: {empty}/{len(val_plates)} = {100*empty/len(val_plates):.1f}%")

    out = {
        "threshold": thr,
        "class_thresholds": class_thr,
        "val": {k: m[k] for k in ("precision", "recall", "f1", "exact_match", "parts_per_plate")},
        "canteen": {k: c[k] for k in ("precision", "recall", "f1", "exact_match", "parts_per_plate")},
        "val_tuned": {k: m_t[k] for k in ("precision", "recall", "f1", "exact_match", "parts_per_plate")},
        "canteen_tuned": {k: c_t[k] for k in ("precision", "recall", "f1", "exact_match", "parts_per_plate")},
        "empty_plates": empty,
        "sweep": [{"threshold": r[0], **{k: r[1][k] for k in ("precision", "recall", "f1", "exact_match")}} for r in rows],
    }
    test_src = SplitDataset(root / "splits/test", CLASS_NAMES, build_transforms(False, False))
    single = evaluate_single(model, test_src, device)
    out["test_single"] = single
    print(f"single test top1 {single['top1']*100:.2f} parts/photo {single['parts_per_photo']:.2f}")

    table = ROOT / "results/tables/mixed_threshold.json"
    table.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", table)

    mixed_json = ROOT / "results/tables/mixed.json"
    if mixed_json.exists():
        data = json.loads(mixed_json.read_text(encoding="utf-8"))
        data["threshold"] = thr
        data["threshold_tuned"] = True
        data["class_thresholds"] = class_thr
        data["val_tuned"] = out["val_tuned"]
        data["canteen_tuned"] = out["canteen_tuned"]
        mixed_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print("updated", mixed_json)

    if args.write:
        ckpt["threshold"] = thr
        ckpt["class_thresholds"] = class_thr
        torch.save(ckpt, ckpt_path)
        print(f"stored threshold {thr:.2f} and per-class cut-offs in {ckpt_path}")

    make_figure(rows, thr, Path(args.figure))


def make_figure(rows, chosen, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    import matplotlib as mpl
    mpl.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    thrs = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    ax.plot(thrs, [100 * r[1]["f1"] for r in rows], color="#2f6df6", marker="o", ms=3, label="component F1 (val)")
    ax.plot(thrs, [100 * r[2]["f1"] for r in rows], color="#e0562a", marker="s", ms=3, label="component F1 (canteen proxy)")
    ax.axvline(chosen, color="#9aa4b2", ls="--", lw=1)
    ax.text(chosen + 0.01, 5, f"thr {chosen:.2f}", color="#6b7686", fontsize=8)
    ax.set_xlabel("detection threshold")
    ax.set_ylabel("F1 (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
