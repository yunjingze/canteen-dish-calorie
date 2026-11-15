"""Measure the open-set gate: can the demo tell "not one of the 20 dishes"?

The 20-class softmax always names something, so a photo of a whole roast chicken
comes back as "烤三文鱼 49 %".  The gate scores three crops of the same photo
(centre, squashed full frame, wide crop) and keeps the *worst* top-1 probability:
a supported dish stays confident in every framing, an unsupported one does not.

This script reports the false-alarm rate on supported photos (val + canteen
proxy) and the detection rate on 20 Food-101 classes that the model never saw,
then draws results/figures/openset_gate.png and writes
results/tables/openset_gate.json.

    python scripts/eval_openset.py            # ~3 min on a GPU
    python scripts/eval_openset.py --cached   # reuse results/tables/openset_gate.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.app import _state, load_model, read_image  # noqa: E402
from foodcls.classes import CLASS_NAMES  # noqa: E402

CACHE = ROOT / "results/tables/openset_gate.npz"
CHICKEN = ROOT / "results/figures/roast_chicken_example.png"
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
VIEWS = [
    ("center", T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), NORM])),
    ("full", T.Compose([T.Resize((224, 224)), T.ToTensor(), NORM])),
    ("wide", T.Compose([T.Resize(288), T.CenterCrop(224), T.ToTensor(), NORM])),
]


@torch.no_grad()
def view_probs(img) -> np.ndarray:
    model, dev = _state["model"], _state["device"]
    temp = max(float(_state.get("temperature", 1.0)), 1e-3)
    batch = torch.cat([tf(img.convert("RGB")).unsqueeze(0) for _, tf in VIEWS]).to(dev)
    return torch.softmax(model(batch).float() / temp, dim=1).cpu().numpy()


def probe(images) -> dict:
    out = {"center": [], "min_view": [], "agree": []}
    for img in images:
        p = view_probs(img)
        out["center"].append(float(p[0].max()))
        out["min_view"].append(float(p.max(axis=1).min()))
        out["agree"].append(float((p.argmax(axis=1) == p.argmax(axis=1)[0]).mean()))
    return {k: np.array(v) for k, v in out.items()}


def collect() -> dict:
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}

    def pick(folder, per):
        return [f for c in CLASS_NAMES for f in sorted((folder / c).glob("*.jpg"))[:per]]

    val = pick(ROOT / "data/splits/val", 20)
    can = pick(ROOT / "data/splits/canteen_clean", 25)
    print(f"scoring {len(val)} val + {len(can)} canteen photos ...", flush=True)
    s_in = probe([read_image(f.read_bytes()) for f in val + can])

    import pyarrow.parquet as pq
    shards = sorted((ROOT / "data/raw_hf").glob("validation-*.parquet"))
    meta = json.loads(pq.ParquetFile(shards[0]).metadata.metadata[b"huggingface"])
    names = meta["info"]["features"]["label"]["names"]
    ood_ids = [i for i, n in enumerate(names) if n not in CLASS_NAMES][:20]
    want = {i: 15 for i in ood_ids}
    blobs = []
    for shard in shards:
        t = pq.read_table(shard, columns=["image", "label"])
        for lab, blob in zip(t["label"].to_pylist(), t["image"].combine_chunks().field("bytes").to_pylist()):
            if lab in want and want[lab] > 0:
                want[lab] -= 1
                blobs.append(blob)
        if all(v == 0 for v in want.values()):
            break
    print(f"scoring {len(blobs)} unseen-dish photos ...", flush=True)
    s_ood = probe([read_image(b) for b in blobs])

    d = {f"in_{k}": v for k, v in s_in.items()}
    d.update({f"ood_{k}": v for k, v in s_ood.items()})
    np.savez(CACHE, **d)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.32)
    args = ap.parse_args()
    load_model()
    d = collect()

    fig_rows = {}
    for key in ("center", "min_view"):
        fin, fout = d[f"in_{key}"], d[f"ood_{key}"]
        auc = roc_auc_score(np.r_[np.ones(len(fin)), np.zeros(len(fout))], np.r_[fin, fout])
        fpr = float((fin < args.threshold).mean())
        tpr = float((fout < args.threshold).mean())
        fig_rows[key] = {"auc": round(float(auc), 3), "fpr": round(fpr, 3), "tpr": round(tpr, 3)}
        print(f"{key:9s} AUC {auc:.3f}  FPR {fpr:.1%}  TPR {tpr:.1%}", flush=True)

    chicken = None
    if CHICKEN.exists():
        p = view_probs(read_image(CHICKEN.read_bytes()))
        chicken = {"center": float(p[0].max()), "min_view": float(p.max(axis=1).min())}
        print(f"roast chicken: centre view {chicken['center']:.3f}, "
              f"worst view {chicken['min_view']:.3f}", flush=True)

    metrics = {"threshold": args.threshold, "views": [n for n, _ in VIEWS],
               "scores": fig_rows, "chicken": chicken,
               "n_in": int(len(d["in_center"])), "n_ood": int(len(d["ood_center"]))}
    (ROOT / "results/tables/openset_gate.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 2.9), constrained_layout=True)
    bins = np.linspace(0, 1, 40)
    ax.hist(d["in_min_view"], bins=bins, density=True, alpha=.62, color="#2f6df6", label="supported dish (40 dishes, 900 photos)")
    ax.hist(d["ood_min_view"], bins=bins, density=True, alpha=.62, color="#e0562a", label="unseen dish (20 other classes, 300 photos)")
    ax.set_yscale("log")
    ax.set_ylim(0.02, 30)
    ax.axvline(args.threshold, color="#1b2430", ls="--", lw=1)
    if chicken:
        ax.axvline(chicken["min_view"], color="#7a5af8", lw=1.4)
        ax.annotate(f"roast chicken\n{chicken['min_view']:.2f}", (chicken["min_view"], 9),
                    textcoords="offset points", xytext=(-7, 0), fontsize=7, color="#5a34d6", va="center", ha="right")
    ax.set_xlabel("worst top-1 probability over three crops")
    ax.set_ylabel("density")
    ax.set_title(f"open-set gate: FPR {fig_rows['min_view']['fpr']:.0%} · TPR {fig_rows['min_view']['tpr']:.0%}", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(ROOT / "results/figures/openset_gate.png", dpi=200)
    print("wrote results/tables/openset_gate.json and results/figures/openset_gate.png", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
