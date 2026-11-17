"""Cache penultimate features of the canteen backbone for a domain head.

The canteen head already ships; a 512-d linear "is this a canteen dish at all"
probe on its pooled features costs nothing extra to deploy.  This script runs the
gate framings of every photo once and dumps the pooled features, so
scripts/fit_domain_head.py can train without touching the GPU again.

Sets
  cn_tr / cn_va / cn_te   canteen dishes (21 classes incl. the other_cn bucket)
  cn_ood                  29 Chinese classes never seen in training
  f101_tr / f101_va / f101_te   the 20-class western menu (Food-101)
  real                    100 hand-shot canteen photos (western menu in practice)
  mixed_real              9 hand-shot trays
  neg                     non-food photos

Usage
  python scripts/extract_domain_features.py [--views 1] [--cap-train 12000]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls import app as A
from foodcls.models import build_model
from foodcls.utils import pick_device

SETS = [
    ("cn_tr", "data/splits/cn241_21/train", 1),
    ("cn_va", "data/splits/cn241_21/val", 1),
    ("cn_te", "data/splits/cn241_21/test", 1),
    ("cn_ood", "data/splits/cn241_21_ood", 1),
    ("f101_tr", "data/splits/train", 0),
    ("f101_va", "data/splits/val", 0),
    ("f101_te", "data/splits/test", 0),
    ("real", "data/splits/canteen_real", 0),
    ("mixed_real", "data/splits/mixed_real", 0),
    ("neg", "data/splits/negatives", 0),
]
CAP = {"cn_tr": 12000, "f101_tr": 12000}


def collect(rel: str, cap: int | None):
    root = ROOT / rel
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if cap and len(files) > cap:
        step = len(files) / cap
        files = [files[int(i * step)] for i in range(cap)]
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/runs/cn241_21/best.pt")
    ap.add_argument("--out", default="results/tables/domain_feats.npz")
    ap.add_argument("--cap-train", type=int, default=0)
    ap.add_argument("--views", type=int, default=1)
    args = ap.parse_args()

    torch.set_num_threads(4)
    device = pick_device()
    ck = torch.load(ROOT / args.ckpt, map_location=device, weights_only=False)
    n_cls = len(ck["class_names"])
    model = build_model(ck["arch"], n_cls, pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    print(f"loaded {args.ckpt} ({ck['arch']}, {n_cls} classes) on {device}", flush=True)

    store: dict = {}
    model.avgpool.register_forward_hook(lambda _m, _i, out: store.__setitem__("f", out))
    tfs = A._transforms()[: max(1, args.views)]

    feats, labels, sets, paths, classes = [], [], [], [], []
    done = 0
    t0 = time.time()
    with torch.no_grad():
        for name, rel, y in SETS:
            cap = CAP.get(name, 0)
            if args.cap_train and name.endswith("_tr"):
                cap = args.cap_train
            files = collect(rel, cap)
            print(f"{name}: {len(files)} images", flush=True)
            for p in files:
                try:
                    img = Image.open(p).convert("RGB")
                except Exception as exc:  # noqa: BLE001 - one broken file must not stop the sweep
                    print("skip", p.name, exc, flush=True)
                    continue
                x = torch.stack([tf(img) for tf in tfs]).to(device)
                model(x)
                feats.append(store["f"].flatten(1).float().cpu().numpy())
                labels.append(y)
                sets.append(name)
                paths.append(str(p.relative_to(ROOT)).replace("\\", "/"))
                classes.append(p.parent.name)
                done += 1
            print(f"  {name} done, {done} total, {done / max(time.time() - t0, 1e-9):.1f} img/s", flush=True)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        feat=np.stack(feats).astype(np.float32),
        y=np.asarray(labels, dtype=np.int8),
        set=np.asarray(sets),
        path=np.asarray(paths),
        cls=np.asarray(classes),
    )
    (out.with_suffix(".json")).write_text(json.dumps({
        "checkpoint": args.ckpt,
        "views": args.views,
        "n": int(len(labels)),
        "by_set": {s: int(sum(1 for v in sets if v == s)) for s in sorted(set(sets))},
    }, indent=2), encoding="utf-8")
    print("wrote", out, len(labels), "rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
