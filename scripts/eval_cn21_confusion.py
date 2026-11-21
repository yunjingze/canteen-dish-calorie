# -*- coding: utf-8 -*-
"""Confusion matrix of the deployed 21-way canteen head on its own test split.

The paper's canteen section needs the same artefact the Food-101 section has: a
row-normalised confusion matrix computed by the deployed decision path (six-view
fusion + temperature) rather than re-derived by hand.  Writes
results/tables/cn241_21_confusion.json plus results/runs/cn241_21_pre/pred_test.npz
so the figure can be redrawn without a second GPU pass.

Run: python scripts/eval_cn21_confusion.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls import app as A  # noqa: E402

TEST = ROOT / "data/splits/cn241_21/test"
OUT = ROOT / "results/tables/cn241_21_confusion.json"
NPZ = ROOT / "results/runs/cn241_21_pre/pred_test.npz"


def class_names() -> list[str]:
    ckpt = torch.load(A.CN_CKPT, map_location="cpu", weights_only=False)
    names = list(ckpt.get("class_names") or A.CN_CLASS_NAMES)
    if len(names) == A.CN_NUM_CLASSES:
        names = names + [A.CN_OTHER_SLUG]
    return names


def main() -> int:
    A._state["device"] = A.pick_device()
    A.load_cn_model()
    names = class_names()
    print(f"{len(names)} classes, device {A._state['device']}", flush=True)
    paths, labels = [], []
    for idx, slug in enumerate(names):
        files = sorted((TEST / slug).glob("*.jpg"))
        if not files:
            raise SystemExit(f"no test images for {slug}")
        paths.extend(files)
        labels.extend([idx] * len(files))
    print(f"{len(paths)} test images", flush=True)
    probs = np.zeros((len(paths), len(names)), dtype=np.float32)
    views_all = np.zeros((len(paths), 6, len(names)), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i, p in enumerate(paths):
            img = Image.open(p).convert("RGB")
            views, fused = A.analyse_cn(img)
            probs[i] = np.asarray(fused, dtype=np.float32)
            views_all[i] = np.asarray(views, dtype=np.float32)
            if (i + 1) % 400 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{len(paths)} {(i+1)/el:.1f} img/s", flush=True)
    y = np.asarray(labels)
    top = probs.argmax(1)
    cm = np.zeros((len(names), len(names)), dtype=int)
    for t, p in zip(y, top):
        cm[t, p] += 1
    per = {names[i]: round(float((top[y == i] == i).mean()), 4)
           for i in range(len(names)) if (y == i).any()}
    order = np.argsort(-probs, axis=1)[:, :5]
    single = views_all[:, 0].argmax(1)
    res = {
        "n": int(len(paths)),
        "top1": round(float((top == y).mean()), 4),
        "top5": round(float(np.mean([y[i] in order[i] for i in range(len(y))])), 4),
        "top1_single_view": round(float((single == y).mean()), 4),
        "top5_single_view": round(float(np.mean(
            [y[i] in np.argsort(-views_all[i, 0])[:5] for i in range(len(y))])), 4),
        "classes": names,
        "confusion": cm.tolist(),
        "per_class": per,
    }
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), "utf-8")
    NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ, probs=probs, views=views_all, labels=y,
                        paths=np.array([str(p.relative_to(ROOT)).replace("\\", "/") for p in paths]))
    print(json.dumps({"n": res["n"], "top1": res["top1"], "top5": res["top5"],
                      "top1_single_view": res["top1_single_view"],
                      "top5_single_view": res["top5_single_view"]}), flush=True)
    pairs: dict[tuple[str, str], int] = {}
    for t, p in zip(y, top):
        if t != p:
            pairs[(names[t], names[p])] = pairs.get((names[t], names[p]), 0) + 1
    for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {a:20s} -> {b:20s} {n}", flush=True)
    print("wrote", OUT.relative_to(ROOT), NPZ.relative_to(ROOT), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
