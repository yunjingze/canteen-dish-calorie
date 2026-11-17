# -*- coding: utf-8 -*-
"""Does the 20-dish model know when it is looking at something else?

A closed 20-class head always returns one of its 20 labels.  The demo needs a
"not on the menu" escape hatch, so measure the confidence distribution on
in-list vs out-of-list food and pick a scope threshold from it.

    python scripts/eval_cn241_scope.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from foodcls.utils import pick_device  # noqa: E402

CKPT = ROOT / "results/runs/cn241_last_stage/best.pt"
IN_LIST = ROOT / "data/splits/cn241/test"
OUT_LIST = ROOT / "data/wide/splits/test"


def view(path):
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 256 / 224)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    with open(path, "rb") as fh:
        return tf(Image.open(fh).convert("RGB"))


def probs_for(model, device, paths, batch=16):
    out = []
    for start in range(0, len(paths), batch):
        x = torch.stack([view(p) for p in paths[start:start + batch]]).to(device)
        with torch.no_grad():
            out.append(torch.softmax(model(x).float(), 1).cpu().numpy())
    return np.concatenate(out)


def main():
    device = pick_device()
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    names = ckpt["class_names"]
    model = build_model(ckpt["arch"], len(names), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    in_paths = [p for c in names for p in sorted((IN_LIST / c).iterdir())][:600]
    rng = random.Random(0)
    out_paths = []
    for cls_dir in sorted(OUT_LIST.iterdir()):
        if cls_dir.is_dir():
            files = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() == ".jpg")
            out_paths += rng.sample(files, min(6, len(files)))
    out_paths = out_paths[:600]
    print(f"in-list {len(in_paths)} / out-of-list {len(out_paths)} from {len(list(OUT_LIST.iterdir()))} classes", flush=True)

    res = {}
    for tag, paths in (("in_list", in_paths), ("out_of_list", out_paths)):
        p = probs_for(model, device, paths)
        top = p.max(1)
        res[tag] = {"mean_top1_prob": round(float(top.mean()), 4),
                    "p50": round(float(np.median(top)), 4),
                    "share_above_0.5": round(float((top > 0.5).mean()), 4),
                    "share_above_0.8": round(float((top > 0.8).mean()), 4),
                    "n": int(len(top))}
        print(tag, res[tag], flush=True)

    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        keep_in = float((probs_for(model, device, in_paths).max(1) > thr).mean())
        keep_out = float((probs_for(model, device, out_paths).max(1) > thr).mean())
        print(f"thr {thr:.1f}: keep in-list {keep_in*100:.1f}% | accept out-of-list {keep_out*100:.1f}%", flush=True)
    out = ROOT / "results/tables/cn241_scope.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
