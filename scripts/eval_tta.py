"""Test-time augmentation for a trained checkpoint.

Averages softmax probabilities over a small set of deterministic views and
reports accuracy/macro-F1/ECE on the official test split and on the canteen
proxy, next to the single-view baseline.

Usage:
    python -m scripts.eval_tta --ckpt results/runs/<run>/best.pt --out results/tables/tta.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, SplitDataset  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from foodcls.utils import (  # noqa: E402
    classification_metrics,
    expected_calibration_error,
    pick_device,
    save_json,
)


def views() -> dict:
    """name -> list of (resize_to, flip, five_crop) deterministic views."""
    def one(resize_to, flip=False, five=False):
        return (int(IMG_SIZE * resize_to / 224), flip, five)

    return {
        "base": [one(256)],
        "hflip": [one(256), one(256, flip=True)],
        "scales": [one(256), one(288), one(320)],
        "scales+hflip": [one(256), one(288), one(320),
                         one(256, True), one(288, True), one(320, True)],
        "fivecrop": [one(256, five=True)],
        "scales+hflip+five": [one(256), one(288), one(320),
                              one(256, True), one(288, True), one(320, True),
                              one(256, five=True)],
    }


class FiveView:
    """Five crops stacked into a (5, C, H, W) tensor."""

    def __init__(self, resize_to: int, flip: bool):
        self.resize = transforms.Resize(resize_to)
        self.crop = transforms.FiveCrop(IMG_SIZE)
        ops = [transforms.RandomHorizontalFlip(p=1.0)] if flip else []
        ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        self.post = transforms.Compose(ops)

    def __call__(self, img):
        return torch.stack([self.post(c) for c in self.crop(self.resize(img))])


def build_tf(resize_to: int, flip: bool, five: bool):
    if five:
        return FiveView(resize_to, flip)
    ops = [transforms.Resize(resize_to), transforms.CenterCrop(IMG_SIZE)]
    if flip:
        ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


def five_crop_forward(model, x, device):
    """x: (B, 5, C, H, W) -> averaged logits."""
    b, n = x.shape[:2]
    flat = x.reshape(b * n, *x.shape[2:]).to(device, non_blocking=True)
    logits = model(flat)
    return logits.reshape(b, n, -1).mean(1)


@torch.no_grad()
def predict_views(model, dataset_root, spec, batch_size, workers, device):
    probs = []
    labels = None
    for resize_to, flip, five in spec:
        ds = SplitDataset(dataset_root, CLASS_NAMES, build_tf(resize_to, flip, five))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=True)
        chunks, lab_chunks = [], []
        for x, y in loader:
            if five:
                logits = five_crop_forward(model, x, device)
            else:
                logits = model(x.to(device, non_blocking=True))
            chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            lab_chunks.append(y.numpy())
        probs.append(np.concatenate(chunks))
        if labels is None:
            labels = np.concatenate(lab_chunks)
    return labels, np.mean(probs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="checkpoint path; repeatable")
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--splits", nargs="+", default=["test", "canteen"])
    ap.add_argument("--out", default=str(ROOT / "results/tables/tta.json"))
    args = ap.parse_args()

    device = pick_device()
    results = {}
    for ckpt_path in args.ckpt:
        ckpt_path = Path(ckpt_path).resolve()
        run = ckpt_path.parent.name
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(ckpt["arch"], len(CLASS_NAMES), pretrained=False).to(device).eval()
        model.load_state_dict(ckpt["model"])
        print(f"[{run}] arch={ckpt['arch']}")
        results[run] = {"arch": ckpt["arch"], "splits": {}}
        for split in args.splits:
            root = Path(args.data_root) / "splits" / split
            entry = {}
            for name, spec in views().items():
                labels, probs = predict_views(model, root, spec, args.batch_size, args.workers, device)
                met = classification_metrics(labels, probs, CLASS_NAMES)
                met["ece"] = expected_calibration_error(labels, probs)
                met.pop("per_class", None)
                met.pop("confusion_matrix", None)
                entry[name] = met
                print(f"  {split:8s} {name:18s} acc={100*met['accuracy']:.2f} "
                      f"f1={100*met['macro_f1']:.2f} ece={met['ece']:.3f}")
            results[run]["splits"][split] = entry
    save_json(args.out, results)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
