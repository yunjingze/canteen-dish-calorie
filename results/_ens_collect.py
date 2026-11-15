"""Zero-cost ensembling study: average the five checkpoints we already have.

Single view = the training-time framing; fused = the app's six views averaged in
logit space.  Everything is cached to results/_ens_<run>_<split>.npz so the
analysis can be re-run without another GPU pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from foodcls.utils import pick_device  # noqa: E402

RUNS = [
    "resnet18_last_stage_aug",             # E4  baseline
    "resnet18_last_stage_aug_f101n",       # E7  +Food-101N
    "resnet18_last_stage_aug_f101n_nodedup",  # E8
    "resnet18_last_stage_aug_f101n_e24",   # E12 longer
    "resnet18_last_stage_aug_seed7",       # E13 reseed
]
SPLITS = ["test", "canteen", "canteen_s7"]
NORM = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def _views():
    def view(resize_to=None, square=False, flip=False):
        ops = ([transforms.Resize((IMG_SIZE, IMG_SIZE))] if square
               else [transforms.Resize(int(IMG_SIZE * resize_to / 224)), transforms.CenterCrop(IMG_SIZE)])
        mirror = [transforms.RandomHorizontalFlip(p=1.0)] if flip else []
        return transforms.Compose(ops + mirror + NORM)
    return [view(256), view(square=True), view(320),
            view(256, flip=True), view(square=True, flip=True), view(320, flip=True)]


VIEWS = _views()


class MultiView(Dataset):
    def __init__(self, root: Path, classes):
        self.items = []
        for ci, c in enumerate(classes):
            for p in sorted((root / c).glob("*.jpg")):
                self.items.append((p, ci))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        return torch.stack([tf(img) for tf in VIEWS]), label, str(path)


def cache_path(run, split):
    return ROOT / "results" / f"_ens_{run}__{split}.npz"


def collect(run, split, device, bs=16):
    out = cache_path(run, split)
    if out.exists():
        return
    ckpt = torch.load(ROOT / "results/runs" / run / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    ds = MultiView(ROOT / "data/splits" / split, CLASS_NAMES)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    logits, labels, paths = [], [], []
    with torch.no_grad():
        for x, y, p in dl:
            n, v = x.shape[:2]
            z = model(x.view(n * v, *x.shape[2:]).to(device, non_blocking=True))
            logits.append(z.float().cpu().view(n, v, -1).numpy())
            labels += y.tolist()
            paths += list(p)
    np.savez_compressed(out, logits=np.concatenate(logits).astype(np.float32),
                        labels=np.array(labels), paths=np.array(paths))
    print("cached", run, split, flush=True)


def main():
    device = pick_device()
    for run in RUNS:
        for split in SPLITS:
            collect(run, split, device)
    print("done caching", flush=True)


if __name__ == "__main__":
    main()
