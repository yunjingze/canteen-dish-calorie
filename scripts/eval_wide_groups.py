"""Measure the wide model on the test split in one parallel pass.

train_wide.py evaluates the test split twice by decoding every JPEG in the
main process (single view, then three views), which on Windows takes close to an
hour for 25k images.  This script decodes each image once inside DataLoader
workers, stacks the three demo views, and writes the same results.json layout,
plus per-group (Food-101 vs external Chinese dishes) and per-class accuracy.

Usage:
    python scripts/eval_wide_groups.py --run results/runs/resnet18_food101_cn116
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes_wide import WIDE_CLASS_NAMES, WIDE_NUM_CLASSES, WIDE_ZH  # noqa: E402
from foodcls.data import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, SplitDataset  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from foodcls.utils import expected_calibration_error  # noqa: E402

F101 = 101  # first rows of the head: original Food-101 classes


def views():
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return [
        transforms.Compose([transforms.Resize(int(IMG_SIZE * 256 / 224)), transforms.CenterCrop(IMG_SIZE)] + norm),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))] + norm),
        transforms.Compose([transforms.Resize(int(IMG_SIZE * 320 / 224)), transforms.CenterCrop(IMG_SIZE)] + norm),
    ]


class Views(Dataset):
    def __init__(self, base: SplitDataset):
        self.base = base
        self.tfs = views()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        return torch.stack([tf(img) for tf in self.tfs]), label


@torch.no_grad()
def score(model, loader, device):
    logits, labels = [], []
    for x, y in loader:
        b = x.shape[0]
        out = model(x.to(device, non_blocking=True).flatten(0, 1)).float().view(b, 3, -1)
        logits.append(out.cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), np.concatenate(labels)


def topk(probs, y):
    order = probs.argsort(axis=1)[:, ::-1][:, :5]
    return {"top1": float((order[:, 0] == y).mean()),
            "top5": float((order == y[:, None]).any(axis=1).mean()),
            "n": int(y.size)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/runs/resnet18_food101_cn116")
    ap.add_argument("--split", default="data/wide/splits/test")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    run = (ROOT / args.run).resolve()
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ckpt.get("arch", "resnet18"), WIDE_NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    temperature = json.loads((run / "calibration.json").read_text())["temperature"]

    ds = SplitDataset(ROOT / args.split, WIDE_CLASS_NAMES, transform=None)
    loader = DataLoader(Views(ds), batch_size=96, shuffle=False, num_workers=args.workers,
                        pin_memory=True, persistent_workers=args.workers > 0)
    logits, y = score(model, loader, device)

    p0 = torch.softmax(torch.tensor(logits[:, 0]), 1).numpy()
    pm = torch.softmax(torch.tensor(logits.mean(1) / temperature), 1).numpy()
    single, tta = topk(p0, y), topk(pm, y)
    single["tta"], tta["tta"] = False, True

    pred = pm.argmax(1)
    per_class = {}
    for k, name in enumerate(WIDE_CLASS_NAMES):
        mask = y == k
        if mask.any():
            per_class[name] = {"zh": WIDE_ZH[name], "n": int(mask.sum()),
                               "top1": round(float((pred[mask] == k).mean()), 4)}
    groups = {}
    for label, ids in (("food101", np.arange(F101)),
                       ("chinese", np.arange(F101, WIDE_NUM_CLASSES))):
        mask = np.isin(y, ids)
        groups[label] = {"n": int(mask.sum()),
                         "top1": float((pred[mask] == y[mask]).mean()) if mask.any() else 0.0}

    history = list(csv.DictReader((run / "history.csv").open(encoding="utf-8")))
    results = {
        "arch": ckpt.get("arch", "resnet18"),
        "num_classes": WIDE_NUM_CLASSES,
        "epochs": len(history),
        "best": {"macro_f1": ckpt.get("val_macro_f1"), "val_acc": ckpt.get("val_acc"),
                 "epoch": ckpt.get("epoch")},
        "test": single, "test_tta": tta,
        "test_groups": groups,
        "temperature": temperature,
        "test_ece": expected_calibration_error(y, p0),
        "test_ece_calibrated": expected_calibration_error(
            y, torch.softmax(torch.tensor(logits[:, 0]) / temperature, 1).numpy()),
        "per_class": per_class,
        "history": history,
    }
    (run / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in results.items() if k not in ("history", "per_class")}, indent=2,
                     ensure_ascii=False), flush=True)
    chinese = {k: v for k, v in per_class.items() if k not in WIDE_CLASS_NAMES[:F101]}
    print("chinese classes:", json.dumps(chinese, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())