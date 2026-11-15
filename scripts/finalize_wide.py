"""Finish the 101-class run after the training script died in its final pass.

The 10 training epochs finished (history.csv, best.pt, calibration.json exist) and
the test numbers were printed, but the last `predict()` call reused six DataLoader
workers and Windows ran out of shared memory ("Couldn't open shared file mapping",
error 1455).  This script redoes only the missing measurement from best.pt, with
workers=0, and writes results.json in the same layout as foodcls.train.

One pass over the test split gives both the single-view and the three-crop numbers,
plus the calibrated ECE, so it costs one decode per image instead of three.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes_wide import WIDE_CLASS_NAMES, WIDE_NUM_CLASSES  # noqa: E402
from foodcls.data import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, SplitDataset  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from foodcls.utils import expected_calibration_error  # noqa: E402


def views():
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return [
        transforms.Compose([transforms.Resize(int(IMG_SIZE * 256 / 224)), transforms.CenterCrop(IMG_SIZE)] + norm),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))] + norm),
        transforms.Compose([transforms.Resize(int(IMG_SIZE * 320 / 224)), transforms.CenterCrop(IMG_SIZE)] + norm),
    ]


def topk_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    order = probs.argsort(axis=1)[:, ::-1][:, :5]
    top1 = float((order[:, 0] == labels).mean())
    top5 = float((order == labels[:, None]).any(axis=1).mean())
    return {"top1": top1, "top5": top5, "n": int(labels.size)}


def main():
    torch.set_num_threads(4)
    run = ROOT / "results/runs/resnet18_food101"
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ckpt.get("arch", "resnet18"), len(ckpt.get("class_names") or WIDE_CLASS_NAMES))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    temperature = json.loads((run / "calibration.json").read_text())["temperature"]

    ds = SplitDataset(ROOT / "data/wide/splits/test", WIDE_CLASS_NAMES, transform=None)
    tfs = views()
    logits_v0, logits_mean, labels = [], [], []

    from PIL import Image

    batch, buf, labs = [], [], []
    with torch.no_grad():
        for path, lab in ds.samples:
            with open(path, "rb") as fh:
                img = Image.open(fh).convert("RGB")
            buf.append(torch.stack([tf(img) for tf in tfs]))
            labs.append(lab)
            if len(buf) < 48:
                continue
            x = torch.stack(buf).to(device)
            out = model(x.flatten(0, 1)).float().view(len(buf), len(tfs), -1)
            logits_v0.append(out[:, 0].cpu())
            logits_mean.append(out.mean(1).cpu())
            labels.append(np.array(labs))
            buf, labs = [], []
        if buf:
            x = torch.stack(buf).to(device)
            out = model(x.flatten(0, 1)).float().view(len(buf), len(tfs), -1)
            logits_v0.append(out[:, 0].cpu())
            logits_mean.append(out.mean(1).cpu())
            labels.append(np.array(labs))

    z0 = torch.cat(logits_v0).numpy()
    zm = torch.cat(logits_mean).numpy()
    y = np.concatenate(labels)
    p0 = torch.softmax(torch.tensor(z0), 1).numpy()
    pm = torch.softmax(torch.tensor(zm / temperature), 1).numpy()
    single = topk_metrics(p0, y)
    tta = topk_metrics(pm, y)
    tta["tta"] = True
    single["tta"] = False

    history = list(csv.DictReader((run / "history.csv").open(encoding="utf-8")))
    last = history[-1]
    results = {
        "arch": ckpt.get("arch", "resnet18"),
        "num_classes": WIDE_NUM_CLASSES,
        "epochs": len(history),
        "best": {"macro_f1": ckpt.get("val_macro_f1"), "val_acc": ckpt.get("val_acc"),
                 "epoch": ckpt.get("epoch")},
        "test": single,
        "test_tta": tta,
        "temperature": temperature,
        "test_ece": expected_calibration_error(y, p0),
        "test_ece_calibrated": expected_calibration_error(
            y, torch.softmax(torch.tensor(z0) / temperature, 1).numpy()),
        "history": history,
        "note": "final pass rebuilt by scripts/finalize_wide.py after a Windows "
                "shared-memory failure in the original DataLoader pass",
    }
    (run / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "history"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
