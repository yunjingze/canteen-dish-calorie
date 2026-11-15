"""Fit a single temperature on the validation split (post-hoc calibration).

Guo et al., "On calibration of modern neural networks", ICML 2017.
The temperature never changes the predicted class, only the reported confidence,
so accuracy and macro-F1 stay identical; ECE changes.

Usage:
    python scripts/fit_temperature.py --run results/runs/resnet18_last_stage_aug
    python scripts/fit_temperature.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.data import SplitDataset, build_transforms  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from foodcls.utils import expected_calibration_error, pick_device, save_json  # noqa: E402

RUNS = [
    "resnet18_frozen_noaug",
    "resnet18_frozen_aug",
    "resnet18_last_stage_noaug",
    "resnet18_last_stage_aug",
    "mobilenet_v3_small_last_stage_aug",
]


@torch.no_grad()
def collect_logits(model, loader, device):
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Minimise validation NLL over a scalar temperature (LBFGS, as in Guo et al.)."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / torch.exp(log_t), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item())


def evaluate(logits: torch.Tensor, labels: torch.Tensor, temperature: float):
    probs = torch.softmax(logits / temperature, dim=1).numpy()
    y = labels.numpy()
    acc = float((probs.argmax(1) == y).mean())
    return {"accuracy": acc, "ece": float(expected_calibration_error(y, probs))}


def run_one(run_dir: Path, data_root: Path, workers: int):
    device = pick_device()
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    data = {}
    for split in ("val", "test", "canteen"):
        ds = SplitDataset(data_root / "splits" / split, CLASS_NAMES, build_transforms(False, False))
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=workers, pin_memory=True)
        data[split] = collect_logits(model, loader, device)

    t = fit_temperature(*data["val"])
    out = {"run": run_dir.name, "temperature": round(t, 4)}
    for split in ("val", "test", "canteen"):
        before = evaluate(*data[split], 1.0)
        after = evaluate(*data[split], t)
        out[split] = {"before": before, "after": after}
    save_json(run_dir / "calibration.json", out)
    print(
        f"{run_dir.name}: T={t:.3f}  "
        f"test ECE {out['test']['before']['ece']:.3f}->{out['test']['after']['ece']:.3f}  "
        f"proxy ECE {out['canteen']['before']['ece']:.3f}->{out['canteen']['after']['ece']:.3f}"
    )
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", default=[])
    p.add_argument("--all", action="store_true")
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args(argv)

    runs = args.run or ([str(ROOT / "results/runs" / r) for r in RUNS] if args.all else [])
    if not runs:
        p.error("pass --run <dir> or --all")

    table = []
    for r in runs:
        table.append(run_one(Path(r).resolve(), Path(args.data_root).resolve(), args.workers))
    save_json(ROOT / "results/tables/calibration.json", table)
    print("wrote results/tables/calibration.json")


if __name__ == "__main__":
    main()
