"""Train one configuration of the dish classifier.

Example:
    python -m foodcls.train --arch resnet18 --mode frozen --color-aug \
        --epochs 12 --out ../results/runs/resnet18_frozen_aug
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .classes import CLASS_NAMES
from .data import SplitDataset, build_transforms, IMG_SIZE
from .models import apply_finetune_policy, build_model, freeze_batchnorm
from .utils import (
    classification_metrics,
    count_params,
    expected_calibration_error,
    pick_device,
    predict,
    save_json,
    set_seed,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="../data")
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v3_small", "efficientnet_b0"])
    p.add_argument("--mode", default="frozen", choices=["frozen", "last_stage", "full"])
    p.add_argument("--color-aug", dest="color_aug", action="store_true")
    p.add_argument("--no-color-aug", dest="color_aug", action="store_false")
    p.set_defaults(color_aug=False)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", dest="no_pretrained", action="store_true",
                   help="random init (from-scratch baseline); use with --mode full")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--extra-train", default=None,
                   help="extra ImageFolder root of noisy images (e.g. data/external/food101n_20cls)")
    p.add_argument("--extra-weight", type=float, default=0.5,
                   help="sampling weight of an extra image relative to one official-train image")
    p.add_argument("--out", required=True)
    p.add_argument("--kd-teacher", default=None,
                   help="run directory whose best.pt is distilled into the student (knowledge distillation)")
    p.add_argument("--kd-temperature", type=float, default=4.0)
    p.add_argument("--kd-alpha", type=float, default=0.5,
                   help="weight of the distillation term; 1-alpha goes to the hard-label loss")
    p.add_argument("--tag", default="")
    return p.parse_args(argv)


def make_loader(ds, batch_size, workers, shuffle):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    labels, probs, _ = predict(model, loader, device)
    met = classification_metrics(labels, probs, CLASS_NAMES)
    met["ece"] = expected_calibration_error(labels, probs)
    return met, labels, probs


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()
    set_seed(args.seed)
    device = pick_device()

    train_ds = SplitDataset(data_root / "splits/train", CLASS_NAMES, build_transforms(True, args.color_aug))
    val_ds = SplitDataset(data_root / "splits/val", CLASS_NAMES, build_transforms(False, False))
    if args.extra_train:
        extra_ds = SplitDataset(Path(args.extra_train), CLASS_NAMES, build_transforms(True, args.color_aug))
        if len(extra_ds) == 0:
            raise SystemExit(f"no images found under {args.extra_train}")
        base_n = len(train_ds)
        train_ds = ConcatDataset([train_ds, extra_ds])
        weights = np.ones(len(train_ds), dtype=np.float64)
        weights[base_n:] = max(1e-3, args.extra_weight)
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=args.workers > 0,
        )
        print(f"[extra-train] {base_n} official + {len(extra_ds)} extra images "
              f"(extra weight {args.extra_weight})")
    else:
        train_loader = make_loader(train_ds, args.batch_size, args.workers, True)
    val_loader = make_loader(val_ds, args.batch_size, args.workers, False)

    model = build_model(args.arch, len(CLASS_NAMES), pretrained=not args.no_pretrained).to(device)
    trainable, total = apply_finetune_policy(model, args.mode)

    teacher = None
    if args.kd_teacher:
        tckpt = torch.load(Path(args.kd_teacher).resolve() / "best.pt", map_location=device, weights_only=False)
        teacher = build_model(tckpt["arch"], len(CLASS_NAMES), pretrained=False).to(device)
        teacher.load_state_dict(tckpt["model"])
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        print(f"[kd] teacher {tckpt['arch']} from {args.kd_teacher} "
              f"(T={args.kd_temperature}, alpha={args.kd_alpha})")

    head = model.fc if hasattr(model, "fc") else model.classifier[-1]
    head_ids = {id(p) for p in head.parameters()}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": list(head.parameters()), "lr": args.lr_head}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.lr_backbone})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = vars(args) | {
        "device": str(device),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "trainable_params": trainable,
        "total_params": total,
        "torch": torch.__version__,
    }
    save_json(out / "config.json", config)
    print(json.dumps(config, indent=2))

    history = []
    best = {"macro_f1": -1.0, "epoch": -1}
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.mode == "frozen":
            freeze_batchnorm(model)
        t0 = time.perf_counter()
        running_loss, seen, correct = 0.0, 0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
                    t = args.kd_temperature
                    kd = nn.functional.kl_div(
                        nn.functional.log_softmax(logits / t, dim=1),
                        nn.functional.softmax(t_logits / t, dim=1),
                        reduction="batchmean",
                    ) * (t * t)
                    loss = (1.0 - args.kd_alpha) * loss + args.kd_alpha * kd
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * x.size(0)
            seen += x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        scheduler.step()
        train_time = time.perf_counter() - t0

        val_met, _, _ = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "train_acc": correct / max(1, seen),
            "val_acc": val_met["accuracy"],
            "val_macro_f1": val_met["macro_f1"],
            "val_ece": val_met["ece"],
            "lr_head": optimizer.param_groups[0]["lr"],
            "epoch_time_s": round(train_time, 1),
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs} loss {row['train_loss']:.4f} "
            f"train@1 {row['train_acc']*100:.2f} val@1 {row['val_acc']*100:.2f} "
            f"valF1 {row['val_macro_f1']*100:.2f} ({train_time:.1f}s)"
        )

        ckpt = {
            "model": model.state_dict(),
            "arch": args.arch,
            "mode": args.mode,
            "color_aug": args.color_aug,
            "class_names": CLASS_NAMES,
            "epoch": epoch,
            "val_macro_f1": val_met["macro_f1"],
            "config": config,
        }
        torch.save(ckpt, out / "last.pt")
        if val_met["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": val_met["macro_f1"], "epoch": epoch, "val_acc": val_met["accuracy"]}
            torch.save(ckpt, out / "best.pt")

    with open(out / "history.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # Final evaluation of the selected checkpoint on the public test split and
    # on the canteen-capture proxy domain.
    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    results = {
        "best": best,
        "arch": args.arch,
        "mode": args.mode,
        "color_aug": args.color_aug,
        "pretrained": not args.no_pretrained,
    }
    for split in ("val", "test", "canteen"):
        ds = SplitDataset(data_root / "splits" / split, CLASS_NAMES, build_transforms(False, False))
        loader = make_loader(ds, 128, args.workers, False)
        met, labels, probs = evaluate(model, loader, device)
        results[split] = met
        np.savez_compressed(
            out / f"pred_{split}.npz",
            labels=labels,
            probs=probs.astype(np.float32),
            paths=np.array([p for p, _ in ds.samples]),
        )
        print(f"[{split}] acc={met['accuracy']*100:.2f} macroF1={met['macro_f1']*100:.2f} ece={met['ece']:.3f}")
    save_json(out / "results.json", results)
    save_json(out / "summary.json", {"tag": args.tag or out.name, **results["val"], "canteen_acc": results["canteen"]["accuracy"]})
    print("done:", out)


if __name__ == "__main__":
    main()
