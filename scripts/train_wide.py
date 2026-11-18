"""Train the wide model: all 101 Food-101 classes on the full training split.

The 20-class canteen protocol stays in foodcls.classes; this model exists so the
demo can name dishes outside that shortlist (peking duck, paella, tacos, ...).  It
uses the same backbone builder and split format, but a wider label head and far
more data (65k train images instead of 12k), so it needs a full fine-tune rather
than the last-stage recipe.

    python scripts/train_wide.py --epochs 10
    python scripts/train_wide.py --arch resnet50 --epochs 8 --batch-size 64
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes_wide import WIDE_CLASS_NAMES, WIDE_NUM_CLASSES  # noqa: E402
from foodcls.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, SplitDataset, build_transforms  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from foodcls.utils import (  # noqa: E402
    classification_metrics,
    expected_calibration_error,
    pick_device,
    save_json,
    set_seed,
)


def make_loader(ds, batch_size, workers, shuffle):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, drop_last=shuffle, persistent_workers=workers > 0)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        probs.append(torch.softmax(model(x).float(), 1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


@torch.no_grad()
def predict_logits(model, loader, device):
    """Raw logits: temperature scaling is defined on logits, not on softmax output."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits.append(model(x).float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(logits)


def tta_transforms():
    from torchvision import transforms

    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return [
        transforms.Compose([transforms.Resize(int(IMG_SIZE * 256 / 224)), transforms.CenterCrop(IMG_SIZE)] + norm),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))] + norm),
        transforms.Compose([transforms.Resize(int(IMG_SIZE * 320 / 224)), transforms.CenterCrop(IMG_SIZE)] + norm),
    ]


def _load(path):
    from PIL import Image

    with open(path, "rb") as fh:
        return Image.open(fh).convert("RGB")


@torch.no_grad()
def evaluate_tta(model, root: Path, device, batch_size=64, tta=True):
    """Top-1/top-5 with the same three-crop average the demo uses."""
    tfs = tta_transforms() if tta else tta_transforms()[:1]
    dataset = SplitDataset(root, WIDE_CLASS_NAMES, transform=None)
    acc1 = acc5 = 0.0
    n = 0
    model.eval()
    for start in range(0, len(dataset.samples), batch_size):
        chunk = dataset.samples[start:start + batch_size]
        batch = torch.stack([torch.stack([tf(_load(p)) for tf in tfs]) for p, _ in chunk]).to(device)
        b, v = batch.shape[:2]
        logits = model(batch.flatten(0, 1)).float().view(b, v, -1).mean(1)
        y = torch.tensor([lab for _, lab in chunk], device=device)
        top5 = logits.topk(5, dim=1).indices
        acc1 += (top5[:, 0] == y).sum().item()
        acc5 += (top5 == y[:, None]).any(1).sum().item()
        n += len(chunk)
    return {"top1": acc1 / n, "top5": acc5 / n, "n": n, "tta": tta}


def fit_temperature(val_logits: np.ndarray, labels: np.ndarray) -> float:
    """Post-hoc temperature scaling, minimising NLL on the validation split."""
    z = torch.tensor(val_logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)
    loss_fn = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(z / torch.exp(log_t), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item())


def warm_start(model, ckpt_path: Path, head_only: bool = False) -> dict:
    """Copy a checkpoint into the wide model, matching the head rows by class name.

    The narrow canteen head and the wide head share 20 dish names but not their
    order (WIDE_CLASS_NAMES is alphabetical, CLASS_NAMES follows the canteen
    menu), so copying fc rows positionally would scramble them.  Rows are placed
    by name instead; every wide class the source does not know keeps the random
    init it was given.

    With head_only the backbone is left alone, which lets a run take its backbone
    from one checkpoint and its classifier from another.
    """
    src = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src_sd = src["model"]
    src_names = list(src.get("class_names") or [])
    tgt = model.state_dict()
    copied = 0
    head_rows = {}
    for key, val in src_sd.items():
        if key not in tgt:
            continue
        if key in ("fc.weight", "fc.bias") and src_names:
            index = {name: i for i, name in enumerate(WIDE_CLASS_NAMES)}
            rows = [(index[n], i) for i, n in enumerate(src_names) if n in index]
            if not rows:
                continue
            for dst, s in rows:
                tgt[key][dst] = val[s]
            head_rows[key] = [src_names[s] for _, s in rows]
            copied += 1
            continue
        if head_only:
            continue
        if tgt[key].shape == val.shape:
            tgt[key] = val
            copied += 1
    model.load_state_dict(tgt)
    return {"tensors": copied, "head_rows": head_rows, "source_classes": src_names,
            "source_epoch": src.get("epoch"), "source_arch": src.get("arch"),
            "head_only": head_only}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "data/wide"))
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50", "mobilenet_v3_small"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "results/runs/resnet18_food101"))
    ap.add_argument("--init", default="",
                    help="warm-start checkpoint(s), comma separated and applied in order; head rows are "
                         "matched by class name, later checkpoints win")
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root).resolve()
    set_seed(args.seed)
    device = pick_device()

    train_ds = SplitDataset(root / "splits/train", WIDE_CLASS_NAMES, build_transforms(True, True))
    val_ds = SplitDataset(root / "splits/val", WIDE_CLASS_NAMES, build_transforms(False, False))
    train_loader = make_loader(train_ds, args.batch_size, args.workers, True)
    val_loader = make_loader(val_ds, args.batch_size, args.workers, False)
    print(f"train {len(train_ds)} / val {len(val_ds)} images, {WIDE_NUM_CLASSES} classes", flush=True)

    model = build_model(args.arch, WIDE_NUM_CLASSES, pretrained=True).to(device)
    warm_infos = []
    for entry in [p.strip() for p in args.init.split(",") if p.strip()]:
        # "path@head" takes only the classifier rows, leaving the backbone alone.
        head_only = entry.endswith("@head")
        init_path = entry[: -len("@head")] if head_only else entry
        info = warm_start(model, Path(init_path), head_only=head_only)
        warm_infos.append(info)
        kept = len(info["head_rows"].get("fc.weight", []))
        print(f"warm start from {init_path}{' (head only)' if head_only else ''}: "
              f"{info['tensors']} tensors, {kept}/{len(info['source_classes'])} head rows "
              f"placed by class name", flush=True)
    warm_info = warm_infos[-1] if warm_infos else None
    head = model.fc if hasattr(model, "fc") else model.classifier[-1]
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": list(head.parameters()), "lr": args.lr_head}]
    if backbone:
        groups.append({"params": backbone, "lr": args.lr_backbone})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = vars(args) | {"device": str(device), "train_size": len(train_ds),
                           "val_size": len(val_ds), "num_classes": WIDE_NUM_CLASSES,
                           "warm_start": [{k: v for k, v in w.items() if k != "head_rows"}
                                          for w in warm_infos] or None,
                           "torch": torch.__version__}
    save_json(out / "config.json", config)

    history, best = [], {"val_macro_f1": -1.0, "epoch": -1}
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running_loss = seen = correct = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * x.size(0)
            seen += x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        scheduler.step()
        dt = time.perf_counter() - t0

        labels, probs = predict(model, val_loader, device)
        met = classification_metrics(labels, probs, WIDE_CLASS_NAMES)
        row = {"epoch": epoch, "train_loss": running_loss / max(1, seen),
               "train_acc": correct / max(1, seen), "val_acc": met["accuracy"],
               "val_macro_f1": met["macro_f1"], "val_ece": expected_calibration_error(labels, probs),
               "epoch_time_s": round(dt, 1)}
        history.append(row)
        print(f"epoch {epoch:02d}/{args.epochs} loss {row['train_loss']:.4f} "
              f"train@1 {row['train_acc']*100:.2f} val@1 {row['val_acc']*100:.2f} "
              f"valF1 {row['val_macro_f1']*100:.2f} ({dt:.0f}s)", flush=True)

        ckpt = {"model": model.state_dict(), "arch": args.arch, "class_names": WIDE_CLASS_NAMES,
                "epoch": epoch, "val_macro_f1": met["macro_f1"], "val_acc": met["accuracy"],
                "config": config}
        torch.save(ckpt, out / "last.pt")
        if met["macro_f1"] > best["val_macro_f1"]:
            best = {"val_macro_f1": met["macro_f1"], "val_acc": met["accuracy"], "epoch": epoch}
            torch.save(ckpt, out / "best.pt")
        with open(out / "history.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)

    # calibration on the validation split
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=False)["model"])
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for x, y in val_loader:
            zs.append(model(x.to(device)).float().cpu().numpy())
            ys.append(y.numpy())
    temperature = fit_temperature(np.concatenate(zs), np.concatenate(ys))
    save_json(out / "calibration.json", {"temperature": round(temperature, 4), "n": len(val_ds)})
    print(f"temperature {temperature:.3f}", flush=True)

    test = evaluate_tta(model, root / "splits/test", device, args.batch_size, tta=False)
    test_tta = evaluate_tta(model, root / "splits/test", device, args.batch_size, tta=True)
    print(f"test top-1 {test['top1']*100:.2f} top-5 {test['top5']*100:.2f} | "
          f"3-crop top-1 {test_tta['top1']*100:.2f} top-5 {test_tta['top5']*100:.2f}", flush=True)

    ds = SplitDataset(root / "splits/test", WIDE_CLASS_NAMES, build_transforms(False, False))
    # Windows: many workers + a persistent loader can exhaust the shared-memory
    # segment ("Couldn't open shared file mapping", error 1455), so keep this pass small.
    labels_test, logits_test = predict_logits(
        model, make_loader(ds, args.batch_size, min(2, args.workers), False), device)
    probs_test = torch.softmax(torch.tensor(logits_test), 1).numpy()
    pred_test = probs_test.argmax(1)
    old_ids = np.arange(101)

    def _group_acc(ids):
        mask = np.isin(labels_test, ids)
        return {"n": int(mask.sum()),
                "top1": float((pred_test[mask] == labels_test[mask]).mean()) if mask.any() else 0.0}

    test_groups = {"food101": _group_acc(old_ids),
                   "chinese": _group_acc(np.arange(101, WIDE_NUM_CLASSES))}
    print(f"test by group: {test_groups}", flush=True)
    results = {
        "arch": args.arch, "num_classes": WIDE_NUM_CLASSES, "epochs": args.epochs,
        "best": best, "test": test, "test_tta": test_tta, "temperature": round(temperature, 4),
        "test_ece": expected_calibration_error(labels_test, probs_test),
        "test_ece_calibrated": expected_calibration_error(
            labels_test, torch.softmax(torch.tensor(logits_test) / temperature, 1).numpy()),
        "test_groups": test_groups, "history": history,
    }
    save_json(out / "results.json", results)
    print(json.dumps({k: v for k, v in results.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
