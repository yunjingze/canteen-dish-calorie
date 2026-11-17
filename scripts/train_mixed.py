"""Train the multi-label plate model on synthetic composites.

Single-dish photos are turned into multi-label plates (1-3 dishes share one
canvas) so the demo can answer "this tray holds fries *and* wings".  With
``--classes plate`` the training pool also includes the 20 canteen dishes from
CNFOOD-241, so one head covers both the Food-101 menu and the campus canteen:
western units come first (0-19), canteen units after (20-39).

The script reports component-level precision/recall/F1 per cuisine on held-out
val plates, Top-1 accuracy on the ordinary single-dish test split (which must not
collapse), and a real-photo canteen proxy when the images are present.

    python scripts/train_mixed.py --classes plate --epochs 10 --workers 0 \
        --init-ckpt results/runs/resnet18_multilabel_aug/best.pt \
        --out results/runs/resnet18_multilabel_40
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
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES                     # noqa: E402
from foodcls.classes_cn import CN_CLASS_NAMES                # noqa: E402
from foodcls.data import SplitDataset, build_transforms      # noqa: E402
from foodcls.mixed import (                                  # noqa: E402
    MixedPlateDataset,
    build_eval_set,
    detect_parts,
)
from foodcls.models import apply_finetune_policy, build_model, freeze_batchnorm  # noqa: E402
from foodcls.plate_classes import (                          # noqa: E402
    CANTEEN_CLASS_NAMES,
    PLATE_CLASS_NAMES,
    PLATE_KCAL_PER_100G,
    PLATE_PORTION_G,
    PLATE_ZH,
    WESTERN_CLASS_NAMES,
)
from foodcls.utils import pick_device, save_json, set_seed   # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--cn-root", default=str(ROOT / "data/splits/cn241_21"),
                   help="canteen splits (only read with --classes plate)")
    p.add_argument("--classes", default="western", choices=["western", "plate"],
                   help="western: 20 Food-101 dishes; plate: those plus 20 canteen dishes")
    p.add_argument("--arch", default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    p.add_argument("--mode", default="last_stage", choices=["frozen", "last_stage", "full"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--n-train", type=int, default=12000)
    p.add_argument("--n-val", type=int, default=600,
                   help="val plates per cuisine (western and canteen)")
    p.add_argument("--n-canteen", type=int, default=300)
    p.add_argument("--n-real", type=int, default=80,
                   help="plates built from real canteen photos, when that split exists")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(ROOT / "results/runs/resnet18_multilabel_aug"))
    p.add_argument("--init-ckpt", default=None,
                   help="warm-start weights (e.g. the single-dish or 20-class plate run)")
    return p.parse_args(argv)


def class_list(kind: str):
    return list(PLATE_CLASS_NAMES) if kind == "plate" else list(CLASS_NAMES)


@torch.no_grad()
def predict_probs(model, loader, device):
    """Sigmoid scores with flip TTA: micro-batch is small, so the extra pass is cheap.

    A small side dish (three wings at the edge of a fry tray) can vanish from
    one view and survive in the mirrored one; taking the element-wise maximum
    keeps it.  The web app uses the same rule, so the calibrated thresholds
    transfer.
    """
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        probs = torch.sigmoid(model(x).float())
        probs = torch.maximum(probs, torch.sigmoid(model(torch.flip(x, dims=[3])).float()))
        out.append(probs.cpu().numpy())
    return np.concatenate(out, axis=0)


class PlateEvalSet(Dataset):
    """Fixed composites for evaluation (module level: Windows spawn pickles it)."""

    def __init__(self, plates, num_classes: int | None = None):
        from torchvision import transforms
        from foodcls.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

        self.plates = plates
        self.num_classes = len(CLASS_NAMES) if num_classes is None else int(num_classes)
        self.tf = transforms.Compose([
            transforms.Resize(int(IMG_SIZE * 256 / 224)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.plates)

    def __getitem__(self, i):
        img, labels = self.plates[i]
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        target[labels] = 1.0
        return self.tf(img), target


class SampleList(Dataset):
    """Flat (path, label) list -- used for single-dish accuracy on either pool."""

    def __init__(self, samples, transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image

        path, label = self.samples[index]
        with open(path, "rb") as fh:
            img = Image.open(fh).convert("RGB")
        return self.transform(img), int(label)


def make_plate_loader(plates, batch_size=64, workers=2, num_classes=None):
    return DataLoader(PlateEvalSet(plates, num_classes=num_classes), batch_size=batch_size,
                      shuffle=False, num_workers=workers)


def component_metrics(targets, probs, threshold, class_thresholds=None, class_names=None):
    class_names = CLASS_NAMES if class_names is None else list(class_names)
    targets = targets.astype(bool)
    if class_thresholds:
        cuts = np.array([class_thresholds.get(c, threshold) for c in class_names])[None, :]
        pred = probs >= np.maximum(cuts, threshold)
    else:
        pred = probs >= threshold
    tp = float((pred & targets).sum())
    fp = float((pred & ~targets).sum())
    fn = float((~pred & targets).sum())
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    exact = float((pred == targets).all(axis=1).mean())
    counts = pred.sum(axis=1)
    return {
        "threshold": round(float(threshold), 2),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact,
        "parts_per_plate": float(counts.mean()),
        "single_pred_rate": float((counts == 1).mean()),
    }


def per_class_thresholds(targets, probs, grid=None, class_names=None):
    """Pick the cut-off that maximises F1 for each dish separately.

    One global threshold punishes side dishes: chicken_wings is usually a
    small blob and scores lower than a full-frame french_fries, so at 0.2
    the wings disappear while the fries sit at 0.99.  Per-class cut-offs are
    fitted on the val plates only and stored next to the weights.
    """
    class_names = CLASS_NAMES if class_names is None else list(class_names)
    if grid is None:
        grid = np.arange(0.10, 0.86, 0.01)
    targets = np.asarray(targets, dtype=bool)
    out = {}
    for c, name in enumerate(class_names):
        y = targets[:, c]
        best_f1, best_t = -1.0, 0.2
        for t in grid:
            pred = probs[:, c] >= t
            tp = float((pred & y).sum())
            fp = float((pred & ~y).sum())
            fn = float((~pred & y).sum())
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
            if f1 > best_f1 + 1e-9:
                best_f1, best_t = f1, float(t)
        out[name] = round(best_t, 2)
    return out


def targets_for(plates, num_classes):
    t = np.zeros((len(plates), int(num_classes)), dtype=np.float32)
    for i, (_, labels) in enumerate(plates):
        t[i, labels] = 1.0
    return t


def evaluate_plates(model, plates, device, batch_size=64, class_thresholds=None, num_classes=None):
    class_names = class_list("plate") if num_classes and num_classes > len(CLASS_NAMES) else CLASS_NAMES
    loader = make_plate_loader(plates, batch_size=batch_size, workers=2, num_classes=num_classes)
    probs = predict_probs(model, loader, device)
    targets = targets_for(plates, len(class_names))
    sweep = [component_metrics(targets, probs, t, class_names=class_names)
             for t in np.arange(0.10, 0.71, 0.05)]
    best = max(sweep, key=lambda m: m["f1"])
    tuned = (component_metrics(targets, probs, 0.2, class_thresholds=class_thresholds,
                               class_names=class_names) if class_thresholds else None)
    return best, sweep, tuned, probs, targets


def evaluate_single(model, ds, device, batch_size=128):
    """Top-1 accuracy on ordinary single-dish photos, plus over-splitting rate."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    probs = predict_probs(model, loader, device)
    labels = np.array([lab for _, lab in ds.samples])
    top1 = float((probs.argmax(1) == labels).mean())
    extra = float((probs >= 0.5).sum(axis=1).mean())
    return {"top1": top1, "parts_per_photo": extra}


def evaluate_samples(model, samples, device, batch_size=128):
    ds = SampleList(samples, build_transforms(False, False))
    return evaluate_single(model, ds, device, batch_size=batch_size)


def real_canteen_samples(data_root: Path):
    """Real phone photos of Food-101 dishes taken in a canteen, on western units.

    These are the closest thing to a real tray photo we have: the same dishes as
    the synthetic composites but shot by hand under canteen light.  They were
    never used for training.
    """
    root = data_root / "splits" / "canteen_real"
    if not root.is_dir():
        return []
    out = []
    for cdir in sorted(root.iterdir()):
        if not cdir.is_dir() or cdir.name not in CLASS_NAMES:
            continue
        unit = CLASS_NAMES.index(cdir.name)
        for p in sorted(cdir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                out.append((str(p), unit))
    return out


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()
    cn_root = Path(args.cn_root).resolve()
    set_seed(args.seed)
    device = pick_device()
    print(f"device {device} classes {args.classes}", flush=True)

    names = class_list(args.classes)
    n_classes = len(names)
    offset = len(CLASS_NAMES)

    train_src = SplitDataset(data_root / "splits/train", CLASS_NAMES, transform=None)
    val_src = SplitDataset(data_root / "splits/val", CLASS_NAMES, transform=None)
    test_src = SplitDataset(data_root / "splits/test", CLASS_NAMES, build_transforms(False, False))
    canteen_src = SplitDataset(data_root / "splits/canteen_clean", CLASS_NAMES, transform=None)

    train_samples = list(train_src.samples)
    val_w = build_eval_set(val_src.samples, args.n_val, seed=1234, num_classes=offset)
    val_c, val_c_real, test_c_samples = [], [], []
    canteen_plates = build_eval_set(canteen_src.samples, args.n_canteen, seed=999, num_classes=offset)
    single_c = None
    if args.classes == "plate":
        cn_train = SplitDataset(cn_root / "train", CN_CLASS_NAMES, transform=None)
        cn_val = SplitDataset(cn_root / "val", CN_CLASS_NAMES, transform=None)
        cn_test = SplitDataset(cn_root / "test", CN_CLASS_NAMES, transform=None)
        label_map = [offset + i for i in range(len(CN_CLASS_NAMES))]
        train_samples += [(p, offset + lab) for p, lab in cn_train.samples]
        val_c = build_eval_set(cn_val.samples, args.n_val, seed=4321,
                               num_classes=len(CN_CLASS_NAMES), label_map=label_map)
        canteen_real = real_canteen_samples(data_root)
        if canteen_real:
            # canteen_real holds only some of the 20 dishes; map the units that
            # are actually present onto compact local labels for plate building
            present = sorted({lab for _, lab in canteen_real})
            remap = {g: i for i, g in enumerate(present)}
            real_local = [(p, remap[lab]) for p, lab in canteen_real]
            val_c_real = build_eval_set(real_local, args.n_real, seed=777,
                                        num_classes=len(present), label_map=present)
        test_c_samples = [(p, offset + lab) for p, lab in cn_test.samples]
    # shuffle so a primary dish is not always drawn from the first pool
    order = np.random.default_rng(args.seed).permutation(len(train_samples))
    train_samples = [train_samples[i] for i in order]
    val_plates = val_w + val_c
    print(f"train pool {len(train_samples)} | val plates {len(val_plates)} "
          f"(west {len(val_w)}, cn {len(val_c)}) | real cn {len(val_c_real)}", flush=True)

    train_ds = MixedPlateDataset(
        train_samples, args.n_train, transform=build_transforms(True, True),
        seed=args.seed, num_classes=n_classes,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.workers > 0,
    )

    model = build_model(args.arch, n_classes, pretrained=True).to(device)
    if args.init_ckpt:
        warm = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        state = warm["model"]
        own = model.state_dict()
        copied, kept = {}, []
        for key, value in state.items():
            if key not in own:
                continue
            if value.shape == own[key].shape:
                copied[key] = value
            elif key in ("fc.weight", "fc.bias") and value.shape[0] <= own[key].shape[0]:
                # a 20-class head warm-starts the first 20 rows of the 40-class
                # one; the western dishes keep their old decision boundaries
                rows = value.shape[0]
                merged = own[key].clone()
                merged[:rows] = value
                copied[key] = merged
                kept.append(key)
        missing, unexpected = model.load_state_dict(copied, strict=False)
        print(f"warm start from {args.init_ckpt}: copied {len(copied)} tensors "
              f"(head rows carried over for {kept}), missing={len(missing)}, "
              f"unexpected={len(unexpected)}", flush=True)
    trainable, total = apply_finetune_policy(model, args.mode)
    head = model.fc if hasattr(model, "fc") else model.classifier[-1]
    head_ids = {id(p) for p in head.parameters()}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": list(head.parameters()), "lr": args.lr_head}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.lr_backbone})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    criterion = nn.BCEWithLogitsLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    config = vars(args) | {
        "device": str(device),
        "classes": args.classes,
        "num_classes": n_classes,
        "class_names": names,
        "trainable_params": trainable,
        "total_params": total,
        "torch": torch.__version__,
        "train_pool": len(train_samples),
        "val_plates": len(val_plates),
        "val_plates_west": len(val_w),
        "val_plates_canteen": len(val_c),
        "val_plates_real": len(val_c_real),
        "canteen_plates": len(canteen_plates),
    }
    save_json(out / "config.json", config)
    print(json.dumps({k: config[k] for k in ("classes", "num_classes", "train_pool", "val_plates")}, indent=2), flush=True)

    history, best = [], {"f1": -1.0, "epoch": -1, "threshold": 0.5}
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.mode == "frozen":
            freeze_batchnorm(model)
        t0 = time.perf_counter()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        scheduler.step()
        train_time = time.perf_counter() - t0

        met, sweep, _, _, _ = evaluate_plates(model, val_plates, device, num_classes=n_classes)
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "val_f1": met["f1"],
            "val_precision": met["precision"],
            "val_recall": met["recall"],
            "val_exact": met["exact_match"],
            "val_parts": met["parts_per_plate"],
            "threshold": met["threshold"],
            "epoch_time_s": round(train_time, 1),
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}/{args.epochs} loss {row['train_loss']:.4f} "
            f"plateF1 {row['val_f1']*100:.2f} P {row['val_precision']*100:.2f} "
            f"R {row['val_recall']*100:.2f} exact {row['val_exact']*100:.2f} "
            f"thr {row['threshold']:.2f} ({train_time:.1f}s)",
            flush=True,
        )
        ckpt = {
            "model": model.state_dict(),
            "arch": args.arch,
            "mode": args.mode,
            "multilabel": True,
            "threshold": met["threshold"],
            "class_names": names,
            "epoch": epoch,
            "val_f1": met["f1"],
            "config": config,
        }
        torch.save(ckpt, out / "last.pt")
        if met["f1"] > best["f1"]:
            best = {"f1": met["f1"], "epoch": epoch, "threshold": met["threshold"]}
            torch.save(ckpt, out / "best.pt")

    with open(out / "history.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    thr = float(ckpt["threshold"])
    val_best, val_sweep, _, val_probs, val_targets = evaluate_plates(
        model, val_plates, device, num_classes=n_classes)
    class_thr = per_class_thresholds(val_targets, val_probs, class_names=names)
    val_tuned = component_metrics(val_targets, val_probs, thr, class_thresholds=class_thr,
                                  class_names=names)
    canteen_best, canteen_sweep, _, can_probs, can_targets = evaluate_plates(
        model, canteen_plates, device, num_classes=n_classes)
    canteen_tuned = component_metrics(can_targets, can_probs, thr, class_thresholds=class_thr,
                                      class_names=names)
    single = evaluate_single(model, test_src, device)
    per_cuisine = {}
    for tag, plates in (("western", val_w), ("canteen", val_c), ("real", val_c_real)):
        if not plates:
            continue
        _, _, _, probs, targets = evaluate_plates(model, plates, device, num_classes=n_classes)
        met = component_metrics(targets, probs, thr, class_thresholds=class_thr, class_names=names)
        per_cuisine[tag] = {k: met[k] for k in ("precision", "recall", "f1", "exact_match", "parts_per_plate")}
    single_cn = evaluate_samples(model, test_c_samples, device) if test_c_samples else None
    # store the per-class cut-offs in the checkpoint the app loads
    ckpt["class_thresholds"] = class_thr
    ckpt["threshold"] = thr
    torch.save(ckpt, out / "best.pt")
    results = {
        "arch": args.arch,
        "mode": args.mode,
        "classes": args.classes,
        "num_classes": n_classes,
        "epochs": args.epochs,
        "n_train_plates": args.n_train,
        "train_pool": len(train_samples),
        "threshold": thr,
        "class_thresholds": class_thr,
        "best_epoch": best["epoch"],
        "val_plates": val_best,
        "val_tuned": val_tuned,
        "canteen_plates": canteen_best,
        "canteen_tuned": canteen_tuned,
        "per_cuisine_tuned": per_cuisine,
        "test_single": single,
        "test_single_canteen": single_cn,
        "val_sweep": val_sweep,
        "canteen_sweep": canteen_sweep,
    }
    save_json(out / "results.json", results)
    table = ROOT / "results/tables/mixed_plate.json" if args.classes == "plate"         else ROOT / "results/tables/mixed.json"
    save_json(table, results)
    print(
        f"val plateF1 {val_best['f1']*100:.2f} (P {val_best['precision']*100:.2f} "
        f"R {val_best['recall']*100:.2f} exact {val_best['exact_match']*100:.2f}) "
        f"| tuned {val_tuned['f1']*100:.2f}",
        f"| single test top1 {single['top1']*100:.2f} parts/photo {single['parts_per_photo']:.2f}",
        flush=True,
    )
    for tag, met in per_cuisine.items():
        print(f"  {tag:12s} F1 {met['f1']*100:6.2f} P {met['precision']*100:6.2f} "
              f"R {met['recall']*100:6.2f} exact {met['exact_match']*100:6.2f}", flush=True)
    if single_cn:
        print(f"  canteen single top1 {single_cn['top1']*100:.2f} parts/photo {single_cn['parts_per_photo']:.2f}",
              flush=True)
    print("done:", out, flush=True)


if __name__ == "__main__":
    main()
