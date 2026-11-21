# -*- coding: utf-8 -*-
"""Train the 20-class canteen-Chinese head on the CNFOOD-241 subset.

Two stages, cheapest first:

    python scripts/train_cn241.py --mode probe        # frozen features + logistic regression
    python scripts/train_cn241.py --mode finetune --epochs 8

The probe answers one question: does the Food-101N backbone already know these
dishes?  If linear separability is already high, a full fine-tune is only worth
it for the last few points.
"""
from __future__ import annotations

import argparse
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

from foodcls.data import IMG_SIZE, SplitDataset, build_transforms  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from foodcls.utils import expected_calibration_error, pick_device, set_seed  # noqa: E402


def fit_temperature(val_logits: np.ndarray, labels: np.ndarray) -> float:
    """Post-hoc temperature scaling fitted by minimising NLL on validation logits."""
    z = torch.tensor(val_logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=80)
    loss_fn = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(z / torch.exp(log_t), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item())


class FocalLoss(nn.Module):
    """Cross-entropy with optional class weights and focal modulation.

    gamma=0 and weight=None is plain (optionally smoothed) cross-entropy, so the
    baseline path stays bit-for-bit the same.  Only used for the class-balance
    A/B on the 21-way canteen head: boiled chicken has 223 training images
    against 842 for most dishes, and a focal term keeps the easy bulk classes
    from dominating the gradient once the weights even the loss out.
    """

    def __init__(self, gamma: float = 0.0, weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("weight", None if weight is None else torch.as_tensor(weight, dtype=torch.float32))

    def forward(self, logits, target):
        ce = torch.nn.functional.cross_entropy(
            logits, target, weight=self.weight, label_smoothing=self.label_smoothing, reduction="none")
        if self.gamma <= 0:
            return ce.mean()
        logp = torch.nn.functional.log_softmax(logits, dim=1)
        pt = logp.gather(1, target[:, None]).squeeze(1).exp()
        return (ce * (1.0 - pt).pow(self.gamma)).mean()


def class_weights(dataset, num_classes: int, cap: float = 4.0, skip=()):
    """Inverse-frequency weights, clipped so one tiny class cannot dominate.

    The skip indices keep weight 1.0 — used for the "other" bucket, whose many
    exemplars make it the anchor of overall accuracy.
    """
    counts = np.bincount([label for _, label in dataset.samples], minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / (num_classes * counts)
    w = np.clip(w, 1.0 / cap, cap)
    for idx in skip:
        w[int(idx)] = 1.0
    return w, counts


def class_table(path: Path):
    table = json.loads(path.read_text(encoding="utf-8"))
    return table, [c["slug"] for c in table["classes"]]


def load_backbone(arch: str, init: str, num_classes: int, device):
    model = build_model(arch, num_classes, pretrained=False)
    if init:
        ckpt = torch.load(init, map_location="cpu", weights_only=False)
        src, tgt, copied = ckpt["model"], model.state_dict(), 0
        for k, v in src.items():
            if k in tgt and tgt[k].shape == v.shape:
                tgt[k] = v
                copied += 1
        model.load_state_dict(tgt)
        print(f"loaded {copied} tensors from {init} (head re-initialised)", flush=True)
    return model.to(device)


@torch.no_grad()
def extract(model, loader, device):
    feats, labels = [], []
    model.eval()
    for x, y in loader:
        f = model(x.to(device, non_blocking=True))
        if f.dim() > 2:
            f = torch.flatten(torch.nn.functional.adaptive_avg_pool2d(f, 1), 1)
        feats.append(f.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def topk_accuracy(probs, labels, names, k=5):
    order = np.argsort(-probs, axis=1)
    top1 = float((order[:, 0] == labels).mean())
    topk = float((order[:, :k] == labels[:, None]).any(1).mean())
    per_class = {}
    for i, name in enumerate(names):
        m = labels == i
        per_class[name] = float((order[m, 0] == i).mean()) if m.any() else 0.0
    return {"top1": top1, f"top{k}": topk, "per_class": per_class, "n": int(len(labels))}


def run_probe(args, names, table, device):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import Normalizer

    root = Path(args.data_root)
    model = load_backbone(args.arch, args.init, len(names), device)
    if args.arch.startswith("resnet"):
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
    else:
        feat_dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
    model.to(device)

    tf = build_transforms(False, False)
    feats = {}
    for split in ("train", "val", "test"):
        ds = SplitDataset(root / split, names, tf)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)
        t0 = time.perf_counter()
        feats[split] = extract(model, dl, device)
        print(f"{split}: {len(ds)} images in {time.perf_counter()-t0:.1f}s", flush=True)

    scaler = Normalizer()
    x_tr, y_tr = feats["train"]
    x_tr = scaler.fit_transform(x_tr)
    clf = LogisticRegression(C=args.C, max_iter=4000, n_jobs=-1)
    clf.fit(x_tr, y_tr)
    print(f"probe trained on {x_tr.shape[0]} x {x_tr.shape[1]} (feat_dim {feat_dim})", flush=True)

    out = {"mode": "probe", "arch": args.arch, "init": args.init, "C": args.C,
           "feat_dim": int(feat_dim), "classes": names, "n_train": int(len(y_tr))}
    for split in ("val", "test"):
        x, y = feats[split]
        probs = clf.predict_proba(scaler.transform(x))
        met = topk_accuracy(probs, y, names, k=5)
        out[split] = met
        print(f"{split}: top-1 {met['top1']*100:.2f} top-5 {met['top5']*100:.2f} (n={met['n']})", flush=True)

    # most-confused pairs on the test split
    x_te, y_te = feats["test"]
    pred = clf.predict(scaler.transform(x_te))
    pairs = {}
    for t, p in zip(y_te, pred):
        if t != p:
            key = f"{names[t]} -> {names[p]}"
            pairs[key] = pairs.get(key, 0) + 1
    out["confusions"] = dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:12])
    return out


def run_finetune(args, names, table, device):
    from foodcls.utils import classification_metrics

    root = Path(args.data_root)
    set_seed(args.seed)
    train_tf = build_transforms(True, True)
    if args.aug == "strong":
        # Canteen photos are shot under mixed lamps and on coloured trays; the
        # baseline jitter assumes daylight and clean plates, which is why the
        # pale dishes (white-cut chicken) were the ones that stayed weak.
        train_tf = build_transforms(True, True, crop_scale=0.45, jitter=0.5, hue=0.08, gray_p=0.10)
    train_ds = SplitDataset(root / "train", names, train_tf)
    val_ds = SplitDataset(root / "val", names, build_transforms(False, False))
    test_ds = SplitDataset(root / "test", names, build_transforms(False, False))
    loaders = {
        # Only the train loader keeps its workers alive: on Windows every worker
        # holds its own copy of torch plus the batch buffers, and the eval loaders
        # used to add three more each (nine in total), which is what tipped the
        # commit limit over.  Eval workers spawn for the pass and exit after it.
        "train": DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                            pin_memory=True, drop_last=True, persistent_workers=args.workers > 0),
        "val": DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.eval_workers,
                          pin_memory=True),
        "test": DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.eval_workers,
                           pin_memory=True),
    }
    print(f"train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}", flush=True)

    model = load_backbone(args.arch, args.init, len(names), device)
    head = model.fc if hasattr(model, "fc") else model.classifier[-1]
    head_ids = {id(p) for p in head.parameters()}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": list(head.parameters()), "lr": args.lr_head}]
    if backbone:
        groups.append({"params": backbone, "lr": args.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    weights = None
    if args.class_balance == "weights":
        skip = [names.index(s.strip()) for s in args.balance_skip.split(",") if s.strip() in names]
        w, counts = class_weights(train_ds, len(names), cap=args.balance_cap, skip=skip)
        weights = torch.tensor(w, dtype=torch.float32, device=device)
        rare = sorted(zip(counts.tolist(), names))[:3]
        print("class weights on (cap %.1f); rarest: %s | weight range %.2f..%.2f | fixed: %s" % (
            args.balance_cap, ", ".join(f"{n} {int(c)}" for c, n in rare), w.min(), w.max(),
            ",".join(names[i] for i in skip) or "none"), flush=True)
    criterion = FocalLoss(gamma=args.focal_gamma, weight=weights, label_smoothing=args.label_smoothing)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"val_acc": -1.0, "epoch": -1, "state": None}
    history = []
    start_epoch = 1
    last_path = Path(args.save_dir) / "last.pt"
    if args.resume and last_path.is_file():
        prev = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(prev["model"])
        opt.load_state_dict(prev["optimizer"])
        sched.load_state_dict(prev["scheduler"])
        best = prev["best"]
        history = prev["history"]
        start_epoch = prev["epoch"] + 1
        print(f"resumed {last_path} at epoch {prev['epoch']} (best val@1 {best['val_acc']*100:.2f})", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, loss_sum, seen, correct = time.perf_counter(), 0.0, 0, 0
        for step, (x, y) in enumerate(loaders["train"]):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * x.size(0)
            seen += x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            if args.debug_every and (step + 1) % args.debug_every == 0:
                print(f"  batch {step+1} {seen} imgs {time.perf_counter()-t0:.0f}s", flush=True)
        sched.step()
        model.eval()
        probs, labs = [], []
        with torch.no_grad():
            for x, y in loaders["val"]:
                probs.append(torch.softmax(model(x.to(device)).float(), 1).cpu().numpy())
                labs.append(y.numpy())
        met = classification_metrics(np.concatenate(labs), np.concatenate(probs), names)
        dt = time.perf_counter() - t0
        row = {"epoch": epoch, "train_loss": loss_sum / max(1, seen), "train_acc": correct / max(1, seen),
               "val_acc": met["accuracy"], "val_macro_f1": met["macro_f1"], "epoch_time_s": round(dt, 1)}
        history.append(row)
        print(f"epoch {epoch:02d}/{args.epochs} loss {row['train_loss']:.4f} train@1 {row['train_acc']*100:.2f} "
              f"val@1 {row['val_acc']*100:.2f} valF1 {row['val_macro_f1']*100:.2f} ({dt:.0f}s)", flush=True)
        if met["accuracy"] > best["val_acc"]:
            best = {"val_acc": met["accuracy"], "epoch": epoch, "state": {k: v.clone() for k, v in model.state_dict().items()}}
        # Written every epoch so an interrupted run resumes instead of restarting.
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "epoch": epoch, "best": best,
                    "history": history, "arch": args.arch, "class_names": names},
                   last_path)
    model.load_state_dict(best["state"])
    model.eval()
    # temperature scaling on the validation split, applied to everything downstream
    def logits_of(loader):
        zs, ys = [], []
        with torch.no_grad():
            for x, y in loader:
                zs.append(model(x.to(device)).float().cpu().numpy())
                ys.append(y.numpy())
        return np.concatenate(zs), np.concatenate(ys)

    z_val, y_val = logits_of(loaders["val"])
    temperature = fit_temperature(z_val, y_val)
    ece_before = expected_calibration_error(y_val, torch.softmax(torch.tensor(z_val, dtype=torch.float64), 1).numpy())
    ece_after = expected_calibration_error(
        y_val, torch.softmax(torch.tensor(z_val, dtype=torch.float64) / temperature, 1).numpy())
    print(f"temperature {temperature:.3f} | val ECE {ece_before*100:.2f} -> {ece_after*100:.2f}", flush=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "arch": args.arch, "class_names": names,
                "class_table": table, "epoch": best["epoch"], "val_acc": best["val_acc"],
                "temperature": temperature, "config": vars(args)}, save_dir / "best.pt")
    (save_dir / "calibration.json").write_text(
        json.dumps({"temperature": round(float(temperature), 4), "n": int(len(y_val)),
                    "val_ece_before": round(float(ece_before), 4),
                    "val_ece_after": round(float(ece_after), 4)}, indent=1), encoding="utf-8")
    print(f"saved {save_dir / 'best.pt'}", flush=True)

    z_test, y_test = logits_of(loaders["test"])
    probs = torch.softmax(torch.tensor(z_test, dtype=torch.float64) / temperature, 1).numpy()
    met = topk_accuracy(probs, y_test, names, k=5)
    met["ece"] = float(expected_calibration_error(y_test, probs))
    met["mean_top1_prob"] = float(probs.max(1).mean())
    print(f"test: top-1 {met['top1']*100:.2f} top-5 {met['top5']*100:.2f} "
          f"ECE {met['ece']*100:.2f} mean p {met['mean_top1_prob']:.3f}", flush=True)
    return {"mode": "finetune", "arch": args.arch, "init": args.init, "epochs": args.epochs,
            "best_epoch": best["epoch"], "best_val_acc": best["val_acc"], "classes": names,
            "history": history, "test": met}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "data/splits/cn241"))
    ap.add_argument("--classes", default=str(ROOT / "results/tables/cn241_classes.json"))
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50", "mobilenet_v3_small"])
    ap.add_argument("--init", default=str(ROOT / "results/runs/resnet18_last_stage_aug_f101n/best.pt"))
    ap.add_argument("--mode", default="probe", choices=["probe", "finetune"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-workers", type=int, default=0)
    ap.add_argument("--debug-every", type=int, default=0, help="print progress every N train batches")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--aug", default="base", choices=["base", "strong"])
    ap.add_argument("--class-balance", default="none", choices=["none", "weights"])
    ap.add_argument("--balance-cap", type=float, default=4.0,
                    help="clip inverse-frequency weights to [1/cap, cap]")
    ap.add_argument("--balance-skip", default="",
                    help="comma-separated class slugs kept at weight 1.0, e.g. other_cn")
    ap.add_argument("--focal-gamma", type=float, default=0.0,
                    help="0 = plain cross-entropy; 1-2 typical for hard-example focus")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results/tables/cn241_probe.json"))
    ap.add_argument("--save-dir", default=str(ROOT / "results/runs/cn241_last_stage"))
    args = ap.parse_args(argv)

    table, names = class_table(Path(args.classes))
    device = pick_device()
    print(f"device {device}, {len(names)} classes", flush=True)
    result = run_probe(args, names, table, device) if args.mode == "probe" else run_finetune(args, names, table, device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
