"""Generate every figure used in the report and the demo."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from foodcls.utils import pick_device  # noqa: E402
from foodcls.calorie import topk_with_calories  # noqa: E402

FIG = ROOT / "results/figures"
RUNS = ROOT / "results/runs"
PALETTE = ["#2f6df6", "#7a5af8", "#16a34a", "#e0562a", "#0891b2", "#374151"]
# Chinese dish names appear in the prediction panel; pick an installed CJK font.
for _f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
    if any(_f == f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
    }
)


RUN_ORDER = [
    "resnet18_frozen_noaug",
    "resnet18_frozen_aug",
    "resnet18_last_stage_noaug",
    "resnet18_last_stage_aug",
    "mobilenet_v3_small_last_stage_aug",
    "resnet18_scratch_noaug",
]


def load_runs(pattern: str = "*") -> list[Path]:
    runs = [p for p in RUNS.glob(pattern) if (p / "results.json").exists()]
    rank = {name: i for i, name in enumerate(RUN_ORDER)}
    return sorted(runs, key=lambda p: (rank.get(p.name, 99), p.name))


def label_of(run: Path) -> str:
    meta = json.loads((run / "results.json").read_text(encoding="utf-8"))
    mode = {"frozen": "Frozen", "last_stage": "Layer4", "full": "Full"}[meta["mode"]]
    prefix = "" if meta["arch"] == "resnet18" else "MBV3 "
    if meta.get("pretrained", True) is False:
        return f"{prefix}from scratch"
    return f"{prefix}{mode}{' + C' if meta['color_aug'] else ''}"


def fig_training_curves(runs):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for i, run in enumerate(runs):
        import csv

        rows = list(csv.DictReader(open(run / "history.csv", encoding="utf-8")))
        ep = [int(r["epoch"]) for r in rows]
        axes[0].plot(ep, [float(r["train_loss"]) for r in rows], color=PALETTE[i], label=label_of(run))
        axes[1].plot(ep, [float(r["val_acc"]) * 100 for r in rows], color=PALETTE[i], label=label_of(run))
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training loss")
    axes[0].set_title("(a) Training loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation accuracy (%)")
    axes[1].set_title("(b) In-domain validation accuracy")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png")
    plt.close(fig)


def fig_comparison(runs):
    labels = [label_of(r) for r in runs]
    test_acc = [json.loads((r / "results.json").read_text())["test"]["accuracy"] * 100 for r in runs]
    cta = [json.loads((r / "results.json").read_text())["canteen"]["accuracy"] * 100 for r in runs]
    test_f1 = [json.loads((r / "results.json").read_text())["test"]["macro_f1"] * 100 for r in runs]
    x = np.arange(len(runs))
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.bar(x - 0.2, test_acc, 0.38, color="#2f6df6", label="Food-101 test (in-domain)")
    ax.bar(x + 0.2, cta, 0.38, color="#e0562a", label="Canteen proxy (cross-domain)")
    ax.plot(x, test_f1, "k_", ms=22, mew=2, label="Test Macro-F1")
    for i, (a, b) in enumerate(zip(test_acc, cta)):
        ax.text(i - 0.2, a + 1, f"{a:.1f}", ha="center", fontsize=7)
        ax.text(i + 0.2, b + 1, f"{b:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x, labels, fontsize=7.8)
    ax.set_ylim(0, 104)
    ax.set_ylabel("Accuracy / F1 (%)")
    ax.legend(frameon=False, fontsize=7.2, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, -0.14))
    ax.set_title("Effect of depth, colour augmentation and pre-training")
    fig.tight_layout()
    fig.savefig(FIG / "comparison.png")
    plt.close(fig)


def fig_confusion(run, split, fname, title):
    res = json.loads((run / "results.json").read_text())
    cm = np.asarray(res[split]["confusion_matrix"], dtype=float)
    cm_n = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.5, 4.4))
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    short = [c.replace("_", " ")[:14] for c in CLASS_NAMES]
    ax.set_xticks(range(len(short)), short, rotation=90, fontsize=7)
    ax.set_yticks(range(len(short)), short, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=8.5)
    ax.set_ylabel("True", fontsize=8.5)
    ax.set_title(title, fontsize=9.5)
    for i in range(len(short)):
        for j in range(len(short)):
            if cm_n[i, j] >= 0.01:
                ax.text(j, i, f"{cm_n[i,j]*100:.0f}", ha="center", va="center", fontsize=5.0,
                        color="white" if cm_n[i, j] > 0.55 else "#243b6b")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Row-normalised rate", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)


def fig_prediction_grid(run, split, fname, n=6, seed=0):
    data = np.load(run / f"pred_{split}.npz", allow_pickle=True)
    probs, paths = data["probs"], data["paths"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), n, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for ax, i in zip(axes.ravel(), idx):
        img = Image.open(paths[i]).convert("RGB").resize((260, 260))
        ax.imshow(img)
        top = topk_with_calories(probs[i], 3)
        ax.set_title(f"{top[0]['zh']} ({top[0]['prob']*100:.0f}%)\n{top[0]['kcal_low']}-{top[0]['kcal_high']} kcal",
                     fontsize=7.5)
        ax.axis("off")
    fig.suptitle("Photo -> Top-1 dish and estimated calorie range", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.layer = model, layer
        self.acts, self.grads = None, None
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.acts = o.detach()

    def _bwd(self, m, gi, go):
        self.grads = go[0].detach()

    def __call__(self, x, cls):
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, cls].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1))[0]
        cam = cam - cam.min()
        return (cam / (cam.max() + 1e-8)).cpu().numpy()


def fig_gradcam(run, split, fname, n=4, seed=3):
    device = pick_device()
    ckpt = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    layer = model.layer4[-1] if hasattr(model, "layer4") else model.features[-1][-1]

    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 256 / 224)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    data = np.load(run / f"pred_{split}.npz", allow_pickle=True)
    paths = data["paths"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paths), n, replace=False)
    cam = GradCAM(model, layer)
    fig, axes = plt.subplots(2, n, figsize=(7.2, 3.9))
    for k, i in enumerate(idx):
        pil = Image.open(paths[i]).convert("RGB")
        x = tf(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(model(x).argmax(1))
        arr = cam(x, pred)
        axes[0, k].imshow(pil.resize((224, 224)))
        axes[0, k].axis("off")
        up = np.asarray(Image.fromarray((arr * 255).astype(np.uint8)).resize((224, 224), Image.BICUBIC)) / 255.0
        axes[1, k].imshow(pil.resize((224, 224)))
        axes[1, k].imshow(up, cmap="jet", alpha=0.45)
        axes[1, k].axis("off")
    axes[0, 0].set_ylabel("Input", fontsize=8)
    fig.suptitle("Grad-CAM: regions supporting the top-1 class", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)


def _reliability(run, split, temperature=1.0):
    data = np.load(run / f"pred_{split}.npz", allow_pickle=True)
    probs, labels = data["probs"], data["labels"]
    if temperature != 1.0:
        # probs are softmax(logits), so softmax(log p / T) == softmax(logits / T)
        logits = np.log(np.clip(probs, 1e-12, 1.0)) / temperature
        probs = np.exp(logits - logits.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == labels).astype(float)
    bins = np.linspace(0, 1, 11)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 20:
            xs.append(conf[m].mean())
            ys.append(acc[m].mean())
    return np.array(xs), np.array(ys), conf, acc


def fig_calibration(runs, fname="calibration.png"):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    for ax, (split, title) in zip(axes, [("test", "(a) Food-101 test (in-domain)"),
                                         ("canteen", "(b) Canteen proxy (cross-domain)")]):
        ax.plot([0, 1], [0, 1], "--", color="#9aa4b2", lw=1, label="perfect calibration")
        for i, run in enumerate(runs):
            res = json.loads((run / "results.json").read_text())
            xs, ys, conf, acc = _reliability(run, split)
            ece = res[split]["ece"]
            ax.plot(xs, ys, marker="o", ms=3.2, lw=1.2, color=PALETTE[i],
                    label=f"{label_of(run)} (ECE {ece:.3f})")
        calib = runs[3] / "calibration.json"
        if calib.exists():
            info = json.loads(calib.read_text())
            t = float(info["temperature"])
            ece_t = info[split]["after"]["ece"]
            xs, ys, _, _ = _reliability(runs[3], split, temperature=t)
            ax.plot(xs, ys, ls="--", lw=1.2, color=PALETTE[3], alpha=0.9,
                    label=f"{label_of(runs[3])}, calibrated (ECE {ece_t:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Confidence")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("Accuracy")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=6.6, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.26, 1, 1))
    fig.savefig(FIG / fname)
    plt.close(fig)


def fig_latency(fname="complexity.png"):
    path = ROOT / "results/tables/complexity.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())
    seen = {}
    for r in rows:
        seen.setdefault(r["arch"], r)
    rows = list(seen.values())
    labels = ["ResNet18" if r["arch"] == "resnet18" else "MobileNetV3-S" for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    x = np.arange(len(rows))
    pv = [r["params_m"] for r in rows]
    mv = [r["macs_g"] for r in rows]
    b1 = axes[0].bar(x - 0.2, pv, 0.38, color="#2f6df6", label="Params (M)")
    b2 = axes[0].bar(x + 0.2, mv, 0.38, color="#7a5af8", label="MACs (G)")
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.03, 40)
    for bars, vals, fmt in ((b1, pv, "{:.2f}"), (b2, mv, "{:.02f}")):
        for rect, v in zip(bars, vals):
            axes[0].text(rect.get_x() + rect.get_width() / 2, v * 1.15, fmt.format(v),
                         ha="center", fontsize=6.5)
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    axes[0].set_ylabel("Count (log)")
    axes[0].set_title("(a) Model complexity")
    axes[1].bar(x - 0.2, [r["gpu_latency_ms_mean"] for r in rows], 0.38, color="#16a34a", label="GPU")
    axes[1].bar(x + 0.2, [r["cpu_latency_ms_mean"] for r in rows], 0.38, color="#e0562a", label="CPU")
    axes[1].set_xticks(x, labels)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Latency per image (ms, log)")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].set_title("(b) Batch-1 latency")
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)


def fig_architecture(fname="architecture.png"):
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.axis("off")
    boxes = [
        ("Input\n224x224", "#e8efff"),
        ("ResNet18\nconv1-layer3\n(frozen)", "#dce6f8"),
        ("layer4\n(fine-tuned)", "#bfd4fb"),
        ("GAP + FC\n(20 classes)", "#9fc0f8"),
        ("Softmax\nTop-3", "#7aa5f2"),
        ("Calorie\ninterval", "#f6d6c2"),
    ]
    w, h, gap = 1.55, 0.95, 0.28
    for i, (txt, col) in enumerate(boxes):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                    fc=col, ec="#5b6b82", lw=1))
        ax.text(x + w / 2, h / 2, txt, ha="center", va="center", fontsize=7.2)
        if i < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, h / 2), (x + w + gap - 0.02, h / 2),
                                         arrowstyle="-|>", mutation_scale=9, color="#5b6b82"))
    ax.set_xlim(-0.1, len(boxes) * (w + gap))
    ax.set_ylim(-0.15, h + 0.15)
    fig.tight_layout()
    fig.savefig(FIG / fname)
    plt.close(fig)


def write_tables(runs):
    import csv

    out = ROOT / "results/tables"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in runs:
        r = json.loads((run / "results.json").read_text())
        rows.append({
            "run": run.name,
            "mode": r["mode"],
            "color_aug": r["color_aug"],
            "best_epoch": r["best"]["epoch"],
            "val_acc": r["val"]["accuracy"],
            "test_acc": r["test"]["accuracy"],
            "test_macro_f1": r["test"]["macro_f1"],
            "test_ece": r["test"]["ece"],
            "canteen_acc": r["canteen"]["accuracy"],
            "canteen_macro_f1": r["canteen"]["macro_f1"],
            "canteen_ece": r["canteen"]["ece"],
            "drop_acc": r["test"]["accuracy"] - r["canteen"]["accuracy"],
        })
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("table ->", out / "summary.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="resnet18_last_stage_aug")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_runs("*")
    if not runs:
        print("no runs found")
        return
    best = RUNS / args.best
    fig_architecture()
    fig_training_curves(runs)
    fig_comparison(runs)
    if (best / "results.json").exists():
        fig_confusion(best, "test", "confusion_test.png", "Confusion matrix - Food-101 test")
        fig_confusion(best, "canteen", "confusion_canteen.png", "Confusion matrix - canteen proxy")
        fig_prediction_grid(best, "canteen", "predictions_canteen.png")
        fig_gradcam(best, "canteen", "gradcam_canteen.png")
    fig_calibration(runs)
    fig_latency()
    write_tables(runs)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
