"""Score a candidate wide checkpoint everywhere the deployed one is scored.

The deployed 116-class head was trained from a plain Food-101 resnet18.  The
20-class demo now rides on the Food-101N clean-label backbone
(resnet18_last_stage_aug_f101n), so the obvious experiment is to retrain the wide
head from that same backbone: one backbone family for the whole app, and a
stronger starting point for the 15 Chinese classes that only have a few hundred
photos each.

This script puts a candidate checkpoint through the same measurements the
deployed one already has, so the numbers can be compared line by line:

* data/wide/splits/test        - 116-class held-out split (Food-101 + Chinese)
* data/splits/test             - the 20-class course protocol, restricted to 20
* data/splits/canteen_clean    - canteen photos, restricted to 20
* data/splits/canteen_real     - 100 hand-labelled real canteen photographs

    python scripts/eval_wide_candidate.py --ckpt results/runs/resnet18_cn116_f101n/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.classes_wide import WIDE_CLASS_NAMES, WIDE_NUM_CLASSES  # noqa: E402
from foodcls.data import SplitDataset, build_transforms  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from foodcls.utils import classification_metrics, expected_calibration_error  # noqa: E402
from scripts.train_wide import evaluate_tta, predict, predict_logits, tta_transforms  # noqa: E402


def load(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt.get("arch", "resnet18"), len(ckpt.get("class_names") or WIDE_CLASS_NAMES),
                        pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    calib = ckpt_path.parent / "calibration.json"
    temperature = json.loads(calib.read_text("utf-8"))["temperature"] if calib.is_file() else 1.0
    return model, float(temperature), ckpt


@torch.no_grad()
def score_dir(model, root: Path, device, temperature: float, batch_size: int = 24) -> dict:
    """Restricted-20 and full-116 top-1 on a folder-per-class tree."""
    from PIL import Image

    tf = tta_transforms()
    items = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name not in CLASS_NAMES:
            continue
        label = CLASS_NAMES.index(folder.name)
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                items.append((p, label))
    if not items:
        raise SystemExit(f"no labelled photographs under {root}")

    subset = torch.tensor([WIDE_CLASS_NAMES.index(c) for c in CLASS_NAMES], device=device)
    label_in_wide = torch.tensor([WIDE_CLASS_NAMES.index(c) for c in CLASS_NAMES], device=device)
    n = correct_full = correct_sub = top5_sub = 0
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        views = torch.stack([torch.stack([t(Image.open(p).convert("RGB")) for t in tf]) for p, _ in chunk])
        views = views.to(device)
        b, v, c, h, w = views.shape
        logits = model(views.view(b * v, c, h, w)).float().view(b, v, -1).mean(1) / max(temperature, 1e-3)
        labels = torch.tensor([lab for _, lab in chunk], device=device)
        correct_full += (logits.argmax(1) == label_in_wide[labels]).sum().item()
        sub_logits = logits.index_select(1, subset)
        correct_sub += (sub_logits.argmax(1) == labels).sum().item()
        top5_sub += (sub_logits.topk(5, dim=1).indices == labels[:, None]).any(1).sum().item()
        n += b
    return {"n": n, "top1_restricted": round(correct_sub / n, 4),
            "top5_restricted": round(top5_sub / n, 4),
            "top1_full116": round(correct_full / n, 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/runs/resnet18_cn116_f101n/best.pt")
    ap.add_argument("--out", default="results/tables/wide_candidate.json")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--skip-test", action="store_true", help="skip the 25,830-image wide test split")
    ap.add_argument("--skip-tta", action="store_true", help="skip only the slow three-crop pass on the test split")
    args = ap.parse_args()

    ckpt_path = ROOT / args.ckpt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, temperature, ckpt = load(ckpt_path, device)
    out = {"ckpt": str(ckpt_path.relative_to(ROOT)), "arch": ckpt.get("arch"),
           "epoch": ckpt.get("epoch"), "temperature": round(temperature, 4),
           "val_macro_f1": ckpt.get("val_macro_f1"), "val_acc": ckpt.get("val_acc")}
    print(f"{out['ckpt']} epoch {out['epoch']} T={temperature:.3f}", flush=True)

    wide_test = ROOT / "data/wide/splits/test"
    if not args.skip_test:
        for tta in ((False,) if args.skip_tta else (False, True)):
            res = evaluate_tta(model, wide_test, device, args.batch_size, tta=tta)
            out["test" if not tta else "test_tta"] = res
            print(f"wide test {'3-crop' if tta else 'single'}: top1 {res['top1']*100:.2f} "
                  f"top5 {res['top5']*100:.2f}", flush=True)

        ds = SplitDataset(wide_test, WIDE_CLASS_NAMES, build_transforms(False, False))
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        labels, logits = predict_logits(model, loader, device)
        probs = torch.softmax(torch.tensor(logits) / max(temperature, 1e-3), 1).numpy()
        met = classification_metrics(labels, probs, WIDE_CLASS_NAMES)
        pred = probs.argmax(1)
        out["test_metrics"] = {"accuracy": met["accuracy"], "macro_f1": met["macro_f1"],
                               "ece": expected_calibration_error(labels, probs)}

        def group(ids):
            mask = np.isin(labels, ids)
            return {"n": int(mask.sum()),
                    "top1": round(float((pred[mask] == labels[mask]).mean()), 4) if mask.any() else 0.0}

        out["test_groups"] = {"food101": group(np.arange(101)),
                              "chinese": group(np.arange(101, WIDE_NUM_CLASSES))}
        print(f"wide test calibrated: acc {met['accuracy']*100:.2f} f1 {met['macro_f1']*100:.2f} "
              f"ece {out['test_metrics']['ece']*100:.2f} | groups {out['test_groups']}", flush=True)

    for name, root in (("course_test", ROOT / "data/splits/test"),
                       ("canteen_clean", ROOT / "data/splits/canteen_clean"),
                       ("canteen_proxy", ROOT / "data/splits/canteen"),
                       ("canteen_real", ROOT / "data/splits/canteen_real")):
        if not root.is_dir():
            print(f"skip {name}: no {root}", flush=True)
            continue
        res = score_dir(model, root, device, temperature, args.batch_size)
        out[name] = res
        print(f"{name:13s} n={res['n']:5d} restricted top1 {res['top1_restricted']*100:.2f} "
              f"top5 {res['top5_restricted']*100:.2f} full116 {res['top1_full116']*100:.2f}", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
