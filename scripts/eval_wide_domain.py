"""How well does the 101-class model do on the 20-class canteen protocol?

The wide model is trained on all 101 Food-101 classes (5x the course data).  This
script scores it on exactly the same images used for the course report:

* data/splits/test          - 20-class official Food-101 test split (5,000)
* data/splits/canteen_clean - canteen photos (1,000)

Two numbers are reported.  "restricted" re-normalises the 101 logits over the 20
course classes, which is the fair comparison against the 20-class model (the
model is not punished for ranking an unseen-but-plausible class first).
"full" is the plain 101-way argmax, i.e. what the demo actually returns.

Usage:
    python scripts/eval_wide_domain.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # so "scripts.train_wide" resolves when run from anywhere

from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.classes_wide import WIDE_CLASS_NAMES, WIDE_NUM_CLASSES  # noqa: E402
from foodcls.data import SplitDataset  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from scripts.train_wide import tta_transforms  # noqa: E402


def load_views(img):
    return [t(img) for t in tta_transforms()]


class Views(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, label = self.base[i]
        return torch.stack(load_views(img)), label


@torch.no_grad()
def score(model, root: Path, device, batch_size: int = 64):
    ds = SplitDataset(root, CLASS_NAMES, transform=None)
    loader = DataLoader(Views(ds), batch_size=batch_size, shuffle=False, num_workers=0)
    subset = [WIDE_CLASS_NAMES.index(c) for c in CLASS_NAMES if c in WIDE_CLASS_NAMES]
    missing = [c for c in CLASS_NAMES if c not in WIDE_CLASS_NAMES]
    if missing:
        raise SystemExit(f"course classes missing from the wide table: {missing}")
    sub = torch.tensor(subset, device=device)
    # SplitDataset labels are positions in CLASS_NAMES; the 101-way argmax is a
    # position in WIDE_CLASS_NAMES, so map the label before comparing.
    label_in_wide = torch.tensor([WIDE_CLASS_NAMES.index(c) for c in CLASS_NAMES], device=device)
    n = correct_full = correct_sub = top5_sub = 0
    for views, labels in loader:
        views = views.to(device, non_blocking=True)
        b, v, c, h, w = views.shape
        logits = model(views.view(b * v, c, h, w)).view(b, v, -1).mean(1)
        labels = labels.to(device)
        correct_full += (logits.argmax(1) == label_in_wide[labels]).sum().item()
        sub_logits = logits.index_select(1, sub)
        pred = sub_logits.argmax(1)
        correct_sub += (pred == labels).sum().item()
        top5 = sub_logits.topk(5, dim=1).indices
        top5_sub += (top5 == labels[:, None]).any(1).sum().item()
        n += b
    return {
        "n": n,
        "top1_restricted": round(correct_sub / n, 4),
        "top5_restricted": round(top5_sub / n, 4),
        "top1_full101": round(correct_full / n, 4),
        "classes_in_wide": len(subset),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="results/runs/resnet18_food101_cn116/best.pt")
    ap.add_argument("--out", default="results/tables/wide_domain.json")
    ap.add_argument("--batch-size", type=int, default=48)
    args = ap.parse_args()

    ckpt_path = ROOT / args.ckpt
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ckpt.get("arch", "resnet18"), WIDE_NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    calib = ckpt_path.parent / "calibration.json"
    temperature = json.loads(calib.read_text())["temperature"] if calib.is_file() else 1.0

    out = {"ckpt": str(ckpt_path.relative_to(ROOT)), "temperature": temperature}
    for name, root in [("test", ROOT / "data/splits/test"),
                       ("canteen_clean", ROOT / "data/splits/canteen_clean"),
                       ("canteen_proxy", ROOT / "data/splits/canteen")]:
        res = score(model, root, device, args.batch_size)
        out[name] = res
        print(f"{name:8s} n={res['n']} restricted top1={res['top1_restricted']:.4f} "
              f"top5={res['top5_restricted']:.4f} full101={res['top1_full101']:.4f}", flush=True)

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {out_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
