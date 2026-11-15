"""Score a checkpoint on real canteen photos taken with a phone.

Two accepted layouts:

1. one sub-folder per class, named with the English class name:
       real_photos/fried_rice/IMG_1234.jpg
       real_photos/dumplings/IMG_1235.jpg

2. a flat folder plus a CSV manifest with columns `path,label`:
       python scripts/eval_real_photos.py --dir real_photos --manifest labels.csv

The report is intentionally small-sample friendly: it prints n, top-1, top-3,
per-class counts and every misclassified file, and writes JSON to
results/tables/real_photos.json.  Nothing here is used for training or model
selection; it is a held-out probe of the deployed model only.

Example:
    python scripts/eval_real_photos.py --dir ..\real_canteen_photos
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from foodcls.utils import pick_device  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic"}


def collect(folder: Path, manifest: Path | None):
    items: list[tuple[Path, str]] = []
    if manifest is not None:
        with open(manifest, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                label = (row.get("label") or "").strip()
                path = Path((row.get("path") or "").strip())
                if not path.is_absolute():
                    path = folder / path
                if label in CLASS_NAMES and path.is_file():
                    items.append((path, label))
    else:
        for cls_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            if cls_dir.name not in CLASS_NAMES:
                print(f"skip unknown class folder: {cls_dir.name}")
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in EXTS:
                    items.append((p, cls_dir.name))
    return items


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="folder with per-class sub-folders or photos")
    ap.add_argument("--manifest", default=None, help="optional CSV with path,label")
    ap.add_argument("--ckpt", default=str(ROOT / "results/runs/resnet18_last_stage_aug/best.pt"))
    ap.add_argument("--out", default=str(ROOT / "results/tables/real_photos.json"))
    args = ap.parse_args(argv)

    items = collect(Path(args.dir).resolve(), Path(args.manifest).resolve() if args.manifest else None)
    if not items:
        print("no labelled photos found -- expected <dir>/<class_name>/*.jpg or a path,label CSV")
        return 1

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model(ckpt["arch"], len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # Match the app: it reports calibrated probabilities from the fused
    # temperature stored next to the checkpoint.  Ranking is unchanged, only the
    # confidence values move, but the report should quote what users see.
    temperature = 1.0
    cal_path = Path(args.ckpt).parent / "calibration.json"
    if cal_path.is_file():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        temperature = float(cal.get("temperature_fused", cal.get("temperature", 1.0)))

    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 256 / 224)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    rows = []
    with torch.no_grad():
        for path, label in items:
            try:
                img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            except Exception as exc:  # unreadable file
                rows.append({"path": str(path), "label": label, "error": str(exc)})
                continue
            logits = model(tf(img).unsqueeze(0).to(device)).float()
            probs = torch.softmax(logits / temperature, 1)[0].cpu().numpy()
            order = np.argsort(-probs)[:3]
            rows.append(
                {
                    "path": str(path),
                    "label": label,
                    "pred": CLASS_NAMES[int(order[0])],
                    "top3": [CLASS_NAMES[int(i)] for i in order],
                    "conf": float(probs[order[0]]),
                }
            )

    scored = [r for r in rows if "pred" in r]
    n = len(scored)
    top1 = sum(r["pred"] == r["label"] for r in scored) / max(1, n)
    top3 = sum(r["label"] in r["top3"] for r in scored) / max(1, n)
    per_class = Counter(r["label"] for r in scored)
    misses = [r for r in scored if r["pred"] != r["label"]]

    print(f"photos={n}  top-1={top1*100:.1f}%  top-3={top3*100:.1f}%")
    print("per class:", dict(sorted(per_class.items())))
    if misses:
        print("misclassified:")
        for r in misses:
            print(f"  {Path(r['path']).name:40s} true={r['label']:18s} pred={r['pred']:18s} p={r['conf']:.2f}")
    else:
        print("all photos classified correctly")

    payload = {
        "checkpoint": str(args.ckpt),
        "n": n,
        "top1": top1,
        "top3": top3,
        "per_class_n": dict(sorted(per_class.items())),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
