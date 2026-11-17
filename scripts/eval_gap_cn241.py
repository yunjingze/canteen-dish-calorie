# -*- coding: utf-8 -*-
"""What do the shipped models say about real canteen dishes?

Runs the two production checkpoints (20-class canteen, 116-class wide) over the
CNFOOD-241 test split of the 20 everyday dishes and reports how confident they
are while being necessarily wrong -- none of these 20 labels exist in either
label space.  This quantifies the gap the user reported: a plate of tomato and
egg comes back as mapo tofu, at high confidence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.classes_wide import WIDE_CLASS_NAMES  # noqa: E402
from foodcls.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from foodcls.utils import pick_device  # noqa: E402

CKPTS = {
    "canteen20": ROOT / "results/runs/resnet18_last_stage_aug/best.pt",
    "wide116": ROOT / "results/runs/resnet18_food101_cn116/best.pt",
}
TEST = ROOT / "data/splits/cn241/test"
TABLE = json.loads((ROOT / "results/tables/cn241_classes.json").read_text(encoding="utf-8"))


def views(path):
    from torchvision import transforms

    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    tfs = [
        transforms.Compose([transforms.Resize(int(IMG_SIZE * 256 / 224)), transforms.CenterCrop(IMG_SIZE)] + norm),
        transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE))] + norm),
        transforms.Compose([transforms.Resize(int(IMG_SIZE * 320 / 224)), transforms.CenterCrop(IMG_SIZE)] + norm),
    ]
    with open(path, "rb") as fh:
        img = Image.open(fh).convert("RGB")
    return torch.stack([tf(img) for tf in tfs])


def main():
    device = pick_device()
    files = []
    for c in TABLE["classes"]:
        d = TEST / c["slug"]
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                files.append((c, p))
    print(f"{len(files)} test images, {len(TABLE['classes'])} classes", flush=True)

    batch = 4  # 3 views x batch fits the 8 GB card alongside other jobs
    report = {}
    for tag, path in CKPTS.items():
        if not path.is_file():
            print(f"missing {path}", flush=True)
            continue
        ckpt = torch.load(path, map_location=device, weights_only=False)
        names = ckpt.get("class_names") or (CLASS_NAMES if tag == "canteen20" else WIDE_CLASS_NAMES)
        model = build_model(ckpt["arch"], len(names), pretrained=False)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        top1, top1p, ents, per_class = [], [], [], {}
        for start in range(0, len(files), batch):
            chunk = files[start:start + batch]
            x = torch.stack([views(p) for _, p in chunk]).to(device)  # B,V,C,H,W
            b, v = x.shape[:2]
            with torch.no_grad():
                logits = model(x.flatten(0, 1)).float().view(b, v, -1).mean(1)
                probs = torch.softmax(logits, 1).cpu().numpy()
            for (cls, _), pr in zip(chunk, probs):
                idx = int(pr.argmax())
                top1.append(names[idx])
                top1p.append(float(pr[idx]))
                per_class.setdefault(cls["slug"], []).append(float(pr[idx]))
        top1p = np.array(top1p)
        counts = {}
        for t in top1:
            counts[t] = counts.get(t, 0) + 1
        report[tag] = {
            "classes": len(names),
            "mean_top1_prob": round(float(top1p.mean()), 4),
            "share_above_0.5": round(float((top1p > 0.5).mean()), 4),
            "share_above_0.8": round(float((top1p > 0.8).mean()), 4),
            "top_predictions": dict(sorted(counts.items(), key=lambda kv: -kv[1])[:10]),
            "mean_prob_per_true_class": {k: round(float(np.mean(v)), 4) for k, v in per_class.items()},
        }
        print(f"{tag}: mean p {report[tag]['mean_top1_prob']:.3f} "
              f">0.5 {report[tag]['share_above_0.5']*100:.1f}% "
              f">0.8 {report[tag]['share_above_0.8']*100:.1f}%", flush=True)

    out = ROOT / "results/tables/cn241_gap.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
