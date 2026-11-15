"""Would the 20-class model fix the 116-class model's real-photo mistakes?

Dumps the wide model's restricted top-5 for each real canteen photo, then joins
with the cached E7 / five-model predictions in real_photos_views.json to see how
many of the wide model's errors a rerank could recover.

    python results/_wide_rerank.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.classes_wide import WIDE_CLASS_NAMES, WIDE_NUM_CLASSES  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from scripts.train_wide import tta_transforms  # noqa: E402

CKPT = ROOT / "results/runs/resnet18_food101_cn116/best.pt"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
REAL = ROOT / "data/splits/canteen_real"


@torch.no_grad()
def predict(model, device) -> list[dict]:
    tf = tta_transforms()
    subset = [WIDE_CLASS_NAMES.index(c) for c in CLASS_NAMES]
    rows = []
    paths = []
    for folder in sorted(REAL.iterdir()):
        if folder.is_dir() and folder.name in CLASS_NAMES:
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in EXTS:
                    paths.append((p, folder.name))
    for start in range(0, len(paths), 24):
        chunk = paths[start:start + 24]
        views = [torch.stack([t(Image.open(p).convert("RGB")) for t in tf]) for p, _ in chunk]
        views = torch.stack(views).to(device)
        b, v, c, h, w = views.shape
        logits = model(views.view(b * v, c, h, w)).view(b, v, -1).mean(1)
        sub = logits[:, subset]
        probs = sub.softmax(1)
        top5 = probs.topk(5, 1)
        for i, (p, label) in enumerate(chunk):
            rows.append({
                "path": str(p.relative_to(ROOT)),
                "label": label,
                "top1": CLASS_NAMES[top5.indices[i, 0].item()],
                "top5": [CLASS_NAMES[j] for j in top5.indices[i].tolist()],
                "p_top1": round(float(top5.values[i, 0]), 4),
            })
    return rows


def main() -> None:
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ckpt.get("arch", "resnet18"), WIDE_NUM_CLASSES)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    rows = predict(model, device)

    cached = json.loads((ROOT / "results/tables/real_photos_views.json").read_text(encoding="utf-8"))
    by_path = {r["path"].replace("/", "\\"): r for r in cached["single_dish"]["rows"]}
    wide_ok = sum(r["top1"] == r["label"] for r in rows)
    fixes_e7 = fixes_ens = 0
    errs = []
    for r in rows:
        if r["top1"] == r["label"]:
            continue
        c = by_path.get(r["path"]) or {}
        e7 = c.get("E7")
        e7 = e7.get("fused") if isinstance(e7, dict) else e7
        ens = c.get("all5")
        ens = ens.get("fused") if isinstance(ens, dict) else ens
        if e7 == r["label"]:
            fixes_e7 += 1
        if ens == r["label"]:
            fixes_ens += 1
        errs.append({"path": r["path"], "label": r["label"], "wide": r["top1"],
                     "p": r["p_top1"], "E7": e7, "all5": ens, "in_top5": r["label"] in r["top5"]})
    out = {"n": len(rows), "wide_top1": wide_ok, "wide_errors": len(errs),
           "e7_would_fix": fixes_e7, "ens_would_fix": fixes_ens, "errors": errs}
    (ROOT / "results/tables/wide_real_preds.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "errors"}, ensure_ascii=False), flush=True)
    for e in errs:
        print(f"  {e['path']:52s} true={e['label']:18s} wide={e['wide']:18s} p={e['p']:.2f} "
              f"E7={e['E7']} all5={e['all5']} in_top5={e['in_top5']}", flush=True)


if __name__ == "__main__":
    main()
