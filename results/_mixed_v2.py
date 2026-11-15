"""Mixed-plate P/R with the v2 (20-class vocab) labels, printing raw sigmoid probs."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageOps

ROOT = Path(r"E:\项目作业\food_project")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "results"))
from foodcls.classes import CLASS_NAMES
from foodcls.models import build_model
from foodcls.utils import pick_device
from _eval_real import VIEWS, load_model, six_views  # reuse transforms

def main():
    device = pick_device()
    model, ck = load_model("resnet18_multilabel_aug", device)
    thresh = float(ck.get("threshold", 0.5))
    class_th = {str(k): float(v) for k, v in (ck.get("class_thresholds") or {}).items()}
    d = ROOT / "data/splits/mixed_real"
    labels = {}
    with open(d / "labels.csv", newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            labels[row["filename"]] = [x.strip() for x in row["dishes"].split(";") if x.strip()]

    tp = fp = fn = 0
    rows = []
    for p in sorted(d.glob("*.jpg")):
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        z = six_views(model, img, device)
        prob = 1.0 / (1.0 + np.exp(-z.mean(0)))
        order = np.argsort(-prob)
        top = [(CLASS_NAMES[int(i)], float(prob[i]), float(class_th.get(CLASS_NAMES[int(i)], thresh))) for i in order[:6]]
        picked = [c for c, pr, ct in top if pr >= max(ct, 0.10)]
        truth = set(labels.get(p.name, []))
        hit = set(picked[:3]) & truth
        tp += len(hit); fp += len(set(picked[:3]) - truth); fn += len(truth - set(picked[:3]))
        rows.append({"file": p.name, "truth": sorted(truth), "picked": picked[:3], "hit": sorted(hit),
                     "top": [[c, round(pr, 3), round(ct, 2)] for c, pr, ct in top]})
        print(f"{p.name} truth={sorted(truth)}")
        print("   top:", [[c, round(pr, 2), round(ct, 2)] for c, pr, ct in top[:4]], " picked:", picked[:3])

    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    print(f"\nmicro P={100*P:.1f} R={100*R:.1f} F1={100*2*P*R/max(P+R,1e-9):.1f}  tp={tp} fp={fp} fn={fn}")
    (ROOT/"results/tables/mixed_v2.json").write_text(json.dumps({"rows": rows, "tp": tp, "fp": fp, "fn": fn,
        "precision": P, "recall": R, "f1": 2*P*R/max(P+R,1e-9)}, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    main()
