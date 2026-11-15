"""Two read-only analyses on the new real-photo batch: gate threshold sweep, mixed P/R."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path(r"E:\项目作业\food_project")
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from foodcls.utils import pick_device  # noqa: E402

T_SINGLE = 0.7364
NORM = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def tf(resize_to=None, square=False):
    ops = ([transforms.Resize((IMG_SIZE, IMG_SIZE))] if square
           else [transforms.Resize(int(IMG_SIZE * resize_to / 224)), transforms.CenterCrop(IMG_SIZE)])
    return transforms.Compose(ops + NORM)


GATE_TFS = [tf(256), tf(square=True), tf(320)]
SIX = [tf(256), tf(square=True), tf(320), tf(256), tf(square=True), tf(320)]


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def load(run, device):
    ckpt = torch.load(ROOT / "results/runs" / run / "best.pt", map_location="cpu", weights_only=False)
    m = build_model(ckpt["arch"], len(CLASS_NAMES), pretrained=False)
    m.load_state_dict(ckpt["model"])
    m.to(device).eval()
    return m, ckpt


@torch.no_grad()
def robust_scores(model, paths, device):
    out = []
    for p in paths:
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        x = torch.stack([t(img) for t in GATE_TFS]).to(device)
        prob = softmax(model(x).float().cpu().numpy() / T_SINGLE)
        out.append(float(min(prob[v].max() for v in range(len(GATE_TFS)))))
    return np.array(out)


def files(folder):
    return sorted(p for p in folder.rglob("*.jpg"))


def main():
    device = pick_device()
    model, _ = load("resnet18_last_stage_aug_f101n", device)

    real = files(ROOT / "data/splits/canteen_real")
    proxy = files(ROOT / "data/splits/canteen")
    nonfood = [ROOT / "data/splits/negatives" / f for f in
               ("005.jpg", "006.jpg", "007.jpg", "008.jpg", "010.jpg", "011.jpg",
                "012.jpg", "013.jpg", "014.jpg")]
    oov_food = [ROOT / "data/splits/negatives" / f for f in
                ("001.jpg", "002.jpg", "003.jpg", "004.jpg", "009.jpg", "015.jpg")]

    s_real = robust_scores(model, real, device)
    s_proxy = robust_scores(model, proxy, device)
    s_nonfood = robust_scores(model, nonfood, device)
    s_oov = robust_scores(model, oov_food, device)

    print("robust score distribution")
    for name, s in (("real web", s_real), ("proxy food", s_proxy),
                    ("non-food", s_nonfood), ("oov food", s_oov)):
        print(f"  {name:12} n={len(s):4}  min={s.min():.2f} p10={np.percentile(s,10):.2f} "
              f"median={np.median(s):.2f} p90={np.percentile(s,90):.2f} max={s.max():.2f}")

    print()
    print("threshold sweep: non-food accepted (want 0) | real accepted | proxy accepted")
    rows = []
    for th in np.arange(0.20, 0.72, 0.02):
        nf = int((s_nonfood >= th).sum())
        rr = int((s_real >= th).sum())
        pr = int((s_proxy >= th).sum())
        rows.append({"threshold": round(float(th), 2), "nonfood_accepted": nf,
                     "real_accepted": rr, "proxy_accepted": pr})
    for r in rows:
        mark = "  <- current" if abs(r["threshold"] - 0.32) < 1e-9 else ""
        print(f"  {r['threshold']:.2f}  {r['nonfood_accepted']}/9  {r['real_accepted']}/86  "
              f"{r['proxy_accepted']}/1000{mark}")

    ok = [r for r in rows if r["nonfood_accepted"] == 0]
    print()
    print("lowest thresholds that reject every non-food image:")
    for r in ok[:5]:
        print(f"  {r['threshold']:.2f}: keeps {r['real_accepted']}/86 real, {r['proxy_accepted']}/1000 proxy")

    # ---------------- mixed P/R with mapped labels ----------------
    mm, mckpt = load("resnet18_multilabel_aug", device)
    thresh = float(mckpt.get("threshold", 0.5))
    class_th = {str(k): float(v) for k, v in (mckpt.get("class_thresholds") or {}).items()}
    mapping = {"chicken wings": "chicken_wings", "pancakes": "pancakes", "waffles": "waffles",
               "ice cream": "ice_cream", "french fries": "french_fries", "pizza": "pizza",
               "dumplings": "dumplings", "fried rice": "fried_rice", "ramen": "ramen",
               "donuts": "donuts", "lasagna": "lasagna", "hamburger": "hamburger",
               "hot dog": "hot_dog", "sushi": "sushi", "steak": "steak", "salad": "caesar_salad"}
    mixed_dir = ROOT / "data/splits/mixed_real"
    labels = {}
    with open(mixed_dir / "labels.csv", newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            labels[row["filename"]] = [x.strip() for x in row["dishes"].split(";") if x.strip()]

    tp = fp = fn = 0
    per_plate = []
    with torch.no_grad():
        for p in sorted(mixed_dir.glob("*.jpg")):
            img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
            z = torch.stack([t(img) for t in SIX]).to(device)
            prob = 1.0 / (1.0 + np.exp(-mm(z).float().cpu().numpy().mean(0)))
            pred = []
            for i in np.argsort(-prob):
                c = CLASS_NAMES[int(i)]
                cut = max(float(class_th.get(c, thresh)), 0.10)
                if prob[i] >= cut and len(pred) < 3:
                    pred.append(c)
            truth = sorted({mapping[t] for t in labels.get(p.name, []) if t in mapping})
            tp += len(set(pred) & set(truth))
            fp += len(set(pred) - set(truth))
            fn += len(set(truth) - set(pred))
            per_plate.append({"file": p.name, "truth": truth, "pred": pred})
            print(f"  {p.name} truth={truth} pred={pred}")
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    print(f"mapped-label mixed P/R/F1 over {len(per_plate)} plates: "
          f"{prec*100:.1f} / {rec*100:.1f} / {f1*100:.1f} (tp={tp} fp={fp} fn={fn})")

    (ROOT / "results/tables/gate_real_probe.json").write_text(json.dumps(
        {"sweep": rows, "scores": {"real": s_real.tolist(), "proxy": s_proxy.tolist(),
                                   "nonfood": s_nonfood.tolist(), "oov_food": s_oov.tolist()},
         "mixed": {"per_plate": per_plate, "precision": prec, "recall": rec, "f1": f1}},
        indent=2), encoding="utf-8")
    print("wrote results/tables/gate_real_probe.json")


if __name__ == "__main__":
    main()
