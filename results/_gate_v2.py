"""Gate analysis v2: robust-score separation with 138 real negatives, per-category FPR."""
from __future__ import annotations
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path(r"E:\项目作业\food_project")
sys.path.insert(0, str(ROOT / "src"))
from foodcls.classes import CLASS_NAMES
from foodcls.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from foodcls.models import build_model
from foodcls.utils import pick_device

NORM = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
def tf(resize_to=None, square=False):
    ops = ([transforms.Resize((IMG_SIZE, IMG_SIZE))] if square
           else [transforms.Resize(int(IMG_SIZE * resize_to / 224)), transforms.CenterCrop(IMG_SIZE)])
    return transforms.Compose(ops + NORM)
VIEWS = [tf(256), tf(square=True), tf(320)]

def softmax(z):
    z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

def load():
    ck = torch.load(ROOT/"results/runs/resnet18_last_stage_aug_f101n/best.pt", map_location="cpu", weights_only=False)
    m = build_model(ck["arch"], len(CLASS_NAMES), pretrained=False); m.load_state_dict(ck["model"])
    return m.to(pick_device()).eval(), float(ck.get("temperature") or 0.7364)

@torch.no_grad()
def robust_of(m, p, T, device):
    img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    x = torch.stack([t(img) for t in VIEWS]).to(device)
    probs = softmax(m(x).float().cpu().numpy() / T)
    return float(min(probs[v].max() for v in range(len(VIEWS))))

def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg]); lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(allv); r = np.empty(len(allv)); r[o] = np.arange(1, len(allv)+1)
    # average ranks for ties
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq)); np.add.at(sums, inv, r)
    r = (sums / cnt)[inv]
    return float((r[lab == 1].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))

def main():
    device = pick_device(); model, T = load()
    cats = {}
    with open(ROOT/"data/splits/negatives/sources.csv", newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            cats[Path(row["file"]).name] = row["category"]

    neg = []
    for p in sorted((ROOT/"data/splits/negatives").glob("*.jpg")):
        neg.append({"file": p.name, "cat": cats.get(p.name, "v1_unknown"), "robust": robust_of(model, p, T, device)})
    real = []
    for p in sorted((ROOT/"data/splits/canteen_real").rglob("*.jpg")):
        real.append({"file": str(p.relative_to(ROOT/"data/splits")), "robust": robust_of(model, p, T, device)})
    print("scored", len(neg), "negatives", len(real), "real", flush=True)

    proxy = json.load(open(ROOT/"results/tables/gate_real_probe.json", encoding="utf-8"))["scores"]["proxy"]
    nr = np.array([x["robust"] for x in neg]); rr = np.array([x["robust"] for x in real]); pr = np.array(proxy)

    print(f"medians  real {np.median(rr):.3f}  proxy {np.median(pr):.3f}  negatives {np.median(nr):.3f}")
    print(f"AUC(real vs neg) {auc(rr, nr):.3f}   AUC(proxy vs neg) {auc(pr, nr):.3f}   AUC(real+proxy vs neg) {auc(np.r_[rr,pr], nr):.3f}")

    by = defaultdict(list)
    for x in neg: by[x["cat"]].append(x["robust"])
    print("\nper-category acceptance (robust >= 0.32):")
    for c in sorted(by, key=lambda k: -np.mean(np.array(by[k]) >= 0.32)):
        a = np.array(by[c]); print(f"  {c:18} n={len(a):3}  accepted {int((a>=0.32).sum()):3}  median {np.median(a):.2f}  max {a.max():.2f}")

    print("\nsweep:")
    for t in (0.32, 0.4, 0.5, 0.6, 0.7):
        print(f"  thr {t:.2f}  real {int((rr>=t).sum()):3}/{len(rr)}  proxy {int((pr>=t).sum()):4}/{len(pr)}  neg accepted {int((nr>=t).sum()):3}/{len(nr)}")

    out = {"T": T, "negatives": neg, "real": real,
           "auc": {"real_vs_neg": auc(rr, nr), "proxy_vs_neg": auc(pr, nr), "food_vs_neg": auc(np.r_[rr, pr], nr)},
           "per_category": {c: {"n": len(v), "accepted": int((np.array(v) >= 0.32).sum()),
                                "median": float(np.median(v))} for c, v in by.items()}}
    (ROOT/"results/tables/gate_v2.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwrote results/tables/gate_v2.json")

if __name__ == "__main__":
    main()
