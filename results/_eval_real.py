"""Score the real-photo set: six-view fusion, five-checkpoint ensemble, negatives, mixed plates."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path(r"E:\项目作业\food_project")
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE  # noqa: E402
from foodcls import packaging  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from foodcls.utils import pick_device  # noqa: E402

RUNS = ["resnet18_last_stage_aug", "resnet18_last_stage_aug_f101n",
        "resnet18_last_stage_aug_f101n_nodedup", "resnet18_last_stage_aug_f101n_e24",
        "resnet18_last_stage_aug_seed7"]
SHORT = dict(zip(RUNS, ["E4", "E7", "E8", "E12", "E13"]))
NORM = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def tf(resize_to=None, square=False, flip=False):
    ops = ([transforms.Resize((IMG_SIZE, IMG_SIZE))] if square
           else [transforms.Resize(int(IMG_SIZE * resize_to / 224)), transforms.CenterCrop(IMG_SIZE)])
    mirror = [transforms.RandomHorizontalFlip(p=1.0)] if flip else []
    return transforms.Compose(ops + mirror + NORM)


VIEWS = [tf(256), tf(square=True), tf(320), tf(256, flip=True), tf(square=True, flip=True), tf(320, flip=True)]


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def load_model(run, device):
    ckpt = torch.load(ROOT / "results/runs" / run / "best.pt", map_location="cpu", weights_only=False)
    m = build_model(ckpt["arch"], len(CLASS_NAMES), pretrained=False)
    m.load_state_dict(ckpt["model"])
    m.to(device).eval()
    return m, ckpt


def items_of(folder):
    items = []
    for cls_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        if cls_dir.name not in CLASS_NAMES:
            print("skip unknown folder", cls_dir.name)
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                items.append((p, cls_dir.name))
    return items


@torch.no_grad()
def six_views(model, img, device):
    x = torch.stack([t(img) for t in VIEWS]).to(device)
    return model(x).float().cpu().numpy()


def main():
    device = pick_device()
    models = {}
    ckpts = {}
    for r in RUNS:
        models[r], ckpts[r] = load_model(r, device)
        print("loaded", SHORT[r], flush=True)

    # ---------------- single-dish set ----------------
    items = items_of(ROOT / "data/splits/canteen_real")
    print(f"single-dish images: {len(items)}", flush=True)
    logits = defaultdict(list)
    labels = []
    for p, label in items:
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        labels.append(CLASS_NAMES.index(label))
        for r in RUNS:
            logits[r].append(six_views(models[r], img, device))
    y = np.array(labels)
    for r in RUNS:
        logits[r] = np.stack(logits[r])

    rows = []
    for k, (p, label) in enumerate(items):
        rec = {"path": str(p.relative_to(ROOT)), "label": label}
        for r in RUNS:
            rec[SHORT[r]] = {
                "single": CLASS_NAMES[int(logits[r][k, 0].argmax())],
                "fused": CLASS_NAMES[int(logits[r][k].mean(0).argmax())],
                "conf": float(softmax(logits[r][k].mean(0))[int(logits[r][k].mean(0).argmax())]),
            }
        ens = np.mean([logits[r][k].mean(0) for r in RUNS], axis=0)
        rec["all5"] = CLASS_NAMES[int(ens.argmax())]
        rec["all5_top3"] = [CLASS_NAMES[int(i)] for i in np.argsort(-ens)[:3]]
        for r in ("resnet18_last_stage_aug_f101n",):
            rec["E7_top3"] = [CLASS_NAMES[int(i)] for i in np.argsort(-logits[r][k].mean(0))[:3]]
        rows.append(rec)

    def rate(pred_key, subset=None):
        idx = [i for i, (p, l) in enumerate(items) if subset is None or l in subset]
        ok = sum(rows[i][pred_key] == CLASS_NAMES[y[i]] for i in idx)
        return ok, len(idx)

    print()

    n = len(items)
    print("per model fused accuracy on", n, "images")
    for r in RUNS:
        fused_hits = sum(rows[i][SHORT[r]]["fused"] == CLASS_NAMES[y[i]] for i in range(n))
        ok_single = sum(rows[i][SHORT[r]]["single"] == CLASS_NAMES[y[i]] for i in range(n))
        print(f"  {SHORT[r]:4} single {100*ok_single/n:5.1f}%  fused {100*fused_hits/n:5.1f}%")
    ok3 = sum(rows[i]["all5"] == CLASS_NAMES[y[i]] for i in range(n))
    print(f"  all5 fused {100*ok3/n:5.1f}%")

    print()
    print("per class (E7 fused / all5 fused):")
    per = defaultdict(lambda: [0, 0, 0])
    for i in range(n):
        c = CLASS_NAMES[y[i]]
        per[c][1] += 1
        per[c][0] += rows[i]["E7"]["fused"] == c
        per[c][2] += rows[i]["all5"] == c
    for c in CLASS_NAMES:
        if c in per:
            a, m, b = per[c]
            print(f"  {c:22} n={m:2}  E7 {a}/{m}  all5 {b}/{m}")

    print()
    print("misses (E7 fused):")
    for i in range(n):
        if rows[i]["E7"]["fused"] != rows[i]["label"]:
            print(f"  {rows[i]['path']:52} true={rows[i]['label']:18} pred={rows[i]['E7']['fused']:18} top3={rows[i]['E7_top3']}")

    # ---------------- negatives ----------------
    neg_files = sorted((ROOT / "data/splits/negatives").glob("*.jpg"))
    neg_rows = []
    for p in neg_files:
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        z = six_views(models["resnet18_last_stage_aug_f101n"], img, device)
        probs = softmax(z)  # per view, uncalibrated ordering; gate uses calibrated
        gate_probs = softmax(z / 0.7364)
        robust = float(min(gate_probs[v].max() for v in range(3)))
        pkg = packaging.signals(img)
        neg_rows.append({"file": p.name, "robust": robust, "in_scope": robust >= 0.32,
                         "top1": CLASS_NAMES[int(gate_probs[0].argmax())],
                         "packaging": pkg.get("detected", False)})
    print()
    print(f"negatives n={len(neg_rows)}  flagged in-scope={sum(r['in_scope'] for r in neg_rows)}  packaging={sum(r['packaging'] for r in neg_rows)}")
    for r in neg_rows:
        print(f"  {r['file']:20} in_scope={str(r['in_scope']):5} pkg={str(r['packaging']):5} robust={r['robust']:.2f} top1={r['top1']}")

    # ---------------- mixed plates ----------------
    mm, mckpt = load_model("resnet18_multilabel_aug", device)
    thresh = float(mckpt.get("threshold", 0.5))
    class_th = {str(k): float(v) for k, v in (mckpt.get("class_thresholds") or {}).items()}
    mapping = {"chicken wings": "chicken_wings", "pancakes": "pancakes", "waffles": "waffles",
               "ice cream": "ice_cream", "french fries": "french_fries", "pizza": "pizza",
               "dumplings": "dumplings", "fried rice": "fried_rice", "ramen": "ramen"}
    mixed_dir = ROOT / "data/splits/mixed_real"
    labels_csv = {}
    with open(mixed_dir / "labels.csv", newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            labels_csv[row["filename"]] = [x.strip() for x in row["dishes"].split(";") if x.strip()]
    mixed_rows = []
    for p in sorted(mixed_dir.glob("*.jpg")):
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        z = six_views(mm, img, device)
        prob = 1.0 / (1.0 + np.exp(-z.mean(0)))
        order = np.argsort(-prob)
        parts = []
        for i in order:
            c = CLASS_NAMES[int(i)]
            cut = float(class_th.get(c, thresh))
            if prob[i] >= max(cut, 0.10):
                parts.append((c, float(prob[i])))
        truth_raw = labels_csv.get(p.name, [])
        truth = {mapping[t] for t in truth_raw if t in mapping}
        got = {c for c, _ in parts[:3]}
        mixed_rows.append({"file": p.name, "truth_raw": truth_raw, "truth_in_vocab": sorted(truth),
                           "pred": [[c, round(p_, 3)] for c, p_ in parts[:5]],
                           "hit": len(got & truth), "miss": sorted(truth - got)})
    print()
    print("mixed plates (top parts after per-class thresholds):")
    for r in mixed_rows:
        print(f"  {r['file']}  truth={r['truth_raw']}")
        print(f"      pred={r['pred']}  in-vocab truth={r['truth_in_vocab']} hit={r['hit']} miss={r['miss']}")

    out = {
        "single_dish": {"n": n, "rows": rows,
                        "accuracy": {SHORT[r]: {"single": None, "fused": None} for r in RUNS}},
        "negatives": neg_rows,
        "mixed": mixed_rows,
    }
    for r in RUNS:
        out["single_dish"]["accuracy"][SHORT[r]]["single"] = sum(rows[i][SHORT[r]]["single"] == CLASS_NAMES[y[i]] for i in range(n)) / n
        out["single_dish"]["accuracy"][SHORT[r]]["fused"] = sum(rows[i][SHORT[r]]["fused"] == CLASS_NAMES[y[i]] for i in range(n)) / n
    out["single_dish"]["accuracy"]["all5"] = {"fused": ok3 / n}
    (ROOT / "results/tables/real_photos_views.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nwrote results/tables/real_photos_views.json")


if __name__ == "__main__":
    main()
