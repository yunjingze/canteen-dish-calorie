
"""Packaging head: a 512-d linear model that flags product-packaging photos.

The numpy heuristic in src/foodcls/packaging.py misses 62/62 supermarket pack
shots, because a photo printed on a pizza box looks like pizza.  This trains a
head on the same avgpool features the food gate uses (view 0 of the six-view
batch, so inference costs nothing extra).

Positives: 17 existing packaging negatives + 62 unique photos from the archived
v3 batch.  Negatives: 3700 food photos (train/clean/real/test).
"""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"E:\项目作业\food_project")
sys.path.insert(0, str(ROOT / "src"))
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from foodcls.classes import CLASS_NAMES
from foodcls.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from foodcls.models import build_model
from foodcls.utils import pick_device

# exactly the app's view 0: Resize(256) then a 224 centre crop
TF = transforms.Compose([transforms.Resize(int(IMG_SIZE * 256 / 224)), transforms.CenterCrop(IMG_SIZE),
                         transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
CK = ROOT/"results/runs/resnet18_last_stage_aug_f101n/best.pt"
NPZ = ROOT/"results/tables/packaging_features.npz"


def packaging_paths():
    neg = ROOT/"data/splits/negatives"
    cats = {}
    with open(neg/"sources.csv", newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            cats[Path(row["file"]).name] = row["category"]
    old = [neg/n for n, c in cats.items() if c == "packaging"]
    old.append(neg/"005.jpg")  # v1: packaging
    seen, new = set(), []
    for p in sorted((ROOT/"results/_discarded_v3").rglob("*.jpg")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h); new.append(p)
    return sorted(old), new


def food_paths():
    sp = ROOT/"data/splits"
    def take(d, n):
        ps = sorted(d.rglob("*.jpg"))
        if len(ps) <= n:
            return ps
        idx = np.linspace(0, len(ps)-1, n).round().astype(int)
        return [ps[i] for i in idx]
    return (take(sp/"train", 2000) + take(sp/"canteen_clean", 1000)
            + take(sp/"canteen_real", 100) + take(sp/"test", 600))


def feats(model, device, store, paths):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            batch = paths[i:i+64]
            x = torch.stack([TF(ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for p in batch]).to(device)
            model(x)
            out.append(store["f"].flatten(1).float().cpu().numpy())
            if i % 512 == 0:
                print("   ", i, "/", len(paths), flush=True)
    return np.concatenate(out)


def main():
    device = pick_device()
    print("device", device, flush=True)
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(ck["model"]); model.to(device).eval()
    store = {}
    model.avgpool.register_forward_hook(lambda m, i, o: store.__setitem__("f", o))

    old, new = packaging_paths()
    food = food_paths()
    print(f"packaging old={len(old)} v3={len(new)} food={len(food)}", flush=True)
    X_old = feats(model, device, store, old)
    X_new = feats(model, device, store, new)
    X_food = feats(model, device, store, food)
    np.savez(NPZ, X_old=X_old, X_new=X_new, X_food=X_food,
             old_names=np.array([str(p.relative_to(ROOT)) for p in old]),
             new_names=np.array([str(p.relative_to(ROOT)) for p in new]),
             food_names=np.array([str(p.relative_to(ROOT)) for p in food]))
    print("saved", NPZ, X_old.shape, X_new.shape, X_food.shape, flush=True)


main()
