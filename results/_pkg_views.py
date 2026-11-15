
"""Per-view features for the packaging head (3 framings, no mirrors)."""
from __future__ import annotations
import csv, hashlib, sys
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

def tfs():
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    def view(resize_to=None, square=False):
        ops = ([transforms.Resize((IMG_SIZE, IMG_SIZE))] if square
               else [transforms.Resize(int(IMG_SIZE * resize_to / 224)), transforms.CenterCrop(IMG_SIZE)])
        return transforms.Compose(ops + norm)
    return [view(256), view(square=True), view(320)]

TF = tfs()
CK = ROOT/"results/runs/resnet18_last_stage_aug_f101n/best.pt"

def packaging_paths():
    neg = ROOT/"data/splits/negatives"
    cats = {}
    with open(neg/"sources.csv", newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            cats[Path(row["file"]).name] = row["category"]
    old = [neg/n for n, c in cats.items() if c == "packaging"] + [neg/"005.jpg"]
    seen, new = set(), []
    for p in sorted((ROOT/"results/_discarded_v3").rglob("*.jpg")):
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h in seen: continue
        seen.add(h); new.append(p)
    return sorted(old), new

def food_paths():
    sp = ROOT/"data/splits"
    def take(d, n):
        ps = sorted(d.rglob("*.jpg"))
        if len(ps) <= n: return ps
        idx = np.linspace(0, len(ps)-1, n).round().astype(int)
        return [ps[i] for i in idx]
    return (take(sp/"train", 2000) + take(sp/"canteen_clean", 1000)
            + take(sp/"canteen_real", 100) + take(sp/"test", 600))

def feats(model, device, store, paths):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 48):
            batch = paths[i:i+48]
            x = torch.stack([torch.stack([tf(ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for tf in TF]) for p in batch]).to(device)
            b, v, c, h, w = x.shape
            model(x.view(b*v, c, h, w))
            f = store["f"].flatten(1).float().cpu().numpy().reshape(b, v, -1)
            out.append(f)
            if i % 480 == 0: print("   ", i, "/", len(paths), flush=True)
    return np.concatenate(out)

device = pick_device()
ck = torch.load(CK, map_location="cpu", weights_only=False)
model = build_model(ck["arch"], len(CLASS_NAMES), pretrained=False)
model.load_state_dict(ck["model"]); model.to(device).eval()
store = {}
model.avgpool.register_forward_hook(lambda m, i, o: store.__setitem__("f", o))
old, new = packaging_paths(); food = food_paths()
print("device", device, "old", len(old), "new", len(new), "food", len(food), flush=True)
np.savez(ROOT/"results/tables/packaging_views.npz",
         X_old=feats(model, device, store, old), X_new=feats(model, device, store, new),
         X_food=feats(model, device, store, food),
         old_names=np.array([str(p.relative_to(ROOT)) for p in old]),
         new_names=np.array([str(p.relative_to(ROOT)) for p in new]),
         food_names=np.array([str(p.relative_to(ROOT)) for p in food]))
print("saved", flush=True)
