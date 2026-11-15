"""Rebuild the gate feature cache after correcting the v1 negative labels.

Seven of the fifteen v1 "negatives" are photographs of food (bread rolls, salad
plates on a tray, food cups, a nugget tray) - the gate was right and the label
was wrong, which is what dragged the v1_unknown slice down to 0.807 AUC.  They
become cross-domain food positives; the gate trains on the remaining 131 real
non-food images.

Only the negative side is re-extracted; the 4000 train / 1000 clean / 100 real
/ 600 test positives are re-used from the previous cache.
"""
from __future__ import annotations
import json, sys
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

TF = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                         transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
CK = ROOT/"results/runs/resnet18_last_stage_aug_f101n/best.pt"

def main():
    device = pick_device()
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    model = build_model(ck["arch"], len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(ck["model"]); model.to(device).eval()
    store = {}
    model.avgpool.register_forward_hook(lambda m, i, o: store.__setitem__("f", o))

    paths = sorted((ROOT/"data/splits/negatives").glob("*.jpg"))
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            batch = paths[i:i+64]
            x = torch.stack([TF(ImageOps.exif_transpose(Image.open(p)).convert("RGB")) for p in batch]).to(device)
            model(x)
            out.append(store["f"].flatten(1).float().cpu().numpy())
            print("  feats", i, "/", len(paths), flush=True)
    Xn = np.concatenate(out)

    v1 = json.load(open(ROOT/"data/splits/negatives/v1_categories.json", encoding="utf-8"))
    old = np.load(ROOT/"results/tables/gate_features.npz", allow_pickle=True)
    # categories for the 123 csv rows; the 15 v1 files come from v1_categories.json
    import csv
    cats_csv = {}
    with open(ROOT/"data/splits/negatives/sources.csv", newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            cats_csv[Path(row["file"]).name] = row["category"]
    cats = np.array([v1.get(p.name, cats_csv.get(p.name, "v1_unknown")) for p in paths], dtype=object)
    food = cats == "food_nonvocab"
    print("negatives", int((~food).sum()), "food", int(food.sum()))

    np.savez(ROOT/"results/tables/gate_features.npz",
             Xn=Xn[~food], Xp_tr=old["Xp_tr"], Xp_cl=old["Xp_cl"], Xp_re=old["Xp_re"], Xp_te=old["Xp_te"],
             X_food_extra=Xn[food], neg_cat=cats[~food], extra_cat=cats[food],
             neg_names=np.array([p.name for p, f in zip(paths, food) if not f]),
             extra_names=np.array([p.name for p, f in zip(paths, food) if f]))
    for c in sorted(set(cats[~food])):
        print(f"  {c:18} {int((cats[~food]==c).sum())}")

main()
