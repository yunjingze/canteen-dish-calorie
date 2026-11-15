import sys, itertools, json
from pathlib import Path
sys.path.insert(0, r"E:\项目作业\food_project\src")
import numpy as np, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from foodcls.app import load_model, _state
from foodcls.classes import CLASS_NAMES
from foodcls.data import SplitDataset, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

ROOT = Path(r"E:\项目作业\food_project")
load_model()
model, dev = _state["model"], _state["device"]

def tf(resize=None, crop=None, square=False, flip=False):
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))] if square else [transforms.Resize(resize)]
    if crop: ops.append(transforms.CenterCrop(crop))
    if flip: ops.append(transforms.RandomHorizontalFlip(p=1.0))
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

VIEWS = [("c256", tf(256, 224)), ("c320", tf(320, 224)), ("c448", tf(448, 224)),
         ("sq", tf(None, None, True)), ("c256f", tf(256, 224, False, True)),
         ("c320f", tf(320, 224, False, True)), ("c448f", tf(448, 224, False, True)),
         ("sqf", tf(None, None, True, True))]

@torch.no_grad()
def logits_for(t, split):
    ds = SplitDataset(ROOT/"data/splits"/split, CLASS_NAMES, t)
    ld = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    zs, ys = [], []
    for x, y in ld:
        zs.append(model(x.to(dev, non_blocking=True)).float().cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(zs), np.concatenate(ys)

out = {}
for split in ("val", "test", "canteen"):
    for name, t in VIEWS:
        z, y = logits_for(t, split)
        out[f"{split}_{name}"] = z
        if split == "val": out["labels_val"] = y
        if split == "test": out["labels_test"] = y
        if split == "canteen": out["labels_canteen"] = y
    print("done", split, flush=True)
np.savez_compressed(ROOT/"results/_viewlogits.npz", **out)

# the packaged-dumpling photo, same views
im = Image.open(ROOT/"results/_probe_crop.png").convert("RGB")
with torch.no_grad():
    dimg = {name: model(t(im).unsqueeze(0).to(dev)).float().cpu().numpy()[0] for name, t in VIEWS}
np.savez_compressed(ROOT/"results/_dumpling_logits.npz", **dimg)
print("saved", flush=True)
