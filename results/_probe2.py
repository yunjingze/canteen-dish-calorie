import sys, json
sys.path.insert(0, r"E:\项目作业\food_project\src")
import numpy as np, torch
from PIL import Image
from foodcls.app import preprocess_views, _state, load_model, analyse
from foodcls.classes import CLASS_NAMES
load_model()
idx = {n: i for i, n in enumerate(CLASS_NAMES)}
im = Image.open(r"E:\项目作业\food_project\results\_probe_crop.png").convert("RGB")
views, fused = analyse(im)
for i, v in enumerate(views):
    print(f"view{i}", " ".join(f"{n}={v[idx[n]]:.3f}" for n in ("dumplings","hot_dog","steak","ice_cream")))
print("fused", " ".join(f"{n}={fused[idx[n]]:.3f}" for n in ("dumplings","hot_dog","steak","ice_cream")))
print("argmax fused", CLASS_NAMES[int(fused.argmax())])
