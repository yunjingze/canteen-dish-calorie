import sys
sys.path.insert(0, r"E:\项目作业\food_project\src")
import numpy as np, torch
from PIL import Image
from torchvision import transforms
from foodcls.app import load_model, _state
from foodcls.classes import CLASS_NAMES
from foodcls.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
load_model()
model = _state["model"]; dev = _state["device"]
idx = {n: i for i, n in enumerate(CLASS_NAMES)}
im = Image.open(r"E:\项目作业\food_project\results\_probe_crop.png").convert("RGB")
print("size", im.size)
def tf(resize, crop=None, square=False):
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))] if square else [transforms.Resize(resize)]
    if crop: ops.append(transforms.CenterCrop(crop))
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
variants = {
 "short256_c224": tf(256, 224), "short320_c224": tf(320, 224), "short384_c224": tf(384, 224),
 "short448_c224": tf(448, 224), "square_full": tf(None, None, True), "frame_crop85": tf(256, 0.85),
}
with torch.no_grad():
    for name, t in variants.items():
        x = t(im).unsqueeze(0).to(dev)
        p = torch.softmax(model(x).float() / 0.671, 1)[0].cpu().numpy()
        order = p.argsort()[::-1][:3]
        print(f"{name:18s}", " ".join(f"{CLASS_NAMES[i]}={p[i]:.3f}" for i in order))
