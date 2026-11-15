import sys, glob, json, random
from pathlib import Path
sys.path.insert(0, str(Path("E:/项目作业/food_project/src")))
import numpy as np, torch
from PIL import Image
from torchvision import transforms
from foodcls.models import build_model
from foodcls.classes_wide import WIDE_CLASS_NAMES
from foodcls.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

root = Path("E:/项目作业/food_project")
ckpt = torch.load(root/"results/runs/resnet18_food101_cn116/best.pt", map_location="cpu", weights_only=False)
model = build_model(ckpt["arch"], len(ckpt["class_names"]))
model.load_state_dict(ckpt["model"]); model.eval()
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(dev)
T = 0.819
tf = transforms.Compose([transforms.Resize(int(IMG_SIZE*256/224)), transforms.CenterCrop(IMG_SIZE),
                         transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
files = []
for ci, c in enumerate(WIDE_CLASS_NAMES):
    fs = sorted((root/"data/wide/splits/test"/c).glob("*.jpg"))[:8]
    files += [(f, ci) for f in fs]
random.Random(0).shuffle(files)
def ece(y, p, bins=15):
    conf = p.max(1); pred = p.argmax(1); acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins+1); out = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.sum(): out += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(out)
zs, ys = [], []
with torch.no_grad():
    for i in range(0, len(files), 64):
        chunk = files[i:i+64]
        xs = torch.stack([tf(Image.open(f).convert("RGB")) for f, _ in chunk]).to(dev)
        zs.append(model(xs).float().cpu().numpy()); ys += [y for _, y in chunk]
z = np.concatenate(zs); y = np.array(ys)
p = torch.softmax(torch.tensor(z), 1).numpy()
pc = torch.softmax(torch.tensor(z)/T, 1).numpy()
pw = torch.softmax(torch.tensor(p).clamp_min(1e-12).log()/T, 1).numpy()
print("n", len(y))
print("raw ECE        %.4f  acc %.4f" % (ece(y, p), (p.argmax(1)==y).mean()))
print("logits/T ECE   %.4f" % ece(y, pc))
print("logp/T   ECE   %.4f" % ece(y, pw))