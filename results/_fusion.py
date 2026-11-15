import sys, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
ROOT = Path(r"E:\项目作业\food_project")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from scripts.eval_tta import build_tf
from foodcls.classes import CLASS_NAMES
from foodcls.data import SplitDataset
from foodcls.models import build_model
from foodcls.utils import pick_device

dev = pick_device()
ck = torch.load(ROOT / "results/runs/resnet18_last_stage_aug_f101n/best.pt", map_location=dev, weights_only=False)
m = build_model(ck["arch"], len(CLASS_NAMES), pretrained=False).to(dev).eval(); m.load_state_dict(ck["model"])

specs = {"v0": (256, False), "v1": ("sq", False), "v2": (320, False)}
def tf(resize_to, flip):
    if resize_to == "sq": return build_tf(256, flip, False)  # placeholder replaced below
    return build_tf(resize_to, flip, False)

from torchvision import transforms
from foodcls.data import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
def square(flip):
    ops=[transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if flip: ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)

@torch.no_grad()
def probs_of(model, root, tfs, bs=64, workers=0):
    outs=[]; labels=None
    for name, t in tfs.items():
        ds=SplitDataset(root, CLASS_NAMES, t); ld=DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers)
        chunks=[]; labs=[]
        for x,y in ld:
            chunks.append(torch.softmax(model(x.to(dev, non_blocking=True)).float(),1).cpu().numpy()); labs.append(y.numpy())
        outs.append(np.concatenate(chunks))
        if labels is None: labels=np.concatenate(labs)
    return labels, outs

for split in ("test","canteen"):
    root = ROOT/"data/splits"/split
    labels, (p0,p1,p2) = probs_of(m, root, {"v0":build_tf(256,False,False), "v1":square(False), "v2":build_tf(320,False,False)})
    labels, (q0,q1,q2) = probs_of(m, root, {"v0f":build_tf(256,True,False), "v1f":square(True), "v2f":build_tf(320,True,False)})
    acc=lambda p: 100*float((p.argmax(1)==labels).mean())
    print(split, "v0=%.2f v1=%.2f v2=%.2f | mean3=%.2f mean6=%.2f v0+flip=%.2f mean_v0v1=%.2f" % (
        acc(p0),acc(p1),acc(p2),acc((p0+p1+p2)/3),acc((p0+p1+p2+q0+q1+q2)/6),acc((p0+q0)/2),acc((p0+p1)/2)))
