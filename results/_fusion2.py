import sys, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
ROOT = Path(r"E:\项目作业\food_project")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from scripts.eval_tta import build_tf
from foodcls.classes import CLASS_NAMES
from foodcls.data import SplitDataset, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from foodcls.models import build_model
from foodcls.utils import pick_device
from torchvision import transforms

dev = pick_device()
ck = torch.load(ROOT / "results/runs/resnet18_last_stage_aug_f101n/best.pt", map_location=dev, weights_only=False)
m = build_model(ck["arch"], len(CLASS_NAMES), pretrained=False).to(dev).eval(); m.load_state_dict(ck["model"])
T = 0.7364

def square(flip):
    ops=[transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if flip: ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)

tfs = {"v0":build_tf(256,False,False), "v1":square(False), "v2":build_tf(320,False,False),
       "v0f":build_tf(256,True,False), "v1f":square(True), "v2f":build_tf(320,True,False)}

@torch.no_grad()
def logits_of(model, root, bs=64):
    outs=[]; labels=None
    for name,t in tfs.items():
        ds=SplitDataset(root, CLASS_NAMES, t); ld=DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
        ch=[]; lb=[]
        for x,y in ld:
            ch.append(model(x.to(dev, non_blocking=True)).float().cpu().numpy()); lb.append(y.numpy())
        outs.append(np.concatenate(ch))
        if labels is None: labels=np.concatenate(lb)
    return labels, np.stack(outs)   # (views, N, C)

def sm(z): 
    z = z - z.max(1, keepdims=True); e=np.exp(z); return e/e.sum(1, keepdims=True)

for split in ("test","canteen"):
    labels, Z = logits_of(m, ROOT/"data/splits"/split)
    A = sm(Z / T).mean(0)            # per-view T then average  (current app)
    B = sm(Z.mean(0) / T)            # average logits then T
    C = sm(Z).mean(0)                # no T
    acc=lambda p: 100*float((p.argmax(1)==labels).mean())
    print(split, "A=%.2f B=%.2f C=%.2f single_v0=%.2f" % (acc(A), acc(B), acc(C), acc(sm(Z[0]))), flush=True)
