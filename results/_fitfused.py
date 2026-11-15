import sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from torchvision import transforms
ROOT = Path(r"E:\项目作业\food_project")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
from scripts.fit_temperature import fit_temperature
from scripts.eval_tta import build_tf
from foodcls.classes import CLASS_NAMES
from foodcls.data import SplitDataset, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from foodcls.models import build_model
from foodcls.utils import expected_calibration_error, pick_device

dev = pick_device()
def square(flip):
    ops=[transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if flip: ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)
tfs = [build_tf(256,False,False), square(False), build_tf(320,False,False),
       build_tf(256,True,False), square(True), build_tf(320,True,False)]

def sm(z):
    z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

@torch.no_grad()
def fused(model, root, bs=64):
    acc=[]; labels=None
    for t in tfs:
        ds=SplitDataset(root, CLASS_NAMES, t); ld=DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
        ch=[]; lb=[]
        for x,y in ld:
            ch.append(model(x.to(dev, non_blocking=True)).float().cpu()); lb.append(y)
        acc.append(torch.cat(ch))
        if labels is None: labels=torch.cat(lb)
    return torch.stack(acc).mean(0), labels   # (N, C) averaged logits

for run in ("resnet18_last_stage_aug", "resnet18_last_stage_aug_f101n"):
    ck = torch.load(ROOT/f"results/runs/{run}/best.pt", map_location=dev, weights_only=False)
    m = build_model(ck["arch"], len(CLASS_NAMES), pretrained=False).to(dev).eval(); m.load_state_dict(ck["model"])
    zv, yv = fused(m, ROOT/"data/splits/val")
    T = fit_temperature(zv, yv)
    line = f"{run}  T_fused={T:.3f}"
    for split in ("test","canteen"):
        z, y = fused(m, ROOT/"data/splits"/split)
        p = sm((z/T).numpy()); yy = y.numpy()
        line += f" | {split} acc={100*(p.argmax(1)==yy).mean():.2f} ece={expected_calibration_error(yy, p):.3f}"
    print(line, flush=True)
