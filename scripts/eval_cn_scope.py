"""Pick the worst-view threshold of the 20-dish canteen head.

In scope: CNFOOD-241 test photos of the 20 supported dishes.
Out of scope: photos of dishes that are NOT among the 20 -- the Chinese and
Food-101 classes of the wide test split, which is where a canteen head meets
the dishes it must refuse (braised beef noodles, fish, soup, western food).

Writes results/tables/cn_scope.json.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcls import app as A
from foodcls.classes_cn import CN_CLASS_NAMES
from foodcls.models import build_model
from foodcls.utils import pick_device

ROOT = Path(__file__).resolve().parents[1]
random.seed(0)
np.random.seed(0)

device = pick_device()
ck = torch.load(ROOT / "results/runs/cn241_full/best.pt", map_location=device, weights_only=False)
model = build_model(ck["arch"], len(CN_CLASS_NAMES), pretrained=False)
model.load_state_dict(ck["model"])
model.to(device).eval()
A._state["device"] = device
A._state["cn_model"] = model
A._state["cn_temperature"] = float(json.loads(
    (ROOT / "results/runs/cn241_full/calibration.json").read_text(encoding="utf-8"))["temperature"])
print("temperature", A._state["cn_temperature"])


def robust_of(path: Path) -> float:
    img = Image.open(path).convert("RGB")
    views, _ = A.analyse_cn(img)
    return float(min(np.max(v) for v in views[:A._GATE_VIEWS]))


inscope = []
for cls in CN_CLASS_NAMES:
    files = sorted((ROOT / "data/splits/cn241_full/test" / cls).glob("*.jpg"))[:40]
    vals = [robust_of(f_) for f_ in files]
    inscope.extend(vals)
    print(f"in  {cls:18s} n={len(files)} mean={np.mean(vals) if vals else 0:.3f}", flush=True)

outscope = []
wide_test = ROOT / "data/wide/splits/test"
names = [p for p in wide_test.iterdir() if p.is_dir() and p.name not in CN_CLASS_NAMES]
random.shuffle(names)
count = 0
for p in names:
    files = sorted(p.glob("*.jpg"))[:8]
    for f_ in files:
        outscope.append(robust_of(f_))
        count += 1
    if count >= 320:
        break

ins = np.array(inscope)
out = np.array(outscope)
rows = []
for th in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    keep = float((ins >= th).mean())
    reject = float((out < th).mean())
    rows.append({"threshold": th, "in_keep": round(keep, 4), "out_reject": round(reject, 4)})
    print(f"th {th:.2f}  in-scope kept {keep*100:5.1f}%   out-of-scope rejected {reject*100:5.1f}%", flush=True)

res = {
    "n_in": len(ins), "n_out": len(out),
    "in_mean": round(float(ins.mean()), 4), "out_mean": round(float(out.mean()), 4),
    "in_p05": round(float(np.percentile(ins, 5)), 4),
    "out_p50": round(float(np.percentile(out, 50)), 4),
    "rows": rows,
}
(ROOT / "results/tables/cn_scope.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=2))
