"""How the canteen head behaves on the real canteen photos we shot.

Those 100 photos are the 20-class western menu (pizza, fries, sushi...), i.e.
exactly what the canteen head is NOT for.  The right answer on them is "not on
my 20-dish list", so this script reports the out-of-scope rate, not accuracy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls import app as A
from foodcls.classes_cn import CN_CLASS_NAMES, CN_ZH
from foodcls.models import build_model
from foodcls.utils import pick_device

device = pick_device()
ck = torch.load(ROOT / "results/runs/cn241_full/best.pt", map_location=device, weights_only=False)
model = build_model(ck["arch"], len(CN_CLASS_NAMES), pretrained=False)
model.load_state_dict(ck["model"])
model.to(device).eval()
A._state["device"] = device
A._state["cn_model"] = model
A._state["cn_temperature"] = json.loads(
    (ROOT / "results/runs/cn241_full/calibration.json").read_text(encoding="utf-8"))["temperature"]

root = ROOT / "data/splits/canteen_real"
files = sorted(p for p in root.rglob("*.jpg"))
rows = []
for p in files:
    img = Image.open(p).convert("RGB")
    views, fused = A.analyse_cn(img)
    robust = float(min(np.max(v) for v in views[:A._GATE_VIEWS]))
    top = int(np.argmax(fused))
    rows.append({"file": p.name, "class": p.parent.name,
                 "robust": round(robust, 4), "cn_top1": CN_CLASS_NAMES[top],
                 "in_scope": bool(robust >= A.CN_SCOPE_THRESHOLD)})

inscope = [r for r in rows if r["in_scope"]]
res = {
    "n": len(rows),
    "threshold": A.CN_SCOPE_THRESHOLD,
    "out_of_scope_rate": round(1 - len(inscope) / len(rows), 4),
    "mean_robust": round(float(np.mean([r["robust"] for r in rows])), 4),
    "in_scope_files": [r["file"] for r in inscope],
}
(ROOT / "results/tables/cn_domain.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
for r in rows:
    print(f"{r['file'][:28]:30s} {r['class']:20s} {r['robust']:.3f} {CN_ZH[r['cn_top1']]:8s} {r['in_scope']}")
print(json.dumps({k: v for k, v in res.items() if k != 'in_scope_files'}, indent=2))
