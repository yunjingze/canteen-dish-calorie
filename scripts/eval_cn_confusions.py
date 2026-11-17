"""Where the 20-dish canteen head still loses: confusion pairs and force-fit.

Test split of CNFOOD-241 (never trained on) plus the 15 Chinese dishes of the
wide tab, which the canteen head has to refuse but tends to absorb.
Writes results/tables/cn_improve.json.

Run: python scripts/eval_cn_confusions.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from foodcls import app as A
from foodcls.classes_cn import CN_CLASS_NAMES, CN_ZH
from foodcls.models import build_model
from foodcls.utils import pick_device

ROOT = Path(__file__).resolve().parents[1]
device = pick_device()
ck = torch.load(ROOT / "results/runs/cn241_full/best.pt", map_location="cpu", weights_only=False)
model = build_model(ck["arch"], len(CN_CLASS_NAMES), pretrained=False)
model.load_state_dict(ck["model"])
model.to(device).eval()
from PIL import Image

T = float(json.loads((ROOT / "results/runs/cn241_full/calibration.json").read_text("utf-8"))["temperature"])
A._state["device"] = device
A._state["cn_model"] = model
A._state["cn_temperature"] = T


@torch.no_grad()
def probs_of(paths, batch=24):
    out = []
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        x = torch.cat([A.preprocess_views(Image.open(p)) for p in chunk]).to(device)
        logits = model(x).float().view(len(chunk), 6, -1)
        fused = torch.softmax(logits.mean(1) / max(T, 1e-3), -1).cpu().numpy()
        out.append(fused)
    return np.concatenate(out)


test = ROOT / "data/splits/cn241_full/test"
paths, labels = [], []
for idx, cls in enumerate(CN_CLASS_NAMES):
    for p in sorted((test / cls).glob("*.jpg")):
        paths.append(p)
        labels.append(idx)
P = probs_of(paths)
y = np.array(labels)
top = P.argmax(1)
acc = float((top == y).mean())
print(f"test n={len(paths)} fused top1={acc*100:.2f}%", flush=True)

pairs = Counter()
for i in range(len(y)):
    if top[i] != y[i]:
        pairs[(CN_CLASS_NAMES[y[i]], CN_CLASS_NAMES[top[i]])] += 1
worst = [(a, b, n) for (a, b), n in pairs.most_common(12)]
print("top confusions (true -> predicted):")
for a, b, n in worst:
    print(f"  {a:16s} -> {b:16s} {n:3d}  ({n/ (y==CN_CLASS_NAMES.index(a)).sum()*100:.0f}% of class)")

# per-class accuracy sorted, to see where the data is thin
per = {}
for idx, cls in enumerate(CN_CLASS_NAMES):
    m = y == idx
    if m.sum():
        per[cls] = round(float((top[m] == idx).mean()), 4)
lows = sorted(per.items(), key=lambda kv: kv[1])[:8]
print("weakest classes:", lows)

# the 15 Chinese dishes of the wide tab, which this head must refuse
wide = ROOT / "data/wide/splits/test"
cn_like = ["mapo_tofu", "kung_pao_chicken", "twice_cooked_pork", "sweet_and_sour_pork",
           "boiled_fish_sichuan", "fish_with_pickled_cabbage", "west_lake_vinegar_fish",
           "steamed_sea_bass", "peking_duck", "soup_dumplings", "dumplings", "spring_rolls",
           "fried_rice", "ramen", "braised_pork_meatballs"]
rows = []
for cls in cn_like:
    files = sorted((wide / cls).glob("*.jpg"))[:20]
    if not files:
        continue
    q = probs_of(files)
    t1 = q.argmax(1)
    topn = Counter(CN_CLASS_NAMES[k] for k in t1)
    robust = float((q.max(1) >= 0.40).mean())
    rows.append({"dish": cls, "n": len(files), "in_scope_rate": round(robust, 4),
                 "top_pred": topn.most_common(3)})
    print(f"{cls:26s} n={len(files):3d} forced-in-scope {robust*100:5.1f}%  "
          f"-> {topn.most_common(2)}", flush=True)

res = {"test_acc": round(acc, 4), "n_test": len(paths),
       "confusions": [{"true": a, "pred": b, "n": n} for a, b, n in worst],
       "per_class": per, "weakest": lows, "chinese_wide": rows}
(ROOT / "results/tables/cn_improve.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), "utf-8")
