"""Corrected-label re-partition: same cached features, names kept for the robust-score join."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(r"E:\项目作业\food_project")
d = np.load(ROOT/"results/tables/gate_features.npz", allow_pickle=True)
Xn, neg_cat = d["Xn"], d["neg_cat"].astype(object)
paths = sorted((ROOT/"data/splits/negatives").glob("*.jpg"))
v1 = json.load(open(ROOT/"data/splits/negatives/v1_categories.json", encoding="utf-8"))
cats = np.array([v1[p.name] if p.name in v1 else c for p, c in zip(paths, neg_cat)], dtype=object)
food = cats == "food_nonvocab"
keep = [p.name for p, f in zip(paths, food) if not f]
np.savez(ROOT/"results/tables/gate_features.npz",
         Xn=Xn[~food], Xp_tr=d["Xp_tr"], Xp_cl=d["Xp_cl"], Xp_re=d["Xp_re"], Xp_te=d["Xp_te"],
         X_food_extra=Xn[food], neg_cat=cats[~food], extra_cat=cats[food],
         neg_names=np.array(keep), extra_names=np.array([p.name for p, f in zip(paths, food) if f]))
print("negatives", len(keep), "extra food", int(food.sum()))
