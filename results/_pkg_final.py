
"""Fit the deployed packaging head on all 79 positives (whole-frame view)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"E:\项目作业\food_project")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

d = np.load(ROOT/"results/tables/packaging_views.npz", allow_pickle=True)
V = 1  # whole frame, square resize: packaging is photographed as an object
Xp = np.concatenate([d["X_old"], d["X_new"]])[:, V].astype(np.float64)
Xn = d["X_food"][:, V].astype(np.float64)
names = np.concatenate([d["old_names"], d["new_names"]])

X = np.vstack([Xp, Xn]); y = np.r_[np.ones(len(Xp)), np.zeros(len(Xn))]
mu, sd = X.mean(0), X.std(0) + 1e-6
Z = (X-mu)/sd
clf = LogisticRegression(C=0.01, class_weight="balanced", max_iter=3000).fit(Z, y)
s = Z @ clf.coef_[0] + float(clf.intercept_[0])
p = 1/(1+np.exp(-s))
print("in-sample AUC", round(roc_auc_score(y, s), 4))
print("food quantiles p99/p99.9", np.quantile(p[len(Xp):], [0.99, 0.999]).round(4))
for t in (0.5, 0.7, 0.8, 0.9, 0.95, 0.99):
    print(f"t={t}: package recall {(p[:len(Xp)]>=t).mean()*100:.0f}%  food fpr {(p[len(Xp):]>=t).mean()*100:.2f}%  "
          f"clean {(p[len(Xp)+2000:len(Xp)+3000]>=t).mean()*100:.2f}%  real {(p[len(Xp)+3000:len(Xp)+3100]>=t).mean()*100:.0f}%")

thr = 0.80
out = {
    "kind": "packaging_head_linear",
    "view": "whole",
    "input": "avgpool 512-d of the whole-frame view (view 1 of 3)",
    "threshold": thr,
    "mean": mu.tolist(), "scale": sd.tolist(),
    "w": clf.coef_[0].tolist(), "b": float(clf.intercept_[0]),
    "n_packaging": int(len(Xp)), "n_food": int(len(Xn)),
    "notes": ("79 packaging photos (17 catalogue negatives + 62 archived v3 product shots) vs "
              "3700 food photos; 5-fold over the 62 held-out: AUC 0.994, recall 79% at the "
              "same operating point, food false-flag 0.5-0.7% on train/clean, 0/100 real."),
}
(ROOT/"results/tables/packaging_linear.json").write_text(json.dumps(out), encoding="utf-8")
print("wrote", ROOT/"results/tables/packaging_linear.json", "bytes", len(json.dumps(out)))
# which food photos would be flagged as packaging at this threshold
flag = p[len(Xp):] >= thr
print("flagged food:", int(flag.sum()), "of", len(Xn))
print([Path(n).as_posix() for n in d["food_names"][flag]][:20])
