"""Train the linear food/non-food gate on all 138 negatives and report operating points + fusion with the robust score."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(r"E:\项目作业\food_project")
d = np.load(ROOT/"results/tables/gate_features.npz", allow_pickle=True)
Xn, Xp_tr, Xp_cl, Xp_re, Xp_te = d["Xn"], d["Xp_tr"], d["Xp_cl"], d["Xp_re"], d["Xp_te"]
neg_cat = d["neg_cat"]

X = np.r_[Xn, Xp_tr, Xp_cl, Xp_re]
y = np.r_[np.zeros(len(Xn)), np.ones(len(Xp_tr)+len(Xp_cl)+len(Xp_re))]
sc = StandardScaler().fit(X)
clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000).fit(sc.transform(X), y)

def p_notfood(A): return clf.predict_proba(sc.transform(A))[:, 1]

neg = p_notfood(Xn); real = p_notfood(Xp_re); proxy = p_notfood(Xp_cl); f101 = p_notfood(Xp_te)
print(f"AUC negatives vs real100  {roc_auc_score(np.r_[np.zeros(len(neg)),np.ones(len(real))], np.r_[neg,real]):.3f}")
print(f"AUC negatives vs proxy1k  {roc_auc_score(np.r_[np.zeros(len(neg)),np.ones(len(proxy))], np.r_[neg,proxy]):.3f}")
print(f"AUC negatives vs f101     {roc_auc_score(np.r_[np.zeros(len(neg)),np.ones(len(f101))], np.r_[neg,f101]):.3f}")

print("\noperating points (reject non-food when p >= t):")
print(f"{'t':>5} {'neg rejected':>13} {'real kept':>10} {'proxy kept':>11} {'f101 kept':>10}")
for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
    print(f"{t:5.1f} {int((neg>=t).sum()):>7}/{len(neg):<5} {int((real<t).sum()):>4}/{len(real):<5} {int((proxy<t).sum()):>5}/{len(proxy):<5} {int((f101<t).sum()):>4}/{len(f101):<5}")

# per-category at t=0.5
print("\nper-category rejected at t=0.5:")
for c in sorted(set(neg_cat)):
    m = neg_cat == c
    print(f"  {c:18} {int((neg[m]>=0.5).sum()):3}/{int(m.sum()):<3}  median p={np.median(neg[m]):.2f}")

# fusion with the v1 robust score
gp = json.load(open(ROOT/"results/tables/gate_real_probe.json", encoding="utf-8"))["scores"]
rob_proxy = np.array(gp["proxy"]); rob_real = np.array(gp["real"])  # 86 v1 images, proxy 1000
# negatives' robust scores from gate_v2.json
g2 = json.load(open(ROOT/"results/tables/gate_v2.json", encoding="utf-8"))
rob_neg = np.array([x["robust"] for x in g2["negatives"]])
print(f"\nrobust-only at 0.32: neg accepted {int((rob_neg>=0.32).sum())}/{len(rob_neg)}")
for t in (0.4, 0.5, 0.6):
    rule = (rob_neg >= 0.32) & (neg < t)
    print(f"  robust>=0.32 AND p_notfood<{t}: neg accepted {int(rule.sum())}/{len(rob_neg)}")

coef = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "w": clf.coef_[0].tolist(), "b": float(clf.intercept_[0]), "C": 0.05}
(ROOT/"results/tables/foodgate_linear.json").write_text(json.dumps(coef), encoding="utf-8")
print("\nwrote results/tables/foodgate_linear.json")
