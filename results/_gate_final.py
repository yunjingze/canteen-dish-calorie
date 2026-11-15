"""Honest food/non-food gate: negatives scored out-of-fold by category, real photos never seen in training.

131 true negatives (the 7 food photographs that were mislabelled in v1 are now
cross-domain positives), 100 real canteen photos held out entirely, and the
proxy domain (canteen_clean 1k) standing in for the training distribution.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

ROOT = Path(r"E:\项目作业\food_project")
d = np.load(ROOT/"results/tables/gate_features.npz", allow_pickle=True)
Xn, Xp_tr, Xp_cl, Xp_re = d["Xn"], d["Xp_tr"], d["Xp_cl"], d["Xp_re"]
X_food = d["X_food_extra"]
neg_cat = d["neg_cat"].astype(object)
neg_names = [str(x) for x in d["neg_names"]]
cats = sorted(set(neg_cat))
N = len(Xn)

def fit(train_mask_neg):
    X = np.r_[Xn[train_mask_neg], Xp_tr, Xp_cl]
    y = np.r_[np.zeros(train_mask_neg.sum()), np.ones(len(Xp_tr)+len(Xp_cl))]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=5000).fit(sc.transform(X), y)
    return sc, clf

def p_food(sc, clf, A):
    return clf.predict_proba(sc.transform(A))[:, 1]

oof = np.zeros(N); fold_auc = {}
for c in cats:
    m = neg_cat == c
    sc, clf = fit(~m)
    oof[m] = p_food(sc, clf, Xn[m])
    fold_auc[c] = float(roc_auc_score(np.r_[np.zeros(int(m.sum())), np.ones(len(Xp_re))],
                                      np.r_[oof[m], p_food(sc, clf, Xp_re)]))

sc, clf = fit(np.ones(N, bool))
real_p = p_food(sc, clf, Xp_re)
proxy_p = p_food(sc, clf, Xp_cl)
food_p = p_food(sc, clf, X_food)
print(f"gate trained on {N} negatives; out-of-fold AUC vs {len(Xp_re)} real photos:")
for c in cats: print(f"  {c:18} {fold_auc[c]:.3f}  (n={int((neg_cat==c).sum())})")
print(f"  mean {np.mean(list(fold_auc.values())):.3f}   min {min(fold_auc.values()):.3f}")
print(f"AUC oof-negatives vs real: {roc_auc_score(np.r_[np.zeros(N),np.ones(len(real_p))], np.r_[oof, real_p]):.3f}")
print(f"AUC oof-negatives vs proxy: {roc_auc_score(np.r_[np.zeros(N),np.ones(len(proxy_p))], np.r_[oof, proxy_p]):.3f}")
print(f"7 relabelled food photos p_food: {np.round(food_p,2).tolist()}")

g2 = {x["name"] if "name" in x else i: x for i, x in
      enumerate(json.load(open(ROOT/'results/tables/gate_v2.json', encoding='utf-8'))["negatives"])}
rob_all = json.load(open(ROOT/'results/tables/gate_v2.json', encoding='utf-8'))["negatives"]
print("\nnegatives in gate_v2.json:", len(rob_all))
rob = np.array([rob_all[i]["robust"] for i in range(len(rob_all))])[:N]

print(f"\nfusion (reject if robust<0.32 OR p_food<t); baseline robust-only rejects {int((rob<0.32).sum())}/{N}")
print(f"{'t':>5} {'neg rejected':>13} {'real kept':>10} {'proxy kept':>11} {'food7 kept':>11}")
for t in (0.1,0.2,0.3,0.4,0.5,0.6):
    rej = (rob < 0.32) | (oof < t)
    print(f"{t:5.1f} {int(rej.sum()):>7}/{N:<5} {int((real_p>=t).sum()):>4}/{len(real_p):<5} "
          f"{int((proxy_p>=t).sum()):>5}/{len(proxy_p):<5} {int((food_p>=t).sum()):>5}/7")
print("\ngate-only:")
for t in (0.3,0.4,0.5,0.6):
    print(f"  t={t:.1f}  neg rejected {int((oof<t).sum()):3}/{N}  real kept {int((real_p>=t).sum()):3}/{len(real_p)}  proxy kept {int((proxy_p>=t).sum()):4}/1000")

coef = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "w": clf.coef_[0].tolist(),
        "b": float(clf.intercept_[0]), "n_negatives": N}
(ROOT/"results/tables/foodgate_linear.json").write_text(json.dumps(coef), encoding="utf-8")
out = {"loco_auc": fold_auc, "oof": oof.tolist(), "real_p": real_p.tolist(),
       "neg_cat": neg_cat.tolist(), "neg_names": neg_names, "robust": rob.tolist(),
       "food_extra_p": food_p.tolist()}
(ROOT/"results/tables/gate_oof.json").write_text(json.dumps(out), encoding="utf-8")
print("\nwrote foodgate_linear.json + gate_oof.json")
