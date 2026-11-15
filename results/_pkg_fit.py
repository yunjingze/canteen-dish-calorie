
"""Fit and validate the packaging head."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"E:\项目作业\food_project")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

d = np.load(ROOT/"results/tables/packaging_features.npz", allow_pickle=True)
Xp_all = np.concatenate([d["X_old"], d["X_new"]]).astype(np.float64)
n_old = len(d["X_old"])
Xn = d["X_food"].astype(np.float64)
print("pos", Xp_all.shape, "neg", Xn.shape)


def fit(X, y, C):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
    clf.fit((X-mu)/sd, y)
    return lambda Z: clf.decision_function((Z-mu)/sd), (mu, sd, clf.coef_[0], float(clf.intercept_[0]))


def report(scores_p, scores_n, label):
    auc = roc_auc_score(np.r_[np.ones(len(scores_p)), np.zeros(len(scores_n))], np.r_[scores_p, scores_n])
    print(f"{label}: AUC {auc:.3f}", end="  ")
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        tpr = (scores_p >= t).mean(); fpr = (scores_n >= t).mean()
        print(f"| t={t:+.1f} rec {tpr*100:.0f}%/{len(scores_p)} fpr {fpr*100:.2f}%", end=" ")
    print()
    return auc


# --- 5-fold CV, picks C on inner split -------------------------------------
best = None
for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    idx = np.arange(len(Xp_all) + len(Xn))
    y = np.r_[np.ones(len(Xp_all)), np.zeros(len(Xn))]
    sp, sn = [], []
    for tr, te in skf.split(idx, y):
        Xtr = np.vstack([Xp_all[tr[tr < len(Xp_all)]], Xn[tr[tr >= len(Xp_all)] - len(Xp_all)]])
        ytr = y[tr]
        f, _ = fit(Xtr, ytr, C)
        sp.append(f(Xp_all[te[te < len(Xp_all)]]))
        sn.append(f(Xn[te[te >= len(Xp_all)] - len(Xp_all)]))
    sp, sn = np.concatenate(sp), np.concatenate(sn)
    auc = report(sp, sn, f"C={C}")
    if best is None or auc > best[0]:
        best = (auc, C, sp, sn)
print("best", best[1], "AUC", round(best[0], 3))

# --- cross-source generalisation ------------------------------------------
Xold, Xnew = d["X_old"].astype(np.float64), d["X_new"].astype(np.float64)
for C in (0.01, 0.1):
    f, _ = fit(np.vstack([Xnew, Xn]), np.r_[np.ones(len(Xnew)), np.zeros(len(Xn))], C)
    s_o = f(Xold)
    print(f"[train v3 -> test old] C={C} recall@0: {(s_o>=0).mean()*100:.0f}% "
          f"@+1: {(s_o>=1).mean()*100:.0f}%  mean score {s_o.mean():+.2f}")
    f2, _ = fit(np.vstack([Xold, Xn]), np.r_[np.ones(len(Xold)), np.zeros(len(Xn))], C)
    s_n = f2(Xnew)
    print(f"[train old -> test v3] C={C} recall@0: {(s_n>=0).mean()*100:.0f}% "
          f"@+1: {(s_n>=1).mean()*100:.0f}%  mean score {s_n.mean():+.2f}")
    print(f"   food fpr@0 {(f2(Xn)>=0).mean()*100:.2f}%")
