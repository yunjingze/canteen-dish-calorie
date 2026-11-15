
"""Compare framings for the packaging head; honest folds only."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"E:\项目作业\food_project")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

d = np.load(ROOT/"results/tables/packaging_views.npz", allow_pickle=True)
Xo, Xn, Xf = d["X_old"], d["X_new"], d["X_food"]      # (N,3,512)
print("shapes", Xo.shape, Xn.shape, Xf.shape)
food_names = d["food_names"]
# split food by slice so FPR per slice is visible
n_tr, n_cl, n_re, n_te = 2000, 1000, 100, 600
sl = {"train": slice(0,n_tr), "clean": slice(n_tr,n_tr+n_cl),
      "real": slice(n_tr+n_cl, n_tr+n_cl+n_re), "test": slice(n_tr+n_cl+n_re, None)}


def prep(X, how):
    if how == "v0": return X[:, 0]
    if how == "v1": return X[:, 1]
    if how == "v2": return X[:, 2]
    if how == "mean": return X.mean(1)
    if how == "cat": return X.reshape(len(X), -1)
    raise ValueError(how)


def fit(X, y, C=0.01):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000)
    clf.fit((X-mu)/sd, y)
    return lambda Z: clf.decision_function((Z-mu)/sd), (mu, sd, clf.coef_[0], float(clf.intercept_[0]))


for how in ("v0", "v1", "v2", "mean", "cat"):
    Xp = prep(np.concatenate([Xo, Xn]), how); Xn_f = prep(Xf, how)
    # held-out folds over the 62 v3 positives only
    skf = StratifiedKFold(5, shuffle=True, random_state=1)
    recs, aucs = [], []
    for tr, te in skf.split(np.arange(62), np.zeros(62)):
        Xtr = np.vstack([Xp[:17], Xp[17+tr], Xn_f])
        ytr = np.r_[np.ones(17+len(tr)), np.zeros(len(Xn_f))]
        f, _ = fit(Xtr, ytr)
        s = f(Xp[17+te]); recs.append(s)
        aucs.append(roc_auc_score(np.r_[np.ones(len(te)), np.zeros(len(Xn_f))], np.r_[s, f(Xn_f)]))
    s = np.concatenate(recs)
    # leave-old-out: train on 62 + food, test the 17
    f2, _ = fit(np.vstack([Xp[17:], Xn_f]), np.r_[np.ones(62), np.zeros(len(Xn_f))])
    so = f2(Xp[:17]); sf = f2(Xn_f)
    line = (f"{how:5} v3-OOF AUC {np.mean(aucs):.3f} rec " +
            " ".join(f"@{t:+.1f}:{(s>=t).mean()*100:3.0f}%" for t in (0,1,2,3)))
    line += " || old-held " + " ".join(f"@{(so>=t).mean()*100:3.0f}%" for t in (0,1,2))
    line += " || food-fpr@+1 " + " ".join(f"{k}:{(sf[sl[k]]>=1).mean()*100:.2f}%" for k in sl)
    print(line, flush=True)
