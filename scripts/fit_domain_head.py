"""Fit the Chinese-vs-western domain probe on the cached canteen features.

A logistic regression on the pooled 512-d feature of the *deployed* canteen
backbone: it costs ~10 KB of JSON, no second network, and answers "is this
photo even a canteen-style Chinese dish?" -- the question the 21-class head
cannot answer because every one of its classes is Chinese.

Threshold sits at the 5th percentile of the Chinese validation scores, the same
false-alarm budget as the other gates (0.32 / 0.18 / 0.3445).

Usage
  python scripts/fit_domain_head.py [--feats results/tables/domain_feats.npz]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HEAD = ROOT / "results/tables/domain_head.json"
REPORT = ROOT / "results/tables/domain_head_metrics.json"


def vectors(feat: np.ndarray, mode: str) -> np.ndarray:
    """(N, 3, 512) -> (N, d).  mean = the three gate framings averaged."""
    if mode == "mean":
        return feat.mean(axis=1)
    if mode == "min":
        return feat.min(axis=1)
    if mode == "meanmin":
        return np.concatenate([feat.mean(axis=1), feat.min(axis=1)], axis=1)
    raise SystemExit(f"unknown mode {mode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="results/tables/domain_feats.npz")
    ap.add_argument("--mode", default="mean", choices=["mean", "min", "meanmin"])
    args = ap.parse_args()

    z = np.load(ROOT / args.feats, allow_pickle=True)
    feat, y, names = z["feat"], z["y"].astype(int), z["set"]
    X = vectors(feat, args.mode)

    tr = np.isin(names, ["cn_tr", "f101_tr"])
    va = np.isin(names, ["cn_va", "f101_va"])
    te = np.isin(names, ["cn_te", "f101_te"])

    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X).astype(np.float64)

    best = None
    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        clf = LogisticRegression(C=C, max_iter=4000, class_weight="balanced")
        clf.fit(Xs[tr], y[tr])
        p = clf.predict_proba(Xs[va])[:, 1]
        auc = roc_auc_score(y[va], p)
        # keep >=95 % of Chinese val: threshold at its 5th percentile
        t = float(np.percentile(p[y[va] == 1], 5))
        keep_cn = float((p[y[va] == 1] >= t).mean())
        rej_west = float((p[y[va] == 0] < t).mean())
        print(f"C={C:<6} val AUC={auc:.4f} keep_cn={keep_cn:.3f} reject_west={rej_west:.3f}", flush=True)
        score = (auc, rej_west)
        if best is None or score > best[0]:
            best = (score, C, clf, t, auc, keep_cn, rej_west)

    _, C, clf, t, auc, keep_cn, rej_west = best
    print(f"picked C={C} t={t:.4f}", flush=True)

    p_all = clf.predict_proba(Xs)[:, 1]
    metrics = {"mode": args.mode, "C": C, "threshold": round(t, 4),
               "val_auc": round(float(auc), 4), "val_keep_cn": round(keep_cn, 4),
               "val_reject_west": round(rej_west, 4), "sets": {}}
    for s in sorted(set(names.tolist())):
        m = names == s
        p, yy = p_all[m], y[m]
        if m.sum() < 2:
            continue
        acc = float(((p >= t) == (yy == 1)).mean())
        row = {
            "n": int(m.sum()),
            "accept": round(float((p >= t).mean()), 4),
            "mean_prob": round(float(p.mean()), 4),
        }
        if len(set(yy.tolist())) == 2:
            row["auc"] = round(float(roc_auc_score(yy, p)), 4)
        else:
            row["accuracy"] = round(acc, 3)
        metrics["sets"][s] = row
        print(f"{s:10s} n={m.sum():5d} accept={row['accept']:.3f} mean={row['mean_prob']:.3f}", flush=True)

    # per-class detail on the western test menu and the hand-shot photos
    for s in ("f101_te", "real"):
        m = names == s
        if not m.any():
            continue
        detail = {}
        for c in sorted(set(z["cls"][m].tolist())):
            mm = m & (z["cls"] == c)
            detail[c] = round(float((p_all[mm] >= t).mean()), 4)
        metrics["per_class_" + s] = detail

    HEAD.write_text(json.dumps({
        "kind": "domain_probe",
        "mode": args.mode,
        "mean": [round(float(v), 6) for v in scaler.mean_],
        "scale": [round(float(v), 6) for v in scaler.scale_],
        "w": [round(float(v), 6) for v in clf.coef_[0]],
        "b": round(float(clf.intercept_[0]), 6),
        "threshold": round(t, 4),
        "note": "p_cn = sigmoid(w . (x - mean)/scale + b); x = mean of the 3 gate-view pooled features",
    }, indent=2), encoding="utf-8")
    REPORT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("wrote", HEAD.relative_to(ROOT), REPORT.relative_to(ROOT), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
