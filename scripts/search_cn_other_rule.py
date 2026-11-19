# -*- coding: utf-8 -*-
"""Offline rule search for the "other Chinese" refusal.

Joins the two evaluation passes (_other_<set>.npz: 21-class argmax, bucket
probability, wide-head veto; _extra_<set>.npz: 20-dish evidence and gate scores)
and scores candidate decision rules without touching the GPU.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls import app as A

OTHER = len(A.CN_CLASS_NAMES)
CN_LIKE = ["mapo_tofu", "kung_pao_chicken", "twice_cooked_pork", "sweet_and_sour_pork",
           "boiled_fish_sichuan", "fish_with_pickled_cabbage", "west_lake_vinegar_fish",
           "steamed_sea_bass", "peking_duck", "soup_dumplings", "dumplings", "spring_rolls",
           "fried_rice", "ramen", "braised_pork_meatballs"]
TAGS = ["in", "out_cn", "out_ood", "out_other", "real", "val"]
MAIN = ("in", "out_cn", "out_ood", "out_other", "real")

TAG = os.environ.get("EVAL_TAG", "")
S = {}
for t in TAGS:
    b = np.load(ROOT / ("results/tables/_extra%s_%s.npz" % (TAG, t)))
    merged = {k: b[k] for k in b.files}
    if t != "val":  # the first pass never dumped the validation split
        a = np.load(ROOT / ("results/tables/_other%s_%s.npz" % (TAG, t)))
        assert len(a["rob"]) == len(b["p20"]) == len(a["p_other"]), t
        merged.update({k: a[k] for k in a.files})
    S[t] = merged

VETO_IDX = np.array([A.WIDE_CLASS_NAMES.index(c) for c in A.CN_VETO_CLASSES])


def veto_mask(d):
    return (np.isin(d["w_top"], VETO_IDX) & (d["w_p"] >= A.CN_VETO_THRESHOLD)
            & (d["w_p"] >= d["top_p"] + A.CN_VETO_MARGIN))


def pct5(key):
    return float(round(np.quantile(S["val"][key], 0.05), 4))


TH = {"rob21": pct5("rob21"), "rob20": pct5("rob20"), "rob20r": pct5("rob20r")}
print("gate thresholds (val 5th pct): " + "  ".join("%s %.4f" % (k, v) for k, v in TH.items()))


def keep_mask(tag, gate="rob21", tau=None, alpha=None, use_top=True):
    d = S[tag]
    keep = d[gate] >= TH[gate]
    if tau is not None:
        hit = d["p_other"] >= tau
        if use_top:
            hit = hit | (d["top"] == OTHER)
        if alpha is not None:
            hit = hit & (d["p20"] < alpha)
        keep = keep & ~hit
    return keep & ~veto_mask(d)


def show(title, **kw):
    vals = {t: float(keep_mask(t, **kw).mean()) for t in MAIN}
    print("%-34s " % title + "  ".join("%s %5.1f" % (t, vals[t] * 100) for t in MAIN))
    return vals


print()
print("== gate only (no bucket) ==")
for g in ("rob21", "rob20", "rob20r"):
    show(g, gate=g)

print()
print("== bucket: (p_other>=tau or top==other) and p20<alpha ==")
for g in ("rob20r", "rob21"):
    for tau in (0.30, 0.40, 0.50, 0.60):
        for alpha in (0.35, 0.50, 0.65, 0.80, 1.01):
            show("%s tau=%.2f alpha=%.2f" % (g, tau, alpha), gate=g, tau=tau, alpha=alpha)

print()
print("== bucket: only p_other>=tau (top==other not used) ==")
for tau in (0.40, 0.50, 0.60):
    for alpha in (0.50, 0.65, 1.01):
        show("rob21 tau=%.2f alpha=%.2f notop" % (tau, alpha),
             gate="rob21", tau=tau, alpha=alpha, use_top=False)

print()
print("== per-class acceptance on out-of-list Chinese dishes ==")
paths = [p for c in CN_LIKE for p in sorted((ROOT / "data/wide/splits/test" / c).glob("*.jpg"))[:24]]
cls = np.array([Path(p).parent.name for p in paths])
RULES = [("shipped-ish rob21 tau.40 a1.0", dict(gate="rob21", tau=0.40, alpha=1.01)),
         ("rob20r tau.40 a.65", dict(gate="rob20r", tau=0.40, alpha=0.65)),
         ("rob20r tau.50 a.65", dict(gate="rob20r", tau=0.50, alpha=0.65))]
print("%-26s" % "dish" + "".join("%22s" % r[0] for r in RULES))
for c in CN_LIKE:
    m = cls == c
    cells = []
    for _, kw in RULES:
        cells.append("%21.1f%%" % (keep_mask("out_cn", **kw)[m].mean() * 100))
    print("%-26s" % c + "".join(cells))

paths_in = [p for c in A.CN_CLASS_NAMES
            for p in sorted((ROOT / "data/splits/cn241_full/test" / c).glob("*.jpg"))]
truth = np.array([A.CN_CLASS_NAMES.index(Path(p).parent.name) for p in paths_in])
print()
print("in-list single-view accuracy: 21-way %.2f%%  20-way %.2f%%  called other %.2f%%" % (
    np.mean(S["in"]["top"] == truth) * 100, np.mean(S["in"]["top20"] == truth) * 100,
    np.mean(S["in"]["top"] == OTHER) * 100))

print()
print("in-list false refusals caused by the bucket (alpha rule, rob21):")
for tau, alpha in ((0.40, 1.01), (0.40, 0.65), (0.50, 0.65)):
    d = S["in"]
    hit = (d["p_other"] >= tau) | (d["top"] == OTHER)
    if alpha is not None:
        hit = hit & (d["p20"] < alpha)
    print("  tau %.2f alpha %.2f -> %5.2f%% of supported photos refused as other" % (
          tau, alpha, hit.mean() * 100))
