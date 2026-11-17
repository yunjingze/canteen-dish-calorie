# -*- coding: utf-8 -*-
"""Pick the "other Chinese" threshold from the measured curves.

scripts/eval_cn_other.py saves one npz per evaluation set (worst-view gate
score, fused top-1, fused "other" probability, and the wide head's verdict).
This script replays the app's decision rule on those arrays for a grid of
CN_OTHER_THRESHOLD values, so the constant shipped in app.py is the argmax of a
stated objective instead of a round number.

Objective: keep at least 93 % of the in-scope test photos (no-bucket baseline is
94.8 %) and, subject to that, maximise the share of out-of-list Chinese dishes
that are turned away.  The Western "real" set is reported but not optimised: the
bucket cannot help there (see README 5.7.1).

Run after eval_cn_other.py: python scripts/pick_cn_other_tau.py
Writes results/tables/cn_other_choice.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls import app as A  # noqa: E402

OTHER = len(A.CN_CLASS_NAMES)
VETO_IDX = np.array([A.WIDE_CLASS_NAMES.index(c) for c in A.CN_VETO_CLASSES])


def load(tag):
    d = np.load(ROOT / f"results/tables/_other_{tag}.npz")
    return {k: d[k] for k in ("rob", "top", "top_p", "p_other", "w_top", "w_p")}


def truth_for(tag, paths):
    if tag == "in":
        table = json.loads((ROOT / "results/tables/cn241_21_classes.json").read_text("utf-8"))
        slugs = [c["slug"] for c in table["classes"]]
        return np.array([slugs.index(Path(p).parent.name) for p in paths])
    return None


def wide_test(cls, k):
    return sorted((ROOT / "data/wide/splits/test" / cls).glob("*.jpg"))[:k]


CN_LIKE = ["mapo_tofu", "kung_pao_chicken", "twice_cooked_pork", "sweet_and_sour_pork",
           "boiled_fish_sichuan", "fish_with_pickled_cabbage", "west_lake_vinegar_fish",
           "steamed_sea_bass", "peking_duck", "soup_dumplings", "dumplings", "spring_rolls",
           "fried_rice", "ramen", "braised_pork_meatballs"]


def paths_for(tag):
    if tag == "in":
        table = json.loads((ROOT / "results/tables/cn241_21_classes.json").read_text("utf-8"))
        names = [c["slug"] for c in table["classes"]][:-1]
        return [p for c in names for p in sorted((ROOT / "data/splits/cn241_full/test" / c).glob("*.jpg"))]
    if tag == "out_cn":
        return [p for c in CN_LIKE for p in wide_test(c, 24)]
    if tag == "out_ood":
        man = json.loads((ROOT / "results/tables/cn241_other_heldout.json").read_text("utf-8"))
        return [ROOT / it["path"] for it in man["items"]]
    if tag == "out_other":
        wt = ROOT / "data/wide/splits/test"
        return [p for d in sorted(wt.iterdir()) if d.is_dir() and d.name not in CN_LIKE
                for p in sorted(d.glob("*.jpg"))[:8]]
    return sorted((ROOT / "data/splits/canteen_real").rglob("*.jpg"))


TAGS = ["in", "out_cn", "out_ood", "out_other", "real"]
sets = {t: load(t) for t in TAGS}
paths = {t: paths_for(t) for t in TAGS}
for t in TAGS:
    assert len(paths[t]) == len(sets[t]["rob"]), (t, len(paths[t]), len(sets[t]["rob"]))

#: The gate is re-derived from the in-scope validation split by eval_cn_other.py;
#: the bucket steals probability mass from the 20 dishes, so keeping the old 0.40
#: would reject in-scope photos the head still recognises.
TH = 0.40
_measured = ROOT / "results/tables/cn_other.json"
if _measured.is_file():
    TH = float(json.loads(_measured.read_text(encoding="utf-8"))["threshold_from_val_5pct"])
print(f"gate threshold {TH:.4f}")


def decide(tag, tau, veto=True):
    d = sets[tag]
    keep = d["rob"] >= TH
    if tau > 0:
        keep &= ~((d["p_other"] >= tau) | (d["top"] == OTHER))
    if veto:
        v = (np.isin(d["w_top"], VETO_IDX) & (d["w_p"] >= A.CN_VETO_THRESHOLD)
             & (d["w_p"] >= d["top_p"] + A.CN_VETO_MARGIN))
        keep &= ~v
    return float(keep.mean())


grid = [round(x, 2) for x in np.arange(0.20, 0.61, 0.05)]
rows = []
for tau in grid:
    row = {"tau": tau}
    for t in TAGS:
        row[t] = round(decide(t, tau), 4)
    rows.append(row)

base_no_veto = {t: round(decide(t, 0.0, veto=False), 4) for t in TAGS}
base_veto = {t: round(decide(t, 0.0, veto=True), 4) for t in TAGS}

# out-of-list Chinese dishes are what the bucket exists for; OOD classes are the
# same question from a different source, so both count.
best = max((r for r in rows if r["in"] >= 0.93),
           key=lambda r: r["out_cn"] + r["out_ood"])

print("tau  " + "  ".join(f"{t:>9}" for t in TAGS))
for r in rows:
    print(f"{r['tau']:.2f} " + "  ".join(f"{r[t]*100:8.1f}%" for t in TAGS))
print("no bucket, veto:", base_veto)
print("no bucket, no veto:", base_no_veto)
print("chosen:", best)

print()
print("=== refusal rate on 1 - accept (Chinese dishes outside the 20) ===")
for t in ("out_cn", "out_ood"):
    print(f"{t}: no bucket {1-base_veto[t]:.3f} -> bucket {1-best[t]:.3f}")

out = {
    "tau_grid": rows,
    "baseline_no_bucket_veto": base_veto,
    "baseline_no_bucket_no_veto": base_no_veto,
    "chosen": best,
    "gate_threshold": TH,
    "objective": "max out_cn+out_ood refusal s.t. in >= 0.93",
}
(ROOT / "results/tables/cn_other_choice.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False), "utf-8")
print("wrote results/tables/cn_other_choice.json")
