# -*- coding: utf-8 -*-
"""Does a cuisine probe on the canteen backbone keep the pizzas out?

The 21-class head only answers "which of the 21".  A logistic regression on the
pooled feature of the same backbone answers "is this a canteen dish at all?"
(scripts/extract_domain_features.py -> scripts/fit_domain_head.py).  This script
runs the *deployed* decision path on five sets -- six views, the wide-head veto,
the "other Chinese" bucket and the domain probe -- and reports what changes when
the probe is switched off.

Run: python scripts/eval_domain_gate.py
Writes results/tables/domain_gate.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls import app as A  # noqa: E402
from foodcls.classes_cn import CN_CLASS_NAMES  # noqa: E402

CN_LIKE = ["mapo_tofu", "kung_pao_chicken", "twice_cooked_pork", "sweet_and_sour_pork",
           "boiled_fish_sichuan", "fish_with_pickled_cabbage", "west_lake_vinegar_fish",
           "steamed_sea_bass", "peking_duck", "soup_dumplings", "dumplings", "spring_rolls",
           "fried_rice", "ramen", "braised_pork_meatballs"]

A._state["device"] = A.pick_device()
A.load_cn_model()
A.load_wide_model()
A.load_domain_head()
TAG = os.environ.get("EVAL_TAG", "")
VAL_ROOT = os.environ.get("EVAL_CN_VAL", "data/splits/cn241_full/val")
IN_ROOT = os.environ.get("EVAL_CN_IN", "data/splits/cn241_full/test")
HEAD = A._state.get("domain_head")
if HEAD is None:
    raise SystemExit("no domain head loaded; run scripts/fit_domain_head.py first")
TH = float(HEAD["threshold"])
print(f"domain probe threshold {TH:.4f}", flush=True)


def wide_test_glob(cls, k):
    return sorted((ROOT / "data/wide/splits/test" / cls).glob("*.jpg"))[:k]


def build_sets():
    sets = {}
    # Same source as test but held out twice: once for the head, and here for
    # tuning a decision rule that is then reported on test.
    sets["val"] = [p for c in CN_NAMES
                   for p in sorted((ROOT / VAL_ROOT / c).glob("*.jpg"))]
    sets["in"] = [p for c in CN_NAMES
                  for p in sorted((ROOT / IN_ROOT / c).glob("*.jpg"))]
    sets["out_cn"] = [p for c in CN_LIKE for p in wide_test_glob(c, 24)]
    ood = json.loads((ROOT / "results/tables/cn241_other_heldout.json").read_text("utf-8"))
    sets["out_ood"] = [ROOT / it["path"] for it in ood["items"]]
    sets["out_other"] = [p for d in sorted((ROOT / "data/wide/splits/test").iterdir())
                         if d.is_dir() and d.name not in CN_LIKE
                         for p in sorted(d.glob("*.jpg"))[:8]]
    sets["real"] = sorted((ROOT / "data/splits/canteen_real").rglob("*.jpg"))
    return sets


def veto_fired(veto, fused):
    """The same predicate scope_of applies, kept here so the dump is exact."""
    if veto is None:
        return False
    return bool(veto["class"] in A.CN_VETO_CLASSES
                and veto["prob"] >= A.CN_VETO_THRESHOLD
                and veto["prob"] >= float(np.max(fused)) + A.CN_VETO_MARGIN)


OTHER_INDEX = A._state.get("cn_other_index")
CN_NAMES = list(A._state.get("cn_names") or A.CN_CLASS_NAMES)


def decide(views, fused, veto, want_domain):
    probs = np.asarray(fused, dtype=np.float64)
    other = float(probs[OTHER_INDEX]) if OTHER_INDEX is not None and probs.size > OTHER_INDEX else None
    other_top = bool(int(np.argmax(probs)) == OTHER_INDEX) if other is not None else False
    dish = probs[:OTHER_INDEX] if other is not None else probs
    dom = A._domain_prob(views["frames"]) if want_domain else None
    sc = A.scope_of(views["views"], probs, cn=True, cn_veto_result=veto,
                    cn_other_prob=other, cn_other_top=other_top, cn_domain_prob=dom)
    return sc, dom, other, int(np.argmax(dish))


def run(paths, tag):
    rows = []
    t0 = time.time()
    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        v, fused, frames = A.analyse_cn(img, want_feats=True)
        veto = A.cn_veto(img)
        packed = {"views": v, "frames": frames}
        on, dom, other, top = decide(packed, fused, veto, True)
        off, _, _, _ = decide(packed, fused, veto, False)
        rows.append({
            "path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "cls": p.parent.name,
            "accept_on": bool(on["in_scope"]),
            "accept_off": bool(off["in_scope"]),
            "p_domain": round(float(dom), 4) if dom is not None else None,
            "robust": on["robust_prob"],
            "p_other": round(float(other), 4) if other is not None else None,
            "top": CN_NAMES[top],
            "domain_fired": bool((on.get("domain") or {}).get("fired")),
            "veto": veto_fired(veto, fused),
            "other_top": bool(OTHER_INDEX is not None and int(np.argmax(fused)) == OTHER_INDEX),
        })
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {tag} {i + 1}/{len(paths)} {(i + 1) / el:.1f} img/s", flush=True)
    # The probe is a pure veto, so accept_on(t) = accept_off & p_domain >= t for
    # any t: keeping the rows lets the threshold be re-swept without another pass.
    np.savez(ROOT / f"results/tables/_domain_gate{TAG}_{tag}.npz",
             path=np.array([r["path"] for r in rows]),
             cls=np.array([r["cls"] for r in rows]),
             top=np.array([r["top"] for r in rows]),
             p_domain=np.array([r["p_domain"] if r["p_domain"] is not None else np.nan for r in rows]),
             robust=np.array([r["robust"] for r in rows]),
             p_other=np.array([r["p_other"] if r["p_other"] is not None else np.nan for r in rows]),
             accept_off=np.array([r["accept_off"] for r in rows]),
             veto=np.array([r["veto"] for r in rows]),
             other_top=np.array([r["other_top"] for r in rows]))
    return rows


def summarise(rows):
    def rate(key):
        return round(float(np.mean([r[key] for r in rows])), 4) if rows else None
    return {"n": len(rows), "accept_domain_off": rate("accept_off"), "accept_domain_on": rate("accept_on")}


sets = build_sets()
res = {"domain_threshold": TH, "sets": {}}
for tag, paths in sets.items():
    print(f"[{tag}] n={len(paths)}", flush=True)
    rows = run(paths, tag)
    res["sets"][tag] = summarise(rows)
    if tag == "in":
        truth = np.array([CN_NAMES.index(r["cls"]) if r["cls"] in CN_NAMES else -1 for r in rows])
        pred = np.array([CN_NAMES.index(r["top"]) for r in rows])
        res["in_top1"] = round(float(np.mean(pred == truth)), 4)
        res["n_dishes"] = len(CN_NAMES)
    if tag in ("out_cn", "real"):
        per = {}
        for r in rows:
            per.setdefault(r["cls"], []).append(r)
        res["sets"][tag]["per_class"] = {
            c: {"n": len(v), "accept_on": round(float(np.mean([x["accept_on"] for x in v])), 3)}
            for c, v in sorted(per.items())}
    print(f"  -> {res['sets'][tag]}", flush=True)

out = ROOT / f"results/tables/domain_gate{TAG}.json"
out.write_text(json.dumps(res, indent=2, ensure_ascii=False), "utf-8")
print("wrote", out.relative_to(ROOT))
