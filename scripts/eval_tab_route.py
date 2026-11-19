# -*- coding: utf-8 -*-
"""Which tab should the "switch tab" button offer, and is the offer honest?

A refusal on one tab can be a photo that belongs to another: a canteen dish
photographed on the 20-class Western tab, or a pizza on the canteen tab.  This
script replays candidate routing rules on the evaluation sets and the 100 real
canteen photos, and reports, per starting tab:

  offers   -- photos the rule would send elsewhere, out of those refused
  correct  -- of those offers, how many name the photo's true dish on arrival
  wrong    -- accepted by the other tab but naming a different dish

The deployed rule is app.suggest_tab(); the variants here differ in what they
gate on (the domain probe, the other tab's top-1 confidence).

Run: python scripts/eval_tab_route.py
Writes results/tables/tab_route.json.
"""
from __future__ import annotations

import json
import random
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
A.load_model()
A.load_cn_model()
A.load_wide_model()
A.load_domain_head()

TAKE = {"val": 150, "in": 150, "out_cn": 150, "out_ood": 100, "out_other": 150, "real": 100}
RNG = random.Random(0)


def wide_test_glob(cls, k):
    return sorted((ROOT / "data/wide/splits/test" / cls).glob("*.jpg"))[:k]


def build_sets():
    sets = {}
    sets["val"] = [p for c in CN_CLASS_NAMES
                   for p in sorted((ROOT / "data/splits/cn241_full/val" / c).glob("*.jpg"))]
    sets["in"] = [p for c in CN_CLASS_NAMES
                  for p in sorted((ROOT / "data/splits/cn241_full/test" / c).glob("*.jpg"))]
    sets["out_cn"] = [p for c in CN_LIKE for p in wide_test_glob(c, 24)]
    ood = json.loads((ROOT / "results/tables/cn241_other_heldout.json").read_text("utf-8"))
    sets["out_ood"] = [ROOT / it["path"] for it in ood["items"]]
    sets["out_other"] = [p for d in sorted((ROOT / "data/wide/splits/test").iterdir())
                         if d.is_dir() and d.name not in CN_LIKE
                         for p in sorted(d.glob("*.jpg"))[:8]]
    sets["real"] = sorted((ROOT / "data/splits/canteen_real").rglob("*.jpg"))
    return sets


def sample(paths, k):
    if len(paths) <= k:
        return list(paths)
    by_cls = {}
    for p in paths:
        by_cls.setdefault(p.parent.name, []).append(p)
    picked = []
    for cls in sorted(by_cls):
        picked.extend(RNG.sample(by_cls[cls], max(1, round(k * len(by_cls[cls]) / len(paths)))))
    RNG.shuffle(picked)
    return picked[:k]


def cn_verdict(img):
    views, fused, frames = A.analyse_cn(img, want_feats=True)
    other = float(fused[A.CN_NUM_CLASSES]) if fused.size > A.CN_NUM_CLASSES else None
    other_top = bool(int(np.argmax(fused)) == A.CN_NUM_CLASSES) if other is not None else False
    dish = fused[:A.CN_NUM_CLASSES] if other is not None else fused
    dom = A._domain_prob(frames)
    sc = A.scope_of(views, dish, cn=True, cn_veto_result=A.cn_veto(img),
                    cn_other_prob=other, cn_other_top=other_top, cn_domain_prob=dom)
    top = A.CN_CLASS_NAMES[int(np.argmax(dish))]
    return {"accept": sc["in_scope"], "top": top, "prob": round(float(np.max(dish)), 4),
            "domain": dom, "domain_fired": bool((sc.get("domain") or {}).get("fired")),
            "reason": A._refusal_reason(sc)}


def single_verdict(img):
    views, fused, frames = A.analyse_full(img)
    pack = A._state.get("packaging")
    sc = A.scope_of(views, fused, food_prob=A._food_prob(frames),
                    packaging_prob=pack.prob(frames) if pack is not None else None)
    return {"accept": sc["in_scope"], "top": A.CLASS_NAMES[int(np.argmax(fused))],
            "prob": round(float(np.max(fused)), 4), "reason": A._refusal_reason(sc)}


def wide_verdict(img):
    views, fused, frames = A.analyse_wide(img, want_feats=True)
    sc = A.scope_of(views, fused, wide=True, food_prob=A._food_prob(frames))
    return {"accept": sc["in_scope"], "top": A.WIDE_CLASS_NAMES[int(np.argmax(fused))],
            "prob": round(float(np.max(fused)), 4), "reason": A._refusal_reason(sc)}


def main():
    sets = build_sets()
    rows = []
    times = {"cn": [], "single": [], "wide": []}
    for tag, paths in sets.items():
        todo = sample(paths, TAKE[tag])
        print(f"{tag}: {len(todo)} photos", flush=True)
        for p in todo:
            img = Image.open(p).convert("RGB")
            v = {}
            for mode, fn in (("cn", cn_verdict), ("single", single_verdict), ("wide", wide_verdict)):
                t0 = time.time()
                v[mode] = fn(img)
                times[mode].append(time.time() - t0)
            rows.append({"set": tag, "cls": p.parent.name, "v": v})

    def offer(cur, row, single_min, cn_min, domain_gate, skip_reasons=()):
        """Return the tab this rule would offer, or None."""
        if cur == "cn" and row["v"]["cn"]["reason"] in skip_reasons:
            return None
        order = ("single", "wide") if cur == "cn" else ("cn",)
        for alt in order:
            o = row["v"][alt]
            if not o["accept"]:
                continue
            if alt == "single":
                if domain_gate and not row["v"]["cn"]["domain_fired"]:
                    continue
                if o["prob"] < single_min:
                    continue
            if alt == "cn" and o["prob"] < cn_min:
                continue
            return alt
        return None

    variants = {
        "deployed": dict(single_min=0.0, cn_min=0.0, domain_gate=False),
        "domain_gated": dict(single_min=0.0, cn_min=0.0, domain_gate=True),
        "domain+0.50": dict(single_min=0.50, cn_min=0.0, domain_gate=True),
        "domain+cn0.50": dict(single_min=0.0, cn_min=0.50, domain_gate=True),
        "domain+0.50+cn0.50": dict(single_min=0.50, cn_min=0.50, domain_gate=True),
        "shipped": dict(single_min=0.0, cn_min=0.0, domain_gate=False,
                        skip_reasons=("veto", "weak")),
    }
    out = {}
    for cur in ("cn", "single"):
        out[cur] = {}
        for name, kw in variants.items():
            per = {}
            for tag in sets:
                rows_ = [r for r in rows if r["set"] == tag]
                refused = [r for r in rows_ if not r["v"][cur]["accept"]]
                offers = [(r, offer(cur, r, **kw)) for r in refused]
                offers = [(r, a) for r, a in offers if a]
                correct = sum(1 for r, a in offers if r["v"][a]["top"] == r["cls"])
                per[tag] = {"n": len(rows_), "refused": len(refused), "offered": len(offers),
                            "correct": correct,
                            "wrong": len(offers) - correct,
                            "share": round(len(offers) / max(1, len(refused)), 3),
                            "acc": round(correct / max(1, len(offers)), 3)}
            out[cur][name] = per
    timing = {m: {"mean_ms": round(1000 * float(np.mean(v)), 1),
                  "max_ms": round(1000 * float(np.max(v)), 1)} for m, v in times.items()}
    payload = {"take": TAKE, "variants": variants, "per_start_tab": out,
               "verdict_time": timing, "rows": rows}
    path = ROOT / "results/tables/tab_route.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")

    for cur in ("cn", "single"):
        print(f"\n== starting tab {cur} ==")
        for name in variants:
            cells = []
            for tag in ("val", "in", "out_cn", "out_ood", "out_other", "real"):
                d = out[cur][name][tag]
                cells.append(f"{tag} {d['offered']}/{d['refused']}o {d['correct']}c {d['acc']:.2f}")
            print(f"{name:18s} " + " | ".join(cells))
    print(json.dumps(timing, ensure_ascii=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
