# -*- coding: utf-8 -*-
"""Score the canteen head on the web-screened external set.

Input is the output of scripts/ingest_web.py apply -- accepted/<slug>/ plus the
manifest that says where every file came from.  Nothing is trained on these
images; they exist to answer one question the report has been missing: how
well do the 20 dishes survive outside CNFOOD-241?

Run: python scripts/eval_web20.py
Writes results/tables/cn_web20.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls import app as A  # noqa: E402

BASE = ROOT / "data/external/web20"
ACCEPTED = BASE / "accepted"
OUT = ROOT / "results/tables/cn_web20.json"


def load_items():
    items = []
    if not ACCEPTED.is_dir():
        return items
    for folder in sorted(ACCEPTED.iterdir()):
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                items.append((folder.name, p))
    return items


def score(items):
    A._state["device"] = A.pick_device()
    A.load_cn_model()
    A.load_wide_model()
    A.load_domain_head()
    rows = []
    t0 = time.time()
    for i, (label, p) in enumerate(items):
        img = Image.open(p).convert("RGB")
        views, fused, frames = A.analyse_cn(img, want_feats=True)
        veto = A.cn_veto(img)
        probs = np.asarray(fused, dtype=np.float64)
        other = float(probs[A.CN_NUM_CLASSES])
        top_idx = int(np.argmax(probs[:A.CN_NUM_CLASSES]))
        top = A.CN_CLASS_NAMES[top_idx]
        sc = A.scope_of(views, probs, cn=True, cn_veto_result=veto,
                        cn_other_prob=other,
                        cn_other_top=bool(int(np.argmax(probs)) == A.CN_NUM_CLASSES),
                        cn_domain_prob=A._domain_prob(frames))
        rows.append({"label": label, "top": top, "accept": bool(sc["in_scope"]),
                     "robust": sc["robust_prob"],
                     "p_other": round(other, 4),
                     "p_domain": None if sc["domain"]["prob"] is None else round(sc["domain"]["prob"], 4),
                     "file": str(p.relative_to(ROOT)).replace("\\", "/")})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(items)} {(i+1)/(time.time()-t0):.1f} img/s", flush=True)
    return rows


def summarise(rows, labels):
    n = len(rows)
    if not n:
        return {"n": 0}
    ok = np.array([r["label"] == r["top"] for r in rows])
    acc = np.array([r["accept"] for r in rows])
    ok_acc = ok & acc
    per = {}
    for slug in labels:
        sub = [i for i, r in enumerate(rows) if r["label"] == slug]
        if not sub:
            continue
        per[slug] = {"n": len(sub),
                     "top1": round(float(ok[sub].mean()), 4),
                     "accept": round(float(acc[sub].mean()), 4)}
    return {"n": n,
            "top1": round(float(ok.mean()), 4),
            "accept_rate": round(float(acc.mean()), 4),
            "top1_accepted": round(float(ok_acc.sum() / max(1, acc.sum())), 4),
            "per_class": per}


def main():
    items = load_items()
    if not items:
        print(f"no images under {ACCEPTED}; run scripts/ingest_web.py apply first")
        return 2
    labels = sorted({label for label, _ in items})
    print(f"{len(items)} images, {len(labels)} dishes")
    rows = score(items)
    res = {"source": "Bing Images, screened by scripts/ingest_web.py",
           "note": "external test set; never used for training",
           "all": summarise(rows, labels),
           "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), "utf-8")
    a = res["all"]
    print(json.dumps({k: v for k, v in a.items() if k != "per_class"}, indent=2, ensure_ascii=False))
    for slug, v in sorted(a["per_class"].items(), key=lambda kv: kv[1]["top1"]):
        print(f"  {slug:20s} n={v['n']:3d} top1={v['top1']:.3f} accept={v['accept']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
