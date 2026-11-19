# -*- coding: utf-8 -*-
"""Out-of-source check for the canteen head, on the only dish 16chinese shares.

The plan was "use the classes ChineseFoodNet and CNFOOD-241 have in common".
The 16-class Kaggle set we have locally overlaps this menu in exactly one dish
(糖醋里脊), so this script reports that one honestly instead of pretending the
whole test split is external.  Before scoring, every image is fingerprinted
against the 21-way head's training images (md5, then an 8x8 dHash within 4 bits):
CNFOOD-241 is partly built from ChineseFoodNet, so "external" has to be proven,
not assumed.

Run: python scripts/eval_external_cn.py
Writes results/tables/cn_external.json.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls import app as A  # noqa: E402

#: 16chinese label -> our slug.  Only same-name, same-dish pairs go here.
#: 小笼包 is not 包子 and 东坡肉 is not 红烧肉, so they stay out.
EXTERNAL_MAP = {"Sweet_and_Sour_Pork": "sweet_sour_pork"}
SPLITS = ("train", "val", "test")
IMAGES = ROOT / "data/external/16chinese/images"


def dhash(img):
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def candidates(label):
    out = []
    for split in SPLITS:
        txt = ROOT / f"data/external/16chinese/{split}.txt"
        for line in txt.read_text("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            base = re.split(r"[\\/]", line[:line.lower().rfind(".jpg")])[-1]
            if re.sub(r"_new_\d+$", "", base) != label:
                continue
            p = IMAGES / f"{base}.jpg"
            if p.exists():
                out.append((split, p))
    return out


def main():
    fp = json.loads((ROOT / "results/tables/_fingerprint_cn241_21.json").read_text("utf-8"))
    md5_seen = {v[0] for v in fp["items"].values()}
    dhash_seen = {}
    for v in fp["items"].values():
        if v[1] is not None:
            dhash_seen.setdefault(int(v[1]), 0)
            dhash_seen[int(v[1])] += 1

    rows = []
    for label, slug in EXTERNAL_MAP.items():
        for split, p in candidates(label):
            b = p.read_bytes()
            import hashlib
            md5 = hashlib.md5(b).hexdigest()
            if md5 in md5_seen:
                rows.append({"split": split, "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                             "dup": "md5"})
                continue
            with Image.open(p) as im:
                dh = dhash(im)
            near = any(bin(dh ^ k).count("1") <= 4 for k in dhash_seen)
            rows.append({"split": split, "path": str(p.relative_to(ROOT)).replace("\\", "/"),
                         "dup": "near" if near else None})
    used = [r for r in rows if r["dup"] is None]
    print(f"[{list(EXTERNAL_MAP)}] candidates {len(rows)} "
          f"dup md5 {sum(r['dup'] == 'md5' for r in rows)} "
          f"dup near {sum(r['dup'] == 'near' for r in rows)} used {len(used)}", flush=True)

    A._state["device"] = A.pick_device()
    A.load_cn_model()
    A.load_wide_model()
    A.load_domain_head()
    t0 = time.time()
    for i, r in enumerate(used):
        img = Image.open(ROOT / r["path"]).convert("RGB")
        views, fused, frames = A.analyse_cn(img, want_feats=True)
        veto = A.cn_veto(img)
        probs = np.asarray(fused, dtype=np.float64)
        other = float(probs[A.CN_NUM_CLASSES])
        top_idx = int(np.argmax(probs[:A.CN_NUM_CLASSES]))
        sc = A.scope_of(views, probs, cn=True, cn_veto_result=veto,
                        cn_other_prob=other, cn_other_top=bool(int(np.argmax(probs)) == A.CN_NUM_CLASSES),
                        cn_domain_prob=A._domain_prob(frames))
        r.update({"top": A.CN_CLASS_NAMES[top_idx], "accept": bool(sc["in_scope"]),
                  "robust": sc["robust_prob"], "p_domain": None if sc["domain"]["prob"] is None else sc["domain"]["prob"],
                  "p_other": round(other, 4)})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(used)} {(i+1)/(time.time()-t0):.1f} img/s", flush=True)

    def summary(rs):
        if not rs:
            return {"n": 0}
        good = [r["top"] == EXTERNAL_MAP["Sweet_and_Sour_Pork"] for r in rs]
        acc = [r["accept"] for r in rs]
        good_acc = [g for g, a in zip(good, acc) if a]
        return {"n": len(rs),
                "top1": round(float(np.mean(good)), 4),
                "accept_rate": round(float(np.mean(acc)), 4),
                "top1_accepted": round(float(np.mean(good_acc)), 4) if good_acc else None}

    res = {"map": EXTERNAL_MAP, "source": "Kaggle 16-famous-chinese-dishes (test/val/train)",
           "dedup": {"md5": sum(r["dup"] == "md5" for r in rows),
                     "near": sum(r["dup"] == "near" for r in rows)},
           "all": summary(used),
           "per_split": {s: summary([r for r in used if r["split"] == s]) for s in SPLITS}}
    out = ROOT / "results/tables/cn_external.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), "utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


main()
