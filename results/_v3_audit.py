"""Audit the v3 batch: unique images, cross-class copies, leakage onto the existing splits."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

ROOT = Path(r"E:\项目作业\food_project")
V3 = ROOT/"data/splits/canteen_real_v3"

def dhash(im, n=8):
    im = im.convert("L").resize((n+1, n), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return sum(1 << (i*n+j) for i in range(n) for j in range(n) if a[i, j] < a[i, j+1])

def hamm(a, b): return bin(a ^ b).count("1")

fs = sorted(V3.rglob("*.jpg"))
by_sha = defaultdict(list)
for f in fs:
    by_sha[hashlib.sha256(f.read_bytes()).hexdigest()].append(f)
uniq = sorted(by_sha.values(), key=lambda v: v[0].as_posix())
print(f"files {len(fs)}  unique {len(uniq)}")

# leakage against everything already in the project
ref = []
for sub in ("train", "val", "test", "canteen_clean", "canteen_real"):
    for p in (ROOT/"data/splits"/sub).rglob("*.jpg"):
        ref.append(p)
print("reference pool", len(ref))
refh = [(p, dhash(Image.open(p))) for p in ref]
hits = []
for grp in uniq:
    h = dhash(Image.open(grp[0]))
    best = min(refh, key=lambda t: hamm(h, t[1]))
    if hamm(h, best[1]) <= 6:
        hits.append((grp[0].relative_to(V3).as_posix(), best[0].relative_to(ROOT/"data/splits").as_posix(),
                     hamm(h, best[1])))
print(f"near-duplicates against existing data: {len(hits)}/{len(uniq)}")
for h in hits[:15]: print("   ", h)
json.dump({"n_files": len(fs), "n_unique": len(uniq),
           "dups": {k: [x.relative_to(V3).as_posix() for x in v] for k, v in by_sha.items() if len(v) > 1},
           "leak": hits}, open(ROOT/"results/tables/v3_audit.json", "w", encoding="utf-8"), indent=1)
