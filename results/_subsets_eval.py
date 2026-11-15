import sys, itertools, json
from pathlib import Path
import numpy as np
sys.path.insert(0, r"E:\项目作业\food_project\src")
from foodcls.classes import CLASS_NAMES
ROOT = Path(r"E:\项目作业\food_project")
D = np.load(ROOT/"results/_viewlogits.npz")
G = np.load(ROOT/"results/_dumpling_logits.npz")
names = ["c256", "c320", "c448", "sq", "c256f", "c320f", "c448f", "sqf"]
yt, yc = D["labels_test"], D["labels_canteen"]
rows = []
for r in range(1, len(names) + 1):
    for combo in itertools.combinations(names, r):
        zt = np.mean([D[f"test_{n}"] for n in combo], 0)
        zc = np.mean([D[f"canteen_{n}"] for n in combo], 0)
        at = 100 * (zt.argmax(1) == yt).mean(); ac = 100 * (zc.argmax(1) == yc).mean()
        zd = np.mean([G[n] for n in combo], 0)
        p = np.exp(zd - zd.max()); p /= p.sum()
        order = p.argsort()[::-1][:3]
        rows.append((at, ac, at + ac, combo, [(CLASS_NAMES[i], round(float(p[i]), 3)) for i in order]))
rows.sort(key=lambda r: -r[2])
print(f"{'test':>6} {'canteen':>7} {'sum':>6}  combo / dumpling-photo top3")
for at, ac, s, combo, top in rows[:12]:
    print(f"{at:6.2f} {ac:7.2f} {s:6.2f}  {'+'.join(combo):32s} {top}")
print("--- best for the dumpling photo (dumplings rank1) ---")
dum = [r for r in rows if r[4][0][0] == "dumplings"]
dum.sort(key=lambda r: -r[2])
for at, ac, s, combo, top in dum[:8]:
    print(f"{at:6.2f} {ac:7.2f} {s:6.2f}  {'+'.join(combo):32s} {top}")
print("--- current six ---")
for r in rows:
    if r[3] == ("c256", "c320", "sq", "c256f", "c320f", "sqf"):
        print(f"{r[0]:6.2f} {r[1]:7.2f} {r[2]:6.2f}  {'+'.join(r[3])} {r[4]}")
