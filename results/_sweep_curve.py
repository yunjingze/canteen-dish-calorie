import json
from pathlib import Path
ROOT = Path(r"E:\项目作业\food_project")
rows = json.load(open(ROOT/"results/tables/api_sweep.json", encoding="utf-8"))
v1 = json.load(open(ROOT/"data/splits/negatives/v1_categories.json", encoding="utf-8"))
neg = [r for r in rows if r["kind"] == "neg" and Path(r["file"]).name not in v1]
real = [r for r in rows if r["kind"] == "real"]
print(f"{'t':>5} {'neg accepted':>13} {'real kept':>10}")
for t in (0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7):
    na = sum(1 for r in neg if r["robust"] >= 0.32 and r["food_prob"] >= t)
    rk = sum(1 for r in real if r["robust"] >= 0.32 and r["food_prob"] >= t)
    print(f"{t:5.2f} {na:>8}/{len(neg):<5} {rk:>6}/{len(real)}")
print()
for thr in (0.25,0.28,0.32,0.36):
    na = sum(1 for r in neg if r["robust"] >= thr and r["food_prob"] >= 0.2)
    rk = sum(1 for r in real if r["robust"] >= thr and r["food_prob"] >= 0.2)
    print(f"robust>={thr:.2f}, p_food>=0.20: neg {na}/{len(neg)}  real {rk}/{len(real)}")
