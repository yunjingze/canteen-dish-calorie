import json
from pathlib import Path
from collections import Counter
ROOT = Path(r"E:\项目作业\food_project")
real = ROOT/"data/splits/canteen_real"
cnt = Counter()
for p in real.rglob("*.jpg"):
    cnt[p.parent.name] += 1
rows = json.load(open(ROOT/"results/tables/api_sweep.json", encoding="utf-8"))
rej = [r for r in rows if r["kind"]=="real" and not r["in_scope"]]
wrong = {}
for r in rows:
    if r["kind"]=="real" and r["in_scope"] and r.get("top1"):
        cls = Path(r["file"]).parent.name
        if r["top1"] != cls:
            wrong[cls] = wrong.get(cls, 0) + 1
print(f"{'class':22}{'n':>4}{'wrong':>7}")
for c in sorted(cnt):
    print(f"{c:22}{cnt[c]:>4}{wrong.get(c,0):>7}")
print("rejected:", [(Path(r['file']).parent.name, round(r['robust'],3), round(r['food_prob'],2)) for r in rej])