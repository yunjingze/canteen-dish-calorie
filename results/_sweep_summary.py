import json, csv
from pathlib import Path
from collections import Counter
ROOT = Path(r"E:\项目作业\food_project")
rows = json.load(open(ROOT/"results/tables/api_sweep.json", encoding="utf-8"))
v1 = json.load(open(ROOT/"data/splits/negatives/v1_categories.json", encoding="utf-8"))
cats = {}
with open(ROOT/"data/splits/negatives/sources.csv", newline="", encoding="utf-8-sig") as fh:
    for r in csv.DictReader(fh):
        cats[Path(r["file"]).name] = r["category"]
neg = [r for r in rows if r["kind"] == "neg"]
for r in neg:
    n = Path(r["file"]).name
    r["cat"] = v1.get(n, cats.get(n, "?"))
food = [r for r in neg if r["cat"] == "food_nonvocab"]
true_neg = [r for r in neg if r["cat"] != "food_nonvocab"]
acc = [r for r in true_neg if r["in_scope"]]
print(f"true negatives accepted {len(acc)}/{len(true_neg)} ({100*len(acc)/len(true_neg):.1f}%)")
print(f"relabelled food accepted {sum(r['in_scope'] for r in food)}/{len(food)}")
print("still accepted true negatives:")
for r in sorted(acc, key=lambda x: -(x["food_prob"] or 0)):
    print(f"  {Path(r['file']).name} cat={r['cat']:16} p_food={r['food_prob']:.3f} robust={r['robust']:.3f} top1={r['top1']}")
print("\nper category accepted:")
for c in sorted({r["cat"] for r in true_neg}):
    m = [r for r in true_neg if r["cat"] == c]
    print(f"  {c:18} {sum(r['in_scope'] for r in m):2}/{len(m):<3}")
real = [r for r in rows if r["kind"] == "real"]
lost = [r for r in real if not r["in_scope"]]
print(f"\nreal kept {len(real)-len(lost)}/{len(real)}; rejected:")
for r in lost:
    print(f"  {r['file']} p_food={r['food_prob']:.3f} robust={r['robust']:.3f} top1={r['top1']}")
