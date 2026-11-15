import json
from pathlib import Path
rows = json.load(open(r"E:\项目作业\food_project\results\tables\v3_api.json", encoding="utf-8"))
n = len(rows)
acc = [r for r in rows if r["in_scope"]]
nf = [r for r in rows if r["not_food"]]
rob_only = [r for r in rows if not r["in_scope"] and not r["not_food"]]
print(f"n={n} in_scope={len(acc)} not_food={len(nf)} out_by_robust={len(rob_only)}")
print("rejected examples:", [(r['file'], r['p_food'], round(r['robust'],2)) for r in nf[:6]])
print("p_food median accepted:", sorted(r['p_food'] for r in acc)[len(acc)//2])
