
import io, sys, csv, json, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
ROOT = Path(r"E:\项目作业\food_project")
neg = ROOT/"data/splits/negatives"
txt = (neg/"sources.csv").read_text(encoding="utf-8-sig", errors="replace").splitlines()
print(len(txt)); print(" | ".join(txt[:3]))
rows = list(csv.DictReader(txt))
print("rows", len(rows), list(rows[0].keys()))
print(collections.Counter((r.get("category") or "").strip() for r in rows))
print("v1cats", (neg/"v1_categories.json").read_text(encoding="utf-8")[:400])
