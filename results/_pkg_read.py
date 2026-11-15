
import json, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"E:\项目作业\food_project")
d = json.loads((ROOT/"results/tables/pkg_sweep.json").read_text(encoding="utf-8"))
neg = d["negatives"]
flagged = [r for r in neg if r["packaging"]]
low = [r for r in flagged if (r["p_food"] or 0) < 0.2]
print("negatives flagged", len(flagged), "of which p_food<0.2:", len(low))
print("  e.g.", [(Path(r["path"]).parent.name, r["p_food"], r["p_pack"]) for r in low[:6]])
ins = [r for r in neg if r["in_scope"]]
print("accepted negatives:", [(r["path"], r["top1"], r["p_pack"], r["p_food"]) for r in ins])
pv = d["pack_v3"]
print("pack_v3 p_food<0.2:", sum((r["p_food"] or 0) < 0.2 for r in pv), "min p_food", min(r["p_food"] for r in pv))
print("pack_v3 p_pack min/median", min(r["p_pack"] for r in pv), sorted(r["p_pack"] for r in pv)[31])
real = d["real"]
print("real rejected:", [(r["path"], round(r["robust"],3), round(r["p_food"],3), r["p_pack"]) for r in real if not r["in_scope"]])
