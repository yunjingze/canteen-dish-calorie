
import json, shutil, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"E:\项目作业\food_project")
src = ROOT/"results/tables/packaging_linear.json"
run = ROOT/"results/runs/resnet18_last_stage_aug_f101n"
shutil.copyfile(src, run/"packaging.json")
d = json.loads((run/"packaging.json").read_text(encoding="utf-8"))
print("published", run/"packaging.json", "threshold", d["threshold"], "w dim", len(d["w"]))
