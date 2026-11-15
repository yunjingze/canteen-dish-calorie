import json, shutil
from pathlib import Path
ROOT = Path(r"E:\项目作业\food_project")
src = ROOT/"results/tables/foodgate_linear.json"
g = json.loads(src.read_text(encoding="utf-8"))
g["threshold"] = 0.2
src.write_text(json.dumps(g), encoding="utf-8")
shutil.copyfile(src, ROOT/"results/runs/resnet18_last_stage_aug_f101n/foodgate.json")
print("threshold", g["threshold"], "n_negatives", g.get("n_negatives"), "bytes", src.stat().st_size)