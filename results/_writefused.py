import json
from pathlib import Path
ROOT = Path(r"E:\项目作业\food_project")
FIT = {"resnet18_last_stage_aug": 0.659, "resnet18_last_stage_aug_f101n": 0.671}
for run, t in FIT.items():
    for p in (ROOT/"results/tables/calibration.json", ROOT/f"results/runs/{run}/calibration.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(d, list):
            for e in d:
                if e.get("run") == run:
                    e["temperature_fused"] = t
        else:
            d["temperature_fused"] = t
        p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        print(p, "ok", flush=True)
