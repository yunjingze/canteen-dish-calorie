import numpy as np
from pathlib import Path
ROOT = Path(r"E:\项目作业\food_project")
D = np.load(ROOT/"results/_pkgfeat.npz", allow_pickle=True)
KEYS = ["white", "glare", "edge", "straight", "sat_block", "colorful"]
F = {s: D[f"{s}_feat"] for s in ("canteen", "test")}
d = D["dumpling_feat"][0]
print("dumpling", dict(zip(KEYS, np.round(d, 4))))
for rule_name, rule in [
    ("straight>=0.015", lambda f: f[:,3] >= 0.015),
    ("straight>=0.020", lambda f: f[:,3] >= 0.020),
    ("straight>=0.025", lambda f: f[:,3] >= 0.025),
    ("str>=0.015 & glare>=0.02", lambda f: (f[:,3] >= 0.015) & (f[:,1] >= 0.02)),
    ("str>=0.015 & white>=0.04", lambda f: (f[:,3] >= 0.015) & (f[:,0] >= 0.04)),
    ("str>=0.012 & glare>=0.03", lambda f: (f[:,3] >= 0.012) & (f[:,1] >= 0.03)),
    ("str>=0.02 & edge<0.06", lambda f: (f[:,3] >= 0.02) & (f[:,2] < 0.06)),
]:
    ic = rule(F["canteen"]).mean(); it = rule(F["test"]).mean()
    hit = bool(rule(d[None, :])[0])
    print(f"{rule_name:28s} canteen_fp={ic*100:5.2f}%  test_fp={it*100:5.2f}%  dumpling={hit}")
