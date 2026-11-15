import numpy as np
from pathlib import Path
ROOT = Path(r"E:\项目作业\food_project")
D = np.load(ROOT/"results/_pkgfeat.npz", allow_pickle=True)
f = D["test_feat"]; paths = D["test_paths"]
hit = (f[:,3] >= 0.02) & (f[:,2] < 0.06)
print("hits", int(hit.sum()))
for p, row in zip(paths[hit], f[hit]):
    print(f"{row[3]:.4f} straight {row[2]:.4f} edge  {Path(p).parent.name}/{Path(p).name}  {p}")
