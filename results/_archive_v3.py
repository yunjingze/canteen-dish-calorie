import shutil
from pathlib import Path
ROOT = Path(r"E:\项目作业\food_project")
src = ROOT/"data/splits/canteen_real_v3"
dst = ROOT/"results/_discarded_v3"
if src.exists():
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))
    n = len(list(dst.rglob("*.jpg")))
    print(f"moved {src} -> {dst}  ({n} jpg)")
else:
    print("already moved; exists:", dst.exists())
print("splits now:", sorted(p.name for p in (ROOT/"data/splits").iterdir()))