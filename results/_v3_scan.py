from pathlib import Path
from collections import Counter
ROOT = Path(r"E:\项目作业\food_project\data\splits\canteen_real_v3")
print("exists:", ROOT.exists())
if ROOT.exists():
    for p in sorted(ROOT.iterdir()):
        if p.is_dir():
            fs = [f for f in p.rglob("*") if f.is_file()]
            print(f"  {p.name:24} {len(fs):4} files  exts={Counter(f.suffix.lower() for f in fs)}")
        else:
            print(f"  {p.name:24} file {p.stat().st_size} B")
    allf = [f for f in ROOT.rglob("*") if f.is_file()]
    print("total files", len(allf), "total MB", round(sum(f.stat().st_size for f in allf)/1e6, 1))