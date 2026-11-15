from pathlib import Path
from collections import Counter
from PIL import Image
ROOT = Path(r"E:\项目作业\food_project\data\splits\canteen_real_v3")
fs = sorted(f for f in ROOT.rglob("*") if f.is_file())
sizes = Counter()
small = []
for f in fs:
    try:
        im = Image.open(f)
        sizes[(im.width, im.height)] += 1
        if min(im.size) < 400: small.append((f.relative_to(ROOT).as_posix(), im.size, f.stat().st_size))
    except Exception as e:
        print("BAD", f.name, e)
print("distinct sizes:", len(sizes))
for (w,h), c in sizes.most_common(20):
    print(f"  {w}x{h}: {c}")
print("small(<400px):", len(small))
for s in small[:15]: print("   ", s)
print("mixed dir:", [p.name for p in (ROOT/"mixed").iterdir()][:20])