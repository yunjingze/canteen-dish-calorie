from pathlib import Path
from collections import defaultdict
import hashlib
from PIL import Image
ROOT = Path(r"E:\项目作业\food_project\data\splits\canteen_real_v3")
fs = sorted(f for f in ROOT.rglob("*.jpg"))
by_sha = defaultdict(list)
for f in fs:
    by_sha[hashlib.sha256(f.read_bytes()).hexdigest()].append(f.relative_to(ROOT).as_posix())
dups = {k: v for k, v in by_sha.items() if len(v) > 1}
print(f"files {len(fs)}  unique bytes {len(by_sha)}  duplicate groups {len(dups)}")
for k, v in sorted(dups.items(), key=lambda kv: -len(kv[1]))[:20]:
    print("  equal:", v)
# EXIF check
with_exif = 0
for f in fs[:40]:
    ex = Image.open(f).getexif()
    if ex and len(ex): with_exif += 1
print("first 40 with EXIF:", with_exif)