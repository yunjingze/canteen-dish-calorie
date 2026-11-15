from __future__ import annotations
import hashlib, json
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw

ROOT = Path(r"E:\项目作业\food_project")
V3 = ROOT/"data/splits/canteen_real_v3"
by_sha = defaultdict(list)
for f in sorted(V3.rglob("*.jpg")):
    by_sha[hashlib.sha256(f.read_bytes()).hexdigest()].append(f)
print(f"unique {len(by_sha)}")
multi = 0
for k, v in by_sha.items():
    labels = {p.parent.name for p in v}
    if len(labels) > 1:
        multi += 1
print(f"images carrying more than one class label: {multi}")
sizes = defaultdict(int)
for k, v in by_sha.items():
    sizes[len({p.parent.name for p in v})] += 1
print("label-count histogram:", dict(sorted(sizes.items())))

cell, cols = 200, 8
items = sorted(by_sha.items(), key=lambda kv: -len(kv[1]))
rows = (len(items) + cols - 1) // cols
sheet = Image.new("RGB", (cols*cell, rows*cell), "white")
d = ImageDraw.Draw(sheet)
for i, (k, v) in enumerate(items):
    im = Image.open(v[0]).convert("RGB")
    im.thumbnail((cell-6, cell-26))
    x, y = (i % cols)*cell, (i//cols)*cell
    sheet.paste(im, (x+3, y+3))
    labels = sorted({p.parent.name for p in v})
    txt = (labels[0] if len(labels) == 1 else f"{labels[0]} +{len(labels)-1}")[:26]
    d.text((x+4, y+cell-20), txt, fill="black")
sheet.save(ROOT/"results/_v3_unique.png")
print("montage", sheet.size)