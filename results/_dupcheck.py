"""Check the new real-photo set for near-duplicates against every split we train/select on."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(r"E:\项目作业\food_project")

def dhash(path, size=8):
    img = ImageOps.exif_transpose(Image.open(path)).convert("L").resize((size+1, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out

def hashes(folder):
    out = []
    for p in sorted(folder.rglob("*.jpg")):
        try:
            out.append((p, dhash(p)))
        except Exception as e:
            print("skip", p, e)
    return out

new = []
for d in ("canteen_real", "mixed_real", "negatives"):
    new += hashes(ROOT/"data/splits"/d)
print("new images:", len(new))

old = []
for d in ("train", "val", "test", "canteen_clean"):
    old += hashes(ROOT/"data/splits"/d)
print("reference images:", len(old))
oh = np.array([h for _, h in old], dtype=np.uint64)
op = [p for p, _ in old]

def ham(a, b):
    return int(np.uint64(a ^ b).bit_count())

# exact duplicates inside the new set
print("\ninternal near-duplicates (<=4 bits):")
for i in range(len(new)):
    for j in range(i+1, len(new)):
        d = ham(new[i][1], new[j][1])
        if d <= 4:
            print(" ", d, new[i][0].name, new[i][0].parent.name, "<->", new[j][0].parent.name, new[j][0].name)

print("\nmatches against train/val/test/canteen_clean (<=6 bits):")
hits = 0
for p, h in new:
    d = np.array([ham(h, int(o)) for o in oh])
    k = d.argsort()[:3]
    if d[k[0]] <= 6:
        hits += 1
        print(f"  {p.parent.name}/{p.name}: {d[k[0]]} bits -> {op[k[0]].relative_to(ROOT/'data/splits')}")
print("total matching images:", hits, "/", len(new))
