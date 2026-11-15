import sys, json
from pathlib import Path
sys.path.insert(0, r"E:\项目作业\food_project\src")
import numpy as np
from PIL import Image

ROOT = Path(r"E:\项目作业\food_project")

def feats(im, size=256):
    im = im.convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(2), a.min(2)
    s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    white = float(((mx > 0.90) & (s < 0.12)).mean())
    glare = float(((mx > 0.97) & (s < 0.18)).mean())
    gray = a.mean(2)
    gx = np.zeros_like(gray); gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    edge = np.hypot(gx, gy)
    strong = edge > 0.18
    hrun = np.zeros_like(strong, dtype=np.int32); run = np.zeros(strong.shape[1], np.int32)
    for i in range(strong.shape[0]):
        run = (run + 1) * strong[i]; hrun[i] = run
    vrun = np.zeros_like(strong, dtype=np.int32); run = np.zeros(strong.shape[0], np.int32)
    for j in range(strong.shape[1]):
        run = (run + 1) * strong[:, j]; vrun[:, j] = run
    straight = float(((hrun > size * 0.12) | (vrun > size * 0.12)).mean())
    sat_block = float(((s > 0.45) & (edge < 0.05)).mean())
    rg, yb = r - g, 0.5 * (r + g) - b
    colorful = float(np.hypot(rg.std(), yb.std()) + np.hypot(rg.mean(), yb.mean()))
    return (white, glare, float((edge > 0.22).mean()), straight, sat_block, colorful)

KEYS = ["white", "glare", "edge", "straight", "sat_block", "colorful"]
out = {}
for split in ("canteen", "test"):
    files = sorted((ROOT/"data/splits"/split).rglob("*.jpg"))
    rows = [feats(Image.open(f)) for f in files]
    out[split + "_feat"] = np.array(rows, dtype=np.float32)
    out[split + "_paths"] = np.array([f.as_posix() for f in files])
    print(split, len(files), flush=True)
out["dumpling_feat"] = np.array([feats(Image.open(ROOT/"results/_probe_crop.png"))], dtype=np.float32)
np.savez_compressed(ROOT/"results/_pkgfeat.npz", **out)
print("saved", flush=True)
