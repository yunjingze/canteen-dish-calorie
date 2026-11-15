import sys, random
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
    v = mx
    s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    white = (v > 0.90) & (s < 0.12)
    glare = (v > 0.97) & (s < 0.18)
    gray = a.mean(2)
    gx = np.zeros_like(gray); gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    edge = np.hypot(gx, gy)
    edge_density = float((edge > 0.22).mean())
    strong = edge > 0.18
    hrun = np.zeros_like(strong, dtype=np.int32); run = np.zeros(strong.shape[1], np.int32)
    for i in range(strong.shape[0]):
        run = (run + 1) * strong[i]
        hrun[i] = run
    vrun = np.zeros_like(strong, dtype=np.int32); run = np.zeros(strong.shape[0], np.int32)
    for j in range(strong.shape[1]):
        run = (run + 1) * strong[:, j]
        vrun[:, j] = run
    straight = float(((hrun > size * 0.12) | (vrun > size * 0.12)).mean())
    sat_block = float(((s > 0.45) & (edge < 0.05)).mean())
    rg, yb = r - g, 0.5 * (r + g) - b
    colorful = float(np.hypot(rg.std(), yb.std()) + np.hypot(rg.mean(), yb.mean()))
    return dict(white=float(white.mean()), glare=float(glare.mean()), edge=edge_density,
                straight=straight, sat_block=sat_block, colorful=colorful)

dumpling = feats(Image.open(ROOT/"results/_probe_crop.png"))
print("dumpling", {k: round(v, 4) for k, v in dumpling.items()}, flush=True)

random.seed(0)
for split, n in (("canteen", 400), ("test", 400)):
    files = sorted((ROOT/"data/splits"/split).rglob("*.jpg"))
    files = random.sample(files, min(n, len(files)))
    rows = [feats(Image.open(f)) for f in files]
    print(split, len(files), flush=True)
    for k in dumpling:
        vals = np.array([r[k] for r in rows])
        q = np.percentile(vals, [50, 90, 99, 99.9])
        rank = float((vals < dumpling[k]).mean())
        print(f"  {k:9s} p50={q[0]:.4f} p90={q[1]:.4f} p99={q[2]:.4f} p999={q[3]:.4f} dumpling_pct={rank*100:.1f}")
