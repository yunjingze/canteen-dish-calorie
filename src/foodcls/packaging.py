"""Cheap "is this a package, not a plate" check.

The demo is trained on plated food: Food-101 (restaurant plating), Food-101N
(web photos) and a canteen set.  A supermarket package is a different domain --
shrink-wrap glare, a printed label with large type, flat saturated ink -- and the
20-class head has nothing sensible to say about it.  Rather than let it answer
anyway, look for the two cues such a photo always has:

* long straight high-contrast runs (label edges, printed rules, seams, the
  rectangular border of the tray), and
* low fine-grained edge density: print is flat and clean, food is textured.

Both are measured on a 256x256 crop with numpy only, so the check costs about a
millisecond and needs no extra model or OCR.

Calibration (results/_pkgfeat_all.py, whole splits):
  canteen 1000 photos: 0 hits with STRAIGHT_MIN=0.020 and EDGE_MAX=0.060
  Food-101 test 5000 photos: 4 hits (0.08%), all plates with strong straight
  rims/placemat edges -- and every one of them is confidently in scope, where the
  flag is not used.  It only ever refines the message of a photo the model has
  already called out of scope, so a false alarm costs nothing.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

STRAIGHT_MIN = 0.020   # fraction of pixels on a >12%-of-width straight strong edge
EDGE_MAX = 0.060       # fine-texture density; printed packaging stays well below


def _features(img: Image.Image, size: int = 256) -> dict:
    small = img.convert("RGB").resize((size, size), Image.BILINEAR)
    a = np.asarray(small, dtype=np.float32) / 255.0
    mx = a.max(2)
    mn = a.min(2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    gray = a.mean(2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    edge = np.hypot(gx, gy)

    strong = edge > 0.18
    hrun = np.zeros(strong.shape, dtype=np.int32)
    run = np.zeros(strong.shape[1], dtype=np.int32)
    for i in range(strong.shape[0]):
        run = (run + 1) * strong[i]
        hrun[i] = run
    vrun = np.zeros(strong.shape, dtype=np.int32)
    run = np.zeros(strong.shape[0], dtype=np.int32)
    for j in range(strong.shape[1]):
        run = (run + 1) * strong[:, j]
        vrun[:, j] = run
    long_min = size * 0.12

    return {
        "straight": float(((hrun > long_min) | (vrun > long_min)).mean()),
        "edge": float((edge > 0.22).mean()),
        "white": float(((mx > 0.90) & (sat < 0.12)).mean()),
        "glare": float(((mx > 0.97) & (sat < 0.18)).mean()),
    }


def signals(img: Image.Image) -> dict:
    """Return the packaging cues and the verdict used by the API payload."""
    f = _features(img)
    straight, edge = f["straight"], f["edge"]
    detected = bool(straight >= STRAIGHT_MIN and edge < EDGE_MAX)
    # one number in [0, 1] for logging: how far past the line the photo is
    score = min(1.0, (straight / STRAIGHT_MIN) / 3.0) if detected else 0.0
    return {"detected": detected, "score": round(score, 3), **f}


def message() -> str:
    return (
        "这看起来是包装食品的照片（包装膜反光、大面积印刷文字），不是餐盘里的菜。"
        "包装照不在这个演示的训练范围里，硬套 20 类只会给出误导的答案，所以这次不给候选。"
        "拆开装到盘子里、或者直接拍食堂/餐厅的餐盘再试一次。"
    )
