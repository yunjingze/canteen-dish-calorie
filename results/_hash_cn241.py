# -*- coding: utf-8 -*-
"""Fingerprint every image the 21-way head was fitted or scored on (train, val,
test) so an outside photo set can be checked for duplicates before it is called
"external". Test matters: an external image that also sits in our own test split
would leak the number we are trying to report."""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/tables/_fingerprint_cn241_21.json"


def dhash(img):
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def main():
    rows = {}
    t0 = time.time()
    files = []
    for split in ("train", "val", "test"):
        files += sorted((ROOT / "data/splits/cn241_21" / split).rglob("*.jpg"))
    print(len(files), "images", flush=True)
    for i, p in enumerate(files):
        b = p.read_bytes()
        h = hashlib.md5(b).hexdigest()
        try:
            with Image.open(p) as im:
                d = dhash(im)
        except Exception:
            d = None
        rows[str(p.relative_to(ROOT)).replace("\\", "/")] = [h, d]
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(files)} {(i+1)/(time.time()-t0):.0f} img/s", flush=True)
    OUT.write_text(json.dumps({"n": len(rows), "items": rows}), "utf-8")
    print("wrote", OUT.relative_to(ROOT), f"{OUT.stat().st_size/1e6:.1f} MB")


main()
