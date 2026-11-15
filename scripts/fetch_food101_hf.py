"""Download the Food-101 parquet shards from the HuggingFace mirror.

Used instead of the 5 GB ETHZ tarball because the mirror is ~6x faster from
this network. The shards are converted to a plain ImageFolder layout by
extract_food101_hf.py.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://hf-mirror.com/datasets/food101/resolve/main/data/"
FILES = [f"train-{i:05d}-of-00008.parquet" for i in range(8)] + [
    f"validation-{i:05d}-of-00003.parquet" for i in range(3)
]


def fetch(name: str, out_dir: Path) -> str:
    dst = out_dir / name
    if dst.exists() and dst.stat().st_size > 1_000_000:
        return f"skip {name} ({dst.stat().st_size/1e6:.0f} MB)"
    tmp = dst.with_suffix(".part")
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "foodcls/1.0"})
    t0, done = time.time(), 0
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as fh:
        total = int(r.headers.get("Content-Length", 0))
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if done % (50 << 20) < (1 << 20):
                mb = done / 1e6
                print(f"  {name}: {mb:.0f}/{total/1e6:.0f} MB  {mb/(time.time()-t0):.1f} MB/s", flush=True)
    tmp.replace(dst)
    return f"done {name} ({dst.stat().st_size/1e6:.0f} MB in {time.time()-t0:.0f}s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw_hf")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for msg in ex.map(lambda n: fetch(n, out), FILES):
            print(msg, flush=True)


if __name__ == "__main__":
    sys.exit(main())
