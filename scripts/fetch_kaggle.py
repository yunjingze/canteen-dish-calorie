"""Download a Kaggle dataset without an API token, in parallel.

Kaggle's public dataset endpoint redirects to a signed storage URL.  A single
connection is slow from some networks, so this script splits the file into byte
ranges and fetches them concurrently, then verifies the size.

Usage:
    python scripts/fetch_kaggle.py --dataset patzer0/16-famous-chinese-dishes \
        --out data/external/16chinese.zip
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

API = "https://www.kaggle.com/api/v1/datasets/download/{ref}"


def resolve(ref: str) -> tuple[str, int]:
    req = urllib.request.Request(API.format(ref=ref), method="GET")
    with urllib.request.urlopen(req, timeout=60) as r:
        final = r.geturl()
        size = int(r.headers.get("Content-Length") or 0)
    if not final.startswith("http"):
        raise RuntimeError("unexpected download URL: " + final[:80])
    return final, size


def fetch_range(url: str, start: int, end: int, path: str) -> int:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    data = urllib.request.urlopen(req, timeout=120).read()
    with open(path, "r+b") as f:
        f.seek(start)
        f.write(data)
    return len(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="owner/slug")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=8)
    args = ap.parse_args()

    url, size = resolve(args.dataset)
    if size <= 0:
        raise SystemExit("server did not report a size")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.truncate(size)

    chunk = args.chunk_mb << 20
    ranges = [(a, min(a + chunk - 1, size - 1)) for a in range(0, size, chunk)]
    done = 0
    started = time.time()
    print(f"{size / 1e6:.1f} MB in {len(ranges)} chunks, {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_range, url, a, b, args.out) for a, b in ranges]
        for fut in futures:
            done += fut.result()
            if done % (64 << 20) < chunk:
                rate = done / 1e6 / max(time.time() - started, 1e-6)
                print(f"  {done / 1e6:6.1f} MB  {rate:4.2f} MB/s", flush=True)
    got = os.path.getsize(args.out)
    if got != size:
        raise SystemExit(f"size mismatch: {got} != {size}")
    print(f"saved {args.out} ({got} bytes) in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
