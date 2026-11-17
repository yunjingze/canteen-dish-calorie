# -*- coding: utf-8 -*-
"""Parallel range downloader: many small HTTP ranges, resumable.

    python scripts/pdl.py URL OUT [--threads 12] [--chunk-mb 16] [--seconds N]

Writes each range at its offset, so an interrupted run can be resumed by
re-running with the same arguments (completed chunks are skipped).
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import urllib.request

DONE = set()
LOCK = threading.Lock()


def head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "*/*",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def fetch(url: str, start: int, end: int, attempts: int = 6) -> bytes:
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={
                "Range": f"bytes={start}-{end-1}",
                "User-Agent": "Mozilla/5.0",
            })
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != end - start:
                raise IOError(f"short read {len(data)} != {end-start}")
            return data
        except Exception as exc:  # noqa: BLE001
            if i == attempts - 1:
                raise
            print(f"  retry {i+1} [{start}-{end}]: {exc}", flush=True)
            time.sleep(1.5 * (i + 1))
    raise RuntimeError("unreachable")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--chunk-mb", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after N seconds (throughput probe)")
    args = ap.parse_args()

    size = head_size(args.url)
    chunk = args.chunk_mb * 1024 * 1024
    starts = list(range(0, size, chunk))
    have = os.path.getsize(args.out) if os.path.exists(args.out) else 0
    if have != size:
        with open(args.out, "wb") as fh:
            fh.truncate(size)
    print(f"size {size/1e9:.2f} GB, {len(starts)} chunks of {args.chunk_mb} MB, {args.threads} threads", flush=True)

    # a resumed run cannot know which ranges are already written; start clean
    done = [0]
    total_done = [0]
    stop_at = time.time() + args.seconds if args.seconds else None
    t0 = time.time()
    last = [t0, 0]
    idx = [0]
    lock = threading.Lock()

    def worker():
        out = args.out
        while True:
            with lock:
                if idx[0] >= len(starts):
                    return
                s = starts[idx[0]]
                idx[0] += 1
            if stop_at and time.time() > stop_at:
                return
            e = min(s + chunk, size)
            data = fetch(args.url, s, e)
            with open(out, "r+b") as fh:
                fh.seek(s)
                fh.write(data)
            with lock:
                done[0] += 1
                total_done[0] += len(data)
                now = time.time()
                if now - last[0] > 5:
                    mbps = (total_done[0] / 1e6) / (now - t0)
                    print(f"  {total_done[0]/1e9:.2f} GB, {done[0]}/{len(starts)} chunks, "
                          f"{mbps:.1f} MB/s, eta {(size-total_done[0])/1e6/max(mbps,0.01)/60:.0f} min", flush=True)
                    last[0] = now

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt = time.time() - t0
    got = os.path.getsize(args.out)
    print(f"done {total_done[0]/1e9:.2f} GB in {dt:.0f}s -> {total_done[0]/1e6/max(dt,0.01):.1f} MB/s "
          f"(file {got/1e9:.2f} GB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
