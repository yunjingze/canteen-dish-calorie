"""Download Food-101 and materialise the train/val/test/canteen splits."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.data import prepare_splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    prepare_splits(args.data_root, CLASS_NAMES, seed=args.seed)
    root = Path(args.data_root) / "splits"
    for split in ("train", "val", "test", "canteen", "canteen_clean"):
        n = sum(1 for _ in (root / split).rglob("*.jpg"))
        print(f"{split:14s} {n:6d} images")


if __name__ == "__main__":
    main()
