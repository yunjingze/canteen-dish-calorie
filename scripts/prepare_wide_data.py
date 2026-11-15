"""Build the wide-model splits (101 classes) from the full Food-101 images.

    python scripts/extract_food101_hf.py --all --dst data/wide/raw/food-101
    python scripts/prepare_wide_data.py

Layout produced under data/wide/:

    splits/train/<class>/*.jpg   650 img/class  (official train split)
    splits/val/<class>/*.jpg     100 img/class  (held out of official train)
    splits/test/<class>/*.jpg    250 img/class  (official test split)

Hard links are used when the filesystem allows it, so this costs no extra disk.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes_wide import WIDE_FOOD101_CLASSES  # noqa: E402
from foodcls.data import prepare_splits  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "data/wide"))
    ap.add_argument("--train", type=int, default=650)
    ap.add_argument("--val", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # only the 101 Food-101 classes live under raw/; the 15 Chinese dishes are
    # written straight into splits/ by scripts/add_chinese_dishes.py, so asking
    # for all 116 here would make prepare_splits miss 15 raw folders.
    prepare_splits(args.data_root, WIDE_FOOD101_CLASSES, seed=args.seed,
                   n_train=args.train, n_val=args.val, n_canteen=0)
    root = Path(args.data_root) / "splits"
    for split in ("train", "val", "test"):
        n = sum(1 for _ in (root / split).rglob("*.jpg"))
        print(f"{split}: {n} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
