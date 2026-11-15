"""Build a clean extra training set for the 20 canteen classes from Food-101N.

Food-101N shares Food-101's 101 class folders, but it is web-crawled, ~20% of
its class labels are wrong, and some of its images are near-duplicates of the
official Food-101 test/val images.  This script

  1. keeps only the 20 classes used by the project,
  2. drops images that are near-duplicates of the evaluation splits
     (test / val / canteen_clean) using a 64-bit difference hash,
  3. drops images whose verification label is 0 (human-marked as mislabelled),
  4. writes an ImageFolder tree plus a manifest with the provenance of every file.

Usage (after unzipping Food-101N_release):

    python scripts/extract_food101n.py
    python scripts/extract_food101n.py --per-class 2000 --max-hamming 6
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# popcount lookup for the byte view of a 64-bit hash
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path: Path, size: int = 8) -> int:
    """64-bit difference hash; robust to resizing and mild re-compression."""
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).reshape(-1)
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distance between two uint64 arrays (len(h1) x len(h2))."""
    x = np.bitwise_xor(h1[:, None], h2[None, :])
    return _POP[x.view(np.uint8).reshape(x.shape[0], x.shape[1], 8)].sum(axis=2)


def read_verification(src: Path) -> dict[str, str]:
    """Map image file name -> verification label ('1' correct, '0' wrong, '-1' unknown).

    Handles the official meta/verified_{train,val}.tsv layout
    (class/img.jpg <TAB> label) as well as a plain CSV with named columns.
    """
    out: dict[str, str] = {}
    files: list[Path] = []
    for pattern in ("*.csv", "*.tsv", "meta/*.csv", "meta/*.tsv"):
        files.extend(sorted(src.glob(pattern)))
    for path in files:
        delim = "\t" if path.suffix.lower() == ".tsv" else ","
        with open(path, newline="", encoding="utf-8", errors="ignore") as fh:
            rows = [r for r in csv.reader(fh, delimiter=delim) if r and len(r) >= 2]
        if not rows:
            continue
        start = 0 if rows[0][1].strip().lstrip("-").isdigit() else 1
        n = 0
        for row in rows[start:]:
            key = row[0].strip()
            if not key:
                continue
            ver = row[1].strip()
            name = key.replace("\\", "/").split("/")[-1]
            out[name] = ver
            out[Path(name).stem] = ver
            n += 1
        print(f"[meta] {path.relative_to(src)}: {n} verification labels (header skipped: {start == 1})")
    return out


def build_reference(data_root: Path) -> tuple[np.ndarray, list[str]]:
    hashes, names = [], []
    for split in ("test", "val", "canteen_clean"):
        folders = [p for p in (data_root / "splits" / split).iterdir() if p.is_dir()]
        for folder in sorted(folders):
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    try:
                        hashes.append(dhash(p))
                        names.append(f"{split}/{folder.name}/{p.name}")
                    except Exception:
                        continue
    return np.array(hashes, dtype=np.uint64), names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/external/food101n/Food-101N_release"),
                    help="unzipped Food-101N_release directory (contains images/<class>/)")
    ap.add_argument("--dst", default=str(ROOT / "data/external/food101n_20cls"))
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--max-hamming", type=int, default=6,
                    help="drop a Food-101N image if its dHash is within this distance of any eval image; "
                         "use -1 to disable de-duplication (ablation)")
    ap.add_argument("--per-class", type=int, default=0, help="0 = keep everything")
    ap.add_argument("--keep-noisy", action="store_true",
                    help="also keep images with verification label 0 (mislabelled)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    img_root = src / "images" if (src / "images").is_dir() else src
    missing = [c for c in CLASS_NAMES if not (img_root / c).is_dir()]
    if missing:
        raise SystemExit(f"missing class folders under {img_root}: {missing}")

    ver = read_verification(src)
    ref_hash, ref_names = build_reference(Path(args.data_root))
    print(f"[ref] {len(ref_hash)} evaluation images (test + val + canteen_clean)")

    dst = Path(args.dst)
    manifest_rows = []
    stats = {"kept": 0, "dup": 0, "noisy": 0, "degenerate": 0, "error": 0}

    for cls in CLASS_NAMES:
        files = sorted(p for p in (img_root / cls).iterdir() if p.suffix.lower() in IMG_EXTS)
        kept_here = 0
        cand_files, cand_hash = [], []
        for p in files:
            v = ver.get(p.name, ver.get(p.stem, ""))
            if v == "0" and not args.keep_noisy:
                stats["noisy"] += 1
                continue
            try:
                h = dhash(p)
            except Exception:
                stats["error"] += 1
                continue
            cand_files.append(p)
            cand_hash.append(h)

        if cand_files and len(ref_hash):
            hh = np.array(cand_hash, dtype=np.uint64)
            min_dist = np.full(len(hh), 64, dtype=np.int32)
            for lo in range(0, len(hh), 2000):
                block = hamming(hh[lo:lo + 2000], ref_hash).min(axis=1)
                min_dist[lo:lo + 2000] = block
        else:
            min_dist = np.zeros(len(cand_files), dtype=np.int32)

        for p, h, d in zip(cand_files, cand_hash, min_dist):
            if d <= args.max_hamming:
                stats["dup"] += 1
                continue
            if args.per_class and kept_here >= args.per_class:
                continue
            v = ver.get(p.name, ver.get(p.stem, ""))
            out = dst / cls / f"f101n_{p.stem}{p.suffix.lower()}"
            if not args.dry_run:
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    out.write_bytes(p.read_bytes())
            manifest_rows.append({
                "file": str(out.relative_to(dst)).replace("\\", "/"),
                "class_name": cls,
                "verification_label": v,
                "min_ref_hamming": int(d),
                "source": str(p),
            })
            kept_here += 1
            stats["kept"] += 1
        print(f"[{cls:22s}] candidates {len(cand_files):5d} -> kept {kept_here:5d}")

    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)
        with open(dst / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["file", "class_name", "verification_label",
                                                    "min_ref_hamming", "source"])
            writer.writeheader()
            writer.writerows(manifest_rows)
    print("summary:", stats)
    print(f"kept {stats['kept']} images -> {dst}")
    if stats["kept"] == 0:
        print("WARNING: nothing kept; check --src and the class folder names", file=sys.stderr)


if __name__ == "__main__":
    main()
