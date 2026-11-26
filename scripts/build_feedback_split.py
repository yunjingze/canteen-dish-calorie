# -*- coding: utf-8 -*-
"""Turn the correction CSV into a trainable split.

The feedback endpoint stores one row per correction: the photo, the dishes
the user says are actually there, and the ones they struck out.  Nothing
consumed that file, so every correction was collected and then ignored --
the loop was open.  This script closes it: it reads the CSV and writes
class-folder splits that SplitDataset can load.

    python scripts/build_feedback_split.py            # write the split
    python scripts/build_feedback_split.py --dry-run  # only report

Four modes, four destinations, because each head has its own label space:

  single  -> one dish, a Food-101 class
  cn      -> one dish, one of the canteen classes (rice included)
  wide    -> one dish, one of the 116 wide classes
  mixed   -> a tray: the corrected dishes are the positives, the struck-out
             ones are dropped from that image's label set

A row whose classes do not fit its mode is skipped and counted, not guessed at.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES                      # noqa: E402
from foodcls.classes_cn import CN_MENU_NAMES                 # noqa: E402
from foodcls.classes_wide import WIDE_CLASS_NAMES            # noqa: E402
from foodcls.plate_classes import PLATE_CLASS_NAMES          # noqa: E402

FEEDBACK_DIR = ROOT / "data/splits/feedback"
FEEDBACK_CSV = FEEDBACK_DIR / "labels.csv"
OUT_ROOT = ROOT / "data/splits/feedback_built"

MODES = {
    "single": set(CLASS_NAMES),
    "cn": set(CN_MENU_NAMES),
    "wide": set(WIDE_CLASS_NAMES),
    "mixed": set(PLATE_CLASS_NAMES),
}


def split_list(raw):
    return [p.strip() for p in (raw or "").split(";") if p.strip()]


def plan(rows, root=None):
    """Return ({mode: [(src, [classes])]}, stats) without touching the disk."""
    root = FEEDBACK_DIR if root is None else Path(root)
    out = {m: [] for m in MODES}
    stats = Counter()
    for row in rows:
        mode = (row.get("mode") or "single").strip() or "single"
        stats["rows"] += 1
        if mode not in MODES:
            stats["unknown_mode"] += 1
            continue
        name = (row.get("file") or "").strip()
        src = root / name
        if not name or not src.is_file():
            stats["missing_image"] += 1
            continue
        corrected = split_list(row.get("corrected", ""))
        removed = split_list(row.get("removed", ""))
        known = MODES[mode]
        if any(c not in known for c in corrected + removed):
            stats["bad_class"] += 1
            continue
        if mode == "mixed":
            keep = [c for c in corrected if c not in removed]
            if not keep:
                # A strike-out-only correction still says something: what the
                # model guessed minus what the user deleted is a tray with the
                # remaining dishes as (weaker, model-proposed) positives.
                predicted = split_list(row.get("predicted", ""))
                keep = [c for c in predicted if c not in removed]
                stats["weak_positive"] += bool(keep)
            if not keep:
                stats["no_positive"] += 1
                continue
            out["mixed"].append((src, keep))
            stats["mixed_with_strike_out"] += bool(removed)
        else:
            if len(corrected) != 1:
                stats["not_single_dish"] += 1
                continue
            out[mode].append((src, corrected))
        stats["used"] += 1
    return out, stats


def write(out, out_root=None):
    """Materialise <mode>/<class>/<n>_<name>.jpg and return the manifest."""
    out_root = OUT_ROOT if out_root is None else Path(out_root)
    written = {}
    for mode, items in out.items():
        for src, classes in items:
            cls = "+".join(classes) if mode == "mixed" else classes[0]
            dest = out_root / mode / cls
            dest.mkdir(parents=True, exist_ok=True)
            n = len([p for p in dest.iterdir() if p.suffix == ".jpg"])
            target = dest / f"{n:04d}_{src.name}"
            if not target.is_file():
                shutil.copyfile(src, target)
            written[str(target.relative_to(out_root))] = classes
    return written


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(FEEDBACK_CSV))
    p.add_argument("--out", default=str(OUT_ROOT))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        print(f"no corrections at {csv_path}")
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out, stats = plan(rows, csv_path.parent)
    by_mode = {m: len(v) for m, v in out.items()}
    print(f"rows {stats['rows']} usable {stats['used']} "
          f"(mixed {by_mode['mixed']}, cn {by_mode['cn']}, "
          f"single {by_mode['single']}, wide {by_mode['wide']})")
    SKIP_KEYS = ("unknown_mode", "missing_image", "bad_class", "no_positive", "not_single_dish")
    NOTE_KEYS = ("weak_positive", "mixed_with_strike_out")
    skipped = {k: stats[k] for k in SKIP_KEYS if stats[k]}
    notes = {k: stats[k] for k in NOTE_KEYS if stats[k]}
    if skipped:
        print("skipped:", json.dumps(skipped, ensure_ascii=False))
    if notes:
        print("notes:  ", json.dumps(notes, ensure_ascii=False))
    if args.dry_run:
        for mode, items in out.items():
            for src, classes in items:
                print(f"  {mode:6s} {src.name} -> {classes}")
        return 0
    out_root = Path(args.out)
    written = write(out, out_root)
    summary = {
        "csv": str(csv_path),
        "out": str(out_root),
        "rows": stats["rows"],
        "used": stats["used"],
        "by_mode": by_mode,
        "skipped": skipped,
        "notes": notes,
    }
    tables = ROOT / "results/tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "feedback_split.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(written)} images under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
