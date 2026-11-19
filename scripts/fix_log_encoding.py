"""Rewrite UTF-16LE logs under results/ as UTF-8 so the read tool can open them.

PS 5.1's ">" / Tee-Object writes UTF-16LE; every training/eval log in this repo
ended up that way, which makes them unreadable without a conversion step.

usage: python scripts/fix_log_encoding.py [--dry-run] [--no-backup] [patterns...]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKUP = ROOT / "results" / "_logs_utf16"


def sniff(raw: bytes) -> str:
    if raw[:2] == b"\xff\xfe":
        return "utf-16-le"
    if raw[:2] == b"\xfe\xff":
        return "utf-16-be"
    if raw[:3] == b"\xef\xbb\xbf":
        return "utf-8-sig"
    if raw and raw.count(b"\x00") > len(raw) // 4:
        return "utf-16-le"
    return "utf-8"


def convert(path: Path, backup: bool, dry: bool) -> str:
    raw = path.read_bytes()
    codec = sniff(raw)
    if codec == "utf-8":
        return "skip (already utf-8)"
    text = raw.decode(codec, errors="replace").lstrip("\ufeff")  # drop the BOM itself
    if backup and not dry:
        BACKUP.mkdir(parents=True, exist_ok=True)
        keep = BACKUP / path.name
        if not keep.exists():  # never overwrite the true UTF-16 original
            shutil.copy2(path, keep)
    payload = text.encode("utf-8")
    if not dry:
        try:
            path.write_bytes(payload)
        except OSError as exc:  # e.g. a running server still has the file open
            return f"locked, skipped ({exc.errno})"
    return f"{codec} -> utf-8 ({len(raw)} -> {len(payload)} B)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("patterns", nargs="*", default=["results/_*.log"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    paths: list[Path] = []
    for pat in args.patterns:
        paths.extend(sorted(ROOT.glob(pat)))
    seen: set[Path] = set()
    changed = 0
    for p in paths:
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        state = convert(p, backup=not args.no_backup, dry=args.dry_run)
        if not state.startswith("skip"):
            changed += 1
            print(f"{p.relative_to(ROOT)}: {state}")
    print(f"\n{changed} file(s) rewritten, {len(seen)} inspected"
          + (" (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
