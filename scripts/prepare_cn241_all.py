# -*- coding: utf-8 -*-
"""Extract all 241 CNFOOD-241 classes, for pretraining a Chinese-dish backbone.

The 20-dish canteen head is fine-tuned straight from a Food-101N backbone, so the
trunk has never seen most Chinese dishes and the weak classes (farmhouse pork,
sweet-and-sour ribs, poached chicken) sit in the low 70s.  Pretraining on the
whole 241-class corpus first is the documented fix; this script builds that
corpus in the SplitDataset layout the trainers already read.

    python scripts/prepare_cn241_all.py

Output:

    data/splits/cn241_all/{train,val,test}/<slug>/<n>.jpg
    results/tables/cn241_all_classes.json

Slugs are the archive folder ids (c000..c240); the Chinese/English names come
from results/_cnfood_classes.txt.  train/val are a 90/10 split of
train600x600.zip, test is all of val600x600.zip -- the same protocol as
scripts/prepare_cn241.py, so the fine-tune test set is never seen in training.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ROOT / "results/_cnfood_classes.txt"
IMAGE_EXT = (".jpg", ".jpeg", ".png")


def read_names(path: Path) -> dict[int, tuple[str, str]]:
    names: dict[int, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            cid = int(parts[0])
        except ValueError:
            continue
        names[cid] = (parts[1].strip(), parts[2].strip())
    return names


def group(zf: zipfile.ZipFile) -> dict[int, list[str]]:
    groups: dict[int, list[str]] = {}
    for n in zf.namelist():
        if n.endswith('/') or not n.lower().endswith(IMAGE_EXT):
            continue
        parts = [p for p in n.split('/') if p]
        if len(parts) < 2:
            continue
        try:
            cid = int(parts[-2])
        except ValueError:
            continue
        groups.setdefault(cid, []).append(n)
    for v in groups.values():
        v.sort()
    return groups


def dump(zf: zipfile.ZipFile, members: list[str], dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(members):
        (dst / f"{i:05d}.jpg").write_bytes(zf.read(name))
    return len(members)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-zip", default=str(ROOT / "data/external/cnfood241_train600x600.zip"))
    ap.add_argument("--val-zip", default=str(ROOT / "data/external/cnfood241_val600x600.zip"))
    ap.add_argument("--out", default=str(ROOT / "data/splits/cn241_all"))
    ap.add_argument("--classes", default=str(ROOT / "results/tables/cn241_all_classes.json"))
    ap.add_argument("--names", default=str(NAMES))
    ap.add_argument("--train-split", default="0.9")
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    rng = random.Random(args.seed)
    tr_frac = float(args.train_split)
    names = read_names(Path(args.names))
    rows, counts = [], {}
    t0 = time.perf_counter()

    with zipfile.ZipFile(args.train_zip) as zt, zipfile.ZipFile(args.val_zip) as zv:
        gtr, gte = group(zt), group(zv)
        print(f"train archive: {len(zt.namelist())} members, {len(gtr)} classes; "
              f"val archive: {len(zv.namelist())} members, {len(gte)} classes", flush=True)
        ids = sorted(set(gtr) | set(gte))
        for k, cid in enumerate(ids, 1):
            slug = f"c{cid:03d}"
            members = list(gtr.get(cid, []))
            rng.shuffle(members)
            if args.cap > 0:
                members = members[: args.cap]
            n_tr = int(round(len(members) * tr_frac))
            row = {"slug": slug, "cnfood_id": cid,
                   "zh": names.get(cid, ("", ""))[0], "en": names.get(cid, ("", ""))[1],
                   "n_train": 0, "n_val": 0, "n_test": 0}
            if not args.dry_run:
                row["n_train"] = dump(zt, members[:n_tr], out / "train" / slug)
                row["n_val"] = dump(zt, members[n_tr:], out / "val" / slug)
                row["n_test"] = dump(zv, list(gte.get(cid, [])), out / "test" / slug)
            else:
                row["n_train"], row["n_val"] = n_tr, len(members) - n_tr
                row["n_test"] = len(gte.get(cid, []))
            rows.append(row)
            if k % 20 == 0 or k == len(ids):
                print(f"  {k}/{len(ids)} classes, {time.perf_counter()-t0:.0f}s", flush=True)

    total = {s: sum(r[f"n_{s}"] for r in rows) for s in ("train", "val", "test")}
    table = {"source": "CNFOOD-241 train600x600 (train/val) + val600x600 (test)",
             "n_classes": len(rows), "splits": {"train": tr_frac, "val": 1 - tr_frac,
                                                "test": "all of val600x600"},
             "cap": args.cap, "totals": total, "classes": rows}
    print(f"{len(rows)} classes: {total}", flush=True)
    if args.dry_run:
        return 0
    dest = Path(args.classes)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
