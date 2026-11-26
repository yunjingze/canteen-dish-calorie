# -*- coding: utf-8 -*-
"""Build the 22-class canteen dataset: the 20 dishes, rice, and "other Chinese".

Rice (CNFOOD-241 class 172) is the single most common item on a canteen tray, but
the 21-way head folds it into the other_cn bucket, which by design only vetoes and
never contributes a name or an energy estimate.  This script moves rice out of the
other pool into a class of its own, so a tray of rice + one dish can price the rice.

    python scripts/prepare_cn241_rice.py --dry-run
    python scripts/prepare_cn241_rice.py

Output (SplitDataset layout; every image is a hard link where the filesystem allows):

    data/splits/cn241_22/{train,val,test}/<slug>/<n>.jpg
    results/tables/cn241_22_classes.json     # 22 rows: 20 dishes, rice, other_cn

The other_cn bucket is re-derived from the archives with exactly the same pool,
held-out split, per-class caps and RNG draw order as scripts/prepare_cn241_other.py,
except that class 172 is skipped when the bucket is written.  Same seed, same
shuffle sequence, so every surviving image is the one the 21-way bucket used; the
only difference is that rice is no longer inside it.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC21 = ROOT / "data/splits/cn241_21"
SRC_ALL = ROOT / "data/splits/cn241_all"
OUT = ROOT / "data/splits/cn241_22"
TABLE21 = ROOT / "results/tables/cn241_21_classes.json"

RICE_ID = 172
RICE_SLUG = "rice"
CAP = {"train": 12, "val": 2, "test": 8}


def group_zip(zf: zipfile.ZipFile) -> dict[int, list[str]]:
    groups: dict[int, list[str]] = {}
    for n in zf.namelist():
        if n.endswith("/") or not n.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        parts = [p for p in n.split("/") if p]
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


def link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "skip"
    try:
        dst.hardlink_to(src)
        return "link"
    except (OSError, AttributeError):
        shutil.copy2(src, dst)
        return "copy"


def dump_zip(zf: zipfile.ZipFile, members: list[str], dst: Path, start: int = 0) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(members):
        (dst / f"{start + i:05d}.jpg").write_bytes(zf.read(name))
    return len(members)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-21", default=str(SRC21))
    ap.add_argument("--src-all", default=str(SRC_ALL))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--table-21", default=str(TABLE21))
    ap.add_argument("--classes", default=str(ROOT / "results/tables/cn241_22_classes.json"))
    ap.add_argument("--train-zip", default=str(ROOT / "data/external/cnfood241_train600x600.zip"))
    ap.add_argument("--val-zip", default=str(ROOT / "data/external/cnfood241_val600x600.zip"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    src21, src_all, out = Path(args.src_21), Path(args.src_all), Path(args.out)
    base = json.loads(Path(args.table_21).read_text(encoding="utf-8"))
    dishes = [c for c in base["classes"] if not c.get("other")]
    assert len(dishes) == 20, len(dishes)
    pool = [int(i) for i in base["other_pool"]]
    heldout = [int(i) for i in base["heldout_ids"]]
    assert RICE_ID in pool, "rice is not in the 21-way other pool; nothing to move"

    # 1. the 20 dishes: hard links, unchanged
    moved = {"link": 0, "copy": 0, "skip": 0}
    for split in ("train", "val", "test"):
        for c in dishes:
            slug = c["slug"]
            for f in sorted((src21 / split / slug).glob("*.jpg")):
                if args.dry_run:
                    continue
                moved[link_or_copy(f, out / split / slug / f.name)] += 1
    print(f"20 dishes linked: {moved}", flush=True)

    # 2. rice: hard links out of the full corpus (811 / 90 / 66)
    rice_counts = {}
    for split in ("train", "val", "test"):
        files = sorted((src_all / split / "c172").glob("*.jpg"))
        rice_counts[split] = len(files)
        if args.dry_run:
            continue
        for f in files:
            link_or_copy(f, out / split / RICE_SLUG / f.name)
    print(f"rice: {rice_counts}", flush=True)

    # 3. other_cn, re-derived with the 21-way draw order but without class 172
    counts = {split: {"other_cn": 0} for split in ("train", "val", "test")}
    dropped = {split: 0 for split in ("train", "val", "test")}
    if not args.dry_run:
        with zipfile.ZipFile(args.train_zip) as zt:
            groups = group_zip(zt)
            print(f"train archive: {len(zt.namelist())} members, {len(groups)} classes", flush=True)
            for cid in pool:
                members = groups.get(cid, [])
                rng.shuffle(members)              # same call order as prepare_cn241_other.py
                n_tr = int(round(len(members) * 0.9))
                per = {"train": members[:n_tr][:CAP["train"]],
                       "val": members[n_tr:][:CAP["val"]]}
                for split, names in per.items():
                    if cid == RICE_ID:
                        dropped[split] += len(names)
                        continue
                    counts[split]["other_cn"] += dump_zip(
                        zt, names, out / split / "other_cn", counts[split]["other_cn"])
        with zipfile.ZipFile(args.val_zip) as zv:
            groups = group_zip(zv)
            print(f"val archive: {len(zv.namelist())} members, {len(groups)} classes", flush=True)
            for cid in pool:
                names = groups.get(cid, [])[:CAP["test"]]
                if cid == RICE_ID:
                    dropped["test"] += len(names)
                    continue
                counts["test"]["other_cn"] += dump_zip(
                    zv, names, out / "test" / "other_cn", counts["test"]["other_cn"])
    print(f"other_cn counts: {counts} (rice dropped from bucket: {dropped})", flush=True)
    _ = heldout  # the held-out manifest is unchanged; reuse results/tables/cn241_other_heldout.json

    rice_row = {"slug": RICE_SLUG, "zh": "米饭", "en": "Rice", "cnfood_id": RICE_ID,
                "kcal_per_100g": 116, "portion_g": 200,
                "n_train": rice_counts["train"], "n_val": rice_counts["val"],
                "n_test": rice_counts["test"]}
    classes = [dict(c) for c in dishes] + [rice_row]
    classes.append({"slug": "other_cn", "zh": "其它中餐", "en": "Other Chinese dish",
                    "cnfood_id": None, "other": True, "n_classes_covered": len(pool) - 1})
    meta = {"source": "CNFOOD-241 train (20 dishes + rice + other) + val (test)",
            "n_classes": 22, "other_class": "other_cn",
            "rice_id": RICE_ID, "other_pool": [i for i in pool if i != RICE_ID],
            "heldout_ids": heldout, "caps": CAP, "per_class": counts, "classes": classes}
    if args.dry_run:
        print(json.dumps({"n_classes": 22, "other_pool": len(pool) - 1}, ensure_ascii=False))
        return 0
    dest = Path(args.classes)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")
    for split in ("train", "val", "test"):
        n = sum(1 for _ in (out / split).rglob("*.jpg"))
        print(f"  {split}: {n} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
