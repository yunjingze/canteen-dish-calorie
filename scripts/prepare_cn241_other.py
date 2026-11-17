# -*- coding: utf-8 -*-
"""Build the 21-class canteen dataset: the 20 dishes plus an "other Chinese" class.

The 20-dish head cannot say "I do not know this dish": on dishes outside its
list it still names one of its 20 classes (scripts/eval_cn_crosscheck.py).  The
documented fix is a 21st class trained on the rest of CNFOOD-241, so the head has
somewhere to put a dish it was never taught.

    python scripts/prepare_cn241_other.py --dry-run
    python scripts/prepare_cn241_other.py

Output (SplitDataset layout, the 20 class folders are hard links, not copies):

    data/splits/cn241_21/{train,val,test}/<slug>/<n>.jpg
    results/tables/cn241_21_classes.json      # 21 rows, other_cn last
    results/tables/cn241_other_heldout.json   # unseen-dish OOD eval manifest

The held-out classes (HELDOUT_IDS) are removed from the "other" pool entirely so
scripts/eval_cn_other.py can ask the harder question: does the head refuse a
Chinese dish it has never seen in any form?
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/splits/cn241_full"
OUT = ROOT / "data/splits/cn241_21"

#: classes of CNFOOD-241 that are not Chinese dishes at all (western food and
#: desserts); putting them in "other Chinese" would make the label a lie.
EXCLUDE_IDS = {10, 147, 153, 195, 196, 197, 198, 199, 200, 201, 202, 203,
               204, 205, 206, 207, 222}

#: per-class caps: enough variety per dish without letting "other" swamp the
#: 20 real classes (each of those has ~835 training images)
CAP = {"train": 12, "val": 2, "test": 8}
HELDOUT_CAP = 8


def read_table(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--classes", default=str(ROOT / "results/tables/cn241_classes_full.json"))
    ap.add_argument("--train-zip", default=str(ROOT / "data/external/cnfood241_train600x600.zip"))
    ap.add_argument("--val-zip", default=str(ROOT / "data/external/cnfood241_val600x600.zip"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    src, out = Path(args.src), Path(args.out)
    table = read_table(Path(args.classes))
    picks = [c["slug"] for c in table["classes"]]
    pick_ids = {int(c["cnfood_id"]) for c in table["classes"]}
    assert len(picks) == 20, picks

    # 1. the 20 real classes: hard links into the new root, same file names
    moved = {"link": 0, "copy": 0, "skip": 0}
    for split in ("train", "val", "test"):
        for slug in picks:
            for f in sorted((src / split / slug).glob("*.jpg")):
                if args.dry_run:
                    continue
                moved[link_or_copy(f, out / split / slug / f.name)] += 1
    print(f"20 classes linked: {moved}", flush=True)

    # 2. which CNFOOD classes carry the "other" mass
    pool = [i for i in range(241) if i not in pick_ids and i not in EXCLUDE_IDS]
    heldout = pool[3::7]
    pool = [i for i in pool if i not in heldout]
    print(f"other pool: {len(pool)} classes, held out: {len(heldout)} classes", flush=True)

    counts = {split: {"other_cn": 0} for split in ("train", "val", "test")}
    if not args.dry_run:
        with zipfile.ZipFile(args.train_zip) as zt:
            groups = group_zip(zt)
            print(f"train archive: {len(zt.namelist())} members, {len(groups)} classes", flush=True)
            for cid in pool:
                members = groups.get(cid, [])
                rng.shuffle(members)
                n_tr = int(round(len(members) * 0.9))
                per = {"train": members[:n_tr][:CAP["train"]],
                       "val": members[n_tr:][:CAP["val"]]}
                for split, names in per.items():
                    counts[split]["other_cn"] += dump_zip(zt, names, out / split / "other_cn",
                                                          counts[split]["other_cn"])
        with zipfile.ZipFile(args.val_zip) as zv:
            groups = group_zip(zv)
            print(f"val archive: {len(zv.namelist())} members, {len(groups)} classes", flush=True)
            for cid in pool:
                counts["test"]["other_cn"] += dump_zip(
                    zv, groups.get(cid, [])[:CAP["test"]], out / "test" / "other_cn",
                    counts["test"]["other_cn"])
            heldout_items = []
            for cid in heldout:
                names = groups.get(cid, [])[:HELDOUT_CAP]
                base = counts["test"]["other_cn"] + len([h for h in heldout_items])
                for i, name in enumerate(names):
                    rel = f"data/splits/cn241_21_ood/{cid:03d}/{i:03d}.jpg"
                    if not args.dry_run:
                        dst = ROOT / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(zv.read(name))
                    heldout_items.append({"path": rel, "cnfood_id": cid})
    print(f"other_cn counts: {counts}", flush=True)

    classes = [dict(c, slug=c["slug"]) for c in table["classes"]]
    classes.append({"slug": "other_cn", "zh": "其它中餐", "en": "Other Chinese dish",
                    "cnfood_id": None, "other": True,
                    "n_classes_covered": len(pool)})
    meta = {"source": "CNFOOD-241 train (20 dishes + other) + val (test)",
            "n_classes": 21, "other_class": "other_cn", "other_pool": pool,
            "heldout_ids": heldout, "caps": CAP, "exclude_ids": sorted(EXCLUDE_IDS),
            "per_class": counts, "classes": classes}
    if args.dry_run:
        print(json.dumps({"n_classes": 21, "other_pool": len(pool),
                          "heldout": len(heldout)}, ensure_ascii=False))
        return 0
    dest = ROOT / "results/tables/cn241_21_classes.json"
    dest.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (ROOT / "results/tables/cn241_other_heldout.json").write_text(
        json.dumps({"heldout_ids": heldout, "items": heldout_items}, indent=1), encoding="utf-8")
    print(f"wrote {dest} and cn241_other_heldout.json ({len(heldout_items)} items)")
    for split in ("train", "val", "test"):
        n = sum(1 for _ in (out / split).rglob("*.jpg"))
        print(f"  {split}: {n} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
