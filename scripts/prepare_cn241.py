# -*- coding: utf-8 -*-
"""Build the canteen-Chinese subset of CNFOOD-241 (20 everyday dishes).

Two modes:

    # prototype: only the small validation archive, 70/10/20
    python scripts/prepare_cn241.py --zip data/external/cnfood241_val600x600.zip

    # full: train archive for train/val (90/10), validation archive as the test set
    python scripts/prepare_cn241.py --zip data/external/cnfood241_val600x600.zip \
        --train-zip data/external/cnfood241_train600x600.zip --out data/splits/cn241_full

Layout is the SplitDataset format the rest of the project uses:

    <out>/{train,val,test}/<slug>/<n>.jpg

CNFOOD-241 stores one numeric folder per class ("000".."240"); the names live in
class_name.xls.  Only the picked folders are read out of the archives.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (CNFOOD-241 row index, slug, zh label, English label)
PICKS = [
    (5,   "hot_sour_potato",   "酸辣土豆丝",   "Potato silk"),
    (9,   "di_san_xian",       "地三鲜",       "Fried potato, green pepper and eggplant"),
    (17,  "hand_torn_cabbage", "手撕包菜",     "Shredded cabbage"),
    (19,  "stir_fried_greens", "炒青菜",       "Sauteed greens"),
    (37,  "dry_fried_beans",   "干煸豆角",     "Dry-fried green beans"),
    (50,  "tomato_egg",        "西红柿炒鸡蛋", "Scrambled egg with tomato"),
    (53,  "steamed_egg",       "鸡蛋羹",       "Steamed egg custard"),
    (58,  "sweet_sour_ribs",   "糖醋排骨",     "Sweet and sour spareribs"),
    (60,  "cola_wings",        "可乐鸡翅",     "Cola chicken wings"),
    (65,  "boiled_chicken",    "白斩鸡",       "Poached chicken"),
    (68,  "braised_chicken",   "黄焖鸡",       "Chicken braised in brown sauce"),
    (70,  "spicy_chicken",     "辣子鸡",       "Spicy chicken"),
    (77,  "braised_pork",      "红烧肉",       "Braised pork belly"),
    (86,  "boiled_pork",       "水煮肉片",     "Pork slices in chilli oil"),
    (87,  "sweet_sour_pork",   "糖醋里脊",     "Sweet and sour tenderloin"),
    (90,  "farmhouse_pork",    "农家小炒肉",   "Farmhouse stir-fried pork"),
    (96,  "pepper_pork",       "青椒肉丝",     "Shredded pork with green pepper"),
    (99,  "mu_xi_pork",        "木须肉",       "Sauteed pork, egg and wood ear"),
    (145, "baozi",             "包子",         "Steamed stuffed bun"),
    (158, "fried_noodles",     "炒面",         "Fried noodles"),
]

IMAGE_EXT = (".jpg", ".jpeg", ".png")


def collect(zf: zipfile.ZipFile, wanted: dict) -> dict:
    """Map class id -> member names, ignoring folder entries and other classes."""
    groups: dict[str, list[str]] = {}
    for n in zf.namelist():
        if n.endswith("/") or not n.lower().endswith(IMAGE_EXT):
            continue
        parts = [p for p in n.split("/") if p]
        if len(parts) < 2:
            continue
        # folders are zero padded ("005"), the class table is not ("5")
        try:
            cls = str(int(parts[-2]))
        except ValueError:
            continue
        if cls in wanted:
            groups.setdefault(cls, []).append(n)
    return groups


def dump(zf: zipfile.ZipFile, members: list[str], dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(members):
        (dst / f"{i:04d}.jpg").write_bytes(zf.read(name))
    return len(members)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(ROOT / "data/external/cnfood241_val600x600.zip"))
    ap.add_argument("--train-zip", default="", help="train600x600.zip; when given, train/val come from it")
    ap.add_argument("--out", default=str(ROOT / "data/splits/cn241"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="0.7,0.1,0.2", help="val-archive-only mode: train,val,test")
    ap.add_argument("--train-split", default="0.9,0.1", help="full mode: train,val of the train archive")
    args = ap.parse_args(argv)

    out = Path(args.out)
    wanted = {str(p[0]): p for p in PICKS}
    rng = random.Random(args.seed)
    per_class = {p[1]: {"train": 0, "val": 0, "test": 0} for p in PICKS}
    counts = {}

    if args.train_split and args.train_zip:
        tr_frac, va_frac = [float(x) for x in args.train_split.split(",")]
        assert abs(tr_frac + va_frac - 1.0) < 1e-6, "train split must add up to 1"
        with zipfile.ZipFile(args.train_zip) as zt:
            groups = collect(zt, wanted)
            missing = [k for k in wanted if k not in groups]
            if missing:
                print("WARNING: no train images for rows", missing)
            for cls, members in sorted(groups.items()):
                _, slug, zh, en = wanted[cls]
                members.sort()
                rng.shuffle(members)
                n = len(members)
                n_tr = int(round(n * tr_frac))
                counts[slug] = {"zh": zh, "en": en, "id": int(cls), "n": n}
                per_class[slug]["train"] = dump(zt, members[:n_tr], out / "train" / slug)
                per_class[slug]["val"] = dump(zt, members[n_tr:], out / "val" / slug)
        # everything in the validation archive becomes the held-out test set
        with zipfile.ZipFile(args.zip) as zv:
            groups = collect(zv, wanted)
            for cls, members in sorted(groups.items()):
                _, slug, _, _ = wanted[cls]
                members.sort()
                per_class[slug]["test"] = dump(zv, members, out / "test" / slug)
        splits = {"train": tr_frac, "val": va_frac, "test": "all of val600x600"}
        source = "CNFOOD-241 train600x600 (train/val) + val600x600 (test)"
    else:
        fracs = [float(x) for x in args.split.split(",")]
        assert abs(sum(fracs) - 1.0) < 1e-6, "split fractions must add up to 1"
        with zipfile.ZipFile(args.zip) as zf:
            groups = collect(zf, wanted)
            print("archive members:", len(zf.namelist()))
            missing = [k for k in wanted if k not in groups]
            if missing:
                print("WARNING: no images for rows", missing)
            for cls, members in sorted(groups.items()):
                _, slug, zh, en = wanted[cls]
                members.sort()
                rng.shuffle(members)
                n = len(members)
                n_tr = int(round(n * fracs[0]))
                n_va = int(round(n * fracs[1]))
                buckets = {"train": members[:n_tr], "val": members[n_tr:n_tr + n_va],
                           "test": members[n_tr + n_va:]}
                counts[slug] = {"zh": zh, "en": en, "id": int(cls), "n": n}
                for split, files in buckets.items():
                    per_class[slug][split] = dump(zf, files, out / split / slug)
        splits = {"train": fracs[0], "val": fracs[1], "test": fracs[2]}
        source = "CNFOOD-241 (Mendeley fspyss5zbb), val600x600"

    table = {
        "source": source,
        "splits": splits,
        "classes": [
            {"slug": p[1], "zh": p[2], "en": p[3], "cnfood_id": p[0], **counts.get(p[1], {})}
            for p in PICKS
        ],
        "per_class": per_class,
    }
    dest = ROOT / ("results/tables/cn241_classes_full.json" if args.train_zip else "results/tables/cn241_classes.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(v["train"] + v["val"] + v["test"] for v in per_class.values())
    print(f"extracted {total} images for {len(per_class)} classes -> {out}")
    for slug, v in per_class.items():
        print(f"  {slug:18s} train {v['train']:4d} val {v['val']:4d} test {v['test']:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
