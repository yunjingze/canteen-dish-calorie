"""Merge the Kaggle "16 Famous Chinese Dishes" set into the wide-model splits.

The download (scripts/fetch_kaggle.py) is a detection-style export: one image per
dish, a YOLO label file per image, and train/val/test lists that still carry the
author's Windows paths.  We only need the classification part, so this script

  * reads the three lists and keeps the file name of every entry,
  * maps the 16 folder prefixes to Food-101-style snake_case names, folding
    Peking_Duck into the existing peking_duck class instead of duplicating it,
  * copies each image to data/wide/splits/<split>/<class>/<file>.

Images are copied rather than extracted wholesale so the wide split stays the one
directory the training/eval code already reads.
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# source prefix -> (wide class name, Chinese label, kcal/100 g, portion g)
MAPPING = {
    "Peking_Duck": ("peking_duck", "北京烤鸭", 340, 200),
    "Sweet_and_Sour_Pork": ("sweet_and_sour_pork", "咕噜肉", 250, 200),
    "Mapo_Tofu": ("mapo_tofu", "麻婆豆腐", 120, 250),
    "Yuxiang_Shredded_Pork": ("yuxiang_shredded_pork", "鱼香肉丝", 190, 200),
    "Husband_and_Wife_Lung_Slices": ("husband_and_wife_lung_slices", "夫妻肺片", 220, 150),
    "Twice-Cooked_Pork": ("twice_cooked_pork", "回锅肉", 290, 200),
    "Kung_Pao_Chicken": ("kung_pao_chicken", "宫保鸡丁", 200, 200),
    "Saliva_Chicken": ("saliva_chicken", "口水鸡", 210, 200),
    "Soup_Dumplings": ("soup_dumplings", "灌汤小笼包", 220, 200),
    "Braised_Pork_Meatballs_in_Brown_Sauce": ("braised_pork_meatballs", "红烧狮子头", 250, 200),
    "Dongpo_Pork": ("dongpo_pork", "东坡肉", 400, 150),
    "West_Lake_Vinegar_Fish": ("west_lake_vinegar_fish", "西湖醋鱼", 130, 250),
    "Buddha_Jumps_Over_the_Wall": ("buddha_jumps_over_wall", "佛跳墙", 150, 300),
    "Steamed_Sea_Bass": ("steamed_sea_bass", "清蒸鲈鱼", 120, 250),
    "Fish_with_Pickled_Cabbage_and_Chili": ("fish_with_pickled_cabbage", "酸菜鱼", 130, 300),
    "Boiled_Fish_with_Sichuan_Peppercorns": ("boiled_fish_sichuan", "水煮鱼", 160, 300),
}


def parse_list(raw: str) -> list[str]:
    """The exported lists hold Windows paths; we only want the trailing name."""
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(line.replace("\\", "/").rsplit("/", 1)[-1])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(ROOT / "data/external/16chinese.zip"))
    ap.add_argument("--data-root", default=str(ROOT / "data/wide"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    zpath = Path(args.zip)
    root = Path(args.data_root).resolve()
    counts: Counter = Counter()
    per_class: dict[str, Counter] = {}
    missing = []

    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
        for split, member in (("train", "train.txt"), ("val", "val.txt"), ("test", "test.txt")):
            for fname in parse_list(zf.read(member).decode("utf-8")):
                prefix = fname.split("_new_")[0]
                if prefix not in MAPPING:
                    missing.append((split, fname))
                    continue
                cls, _zh, _kcal, _g = MAPPING[prefix]
                src = f"images/{fname}"
                if src not in names:
                    missing.append((split, fname))
                    continue
                dst = root / "splits" / split / cls / fname
                if not args.dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists():
                        with zf.open(src) as fh, open(dst, "wb") as out:
                            shutil.copyfileobj(fh, out)
                counts[(split, cls)] += 1
                per_class.setdefault(cls, Counter())[split] += 1

    print(f"{'dry-run' if args.dry_run else 'copied'}: {sum(counts.values())} images, "
          f"{len(per_class)} classes", flush=True)
    for cls in sorted(per_class):
        c = per_class[cls]
        print(f"  {cls:36s} train {c['train']:4d}  val {c['val']:3d}  test {c['test']:3d}", flush=True)
    if missing:
        print(f"skipped {len(missing)} entries, e.g. {missing[:3]}", flush=True)

    out = ROOT / "results/tables/chinese_dishes.json"
    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = {
            cls: {"zh": MAPPING[next(k for k, v in MAPPING.items() if v[0] == cls)][1],
                  "kcal_per_100g": MAPPING[next(k for k, v in MAPPING.items() if v[0] == cls)][2],
                  "portion_g": MAPPING[next(k for k, v in MAPPING.items() if v[0] == cls)][3],
                  "train": per_class[cls]["train"], "val": per_class[cls]["val"],
                  "test": per_class[cls]["test"]}
            for cls in sorted(per_class)
        }
        out.write_text(json.dumps({"source": "kaggle:patzer0/16-famous-chinese-dishes",
                                   "classes": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
