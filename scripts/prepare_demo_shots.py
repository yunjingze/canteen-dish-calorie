"""Copy the hand-picked demo photos into one numbered folder.

The demo video walks through a fixed list of photos.  Keeping the list here
(rather than in a slide deck) means the shots can be rebuilt on any machine
that has the splits, and that every photo is traceable to a split that the
model never trained on:

  01-04  data/external/web20/accepted  -- 623 screened web photos, no training
  05     data/splits/test/pizza        -- Food-101 test split
  06     data/splits/negatives        -- hand-shot non-food photos
  07     data/splits/cn241_21_ood      -- Chinese dishes outside the 21-way head
  08-09  wide-model samples            -- Food-101 test + ChineseFoodNet photos
  10-11  data/splits/mixed             -- composite trays

Usage: python scripts/prepare_demo_shots.py [--out data/external/demo_shots]
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (output name, source path relative to the project root, what the shot shows)
SHOTS = [
    ("01_hot_sour_potato.jpg",  "data/external/web20/accepted/hot_sour_potato/0008.jpg",
     "canteen tab, clean hit: 酸辣土豆丝 95.1%"),
    ("02_braised_pork.jpg",     "data/external/web20/accepted/braised_pork/0010.jpg",
     "canteen tab: 红烧肉 99.3%, the widest calorie band"),
    ("03_di_san_xian.jpg",      "data/external/web20/accepted/di_san_xian/0003.jpg",
     "canteen tab: 地三鲜 98.1%"),
    ("04_baozi.jpg",            "data/external/web20/accepted/baozi/0015.jpg",
     "canteen tab: 包子 94.0%, domain probe only 0.86 so no threshold relax"),
    ("05_pizza.jpg",            "data/splits/test/pizza/test_0-of-00003_1001116.jpg",
     "canteen tab refuses a Western dish, offers 换到「单菜识别」"),
    ("06_not_food.jpg",         "data/splits/negatives/005.jpg",
     "single-dish tab: 这张照片里没找到菜"),
    ("07_off_list_cn.jpg",      "data/splits/cn241_21_ood/013/000.jpg",
     "canteen tab: 清单外的中餐 (鱼香肉丝), refused by the other_cn bucket"),
    ("08_mapo_tofu.jpg",        "data/external/16chinese/images/Mapo_Tofu_new_020.jpg",
     "wide tab: 麻婆豆腐 79%"),
    ("09_peking_duck.jpg",      "data/external/16chinese/images/Peking_Duck_new_001.jpg",
     "wide tab: 北京烤鸭 63%"),
    ("10_mixed_burger_fries.jpg", "data/splits/mixed/mix_02_burger_fries.jpg",
     "tray tab: 汉堡 + 薯条 + 牛排, plate total"),
    ("11_mixed_cn_tray.jpg",    "data/splits/mixed/mix_13_cn_spicy_chicken_beans.jpg",
     "tray tab: canteen tray, 辣子鸡 + 干煸豆角"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/external/demo_shots")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    missing = []
    for name, src, why in SHOTS:
        s = ROOT / src
        if not s.exists():
            missing.append(src)
            continue
        shutil.copy2(s, out / name)
        print(f"{name:28s} <- {src}")
    print(f"\n{len(SHOTS) - len(missing)}/{len(SHOTS)} copied to {out}")
    for m in missing:
        print("  MISSING", m)


if __name__ == "__main__":
    main()
