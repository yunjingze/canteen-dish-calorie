"""Canteen-Chinese class definitions (CNFOOD-241 subset).

These 20 dishes are the everyday canteen menu: stir-fried vegetables, egg and
pork dishes, two staples.  Calorie priors follow the same convention as
foodcls.classes -- typical kcal per 100 g of the dish as served, coarse values
from public food-composition tables (China Food Composition Tables 6th ed.),
used only to print a rough range.

Slugs match the folder names produced by scripts/prepare_cn241.py, so the
checkpoint's class_names are these slugs in this order.
"""

# (slug, zh_label, kcal_per_100g, typical_portion_g)
CLASS_SPEC_CN = [
    ("hot_sour_potato",   "酸辣土豆丝",   120, 150),
    ("di_san_xian",       "地三鲜",       160, 200),
    ("hand_torn_cabbage", "手撕包菜",      70, 150),
    ("stir_fried_greens", "炒青菜",        65, 150),
    ("dry_fried_beans",   "干煸豆角",     150, 150),
    ("tomato_egg",        "西红柿炒鸡蛋", 100, 200),
    ("steamed_egg",       "鸡蛋羹",        70, 180),
    ("sweet_sour_ribs",   "糖醋排骨",     280, 150),
    ("cola_wings",        "可乐鸡翅",     240, 150),
    ("boiled_chicken",    "白斩鸡",       190, 150),
    ("braised_chicken",   "黄焖鸡",       180, 250),
    ("spicy_chicken",     "辣子鸡",       250, 150),
    ("braised_pork",      "红烧肉",       380, 120),
    ("boiled_pork",       "水煮肉片",     200, 250),
    ("sweet_sour_pork",   "糖醋里脊",     290, 150),
    ("farmhouse_pork",    "农家小炒肉",   230, 150),
    ("pepper_pork",       "青椒肉丝",     160, 150),
    ("mu_xi_pork",        "木须肉",       150, 150),
    ("baozi",             "包子",         230, 150),
    ("fried_noodles",     "炒面",         180, 300),
]

CN_CLASS_NAMES = [c[0] for c in CLASS_SPEC_CN]
CN_ZH = {c[0]: c[1] for c in CLASS_SPEC_CN}
CN_KCAL_PER_100G = {c[0]: c[2] for c in CLASS_SPEC_CN}
CN_PORTION_G = {c[0]: c[3] for c in CLASS_SPEC_CN}
CN_NUM_CLASSES = len(CN_CLASS_NAMES)

# The 21st dish is rice: it is the one item on every canteen tray, it was folded
# into the 'other Chinese' bucket until the 22-class head gave it a slot of its
# own, and it is the only canteen class whose slug is not in CN_CLASS_NAMES.
# It is kept out of CLASS_SPEC_CN on purpose: plate_classes.py derives the 40-
# class tray table from CN_CLASS_NAMES, and a 41st slug would invalidate the
# plate40 checkpoint.  The wider table below is what the optional 22-class
# canteen head reports with (see load_cn_model() in app.py).
RICE_SPEC = ("rice", "米饭", 116, 200)

CN_MENU_NAMES = CN_CLASS_NAMES + [RICE_SPEC[0]]
CN_MENU_ZH = {**CN_ZH, RICE_SPEC[0]: RICE_SPEC[1]}
CN_MENU_KCAL_PER_100G = {**CN_KCAL_PER_100G, RICE_SPEC[0]: RICE_SPEC[2]}
CN_MENU_PORTION_G = {**CN_PORTION_G, RICE_SPEC[0]: RICE_SPEC[3]}
