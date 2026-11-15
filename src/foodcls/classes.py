"""Class definitions: Food-101 class names, Chinese labels and nutrition priors.

Calorie priors are typical values per 100 g of the *as-served* dish, taken from
public food-composition tables (China Food Composition Tables 6th ed.; USDA
FoodData Central).  They are coarse priors used only to display a rough range,
not a dietary or medical measurement.
"""

# (food101_name, zh_label, kcal_per_100g, typical_portion_g)
CLASS_SPEC = [
    ("fried_rice",         "蛋炒饭",   175, 350),
    ("dumplings",          "饺子",     220, 200),
    ("ramen",              "汤面",      85, 550),
    ("sushi",              "寿司",     145, 200),
    ("pizza",              "披萨",     266, 200),
    ("hamburger",          "汉堡",     295, 180),
    ("french_fries",       "薯条",     312, 120),
    ("hot_dog",            "热狗",     290, 110),
    ("caesar_salad",       "凯撒沙拉", 130, 200),
    ("greek_salad",        "希腊沙拉", 105, 200),
    ("grilled_salmon",     "烤三文鱼", 208, 150),
    ("steak",              "牛排",     250, 180),
    ("spaghetti_bolognese","肉酱意面", 160, 350),
    ("lasagna",            "千层面",   180, 300),
    ("chicken_wings",      "鸡翅",     240, 150),
    ("pancakes",           "煎饼",     230, 180),
    ("waffles",            "华夫饼",   290, 110),
    ("donuts",             "甜甜圈",   450,  75),
    ("ice_cream",          "冰淇淋",   207, 100),
    ("chocolate_cake",     "巧克力蛋糕",350, 100),
]

CLASS_NAMES = [c[0] for c in CLASS_SPEC]
ZH_LABELS = {c[0]: c[1] for c in CLASS_SPEC}
KCAL_PER_100G = {c[0]: c[2] for c in CLASS_SPEC}
PORTION_G = {c[0]: c[3] for c in CLASS_SPEC}
NUM_CLASSES = len(CLASS_NAMES)
