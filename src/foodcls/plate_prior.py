"""A co-occurrence prior for the synthetic tray plates.

A tray is not a uniform random draw from 40 dishes: a canteen tray is a staple
plus one or two dishes, a burger comes with fries, and nobody eats a chocolate
cake next to a bowl of ramen.  Training on uniform random combinations teaches
the model to score dishes independently, which is exactly what makes it report
``steak`` on a tray that holds a burger and fries: the parts never constrain
each other.  This module supplies the missing constraint as a sampling prior,
used both when building training plates and when building the realistic
evaluation plates.

The prior is deliberately soft: every pair keeps a floor weight, so the model
still sees unusual combinations and does not collapse to a handful of templates.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .plate_classes import PLATE_CLASS_NAMES

#: what a dish is *for* on a tray; the prior is about roles, not identities
ROLES: Dict[str, str] = {}

_STAPLE_WESTERN = ("fried_rice", "pizza", "hamburger", "hot_dog",
                   "spaghetti_bolognese", "lasagna", "pancakes", "waffles")
_MAIN_WESTERN = ("steak", "grilled_salmon", "chicken_wings", "sushi",
                 "dumplings", "ramen")
_SIDE_WESTERN = ("french_fries", "caesar_salad", "greek_salad")
_DESSERT_WESTERN = ("donuts", "ice_cream", "chocolate_cake")

_STAPLE_CN = ("fried_noodles", "baozi")
_MAIN_CN = ("braised_pork", "boiled_pork", "sweet_sour_pork", "farmhouse_pork",
            "pepper_pork", "mu_xi_pork", "spicy_chicken", "braised_chicken",
            "boiled_chicken", "cola_wings", "sweet_sour_ribs")
_SIDE_CN = ("hot_sour_potato", "di_san_xian", "hand_torn_cabbage",
            "stir_fried_greens", "dry_fried_beans", "tomato_egg", "steamed_egg")

for _names, _role in ((_STAPLE_WESTERN, "staple"), (_MAIN_WESTERN, "main"),
                      (_SIDE_WESTERN, "side"), (_DESSERT_WESTERN, "dessert"),
                      (_STAPLE_CN, "staple"), (_MAIN_CN, "main"),
                      (_SIDE_CN, "side")):
    for _n in _names:
        ROLES[_n] = _role

#: cuisine of every plate class, from the western/canteen split of the tables
CUISINE: Dict[str, str] = {}
for _n in _STAPLE_WESTERN + _MAIN_WESTERN + _SIDE_WESTERN + _DESSERT_WESTERN:
    CUISINE[_n] = "western"
for _n in _STAPLE_CN + _MAIN_CN + _SIDE_CN:
    CUISINE[_n] = "canteen"

#: how often two roles share a plate, before the cross-cuisine penalty
ROLE_PAIR: Dict[tuple, float] = {}
for _a, _b, _w in (
    ("staple", "main", 1.00),
    ("main", "side", 0.90),
    ("staple", "side", 0.75),
    ("main", "main", 0.55),
    ("side", "side", 0.50),
    ("staple", "staple", 0.35),
    ("dessert", "dessert", 0.70),
    ("dessert", "side", 0.20),
    ("dessert", "staple", 0.05),
    ("dessert", "main", 0.05),
):
    ROLE_PAIR[(_a, _b)] = _w
    ROLE_PAIR[(_b, _a)] = _w

#: a western dish next to a canteen dish is a real tray (a burger with a stir
#: fry) but rarer than a coherent menu; the penalty keeps it in the data
CROSS_CUISINE = 0.25
#: nothing is impossible: the floor keeps unusual pairs in the training stream
FLOOR = 0.02


def role_weight(a: str, b: str) -> float:
    """Raw compatibility in [0, 1] for two plate classes."""
    ra, rb = ROLES.get(a, "main"), ROLES.get(b, "main")
    w = ROLE_PAIR.get((ra, rb), 0.5)
    if CUISINE.get(a) != CUISINE.get(b):
        w *= CROSS_CUISINE
    return float(w)


def weights_for(class_names: Sequence[str], anchor: str,
                exclude: Sequence[int] = ()) -> np.ndarray:
    """Sampling weights over ``class_names`` for the dishes that join ``anchor``.

    Excluded indices get zero; the rest get ``role_weight`` plus the floor.  The
    vector is *not* normalised so callers can inspect the raw scores.
    """
    w = np.array([role_weight(anchor, c) + FLOOR for c in class_names], dtype=float)
    for i in exclude:
        if 0 <= int(i) < w.size:
            w[int(i)] = 0.0
    return w


def weights_for_many(class_names: Sequence[str], anchors: Sequence[str],
                     exclude: Sequence[int] = ()) -> np.ndarray:
    """Weights for a dish that joins *all* ``anchors`` at once.

    A three-part tray is not three independent choices: once a burger and a
    portion of fries are on the plate, the third part should be another side,
    not a steak.  Averaging the pairwise weights keeps that pressure without
    ever making a combination impossible.
    """
    if not anchors:
        w = np.full(len(class_names), 1.0)
    else:
        w = np.zeros(len(class_names), dtype=float)
        for a in anchors:
            w += np.array([role_weight(a, c) for c in class_names], dtype=float)
        w /= float(len(anchors))
    w = w + FLOOR
    for i in exclude:
        if 0 <= int(i) < w.size:
            w[int(i)] = 0.0
    return w


def pick_addition(rng: np.random.Generator, class_names: Sequence[str],
                  anchor: str | Sequence[str], exclude: Sequence[int] = ()) -> int:
    """Draw one class to put next to ``anchor`` (or next to a set of them)."""
    anchors = [anchor] if isinstance(anchor, str) else list(anchor)
    w = weights_for_many(class_names, anchors, exclude)
    total = float(w.sum())
    if total <= 0.0:                                    # every class excluded
        return int(rng.integers(len(class_names)))
    return int(rng.choice(len(class_names), p=w / total))


def default_names(num_classes: int | None = None) -> List[str]:
    """The plate class list, truncated for the old 20-class western head."""
    names = list(PLATE_CLASS_NAMES)
    if num_classes is None or int(num_classes) >= len(names):
        return names
    return names[: int(num_classes)]

__all__ = [
    "ROLES",
    "CUISINE",
    "ROLE_PAIR",
    "CROSS_CUISINE",
    "FLOOR",
    "role_weight",
    "weights_for",
    "weights_for_many",
    "pick_addition",
    "default_names",
]