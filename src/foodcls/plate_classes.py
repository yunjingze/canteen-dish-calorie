"""Class table for the multi-label tray model.

The tray head answers "which dishes are on this plate".  A campus tray can hold
either the Food-101 dishes the single-dish demo knows or the 20 canteen dishes
from the Chinese head, so the tray model is trained on composites built from
both pools and its class order is: the 20 Food-101 slugs first, then the 20
canteen slugs.  That keeps the old 20-class checkpoints usable (their class
list is just the first half) and keeps slug -> Chinese label / calorie lookups
in one place.
"""
from .classes import CLASS_NAMES, KCAL_PER_100G, PORTION_G, ZH_LABELS
from .classes_cn import CN_CLASS_NAMES, CN_KCAL_PER_100G, CN_PORTION_G, CN_ZH

WESTERN_CLASS_NAMES = list(CLASS_NAMES)
CANTEEN_CLASS_NAMES = list(CN_CLASS_NAMES)
PLATE_CLASS_NAMES = WESTERN_CLASS_NAMES + CANTEEN_CLASS_NAMES
PLATE_NUM_CLASSES = len(PLATE_CLASS_NAMES)

PLATE_ZH = {**ZH_LABELS, **CN_ZH}
PLATE_KCAL_PER_100G = {**KCAL_PER_100G, **CN_KCAL_PER_100G}
PLATE_PORTION_G = {**PORTION_G, **CN_PORTION_G}

#: which single-image head knows this slug; used by reports and by the app when
#: it has to explain why a tray part is served by one head or the other
PLATE_CUISINE = {**{c: "western" for c in WESTERN_CLASS_NAMES},
                 **{c: "canteen" for c in CANTEEN_CLASS_NAMES}}

# The two pools must not share a slug: a collision would silently map two
# different dishes onto one output unit.
_overlap = set(WESTERN_CLASS_NAMES) & set(CANTEEN_CLASS_NAMES)
if _overlap:                                        # pragma: no cover
    raise RuntimeError(f"duplicate dish slugs across heads: {sorted(_overlap)}")

__all__ = [
    "WESTERN_CLASS_NAMES",
    "CANTEEN_CLASS_NAMES",
    "PLATE_CLASS_NAMES",
    "PLATE_NUM_CLASSES",
    "PLATE_ZH",
    "PLATE_KCAL_PER_100G",
    "PLATE_PORTION_G",
    "PLATE_CUISINE",
]
