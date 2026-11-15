"""Translate class probabilities into a coarse calorie interval.

The display rule is deliberately conservative: start from a typical portion
size, apply a +/-15 % band for recipe/portion variability, and widen the band
when the classifier is not confident.  The output is a *range*, never a single
clinical number.  A portion multiplier lets the user say whether the serving on
the tray looks small, normal or large, and an explicit weight in grams overrides
that estimate when the user knows how much is on the plate.
"""
from __future__ import annotations

from typing import Dict, List

from .classes import CLASS_NAMES, KCAL_PER_100G, PORTION_G, ZH_LABELS

# The 20-class canteen tables are the default; the wide (116-class) model passes
# its own tables so the same display rule applies to every dish.

BASE_REL_SPREAD = 0.15   # recipe / portion variability
LOW_CONF_THRESHOLD = 0.55
LOW_CONF_EXTRA = 0.10    # extra relative spread for uncertain predictions

#: portion multipliers exposed by the web interface
PORTION_SCALES = {"small": 0.7, "normal": 1.0, "large": 1.35}
PORTION_LABELS = {"small": "小份", "normal": "常规", "large": "大份"}

#: candidates below this probability are not shown at all
MIN_TOP_CONF = 0.10

#: accepted range for a user-supplied serving weight (grams)
MIN_USER_GRAMS = 10.0
MAX_USER_GRAMS = 2000.0


def portion_scale(name: str | float | None) -> float:
    """Accept a named portion ("small"/"normal"/"large") or a numeric factor."""
    if name is None:
        return 1.0
    if isinstance(name, (int, float)):
        return max(0.3, min(3.0, float(name)))
    return PORTION_SCALES.get(str(name).lower(), 1.0)


def user_grams(value) -> float | None:
    """Normalise a user-supplied weight; None means "use the portion prior"."""
    if value is None or value == "":
        return None
    try:
        grams = float(value)
    except (TypeError, ValueError):
        return None
    if grams <= 0:
        return None
    return max(MIN_USER_GRAMS, min(MAX_USER_GRAMS, grams))


def class_calorie_range(
    class_name: str,
    confidence: float,
    scale: float = 1.0,
    grams: float | None = None,
    kcal_table: Dict[str, float] | None = None,
    portion_table: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Return the estimated kcal interval for one predicted class.

    An explicit weight (grams) is the actual serving mass typed by the user; it
    takes precedence over the class prior scaled by the portion multiplier.
    """
    if isinstance(scale, str):
        scale = portion_scale(scale)
    kcal_table = KCAL_PER_100G if kcal_table is None else kcal_table
    portion_table = PORTION_G if portion_table is None else portion_table
    kcal100 = kcal_table[class_name]
    explicit = user_grams(grams)
    if explicit is None:
        grams_used = portion_table[class_name] * scale
        weight_source = "portion"
    else:
        grams_used = explicit
        weight_source = "user"
    point = kcal100 * grams_used / 100.0
    spread = BASE_REL_SPREAD + (LOW_CONF_EXTRA if confidence < LOW_CONF_THRESHOLD else 0.0)
    low = _round10(point * (1.0 - spread))
    high = _round10(point * (1.0 + spread))
    return {
        "point": round(point),
        "low": low,
        "high": high,
        "kcal_per_100g": kcal100,
        "portion_g": int(round(grams_used)),
        "base_portion_g": portion_table[class_name],
        "spread": spread,
        "weight_source": weight_source,
    }


def _round10(x: float) -> int:
    return int(round(x / 10.0) * 10)


def topk_with_calories(
    probs,
    k: int = 3,
    portion: str | float | None = None,
    grams: float | None = None,
    class_names=None,
    zh_table: Dict[str, str] | None = None,
    kcal_table: Dict[str, float] | None = None,
    portion_table: Dict[str, float] | None = None,
) -> List[Dict]:
    """Build the Top-k payload used by the web demo and the report figures."""
    import numpy as np

    class_names = CLASS_NAMES if class_names is None else class_names
    zh_table = ZH_LABELS if zh_table is None else zh_table
    scale = portion_scale(portion)
    probs = np.asarray(probs, dtype=float).ravel()
    order = np.argsort(-probs)[:k]
    out = []
    for idx in order:
        name = class_names[int(idx)]
        conf = float(probs[int(idx)])
        if conf < MIN_TOP_CONF:      # a 3 % "guess" is noise, not a candidate
            continue
        rng = class_calorie_range(name, conf, scale, grams=grams,
                                  kcal_table=kcal_table, portion_table=portion_table)
        out.append(
            {
                "class": name,
                "zh": zh_table[name],
                "prob": conf,
                "kcal_low": rng["low"],
                "kcal_high": rng["high"],
                "kcal_point": rng["point"],
                "kcal_per_100g": rng["kcal_per_100g"],
                "portion_g": rng["portion_g"],
                "base_portion_g": rng["base_portion_g"],
                "spread": rng["spread"],
                "weight_source": rng["weight_source"],
            }
        )
    return out


def format_range(item: Dict) -> str:
    return f"{item['kcal_low']}-{item['kcal_high']} kcal"
