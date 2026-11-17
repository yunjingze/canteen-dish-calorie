"""Bake a handful of mixed-plate demo photos into data/splits/mixed/.

The web demo needs *realistic* composites with known ground truth: two or three
test photos blended on one canvas, with a labels.json next to them so the
interface can show "truth vs. prediction".  Images come from the test split,
which the multi-label model never trains on.  Ten of the plates are western
dishes; seven more are canteen trays (plus one rice + Chinese dish mix), so the
gallery covers both halves of the 40-class head.

    python scripts/make_mixed_samples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES                     # noqa: E402
from foodcls.mixed import MixedPlateDataset, build_composite  # noqa: E402
from foodcls.plate_classes import (                         # noqa: E402
    CANTEEN_CLASS_NAMES,
    PLATE_ZH,
)

OUT = ROOT / "data/splits/mixed"
WEST_TEST = ROOT / "data/splits/test"
CN_TEST = ROOT / "data/splits/cn241_21/test"
SIZE = 384

#: hand-picked plates that read like a canteen tray (the first one is the brief)
PLATES = [
    ("fries_wings", ["french_fries", "chicken_wings"]),
    ("burger_fries", ["hamburger", "french_fries"]),
    ("rice_wings", ["fried_rice", "chicken_wings"]),
    ("pizza_salad", ["pizza", "caesar_salad"]),
    ("ramen_dumplings", ["ramen", "dumplings"]),
    ("pasta_salmon", ["spaghetti_bolognese", "grilled_salmon"]),
    ("pancakes_icecream", ["pancakes", "ice_cream"]),
    ("sushi_salad", ["sushi", "greek_salad"]),
    ("cake_donut_icecream", ["chocolate_cake", "donuts", "ice_cream"]),
    ("rice_steak_salad", ["fried_rice", "steak", "caesar_salad"]),
]

#: canteen trays: two or three dishes from the 20-dish menu, plus one west+cn mix
CN_PLATES = [
    ("cn_tomato_egg_greens", ["tomato_egg", "stir_fried_greens"]),
    ("cn_braised_pork_potato", ["braised_pork", "hot_sour_potato"]),
    ("cn_spicy_chicken_beans", ["spicy_chicken", "dry_fried_beans"]),
    ("cn_sweet_sour_pork_cabbage", ["sweet_sour_pork", "hand_torn_cabbage"]),
    ("cn_pepper_pork_egg", ["pepper_pork", "steamed_egg"]),
    ("cn_cola_wings_noodles", ["cola_wings", "fried_noodles"]),
    ("cn_rice_tomato_egg", ["fried_rice", "tomato_egg"]),
]


def by_class() -> dict[str, list[Path]]:
    """Test images per dish, for both halves of the 40-class head."""
    table: dict[str, list[Path]] = {}
    for name in CLASS_NAMES:
        files = sorted((WEST_TEST / name).glob("*.jpg"))
        if files:
            table[name] = files
    for name in CANTEEN_CLASS_NAMES:
        files = sorted((CN_TEST / name).glob("*.jpg"))
        if files:
            table[name] = files
    return table


def cuisine_of(classes) -> str:
    west = [c for c in classes if c in CLASS_NAMES]
    cn = [c for c in classes if c in CANTEEN_CLASS_NAMES]
    if west and cn:
        return "mixed"
    return "canteen" if cn else "western"


def main() -> None:
    table = by_class()
    plates = list(PLATES) + list(CN_PLATES)
    missing = {c for _, classes in plates for c in classes if c not in table}
    if missing:
        raise SystemExit(f"missing test images for: {sorted(missing)}")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2024)
    labels = []
    for i, (slug, classes) in enumerate(plates):
        # deterministic but not always the first file of the class
        picks = [table[c][(i * 3 + k) % len(table[c])] for k, c in enumerate(classes)]
        parts = [MixedPlateDataset._load(str(p)) for p in picks]
        plate = build_composite(parts, rng, size=SIZE)
        name = f"mix_{i + 1:02d}_{slug}.jpg"
        plate.save(OUT / name, quality=92)
        labels.append(
            {
                "name": name,
                "classes": classes,
                "zh": [PLATE_ZH[c] for c in classes],
                "cuisine": cuisine_of(classes),
                "sources": [p.name for p in picks],
            }
        )
        print(f"{name}: {' + '.join(PLATE_ZH[c] for c in classes)}", flush=True)
    (OUT / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(labels)} plates to {OUT}", flush=True)


if __name__ == "__main__":
    main()