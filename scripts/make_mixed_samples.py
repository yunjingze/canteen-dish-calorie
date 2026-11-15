"""Bake a handful of mixed-plate demo photos into data/splits/mixed/.

The web demo needs *realistic* composites with known ground truth: two or three
Food-101 test photos blended on one canvas, with a labels.json next to them so
the interface can show "truth vs. prediction".  Images come from the test split,
which the multi-label model never trains on.

    python scripts/make_mixed_samples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES, ZH_LABELS      # noqa: E402
from foodcls.mixed import MixedPlateDataset, build_composite  # noqa: E402

OUT = ROOT / "data/splits/mixed"
TEST = ROOT / "data/splits/test"
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


def by_class() -> dict[str, list[Path]]:
    table: dict[str, list[Path]] = {}
    for name in CLASS_NAMES:
        files = sorted((TEST / name).glob("*.jpg"))
        if files:
            table[name] = files
    return table


def main() -> None:
    table = by_class()
    missing = {c for _, classes in PLATES for c in classes if c not in table}
    if missing:
        raise SystemExit(f"missing test images for: {sorted(missing)}")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2024)
    labels = []
    for i, (slug, classes) in enumerate(PLATES):
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
                "zh": [ZH_LABELS[c] for c in classes],
                "sources": [p.name for p in picks],
            }
        )
        print(f"{name}: {' + '.join(ZH_LABELS[c] for c in classes)}", flush=True)
    (OUT / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(labels)} plates to {OUT}", flush=True)


if __name__ == "__main__":
    main()
