"""Convert the HuggingFace Food-101 parquet shards into an ImageFolder tree.

Only the 20 classes used by this project are materialised.  The official split
is preserved: rows of the *train* shards go to images/, rows of the *validation*
shards (which are the official Food-101 test images) go to images_test/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.classes import CLASS_NAMES  # noqa: E402

FALLBACK_NAMES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad",
    "beignets", "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad",
    "cannoli", "caprese_salad", "carrot_cake", "ceviche", "cheesecake", "cheese_plate",
    "chicken_curry", "chicken_quesadilla", "chicken_wings", "chocolate_cake", "chocolate_mousse",
    "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame",
    "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
    "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup",
    "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi",
    "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger",
    "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes",
    "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto", "samosas", "sashimi",
    "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara",
    "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu",
    "tuna_tartare", "waffles",
]


def label_names(path: Path) -> list[str]:
    md = pq.ParquetFile(path).schema_arrow.metadata or {}
    raw = md.get(b"huggingface")
    if raw:
        info = json.loads(raw.decode("utf-8"))
        feats = info.get("info", {}).get("features", {})
        names = feats.get("label", {}).get("names")
        if isinstance(names, dict):  # HF may serialise ClassLabel.names as {str: name}
            names = [names[str(i)] for i in range(len(names))]
        if names and len(names) == 101:
            return list(names)
    return FALLBACK_NAMES


def convert(shard: Path, dst_root: Path, split: str, only=None) -> int:
    names = label_names(shard)
    pf = pq.ParquetFile(shard)
    written = 0
    for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
        images = batch.column(0).to_pylist()
        labels = batch.column(1).to_pylist()
        for rec, lab in zip(images, labels):
            cls = names[int(lab)]
            if only is not None and cls not in only:
                continue
            src_path = rec.get("path") or ""
            fname = Path(src_path).name or f"{written}.jpg"
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fname += ".jpg"
            out = dst_root / cls / f"{split}_{shard.stem[-10:]}_{fname}"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                out.write_bytes(rec["bytes"])
            written += 1
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/raw_hf"))
    ap.add_argument("--dst", default=str(ROOT / "data/raw/food-101"))
    ap.add_argument("--all", action="store_true",
                    help="materialise all 101 Food-101 classes (wide model) instead of the 20 canteen ones")
    args = ap.parse_args()
    only = None if args.all else CLASS_NAMES
    src, dst = Path(args.src), Path(args.dst)
    total = 0
    for shard in sorted(src.glob("*.parquet")):
        split = "test" if shard.name.startswith("validation") else "train"
        sub = dst / ("images_test" if split == "test" else "images")
        n = convert(shard, sub, split, only=only)
        total += n
        scope = "all 101 classes" if only is None else "the 20 canteen classes"
        print(f"{shard.name}: {n} images ({scope}) -> {sub}", flush=True)
    print(f"total written: {total}")


if __name__ == "__main__":
    main()
