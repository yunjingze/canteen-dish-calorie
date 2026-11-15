"""Chain-menu nutrition references for the demo.

The course priors describe a *generic* dish (175 kcal per 100 g of fried rice).
Chain menus publish measured per-serving values, which anchor the fast-food
classes much better: a Big Mac is 513 kcal for 220 g, not "a burger is roughly
550".  This module turns the scraped table in data/external/brand_menu.json into

  * a searchable menu for the "brand calories" panel of the web page, and
  * per-class references shown under a prediction (hamburger -> Big Mac, ...).

Provenance matters, so every row keeps its brand, serving and source URL.  Only
McDonald's China publishes a machine-readable table; KFC China does not, and we
do not invent numbers for it.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
MENU_PATH = ROOT / "data" / "external" / "brand_menu.json"

#: which menu items are a sensible reference for a model class.  Names must match
#: the scraped table exactly; missing items are skipped rather than invented.
CLASS_ITEMS: Dict[str, List[str]] = {
    "hamburger": ["巨无霸", "板烧鸡腿堡", "麦辣鸡腿汉堡", "双层吉士汉堡", "吉士汉堡包"],
    "french_fries": ["脆脆薯条", "薯条"],
    "chicken_wings": ["麦辣鸡翅", "薄皮焦香V翅"],
    "ice_cream": ["圆筒冰淇淋", "奥利奥麦旋风", "朱古力新地"],
}

#: panel order: dishes first, drinks/desserts next, combos last
CATEGORY_ORDER = [
    "汉堡和卷", "小食", "甜品", "饮品", "早餐",
    "McCafé 饮品及其他", "500大卡套餐", "开心乐园餐",
]

#: short note shown next to the references
BRAND_NOTE = "官网每份实测值；份量与配方会随门店调整，仅作对照"


@lru_cache(maxsize=1)
def _table() -> dict:
    try:
        return json.loads(MENU_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"items": [], "sources": []}


@lru_cache(maxsize=1)
def _by_name() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row in _table().get("items", []):
        out.setdefault(row["item"], row)
    return out


def menu() -> dict:
    """Everything the front-end needs: rows, sources and the class mapping."""
    data = _table()
    items = data.get("items", [])
    rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    items = sorted(items, key=lambda r: (rank.get(r.get("category", ""), len(rank)), r["item"]))
    return {
        "count": len(items),
        "generated_at": data.get("generated_at"),
        "sources": data.get("sources", []),
        "note": BRAND_NOTE,
        "class_map": {k: [n for n in v if n in _by_name()] for k, v in CLASS_ITEMS.items()},
        "items": [
            {
                "brand_zh": r.get("brand_zh", ""),
                "item": r["item"],
                "category": r.get("category", ""),
                "serving": r.get("serving", ""),
                "grams": r.get("grams"),
                "kcal": r.get("kcal"),
                "kcal_per_100g": r.get("kcal_per_100g"),
                "protein_g": r.get("protein_g"),
                "fat_g": r.get("fat_g"),
                "carbs_g": r.get("carbs_g"),
                "sodium_mg": r.get("sodium_mg"),
            }
            for r in items
        ],
    }


def refs_for_class(class_name: str, limit: int = 3, per_100g_scale: float = 1.0) -> List[dict]:
    """Reference servings for one class.

    The per_100g_scale argument lets the caller rescale the portion to the one
    the user selected (small = 0.7x), so the comparison is apples to apples.
    """
    names = CLASS_ITEMS.get(class_name) or []
    out: List[dict] = []
    for name in names:
        row = _by_name().get(name)
        if not row:
            continue
        grams = row.get("grams")
        kcal = row.get("kcal")
        if grams and kcal and per_100g_scale != 1.0:
            grams = int(round(grams * per_100g_scale))
            kcal = int(round(kcal * per_100g_scale))
        out.append(
            {
                "brand": row.get("brand"),
                "brand_zh": row.get("brand_zh", ""),
                "item": name,
                "serving": row.get("serving", ""),
                "grams": grams,
                "kcal": kcal,
                "kcal_per_100g": row.get("kcal_per_100g"),
                "source_url": row.get("source_url", ""),
            }
        )
        if len(out) >= limit:
            break
    return out


def reference_span(class_name: str):
    """Min/max per-100 g of the matched chain items, for the README cross-check."""
    rows = [r for r in refs_for_class(class_name, limit=99) if r.get("kcal_per_100g")]
    if not rows:
        return None
    vals = [r["kcal_per_100g"] for r in rows]
    return {"min": min(vals), "max": max(vals), "n": len(vals)}
