"""Collect official chain-menu nutrition figures into one JSON table.

Why this exists
---------------
The course model estimates calories from a generic per-100 g table.  Chain
menus publish measured values per serving, which are a much better anchor for
fast food: a Big Mac is 513 kcal, not "a burger is roughly 550".  This script
scrapes the McDonald's China nutrition calculator (the only chain we found that
serves its full menu as parseable data) and writes data/external/brand_menu.json.

Provenance
----------
* McDonald's China, 营养计算器: https://www.mcdonalds.com.cn/nutrition_calculator
  The page embeds raw_data with Product / ProductType; each product type
  carries [grams, kJ, protein_g, fat_g, carbs_g, sodium_mg, ...].
* KFC China does not publish item-level nutrition on kfc.com.cn (checked
  2026-11: /nutrition* all 404, the ordering SPA has no nutrition endpoint),
  so only McDonald's entries are collected automatically.  brands.py still
  supports a kfc_cn block if one is added by hand from the in-store brochure.

Usage
-----
    python scripts/fetch_brand_nutrition.py            # refresh the table
    python scripts/fetch_brand_nutrition.py --check    # validate an existing file
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import ssl
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "brand_menu.json"
SOURCE = "https://www.mcdonalds.com.cn/nutrition_calculator"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120 Safari/537.36")

#: nutrition array layout of the embedded raw_data
FIELDS = ("grams", "kj", "protein_g", "fat_g", "carbs_g", "sodium_mg")


def fetch_html(url: str = SOURCE, timeout: int = 40) -> str:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return resp.read().decode("utf-8", "ignore")


def extract_raw_data(html: str) -> dict:
    """Pull the raw_data = {...}; object out of the page."""
    marker = html.find("raw_data =")
    if marker < 0:
        raise RuntimeError("raw_data block not found; the page layout changed")
    start = html.find("{", marker)
    depth = 0
    in_str = False
    esc = False
    for pos in range(start, len(html)):
        ch = html[pos]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(html[start:pos + 1])
    raise RuntimeError("unterminated raw_data object")


def _num(values: list, idx: int):
    try:
        raw = values[idx]
    except (IndexError, TypeError):
        return None
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def build_table(raw: dict) -> list:
    products = raw.get("Product") or {}
    types = raw.get("ProductType") or {}
    category_of = {}
    for block in raw.get("category_list") or []:
        cat = (block.get("ProductCategory") or {}).get("title") or ""
        for pid in block.get("Product") or []:
            category_of.setdefault(str(pid), cat)

    rows = []
    seen = set()
    for pid, prod in products.items():
        title = (prod.get("title") or "").strip()
        type_ids = prod.get("ProductType") or []
        if not title or not type_ids:
            continue
        ptype = types.get(str(type_ids[0])) or {}
        values = ptype.get("nutrition") or []
        serving = (ptype.get("title") or "").strip()
        kj = _num(values, 1)
        grams = _num(values, 0)
        if kj is None or not grams:
            continue
        key = (title, serving)
        if key in seen:
            continue
        seen.add(key)
        kcal = round(kj / 4.184)
        row = {
            "brand": "mcdonalds_cn",
            "brand_zh": "麦当劳（中国）",
            "item": title,
            "category": category_of.get(str(pid), ""),
            "serving": serving or "1份",
            "grams": int(round(grams)),
            "kj": round(kj),
            "kcal": kcal,
            "kcal_per_100g": round(kcal * 100.0 / grams),
            "source": "official",
            "source_url": SOURCE,
        }
        for idx, field in enumerate(FIELDS[2:], start=2):
            val = _num(values, idx)
            if val is not None:
                row[field] = round(val, 1)
        rows.append(row)

    rows.sort(key=lambda r: (r["category"], r["item"]))
    return rows


def write_table(rows: list) -> Path:
    payload = {
        "schema": 1,
        "generated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "sources": [
            {
                "brand": "mcdonalds_cn",
                "name_zh": "麦当劳（中国）营养计算器",
                "url": SOURCE,
                "note": "Per-serving values embedded in the official nutrition calculator; grams are the serving weight.",
            }
        ],
        "items": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return OUT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="validate the existing file instead of scraping")
    ap.add_argument("--min-items", type=int, default=60,
                    help="sanity floor for the scraped table")
    args = ap.parse_args(argv)

    if args.check:
        data = json.loads(OUT.read_text(encoding="utf-8"))
        rows = data["items"]
        assert len(rows) >= args.min_items, "only %d items" % len(rows)
        bad = [r for r in rows
               if not (50 <= r["kcal"] <= 3000) or not (10 <= r["grams"] <= 2000)]
        assert not bad, "implausible rows: %r" % bad[:3]
        print("OK %d items from %s" % (len(rows), OUT.relative_to(ROOT)))
        return 0

    html = fetch_html()
    rows = build_table(extract_raw_data(html))
    if len(rows) < args.min_items:
        print("only %d items parsed, refusing to overwrite" % len(rows), file=sys.stderr)
        return 1
    path = write_table(rows)
    cats = sorted({r["category"] for r in rows})
    print("wrote %s: %d items, %d categories" % (path.relative_to(ROOT), len(rows), len(cats)))
    for cat in cats:
        sub = [r for r in rows if r["category"] == cat]
        print("  %-14s %3d items" % (cat or "(none)", len(sub)))
    top = sorted(rows, key=lambda r: -r["kcal"])[:5]
    print("  heaviest: " + ", ".join("%s %d kcal" % (r["item"], r["kcal"]) for r in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
