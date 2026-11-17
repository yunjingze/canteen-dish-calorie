"""The tray head now spans both menus: 20 western dishes + 20 canteen dishes.

The 40-class multi-label checkpoint must load next to the old 20-class one, and
the parts it reports must carry the right Chinese names, the right calorie
table and a cuisine tag, so one plate can mix a stir fry with fries.  These
tests pin that contract on the app side (the trainer/eval side lives in
scripts/train_mixed.py).
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls import app as A  # noqa: E402
from foodcls.classes import CLASS_NAMES, KCAL_PER_100G, PORTION_G, ZH_LABELS  # noqa: E402
from foodcls.classes_cn import CN_CLASS_NAMES  # noqa: E402
from foodcls.plate_classes import (  # noqa: E402
    PLATE_CLASS_NAMES,
    PLATE_KCAL_PER_100G,
    PLATE_PORTION_G,
    PLATE_ZH,
)

OFFSET = len(CLASS_NAMES)


def plate_state(names):
    """Install the tables load_mixed_model() would have installed."""
    if names == list(CLASS_NAMES):
        zh, kcal, grams = ZH_LABELS, KCAL_PER_100G, PORTION_G
    else:
        zh, kcal, grams = PLATE_ZH, PLATE_KCAL_PER_100G, PLATE_PORTION_G
    return {
        "mixed_names": list(names),
        "mixed_zh": zh,
        "mixed_kcal": kcal,
        "mixed_portion": grams,
        "mixed_threshold": 0.5,
        "mixed_class_thresholds": {},
    }


class _State:
    """Swap a handful of _state keys for the duration of one test."""

    KEYS = ("mixed_names", "mixed_zh", "mixed_kcal", "mixed_portion",
            "mixed_threshold", "mixed_class_thresholds")

    def __init__(self, names):
        self.new = plate_state(names)

    def __enter__(self):
        self.saved = {k: A._state.get(k) for k in self.KEYS}
        A._state.update(self.new)
        return A._state

    def __exit__(self, *exc):
        for k in self.KEYS:
            if self.saved[k] is None:
                A._state.pop(k, None)
            else:
                A._state[k] = self.saved[k]
        return False


def tray(hits):
    """Sigmoid scores over 40 units; ``hits`` maps class slug -> score."""
    p = np.full(len(PLATE_CLASS_NAMES), 0.02, dtype=np.float64)
    for slug, score in hits.items():
        p[PLATE_CLASS_NAMES.index(slug)] = score
    return p


class MixedTablesTest(unittest.TestCase):
    def test_twenty_class_checkpoint_keeps_the_old_tables(self):
        names, zh, kcal, grams = A._mixed_tables({"class_names": list(CLASS_NAMES)})
        self.assertEqual(names, list(CLASS_NAMES))
        self.assertIs(zh, ZH_LABELS)
        self.assertIs(kcal, KCAL_PER_100G)
        self.assertIs(grams, PORTION_G)

    def test_forty_class_checkpoint_uses_the_bilingual_tables(self):
        names, zh, kcal, grams = A._mixed_tables({"class_names": list(PLATE_CLASS_NAMES)})
        self.assertEqual(len(names), 40)
        self.assertEqual(names[OFFSET], CN_CLASS_NAMES[0])
        self.assertIs(zh, PLATE_ZH)
        self.assertIs(kcal, PLATE_KCAL_PER_100G)
        self.assertIn(CN_CLASS_NAMES[0], kcal)

    def test_a_renamed_class_list_is_refused(self):
        names = list(PLATE_CLASS_NAMES)
        names[-1] = "not_a_dish"
        with self.assertRaises(RuntimeError) as ctx:
            A._mixed_tables({"class_names": names})
        self.assertIn("refusing to load", str(ctx.exception))


class MixedPayloadTest(unittest.TestCase):
    def test_canteen_dish_is_named_and_counted(self):
        cn_slug = CN_CLASS_NAMES[0]
        # if the first dish is also a western slug this test is meaningless
        self.assertNotIn(cn_slug, CLASS_NAMES)
        with _State(PLATE_CLASS_NAMES):
            out = A.mixed_payload(tray({cn_slug: 0.93, "french_fries": 0.71}), source="unit-test")
        classes = [p["class"] for p in out["parts"]]
        self.assertEqual(classes[0], cn_slug)
        canteen = out["parts"][0]
        self.assertEqual(canteen["cuisine"], "canteen")
        self.assertEqual(canteen["zh"], PLATE_ZH[cn_slug])
        self.assertEqual(out["parts"][1]["cuisine"], "western")
        rows = {r["class"]: r for r in out["plate"]["parts"]}
        self.assertIn(cn_slug, rows)
        self.assertAlmostEqual(rows[cn_slug]["kcal_per_100g"], PLATE_KCAL_PER_100G[cn_slug])
        self.assertGreater(out["plate"]["total"]["kcal_point"], 0)
        self.assertIn("中西混搭", out["note"])

    def test_canteen_only_tray_gets_the_canteen_note(self):
        with _State(PLATE_CLASS_NAMES):
            out = A.mixed_payload(tray({CN_CLASS_NAMES[1]: 0.9}), source="unit-test")
        self.assertEqual([p["cuisine"] for p in out["parts"]], ["canteen"])
        self.assertIn("食堂托盘", out["note"])

    def test_twenty_class_payload_is_unchanged(self):
        p = np.full(len(CLASS_NAMES), 0.02)
        p[CLASS_NAMES.index("pizza")] = 0.9
        with _State(CLASS_NAMES):
            out = A.mixed_payload(p, source="unit-test")
        self.assertEqual([x["class"] for x in out["parts"]], ["pizza"])
        self.assertEqual(out["parts"][0]["zh"], ZH_LABELS["pizza"])
        self.assertNotIn("中西混搭", out["note"])


class MixedTotalEndpointTest(unittest.TestCase):
    def test_total_accepts_a_canteen_slug(self):
        cn_slug = CN_CLASS_NAMES[2]
        body = {"parts": [{"class": cn_slug, "prob": 0.9, "portion": "normal"}]}
        with _State(PLATE_CLASS_NAMES):
            out = asyncio.run(A.mixed_total(body))
        self.assertEqual(out["parts"][0]["class"], cn_slug)
        self.assertGreater(out["total"]["kcal_point"], 0)

    def test_total_still_accepts_western_slugs(self):
        body = {"parts": [{"class": "french_fries", "prob": 0.8}]}
        with _State(PLATE_CLASS_NAMES):
            out = asyncio.run(A.mixed_total(body))
        self.assertEqual(out["parts"][0]["class"], "french_fries")

    def test_health_reports_the_bilingual_head(self):
        from fastapi.testclient import TestClient
        with _State(PLATE_CLASS_NAMES):
            body = TestClient(A.app).get("/api/health").json()
        self.assertEqual(body["mixed_classes"], 40)
        self.assertEqual(body["mixed_canteen_classes"], 20)


if __name__ == "__main__":
    unittest.main()