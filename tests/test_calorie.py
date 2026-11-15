"""Unit tests for the calorie display rule (no torch / no dataset needed)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.calorie import (  # noqa: E402
    BASE_REL_SPREAD,
    LOW_CONF_EXTRA,
    MIN_TOP_CONF,
    class_calorie_range,
    format_range,
    portion_scale,
    topk_with_calories,
    user_grams,
)
from foodcls.classes import CLASS_NAMES, KCAL_PER_100G, PORTION_G  # noqa: E402


class PortionScaleTest(unittest.TestCase):
    def test_named_portions(self):
        self.assertAlmostEqual(portion_scale("small"), 0.7)
        self.assertAlmostEqual(portion_scale("normal"), 1.0)
        self.assertAlmostEqual(portion_scale("large"), 1.35)
        self.assertAlmostEqual(portion_scale("LARGE"), 1.35)

    def test_unknown_and_missing(self):
        self.assertEqual(portion_scale(None), 1.0)
        self.assertEqual(portion_scale("gigantic"), 1.0)

    def test_numeric_factor_is_clamped(self):
        self.assertAlmostEqual(portion_scale(1.5), 1.5)
        self.assertAlmostEqual(portion_scale(10), 3.0)
        self.assertAlmostEqual(portion_scale(0.01), 0.3)


class UserGramsTest(unittest.TestCase):
    def test_valid_values(self):
        self.assertAlmostEqual(user_grams("150"), 150.0)
        self.assertAlmostEqual(user_grams(80), 80.0)

    def test_missing_and_garbage(self):
        self.assertIsNone(user_grams(None))
        self.assertIsNone(user_grams(""))
        self.assertIsNone(user_grams("about 200"))
        self.assertIsNone(user_grams(0))
        self.assertIsNone(user_grams(-40))

    def test_range_is_clamped(self):
        self.assertAlmostEqual(user_grams(1), 10.0)
        self.assertAlmostEqual(user_grams(5000), 2000.0)


class ClassCalorieRangeTest(unittest.TestCase):
    cls = "fried_rice"

    def test_interval_brackets_the_point_estimate(self):
        rng = class_calorie_range(self.cls, 0.95)
        self.assertLess(rng["low"], rng["point"])
        self.assertLess(rng["point"], rng["high"])
        self.assertAlmostEqual(rng["point"], KCAL_PER_100G[self.cls] * PORTION_G[self.cls] / 100, places=0)
        self.assertEqual(rng["portion_g"], PORTION_G[self.cls])
        self.assertEqual(rng["weight_source"], "portion")

    def test_low_confidence_widens_the_band(self):
        confident = class_calorie_range(self.cls, 0.9)
        unsure = class_calorie_range(self.cls, 0.4)
        self.assertAlmostEqual(confident["spread"], BASE_REL_SPREAD)
        self.assertAlmostEqual(unsure["spread"], BASE_REL_SPREAD + LOW_CONF_EXTRA)
        width = lambda r: r["high"] - r["low"]  # noqa: E731
        self.assertGreater(width(unsure), width(confident))

    def test_portion_multiplier_scales_the_interval(self):
        normal = class_calorie_range(self.cls, 0.9)
        large = class_calorie_range(self.cls, 0.9, scale="large")
        self.assertGreater(large["point"], normal["point"])
        self.assertAlmostEqual(large["point"] / normal["point"], 1.35, places=1)

    def test_user_weight_overrides_the_prior(self):
        rng = class_calorie_range(self.cls, 0.9, grams=100)
        self.assertEqual(rng["weight_source"], "user")
        self.assertEqual(rng["portion_g"], 100)
        self.assertAlmostEqual(rng["point"], KCAL_PER_100G[self.cls], places=0)


class TopKTest(unittest.TestCase):
    def _probs(self, head):
        rest = (1.0 - sum(head)) / (len(CLASS_NAMES) - len(head))
        return list(head) + [rest] * (len(CLASS_NAMES) - len(head))

    def test_returns_descending_top_k(self):
        probs = self._probs([0.5, 0.3, 0.15])
        top = topk_with_calories(probs, k=3)
        self.assertEqual([t["class"] for t in top], CLASS_NAMES[:3])
        self.assertEqual([t["prob"] for t in top], sorted([t["prob"] for t in top], reverse=True))

    def test_noise_candidates_are_dropped(self):
        probs = self._probs([0.5, 0.3, MIN_TOP_CONF / 2])
        top = topk_with_calories(probs, k=3)
        self.assertEqual(len(top), 2)

    def test_format_range(self):
        item = {"kcal_low": 520, "kcal_high": 700}
        self.assertEqual(format_range(item), "520-700 kcal")


if __name__ == "__main__":
    unittest.main()
