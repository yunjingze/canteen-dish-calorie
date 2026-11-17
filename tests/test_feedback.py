"""Contract tests: what a correction may name, and how a list is split."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls.app import FEEDBACK_CLASSES, FEEDBACK_SPECIAL, parse_correction  # noqa: E402
from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.classes_cn import CN_CLASS_NAMES  # noqa: E402
from foodcls.classes_wide import WIDE_CLASS_NAMES  # noqa: E402


class CorrectionTest(unittest.TestCase):
    def test_every_model_can_be_corrected(self):
        self.assertTrue(set(CLASS_NAMES) <= FEEDBACK_CLASSES)
        self.assertTrue(set(WIDE_CLASS_NAMES) <= FEEDBACK_CLASSES)
        self.assertTrue(set(CN_CLASS_NAMES) <= FEEDBACK_CLASSES)
        self.assertTrue(set(FEEDBACK_SPECIAL) <= FEEDBACK_CLASSES)

    def test_canteen_only_slugs_are_valid(self):
        # Regression: "tomato_egg" is a canteen class that neither the 20-class
        # nor the 116-class table contains, so this used to fail validation and
        # every correction made in the canteen tab was thrown away.
        self.assertEqual(parse_correction("tomato_egg"), ["tomato_egg"])
        self.assertEqual(parse_correction("tomato_egg;mapo_tofu"),
                         ["tomato_egg", "mapo_tofu"])

    def test_single_class(self):
        self.assertEqual(parse_correction("pizza"), ["pizza"])

    def test_tray_names_several_dishes(self):
        self.assertEqual(parse_correction(" chicken_wings ; greek_salad "),
                         ["chicken_wings", "greek_salad"])

    def test_empty_and_unknown_are_rejected(self):
        for bad in ("", "  ", ";", "not_a_class", "pizza;nope"):
            with self.assertRaises(ValueError):
                parse_correction(bad)

    def test_strike_out_may_be_empty(self):
        self.assertEqual(parse_correction("", allow_empty=True), [])
        self.assertEqual(parse_correction(" ; ", allow_empty=True), [])

    def test_strike_out_still_validates(self):
        with self.assertRaises(ValueError):
            parse_correction("nope", allow_empty=True)

    def test_too_many_parts(self):
        with self.assertRaises(ValueError):
            parse_correction(";".join(list(CLASS_NAMES)[:7]))


if __name__ == "__main__":
    unittest.main()
