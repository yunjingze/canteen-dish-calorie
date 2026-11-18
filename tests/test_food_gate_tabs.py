"""The food gate now speaks for the wide and tray tabs too.

A menu board, a napkin or an empty cup is not a dish, whichever head is asked.
The gate was fitted on the narrow backbone's pooled feature, and both other
tabs already run that backbone, so the same answer is free.  These tests pin the
contract:

* a wide verdict can be taken away by the gate, never added to;
* the tray tab reports no parts at all when the gate says "no food here";
* without a gate loaded, both tabs behave exactly as before.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls import app as A  # noqa: E402
from foodcls.plate_classes import (  # noqa: E402
    PLATE_CLASS_NAMES,
    PLATE_KCAL_PER_100G,
    PLATE_PORTION_G,
    PLATE_ZH,
)


class FakeGate:
    """Stands in for FoodGate: one threshold, one dot product."""

    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold

    def prob(self, feats):
        return float(np.asarray(feats, dtype=np.float64).sum())


def wide_views(prob: float, index: int = 0):
    n = len(A.WIDE_CLASS_NAMES)
    v = np.full((6, n), (1.0 - prob) / (n - 1), dtype=np.float64)
    v[:, index] = prob
    v[3:, index] = prob * 0.7
    return v


def wide_fused(prob: float, index: int = 0):
    n = len(A.WIDE_CLASS_NAMES)
    p = np.full(n, (1.0 - prob) / (n - 1), dtype=np.float64)
    p[index] = prob
    return p


class _Gate:
    """Install a fake gate for the duration of one test."""

    def __init__(self, gate):
        self.gate = gate

    def __enter__(self):
        self.saved = A._state.get("foodgate")
        A._state["foodgate"] = self.gate
        return self.gate

    def __exit__(self, *exc):
        if self.saved is None:
            A._state.pop("foodgate", None)
        else:
            A._state["foodgate"] = self.saved
        return False


class _PlateTables:
    """Install the 40-class tables load_mixed_model() would have installed."""

    KEYS = ("mixed_names", "mixed_zh", "mixed_kcal", "mixed_portion",
            "mixed_threshold", "mixed_class_thresholds")

    def __enter__(self):
        self.saved = {k: A._state.get(k) for k in self.KEYS}
        A._state.update({
            "mixed_names": list(PLATE_CLASS_NAMES),
            "mixed_zh": PLATE_ZH,
            "mixed_kcal": PLATE_KCAL_PER_100G,
            "mixed_portion": PLATE_PORTION_G,
            "mixed_threshold": 0.5,
            "mixed_class_thresholds": {},
        })
        return A._state

    def __exit__(self, *exc):
        for k in self.KEYS:
            if self.saved[k] is None:
                A._state.pop(k, None)
            else:
                A._state[k] = self.saved[k]
        return False


def tray(hits):
    p = np.full(len(PLATE_CLASS_NAMES), 0.02, dtype=np.float64)
    for slug, score in hits.items():
        p[PLATE_CLASS_NAMES.index(slug)] = score
    return p


class FoodProbTest(unittest.TestCase):
    def test_no_frames_and_no_gate_are_both_none(self):
        self.assertIsNone(A._food_prob(None))
        with _Gate(None):
            self.assertIsNone(A._food_prob(np.zeros((3, 512))))

    def test_the_gate_reads_the_first_frame(self):
        with _Gate(FakeGate()):
            frames = np.array([[0.25, 0.25], [9.0, 9.0]])
            self.assertAlmostEqual(A._food_prob(frames), 0.5)


class WideGateTest(unittest.TestCase):
    def test_a_low_food_prob_takes_a_wide_verdict_away(self):
        with _Gate(FakeGate(0.2)):
            sc = A.scope_of(wide_views(0.9), wide_fused(0.9), wide=True, food_prob=0.05)
        self.assertFalse(sc["in_scope"])
        self.assertTrue(sc["not_food"])
        self.assertIn("没找到菜", sc["message"])
        self.assertEqual(sc["food_prob"], 0.05)
        self.assertEqual(sc["food_threshold"], 0.2)

    def test_a_confident_food_prob_leaves_the_wide_verdict_alone(self):
        with _Gate(FakeGate(0.2)):
            sc = A.scope_of(wide_views(0.9), wide_fused(0.9), wide=True, food_prob=0.95)
        self.assertTrue(sc["in_scope"])
        self.assertNotIn("not_food", sc)

    def test_without_a_gate_the_wide_tab_is_unchanged(self):
        with _Gate(None):
            sc = A.scope_of(wide_views(0.9), wide_fused(0.9), wide=True)
        self.assertTrue(sc["in_scope"])
        self.assertNotIn("food_prob", sc)


class TrayGateTest(unittest.TestCase):
    def test_no_food_means_no_parts_and_no_calories(self):
        with _PlateTables():
            out = A.mixed_payload(tray({"tomato_egg": 0.93, "french_fries": 0.71}),
                                  source="unit-test", scope={"not_food": True})
        self.assertEqual(out["parts"], [])
        self.assertEqual(out["plate"]["total"]["kcal_point"], 0)
        self.assertEqual(out["plate"]["total"]["grams"], 0)
        self.assertIn("没找到菜", out["note"])

    def test_food_present_keeps_the_parts(self):
        with _PlateTables():
            out = A.mixed_payload(tray({"tomato_egg": 0.93, "french_fries": 0.71}),
                                  source="unit-test", scope={"in_scope": True})
        self.assertEqual(len(out["parts"]), 2)
        self.assertGreater(out["plate"]["total"]["kcal_point"], 0)


if __name__ == "__main__":
    unittest.main()
