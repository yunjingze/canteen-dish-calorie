"""The 21st canteen class: "some other Chinese dish" is a refusal, not a name.

The 20-dish head force-fits every photo into its 20 classes, so a dish it never
learned still gets a confident name.  The head is now also trained on a bucket
class built from the rest of CNFOOD-241; these tests pin the contract:

* the bucket only takes a verdict away (never adds one);
* it never reaches the calorie table (no name, no kcal, no portion);
* "closest" still names a real dish, so the refusal message is actionable;
* a 20-class checkpoint from before the retrain still loads.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls import app as A  # noqa: E402
from foodcls.classes_cn import CN_CLASS_NAMES, CN_NUM_CLASSES  # noqa: E402

OTHER = CN_NUM_CLASSES  # the bucket sits last


def views_with(prob: float, index: int = 0):
    """Six view posteriors whose first three views all peak at ``prob``."""
    v = np.full((6, CN_NUM_CLASSES + 1), (1.0 - prob) / CN_NUM_CLASSES, dtype=np.float64)
    v[:, index] = prob
    v[3:, index] = prob * 0.7  # mirrors are weaker, and must not decide the gate
    return v


def fused(top: int, top_p: float, other_p: float):
    p = np.full(CN_NUM_CLASSES + 1, (1.0 - top_p - other_p) / (CN_NUM_CLASSES - 1))
    p[top] = top_p
    p[OTHER] = other_p
    return p


class OtherBucketGateTest(unittest.TestCase):
    def test_quiet_bucket_keeps_a_confident_verdict(self):
        sc = A.scope_of(views_with(0.9), fused(3, 0.7, 0.05), cn=True, cn_other_prob=0.05)
        self.assertTrue(sc["in_scope"])
        self.assertFalse(sc["other"]["fired"])
        self.assertEqual(sc["other"]["zh"], "其它中餐")

    def test_loud_bucket_takes_the_verdict_away(self):
        loud = min(0.95, A.CN_OTHER_THRESHOLD + 0.10)
        sc = A.scope_of(views_with(0.9), fused(3, 0.42, loud), cn=True, cn_other_prob=loud)
        self.assertFalse(sc["in_scope"])
        self.assertTrue(sc["other"]["fired"])
        self.assertIn("其它中餐", sc["message"])
        self.assertNotIn("最接近的「其它中餐」", sc["message"])

    def test_bucket_cannot_put_a_weak_photo_in_scope(self):
        """A high in-list score with a dead bucket is still gate-limited."""
        v = views_with(0.2)  # worst view 0.2 < CN_SCOPE_THRESHOLD
        sc = A.scope_of(v, fused(3, 0.9, 0.01), cn=True, cn_other_prob=0.01)
        self.assertFalse(sc["in_scope"])

    def test_bucket_never_reaches_the_calorie_table(self):
        # three in-list candidates above MIN_TOP_CONF, and a bucket that fires
        p = np.full(CN_NUM_CLASSES + 1, 0.01)
        p[1], p[2], p[3] = 0.30, 0.20, 0.15
        p[OTHER] = min(0.95, A.CN_OTHER_THRESHOLD + 0.10)
        out = A.payload(p, source="unit-test", cn=True, img=None)
        self.assertEqual(len(out["top3"]), 3)
        names = [it["class"] for it in out["top3"]]
        self.assertTrue(set(names) <= set(CN_CLASS_NAMES), names)
        self.assertNotIn(A.CN_OTHER_SLUG, names)
        self.assertFalse(out["scope"]["in_scope"])
        self.assertAlmostEqual(out["scope"]["other"]["prob"], float(p[OTHER]), places=4)

    def test_bucket_low_keeps_top3(self):
        p = fused(3, 0.60, 0.02)
        out = A.payload(p, source="unit-test", cn=True, img=None)
        self.assertTrue(out["scope"]["in_scope"])
        self.assertEqual(out["top3"][0]["class"], CN_CLASS_NAMES[3])


class CheckpointShapeTest(unittest.TestCase):
    """load_cn_model() must accept both list shapes and refuse a renamed class."""

    @staticmethod
    def _has_other(names):
        return (len(names) == CN_NUM_CLASSES + 1
                and names[:CN_NUM_CLASSES] == list(CN_CLASS_NAMES)
                and names[-1] == A.CN_OTHER_SLUG)

    def test_both_shapes(self):
        self.assertFalse(self._has_other(list(CN_CLASS_NAMES)))
        self.assertTrue(self._has_other(list(CN_CLASS_NAMES) + [A.CN_OTHER_SLUG]))

    def test_renamed_class_is_rejected(self):
        names = list(CN_CLASS_NAMES)
        names[0] = "not_a_dish"
        self.assertFalse(self._has_other(names))

    def test_health_reports_the_bucket(self):
        from fastapi.testclient import TestClient
        body = TestClient(A.app).get("/api/health").json()
        self.assertEqual(body["cn_classes"], CN_NUM_CLASSES)
        self.assertEqual(body["cn_other"]["zh"], "其它中餐")
        self.assertIn("threshold", body["cn_other"])


class RenormalisedGateTest(unittest.TestCase):
    """The gate is measured on the dish mass, the bucket answers separately."""

    def test_bucket_mass_does_not_sink_a_supported_dish(self):
        # 33 % on the right dish and 35 % on the bucket: the old 0.40 gate would
        # throw this away, the renormalised score is 0.33 / 0.65 = 0.51.
        v = np.full((6, CN_NUM_CLASSES + 1), 0.004, dtype=np.float64)
        v[:, 3] = 0.33
        v[:, OTHER] = 0.35
        sc = A.scope_of(v, fused(3, 0.33, 0.35), cn=True, cn_other_prob=0.35)
        self.assertGreater(sc["robust_prob"], A.CN_SCOPE_THRESHOLD)
        self.assertFalse(sc["other"]["fired"])
        self.assertTrue(sc["in_scope"])

    def test_bucket_that_wins_the_argmax_refuses_even_below_threshold(self):
        # p_other 0.30 is under the threshold, but it is still the top class:
        # "more of this photo looks like a dish we never learned" is a refusal.
        v = np.full((6, CN_NUM_CLASSES + 1), 0.004, dtype=np.float64)
        v[:, 3] = 0.25
        v[:, OTHER] = 0.30
        sc = A.scope_of(v, fused(3, 0.25, 0.30), cn=True,
                        cn_other_prob=0.30, cn_other_top=True)
        self.assertFalse(sc["in_scope"])
        self.assertTrue(sc["other"]["fired"])
        self.assertEqual(sc["closest"], A.CN_ZH[CN_CLASS_NAMES[3]])

    def test_classes_endpoint_reports_the_bucket(self):
        from fastapi.testclient import TestClient
        body = TestClient(A.app).get("/api/cn/classes").json()
        self.assertEqual(body["count"], CN_NUM_CLASSES)
        self.assertEqual(body["other"]["zh"], "其它中餐")
        self.assertAlmostEqual(body["other"]["threshold"], A.CN_OTHER_THRESHOLD)
        # the gate threshold is the 21-class one only when that checkpoint is loaded
        self.assertIn(body["threshold"], (A.CN_SCOPE_THRESHOLD, A.CN_SCOPE_THRESHOLD_21))

if __name__ == "__main__":
    unittest.main()