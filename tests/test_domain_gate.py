"""The domain probe: "is this even a canteen dish?" before "which one?".

The 21-class head answers only the second question, so a pizza comes back as a
stir-fry.  A logistic regression on the same backbone's pooled feature asks the
first question -- it is not a second network, just one dot product over the
features the forward pass already produced.  These tests pin the contract:

* the probe can only take a verdict away;
* it does not need its own forward pass (it reads the frames analyse_cn returns);
* its message is about cuisine, not about "some other Chinese dish";
* with no probe file loaded, the canteen gate behaves exactly as before.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls import app as A  # noqa: E402
from foodcls.classes_cn import CN_NUM_CLASSES  # noqa: E402

OTHER = CN_NUM_CLASSES


def views_with(prob: float, index: int = 0):
    v = np.full((6, CN_NUM_CLASSES + 1), (1.0 - prob) / CN_NUM_CLASSES, dtype=np.float64)
    v[:, index] = prob
    v[3:, index] = prob * 0.7
    return v


def fused(top: int, top_p: float, other_p: float):
    p = np.full(CN_NUM_CLASSES + 1, (1.0 - top_p - other_p) / (CN_NUM_CLASSES - 1))
    p[top] = top_p
    p[OTHER] = other_p
    return p


def head(threshold: float = 0.5, dim: int = 4):
    return {
        "w": np.array([1.0, -1.0] + [0.0] * (dim - 2)),
        "b": 0.0,
        "mean": np.zeros(dim),
        "scale": np.ones(dim),
        "threshold": threshold,
    }


class DomainProbTest(unittest.TestCase):
    def setUp(self):
        self._saved = A._state.get("domain_head")
        A._state["domain_head"] = head()

    def tearDown(self):
        A._state["domain_head"] = self._saved

    def test_sigmoid_follows_the_weight_sign(self):
        cn = A._domain_prob(np.array([[1.0, -1.0, 0.0, 0.0]]))
        west = A._domain_prob(np.array([[-1.0, 1.0, 0.0, 0.0]]))
        self.assertGreater(cn, 0.8)
        self.assertLess(west, 0.2)

    def test_views_are_averaged_before_the_dot_product(self):
        frames = np.array([[2.0, -2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        mixed = A._domain_prob(frames)
        self.assertAlmostEqual(mixed, A._domain_prob(np.array([[1.0, -1.0, 0.0, 0.0]])), places=9)

    def test_no_head_and_no_frames_are_both_none(self):
        self.assertIsNone(A._domain_prob(None))
        A._state["domain_head"] = None
        self.assertIsNone(A._domain_prob(np.zeros((1, 4))))


class DomainGateTest(unittest.TestCase):
    def test_confident_probe_keeps_a_confident_verdict(self):
        sc = A.scope_of(views_with(0.9), fused(3, 0.7, 0.05), cn=True,
                        cn_other_prob=0.05, cn_domain_prob=0.9)
        self.assertTrue(sc["in_scope"])
        self.assertFalse(sc["domain"]["fired"])
        self.assertEqual(sc["domain"]["zh"], "食堂中餐")

    def test_probe_takes_the_verdict_away(self):
        sc = A.scope_of(views_with(0.9), fused(3, 0.7, 0.05), cn=True,
                        cn_other_prob=0.05, cn_domain_prob=0.1)
        self.assertFalse(sc["in_scope"])
        self.assertTrue(sc["domain"]["fired"])
        self.assertIn("食堂中餐", sc["message"])
        self.assertIn("单菜识别", sc["message"])
        self.assertNotIn("其它中餐", sc["message"])

    def test_cuisine_message_beats_the_bucket_message(self):
        """A western photo that also trips the bucket should still read as cuisine."""
        sc = A.scope_of(views_with(0.9), fused(3, 0.4, min(0.95, A.CN_OTHER_THRESHOLD + 0.1)),
                        cn=True, cn_other_prob=min(0.95, A.CN_OTHER_THRESHOLD + 0.1),
                        cn_domain_prob=0.05)
        self.assertTrue(sc["other"]["fired"])
        self.assertTrue(sc["domain"]["fired"])
        self.assertIn("食堂中餐", sc["message"])

    def test_probe_cannot_put_a_weak_photo_in_scope(self):
        sc = A.scope_of(views_with(0.2), fused(3, 0.9, 0.01), cn=True,
                        cn_other_prob=0.01, cn_domain_prob=0.99)
        self.assertFalse(sc["in_scope"])

    def test_trusted_probe_buys_a_lower_gate(self):
        weak = A.scope_of(views_with(0.37), fused(3, 0.6, 0.05), cn=True,
                          cn_other_prob=0.05, cn_domain_prob=0.6)
        self.assertFalse(weak["in_scope"])
        self.assertAlmostEqual(weak["threshold"], A.CN_SCOPE_THRESHOLD_21, places=6)
        strong = A.scope_of(views_with(0.37), fused(3, 0.6, 0.05), cn=True,
                            cn_other_prob=0.05, cn_domain_prob=0.95)
        self.assertTrue(strong["in_scope"])
        self.assertAlmostEqual(strong["threshold"], 0.36, places=6)

    def test_trusted_probe_still_fails_the_other_gates(self):
        sc = A.scope_of(views_with(0.37), fused(3, 0.4, min(0.95, A.CN_OTHER_THRESHOLD + 0.1)),
                        cn=True, cn_other_prob=min(0.95, A.CN_OTHER_THRESHOLD + 0.1),
                        cn_domain_prob=0.99)
        self.assertFalse(sc["in_scope"])

    def test_payload_reads_the_same_frames_once(self):
        """The probe must not need a second forward pass: payload takes the frames."""
        p = np.full(CN_NUM_CLASSES + 1, 0.01)
        p[1], p[2], p[3] = 0.30, 0.20, 0.15
        frames = np.array([[1.0, -1.0, 0.0, 0.0]])
        saved = A._state.get("domain_head")
        A._state["domain_head"] = head()
        try:
            expected = round(A._domain_prob(frames), 4)
            out = A.payload(p, source="unit-test", cn=True, img=None, cn_frames=frames)
        finally:
            A._state["domain_head"] = saved
        self.assertEqual(out["scope"]["domain"]["prob"], expected)
        self.assertNotIn("domain", A.payload(p, source="unit-test", cn=True, img=None)["scope"])

    def test_without_the_probe_the_gate_is_unchanged(self):
        saved = A._state.get("domain_head")
        A._state["domain_head"] = None
        try:
            sc = A.scope_of(views_with(0.9), fused(3, 0.7, 0.05), cn=True, cn_other_prob=0.05)
        finally:
            A._state["domain_head"] = saved
        self.assertTrue(sc["in_scope"])
        self.assertNotIn("domain", sc)
