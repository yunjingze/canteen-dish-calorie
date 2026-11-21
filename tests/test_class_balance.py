"""Contract tests for the class-balance / focal options of train_cn241."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

spec = importlib.util.spec_from_file_location("train_cn241", ROOT / "scripts" / "train_cn241.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
FocalLoss, class_weights = train.FocalLoss, train.class_weights


def fake_dataset(counts):
    samples = []
    for label, n in enumerate(counts):
        samples += [(f"{label}/{i}.jpg", label) for i in range(n)]
    return SimpleNamespace(samples=samples)


class ClassWeightsTest(unittest.TestCase):
    def test_inverse_frequency_and_clipping(self):
        ds = fake_dataset([100] * 19 + [1])
        w, counts = class_weights(ds, 20, cap=4.0)
        self.assertEqual(len(w), 20)
        self.assertLess(w[0], w[19])          # the rare class gets a bigger weight
        self.assertAlmostEqual(float(w[0]), 1901 / (20 * 100), places=6)
        self.assertAlmostEqual(float(w.max()), 4.0, places=6)  # capped at 4, not 95x
        self.assertEqual(int(counts[19]), 1)

    def test_equal_counts_give_unit_weights(self):
        w, _ = class_weights(fake_dataset([7, 7, 7]), 3)
        self.assertTrue(torch.allclose(torch.tensor(w), torch.ones(3, dtype=torch.float64)))

    def test_skip_pins_a_class_to_unit_weight(self):
        ds = fake_dataset([2100, 900, 223])
        plain, _ = class_weights(ds, 3, cap=4.0)
        self.assertLess(float(plain[0]), 1.0)          # the big "other" bucket is down-weighted
        w, _ = class_weights(ds, 3, cap=4.0, skip=[0])
        self.assertEqual(float(w[0]), 1.0)             # skipped class untouched
        self.assertAlmostEqual(float(w[1]), float(plain[1]), places=9)  # others unchanged
        self.assertEqual(float(w[2]), 4.0)             # rarest still capped


class FocalLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(6, 5) * 3
        self.target = torch.tensor([0, 1, 2, 3, 4, 0])

    def test_gamma_zero_matches_cross_entropy(self):
        ref = torch.nn.CrossEntropyLoss(label_smoothing=0.1)(self.logits, self.target)
        got = FocalLoss(gamma=0.0, label_smoothing=0.1)(self.logits, self.target)
        self.assertAlmostEqual(float(ref), float(got), places=6)

    def test_weights_are_applied(self):
        w = [1.0, 1.0, 1.0, 1.0, 8.0]
        plain = FocalLoss(gamma=0.0)(self.logits, self.target)
        weighted = FocalLoss(gamma=0.0, weight=w)(self.logits, self.target)
        self.assertGreater(float(weighted), float(plain))

    def test_focal_downweights_confident_examples(self):
        logits = self.logits.clone()
        logits[torch.arange(6), self.target] = 20.0  # everything is easy
        ce = FocalLoss(gamma=0.0)(logits, self.target)
        fl = FocalLoss(gamma=2.0)(logits, self.target)
        self.assertLess(float(fl), float(ce) * 0.01)


if __name__ == "__main__":
    unittest.main()
