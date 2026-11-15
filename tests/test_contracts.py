"""Contract tests: class tables, transforms and the model factory."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from foodcls.classes import CLASS_NAMES, KCAL_PER_100G, NUM_CLASSES, PORTION_G, ZH_LABELS  # noqa: E402
from foodcls.data import build_transforms  # noqa: E402
from foodcls.models import apply_finetune_policy, build_model  # noqa: E402


class ClassTableTest(unittest.TestCase):
    def test_tables_are_consistent(self):
        self.assertEqual(NUM_CLASSES, len(CLASS_NAMES))
        self.assertEqual(len(set(CLASS_NAMES)), len(CLASS_NAMES))
        for table in (ZH_LABELS, KCAL_PER_100G, PORTION_G):
            self.assertEqual(set(table), set(CLASS_NAMES))

    def test_priors_are_plausible(self):
        for name in CLASS_NAMES:
            self.assertGreaterEqual(KCAL_PER_100G[name], 50)
            self.assertLessEqual(KCAL_PER_100G[name], 600)
            self.assertGreaterEqual(PORTION_G[name], 50)
            self.assertLessEqual(PORTION_G[name], 1000)


class TransformTest(unittest.TestCase):
    def test_eval_transform_shape(self):
        img = Image.new("RGB", (300, 260), color=(120, 80, 40))
        out = build_transforms(False, False)(img)
        self.assertEqual(tuple(out.shape), (3, 224, 224))
        self.assertTrue(torch.isfinite(out).all())

    def test_train_transform_shape(self):
        img = Image.new("RGB", (400, 400), color=(10, 200, 90))
        out = build_transforms(True, True)(img)
        self.assertEqual(tuple(out.shape), (3, 224, 224))


class ModelFactoryTest(unittest.TestCase):
    def test_every_architecture_maps_to_the_head(self):
        for arch in ("resnet18", "mobilenet_v3_small", "efficientnet_b0"):
            model = build_model(arch, NUM_CLASSES, pretrained=False).eval()
            with torch.no_grad():
                out = model(torch.zeros(2, 3, 224, 224))
            self.assertEqual(tuple(out.shape), (2, NUM_CLASSES), arch)

    def test_finetune_policy_freezes_the_backbone(self):
        model = build_model("resnet18", NUM_CLASSES, pretrained=False)
        trainable, total = apply_finetune_policy(model, "frozen")
        self.assertLess(trainable, total)
        self.assertTrue(all(p.requires_grad for p in model.fc.parameters()))
        self.assertFalse(model.conv1.weight.requires_grad)

    def test_full_policy_trains_everything(self):
        model = build_model("resnet18", NUM_CLASSES, pretrained=False)
        trainable, total = apply_finetune_policy(model, "full")
        self.assertEqual(trainable, total)

    def test_unknown_architecture_is_rejected(self):
        with self.assertRaises(ValueError):
            build_model("vgg16", NUM_CLASSES, pretrained=False)


if __name__ == "__main__":
    unittest.main()
