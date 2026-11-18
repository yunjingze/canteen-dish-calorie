"""Warm-starting the 116-class head: the head rows must follow the class names.

The narrow canteen head and the wide head share 20 dish names but not their
order (WIDE_CLASS_NAMES is alphabetical, CLASS_NAMES follows the canteen menu),
so the old positional copy scrambled them.  These tests pin the by-name copy and
the two switches the backbone study needed: "@head" (classifier only, keep the
backbone) and several --init checkpoints applied in order.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from foodcls.classes import CLASS_NAMES  # noqa: E402
from foodcls.classes_wide import WIDE_CLASS_NAMES  # noqa: E402
from foodcls.models import build_model  # noqa: E402
from scripts.train_wide import warm_start  # noqa: E402


def fake_ckpt(model, names, zero: float) -> dict:
    """A checkpoint whose fc rows are constant and easy to trace back."""
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    width = sd["fc.weight"].shape[1]
    sd["fc.weight"] = torch.stack([torch.full((width,), zero) for _ in range(len(names))])
    sd["fc.bias"] = torch.tensor([zero for _ in range(len(names))])
    return {"model": sd, "class_names": list(names), "epoch": 1, "arch": "resnet18"}


class WarmStartTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def save(self, ckpt, name):
        path = self.dir / name
        torch.save(ckpt, path)
        return path

    def test_head_rows_land_on_the_right_classes(self):
        model = build_model("resnet18", len(WIDE_CLASS_NAMES), pretrained=False)
        src = fake_ckpt(model, CLASS_NAMES, zero=7.0)
        path = self.save(src, "narrow.pt")

        before = model.fc.weight.detach().clone()
        info = warm_start(model, path)

        kept = info["head_rows"]["fc.weight"]
        self.assertEqual(sorted(kept), sorted(CLASS_NAMES))
        for name in CLASS_NAMES:
            row = model.fc.weight.detach()[WIDE_CLASS_NAMES.index(name)]
            self.assertEqual(float(row[0]), 7.0, name)
        # A class the source does not know keeps whatever it had.
        stranger = next(n for n in WIDE_CLASS_NAMES if n not in set(CLASS_NAMES))
        self.assertTrue(torch.equal(model.fc.weight[WIDE_CLASS_NAMES.index(stranger)],
                                    before[WIDE_CLASS_NAMES.index(stranger)]))

    def test_head_only_leaves_the_backbone_alone(self):
        model = build_model("resnet18", len(WIDE_CLASS_NAMES), pretrained=False)
        model.fc.weight.data.fill_(0.0)
        path = self.save(fake_ckpt(model, CLASS_NAMES, zero=3.0), "narrow.pt")

        backbone = model.conv1.weight.detach().clone()
        info = warm_start(model, path, head_only=True)

        self.assertEqual(info["tensors"], 2)
        self.assertTrue(info["head_only"])
        self.assertTrue(torch.equal(model.conv1.weight, backbone))
        self.assertEqual(float(model.fc.weight.detach()[WIDE_CLASS_NAMES.index("dumplings")][0]), 3.0)

    def test_later_checkpoint_wins_on_the_head(self):
        model = build_model("resnet18", len(WIDE_CLASS_NAMES), pretrained=False)
        backbone_path = self.save(fake_ckpt(model, CLASS_NAMES, zero=1.0), "backbone.pt")
        head_path = self.save(fake_ckpt(model, WIDE_CLASS_NAMES, zero=9.0), "wide.pt")

        backbone = model.conv1.weight.detach().clone()
        warm_start(model, backbone_path)
        warm_start(model, head_path, head_only=True)

        self.assertTrue(torch.equal(model.conv1.weight, backbone))
        self.assertEqual(float(model.fc.weight.detach()[WIDE_CLASS_NAMES.index("apple_pie")][0]), 9.0)
        self.assertEqual(float(model.fc.bias.detach()[0]), 9.0)

    def test_unknown_source_classes_are_skipped(self):
        model = build_model("resnet18", len(WIDE_CLASS_NAMES), pretrained=False)
        before = model.fc.weight.detach().clone()
        path = self.save(fake_ckpt(model, ["not_on_the_menu"], zero=5.0), "junk.pt")

        info = warm_start(model, path)

        self.assertEqual(info["head_rows"], {})
        self.assertTrue(torch.equal(model.fc.weight, before))


if __name__ == "__main__":
    unittest.main()
