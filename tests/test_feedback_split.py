"""Contract tests for turning the correction CSV into a trainable split."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_feedback_split import plan, write  # noqa: E402


def row(name, predicted="", corrected="", removed="", mode="single"):
    return {
        "timestamp": "2026-11-26T00:00:00",
        "file": name,
        "predicted": predicted,
        "corrected": corrected,
        "removed": removed,
        "note": "",
        "source": "upload",
        "mode": mode,
    }


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        for n in ("a.jpg", "b.jpg", "c.jpg"):
            (self.dir / n).write_bytes(b"x")

    def tearDown(self):
        self.tmp.cleanup()

    def test_single_dish_row_lands_in_its_mode(self):
        out, stats = plan([row("a.jpg", corrected="pizza")], self.dir)
        self.assertEqual([Path(p).name for p, _ in out["single"]], ["a.jpg"])
        self.assertEqual(stats["used"], 1)

    def test_canteen_slug_is_accepted_and_rice_too(self):
        out, _ = plan([row("a.jpg", corrected="tomato_egg", mode="cn"),
                       row("b.jpg", corrected="rice", mode="cn")], self.dir)
        self.assertEqual(len(out["cn"]), 2)

    def test_tray_keeps_several_dishes(self):
        out, _ = plan([row("a.jpg", corrected="hamburger;french_fries", mode="mixed")],
                      self.dir)
        self.assertEqual(out["mixed"][0][1], ["hamburger", "french_fries"])

    def test_struck_out_dish_is_not_a_positive(self):
        out, _ = plan([row("a.jpg", corrected="hamburger;french_fries",
                            removed="french_fries", mode="mixed")], self.dir)
        self.assertEqual(out["mixed"][0][1], ["hamburger"])

    def test_strike_out_only_row_falls_back_to_the_prediction(self):
        out, stats = plan([row("a.jpg", predicted="stir_fried_greens;spicy_chicken",
                            removed="stir_fried_greens", mode="mixed")], self.dir)
        self.assertEqual(out["mixed"][0][1], ["spicy_chicken"])
        self.assertEqual(stats["weak_positive"], 1)

    def test_row_with_no_positive_is_skipped(self):
        out, stats = plan([row("a.jpg", predicted="", removed="pizza", mode="mixed")],
                           self.dir)
        self.assertEqual(out["mixed"], [])
        self.assertEqual(stats["no_positive"], 1)

    def test_unknown_class_is_skipped_not_guessed(self):
        _, stats = plan([row("a.jpg", corrected="not_a_dish")], self.dir)
        self.assertEqual(stats["bad_class"], 1)
        self.assertEqual(stats["used"], 0)

    def test_missing_image_is_counted(self):
        out, stats = plan([row("nope.jpg", corrected="pizza")], self.dir)
        self.assertEqual(out["single"], [])
        self.assertEqual(stats["missing_image"], 1)

    def test_two_dishes_in_a_single_mode_row_is_skipped(self):
        _, stats = plan([row("a.jpg", corrected="pizza;sushi")], self.dir)
        self.assertEqual(stats["not_single_dish"], 1)


class WriteTest(unittest.TestCase):
    def test_images_land_in_class_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            (src / "a.jpg").write_bytes(b"x")
            (src / "b.jpg").write_bytes(b"x")
            out, _ = plan([row("a.jpg", corrected="pizza"),
                           row("b.jpg", corrected="pizza")], src)
            written = write(out, Path(tmp) / "built")
            self.assertEqual(len(written), 2)
            dest = Path(tmp) / "built" / "single" / "pizza"
            self.assertEqual(len(list(dest.glob("*.jpg"))), 2)

    def test_manifest_keeps_the_label_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            (src / "a.jpg").write_bytes(b"x")
            out, _ = plan([row("a.jpg", corrected="hamburger;french_fries",
                                mode="mixed")], src)
            written = write(out, Path(tmp) / "built")
            self.assertEqual(list(written.values())[0], ["hamburger", "french_fries"])


if __name__ == "__main__":
    unittest.main()
