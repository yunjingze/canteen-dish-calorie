"""The "switch tab" button: when is another tab's verdict a promise we can keep?

A refusal on one tab often means the photo belongs to another -- a pizza on the
canteen tab, a 红烧肉 on the single-dish tab.  The button is only offered when
the other tab's *own* gate accepts the photo, and the canteen tab stays quiet
when its own cues say "this is canteen food I could not name" (veto/weak).
These tests pin that contract without loading any weights.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcls import app as A  # noqa: E402


class RefusalReasonTest(unittest.TestCase):
    def test_cues_are_read_in_the_order_scope_of_checks_them(self):
        self.assertEqual(A._refusal_reason({"domain": {"fired": True},
                                            "other": {"fired": True},
                                            "veto": {"fired": True}}), "domain")
        self.assertEqual(A._refusal_reason({"other": {"fired": True},
                                            "veto": {"fired": True}}), "other")
        self.assertEqual(A._refusal_reason({"veto": {"fired": True}}), "veto")
        self.assertEqual(A._refusal_reason({}), "weak")
        self.assertEqual(A._refusal_reason({"domain": {"fired": False}}), "weak")


class TabOrderTest(unittest.TestCase):
    def test_every_mode_points_at_other_classifier_tabs(self):
        for cur, alts in A._TAB_ORDER.items():
            self.assertTrue(alts, cur)
            self.assertNotIn(cur, alts)
            self.assertNotIn("tray", alts)  # the tray tab is a correction tool
        self.assertEqual(set(A._TAB_ORDER), {"single", "wide", "cn"})

    def test_scope_specs_name_a_tab_for_every_mode(self):
        for mode, spec in A._SCOPE_SPECS.items():
            self.assertIn(spec["tab"], ("单菜识别", "116 类", "食堂菜"), mode)


class SuggestTest(unittest.TestCase):
    def setUp(self):
        self._saved = A._tab_scope
        self.calls = []

    def tearDown(self):
        A._tab_scope = self._saved

    def _tab_accepts(self, img, mode):
        self.calls.append(mode)
        return {"in_scope": True}

    def test_nothing_to_suggest_when_the_tab_accepted(self):
        self.assertIsNone(A.suggest_tab(object(), "cn", {"in_scope": True}))
        self.assertEqual(self.calls, [])

    def test_a_photo_missing_a_dish_is_not_a_tab_problem(self):
        for scope in ({"in_scope": False, "not_food": True},
                      {"in_scope": False, "packaging": True}):
            self.assertIsNone(A.suggest_tab(object(), "cn", scope))
        self.assertEqual(self.calls, [])

    def test_canteen_refusal_with_no_evidence_offers_no_button(self):
        A._tab_scope = self._tab_accepts
        self.assertIsNone(A.suggest_tab(object(), "cn", {"in_scope": False}))
        self.assertIsNone(A.suggest_tab(object(), "cn",
                                        {"in_scope": False, "veto": {"fired": True}}))
        self.assertEqual(self.calls, [])

    def test_western_food_on_the_canteen_tab_offers_the_single_tab(self):
        A._tab_scope = self._tab_accepts
        got = A.suggest_tab(object(), "cn", {"in_scope": False, "domain": {"fired": True}})
        self.assertEqual(got["mode"], "single")
        self.assertEqual(got["tab"], "单菜识别")
        self.assertEqual(got["reason"], "domain")

    def test_canteen_dish_on_the_single_tab_offers_the_canteen_tab(self):
        A._tab_scope = self._tab_accepts
        got = A.suggest_tab(object(), "single", {"in_scope": False})
        self.assertEqual(got["mode"], "cn")
        self.assertEqual(got["tab"], "食堂菜")
        self.assertEqual(got["reason"], "weak")

    def test_the_other_tab_has_to_accept_it_itself(self):
        A._tab_scope = lambda img, mode: {"in_scope": False}
        self.assertIsNone(A.suggest_tab(object(), "single", {"in_scope": False}))


if __name__ == "__main__":
    unittest.main()
