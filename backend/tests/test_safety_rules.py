"""Custom safety rules: add/delete, live (never-cached) enforcement, and the
guarantee that built-in protections can't be removed."""
import tempfile
import unittest
from pathlib import Path

from app.core.db import Database
from app.core.safety import SafetyFilter, SafetyRuleError, SafetyRuleStore


class SafetyRuleStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "t.sqlite3")
        self.store = SafetyRuleStore(self.db)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_add_list_delete(self):
        self.assertEqual(self.store.list(), [])
        rule = self.store.add("velociraptor", "no dinosaurs")
        self.assertEqual(rule["pattern"], "velociraptor")
        self.assertEqual(rule["category"], "custom")
        self.assertEqual(len(self.store.list()), 1)
        self.assertTrue(self.store.delete(rule["id"]))
        self.assertEqual(self.store.list(), [])
        self.assertFalse(self.store.delete(rule["id"]))  # already gone

    def test_reserved_category_rejected(self):
        for cat in ("minors", "exposure", "deepfake", "nonconsensual"):
            with self.subTest(cat=cat), self.assertRaises(SafetyRuleError):
                self.store.add("something", "x", category=cat)

    def test_invalid_regex_rejected_and_empty_rejected(self):
        with self.assertRaises(SafetyRuleError):
            self.store.add("(unclosed", "x")
        with self.assertRaises(SafetyRuleError):
            self.store.add("   ", "x")

    def test_default_reason_supplied(self):
        rule = self.store.add("bananas")
        self.assertIn("custom", rule["reason"])


class LiveEnforcementTests(unittest.TestCase):
    """The filter must reflect add/delete on the very next check — proving the
    rules are read live and never served from a cache."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(Path(self.tmp.name) / "t.sqlite3")
        self.store = SafetyRuleStore(self.db)
        self.filter = SafetyFilter(custom_provider=self.store.compiled)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_custom_rule_takes_effect_immediately(self):
        self.assertTrue(self.filter.check("a photo of a unicorn").allowed)
        rule = self.store.add("unicorn", "no unicorns", category="mythical")
        v = self.filter.check("a photo of a unicorn")
        self.assertFalse(v.allowed)
        self.assertEqual(v.category, "mythical")
        self.assertEqual(v.reason, "no unicorns")
        # ...and removing it re-allows on the next check, no restart.
        self.store.delete(rule["id"])
        self.assertTrue(self.filter.check("a photo of a unicorn").allowed)

    def test_whole_word_match_no_substring_false_positive(self):
        self.store.add("gun")
        self.assertFalse(self.filter.check("holding a gun").allowed)
        self.assertTrue(self.filter.check("the race had begun").allowed)

    def test_custom_rules_cannot_disable_builtins(self):
        # Even with custom rules present, the locked protections still fire.
        self.store.add("teapot")
        self.assertFalse(self.filter.check("undress the woman",
                                           editing=True).allowed)
        self.assertFalse(self.filter.check("face swap with a celebrity").allowed)
        self.assertFalse(self.filter.check("nude teen").allowed)

    def test_broken_provider_never_disables_builtins(self):
        def boom():
            raise RuntimeError("db exploded")
        f = SafetyFilter(custom_provider=boom)
        self.assertFalse(f.check("undress the woman", editing=True).allowed)

    def test_builtin_summary_lists_locked_categories(self):
        cats = {r["category"] for r in SafetyFilter.builtin_summary()}
        self.assertEqual(cats,
                         {"minors", "exposure", "deepfake", "nonconsensual"})
        self.assertTrue(all(r["locked"] for r in SafetyFilter.builtin_summary()))


if __name__ == "__main__":
    unittest.main()
