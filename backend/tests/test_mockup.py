"""The layout engine: interface mock-ups drawn, not diffused.

The live request behind this module: "based on the layout and style from
this menu, create a mock-up for a new menu for item upgrades with tabs
for killstreaks, perks, hunter shop, zombie shop and rank shop, each tab
has multiple items that can be upgraded with Essence from lvl 1 to 3,
use paging" — text-dense by construction, which diffusion letters as
gibberish. Here every glyph is drawn by Pillow."""
import inspect
import math
import unittest

from PIL import Image

from app.core import mockup

LIVE = ("based on the layout and style from this menu, create a mock-up "
        "for a new menu for item upgrades with tabs for killstreaks, "
        "perks, hunter shop, zombie shop and rank shop, each tab has "
        "multiple items that can be upgraded with Essence from lvl 1 "
        "to 3, use paging")


class _SpecLLM:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, prompt, max_tokens=4096, **kw):
        class R:
            text = self.payload
        return R()


def _spec(tabs=None):
    import json
    return json.dumps({
        "title": "Item Upgrades", "currency": "Essence", "max_level": 3,
        "active_tab": "Killstreaks",
        "tabs": tabs if tabs is not None else [
            {"name": name, "items": [
                {"name": f"{name} item {i}", "level": i % 4,
                 "cost": 100 * (i + 1)} for i in range(8)]}
            for name in ("Killstreaks", "Perks", "Hunter Shop",
                         "Zombie Shop", "Rank Shop")]})


class RequestedTabsTests(unittest.TestCase):
    def test_the_live_prompt_parses_to_its_five_tabs(self):
        self.assertEqual(
            mockup.requested_tabs(LIVE),
            ["killstreaks", "perks", "hunter shop", "zombie shop",
             "rank shop"])

    def test_missing_tabs_names_what_the_plan_dropped(self):
        spec = {"tabs": [{"name": "Killstreaks"}, {"name": "Perks"}]}
        self.assertEqual(
            mockup.missing_tabs(LIVE, spec),
            ["hunter shop", "zombie shop", "rank shop"])
        full = {"tabs": [{"name": n} for n in (
            "Killstreaks", "Perks", "Hunter Shop", "Zombie Shop",
            "Rank Shop")]}
        self.assertEqual(mockup.missing_tabs(LIVE, full), [])


class SpecTests(unittest.TestCase):
    def test_a_schema_reply_normalizes(self):
        spec = mockup.mockup_spec(_SpecLLM(_spec()), LIVE)
        self.assertEqual(len(spec["tabs"]), 5)
        self.assertEqual(spec["currency"], "Essence")
        self.assertEqual(spec["max_level"], 3)
        for tab in spec["tabs"]:
            for item in tab["items"]:
                self.assertLessEqual(item["level"], 3)
                self.assertGreaterEqual(item["level"], 0)

    def test_no_planner_or_garbage_means_none(self):
        self.assertIsNone(mockup.mockup_spec(None, LIVE))
        self.assertIsNone(mockup.mockup_spec(_SpecLLM("not json"), LIVE))
        self.assertIsNone(
            mockup.mockup_spec(_SpecLLM('{"title": "x"}'), LIVE))


class StyleTests(unittest.TestCase):
    def test_valid_hexes_land_and_junk_keeps_defaults(self):
        class C:
            def ask(self, image, q, schema=None):
                return ('{"background": "#101418", "panel": "#nothex", '
                        '"border": "#334", "accent": "#ffcc00", '
                        '"text": "#f0f0f0", "vibe": "pixel"}')

        style = mockup.style_from_reference(C(), Image.new("RGB", (8, 8)))
        self.assertEqual(style["background"], "#101418")
        self.assertEqual(style["accent"], "#ffcc00")
        self.assertEqual(style["panel"], mockup.DEFAULT_STYLE["panel"])
        self.assertEqual(style["border"], mockup.DEFAULT_STYLE["border"])
        self.assertEqual(style["vibe"], "pixel")

    def test_no_critic_is_the_default_style(self):
        self.assertEqual(mockup.style_from_reference(None, None),
                         mockup.DEFAULT_STYLE)


class RenderTests(unittest.TestCase):
    def test_the_live_spec_draws_in_both_vibes(self):
        spec = mockup.mockup_spec(_SpecLLM(_spec()), LIVE)
        for vibe in ("rounded", "pixel"):
            img = mockup.render_mockup(
                spec, {**mockup.DEFAULT_STYLE, "vibe": vibe})
            self.assertEqual(img.size, (1280, 832))
            # the accent must actually appear (active tab, pips, chips)
            accent = mockup.DEFAULT_STYLE["accent"].lstrip("#")
            target = tuple(int(accent[i:i + 2], 16) for i in (0, 2, 4))
            self.assertIn(target, [p for p in img.getdata()][::7])

    def test_paging_is_real_arithmetic(self):
        self.assertEqual(math.ceil(8 / mockup.PER_PAGE), 2)
        self.assertEqual(math.ceil(14 / mockup.PER_PAGE), 3)


class RoutingTests(unittest.TestCase):
    def test_mockups_route_to_the_layout_engine_before_comfyui(self):
        from app.core.services import Services
        src = inspect.getsource(Services._workflow_inner)
        head = src.split("self._require_comfy(job)")[0]
        self.assertIn("quality.ui_mockup_intent(prompt)", head)
        self.assertIn("self._render_ui_mockup(job, prompt)", head)
        rsrc = inspect.getsource(Services._render_ui_mockup)
        self.assertIn("missing_tabs", rsrc)
        self.assertIn("every tab you named", rsrc)


if __name__ == "__main__":
    unittest.main()
