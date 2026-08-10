"""Getting the inpaint right on the FIRST attempt, and never paying for a
retry the pixels can disprove.

Retrying was measured at 2.5x-12.7x the cost of the render it retries — a
quarter of all pipeline time — and the vision judge that triggers it scored
the SAME image 20 and 70 on two runs. Three defences, each deterministic:

  conditioning   a change-request leads with the TARGET STATE ("a red
                 shirt"), not the instruction ("change the shirt to red"),
                 and the displaced state goes to the negative.

  colour proof   when the requirement names a colour, a hue count over the
                 masked pixels settles it — in both directions.

  no-op proof    a masked region that came back untouched is a failed
                 recipe; the retry must change the recipe, not the adverbs.
"""
import inspect
import unittest

from PIL import Image, ImageDraw

from app.core import quality


class SegmentTheSource(unittest.TestCase):
    """The text engine unions its phrases, so segmentation must look for the
    SOURCE of a change, never the destination. Measured live: "blue" as a
    phrase lit up the entire blue background of a test image at peak 0.956
    and 89% coverage — an edit mask that would repaint the sky along with
    the car it was meant to select."""

    def test_change_to_segments_only_the_source(self):
        self.assertEqual(quality.mask_phrases("change the red car to blue"),
                         ["red car"])

    def test_replace_with_segments_only_the_source(self):
        self.assertEqual(
            quality.mask_phrases("replace the wooden fence with a hedge"),
            ["wooden fence"])

    def test_make_state_segments_only_the_source(self):
        # A non-garment noun: garment requests route to the part vocabulary
        # first ("shirt" would come back as its part label), by design.
        self.assertEqual(quality.mask_phrases("make the wall green"),
                         ["wall"])

    def test_non_change_requests_keep_the_word_union(self):
        self.assertIn("hat", quality.mask_phrases("remove the hat"))

    def test_garment_requests_never_carry_the_destination(self):
        phrases = quality.mask_phrases("change her red dress to blue")
        self.assertNotIn("blue", " ".join(phrases).lower())
        self.assertTrue(any("dress" in p.lower() for p in phrases), phrases)

    def test_render_paths_target_prefix_does_not_defeat_the_parse(self):
        """The render hands the chooser target + instruction; the prefix
        used to break the anchored parse and put the destination back
        among the phrases."""
        request = quality.chooser_request("car", "change the red car to blue")
        self.assertEqual(request, "change the red car to blue")
        self.assertEqual(quality.mask_phrases(request), ["red car"])

    def test_non_change_steps_keep_the_target_prefix(self):
        self.assertEqual(quality.chooser_request("hat", "remove the hat"),
                         "hat remove the hat")


class AttributeConditioning(unittest.TestCase):

    def cond(self, instruction, target=""):
        return quality.attribute_conditioning(
            instruction, target, f"{instruction}, photorealistic",
            "blurry, low quality")

    def test_change_to_bare_colour_leads_with_the_state(self):
        out = self.cond("change the shirt to red", "shirt")
        self.assertTrue(out["positive"].startswith("(a red shirt:1.2)"))
        self.assertIn("change the shirt to red", out["positive"])

    def test_the_displaced_attribute_goes_to_the_negative(self):
        out = self.cond("change the blue shirt to red", "shirt")
        self.assertIn("blue shirt", out["negative"])
        self.assertTrue(out["positive"].startswith("(a red shirt:1.2)"))

    def test_a_bare_noun_source_is_never_negated(self):
        """Negating "shirt" would fight the red shirt being asked for."""
        out = self.cond("change the shirt to red", "shirt")
        self.assertNotIn("shirt", out["negative"])

    def test_replace_with_keeps_the_destination_as_written(self):
        out = self.cond("replace the cap with a red leather jacket", "cap")
        self.assertTrue(out["positive"].startswith(
            "(a red leather jacket:1.2)"))
        self.assertIn("replace the cap", out["positive"])

    def test_make_plus_state_word_parses(self):
        out = self.cond("make the dress green", "dress")
        self.assertTrue(out["positive"].startswith("(a green dress:1.2)"))

    def test_make_plus_action_does_not_parse(self):
        """'make her smile at the camera' is not an attribute change; forcing
        a state-first prompt onto it would mangle the request."""
        self.assertIsNone(self.cond("make her smile at the camera"))

    def test_removals_and_additions_are_not_claimed(self):
        self.assertIsNone(self.cond("remove the hat", "hat"))
        self.assertIsNone(self.cond("add a necklace", "necklace"))

    def test_trailing_colour_word_is_trimmed(self):
        out = self.cond("change the car to a red color", "car")
        self.assertTrue(out["positive"].startswith("(a red car:1.2)"))

    def test_the_users_words_always_survive(self):
        for text in ("change the shirt to red",
                     "replace the cap with a beret",
                     "make the dress black"):
            out = self.cond(text)
            self.assertIn(text, out["positive"], text)


def region(size=(120, 120), box=(30, 30, 90, 90)):
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask


class ColourProof(unittest.TestCase):

    def canvas(self, colour, size=(120, 120), box=(30, 30, 90, 90)):
        img = Image.new("RGB", size, (128, 128, 128))
        ImageDraw.Draw(img).rectangle(box, fill=colour)
        return img

    def test_a_red_region_is_red(self):
        self.assertIs(quality.colour_delivered(
            self.canvas((200, 30, 30)), region(), "red"), True)

    def test_a_red_region_is_not_blue(self):
        self.assertIs(quality.colour_delivered(
            self.canvas((200, 30, 30)), region(), "blue"), False)

    def test_black_and_white_are_judged_on_light_not_hue(self):
        self.assertIs(quality.colour_delivered(
            self.canvas((15, 15, 15)), region(), "black"), True)
        self.assertIs(quality.colour_delivered(
            self.canvas((245, 245, 245)), region(), "white"), True)

    def test_a_mixed_region_is_inconclusive_not_failed(self):
        """A garment half in shadow must not be called 'not red' — the model
        verdict stands in the grey zone."""
        img = self.canvas((200, 30, 30))
        ImageDraw.Draw(img).rectangle((30, 30, 90, 60), fill=(40, 40, 40))
        share = quality._colour_share(img, region(), "red")
        self.assertTrue(0.08 < share < 0.65)

    def test_requirement_colour_finds_the_word(self):
        self.assertEqual(quality.requirement_colour(
            "the shirt being bright red"), "red")
        self.assertEqual(quality.requirement_colour("a golden ring"), "gold")
        self.assertIsNone(quality.requirement_colour("a bigger hat"))


class NoOpProof(unittest.TestCase):

    def test_an_untouched_region_measures_zero(self):
        img = Image.new("RGB", (120, 120), (90, 90, 90))
        self.assertLess(quality.region_change(img, img.copy(), region()),
                        0.001)

    def test_a_repainted_region_measures_large(self):
        before = Image.new("RGB", (120, 120), (90, 90, 90))
        after = before.copy()
        ImageDraw.Draw(after).rectangle((30, 30, 90, 90), fill=(220, 40, 40))
        self.assertGreater(quality.region_change(before, after, region()),
                           0.1)

    def test_change_outside_the_mask_does_not_count(self):
        before = Image.new("RGB", (120, 120), (90, 90, 90))
        after = before.copy()
        ImageDraw.Draw(after).rectangle((0, 0, 20, 20), fill=(255, 255, 255))
        self.assertLess(quality.region_change(before, after, region()),
                        0.001)


class Wiring(unittest.TestCase):
    """The measurements must actually gate the ladder."""

    def source(self):
        from app.core.services import Services
        return inspect.getsource(Services._handle_image_edit)

    def test_the_first_attempt_leads_with_the_target_state(self):
        self.assertIn("quality.attribute_conditioning(", self.source())

    def test_colour_is_settled_before_and_during_retries(self):
        src = self.source()
        self.assertIn("settle_colour(current, best_missing)", src)
        self.assertIn("settle_colour(\n                        candidate",
                      src)

    def test_an_unchanged_region_skips_the_same_recipe_round(self):
        src = self.source()
        self.assertIn("region_change(final_input, current", src)
        self.assertIn("(rounds > 1 or force_swap) and can_swap_model", src)


if __name__ == "__main__":
    unittest.main()
