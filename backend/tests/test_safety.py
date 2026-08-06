import unittest

from app.core.safety import SafetyFilter


class SafetyFilterTests(unittest.TestCase):
    def setUp(self):
        self.f = SafetyFilter()

    def test_normal_edit_prompts_are_allowed(self):
        for prompt in [
            "remove the chair",
            "change the sky to sunset",
            "replace the background with a studio wall",
            "make the grass greener",
            "remove the person in the background",  # removal, not manipulation
        ]:
            with self.subTest(prompt=prompt):
                self.assertTrue(self.f.check(prompt).allowed)

    def test_empty_prompt_rejected(self):
        v = self.f.check("   ")
        self.assertFalse(v.allowed)
        self.assertEqual(v.category, "empty")

    def test_general_adult_content_is_allowed(self):
        # Adult mode / the general adult filter were removed from this build.
        for prompt in ["a nude portrait", "nsfw version please",
                       "make her nude"]:
            with self.subTest(prompt=prompt):
                self.assertTrue(self.f.check(prompt).allowed)

    def test_exposure_edits_of_photos_still_blocked(self):
        # Undressing an existing photo of a person is NCII — always blocked.
        for prompt in ["remove his shirt", "undress the woman",
                       "remove her clothes"]:
            with self.subTest(prompt=prompt):
                v = self.f.check(prompt, editing=True)
                self.assertFalse(v.allowed)
                self.assertEqual(v.category, "exposure")

    def test_deepfake_prompts_blocked(self):
        for prompt in ["face swap with my boss", "deepfake this video",
                       "put my face on this body", "replace her face"]:
            with self.subTest(prompt=prompt):
                v = self.f.check(prompt)
                self.assertFalse(v.allowed)
                self.assertEqual(v.category, "deepfake")

    def test_nonconsensual_blocked(self):
        v = self.f.check("edit this photo taken without her consent")
        self.assertFalse(v.allowed)
        self.assertEqual(v.category, "nonconsensual")

    def test_minor_related_body_edits_blocked(self):
        v = self.f.check("change the child's clothes to something else")
        self.assertFalse(v.allowed)
        self.assertEqual(v.category, "minors")

    def test_sexual_plus_minor_is_minors_category(self):
        v = self.f.check("nude teen")
        self.assertFalse(v.allowed)
        self.assertEqual(v.category, "minors")

    def test_trivial_obfuscation_of_exposure_caught(self):
        v = self.f.check("u.n.d.r.e.s.s her", editing=True)
        # "undress" reconstructs after separator stripping
        self.assertFalse(v.allowed)

    def test_no_substring_false_positives(self):
        # "grass" contains "ass"-like substrings; word boundaries must protect these
        for prompt in ["add more grass", "sharpen the classic car",
                       "brighten the skyline photo of the city"]:
            with self.subTest(prompt=prompt):
                self.assertTrue(self.f.check(prompt).allowed)


if __name__ == "__main__":
    unittest.main()
