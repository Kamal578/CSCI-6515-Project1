import unittest

from src.spell_utils import generate_az_variants, generate_az_variants_with_edits


class TestSpellUtils(unittest.TestCase):
    def test_generate_variants_includes_identity_and_replacement(self):
        variants = generate_az_variants("cay", max_edits=1, max_candidates=20)
        self.assertIn("cay", variants)
        self.assertIn("çay", variants)

    def test_generate_variants_handles_digraphs(self):
        variants = generate_az_variants("chay", max_edits=1, max_candidates=20)
        self.assertIn("çay", variants)

    def test_generate_variants_respects_max_candidates(self):
        variants = generate_az_variants("sherbet", max_edits=3, max_candidates=5)
        self.assertLessEqual(len(variants), 5)

    def test_generate_variants_with_edits(self):
        variants = dict(generate_az_variants_with_edits("cay", max_edits=1, max_candidates=20))
        self.assertEqual(variants.get("cay"), 0)
        self.assertEqual(variants.get("çay"), 1)


if __name__ == "__main__":
    unittest.main()
