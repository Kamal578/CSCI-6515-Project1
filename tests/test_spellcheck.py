import unittest
from collections import Counter

from src.spellcheck import suggest
from src.serve_spellcheck import expand_suggest


class TestSpellcheck(unittest.TestCase):
    def test_suggest_ranks_by_distance_then_freq(self):
        vocab = Counter({"azərbaycan": 10, "azerbaycan": 5, "kitab": 3})
        cands = suggest("azrbaycan", vocab, max_dist=2, top_k=3)
        self.assertTrue(len(cands) >= 1)
        top_word, _ = cands[0]
        self.assertEqual(top_word, "azərbaycan")

    def test_expand_suggest_uses_variants(self):
        vocab = Counter({"çay": 10, "cay": 1})
        cands = expand_suggest("cay", vocab, max_dist=1, top_k=5, weights=None, max_variant_edits=2, max_variant_candidates=20)
        tokens = [tok for tok, _ in cands]
        self.assertIn("çay", tokens)

    def test_expand_suggest_sherbet(self):
        vocab = Counter({"şərbət": 8})
        cands = expand_suggest("sherbet", vocab, max_dist=2, top_k=5, weights=None, max_variant_edits=3, max_variant_candidates=40)
        tokens = [tok for tok, _ in cands]
        self.assertIn("şərbət", tokens)


if __name__ == "__main__":
    unittest.main()
