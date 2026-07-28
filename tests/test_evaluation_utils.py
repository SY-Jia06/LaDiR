from __future__ import annotations

import unittest

from evaluate import (
    exact_checker,
    last_number,
    last_number_checker,
    normalize_exact,
    parse_ks,
)


class EvaluationUtilsTest(unittest.TestCase):
    def test_exact_normalization(self):
        self.assertEqual(normalize_exact("  The   Answer IS: 7 "), "the answer is: 7")
        self.assertTrue(exact_checker("A  B", "a b", {}))

    def test_last_number_matching(self):
        self.assertEqual(last_number("x=1,234.5"), "1234.5")
        self.assertTrue(
            last_number_checker(
                "work ... The answer is 70.",
                "The answer is: 70.",
                {},
            )
        )
        self.assertFalse(last_number_checker("no numeric answer", "7", {}))

    def test_pass_at_k_is_clipped_to_available_samples(self):
        self.assertEqual(parse_ks("1,10,25,100", 25), [1, 10, 25])
        self.assertEqual(parse_ks("100", 8), [8])


if __name__ == "__main__":
    unittest.main()
