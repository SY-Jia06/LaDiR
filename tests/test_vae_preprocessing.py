"""Pure preprocessing regression tests (no model download required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


# data_vae imports Hugging Face datasets for the full input pipeline. Stub the
# symbol here so the pure split functions can be tested in lightweight CI.
datasets_stub = types.ModuleType("datasets")
datasets_stub.Dataset = object
sys.modules.setdefault("datasets", datasets_stub)

MODULE_PATH = Path(__file__).parents[1] / "vae" / "data_vae.py"
spec = importlib.util.spec_from_file_location("data_vae", MODULE_PATH)
data_vae = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(data_vae)


class VAEPreprocessingTest(unittest.TestCase):
    def test_answer_prefix_split(self):
        cot, answer = data_vae.split_cot_and_answer(
            "First compute 2 + 3 = 5. The answer is: 5."
        )
        self.assertEqual(cot, "First compute 2 + 3 = 5.")
        self.assertEqual(answer, "The answer is: 5.")

    def test_sentence_blockization_preserves_decimals(self):
        blocks = data_vae.split_sentence_blocks(
            "Let x = 3.5. Then 2x = 7.\nTherefore x is valid."
        )
        self.assertEqual(
            blocks,
            ["Let x = 3.5.", "Then 2x = 7.", "Therefore x is valid."],
        )

    def test_sentence_blockization_preserves_common_abbreviations(self):
        blocks = data_vae.split_sentence_blocks(
            "Use Eq. 3, e.g. the quadratic formula. Then simplify."
        )
        self.assertEqual(
            blocks,
            ["Use Eq. 3, e.g. the quadratic formula.", "Then simplify."],
        )

    def test_sentence_blockization_preserves_initialisms(self):
        blocks = data_vae.split_sentence_blocks(
            "The U.S. value is 4. Therefore the result follows."
        )
        self.assertEqual(
            blocks,
            ["The U.S. value is 4.", "Therefore the result follows."],
        )

    def test_missing_answer_prefix_is_explicit(self):
        with self.assertRaises(ValueError):
            data_vae.split_cot_and_answer("A reasoning trace without an answer.")


if __name__ == "__main__":
    unittest.main()
