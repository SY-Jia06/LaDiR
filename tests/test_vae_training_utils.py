"""Regression tests for the VAE teacher-forcing interface."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch


# Keep this unit test independent of optional runtime integrations.
wandb_stub = types.ModuleType("wandb")
wandb_stub.run = None
wandb_stub.log = lambda *args, **kwargs: None
sys.modules.setdefault("wandb", wandb_stub)

transformers_stub = types.ModuleType("transformers")
transformers_stub.Trainer = object
trainer_utils_stub = types.ModuleType("transformers.trainer_utils")
trainer_utils_stub.get_last_checkpoint = lambda _: None
sys.modules.setdefault("transformers", transformers_stub)
sys.modules.setdefault("transformers.trainer_utils", trainer_utils_stub)

MODULE_PATH = Path(__file__).parents[1] / "vae" / "training_utils.py"
spec = importlib.util.spec_from_file_location("training_utils", MODULE_PATH)
training_utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(training_utils)


class FakeTokenizer:
    """Minimal batch tokenizer with deterministic token IDs."""

    def __call__(self, texts, **kwargs):
        del kwargs
        return {
            "input_ids": [
                [10 + index for index, _ in enumerate(text.split())]
                for text in texts
            ]
        }


class VAETeacherForcingTest(unittest.TestCase):
    def test_latents_are_followed_directly_by_text(self):
        result = training_utils.pretrain_tokenize_function(
            {
                "cot_only": ["alpha beta"],
                "chain_of_thought": ["alpha beta"],
            },
            tokenizer=FakeTokenizer(),
            model_max_length=32,
            mem_size=4,
            min_tokens_for_lm=64,
            mean_compression_rate=1,
            add_special_token_for_lm=False,
            leave_tokens_for_lm=8,
            ae_token_id=999,
            eos_id=2,
            mem=[100, 101, 102, 103],
            input_type="cot_only",
            lm_ratio=0.0,
        )

        self.assertEqual(result["input_ids"], [[10, 11]])
        self.assertEqual(
            result["prompt_answer_ids"],
            [[100, 101, 102, 103, 10, 11, 2]],
        )
        self.assertEqual(
            result["labels"],
            [[-100, -100, -100, -100, 10, 11, 2]],
        )
        self.assertNotIn(999, result["prompt_answer_ids"][0])

    def test_dynamic_padding_uses_separate_fill_values(self):
        collator = training_utils.DataCollatorForDynamicPadding(pad_token_id=32000)
        batch = collator(
            [
                {
                    "input_ids": [1, 2],
                    "prompt_answer_ids": [100, 10, 2],
                    "labels": [-100, 10, 2],
                },
                {
                    "input_ids": [1],
                    "prompt_answer_ids": [100, 2],
                    "labels": [-100, 2],
                },
            ]
        )

        self.assertTrue(torch.equal(batch["input_ids"][1], torch.tensor([1, 32000])))
        self.assertTrue(
            torch.equal(
                batch["prompt_answer_ids"][1],
                torch.tensor([100, 2, 32000]),
            )
        )
        self.assertTrue(torch.equal(batch["labels"][1], torch.tensor([-100, 2, -100])))


if __name__ == "__main__":
    unittest.main()
