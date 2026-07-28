from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from dataset import ReasoningDataCollator, ReasoningLatentDataset


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [1] + [2 + (ord(char) % 13) for char in text]}


class ReasonerDatasetTest(unittest.TestCase):
    def test_memmap_ranges_and_collation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latents = np.arange(5 * 2 * 3, dtype=np.float16).reshape(5, 2, 3)
            np.save(root / "latents.npy", latents)
            records = [
                {
                    "input": "q1",
                    "answer": "a1",
                    "latent_start": 0,
                    "num_blocks": 2,
                },
                {
                    "input": "q2",
                    "answer": "a2",
                    "latent_start": 2,
                    "num_blocks": 3,
                },
            ]
            with (root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            metadata = {
                "latent_tokens_per_block": 2,
                "latent_dim": 3,
                "dtype": "float16",
                "num_blocks": 5,
                "num_samples": 2,
                "latent_mode": "sample",
            }
            (root / "metadata.json").write_text(json.dumps(metadata))

            dataset = ReasoningLatentDataset(
                TinyTokenizer(),
                str(root / "manifest.jsonl"),
                str(root / "latents.npy"),
                metadata_path=str(root / "metadata.json"),
            )
            batch = ReasoningDataCollator(0)([dataset[0], dataset[1]])
            self.assertEqual(
                tuple(batch["oracle_latents"].shape), (2, 3, 2, 3)
            )
            self.assertTrue(
                torch.equal(
                    batch["block_mask"][0],
                    torch.tensor([1, 1, 0], dtype=torch.bool),
                )
            )
            self.assertTrue(
                torch.equal(
                    batch["block_mask"][1],
                    torch.tensor([1, 1, 1], dtype=torch.bool),
                )
            )
            self.assertTrue(batch["question_attention_mask"].all())
            self.assertTrue(batch["answer_attention_mask"].all())


if __name__ == "__main__":
    unittest.main()
