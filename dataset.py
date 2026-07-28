"""Datasets for paper-aligned LaDiR reasoner training.

Reasoner training consumes one question, a variable number of VAE latent blocks,
and the final answer. The recommended path is to precompute the frozen VAE
outputs into one NumPy ``.npy`` array and keep sample metadata in JSONL. This
keeps the two 8B VAE copies out of Stage-1/Stage-2 training.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LatentStoreMetadata:
    latent_tokens_per_block: int
    latent_dim: int
    dtype: str
    num_blocks: int
    num_samples: int
    latent_mode: str = "sample"

    @classmethod
    def from_json(cls, path: str | Path) -> "LatentStoreMetadata":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(**payload)


class ReasoningLatentDataset(Dataset):
    """Question/answer records backed by a memory-mapped latent array."""

    def __init__(
        self,
        tokenizer: Any,
        manifest_path: str,
        latent_path: str,
        *,
        metadata_path: Optional[str] = None,
        max_question_length: int = 1024,
        max_answer_length: int = 512,
        max_blocks: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.manifest_path = Path(manifest_path)
        self.latent_path = Path(latent_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        if not self.latent_path.is_file():
            raise FileNotFoundError(self.latent_path)
        if max_question_length <= 0 or max_answer_length <= 0:
            raise ValueError("token length limits must be positive")
        if max_blocks is not None and max_blocks <= 0:
            raise ValueError("max_blocks must be positive when supplied")
        self.max_question_length = int(max_question_length)
        self.max_answer_length = int(max_answer_length)
        self.max_blocks = max_blocks

        self.records: list[dict[str, Any]] = []
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                missing = {
                    "input",
                    "answer",
                    "latent_start",
                    "num_blocks",
                } - record.keys()
                if missing:
                    raise ValueError(
                        f"manifest line {line_number} is missing {sorted(missing)}"
                    )
                record["latent_start"] = int(record["latent_start"])
                record["num_blocks"] = int(record["num_blocks"])
                if record["latent_start"] < 0 or record["num_blocks"] <= 0:
                    raise ValueError(
                        f"invalid latent range at manifest line {line_number}"
                    )
                self.records.append(record)
        if not self.records:
            raise ValueError("reasoner manifest is empty")

        self.latents = np.load(self.latent_path, mmap_mode="r")
        if self.latents.ndim != 3:
            raise ValueError(
                "latent array must have shape [total_blocks, latent_tokens, latent_dim]"
            )
        self.latent_tokens_per_block = int(self.latents.shape[1])
        self.latent_dim = int(self.latents.shape[2])

        if metadata_path:
            metadata = LatentStoreMetadata.from_json(metadata_path)
            expected = (
                metadata.num_blocks,
                metadata.latent_tokens_per_block,
                metadata.latent_dim,
            )
            if tuple(self.latents.shape) != expected:
                raise ValueError(
                    f"latent metadata expects shape {expected}, got {self.latents.shape}"
                )
            if metadata.num_samples != len(self.records):
                raise ValueError(
                    "latent metadata sample count does not match the manifest"
                )

        final_end = max(
            record["latent_start"] + record["num_blocks"]
            for record in self.records
        )
        if final_end > len(self.latents):
            raise ValueError(
                f"manifest references latent block {final_end - 1}, but the store "
                f"contains only {len(self.latents)} blocks"
            )

    def __len__(self) -> int:
        return len(self.records)

    def _tokenize_question(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_question_length,
            return_attention_mask=False,
        )["input_ids"]
        if not encoded:
            raise ValueError("tokenizer produced an empty question sequence")
        return torch.tensor(encoded, dtype=torch.long)

    def _tokenize_answer(self, text: str) -> torch.Tensor:
        # The question already owns the sequence BOS. Answer supervision begins
        # after <SOA>, so do not inject a second BOS; append EOS explicitly.
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("tokenizer must define eos_token_id")
        budget = max(self.max_answer_length - 1, 1)
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=budget,
            return_attention_mask=False,
        )["input_ids"]
        encoded = list(encoded)[:budget] + [eos_id]
        return torch.tensor(encoded, dtype=torch.long)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        num_blocks = record["num_blocks"]
        if self.max_blocks is not None:
            num_blocks = min(num_blocks, self.max_blocks)
        start = record["latent_start"]
        # Copy the mmap view so workers never expose a non-writable NumPy view.
        latents = torch.tensor(
            np.asarray(self.latents[start : start + num_blocks]),
            dtype=torch.float32,
        )
        return {
            "question_input_ids": self._tokenize_question(str(record["input"])),
            "answer_input_ids": self._tokenize_answer(str(record["answer"])),
            "oracle_latents": latents,
            "num_blocks": num_blocks,
            "source_index": int(record.get("source_index", index)),
        }


class ReasoningDataCollator:
    """Pad text and latent-block dimensions for one reasoner batch."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("cannot collate an empty batch")
        question_ids = pad_sequence(
            [item["question_input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        answer_ids = pad_sequence(
            [item["answer_input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        # Length-derived masks remain correct when pad_token_id == eos_token_id.
        question_mask = torch.zeros_like(question_ids, dtype=torch.bool)
        answer_mask = torch.zeros_like(answer_ids, dtype=torch.bool)
        for batch_index, item in enumerate(batch):
            question_mask[batch_index, : item["question_input_ids"].numel()] = True
            answer_mask[batch_index, : item["answer_input_ids"].numel()] = True

        first_shape = batch[0]["oracle_latents"].shape[1:]
        if len(first_shape) != 2:
            raise ValueError("each latent record must have shape [blocks, tokens, dim]")
        max_blocks = max(item["oracle_latents"].size(0) for item in batch)
        oracle_latents = torch.zeros(
            len(batch), max_blocks, *first_shape, dtype=torch.float32
        )
        block_mask = torch.zeros(len(batch), max_blocks, dtype=torch.bool)
        for batch_index, item in enumerate(batch):
            latent = item["oracle_latents"]
            if latent.shape[1:] != first_shape:
                raise ValueError("all latent blocks in a batch must share shape")
            count = latent.size(0)
            oracle_latents[batch_index, :count] = latent
            block_mask[batch_index, :count] = True

        return {
            "question_input_ids": question_ids,
            "question_attention_mask": question_mask,
            "block_mask": block_mask,
            "answer_input_ids": answer_ids,
            "answer_attention_mask": answer_mask,
            "oracle_latents": oracle_latents,
            "source_index": torch.tensor(
                [item["source_index"] for item in batch], dtype=torch.long
            ),
        }


# Backward-compatible names used by the released training script.
ThoughtDataset = ReasoningLatentDataset
ThoughtDataCollator = ReasoningDataCollator
