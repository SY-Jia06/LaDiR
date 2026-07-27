"""Dataset preparation for paper-aligned LaDiR VAE pretraining.

The LaDiR paper separates the final answer with the literal prefix
``The answer is`` and treats every preceding reasoning sentence as an
independent VAE block. This module performs that blockization before training,
rather than asking the VAE to length-chunk a complete solution.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

from datasets import Dataset


ANSWER_PREFIX = "The answer is"
# Mathematical decimals such as ``3.14`` are not split because the period is
# not followed by whitespace. Newlines are always treated as block boundaries.
_SENTENCE_BOUNDARY = re.compile(r"(?:\r?\n)+|(?<=[.!?])\s+")
_PERIOD_PLACEHOLDER = "\ue000"
_COMMON_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "etc.",
    "vs.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "Eq.",
    "Eqs.",
    "Fig.",
    "Figs.",
    "No.",
)


def split_cot_and_answer(
    output: str,
    answer_prefix: str = ANSWER_PREFIX,
    *,
    require_answer_prefix: bool = True,
) -> tuple[str, str]:
    """Split a solution into CoT and final answer using the paper's prefix."""
    output = output.strip()
    prefix_index = output.find(answer_prefix)
    if prefix_index < 0:
        if require_answer_prefix:
            raise ValueError(
                f"solution does not contain the required answer prefix "
                f"{answer_prefix!r}: {output[:160]!r}"
            )
        return output, ""

    cot = output[:prefix_index].strip()
    answer = output[prefix_index:].strip()
    if not cot:
        raise ValueError("solution contains an answer prefix but no CoT text")
    return cot, answer


def _protect_abbreviation_periods(text: str) -> str:
    """Hide periods that should not act as sentence boundaries."""
    protected = text
    for abbreviation in sorted(_COMMON_ABBREVIATIONS, key=len, reverse=True):
        pattern = re.compile(re.escape(abbreviation), flags=re.IGNORECASE)
        protected = pattern.sub(
            lambda match: match.group(0).replace(".", _PERIOD_PLACEHOLDER),
            protected,
        )

    # Preserve common initialisms such as U.S. and U.K. when followed by text.
    protected = re.sub(
        r"\b(?:[A-Za-z]\.){2,}",
        lambda match: match.group(0).replace(".", _PERIOD_PLACEHOLDER),
        protected,
    )
    return protected


def split_sentence_blocks(cot: str) -> list[str]:
    """Split CoT text into one-sentence blocks without external NLP models.

    Boundaries are newlines or terminal punctuation followed by whitespace.
    Common abbreviations and initialisms are protected before splitting, and
    decimals are naturally preserved. This remains a deterministic heuristic:
    domain-specific abbreviations not listed above and punctuation without
    whitespace may require upstream normalization. Run the block audit utility
    on the actual training JSONL before launching the full reproduction.
    """
    normalized = re.sub(r"[ \t]+", " ", cot.strip())
    protected = _protect_abbreviation_periods(normalized)
    blocks = [piece.strip() for piece in _SENTENCE_BOUNDARY.split(protected)]
    return [
        block.replace(_PERIOD_PLACEHOLDER, ".")
        for block in blocks
        if block
    ]


def _iter_block_records(
    jsonl_path: str,
    answer_prefix: str,
    require_answer_prefix: bool,
) -> Iterator[dict]:
    path = Path(jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"VAE dataset not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            sample = json.loads(line)
            question = str(sample["input"]).strip()
            cot, final_answer = split_cot_and_answer(
                str(sample["output"]),
                answer_prefix,
                require_answer_prefix=require_answer_prefix,
            )
            blocks = split_sentence_blocks(cot)
            if not blocks:
                raise ValueError(
                    f"sample {source_index} in {path} produced no CoT blocks"
                )

            for block_index, block in enumerate(blocks):
                yield {
                    "question": question,
                    "chain_of_thought": block,
                    "cot_only": block,
                    # Kept as an explicit debug option. The paper-aligned recipe
                    # trains with input_type=cot_only.
                    "full_format": f"{question}\n{block}",
                    "final_answer": final_answer,
                    "block_index": block_index,
                    "num_blocks": len(blocks),
                    "source_index": source_index,
                }


def _dataset_from_jsonl(
    path: str,
    answer_prefix: str,
    require_answer_prefix: bool,
) -> Dataset:
    return Dataset.from_generator(
        _iter_block_records,
        gen_kwargs={
            "jsonl_path": path,
            "answer_prefix": answer_prefix,
            "require_answer_prefix": require_answer_prefix,
        },
    )


def load_data(
    test_size: int | float = 100,
    input_type: str = "cot_only",
    *,
    train_file: str = "data/vae_train.jsonl",
    val_file: str | None = "data/vae_val.jsonl",
    answer_prefix: str = ANSWER_PREFIX,
    require_answer_prefix: bool = True,
    seed: int = 42,
):
    """Load and blockize VAE data.

    A dedicated validation JSONL is used when present. Otherwise the training
    block dataset is split deterministically using ``test_size``.
    """
    if input_type not in {"cot_only", "full_format"}:
        raise ValueError("input_type must be 'cot_only' or 'full_format'")

    print(f"Loading and sentence-blockizing VAE data from {train_file}...")
    train_dataset = _dataset_from_jsonl(
        train_file,
        answer_prefix,
        require_answer_prefix,
    )

    val_path = Path(val_file) if val_file else None
    if val_path is not None and val_path.is_file():
        eval_dataset = _dataset_from_jsonl(
            str(val_path),
            answer_prefix,
            require_answer_prefix,
        )
    else:
        split_dataset = train_dataset.train_test_split(
            test_size=test_size,
            seed=seed,
            shuffle=True,
        )
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]

    train_dataset = train_dataset.shuffle(seed=seed)
    eval_dataset = eval_dataset.shuffle(seed=seed)

    inference_count = min(200, len(eval_dataset))
    inference_examples = [
        eval_dataset[index][input_type] for index in range(inference_count)
    ]

    print(
        "VAE data loaded: "
        f"{len(train_dataset):,} train blocks, {len(eval_dataset):,} eval blocks"
    )
    return train_dataset, eval_dataset, inference_examples
