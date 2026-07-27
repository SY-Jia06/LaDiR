#!/usr/bin/env python3
"""Audit LaDiR sentence blockization before an expensive VAE training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import datasets  # noqa: F401
except ModuleNotFoundError:
    import types

    datasets_stub = types.ModuleType("datasets")
    datasets_stub.Dataset = object
    sys.modules["datasets"] = datasets_stub

from vae.data_vae import ANSWER_PREFIX, split_cot_and_answer, split_sentence_blocks


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/vae_train.jsonl")
    parser.add_argument("--answer-prefix", default=ANSWER_PREFIX)
    parser.add_argument("--allow-missing-prefix", action="store_true")
    parser.add_argument("--show", type=int, default=10)
    parser.add_argument("--short-words", type=int, default=3)
    parser.add_argument("--long-words", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    if not path.is_file():
        raise FileNotFoundError(path)

    block_counts: list[int] = []
    word_counts: list[int] = []
    suspicious: list[tuple[int, int, str]] = []
    missing_prefix = 0
    samples = 0

    with path.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            samples += 1
            sample = json.loads(line)
            output = str(sample["output"])
            if args.answer_prefix not in output:
                missing_prefix += 1
            cot, _ = split_cot_and_answer(
                output,
                args.answer_prefix,
                require_answer_prefix=not args.allow_missing_prefix,
            )

            blocks = split_sentence_blocks(cot)
            block_counts.append(len(blocks))
            for block_index, block in enumerate(blocks):
                words = len(block.split())
                word_counts.append(words)
                if words <= args.short_words or words >= args.long_words:
                    suspicious.append((source_index, block_index, block))

    print(f"samples: {samples:,}")
    print(f"blocks: {len(word_counts):,}")
    print(f"missing answer prefix: {missing_prefix:,}")
    if block_counts:
        print(f"blocks/sample mean: {mean(block_counts):.2f}")
        print(f"blocks/sample median: {median(block_counts):.1f}")
        print(f"blocks/sample p95: {percentile(block_counts, 0.95)}")
    if word_counts:
        print(f"words/block mean: {mean(word_counts):.2f}")
        print(f"words/block median: {median(word_counts):.1f}")
        print(f"words/block p95: {percentile(word_counts, 0.95)}")
        print(f"words/block max: {max(word_counts)}")

    print(
        f"suspicious blocks (<= {args.short_words} or >= {args.long_words} words): "
        f"{len(suspicious):,}"
    )
    for source_index, block_index, block in suspicious[: args.show]:
        print(f"[{source_index}:{block_index}] {block}")


if __name__ == "__main__":
    main()
