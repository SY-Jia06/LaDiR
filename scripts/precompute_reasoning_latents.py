#!/usr/bin/env python3
"""Precompute frozen LaDiR VAE blocks for reasoner training.

Outputs ``manifest.jsonl``, ``latents.npy``, and ``metadata.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
import sys

import numpy as np
import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vae.data_vae import ANSWER_PREFIX, split_cot_and_answer, split_sentence_blocks
from vae.model_vae import VAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw input/output JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument(
        "--model-name-or-path", default="meta-llama/Llama-3.1-8B"
    )
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--latent-tokens-per-block", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--latent-mode", choices=["sample", "mean"], default="sample"
    )
    parser.add_argument("--answer-prefix", default=ANSWER_PREFIX)
    parser.add_argument("--allow-missing-prefix", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_samples(
    path: Path,
    answer_prefix: str,
    require_answer_prefix: bool,
) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            sample = json.loads(line)
            question = str(sample["input"]).strip()
            cot, answer = split_cot_and_answer(
                str(sample["output"]),
                answer_prefix,
                require_answer_prefix=require_answer_prefix,
            )
            blocks = split_sentence_blocks(cot)
            if not blocks:
                raise ValueError(
                    f"sample {source_index} produced no reasoning blocks"
                )
            yield {
                "source_index": source_index,
                "input": question,
                "answer": answer,
                "blocks": blocks,
            }


def count_records(
    path: Path,
    answer_prefix: str,
    require_answer_prefix: bool,
) -> tuple[int, int]:
    num_samples = 0
    num_blocks = 0
    for sample in iter_samples(path, answer_prefix, require_answer_prefix):
        num_samples += 1
        num_blocks += len(sample["blocks"])
    if num_samples == 0:
        raise ValueError("input JSONL contains no samples")
    return num_samples, num_blocks


def load_vae(args: argparse.Namespace) -> VAE:
    model_args = SimpleNamespace(
        model_name_or_path=args.model_name_or_path,
        paper_block_mode=True,
        latent_dim=args.latent_dim,
        beta=1e-5,
        latent_noise_std=0.0,
        token_substitution_prob=0.0,
        use_lora=False,
    )
    training_args = SimpleNamespace(
        fixed_mem_size=args.latent_tokens_per_block,
        mean_compression_rate=1,
        restore_from="",
    )
    vae = VAE(model_args, training_args)
    incompatible = vae.load_state_dict(
        load_file(args.vae_checkpoint), strict=False
    )
    missing_non_decoder = [
        key for key in incompatible.missing_keys if not key.startswith("decoder.")
    ]
    if missing_non_decoder or incompatible.unexpected_keys:
        raise RuntimeError(
            "VAE checkpoint mismatch: "
            f"missing_non_decoder={missing_non_decoder[:20]}, "
            f"unexpected={incompatible.unexpected_keys[:20]}"
        )

    # Encoding does not use the frozen decoder; release that LLM copy before
    # moving the encoder to the GPU.
    vae.decoder = torch.nn.Identity()
    vae.to(device=torch.device(args.device), dtype=torch.bfloat16)
    vae.eval()
    return vae


def encode_batch(
    vae: VAE,
    texts: list[str],
    latent_mode: str,
    max_length: int,
) -> np.ndarray:
    input_ids = vae._tokenize_batch(texts, max_length=max_length)
    with torch.inference_mode():
        latents = vae._compress(input_ids, return_sample=latent_mode)
    return latents.float().cpu().numpy().astype(np.float16, copy=False)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_length <= 0:
        raise ValueError("batch size and max length must be positive")
    input_path = Path(args.input)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    require_answer_prefix = not args.allow_missing_prefix
    num_samples, num_blocks = count_records(
        input_path, args.answer_prefix, require_answer_prefix
    )
    print(f"samples={num_samples:,} blocks={num_blocks:,}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    vae = load_vae(args)

    latent_path = output_dir / "latents.npy"
    manifest_path = output_dir / "manifest.jsonl"
    metadata_path = output_dir / "metadata.json"
    latent_store = np.lib.format.open_memmap(
        latent_path,
        mode="w+",
        dtype=np.float16,
        shape=(num_blocks, args.latent_tokens_per_block, args.latent_dim),
    )

    block_cursor = 0
    write_cursor = 0
    text_buffer: list[str] = []

    def flush() -> None:
        nonlocal write_cursor, text_buffer
        if not text_buffer:
            return
        encoded = encode_batch(
            vae, text_buffer, args.latent_mode, args.max_length
        )
        expected = (
            len(text_buffer),
            args.latent_tokens_per_block,
            args.latent_dim,
        )
        if encoded.shape != expected:
            raise RuntimeError(
                f"VAE produced shape {encoded.shape}; expected {expected}"
            )
        latent_store[write_cursor : write_cursor + len(text_buffer)] = encoded
        write_cursor += len(text_buffer)
        text_buffer = []
        if write_cursor % max(args.batch_size * 100, 1) == 0:
            print(f"encoded {write_cursor:,}/{num_blocks:,} blocks")

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for sample in iter_samples(
            input_path, args.answer_prefix, require_answer_prefix
        ):
            blocks = sample.pop("blocks")
            manifest.write(
                json.dumps(
                    {
                        **sample,
                        "latent_start": block_cursor,
                        "num_blocks": len(blocks),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            block_cursor += len(blocks)
            for block in blocks:
                text_buffer.append(block)
                if len(text_buffer) >= args.batch_size:
                    flush()
        flush()

    latent_store.flush()
    if block_cursor != num_blocks or write_cursor != num_blocks:
        raise RuntimeError(
            f"block accounting mismatch: counted={num_blocks}, "
            f"manifest={block_cursor}, encoded={write_cursor}"
        )
    metadata = {
        "latent_tokens_per_block": args.latent_tokens_per_block,
        "latent_dim": args.latent_dim,
        "dtype": "float16",
        "num_blocks": num_blocks,
        "num_samples": num_samples,
        "latent_mode": args.latent_mode,
        "model_name_or_path": args.model_name_or_path,
        "vae_checkpoint": str(Path(args.vae_checkpoint).resolve()),
        "answer_prefix": args.answer_prefix,
        "seed": args.seed,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"wrote {manifest_path}")
    print(f"wrote {latent_path}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
