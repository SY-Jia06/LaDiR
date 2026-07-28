#!/usr/bin/env python3
"""Evaluate LaDiR or an autoregressive baseline with one result schema.

The generic built-in checkers are intentionally modest.  Exact-answer and
last-number matching are useful smoke metrics; benchmark submissions should
pass a task-specific ``module:function`` checker and keep the raw generations
written by this script for auditability.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
import importlib
import json
from pathlib import Path
import re
from typing import Any, Optional

from omegaconf import OmegaConf
from safetensors.torch import load_file
import torch
from model import LaDiRReasoner, configure_reasoner_tokenizer


Checker = Callable[[str, str, dict[str, Any]], bool]
_NUMBER = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/[+-]?\d+)?")


def normalize_exact(text: str) -> str:
    return " ".join(text.strip().lower().split())


def last_number(text: str) -> Optional[str]:
    matches = _NUMBER.findall(text)
    return matches[-1].replace(",", "") if matches else None


def exact_checker(prediction: str, reference: str, record: dict[str, Any]) -> bool:
    del record
    return normalize_exact(prediction) == normalize_exact(reference)


def last_number_checker(
    prediction: str, reference: str, record: dict[str, Any]
) -> bool:
    del record
    predicted = last_number(prediction)
    expected = last_number(reference)
    return predicted is not None and predicted == expected


def load_checker(spec: str) -> Checker:
    builtins: dict[str, Checker] = {
        "exact": exact_checker,
        "last_number": last_number_checker,
    }
    if spec in builtins:
        return builtins[spec]
    if ":" not in spec:
        raise ValueError(
            "checker must be 'exact', 'last_number', or 'module:function'"
        )
    module_name, function_name = spec.split(":", 1)
    checker = getattr(importlib.import_module(module_name), function_name)
    if not callable(checker):
        raise TypeError(f"checker {spec!r} is not callable")
    return checker


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            if "input" not in record:
                raise ValueError(f"record {source_index} has no 'input' field")
            if "answer" in record:
                reference = str(record["answer"])
            elif "output" in record:
                from vae.data_vae import split_cot_and_answer

                _, reference = split_cot_and_answer(str(record["output"]))
            else:
                raise ValueError(
                    f"record {source_index} needs 'answer' or 'output'"
                )
            yield {
                **record,
                "source_index": int(record.get("source_index", source_index)),
                "reference": reference,
            }


def parse_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Evaluation JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--sample-batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--checker", default="last_number")
    parser.add_argument("--pass-at-k", default="1,10,25,50,100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--strict-checkpoint", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="method", required=True)

    ladir = subparsers.add_parser("ladir")
    parse_common(ladir)
    ladir.add_argument("--config", default="configs/reasoner_stage2.yaml")
    ladir.add_argument("--min-blocks", type=int, default=1)
    ladir.add_argument("--max-blocks", type=int, default=16)
    ladir.add_argument("--num-steps", type=int, default=50)
    ladir.add_argument("--guidance-scale", type=float, default=4.0)
    ladir.add_argument("--initial-noise-scale", type=float, default=2.0)
    ladir.add_argument("--diversity-scale", type=float, default=0.8)

    ar = subparsers.add_parser("ar")
    parse_common(ar)
    ar.add_argument("--prompt-template", default="{input}\n")
    return parser.parse_args()


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype {name!r}")
    dtype = mapping[name]
    if device.type == "cpu" and dtype != torch.float32:
        return torch.float32
    return dtype


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is unavailable; evaluating on CPU")
        return torch.device("cpu")
    return torch.device(name)


def load_state(
    model: torch.nn.Module, checkpoint: Optional[str], strict: bool
) -> None:
    if not checkpoint:
        return
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    incompatible = model.load_state_dict(load_file(str(path)), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        message = (
            f"checkpoint mismatch: missing={incompatible.missing_keys[:30]}, "
            f"unexpected={incompatible.unexpected_keys[:30]}"
        )
        if strict:
            raise RuntimeError(message)
        print(message)


def load_ladir(args: argparse.Namespace, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = OmegaConf.load(args.config)
    model_name = args.model_name_or_path or str(cfg.model.model_name_or_path)
    dtype = resolve_dtype(args.torch_dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    backbone = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    token_ids = configure_reasoner_tokenizer(tokenizer, backbone)
    model = LaDiRReasoner(backbone, tokenizer, cfg, token_ids=token_ids)
    load_state(model, args.checkpoint, args.strict_checkpoint)
    model.to(device).eval()
    return model, tokenizer


def load_ar(args: argparse.Namespace, device: torch.device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_source = (
        args.checkpoint
        if args.checkpoint and Path(args.checkpoint).is_dir()
        else args.model_name_or_path
    )
    if not model_source:
        raise ValueError(
            "AR evaluation requires --model-name-or-path or a checkpoint directory"
        )
    tokenizer_source = args.model_name_or_path or model_source
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = resolve_dtype(args.torch_dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        model_source if Path(str(model_source)).is_dir() else args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="eager",
    )
    if args.checkpoint and Path(args.checkpoint).is_file():
        load_state(model, args.checkpoint, args.strict_checkpoint)
    model.to(device).eval()
    return model, tokenizer


def chunk_count(total: int, chunk_size: int) -> Iterator[int]:
    remaining = total
    while remaining:
        current = min(remaining, chunk_size)
        yield current
        remaining -= current


def generate_ladir(
    model: LaDiRReasoner,
    tokenizer: Any,
    question: str,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[list[str], list[int]]:
    encoded = tokenizer(
        question,
        add_special_tokens=True,
        return_tensors="pt",
    )
    base_ids = encoded["input_ids"].to(device)
    base_mask = encoded["attention_mask"].to(device)
    texts: list[str] = []
    block_counts: list[int] = []
    for count in chunk_count(args.num_samples, args.sample_batch_size):
        input_ids = base_ids.expand(count, -1)
        attention_mask = base_mask.expand(count, -1)
        output = model.generate(
            input_ids,
            attention_mask,
            min_blocks=args.min_blocks,
            max_blocks=args.max_blocks,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            initial_noise_scale=args.initial_noise_scale,
            diversity_scale=args.diversity_scale,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            generator=generator,
        )
        texts.extend(output["texts"])
        block_counts.extend(output["block_mask"].sum(dim=1).cpu().tolist())
    return texts, block_counts


def generate_ar(
    model: torch.nn.Module,
    tokenizer: Any,
    question: str,
    args: argparse.Namespace,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[list[str], list[int]]:
    del generator
    prompt = args.prompt_template.format(input=question)
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_length = encoded["input_ids"].size(1)
    texts: list[str] = []
    for count in chunk_count(args.num_samples, args.sample_batch_size):
        input_ids = encoded["input_ids"].expand(count, -1)
        attention_mask = encoded["attention_mask"].expand(count, -1)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-6),
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        texts.extend(
            tokenizer.batch_decode(
                output[:, prompt_length:], skip_special_tokens=True
            )
        )
    return texts, [0] * len(texts)


def parse_ks(text: str, num_samples: int) -> list[int]:
    values = sorted({int(value) for value in text.split(",") if value.strip()})
    if not values or values[0] <= 0:
        raise ValueError("pass-at-k values must be positive")
    return [value for value in values if value <= num_samples] or [num_samples]


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.sample_batch_size <= 0:
        raise ValueError("sample counts must be positive")
    device = resolve_device(args.device)
    checker = load_checker(args.checker)
    ks = parse_ks(args.pass_at_k, args.num_samples)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    if args.method == "ladir":
        model, tokenizer = load_ladir(args, device)
        generate_one = generate_ladir
    else:
        model, tokenizer = load_ar(args, device)
        generate_one = generate_ar

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_path = output_dir / "generations.jsonl"
    totals = {k: 0 for k in ks}
    num_examples = 0
    unique_counts: list[int] = []
    correct_unique_counts: list[int] = []

    with generation_path.open("w", encoding="utf-8") as handle:
        for record in iter_records(Path(args.input)):
            if args.max_examples is not None and num_examples >= args.max_examples:
                break
            predictions, block_counts = generate_one(
                model,
                tokenizer,
                str(record["input"]),
                args,
                device,
                generator,
            )
            correct = [
                checker(prediction, record["reference"], record)
                for prediction in predictions
            ]
            for k in ks:
                totals[k] += int(any(correct[:k]))
            normalized = [normalize_exact(text) for text in predictions]
            unique_counts.append(len(set(normalized)))
            correct_unique_counts.append(
                len({text for text, hit in zip(normalized, correct) if hit})
            )
            handle.write(
                json.dumps(
                    {
                        "source_index": record["source_index"],
                        "input": record["input"],
                        "reference": record["reference"],
                        "predictions": predictions,
                        "correct": correct,
                        "block_counts": block_counts,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            num_examples += 1
            print(
                f"evaluated {num_examples}: pass@1={totals.get(1, 0) / num_examples:.4f}"
            )

    if num_examples == 0:
        raise ValueError("evaluation input produced no examples")
    summary = {
        "method": args.method,
        "num_examples": num_examples,
        "num_samples": args.num_samples,
        "checker": args.checker,
        "pass_at_k": {str(k): totals[k] / num_examples for k in ks},
        "mean_unique_outputs": sum(unique_counts) / num_examples,
        "mean_unique_correct_outputs": sum(correct_unique_counts) / num_examples,
        "arguments": vars(args),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
