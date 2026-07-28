#!/usr/bin/env python3
"""Train the LaDiR reasoning model (Stage 1 or Stage 2)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from omegaconf import OmegaConf
from safetensors.torch import load_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from dataset import ReasoningDataCollator, ReasoningLatentDataset
from model import LaDiRReasoner, configure_reasoner_tokenizer


def parse_cli() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/reasoner_stage1.yaml",
        help="OmegaConf YAML file",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="Optional Stage-1/Stage-2 model.safetensors checkpoint",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported torch dtype {name!r}")
    return mapping[name]


def load_config(path: str, overrides: list[str]) -> Any:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def make_dataset(
    tokenizer: Any, cfg: Any, split: str
) -> Optional[ReasoningLatentDataset]:
    section = cfg.data
    manifest = section.get(f"{split}_manifest")
    latents = section.get(f"{split}_latents")
    if not manifest or not latents:
        return None
    metadata = section.get(f"{split}_metadata")
    return ReasoningLatentDataset(
        tokenizer,
        str(manifest),
        str(latents),
        metadata_path=str(metadata) if metadata else None,
        max_question_length=int(section.max_question_length),
        max_answer_length=int(section.max_answer_length),
        max_blocks=(
            int(section.max_blocks)
            if section.get("max_blocks") is not None
            else None
        ),
    )


def load_initial_checkpoint(model: torch.nn.Module, checkpoint: str) -> None:
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    incompatible = model.load_state_dict(load_file(str(path)), strict=False)
    if incompatible.missing_keys:
        print(f"initial checkpoint missing keys: {incompatible.missing_keys[:30]}")
    if incompatible.unexpected_keys:
        print(
            f"initial checkpoint unexpected keys: "
            f"{incompatible.unexpected_keys[:30]}"
        )


def main() -> None:
    args, overrides = parse_cli()
    cfg = load_config(args.config, overrides)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    model_name = str(cfg.model.model_name_or_path)
    dtype = resolve_dtype(str(cfg.model.get("torch_dtype", "bfloat16")))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=bool(cfg.model.get("use_fast_tokenizer", False)),
    )
    causal_lm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        # Llama's eager path accepts the explicit 4D hybrid additive mask.
        attn_implementation=str(cfg.model.get("attn_implementation", "eager")),
    )
    token_ids = configure_reasoner_tokenizer(tokenizer, causal_lm)
    model = LaDiRReasoner(causal_lm, tokenizer, cfg, token_ids=token_ids)

    init_from = args.init_from or cfg.model.get("init_from")
    if init_from:
        print(f"loading initial reasoner checkpoint from {init_from}")
        load_initial_checkpoint(model, str(init_from))

    train_dataset = make_dataset(tokenizer, cfg, "train")
    if train_dataset is None:
        raise ValueError("data.train_manifest and data.train_latents are required")
    eval_dataset = make_dataset(tokenizer, cfg, "validation")
    if (
        train_dataset.latent_dim != model.latent_dim
        or train_dataset.latent_tokens_per_block
        != model.latent_tokens_per_block
    ):
        raise ValueError(
            "precomputed VAE shape does not match the reasoner config: "
            f"dataset=({train_dataset.latent_tokens_per_block}, "
            f"{train_dataset.latent_dim}), model=("
            f"{model.latent_tokens_per_block}, {model.latent_dim})"
        )

    training_payload = OmegaConf.to_container(cfg.trainer, resolve=True)
    if not isinstance(training_payload, dict):
        raise TypeError("trainer config must be a mapping")
    training_args = TrainingArguments(**training_payload)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=ReasoningDataCollator(tokenizer.pad_token_id),
    )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"trainable parameters: {trainable:,}/{total:,}")
    print(
        f"stage={model.stage} objective={model.objective} "
        f"train_samples={len(train_dataset):,}"
    )

    output_dir = Path(training_args.output_dir)
    has_checkpoint = any(output_dir.glob("checkpoint-*"))
    resume = bool(cfg.get("allow_resume", True) and has_checkpoint)
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_model()
    trainer.save_state()
    if trainer.is_world_process_zero():
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "resolved_config.yaml").open(
            "w", encoding="utf-8"
        ) as handle:
            handle.write(OmegaConf.to_yaml(cfg, resolve=True))
        with (output_dir / "train_metrics.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(result.metrics, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
