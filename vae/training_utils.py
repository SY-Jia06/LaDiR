"""Training and tokenization utilities for the LaDiR VAE stage."""

from __future__ import annotations

import pathlib
from typing import Iterable

import torch
import wandb
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint


def run_inference(model, lines: Iterable[str]) -> list[str]:
    """Reconstruct a small collection of thought blocks for diagnostics."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for line in lines:
            outputs.append(model.run_inference(line))
    return outputs


def train_model(
    args,
    notes,
    model,
    train_dataset,
    eval_dataset,
    model_args,
    training_args,
    lines,
    data_collator=None,
):
    """Train with Hugging Face Trainer and save an FSDP-safe checkpoint."""
    del args, notes, model_args, lines

    output_dir = pathlib.Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    last_checkpoint = get_last_checkpoint(str(output_dir))
    if last_checkpoint is not None:
        print(f"Resuming from {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("Training the paper-aligned VAE from the pretrained backbone.")
        trainer.train()

    trainer.save_model(str(output_dir))
    trainer.save_state()

    if wandb.run is not None:
        wandb.log({"train/finished": 1})

    return trainer


def pretrain_tokenize_function(
    examples,
    tokenizer,
    model_max_length,
    mem_size,
    min_tokens_for_lm,
    mean_compression_rate,
    add_special_token_for_lm,
    leave_tokens_for_lm,
    ae_token_id,
    eos_id,
    mem,
    input_type,
    lm_ratio=0.0,
):
    """Tokenize one sentence per fixed-size latent block.

    The encoder and decoder contexts are limited to ``model_max_length`` after
    accounting for the memory prefix; the decoder budget also includes a
    terminal EOS token. Several arguments are retained for call-site
    compatibility with the original release. The paper-aligned recipe is a pure
    autoencoding objective (``lm_ratio=0``) and does not insert the release-only
    AE token.
    """
    del (
        min_tokens_for_lm,
        mean_compression_rate,
        add_special_token_for_lm,
        leave_tokens_for_lm,
        ae_token_id,
    )
    if lm_ratio != 0:
        raise ValueError("Paper-aligned VAE pretraining requires lm_ratio=0")
    if len(mem) != mem_size:
        raise ValueError(
            f"mem_size={mem_size} requires exactly {mem_size} memory IDs, "
            f"received {len(mem)}"
        )
    if model_max_length <= mem_size:
        raise ValueError(
            "model_max_length must be greater than mem_size so the encoder and "
            "decoder have room for text tokens"
        )

    encoder_texts = examples[input_type]
    target_texts = examples["chain_of_thought"]
    # Both encoder and decoder append the fixed memory prefix, so reserve it
    # inside the configured context limit on both sides.
    text_budget = model_max_length - mem_size
    encoder_outputs = tokenizer(
        encoder_texts,
        truncation=True,
        max_length=text_budget,
        padding=False,
        return_attention_mask=False,
    )

    # EOS is enforced below by replacing the final target token when the budget
    # is already full.
    target_budget = text_budget
    target_outputs = tokenizer(
        target_texts,
        truncation=True,
        max_length=target_budget,
        padding=False,
        return_attention_mask=False,
    )

    prompt_answer_ids = []
    labels = []
    for target_ids in target_outputs["input_ids"]:
        target_ids = list(target_ids[:target_budget])
        if not target_ids:
            target_ids = [eos_id]
        elif target_ids[-1] != eos_id:
            if len(target_ids) == target_budget:
                target_ids[-1] = eos_id
            else:
                target_ids.append(eos_id)

        # Figure 7 in the paper conditions the frozen decoder directly on
        # [latent thought tokens, teacher-forced text embeddings]. There is no
        # additional AE delimiter between the two.
        decoder_ids = list(mem) + target_ids
        decoder_labels = [-100] * mem_size + target_ids
        if len(decoder_ids) > model_max_length:
            raise AssertionError("decoder sequence exceeded model_max_length")
        prompt_answer_ids.append(decoder_ids)
        labels.append(decoder_labels)

    return {
        "input_ids": encoder_outputs["input_ids"],
        "prompt_answer_ids": prompt_answer_ids,
        "labels": labels,
    }


class DataCollatorForDynamicPadding:
    """Right-pad encoder IDs, decoder IDs, and labels independently."""

    def __init__(self, pad_token_id: int, pad_to_multiple_of: int | None = None):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, examples):
        input_ids = [
            torch.tensor(example["input_ids"], dtype=torch.long)
            for example in examples
        ]
        labels = [
            torch.tensor(example["labels"], dtype=torch.long)
            for example in examples
        ]
        prompt_answer_ids = [
            torch.tensor(example["prompt_answer_ids"], dtype=torch.long)
            for example in examples
        ]
        return {
            "input_ids": self.dynamic_padding(
                input_ids,
                fill_value=self.pad_token_id,
            ),
            "prompt_answer_ids": self.dynamic_padding(
                prompt_answer_ids,
                fill_value=self.pad_token_id,
            ),
            "labels": self.dynamic_padding(labels, fill_value=-100),
        }

    def dynamic_padding(self, sequences, fill_value=-100):
        if not sequences:
            raise ValueError("cannot pad an empty batch")
        max_length = max(len(sequence) for sequence in sequences)
        if self.pad_to_multiple_of is not None:
            if self.pad_to_multiple_of <= 0:
                raise ValueError("pad_to_multiple_of must be positive")
            max_length = (
                (max_length - 1) // self.pad_to_multiple_of + 1
            ) * self.pad_to_multiple_of
        padded = torch.full(
            (len(sequences), max_length),
            fill_value,
            dtype=torch.long,
        )
        for index, sequence in enumerate(sequences):
            padded[index, : len(sequence)] = sequence
        return padded
