"""Training and tokenization utilities for the LaDiR VAE stage."""

from __future__ import annotations

import pathlib
from typing import Iterable

import torch
import wandb
from transformers import Trainer


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

    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    if checkpoints:
        print(f"Resuming from {checkpoints[-1]}")
        trainer.train(resume_from_checkpoint=str(checkpoints[-1]))
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

    Several arguments are retained for call-site compatibility with the
    original release.  The paper-aligned recipe is a pure autoencoding
    objective (``lm_ratio=0``) and does not insert the release-only AE token.
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
        raise ValueError(f"expected {mem_size} memory IDs, received {len(mem)}")

    encoder_texts = examples[input_type]
    target_texts = examples["chain_of_thought"]
    encoder_outputs = tokenizer(
        encoder_texts,
        truncation=True,
        max_length=model_max_length,
        padding=False,
        return_attention_mask=False,
    )
    target_outputs = tokenizer(
        target_texts,
        truncation=True,
        max_length=model_max_length,
        padding=False,
        return_attention_mask=False,
    )

    prompt_answer_ids = []
    labels = []
    for target_ids in target_outputs["input_ids"]:
        target_ids = list(target_ids)
        if not target_ids or target_ids[-1] != eos_id:
            target_ids.append(eos_id)

        # Figure 7 in the paper conditions the frozen decoder directly on
        # [latent thought tokens, teacher-forced text embeddings].  There is no
        # additional AE delimiter between the two.
        decoder_ids = list(mem) + target_ids
        decoder_labels = [-100] * mem_size + target_ids
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
        max_length = max(len(sequence) for sequence in sequences)
        if self.pad_to_multiple_of:
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
