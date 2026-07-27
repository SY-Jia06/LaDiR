"""Train the LaDiR variational autoencoder with the paper recipe."""

from dataclasses import dataclass, field
from typing import Optional

import transformers
from peft import LoraConfig
from transformers import HfArgumentParser

from data_vae import ANSWER_PREFIX, load_data
from model_vae import VAE
from training_utils import (
    DataCollatorForDynamicPadding,
    pretrain_tokenize_function,
    train_model,
)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Llama-3.1-8B")
    train: bool = field(default=True)
    beta: float = field(default=1e-5)
    latent_dim: int = field(default=512)
    latent_noise_std: float = field(default=3.0)
    token_substitution_prob: float = field(default=0.3)
    paper_block_mode: bool = field(default=True)

    # Compatibility path only. The paper fine-tunes every encoder parameter.
    use_lora: bool = field(default=False)
    lora_r: int = field(default=512)
    lora_alpha: int = field(default=256)
    lora_dropout: float = field(default=0.05)


@dataclass
class DataArguments:
    train_file: str = field(default="data/vae_train.jsonl")
    val_file: Optional[str] = field(default="data/vae_val.jsonl")
    input_type: str = field(default="cot_only")
    test_size: int = field(default=100)
    answer_prefix: str = field(default=ANSWER_PREFIX)
    require_answer_prefix: bool = field(default=True)
    preprocessing_num_workers: int = field(default=16)
    preprocessing_batch_size: int = field(default=256)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="checkpoints/vae_paper")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)
    fixed_mem_size: int = field(default=4)
    mean_compression_rate: int = field(default=1)
    restore_from: str = field(default="")

    # Kept for backward-compatible tokenization signatures; unused by the
    # paper's pure autoencoding objective.
    min_tokens_for_lm: int = field(default=64)
    leave_tokens_for_lm: int = field(default=8)
    lm_ratio: float = field(default=0.0)
    add_special_token_for_lm: bool = field(default=False)

    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)
    num_train_epochs: float = field(default=2.0)
    max_steps: int = field(default=-1)
    learning_rate: float = field(default=2e-5)
    lr_scheduler_type: str = field(default="cosine")
    warmup_steps: int = field(default=1000)
    weight_decay: float = field(default=0.03)
    max_grad_norm: float = field(default=1.0)
    bf16: bool = field(default=True)
    save_strategy: str = field(default="epoch")
    logging_steps: float = field(default=10)
    eval_strategy: str = field(default="no")
    remove_unused_columns: bool = field(default=False)
    dataloader_num_workers: int = field(default=8)
    report_to: str = field(default="wandb")
    ignore_data_skip: bool = field(default=True)


def main(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> None:
    train_dataset, eval_dataset, inference_examples = load_data(
        test_size=data_args.test_size,
        input_type=data_args.input_type,
        train_file=data_args.train_file,
        val_file=data_args.val_file,
        answer_prefix=data_args.answer_prefix,
        require_answer_prefix=data_args.require_answer_prefix,
    )

    lora_config = None
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    print("Loading VAE encoder and frozen decoder...")
    model = VAE(model_args, training_args, lora_config)
    memory_ids = list(
        range(model.vocab_size, model.vocab_size + model.mem_size)
    )

    tokenize_kwargs = {
        "tokenizer": model.tokenizer,
        "model_max_length": training_args.model_max_length,
        "mem_size": model.mem_size,
        "min_tokens_for_lm": training_args.min_tokens_for_lm,
        "mean_compression_rate": model.mean_compression_rate,
        "add_special_token_for_lm": training_args.add_special_token_for_lm,
        "leave_tokens_for_lm": training_args.leave_tokens_for_lm,
        "ae_token_id": model.ae_token_id,
        "eos_id": model.eos_id,
        "mem": memory_ids,
        "input_type": data_args.input_type,
        "lm_ratio": training_args.lm_ratio,
    }

    map_kwargs = {
        "function": pretrain_tokenize_function,
        "batched": True,
        "batch_size": data_args.preprocessing_batch_size,
        "fn_kwargs": tokenize_kwargs,
        "remove_columns": train_dataset.column_names,
    }
    if data_args.preprocessing_num_workers > 1:
        map_kwargs["num_proc"] = data_args.preprocessing_num_workers

    print("Tokenizing train blocks...")
    tokenized_train = train_dataset.map(**map_kwargs)
    print("Tokenizing eval blocks...")
    eval_map_kwargs = dict(map_kwargs)
    eval_map_kwargs["remove_columns"] = eval_dataset.column_names
    tokenized_eval = eval_dataset.map(**eval_map_kwargs)

    data_collator = DataCollatorForDynamicPadding(model.pad_token_id)
    train_model(
        data_args,
        None,
        model,
        tokenized_train,
        tokenized_eval,
        model_args,
        training_args,
        inference_examples,
        data_collator,
    )


def parse_args():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    return parser.parse_args_into_dataclasses()


if __name__ == "__main__":
    main(*parse_args())
