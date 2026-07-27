"""Train the paper-aligned LaDiR variational autoencoder."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import sys
from typing import Optional

import transformers
from peft import LoraConfig
from transformers import HfArgumentParser, Trainer, set_seed

from data_vae import VAEDataCollator, load_data
from model_vae import VAE


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="meta-llama/Llama-3.1-8B",
        metadata={"help": "Encoder/decoder initialization checkpoint."},
    )
    latent_dim: int = field(
        default=512,
        metadata={"help": "Dimension of each continuous thought token."},
    )
    num_latent_tokens: int = field(
        default=4,
        metadata={"help": "Number of latent thought tokens per reasoning block."},
    )
    beta: float = field(default=1e-5, metadata={"help": "KL-loss weight."})
    token_substitution_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of replacing each encoder text token."},
    )
    latent_noise_std: float = field(
        default=3.0,
        metadata={"help": "Std. dev. of Gaussian latent robustness noise."},
    )
    encoder_tuning: str = field(
        default="full",
        metadata={"help": "Use 'full' for the paper setup or 'lora' for a cheaper run."},
    )
    lora_r: int = 128
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Transformers attention implementation."},
    )
    trust_remote_code: bool = False


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Hugging Face dataset identifier."},
    )
    dataset_config: Optional[str] = None
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "Local JSON, JSONL, or Parquet training file."},
    )
    validation_file: Optional[str] = None
    split: str = "train"
    streaming: bool = False
    max_source_samples: Optional[int] = None
    validation_split_percentage: float = 1.0

    question_field: Optional[str] = field(
        default=None,
        metadata={"help": "Explicit question field; omit to infer it."},
    )
    response_field: Optional[str] = field(
        default=None,
        metadata={"help": "Explicit response field; omit to infer it."},
    )
    messages_field: str = "messages"
    response_suffix: str = "_responses"
    all_responses: bool = field(
        default=False,
        metadata={"help": "Expand every response in list-valued response columns."},
    )
    reasoning_extraction: str = field(
        default="think_or_full",
        metadata={"help": "One of: think_or_full, full, before_answer."},
    )
    block_mode: str = field(
        default="sentence",
        metadata={"help": "One of: sentence, line, response."},
    )
    min_block_chars: int = 2
    max_block_chars: int = 4096
    max_block_tokens: int = 256
    preprocessing_num_workers: int = 1


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = "checkpoints/vae-paper"
    model_max_length: int = 512
    num_train_epochs: float = 2.0
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    bf16: bool = True
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 1000
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    remove_unused_columns: bool = True
    report_to: str = "wandb"
    ddp_find_unused_parameters: bool = False
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Optional Trainer checkpoint directory."},
    )


def parse_args():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    return parser.parse_args_into_dataclasses()


def main() -> None:
    model_args, data_args, training_args = parse_args()
    set_seed(training_args.seed)

    if not data_args.dataset_name and not data_args.train_file:
        raise ValueError("Set either --dataset_name or --train_file")
    if data_args.dataset_name and data_args.train_file:
        raise ValueError("Use only one of --dataset_name and --train_file")

    lora_config = None
    if model_args.encoder_tuning == "lora":
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

    model = VAE(model_args, training_args, lora_config=lora_config)

    train_dataset, eval_dataset, preview = load_data(
        tokenizer=model.tokenizer,
        dataset_name=data_args.dataset_name,
        dataset_config=data_args.dataset_config,
        train_file=data_args.train_file,
        validation_file=data_args.validation_file,
        split=data_args.split,
        streaming=data_args.streaming,
        max_source_samples=data_args.max_source_samples,
        validation_split_percentage=data_args.validation_split_percentage,
        seed=training_args.data_seed or training_args.seed,
        question_field=data_args.question_field,
        response_field=data_args.response_field,
        messages_field=data_args.messages_field,
        response_suffix=data_args.response_suffix,
        all_responses=data_args.all_responses,
        reasoning_extraction=data_args.reasoning_extraction,
        block_mode=data_args.block_mode,
        min_block_chars=data_args.min_block_chars,
        max_block_chars=data_args.max_block_chars,
        max_block_tokens=data_args.max_block_tokens,
        num_proc=data_args.preprocessing_num_workers,
    )

    if training_args.local_rank in {-1, 0}:
        print(f"Prepared {len(train_dataset):,} training blocks")
        print(f"Prepared {len(eval_dataset):,} validation blocks")
        for index, block in enumerate(preview[:5]):
            print(f"preview block {index}: {block}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=VAEDataCollator(model.tokenizer),
        processing_class=model.tokenizer,
    )

    if training_args.do_train:
        train_result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        trainer.save_model()
        trainer.save_state()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
