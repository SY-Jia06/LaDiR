"""Generic reasoning-data loader for paper-aligned VAE training.

The loader intentionally depends on schema rather than dataset identity. It
supports local JSON/JSONL/Parquet files and Hugging Face datasets with any of
these common layouts:

* ``input`` / ``output`` (or similar scalar fields)
* chat ``messages`` containing user and assistant turns
* a scalar prompt plus one or more list-valued ``*_responses`` columns

Each response is reduced to its reasoning trace and split into sentence-level
blocks, matching the one-sentence-per-block VAE training described in LaDiR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset, load_dataset


QUESTION_CANDIDATES = ("prompt", "question", "input", "problem", "instruction")
RESPONSE_CANDIDATES = (
    "chain_of_thought",
    "reasoning",
    "output",
    "response",
    "completion",
    "solution",
    "answer",
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _extract_message(messages: Any, roles: Sequence[str]) -> str:
    if not isinstance(messages, (list, tuple)):
        return ""
    normalized_roles = {role.lower() for role in roles}
    selected: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = _clean_text(message.get("role")).lower()
        if role in normalized_roles:
            content = _clean_text(message.get("content"))
            if content:
                selected.append(content)
    return "\n".join(selected).strip()


def infer_question(record: Dict[str, Any], question_field: Optional[str]) -> str:
    if question_field:
        question = _clean_text(record.get(question_field))
        if question:
            return question

    messages = record.get("messages")
    question = _extract_message(messages, ("user", "human"))
    if question:
        return question

    for field in QUESTION_CANDIDATES:
        question = _clean_text(record.get(field))
        if question:
            return question
    return ""


def _flatten_responses(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        responses: List[str] = []
        for item in value:
            if isinstance(item, dict):
                content = _clean_text(item.get("content"))
                if content:
                    responses.append(content)
            else:
                content = _clean_text(item)
                if content:
                    responses.append(content)
        return responses
    text = _clean_text(value)
    return [text] if text else []


def infer_responses(
    record: Dict[str, Any],
    response_field: Optional[str],
    messages_field: str = "messages",
    response_suffix: str = "_responses",
    all_responses: bool = False,
) -> List[str]:
    if response_field:
        return _flatten_responses(record.get(response_field))

    assistant = _extract_message(record.get(messages_field), ("assistant", "model", "gpt"))
    if assistant:
        return [assistant]

    for field in RESPONSE_CANDIDATES:
        responses = _flatten_responses(record.get(field))
        if responses:
            return responses if all_responses else responses[:1]

    response_fields = sorted(
        field
        for field in record
        if field.endswith(response_suffix) and record.get(field) is not None
    )
    collected: List[str] = []
    for field in response_fields:
        values = _flatten_responses(record.get(field))
        if not values:
            continue
        if not all_responses:
            return values[:1]
        collected.extend(values)
    return collected


def extract_reasoning(response: str, mode: str = "think_or_full") -> str:
    """Extract reasoning without depending on a particular model or dataset."""
    text = _clean_text(response)
    if not text:
        return ""

    if mode not in {"think_or_full", "full", "before_answer"}:
        raise ValueError(f"Unknown reasoning extraction mode: {mode}")

    if mode == "think_or_full":
        complete = re.search(r"<think\b[^>]*>(.*?)</think>", text, flags=re.I | re.S)
        if complete:
            return complete.group(1).strip()
        open_tag = re.search(r"<think\b[^>]*>", text, flags=re.I)
        if open_tag:
            tail = text[open_tag.end() :]
            tail = re.split(r"</think>|<final\b[^>]*>", tail, maxsplit=1, flags=re.I)[0]
            if tail.strip():
                return tail.strip()
        return text

    if mode == "before_answer":
        parts = re.split(
            r"\n\s*(?:the\s+answer\s+is|final\s+answer|answer\s*:)",
            text,
            maxsplit=1,
            flags=re.I,
        )
        return parts[0].strip()

    return text


def _split_long_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    pieces: List[str] = []
    remaining = text.strip()
    while len(remaining) > max_chars:
        cut = remaining.rfind(" ", 0, max_chars)
        if cut < max_chars // 2:
            cut = max_chars
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        pieces.append(remaining)
    return pieces


def split_reasoning_blocks(
    reasoning: str,
    mode: str = "sentence",
    min_chars: int = 2,
    max_chars: int = 4096,
) -> List[str]:
    """Split a trace into localized reasoning blocks.

    Code fences are kept intact. Outside code fences, sentence punctuation,
    paragraph boundaries, and explicit line breaks are treated as candidate
    boundaries. This is deliberately lightweight and language-agnostic.
    """
    text = _clean_text(reasoning).replace("\r\n", "\n").replace("\r", "\n")
    if not text:
        return []
    if mode not in {"sentence", "line", "response"}:
        raise ValueError(f"Unknown block mode: {mode}")
    if mode == "response":
        return _split_long_text(text, max_chars)

    chunks = re.split(r"(```.*?```)", text, flags=re.S)
    blocks: List[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith("```") and chunk.endswith("```"):
            blocks.extend(_split_long_text(chunk, max_chars))
            continue

        if mode == "line":
            candidates = re.split(r"\n+", chunk)
        else:
            candidates = re.split(
                r"(?<=[.!?。！？])\s+|\n\s*\n+|\n+(?=\s*(?:[-*•]|\d+[.)]))|\n+",
                chunk,
            )

        for candidate in candidates:
            candidate = re.sub(r"[ \t]+", " ", candidate).strip()
            if len(candidate) < min_chars:
                if candidate and blocks:
                    blocks[-1] = f"{blocks[-1]} {candidate}".strip()
                continue
            blocks.extend(_split_long_text(candidate, max_chars))

    return [block for block in blocks if len(block) >= min_chars]


def _normalize_batch(
    batch: Dict[str, List[Any]],
    question_field: Optional[str],
    response_field: Optional[str],
    messages_field: str,
    response_suffix: str,
    all_responses: bool,
    reasoning_extraction: str,
    block_mode: str,
    min_block_chars: int,
    max_block_chars: int,
) -> Dict[str, List[Any]]:
    output: Dict[str, List[Any]] = {
        "question": [],
        "reasoning_block": [],
        "block_index": [],
        "num_blocks": [],
    }
    if not batch:
        return output

    row_count = len(next(iter(batch.values())))
    for index in range(row_count):
        record = {field: values[index] for field, values in batch.items()}
        question = infer_question(record, question_field)
        responses = infer_responses(
            record,
            response_field=response_field,
            messages_field=messages_field,
            response_suffix=response_suffix,
            all_responses=all_responses,
        )
        for response in responses:
            reasoning = extract_reasoning(response, reasoning_extraction)
            blocks = split_reasoning_blocks(
                reasoning,
                mode=block_mode,
                min_chars=min_block_chars,
                max_chars=max_block_chars,
            )
            for block_index, block in enumerate(blocks):
                output["question"].append(question)
                output["reasoning_block"].append(block)
                output["block_index"].append(block_index)
                output["num_blocks"].append(len(blocks))
    return output


def _dataset_loader_name(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    raise ValueError(f"Unsupported data file type: {suffix}")


def _load_source(
    dataset_name: Optional[str],
    dataset_config: Optional[str],
    split: str,
    data_file: Optional[str],
    streaming: bool,
):
    if bool(dataset_name) == bool(data_file):
        raise ValueError("Provide exactly one of dataset_name or data_file")
    if dataset_name:
        return load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=streaming,
        )
    return load_dataset(
        _dataset_loader_name(data_file),
        data_files=data_file,
        split="train",
        streaming=streaming,
    )


def _materialize_stream(dataset, max_source_samples: Optional[int]) -> Dataset:
    if max_source_samples is None:
        raise ValueError(
            "Streaming mode requires max_source_samples so block expansion remains bounded"
        )
    return Dataset.from_list(list(dataset.take(max_source_samples)))


def _select_limit(dataset: Dataset, max_source_samples: Optional[int]) -> Dataset:
    if max_source_samples is None:
        return dataset
    return dataset.select(range(min(max_source_samples, len(dataset))))


def _tokenize_batch(batch, tokenizer, max_block_tokens: int):
    tokenized = tokenizer(
        batch["reasoning_block"],
        truncation=True,
        max_length=max_block_tokens,
        add_special_tokens=True,
        padding=False,
    )
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }


def prepare_block_dataset(
    source,
    tokenizer,
    *,
    question_field: Optional[str] = None,
    response_field: Optional[str] = None,
    messages_field: str = "messages",
    response_suffix: str = "_responses",
    all_responses: bool = False,
    reasoning_extraction: str = "think_or_full",
    block_mode: str = "sentence",
    min_block_chars: int = 2,
    max_block_chars: int = 4096,
    max_block_tokens: int = 256,
    num_proc: int = 1,
) -> Dataset:
    source_columns = source.column_names
    blocks = source.map(
        _normalize_batch,
        batched=True,
        batch_size=64,
        remove_columns=source_columns,
        fn_kwargs={
            "question_field": question_field,
            "response_field": response_field,
            "messages_field": messages_field,
            "response_suffix": response_suffix,
            "all_responses": all_responses,
            "reasoning_extraction": reasoning_extraction,
            "block_mode": block_mode,
            "min_block_chars": min_block_chars,
            "max_block_chars": max_block_chars,
        },
        num_proc=max(1, num_proc),
        desc="Splitting reasoning traces into blocks",
    )
    blocks = blocks.filter(
        lambda example: bool(example["reasoning_block"].strip()),
        num_proc=max(1, num_proc),
        desc="Removing empty reasoning blocks",
    )
    return blocks.map(
        _tokenize_batch,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer, "max_block_tokens": max_block_tokens},
        num_proc=max(1, num_proc),
        desc="Tokenizing reasoning blocks",
    )


def load_data(
    tokenizer,
    *,
    dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None,
    train_file: Optional[str] = None,
    validation_file: Optional[str] = None,
    split: str = "train",
    streaming: bool = False,
    max_source_samples: Optional[int] = None,
    validation_split_percentage: float = 1.0,
    seed: int = 42,
    question_field: Optional[str] = None,
    response_field: Optional[str] = None,
    messages_field: str = "messages",
    response_suffix: str = "_responses",
    all_responses: bool = False,
    reasoning_extraction: str = "think_or_full",
    block_mode: str = "sentence",
    min_block_chars: int = 2,
    max_block_chars: int = 4096,
    max_block_tokens: int = 256,
    num_proc: int = 1,
):
    train_source = _load_source(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        data_file=train_file,
        streaming=streaming,
    )
    if streaming:
        train_source = _materialize_stream(train_source, max_source_samples)
    else:
        train_source = _select_limit(train_source, max_source_samples)

    if validation_file:
        validation_source = _load_source(
            dataset_name=None,
            dataset_config=None,
            split="train",
            data_file=validation_file,
            streaming=streaming,
        )
        if streaming:
            validation_source = _materialize_stream(validation_source, max_source_samples)
        else:
            validation_source = _select_limit(validation_source, max_source_samples)
    else:
        percentage = float(validation_split_percentage)
        if not 0 < percentage < 100:
            raise ValueError("validation_split_percentage must be between 0 and 100")
        split_dataset = train_source.train_test_split(
            test_size=percentage / 100.0,
            seed=seed,
        )
        train_source = split_dataset["train"]
        validation_source = split_dataset["test"]

    common_kwargs = {
        "tokenizer": tokenizer,
        "question_field": question_field,
        "response_field": response_field,
        "messages_field": messages_field,
        "response_suffix": response_suffix,
        "all_responses": all_responses,
        "reasoning_extraction": reasoning_extraction,
        "block_mode": block_mode,
        "min_block_chars": min_block_chars,
        "max_block_chars": max_block_chars,
        "max_block_tokens": max_block_tokens,
        "num_proc": num_proc,
    }
    train_dataset = prepare_block_dataset(train_source, **common_kwargs)
    validation_dataset = prepare_block_dataset(validation_source, **common_kwargs)
    preview = validation_dataset["reasoning_block"][: min(20, len(validation_dataset))]
    return train_dataset, validation_dataset, preview


@dataclass
class VAEDataCollator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        model_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        return self.tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
