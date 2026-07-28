"""Sequence construction and hybrid blockwise attention for LaDiR."""

from __future__ import annotations

from typing import Any, Optional

import torch

from .common import SequenceBatch


class SequenceMixin:
    def _special_embedding(self, token_id: int) -> torch.Tensor:
        ids = torch.tensor([token_id], device=self.model_device, dtype=torch.long)
        return self.text_llama.get_input_embeddings()(ids)[0]

    def _make_attention_mask(
        self,
        valid_lengths: list[int],
        block_spans: list[list[tuple[int, int]]],
        max_length: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(valid_lengths)
        allowed = torch.zeros(
            batch_size, max_length, max_length, dtype=torch.bool, device=device
        )
        valid_mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=device)
        position_ids = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        for batch_index, length in enumerate(valid_lengths):
            valid_mask[batch_index, :length] = True
            position_ids[batch_index, :length] = torch.arange(length, device=device)
            allowed[batch_index, :length, :length] = torch.tril(
                torch.ones(length, length, dtype=torch.bool, device=device)
            )
            for start, end in block_spans[batch_index]:
                allowed[batch_index, start : end + 1, start : end + 1] = True
            # Avoid all-masked padded query rows, which can produce SDPA NaNs.
            if length < max_length:
                allowed[batch_index, length:, 0] = True

        additive = torch.full(
            allowed.shape, torch.finfo(dtype).min, dtype=dtype, device=device
        )
        additive.masked_fill_(allowed, 0.0)
        return additive.unsqueeze(1), position_ids, valid_mask

    def build_blockwise_attention_mask(
        self,
        valid_lengths: list[int],
        block_spans: list[list[tuple[int, int]]],
        max_length: Optional[int] = None,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Public test/debug helper for the hybrid attention contract."""
        if len(valid_lengths) != len(block_spans):
            raise ValueError("valid_lengths and block_spans must have equal length")
        max_length = max_length or max(valid_lengths)
        mask, _, _ = self._make_attention_mask(
            valid_lengths,
            block_spans,
            max_length,
            dtype or self.model_dtype,
            device or self.model_device,
        )
        return mask

    def _build_sequence_batch(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        context_blocks: torch.Tensor,
        context_block_mask: torch.Tensor,
        *,
        current_block: Optional[torch.Tensor] = None,
        current_timesteps: Optional[torch.Tensor] = None,
        answer_input_ids: Optional[torch.Tensor] = None,
        answer_attention_mask: Optional[torch.Tensor] = None,
        drop_condition_mask: Optional[torch.Tensor] = None,
    ) -> SequenceBatch:
        batch_size = question_input_ids.size(0)
        if context_blocks.ndim != 4:
            raise ValueError("context_blocks must have shape [B, K, N, D]")
        if context_blocks.size(0) != batch_size:
            raise ValueError("context block batch size mismatch")
        if context_blocks.shape[2:] != (
            self.latent_tokens_per_block,
            self.latent_dim,
        ):
            raise ValueError("context block shape does not match model config")
        if current_block is not None:
            expected = (
                batch_size,
                self.latent_tokens_per_block,
                self.latent_dim,
            )
            if current_block.shape != expected:
                raise ValueError(f"current_block must have shape {expected}")
            if current_timesteps is None or current_timesteps.shape != (batch_size,):
                raise ValueError("current_timesteps must have shape [batch]")
        if answer_input_ids is not None and answer_attention_mask is None:
            raise ValueError("answer_attention_mask is required with answer_input_ids")

        token_embedding = self.text_llama.get_input_embeddings()
        bot = self._special_embedding(self.token_ids.bot)
        eot = self._special_embedding(self.token_ids.eot)
        soa = self._special_embedding(self.token_ids.soa)

        sample_embeddings: list[torch.Tensor] = []
        block_spans: list[list[tuple[int, int]]] = []
        current_positions: list[torch.Tensor] = []
        eot_batch_indices: list[int] = []
        eot_positions: list[int] = []
        special_targets: list[int] = []
        sample_labels: list[torch.Tensor] = []
        valid_lengths: list[int] = []

        for batch_index in range(batch_size):
            q_ids = question_input_ids[batch_index][
                question_attention_mask[batch_index].bool()
            ]
            if q_ids.numel() == 0:
                raise ValueError("every example needs at least one question token")
            q_emb = token_embedding(q_ids)
            if drop_condition_mask is not None and bool(drop_condition_mask[batch_index]):
                q_emb = q_emb.clone()
                if q_emb.size(0) > 1:
                    q_emb[1:] = self.null_condition.to(q_emb.dtype)
                else:
                    q_emb[0] = self.null_condition.to(q_emb.dtype)

            parts: list[torch.Tensor] = [q_emb]
            spans: list[tuple[int, int]] = []
            labels: list[int] = [-100] * q_emb.size(0)
            cursor = q_emb.size(0)
            valid_indices = torch.nonzero(
                context_block_mask[batch_index], as_tuple=False
            ).flatten().tolist()
            for local_index, block_index in enumerate(valid_indices):
                projected = self.latent_in(
                    context_blocks[batch_index, block_index].to(self.model_dtype)
                )
                start = cursor
                parts.extend([bot.unsqueeze(0), projected, eot.unsqueeze(0)])
                cursor += self.latent_tokens_per_block + 2
                spans.append((start, cursor - 1))
                eot_batch_indices.append(batch_index)
                eot_positions.append(cursor - 1)
                final_oracle_block = (
                    local_index == len(valid_indices) - 1 and current_block is None
                )
                special_targets.append(1 if final_oracle_block else 0)
                labels.extend([-100] * (self.latent_tokens_per_block + 2))

            if current_block is not None:
                projected = self.latent_in(current_block[batch_index].to(self.model_dtype))
                time = self.time_embedding(
                    current_timesteps[batch_index : batch_index + 1], self.model_dtype
                )
                start = cursor
                parts.extend([bot.unsqueeze(0), time, projected, eot.unsqueeze(0)])
                latent_start = cursor + 2
                current_positions.append(
                    torch.arange(
                        latent_start,
                        latent_start + self.latent_tokens_per_block,
                        device=question_input_ids.device,
                    )
                )
                cursor += self.latent_tokens_per_block + 3
                spans.append((start, cursor - 1))
                labels.extend([-100] * (self.latent_tokens_per_block + 3))

            if answer_input_ids is not None:
                parts.append(soa.unsqueeze(0))
                labels.append(-100)
                cursor += 1
                a_ids = answer_input_ids[batch_index][
                    answer_attention_mask[batch_index].bool()
                ]
                if a_ids.numel():
                    parts.append(token_embedding(a_ids))
                    labels.extend(a_ids.tolist())
                    cursor += a_ids.numel()

            sample = torch.cat(parts, dim=0).to(self.model_dtype)
            sample_embeddings.append(sample)
            block_spans.append(spans)
            sample_labels.append(
                torch.tensor(labels, device=sample.device, dtype=torch.long)
            )
            valid_lengths.append(cursor)

        max_length = max(valid_lengths)
        embeddings = torch.zeros(
            batch_size,
            max_length,
            self.hidden_size,
            device=question_input_ids.device,
            dtype=self.model_dtype,
        )
        labels_tensor = (
            torch.full(
                (batch_size, max_length),
                -100,
                device=question_input_ids.device,
                dtype=torch.long,
            )
            if answer_input_ids is not None
            else None
        )
        for batch_index, sample in enumerate(sample_embeddings):
            embeddings[batch_index, : sample.size(0)] = sample
            if labels_tensor is not None:
                labels_tensor[batch_index, : sample.size(0)] = sample_labels[batch_index]

        attention_mask, position_ids, valid_mask = self._make_attention_mask(
            valid_lengths, block_spans, max_length, embeddings.dtype, embeddings.device
        )
        return SequenceBatch(
            embeddings=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            valid_mask=valid_mask,
            current_latent_positions=(
                torch.stack(current_positions) if current_positions else None
            ),
            eot_batch_indices=torch.tensor(
                eot_batch_indices, device=embeddings.device, dtype=torch.long
            ),
            eot_positions=torch.tensor(
                eot_positions, device=embeddings.device, dtype=torch.long
            ),
            special_targets=torch.tensor(
                special_targets, device=embeddings.device, dtype=torch.long
            ),
            labels=labels_tensor,
        )

    def _run_backbone(
        self, sequence: SequenceBatch, *, use_cache: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, Any]:
        outputs = self.text_llama(
            inputs_embeds=sequence.embeddings,
            attention_mask=sequence.attention_mask,
            position_ids=sequence.position_ids,
            output_hidden_states=True,
            return_dict=True,
            use_cache=use_cache,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        hidden = hidden_states[-1] if hidden_states else getattr(
            outputs, "last_hidden_state", None
        )
        if hidden is None:
            raise RuntimeError("causal LM returned no hidden states")
        logits = getattr(outputs, "logits", None)
        if logits is None:
            output_embeddings = self.text_llama.get_output_embeddings()
            if output_embeddings is None:
                raise RuntimeError("causal LM returned no logits and has no LM head")
            logits = output_embeddings(hidden)
        return hidden, logits, outputs

    def _predict_current_block(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        context_blocks: torch.Tensor,
        context_block_mask: torch.Tensor,
        current_block: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        drop_condition_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sequence = self._build_sequence_batch(
            question_input_ids,
            question_attention_mask,
            context_blocks,
            context_block_mask,
            current_block=current_block,
            current_timesteps=timesteps,
            drop_condition_mask=drop_condition_mask,
        )
        hidden, _, _ = self._run_backbone(sequence)
        if sequence.current_latent_positions is None:
            raise RuntimeError("current latent positions were not constructed")
        batch_indices = torch.arange(hidden.size(0), device=hidden.device).unsqueeze(1)
        latent_hidden = hidden[batch_indices, sequence.current_latent_positions]
        return self.latent_out(latent_hidden)
