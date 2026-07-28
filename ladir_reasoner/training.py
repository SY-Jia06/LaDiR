"""Stage-1 and Stage-2 objectives for the LaDiR reasoner."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F


class TrainingMixin:
    def _encode_online_oracle_latents(
        self, block_input_ids: torch.Tensor, block_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.autoencoder is None:
            raise ValueError(
                "oracle_latents were not supplied and no online autoencoder is attached"
            )
        valid = block_mask.bool()
        flat_ids = block_input_ids[valid]
        if flat_ids.numel() == 0:
            raise ValueError("batch contains no valid reasoning blocks")
        self.autoencoder.eval()
        with torch.no_grad():
            flat_latents = self.autoencoder._compress(
                flat_ids, return_sample=self.oracle_latent_mode
            )
        batch_size, num_blocks = block_mask.shape
        result = torch.zeros(
            batch_size,
            num_blocks,
            self.latent_tokens_per_block,
            self.latent_dim,
            device=flat_latents.device,
            dtype=flat_latents.dtype,
        )
        result[valid] = flat_latents
        return result

    def _select_flow_tasks(
        self, block_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.target_block_sampling == "all":
            indices = torch.nonzero(block_mask, as_tuple=False)
            if not len(indices):
                raise ValueError("batch contains no valid blocks")
            return indices[:, 0], indices[:, 1]

        batch_indices: list[int] = []
        block_indices: list[int] = []
        for batch_index in range(block_mask.size(0)):
            valid = torch.nonzero(block_mask[batch_index], as_tuple=False).flatten()
            if not len(valid):
                continue
            choice = valid[torch.randint(len(valid), (1,), device=valid.device)]
            batch_indices.append(batch_index)
            block_indices.append(int(choice))
        if not batch_indices:
            raise ValueError("batch contains no valid blocks")
        return (
            torch.tensor(batch_indices, device=block_mask.device),
            torch.tensor(block_indices, device=block_mask.device),
        )

    def _flow_loss(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        oracle_latents: torch.Tensor,
        block_mask: torch.Tensor,
        context_latents: torch.Tensor,
    ) -> torch.Tensor:
        batch_indices, target_indices = self._select_flow_tasks(block_mask)
        task_questions = question_input_ids[batch_indices]
        task_question_mask = question_attention_mask[batch_indices]
        task_context = context_latents[batch_indices]
        task_context_mask = block_mask[batch_indices].clone()
        positions = torch.arange(block_mask.size(1), device=block_mask.device)
        task_context_mask &= positions.unsqueeze(0) < target_indices.unsqueeze(1)
        clean = oracle_latents[batch_indices, target_indices]

        noisy, target, timesteps, _ = self.scheduler.training_pair(clean)
        drop_mask = None
        if self.training and self.condition_dropout_prob > 0:
            drop_mask = (
                torch.rand(len(batch_indices), device=clean.device)
                < self.condition_dropout_prob
            )
        prediction = self._predict_current_block(
            task_questions,
            task_question_mask,
            task_context,
            task_context_mask,
            noisy,
            timesteps,
            drop_condition_mask=drop_mask,
        )
        return F.mse_loss(prediction.float(), target.float())

    def _answer_and_special_losses(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        latent_blocks: torch.Tensor,
        block_mask: torch.Tensor,
        answer_input_ids: torch.Tensor,
        answer_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self._build_sequence_batch(
            question_input_ids,
            question_attention_mask,
            latent_blocks,
            block_mask,
            answer_input_ids=answer_input_ids,
            answer_attention_mask=answer_attention_mask,
        )
        hidden, logits, _ = self._run_backbone(sequence)
        if sequence.labels is None:
            raise RuntimeError("answer labels were not constructed")
        answer_loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.size(-1)),
            sequence.labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        if sequence.eot_positions.numel() == 0:
            special_loss = hidden.sum() * 0.0
        else:
            eot_hidden = hidden[
                sequence.eot_batch_indices, sequence.eot_positions
            ]
            special_loss = F.cross_entropy(
                self.special_head(eot_hidden).float(), sequence.special_targets
            )
        return answer_loss, special_loss

    def _rollout_stage2(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        block_mask: torch.Tensor,
    ) -> torch.Tensor:
        generated: list[torch.Tensor] = []
        batch_size, max_blocks = block_mask.shape
        for block_index in range(max_blocks):
            context = (
                torch.stack(generated, dim=1)
                if generated
                else torch.zeros(
                    batch_size,
                    0,
                    self.latent_tokens_per_block,
                    self.latent_dim,
                    device=question_input_ids.device,
                    dtype=self.model_dtype,
                )
            )
            block = self.sample_block(
                question_input_ids,
                question_attention_mask,
                context,
                block_mask[:, :block_index],
                num_steps=self.rollout_steps,
                guidance_scale=1.0,
                initial_noise_scale=self.rollout_initial_noise_scale,
            )
            active = block_mask[:, block_index].view(batch_size, 1, 1)
            generated.append(torch.where(active, block, torch.zeros_like(block)))
        return torch.stack(generated, dim=1)

    def forward(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        block_mask: torch.Tensor,
        answer_input_ids: torch.Tensor,
        answer_attention_mask: torch.Tensor,
        oracle_latents: Optional[torch.Tensor] = None,
        block_input_ids: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        block_mask = block_mask.bool()
        if oracle_latents is None:
            if block_input_ids is None:
                raise ValueError("provide oracle_latents or block_input_ids")
            oracle_latents = self._encode_online_oracle_latents(
                block_input_ids, block_mask
            )
        oracle_latents = oracle_latents.to(
            device=question_input_ids.device, dtype=self.model_dtype
        )
        expected = (
            question_input_ids.size(0),
            block_mask.size(1),
            self.latent_tokens_per_block,
            self.latent_dim,
        )
        if oracle_latents.shape != expected:
            raise ValueError(
                f"oracle_latents must have shape {expected}, got "
                f"{tuple(oracle_latents.shape)}"
            )

        if self.stage == 2:
            context_latents = self._rollout_stage2(
                question_input_ids, question_attention_mask, block_mask
            )
            answer_latents = context_latents
        else:
            context_latents = oracle_latents
            answer_latents = oracle_latents

        flow_loss = self._flow_loss(
            question_input_ids,
            question_attention_mask,
            oracle_latents,
            block_mask,
            context_latents,
        )
        answer_loss, special_loss = self._answer_and_special_losses(
            question_input_ids,
            question_attention_mask,
            answer_latents,
            block_mask,
            answer_input_ids,
            answer_attention_mask,
        )
        loss = (
            self.lambda_flow * flow_loss
            + self.lambda_answer * answer_loss
            + self.lambda_special * special_loss
        )
        return {
            "loss": loss,
            "flow_loss": flow_loss.detach(),
            "answer_loss": answer_loss.detach(),
            "special_loss": special_loss.detach(),
        }
