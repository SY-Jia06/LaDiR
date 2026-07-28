"""Denoising, stopping, diversity guidance, and answer generation."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F


class InferenceMixin:
    @staticmethod
    def diversity_repulsion(latents: torch.Tensor) -> torch.Tensor:
        """Repel parallel latent trajectories using the paper's RBF force."""
        if latents.size(0) <= 1:
            return torch.zeros_like(latents)
        flat = latents.float().reshape(latents.size(0), -1)
        diff = flat[:, None, :] - flat[None, :, :]
        distance_sq = diff.square().sum(dim=-1)
        nonzero = distance_sq[distance_sq > 0]
        if nonzero.numel() == 0:
            return torch.zeros_like(latents)
        sigma_sq = nonzero.sqrt().median().square().clamp_min(1e-6)
        ratio = distance_sq / sigma_sq
        weights = 2.0 * (1.0 - ratio) * torch.exp(-ratio)
        weights.fill_diagonal_(0.0)
        force = (weights.unsqueeze(-1) * diff).sum(dim=1)
        return force.reshape_as(latents).to(latents.dtype)

    def sample_block(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        context_blocks: torch.Tensor,
        context_block_mask: torch.Tensor,
        *,
        num_steps: int = 50,
        guidance_scale: float = 1.0,
        initial_noise_scale: float = 1.0,
        diversity_scale: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        batch_size = question_input_ids.size(0)
        sample = torch.randn(
            batch_size,
            self.latent_tokens_per_block,
            self.latent_dim,
            device=question_input_ids.device,
            dtype=self.model_dtype,
            generator=generator,
        ) * initial_noise_scale
        self.scheduler.set_timesteps(num_steps, device=sample.device)
        if self.scheduler.timesteps is None:
            raise RuntimeError("scheduler did not create an inference grid")

        for step_index in range(num_steps):
            timestep = self.scheduler.timesteps[step_index]
            next_timestep = self.scheduler.timesteps[step_index + 1]
            t_batch = timestep.expand(batch_size)
            conditional = self._predict_current_block(
                question_input_ids,
                question_attention_mask,
                context_blocks,
                context_block_mask,
                sample,
                t_batch,
            )
            prediction = conditional
            if guidance_scale != 1.0:
                unconditional = self._predict_current_block(
                    question_input_ids,
                    question_attention_mask,
                    context_blocks,
                    context_block_mask,
                    sample,
                    t_batch,
                    drop_condition_mask=torch.ones(
                        batch_size, device=sample.device, dtype=torch.bool
                    ),
                )
                prediction = unconditional + guidance_scale * (
                    conditional - unconditional
                )
            if diversity_scale and self.objective == "flow":
                prediction = prediction + (
                    diversity_scale
                    * float(timestep)
                    * self.diversity_repulsion(sample)
                )
            sample = self.scheduler.step(
                prediction,
                timestep,
                sample,
                next_timestep=next_timestep,
            ).prev_sample
        return sample

    def predict_stop(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        latent_blocks: torch.Tensor,
        block_mask: torch.Tensor,
    ) -> torch.Tensor:
        sequence = self._build_sequence_batch(
            question_input_ids,
            question_attention_mask,
            latent_blocks,
            block_mask,
        )
        hidden, _, _ = self._run_backbone(sequence)
        logits = torch.full(
            (question_input_ids.size(0), 2),
            float("-inf"),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        for batch_index in range(question_input_ids.size(0)):
            matching = sequence.eot_batch_indices == batch_index
            if matching.any():
                final_pos = sequence.eot_positions[matching][-1]
                logits[batch_index] = self.special_head(hidden[batch_index, final_pos])
        return logits

    @torch.no_grad()
    def generate_latent_reasoning(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        *,
        min_blocks: int = 1,
        max_blocks: int = 16,
        num_steps: int = 50,
        guidance_scale: float = 4.0,
        initial_noise_scale: float = 2.0,
        diversity_scale: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if min_blocks <= 0 or max_blocks < min_blocks:
            raise ValueError("require 0 < min_blocks <= max_blocks")
        batch_size = question_input_ids.size(0)
        generated: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        active = torch.ones(
            batch_size, device=question_input_ids.device, dtype=torch.bool
        )

        for block_index in range(max_blocks):
            if generated:
                context = torch.stack(generated, dim=1)
                context_mask = torch.stack(masks, dim=1)
            else:
                context = torch.zeros(
                    batch_size,
                    0,
                    self.latent_tokens_per_block,
                    self.latent_dim,
                    device=question_input_ids.device,
                    dtype=self.model_dtype,
                )
                context_mask = torch.zeros(
                    batch_size,
                    0,
                    device=question_input_ids.device,
                    dtype=torch.bool,
                )
            block = self.sample_block(
                question_input_ids,
                question_attention_mask,
                context,
                context_mask,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                initial_noise_scale=initial_noise_scale,
                diversity_scale=diversity_scale,
                generator=generator,
            )
            generated.append(
                torch.where(active[:, None, None], block, torch.zeros_like(block))
            )
            masks.append(active.clone())
            stacked = torch.stack(generated, dim=1)
            stacked_mask = torch.stack(masks, dim=1)
            should_stop = self.predict_stop(
                question_input_ids,
                question_attention_mask,
                stacked,
                stacked_mask,
            ).argmax(dim=-1).eq(1)
            if block_index + 1 >= min_blocks:
                active &= ~should_stop
            if not active.any():
                break
        return torch.stack(generated, dim=1), torch.stack(masks, dim=1)

    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        *,
        temperature: float,
        top_p: float,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1)
        logits = logits / temperature
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probabilities = F.softmax(sorted_logits, dim=-1)
            cumulative = probabilities.cumsum(dim=-1)
            remove = cumulative - probabilities > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            filtered = torch.full_like(logits, float("-inf"))
            filtered.scatter_(1, sorted_indices, sorted_logits)
            logits = filtered
        return torch.multinomial(
            F.softmax(logits, dim=-1), 1, generator=generator
        ).squeeze(1)

    @torch.no_grad()
    def generate_answer(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        latent_blocks: torch.Tensor,
        block_mask: torch.Tensor,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        batch_size = question_input_ids.size(0)
        generated = torch.empty(
            batch_size, 0, device=question_input_ids.device, dtype=torch.long
        )
        generated_mask = torch.empty(
            batch_size, 0, device=question_input_ids.device, dtype=torch.bool
        )
        finished = torch.zeros(
            batch_size, device=question_input_ids.device, dtype=torch.bool
        )
        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("tokenizer must define eos_token_id")

        # Correctness-first implementation; add a KV-cache path only after
        # numerical parity tests for the custom blockwise mask.
        for _ in range(max_new_tokens):
            sequence = self._build_sequence_batch(
                question_input_ids,
                question_attention_mask,
                latent_blocks,
                block_mask,
                answer_input_ids=generated,
                answer_attention_mask=generated_mask,
            )
            _, logits, _ = self._run_backbone(sequence)
            last_positions = sequence.valid_mask.long().sum(dim=1) - 1
            next_logits = logits[
                torch.arange(batch_size, device=logits.device), last_positions
            ]
            next_ids = self._sample_next_token(
                next_logits,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            next_ids = torch.where(
                finished, torch.full_like(next_ids, eos_id), next_ids
            )
            generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)
            generated_mask = torch.cat(
                [generated_mask, (~finished).unsqueeze(1)], dim=1
            )
            finished |= next_ids.eq(eos_id)
            if finished.all():
                break
        return generated

    @torch.no_grad()
    def generate(
        self,
        question_input_ids: torch.Tensor,
        question_attention_mask: torch.Tensor,
        **generation_kwargs: Any,
    ) -> dict[str, Any]:
        latent_keys = {
            "min_blocks",
            "max_blocks",
            "num_steps",
            "guidance_scale",
            "initial_noise_scale",
            "diversity_scale",
            "generator",
        }
        latent_kwargs = {
            key: value
            for key, value in generation_kwargs.items()
            if key in latent_keys
        }
        answer_kwargs = {
            key: value
            for key, value in generation_kwargs.items()
            if key in {"max_new_tokens", "temperature", "top_p", "generator"}
        }
        blocks, block_mask = self.generate_latent_reasoning(
            question_input_ids, question_attention_mask, **latent_kwargs
        )
        answer_ids = self.generate_answer(
            question_input_ids,
            question_attention_mask,
            blocks,
            block_mask,
            **answer_kwargs,
        )
        return {
            "latent_blocks": blocks,
            "block_mask": block_mask,
            "answer_ids": answer_ids,
            "texts": self.tokenizer.batch_decode(
                answer_ids, skip_special_tokens=True
            ),
        }
