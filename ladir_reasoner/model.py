"""Paper-aligned non-VAE LaDiR reasoning model."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from fm_noise_scheduler import FlowMatchEulerDiscreteScheduler
from .common import (
    FourierTimeEmbedding,
    ReasonerTokenIds,
    cfg_get,
    configure_reasoner_tokenizer,
)
from .inference import InferenceMixin
from .sequence import SequenceMixin
from .training import TrainingMixin


class LaDiRReasoner(TrainingMixin, InferenceMixin, SequenceMixin, nn.Module):
    """Latent block diffusion plus autoregressive answer generation."""

    def __init__(
        self,
        causal_lm: nn.Module,
        tokenizer: Any,
        config: Any,
        *,
        token_ids: Optional[ReasonerTokenIds] = None,
        autoencoder: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.text_llama = causal_lm  # release compatibility name
        self.tokenizer = tokenizer
        self.config = config
        self.token_ids = token_ids or configure_reasoner_tokenizer(
            tokenizer, causal_lm
        )
        self.autoencoder = autoencoder
        if self.autoencoder is not None:
            for parameter in self.autoencoder.parameters():
                parameter.requires_grad = False
            self.autoencoder.eval()

        self.hidden_size = int(causal_lm.config.hidden_size)
        self.latent_dim = int(
            cfg_get(config, "model.latent_dim", getattr(autoencoder, "dim", 512))
        )
        self.latent_tokens_per_block = int(
            cfg_get(
                config,
                "model.latent_tokens_per_block",
                getattr(autoencoder, "mem_size", 4),
            )
        )
        if self.latent_dim <= 0 or self.latent_tokens_per_block <= 0:
            raise ValueError("latent dimensions and block size must be positive")

        self.stage = int(cfg_get(config, "model.stage", 1))
        if self.stage not in {1, 2}:
            raise ValueError("model.stage must be 1 or 2")
        self.objective = str(cfg_get(config, "model.objective", "flow"))
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=int(
                cfg_get(config, "model.num_train_timesteps", 1000)
            ),
            objective=self.objective,
        )
        self.lambda_flow = float(cfg_get(config, "model.lambda_flow", 5.0))
        self.lambda_answer = float(cfg_get(config, "model.lambda_answer", 1.0))
        self.lambda_special = float(cfg_get(config, "model.lambda_special", 1.0))
        self.condition_dropout_prob = float(
            cfg_get(config, "model.condition_dropout_prob", 0.1)
        )
        self.target_block_sampling = str(
            cfg_get(config, "model.target_block_sampling", "all")
        )
        if self.target_block_sampling not in {"all", "random"}:
            raise ValueError("target_block_sampling must be 'all' or 'random'")
        self.oracle_latent_mode = str(
            cfg_get(config, "model.oracle_latent_mode", "sample")
        )
        self.rollout_steps = int(cfg_get(config, "model.rollout_steps", 10))
        self.rollout_initial_noise_scale = float(
            cfg_get(config, "model.rollout_initial_noise_scale", 1.0)
        )

        self.latent_in = nn.Linear(self.latent_dim, self.hidden_size)
        self.latent_out = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.latent_dim),
        )
        self.time_embedding = FourierTimeEmbedding(self.hidden_size)
        self.special_head = nn.Linear(self.hidden_size, 2)
        self.null_condition = nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.null_condition, std=0.02)
        nn.init.zeros_(self.latent_out[-1].bias)
        if hasattr(self.text_llama.config, "use_cache"):
            self.text_llama.config.use_cache = False

    @property
    def model_dtype(self) -> torch.dtype:
        return self.text_llama.get_input_embeddings().weight.dtype

    @property
    def model_device(self) -> torch.device:
        return self.text_llama.get_input_embeddings().weight.device

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs=None
    ) -> None:
        if hasattr(self.text_llama, "gradient_checkpointing_enable"):
            self.text_llama.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {}
            )
