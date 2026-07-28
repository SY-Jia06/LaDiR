"""Shared LaDiR reasoner data structures and embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn


@dataclass(frozen=True)
class ReasonerSpecialTokens:
    bot: str = "<BOT>"
    eot: str = "<EOT>"
    soa: str = "<SOA>"
    timestep: str = "<TIMESTEP>"
    latent: str = "<LATENT>"


@dataclass(frozen=True)
class ReasonerTokenIds:
    bot: int
    eot: int
    soa: int
    timestep: int
    latent: int


def cfg_get(config: Any, path: str, default: Any) -> Any:
    value = config
    for key in path.split("."):
        if value is None:
            return default
        if isinstance(value, dict):
            if key not in value:
                return default
            value = value[key]
        else:
            if not hasattr(value, key):
                return default
            value = getattr(value, key)
    return value


def configure_reasoner_tokenizer(
    tokenizer: Any,
    causal_lm: Optional[nn.Module] = None,
    tokens: ReasonerSpecialTokens = ReasonerSpecialTokens(),
) -> ReasonerTokenIds:
    """Add control tokens and mean-initialize new LM embedding rows."""
    old_size = len(tokenizer)
    tokenizer.add_special_tokens(
        {
            "additional_special_tokens": [
                tokens.bot,
                tokens.eot,
                tokens.soa,
                tokens.timestep,
                tokens.latent,
            ]
        }
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    if causal_lm is not None and len(tokenizer) != old_size:
        old_weight = causal_lm.get_input_embeddings().weight.detach().clone()
        causal_lm.resize_token_embeddings(len(tokenizer))
        new_input = causal_lm.get_input_embeddings().weight
        with torch.no_grad():
            mean = old_weight.mean(dim=0, keepdim=True).to(new_input.dtype)
            new_input[old_size:].copy_(mean.expand(len(tokenizer) - old_size, -1))
            output = causal_lm.get_output_embeddings()
            if output is not None and output.weight.shape[0] == len(tokenizer):
                output.weight[old_size:].copy_(
                    mean.to(output.weight.dtype).expand(len(tokenizer) - old_size, -1)
                )

    ids = ReasonerTokenIds(
        bot=tokenizer.convert_tokens_to_ids(tokens.bot),
        eot=tokenizer.convert_tokens_to_ids(tokens.eot),
        soa=tokenizer.convert_tokens_to_ids(tokens.soa),
        timestep=tokenizer.convert_tokens_to_ids(tokens.timestep),
        latent=tokenizer.convert_tokens_to_ids(tokens.latent),
    )
    if len({ids.bot, ids.eot, ids.soa, ids.timestep, ids.latent}) != 5:
        raise RuntimeError("LaDiR special tokens did not receive distinct IDs")
    return ids


class FourierTimeEmbedding(nn.Module):
    """Continuous sinusoidal timestep embedding followed by a two-layer MLP."""

    def __init__(self, hidden_size: int, frequency_dim: int = 256) -> None:
        super().__init__()
        if frequency_dim % 2:
            raise ValueError("frequency_dim must be even")
        self.frequency_dim = frequency_dim
        self.proj = nn.Sequential(
            nn.Linear(frequency_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timesteps: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        half = self.frequency_dim // 2
        frequencies = torch.exp(
            -torch.log(torch.tensor(10000.0, device=timesteps.device))
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / half
        )
        angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.cos(), angles.sin()], dim=-1).to(dtype)
        return self.proj(embedding)


@dataclass
class SequenceBatch:
    embeddings: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    valid_mask: torch.Tensor
    current_latent_positions: Optional[torch.Tensor]
    eot_batch_indices: torch.Tensor
    eot_positions: torch.Tensor
    special_targets: torch.Tensor
    labels: Optional[torch.Tensor]
