"""Noise schedules used by the LaDiR latent reasoner.

The paper uses rectified-flow matching by default and compares it with x0,
epsilon, and v parameterizations. This module keeps those choices behind one
small interface so training and inference use the same path definition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Optional

import torch


@dataclass
class SchedulerStepOutput:
    """Output of one deterministic reverse-process step."""

    prev_sample: torch.Tensor


class FlowMatchEulerDiscreteScheduler:
    """Continuous-time scheduler for LaDiR latent blocks.

    ``objective='flow'`` follows the paper's linear path

        z_t = (1 - t) z_0 + t epsilon, t in [0, 1]

    and learns the constant velocity ``epsilon - z_0``. The DDIM-style
    ``epsilon`` and ``v`` objectives use a cosine alpha-bar schedule. ``x0``
    uses the same linear path as flow matching.
    """

    SUPPORTED_OBJECTIVES = {"flow", "x0", "epsilon", "v", "mse"}

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        objective: str = "flow",
        min_t: float = 1e-4,
    ) -> None:
        if objective not in self.SUPPORTED_OBJECTIVES:
            raise ValueError(
                f"objective must be one of {sorted(self.SUPPORTED_OBJECTIVES)}, "
                f"got {objective!r}"
            )
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        if not 0.0 < min_t < 0.5:
            raise ValueError("min_t must be in (0, 0.5)")

        self.num_train_timesteps = int(num_train_timesteps)
        self.objective = objective
        self.min_t = float(min_t)
        prediction_type = {
            "flow": "flow",
            "x0": "sample",
            "epsilon": "epsilon",
            "v": "v_prediction",
            "mse": "sample",
        }[objective]
        self.config = SimpleNamespace(prediction_type=prediction_type)
        self.timesteps: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None
        self._step_index: Optional[int] = None

    @staticmethod
    def _expand_time(t: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        while t.ndim < sample.ndim:
            t = t.unsqueeze(-1)
        return t.to(device=sample.device, dtype=sample.dtype)

    @staticmethod
    def _cosine_alpha_bar(t: torch.Tensor) -> torch.Tensor:
        s = 0.008
        angle = ((t + s) / (1.0 + s)) * (math.pi / 2.0)
        alpha_bar = torch.cos(angle).square()
        alpha_bar_0 = math.cos((s / (1.0 + s)) * (math.pi / 2.0)) ** 2
        result = (alpha_bar / alpha_bar_0).clamp(1e-5, 1.0)
        return torch.where(t <= 0, torch.ones_like(result), result)

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        t = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
        return t.clamp(self.min_t, 1.0 - self.min_t)

    def training_pair(
        self,
        clean: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(noisy, target, timesteps, noise)`` for one objective."""
        if clean.ndim < 2:
            raise ValueError("clean latents must have a batch dimension")
        if noise is None:
            noise = torch.randn(
                clean.shape,
                device=clean.device,
                dtype=clean.dtype,
                generator=generator,
            )
        if noise.shape != clean.shape:
            raise ValueError("noise must have the same shape as clean")
        if timesteps is None:
            timesteps = self.sample_timesteps(
                clean.size(0), clean.device, torch.float32, generator
            )
        if timesteps.shape != (clean.size(0),):
            raise ValueError("timesteps must have shape [batch]")

        t = self._expand_time(timesteps, clean)
        if self.objective in {"flow", "x0"}:
            noisy = (1.0 - t) * clean + t * noise
            target = noise - clean if self.objective == "flow" else clean
        elif self.objective in {"epsilon", "v"}:
            alpha = self._expand_time(self._cosine_alpha_bar(timesteps), clean)
            sqrt_alpha = alpha.sqrt()
            sqrt_one_minus = (1.0 - alpha).sqrt()
            noisy = sqrt_alpha * clean + sqrt_one_minus * noise
            if self.objective == "epsilon":
                target = noise
            else:
                target = sqrt_alpha * noise - sqrt_one_minus * clean
        else:
            noisy = torch.zeros_like(clean)
            target = clean
        return noisy, target, timesteps, noise

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self._normalize_legacy_timesteps(timesteps, original_samples)
        noisy, _, _, _ = self.training_pair(
            original_samples, noise=noise, timesteps=normalized
        )
        self.timestep = normalized
        return noisy

    def _normalize_legacy_timesteps(
        self, timesteps: torch.Tensor, sample: torch.Tensor
    ) -> torch.Tensor:
        timesteps = timesteps.to(device=sample.device, dtype=torch.float32)
        if timesteps.numel() and timesteps.max() > 1.0:
            timesteps = timesteps / float(max(self.num_train_timesteps - 1, 1))
        return timesteps.clamp(self.min_t, 1.0 - self.min_t)

    def get_velocity(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        timesteps = self._normalize_legacy_timesteps(timesteps, clean)
        alpha = self._expand_time(self._cosine_alpha_bar(timesteps), clean)
        return alpha.sqrt() * noise - (1.0 - alpha).sqrt() * clean

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Optional[torch.device | str] = None,
    ) -> None:
        if num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        self.timesteps = torch.linspace(
            1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32
        )
        self.sigmas = self.timesteps.clone()
        self._step_index = 0

    def _predict_x0_and_epsilon(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.as_tensor(timestep, device=sample.device, dtype=torch.float32)
        if t.ndim == 0:
            t = t.expand(sample.size(0))
        t_expanded = self._expand_time(t.clamp(self.min_t, 1.0 - self.min_t), sample)

        if self.objective == "x0":
            x0 = model_output
            epsilon = (sample - (1.0 - t_expanded) * x0) / t_expanded
        elif self.objective == "epsilon":
            alpha = self._expand_time(self._cosine_alpha_bar(t), sample)
            epsilon = model_output
            x0 = (sample - (1.0 - alpha).sqrt() * epsilon) / alpha.sqrt()
        elif self.objective == "v":
            alpha = self._expand_time(self._cosine_alpha_bar(t), sample)
            sqrt_alpha = alpha.sqrt()
            sqrt_one_minus = (1.0 - alpha).sqrt()
            x0 = sqrt_alpha * sample - sqrt_one_minus * model_output
            epsilon = sqrt_one_minus * sample + sqrt_alpha * model_output
        elif self.objective == "mse":
            x0 = model_output
            epsilon = torch.zeros_like(model_output)
        else:
            raise RuntimeError("flow objective does not use x0/epsilon conversion")
        return x0, epsilon

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
        *,
        next_timestep: Optional[torch.Tensor | float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> SchedulerStepOutput:
        del generator
        t = torch.as_tensor(timestep, device=sample.device, dtype=torch.float32)
        if next_timestep is None:
            if self.timesteps is None or self._step_index is None:
                raise RuntimeError("set_timesteps must be called before step")
            if self._step_index + 1 >= len(self.timesteps):
                raise IndexError("sampling is already at the final timestep")
            next_t = self.timesteps[self._step_index + 1].to(sample.device)
            self._step_index += 1
        else:
            next_t = torch.as_tensor(
                next_timestep, device=sample.device, dtype=torch.float32
            )

        if self.objective == "flow":
            prev = sample + self._expand_time(next_t - t, sample) * model_output
        elif self.objective == "x0":
            t_safe = self._expand_time(t.clamp_min(self.min_t), sample)
            velocity = (sample - model_output) / t_safe
            prev = sample + self._expand_time(next_t - t, sample) * velocity
        elif self.objective in {"epsilon", "v"}:
            x0, epsilon = self._predict_x0_and_epsilon(model_output, sample, t)
            next_alpha = self._expand_time(
                self._cosine_alpha_bar(next_t), sample
            )
            prev = next_alpha.sqrt() * x0 + (1.0 - next_alpha).sqrt() * epsilon
        else:
            prev = model_output
        return SchedulerStepOutput(prev_sample=prev)
