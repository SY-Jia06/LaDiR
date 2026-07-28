from __future__ import annotations

import unittest

import torch

from fm_noise_scheduler import FlowMatchEulerDiscreteScheduler


class FlowSchedulerTest(unittest.TestCase):
    def test_perfect_flow_velocity_recovers_clean_sample(self):
        scheduler = FlowMatchEulerDiscreteScheduler(objective="flow")
        clean = torch.randn(2, 3, 4)
        noise = torch.randn_like(clean)
        timesteps = torch.tensor([0.3, 0.8])
        noisy, target, _, _ = scheduler.training_pair(
            clean, noise=noise, timesteps=timesteps
        )
        recovered = scheduler.step(
            target,
            timesteps,
            noisy,
            next_timestep=torch.zeros_like(timesteps),
        ).prev_sample
        self.assertTrue(torch.allclose(recovered, clean, atol=1e-5, rtol=1e-5))

    def test_x0_epsilon_and_v_end_at_predicted_clean_sample(self):
        clean = torch.randn(2, 2, 3)
        noise = torch.randn_like(clean)
        timesteps = torch.tensor([0.2, 0.7])
        for objective in ("x0", "epsilon", "v"):
            with self.subTest(objective=objective):
                scheduler = FlowMatchEulerDiscreteScheduler(objective=objective)
                noisy, target, _, _ = scheduler.training_pair(
                    clean, noise=noise, timesteps=timesteps
                )
                recovered = scheduler.step(
                    target,
                    timesteps,
                    noisy,
                    next_timestep=torch.zeros_like(timesteps),
                ).prev_sample
                self.assertTrue(
                    torch.allclose(recovered, clean, atol=2e-4, rtol=2e-4),
                    (recovered - clean).abs().max().item(),
                )

    def test_inference_grid_includes_both_endpoints(self):
        scheduler = FlowMatchEulerDiscreteScheduler(objective="flow")
        scheduler.set_timesteps(10)
        self.assertEqual(len(scheduler.timesteps), 11)
        self.assertEqual(float(scheduler.timesteps[0]), 1.0)
        self.assertEqual(float(scheduler.timesteps[-1]), 0.0)


if __name__ == "__main__":
    unittest.main()
