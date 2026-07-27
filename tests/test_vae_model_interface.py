"""Lightweight VAE model-interface smoke tests with mocked HF models."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types
import unittest

import torch
from torch import nn


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        del kwargs
        if isinstance(text, list):
            return {"input_ids": [[3, 4] for _ in text]}
        return {"input_ids": [3, 4]}

    def decode(self, tokens, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(map(str, tokens))


class FakeBackbone(nn.Module):
    def forward(self, inputs_embeds, **kwargs):
        del kwargs
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class FakeCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=32,
            hidden_size=8,
            model_type="llama",
        )
        self.embed = nn.Embedding(32, 8, dtype=torch.bfloat16)
        self.model = FakeBackbone()

    def get_input_embeddings(self):
        return self.embed

    def resize_token_embeddings(self, size):
        old = self.embed
        self.embed = nn.Embedding(size, 8, dtype=torch.bfloat16)
        with torch.no_grad():
            self.embed.weight[: old.num_embeddings].copy_(old.weight)
        return self.embed

    def forward(self, inputs_embeds, use_cache=False, **kwargs):
        del kwargs
        batch, length, _ = inputs_embeds.shape
        logits = torch.zeros(batch, length, 32, device=inputs_embeds.device)
        logits[..., 2] = 1.0
        return SimpleNamespace(
            logits=logits,
            past_key_values=() if use_cache else None,
        )


transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoTokenizer = SimpleNamespace(
    from_pretrained=lambda *args, **kwargs: FakeTokenizer()
)
transformers_stub.AutoModelForCausalLM = SimpleNamespace(
    from_pretrained=lambda *args, **kwargs: FakeCausalLM()
)
sys.modules["transformers"] = transformers_stub

peft_stub = types.ModuleType("peft")
peft_stub.get_peft_model = lambda model, config: model
sys.modules["peft"] = peft_stub

safetensors_stub = types.ModuleType("safetensors")
safetensors_torch_stub = types.ModuleType("safetensors.torch")
safetensors_torch_stub.load_file = lambda path: {}
sys.modules["safetensors"] = safetensors_stub
sys.modules["safetensors.torch"] = safetensors_torch_stub

MODULE_PATH = Path(__file__).parents[1] / "vae" / "model_vae.py"
spec = importlib.util.spec_from_file_location("model_vae", MODULE_PATH)
model_vae = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(model_vae)


def make_args(**overrides):
    model_args = {
        "model_name_or_path": "fake-llama",
        "paper_block_mode": True,
        "latent_dim": 512,
        "beta": 1e-5,
        "latent_noise_std": 0.0,
        "token_substitution_prob": 0.0,
        "use_lora": False,
    }
    model_args.update(overrides)
    training_args = {
        "fixed_mem_size": 4,
        "mean_compression_rate": 1,
        "restore_from": "",
    }
    return SimpleNamespace(**model_args), SimpleNamespace(**training_args)


class VAEModelInterfaceTest(unittest.TestCase):
    def test_rejects_release_multi_segment_mode(self):
        model_args, training_args = make_args(paper_block_mode=False)
        with self.assertRaisesRegex(ValueError, "paper_block_mode"):
            model_vae.VAE(model_args, training_args)

    def test_forward_and_posterior_shapes(self):
        model_args, training_args = make_args()
        model = model_vae.VAE(model_args, training_args)
        model.train()

        input_ids = torch.tensor([[3, 4, 32], [5, 6, 7]])
        memory_ids = [33, 34, 35, 36]
        prompt_answer_ids = torch.tensor(
            [memory_ids + [3, 4, 2], memory_ids + [5, 6, 2]]
        )
        labels = torch.tensor(
            [[-100] * 4 + [3, 4, 2], [-100] * 4 + [5, 6, 2]]
        )

        outputs = model(input_ids, prompt_answer_ids, labels)
        self.assertEqual(outputs["logits"].shape, (2, 7, 32))
        self.assertTrue(torch.isfinite(outputs["loss"]))

        mean, logvar = model._compress(input_ids, return_sample="parameters")
        self.assertEqual(mean.shape, (2, 4, 512))
        self.assertEqual(logvar.shape, (2, 4, 512))


if __name__ == "__main__":
    unittest.main()
