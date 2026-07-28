from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch
from torch import nn

from model import LaDiRReasoner, configure_reasoner_tokenizer


class FakeTokenizer:
    def __init__(self):
        self.vocab = {f"t{index}": index for index in range(20)}
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.pad_token = "t0"

    def __len__(self):
        return len(self.vocab)

    def add_special_tokens(self, mapping):
        for token in mapping["additional_special_tokens"]:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab[token]

    def batch_decode(self, batch, skip_special_tokens=True):
        del skip_special_tokens
        return [" ".join(map(str, row.tolist())) for row in batch]


class FakeCausalLM(nn.Module):
    def __init__(self, vocab_size=20, hidden_size=12):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, use_cache=False)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.mix = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, size):
        old_embed = self.embed
        old_head = self.lm_head
        self.embed = nn.Embedding(size, old_embed.embedding_dim)
        self.lm_head = nn.Linear(old_embed.embedding_dim, size, bias=False)
        with torch.no_grad():
            self.embed.weight[: old_embed.num_embeddings].copy_(old_embed.weight)
            self.lm_head.weight[: old_head.out_features].copy_(old_head.weight)
        return self.embed

    def forward(
        self, inputs_embeds, attention_mask=None, position_ids=None, **kwargs
    ):
        del attention_mask, position_ids, kwargs
        hidden = torch.tanh(self.mix(inputs_embeds))
        logits = self.lm_head(hidden)
        return SimpleNamespace(hidden_states=(hidden,), logits=logits)


class ReasonerModelTest(unittest.TestCase):
    def make_model(self, stage=1):
        tokenizer = FakeTokenizer()
        lm = FakeCausalLM()
        ids = configure_reasoner_tokenizer(tokenizer, lm)
        config = {
            "model": {
                "latent_dim": 4,
                "latent_tokens_per_block": 2,
                "stage": stage,
                "objective": "flow",
                "lambda_flow": 5.0,
                "lambda_answer": 1.0,
                "lambda_special": 1.0,
                "condition_dropout_prob": 0.0,
                "target_block_sampling": "all",
                "rollout_steps": 2,
            }
        }
        return LaDiRReasoner(lm, tokenizer, config, token_ids=ids)

    def make_batch(self):
        return {
            "question_input_ids": torch.tensor([[1, 3, 4], [1, 5, 0]]),
            "question_attention_mask": torch.tensor(
                [[1, 1, 1], [1, 1, 0]], dtype=torch.bool
            ),
            "block_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
            "answer_input_ids": torch.tensor([[6, 7, 2], [8, 2, 0]]),
            "answer_attention_mask": torch.tensor(
                [[1, 1, 1], [1, 1, 0]], dtype=torch.bool
            ),
            "oracle_latents": torch.randn(2, 2, 2, 4),
        }

    def test_hybrid_mask_opens_only_the_declared_block(self):
        model = self.make_model()
        mask = model.build_blockwise_attention_mask(
            [6],
            [[(2, 4)]],
            max_length=6,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )[0, 0]
        self.assertEqual(float(mask[2, 4]), 0.0)
        self.assertLess(float(mask[1, 4]), -1e20)
        self.assertEqual(float(mask[5, 4]), 0.0)

    def test_stage1_joint_losses_are_finite_and_backward(self):
        model = self.make_model(stage=1)
        outputs = model(**self.make_batch())
        for key in ("loss", "flow_loss", "answer_loss", "special_loss"):
            self.assertTrue(torch.isfinite(outputs[key]).all(), key)
        outputs["loss"].backward()
        self.assertIsNotNone(model.latent_in.weight.grad)
        self.assertIsNotNone(model.special_head.weight.grad)

    def test_stage2_rollout_remains_differentiable(self):
        model = self.make_model(stage=2)
        batch = self.make_batch()
        batch["question_input_ids"] = batch["question_input_ids"][:1]
        batch["question_attention_mask"] = batch["question_attention_mask"][:1]
        batch["block_mask"] = batch["block_mask"][:1, :1]
        batch["answer_input_ids"] = batch["answer_input_ids"][:1]
        batch["answer_attention_mask"] = batch["answer_attention_mask"][:1]
        batch["oracle_latents"] = batch["oracle_latents"][:1, :1]
        outputs = model(**batch)
        outputs["loss"].backward()
        self.assertIsNotNone(model.latent_out[-1].weight.grad)


if __name__ == "__main__":
    unittest.main()
