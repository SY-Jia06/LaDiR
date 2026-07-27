"""Paper-aligned variational autoencoder for LaDiR thought blocks.

The encoder and decoder are initialized from the same causal language model.
A reasoning sentence is encoded together with learnable latent-query embeddings.
The hidden states at the query positions parameterize a diagonal Gaussian over
latent thought tokens. A frozen language-model decoder reconstructs the sentence
under teacher forcing from the sampled latent prefix.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import get_peft_model
except ImportError:  # pragma: no cover - PEFT is optional for full fine-tuning.
    get_peft_model = None


def _getattr(obj, name: str, default):
    return getattr(obj, name, default)


def freeze_module(module: nn.Module) -> None:
    module.requires_grad_(False)
    module.eval()


def print_trainable_parameters(model: nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    ratio = 100.0 * trainable / max(total, 1)
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {ratio:.4f}")


class VAE(nn.Module):
    """LLM-based beta-VAE for one reasoning block at a time.

    Defaults follow the paper's main VAE configuration: four latent tokens,
    512 latent dimensions, beta=1e-5, token substitution probability 0.3,
    and latent Gaussian augmentation with standard deviation 3.0. All values
    remain configurable because the paper reports task-specific block sizes.

    The constructor keeps the original repository signature so existing loading
    code can continue to instantiate ``VAE(model_args, training_args, lora_config)``.
    """

    def __init__(self, model_args, training_args, lora_config=None):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path

        self.dim = int(_getattr(model_args, "latent_dim", 512))
        self.mem_size = int(
            _getattr(
                model_args,
                "num_latent_tokens",
                _getattr(training_args, "fixed_mem_size", 4),
            )
        )
        self.fixed_mem_size = self.mem_size
        self.beta = float(_getattr(model_args, "beta", 1e-5))
        self.token_substitution_prob = float(
            _getattr(model_args, "token_substitution_prob", 0.3)
        )
        self.latent_noise_std = float(_getattr(model_args, "latent_noise_std", 3.0))
        self.encoder_tuning = str(_getattr(model_args, "encoder_tuning", "full"))

        use_bf16 = bool(_getattr(training_args, "bf16", True))
        use_fp16 = bool(_getattr(training_args, "fp16", False))
        if use_bf16:
            model_dtype = torch.bfloat16
        elif use_fp16:
            model_dtype = torch.float16
        else:
            model_dtype = torch.float32

        load_kwargs = {
            "torch_dtype": model_dtype,
            "trust_remote_code": bool(_getattr(model_args, "trust_remote_code", False)),
        }
        attn_implementation = _getattr(model_args, "attn_implementation", None)
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        self.encoder = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)
        self.decoder = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kwargs)

        if self.encoder_tuning == "lora":
            if get_peft_model is None:
                raise ImportError("encoder_tuning='lora' requires the peft package")
            if lora_config is None:
                raise ValueError("A LoRA config is required when encoder_tuning='lora'")
            self.encoder = get_peft_model(self.encoder, lora_config)
        elif self.encoder_tuning != "full":
            raise ValueError("encoder_tuning must be either 'full' or 'lora'")

        # Compatibility alias used by parts of the original repository.
        self.icae = self.encoder

        freeze_module(self.decoder)
        self.encoder.config.use_cache = False
        self.decoder.config.use_cache = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False,
            trust_remote_code=bool(_getattr(model_args, "trust_remote_code", False)),
        )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("The tokenizer must define either a pad token or an EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.eos_id = int(self.tokenizer.eos_token_id)
        self.bos_id = self.tokenizer.bos_token_id
        self.vocab_size = int(self.encoder.config.vocab_size)

        hidden_size = int(self.encoder.config.hidden_size)
        self.hidden_size = hidden_size

        # Learnable embeddings are appended after the text. Since the encoder is
        # causal, each query can attend to the complete reasoning block.
        self.latent_queries = nn.Parameter(torch.empty(self.mem_size, hidden_size))
        nn.init.normal_(self.latent_queries, mean=0.0, std=0.02)

        self.mean = nn.Linear(hidden_size, self.dim, dtype=model_dtype)
        self.log_var = nn.Linear(hidden_size, self.dim, dtype=model_dtype)
        self.decompress_layer = nn.Linear(self.dim, hidden_size, dtype=model_dtype)

        if bool(_getattr(training_args, "gradient_checkpointing", False)):
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        print_trainable_parameters(self)

    def train(self, mode: bool = True):
        super().train(mode)
        # The paper freezes the decoder throughout VAE training.
        self.decoder.eval()
        return self

    def _special_token_ids(self, device: torch.device) -> torch.Tensor:
        ids = [self.pad_token_id, self.eos_id]
        if self.bos_id is not None:
            ids.append(int(self.bos_id))
        return torch.tensor(sorted(set(ids)), device=device, dtype=torch.long)

    def _substitute_input_tokens(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.LongTensor:
        """Randomly substitute encoder tokens as the paper's robustness augmentation."""
        if not self.training or self.token_substitution_prob <= 0:
            return input_ids

        replace_mask = torch.rand(input_ids.shape, device=input_ids.device)
        replace_mask = (replace_mask < self.token_substitution_prob) & attention_mask.bool()

        special_ids = self._special_token_ids(input_ids.device)
        for token_id in special_ids:
            replace_mask &= input_ids.ne(token_id)

        random_ids = torch.randint(
            low=0,
            high=self.vocab_size,
            size=input_ids.shape,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        return torch.where(replace_mask, random_ids, input_ids)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    @staticmethod
    def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())

    def encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        sample: bool = True,
        add_robustness_noise: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id).long()

        encoder_ids = self._substitute_input_tokens(input_ids, attention_mask)
        text_embeddings = self.encoder.get_input_embeddings()(encoder_ids)

        batch_size = input_ids.size(0)
        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries.to(dtype=text_embeddings.dtype, device=text_embeddings.device)
        encoder_embeddings = torch.cat([text_embeddings, queries], dim=1)

        query_mask = torch.ones(
            batch_size,
            self.mem_size,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        encoder_attention_mask = torch.cat([attention_mask, query_mask], dim=1)

        outputs = self.encoder(
            inputs_embeds=encoder_embeddings,
            attention_mask=encoder_attention_mask,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        query_hidden = outputs.hidden_states[-1][:, -self.mem_size :, :]
        mu = self.mean(query_hidden)
        logvar = self.log_var(query_hidden)

        z = self.reparameterize(mu, logvar) if sample else mu
        if add_robustness_noise is None:
            add_robustness_noise = self.training
        if add_robustness_noise and self.latent_noise_std > 0:
            z = z + torch.randn_like(z) * self.latent_noise_std

        return {"latent": z, "mu": mu, "logvar": logvar}

    def _decode_teacher_forcing(
        self,
        z: torch.Tensor,
        target_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct the block from a latent prefix and shifted gold tokens."""
        latent_prefix = self.decompress_layer(z).to(self.decoder.dtype)
        batch_size, target_length = target_ids.shape

        # z_1 ... z_k, w_1 ... w_{L-1} -> predict w_1 ... w_L
        if target_length > 1:
            shifted_embeddings = self.decoder.get_input_embeddings()(target_ids[:, :-1])
            decoder_embeddings = torch.cat([latent_prefix, shifted_embeddings], dim=1)
            decoder_attention_mask = torch.cat(
                [
                    torch.ones(
                        batch_size,
                        self.mem_size,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                    attention_mask[:, :-1],
                ],
                dim=1,
            )
        else:
            decoder_embeddings = latent_prefix
            decoder_attention_mask = torch.ones(
                batch_size,
                self.mem_size,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )

        decoder_outputs = self.decoder(
            inputs_embeds=decoder_embeddings,
            attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )

        start = self.mem_size - 1
        reconstruction_logits = decoder_outputs.logits[:, start : start + target_length, :]
        labels = target_ids.masked_fill(~attention_mask.bool(), -100)
        ce_loss = F.cross_entropy(
            reconstruction_logits.float().reshape(-1, reconstruction_logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        return reconstruction_logits, ce_loss

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        **_: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id).long()

        encoded = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sample=True,
            add_robustness_noise=True,
        )
        logits, ce_loss = self._decode_teacher_forcing(
            encoded["latent"], input_ids, attention_mask
        )
        kl_loss = self.kl_divergence(encoded["mu"], encoded["logvar"])
        loss = ce_loss + self.beta * kl_loss
        return {
            "loss": loss,
            "logits": logits,
            "ce_loss": ce_loss,
            "kl_loss": kl_loss,
            "latent": encoded["latent"],
            "mu": encoded["mu"],
            "logvar": encoded["logvar"],
        }

    def _compress(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        return_sample: Optional[str] = None,
    ):
        """Compatibility entry point used by the original diffusion code."""
        if attention_mask is None:
            attention_mask = input_ids.ne(self.pad_token_id).long()
        if return_sample == "parameters":
            encoded = self.encode(input_ids, attention_mask, sample=False, add_robustness_noise=False)
            return encoded["mu"], encoded["logvar"]
        if return_sample == "sample":
            return self.encode(
                input_ids,
                attention_mask,
                sample=True,
                add_robustness_noise=False,
            )["latent"]
        return self.encode(
            input_ids,
            attention_mask,
            sample=False,
            add_robustness_noise=False,
        )["latent"]

    @torch.no_grad()
    def encode_text(self, text: str, return_sample: str = ""):
        self.eval()
        batch = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=int(_getattr(self.training_args, "model_max_length", 512)),
        )
        device = next(self.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        return self._compress(input_ids, attention_mask, return_sample or None)

    @torch.no_grad()
    def encode_batch_text(self, text_list: List[str], return_sample: str = ""):
        self.eval()
        batch = self.tokenizer(
            text_list,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(_getattr(self.training_args, "model_max_length", 512)),
        )
        device = next(self.parameters()).device
        return self._compress(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            return_sample or None,
        )

    @torch.no_grad()
    def decode_text_batch(
        self,
        memory_slots: torch.Tensor,
        max_new_tokens: int = 256,
    ) -> List[str]:
        self.eval()
        device = next(self.parameters()).device
        z = memory_slots.to(device=device, dtype=self.decompress_layer.weight.dtype)
        prefix = self.decompress_layer(z).to(self.decoder.dtype)

        batch_size = z.size(0)
        generated: List[List[int]] = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        outputs = self.decoder(inputs_embeds=prefix, use_cache=True, return_dict=True)
        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        for _ in range(max_new_tokens):
            next_ids = next_logits.argmax(dim=-1)
            next_ids = torch.where(
                finished,
                torch.full_like(next_ids, self.eos_id),
                next_ids,
            )
            for index, token_id in enumerate(next_ids.tolist()):
                if not finished[index]:
                    generated[index].append(token_id)
            finished |= next_ids.eq(self.eos_id)
            if finished.all():
                break

            next_embeddings = self.decoder.get_input_embeddings()(next_ids).unsqueeze(1)
            outputs = self.decoder(
                inputs_embeds=next_embeddings,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]

        return [
            self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for ids in generated
        ]

    @torch.no_grad()
    def decode_text(self, memory_slots: torch.Tensor, max_new_tokens: int = 256) -> str:
        return self.decode_text_batch(memory_slots, max_new_tokens=max_new_tokens)[0]

    @torch.no_grad()
    def run_inference(self, text: str) -> str:
        return self.decode_text(self.encode_text(text))
