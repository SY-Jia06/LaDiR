"""Paper-aligned variational autoencoder for LaDiR thought blocks.

The implementation keeps the released ICAE-style design: a pretrained causal
LLM encodes one reasoning sentence followed by learnable memory tokens, linear
heads parameterize a Gaussian latent distribution, and a separate frozen copy
of the pretrained LLM reconstructs the sentence under teacher forcing.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from peft import get_peft_model
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def print_trainable_parameters(model: nn.Module) -> None:
    trainable_parameters = 0
    all_parameters = 0
    for parameter in model.parameters():
        all_parameters += parameter.numel()
        if parameter.requires_grad:
            trainable_parameters += parameter.numel()
    percentage = 100 * trainable_parameters / max(all_parameters, 1)
    print(
        f"trainable params: {trainable_parameters} || all params: {all_parameters} "
        f"|| trainable%: {percentage}"
    )


def freeze_model(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


class VAE(nn.Module):
    """Variational autoencoder for one-sentence LaDiR thought blocks.

    The reproduction path intentionally supports the paper's Llama-3.1 causal
    backbone and fixed one-sentence blocks only. LoRA remains an optional encoder
    optimization, but the paper recipe fine-tunes the full encoder.
    """

    def __init__(self, model_args, training_args, lora_config=None):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.model_name = model_args.model_name_or_path

        self.paper_block_mode = bool(getattr(model_args, "paper_block_mode", True))
        if not self.paper_block_mode:
            raise ValueError(
                "paper_block_mode=False is not supported by the paper-aligned "
                "decoder interface. Pre-blockize each CoT sentence and keep "
                "paper_block_mode=True."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.icae = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
        )
        self.decoder = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
        )

        model_type = getattr(self.icae.config, "model_type", None)
        if model_type != "llama" or not hasattr(self.icae, "model"):
            raise ValueError(
                "This reproduction currently supports a LlamaForCausalLM "
                "backbone only; use the paper's meta-llama/Llama-3.1-8B model."
            )

        self.base_vocab_size = self.icae.config.vocab_size
        self.vocab_size = self.base_vocab_size + 1  # reserve one ID for padding
        self.pad_token_id = self.base_vocab_size
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id
        if self.eos_id is None:
            raise ValueError("The selected tokenizer must define eos_token_id.")

        self.mem_size = int(training_args.fixed_mem_size)
        if self.mem_size <= 0:
            raise ValueError("fixed_mem_size must be positive")
        self.mean_compression_rate = int(training_args.mean_compression_rate)
        self.dim = int(getattr(model_args, "latent_dim", 512))
        self.beta = float(model_args.beta)
        self.latent_noise_std = float(getattr(model_args, "latent_noise_std", 3.0))
        self.token_substitution_prob = float(
            getattr(model_args, "token_substitution_prob", 0.3)
        )
        if not 0.0 <= self.token_substitution_prob <= 1.0:
            raise ValueError("token_substitution_prob must be in [0, 1]")
        self.use_lora = bool(getattr(model_args, "use_lora", False))

        self.vocab_size_with_mem = self.vocab_size + self.mem_size
        # Kept as a compatibility attribute; the paper path does not insert it.
        self.ae_token_id = self.vocab_size_with_mem
        self.icae.resize_token_embeddings(self.vocab_size_with_mem + 1)

        if self.use_lora:
            if lora_config is None:
                raise ValueError("lora_config is required when use_lora=True")
            self.icae = get_peft_model(self.icae, lora_config)

        hidden_size = self.icae.config.hidden_size
        self.mean = nn.Linear(hidden_size, self.dim, dtype=torch.bfloat16)
        self.log_var = nn.Linear(hidden_size, self.dim, dtype=torch.bfloat16)
        self.decompress_layer = nn.Linear(self.dim, hidden_size, dtype=torch.bfloat16)
        self.memory_token_embed = nn.Embedding(
            self.mem_size + 1,
            self.dim,
            padding_idx=None,
            dtype=torch.bfloat16,
        )
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        self.register_buffer(
            "append_sequence",
            torch.arange(self.vocab_size, self.vocab_size + self.mem_size).unsqueeze(0),
            persistent=False,
        )

        freeze_model(self.decoder)
        self.decoder.eval()
        self._restore_if_requested()
        print_trainable_parameters(self)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def train(self, mode: bool = True):
        """Keep the frozen decoder deterministic while the encoder is trained."""
        super().train(mode)
        self.decoder.eval()
        return self

    def _restore_if_requested(self) -> None:
        restore_from = getattr(self.training_args, "restore_from", "")
        if not restore_from:
            return
        print(f"Loading from the pretrained checkpoint: {restore_from}...")
        state_dict = load_file(restore_from)
        self.load_state_dict(state_dict)
        print(f"Finished loading from {restore_from}")

    def _encoder_model(self):
        return self.icae.get_base_model() if self.use_lora else self.icae

    def _encoder_embedding_layer(self):
        return self._encoder_model().get_input_embeddings()

    def _encoder_backbone(self):
        """Return the validated Llama transformer without its unused LM head."""
        encoder_model = self._encoder_model()
        backbone = getattr(encoder_model, "model", None)
        if backbone is None:
            raise RuntimeError(
                "Expected a LlamaForCausalLM-style `.model` transformer backbone."
            )
        return backbone

    def _decoder_embedding_layer(self):
        return self.decoder.get_input_embeddings()

    @staticmethod
    def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        return position_ids.clamp_min_(0)

    def _decoder_text_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed ordinary decoder tokens without indexing custom VAE IDs."""
        safe_ids = token_ids.clone()
        custom_or_pad = safe_ids >= self.base_vocab_size
        safe_ids[custom_or_pad] = 0
        embeddings = self._decoder_embedding_layer()(safe_ids)
        embeddings[token_ids == self.pad_token_id] = 0
        return embeddings

    def _apply_token_substitution(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.training or self.token_substitution_prob <= 0:
            return input_ids
        valid = (input_ids >= 0) & (input_ids < self.base_vocab_size)
        replace = (
            torch.rand(input_ids.shape, device=input_ids.device)
            < self.token_substitution_prob
        ) & valid
        random_ids = torch.randint(
            low=0,
            high=self.base_vocab_size,
            size=input_ids.shape,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        return torch.where(replace, random_ids, input_ids)

    def compute_num_segments(self, total_length: int) -> int:
        """Return the single paper block; length chunking is deliberately disabled."""
        if total_length <= 0:
            raise ValueError("total_length must be positive")
        return 1

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _encode_block(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = input_ids.size(0)
        memory_ids = self.append_sequence.to(input_ids.device).expand(batch_size, -1)
        encoder_ids = torch.cat([input_ids, memory_ids], dim=1)
        memory_mask = encoder_ids >= self.vocab_size
        attention_mask = encoder_ids.ne(self.pad_token_id)

        embeddings = self._encoder_embedding_layer()(encoder_ids)
        memory_embeddings = self.decompress_layer(
            self.memory_token_embed(
                encoder_ids[memory_mask] - self.vocab_size
            ).to(embeddings.dtype)
        )
        embeddings[memory_mask] = memory_embeddings

        outputs = self._encoder_backbone()(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=self._position_ids(attention_mask),
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden_states = getattr(outputs, "hidden_states", None)
            if not hidden_states:
                raise RuntimeError("encoder backbone returned no hidden states")
            hidden = hidden_states[-1]
        memory_hidden = hidden[memory_mask].view(batch_size, self.mem_size, -1)
        return self.mean(memory_hidden), self.log_var(memory_hidden)

    def _memory_mask(
        self,
        prompt_answer_ids: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        mask = (
            (prompt_answer_ids >= self.vocab_size)
            & (prompt_answer_ids < self.vocab_size + self.mem_size)
        )
        expected = batch_size * self.mem_size
        actual = int(mask.sum())
        if actual != expected:
            raise ValueError(
                "Decoder prompt must contain exactly one paper block of memory "
                f"tokens per example: expected {expected}, found {actual}."
            )
        return mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        prompt_answer_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
    ):
        if labels is None:
            raise ValueError("labels are required for VAE training")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        input_ids = self._apply_token_substitution(input_ids)
        batch_size = input_ids.size(0)
        prompt_answer_ids = prompt_answer_ids.reshape(batch_size, -1)

        mu, logvar = self._encode_block(input_ids)
        compressed = self.reparameterize(mu, logvar)
        if self.training and self.latent_noise_std > 0:
            compressed = compressed + torch.randn_like(compressed) * self.latent_noise_std
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        prompt_answer_embs = self._decoder_text_embeddings(prompt_answer_ids)
        decoder_mem_mask = self._memory_mask(prompt_answer_ids, batch_size)
        decompressed = self.decompress_layer(compressed)
        prompt_answer_embs[decoder_mem_mask] = decompressed.reshape(
            -1, decompressed.size(-1)
        )

        decoder_attention_mask = prompt_answer_ids.ne(self.pad_token_id)
        decoder_outputs = self.decoder(
            inputs_embeds=prompt_answer_embs,
            attention_mask=decoder_attention_mask,
            position_ids=self._position_ids(decoder_attention_mask),
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        logits = decoder_outputs.logits
        effective_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        target_ids = labels[:, 1:].reshape(-1)
        ce_loss = self.loss_fct(effective_logits, target_ids)
        loss = ce_loss + self.beta * kl_loss
        return {
            "loss": loss,
            "logits": logits,
            "kl_loss": kl_loss,
            "ce_loss": ce_loss,
        }

    def decoder_loss(
        self,
        memory_slots: torch.Tensor,
        prompt_answer_ids: torch.LongTensor,
        labels: torch.LongTensor,
    ):
        batch_size = memory_slots.size(0)
        if memory_slots.size(1) != self.mem_size:
            raise ValueError(
                f"expected {self.mem_size} latent slots, received {memory_slots.size(1)}"
            )
        prompt_answer_ids = prompt_answer_ids.reshape(batch_size, -1)
        prompt_answer_embs = self._decoder_text_embeddings(prompt_answer_ids)
        decoder_mem_mask = self._memory_mask(prompt_answer_ids, batch_size)
        decompressed = self.decompress_layer(
            memory_slots.to(self.decompress_layer.weight.dtype)
        )
        prompt_answer_embs[decoder_mem_mask] = decompressed.reshape(
            -1, decompressed.size(-1)
        )

        attention_mask = prompt_answer_ids.ne(self.pad_token_id)
        outputs = self.decoder(
            inputs_embeds=prompt_answer_embs,
            attention_mask=attention_mask,
            position_ids=self._position_ids(attention_mask),
            use_cache=False,
            return_dict=True,
        )
        logits = outputs.logits
        ce_loss = self.loss_fct(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
        )
        return {"loss": ce_loss, "logits": logits}

    def tokens_to_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map ordinary and VAE special IDs into frozen-decoder hidden space."""
        embeddings = self._decoder_text_embeddings(token_ids)
        special = token_ids >= self.vocab_size
        if special.any():
            indices = token_ids[special] - self.vocab_size
            if int(indices.max()) > self.mem_size:
                raise ValueError("unknown VAE special token ID")
            latent_embeddings = self.memory_token_embed(indices).to(embeddings.dtype)
            embeddings[special] = self.decompress_layer(latent_embeddings)
        return embeddings

    def _compress(
        self,
        input_ids: torch.LongTensor,
        return_sample: Optional[str] = None,
    ):
        """Encode one sentence block as posterior parameters, sample, or mean.

        ``return_sample='sample'`` applies only VAE reparameterization. The
        robustness noise with std=3 is a training augmentation and is not
        applied when exporting oracle latents for diffusion training.
        """
        mu, logvar = self._encode_block(input_ids)
        if return_sample == "parameters":
            return mu, logvar
        if return_sample in (None, "", "sample"):
            return self.reparameterize(mu, logvar)
        if return_sample == "mean":
            return mu
        raise ValueError(
            "return_sample must be one of None, '', 'sample', 'mean', or 'parameters'"
        )

    def _tokenize_batch(self, texts: list[str], max_length: int = 5120) -> torch.Tensor:
        if not texts:
            raise ValueError("texts must not be empty")
        encoded = [
            self.tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_attention_mask=False,
            )["input_ids"]
            for text in texts
        ]
        max_len = max(len(ids) for ids in encoded)
        if max_len <= 0:
            raise ValueError("tokenizer produced an empty sequence")
        input_ids = torch.full(
            (len(encoded), max_len),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        for index, ids in enumerate(encoded):
            input_ids[index, : len(ids)] = torch.tensor(ids, device=self.device)
        return input_ids

    def encode_text(self, text: str, return_sample: str = ""):
        self.eval()
        with torch.no_grad():
            input_ids = self._tokenize_batch([text])
            return self._compress(input_ids, return_sample=return_sample)

    def encode_batch_text(self, text_list: list[str], return_sample: str = ""):
        self.eval()
        with torch.no_grad():
            input_ids = self._tokenize_batch(text_list)
            return self._compress(input_ids, return_sample=return_sample)

    def _greedy_decode(
        self,
        memory_slots: torch.Tensor,
        max_new_tokens: int = 256,
    ) -> list[str]:
        if memory_slots.ndim != 3 or memory_slots.size(1) != self.mem_size:
            raise ValueError(
                f"memory_slots must have shape [batch, {self.mem_size}, latent_dim]"
            )
        batch_size = memory_slots.size(0)
        memory_slots = memory_slots.to(
            device=self.device,
            dtype=self.decompress_layer.weight.dtype,
        )
        prefix = self.decompress_layer(memory_slots)

        outputs = self.decoder(inputs_embeds=prefix, use_cache=True, return_dict=True)
        past_key_values = outputs.past_key_values
        generated = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_new_tokens):
            next_ids = outputs.logits[:, -1, : self.base_vocab_size].argmax(dim=-1)
            active = ~finished
            for index in range(batch_size):
                if active[index] and next_ids[index].item() != self.eos_id:
                    generated[index].append(next_ids[index].item())
            finished |= next_ids.eq(self.eos_id)
            if finished.all():
                break

            decoder_ids = torch.where(
                finished,
                torch.full_like(next_ids, self.eos_id),
                next_ids,
            )
            next_embeddings = self._decoder_embedding_layer()(decoder_ids).unsqueeze(1)
            outputs = self.decoder(
                inputs_embeds=next_embeddings,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values

        return [
            self.tokenizer.decode(tokens, skip_special_tokens=True)
            for tokens in generated
        ]

    def run_inference(self, text: str) -> str:
        self.eval()
        with torch.no_grad():
            memory_slots = self.encode_text(text, return_sample="sample")
            return self._greedy_decode(memory_slots)[0]

    def decode_text(self, memory_slots: torch.Tensor):
        self.eval()
        with torch.no_grad():
            decoded = self._greedy_decode(memory_slots)
        return decoded[0] if len(decoded) == 1 else decoded

    def decode_text_batch(self, memory_slots: torch.Tensor) -> list[str]:
        self.eval()
        with torch.no_grad():
            return self._greedy_decode(memory_slots)
