# LaDiR VAE Reproduction Notes

This audit treats the current ICLR 2026 paper (arXiv:2510.04573v6) as the
reference and modifies the released VAE implementation in place. It does not
replace the repository with a new VAE architecture.

## Paper-to-code alignment

| Item | Paper recipe | Released code | Reproduction change |
|---|---|---|---|
| Training unit | One CoT sentence per block | Whole question/solution with length-driven segmentation | Split on `The answer is`, then sentence-blockize before tokenization |
| Encoder | Pretrained LLM, all parameters fine-tuned | LoRA-only encoder | Full fine-tuning by default; LoRA retained only as an opt-in compatibility path |
| Decoder | Separate frozen pretrained LLM | Separate decoder in training, encoder-with-disabled-adapter in inference | Always use the same separate frozen decoder |
| Latent size | 512 dimensions | 128 dimensions | Default changed to 512 |
| Block size | 4 latent tokens in Table 13 and Table 7 | 3 memory tokens | Default changed to 4 |
| Decoder conditioning | Latents followed directly by teacher-forced text embeddings | Extra `<AE>` embedding inserted between latents and text | Remove the release-only delimiter from the paper path |
| Latent augmentation | Gaussian noise with standard deviation `k=3` | Gaussian noise scaled by `0.3` | Use standard deviation 3 during VAE training only |
| Input augmentation | Uniform token substitution with probability `p=0.3` | Missing | Implement encoder-input substitution |
| Training recipe | LR `2e-5`, batch 128, 2 epochs, beta `1e-5` | LR/epochs/batch differed across script/config/defaults | Align launcher, Python defaults, and YAML |
| Token IDs | Backbone tokenizer IDs | Hard-coded BOS/EOS IDs from older LLaMA versions | Read special IDs from the tokenizer |
| Padding | Padding-aware batched encoding and decoding | No attention masks | Add masks and consistent position IDs |

## Paper inconsistency retained as an explicit choice

Appendix D.1 states that math reasoning uses six latent tokens, while the later
complete hyperparameter Table 13 lists four. The blockization ablation in Table
7 also identifies one sentence with four latent tokens as the best balance.
This reproduction therefore defaults to four, while exposing
`--fixed_mem_size` for a six-token ablation.

## Information the paper does not provide

The downstream YAML contains `scale_factor=0.2154` and `shift_factor=0.2192`,
but the paper does not define how these statistics were estimated. They are
therefore documented as release-checkpoint values, not treated as valid
statistics for a newly trained VAE. Before diffusion training, measure the new
checkpoint's latent distribution and verify the downstream affine convention.

Likewise, the paper gives the principal VAE hyperparameters but does not state
all optimizer details such as warmup length and weight decay. The launcher keeps
those secondary values explicit rather than presenting them as paper claims.

## Scope and verification

The patch covers the VAE data, model, teacher-forcing objective, inference
path, dependencies, and launcher. Static compilation, shell validation,
preprocessing tests, teacher-forcing tests, and a tiny mocked model smoke test
pass. It does not claim the published benchmark numbers until the full 8B
multi-GPU training run has completed on the paper data.
