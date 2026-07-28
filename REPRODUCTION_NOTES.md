# LaDiR Reproduction Notes

This audit treats the current ICLR 2026 paper (arXiv:2510.04573v6) as the
reference and modifies the released implementation in place. It does not claim
published benchmark numbers without full 8B multi-GPU training.

## VAE paper-to-code alignment

| Item | Paper recipe | Released code | Reproduction change |
|---|---|---|---|
| Training unit | One CoT sentence per block | Whole question/solution with length-driven segmentation | Split on `The answer is`, then sentence-blockize before tokenization |
| Encoder | Pretrained LLM, all parameters fine-tuned | LoRA-only encoder | Full fine-tuning by default; LoRA retained only as an opt-in optimization |
| Decoder | Separate frozen pretrained LLM | Separate decoder in training, encoder-with-disabled-adapter in inference | Always use the same separate frozen decoder |
| Latent size | 512 dimensions | 128 dimensions | Default changed to 512 |
| Block size | 4 latent tokens in Table 13 and Table 7 | 3 memory tokens | Default changed to 4 |
| Decoder conditioning | Latents followed directly by teacher-forced text embeddings | Extra `<AE>` embedding inserted between latents and text | Remove the release-only delimiter from the paper path |
| Latent augmentation | Gaussian noise with standard deviation `k=3` | Gaussian noise scaled by `0.3` | Use standard deviation 3 during VAE training only |
| Input augmentation | Uniform token substitution with probability `p=0.3` | Missing | Implement encoder-input substitution |
| Training recipe | LR `2e-5`, batch 128, 2 epochs, beta `1e-5` | LR/epochs/batch differed across script/config/defaults | Align launcher, Python defaults, and YAML |
| Token IDs | Backbone tokenizer IDs | Hard-coded BOS/EOS IDs from older LLaMA versions | Read special IDs from the tokenizer |
| Padding | Padding-aware batched encoding and decoding | No attention masks | Add masks and consistent position IDs |
| Decoder context | Latent prefix and target must fit the model context | Target truncated before adding memory tokens and EOS | Reserve the memory prefix inside `model_max_length` and force EOS within budget |
| Backbone interface | Paper uses Llama-3.1-8B | Generic `AutoModelForCausalLM` path implied broader support | Validate Llama model type and use its transformer backbone explicitly |

## Reasoner paper-to-code alignment

| Item | Paper recipe | Released code | Reproduction change |
|---|---|---|---|
| Training data | Ordered one-sentence VAE blocks | Whole question/solution sent through `_compress` | Precompute ordered frozen-VAE blocks in a validated memory-mapped store |
| Latent dimension | 512 | Hard-coded 128 | Infer/validate 512 from config and latent metadata |
| Block attention | Bidirectional inside a block, causal across blocks | Global handling of every `<tht>` token as one bidirectional set | Build a separate explicit span for each block |
| Denoising objective | Flow matching by default | Placeholder additive scheduler and fixed update | Implement the linear rectified-flow path and Euler integration |
| Objective ablations | MSE, x0, epsilon, v, flow | Not executable | One scheduler interface exposes all five objectives |
| Answer supervision | CE from the same LLM backbone | Commented out | Train answer tokens after `<SOA>` with EOS supervision |
| Variable block count | Binary `<BOT>`/`<SOA>` head at `<EOT>` | Missing | Add the binary head, loss, and inference stopping rule |
| Stage 1 | Oracle previous latent blocks | One flattened thought target | Train every target block conditioned on earlier oracle blocks |
| Stage 2 | Self-generated previous blocks, 10 denoising steps, gradients retained | Missing | Differentiable Euler rollouts with the ground-truth block count |
| CFG/diversity | CFG plus decaying repulsive guidance | Partial debug methods | Expose both in the common block sampler |
| Training recipe | Stage 1 batch 64; Stage 2 batch 12; LR `1e-5`; 20 epochs | Inconsistent VAE-oriented config | Separate Stage-1 and Stage-2 YAML/launchers |

## Deliberately unsupported VAE release path

The release's length-driven multi-segment encoder reused one set of memory-token
IDs while producing multiple latent groups. That layout cannot satisfy the
paper decoder interface, which expects one fixed latent block before the target
sentence. The reproduction therefore fails fast when `paper_block_mode=False`.

## Sentence blockization

The paper defines one sentence as one VAE block but does not publish its exact
sentence-segmentation implementation. The repository uses a deterministic
heuristic based on newlines and terminal punctuation. It protects decimals,
common abbreviations such as `e.g.` and `Eq.`, and initialisms such as `U.S.`.
It can still require upstream normalization for domain-specific abbreviations or
punctuation without whitespace.

Before full training, inspect the real data with:

```bash
python scripts/audit_vae_blocks.py --input data/vae_train.jsonl --show 20
```

## Paper ambiguities and explicit choices

### Four versus six latent tokens

Appendix D.1 states that math reasoning uses six latent tokens, while the later
complete hyperparameter Table 13 lists four. The blockization ablation in Table
7 also identifies one sentence with four latent tokens as the best balance.
This reproduction defaults to four and leaves the value configurable.

### Oracle latent sampling

The method samples VAE latents from the posterior, but it does not say whether a
fresh posterior sample is drawn every reasoner epoch. The practical default
precomputes one posterior sample per block. `--latent-mode mean` provides a
deterministic comparison; online resampling is deliberately not the default
because it would keep the large VAE resident during reasoner training.

### Target-block batching

The paper does not state whether all target blocks from one solution are trained
in one forward pass or sampled as separate instances. The implementation can
use all valid target blocks (`target_block_sampling=all`) or one random target
per sample (`random`) so this choice can be measured rather than hidden.

### Secondary optimizer and CFG details

The paper gives principal learning rates, batch sizes, epochs, loss weights,
and inference CFG scale, but not every optimizer, warmup, or condition-dropout
detail. Those settings remain explicit in YAML and are not presented as paper
claims. The release checkpoint's latent `scale_factor` and `shift_factor` are
also not applied to a newly trained VAE without re-estimation.

## Scope and verification

The patch covers VAE preprocessing/model training and the non-VAE reasoner:
latent precomputation, blockwise masking, objective parameterizations, joint
losses, Stage-1/Stage-2 training, stopping, answer generation, CFG, and diversity
guidance. Static compilation, shell validation, scheduler tests, latent-store
tests, Stage-1 backward tests, and differentiable Stage-2 rollout tests pass.

It does not claim numerical reproduction until the real data, gated backbone,
trained VAE, long multi-GPU runs, and benchmark-specific evaluators have been
executed and audited.
