# LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning

Implementation of **LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning**.

This fork contains a paper-aligned VAE training path on the
`paper-aligned-vae` branch. The original public implementation used an
ICAE-style length compressor; the updated path trains one VAE latent block for
one reasoning sentence, as described in the paper.

## Paper-aligned VAE

For each reasoning block, the data flow is:

```text
reasoning sentence tokens
        +
learnable latent-query embeddings
        |
        v
trainable causal LLM encoder
        |
        v
query-position hidden states
        |
        +--> Linear(mu)
        +--> Linear(log variance)
        |
        v
z = mu + sigma * epsilon
        |
        + optional Gaussian robustness noise
        |
        v
Linear(latent_dim -> LLM hidden_dim)
        |
        v
frozen causal LLM decoder
        |
        v
teacher-forced reconstruction of the sentence
```

Default settings follow the paper's main VAE table:

- latent dimension: `512`
- latent tokens per block: `4`
- KL weight: `1e-5`
- encoder token-substitution probability: `0.3`
- latent robustness-noise standard deviation: `3.0`
- encoder tuning: full-parameter
- decoder: frozen

The latent-token count is configurable because the paper also reports
experiment-specific values.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Supported data layouts

The VAE loader is schema-driven and does not contain a hard-coded dataset
identifier. It accepts a Hugging Face dataset or a local JSON, JSONL, or Parquet
file.

### Scalar input and output

```json
{"input": "question text", "output": "reasoning trace"}
```

### Chat messages

```json
{
  "messages": [
    {"role": "user", "content": "question text"},
    {"role": "assistant", "content": "<think>reasoning trace</think> final answer"}
  ]
}
```

### Prompt plus sampled response lists

```json
{
  "prompt": "question text",
  "model_a_responses": ["<think>trace one</think> answer"],
  "model_b_responses": ["<think>trace two</think> answer"]
}
```

When no response field is supplied, the loader checks scalar response fields,
chat messages, and then any non-empty field ending in `_responses`. Set
`--all_responses true` to expand every sampled response rather than selecting
the first available one.

Reasoning enclosed in `<think>...</think>` is extracted automatically. If no
such region exists, the complete response is used. The trace is then split into
sentence-level blocks; fenced code is preserved as a single block.

## Training

### Hugging Face dataset

```bash
DATASET_NAME=org/reasoning-dataset \
MODEL_NAME=meta-llama/Llama-3.1-8B \
bash scripts/train_vae.sh
```

For a specific response column:

```bash
DATASET_NAME=org/reasoning-dataset \
RESPONSE_FIELD=model_responses \
bash scripts/train_vae.sh
```

### Local file

```bash
TRAIN_FILE=/path/to/train.jsonl \
bash scripts/train_vae.sh
```

Important environment overrides include:

```text
LATENT_DIM
NUM_LATENT_TOKENS
LATENT_NOISE_STD
TOKEN_SUBSTITUTION_PROB
BETA
ENCODER_TUNING
MAX_BLOCK_TOKENS
PER_DEVICE_BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS
NUM_EPOCHS
LEARNING_RATE
```

The launcher defaults to eight processes. Override `NUM_GPUS` when needed.
For a bounded streaming run, invoke `vae/train_vae.py` directly with
`--streaming true --max_source_samples N`.

## Tests

```bash
python -m unittest tests/test_data_vae.py
```

## Repository status

The VAE encoder, decoder, augmentations, and sentence-block data path are
aligned with the paper description. The downstream diffusion model in the
upstream repository still contains several hard-coded legacy latent dimensions;
those should be made configuration-driven before using a non-legacy VAE
checkpoint end to end.

## Citation

```bibtex
@article{kang2025ladir,
  title={LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning},
  author={Kang, Haoqiang and Zhang, Yizhe and Kuang, Nikki Lijing and Majamäki, Nicklas and Jaitly, Navdeep and Ma, Yi-An and Qin, Lianhui},
  journal={arXiv preprint arXiv:2510.04573},
  year={2025}
}
```
