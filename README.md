# LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning

Paper-aligned reproduction code for:

**[LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning](https://arxiv.org/abs/2510.04573)**  
Published at ICLR 2026.

LaDiR first trains a variational autoencoder (VAE) that maps one reasoning
sentence to a fixed block of continuous thought tokens. A latent diffusion
reasoner is then trained over these blocks and generates the final answer with
the same causal-LM backbone.

## Installation

```bash
git clone https://github.com/SY-Jia06/LaDiR.git
cd LaDiR
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Access to the selected Hugging Face backbone is required. The paper recipe and
this audited implementation use `meta-llama/Llama-3.1-8B`; the custom hybrid
attention mask is run through Transformers' eager attention implementation.

## Data

Create `data/vae_train.jsonl` and, optionally, `data/vae_val.jsonl`:

```json
{"input": "question text", "output": "First reasoning sentence. Second reasoning sentence. The answer is: final answer."}
```

The VAE and reasoner preprocessing follow the paper's blockization procedure:

1. split CoT from the final answer with the literal prefix `The answer is`;
2. split the CoT into sentences;
3. encode every sentence as one independent latent block;
4. retain the ordered blocks, question, and final answer for reasoner training.

By default, examples missing the answer prefix raise an error instead of being
silently trained with a different data format. Sentence splitting is a
reproducible heuristic that protects decimals, common abbreviations, and
initialisms. Audit the real data before an expensive training run:

```bash
python scripts/audit_vae_blocks.py --input data/vae_train.jsonl --show 20
```

## Stage 0: train the VAE

```bash
bash scripts/train_vae.sh
```

The launcher defaults to the paper recipe:

- full-parameter fine-tuning of the LLM encoder;
- a separate frozen pretrained LLM decoder;
- latent dimension 512;
- four latent tokens per sentence block;
- KL weight `1e-5`;
- latent Gaussian augmentation with standard deviation 3;
- encoder-token substitution probability 0.3;
- learning rate `2e-5`, global batch size 128, two epochs.

The paper path accepts one pre-blockized sentence at a time. The release's
length-driven multi-segment mode is deliberately rejected because its repeated
memory-token interface does not match the paper decoder layout.

The number of GPUs and paths can be overridden without editing the script:

```bash
NUM_GPUS=8 \
MODEL_NAME_OR_PATH=/path/to/Llama-3.1-8B \
OUTPUT_DIR=/path/to/vae_ckpt \
REPORT_TO=none \
bash scripts/train_vae.sh
```

`configs/cd_formal_8B_VAE_conn.yaml` records the same paper-aligned values for
experiments that use OmegaConf. Its latent `scale_factor` and `shift_factor`
are retained release-checkpoint values: the paper does not explain how they
were estimated, so recompute and verify them before using a newly trained VAE.

## Precompute oracle latent blocks

The reasoner should not keep the two frozen 8B VAE copies resident during
training. Export the encoder outputs once into a memory-mapped array:

```bash
INPUT=data/vae_train.jsonl \
VAE_CHECKPOINT=checkpoints/vae/model.safetensors \
OUTPUT_DIR=data/reasoner_train \
bash scripts/precompute_reasoning_latents.sh
```

This writes `manifest.jsonl`, `latents.npy`, and `metadata.json`. The default
uses posterior samples, matching the VAE sampling formulation; set
`LATENT_MODE=mean` for a deterministic ablation.

## Stage 1: teacher-forcing reasoner training

```bash
bash scripts/train_reasoner_stage1.sh
```

The implementation jointly optimizes:

- blockwise flow matching on VAE latent tokens;
- autoregressive answer-token cross entropy;
- a binary `<BOT>`/`<SOA>` stopping loss at every `<EOT>` state.

The hybrid attention mask is causal across blocks and bidirectional within each
current latent block. `configs/reasoner_stage1.yaml` records the paper's main
weights (`lambda_flow=5`, `lambda_answer=1`, `lambda_special=1`), learning rate
`1e-5`, 20 epochs, and global batch size 64. The launcher computes gradient
accumulation from `NUM_GPUS`, `PER_DEVICE_BATCH`, and `GLOBAL_BATCH_SIZE`.

## Stage 2: differentiable rollout training

```bash
bash scripts/train_reasoner_stage2.sh
```

Stage 2 initializes from the Stage-1 checkpoint, keeps the ground-truth number
of blocks, replaces oracle context blocks with self-generated blocks, and
backpropagates answer supervision through a 10-step Euler denoising trajectory.
The flow-matching loss remains active to reduce latent collapse. The default
launcher uses 4 GPUs so batch 12 is represented exactly; override the launcher
variables for another topology.

## Inference API

`LaDiRReasoner.generate()` first denoises variable-length latent blocks until
the stopping head predicts `<SOA>`, then generates answer tokens
autoregressively. It exposes denoising steps, initial-noise scale,
classifier-free guidance, and diversity guidance as inference arguments.

## Validation performed in this repository

The lightweight tests do not download a model:

```bash
python -m unittest -v \
  tests/test_vae_preprocessing.py \
  tests/test_vae_training_utils.py \
  tests/test_vae_model_interface.py \
  tests/test_flow_scheduler.py \
  tests/test_reasoner_dataset.py \
  tests/test_reasoner_model.py
```

The reasoner suite covers exact scheduler conversions, latent-store collation,
the blockwise attention contract, joint Stage-1 losses, and differentiable
Stage-2 rollout with a mocked causal LM. Python files compile and shell
launchers pass `bash -n`.

A full numerical reproduction still requires the paper dataset, the gated 8B
backbone, the trained VAE, and multi-GPU training. Passing the lightweight tests
does not establish the published benchmark numbers.

## Reproduction audit

See `REPRODUCTION_NOTES.md` for the paper-to-release discrepancy table, the
four-versus-six latent-token ambiguity, reasoner implementation choices, and
the boundaries of what can be verified without the original training run.

## Citation

```bibtex
@inproceedings{kang2026ladir,
  title={LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning},
  author={Kang, Haoqiang and Zhang, Yizhe and Kuang, Nikki Lijing and Majamaki, Nicklas and Jaitly, Navdeep and Ma, Yi-An and Qin, Lianhui},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```
