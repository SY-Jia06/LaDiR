# LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning

Paper-aligned reproduction code for:

**[LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning](https://arxiv.org/abs/2510.04573)**  
Published at ICLR 2026.

LaDiR first trains a variational autoencoder (VAE) that maps one reasoning
sentence to a fixed block of continuous thought tokens. A latent diffusion
reasoner is then trained over these blocks.

## Installation

```bash
git clone https://github.com/SY-Jia06/LaDiR.git
cd LaDiR
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Access to the selected Hugging Face backbone is required. The paper recipe uses
`meta-llama/Llama-3.1-8B`.

## Data

Create `data/vae_train.jsonl` and, optionally, `data/vae_val.jsonl`:

```json
{"input": "question text", "output": "First reasoning sentence. Second reasoning sentence. The answer is: final answer."}
```

The VAE loader follows the paper's blockization procedure:

1. split CoT from the final answer with the literal prefix `The answer is`;
2. split the CoT into sentences;
3. train the VAE on each sentence as one independent latent block.

By default, examples missing the answer prefix raise an error instead of being
silently trained with a different data format.

## Train the VAE

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

The number of GPUs and paths can be overridden without editing the script:

```bash
NUM_GPUS=8 \
MODEL_NAME_OR_PATH=/path/to/Llama-3.1-8B \
OUTPUT_DIR=/path/to/vae_ckpt \
REPORT_TO=none \
bash scripts/train_vae.sh
```

`configs/cd_formal_8B_VAE_conn.yaml` records the same paper-aligned values for
experiments that use OmegaConf.

## Validation performed in this repository

The lightweight preprocessing tests do not download a model:

```bash
python -m unittest -v tests/test_vae_preprocessing.py
```

A full numerical reproduction still requires the paper dataset, the gated 8B
backbone, and multi-GPU training.

## Citation

```bibtex
@inproceedings{kang2026ladir,
  title={LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning},
  author={Kang, Haoqiang and Zhang, Yizhe and Kuang, Nikki Lijing and Majamaki, Nicklas and Jaitly, Navdeep and Ma, Yi-An and Qin, Lianhui},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```
