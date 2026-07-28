# Format-Constrained Knowledge Distillation (FC-KD) for SLMs

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-b31b1b.svg)](#citation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official repository for:**
> *Format-Constrained Knowledge Distillation for Topology-Preserving Knowledge Graph Reasoning in Small Language Models*
> Olusegun Oladele Odewole, Seren Başaran — Near East University

## Overview

This repository implements **Format-Constrained Sequence-Level Distillation (FC-KD)**, a training and evaluation pipeline studying the *generative bottleneck* in Small Language Models (SLMs) reasoning over structured biomedical knowledge graphs.

Standard autoregressive fine-tuning forces a model to simultaneously learn graph topology and string-level surface form, which we show degrades structural fidelity even when the underlying relational knowledge is well-represented internally. This pipeline trains a 1.3B-parameter student (`microsoft/phi-1_5`) via a three-stage parallel curriculum on **PrimeKG**, then evaluates zero-shot generalization on **PharmGKB**, a structurally distinct pharmacogenomic graph the model never sees during training.

### Core Capabilities
- **Training curriculum:** Structural SFT → Topology-Anchored ORPO / Format-Constrained KD (parallel branches from a shared SFT checkpoint), all under 4-bit NF4 + LoRA.
- **Cross-domain evaluation:** Dual-modality zero-shot testing — generative extraction (Ontological Validity, Strict Exact Match) and discriminative adversarial ranking — measured on a graph never seen in training.

---

## Quickstart

The pipeline has five stages, run one at a time by editing `ACTIVE_STAGE` in `config.py`. Two of them (Stage 3 and Stage 3b) need a dataset produced by a separate generation script — generate those before touching the numbered stages. Running a stage before its dependency exists stops with an error naming the missing artifact and the command needed to produce it.

| Step | What to run | What it produces |
|---|---|---|
| 1 | `export HF_HUB_USERNAME=your-username` | — sets where all models/datasets are read from and pushed to |
| 2 | `python data/black_box_distill.py` | `primekg-phi3-distilled-sft-seed{N}` — required for Stage 3 |
| 3 | `python data/distill_vanilla_kd.py` | `primekg-phi3-vanillakd-sft-seed{N}` — required for Stage 3b |
| 4 | `ACTIVE_STAGE = 1` → run the pipeline | `phi-1.5-primekg-sft-seed{N}` (SFT checkpoint) |
| 5 | `ACTIVE_STAGE = 2` → run the pipeline | `phi-1.5-primekg-orpo-seed{N}` (ORPO checkpoint) |
| 6 | `ACTIVE_STAGE = 3` → run the pipeline | `phi-1.5-primekg-dkd-seed{N}` (FC-KD checkpoint) |
| 7 | `ACTIVE_STAGE = "3b"` → run the pipeline | `phi-1.5-primekg-vanillakd-seed{N}` (ablation checkpoint) |
| 8 | `ACTIVE_STAGE = 4` → run the pipeline | Evaluation metrics in `EVAL_CONFIG["output_dir"]` |

Steps 4–7 each need to be repeated once per seed (`42`, `101`, `999`) by changing `CONFIG["RUN_SEED"]` between runs, to reproduce the paper's three-seed averages. Stage 3 and 3b both branch from the Stage 1 checkpoint, so Stage 1 must complete first; Stage 4 evaluation needs whichever checkpoints you want to evaluate to already exist for that seed.

"Run the pipeline" means `./run_pipeline.sh` (local) or the equivalent `docker run` command (Docker) — see below for exact commands for each.

W&B logging is optional. If `WANDB_API_KEY` is not set, the pipeline runs with local-only logging and no W&B account is required.

---

## Cross-Domain Evaluation Protocol

**Training and evaluation use two different, non-overlapping knowledge graphs.**

| Dataset | Role | Exposure |
|---|---|---|
| **PrimeKG** | Training (Stages 1–3) | Used for all fine-tuning; the student model only ever sees PrimeKG triples during training. |
| **PharmGKB** | Evaluation (Stage 4) | Held out entirely. Zero PharmGKB entities, relations, or naming conventions are seen during any training stage. |

This is a deliberate design choice, not an accidental default. Evaluating a PrimeKG-trained model on PrimeKG's own held-out split only tests memorization of one graph's local vocabulary. Evaluating on PharmGKB instead tests whether the *relational competence itself* — the model's latent ability to distinguish topologically valid from invalid structure — transfers to an unseen graph schema. This is what `EVAL_MODE="cross_domain"` in `config.py` runs, and it is the setting used for every number reported in the manuscript.

An in-domain `EVAL_MODE="standard"` (held-out PrimeKG split) is also implemented, but exists only as an internal sanity check — it is **not** used for any reported result.

Because PharmGKB's entity-naming conventions differ from PrimeKG's, some fraction of Strict Exact Match failures reflect cross-graph lexical shift as well as pure surface-form generation failure. See the paper's Limitations section for a full discussion of this confound.

---

## Repository Structure

```text
fc-kd-repo/
│
├── data/                     # Autogenerated cache for PrimeKG / PharmGKB downloads
│
├── config.py                 # Training + evaluation hyperparameters, seeds, environment setup
├── data_utils.py              # PrimeKG training-data loading, tokenization, SFT/ORPO alignment
├── models.py                  # 4-bit NF4 quantization + LoRA/PEFT model construction
├── trainers.py                 # Stage 1 (SFT), Stage 2 (ORPO), Stage 3 (FC-KD) training classes
├── evaluate.py                 # Stage 4: dual-modality cross-domain evaluation (PharmGKB)
├── main.py                     # Single entrypoint routing all four stages
│
├── run_pipeline.sh             # Environment verification + execution wrapper
├── requirements.txt            # Pinned dependencies
└── README.md
```

---

## Setup & Installation

Two ways to run this pipeline: Docker (recommended, for reproducible FlashAttention compilation and no CUDA/PyTorch dependency conflicts) or a local Conda environment. Pick one and follow it end-to-end — the steps below are self-contained per option.

### Option A: Docker

**1. Build the image.** This installs all system dependencies, the correct PyTorch wheel, and compiles FlashAttention for the target architecture. The container is already set up to execute `run_pipeline.sh` on its own, so no manual `chmod` is needed on this path.

```bash
docker build -t format-constrained-kd .
```

**2. Set your Hugging Face username** (used to build every model/dataset path the pipeline reads from or pushes to):

```bash
export HF_HUB_USERNAME=your-username-here
```

**3. Generate the Stage 3 / 3b datasets.** Stages 3 and 3b train on teacher-model completions rather than on PrimeKG directly, so these need to exist before you run those stages:

```bash
# Vanilla-KD teacher-generation script:
docker run --gpus all -it --rm \
  -e HF_TOKEN="your_hf_token_here" \
  -e WANDB_API_KEY="your_wandb_token_here" \
  -e HF_HUB_USERNAME="$HF_HUB_USERNAME" \
  -v $(pwd):/workspace \
  format-constrained-kd bash run_pipeline.sh data/distill_vanilla_kd.py

# FC-KD teacher-generation script:
docker run --gpus all -it --rm \
  -e HF_TOKEN="your_hf_token_here" \
  -e WANDB_API_KEY="your_wandb_token_here" \
  -e HF_HUB_USERNAME="$HF_HUB_USERNAME" \
  -v $(pwd):/workspace \
  format-constrained-kd bash run_pipeline.sh data/black_box_distill.py
```

Each script pushes its output dataset to `{HF_HUB_USERNAME}/...` on the Hub. If you already have access to someone else's copy of these datasets, point to their account instead of generating your own: `-e SYNTHETIC_DATA_OWNER="their-account-name"`.

**4. Set `ACTIVE_STAGE` in `config.py`** to the stage you want to run (see the Quickstart table above), then run the main pipeline:

```bash
docker run --gpus all -it --rm \
  -e HF_TOKEN="your_hf_token_here" \
  -e WANDB_API_KEY="your_wandb_token_here" \
  -e HF_HUB_USERNAME="$HF_HUB_USERNAME" \
  -v $(pwd):/workspace \
  format-constrained-kd bash run_pipeline.sh
```

`WANDB_API_KEY` is optional — omit it to run with local-only logging. `-v $(pwd):/workspace` live-mounts your working directory, so edits to `config.py` on the host (e.g. changing `ACTIVE_STAGE` or `CONFIG["RUN_SEED"]`) take effect on the next run without rebuilding the image. PrimeKG and PharmGKB are downloaded and cached to `./data` automatically on first use.

### Option B: Local Conda Environment

**1. Create the environment and install dependencies:**

```bash
conda create -n fc-kd python=3.10 -y
conda activate fc-kd

# Fix for "No module named 'pkg_resources'"
pip install "setuptools<70.0.0" wheel packaging ninja

# Install PyTorch 2.7.0 strictly compiled for CUDA 12.8 (+cu128)
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install baseline requirements
pip install -r requirements.txt

# Pin core AI toolchains (grouped together so pip cannot silently downgrade them)
pip install transformers==4.57.6 trl==0.11.4 peft==0.19.1 accelerate==1.13.0

# Install Flash Attention
pip install flash-attn==2.8.3 --no-build-isolation
```

**2. Make the wrapper script executable** (only needed once — this is a host-side permission, not something Docker needs since the image sets it during build):

```bash
chmod +x run_pipeline.sh
```

**3. Set your environment variables:**

```bash
export HF_HUB_USERNAME=your-username-here
export HF_TOKEN=your_hf_token_here
export WANDB_API_KEY=your_wandb_token_here   # optional — omit to run with local-only logging
```

**4. Generate the Stage 3 / 3b datasets** (required before running those stages; skip if you already have them or are pointing to `SYNTHETIC_DATA_OWNER`):

```bash
./run_pipeline.sh data/black_box_distill.py     # produces the Stage 3 (FC-KD) dataset
./run_pipeline.sh data/distill_vanilla_kd.py    # produces the Stage 3b (Vanilla-KD) dataset
```

**5. Set `ACTIVE_STAGE` in `config.py`** to the stage you want to run (see the Quickstart table above), then run the pipeline:

```bash
./run_pipeline.sh
```

PrimeKG and PharmGKB are downloaded and cached to `./data` automatically on first use.

---

## Reproducibility Notes

- **Seeds:** All reported results are averaged over three independent seeds (`42`, `101`, `999`), set via `CONFIG["RUN_SEED"]` (training) and `EVAL_CONFIG["SEEDS_TO_EVAL"]` (evaluation).
- **Ground-truth normalization:** Strict Exact Match comparisons normalize both the predicted and ground-truth strings identically (`.strip().lower()`) before comparison, avoiding false mismatches from incidental whitespace or casing differences in the source data.
- **Attention implementation:** `flash_attention_2` is selected automatically when the host GPU reports compute capability ≥ 8.0 (`models.get_attn_impl()`), falling back to eager attention otherwise. This is applied consistently across all three training stages and affects throughput only, not the numerical results reported in the paper.
- **Stage branching:** Both Stage 2 (ORPO) and Stage 3 (FC-KD) load their base adapter from the **Stage 1** checkpoint, not from each other — this is intentional, to keep the two optimization branches isolated for the ablation comparison (see paper Section 3.1).

---

## Memory Profile & Hardware Requirements

| Model | Size | Configuration | Max Sequence | Peak VRAM |
|---|---|---|---|---|
| `microsoft/phi-1_5` | 1.3B | 4-bit NF4 + LoRA (r=32) | 512 | ~4.8 GB |
| `microsoft/Phi-3.5-mini-instruct` (control) | 3.8B | 4-bit NF4 | 512 | ~7.2 GB |

Reference hardware: 1× NVIDIA RTX 3090 Ti (24GB VRAM), 64GB system RAM, Ubuntu 22.04 LTS.

---

## Citation

```bibtex
@article{odewole2026formatconstrained,
  title={Format-Constrained Knowledge Distillation for Topology-Preserving Knowledge Graph Reasoning in Small Language Models},
  author={Odewole, Olusegun Oladele and Ba{\c{s}}aran, Seren},
  journal={Under review},
  year={2026},
  institution={Near East University, Faculty of Artificial Intelligence and Informatics}
}
```

*(Update with the final venue and DOI upon acceptance.)*

## Disclaimer

The data, models, and scripts in this repository are for **academic and computational research evaluation only**. They are not intended for clinical use, diagnostic applications, or real-world medical decision-making. Cross-domain evaluation results reflect PrimeKG-to-PharmGKB generalization only and should not be interpreted as validated performance on any specific downstream pharmacogenomic application.

## License

MIT — see `LICENSE`.
