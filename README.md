# Format-Constrained Knowledge Distillation (FC-KD) for SLMs

[![Paper](https://img.shields.io/badge/Paper-Knowledge--Based%20Systems-0066cc.svg)](https://doi.org/10.1016/j.knosys.2026.117372)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Official repository for:**
> *Evaluating format-constrained knowledge distillation for knowledge graph reasoning in small language models: A controlled cross-graph study*
> Olusegun Oladele Odewole, Seren Başaran — Near East University
> Published in *Knowledge-Based Systems* (2026), Vol. 117372. DOI: [10.1016/j.knosys.2026.117372](https://doi.org/10.1016/j.knosys.2026.117372)

## Overview

This repository implements the **Format-Constrained Knowledge Distillation (FC-KD)** pipeline, a training and evaluation framework for studying the *generative bottleneck* in Small Language Models (SLMs) reasoning over structured biomedical knowledge graphs.

Two student backbones — `microsoft/phi-1_5` (1.3B) and `Qwen/Qwen2.5-1.5B` (1.5B, cross-model validation) — are trained on **PrimeKG** through a parallel curriculum and evaluated zero-shot on **PharmGKB**, a structurally distinct pharmacogenomic graph never seen in training, as well as on a PrimeKG evaluation pool.

### Core Capabilities
- **Training curriculum:** Structural SFT → Topology-Anchored ORPO / FC-KD / Vanilla-KD (parallel branches from a shared SFT checkpoint), all under 4-bit NF4 + LoRA, plus two matched control arms that isolate what the FC-KD stage adds.
- **Dual-modality evaluation:** generative extraction (template compliance, Ontological Validity, Strict Exact Match) and discriminative adversarial ranking, cross-domain (PharmGKB), in-domain (PrimeKG), and on a held-out PrimeKG set that excludes every training edge.
- **Additional analyses:** paired significance tests over five seeds, noise robustness (four noise types), constrained decoding, error decomposition, trained-vs-unseen edges, an incremental-update pilot, Vanilla-KD corpus statistics, and inter-rater / LLM-judge validation.

---

## What the FC-KD Stage Actually Trains On (read before interpreting results)

An audit of the released code and datasets (`audit_fckd_corpus.py`) established the following. The paper's method section reflects these facts.

- **FC-KD corpus composition.** `data/black_box_distill.py` asks the teacher (`microsoft/Phi-3-mini-4k-instruct`) for the target of each PrimeKG edge. A teacher output is kept only if it parses and its entity name substring-matches the ground-truth target; otherwise the **ground-truth target is substituted**. Across all five seeds, ~95.0% of FC-KD training targets are identical to the ground truth, ~4.9% are teacher name variants accepted by the substring filter, and <0.1% differ only in type.
- **Prompts.** The FC-KD student text uses the **same prompt as evaluation** (`"Task: Analyze the biomedical relationship and output ONLY ..."`). SFT, ORPO, and Vanilla-KD train on the Stage 1 prompt (`"Analyze the biomedical relationship.\n..."`).
- **Vanilla-KD** trains on free-form teacher answers (the teacher is asked to explain its reasoning), with no ground-truth fallback and the Stage 1 prompt. It therefore differs from FC-KD in label correctness, target format, and prompt — not in format alone.
- **Same rows.** SFT, ORPO, FC-KD, Vanilla-KD, and both control arms draw the identical per-relation PrimeKG sample for a given seed, and SFT, FC-KD, and the control arms train on the identical 90% split.
- **Control arms** (`train_control_arms.py`) exist to separate these factors; see below.

---

## Quickstart

Training stages are selected with `ACTIVE_STAGE` in `config.py` and run through `main.py` / `run_pipeline.sh`. The backbone and evaluation mode can be set per command with environment variables, so `config.py` does not need editing between backbones:

```bash
export HF_HUB_USERNAME=your-username
export STUDENT_NAME="microsoft/phi-1_5"      # or "Qwen/Qwen2.5-1.5B"
export EVAL_MODE=cross_domain                # or "standard"
```

### A. Training (once per backbone)

| Step | What to run | What it produces |
|---|---|---|
| 1 | `python data/black_box_distill.py` | `primekg-phi3-distilled-sft-seed{N}` — FC-KD corpus (backbone-independent) |
| 2 | `python data/distill_vanilla_kd.py` | `primekg-phi3-vanillakd-sft-seed{N}` — Vanilla-KD corpus (backbone-independent) |
| 3 | `ACTIVE_STAGE = 1` → run the pipeline | `{student}-primekg-sft-seed{N}` |
| 4 | `ACTIVE_STAGE = 2` → run the pipeline | `{student}-primekg-orpo-seed{N}` |
| 5 | `ACTIVE_STAGE = 3` → run the pipeline | `{student}-primekg-dkd-seed{N}` (FC-KD) |
| 6 | `ACTIVE_STAGE = "3b"` → run the pipeline | `{student}-primekg-vanillakd-seed{N}` |
| 7 | `python train_control_arms.py --arm evalprompt` | `{student}-primekg-ctl-evalprompt-seed{N}` |
| 8 | `python train_control_arms.py --arm trainprompt` | `{student}-primekg-ctl-trainprompt-seed{N}` |

Stages 2, 3, 3b and both control arms branch from the Stage 1 checkpoint, so Stage 1 must finish first. Seeds come from `CONFIG["RUN_SEEDS"]` (`42, 101, 999, 7, 2024`); `train_control_arms.py` loops over them automatically (use `--seeds` to restrict). Check `main.py` for how the numbered stages iterate over seeds in your version.

### B. Evaluation and analysis

`run_revision_pipeline.sh` runs the main evaluations and analyses for both backbones in order:

```bash
bash run_revision_pipeline.sh 2>&1 | tee pipeline.log
```

| Step | Command | Meaning |
|---|---|---|
| 0 | (automatic) | Moves old `eval_outputs*` folders **outside** the project so aggregation cannot mix stale reports with new ones |
| 1 | `train_control_arms.py --arm evalprompt / trainprompt` | Control arms (skip if already trained) |
| 2 | `EVAL_MODE=cross_domain python evaluate.py` | PharmGKB evaluation: 7 arms × 5 seeds |
| 3 | `EVAL_MODE=standard python evaluate.py` | PrimeKG in-domain pool: 7 arms × 5 seeds |
| 4 | `python evaluate_expanded_noise_robustness.py` | Four noise types × 7 arms × 5 seeds |
| 5 | `python evaluate_constrained_decoding_baseline.py` | Template-constrained decoding on SFT and FC-KD |
| 6 | `python aggregate_and_test_all_results.py .` | Averaged tables, paired tests, Vanilla-KD breakdown → `stage12_output.txt` |
| 7 | `python format_compliance_analysis.py` | Template compliance, error decomposition, paired tests |
| 8 | `python seen_unseen_standard_analysis.py` | Trained vs. unseen edges within the in-domain pool |

Further experiments, run separately (per backbone where a GPU is used):

| Command | Meaning |
|---|---|
| `EVAL_MODE=heldout python evaluate.py` | Held-out PrimeKG: each seed's exact training sample and all reverse edges are excluded before sampling |
| `python pilot_incremental_update.py` | Incremental-update pilot (acquisition / retention / transfer); resumes after interruption |
| `python vanillakd_corpus_stats.py` | Vanilla-KD teacher-corpus statistics vs. student template retention (CPU) |
| `python audit_fckd_corpus.py` | FC-KD corpus composition audit (CPU) |
| `python llm_judge_factual_correctness.py` | LLM-judge refinement of factual correctness (GPT-4o by default; needs `OPENAI_API_KEY`) |
| `python make_manuscript_figures.py` | Robustness and error-decomposition figures from the evaluation CSVs |
| `python figure_scripts/make_figure0_overview.py`, `make_figure1_flowchart.py`, `make_graphical_abstract.py` | Conceptual overview, architecture diagram, graphical abstract |

Steps 2–5 and the held-out evaluation run once per backbone. A complete main run produces **440** report files (per backbone: 7 arms × 5 seeds × 6 evaluation modes, plus 10 constrained-decoding reports); the held-out evaluation adds 70.

**Speed without changing results.** `evaluate.py` loads the base model once and swaps LoRA adapters (`EVAL_CONFIG["reuse_base_model"]`), computes the ranking loss in chunks to limit memory, and can be split across processes or GPUs:

```bash
EVAL_SHARD=0/2 CUDA_VISIBLE_DEVICES=0 python evaluate.py &
EVAL_SHARD=1/2 CUDA_VISIBLE_DEVICES=1 python evaluate.py
```

In Jupyter, `!cmd &` is not supported; use a `%%bash` cell with `&` and `wait`. Do **not** change `batch_size` to gain speed: adversarial-trap sampling is keyed to batch position, so every arm, seed, backbone, and mode must use the same value (24). Set `force_rerun: False` to resume an interrupted evaluation without regenerating completed files.

W&B logging is optional. If `WANDB_API_KEY` is not set, the pipeline runs with local-only logging.

## Evaluation Protocol

| Dataset | Role | Exposure |
|---|---|---|
| **PrimeKG** | Training; in-domain evaluation (`EVAL_MODE="standard"`) | Training uses up to 2,000 edges per relation per seed. |
| **PrimeKG, held-out** | Held-out evaluation (`EVAL_MODE="heldout"`) | Every training-sample edge and its reverse excluded per seed. |
| **PharmGKB** | Cross-domain evaluation (`EVAL_MODE="cross_domain"`) | Never seen in training. |

**Cross-domain** evaluation tests whether relational competence transfers to an unseen graph schema. Because PharmGKB's entity names differ from PrimeKG's, extraction failures there mix generative failure with cross-graph vocabulary shift.

**In-domain** evaluation draws up to 10,000 PrimeKG edges per relation with the same seed as training, so the 2,000-edge training sample is contained in this pool. Of the 500 evaluated edges per seed, ~27% were in the training split and ~70% were never sampled for training; `seen_unseen_standard_analysis.py` reports results for each group separately. 

**Held-out** evaluation (`EVAL_MODE="heldout"`) removes, for each seed, the exact training sample shared by all arms together with the reverse direction of every sampled edge, and then samples 500 evaluation rows with the same relation-stratified procedure, so no evaluated edge was available to any training arm.

**Metrics.**
- *Template compliance:* output matches `[Name] (Type: [Class])` with a non-empty name.
- *Ontological Validity (OV):* reported as an F1; an output is valid if its entity exists in the evaluation graph and its stated type (when present) matches. Valid outputs count as true positives whether or not they are the ground-truth target.
- *Strict Exact Match (SEM):* F1 against the ground-truth target (case-insensitive).
- *Ranking Accuracy:* the model scores the ground-truth target against a same-type factual trap, a wrong-type topology trap, and a non-existent entity; chance is 25%.

**Caveat.** PharmGKB includes "not associated" and "ambiguous" relations, for which the listed target is not a uniquely correct completion. All metrics treat the listed target as ground truth for every relation type.

---

## Control Arms

`train_control_arms.py` trains two arms that are identical to Stage 3 (FC-KD) in branch point (Stage 1 adapter), learning rate, epochs, batch size, maximum length, rows, and split — only the training data differ:

| Arm | Training data | Contrast | Isolates |
|---|---|---|---|
| `ctl-evalprompt` | Ground-truth targets, evaluation prompt | FC-KD − ctl-evalprompt | Contribution of the teacher |
| `ctl-trainprompt` | Ground-truth targets, Stage 1 prompt | ctl-evalprompt − ctl-trainprompt | Training on the evaluation prompt |
| | | ctl-trainprompt − SFT | Three extra epochs on the same rows |

Datasets are pushed to `{HF_HUB_USERNAME}/primekg-gt-{arm}-seed{N}` and adapters to `{student}-primekg-ctl-{arm}-seed{N}`. Arm names avoid the substrings `sft`, `dkd`, `orpo`, `vanillakd`, and `control`, so filename-based arm detection in the analysis scripts cannot misclassify them. Set `CONFIG["INCLUDE_CONTROL_ARMS"] = False` to evaluate only the original arms.

---

## Extended Analysis Scripts

- **`evaluate_constrained_decoding_baseline.py`:** applies a template-constrained decoder (forces the `(Type: …)` scaffold and end-of-sequence) to the SFT and FC-KD checkpoints. The constraint acts only on generation, so Ranking Accuracy must equal the unconstrained Cross-Domain value for the same arm and seed — a built-in integrity check. It constrains syntax only, not entity vocabulary.
- **`evaluate_expanded_noise_robustness.py`:** four noise types, each evaluated separately — character swap, entity-type corruption, predicate corruption (a different real relation), and invalid source/target (source replaced with a graph entity of a different type; a best-effort approximation). Noise uses a dedicated RNG keyed to seed and batch position, and trap options match the clean evaluation, so differences isolate the noise. Corrupted inputs are logged per row (`Noisy_*` columns).
- **`aggregate_and_test_all_results.py`:** rescoring of OV from CSVs, the averaged results table for all arms and modes, exact paired Wilcoxon tests with bootstrap CIs (FC-KD vs SFT, ORPO, Vanilla-KD, Ctl-EvalPrompt; Ctl-EvalPrompt vs Ctl-TrainPrompt; Ctl-TrainPrompt vs SFT), and the Vanilla-KD per-seed breakdown. Supersedes `run_significance_tests.py`.
- **`format_compliance_analysis.py`:** template compliance and the mutually exclusive decomposition (syntactically invalid / valid syntax, invalid ontology / ontologically valid), per arm, seed, backbone, and mode, with paired tests; also tabulates LLM-judge verdicts by relation from existing audit files.
- **`seen_unseen_standard_analysis.py`:** rebuilds each seed's training split exactly as `data_utils.py` does and splits the in-domain results into trained, sampled-not-trained, and unseen edges, with paired tests. Serves as the controlled version of the incremental-knowledge pilot.
- **`audit_fckd_corpus.py`:** reports, per seed, how many FC-KD training targets equal the ground truth.
- **`llm_judge_factual_correctness.py`:** judges whether ontologically valid, non-matching outputs are correct alternative completions. Uses an independent OpenAI judge by default (`OPENAI_JUDGE_MODEL`, default `gpt-4o`; responses cached in `llm_judge_outputs/`), or the teacher with `JUDGE_BACKEND=local_teacher`. Verdicts are informative mainly for "associated" relations; the paper reports factual correctness as an upper bound.
- **`analyze_vanillakd_variance.py`:** per-seed Vanilla-KD error-class, ranking, and output-length breakdown for one backbone (its "OV" line counts `tp` only; the aggregator's Step 4 gives the corrected `tp + fp_fn` rate for both backbones).
- **`pilot_incremental_update.py`:** incremental-update pilot. A trained checkpoint (FC-KD and SFT by default) is fine-tuned for three epochs on about 480 new PrimeKG edges outside its training sample; an update set, an old set (original training edges), and a separate held-out set are evaluated before and after through the `Evaluator` pipeline, measuring acquisition, retention, and transfer. Saves after every (arm, seed) and resumes.
- **`vanillakd_corpus_stats.py`:** per-seed statistics of the Vanilla-KD teacher corpus (template-conforming first lines, explanatory text, length, completions containing a correct target) related to the students' template retention.
- **`make_manuscript_figures.py`:** robustness and error-decomposition figures computed directly from the evaluation CSVs, with an explicit folder per evaluation condition.

---

## Repository Structure

```text
fc-kd-repo/
│
├── data/
│   ├── black_box_distill.py            # FC-KD corpus: teacher + ground-truth filter and fallback
│   ├── distill_vanilla_kd.py           # Vanilla-KD corpus: free-form teacher answers (+ generation stats)
│   ├── relationships.zip               # PharmGKB relationships (auto-downloaded if absent)
│   └── CREATED_2026-07-05.txt, info.txt, LICENSE.txt, README.pdf   # PharmGKB release files
│
├── config.py                  # Hyperparameters, seeds, arms, evaluation settings (STUDENT_NAME / EVAL_MODE env overrides)
├── data_utils.py              # PrimeKG sampling and SFT/ORPO dataset construction
├── models.py                  # 4-bit NF4 quantization + LoRA/PEFT construction
├── trainers.py                # Stage 1 (SFT), 2 (ORPO), 3 (FC-KD), 3b (Vanilla-KD)
├── train_control_arms.py      # Ctl-EvalPrompt / Ctl-TrainPrompt datasets + training
├── main.py                    # Entry point routing ACTIVE_STAGE
│
├── evaluate.py                                # Dual-modality evaluation (cross_domain / standard / heldout; NF4; adapter reuse; EVAL_SHARD)
├── evaluate_expanded_noise_robustness.py      # Four noise types
├── evaluate_constrained_decoding_baseline.py  # Template-constrained decoding on SFT / FC-KD
├── pilot_incremental_update.py                # Incremental-update pilot (acquisition / retention / transfer)
├── llm_judge_factual_correctness.py           # LLM-judge refinement of factual correctness
├── run_indomain_ablation.py                   # In-domain vs cross-domain side-by-side
│
├── aggregate_and_test_all_results.py  # Averaged tables + paired tests (all arms, both backbones, all modes)
├── format_compliance_analysis.py      # Template compliance + error decomposition + paired tests
├── seen_unseen_standard_analysis.py   # Trained vs unseen edges in the in-domain pool
├── vanillakd_corpus_stats.py          # Vanilla-KD corpus statistics vs template retention
├── audit_fckd_corpus.py               # FC-KD corpus composition audit
├── analyze_vanillakd_variance.py      # Vanilla-KD per-seed breakdown (single backbone)
├── make_manuscript_figures.py         # Robustness + error-decomposition figures from CSVs
│
├── figure_scripts/
│   ├── make_figure0_overview.py       # Conceptual overview (paper Figure 1)
│   ├── make_figure1_flowchart.py      # Training arms and evaluation (paper Figure 2)
│   └── make_graphical_abstract.py     # Graphical abstract
│
├── run_revision_pipeline.sh   # Main evaluation + analysis run, both backbones
├── run_pipeline.sh            # Environment verification + ACTIVE_STAGE execution wrapper
├── Dockerfile
├── requirements.txt
└── README.md
```

Generated at run time (not versioned): `eval_outputs/`, `eval_outputs_noise_<type>/`, `eval_outputs_constrained_decoding_<tag>/`, held-out reports (`*_heldout_*` in the evaluation output folder), `format_compliance_outputs/`, `seen_unseen_outputs/`, `llm_judge_outputs/`, `pilot_incremental_update_outputs/`, `Final_Metrics_Averaged.csv`, `Raw_Metrics_Long.csv`, `Manuscript_Figure_Data.csv`, `vanillakd_corpus_stats.csv`.

Superseded scripts kept out of the release: `run_significance_tests.py` (replaced by `aggregate_and_test_all_results.py`), `pilot_incremental_kg_update.py` (replaced by `pilot_incremental_update.py`), `end_to_end_ranking_workflow.py`, `compare_pilot_to_indomain.py`, `test_pilot_degradation_significance.py`, `measure_ranking_noise.py`, `evaluate_segestic_partial*.py`.

## Setup & Installation

Two ways to run this pipeline: Docker (recommended, for reproducible FlashAttention compilation and no CUDA/PyTorch dependency conflicts) or a local Conda environment.

### Option A: Docker

**1. Build the image.**

```bash
docker build -t format-constrained-kd .
```

**2. Set your Hugging Face username:**

```bash
export HF_HUB_USERNAME=your-username-here
```

**3. Generate the Stage 3 / 3b datasets:**

```bash
docker run --gpus all -it --rm \
  -e HF_TOKEN="your_hf_token_here" \
  -e HF_HUB_USERNAME="$HF_HUB_USERNAME" \
  -v $(pwd):/workspace \
  format-constrained-kd bash run_pipeline.sh data/black_box_distill.py

docker run --gpus all -it --rm \
  -e HF_TOKEN="your_hf_token_here" \
  -e HF_HUB_USERNAME="$HF_HUB_USERNAME" \
  -v $(pwd):/workspace \
  format-constrained-kd bash run_pipeline.sh data/distill_vanilla_kd.py
```

To use someone else's copy of these datasets instead, add `-e SYNTHETIC_DATA_OWNER="their-account-name"`.

**4. Run a training stage** (set `ACTIVE_STAGE` in `config.py`; add `-e STUDENT_NAME=...` to select the backbone):

```bash
docker run --gpus all -it --rm \
  -e HF_TOKEN="your_hf_token_here" \
  -e HF_HUB_USERNAME="$HF_HUB_USERNAME" \
  -e STUDENT_NAME="microsoft/phi-1_5" \
  -v $(pwd):/workspace \
  format-constrained-kd bash run_pipeline.sh
```

**5. Run the evaluation and analysis:**

```bash
docker run --gpus all -it --rm \
  -e HF_TOKEN="your_hf_token_here" \
  -e HF_HUB_USERNAME="$HF_HUB_USERNAME" \
  -v $(pwd):/workspace \
  format-constrained-kd bash run_revision_pipeline.sh
```

`WANDB_API_KEY` is optional. `-v $(pwd):/workspace` live-mounts the working directory, so edits to `config.py` take effect without rebuilding. PrimeKG and PharmGKB are cached to `./data` on first use.

### Option B: Local Conda Environment

```bash
conda create -n fc-kd python=3.10 -y
conda activate fc-kd

# Fix for "No module named 'pkg_resources'"
pip install "setuptools<70.0.0" wheel packaging ninja

# PyTorch 2.7.0 for CUDA 12.8
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt

# Pin core toolchains together so pip cannot silently downgrade them
pip install transformers==4.57.6 trl==0.11.4 peft==0.19.1 accelerate==1.13.0

pip install flash-attn==2.8.3 --no-build-isolation

chmod +x run_pipeline.sh

export HF_HUB_USERNAME=your-username-here
export HF_TOKEN=your_hf_token_here
export WANDB_API_KEY=your_wandb_token_here   # optional

./run_pipeline.sh data/black_box_distill.py
./run_pipeline.sh data/distill_vanilla_kd.py
./run_pipeline.sh                            # training stage set by ACTIVE_STAGE
bash run_revision_pipeline.sh                # evaluation + analysis
```

---

## Reproducibility Notes

- **Seeds.** All results are reported over five seeds (`42, 101, 999, 7, 2024`; `CONFIG["RUN_SEEDS"]`, `EVAL_CONFIG["SEEDS_TO_EVAL"]`). `main.py` calls `set_global_seed(seed)` before training, so each seed determines the training sample and split, the evaluation sample, and the LoRA adapter initialization. The trainers' own seed (example shuffling and dropout) is left at its default (42); the training procedure is otherwise identical across seeds.
- **Determinism.** `evaluate.py` enables deterministic PyTorch/cuDNN algorithms and sets `CUBLAS_WORKSPACE_CONFIG`. Adversarial traps and robustness noise are drawn from dedicated RNGs keyed to `(seed, batch start index)`, and ontology lists are sorted, so a given (arm, seed) evaluates identically in isolation and inside a full run.
- **Batch size.** Evaluation uses `batch_size = 24` for every arm, seed, backbone, and mode. Changing it changes which traps each example receives and slightly changes GPU numerics, so results at different batch sizes are not comparable.
- **Hardware.** The cross-domain, in-domain, noise, and constrained-decoding evaluations were produced in a single pass on one NVIDIA GeForce RTX 3090 (24 GB); the held-out evaluation and the update pilot used the same code and settings on a separate machine. Arm means can shift by up to about 2.4pp across hardware and batch-size configurations, so compare arms within one machine.
- **Quantization.** Training and evaluation both load the base model in 4-bit **NF4** with bf16 compute. (Earlier versions of `evaluate.py` did not set the quantization type and therefore evaluated in FP4; all reported results use NF4.)
- **Attention.** Training selects `flash_attention_2` when the GPU has compute capability ≥ 8.0 (`config.get_attn_impl()`), otherwise eager. `evaluate.py` uses the Transformers default (SDPA). The difference is numerical at floating-point level only.
- **String normalization.** Strict Exact Match and vocabulary lookups normalize with `.strip().lower()`.
- **Stage branching.** ORPO, FC-KD, Vanilla-KD, and both control arms load the **Stage 1** adapter, not each other.

---

## Memory Profile & Hardware Requirements

| Model | Role | Size | Configuration | Max Sequence |
|---|---|---|---|---|
| `microsoft/phi-1_5` | Student | 1.3B | 4-bit NF4 + LoRA (r=32, α=64) | 512 |
| `Qwen/Qwen2.5-1.5B` | Student (cross-model validation) | 1.5B | 4-bit NF4 + LoRA (r=32, α=64) | 512 |
| `microsoft/Phi-3-mini-4k-instruct` | Teacher / LLM judge | 3.8B | 4-bit NF4 | — |

Evaluation at `batch_size = 24` fits both students on a 24 GB GPU; Qwen2.5-1.5B's 151k-token vocabulary makes it the memory-limiting backbone for the ranking pass. Evaluation hardware: 1× NVIDIA GeForce RTX 3090 (24 GB).

---

## Citation

```bibtex
@article{ODEWOLE2026117372,
  title = {Evaluating format-constrained knowledge distillation for knowledge graph reasoning in small language models: A controlled cross-graph study},
  author = {Odewole, Olusegun Oladele and Ba{\c{s}}aran, Seren},
  journal = {Knowledge-Based Systems},
  pages = {117372},
  year = {2026},
  issn = {0950-7051},
  doi = {https://doi.org/10.1016/j.knosys.2026.117372},
  url = {https://www.sciencedirect.com/science/article/pii/S0950705126020988},
  keywords = {Knowledge graph completion, Small language models, Knowledge distillation, Cross-graph generalization, Generative bottleneck, Controlled evaluation}
}
```

---

## Disclaimer

The data, models, and scripts in this repository are for **academic and computational research evaluation only**. They are not intended for clinical use, diagnostic applications, or real-world medical decision-making. Results reflect PrimeKG and PrimeKG-to-PharmGKB evaluation only and should not be interpreted as validated performance on any downstream pharmacogenomic application.

## License

MIT — see `LICENSE`.
