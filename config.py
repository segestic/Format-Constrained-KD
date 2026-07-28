"""
config.py
=========
Master configuration for the FC-KD pipeline: training hyperparameters,
environment setup, and evaluation protocol.

Design note (Cross-Domain Evaluation Protocol)
-----------------------------------------------
Models are trained EXCLUSIVELY on PrimeKG (Stages 1-3). Evaluation is run
EXCLUSIVELY on PharmGKB (Stage 4), a structurally distinct pharmacogenomic
knowledge graph that receives zero exposure during any training stage.
This is an intentional design choice, not an accidental default -- it tests
whether relational competence transfers across graph schemas (H1), rather
than merely across a held-out split of the same graph. See README.md
"Cross-Domain Evaluation Protocol" for details.
"""

import os
import random

import numpy as np
import torch

# ==============================================================================
# TRAINING CONFIGURATION (Stages 1-3)
# ==============================================================================
# Note: Use integers for 1, 2, 4. Only stage "3b" requires quotes.
ACTIVE_STAGE = "3b"  # 1 = SFT, 2 = ORPO, 3 = FC-KD, "3b" = Vanilla-KD,  4 = Evaluation

CONFIG = {
    "RUN_SEED": 999, #42, 101, 999
    "GLOBAL_SAMPLE_CAP": 2000,

    "teacher_name": "microsoft/Phi-3-mini-4k-instruct",
    "student_name": "microsoft/phi-1_5",

    "push_to_hub": True,
    # Override via `export HF_HUB_USERNAME=your-username` to push/pull from a
    # different account. Defaults to the author's account for local dev use.
    "hub_username": os.environ.get("HF_HUB_USERNAME", "olusegunola"),

    # Owner of the pre-generated synthetic teacher-completion datasets used
    # by Stage 3 / Stage 3b. Defaults to hub_username. Override with
    # `export SYNTHETIC_DATA_OWNER=some-account` only if pulling someone
    # else's already-pushed synthetic datasets.
    "synthetic_data_owner": os.environ.get("SYNTHETIC_DATA_OWNER", os.environ.get("HF_HUB_USERNAME", "olusegunola")),

    "GLOBAL_EPOCHS": 3,
    "GLOBAL_BATCH_SIZE": 16,
    "GLOBAL_GRAD_ACCUM": 2,

    # --- Stage 1: Structural SFT Warm-Up ---
    "s1_lr": 2e-4,
    "s1_max_length": 512,
    "s1_output_dir": "phi-1.5-primekg-sft",

    # --- Stage 2: Topology-Anchored ORPO ---
    "s2_lr": 8e-6,
    "s2_beta": 0.1,
    "s2_max_length": 512,
    "s2_max_prompt_length": 256,
    "s2_output_dir": "phi-1.5-primekg-orpo",

    # --- Stage 3: Format-Constrained Sequence-Level Distillation (FC-KD) ---
    "s3_lr": 5e-5,
    "s3_max_length": 512,
    "s3_output_dir": "phi-1.5-primekg-dkd",

    # --- Stage 3b: Vanilla (Unconstrained) Sequence-Level Distillation ---
    # Ablation arm: same teacher, same student, same loss, same seeds, same
    # LoRA branch point as Stage 3 -- the ONLY thing that differs is the
    # dataset it trains on (unconstrained teacher completions instead of
    # format-constrained/regex-filtered ones). All hyperparameters are kept
    # identical to Stage 3 on purpose, so a performance gap can only be
    # attributed to the training data's format constraint, not to a
    # confounded change in learning rate, sequence length, or epochs.
    "s3b_lr": 5e-5,
    "s3b_max_length": 512,
    "s3b_output_dir": "phi-1.5-primekg-vanillakd",
}

# ==============================================================================
# EVALUATION CONFIGURATION (Stage 4)
# ==============================================================================
EVAL_CONFIG = {
    "base_model_name": "microsoft/phi-1_5",
    "num_eval_samples": 500,
    "max_new_tokens": 20,

    # "cross_domain" (PharmGKB, out-of-distribution) is the paper's reported
    # setting. "standard" (PrimeKG held-out split) is provided only as an
    # internal sanity-check mode and is NOT used for any number reported
    # in the manuscript.
    "EVAL_MODE": "cross_domain",
    "noise_level": 0.10,  # robustness mode: character-swap noise fraction

    "SEEDS_TO_EVAL": [42, 101, 999],
    "CONTROL_MODEL": "microsoft/phi-1_5",
    "CUSTOM_MODELS": [
        f"{os.environ.get('HF_HUB_USERNAME', 'olusegunola')}/phi-1.5-primekg-dkd",
        f"{os.environ.get('HF_HUB_USERNAME', 'olusegunola')}/phi-1.5-primekg-vanillakd",   # <-- Stage 3b ablation arm
        f"{os.environ.get('HF_HUB_USERNAME', 'olusegunola')}/phi-1.5-primekg-sft",
        f"{os.environ.get('HF_HUB_USERNAME', 'olusegunola')}/phi-1.5-primekg-orpo",
    ],
    "output_dir": "./eval_outputs",
    # --- NEW: speed settings ---
    "batch_size": 16,   # raise to 32/64 if VRAM allows
    "use_4bit": True,   # keep True to stay consistent with prior 4-bit runs;
                         # flip to False only if re-running ALL
}
EVAL_CONFIG["ALL_MODELS"] = EVAL_CONFIG["CUSTOM_MODELS"] + [EVAL_CONFIG["CONTROL_MODEL"]]


# ==============================================================================
# SYSTEM INITIALIZATION
# ==============================================================================
def set_global_seed(seed: int):
    """Enforces deterministic execution globally."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_environment():
    """Configures system-level optimizations and W&B logging."""
    os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # If no WANDB_API_KEY is set, disable W&B entirely rather than letting
    # wandb.login() hang on an interactive prompt or crash inside a
    # non-interactive Docker container. WANDB_MODE="disabled" makes every
    # wandb call (including the trainers' automatic logging integration) a
    # silent no-op -- training runs identically, just without remote logging.
    if not os.environ.get("WANDB_API_KEY"):
        print("ℹ️  WANDB_API_KEY not set -- W&B logging disabled for this run.")
        os.environ["WANDB_MODE"] = "disabled"
    else:
        os.environ["WANDB_PROJECT"] = "phi_1_5_distillation_NEW_graph1_v2"
        os.environ["WANDB_ENTITY"] = os.environ.get("WANDB_ENTITY", "olusegunodewole1")
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    # Accelerate monkey patch for a known optimizer compatibility bug
    if not hasattr(torch.optim.Optimizer, "train"):
        torch.optim.Optimizer.train = lambda self: None
    if not hasattr(torch.optim.Optimizer, "eval"):
        torch.optim.Optimizer.eval = lambda self: None


def get_attn_impl():
    """Determines the optimal attention implementation for the current GPU."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return "flash_attention_2"
    return "eager"
