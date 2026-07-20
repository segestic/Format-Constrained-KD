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
ACTIVE_STAGE = 3  # 1 = SFT, 2 = ORPO, 3 = FC-KD, 4 = Evaluation

CONFIG = {
    "RUN_SEED": 999,
    "GLOBAL_SAMPLE_CAP": 2000,

    "teacher_name": "microsoft/Phi-3-mini-4k-instruct",
    "student_name": "microsoft/phi-1_5",

    "push_to_hub": True,
    "hub_username": "olusegunola",

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
    "CONTROL_MODEL": "microsoft/Phi-3.5-mini-instruct",
    "CUSTOM_MODELS": [
        "olusegunola/phi-1.5-primekg-dkd",
        # "olusegunola/phi-1.5-primekg-sft",
        # "olusegunola/phi-1.5-primekg-orpo",
    ],
    "output_dir": "./eval_outputs",
    # --- NEW: speed settings ---
    "batch_size": 16,   # raise to 32/64 if VRAM allows
    "use_4bit": True,   # keep True to stay consistent with prior 4-bit runs;
                         # flip to False only if re-running ALL arms fresh
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
    os.environ["WANDB_PROJECT"] = "phi_1_5_distillation_NEW_graph1_v2"
    os.environ["WANDB_ENTITY"] = "olusegunodewole1"
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
