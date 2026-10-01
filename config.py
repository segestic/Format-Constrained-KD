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
# Note: Use integers for 1, 2, 4-12. Only stage "3b" requires quotes.
ACTIVE_STAGE = 4
# 1 = SFT, 2 = ORPO, 3 = FC-KD, "3b" = Vanilla-KD, 4 = Evaluation,
# 5 = In-Domain vs Cross-Domain Ablation (review #3, post-hoc, needs Stage 4 done),
# 6 = Constrained-Decoding Baseline + FC-KD Hybrid (review #6, post-hoc, needs Stages 1 & 3),
# 7 = Vanilla-KD Seed-Variance Analysis (review #5, post-hoc, needs Stage 4 done for "3b"),
# 8 = Expanded Noise Robustness (review #9, post-hoc, needs Stage 4 checkpoints),
# 9 = LLM-as-Judge Four-Way Metric (review #4, post-hoc, needs Stage 4 CSVs),
# 10 = End-to-End Ranking Workflow (review #10, post-hoc, needs Stage 3 FC-KD checkpoint),
# 11 = Incremental KG-Update Pilot (review #8, post-hoc, needs Stage 3 FC-KD checkpoint),
# 12 = Paired Significance Testing (review #1, post-hoc, needs Stage 4 done for ALL 5 seeds)
# Control arms (ctl-evalprompt / ctl-trainprompt) are trained with
# train_control_arms.py, not through ACTIVE_STAGE.

CONFIG = {
    # Seeds select the per-relation PrimeKG sample and its 90/10 split. The
    # Trainers are not passed a seed, so optimization used the Trainer default
    # (42) in every run; state this in the paper.
    "RUN_SEEDS": [42, 101, 999, 7, 2024],
    "GLOBAL_SAMPLE_CAP": 2000,

    "teacher_name": "microsoft/Phi-3-mini-4k-instruct",

    # --- Student backbone ---
    # Switch between the two backbones and run every training/eval step once for each:
    #   "microsoft/phi-1_5"   (primary)
    #   "Qwen/Qwen2.5-1.5B"   (second backbone, review comment #2)
    # Use BASE (non-instruct) variants. Before adding a new family, add its LoRA
    # target_modules to LORA_TARGET_MODULES_BY_ARCHITECTURE in models.py.
    # Can be overridden per run: STUDENT_NAME="Qwen/Qwen2.5-1.5B" python ...
    "student_name": os.environ.get("STUDENT_NAME", "microsoft/phi-1_5"),

    "push_to_hub": True,
    # Override via `export HF_HUB_USERNAME=your-username`.
    "hub_username": os.environ.get("HF_HUB_USERNAME", "olusegunola"),

    # Owner of the pre-generated teacher-completion datasets used by Stage 3 / 3b.
    "synthetic_data_owner": os.environ.get("SYNTHETIC_DATA_OWNER", os.environ.get("HF_HUB_USERNAME", "olusegunola")),

    "GLOBAL_EPOCHS": 3,
    "GLOBAL_BATCH_SIZE": 8,
    "GLOBAL_GRAD_ACCUM": 4,

    # --- Stage 1: Structural SFT Warm-Up ---
    "s1_lr": 2e-4,
    "s1_max_length": 512,

    # --- Stage 2: Topology-Anchored ORPO ---
    "s2_lr": 8e-6,
    "s2_beta": 0.1,
    "s2_max_length": 512,
    "s2_max_prompt_length": 256,

    # --- Stage 3: FC-KD ---
    # NOTE (audit): the FC-KD corpus is ~95% identical to the PrimeKG ground
    # truth (teacher output kept only if it substring-matches the ground truth,
    # ground truth substituted otherwise), and its student text uses the
    # evaluation prompt. See train_control_arms.py.
    "s3_lr": 5e-5,
    "s3_max_length": 512,

    # --- Stage 3b: Vanilla-KD ---
    # Same hyperparameters and branch point as Stage 3. The DATA differs in more
    # than format: free-form teacher answers (asked to explain its reasoning),
    # no ground-truth fallback, and the Stage 1 prompt rather than the
    # evaluation prompt.
    "s3b_lr": 5e-5,
    "s3b_max_length": 512,

    # --- Control arms (train_control_arms.py) ---
    # Identical to Stage 3 (branch point, lr, epochs, batch, max length, rows,
    # split) except the data: pure ground-truth targets.
    #   ctl-evalprompt : evaluation prompt (FC-KD's template)  -> isolates the teacher
    #   ctl-trainprompt: Stage 1 prompt (SFT/ORPO template)     -> isolates the prompt
    # Set to False to evaluate only the original arms (e.g. before the controls
    # have been trained and pushed for the current backbone).
    "INCLUDE_CONTROL_ARMS": True,
}

# ==============================================================================
# STUDENT-DEPENDENT OUTPUT/REPO NAMING
# ==============================================================================
# phi-1_5 is special-cased to preserve the slug ("phi-1.5") used by every
# checkpoint and dataset already pushed.
_KNOWN_SLUG_OVERRIDES = {
    "microsoft/phi-1_5": "phi-1.5",
}
_student_slug = _KNOWN_SLUG_OVERRIDES.get(
    CONFIG["student_name"],
    CONFIG["student_name"].split("/")[-1].lower(),
)
CONFIG["student_slug"] = _student_slug
CONFIG["s1_output_dir"] = f"{_student_slug}-primekg-sft"
CONFIG["s2_output_dir"] = f"{_student_slug}-primekg-orpo"
CONFIG["s3_output_dir"] = f"{_student_slug}-primekg-dkd"
CONFIG["s3b_output_dir"] = f"{_student_slug}-primekg-vanillakd"
# Control-arm names deliberately avoid "sft"/"dkd"/"orpo"/"vanillakd"/"control"
# so filename-based arm detection in the analysis scripts cannot misclassify them.
CONFIG["ctl_eval_output_dir"] = f"{_student_slug}-primekg-ctl-evalprompt"
CONFIG["ctl_train_output_dir"] = f"{_student_slug}-primekg-ctl-trainprompt"

# ==============================================================================
# EVALUATION CONFIGURATION (Stage 4)
# ==============================================================================
_HUB = CONFIG["hub_username"]
EVAL_CONFIG = {
    "base_model_name": CONFIG["student_name"],
    "force_rerun": True,  # regenerate every (model, seed); skip-if-exists is off
    "reuse_base_model": True,  # evaluate.py: load base once, swap adapters
    "num_eval_samples": 500,
    "max_new_tokens": 20,

    # "cross_domain": PharmGKB (primary). "standard": PrimeKG evaluation pool,
    # ~70% of whose edges were never in the training sample and ~27% were
    # (see seen_unseen_standard_analysis.py); reported as the in-domain comparison.
    # Can be overridden per run: EVAL_MODE=standard python evaluate.py
    "EVAL_MODE": os.environ.get("EVAL_MODE", "cross_domain"),  # "cross_domain" | "standard"
    "noise_level": 0.10,  # character-swap noise fraction

    "SEEDS_TO_EVAL": [42, 101, 999, 7, 2024],
    "CONTROL_MODEL": CONFIG["student_name"],
    "CUSTOM_MODELS": [
        f"{_HUB}/{CONFIG['s3_output_dir']}",
        f"{_HUB}/{CONFIG['s3b_output_dir']}",
        f"{_HUB}/{CONFIG['s1_output_dir']}",
        f"{_HUB}/{CONFIG['s2_output_dir']}",
    ] + ([
        f"{_HUB}/{CONFIG['ctl_eval_output_dir']}",
        f"{_HUB}/{CONFIG['ctl_train_output_dir']}",
    ] if CONFIG["INCLUDE_CONTROL_ARMS"] else []),
    "output_dir": "./eval_outputs",
    # Must be identical for every arm, seed, backbone, and eval mode: trap
    # sampling is keyed to batch position. 24 fits Qwen2.5-1.5B on a 24 GB GPU.
    "batch_size": 24,
    "use_4bit": True,  # evaluate.py loads NF4, matching training
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
