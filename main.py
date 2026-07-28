"""
main.py
=======
Master orchestration router. Set ACTIVE_STAGE in config.py to select which
stage of the pipeline to run:
    1    = Structural SFT Warm-Up          (trains on PrimeKG)
    2    = Topology-Anchored ORPO          (trains on PrimeKG)
    3    = Format-Constrained KD (FC-KD)   (trains on PrimeKG)
    "3b" = Vanilla (Unconstrained) KD      (trains on PrimeKG, ablation arm)
    4    = Dual-Modality Evaluation        (evaluates on PharmGKB, cross-domain)
"""

import os
import wandb
from huggingface_hub import login, dataset_info
from config import ACTIVE_STAGE, CONFIG, EVAL_CONFIG, set_global_seed, setup_environment
from data_utils import PrimeKGDataLoader
from evaluate import Evaluator
from trainers import PipelineTrainer
from transformers import logging as hf_logging

# Force Hugging Face to only print fatal errors, hiding the deprecation spam
hf_logging.set_verbosity_error()


def authenticate_services():
    try:
        login(token=os.environ.get("HF_TOKEN", ""))
    except Exception as e:
        print(f"⚠️ Hugging Face Authentication Error: {e}")

    # Only attempt W&B login if a key was actually provided. Calling
    # wandb.login() with no key can hang waiting for interactive input inside
    # a non-interactive Docker container -- setup_environment() has already
    # set WANDB_MODE="disabled" in that case, so we skip login entirely.
    if os.environ.get("WANDB_API_KEY"):
        try:
            wandb.login(key=os.environ["WANDB_API_KEY"])
        except Exception as e:
            print(f"⚠️ W&B Authentication Error: {e}")


def check_synthetic_dataset_exists(seed: int, stage: str):
    """Preflight check for Stage 3 / 3b: fail fast with a clear message if the
    required synthetic teacher dataset hasn't been generated/pushed yet,
    instead of letting the run proceed through slow model loading first.
    """
    repo_map = {
        "3": f"{CONFIG['synthetic_data_owner']}/primekg-phi3-distilled-sft-seed{seed}",
        "3b": f"{CONFIG['synthetic_data_owner']}/primekg-phi3-vanillakd-sft-seed{seed}",
    }
    script_map = {
        "3": "data/black_box_distill.py",
        "3b": "data/distill_vanilla_kd.py",
    }
    repo_id = repo_map[stage]
    try:
        dataset_info(repo_id)
    except Exception:
        raise SystemExit(
            f"\n❌ Required synthetic dataset '{repo_id}' was not found on the Hub.\n"
            f"   Stage {stage} trains on pre-generated teacher completions, not on PrimeKG directly.\n"
            f"   Fix: run `python {script_map[stage]}` first to generate and push this dataset\n"
            f"   under your own HF_HUB_USERNAME, then re-run this stage. If someone else already\n"
            f"   has a copy, set `export SYNTHETIC_DATA_OWNER=their-account` instead.\n"
        )


if __name__ == "__main__":
    setup_environment()
    set_global_seed(CONFIG["RUN_SEED"])

    print(f"\nInitializing Pipeline. Active Stage: {ACTIVE_STAGE} | Seed: {CONFIG['RUN_SEED']}")

    if ACTIVE_STAGE in [1, 2, 3, "3b"]:
        authenticate_services()

        if ACTIVE_STAGE in [3, "3b"]:
            check_synthetic_dataset_exists(CONFIG["RUN_SEED"], str(ACTIVE_STAGE))

        data_loader = PrimeKGDataLoader(seed=CONFIG["RUN_SEED"], sample_cap=CONFIG["GLOBAL_SAMPLE_CAP"])
        trainer = PipelineTrainer(seed=CONFIG["RUN_SEED"])

        if ACTIVE_STAGE in [1, 2]:
            sft_data, orpo_data = data_loader.get_aligned_datasets()
            if ACTIVE_STAGE == 1:
                trainer.run_stage_1_sft(sft_data["train"], sft_data["test"])
            else:
                trainer.run_stage_2_orpo(orpo_data["train"])
        elif ACTIVE_STAGE == 3:
            trainer.run_stage_3_distillation()
        else:  # ACTIVE_STAGE == "3b"
            trainer.run_stage_3b_vanilla_kd()

        if os.environ.get("WANDB_API_KEY"):
            wandb.finish()

    elif ACTIVE_STAGE == 4:
        print(f"\nRunning Stage 4 Evaluation | Mode: {EVAL_CONFIG['EVAL_MODE']}")
        evaluator = Evaluator(EVAL_CONFIG)
        evaluator.prepare_data()
        evaluator.run_evaluation()

    else:
        print(f"❌ Invalid ACTIVE_STAGE selected: {ACTIVE_STAGE}. Must be 1, 2, 3, \"3b\", or 4.")
