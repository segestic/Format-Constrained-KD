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
from huggingface_hub import login
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
        wandb.login(key=os.environ.get("WANDB_API_KEY", ""))
    except Exception as e:
        print(f"⚠️ Authentication Error: {e}")


if __name__ == "__main__":
    setup_environment()
    set_global_seed(CONFIG["RUN_SEED"])

    print(f"\nInitializing Pipeline. Active Stage: {ACTIVE_STAGE} | Seed: {CONFIG['RUN_SEED']}")

    if ACTIVE_STAGE in [1, 2, 3, "3b"]:
        authenticate_services()
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

        wandb.finish()

    elif ACTIVE_STAGE == 4:
        print(f"\nRunning Stage 4 Evaluation | Mode: {EVAL_CONFIG['EVAL_MODE']}")
        evaluator = Evaluator(EVAL_CONFIG)
        evaluator.prepare_data()
        evaluator.run_evaluation()

    else:
        print(f"❌ Invalid ACTIVE_STAGE selected: {ACTIVE_STAGE}. Must be 1, 2, 3, \"3b\", or 4.")
