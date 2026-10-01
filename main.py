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
    5    = In-Domain vs Cross-Domain Ablation   (review comment #3)
    6    = Constrained-Decoding Baseline + Hybrid (review comment #6)
    7    = Vanilla-KD Seed-Variance Analysis     (review comment #5)
    8    = Expanded Noise Robustness              (review comment #9)
    9    = LLM-as-Judge Four-Way Metric           (review comment #4)
    10   = End-to-End Ranking Workflow            (review comment #10)
    11   = Incremental KG-Update Pilot            (review comment #8)
    12   = Paired Significance Testing            (review comment #1)

Stages 5-11 are post-hoc analyses over artifacts produced by stages 1-4 --
they do not train anything and require the relevant checkpoints/evaluation
CSVs to already exist.
"""

import os
import gc
import torch
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
    
    # ==============================================================================
    # TRAINING STAGES (1, 2, 3, "3b") - Now looped safely across all seeds
    # ==============================================================================
    if ACTIVE_STAGE in [1, 2, 3, "3b"]:
        authenticate_services()
        
        for current_seed in CONFIG["RUN_SEEDS"]:
            print(f"\n{'='*60}")
            print(f"🚀 Starting Pipeline Stage: {ACTIVE_STAGE} | Seed: {current_seed}")
            print(f"{'='*60}\n")
            
            # 1. ENFORCE DETERMINISM FOR THIS ITERATION
            set_global_seed(current_seed)
            
            # (Note: We rely on PipelineTrainer to append the -seed suffix internally
            # to the base dirs defined in config.py, avoiding the -seed42-seed42 bug)

            if ACTIVE_STAGE in [3, "3b"]:
                check_synthetic_dataset_exists(current_seed, str(ACTIVE_STAGE))

            # 2. RE-INITIALIZE DATA AND TRAINER FRESH FOR THIS SEED
            data_loader = PrimeKGDataLoader(seed=current_seed, sample_cap=CONFIG["GLOBAL_SAMPLE_CAP"])
            trainer = PipelineTrainer(seed=current_seed)

            # 3. EXECUTE TRAINING
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

            # 4. CLEAN UP STATE AND MEMORY BEFORE NEXT SEED
            if os.environ.get("WANDB_API_KEY") and wandb.run is not None:
                wandb.finish() # Closes the current W&B run so the next seed starts a fresh one
            
            del data_loader
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ==============================================================================
    # EVALUATION & POST-HOC ANALYSIS STAGES (4-12)
    # ==============================================================================
    elif ACTIVE_STAGE == 4:
        print(f"\nRunning Stage 4 Evaluation | Mode: {EVAL_CONFIG['EVAL_MODE']}")
        evaluator = Evaluator(EVAL_CONFIG)
        evaluator.prepare_data()
        evaluator.run_evaluation()

    elif ACTIVE_STAGE == 5:
        print("\nRunning Stage 5: In-Domain vs Cross-Domain Ablation (review comment #3)")
        from run_indomain_ablation import run_indomain_ablation
        run_indomain_ablation()

    elif ACTIVE_STAGE == 6:
        print("\nRunning Stage 6: Constrained-Decoding Baseline + FC-KD Hybrid (review comment #6)")
        from evaluate_constrained_decoding_baseline import run_constrained_decoding_experiments
        run_constrained_decoding_experiments()

    elif ACTIVE_STAGE == 7:
        print("\nRunning Stage 7: Vanilla-KD Seed-Variance Analysis (review comment #5)")
        from analyze_vanillakd_variance import run_vanillakd_variance_analysis
        run_vanillakd_variance_analysis()

    elif ACTIVE_STAGE == 8:
        print("\nRunning Stage 8: Expanded Noise Robustness (review comment #9)")
        from evaluate_expanded_noise_robustness import run_expanded_noise_robustness
        run_expanded_noise_robustness()

    elif ACTIVE_STAGE == 9:
        print("\nRunning Stage 9: LLM-as-Judge Four-Way Metric (review comment #4)")
        from llm_judge_factual_correctness import run_llm_judge_rescoring
        run_llm_judge_rescoring()

    elif ACTIVE_STAGE == 10:
        print("\nRunning Stage 10: End-to-End Ranking Workflow (review comment #10)")
        from end_to_end_ranking_workflow import run_end_to_end_workflow
        run_end_to_end_workflow()

    elif ACTIVE_STAGE == 11:
        print("\nRunning Stage 11: Incremental KG-Update Pilot (review comment #8)")
        from pilot_incremental_kg_update import run_incremental_update_pilot
        run_incremental_update_pilot()

    elif ACTIVE_STAGE == 12:
        print("\nRunning Stage 12: Paired Significance Testing (review comment #1)")
        from run_significance_tests import run_significance_tests
        run_significance_tests()

    else:
        print(f"❌ Invalid ACTIVE_STAGE selected: {ACTIVE_STAGE}. Must be 1, 2, 3, \"3b\", 4-12.")