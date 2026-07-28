"""
trainers.py
===========
Isolated training-stage implementations: Structural SFT Warm-Up (Stage 1),
Topology-Anchored ORPO (Stage 2), and Format-Constrained Sequence-Level
Distillation / FC-KD (Stage 3). All three stages branch from the Stage 1
checkpoint to preserve preference isolation between the ORPO and FC-KD
optimization paths (see paper Section 3.1, "Parallel Alignment Curriculum").
"""

import gc

import torch
from datasets import load_dataset
from peft import PeftModel
from trl import SFTConfig, SFTTrainer

try:
    from trl import ORPOConfig, ORPOTrainer
except ImportError:
    from trl.experimental.orpo import ORPOConfig, ORPOTrainer

from config import CONFIG
from models import get_quantized_base_model, get_tokenizer, setup_peft_model


def clear_gpu_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class PipelineTrainer:
    def __init__(self, seed: int):
        self.seed = seed

    def push_artifacts(self, trainer, tokenizer, repo_id):
        if CONFIG.get("push_to_hub"):
            trainer.model.push_to_hub(repo_id)
            tokenizer.push_to_hub(repo_id)

    def run_stage_1_sft(self, train_data, eval_data):
        print(f"\n🚀 STAGE 1: SFT WARM-UP | SEED: {self.seed}")
        clear_gpu_cache()
        output_dir = f"./{CONFIG['s1_output_dir']}-seed{self.seed}"

        tokenizer = get_tokenizer(CONFIG["student_name"])
        base_model = get_quantized_base_model(CONFIG["student_name"])
        model = setup_peft_model(base_model)

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            tokenizer=tokenizer,
            args=SFTConfig(
                dataset_text_field="text",
                max_seq_length=CONFIG["s1_max_length"],
                output_dir=output_dir,
                num_train_epochs=CONFIG["GLOBAL_EPOCHS"],
                per_device_train_batch_size=CONFIG["GLOBAL_BATCH_SIZE"],
                gradient_accumulation_steps=CONFIG["GLOBAL_GRAD_ACCUM"],
                learning_rate=CONFIG["s1_lr"],
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=50,
                eval_strategy="steps",
                eval_steps=200,
                save_strategy="no",
                optim="paged_adamw_8bit",
            ),
        )
        trainer.train()
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        self.push_artifacts(trainer, tokenizer, f"{CONFIG['hub_username']}/{CONFIG['s1_output_dir']}-seed{self.seed}")

    def run_stage_2_orpo(self, train_data):
        print(f"\n🚀 STAGE 2: TOPOLOGY ORPO | SEED: {self.seed}")
        clear_gpu_cache()
        input_dir = f"./{CONFIG['s1_output_dir']}-seed{self.seed}"
        output_dir = f"./{CONFIG['s2_output_dir']}-seed{self.seed}"

        tokenizer = get_tokenizer(input_dir)
        base_model = get_quantized_base_model(CONFIG["student_name"])
        model = PeftModel.from_pretrained(base_model, input_dir, is_trainable=True)

        # ====================================================================
        # FIX FOR TRL/TRANSFORMERS VERSION MISMATCHES (v5.8.1 compat)
        # ====================================================================
        from trl import ORPOTrainer
        from transformers import Trainer
        
        # 1. Patch get_batch_samples (Name Collision Fix)
        original_orpo_get_batch_samples = ORPOTrainer.get_batch_samples
        original_trainer_get_batch_samples = Trainer.get_batch_samples
        
        def smart_get_batch_samples(self, *args, **kwargs):
            if len(args) >= 3:
                return original_trainer_get_batch_samples(self, *args, **kwargs)
            return original_orpo_get_batch_samples(self, *args, **kwargs)
            
        ORPOTrainer.get_batch_samples = smart_get_batch_samples

        # 2. Patch compute_loss (Unexpected Keyword Fix)
        original_compute_loss = ORPOTrainer.compute_loss
        
        def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            return original_compute_loss(self, model, inputs, return_outputs=return_outputs)
            
        ORPOTrainer.compute_loss = patched_compute_loss

        # 3. Patch log (Unexpected start_time Fix)
        original_log = ORPOTrainer.log

        def patched_log(self, logs, *args, **kwargs):
            # transformers passes (logs, start_time), but ORPOTrainer only expects (logs)
            return original_log(self, logs)

        ORPOTrainer.log = patched_log
        # ====================================================================

        trainer = ORPOTrainer(
            model=model,
            train_dataset=train_data,
            tokenizer=tokenizer,
            args=ORPOConfig(
                output_dir=output_dir,
                num_train_epochs=CONFIG["GLOBAL_EPOCHS"],
                per_device_train_batch_size=CONFIG["GLOBAL_BATCH_SIZE"],
                gradient_accumulation_steps=CONFIG["GLOBAL_GRAD_ACCUM"],
                learning_rate=CONFIG["s2_lr"],
                beta=CONFIG["s2_beta"],
                max_length=CONFIG["s2_max_length"],
                max_completion_length=CONFIG["s2_max_prompt_length"],
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=50,
                remove_unused_columns=False,
                optim="paged_adamw_8bit",
            ),
        )
        trainer.train()
        trainer.model.save_pretrained(output_dir)

        self.push_artifacts(trainer, tokenizer, f"{CONFIG['hub_username']}/{CONFIG['s2_output_dir']}-seed{self.seed}")

    def run_stage_3_distillation(self):
        print(f"\n🚀 STAGE 3: FORMAT-CONSTRAINED DISTILLATION (FC-KD) | SEED: {self.seed}")
        clear_gpu_cache()
        input_repo_id = f"{CONFIG['hub_username']}/{CONFIG['s1_output_dir']}-seed{self.seed}"
        output_dir = f"./{CONFIG['s3_output_dir']}-seed{self.seed}"
        synthetic_data_repo = f"{CONFIG['synthetic_data_owner']}/primekg-phi3-distilled-sft-seed{self.seed}"

        tokenizer = get_tokenizer(input_repo_id)

        print(f"📥 Loading synthetic teacher dataset from {synthetic_data_repo}...")
        try:
            full_synthetic_dataset = load_dataset(synthetic_data_repo, split="train")
        except Exception as e:
            raise RuntimeError(
                f"Could not load synthetic teacher dataset '{synthetic_data_repo}'. "
                f"This dataset is generated by data/black_box_distill.py and pushed to your own "
                f"Hugging Face account -- it is not bundled with this repo. Either (1) run "
                f"data/black_box_distill.py first to generate and push it under your own "
                f"HF_HUB_USERNAME, or (2) if you have access to an existing copy under a "
                f"different account, set SYNTHETIC_DATA_OWNER to that account name. "
                f"Original error: {e}"
            ) from e
        syn_split = full_synthetic_dataset.train_test_split(test_size=0.1, seed=self.seed)

        base_student = get_quantized_base_model(CONFIG["student_name"])
        # Branch from the Stage 1 adapter (not Stage 2) to preserve preference isolation
        model = PeftModel.from_pretrained(base_student, input_repo_id, is_trainable=True)

        trainer = SFTTrainer(
            model=model,
            train_dataset=syn_split["train"],
            eval_dataset=syn_split["test"],
            tokenizer=tokenizer,
            args=SFTConfig(
                dataset_text_field="text",
                max_seq_length=CONFIG["s3_max_length"],
                output_dir=output_dir,
                num_train_epochs=CONFIG["GLOBAL_EPOCHS"],
                per_device_train_batch_size=CONFIG["GLOBAL_BATCH_SIZE"],
                gradient_accumulation_steps=CONFIG["GLOBAL_GRAD_ACCUM"],
                learning_rate=CONFIG["s3_lr"],
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=50,
                eval_strategy="steps",
                eval_steps=200,
                save_strategy="no",
                optim="paged_adamw_8bit",
            ),
        )
        trainer.train()
        trainer.model.save_pretrained(output_dir)

        self.push_artifacts(trainer, tokenizer, f"{CONFIG['hub_username']}/{CONFIG['s3_output_dir']}-seed{self.seed}")

    #new
    def run_stage_3b_vanilla_kd(self):
        print(f"\n🚀 STAGE 3b: VANILLA (UNCONSTRAINED) DISTILLATION | SEED: {self.seed}")
        clear_gpu_cache()

        input_repo_id = f"{CONFIG['hub_username']}/{CONFIG['s1_output_dir']}-seed{self.seed}"
        output_dir = f"./{CONFIG['s3b_output_dir']}-seed{self.seed}"
        synthetic_data_repo = f"{CONFIG['synthetic_data_owner']}/primekg-phi3-vanillakd-sft-seed{self.seed}"

        tokenizer = get_tokenizer(input_repo_id)

        print(f"📥 Loading unconstrained teacher dataset from {synthetic_data_repo}...")
        try:
            full_synthetic_dataset = load_dataset(synthetic_data_repo, split="train")
        except Exception as e:
            raise RuntimeError(
                f"Could not load unconstrained teacher dataset '{synthetic_data_repo}'. "
                f"This dataset is generated by data/distill_vanilla_kd.py and pushed to your own "
                f"Hugging Face account -- it is not bundled with this repo. Either (1) run "
                f"data/distill_vanilla_kd.py first to generate and push it under your own "
                f"HF_HUB_USERNAME, or (2) if you have access to an existing copy under a "
                f"different account, set SYNTHETIC_DATA_OWNER to that account name. "
                f"Original error: {e}"
            ) from e
        syn_split = full_synthetic_dataset.train_test_split(test_size=0.1, seed=self.seed)

        base_student = get_quantized_base_model(CONFIG["student_name"])
        # Branch from the same Stage 1 adapter as Stage 3, so both distillation
        # arms start from an identical point in parameter space.
        model = PeftModel.from_pretrained(base_student, input_repo_id, is_trainable=True)

        trainer = SFTTrainer(
            model=model,
            train_dataset=syn_split["train"],
            eval_dataset=syn_split["test"],
            tokenizer=tokenizer,
            args=SFTConfig(
                dataset_text_field="text",
                max_seq_length=CONFIG["s3b_max_length"],
                output_dir=output_dir,
                num_train_epochs=CONFIG["GLOBAL_EPOCHS"],
                per_device_train_batch_size=CONFIG["GLOBAL_BATCH_SIZE"],
                gradient_accumulation_steps=CONFIG["GLOBAL_GRAD_ACCUM"],
                learning_rate=CONFIG["s3b_lr"],
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=50,
                eval_strategy="steps",
                eval_steps=200,
                save_strategy="no",
                optim="paged_adamw_8bit",
            ),
        )
        trainer.train()
        trainer.model.save_pretrained(output_dir)
        self.push_artifacts(
            trainer, tokenizer,
            f"{CONFIG['hub_username']}/{CONFIG['s3b_output_dir']}-seed{self.seed}"
        )
