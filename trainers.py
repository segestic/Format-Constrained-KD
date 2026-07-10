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
            processing_class=tokenizer,
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

        trainer = ORPOTrainer(
            model=model,
            train_dataset=train_data,
            processing_class=tokenizer,
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
        synthetic_data_repo = f"{CONFIG['hub_username']}/primekg-phi3-distilled-sft-seed{self.seed}"

        tokenizer = get_tokenizer(input_repo_id)

        print(f"📥 Loading synthetic teacher dataset from {synthetic_data_repo}...")
        full_synthetic_dataset = load_dataset(synthetic_data_repo, split="train")
        syn_split = full_synthetic_dataset.train_test_split(test_size=0.1, seed=self.seed)

        base_student = get_quantized_base_model(CONFIG["student_name"])
        # Branch from the Stage 1 adapter (not Stage 2) to preserve preference isolation
        model = PeftModel.from_pretrained(base_student, input_repo_id, is_trainable=True)

        trainer = SFTTrainer(
            model=model,
            train_dataset=syn_split["train"],
            eval_dataset=syn_split["test"],
            processing_class=tokenizer,
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
