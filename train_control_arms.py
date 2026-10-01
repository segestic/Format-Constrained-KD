"""
train_control_arms.py
======================
Control arms that isolate what FC-KD's Stage 3 actually adds.

Audit facts this design rests on (black_box_distill.py, trainers.py):
  - FC-KD's corpus is ~95% identical to the PrimeKG ground truth (teacher
    output kept only when it substring-matches the ground truth; ground
    truth substituted otherwise).
  - FC-KD's student text uses the EVALUATION prompt ("Task: Analyze ...
    Target Node: "), while SFT, ORPO, and Vanilla-KD use the Stage 1 prompt
    ("Analyze the biomedical relationship.\n...").
  - Stage 3 starts from the Stage 1 adapter and trains 3 more epochs on the
    same 90% of the same rows (identical length and split seed), at s3_lr.

Two controls, both identical to Stage 3 in every respect except the data:
  ctl-evalprompt  : ground-truth targets, EVALUATION prompt (FC-KD's template).
                    FC-KD minus this arm = effect of the teacher (~5% of targets).
  ctl-trainprompt : ground-truth targets, Stage 1 prompt (SFT/ORPO template).
                    ctl-evalprompt minus this arm = effect of training on the
                    evaluation prompt; this arm minus SFT = effect of extra epochs.

Rows, row order, dataset length, split seed, starting adapter, optimizer,
learning rate, epochs, batch size, and max length all match Stage 3, so the
90/10 split selects exactly the same training rows as SFT and FC-KD.

USAGE (per backbone, i.e. with student_name set in config.py):
    python train_control_arms.py --arm evalprompt
    python train_control_arms.py --arm trainprompt
    python train_control_arms.py --arm evalprompt --seeds 42 101
Datasets are pushed to <hub_user>/primekg-gt-<arm>-seed<SEED> and adapters to
<hub_user>/<CONFIG ctl_eval_output_dir | ctl_train_output_dir>-seed<SEED>. Repo names deliberately avoid
"sft"/"dkd"/"orpo"/"vanillakd" so existing filename-based arm detection
cannot misclassify them.
"""

import argparse
import gc

import torch
from datasets import Dataset, load_dataset
from peft import PeftModel
from tdc.resource import PrimeKG
from trl import SFTConfig, SFTTrainer

from config import CONFIG, EVAL_CONFIG, set_global_seed, setup_environment
from models import get_quantized_base_model, get_tokenizer

TEMPLATES = {
    # Identical to black_box_distill.py's sft_string (FC-KD student text).
    "evalprompt": (
        "Task: Analyze the biomedical relationship and output ONLY the Target Node and its Type. "
        "Do not write explanations, exercises, or extra text.\n\n"
        "Source Node: {x_name} (Type: {x_type})\n"
        "Relationship: {relation}\n"
        "Target Node: {y_name} (Type: {y_type})"
    ),
    # Identical to data_utils._build_sft_text (SFT student text).
    "trainprompt": (
        "Analyze the biomedical relationship.\n"
        "Source Node: {x_name} (Type: {x_type})\n"
        "Relationship: {relation}\n"
        "Target Node: {y_name} (Type: {y_type})"
    ),
}


def build_and_push_dataset(df_base, arm: str, seed: int) -> str:
    # Same expression and row order as data_utils.py and black_box_distill.py.
    df_sample = df_base.groupby("relation", group_keys=False).apply(
        lambda x: x.sample(min(len(x), CONFIG["GLOBAL_SAMPLE_CAP"]), random_state=seed)
    ).copy()
    texts = [TEMPLATES[arm].format(**row) for _, row in df_sample.iterrows()]
    repo = f"{CONFIG['hub_username']}/primekg-gt-{arm}-seed{seed}"
    Dataset.from_dict({"text": texts}).push_to_hub(repo)
    print(f"  pushed {len(texts)} texts to {repo}")
    return repo


def train_control(arm: str, seed: int, dataset_repo: str):
    set_global_seed(seed)
    input_repo_id = f"{CONFIG['hub_username']}/{CONFIG['s1_output_dir']}-seed{seed}"
    out_name = CONFIG["ctl_eval_output_dir"] if arm == "evalprompt" else CONFIG["ctl_train_output_dir"]
    output_dir = f"./{out_name}-seed{seed}"

    tokenizer = get_tokenizer(input_repo_id)
    full = load_dataset(dataset_repo, split="train")
    split = full.train_test_split(test_size=0.1, seed=seed)  # same rows as SFT / FC-KD

    base = get_quantized_base_model(CONFIG["student_name"])  # NF4, as in Stage 3
    model = PeftModel.from_pretrained(base, input_repo_id, is_trainable=True)

    trainer = SFTTrainer(
        model=model,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer,
        args=SFTConfig(  # every value copied from run_stage_3_distillation
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
            # No `seed=` on purpose: run_stage_3_distillation does not pass one,
            # so the Trainer used its default (42). Matching that keeps the
            # controls identical to FC-KD.
        ),
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    if CONFIG.get("push_to_hub"):
        repo_id = f"{CONFIG['hub_username']}/{out_name}-seed{seed}"
        trainer.model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print(f"  pushed adapter to {repo_id}")

    del trainer, model, base, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=sorted(TEMPLATES))
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()
    seeds = args.seeds or CONFIG["RUN_SEEDS"]

    setup_environment()
    df_base = PrimeKG(path="./data").get_data().dropna(subset=["x_name", "y_name"])
    for seed in seeds:
        print(f"\n{'=' * 60}\nCONTROL ARM ctl-{args.arm} | {CONFIG['student_name']} | seed {seed}\n{'=' * 60}")
        repo = build_and_push_dataset(df_base, args.arm, seed)
        train_control(args.arm, seed, repo)


if __name__ == "__main__":
    main()
