#%%
# distillation from teacher — vanilla (unconstrained) KD ablation arm
#
# This script is a deliberate near-duplicate of the FC-KD teacher-generation
# script. Everything is held identical (teacher model, seeds, sampling rows,
# decoding determinism, batch size, output dataset schema) EXCEPT:
#   1. The teacher prompt does NOT enforce a rigid output template.
#   2. There is NO regex-based parsing/rejection filter.
#   3. There is NO fallback-to-ground-truth substitution on parse failure
#      (since there is no parse step to fail).
#
# This isolates "format-constrained teacher" vs. "unconstrained teacher" as
# the single varying factor between this dataset and the FC-KD dataset,
# so that Stage 3 (FC-KD) vs Stage 3b (vanilla KD) becomes a clean ablation
# of the format-constraint mechanism itself, rather than a comparison that
# also confounds in "distillation vs. no distillation."


import os
import gc
import torch
import random
import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tdc.resource import PrimeKG

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
SEEDS = [42, 101, 999]
GLOBAL_SAMPLE_CAP = 2000
TEACHER_MODEL = "microsoft/Phi-3-mini-4k-instruct"
HUB_USERNAME = "olusegunola"

# Vanilla-KD completions are free-form, so they need more headroom than the
# 25-token budget used for the strict "[Node] (Type: [Class])" FC-KD target.
INFERENCE_BATCH_SIZE = 64
MAX_NEW_TOKENS_VANILLA = 80

# ==============================================================================
# 1. LOAD DATA ONCE (identical source rows to the FC-KD run for comparability)
# ==============================================================================
print(f"📚 Loading PrimeKG base matrix...")
data = PrimeKG(path='./data')
df_base = data.get_data().dropna(subset=['x_name', 'y_name'])
print(f"✅ Loaded {len(df_base)} total PrimeKG rows.")

# ==============================================================================
# 2. LOAD TEACHER MODEL ONCE (identical model/quantization to FC-KD)
# ==============================================================================
print(f"\n🧠 Loading Teacher Model ({TEACHER_MODEL}) into VRAM...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

attn_impl = "eager"
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_impl,
    trust_remote_code=False
)
teacher.eval()
print("✅ Teacher model loaded and configured for Left-Padded Batching.")

# ==============================================================================
# 3. THE MULTI-SEED LOOP (BATCHED, UNCONSTRAINED PROMPT)
# ==============================================================================
for SEED in SEEDS:
    print("\n" + "="*60)
    print(f"🚀 STARTING VANILLA-KD DISTILLATION RUN FOR SEED: {SEED}")
    print("="*60)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # Same stratified sampling procedure and seed as the FC-KD run, so the
    # exact same (source, relation, target) rows are used in both arms.
    df_sample = df_base.groupby('relation', group_keys=False).apply(
        lambda x: x.sample(min(len(x), GLOBAL_SAMPLE_CAP), random_state=SEED)
    ).copy()

    print(f"✅ Isolated {len(df_sample)} rows for Seed {SEED}.")

    distilled_texts = []
    empty_completion_count = 0

    for i in tqdm(range(0, len(df_sample), INFERENCE_BATCH_SIZE), desc=f"Vanilla-KD Seed {SEED}"):

        chunk = df_sample.iloc[i : i + INFERENCE_BATCH_SIZE]
        batch_prompts = []

        # 1. Build unconstrained prompts for the whole batch.
        #    No "Output ONLY..." instruction, no fixed template to fill in.
        #    The teacher is free to answer in whatever form it prefers.
        for _, row in chunk.iterrows():
            prompt = (
                f"<|user|>\n"
                f"You are a biomedical knowledge assistant. Given a Source Node "
                f"and a Relation, identify the Target Node that completes this "
                f"biomedical relationship, and briefly explain your reasoning.\n\n"
                f"Source: {row['x_name']} (Type: {row['x_type']})\n"
                f"Relation: {row['relation']}\n"
                f"What is the Target Node?<|end|>\n<|assistant|>\n"
            )
            batch_prompts.append(prompt)

        # 2. Tokenize the entire batch at once (Left-Padded)
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(teacher.device)

        # 3. Generate across the entire batch matrix simultaneously.
        #    Decoding stays deterministic (do_sample=False) to match FC-KD --
        #    the only varying factor is prompt strictness and post-processing.
        with torch.no_grad():
            outputs = teacher.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS_VANILLA,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # 4. Slice off the padded prompts and decode the new tokens
        generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # 5. NO regex filter, NO rejection, NO fallback substitution.
        #    Whatever the teacher produced is accepted as-is (after basic
        #    whitespace trimming). This is the defining property of the
        #    "unconstrained" arm: the student learns from raw teacher output,
        #    not from a deterministically enforced schema.
        for gen_text, (_, row) in zip(generated_texts, chunk.iterrows()):
            clean_completion = gen_text.strip()

            if not clean_completion:
                # Only exclusion rule kept: an empty generation carries no
                # training signal at all, so it is dropped (not substituted).
                empty_completion_count += 1
                continue

            # Prompt scaffold kept identical to FC-KD's SFT text template so
            # that downstream tokenization/eval extraction logic is unchanged;
            # only the completion content differs (free-form vs. constrained).
            sft_string = (
                f"Analyze the biomedical relationship.\n"
                f"Source Node: {row['x_name']} (Type: {row['x_type']})\n"
                f"Relationship: {row['relation']}\n"
                f"Target Node: {clean_completion}"
            )

            distilled_texts.append(sft_string)

    print(f"\n✅ Vanilla-KD distillation for Seed {SEED} Complete!")
    print(f"⚠️ Empty completions dropped: {empty_completion_count}/{len(df_sample)} "
          f"({(empty_completion_count/len(df_sample))*100:.2f}%)")

    # Push to HuggingFace Hub under a distinct repo name (parallel to the
    # FC-KD dataset naming convention, distinguished by "vanillakd").
    dataset_repo = f"{HUB_USERNAME}/primekg-phi3-vanillakd-sft-seed{SEED}"
    print(f"🚀 Pushing Vanilla-KD Dataset to Hub: {dataset_repo}...")
    hf_dataset = Dataset.from_dict({"text": distilled_texts})
    hf_dataset.push_to_hub(dataset_repo)

    # STRICT MEMORY MANAGEMENT
    print(f"🧹 Sweeping memory buffers for Seed {SEED}...")
    del df_sample
    del distilled_texts
    del hf_dataset
    del inputs
    del outputs
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ Memory cleared. Ready for next seed.\n")

# ==============================================================================
# 4. FINAL CLEANUP
# ==============================================================================
print("\n" + "="*60)
print("🎉 ALL SEEDS DISTILLED (VANILLA-KD) SUCCESSFULLY!")
print("="*60)
print("🧹 Removing Teacher model from VRAM...")
del teacher
del tokenizer
gc.collect()
torch.cuda.empty_cache()
print("✅ VRAM fully released. Pipeline complete.")
