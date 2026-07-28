import os
import re
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

# 🚀 HIGH-SPEED INFERENCE SETTING
# If you have 24GB VRAM, try 32 or 64. If you have 80GB VRAM, try 128 or 256.
INFERENCE_BATCH_SIZE = 64 

# ==============================================================================
# 1. LOAD DATA ONCE
# ==============================================================================
print(f"📚 Loading PrimeKG base matrix...")
data = PrimeKG(path='./data')
df_base = data.get_data().dropna(subset=['x_name', 'y_name'])
print(f"✅ Loaded {len(df_base)} total PrimeKG rows.")

# ==============================================================================
# 2. LOAD TEACHER MODEL ONCE
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

# 🚨 CRITICAL FOR BATCHED GENERATION: Models must be left-padded
tokenizer.padding_side = "left" 

#attn_impl = "flash_attention_2" if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else "eager"
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
# 3. THE MULTI-SEED LOOP (BATCHED)
# ==============================================================================
for SEED in SEEDS:
    print("\n" + "="*60)
    print(f"🚀 STARTING FAST DISTILLATION RUN FOR SEED: {SEED}")
    print("="*60)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    df_sample = df_base.groupby('relation', group_keys=False).apply(
        lambda x: x.sample(min(len(x), GLOBAL_SAMPLE_CAP), random_state=SEED)
    ).copy()
    
    print(f"✅ Isolated {len(df_sample)} rows for Seed {SEED}.")

    distilled_texts = []
    fallback_count = 0

    # 🚀 Process in high-speed batches instead of row-by-row
    for i in tqdm(range(0, len(df_sample), INFERENCE_BATCH_SIZE), desc=f"Distilling Seed {SEED}"):
        
        chunk = df_sample.iloc[i : i + INFERENCE_BATCH_SIZE]
        batch_prompts = []
        
        # 1. Build the prompts for the whole batch
        for _, row in chunk.iterrows():
            prompt = (
                f"<|user|>\n"
                f"You are a strict biomedical extraction system. Given a Source Node and a Relation, identify the Target Node.\n"
                f"CRITICAL INSTRUCTION: Output ONLY the exact Target Node and its Type in the following format: [Node Name] (Type: [Class]). Do not write any explanations, greetings, or filler text.\n\n"
                f"Source: {row['x_name']} (Type: {row['x_type']})\n"
                f"Relation: {row['relation']}\n"
                f"Target Node:<|end|>\n<|assistant|>\n"
            )
            batch_prompts.append(prompt)

        # 2. Tokenize the entire batch at once (Left-Padded)
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(teacher.device)

        # 3. Generate across the entire batch matrix simultaneously
        with torch.no_grad():
            outputs = teacher.generate(
                **inputs,
                max_new_tokens=25, 
                temperature=0.1,   
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        # 4. Slice off the padded prompts and decode the new tokens
        generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # 5. Apply the regex filter mapped back to each original row
        for gen_text, (_, row) in zip(generated_texts, chunk.iterrows()):
            match = re.search(r'^(.*?)\s*\(\s*Type:\s*(.*?)\s*\)', gen_text.strip(), re.IGNORECASE)
            
            if match:
                clean_node = match.group(1).strip()
                clean_type = match.group(2).strip()
            else:
                clean_node = row['y_name']
                clean_type = row['y_type']
                fallback_count += 1

            sft_string = (
                f"Analyze the biomedical relationship.\n"
                f"Source Node: {row['x_name']} (Type: {row['x_type']})\n"
                f"Relationship: {row['relation']}\n"
                f"Target Node: {clean_node} (Type: {clean_type})"
            )
            
            distilled_texts.append(sft_string)

    print(f"\n✅ Distillation for Seed {SEED} Complete!")
    print(f"⚠️ Regex Fallbacks triggered: {fallback_count}/{len(df_sample)} ({(fallback_count/len(df_sample))*100:.2f}%)")

    # Push to HuggingFace Hub
    dataset_repo = f"{HUB_USERNAME}/primekg-phi3-distilled-sft-seed{SEED}"
    print(f"🚀 Pushing Distilled Dataset to Hub: {dataset_repo}...")
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
print("🎉 ALL SEEDS DISTILLED SUCCESSFULLY!")
print("="*60)

print("🧹 Removing Teacher model from VRAM...")
del teacher
del tokenizer
gc.collect()
torch.cuda.empty_cache()
print("✅ VRAM fully released. Pipeline complete.")
