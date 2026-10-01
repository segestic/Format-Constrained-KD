import os
import sys
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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import CONFIG, EVAL_CONFIG, setup_environment, set_global_seed, get_attn_impl

# ==============================================================================
# 0. CONFIGURATION & HELPERS
# ==============================================================================
setup_environment()

SEEDS = EVAL_CONFIG["SEEDS_TO_EVAL"]
GLOBAL_SAMPLE_CAP = CONFIG["GLOBAL_SAMPLE_CAP"]
TEACHER_MODEL = CONFIG["teacher_name"]
HUB_USERNAME = CONFIG["hub_username"]
# 🚀 HIGH-SPEED INFERENCE SETTING
# If you have 24GB VRAM, try 32 or 64. If you have 80GB VRAM, try 128 or 256.
INFERENCE_BATCH_SIZE = EVAL_CONFIG.get("batch_size", 32)
# =======================================================
INFERENCE_BATCH_SIZE = EVAL_CONFIG.get("batch_size", 32)

def parse_constrained_output(gen_text: str):
    """
    Robust parser for the constrained format:
    [Node Name] (Type: [Class])
    Returns (node, type) or (None, None) on failure.
    """
    text = gen_text.strip()
    
    # Remove common leading junk the model sometimes adds
    text = re.sub(r'^[\s\n\r:.\-–—]*', '', text)
    text = re.sub(r'^(Target Node|Answer|Output)\s*[:\-]?\s*', '', text, flags=re.IGNORECASE)
    
    # Primary pattern – very tolerant of whitespace and optional brackets
    patterns = [
        # Ideal: [Name] (Type: Class)   or   Name (Type: Class)
        r'^[\[\"]?(.*?)[\]\"]?\s*\(\s*Type\s*:\s*(.*?)\s*\)',
        # Fallback: Name (Class) without the word "Type"
        r'^[\[\"]?(.*?)[\]\"]?\s*\(\s*(.*?)\s*\)',
        # Last resort: just capture until first parenthesis or end
        r'^[\[\"]?(.*?)[\]\"]?(?:\s*\(|$)',
    ]
    
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            node = m.group(1).strip().strip('[]"\'')
            typ  = m.group(2).strip().strip('[]"\'') if m.lastindex >= 2 else None
            if node and len(node) > 1:          # basic sanity
                return node, typ
    return None, None

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
                f"CRITICAL INSTRUCTION: Output ONLY the exact Target Node and its Type in the following format and nothing else:\n"
                f"[Node Name] (Type: [Class])\n"
                f"Do not write any explanations, greetings, or filler text.\n\n"
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
                max_new_tokens=48,           
                do_sample=False,             
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # 4. Slice off the padded prompts and decode the new tokens
        generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # 5. Apply the robust parser and quality filters mapped back to each original row
        for gen_text, (_, row) in zip(generated_texts, chunk.iterrows()):
            clean_node, clean_type = parse_constrained_output(gen_text)
            
            # Condition A: Strict format constraint check + safety logic
            valid_format = not (clean_node is None or clean_type is None or len(clean_node) < 2 or len(clean_type) < 2)
            
            # Condition B: Quality Filter (Are we accepting a hallucination?)
            # We check if the teacher's prediction is at least a partial match for the ground truth
            gt_node = str(row['y_name']).lower()
            is_accurate = False
            if valid_format:
                pred_node_lower = clean_node.lower()
                if (pred_node_lower == gt_node) or (pred_node_lower in gt_node) or (gt_node in pred_node_lower):
                    is_accurate = True

            # Evaluate final acceptance
            if not (valid_format and is_accurate):
                # Print the first 5 failures to see what's going wrong
                if fallback_count < 5:
                    print(f"\n--- REJECTED TEACHER OUTPUT ---")
                    print(f"Expected: {gt_node}")
                    print(f"Teacher Generated: {gen_text.strip()}")
                    print(f"Reason: Format Valid? {valid_format}, Accurate? {is_accurate}")
                clean_node = row['y_name']
                clean_type = row['y_type']
                fallback_count += 1

            sft_string = (
                f"Task: Analyze the biomedical relationship and output ONLY the Target Node and its Type. "
                f"Do not write explanations, exercises, or extra text.\n\n"
                f"Source Node: {row['x_name']} (Type: {row['x_type']})\n"
                f"Relationship: {row['relation']}\n"
                f"Target Node: {clean_node} (Type: {clean_type})"
            )
            
            distilled_texts.append(sft_string)

    print(f"\n✅ Distillation for Seed {SEED} Complete!")
    print(f"⚠️ Fallbacks to Ground Truth triggered: {fallback_count}/{len(df_sample)} ({(fallback_count/len(df_sample))*100:.2f}%)")

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