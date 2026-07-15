"""
Phi1.5_Base (Seed 42) -> /content/olusegunola_phi-1.5-base-seed42_cross_domain_data.csv
   Mapped Control_Phi1.5_Base (Seed 101) -> /content/olusegunola_phi-1.5-base-seed101_cross_domain_data.csv
   Mapped Control_Phi1.5_Base (Seed 999) -> /content/olusegunola_phi-1.5-base-seed999_cross_domain_data.csv
   Mapped SFT_Phi1.5 (Seed 42) -> /content/olusegunola_phi-1.5-primekg-sft-seed42_cross_domain_data.csv
   Mapped SFT_Phi1.5 (Seed 101) -> /content/olusegunola_phi-1.5-primekg-sft-seed101_cross_domain_data.csv
   Mapped SFT_Phi1.5 (Seed 999) -> /content/olusegunola_phi-1.5-primekg-sft-seed999_cross_domain_data.csv
   Mapped ORPO_Phi1.5 (Seed 42) -> /content/olusegunola_phi-1.5-primekg-orpo-seed42_cross_domain_data.csv
   Mapped ORPO_Phi1.5 (Seed 101) -> /content/olusegunola_phi-1.5-primekg-orpo-seed101_cross_domain_data.csv
   Mapped ORPO_Phi1.5 (Seed 999) -> /content/olusegunola_phi-1.5-primekg-orpo-seed999_cross_domain_data.csv
   Mapped DKD_Phi1.5 (Seed 42) -> /content/olusegunola_phi-1.5-primekg-dkd-seed42_cross_domain_data.csv
   Mapped DKD_Phi1.5 (Seed 101) -> /content/olusegunola_phi-1.5-primekg-dkd-seed101_cross_domain_data.csv
   Mapped DKD_Phi1.5 (Seed 999) -> /content/olusegunola_phi-1.5-primekg-dkd-seed999_cross_domain_data.csv
"""

!mkdir -p data
!wget -O data/relationships.zip "https://api.pharmgkb.org/v1/download/file/data/relationships.zip"
!unzip -j data/relationships.zip '*relationships.tsv' -d data
#rm data/relationships.zip


import pandas as pd
import numpy as np
import os

# ==============================================================================
# 1. CONFIGURATION & FILE GENERATION
# ==============================================================================
SEEDS = [42, 101, 999]

MODELS = {
    "Control_Phi1.5_Base": "olusegunola/phi-1.5-base",
    "SFT_Phi1.5": "olusegunola/phi-1.5-primekg-sft",
    "ORPO_Phi1.5": "olusegunola/phi-1.5-primekg-orpo",
    "DKD_Phi1.5": "olusegunola/phi-1.5-primekg-dkd"
}

FILES = {}
print("🔍 Resolving file paths...")
for model_label, repo_path in MODELS.items():
    FILES[model_label] = {}
    for seed in SEEDS:
        safe_name = f"{repo_path}-seed{seed}".replace('/', '_')
        # Pointing to the exact files you listed
        filename = f"/content/{safe_name}_cross_domain_data.csv"

        if os.path.exists(filename):
            FILES[model_label][seed] = filename
            print(f"   Mapped {model_label} (Seed {seed}) -> {filename}")
        else:
            filename_local = f"{safe_name}_cross_domain_data.csv"
            if os.path.exists(filename_local):
                FILES[model_label][seed] = filename_local
                print(f"   Mapped {model_label} (Seed {seed}) -> {filename_local}")
            else:
                print(f"   ⚠️ Missing {model_label} (Seed {seed}) -> {filename}")

# ==============================================================================
# 2. LOAD VOCABULARY FOR HALLUCINATION CHECKING (Restored to fix your scores)
# ==============================================================================
print(f"\nLoading validation vocabulary to establish Ontological Validity...")
try:
    # Restoring your original logic to get the 5.27% and 11.13% math correct
    tsv_path = './data/relationships.tsv'
    if os.path.exists(tsv_path):
        df_kg = pd.read_csv(tsv_path, sep='\t')
        valid_list = df_kg['Entity1_name'].dropna().astype(str).tolist() + df_kg['Entity2_name'].dropna().astype(str).tolist()
        VALID_NODES = set(x.strip().lower() for x in valid_list)
        print(f"✅ Loaded {len(VALID_NODES)} unique valid nodes for hallucination checking.")
    else:
        print(f"⚠️ TSV not found at {tsv_path}. Validity checks will fail. Please ensure the data folder is present.")
        VALID_NODES = set()
except Exception as e:
    print(f"⚠️ Error loading dataset: {e}. Defaulting to empty set.")
    VALID_NODES = set()

# ==============================================================================
# 3. DATA PROCESSING & PER-SEED SCORE EXTRACTION
# ==============================================================================
results = {
    "Modality 1: Ontological Validity": {m: {} for m in MODELS},
    "Modality 1: Strict Accuracy": {m: {} for m in MODELS},
    "Modality 2: Adversarial Ranking": {m: {} for m in MODELS}
}

print("\nProcessing files and extracting raw seed scores...")

for model_name, seed_files in FILES.items():
    for seed, path in seed_files.items():
        try:
            df = pd.read_csv(path)

            pred_norm = df["Ext_Predicted"].astype(str).str.strip().str.lower()
            truth_norm = df["Ground_Truth"].astype(str).str.strip().str.lower()

            # Metric A: Ontological Validity
            is_valid = np.where((pred_norm != "failed") & (pred_norm.isin(VALID_NODES)), 1, 0)
            results["Modality 1: Ontological Validity"][model_name][seed] = np.mean(is_valid) * 100

            # Metric B: Strict Accuracy
            is_accurate = np.where(pred_norm == truth_norm, 1, 0)
            results["Modality 1: Strict Accuracy"][model_name][seed] = np.mean(is_accurate) * 100

            # Metric C: Adversarial Ranking
            is_ranked_correctly = df["Rank_Is_Correct"].astype(int).values
            results["Modality 2: Adversarial Ranking"][model_name][seed] = np.mean(is_ranked_correctly) * 100

        except Exception as e:
            print(f"⚠️ Error processing {path}: {e}")

# ==============================================================================
# 4. FORMAT OUTPUT AND EXPORT (Reviewer-Proof)
# ==============================================================================
csv_export_data = []

for metric, model_data in results.items():
    for model, seed_scores in model_data.items():
        if not seed_scores:
            continue

        s1 = seed_scores.get(SEEDS[0], 0.0)
        s2 = seed_scores.get(SEEDS[1], 0.0)
        s3 = seed_scores.get(SEEDS[2], 0.0)

        seed_array = [s1, s2, s3]
        mean_val = np.mean(seed_array)
        sd_val = np.std(seed_array, ddof=1) if len(seed_array) > 1 else 0.0

        csv_export_data.append({
            "Metric": metric,
            "Model": model,
            f"Seed_{SEEDS[0]}_%": round(s1, 2),
            f"Seed_{SEEDS[1]}_%": round(s2, 2),
            f"Seed_{SEEDS[2]}_%": round(s3, 2),
            "Mean_%": round(mean_val, 2),
            "Mean_±_SD": f"{mean_val:.2f}% ± {sd_val:.2f}%"
        })

output_filename = "Manuscript_Descriptive_Results.csv"
stats_df = pd.DataFrame(csv_export_data)

print(f"\n{'='*100}")
print(stats_df.to_string(index=False))
print(f"{'='*100}")

stats_df.to_csv(output_filename, index=False)
print(f"✅ Reviewer-compliant descriptive results saved to: {output_filename}")
