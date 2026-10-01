"""
aggregate_and_test_all_results.py
===================================
Single entry point that covers every reviewer item answerable from data
you've already generated (no GPU / model inference required):

  #1  Descriptive mean +/- SD across seeds, AND paired Wilcoxon +
      bootstrap-CI significance testing (FC-KD vs. each baseline) --
      now covers BOTH Ontological Validity AND Ranking Accuracy.
  #3  In-domain ("standard") vs. cross-domain comparison
  #5  Vanilla-KD variance breakdown (Ext_Class distribution per seed)
  #6  Constrained-decoding hybrid vs. baseline (if those folders exist)
  #9  Expanded noise-robustness, reported separately per noise type

NOT covered here -- these need a model run, a different LLM, or a human
annotator, and are handled by their own separate scripts:
  #2  cross-model/backbone validation      -> re-run Stage 3/4 on Qwen etc.
  #4  LLM-as-judge factual correctness     -> llm_judge_factual_correctness.py
  #7  inter-rater agreement                -> blinded annotation spreadsheet
  #8  incremental knowledge-update pilot   -> pilot_incremental_kg_update.py
  #10 end-to-end ranking workflow          -> end_to_end_ranking_workflow.py

WHAT THIS SCRIPT DOES, IN ORDER
--------------------------------
STEP 1 -- RESCORE (fixes stale Ontological Validity numbers in place)
    Recomputes OV correctly as (tp + fp_fn) / ... directly from each
    CSV's Ext_Class column -- no model, no GPU, no regeneration -- and
    rewrites the "Ontological Validity ... F1:" line in the matching
    _report.txt IN PLACE. A backup of each original .txt is kept
    alongside it the first time it's touched, so this is safe to re-run.

STEP 2 -- AGGREGATE
    Walks every eval_outputs*/ folder, tags each report by Eval_Type and
    Model, and pulls out BOTH Ontological Validity F1 and Strict Exact
    Match F1 as separate rows (matched by full label text, not by which
    "...F1:" line happens to appear first). Produces a mean +/- SD table
    across seeds plus a raw long-format CSV for further analysis.

STEP 3 -- SIGNIFICANCE TESTING (review comment #1)
    FIX: now runs paired Wilcoxon signed-rank + bootstrap 95% CI on BOTH
    metrics -- Ontological Validity AND Ranking Accuracy -- using the
    "Cross Domain" values collected in Step 2 (the paper's reported
    setting). Previously this step only tested OV; Ranking Accuracy
    differences were being described in prose without the same
    statistical backing OV received, which is the same
    "descriptive-not-validated" pattern review comment #1 flagged in the
    original submission. Both metrics now get identical treatment.

STEP 4 -- VANILLA-KD VARIANCE BREAKDOWN (review comment #5)
    For the Vanilla-KD arm specifically, prints each seed's full Ext_Class
    distribution, so a variance outlier can be diagnosed as "wins via more
    tp" vs. "wins via a different failure mix" rather than reported as
    unexplained variance.

USAGE:
    python aggregate_and_test_all_results.py [BASE_DIR]
    BASE_DIR defaults to "." -- point it at the directory containing all
    your eval_outputs*/ folders.
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
SEEDS = [7, 42, 101, 999, 2024]
# ORDER MATTERS: 'vanillakd' MUST come before 'dkd' so it doesn't get
# misclassified (kept from the original script -- "dkd" is a substring
# that could otherwise mismatch against "vanillakd" filenames).
# Control arms first: their names contain none of the other substrings, but
# keeping the most specific names first avoids future collisions.
MODELS = ["ctl-evalprompt", "ctl-trainprompt", "vanillakd", "dkd", "sft", "orpo", "control"]
BACKBONES = ["phi", "qwen"]  # ADDED: explicitly track model backbones

OV_LINE_RE = re.compile(r"Ontological Validity.*?F1:\s*[\d.]+")

METRIC_PATTERNS = {
    "1a. Extraction: Ontological Validity F1": r"Ontological Validity.*?F1:\s*([\d.]+)",
    "1b. Extraction: Strict Exact Match F1": r"Strict Exact Match F1:\s*([\d.]+)",
    "2. Extraction: Vocabulary Hallucinations (%)": r"Vocabulary Hallucinations:\s*([\d.]+)",
    "3. Extraction: Structural Schema Faults (%)": r"Structural Schema Faults:\s*([\d.]+)",
    "4. Ranking: Overall Accuracy (%)": r"Ranking Accuracy:\s*([\d.]+)",
    "5. Ranking: Factual Errors (%)": r"Factual Errors:\s*([\d.]+)",
    "6. Ranking: Topology Errors (%)": r"Topology Errors:\s*([\d.]+)",
    "7. Ranking: Hallucination Errors (%)": r"Hallucination Errors:\s*([\d.]+)",
}

# ==============================================================================
# UTILITY: CLASSIFY MODEL STRATEGY + BACKBONE
# ==============================================================================
def classify_model(filename: str):
    fname = filename.lower()
    arm = next((m for m in MODELS if m in fname), None)
    backbone = next((b for b in BACKBONES if b in fname), None)
    
    if arm is None:
        return None
    if backbone is None:
        return arm.upper()  # fallback, e.g., if just "control"
    
    return f"{backbone.upper()}-{arm.upper()}"


# ==============================================================================
# STEP 1: RESCORE STALE OV VALUES IN PLACE (no GPU, no regeneration)
# ==============================================================================
def compute_correct_ov_f1(csv_path: str):
    df = pd.read_csv(csv_path)
    if "Ext_Class" not in df.columns:
        return None

    counts = df["Ext_Class"].value_counts()
    tp = int(counts.get("tp", 0))
    fp_fn = int(counts.get("fp_fn", 0))
    hallucinations = int(counts.get("hallucination", 0))
    topology_violations = int(counts.get("topology_violation", 0))
    fn = int(counts.get("fn", 0))

    ov_tp = tp + fp_fn
    ov_fp = hallucinations + topology_violations
    ov_fn = fn

    ov_p = ov_tp / (ov_tp + ov_fp) if (ov_tp + ov_fp) > 0 else 0.0
    ov_r = ov_tp / (ov_tp + ov_fn) if (ov_tp + ov_fn) > 0 else 0.0
    ov_f1 = 2 * (ov_p * ov_r) / (ov_p + ov_r) if (ov_p + ov_r) > 0 else 0.0
    return ov_f1


def rescore_all_reports(base_dir: str):
    print("=" * 80)
    print("STEP 1: RESCORING STALE ONTOLOGICAL VALIDITY VALUES (in place)")
    print("=" * 80)

    report_paths = sorted(glob.glob(os.path.join(base_dir, "eval_outputs*", "*_report.txt")))
    if not report_paths:
        print(f"  (no *_report.txt files found under {base_dir}/eval_outputs*/ -- skipping)")
        return

    fixed, unchanged, skipped = 0, 0, 0
    for report_path in report_paths:
        csv_path = report_path.replace("_report.txt", "_data.csv")
        if not os.path.exists(csv_path):
            skipped += 1
            continue

        ov_f1 = compute_correct_ov_f1(csv_path)
        if ov_f1 is None:
            skipped += 1
            continue

        with open(report_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not OV_LINE_RE.search(text):
            skipped += 1
            continue

        new_line = f"Ontological Validity (relaxed structural match) F1: {ov_f1:.4f}"
        current_match = OV_LINE_RE.search(text).group(0)
        if current_match == new_line:
            unchanged += 1
            continue

        new_text = OV_LINE_RE.sub(new_line, text, count=1)

        backup_path = report_path + ".pre_ov_fix.bak"
        if not os.path.exists(backup_path):
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(text)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(new_text)
        fixed += 1

    print(f"  Rescored (value changed): {fixed}")
    print(f"  Already correct:          {unchanged}")
    print(f"  Skipped (no CSV/match):   {skipped}")
    print()


# ==============================================================================
# STEP 2: AGGREGATE ACROSS ALL eval_outputs*/ FOLDERS
# ==============================================================================
def classify_eval_type(filename: str, root: str) -> str:
    fname = filename.lower()
    root_lower = root.lower()

    if "cross_domain" in fname:
        if "constrained_decoding_fckd_hybrid" in root_lower:
            return "Constrained Decoding: FC-KD Hybrid"
        if "constrained_decoding_sft_baseline" in root_lower:
            return "Constrained Decoding: SFT Baseline"
        return "Cross Domain"
    if "heldout" in fname:
        return "Held-Out (PrimeKG)"
    if "standard" in fname:
        return "Standard"
    if "robustness" in fname:
        if "char_swap" in root_lower:
            return "Robustness: Char Swap"
        if "entity_type_corruption" in root_lower:
            return "Robustness: Entity Type Corruption"
        if "invalid_source_target" in root_lower:
            return "Robustness: Invalid Source Target"
        if "predicate_corruption" in root_lower:
            return "Robustness: Predicate Corruption"
        return "Robustness: Other"
    return "unknown"


def aggregate_all_results(base_dir: str):
    print("=" * 80)
    print("STEP 2: EXTRACTING METRICS FROM TXT LOGS & CALCULATING STATISTICS")
    print("=" * 80 + "\n")

    parsed_data = []
    files_found = 0
    files_with_data = 0

    for root, _dirs, files in os.walk(base_dir):
        for file in files:
            if not file.endswith("_report.txt"):
                continue

            seed_match = re.search(r"seed(\d+)", file.lower())
            if not seed_match:
                continue
            seed = int(seed_match.group(1))
            if seed not in SEEDS:
                continue

            # FIX: Use backbone-aware classification
            model_type = classify_model(file)
            if not model_type:
                continue

            files_found += 1
            file_path = os.path.join(root, file)
            eval_type = classify_eval_type(file, root)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            found_metric_in_file = False
            for metric_name, pattern in METRIC_PATTERNS.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    found_metric_in_file = True
                    value = float(match.group(1))
                    parsed_data.append({
                        "Eval_Type": eval_type,
                        "Model": model_type.upper(),
                        "Seed": seed,
                        "Metric": metric_name,
                        "Value": value,
                    })

            if found_metric_in_file:
                files_with_data += 1

    print(f"📁 Text files found matching models/seeds: {files_found}")
    print(f"✅ Files successfully parsed for metrics:  {files_with_data}\n")

    df_raw = pd.DataFrame(parsed_data)
    if df_raw.empty:
        print("❌ ERROR: Found files, but could not find the exact text inside them to extract numbers.")
        return df_raw, None

    df_raw.to_csv(os.path.join(base_dir, "Raw_Metrics_Long.csv"), index=False)

    df_agg = df_raw.groupby(["Eval_Type", "Metric", "Model"])["Value"].agg(["mean", "std"]).reset_index()
    df_agg["std"] = df_agg["std"].fillna(0.0)

    def format_stats(row):
        if "F1" in row["Metric"]:
            return f"{row['mean']:.4f} ± {row['std']:.4f}"
        return f"{row['mean']:.2f} ± {row['std']:.2f}"

    df_agg["Formatted_Value"] = df_agg.apply(format_stats, axis=1)

    df_final = df_agg.pivot(index=["Eval_Type", "Metric"], columns="Model", values="Formatted_Value")
    
    # FIX: Dynamically build column order combining both Backbones and Models
    expected_models = [f"{b.upper()}-{m.upper()}" for b in BACKBONES for m in MODELS] + [m.upper() for m in MODELS]
    available_models = [m for m in expected_models if m in df_final.columns]
    
    df_final = df_final[available_models]
    df_final = df_final.sort_index().reset_index()
    df_final["Metric"] = df_final["Metric"].apply(lambda x: x.split(". ", 1)[1] if ". " in x else x)

    print("📊 FINAL AVERAGED RESULTS ACROSS ALL SEEDS")
    print("-" * 120)
    print(df_final.to_markdown(index=False))
    print("-" * 120)

    csv_filename = os.path.join(base_dir, "Final_Metrics_Averaged.csv")
    df_final.to_csv(csv_filename, index=False)
    print(f"\n✅ Data successfully aggregated and saved to '{csv_filename}'")
    print(f"✅ Raw long-format data saved to 'Raw_Metrics_Long.csv' (used by Steps 3-4 below)\n")

    return df_raw, df_final


# ==============================================================================
# STEP 3: PAIRED SIGNIFICANCE TESTING 
# FIX: Backbone-aware testing for BOTH Ontological Validity AND Ranking Accuracy
# ==============================================================================
def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    diffs = a - b
    n = len(diffs)
    boot_means = np.array([rng.choice(diffs, size=n, replace=True).mean() for _ in range(n_boot)])
    lower, upper = np.percentile(boot_means, [2.5, 97.5])
    return lower, upper


def compare_metric(fckd_vals: dict, base_vals: dict, baseline_name: str, metric_label: str, decimals: int = 4, a_name: str = "FC-KD"):
    common_seeds = sorted(set(fckd_vals.keys()) & set(base_vals.keys()))
    if len(common_seeds) < 3:
        print(f"  ⚠️ Only {len(common_seeds)} seeds with both {a_name} and {baseline_name} results for "
              f"{metric_label} -- too few for a meaningful test.")
        return None

    a = np.array([fckd_vals[s] for s in common_seeds])
    b = np.array([base_vals[s] for s in common_seeds])

    try:
        if np.all(a - b == 0):
            p_value = 1.0
        else:
            try:
                _stat, p_value = stats.wilcoxon(a, b, method="exact")
            except TypeError:  # older SciPy
                _stat, p_value = stats.wilcoxon(a, b, mode="exact")
    except ValueError as e:
        print(f"  ⚠️ Wilcoxon test could not be computed ({e}) -- likely all-tied differences.")
        p_value = None

    ci_lower, ci_upper = paired_bootstrap_ci(a, b)
    mean_diff = (a - b).mean()

    fmt = f"{{:.{decimals}f}}"
    print(f"  [{metric_label}] {a_name} vs {baseline_name} (n={len(common_seeds)} seeds: {common_seeds}):")
    print(f"    {a_name}: {fmt.format(a.mean())} (values: {list(a)})")
    print(f"    {baseline_name}: {fmt.format(b.mean())} (values: {list(b)})")
    print(f"    Mean difference: {fmt.format(mean_diff)}")
    print(f"    95% bootstrap CI on difference: [{fmt.format(ci_lower)}, {fmt.format(ci_upper)}]")
    if p_value is not None:
        sig = "significant at alpha=0.05" if p_value < 0.05 else "NOT significant at alpha=0.05"
        print(f"    Wilcoxon signed-rank p-value: {p_value:.4f} ({sig}; min possible p at N={len(common_seeds)} "
              f"is {2 / (2 ** len(common_seeds)):.4f})")
    print()

    return {
        "baseline": baseline_name,
        "metric": metric_label,
        "n_seeds": len(common_seeds),
        "seeds_used": common_seeds,
        "fckd_mean": float(a.mean()),
        "baseline_mean": float(b.mean()),
        "mean_diff": float(mean_diff),
        "ci_95": [float(ci_lower), float(ci_upper)],
        "wilcoxon_p": float(p_value) if p_value is not None else None,
    }


def run_significance_tests(df_raw: pd.DataFrame):
    print("=" * 80)
    print("STEP 3: PAIRED SIGNIFICANCE TESTING")
    print("=" * 80)
    print("Now iterating through available backbones to compare FC-KD against baselines.\n")

    def seed_map(metric_key: str, model: str):
        rows = df_raw[
            (df_raw["Eval_Type"] == "Cross Domain")
            & (df_raw["Metric"] == metric_key)
            & (df_raw["Model"] == model)
        ]
        return dict(zip(rows["Seed"], rows["Value"]))

    ov_key = "1a. Extraction: Ontological Validity F1"
    rank_key = "4. Ranking: Overall Accuracy (%)"

    ov_available = not df_raw[df_raw["Metric"] == ov_key].empty
    rank_available = not df_raw[df_raw["Metric"] == rank_key].empty

    if not ov_available and not rank_available:
        print("  ⚠️ No Cross Domain rows found for OV or Ranking -- skipping tests.")
        return {}

    results = {}
    
    # FIX: Loop over each backbone explicitly
    for backbone in [b.upper() for b in BACKBONES]:
        fckd_model_name = f"{backbone}-DKD"
        
        # Check if this backbone has DKD run
        if df_raw[df_raw["Model"] == fckd_model_name].empty:
            continue
            
        print(f"\n{'='*40}")
        print(f"🧪 SIGNIFICANCE TESTS FOR: {backbone}")
        print(f"{'='*40}\n")
        
        contrasts = [
            ("DKD", "FC-KD", "SFT", "SFT"),
            ("DKD", "FC-KD", "ORPO", "ORPO"),
            ("DKD", "FC-KD", "VANILLAKD", "Vanilla-KD"),
            # Control arms (train_control_arms.py): what Stage 3 actually adds.
            ("DKD", "FC-KD", "CTL-EVALPROMPT", "Ctl-EvalPrompt"),               # teacher effect
            ("CTL-EVALPROMPT", "Ctl-EvalPrompt", "CTL-TRAINPROMPT", "Ctl-TrainPrompt"),  # eval-prompt effect
            ("CTL-TRAINPROMPT", "Ctl-TrainPrompt", "SFT", "SFT"),                # extra-epochs effect
        ]
        for a_key, a_name, baseline_key, baseline_name in contrasts:
            fckd_model_name = f"{backbone}-{a_key}"
            baseline_model_name = f"{backbone}-{baseline_key}"
            if df_raw[df_raw["Model"] == fckd_model_name].empty or df_raw[df_raw["Model"] == baseline_model_name].empty:
                continue
            
            if ov_available:
                fckd_ov = seed_map(ov_key, fckd_model_name)
                base_ov = seed_map(ov_key, baseline_model_name)
                if fckd_ov and base_ov:
                    results[f"{backbone}_{a_key.lower()}_vs_{baseline_key.lower()}_ov"] = compare_metric(
                        fckd_ov, base_ov, baseline_name, "Ontological Validity (OV)", decimals=4, a_name=a_name
                    )
            
            if rank_available:
                fckd_rank = seed_map(rank_key, fckd_model_name)
                base_rank = seed_map(rank_key, baseline_model_name)
                if fckd_rank and base_rank:
                    results[f"{backbone}_{a_key.lower()}_vs_{baseline_key.lower()}_rank"] = compare_metric(
                        fckd_rank, base_rank, baseline_name, "Ranking Accuracy", decimals=2, a_name=a_name
                    )

    return results


# ==============================================================================
# STEP 4: VANILLA-KD VARIANCE BREAKDOWN (review comment #5)
# ==============================================================================
def run_vanillakd_variance_breakdown(base_dir: str):
    print("=" * 80)
    print("STEP 4: VANILLA-KD VARIANCE BREAKDOWN (review comment #5)")
    print("=" * 80)

    # Note: Using glob, this will grab Vanilla-KD runs for ALL backbones.
    csv_paths = sorted(glob.glob(os.path.join(base_dir, "eval_outputs*", "*vanillakd*cross_domain_data.csv")))
    if not csv_paths:
        print("  ⚠️ No Vanilla-KD cross_domain CSVs found under eval_outputs*/ -- skipping.\n")
        return

    for csv_path in csv_paths:
        seed_match = re.search(r"seed(\d+)", os.path.basename(csv_path).lower())
        seed = seed_match.group(1) if seed_match else "?"
        
        # Identify backbone for printing
        bb_match = next((b.upper() for b in BACKBONES if b in os.path.basename(csv_path).lower()), "UNKNOWN")
        
        df = pd.read_csv(csv_path)
        dist = (df["Ext_Class"].value_counts(normalize=True) * 100).round(2).to_dict()
        ov_rate = round(df["Ext_Class"].isin(["tp", "fp_fn"]).mean() * 100, 2)
        print(f"  [{bb_match}] Seed {seed}: OV rate (tp+fp_fn)={ov_rate}%  |  Ext_Class distribution: {dist}")

    print("\n  Look at whether any outlier seed wins via more 'tp'/'fp_fn' or via a")
    print("  different failure mix -- see analyze_vanillakd_variance.py for the fuller")
    print("  version of this analysis (raw-output length distribution, ranking")
    print("  breakdown, and optional generation-time stats).\n")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "."

    rescore_all_reports(base_dir)
    df_raw, df_final = aggregate_all_results(base_dir)

    if df_raw is not None and not df_raw.empty:
        run_significance_tests(df_raw)
        run_vanillakd_variance_breakdown(base_dir)

    print("=" * 80)
    print("NOT COVERED BY THIS SCRIPT (require model inference / human annotation):")
    print("  #2  cross-model backbone validation   -> re-run training on e.g. Qwen2.5-1.5B")
    print("  #4  LLM-as-judge factual correctness   -> llm_judge_factual_correctness.py")
    print("  #7  inter-rater agreement               -> blinded annotation spreadsheet")
    print("  #8  incremental knowledge-update pilot  -> pilot_incremental_kg_update.py")
    print("  #10 end-to-end ranking workflow          -> end_to_end_ranking_workflow.py")
    print("=" * 80)