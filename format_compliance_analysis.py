"""
format_compliance_analysis.py
==============================
CPU-only. No model loading, no GPU, no config switching: reads the existing
Stage 4 CSVs for BOTH backbones and reports, per arm and seed:

  - format_compliance_pct : output matches the "[Name] (Type: [Class])"
                            template WITH a non-empty name (same strict regex
                            as evaluate.parse_extraction_output)
  - cat1_syntactically_invalid_pct        = 100 - format_compliance_pct
  - cat2_valid_syntax_invalid_ontology_pct: template OK, Ext_Class in
                                            {hallucination, topology_violation}
  - cat34_ontologically_valid_pct         : template OK, Ext_Class in {tp, fp_fn}
                                            (categories 3 and 4 MERGED -- the
                                            LLM-judge split is not relied on;
                                            see judge-by-relation table below)
  - exact_match_pct                       : Ext_Class == tp

Then runs paired tests (exact two-sided Wilcoxon + paired bootstrap 95% CI)
of FC-KD vs SFT / ORPO / Vanilla-KD / Control on format compliance, per
backbone and eval mode.

If ./llm_judge_outputs/judge_audit_*.csv exist (from
llm_judge_factual_correctness.py), it also prints the judge verdicts
broken down by relation, as evidence for how the judge behaves.

USAGE:
    python format_compliance_analysis.py
Outputs land in ./format_compliance_outputs/.
"""

import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

OUTPUT_DIR = "./eval_outputs"
RESULTS_DIR = "./format_compliance_outputs"
HUB_USER = os.environ.get("HF_HUB_USERNAME", "olusegunola")
SEEDS = [7, 42, 101, 999, 2024]
EVAL_MODES = ["cross_domain", "standard"]

# backbone label -> (student slug used in adapter repo names, control file slug)
BACKBONES = {
    "PHI": ("phi-1.5", "phi-1_5"),
    "QWEN": ("qwen2.5-1.5b", "Qwen2.5-1.5B"),
}
ARMS = {
    "FC-KD": "dkd",
    "SFT": "sft",
    "ORPO": "orpo",
    "Vanilla-KD": "vanillakd",
    "Control": None,
    "Ctl-EvalPrompt": "ctl-evalprompt",
    "Ctl-TrainPrompt": "ctl-trainprompt",
}
BASELINES = ["SFT", "ORPO", "Vanilla-KD", "Control", "Ctl-EvalPrompt", "Ctl-TrainPrompt"]

TEMPLATE_RE = re.compile(r"^(.*?)\s*\(\s*Type:\s*(.*?)\s*\)", re.IGNORECASE)


def template_ok(raw_output) -> bool:
    """Template present AND a non-empty entity name, so that
    format_compliance + cat1 always sums to 100."""
    if raw_output is None or (isinstance(raw_output, float) and np.isnan(raw_output)):
        return False
    m = TEMPLATE_RE.search(str(raw_output).strip())
    return bool(m and m.group(1).strip())


def csv_path(backbone: str, arm: str, seed: int, mode: str) -> str:
    student_slug, control_slug = BACKBONES[backbone]
    if ARMS[arm] is None:
        prefix = f"control-{control_slug}-seed{seed}"
    else:
        prefix = f"{HUB_USER}_{student_slug}-primekg-{ARMS[arm]}-seed{seed}"
    return os.path.join(OUTPUT_DIR, f"{prefix}_{mode}_data.csv")


def score_csv(path: str) -> dict:
    df = pd.read_csv(path)
    n = len(df)
    fmt = df["Raw_Output"].apply(template_ok)
    ext = df["Ext_Class"]
    cat1 = (~fmt) | (ext == "fn")
    return {
        "n": n,
        "format_compliance_pct": 100.0 * (~cat1).sum() / n,
        "cat1_syntactically_invalid_pct": 100.0 * cat1.sum() / n,
        "cat2_valid_syntax_invalid_ontology_pct": 100.0 * ((~cat1) & ext.isin(["hallucination", "topology_violation"])).sum() / n,
        "cat34_ontologically_valid_pct": 100.0 * ((~cat1) & ext.isin(["tp", "fp_fn"])).sum() / n,
        "exact_match_pct": 100.0 * (ext == "tp").sum() / n,
    }


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n_boot)])
    return np.percentile(means, [2.5, 97.5])


def exact_wilcoxon(a: np.ndarray, b: np.ndarray):
    diffs = a - b
    if np.all(diffs == 0):
        return 1.0
    try:
        return float(stats.wilcoxon(a, b, method="exact").pvalue)
    except TypeError:  # older SciPy
        return float(stats.wilcoxon(a, b, mode="exact").pvalue)
    except ValueError:
        return float("nan")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    for mode in EVAL_MODES:
        for backbone in BACKBONES:
            for arm in ARMS:
                for seed in SEEDS:
                    path = csv_path(backbone, arm, seed, mode)
                    if not os.path.exists(path):
                        print(f"⚠️ Missing: {path}")
                        continue
                    rows.append({"mode": mode, "backbone": backbone, "arm": arm, "seed": seed, **score_csv(path)})

    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        print("No CSVs found -- check OUTPUT_DIR / HF_HUB_USERNAME.")
        return
    per_seed.to_csv(os.path.join(RESULTS_DIR, "per_seed.csv"), index=False)

    metric_cols = [c for c in per_seed.columns if c.endswith("_pct")]
    summary = per_seed.groupby(["mode", "backbone", "arm"])[metric_cols].agg(["mean", "std"]).round(2)
    summary.to_csv(os.path.join(RESULTS_DIR, "mean_sd.csv"))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\n" + "=" * 90 + "\nFOUR-WAY BREAKDOWN, MEAN ± SD ACROSS SEEDS (categories 3+4 merged)\n" + "=" * 90)
    print(summary.to_string())

    # --- per-seed format compliance, for transparency ---
    print("\n" + "=" * 90 + "\nPER-SEED FORMAT COMPLIANCE (%)  seeds: " + str(SEEDS) + "\n" + "=" * 90)
    for (mode, backbone), g in per_seed.groupby(["mode", "backbone"]):
        piv = g.pivot(index="arm", columns="seed", values="format_compliance_pct").reindex(columns=SEEDS).round(1)
        print(f"\n[{mode} | {backbone}]\n{piv.to_string()}")

    # --- paired tests on format compliance ---
    test_rows = []
    print("\n" + "=" * 90 + "\nPAIRED TESTS ON FORMAT COMPLIANCE: FC-KD vs each baseline\n" + "=" * 90)
    for (mode, backbone), g in per_seed.groupby(["mode", "backbone"]):
        fckd = g[g["arm"] == "FC-KD"].set_index("seed")["format_compliance_pct"]
        for base in BASELINES:
            other = g[g["arm"] == base].set_index("seed")["format_compliance_pct"]
            common = sorted(set(fckd.index) & set(other.index))
            if len(common) < 3:
                continue
            a = fckd.loc[common].to_numpy()
            b = other.loc[common].to_numpy()
            diffs = a - b
            lo, hi = paired_bootstrap_ci(diffs)
            p = exact_wilcoxon(a, b)
            n_pos = int((diffs > 0).sum())
            n_zero = int((diffs == 0).sum())
            test_rows.append({
                "mode": mode, "backbone": backbone, "comparison": f"FC-KD vs {base}",
                "n_seeds": len(common), "fckd_mean": a.mean(), "baseline_mean": b.mean(),
                "mean_diff_pp": diffs.mean(), "ci_low": lo, "ci_high": hi,
                "seeds_favouring_fckd": n_pos, "tied_seeds": n_zero, "wilcoxon_p": p,
            })
            print(f"[{mode} | {backbone}] FC-KD vs {base:<10} "
                  f"diff={diffs.mean():+7.2f}pp  CI=[{lo:+.2f}, {hi:+.2f}]  "
                  f"FC-KD higher on {n_pos}/{len(common)} seeds (ties: {n_zero})  p={p:.4f}")
    pd.DataFrame(test_rows).round(4).to_csv(os.path.join(RESULTS_DIR, "paired_tests.csv"), index=False)
    print("\nNote: exact two-sided Wilcoxon; minimum attainable p at n=5 is 0.0625. "
          "Tied seeds are dropped by the test, which raises the minimum attainable p.")

    # --- judge verdicts by relation, if audit files exist ---
    audits = sorted(glob.glob("./llm_judge_outputs/judge_audit_*.csv"))
    if audits:
        print("\n" + "=" * 90 + "\nLLM-JUDGE VERDICTS BY RELATION (from existing audit files)\n" + "=" * 90)
        for path in audits:
            audit = pd.read_csv(path)
            if audit.empty:
                continue
            # Keep only rows still template-compliant under the non-empty-name rule.
            if "Ext_Predicted" in audit.columns:
                audit = audit[audit["Ext_Predicted"].astype(str).str.strip() != ""]
            table = pd.crosstab(audit["Relation"], audit["verdict"], margins=True)
            print(f"\n{os.path.basename(path)}\n{table.to_string()}")
            table.to_csv(os.path.join(RESULTS_DIR, f"judge_by_relation_{os.path.basename(path)}"))

    print(f"\n✅ Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
