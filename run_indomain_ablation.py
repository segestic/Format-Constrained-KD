"""
run_indomain_ablation.py
=========================
Reviewer-requested ablation (review comment #3): disentangle cross-graph
vocabulary shift from the intrinsic generation-vs-discrimination bottleneck
by running the SAME evaluation pipeline in both modes already implemented
in evaluate.py:
    - "cross_domain"  : PrimeKG-trained models evaluated on PharmGKB
                         (the paper's reported setting; SEM = 0.00% here)
    - "standard"       : PrimeKG-trained models evaluated on a held-out
                          PrimeKG split (in-domain; same vocabulary as
                          training)

Logic: if SEM is still ~0% in-domain, the generative bottleneck is the
dominant cause (vocabulary shift is not required to produce the failure).
If SEM is meaningfully higher in-domain, that gap quantifies how much of
the cross-domain SEM=0.00% is attributable to lexical divergence between
PrimeKG and PharmGKB specifically, rather than the format-generation
problem the paper's central claim rests on.

This script makes NO changes to evaluate.py -- both modes were already
implemented and are already routed correctly by Evaluator.prepare_data().
It just runs both configurations back-to-back and prints a side-by-side
summary, using the same report/csv outputs evaluate.py already writes.
"""

import copy
import os
import re

from config import EVAL_CONFIG
from evaluate import Evaluator


def parse_sem_ov_from_report(report_path: str):
    """Pulls Strict Exact Match TP count and Ontological Validity F1 back out
    of the .txt report evaluate.py already writes, so this script doesn't
    need to duplicate any scoring logic -- it just reads what Evaluator
    already computed and saved."""
    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()
    sem_match = re.search(r"Strict Exact Match TP:\s*(\d+)", text)
    ov_match = re.search(r"Ontological Validity.*?F1:\s*([\d.]+)", text)
    return {
        "sem_tp": int(sem_match.group(1)) if sem_match else None,
        "ov_f1": float(ov_match.group(1)) if ov_match else None,
    }


def run_mode(mode: str):
    mode_config = copy.deepcopy(EVAL_CONFIG)
    mode_config["EVAL_MODE"] = mode
    mode_config["output_dir"] = EVAL_CONFIG.get("output_dir", "./eval_outputs")  # same dir; filenames are tagged by mode

    print(f"\n{'#' * 60}\n# RUNNING MODE: {mode.upper()}\n{'#' * 60}")
    evaluator = Evaluator(mode_config)
    evaluator.prepare_data()
    generated_files = evaluator.run_evaluation()
    return [f for f in generated_files if f.endswith(f"_{mode}_report.txt")]


def run_indomain_ablation():
    cross_domain_reports = run_mode("cross_domain")
    standard_reports = run_mode("standard")

    print("\n" + "=" * 70)
    print("IN-DOMAIN vs CROSS-DOMAIN COMPARISON (review comment #3)")
    print("=" * 70)
    print(f"{'Model/Seed':<45} {'SEM TP (cross)':<16} {'SEM TP (std)':<14} {'OV F1 (cross)':<15} {'OV F1 (std)'}")

    # Match cross-domain and standard reports by their shared file_prefix
    # (everything before "_{mode}_report.txt").
    for cd_report in cross_domain_reports:
        prefix = os.path.basename(cd_report).replace("_cross_domain_report.txt", "")
        std_report = next(
            (r for r in standard_reports if os.path.basename(r).replace("_standard_report.txt", "") == prefix),
            None,
        )
        cd_metrics = parse_sem_ov_from_report(cd_report)
        std_metrics = parse_sem_ov_from_report(std_report) if std_report else {"sem_tp": None, "ov_f1": None}
        print(
            f"{prefix:<45} "
            f"{str(cd_metrics['sem_tp']):<16} "
            f"{str(std_metrics['sem_tp']):<14} "
            f"{str(cd_metrics['ov_f1']):<15} "
            f"{str(std_metrics['ov_f1'])}"
        )

    print("\nInterpretation: if SEM TP (std) is also ~0, the bottleneck is generative,")
    print("not primarily a vocabulary-shift artifact. If SEM TP (std) is notably higher")
    print("than SEM TP (cross), that gap is attributable to cross-graph lexical divergence.")

    return {"cross_domain_reports": cross_domain_reports, "standard_reports": standard_reports}


if __name__ == "__main__":
    run_indomain_ablation()
