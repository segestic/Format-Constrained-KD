#!/usr/bin/env bash
# run_revision_pipeline.sh -- full revision run, both backbones.
# Usage:  bash run_revision_pipeline.sh 2>&1 | tee pipeline.log
# Resume a single step by commenting out the steps already done.
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---------------------------------------------------------------------------
# 0. Archive previous evaluation outputs OUTSIDE the project directory, so the
#    aggregator (which walks the whole project) cannot pick up stale reports.
# ---------------------------------------------------------------------------
ARCHIVE="../eval_archive_$(date +%Y%m%d_%H%M)"
if compgen -G "eval_outputs*" > /dev/null; then
  mkdir -p "$ARCHIVE" && mv eval_outputs* "$ARCHIVE"/
  echo "Archived old outputs to $ARCHIVE"
fi

for STUDENT in "microsoft/phi-1_5" "Qwen/Qwen2.5-1.5B"; do
  export STUDENT_NAME="$STUDENT"
  echo "=================== BACKBONE: $STUDENT_NAME ==================="

  # 1. Train the two control arms (GPU; 5 seeds each).
  python train_control_arms.py --arm evalprompt
  python train_control_arms.py --arm trainprompt

  # 2. Main evaluation, both modes (GPU). All 7 arms x 5 seeds, NF4, batch 24.
  EVAL_MODE=cross_domain python evaluate.py
  EVAL_MODE=standard     python evaluate.py

  # 3. Noise robustness, 4 noise types x 7 arms x 5 seeds (GPU).
  python evaluate_expanded_noise_robustness.py

  # 4. Constrained decoding on SFT and FC-KD (GPU).
  python evaluate_constrained_decoding_baseline.py
done

# ---------------------------------------------------------------------------
# 5. CPU-only analyses over everything produced above.
# ---------------------------------------------------------------------------
python aggregate_and_test_all_results.py . > stage12_output.txt
python format_compliance_analysis.py        > format_compliance_output.txt
python seen_unseen_standard_analysis.py     > seen_unseen_output.txt
echo "Done. Send stage12_output.txt, format_compliance_output.txt, seen_unseen_output.txt."
