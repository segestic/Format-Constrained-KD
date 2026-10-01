"""
seen_unseen_standard_analysis.py
=================================
CPU-only. Reproduces each seed's training sample exactly as data_utils.py
builds it, then tags every row of the existing Standard-mode (in-domain)
evaluation CSVs as:

    trained  -- the (x, relation, y) triple, in either direction, is in the
                SFT/ORPO TRAIN split for that seed
    sampled  -- in that seed's training sample but in its 10% test split
                (never trained on)
    unseen   -- not in that seed's training sample at all

and reports OV F1, OV rate, and Ranking Accuracy per group, per arm, per
seed, plus paired tests of unseen vs trained. This answers two questions
without any new inference:

  1. How much of the "held-out" Standard evaluation was actually trained on?
     (evaluate.prepare_data samples up to 10,000 edges per relation with the
     same random_state that data_utils uses for its 2,000-edge training
     sample, and pandas draws both from the same permutation, so the training
     sample is contained in the Standard evaluation pool.)
  2. A within-evaluation version of the review comment #8 pilot: does a
     frozen checkpoint perform differently on trained vs never-seen edges
     from the same graph, same prompts, same batches?

LIMITATION: FC-KD and Vanilla-KD were additionally trained on teacher
completions built by the distillation scripts. If those scripts sample
PrimeKG rows differently from data_utils.py, "unseen" is only guaranteed
unseen for SFT/ORPO training data. Check black_box_distill.py and
distill_vanilla_kd.py.

USAGE:
    python seen_unseen_standard_analysis.py
Outputs land in ./seen_unseen_outputs/.
"""

import os

import numpy as np
import pandas as pd
from datasets import Dataset
from scipy import stats
from tdc.resource import PrimeKG

OUTPUT_DIR = "./eval_outputs"
RESULTS_DIR = "./seen_unseen_outputs"
HUB_USER = os.environ.get("HF_HUB_USERNAME", "olusegunola")
SEEDS = [7, 42, 101, 999, 2024]
SAMPLE_CAP = 2000  # CONFIG["GLOBAL_SAMPLE_CAP"]
BACKBONES = {"PHI": ("phi-1.5", "phi-1_5"), "QWEN": ("qwen2.5-1.5b", "Qwen2.5-1.5B")}
ARMS = {"FC-KD": "dkd", "SFT": "sft", "ORPO": "orpo", "Vanilla-KD": "vanillakd", "Control": None,
        "Ctl-EvalPrompt": "ctl-evalprompt", "Ctl-TrainPrompt": "ctl-trainprompt"}


def training_keys(df: pd.DataFrame, seed: int):
    """Exactly data_utils.PrimeKGDataLoader: per-relation sample, then a
    Dataset.train_test_split(test_size=0.1, seed=seed) on rows in that order.
    Returns (train_keys, sample_keys), each containing both edge directions."""
    parts = [g.sample(min(len(g), SAMPLE_CAP), random_state=seed) for _, g in df.groupby("relation", sort=True)]
    df_sample = pd.concat(parts)
    # train_test_split's permutation depends only on length and seed, not content.
    split = Dataset.from_dict({"i": list(range(len(df_sample)))}).train_test_split(test_size=0.1, seed=seed)
    train_rows = df_sample.iloc[split["train"]["i"]]

    def keys(frame):
        k = set(zip(frame["x_name"], frame["relation"], frame["y_name"]))
        return k | {(y, r, x) for (x, r, y) in k}

    return keys(train_rows), keys(df_sample)


def score(df: pd.DataFrame) -> dict:
    ext = df["Ext_Class"]
    n = len(df)
    ov_tp = ext.isin(["tp", "fp_fn"]).sum()
    ov_fp = ext.isin(["hallucination", "topology_violation"]).sum()
    ov_fn = (ext == "fn").sum()
    p = ov_tp / (ov_tp + ov_fp) if (ov_tp + ov_fp) else 0.0
    r = ov_tp / (ov_tp + ov_fn) if (ov_tp + ov_fn) else 0.0
    return {
        "n": int(n),
        "ov_f1_pct": 100 * (2 * p * r / (p + r) if (p + r) else 0.0),
        "ov_rate_pct": 100 * ov_tp / n if n else np.nan,
        "ranking_acc_pct": 100 * df["Rank_Is_Correct"].mean() if n else np.nan,
    }


def csv_path(backbone, arm, seed):
    student_slug, control_slug = BACKBONES[backbone]
    prefix = (f"control-{control_slug}-seed{seed}" if ARMS[arm] is None
              else f"{HUB_USER}_{student_slug}-primekg-{ARMS[arm]}-seed{seed}")
    return os.path.join(OUTPUT_DIR, f"{prefix}_standard_data.csv")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Loading PrimeKG...")
    df = PrimeKG(path="./data").get_data().dropna(subset=["x_name", "y_name"]).astype(str)

    rows, relation_rows = [], []
    for seed in SEEDS:
        train_k, sample_k = training_keys(df, seed)
        for backbone in BACKBONES:
            for arm in ARMS:
                path = csv_path(backbone, arm, seed)
                if not os.path.exists(path):
                    print(f"⚠️ Missing: {path}")
                    continue
                ev = pd.read_csv(path, dtype={"Source_Node": str, "Relation": str, "Ground_Truth": str})
                triples = list(zip(ev["Source_Node"], ev["Relation"], ev["Ground_Truth"]))
                ev["group"] = ["trained" if t in train_k else ("sampled" if t in sample_k else "unseen") for t in triples]
                for g in ["trained", "sampled", "unseen", "all"]:
                    sub = ev if g == "all" else ev[ev["group"] == g]
                    rows.append({"backbone": backbone, "arm": arm, "seed": seed, "group": g, **score(sub)})
                if arm == "FC-KD":
                    rc = ev.groupby(["Relation", "group"]).size().unstack(fill_value=0)
                    rc["backbone"], rc["seed"] = backbone, seed
                    relation_rows.append(rc.reset_index())

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS_DIR, "per_seed_by_group.csv"), index=False)
    if relation_rows:
        pd.concat(relation_rows).to_csv(os.path.join(RESULTS_DIR, "fckd_relation_by_group_counts.csv"), index=False)

    pd.set_option("display.width", 220)
    share = res[res["group"] != "all"].pivot_table(index=["backbone", "arm", "seed"], columns="group", values="n", fill_value=0)
    share = share.div(share.sum(axis=1), axis=0).mul(100).round(1)
    print("\n" + "=" * 80 + "\nSHARE OF STANDARD-MODE EVAL ROWS BY GROUP (%)\n" + "=" * 80)
    print(share.groupby(level=["backbone", "arm"]).mean().round(1).to_string())

    summ = res.groupby(["backbone", "arm", "group"])[["n", "ov_f1_pct", "ov_rate_pct", "ranking_acc_pct"]].agg(["mean", "std"]).round(2)
    summ.to_csv(os.path.join(RESULTS_DIR, "mean_sd_by_group.csv"))
    print("\n" + "=" * 80 + "\nMETRICS BY GROUP, MEAN ± SD OVER SEEDS\n" + "=" * 80)
    print(summ.to_string())

    print("\n" + "=" * 80 + "\nPAIRED: unseen minus trained, per arm (exact Wilcoxon over seeds)\n" + "=" * 80)
    for (backbone, arm), g in res.groupby(["backbone", "arm"]):
        wide = g.pivot(index="seed", columns="group", values=["ov_rate_pct", "ranking_acc_pct"])
        for metric in ["ov_rate_pct", "ranking_acc_pct"]:
            if ("unseen" not in wide[metric]) or ("trained" not in wide[metric]):
                continue
            pair = wide[metric][["unseen", "trained"]].dropna()
            if len(pair) < 3:
                continue
            d = pair["unseen"] - pair["trained"]
            if np.all(d == 0):
                p = 1.0
            else:
                try:
                    p = stats.wilcoxon(pair["unseen"], pair["trained"], method="exact").pvalue
                except TypeError:
                    p = stats.wilcoxon(pair["unseen"], pair["trained"], mode="exact").pvalue
            print(f"[{backbone} | {arm:<10}] {metric:<16} diff={d.mean():+6.2f}pp  "
                  f"unseen higher on {(d > 0).sum()}/{len(d)}  p={p:.4f}")
    print("\nNote: group sizes differ by relation; see fckd_relation_by_group_counts.csv "
          "before interpreting differences as a pure seen/unseen effect.")
    print(f"\n✅ Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
