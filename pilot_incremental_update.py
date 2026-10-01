"""
pilot_incremental_update.py
============================
Pilot of incremental knowledge updating (review comment #8): a trained
checkpoint receives a small "update" of new PrimeKG edges by brief continued
fine-tuning, and we measure

  acquisition  change on the UPDATE edges U (learned during the update)
  retention    change on OLD edges O (from the original training split)
  transfer     change on HELD-OUT edges H (never seen, before or after)

All three sets are disjoint, relation-stratified, and fixed per seed:
  O  sampled from the seed's original SFT training split
  U  and  H  sampled from edges outside the seed's training sample (both
     edge directions excluded), as in evaluate.py EVAL_MODE="heldout"

Metrics come from evaluate.Evaluator's own batched pipeline (same prompt,
traps, classification, batch size): Ontological Validity rate (tp + fp_fn),
Strict Exact Match rate, and Ranking Accuracy. Paired before/after tests are
exact Wilcoxon over seeds.

Scope (state in the paper): a pilot of update-by-fine-tuning on one graph
schema; no new entity types or relation types, no retrieval-based updating.

USAGE (per backbone):
    python pilot_incremental_update.py                 # arms FC-KD and SFT
    PILOT_ARMS=dkd,ctl-evalprompt python pilot_incremental_update.py
Writes ./pilot_incremental_update_outputs/<backbone>_pilot.csv
"""

import copy
import gc
import os
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from scipy import stats
from trl import SFTConfig, SFTTrainer

from config import CONFIG, EVAL_CONFIG, set_global_seed
from evaluate import Evaluator
from models import get_quantized_base_model, get_tokenizer

N_PER_SET = int(os.environ.get("PILOT_N", 500))
UPDATE_EPOCHS = int(os.environ.get("PILOT_EPOCHS", 3))
ARMS = os.environ.get("PILOT_ARMS", "dkd,sft").split(",")
OUT_DIR = "./pilot_incremental_update_outputs"

TEXT = ("Task: Analyze the biomedical relationship and output ONLY the Target Node and its Type. "
        "Do not write explanations, exercises, or extra text.\n\n"
        "Source Node: {x_name} (Type: {x_type})\nRelationship: {relation}\n"
        "Target Node: {y_name} (Type: {y_type})")


def per_relation(df, n_total, seed):
    per = max(1, n_total // max(1, df["relation"].nunique()))
    parts = [g.sample(min(len(g), per), random_state=seed) for _, g in df.groupby("relation", sort=True)]
    out = pd.concat(parts)
    return out.sample(n=min(n_total, len(out)), random_state=seed).sort_index()


def build_sets(df, seed):
    cap = CONFIG["GLOBAL_SAMPLE_CAP"]
    sample = pd.concat([g.sample(min(len(g), cap), random_state=seed)
                        for _, g in df.groupby("relation", sort=True)])
    # SFT training split exactly as data_utils: train_test_split on row order
    idx = Dataset.from_dict({"i": list(range(len(sample)))}).train_test_split(test_size=0.1, seed=seed)["train"]["i"]
    trained = sample.iloc[idx]
    keys = set(zip(sample["x_name"], sample["relation"], sample["y_name"]))
    keys |= {(y, r, x) for (x, r, y) in keys}
    rest = df[~df.index.isin(sample.index)]
    unseen = rest[[k not in keys for k in zip(rest["x_name"], rest["relation"], rest["y_name"])]]
    U = per_relation(unseen, N_PER_SET, seed)
    uk = set(zip(U["x_name"], U["relation"], U["y_name"])); uk |= {(y, r, x) for (x, r, y) in uk}
    remaining = unseen[[k not in uk for k in zip(unseen["x_name"], unseen["relation"], unseen["y_name"])]]
    H = per_relation(remaining.drop(U.index, errors="ignore"), N_PER_SET, seed + 1)
    O = per_relation(trained, N_PER_SET, seed)
    return {"old": O, "update": U, "heldout": H}


def score(ev, model, tok, records, seed):
    rows, _, _, _ = ev._evaluate_dataset_batched(records, model, tok, seed)
    d = pd.DataFrame(rows)
    return {"ov_rate": 100 * d["Ext_Class"].isin(["tp", "fp_fn"]).mean(),
            "em": 100 * (d["Ext_Class"] == "tp").mean(),
            "rank": 100 * d["Rank_Is_Correct"].mean()}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = copy.deepcopy(EVAL_CONFIG); cfg["EVAL_MODE"] = "heldout"
    ev = Evaluator(cfg)
    ev.seeds = []                       # only vocabulary / ontology are needed
    ev.prepare_data()
    df = ev._load_primekg()
    slug = CONFIG["s1_output_dir"].rsplit("-primekg-sft", 1)[0]
    backbone = CONFIG["student_name"].split("/")[-1]
    out_csv = os.path.join(OUT_DIR, f"{backbone}_pilot.csv")
    # Resume: keep finished (arm, seed) runs and skip them.
    results = pd.read_csv(out_csv).to_dict("records") if os.path.exists(out_csv) else []
    done = {(r["arm"], int(r["seed"])) for r in results}

    for arm in ARMS:
        for seed in cfg["SEEDS_TO_EVAL"]:
            if (arm, seed) in done:
                print(f"  ⏭️  {arm} seed {seed} already done -- skipping")
                continue
            set_global_seed(seed)
            sets = build_sets(df, seed)
            recs = {k: v.astype(str).to_dict("records") for k, v in sets.items()}
            repo = f"{CONFIG['hub_username']}/{slug}-primekg-{arm}-seed{seed}"
            print(f"\n=== {repo} ===")
            tok = get_tokenizer(repo)
            base = get_quantized_base_model(CONFIG["student_name"])
            model = PeftModel.from_pretrained(base, repo, is_trainable=True)
            model.eval()
            before = {k: score(ev, model, tok, r, seed) for k, r in recs.items()}

            train_ds = Dataset.from_dict({"text": [TEXT.format(**r) for r in recs["update"]]})
            trainer = SFTTrainer(
                model=model, train_dataset=train_ds, tokenizer=tok,
                args=SFTConfig(dataset_text_field="text", max_seq_length=CONFIG["s3_max_length"],
                               output_dir=f"./tmp_pilot_{arm}_{seed}", num_train_epochs=UPDATE_EPOCHS,
                               per_device_train_batch_size=CONFIG["GLOBAL_BATCH_SIZE"],
                               gradient_accumulation_steps=1, learning_rate=CONFIG["s3_lr"],
                               bf16=torch.cuda.is_bf16_supported(), logging_steps=20,
                               save_strategy="no", optim="paged_adamw_8bit", report_to=[]),
            )
            trainer.train()
            # Release optimizer state and gradients before evaluating again.
            model = trainer.model
            trainer.optimizer = None
            trainer.lr_scheduler = None
            model.zero_grad(set_to_none=True)
            del trainer
            trainer = None
            gc.collect(); torch.cuda.empty_cache()
            model.eval()
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            after = {k: score(ev, model, tok, r, seed) for k, r in recs.items()}

            for s in recs:
                for m in ["ov_rate", "em", "rank"]:
                    results.append({"arm": arm, "seed": seed, "set": s, "metric": m,
                                    "before": before[s][m], "after": after[s][m]})
            pd.DataFrame(results).to_csv(out_csv, index=False)   # save after every run
            del trainer, model, base, tok
            gc.collect(); torch.cuda.empty_cache()

    res = pd.DataFrame(results)
    res.to_csv(out_csv, index=False)
    print("\nMEAN BEFORE -> AFTER (paired exact Wilcoxon over seeds)")
    for (arm, s, m), g in res.groupby(["arm", "set", "metric"]):
        d = g["after"] - g["before"]
        try:
            p = 1.0 if np.all(d == 0) else stats.wilcoxon(g["after"], g["before"], method="exact").pvalue
        except TypeError:
            p = stats.wilcoxon(g["after"], g["before"], mode="exact").pvalue
        print(f"  {arm:<16} {s:<8} {m:<8} {g['before'].mean():6.2f} -> {g['after'].mean():6.2f} "
              f"(Δ {d.mean():+6.2f}; up on {(d > 0).sum()}/{len(d)}; p={p:.4f})")


if __name__ == "__main__":
    main()
