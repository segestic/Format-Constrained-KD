"""
audit_fckd_corpus.py
=====================
CPU-only audit of the FC-KD distillation datasets pushed by black_box_distill.py
(<hub_user>/primekg-phi3-distilled-sft-seed<SEED>). Rebuilds each seed's PrimeKG
sample exactly as black_box_distill.py does (same rows, same order), parses the
"Target Node: ... (Type: ...)" line of every training text, and compares it with
the ground-truth row:

  identical_to_ground_truth : name AND type equal the PrimeKG target
                              (either the teacher reproduced it exactly, or the
                              ground-truth fallback was used -- indistinguishable)
  teacher_partial_name      : name differs from ground truth (accepted by the
                              substring filter, e.g. a truncated or extended name)
  same_name_other_type      : name equals ground truth, type differs (teacher's type)

USAGE:
    python audit_fckd_corpus.py
"""

import os
import re

import pandas as pd
from datasets import load_dataset
from tdc.resource import PrimeKG

HUB_USER = os.environ.get("HF_HUB_USERNAME", "olusegunola")
SEEDS = [42, 101, 999, 7, 2024]
SAMPLE_CAP = 2000
TARGET_RE = re.compile(r"Target Node: (.*) \(Type: (.*)\)\s*$", re.DOTALL)

df_base = PrimeKG(path="./data").get_data().dropna(subset=["x_name", "y_name"])

rows = []
for seed in SEEDS:
    df_sample = df_base.groupby("relation", group_keys=False).apply(
        lambda x: x.sample(min(len(x), SAMPLE_CAP), random_state=seed)
    ).copy()
    ds = load_dataset(f"{HUB_USER}/primekg-phi3-distilled-sft-seed{seed}", split="train")
    assert len(ds) == len(df_sample), f"seed {seed}: {len(ds)} texts vs {len(df_sample)} sampled rows"

    counts = {"identical_to_ground_truth": 0, "teacher_partial_name": 0,
              "same_name_other_type": 0, "unparsed": 0}
    examples = []
    for text, (_, gt) in zip(ds["text"], df_sample.iterrows()):
        m = TARGET_RE.search(text)
        if not m:
            counts["unparsed"] += 1
            continue
        name, typ = m.group(1).strip(), m.group(2).strip()
        if name == str(gt["y_name"]) and typ == str(gt["y_type"]):
            counts["identical_to_ground_truth"] += 1
        elif name == str(gt["y_name"]):
            counts["same_name_other_type"] += 1
        else:
            counts["teacher_partial_name"] += 1
            if len(examples) < 5:
                examples.append((gt["y_name"], name))
    n = len(ds)
    rows.append({"seed": seed, "n": n, **{k: round(100 * v / n, 2) for k, v in counts.items()}})
    print(f"\nSeed {seed}: {rows[-1]}")
    for gt_name, used in examples:
        print(f"   ground truth: {gt_name!r:<45} training target: {used!r}")

print("\n" + pd.DataFrame(rows).to_string(index=False))
print("\n(% of training texts per seed)")
