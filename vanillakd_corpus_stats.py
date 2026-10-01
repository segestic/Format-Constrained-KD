"""
vanillakd_corpus_stats.py
==========================
CPU-only. Characterizes the Vanilla-KD training corpus of each seed
(<hub_user>/primekg-phi3-vanillakd-sft-seed<N>) and relates it to how much the
trained students retain the output template (review comment #5).

Per seed, over the teacher completions (text after "Target Node: "):
  kept                 completions kept (empty ones were dropped at generation)
  first_line_template  first line matches "[Name] (Type: [Class])" with a name
  any_type_tag         "(Type:" appears anywhere
  multi_sentence       contains an explanation (a sentence break after line 1,
                       or more than one line)
  len_mean / len_sd    completion length in characters
  first_line_len_mean  length of the first line (what the parser reads)
  contains_gt          a ground-truth target for that (source, relation)
                       appears in the completion (case-insensitive)

Then joins per-seed downstream template compliance of the Vanilla-KD arm from
format_compliance_outputs/per_seed.csv (run format_compliance_analysis.py
first) and prints Spearman correlations over the five seeds. With n = 5 these
correlations are descriptive only.

USAGE:  python vanillakd_corpus_stats.py
"""

import os
import re

import pandas as pd
from datasets import load_dataset
from scipy import stats
from tdc.resource import PrimeKG

HUB = os.environ.get("HF_HUB_USERNAME", "olusegunola")
SEEDS = [7, 42, 101, 999, 2024]
TEMPLATE = re.compile(r"^(.*?)\s*\(\s*Type:\s*(.*?)\s*\)", re.IGNORECASE)
PARSE = re.compile(r"Source Node: (.*) \(Type: .*?\)\nRelationship: (.*)\nTarget Node: (.*)", re.DOTALL)

kg = PrimeKG(path="./data").get_data().dropna(subset=["x_name", "y_name"])
gt = kg.groupby(["x_name", "relation"])["y_name"].apply(lambda s: [str(v).lower() for v in s]).to_dict()

rows = []
for seed in SEEDS:
    ds = load_dataset(f"{HUB}/primekg-phi3-vanillakd-sft-seed{seed}", split="train")
    stats_rows = []
    for text in ds["text"]:
        m = PARSE.search(text)
        if not m:
            continue
        src, rel, comp = m.group(1), m.group(2), m.group(3).strip()
        first = comp.split("\n")[0].strip()
        tm = TEMPLATE.search(first)
        targets = gt.get((src, rel), [])
        low = comp.lower()
        stats_rows.append({
            "first_line_template": bool(tm and tm.group(1).strip()),
            "any_type_tag": "(type:" in low,
            "multi_sentence": ("\n" in comp) or bool(re.search(r"[.!?]\s+\S", first)),
            "len": len(comp),
            "first_line_len": len(first),
            "contains_gt": any(t and t in low for t in targets),
        })
    s = pd.DataFrame(stats_rows)
    rows.append({"seed": seed, "kept": len(ds),
                 "first_line_template_%": 100 * s["first_line_template"].mean(),
                 "any_type_tag_%": 100 * s["any_type_tag"].mean(),
                 "multi_sentence_%": 100 * s["multi_sentence"].mean(),
                 "len_mean": s["len"].mean(), "len_sd": s["len"].std(),
                 "first_line_len_mean": s["first_line_len"].mean(),
                 "contains_gt_%": 100 * s["contains_gt"].mean()})

corpus = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\nVANILLA-KD CORPUS, PER SEED\n", corpus.round(2).to_string(index=False))
corpus.to_csv("vanillakd_corpus_stats.csv", index=False)

path = "format_compliance_outputs/per_seed.csv"
if os.path.exists(path):
    fc = pd.read_csv(path)
    fc = fc[fc["arm"] == "Vanilla-KD"][["mode", "backbone", "seed", "format_compliance_pct"]]
    print("\nSPEARMAN (n = 5 seeds, descriptive): corpus statistic vs. student template compliance")
    for (mode, bb), g in fc.groupby(["mode", "backbone"]):
        j = corpus.merge(g, on="seed")
        for col in ["first_line_template_%", "any_type_tag_%", "multi_sentence_%", "len_mean",
                    "first_line_len_mean", "contains_gt_%"]:
            rho, p = stats.spearmanr(j[col], j["format_compliance_pct"])
            print(f"  [{mode} | {bb}] {col:<22} rho={rho:+.2f}  p={p:.3f}")
else:
    print(f"\n(Run format_compliance_analysis.py first to create {path}.)")
