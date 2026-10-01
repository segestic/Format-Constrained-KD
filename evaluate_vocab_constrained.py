"""
evaluate_vocab_constrained.py
==============================
New study: vocabulary-constrained decoding for knowledge-graph triple generation.

Idea. The revision showed that cross-graph generation fails lexically: models
emit well-formed "[Name] (Type: [Class])" triples naming entities that do not
exist in the target graph. This script restricts generation, token by token,
to continuations that spell a real "<entity name> (Type: <entity type>)" string
from the TARGET graph (a prefix trie over the graph's vocabulary, tokenized with
each backbone's tokenizer). Ontological Validity becomes 100% by construction,
so the question becomes whether the model then selects the CORRECT entity.

Pre-specified outcomes (fix these before looking at results):
  PRIMARY    exact match (EM) of the top-1 constrained output vs. the ground
             truth, cross-domain (PharmGKB), per arm and backbone, compared with
             unconstrained EM from the existing Stage 4 CSVs (same rows/seeds).
  SECONDARY  Hits@k (k = 5, beam search), type accuracy, the same on PrimeKG
             in-domain, and the "associated"-relation subset.
  TEST       per seed: exact McNemar test on paired rows; across seeds: exact
             Wilcoxon on per-seed EM (min p = 0.0625 at n = 5). Report both.
  REFERENCE  chance EM under the constraint = mean over rows of
             1 / (number of vocabulary entries sharing the ground-truth type).

No training is needed: every checkpoint from the FC-KD study is reused.

USAGE (per backbone, as with evaluate.py):
    python evaluate_vocab_constrained.py                      # cross-domain
    EVAL_MODE=standard python evaluate_vocab_constrained.py   # in-domain
Outputs: ./eval_outputs_vocab_constrained[_standard]/<model>-seed<N>_vocab.csv
         and a summary printed at the end.
"""

import copy
import gc
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from config import EVAL_CONFIG
from evaluate import Evaluator

NUM_BEAMS = int(os.environ.get("VOCAB_NUM_BEAMS", 5))   # k for Hits@k
MAX_NEW_TOKENS = int(os.environ.get("VOCAB_MAX_NEW_TOKENS", 48))


# ==============================================================================
# PREFIX TRIE OVER THE TARGET GRAPH'S "<name> (Type: <type>)" STRINGS
# ==============================================================================
class TokenTrie:
    def __init__(self):
        self.root = {}
        self.END = -1  # key marking "a complete candidate ends here"

    def add(self, ids):
        node = self.root
        for t in ids:
            node = node.setdefault(t, {})
        node[self.END] = True

    def walk(self, ids):
        node = self.root
        for t in ids:
            node = node.get(t)
            if node is None:
                return None
        return node


def candidate_strings(df: pd.DataFrame):
    """All (name, type) pairs of the graph, as the exact target strings."""
    pairs = set(zip(df["x_name"], df["x_type"])) | set(zip(df["y_name"], df["y_type"]))
    return sorted(f"{n} (Type: {t})" for n, t in pairs if str(n).strip())


def build_trie(tokenizer, candidates, prompt_example):
    """Tokenize each candidate exactly as it would follow the prompt.

    The evaluation prompt ends with "Target Node: " (trailing space). We tokenize
    prompt+candidate and keep the tokens after the prompt's own tokenization, so
    the trie matches what the model would actually produce after this prompt,
    whatever the tokenizer does with the trailing space. Candidates whose
    tokenization is not prompt-prefix-stable are counted and reported.
    """
    trie = TokenTrie()
    p_ids = tokenizer(prompt_example, add_special_tokens=False)["input_ids"]
    unstable = 0
    for cand in candidates:
        full = tokenizer(prompt_example + cand, add_special_tokens=False)["input_ids"]
        if full[:len(p_ids)] != p_ids:
            unstable += 1
            continue
        trie.add(full[len(p_ids):])
    return trie, unstable


class TrieConstrainedLogitsProcessor(LogitsProcessor):
    """Allows only tokens that continue a candidate in the trie; allows EOS only
    at the end of a complete candidate. Works with greedy and beam search."""

    def __init__(self, trie: TokenTrie, prompt_len: int, eos_id: int):
        self.trie, self.prompt_len, self.eos_id = trie, prompt_len, eos_id

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        gen = input_ids[:, self.prompt_len:].tolist()
        for b, ids in enumerate(gen):
            # once EOS has been emitted (finished beam), keep emitting EOS
            if self.eos_id in ids:
                mask[b, self.eos_id] = 0.0
                continue
            node = self.trie.walk(ids)
            if node is None:           # cannot happen unless the trie is empty
                mask[b, self.eos_id] = 0.0
                continue
            allowed = [t for t in node if t != self.trie.END]
            if allowed:
                mask[b, allowed] = 0.0
            if self.trie.END in node:
                mask[b, self.eos_id] = 0.0
        return scores + mask


# ==============================================================================
# EVALUATOR
# ==============================================================================
def parse_candidate(text: str):
    text = text.strip()
    if text.endswith(")") and " (Type: " in text:
        name, typ = text[:-1].rsplit(" (Type: ", 1)
        return name.strip(), typ.strip()
    return text, None


class VocabConstrainedEvaluator(Evaluator):

    def prepare_data(self):
        # Keep the full evaluation graph so the trie covers its whole vocabulary.
        orig_pharm, orig_prime = self._load_pharmgkb, self._load_primekg

        def keep(loader):
            def wrapped():
                self.graph_df = loader()
                return self.graph_df
            return wrapped
        self._load_pharmgkb, self._load_primekg = keep(orig_pharm), keep(orig_prime)
        super().prepare_data()
        self.candidates = candidate_strings(self.graph_df)
        types = self.graph_df[["y_name", "y_type"]].drop_duplicates()
        self.n_per_type = pd.concat([
            self.graph_df[["x_name", "x_type"]].rename(columns={"x_name": "n", "x_type": "t"}),
            self.graph_df[["y_name", "y_type"]].rename(columns={"y_name": "n", "y_type": "t"}),
        ]).drop_duplicates().groupby("t").size().to_dict()
        print(f"  Target vocabulary: {len(self.candidates)} (name, type) candidates.")

    @staticmethod
    def prompt_for(example):
        return (
            "Task: Analyze the biomedical relationship and output ONLY the Target Node and its Type. "
            "Do not write explanations, exercises, or extra text.\n\n"
            f"Source Node: {example['x_name']} (Type: {example['x_type']})\n"
            f"Relationship: {example['relation']}\n"
            f"Target Node: "
        )

    def evaluate_seed(self, model, tokenizer, seed):
        dataset = self.datasets_by_seed[seed]
        trie, unstable = build_trie(tokenizer, self.candidates, self.prompt_for(dataset[0]))
        if unstable:
            print(f"    ⚠️ {unstable} candidates had prompt-unstable tokenization and were excluded.")
        rows = []
        tokenizer.padding_side = "left"
        bs = max(1, self.config.get("batch_size", 16) // NUM_BEAMS)
        for i in range(0, len(dataset), bs):
            batch = dataset[i:i + bs]
            prompts = [self.prompt_for(ex) for ex in batch]
            enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            proc = TrieConstrainedLogitsProcessor(trie, enc["input_ids"].shape[1], tokenizer.eos_token_id)
            with torch.no_grad():
                out = model.generate(
                    **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    num_beams=NUM_BEAMS, num_return_sequences=NUM_BEAMS, early_stopping=True,
                    pad_token_id=tokenizer.eos_token_id,
                    logits_processor=LogitsProcessorList([proc]),
                )
            texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for b, ex in enumerate(batch):
                beams = [parse_candidate(t) for t in texts[b * NUM_BEAMS:(b + 1) * NUM_BEAMS]]
                gt = str(ex["y_name"]).strip().lower()
                names = [n.strip().lower() for n, _ in beams]
                top_name, top_type = beams[0]
                rows.append({
                    "Source_Node": ex["x_name"], "Relation": ex["relation"],
                    "Ground_Truth": ex["y_name"], "Ground_Truth_Type": ex["y_type"],
                    "Top1_Name": top_name, "Top1_Type": top_type,
                    "TopK_Names": " || ".join(n for n, _ in beams),
                    "EM_Top1": int(names[0] == gt),
                    f"Hits@{NUM_BEAMS}": int(gt in names),
                    "Type_Correct": int(top_type == ex["y_type"]),
                    "Valid_Candidate": int(top_type is not None),
                    "Chance_EM": 1.0 / max(1, self.n_per_type.get(ex["y_type"], 1)),
                })
            print(f"    {min(i + bs, len(dataset))}/{len(dataset)}")
        return pd.DataFrame(rows)

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        summary = []
        for base_model_path in self.all_models:
            for seed in self.seeds:
                tag = (f"control-{base_model_path.split('/')[-1]}" if base_model_path == self.config["CONTROL_MODEL"]
                       else base_model_path.replace("/", "_"))
                out_csv = os.path.join(self.output_dir, f"{tag}-seed{seed}_{self.eval_mode}_vocab.csv")
                if os.path.exists(out_csv) and not self.force_rerun:
                    df = pd.read_csv(out_csv)
                else:
                    model, base, tok, name = self._load_model(base_model_path, seed)
                    model.eval()
                    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
                    t0 = time.time()
                    df = self.evaluate_seed(model, tok, seed)
                    df.to_csv(out_csv, index=False)
                    print(f"  ✅ {out_csv}  ({time.time() - t0:.0f}s)")
                    del model, base, tok
                    gc.collect(); torch.cuda.empty_cache()
                summary.append({"model": tag, "seed": seed,
                                "EM_Top1_%": 100 * df["EM_Top1"].mean(),
                                f"Hits@{NUM_BEAMS}_%": 100 * df[f"Hits@{NUM_BEAMS}"].mean(),
                                "Type_Acc_%": 100 * df["Type_Correct"].mean(),
                                "Valid_%": 100 * df["Valid_Candidate"].mean(),
                                "Chance_EM_%": 100 * df["Chance_EM"].mean()})
        s = pd.DataFrame(summary)
        s.to_csv(os.path.join(self.output_dir, f"summary_{self.eval_mode}.csv"), index=False)
        print(s.groupby("model").agg(["mean", "std"]).round(2).to_string())


if __name__ == "__main__":
    cfg = copy.deepcopy(EVAL_CONFIG)
    suffix = "" if cfg["EVAL_MODE"] == "cross_domain" else f"_{cfg['EVAL_MODE']}"
    cfg["output_dir"] = EVAL_CONFIG.get("output_dir", "./eval_outputs") + f"_vocab_constrained{suffix}"
    ev = VocabConstrainedEvaluator(cfg)
    ev.prepare_data()
    ev.run()
