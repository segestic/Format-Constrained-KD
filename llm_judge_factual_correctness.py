"""
llm_judge_factual_correctness.py
==================================
Review comment #4: OV only checks syntax compliance and vocabulary
membership -- it cannot distinguish a semantically plausible-but-wrong
output from a completely unreasonable one, and a knowledge graph can have
MORE than one factually correct target for a given (source, relation) pair,
while exact-match scoring only credits the single row sampled as ground truth.

Four hierarchical categories (each row lands in exactly one):
    1. Syntactically invalid            -- empty output (Ext_Class == "fn")
                                            OR the output does not follow the
                                            "[Name] (Type: [Class])" template.
                                            Checked with the SAME regex that
                                            evaluate.parse_extraction_output
                                            uses for a strict match.
    2. Syntactically valid,
       ontologically invalid            -- template OK, but Ext_Class in
                                            {"hallucination", "topology_violation"}
    3. Ontologically valid,
       factually incorrect              -- template OK, Ext_Class == "fp_fn",
                                            judge says INCORRECT
    4. Factually correct                -- template OK and Ext_Class == "tp",
                                            PLUS template-OK "fp_fn" rows the
                                            judge says CORRECT
    (fp_fn rows whose judge answer cannot be parsed are reported separately
     as "unparseable" and are NOT counted in 3 or 4.)

FIXES vs. previous version:
  - Verdict parsing: previously `"CORRECT" in response`, which is also True
    for "INCORRECT", so every judged row counted as correct. Now parsed with
    word boundaries; unparseable answers are counted, not defaulted.
  - Category 1 previously counted only empty outputs; outputs without the
    template were misfiled as "syntactically valid".
  - Control-model CSVs were never found (wrong filename prefix); the prefix
    now mirrors evaluate.Evaluator._report_paths.
  - Judge prompt built with tokenizer.apply_chat_template.
  - Every judgment (with the raw response) is saved for audit, and verdicts
    are broken down by relation, because "correct completion" is ill-defined
    for "not associated" / "ambiguous" PharmGKB edges.

JUDGE BACKEND: defaults to the pipeline's teacher model
(microsoft/Phi-3-mini-4k-instruct). CAVEAT: the same model family generated
the FC-KD / Vanilla-KD training data, so self-preference bias is possible.
An independent judge (JUDGE_BACKEND="api") is the stronger choice.

Evaluates the backbone currently set in config.py; run once per backbone.
"""

import json
import os
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import CONFIG, EVAL_CONFIG

# "api" = independent OpenAI judge (needs OPENAI_API_KEY); "local_teacher" = Phi-3-mini.
JUDGE_BACKEND = os.environ.get("JUDGE_BACKEND", "api")
# Any chat model your key can access; override with OPENAI_JUDGE_MODEL.
OPENAI_JUDGE_MODEL = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-4o")
JUDGE_MODEL_NAME = CONFIG["teacher_name"] if JUDGE_BACKEND == "local_teacher" else OPENAI_JUDGE_MODEL
# Every API answer is cached here, keyed by the exact prompt, so re-runs are
# free and reproducible (temperature 0 is not bit-deterministic on the API side).
API_CACHE_PATH = "./llm_judge_outputs/openai_judge_cache.json"
AUDIT_DIR = "./llm_judge_outputs"

# Same strict-template regex as evaluate.parse_extraction_output.
TEMPLATE_RE = re.compile(r"^(.*?)\s*\(\s*Type:\s*(.*?)\s*\)", re.IGNORECASE)
VERDICT_RE = re.compile(r"\b(INCORRECT|CORRECT)\b")

JUDGE_QUESTION = (
    "You are a biomedical fact-checking assistant. Given a source entity, a "
    "relationship, and a PROPOSED target entity, decide whether the proposed "
    "target is a factually correct completion of this biomedical relationship "
    "-- even if it differs from a specific known answer, multiple targets can "
    "be correct for the same source and relationship.\n\n"
    "Source: {source_name} (Type: {source_type})\n"
    "Relationship: {relation}\n"
    "Proposed Target: {predicted_name} (Type: {predicted_type})\n\n"
    "Is the proposed target a factually correct completion of this "
    "relationship? Answer with exactly one word: CORRECT or INCORRECT."
)


def parse_verdict(response: str) -> str:
    """Returns 'CORRECT', 'INCORRECT', or 'UNPARSEABLE'. INCORRECT is listed
    first in the alternation and word boundaries prevent 'CORRECT' matching
    inside 'INCORRECT'."""
    m = VERDICT_RE.search(response.strip().upper())
    return m.group(1) if m else "UNPARSEABLE"


def load_local_judge():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_NAME, quantization_config=bnb_config, device_map="auto", trust_remote_code=False
    )
    model.eval()
    return model, tokenizer


def judge_with_local_teacher(model, tokenizer, source_name, source_type, relation, predicted_name, predicted_type):
    messages = [{"role": "user", "content": JUDGE_QUESTION.format(
        source_name=source_name, source_type=source_type, relation=relation,
        predicted_name=predicted_name, predicted_type=predicted_type,
    )}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
    return parse_verdict(response), response


_api_client = None
_api_cache = None


def _load_cache():
    global _api_cache
    if _api_cache is None:
        _api_cache = {}
        if os.path.exists(API_CACHE_PATH):
            with open(API_CACHE_PATH, "r", encoding="utf-8") as f:
                _api_cache = json.load(f)
    return _api_cache


def _save_cache():
    os.makedirs(os.path.dirname(API_CACHE_PATH), exist_ok=True)
    with open(API_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(_api_cache, f, indent=1)


def call_external_judge_api(source_name, source_type, relation, predicted_name, predicted_type):
    """Independent judge via the OpenAI API. Returns (verdict, raw_response)."""
    global _api_client
    import time
    from openai import OpenAI  # pip install openai

    if _api_client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY (or JUDGE_BACKEND=local_teacher).")
        _api_client = OpenAI()

    question = JUDGE_QUESTION.format(
        source_name=source_name, source_type=source_type, relation=relation,
        predicted_name=predicted_name, predicted_type=predicted_type,
    )
    cache = _load_cache()
    key = f"{OPENAI_JUDGE_MODEL}||{question}"
    if key in cache:
        raw = cache[key]
        return parse_verdict(raw), raw

    messages = [{"role": "user", "content": question}]
    for attempt in range(6):
        try:
            try:
                resp = _api_client.chat.completions.create(
                    model=OPENAI_JUDGE_MODEL, messages=messages, temperature=0, max_tokens=5,
                )
            except Exception as e:  # newer reasoning models reject temperature / max_tokens
                if "temperature" in str(e) or "max_tokens" in str(e):
                    resp = _api_client.chat.completions.create(
                        model=OPENAI_JUDGE_MODEL, messages=messages, max_completion_tokens=256,
                    )
                else:
                    raise
            raw = (resp.choices[0].message.content or "").strip()
            cache[key] = raw
            _save_cache()
            return parse_verdict(raw), raw
        except Exception as e:
            wait = 2 ** attempt
            print(f"    API error ({e.__class__.__name__}: {e}); retrying in {wait}s")
            time.sleep(wait)
    return "UNPARSEABLE", "API_FAILED"


def csv_path_for(model_repo: str, seed: int, output_dir: str) -> str:
    # Mirrors evaluate.Evaluator._report_paths.
    if model_repo == EVAL_CONFIG["CONTROL_MODEL"]:
        prefix = f"control-{model_repo.split('/')[-1]}-seed{seed}"
    else:
        prefix = f"{model_repo.replace('/', '_')}-seed{seed}"
    return os.path.join(output_dir, f"{prefix}_cross_domain_data.csv")


def find_eval_csvs(output_dir: str, models: list, seeds: list):
    found, missing = [], []
    for model_repo in models:
        for seed in seeds:
            path = csv_path_for(model_repo, seed, output_dir)
            (found if os.path.exists(path) else missing).append((model_repo, seed, path))
    for _, _, path in missing:
        print(f"⚠️ Missing: {path}")
    return found


def template_ok(raw_output) -> bool:
    if raw_output is None or (isinstance(raw_output, float) and np.isnan(raw_output)):
        return False
    m = TEMPLATE_RE.search(str(raw_output).strip())
    return bool(m and m.group(1).strip())  # template AND a non-empty entity name


def run_llm_judge_rescoring():
    output_dir = EVAL_CONFIG.get("output_dir", "./eval_outputs")
    os.makedirs(AUDIT_DIR, exist_ok=True)
    csv_locations = find_eval_csvs(output_dir, EVAL_CONFIG["ALL_MODELS"], EVAL_CONFIG["SEEDS_TO_EVAL"])

    if not csv_locations:
        print(f"⚠️ No evaluation CSVs found under {output_dir}. Run Stage 4 first.")
        return {}

    judge_model, judge_tokenizer = (load_local_judge() if JUDGE_BACKEND == "local_teacher" else (None, None))

    all_summaries = {}
    audit_rows = []

    for model_repo, seed, csv_path in csv_locations:
        print(f"\n{'=' * 60}\n{model_repo} seed {seed}\n{'=' * 60}")
        df = pd.read_csv(csv_path)
        n = len(df)
        fmt_ok = df["Raw_Output"].apply(template_ok)
        ext = df["Ext_Class"]

        cat1 = (ext == "fn") | (~fmt_ok)
        cat2 = (~cat1) & ext.isin(["hallucination", "topology_violation"])
        exact_tp = (~cat1) & (ext == "tp")
        to_judge = df[(~cat1) & (ext == "fp_fn")]

        verdicts = {"CORRECT": 0, "INCORRECT": 0, "UNPARSEABLE": 0}
        by_relation = {}

        for _, row in to_judge.iterrows():
            args = (
                row["Source_Node"], row.get("Source_Type", "Unknown"), row["Relation"],
                row["Ext_Predicted"], row.get("Ext_Predicted_Type", "Unknown"),
            )
            if JUDGE_BACKEND == "local_teacher":
                verdict, raw = judge_with_local_teacher(judge_model, judge_tokenizer, *args)
            else:
                verdict, raw = call_external_judge_api(*args)

            verdicts[verdict] += 1
            rel = str(row["Relation"])
            by_relation.setdefault(rel, {"CORRECT": 0, "INCORRECT": 0, "UNPARSEABLE": 0})[verdict] += 1
            audit_rows.append({
                "model": model_repo, "seed": seed,
                "Source_Node": row["Source_Node"], "Relation": row["Relation"],
                "Ext_Predicted": row["Ext_Predicted"], "Ext_Predicted_Type": row.get("Ext_Predicted_Type", ""),
                "Ground_Truth": row["Ground_Truth"],
                "verdict": verdict, "raw_judge_response": raw,
            })

        pct = lambda k: round(k / n * 100, 2) if n else 0.0
        summary = {
            "model": model_repo,
            "seed": seed,
            "n_samples": n,
            "format_compliance_pct": pct(int(fmt_ok.sum())),
            "1_syntactically_invalid_pct": pct(int(cat1.sum())),
            "2_syntactically_valid_ontologically_invalid_pct": pct(int(cat2.sum())),
            "3_ontologically_valid_factually_incorrect_pct": pct(verdicts["INCORRECT"]),
            "4_factually_correct_pct": pct(int(exact_tp.sum()) + verdicts["CORRECT"]),
            "judge_unparseable_pct": pct(verdicts["UNPARSEABLE"]),
            "n_judged": int(len(to_judge)),
            "judge_verdict_counts": verdicts,
            "judge_verdicts_by_relation": by_relation,
            "judge_backend": JUDGE_BACKEND,
            "judge_model": JUDGE_MODEL_NAME,
        }
        all_summaries[f"{model_repo}-seed{seed}"] = summary
        print(json.dumps(summary, indent=2))

    # --- audit + aggregate outputs ---
    # Backend-specific file names, so API and local-teacher audits never overwrite each other.
    backbone = CONFIG["student_name"].split("/")[-1] + ("" if JUDGE_BACKEND == "local_teacher" else f"_{OPENAI_JUDGE_MODEL}")
    audit_path = os.path.join(AUDIT_DIR, f"judge_audit_{backbone}.csv")
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)

    summary_df = pd.DataFrame([
        {k: v for k, v in s.items() if not isinstance(v, dict)} for s in all_summaries.values()
    ])
    metric_cols = [c for c in summary_df.columns if c.endswith("_pct")]
    agg = summary_df.groupby("model")[metric_cols].agg(["mean", "std"]).round(2)

    summary_df.to_csv(os.path.join(AUDIT_DIR, f"judge_per_seed_{backbone}.csv"), index=False)
    agg.to_csv(os.path.join(AUDIT_DIR, f"judge_mean_sd_{backbone}.csv"))
    with open(os.path.join(AUDIT_DIR, f"judge_summaries_{backbone}.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'=' * 60}\nMEAN / SD ACROSS SEEDS ({backbone})\n{'=' * 60}")
    print(agg.to_string())
    print(f"\n📝 Every judge call (with raw response) saved to {audit_path} -- inspect it before reporting.")

    return all_summaries


if __name__ == "__main__":
    run_llm_judge_rescoring()
