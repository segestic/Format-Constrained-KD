"""
evaluate_constrained_decoding_baseline.py
==========================================

USAGE:
    python evaluate_constrained_decoding_baseline.py

Runs both comparisons above, across all seeds in EVAL_CONFIG['SEEDS_TO_EVAL'].
Output files land in EVAL_CONFIG['output_dir'] + '_constrained_decoding',
tagged by which checkpoint was used, so they sit alongside your existing
reports without colliding.

CHANGES (output-preserving unless noted):
  - Logits processor: one GPU->CPU transfer per generation step instead of
    one per batch row. Constraint logic is unchanged.
  - Robustness-mode character swaps use the same per-batch noise RNG as
    evaluate.py (changes EVAL_MODE="robustness" outputs only).
  - prepare_data() runs once and its results are shared by both tags.
  - Base-model reuse and EVAL_SHARD sharding are inherited from evaluate.py.

INTEGRITY CHECK: the logits processor only affects Pass A, so Ranking
Accuracy here must exactly equal the unconstrained Cross Domain ranking for
the same arm/seed at the same batch_size. Any difference means something is
out of sync with evaluate.py.
"""

import copy
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from config import CONFIG, EVAL_CONFIG
from evaluate import Evaluator


# ==============================================================================
# GRAMMAR-CONSTRAINED DECODER
# ==============================================================================
# Forces the literal scaffold " (Type: " ... ")" around free-form spans for
# the entity name and type, so the model CANNOT produce a malformed triple.
# It can still name the wrong entity or the wrong type -- this constrains
# SYNTAX only, exactly the property Section 5.5 argues constrained decoding
# does and does not give you, as distinct from FC-KD's training-time effect.
class TemplateConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, prompt_len: int, max_name_tokens: int = 12, max_type_tokens: int = 6):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.max_name_tokens = max_name_tokens
        self.max_type_tokens = max_type_tokens
        self.literal_open = tokenizer(" (Type: ", add_special_tokens=False).input_ids
        self.literal_close = tokenizer(")", add_special_tokens=False).input_ids
        self.eos_id = tokenizer.eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # SPEED: single GPU->CPU transfer for the whole batch per step
        # (previously one .tolist() per row). Logic below is unchanged.
        generated_all = input_ids[:, self.prompt_len:].tolist()
        open_tok = self.literal_open[0]
        close_tok = self.literal_close[0]

        for b, generated in enumerate(generated_all):
            n_gen = len(generated)

            if n_gen >= self.max_name_tokens and open_tok not in generated:
                scores[b, :] = -float("inf")
                scores[b, open_tok] = 0.0
                continue

            if open_tok in generated:
                open_idx = generated.index(open_tok)
                tokens_after_open = n_gen - (open_idx + len(self.literal_open))
                if tokens_after_open >= self.max_type_tokens and close_tok not in generated[open_idx:]:
                    scores[b, :] = -float("inf")
                    scores[b, close_tok] = 0.0
                    continue
                if close_tok in generated[open_idx:]:
                    scores[b, :] = -float("inf")
                    scores[b, self.eos_id] = 0.0
        return scores


# ==============================================================================
# EVALUATOR SUBCLASS -- reuses the real pipeline, overrides only Pass A's generate()
# ==============================================================================
class ConstrainedDecodingEvaluator(Evaluator):
    """
    Identical to Evaluator._evaluate_dataset_batched, with exactly one change:
    Pass A's model.generate() call is given a TemplateConstrainedLogitsProcessor.
    Everything else (Pass B ranking, metric bucketing, report text) is copied
    verbatim from evaluate.py so results stay comparable to your other arms.

    SYNC WARNING: if evaluate.py's _evaluate_dataset_batched changes in the
    future, this override needs to be re-synced with it -- it is a copy, not
    a wrapper, because the generate() call is embedded inside the larger
    batched loop with no clean seam to hook into otherwise.
    """

    def _evaluate_dataset_batched(self, dataset, model, tokenizer, seed):
        import random
        import time

        batch_size = self.config.get("batch_size", 16)
        csv_results = []
        ext_metrics = {"tp": 0, "fp": 0, "fn": 0, "hallucinations": 0, "topology_violations": 0, "fp_fn_count": 0}
        rank_metrics = {"correct": 0, "factual_errors": 0, "topology_errors": 0, "hallucination_errors": 0}
        total_time = 0.0

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            start_time = time.time()

            # Synced with evaluate.py: robustness noise RNG keyed to (seed, i).
            noise_random = random.Random(seed * 1_000_003 + i + 500_009)

            prompts, ground_truths = [], []
            for example in batch:
                source_node = example["x_name"]
                if self.eval_mode == "robustness":
                    source_node = self.inject_character_swaps(
                        source_node, self.config["noise_level"], rng=noise_random
                    )
                prompt = (
                    "Task: Analyze the biomedical relationship and output ONLY the Target Node and its Type. "
                    "Do not write explanations, exercises, or extra text.\n\n"
                    f"Source Node: {source_node} (Type: {example['x_type']})\n"
                    f"Relationship: {example['relation']}\n"
                    f"Target Node: "
                )
                prompts.append(prompt)
                ground_truths.append((example["y_name"], example["y_type"]))

            # --- PASS A: batched zero-shot generative extraction ---
            # ONLY CHANGE FROM evaluate.py: logits_processor added below.
            tokenizer.padding_side = "left"
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            constraint = TemplateConstrainedLogitsProcessor(tokenizer, prompt_len)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config["max_new_tokens"],
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    logits_processor=LogitsProcessorList([constraint]),
                )
            gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
            gen_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

            ext_data = []
            for b_idx, gen_text in enumerate(gen_texts):
                gt_name, gt_type = ground_truths[b_idx]
                ext_name, ext_type, raw_gen = self.parse_extraction_output(gen_text, "")
                ext_name_norm = ext_name.strip().lower() if ext_name else None
                gt_name_norm = gt_name.strip().lower()
                ext_is_correct = False
                if not ext_name:
                    ext_class = "fn"
                elif ext_name_norm not in self.valid_nodes:
                    ext_class = "hallucination"
                elif (ext_name_norm in self.valid_node_types
                      and self.valid_node_types[ext_name_norm] != ext_type
                      and ext_type != "Unknown"):
                    ext_class = "topology_violation"
                elif ext_name_norm == gt_name_norm:
                    ext_class = "tp"
                    ext_is_correct = True
                else:
                    ext_class = "fp_fn"
                ext_data.append({
                    "Ext_Predicted": ext_name if ext_name else "FAILED",
                    "Ext_Predicted_Type": ext_type if ext_name else "Unknown",
                    "Ext_Class": ext_class,
                    "Ext_Is_Correct": int(ext_is_correct),
                    "Raw_Output": raw_gen,
                })

            # --- PASS B: batched adversarial generative ranking ---
            # FIX: explicit per-batch reseed, matching evaluate.py
            batch_random = random.Random(seed * 1_000_003 + i)

            all_rank_prompts, all_rank_targets, all_rank_mappings = [], [], []
            for b_idx, example in enumerate(batch):
                gt_name, gt_type = ground_truths[b_idx]
                same_type_nodes = [n for n in self.ontology_dict[gt_type] if n != gt_name]
                factual_trap = batch_random.choice(same_type_nodes) if same_type_nodes else "Generic_Biomolecule"
                wrong_types = [t for t in self.ontology_dict.keys() if t != gt_type]
                if wrong_types:
                    random_wrong_type = batch_random.choice(wrong_types)
                    topology_trap = batch_random.choice(self.ontology_dict[random_wrong_type])
                    topology_trap_type = random_wrong_type
                else:
                    topology_trap = f"Topology_Violation_Sim_{batch_random.randint(1000, 9999)}"
                    topology_trap_type = "Simulated_Type"
                hallucination_trap = f"BioNode_{batch_random.randint(10000, 99999)}"

                options = [
                    (f"{gt_name} (Type: {gt_type})", "Correct"),
                    (f"{factual_trap} (Type: {gt_type})", "Factual_Error"),
                    (f"{topology_trap} (Type: {topology_trap_type})", "Topology_Error"),
                    (f"{hallucination_trap} (Type: Unknown)", "Hallucination_Error"),
                ]
                batch_random.shuffle(options)
                for target_text, error_type in options:
                    all_rank_prompts.append(prompts[b_idx])
                    all_rank_targets.append(target_text)
                    all_rank_mappings.append((b_idx, target_text, error_type))

            losses = self.calculate_batched_target_losses(all_rank_prompts, all_rank_targets, model, tokenizer)
            batch_loss_results = {b: [] for b in range(len(batch))}
            for loss, (b_idx, target_text, error_type) in zip(losses, all_rank_mappings):
                batch_loss_results[b_idx].append((loss, target_text, error_type))

            for b_idx in range(len(batch)):
                _, best_target, rank_error_type = min(batch_loss_results[b_idx], key=lambda x: x[0])
                row_data = {
                    "Source_Node": batch[b_idx]["x_name"],
                    "Source_Type": batch[b_idx]["x_type"],
                    "Relation": batch[b_idx]["relation"],
                    "Ground_Truth": ground_truths[b_idx][0],
                    "Ext_Predicted": ext_data[b_idx]["Ext_Predicted"],
                    "Ext_Predicted_Type": ext_data[b_idx]["Ext_Predicted_Type"],
                    "Ext_Class": ext_data[b_idx]["Ext_Class"],
                    "Ext_Is_Correct": ext_data[b_idx]["Ext_Is_Correct"],
                    "Raw_Output": ext_data[b_idx]["Raw_Output"],
                    "Rank_Predicted": best_target.split(" (Type:")[0],
                    "Rank_Error_Type": rank_error_type,
                    "Rank_Is_Correct": int(rank_error_type == "Correct"),
                }
                csv_results.append(row_data)

                ext_class = row_data["Ext_Class"]
                if ext_class == "fn":
                    ext_metrics["fn"] += 1
                elif ext_class == "hallucination":
                    ext_metrics["hallucinations"] += 1
                    ext_metrics["fp"] += 1
                elif ext_class == "topology_violation":
                    ext_metrics["topology_violations"] += 1
                    ext_metrics["fp"] += 1
                elif ext_class == "tp":
                    ext_metrics["tp"] += 1
                else:
                    ext_metrics["fp"] += 1
                    ext_metrics["fn"] += 1
                    ext_metrics["fp_fn_count"] += 1

                if rank_error_type == "Correct":
                    rank_metrics["correct"] += 1
                elif rank_error_type == "Factual_Error":
                    rank_metrics["factual_errors"] += 1
                elif rank_error_type == "Topology_Error":
                    rank_metrics["topology_errors"] += 1
                elif rank_error_type == "Hallucination_Error":
                    rank_metrics["hallucination_errors"] += 1

            total_time += (time.time() - start_time)
            print(f"    Processed {min(i + batch_size, len(dataset))}/{len(dataset)} iterations...")

        return csv_results, ext_metrics, rank_metrics, total_time


def run_constrained_decoding_on(checkpoint_output_dir_key: str, tag: str, prepared_from=None):
    """
    checkpoint_output_dir_key: the CONFIG key naming the checkpoint's output
        dir, e.g. 's1_output_dir' (SFT) or 's3_output_dir' (FC-KD).
    tag: short label used in the output directory name to distinguish this
        run's reports from the other comparison's.
    prepared_from: optional evaluator whose prepare_data() results are
        reused (same EVAL_MODE and seeds), so data prep runs only once.
    Returns (generated_files, evaluator).
    """
    baseline_config = copy.deepcopy(EVAL_CONFIG)
    repo = f"{CONFIG['hub_username']}/{CONFIG[checkpoint_output_dir_key]}"
    baseline_config["CUSTOM_MODELS"] = [repo]
    baseline_config["ALL_MODELS"] = [repo]  # exclude CONTROL_MODEL -- base model has no adapter to meaningfully constrain
    baseline_config["output_dir"] = EVAL_CONFIG.get("output_dir", "./eval_outputs") + f"_constrained_decoding_{tag}"

    print(f"\n{'#' * 60}\n# CONSTRAINED DECODING ON: {repo}  (tag: {tag})\n{'#' * 60}")
    evaluator = ConstrainedDecodingEvaluator(baseline_config)

    if prepared_from is not None:
        evaluator.datasets_by_seed = prepared_from.datasets_by_seed
        evaluator.ontology_dict = prepared_from.ontology_dict
        evaluator.valid_nodes = prepared_from.valid_nodes
        evaluator.valid_node_types = prepared_from.valid_node_types
    else:
        evaluator.prepare_data()

    return evaluator.run_evaluation(), evaluator


def run_constrained_decoding_experiments():
    # Comparison 1 (Section 5.5's baseline): constrained decoding on the SFT
    # checkpoint alone -- the same branch point FC-KD starts from, so this
    # answers "is inference-time constraint alone enough to close the gap?"
    sft_results, sft_evaluator = run_constrained_decoding_on("s1_output_dir", tag="sft_baseline")

    # Comparison 2 (review comment #6's hybrid): constrained decoding ON TOP
    # OF an already-FC-KD-trained checkpoint -- answers "do the two
    # techniques compound, as Section 5.5 speculates but does not test?"
    fckd_results, _ = run_constrained_decoding_on("s3_output_dir", tag="fckd_hybrid", prepared_from=sft_evaluator)

    print("\n" + "=" * 70)
    print("SUMMARY: Constrained Decoding Baseline")
    print("Compare these results directly against your verified cross-domain")
    print("FC-KD and SFT numbers in eval_outputs/ to see if inference-time")
    print("constraints alone can close the gap.")
    print("=" * 70)
    print("SFT + constrained decoding reports:", sft_results)
    print("FC-KD + constrained decoding reports:", fckd_results)

    return {"sft_baseline": sft_results, "fckd_hybrid": fckd_results}


if __name__ == "__main__":
    run_constrained_decoding_experiments()
