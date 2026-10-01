"""
evaluate_expanded_noise_robustness.py
=======================================
Review comment #9: the existing robustness test (evaluate.py's
EVAL_MODE="robustness") only corrupts entity strings via randomized
character swaps. This script adds two more realistic biomedical-data noise
types -- corrupted entity-type labels and corrupted relation predicates --
and runs each as its OWN separate evaluation pass, so results are reported
per noise type rather than combined into a single number, per the
reviewer's explicit instruction.

DESIGN: subclasses evaluate.Evaluator the same way
evaluate_constrained_decoding_baseline.py does -- copies
_evaluate_dataset_batched verbatim and changes only the noise-injection
step, so results stay comparable to the existing character-swap robustness
numbers (Figure 2) computed by the same underlying pipeline.

Noise types implemented:
    - "char_swap"              : existing method, included here as the
                                  baseline for comparison (same as
                                  EVAL_MODE="robustness" already produces).
    - "entity_type_corruption" : the source node's stated Type is replaced
                                  with a DIFFERENT, incorrect type label
                                  from the ontology, while the entity name
                                  itself stays correct. Tests whether the
                                  model is misled by an incorrect type cue.
    - "predicate_corruption"   : the relation string is replaced with a
                                  different, real relation from elsewhere in
                                  the graph, unrelated to the true
                                  source-target pair.
    - "invalid_source_target"  : the source entity is swapped for a
                                  different entity whose type is
                                  incompatible with the stated relation
                                  (best-effort -- see caveat in the
                                  function docstring below).

CHANGES:
  - Relation pool for predicate_corruption is now sorted() and built once
    in prepare_data() from the union of ALL seeds' eval sets. Previously it
    was list(set(...)) -- order depended on PYTHONHASHSEED, so corruption
    was not reproducible across processes -- and was cached from whichever
    seed's dataset happened to run first. (Changes predicate_corruption
    outputs; rerun all noise types.)
  - Corrupted inputs are logged per row (Noisy_Source_Node,
    Noisy_Source_Type, Noisy_Relation). No metric is affected.
  - prepare_data() runs once and is shared across the four noise types.
  - Duplicate inject_character_swaps override removed (parent accepts rng=).
  - Base-model reuse and EVAL_SHARD sharding are inherited from evaluate.py.

USAGE:
    python evaluate_expanded_noise_robustness.py
Runs all four noise types (including char_swap for reference) across all
models in EVAL_CONFIG['ALL_MODELS'] and EVAL_CONFIG['SEEDS_TO_EVAL'].
"""

import copy
import random
import time

import torch

from config import EVAL_CONFIG
from evaluate import Evaluator


class ExpandedNoiseEvaluator(Evaluator):
    """
    Identical to Evaluator._evaluate_dataset_batched, except the prompt
    construction step applies self.noise_type-specific corruption instead
    of the single hardcoded character-swap. self.noise_type and
    self.noise_level are set via __init__ before prepare_data()/
    run_evaluation() are called.

    SYNC WARNING: same caveat as evaluate_constrained_decoding_baseline.py --
    this is a copy of _evaluate_dataset_batched, not a wrapper, so re-sync
    manually if evaluate.py's version changes.
    """

    def __init__(self, eval_config: dict, noise_type: str):
        super().__init__(eval_config)
        self.noise_type = noise_type
        self._all_relations = None

    def prepare_data(self):
        super().prepare_data()
        self._build_relation_pool()

    def _build_relation_pool(self):
        # Deterministic, seed-order-independent, shard-independent pool.
        self._all_relations = sorted(
            {row["relation"] for ds in self.datasets_by_seed.values() for row in ds}
        )

    def share_prepared_data_from(self, other: "ExpandedNoiseEvaluator"):
        self.datasets_by_seed = other.datasets_by_seed
        self.ontology_dict = other.ontology_dict
        self.valid_nodes = other.valid_nodes
        self.valid_node_types = other.valid_node_types
        self._all_relations = other._all_relations

    def _get_all_types(self):
        # ontology_dict keys are in deterministic first-appearance order.
        return list(self.ontology_dict.keys())

    def corrupt_entity_type(self, true_type: str, noise_rng) -> str:
        """Returns a DIFFERENT type label than true_type, from the ontology."""
        wrong_types = [t for t in self._get_all_types() if t != true_type]
        return noise_rng.choice(wrong_types) if wrong_types else true_type

    def corrupt_predicate(self, true_relation: str, noise_rng) -> str:
        """Returns a DIFFERENT relation string than true_relation."""
        wrong_relations = [r for r in self._all_relations if r != true_relation]
        return noise_rng.choice(wrong_relations) if wrong_relations else true_relation

    def corrupt_source_target(self, true_source_name: str, true_source_type: str, noise_rng):
        """
        Best-effort implementation, flagged explicitly rather than silently
        approximated: swaps the source entity for a different entity of a
        DIFFERENT type, producing a source-relation pairing that would not
        occur in the real graph. This is an approximation of "invalid
        source-target combination" -- constructing a combination guaranteed
        to violate the relation's actual type signature would require
        relation-specific type-signature metadata this pipeline does not
        currently extract from PharmGKB/PrimeKG. Treat this noise type's
        results as indicative, not as precisely as validated as the other
        two -- flag this in the paper's Discussion if using this result.
        """
        wrong_types = [t for t in self._get_all_types() if t != true_source_type]
        if not wrong_types:
            return true_source_name
        wrong_type = noise_rng.choice(wrong_types)
        candidates = self.ontology_dict.get(wrong_type, [])
        return noise_rng.choice(candidates) if candidates else true_source_name

    def _evaluate_dataset_batched(self, dataset, model, tokenizer, seed):
        if self._all_relations is None:
            self._build_relation_pool()

        batch_size = self.config.get("batch_size", 16)
        csv_results = []
        ext_metrics = {"tp": 0, "fp": 0, "fn": 0, "hallucinations": 0, "topology_violations": 0, "fp_fn_count": 0}
        rank_metrics = {"correct": 0, "factual_errors": 0, "topology_errors": 0, "hallucination_errors": 0}
        total_time = 0.0

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            start_time = time.time()
            prompts, ground_truths, noisy_inputs = [], [], []

            # Dedicated RNG for noise generation, keyed to (seed, batch index).
            noise_random = random.Random(seed * 1_000_003 + i + 500_009)

            for example in batch:
                source_node = example["x_name"]
                source_type = example["x_type"]
                relation = example["relation"]

                if self.noise_type == "char_swap":
                    source_node = self.inject_character_swaps(
                        source_node, self.config["noise_level"], rng=noise_random
                    )
                elif self.noise_type == "entity_type_corruption":
                    source_type = self.corrupt_entity_type(source_type, noise_random)
                elif self.noise_type == "predicate_corruption":
                    relation = self.corrupt_predicate(relation, noise_random)
                elif self.noise_type == "invalid_source_target":
                    source_node = self.corrupt_source_target(source_node, source_type, noise_random)

                prompt = (
                    "Task: Analyze the biomedical relationship and output ONLY the Target Node and its Type. "
                    "Do not write explanations, exercises, or extra text.\n\n"
                    f"Source Node: {source_node} (Type: {source_type})\n"
                    f"Relationship: {relation}\n"
                    f"Target Node: "
                )
                prompts.append(prompt)
                ground_truths.append((example["y_name"], example["y_type"]))
                noisy_inputs.append((source_node, source_type, relation))

            tokenizer.padding_side = "left"
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config["max_new_tokens"],
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
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

            # --- PASS B: identical to evaluate.py ---
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
                    "Noisy_Source_Node": noisy_inputs[b_idx][0],
                    "Noisy_Source_Type": noisy_inputs[b_idx][1],
                    "Noisy_Relation": noisy_inputs[b_idx][2],
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


def run_expanded_noise_robustness():
    noise_types = ["char_swap", "entity_type_corruption", "predicate_corruption", "invalid_source_target"]
    all_results = {}
    prepared = None

    for noise_type in noise_types:
        print(f"\n{'#' * 60}\n# NOISE TYPE: {noise_type}\n{'#' * 60}")
        config_copy = copy.deepcopy(EVAL_CONFIG)
        config_copy["EVAL_MODE"] = "robustness"  # reuses PharmGKB cross-domain data loading
        config_copy["output_dir"] = EVAL_CONFIG.get("output_dir", "./eval_outputs") + f"_noise_{noise_type}"

        evaluator = ExpandedNoiseEvaluator(config_copy, noise_type=noise_type)
        if prepared is None:
            evaluator.prepare_data()
            prepared = evaluator
        else:
            evaluator.share_prepared_data_from(prepared)

        generated_files = evaluator.run_evaluation()
        all_results[noise_type] = generated_files

    print("\n" + "=" * 70)
    print("EXPANDED NOISE ROBUSTNESS -- results reported SEPARATELY per noise type")
    print("(review comment #9 explicitly asks these not be combined into one measure)")
    print("=" * 70)
    for noise_type, files in all_results.items():
        print(f"\n{noise_type}:")
        for f in files:
            print(f"  - {f}")

    print("\nCaveat: 'invalid_source_target' is a best-effort approximation --")
    print("see ExpandedNoiseEvaluator.corrupt_source_target()'s docstring before")
    print("reporting this noise type's results with the same confidence as the other two.")

    return all_results


if __name__ == "__main__":
    run_expanded_noise_robustness()
