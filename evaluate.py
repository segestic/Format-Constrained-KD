"""
evaluate.py
===========
Dual-modality evaluation: (1) zero-shot generative extraction (Strict Exact
Match / Ontological Validity) and (2) adversarial generative ranking.

Cross-Domain Protocol
----------------------
EVAL_MODE="cross_domain" (the default, and the setting used for every number
reported in the manuscript) evaluates PrimeKG-trained models on PharmGKB, a
graph never seen during training. EVAL_MODE="standard" evaluates against a
held-out PrimeKG split instead, and exists only as an internal sanity check
against in-domain performance -- it is NOT used for any reported result.

Bug fix vs. earlier draft
--------------------------
Ground-truth strings are now normalized with the same `.strip().lower()`
as the parsed model prediction. The previous version stripped only the
prediction, so trailing/leading whitespace in source TSV fields could
cause false exact-match failures independent of true model behavior.
"""

import gc
import os
import random
import re
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class Evaluator:
    def __init__(self, eval_config: dict):
        self.config = eval_config
        self.seeds = eval_config["SEEDS_TO_EVAL"]
        self.eval_mode = eval_config["EVAL_MODE"]
        self.all_models = eval_config["ALL_MODELS"]
        self.output_dir = eval_config.get("output_dir", "./eval_outputs")

        self.datasets_by_seed = {}
        self.ontology_dict = {}
        self.valid_nodes = set()
        self.valid_node_types = {}

        os.makedirs(self.output_dir, exist_ok=True)

    # --------------------------------------------------------------------
    # Data preparation
    # --------------------------------------------------------------------
    def prepare_data(self):
        print(f"\n[Mode: {self.eval_mode.upper()}] Loading evaluation data matrix...")

        if self.eval_mode in ["cross_domain", "robustness"]:
            df = self._load_pharmgkb()
        else:
            df = self._load_primekg()

        self.valid_nodes = set(
            df["x_name"].str.strip().str.lower().tolist() + df["y_name"].str.strip().str.lower().tolist()
        )
        self.valid_node_types = dict(zip(df["x_name"].str.strip().str.lower(), df["x_type"]))
        self.valid_node_types.update(dict(zip(df["y_name"].str.strip().str.lower(), df["y_type"])))

        for _, row in df.iterrows():
            self.ontology_dict.setdefault(row["y_type"], set()).add(row["y_name"])
        self.ontology_dict = {k: list(v) for k, v in self.ontology_dict.items()}

        for seed in self.seeds:
            random.seed(seed)
            np.random.seed(seed)
            df_stratified = (
                df.groupby("relation", group_keys=False)
                .apply(lambda x: x.sample(min(len(x), 10000), random_state=seed))
                .copy()
                .astype(str)
            )
            split = Dataset.from_pandas(df_stratified).train_test_split(test_size=0.1, seed=seed)
            df_test = split["test"].to_pandas()

            n_samples = min(self.config["num_eval_samples"], len(df_test))
            df_eval = df_test.sample(n=n_samples, random_state=seed).copy()
            self.datasets_by_seed[seed] = df_eval.to_dict("records")
            print(f"  - Seed {seed}: isolated {len(self.datasets_by_seed[seed])} test samples.")

    def _load_pharmgkb(self) -> pd.DataFrame:
        os.makedirs("./data", exist_ok=True)
        zip_path = "./data/relationships.zip"
        tsv_path = "./data/relationships.tsv"

        if not os.path.exists(zip_path):
            print("📥 Downloading PharmGKB relationships.zip...")
            url = "https://api.pharmgkb.org/v1/download/file/data/relationships.zip"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
                out_file.write(response.read())

        if not os.path.exists(tsv_path):
            print("📦 Unzipping PharmGKB relationships...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall("./data")

        df = pd.read_csv(tsv_path, sep="\t")
        df = (
            df.rename(
                columns={
                    "Entity1_name": "x_name",
                    "Entity1_type": "x_type",
                    "Association": "relation",
                    "Entity2_name": "y_name",
                    "Entity2_type": "y_type",
                }
            )
            .dropna(subset=["x_name", "y_name"])
            .copy()
            .astype(str)
        )
        print(f"✅ Ingested {len(df)} PharmGKB cross-domain evaluation edges.")
        return df

    def _load_primekg(self) -> pd.DataFrame:
        from tdc.resource import PrimeKG

        data = PrimeKG(path="./data")
        df = data.get_data().dropna(subset=["x_name", "y_name"]).copy().astype(str)
        print(f"✅ Ingested {len(df)} PrimeKG in-domain sanity-check edges.")
        return df

    # --------------------------------------------------------------------
    # Utility functions
    # --------------------------------------------------------------------
    @staticmethod
    def inject_character_swaps(text: str, noise_level: float) -> str:
        if len(text) < 2 or noise_level <= 0:
            return text
        text_list = list(text)
        num_swaps = max(1, int(len(text) * noise_level))
        for _ in range(num_swaps):
            idx = random.randint(0, len(text_list) - 2)
            text_list[idx], text_list[idx + 1] = text_list[idx + 1], text_list[idx]
        return "".join(text_list)

    @staticmethod
    def parse_extraction_output(full_text: str, prompt_text: str):
        if full_text.startswith(prompt_text):
            generated_part = full_text[len(prompt_text):].strip()
        else:
            generated_part = full_text.split("Target Node:")[-1].strip()

        first_line = generated_part.split("\n")[0].strip()
        match_strict = re.search(r"^(.*?)\s*\(\s*Type:\s*(.*?)\s*\)", first_line, re.IGNORECASE)
        if match_strict:
            return match_strict.group(1).strip(), match_strict.group(2).strip(), first_line

        raw_node = re.sub(r"[,.]+$", "", first_line).strip()
        return raw_node, "Unknown", first_line

    @staticmethod
    def calculate_target_loss(context_text, target_text, model, tokenizer):
        full_text = context_text + target_text
        inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
        context_ids = tokenizer(context_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

        labels = inputs["input_ids"].clone()
        mask_length = min(len(context_ids), labels.shape[1] - 1)
        labels[0, :mask_length] = -100

        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], labels=labels)
            return outputs.loss.item()

    @staticmethod
    def calculate_batched_target_losses(prompts, targets, model, tokenizer):
        """Batched equivalent of calculate_target_loss: computes per-sequence
        loss for a whole batch of (prompt, target) pairs in one forward pass."""
        tokenizer.padding_side = "right"  # required for correct causal masking here
        full_texts = [p + t for p, t in zip(prompts, targets)]
        inputs = tokenizer(full_texts, return_tensors="pt", padding=True).to(model.device)
        labels = inputs["input_ids"].clone()

        for i, p in enumerate(prompts):
            p_ids = tokenizer(p, add_special_tokens=False)["input_ids"]
            p_len = len(p_ids)
            labels[i, :p_len] = -100
            labels[i, inputs["input_ids"][i] == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())
            seq_loss = loss.sum(dim=1) / (shift_labels != -100).sum(dim=1)

        return seq_loss.tolist()

    # --------------------------------------------------------------------
    # Main evaluation loop
    # --------------------------------------------------------------------
    def run_evaluation(self):
        all_generated_files = []

        for base_model_path in self.all_models:
            print(f"\n{'=' * 60}\n🚀 MODEL FAMILY: {base_model_path}\n{'=' * 60}")

            for seed in self.seeds:
                model, base_model, tokenizer, model_name = self._load_model(base_model_path, seed)
                model.eval()

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                dataset = self.datasets_by_seed[seed]
                torch.cuda.reset_peak_memory_stats()

                # Call the newly batched evaluation method
                csv_results, ext_metrics, rank_metrics, total_time = self._evaluate_dataset_batched(
                    dataset, model, tokenizer
                )

                txt_filename, csv_filename = self._write_reports(
                    model_name, base_model_path, seed, csv_results, ext_metrics, rank_metrics, total_time
                )
                all_generated_files.extend([txt_filename, csv_filename])
                print(f"  ✅ Saved: {txt_filename} & {csv_filename}")

                del model, base_model, tokenizer
                gc.collect()
                torch.cuda.empty_cache()

        print("\n" + "=" * 60)
        print("🎉 EVALUATION COMPLETE. Generated files:")
        for f in all_generated_files:
            print(f"  - {f}")
        return all_generated_files

    def _load_model(self, base_model_path, seed):
        if self.config.get("use_4bit", True):
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            load_kwargs = {"quantization_config": bnb_config, "device_map": "auto"}
        else:
            load_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}

        if base_model_path == self.config["CONTROL_MODEL"]:
            model_name = self.config["CONTROL_MODEL"]
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            model = base_model
        else:
            model_name = f"{base_model_path}-seed{seed}"
            tokenizer = AutoTokenizer.from_pretrained(self.config["base_model_name"])
            base_model = AutoModelForCausalLM.from_pretrained(self.config["base_model_name"], **load_kwargs)
            model = PeftModel.from_pretrained(base_model, model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, base_model, tokenizer, model_name

    def _evaluate_dataset_batched(self, dataset, model, tokenizer):
        batch_size = self.config.get("batch_size", 16)
        csv_results = []
        ext_metrics = {"tp": 0, "fp": 0, "fn": 0, "hallucinations": 0, "topology_violations": 0}
        rank_metrics = {"correct": 0, "factual_errors": 0, "topology_errors": 0, "hallucination_errors": 0}
        total_time = 0.0

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            start_time = time.time()

            prompts, ground_truths = [], []
            for example in batch:
                source_node = example["x_name"]
                if self.eval_mode == "robustness":
                    source_node = self.inject_character_swaps(source_node, self.config["noise_level"])
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
                    "Ext_Class": ext_class,
                    "Ext_Is_Correct": int(ext_is_correct),
                    "Raw_Output": raw_gen,
                })

            # --- PASS B: batched adversarial generative ranking ---
            all_rank_prompts, all_rank_targets, all_rank_mappings = [], [], []
            for b_idx, example in enumerate(batch):
                gt_name, gt_type = ground_truths[b_idx]
                same_type_nodes = [n for n in self.ontology_dict[gt_type] if n != gt_name]
                factual_trap = random.choice(same_type_nodes) if same_type_nodes else "Generic_Biomolecule"
                wrong_types = [t for t in self.ontology_dict.keys() if t != gt_type]
                if wrong_types:
                    random_wrong_type = random.choice(wrong_types)
                    topology_trap = random.choice(self.ontology_dict[random_wrong_type])
                    topology_trap_type = random_wrong_type
                else:
                    topology_trap = f"Topology_Violation_Sim_{random.randint(1000, 9999)}"
                    topology_trap_type = "Simulated_Type"
                hallucination_trap = f"BioNode_{random.randint(10000, 99999)}"

                options = [
                    (f"{gt_name} (Type: {gt_type})", "Correct"),
                    (f"{factual_trap} (Type: {gt_type})", "Factual_Error"),
                    (f"{topology_trap} (Type: {topology_trap_type})", "Topology_Error"),
                    (f"{hallucination_trap} (Type: Unknown)", "Hallucination_Error"),
                ]
                random.shuffle(options)
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
                    "Relation": batch[b_idx]["relation"],
                    "Ground_Truth": ground_truths[b_idx][0],
                    "Ext_Predicted": ext_data[b_idx]["Ext_Predicted"],
                    "Ext_Class": ext_data[b_idx]["Ext_Class"],
                    "Ext_Is_Correct": ext_data[b_idx]["Ext_Is_Correct"],
                    "Raw_Output": ext_data[b_idx]["Raw_Output"],
                    "Rank_Predicted": best_target.split(" (Type:")[0],
                    "Rank_Error_Type": rank_error_type,
                    "Rank_Is_Correct": int(rank_error_type == "Correct"),
                }
                csv_results.append(row_data)

                # --- Inlined Metrics Update ---
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
                else:  # fp_fn
                    ext_metrics["fp"] += 1
                    ext_metrics["fn"] += 1

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

    def _write_reports(self, model_name, base_model_path, seed, csv_results, ext_metrics, rank_metrics, total_time):
        N = len(csv_results)
        max_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        ext_p = ext_metrics["tp"] / (ext_metrics["tp"] + ext_metrics["fp"]) if (ext_metrics["tp"] + ext_metrics["fp"]) > 0 else 0
        ext_r = ext_metrics["tp"] / (ext_metrics["tp"] + ext_metrics["fn"]) if (ext_metrics["tp"] + ext_metrics["fn"]) > 0 else 0
        ext_f1 = 2 * (ext_p * ext_r) / (ext_p + ext_r) if (ext_p + ext_r) > 0 else 0
        global_rank_acc = (rank_metrics["correct"] / N) * 100 if N else 0

        results_df = pd.DataFrame(csv_results)

        class_metrics = []
        for relation, group in results_df.groupby("Relation"):
            count = len(group)
            class_metrics.append(
                {
                    "Relation": relation,
                    "N_Samples": count,
                    "Extraction_Acc_%": round((group["Ext_Is_Correct"].sum() / count) * 100, 2),
                    "Ranking_Acc_%": round((group["Rank_Is_Correct"].sum() / count) * 100, 2),
                }
            )
        class_df = pd.DataFrame(class_metrics).sort_values(by="N_Samples", ascending=False)

        report_text = f"""
==================================================
RESULTS
MODEL: {model_name}
MODE: {self.eval_mode.upper()} | SEED: {seed}
==================================================
Total Evaluation Time: {total_time:.2f} seconds
Peak VRAM: {max_vram_mb:.2f} MB

[MODALITY 1: ZERO-SHOT GENERATIVE EXTRACTION]
Ontological Validity (relaxed structural match) F1: {ext_f1:.4f}
Strict Exact Match TP: {ext_metrics['tp']}
Vocabulary Hallucinations: {(ext_metrics['hallucinations'] / N) * 100:.2f}%
Structural Schema Faults:  {(ext_metrics['topology_violations'] / N) * 100:.2f}%

[MODALITY 2: ADVERSARIAL GENERATIVE RANKING]
Ranking Accuracy:          {global_rank_acc:.2f}%
Factual Errors:            {(rank_metrics['factual_errors'] / N) * 100:.2f}%
Topology Errors:           {(rank_metrics['topology_errors'] / N) * 100:.2f}%
Hallucination Errors:      {(rank_metrics['hallucination_errors'] / N) * 100:.2f}%
==================================================

CLASS-WISE METRIC BREAKDOWN:
{class_df.to_string(index=False)}
"""

        if base_model_path == self.config["CONTROL_MODEL"]:
            file_prefix = f"control-{base_model_path.split('/')[-1]}-seed{seed}"
        else:
            file_prefix = f"{base_model_path.replace('/', '_')}-seed{seed}"

        txt_filename = os.path.join(self.output_dir, f"{file_prefix}_{self.eval_mode}_report.txt")
        csv_filename = os.path.join(self.output_dir, f"{file_prefix}_{self.eval_mode}_data.csv")

        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(report_text)
        results_df.to_csv(csv_filename, index=False)

        return txt_filename, csv_filename


if __name__ == "__main__":
    from config import EVAL_CONFIG

    evaluator = Evaluator(EVAL_CONFIG)
    evaluator.prepare_data()
    evaluator.run_evaluation()
