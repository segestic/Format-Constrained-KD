"""
evaluate.py
#export CUBLAS_WORKSPACE_CONFIG=:4096:8
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

FIX (post-review): Ontological Validity vs. Strict Exact Match
----------------------------------------------------------------
OV is scored separately from SEM: a prediction counts as "ontologically
valid" if it names a real, correctly-typed graph entity, REGARDLESS of
whether it's the one specific ground-truth target for that edge (both
"tp" and "fp_fn" count toward OV). Only "hallucination" and
"topology_violation" count against OV precision; only "fn" counts against
OV recall. SEM keeps its original, stricter exact-match definition.

FIX (post-review): skip already-completed (model, seed) evaluations
-----------------------------------------------------------------------
run_evaluation() checks whether a report/CSV pair already exists for a
given (base_model_path, seed, eval_mode) before loading the model and
running generation, and skips straight to reusing the existing files if
so. Set EVAL_CONFIG["force_rerun"] = True to force every (model, seed) to
be re-evaluated from scratch.

FIX (post-review): Ranking Accuracy run-to-run reproducibility
-----------------------------------------------------------------------
Ranking Accuracy (Modality 2) was observed to vary by several percentage
points across separate `python evaluate.py` invocations of the SAME
checkpoint/seed, while Ontological Validity (Modality 1) stayed bit-
identical every time. Isolated single-model reproduction tests (same
checkpoint, same seed, in two completely separate process launches) came
back bit-identical, which rules out GPU kernel non-determinism as the
sole cause -- yet full multi-model pipeline runs still drifted relative
to those isolated tests. Since the exact remaining mechanism could not be
conclusively isolated without further live testing, this fix closes every
plausible source at once rather than guessing at a single root cause:

  1. self.ontology_dict values are now sorted() (previously plain list()
     over a set, which is only stable if the environment's PYTHONHASHSEED
     happens to be fixed -- true in this container by chance, but not a
     property the code should silently depend on).
  2. torch.use_deterministic_algorithms(True) and cudnn determinism flags
     are set at import time, trading a small amount of speed for
     eliminating GPU-kernel-level non-determinism as a variable.
  3. Pass B (adversarial ranking) now explicitly reseeds `random` and
     `numpy.random` per BATCH, keyed to (seed, batch_start_index) rather
     than relying solely on the single reseed at the top of
     run_evaluation(). This removes any dependency on what ran earlier in
     the same process (other models, other seeds, upstream RNG
     consumption in prepare_data() or dataset construction) --
     trap-option selection for a given batch is now a pure function of
     (seed, batch_start_index) alone.

Together, these mean Ranking Accuracy for a given (model, seed) should now
be reproducible both in isolation and as part of a full multi-model run.
If you still observe drift after this fix, the remaining candidate is
non-determinism inside a specific PyTorch/bitsandbytes op that
use_deterministic_algorithms(True) does not cover for this model's ops --
in that case CHECK the console for a RuntimeError naming the offending op
(deterministic mode raises rather than silently falling back), rather than
assuming silent success.

SPEED (output-preserving) changes
-----------------------------------------------------------------------
  A. Base-model reuse: the (4-bit) base model and tokenizer are loaded ONCE
     per run_evaluation() call. Each (arm, seed) LoRA adapter is attached
     with load_adapter()/set_adapter() and removed after use; the control
     arm runs under disable_adapter(). Previously the base model was
     reloaded and re-quantized for every (arm, seed).
     Disable with EVAL_CONFIG["reuse_base_model"] = False to fall back to
     the original per-(arm, seed) loading. VERIFY once before trusting it:
     run one (arm, seed) both ways and diff the two CSVs.
  B. Sharding: set EVAL_SHARD="k/n" (e.g. "0/2", "1/2") to run only every
     n-th (model, seed) job starting at k, so several processes / GPUs
     (CUDA_VISIBLE_DEVICES) can split one run. Results are unaffected
     because all randomness is keyed to (seed, batch index).
  C. ontology_dict is built with groupby(sort=False) instead of iterrows().
     sort=False keeps the original first-appearance key order, which
     matters because Pass B draws from list(ontology_dict.keys()).
  D. Pass B tokenizes each unique prompt once instead of once per option.

FIX: robustness-mode character swaps now use their own RNG keyed to
(seed, batch index) instead of the module-level `random`. This changes
EVAL_MODE="robustness" outputs of THIS script only (the reported noise
rows come from evaluate_expanded_noise_robustness.py).
"""

import contextlib
import gc
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# --------------------------------------------------------------------------
# FIX: full GPU determinism. Requires CUBLAS_WORKSPACE_CONFIG=:4096:8 (or
# :16:8) to be set in the environment BEFORE this process starts -- set it
# in your shell (export CUBLAS_WORKSPACE_CONFIG=:4096:8) before running
# `python evaluate.py`, since it cannot be set reliably from inside Python
# once CUDA is initialized.
# --------------------------------------------------------------------------

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

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
try:
    torch.use_deterministic_algorithms(True)
except Exception as e:
    print(f"⚠️  torch.use_deterministic_algorithms(True) could not be fully enabled: {e}")
    print("    Continuing without it -- Ranking Accuracy reproducibility may be reduced.")


class Evaluator:
    def __init__(self, eval_config: dict):
        self.config = eval_config
        self.seeds = eval_config["SEEDS_TO_EVAL"]
        self.eval_mode = eval_config["EVAL_MODE"]
        self.all_models = eval_config["ALL_MODELS"]
        self.output_dir = eval_config.get("output_dir", "./eval_outputs")
        self.force_rerun = eval_config.get("force_rerun", False)
        self.reuse_base_model = eval_config.get("reuse_base_model", True)

        self.datasets_by_seed = {}
        self.ontology_dict = {}
        self.valid_nodes = set()
        self.valid_node_types = {}

        # Shared-base state (SPEED change A)
        self._base = None
        self._tokenizer = None
        self._peft = None
        self._anchor_adapter = None

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

        # SPEED change C: vectorised replacement for the iterrows() loop.
        # sort=False preserves first-appearance key order (identical to the
        # old setdefault() insertion order); values are sorted() as before.
        self.ontology_dict = {
            k: sorted(set(v))
            for k, v in df.groupby("y_type", sort=False)["y_name"]
        }

        for seed in self.seeds:
            random.seed(seed)
            np.random.seed(seed)
            df_pool = self._exclude_training_edges(df, seed) if self.eval_mode == "heldout" else df
            df_stratified = (
            df_pool.groupby("relation", group_keys=False)
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

    def _exclude_training_edges(self, df: pd.DataFrame, seed: int) -> pd.DataFrame:
        """EVAL_MODE="heldout": remove every PrimeKG edge that any training arm
        could have seen for this seed. All arms (SFT, ORPO, FC-KD, Vanilla-KD,
        controls) draw the same per-relation sample, built exactly as in
        data_utils.PrimeKGDataLoader; both edge directions are excluded."""
        from config import CONFIG
        cap = CONFIG["GLOBAL_SAMPLE_CAP"]
        # Same per-group sampling as data_utils (each relation sampled with the
        # same random_state); written as a loop so it works on every pandas version.
        sample = pd.concat([g.sample(min(len(g), cap), random_state=seed)
                            for _, g in df.groupby("relation", sort=True)])
        keys = set(zip(sample["x_name"], sample["relation"], sample["y_name"]))
        keys |= {(y, r, x) for (x, r, y) in keys}
        rest = df[~df.index.isin(sample.index)]
        keep = [k not in keys for k in zip(rest["x_name"], rest["relation"], rest["y_name"])]
        pool = rest[keep]
        print(f"  - Seed {seed}: held-out pool {len(pool)} edges "
              f"({len(df) - len(pool)} excluded as training-sample edges or their reverses).")
        return pool

    def _load_pharmgkb(self) -> pd.DataFrame:
        os.makedirs("./data", exist_ok=True)
        zip_path = "./data/relationships.zip"
        tsv_path = "./data/relationships.tsv"

        if not os.path.exists(zip_path):
            print("📥 Downloading PharmGKB relationships.zip...")
            url = "https://api.clinpgx.org/v1/download/file/data/relationships.zip"
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
    def inject_character_swaps(text: str, noise_level: float, rng=None) -> str:
        # rng defaults to the module-level `random` so existing subclasses
        # calling inject_character_swaps(text, noise_level) keep working.
        rng = rng if rng is not None else random
        if len(text) < 2 or noise_level <= 0:
            return text
        text_list = list(text)
        num_swaps = max(1, int(len(text) * noise_level))
        for _ in range(num_swaps):
            idx = rng.randint(0, len(text_list) - 2)
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

        # SPEED change D: each prompt appears once per option (4x); tokenize
        # each unique prompt only once. Identical lengths, identical masking.
        prompt_len_cache = {}
        for i, p in enumerate(prompts):
            if p not in prompt_len_cache:
                prompt_len_cache[p] = len(tokenizer(p, add_special_tokens=False)["input_ids"])
            p_len = prompt_len_cache[p]
            labels[i, :p_len] = -100
            labels[i, inputs["input_ids"][i] == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            # labels are NOT passed to the model: its internal loss (a float32 copy of
            # the full-vocabulary logits) was computed and discarded, costing ~8 GB
            # on Qwen2.5-1.5B. The per-sequence loss below is computed from the same
            # logits, so results are unchanged.
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            # Per-token cross-entropy computed in chunks of sequences. Each token's
            # loss depends only on its own logits row, so chunking changes peak
            # memory (the float32 upcast of full-vocabulary logits), not the values.
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            chunk = 16
            parts = []
            for s0 in range(0, logits.size(0), chunk):
                lg = logits[s0:s0 + chunk, :-1, :].contiguous()
                lb = labels[s0:s0 + chunk, 1:].contiguous()
                tok_loss = loss_fct(lg.view(-1, lg.size(-1)), lb.view(-1)).view(lb.size())
                parts.append(tok_loss.sum(dim=1) / (lb != -100).sum(dim=1))
                del lg, tok_loss
            seq_loss = torch.cat(parts)

        return seq_loss.tolist()

    # --------------------------------------------------------------------
    # Output-path helper (shared by the skip-if-exists check and by
    # _write_reports, so the two can never drift out of sync on naming)
    # --------------------------------------------------------------------
    def _report_paths(self, base_model_path, seed):
        if base_model_path == self.config["CONTROL_MODEL"]:
            file_prefix = f"control-{base_model_path.split('/')[-1]}-seed{seed}"
        else:
            file_prefix = f"{base_model_path.replace('/', '_')}-seed{seed}"

        txt_filename = os.path.join(self.output_dir, f"{file_prefix}_{self.eval_mode}_report.txt")
        csv_filename = os.path.join(self.output_dir, f"{file_prefix}_{self.eval_mode}_data.csv")
        return txt_filename, csv_filename

    # --------------------------------------------------------------------
    # SPEED change B: optional sharding across processes / GPUs
    # --------------------------------------------------------------------
    @staticmethod
    def _select_shard(jobs):
        shard = os.environ.get("EVAL_SHARD")
        if not shard:
            return jobs
        k, n = (int(x) for x in shard.split("/"))
        if not (0 <= k < n):
            raise ValueError(f"EVAL_SHARD must be 'k/n' with 0 <= k < n, got {shard!r}")
        selected = [job for idx, job in enumerate(jobs) if idx % n == k]
        print(f"🔀 EVAL_SHARD={shard}: running {len(selected)}/{len(jobs)} (model, seed) jobs in this process.")
        return selected

    # --------------------------------------------------------------------
    # Main evaluation loop
    # --------------------------------------------------------------------
    def run_evaluation(self):
        all_generated_files = []

        jobs = [(base_model_path, seed) for base_model_path in self.all_models for seed in self.seeds]
        jobs = self._select_shard(jobs)

        current_family = None
        for base_model_path, seed in jobs:
            if base_model_path != current_family:
                print(f"\n{'=' * 60}\n🚀 MODEL FAMILY: {base_model_path}\n{'=' * 60}")
                current_family = base_model_path

            txt_filename, csv_filename = self._report_paths(base_model_path, seed)

            if (not self.force_rerun) and os.path.exists(txt_filename) and os.path.exists(csv_filename):
                print(f"  ⏭️  Skipping seed {seed} for {base_model_path} -- "
                      f"report + CSV already exist ({os.path.basename(txt_filename)}). "
                      f"Set EVAL_CONFIG['force_rerun']=True to redo it.")
                all_generated_files.extend([txt_filename, csv_filename])
                continue

            if self.reuse_base_model:
                model, tokenizer, model_name, adapter_name, run_ctx = self._get_shared_model(base_model_path, seed)
                base_model = None
            else:
                model, base_model, tokenizer, model_name = self._load_model(base_model_path, seed)
                adapter_name, run_ctx = None, contextlib.nullcontext()
            model.eval()

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            dataset = self.datasets_by_seed[seed]
            torch.cuda.reset_peak_memory_stats()

            with run_ctx:
                csv_results, ext_metrics, rank_metrics, total_time = self._evaluate_dataset_batched(
                    dataset, model, tokenizer, seed
                )

            txt_filename, csv_filename = self._write_reports(
                model_name, base_model_path, seed, csv_results, ext_metrics, rank_metrics, total_time
            )
            all_generated_files.extend([txt_filename, csv_filename])
            print(f"  ✅ Saved: {txt_filename} & {csv_filename}")

            if self.reuse_base_model:
                self._release_adapter(adapter_name)
                del model, tokenizer
            else:
                del model, base_model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

        self._unload_shared_base()

        print("\n" + "=" * 60)
        print("🎉 EVALUATION COMPLETE. Generated files:")
        for f in all_generated_files:
            print(f"  - {f}")
        return all_generated_files

    # --------------------------------------------------------------------
    # Model loading
    # --------------------------------------------------------------------
    def _load_kwargs(self):
        if self.config.get("use_4bit", True):
            # NF4 to match training (models.get_quantized_base_model). Previously no
            # quant type was set, so bitsandbytes defaulted to FP4.
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
            return {"quantization_config": bnb_config, "device_map": "auto"}
        return {"torch_dtype": torch.bfloat16, "device_map": "auto"}

    def _load_model(self, base_model_path, seed):
        """Original per-(arm, seed) loader. Used when reuse_base_model=False."""
        load_kwargs = self._load_kwargs()

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

    def _ensure_shared_base(self):
        if self._base is not None:
            return
        base_name = self.config["base_model_name"]
        if self.config["CONTROL_MODEL"] != base_name:
            raise ValueError(
                f"reuse_base_model=True requires CONTROL_MODEL == base_model_name "
                f"(got {self.config['CONTROL_MODEL']!r} vs {base_name!r}). "
                f"Set EVAL_CONFIG['reuse_base_model']=False for this configuration."
            )
        print(f"📦 Loading shared base model once: {base_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(base_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._base = AutoModelForCausalLM.from_pretrained(base_name, **self._load_kwargs())

    def _get_shared_model(self, base_model_path, seed):
        """Returns (model, tokenizer, model_name, adapter_name, run_ctx)."""
        self._ensure_shared_base()

        if base_model_path == self.config["CONTROL_MODEL"]:
            model_name = self.config["CONTROL_MODEL"]
            if self._peft is None:
                # No LoRA layers injected yet: the raw base IS the control model.
                return self._base, self._tokenizer, model_name, None, contextlib.nullcontext()
            # LoRA layers are injected into the base in place, so the control
            # arm must run with every adapter disabled.
            return self._peft, self._tokenizer, model_name, None, self._peft.disable_adapter()

        model_name = f"{base_model_path}-seed{seed}"
        adapter_name = re.sub(r"[^A-Za-z0-9_]", "_", model_name)

        if self._peft is None:
            self._peft = PeftModel.from_pretrained(self._base, model_name, adapter_name=adapter_name)
            # The first adapter is kept loaded as an anchor so the PeftModel
            # never ends up with zero adapters; it is inactive whenever
            # another adapter is set, so it does not affect other arms.
            self._anchor_adapter = adapter_name
        else:
            self._peft.load_adapter(model_name, adapter_name=adapter_name)

        self._peft.set_adapter(adapter_name)
        return self._peft, self._tokenizer, model_name, adapter_name, contextlib.nullcontext()

    def _release_adapter(self, adapter_name):
        if adapter_name is None or self._peft is None or adapter_name == self._anchor_adapter:
            return
        self._peft.set_adapter(self._anchor_adapter)
        self._peft.delete_adapter(adapter_name)

    def _unload_shared_base(self):
        self._peft = None
        self._base = None
        self._tokenizer = None
        self._anchor_adapter = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------------------
    # Evaluation loop
    # --------------------------------------------------------------------
    def _evaluate_dataset_batched(self, dataset, model, tokenizer, seed):
        batch_size = self.config.get("batch_size", 16)
        csv_results = []
        ext_metrics = {
            "tp": 0, "fp": 0, "fn": 0,
            "hallucinations": 0, "topology_violations": 0,
            "fp_fn_count": 0,  # valid entity, right/unknown type, wrong specific target
        }
        rank_metrics = {"correct": 0, "factual_errors": 0, "topology_errors": 0, "hallucination_errors": 0}
        total_time = 0.0

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            start_time = time.time()

            # FIX: robustness noise gets its own RNG keyed to (seed, batch
            # start index), independent of Pass B and of process history.
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

            # --- PASS B: batched adversarial generative ranking ---
            # FIX: explicit per-batch reseed, keyed to (seed, batch start
            # index i) alone -- makes trap-option selection for this batch
            # independent of anything that ran earlier in this process
            # (other models, other seeds, upstream RNG consumption).
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
                else:  # fp_fn
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

    def _write_reports(self, model_name, base_model_path, seed, csv_results, ext_metrics, rank_metrics, total_time):
        N = len(csv_results)
        max_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        sem_p = ext_metrics["tp"] / (ext_metrics["tp"] + ext_metrics["fp"]) if (ext_metrics["tp"] + ext_metrics["fp"]) > 0 else 0
        sem_r = ext_metrics["tp"] / (ext_metrics["tp"] + ext_metrics["fn"]) if (ext_metrics["tp"] + ext_metrics["fn"]) > 0 else 0
        sem_f1 = 2 * (sem_p * sem_r) / (sem_p + sem_r) if (sem_p + sem_r) > 0 else 0

        ov_tp = ext_metrics["tp"] + ext_metrics["fp_fn_count"]
        ov_fp = ext_metrics["hallucinations"] + ext_metrics["topology_violations"]

        # --- FIX APPLIED HERE ---
        ov_fn = ext_metrics["fn"] - ext_metrics["fp_fn_count"]

        ov_p = ov_tp / (ov_tp + ov_fp) if (ov_tp + ov_fp) > 0 else 0
        ov_r = ov_tp / (ov_tp + ov_fn) if (ov_tp + ov_fn) > 0 else 0
        ov_f1 = 2 * (ov_p * ov_r) / (ov_p + ov_r) if (ov_p + ov_r) > 0 else 0

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
Ontological Validity (relaxed structural match) F1: {ov_f1:.4f}
Strict Exact Match F1: {sem_f1:.4f}
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

        txt_filename, csv_filename = self._report_paths(base_model_path, seed)

        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(report_text)
        results_df.to_csv(csv_filename, index=False)

        return txt_filename, csv_filename


if __name__ == "__main__":
    from config import EVAL_CONFIG

    evaluator = Evaluator(EVAL_CONFIG)
    evaluator.prepare_data()
    evaluator.run_evaluation()
