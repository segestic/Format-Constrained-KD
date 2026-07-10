"""
data_utils.py
=============
Training-time data utilities. Sourced exclusively from PrimeKG (via TDC).
PharmGKB data loading for cross-domain evaluation lives separately in
evaluate.py, so that the training-data path and evaluation-data path can
never accidentally share state.
"""

import random

from datasets import Dataset
from tdc.resource import PrimeKG


class PrimeKGDataLoader:
    """Generates strictly aligned SFT / ORPO datasets from PrimeKG."""

    def __init__(self, seed: int, sample_cap: int):
        self.seed = seed
        self.sample_cap = sample_cap

    def get_aligned_datasets(self):
        """Generates strictly aligned train/test datasets for SFT and ORPO.

        Both datasets are built from the same underlying sample and split
        with the same seed, guaranteeing zero data leakage and a perfect
        1-to-1 row correspondence between the SFT and ORPO training sets.
        """
        print(f"\n📚 Formatting Aligned PrimeKG Datasets (Seed: {self.seed}, Cap: {self.sample_cap})...")

        data = PrimeKG(path="./data")
        df = data.get_data().dropna(subset=["x_name", "y_name"])
        df_sample = (
            df.groupby("relation", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), self.sample_cap), random_state=self.seed))
            .copy()
        )

        sft_texts, orpo_prompts, orpo_chosens, orpo_rejecteds = [], [], [], []

        for _, row in df_sample.iterrows():
            sft_texts.append(self._build_sft_text(row))
            prompt, chosen, rejected = self._build_orpo_components(row)
            orpo_prompts.append(prompt)
            orpo_chosens.append(chosen)
            orpo_rejecteds.append(rejected)

        sft_dataset = Dataset.from_dict({"text": sft_texts})
        orpo_dataset = Dataset.from_dict(
            {"prompt": orpo_prompts, "chosen": orpo_chosens, "rejected": orpo_rejecteds}
        )

        sft_split = sft_dataset.train_test_split(test_size=0.1, seed=self.seed)
        orpo_split = orpo_dataset.train_test_split(test_size=0.1, seed=self.seed)

        return sft_split, orpo_split

    def _build_sft_text(self, row):
        return (
            f"Analyze the biomedical relationship.\n"
            f"Source Node: {row['x_name']} (Type: {row['x_type']})\n"
            f"Relationship: {row['relation']}\n"
            f"Target Node: {row['y_name']} (Type: {row['y_type']})"
        )

    def _build_orpo_components(self, row):
        prompt = (
            f"Analyze the biomedical relationship.\n"
            f"Source Node: {row['x_name']} (Type: {row['x_type']})\n"
            f"Relationship: {row['relation']}\n"
            f"Target Node:"
        )
        chosen = f" {row['y_name']} (Type: {row['y_type']})"

        violation_type = random.choice(["self_loop", "type_mismatch", "reversed_flow"])
        if violation_type == "self_loop":
            rejected = f" {row['x_name']} (Type: {row['x_type']})"
        elif violation_type == "type_mismatch":
            rejected = f" {row['y_name']} (Type: {'disease' if row['y_type'] != 'disease' else 'drug'})"
        else:
            rejected = f" {row['x_name']} (Type: {row['y_type']})"

        return prompt, chosen, rejected
