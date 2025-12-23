"""CLISTER dataset adapter for French clinical semantic similarity."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.similarity import SimilarityTask


class CLISTERAdapter(DatasetAdapter):
    """Adapter for CLISTER French clinical semantic similarity dataset."""

    @property
    def name(self) -> str:
        return "clister"

    def load_splits(self) -> dict[str, Dataset]:
        """Load train/val/test splits from HuggingFace."""
        hf_ds = load_dataset("DrBenchmark/CLISTER", trust_remote_code=True)

        splits: dict[str, Dataset] = {}
        split_map = {"train": "train", "validation": "val", "test": "test"}

        for hf_split, local_split in split_map.items():
            if hf_split not in hf_ds:
                continue

            records = []
            for item in hf_ds[hf_split]:
                text_1 = item.get("text_1", "")
                text_2 = item.get("text_2", "")
                sim = float(item.get("label", 0))

                if text_1 and text_2:
                    records.append({
                        "text_1": text_1,
                        "text_2": text_2,
                        "similarity": sim,
                    })

            if records:
                splits[local_split] = Dataset.from_list(records)

        return splits

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, SimilarityTask):
            raise ValueError(f"CLISTER expects SimilarityTask, got {type(task)}")

        def map_example(example: dict[str, Any]) -> dict[str, Any]:
            return {
                "text_left": example["text_1"],
                "text_right": example["text_2"],
                "labels": example["similarity"],
            }

        processed = dataset.map(map_example, remove_columns=dataset.column_names)
        processed.set_format(type="python")
        return processed
