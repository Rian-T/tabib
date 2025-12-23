"""MORFITT multi-label dataset adapter for medical speciality classification."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.multilabel import MultiLabelTask

# Medical specialties from MORFITT dataset (12 classes)
MORFITT_SPECIALTIES = [
    'microbiology', 'etiology', 'virology', 'physiology', 'immunology',
    'parasitology', 'genetics', 'chemistry', 'veterinary', 'surgery',
    'pharmacology', 'psychology'
]


class MORFITTAdapter(DatasetAdapter):
    """Multi-label adapter for MORFITT medical speciality classification.

    This is a multi-label task where each abstract can have multiple
    medical specialties assigned. Uses specialities_one_hot from the
    HuggingFace dataset for proper multi-label classification.

    Reference:
        Labrak et al. (2023) "MORFITT: A multi-label corpus of French
        scientific articles in the biomedical domain" (TALN 2023)
    """

    def __init__(self) -> None:
        self._label_vocab: list[str] = MORFITT_SPECIALTIES

    @property
    def name(self) -> str:
        return "morfitt"

    def load_splits(self) -> dict[str, Dataset]:
        """Load train/val/test splits from HuggingFace with multi-hot labels."""
        hf_ds = load_dataset("DrBenchmark/MORFITT", "source", trust_remote_code=True)

        splits: dict[str, Dataset] = {}
        split_map = {"train": "train", "validation": "val", "test": "test"}

        for hf_split, local_split in split_map.items():
            if hf_split not in hf_ds:
                continue

            records = []
            for item in hf_ds[hf_split]:
                text = (item.get("abstract") or "").strip()
                # Use specialities_one_hot directly from HF dataset
                one_hot = item.get("specialities_one_hot")

                if not text or one_hot is None:
                    continue

                # Convert to float for BCE loss
                records.append({
                    "text": text,
                    "labels": [float(x) for x in one_hot],
                })

            if records:
                splits[local_split] = Dataset.from_list(records)

        return splits

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        """Preprocess dataset for multi-label classification.

        Labels are already in multi-hot format from load_splits().
        """
        if not isinstance(task, MultiLabelTask):
            raise ValueError(
                f"MORFITT expects MultiLabelTask, got {type(task)}"
            )

        # Set label vocabulary
        task.ensure_labels(self._label_vocab)

        # Labels already preprocessed in load_splits
        return dataset
