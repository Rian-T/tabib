"""QUAERO MEDLINE dataset adapter (HuggingFace format) - Token-level.

This adapter outputs token-level BIO tags directly (no span conversion).
Uses NERTokenTask for evaluation with seqeval.
"""

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from tabib.data.base import DatasetAdapter


# Entity type mapping from ClassLabel indices
ENTITY_TYPES = [
    "ANAT", "CHEM", "DEVI", "DISO", "GEOG",
    "LIVB", "OBJC", "PHEN", "PHYS", "PROC"
]

# Tag names from dataset (21 classes including O)
TAG_NAMES = [
    "O", "B-ANAT", "I-ANAT", "B-CHEM", "I-CHEM",
    "B-DEVI", "I-DEVI", "B-DISO", "I-DISO", "B-GEOG", "I-GEOG",
    "B-LIVB", "I-LIVB", "B-OBJC", "I-OBJC", "B-PHEN", "I-PHEN",
    "B-PHYS", "I-PHYS", "B-PROC", "I-PROC"
]


class QuaeroMEDLINETokenAdapter(DatasetAdapter):
    """Adapter for QUAERO MEDLINE dataset from HuggingFace (token-level).

    Uses the rntc/quaero-frenchmed-ner-medline dataset which has:
    - tokens: list of tokens
    - ner_tags: list of BIO tag indices

    Already preprocessed with coarse-grained entities and sentence splitting.
    This adapter outputs token-level format directly for NERTokenTask.
    """

    def __init__(self, data_dir: str | Path | None = None):
        """Initialize adapter.

        Args:
            data_dir: Optional local path to dataset. If None, loads from HF.
        """
        self._data_dir = Path(data_dir) if data_dir else None
        self._entity_types = ENTITY_TYPES

    @property
    def name(self) -> str:
        return "quaero_medline_token"

    @property
    def entity_types(self) -> list[str]:
        return self._entity_types

    def load_splits(self) -> dict[str, Dataset]:
        """Load train/dev/test splits."""
        if self._data_dir:
            ds = load_dataset("parquet", data_dir=str(self._data_dir / "data"))
        else:
            ds = load_dataset("rntc/quaero-frenchmed-ner-medline")

        # Convert ner_tags to labels (same integer indices)
        def convert_to_token_format(example, idx):
            return {
                "doc_id": f"medline_{idx}",
                "tokens": example["tokens"],
                "labels": example["ner_tags"],  # Keep as integers
            }

        train_docs = [convert_to_token_format(sample, i) for i, sample in enumerate(ds["train"])]
        dev_docs = [convert_to_token_format(sample, i) for i, sample in enumerate(ds["validation"])]
        test_docs = [convert_to_token_format(sample, i) for i, sample in enumerate(ds["test"])]

        return {
            "train": Dataset.from_list(train_docs),
            "dev": Dataset.from_list(dev_docs),
            "test": Dataset.from_list(test_docs),
        }

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        """Preprocess dataset for the task."""
        from tabib.tasks.ner_token import NERTokenTask

        if isinstance(task, NERTokenTask):
            # Set up the label space with QUAERO tags
            task.set_label_list(TAG_NAMES)

        return dataset
