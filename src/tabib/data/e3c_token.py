"""E3C dataset adapter (HuggingFace format) - Token-level.

This adapter outputs token-level IOB2 tags directly (no span conversion).
Uses NERTokenTask for evaluation with seqeval.
"""

import os
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from tabib.data.base import DatasetAdapter


# E3C has a single entity type: CLINENTITY
ENTITY_TYPES = ["CLINENTITY"]

# Tag names (3 classes: O, B-CLINENTITY, I-CLINENTITY)
TAG_NAMES = ["O", "B-CLINENTITY", "I-CLINENTITY"]


class E3CTokenAdapter(DatasetAdapter):
    """Adapter for E3C dataset from HuggingFace (token-level).

    Uses the rntc/e3c-legacy-ner-fr dataset which has:
    - tokens: list of tokens
    - clinical_entity_tags: list of IOB2 tag indices (0=O, 1=B, 2=I)

    This adapter outputs token-level format directly for NERTokenTask.
    """

    def __init__(self, data_dir: str | Path | None = None):
        """Initialize adapter.

        Args:
            data_dir: Optional local path to dataset. If None, uses default location.
        """
        if data_dir:
            self._data_dir = Path(data_dir)
        else:
            # Default location
            scratch = os.environ.get('SCRATCH', '/lustre/fsn1/projects/rech/rua/uvb79kr')
            self._data_dir = Path(scratch) / "tabib" / "data" / "e3c-legacy-ner-fr"
        self._entity_types = ENTITY_TYPES

    @property
    def name(self) -> str:
        return "e3c_token"

    @property
    def entity_types(self) -> list[str]:
        return self._entity_types

    def load_splits(self) -> dict[str, Dataset]:
        """Load train/dev/test splits."""
        # Always load from local parquet files
        ds = load_dataset("parquet", data_dir=str(self._data_dir / "data"))

        # Convert clinical_entity_tags to labels
        def convert_to_token_format(example, idx):
            return {
                "doc_id": f"e3c_{idx}",
                "tokens": example["tokens"],
                "labels": example["clinical_entity_tags"],  # Already 0=O, 1=B, 2=I
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
            # Set up the label space with E3C tags
            task.set_label_list(TAG_NAMES)

        return dataset
