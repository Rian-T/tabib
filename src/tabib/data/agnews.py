"""AG News dataset adapter for text classification."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.classification import ClassificationTask


class AGNewsAdapter(DatasetAdapter):
    """Adapter for the AG News single-label classification dataset."""

    def __init__(self, validation_size: float = 0.1, seed: int = 42) -> None:
        self.validation_size = validation_size
        self.seed = seed

    @property
    def name(self) -> str:
        return "ag_news"

    def load_splits(self) -> dict[str, Dataset]:
        dataset: DatasetDict = load_dataset("ag_news")
        train_split: Dataset = dataset["train"]

        if self.validation_size > 0:
            split: DatasetDict = train_split.train_test_split(
                test_size=self.validation_size,
                seed=self.seed,
            )
            train_dataset, val_dataset = split["train"], split["test"]
        else:
            train_dataset, val_dataset = train_split, None

        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": dataset["test"],
        }

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, ClassificationTask):
            raise ValueError(f"Expected ClassificationTask, got {type(task)}")

        if not task.label_list:
            label_names = dataset.features["label"].names
            task.ensure_labels(label_names)

        def prepare(example: dict[str, Any]) -> dict[str, Any]:
            return {
                "text": example["text"],
                "labels": example["label"],
            }

        processed = dataset.map(prepare, remove_columns=dataset.column_names)
        processed.set_format(type="python")
        return processed
