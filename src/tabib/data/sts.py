"""Semantic Textual Similarity (STS) dataset adapter."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.similarity import SimilarityTask


class STSAdapter(DatasetAdapter):
    """Adapter for the GLUE STS-B dataset."""

    def __init__(self) -> None:
        self._config_name = "stsb"

    @property
    def name(self) -> str:
        return "sts"

    def load_splits(self) -> dict[str, Dataset]:
        dataset: DatasetDict = load_dataset("glue", self._config_name)
        return {
            "train": dataset["train"],
            "val": dataset["validation"],
            "test": dataset["test"],
        }

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, SimilarityTask):
            raise ValueError(f"Expected SimilarityTask, got {type(task)}")

        def prepare(example: dict[str, Any]) -> dict[str, Any]:
            return {
                "text_left": example["sentence1"],
                "text_right": example["sentence2"],
                "labels": float(example["label"]),
            }

        processed = dataset.map(prepare, remove_columns=dataset.column_names)
        processed.set_format(type="python")
        return processed
