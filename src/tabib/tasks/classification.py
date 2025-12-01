"""Text classification task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import evaluate
import numpy as np

from tabib.tasks.base import Task


@dataclass
class ClassificationMetrics:
    accuracy: Any
    precision: Any
    recall: Any
    f1: Any


class ClassificationTask(Task):
    """Text classification task supporting single-label classification."""

    def __init__(self, label_list: Sequence[str] | None = None) -> None:
        self._metrics = ClassificationMetrics(
            accuracy=evaluate.load("accuracy"),
            precision=evaluate.load("precision"),
            recall=evaluate.load("recall"),
            f1=evaluate.load("f1"),
        )
        self._label_list = list(label_list) if label_list is not None else []
        self._label_to_id = {label: idx for idx, label in enumerate(self._label_list)}

    @property
    def name(self) -> str:
        return "classification"

    @property
    def label_space(self) -> dict[str, int]:
        return self._label_to_id

    @property
    def label_list(self) -> list[str]:
        return self._label_list

    @property
    def num_labels(self) -> int:
        return len(self._label_list)

    def set_label_list(self, labels: Sequence[str]) -> None:
        self._label_list = list(labels)
        self._label_to_id = {label: idx for idx, label in enumerate(self._label_list)}

    def ensure_labels(self, labels: Sequence[str]) -> None:
        if not self._label_list:
            self.set_label_list(labels)

    def compute_metrics(self, predictions: Any, references: Any) -> dict[str, float]:
        logits = predictions["predictions"] if isinstance(predictions, dict) else predictions
        label_ids = predictions.get("label_ids") if isinstance(predictions, dict) else references

        logits_array = np.array(logits)
        if logits_array.ndim == 1:
            pred_ids = logits_array.astype(int)
        else:
            pred_ids = logits_array.argmax(axis=-1)
            logits_array = None  # free reference

        results = {
            "accuracy": self._metrics.accuracy.compute(predictions=pred_ids, references=label_ids)["accuracy"],
            "precision": self._metrics.precision.compute(predictions=pred_ids, references=label_ids, average="macro")["precision"],
            "recall": self._metrics.recall.compute(predictions=pred_ids, references=label_ids, average="macro")["recall"],
            "f1": self._metrics.f1.compute(predictions=pred_ids, references=label_ids, average="macro")["f1"],
        }
        return results

    def format_output(self, predictions: Any) -> list[str]:
        logits = predictions["predictions"] if isinstance(predictions, dict) else predictions
        logits_array = np.array(logits)
        if logits_array.ndim == 1:
            pred_ids = logits_array.astype(int)
        else:
            pred_ids = logits_array.argmax(axis=-1)
        if not self._label_list:
            return pred_ids.tolist()
        return [self._label_list[int(idx)] for idx in pred_ids]
