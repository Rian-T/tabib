"""Semantic similarity task using STS metrics."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import evaluate

from tabib.tasks.base import Task


class SimilarityTask(Task):
    """Semantic textual similarity regression task."""

    def __init__(self, score_range: tuple[float, float] = (0.0, 5.0)) -> None:
        self._metric = evaluate.load("glue", "stsb")
        self._mse = evaluate.load("mse")
        self._score_range = score_range

    @property
    def name(self) -> str:
        return "similarity"

    @property
    def label_space(self) -> dict[str, Any]:
        return {
            "type": "regression",
            "range": self._score_range,
        }

    def compute_metrics(self, predictions: Any, references: Any) -> dict[str, float]:
        preds = self._extract_predictions(predictions)
        refs = self._extract_references(predictions, references)

        glue_metrics = self._metric.compute(predictions=preds, references=refs)
        mse = self._mse.compute(predictions=preds, references=refs)["mse"]

        return {
            "pearson": glue_metrics.get("pearson", float("nan")),
            "spearman": glue_metrics.get("spearmanr", float("nan")),
            "mse": mse,
        }

    def format_output(self, predictions: Any) -> list[float]:
        preds = self._extract_predictions(predictions)
        return preds.tolist()

    def _extract_predictions(self, predictions: Any) -> np.ndarray:
        if isinstance(predictions, dict):
            preds = predictions.get("predictions", predictions)
        else:
            preds = predictions

        preds_array = np.array(preds)
        if preds_array.ndim > 1:
            preds_array = preds_array.squeeze(-1)
        return preds_array.astype(np.float32)

    def _extract_references(self, predictions: Any, references: Any) -> np.ndarray:
        if isinstance(predictions, dict) and "label_ids" in predictions:
            refs = predictions["label_ids"]
        else:
            refs = references

        return np.array(refs, dtype=np.float32)
