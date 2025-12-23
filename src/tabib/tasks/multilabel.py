"""Multi-label classification task."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from tabib.tasks.base import Task

# Default threshold for multi-label classification
# With weighted BCE loss, use higher threshold (0.5) to balance precision/recall
DEFAULT_THRESHOLD = 0.5


class MultiLabelTask(Task):
    """Multi-label classification task with BCE loss.

    Each sample can have multiple labels assigned simultaneously.
    Uses sigmoid activation + binary cross-entropy loss.
    """

    def __init__(self, label_list: Sequence[str] | None = None) -> None:
        """Initialize task.

        Args:
            label_list: Optional list of label names
        """
        self._label_list = list(label_list) if label_list else []
        self._label_to_id = {label: idx for idx, label in enumerate(self._label_list)}

    @property
    def name(self) -> str:
        return "multilabel"

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
        """Set the label list.

        Args:
            labels: List of label names
        """
        self._label_list = list(labels)
        self._label_to_id = {label: idx for idx, label in enumerate(self._label_list)}

    def ensure_labels(self, labels: Sequence[str]) -> None:
        """Ensure labels are set (set if empty).

        Args:
            labels: List of label names
        """
        if not self._label_list:
            self.set_label_list(labels)

    def compute_metrics(self, predictions: Any, references: Any, threshold: float = DEFAULT_THRESHOLD) -> dict[str, float]:
        """Compute multi-label classification metrics.

        Args:
            predictions: Dict with 'predictions' (logits) and optionally 'label_ids'
            references: Ground truth labels (multi-hot vectors)
            threshold: Probability threshold for positive prediction.
                       Using 0.3 instead of 0.5 to account for class imbalance.

        Returns:
            Dictionary with f1_micro, f1_macro, f1_samples, precision_micro, recall_micro
        """
        logits = predictions["predictions"] if isinstance(predictions, dict) else predictions
        label_ids = predictions.get("label_ids") if isinstance(predictions, dict) else references

        logits_array = np.array(logits)
        labels_array = np.array(label_ids)

        # Sigmoid + threshold
        probs = 1 / (1 + np.exp(-logits_array))
        pred_binary = (probs > threshold).astype(int)

        # Fallback: if no predictions for a sample, predict top-1
        for i in range(len(pred_binary)):
            if pred_binary[i].sum() == 0:
                top_idx = probs[i].argmax()
                pred_binary[i, top_idx] = 1

        return {
            "f1_micro": float(f1_score(labels_array, pred_binary, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(labels_array, pred_binary, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(labels_array, pred_binary, average="weighted", zero_division=0)),
            "f1_samples": float(f1_score(labels_array, pred_binary, average="samples", zero_division=0)),
            "precision_micro": float(precision_score(labels_array, pred_binary, average="micro", zero_division=0)),
            "recall_micro": float(recall_score(labels_array, pred_binary, average="micro", zero_division=0)),
        }

    def format_output(self, predictions: Any, threshold: float = DEFAULT_THRESHOLD) -> list[list[str]]:
        """Format predictions as list of predicted labels per sample.

        Args:
            predictions: Raw model logits
            threshold: Probability threshold for positive prediction

        Returns:
            List of lists, each containing predicted label names
        """
        logits = predictions["predictions"] if isinstance(predictions, dict) else predictions
        logits_array = np.array(logits)

        # Sigmoid + threshold
        probs = 1 / (1 + np.exp(-logits_array))
        pred_binary = probs > threshold

        results = []
        for i, sample_preds in enumerate(pred_binary):
            sample_labels = [
                self._label_list[j]
                for j, is_positive in enumerate(sample_preds)
                if is_positive and j < len(self._label_list)
            ]
            # Fallback: predict top-1 if no predictions
            if not sample_labels:
                top_idx = probs[i].argmax()
                if top_idx < len(self._label_list):
                    sample_labels = [self._label_list[top_idx]]
            results.append(sample_labels)
        return results
