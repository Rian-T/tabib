"""Multiple-choice question answering task."""

from __future__ import annotations

from typing import Any, Iterable

from tabib.tasks.classification import ClassificationTask


class MultipleChoiceTask(ClassificationTask):
    """Multiple-choice question answering task.

    This task reuses the classification infrastructure but exposes a distinct
    task name and augments the reported metrics with exact match. Labels are
    stored as whitespace-separated combinations of answer letter identifiers
    (e.g. ``\"a b\"``).
    """

    @property
    def name(self) -> str:
        return "mcqa"

    def compute_metrics(self, predictions: Any, references: Any) -> dict[str, float]:
        metrics = super().compute_metrics(predictions, references)
        # Accuracy corresponds to exact match in the multi-answer setup.
        if "accuracy" in metrics:
            metrics["exact_match"] = metrics["accuracy"]
        else:
            metrics["exact_match"] = 0.0

        # Additional multi-label metrics (Hamming score and EMR) when label
        # strings are available (e.g., "A B").
        label_list = getattr(self, "_label_list", None) or []
        logits = predictions["predictions"] if isinstance(predictions, dict) else predictions
        label_ids = predictions.get("label_ids") if isinstance(predictions, dict) else references

        if label_list and label_ids is not None:
            pred_ids = logits.argmax(axis=-1) if hasattr(logits, "shape") and len(getattr(logits, "shape", [])) > 1 else logits

            def _combo_to_set(label: str) -> set[str]:
                return {part.strip().upper() for part in str(label).split() if part.strip()}

            # Stable universe from configured label list, not just batch labels.
            universe: set[str] = set().union(*[_combo_to_set(lbl) for lbl in label_list]) if label_list else set()
            universe_list = sorted(universe)

            true_sets = [_combo_to_set(label_list[int(idx)]) for idx in label_ids]
            pred_sets = [_combo_to_set(label_list[int(idx)]) for idx in pred_ids]

            def _hamming_sample(pred: set[str], true: set[str], labels: Iterable[str], n_labels: int) -> float:
                if n_labels == 0:
                    return 0.0
                matches = sum(((lbl in pred) == (lbl in true)) for lbl in labels)
                return matches / n_labels

            if universe_list:
                hamming_scores = [
                    _hamming_sample(p, t, universe_list, len(universe_list)) for p, t in zip(pred_sets, true_sets)
                ]
                metrics["hamming_score"] = float(sum(hamming_scores) / len(hamming_scores))
            else:
                metrics["hamming_score"] = 0.0

            emr = sum(1 for p, t in zip(pred_sets, true_sets) if p == t) / len(true_sets)
            metrics["emr"] = float(emr)
        else:
            metrics["hamming_score"] = 0.0
            metrics["emr"] = metrics["exact_match"]
        return metrics

    def format_output(self, predictions: Any) -> list[list[str]]:
        label_strings = super().format_output(predictions)
        formatted: list[list[str]] = []
        for label in label_strings:
            if isinstance(label, str):
                formatted.append(label.split())
            else:
                formatted.append([str(label)])
        return formatted
