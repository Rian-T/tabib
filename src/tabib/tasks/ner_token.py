"""Token-level NER task (non-nested).

Provides a configurable BIO label space so downstream datasets can define
their own entity inventory (e.g., biomedical entities) while keeping the
original CoNLL defaults as a fallback.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from seqeval.metrics import accuracy_score, classification_report, f1_score

from tabib.tasks.base import Task


class NERTokenTask(Task):
    """Token-level NER task using BIO tagging scheme."""

    _DEFAULT_LABELS: tuple[str, ...] = (
        "O",
        "B-PER",
        "I-PER",
        "B-ORG",
        "I-ORG",
        "B-LOC",
        "I-LOC",
        "B-MISC",
        "I-MISC",
    )

    def __init__(self, label_list: Sequence[str] | None = None) -> None:
        """Initialize NER task.

        Args:
            label_list: Optional custom BIO label list. When omitted, defaults to
                the CoNLL-2003 tag set. The provided list does not need to
                include ``"O"``; it will be inserted automatically.
        """

        self._label_list: list[str] = []
        self._label_to_id: dict[str, int] = {}
        self._id_to_label: dict[int, str] = {}

        initial_labels = label_list if label_list is not None else self._DEFAULT_LABELS
        self.set_label_list(initial_labels)
    
    @property
    def name(self) -> str:
        """Return the task name."""
        return "ner_token"
    
    @property
    def label_space(self) -> dict[str, int]:
        """Return the label space mapping."""
        return self._label_to_id
    
    @property
    def label_list(self) -> list[str]:
        """Return the list of labels."""
        return self._label_list
    
    @property
    def num_labels(self) -> int:
        """Return the number of labels."""
        return len(self._label_list)
    
    def set_label_list(self, labels: Sequence[str]) -> None:
        """Replace the task label space with a new BIO label list.

        Args:
            labels: Sequence of BIO labels (e.g., ``["O", "B-DISEASE", ...]``).
                Duplicate labels are ignored while preserving the first
                occurrence order. ``"O"`` is inserted at position 0 when
                missing.
        """

        normalized = self._normalize_labels(labels)
        self._label_list = normalized
        self._label_to_id = {label: idx for idx, label in enumerate(self._label_list)}
        self._id_to_label = {idx: label for label, idx in self._label_to_id.items()}

    def ensure_labels(self, labels: Sequence[str]) -> None:
        """Ensure that the given labels exist in the task label space.

        Any missing labels are appended in the order received. ``"O"`` is
        preserved at index 0.

        Args:
            labels: Sequence of BIO labels to register.
        """

        missing = [str(label) for label in labels if str(label) not in self._label_to_id]
        if not missing:
            return

        extended = list(self._label_list) + missing
        self.set_label_list(extended)

    def label_to_id(self, label: str) -> int:
        """Convert label to ID."""
        return self._label_to_id[label]
    
    def id_to_label(self, label_id: int) -> str:
        """Convert ID to label."""
        return self._id_to_label[label_id]

    def has_label(self, label: str) -> bool:
        """Return True if the label is part of the task label space."""

        return label in self._label_to_id
    
    def compute_metrics(self, predictions: Any, references: Any) -> dict[str, float]:
        """Compute NER metrics using seqeval.
        
        Args:
            predictions: Model predictions (can be logits or label IDs)
            references: Ground truth labels
            
        Returns:
            Dictionary with f1, precision, recall, accuracy
        """
        # Handle different prediction formats
        if isinstance(predictions, dict):
            # If predictions is a dict with 'predictions' key
            preds = predictions.get("predictions", predictions)
        else:
            preds = predictions
        
        # Convert predictions to label IDs if they're logits
        if isinstance(preds, np.ndarray) and len(preds.shape) > 1:
            preds = np.argmax(preds, axis=-1)
        
        # Extract labels from references (handle dataset format)
        if hasattr(references, "features"):
            # HuggingFace dataset
            if "labels" in references.features:
                ref_labels = references["labels"]
            elif "ner_tags" in references.features:
                ref_labels = references["ner_tags"]
            else:
                raise ValueError("Dataset must have 'labels' or 'ner_tags' field")
        elif isinstance(references, dict) and "labels" in references:
            # Dict with labels list
            ref_labels = references["labels"]
        elif isinstance(references, list):
            # List of label sequences
            ref_labels = references
        else:
            raise ValueError(f"Unexpected references format: {type(references)}")
        
        # Convert to list of lists of label strings
        pred_labels = [[self.id_to_label(int(p)) for p in pred] for pred in preds]
        true_labels = [[self.id_to_label(int(r)) for r in ref] for ref in ref_labels]
        
        # Compute metrics
        f1 = f1_score(true_labels, pred_labels)
        accuracy = accuracy_score(true_labels, pred_labels)
        
        # Get detailed report
        report = classification_report(true_labels, pred_labels, output_dict=True)
        
        metrics = {
            "f1": f1,
            "accuracy": accuracy,
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
        }
        
        # Add per-entity metrics
        for label in ["PER", "ORG", "LOC", "MISC"]:
            if f"B-{label}" in report:
                metrics[f"{label.lower()}_f1"] = report[f"B-{label}"]["f1-score"]
        
        return metrics
    
    def format_output(self, predictions: Any) -> Any:
        """Format predictions for output.
        
        Args:
            predictions: Raw model predictions
            
        Returns:
            Formatted predictions as list of label strings
        """
        if isinstance(predictions, dict):
            preds = predictions.get("predictions", predictions)
        else:
            preds = predictions
        
        if isinstance(preds, np.ndarray) and len(preds.shape) > 1:
            preds = np.argmax(preds, axis=-1)
        
        return [[self.id_to_label(int(p)) for p in pred] for pred in preds]

    @staticmethod
    def _normalize_labels(labels: Sequence[str]) -> list[str]:
        """Normalize a BIO label list, enforcing ``"O"`` at position 0."""

        seen: set[str] = set()
        ordered: list[str] = []
        for label in labels:
            label_str = str(label)
            if label_str not in seen:
                seen.add(label_str)
                ordered.append(label_str)

        if "O" in seen:
            ordered = ["O"] + [label for label in ordered if label != "O"]
        else:
            ordered.insert(0, "O")

        return ordered

