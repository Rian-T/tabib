"""Span-based NER task with character offsets and nested entity support."""

from typing import Any

from tabib.evaluation.span_evaluator import SpanEvaluator
from tabib.tasks.base import Task


class NERSpanTask(Task):
    """Span-based NER task using character offsets.

    Supports nested entities and evaluates with exact/partial match F1.
    Also computes seqeval F1 for comparison with standard benchmarks.
    """

    def __init__(self, entity_types: list[str] | None = None):
        """Initialize span NER task.

        Args:
            entity_types: Optional list of entity types. Auto-detected if None.
        """
        self._entity_types = entity_types if entity_types else []
        self._evaluator = SpanEvaluator()

    @property
    def name(self) -> str:
        return "ner_span"

    @property
    def label_space(self) -> list[str]:
        """Return entity types."""
        return self._entity_types

    def ensure_entity_types(self, entity_types: list[str]) -> None:
        """Add entity types to the task."""
        for et in entity_types:
            if et not in self._entity_types:
                self._entity_types.append(et)
        self._entity_types.sort()

    def compute_metrics(self, predictions: Any, references: Any) -> dict[str, float]:
        """Compute span-based NER metrics.

        Args:
            predictions: Predicted spans (document-level) or dict with IOB2 data
            references: Gold spans (document-level)

        Returns:
            Dictionary with exact_f1, partial_f1, seqeval_f1, and per-entity metrics
        """
        # Handle new format with IOB2 data for seqeval
        iob2_preds = None
        iob2_refs = None

        if isinstance(predictions, dict) and 'documents' in predictions:
            iob2_preds = predictions.get('iob2_predictions')
            iob2_refs = predictions.get('iob2_references')
            predictions = predictions['documents']

        # Compute span-based metrics (exact and partial match)
        results = self._evaluator.evaluate(predictions, references)

        # Compute seqeval F1 if IOB2 data is available
        if iob2_preds and iob2_refs:
            seqeval_metrics = self._compute_seqeval_metrics(iob2_preds, iob2_refs)
            results.update(seqeval_metrics)

        return results

    def _compute_seqeval_metrics(
        self, predictions: list[list[str]], references: list[list[str]]
    ) -> dict[str, float]:
        """Compute seqeval metrics for comparison with standard benchmarks.

        This uses the same evaluation methodology as CamemBERT-bio and DrBenchmark papers.

        Args:
            predictions: List of predicted IOB2 label sequences
            references: List of gold IOB2 label sequences

        Returns:
            Dictionary with seqeval_f1, seqeval_precision, seqeval_recall
        """
        try:
            from seqeval.metrics import f1_score, precision_score, recall_score
            from seqeval.scheme import IOB2

            # seqeval expects list of list of strings - use strict mode with IOB2 scheme
            f1 = f1_score(references, predictions, mode='strict', scheme=IOB2)
            precision = precision_score(references, predictions, mode='strict', scheme=IOB2)
            recall = recall_score(references, predictions, mode='strict', scheme=IOB2)

            return {
                'seqeval_f1': f1,
                'seqeval_precision': precision,
                'seqeval_recall': recall,
            }
        except ImportError:
            return {'seqeval_f1': 0.0, 'seqeval_precision': 0.0, 'seqeval_recall': 0.0}
        except Exception as e:
            print(f"Warning: seqeval computation failed: {e}")
            return {'seqeval_f1': 0.0, 'seqeval_precision': 0.0, 'seqeval_recall': 0.0}

    def format_output(self, predictions: Any) -> Any:
        """Format predictions for output."""
        return predictions

