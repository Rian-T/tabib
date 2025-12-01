"""Span-based NER task with character offsets and nested entity support."""

from typing import Any

from tabib.evaluation.span_evaluator import SpanEvaluator
from tabib.tasks.base import Task


class NERSpanTask(Task):
    """Span-based NER task using character offsets.
    
    Supports nested entities and evaluates with exact/partial match F1.
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
            predictions: Predicted spans (document-level)
            references: Gold spans (document-level)
            
        Returns:
            Dictionary with exact_f1, partial_f1, and per-entity metrics
        """
        return self._evaluator.evaluate(predictions, references)
    
    def format_output(self, predictions: Any) -> Any:
        """Format predictions for output."""
        return predictions

