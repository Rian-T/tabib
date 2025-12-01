"""Base evaluator abstraction."""

from abc import ABC, abstractmethod
from typing import Any


class Evaluator(ABC):
    """Base class for evaluators.
    
    Evaluators compute metrics by comparing predictions to references.
    """
    
    @abstractmethod
    def evaluate(
        self, predictions: Any, references: Any, **kwargs: Any
    ) -> dict[str, float]:
        """Evaluate predictions against references.
        
        Args:
            predictions: Model predictions
            references: Ground truth references
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of metric names to values
        """
        pass

