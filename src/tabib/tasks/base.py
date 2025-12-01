"""Base task abstraction.

A Task defines the label space, metrics, and I/O format.
"""

from abc import ABC, abstractmethod
from typing import Any


class Task(ABC):
    """Base class for all tasks.
    
    A task defines:
    - Label space (what labels are valid)
    - Metrics (how to evaluate)
    - I/O format (how data flows in/out)
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the task name."""
        pass
    
    @property
    @abstractmethod
    def label_space(self) -> Any:
        """Return the label space definition."""
        pass
    
    @abstractmethod
    def compute_metrics(self, predictions: Any, references: Any) -> dict[str, float]:
        """Compute task-specific metrics.
        
        Args:
            predictions: Model predictions
            references: Ground truth labels
            
        Returns:
            Dictionary of metric names to values
        """
        pass
    
    @abstractmethod
    def format_output(self, predictions: Any) -> Any:
        """Format predictions for output.
        
        Args:
            predictions: Raw model predictions
            
        Returns:
            Formatted predictions
        """
        pass

