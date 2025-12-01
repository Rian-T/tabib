"""Base dataset adapter abstraction.

A DatasetAdapter loads splits and preprocesses data to match the task.
"""

from abc import ABC, abstractmethod
from typing import Any


class DatasetAdapter(ABC):
    """Base class for all dataset adapters.
    
    A dataset adapter:
    - Loads train/val/test splits
    - Preprocesses data to match the task format
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the dataset name."""
        pass
    
    @abstractmethod
    def load_splits(self) -> dict[str, Any]:
        """Load dataset splits.
        
        Returns:
            Dictionary mapping split names (e.g., 'train', 'val', 'test') 
            to datasets
        """
        pass
    
    @abstractmethod
    def preprocess(self, dataset: Any, task: Any) -> Any:
        """Preprocess dataset to match task format.
        
        Args:
            dataset: Raw dataset split
            task: Task instance to match format against
            
        Returns:
            Preprocessed dataset
        """
        pass

