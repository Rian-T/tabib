"""Base preprocessor abstraction."""

from abc import ABC, abstractmethod
from typing import Any


class Preprocessor(ABC):
    """Base class for dataset preprocessors.
    
    Preprocessors transform datasets before model processing,
    e.g., chunking long documents to fit model context windows.
    """
    
    @abstractmethod
    def preprocess(self, dataset: Any, max_length: int) -> Any:
        """Preprocess dataset.
        
        Args:
            dataset: Input dataset
            max_length: Maximum sequence length for model
            
        Returns:
            Preprocessed dataset with metadata for reconstruction
        """
        pass

