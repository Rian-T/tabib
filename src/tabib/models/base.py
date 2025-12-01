"""Base model adapter abstraction.

A ModelAdapter builds the model and exposes training/inference interfaces.
"""

from abc import ABC, abstractmethod
from typing import Any

from transformers import Trainer


class ModelAdapter(ABC):
    """Base class for all model adapters.
    
    A model adapter:
    - Builds the model
    - Exposes `supports_finetune` property
    - Returns a `Trainer` if trainable, or a `predict` method for inference
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name."""
        pass
    
    @property
    @abstractmethod
    def supports_finetune(self) -> bool:
        """Whether this model supports fine-tuning."""
        pass
    
    @abstractmethod
    def build_model(self, task: Any, **kwargs: Any) -> Any:
        """Build the model.
        
        Args:
            task: Task instance
            **kwargs: Additional model configuration
            
        Returns:
            Model instance
        """
        pass
    
    @abstractmethod
    def get_trainer(
        self, 
        model: Any, 
        train_dataset: Any, 
        eval_dataset: Any | None = None,
        **kwargs: Any
    ) -> Trainer | None:
        """Get a Trainer instance for fine-tuning.
        
        Args:
            model: Model instance
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            **kwargs: Additional training configuration
            
        Returns:
            Trainer instance if trainable, None otherwise
        """
        pass
    
    @abstractmethod
    def predict(self, model: Any, inputs: Any, **kwargs: Any) -> Any:
        """Run inference.
        
        Args:
            model: Model instance
            inputs: Input data
            **kwargs: Additional inference configuration
            
        Returns:
            Model predictions
        """
        pass

