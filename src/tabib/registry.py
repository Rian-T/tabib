"""Registry for tasks, datasets, and models."""

from typing import Type

from tabib.data.base import DatasetAdapter
from tabib.models.base import ModelAdapter
from tabib.tasks.base import Task


# Registries
_tasks: dict[str, Type[Task]] = {}
_datasets: dict[str, Type[DatasetAdapter]] = {}
_models: dict[str, Type[ModelAdapter]] = {}


def register_task(name: str, task_class: Type[Task]) -> None:
    """Register a task class.
    
    Args:
        name: Task name
        task_class: Task class
    """
    _tasks[name] = task_class


def register_dataset(name: str, dataset_class: Type[DatasetAdapter]) -> None:
    """Register a dataset adapter class.
    
    Args:
        name: Dataset name
        dataset_class: DatasetAdapter class
    """
    _datasets[name] = dataset_class


def register_model(name: str, model_class: Type[ModelAdapter]) -> None:
    """Register a model adapter class.
    
    Args:
        name: Model name
        model_class: ModelAdapter class
    """
    _models[name] = model_class


def get_task(name: str) -> Type[Task]:
    """Get a task class by name.
    
    Args:
        name: Task name
        
    Returns:
        Task class
        
    Raises:
        KeyError: If task not found
    """
    if name not in _tasks:
        raise KeyError(f"Task '{name}' not found. Available: {list(_tasks.keys())}")
    return _tasks[name]


def get_dataset(name: str) -> Type[DatasetAdapter]:
    """Get a dataset adapter class by name.
    
    Args:
        name: Dataset name
        
    Returns:
        DatasetAdapter class
        
    Raises:
        KeyError: If dataset not found
    """
    if name not in _datasets:
        raise KeyError(f"Dataset '{name}' not found. Available: {list(_datasets.keys())}")
    return _datasets[name]


def get_model(name: str) -> Type[ModelAdapter]:
    """Get a model adapter class by name.
    
    Args:
        name: Model name
        
    Returns:
        ModelAdapter class
        
    Raises:
        KeyError: If model not found
    """
    if name not in _models:
        raise KeyError(f"Model '{name}' not found. Available: {list(_models.keys())}")
    return _models[name]

