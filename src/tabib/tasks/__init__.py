"""Task abstractions."""

from tabib.tasks.base import Task
from tabib.tasks.classification import ClassificationTask
from tabib.tasks.mcqa import MultipleChoiceTask
from tabib.tasks.open_qa import OpenQATask
from tabib.tasks.ner_token import NERTokenTask
from tabib.tasks.ner_span import NERSpanTask
from tabib.tasks.similarity import SimilarityTask

__all__ = [
    "Task",
    "NERTokenTask",
    "NERSpanTask",
    "ClassificationTask",
    "SimilarityTask",
    "MultipleChoiceTask",
    "OpenQATask",
]

