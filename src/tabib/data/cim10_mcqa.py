"""Adapter for the CIM10-MCQA-FR dataset (ICD-10 coding questions)."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.mcqa import MultipleChoiceTask

_CHOICE_ORDER = ("A", "B", "C", "D")


class CIM10MCQAAdapter(DatasetAdapter):
    """Dataset adapter for rntc/cim10-mcqa-fr.

    This dataset contains 160K French multiple-choice questions about ICD-10
    diagnostic coding. Each question has 4 choices (A, B, C, D) with a single
    correct answer.
    """

    def __init__(self, max_samples: int | None = None) -> None:
        """Initialize the adapter.

        Args:
            max_samples: Optional limit on number of samples to use.
                        If None, uses the full dataset.
        """
        self.max_samples = max_samples
        self._label_list: list[str] = list(_CHOICE_ORDER)

    @property
    def name(self) -> str:
        return "cim10_mcqa"

    def load_splits(self) -> dict[str, Dataset]:
        dataset = load_dataset("rntc/cim10-mcqa-fr", split="train")

        if self.max_samples is not None:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))

        # Split into train/val/test (80/10/10)
        total = len(dataset)
        train_size = int(0.8 * total)
        val_size = int(0.1 * total)

        train_ds = dataset.select(range(train_size))
        val_ds = dataset.select(range(train_size, train_size + val_size))
        test_ds = dataset.select(range(train_size + val_size, total))

        return {
            "train": train_ds,
            "val": val_ds,
            "test": test_ds,
        }

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, MultipleChoiceTask):
            raise ValueError(f"Expected MultipleChoiceTask, got {type(task)}")

        task.set_label_list(self._label_list)
        label_to_id = task.label_space

        def prepare(example: dict[str, Any]) -> dict[str, Any]:
            # Build text with question and choices
            question = example["question"].strip()
            choices = example["choices"]

            option_lines = []
            for letter in _CHOICE_ORDER:
                choice_text = choices.get(letter, "")
                if choice_text:
                    # Remove leading letter if present (e.g., "A. ", "A) ", "A " prefix)
                    text = choice_text.strip()
                    # Handle various prefixes: "A. ", "A) ", "A "
                    for prefix in [f"{letter}. ", f"{letter}) ", f"{letter} "]:
                        if text.startswith(prefix):
                            text = text[len(prefix):].strip()
                            break
                    option_lines.append(f"({letter}) {text}")

            text = f"{question}\n" + "\n".join(option_lines)
            correct = example["correct_answer"].strip().upper()

            return {
                "text": text,
                "labels": label_to_id[correct],
                "label_text": correct,
            }

        processed = dataset.map(prepare, remove_columns=dataset.column_names)
        processed.set_format(type="python")
        return processed
