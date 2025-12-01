"""Adapter for the FrenchMedMCQA-extended dataset."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.classification import ClassificationTask

_CHOICE_ORDER = ("a", "b", "c", "d", "e")


class FrenchMedMCQAExtendedAdapter(DatasetAdapter):
    """Dataset adapter for FrenchMedMCQA-extended.

    Each example is converted into a single text string composed of the question
    followed by the possible answers, while the label encodes the set of correct
    options as a whitespace-separated string (e.g. ``\"a c\"``). The adapter
    aggregates the full label space across splits to guarantee a consistent
    mapping between string combinations and numeric identifiers.
    """

    def __init__(self, data_files: dict[str, str] | None = None) -> None:
        if data_files is None:
            data_files = {
                "train": "hf://datasets/uy-rrodriguez/FrenchMedMCQA-extended/train.json.gz",
                "val": "hf://datasets/uy-rrodriguez/FrenchMedMCQA-extended/dev.json.gz",
                "test": "hf://datasets/uy-rrodriguez/FrenchMedMCQA-extended/test.json.gz",
            }
        self.data_files = data_files
        self._label_list: list[str] | None = None

    @property
    def name(self) -> str:
        return "french_med_mcqa_extended"

    def load_splits(self) -> dict[str, Dataset]:
        dataset_dict: DatasetDict = load_dataset("json", data_files=self.data_files)
        # Compute the global label list once to keep encoding consistent.
        label_set: set[str] = set()
        for split in dataset_dict.values():
            for example in split:
                combo = " ".join(
                    sorted(answer.upper() for answer in example["correct_answers"])
                )
                label_set.add(combo)
        self._label_list = sorted(label_set)
        return {split: dataset_dict[split] for split in dataset_dict}

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, ClassificationTask):
            raise ValueError(f"Expected ClassificationTask-compatible task, got {type(task)}")

        if not self._label_list:
            # As a safeguard, recompute the labels from the current split.
            label_set = {
                " ".join(sorted(answer.upper() for answer in example["correct_answers"]))
                for example in dataset
            }
            self._label_list = sorted(label_set)

        task.set_label_list(self._label_list)
        label_to_id = task.label_space

        def format_example(example: dict[str, Any]) -> dict[str, Any]:
            answers: dict[str, str] = example["answers"]
            option_lines = []
            for key in _CHOICE_ORDER:
                answer_text = answers.get(key)
                if answer_text:
                    option_lines.append(f"({key.upper()}) {answer_text}")
            question_text = example["question"].strip()
            text = question_text
            if option_lines:
                text = f"{question_text}\n" + "\n".join(option_lines)

            combo = " ".join(sorted(answer.upper() for answer in example["correct_answers"]))
            return {
                "text": text,
                "labels": label_to_id[combo],
                "label_text": combo,  # Keep text label for few-shot examples
            }

        processed = dataset.map(
            format_example,
            remove_columns=dataset.column_names,
        )
        processed.set_format(type="python")
        return processed

