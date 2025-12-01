"""Adapters for the MediQAl dataset family."""

from __future__ import annotations

from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.mcqa import MultipleChoiceTask
from tabib.tasks.open_qa import OpenQATask

_CHOICE_ORDER = ("a", "b", "c", "d", "e")


def _normalize_combo(value: str) -> str:
    letters = [
        token.strip().upper()
        for token in value.split(",")
        if token.strip()
    ]
    return " ".join(sorted(letters))


def _build_mcq_text(example: dict[str, Any]) -> str:
    parts: list[str] = []
    clinical_case = example.get("clinical_case")
    if clinical_case:
        parts.append(str(clinical_case).strip())

    question = example.get("question")
    if question:
        parts.append(str(question).strip())

    base_text = "\n\n".join(part for part in parts if part)

    option_lines = []
    for letter in _CHOICE_ORDER:
        answer_text = example.get(f"answer_{letter}")
        if answer_text:
            option_lines.append(f"({letter.upper()}) {str(answer_text).strip()}")

    if option_lines:
        if base_text:
            return f"{base_text}\n" + "\n".join(option_lines)
        return "\n".join(option_lines)

    return base_text


class _BaseMediQAlMCQAdapter(DatasetAdapter):
    config_name: str

    def __init__(self) -> None:
        self._label_list: list[str] | None = None

    @property
    def name(self) -> str:
        return f"mediqal_{self.config_name}"

    def load_splits(self) -> dict[str, Dataset]:
        dataset_dict: DatasetDict = load_dataset("ANR-MALADES/MediQAl", self.config_name)
        label_set: set[str] = set()
        for split in dataset_dict.values():
            label_set.update(self._iter_split_labels(split))
        self._label_list = sorted(label_set)
        return dict(dataset_dict)

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, MultipleChoiceTask):
            raise ValueError(f"Expected MultipleChoiceTask, got {type(task)}")

        if not self._label_list:
            self._label_list = sorted(self._iter_split_labels(dataset))

        task.set_label_list(self._label_list)
        label_to_id = task.label_space

        def prepare(example: dict[str, Any]) -> dict[str, Any]:
            combo = _normalize_combo(example.get("correct_answers", ""))
            text = _build_mcq_text(example)
            return {
                "text": text,
                "labels": label_to_id[combo],
                "label_text": combo,  # Keep text label for few-shot examples
            }

        processed = dataset.map(prepare, remove_columns=dataset.column_names)
        processed.set_format(type="python")
        return processed

    def _iter_split_labels(self, split: Dataset) -> Iterable[str]:
        for example in split:
            yield _normalize_combo(example.get("correct_answers", ""))


class MediQAlMCQUAdapter(_BaseMediQAlMCQAdapter):
    """Adapter for the MediQAl MCQU (single-answer multiple-choice) split."""

    config_name = "mcqu"


class MediQAlMCQMAdapter(_BaseMediQAlMCQAdapter):
    """Adapter for the MediQAl MCQM (multi-answer multiple-choice) split."""

    config_name = "mcqm"


class MediQAlOEQAdapter(DatasetAdapter):
    """Adapter for the MediQAl OEQ (open-ended questions) split."""

    def __init__(self) -> None:
        self.config_name = "oeq"

    @property
    def name(self) -> str:
        return "mediqal_oeq"

    def load_splits(self) -> dict[str, Dataset]:
        dataset_dict: DatasetDict = load_dataset("ANR-MALADES/MediQAl", self.config_name)
        return dict(dataset_dict)

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, OpenQATask):
            raise ValueError(f"Expected OpenQATask, got {type(task)}")

        def prepare(example: dict[str, Any]) -> dict[str, Any]:
            parts = []
            clinical_case = example.get("clinical_case")
            if clinical_case:
                parts.append(str(clinical_case).strip())
            question_number = example.get("cc_question_number")
            question_text = example.get("question")
            if question_number:
                parts.append(f"Question {question_number}: {str(question_text).strip()}")
            elif question_text:
                parts.append(str(question_text).strip())

            text = "\n\n".join(parts)
            answer = str(example.get("answer", "")).strip()
            return {
                "text": text,
                "labels": answer,
            }

        processed = dataset.map(prepare, remove_columns=dataset.column_names)
        processed.set_format(type="python")
        return processed

