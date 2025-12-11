"""Mixed MCQA adapter combining MedIQAL and FrenchMedMCQA datasets."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, concatenate_datasets

from tabib.data.base import DatasetAdapter
from tabib.data.mediqal import MediQAlMCQMAdapter, MediQAlMCQUAdapter
from tabib.data.french_med_mcqa_extended import FrenchMedMCQAExtendedAdapter
from tabib.tasks.classification import ClassificationTask


class MixedMCQAAdapter(DatasetAdapter):
    """Combined adapter for MedIQAL (MCQM + MCQU) and FrenchMedMCQA-extended.

    Concatenates training splits from all three datasets and builds
    a unified label space for consistent training.
    """

    def __init__(self) -> None:
        self._label_list: list[str] | None = None
        self._mcqm = MediQAlMCQMAdapter()
        self._mcqu = MediQAlMCQUAdapter()
        self._frenchmed = FrenchMedMCQAExtendedAdapter()

    @property
    def name(self) -> str:
        return "mixed_mcqa"

    def load_splits(self) -> dict[str, Dataset]:
        # Load all source datasets
        mcqm_splits = self._mcqm.load_splits()
        mcqu_splits = self._mcqu.load_splits()
        frenchmed_splits = self._frenchmed.load_splits()

        # Build unified label space from all datasets
        label_set: set[str] = set()
        if self._mcqm._label_list:
            label_set.update(self._mcqm._label_list)
        if self._mcqu._label_list:
            label_set.update(self._mcqu._label_list)
        if self._frenchmed._label_list:
            label_set.update(self._frenchmed._label_list)
        self._label_list = sorted(label_set)

        # Normalize each dataset to common format before concatenating
        def normalize_split(ds: Dataset, source: str) -> Dataset:
            """Convert dataset to common text/label_text format."""
            def format_example(example: dict) -> dict:
                # Handle correct_answers
                correct = example.get("correct_answers", "")
                if isinstance(correct, list):
                    combo = " ".join(sorted(a.upper() for a in correct))
                else:
                    letters = [t.strip().upper() for t in str(correct).split(",") if t.strip()]
                    combo = " ".join(sorted(letters))

                text = self._build_text(example)
                return {"text": text, "label_text": combo, "source": source}

            return ds.map(format_example, remove_columns=ds.column_names)

        # Combine train splits
        train_datasets = []
        for splits, adapter_name in [
            (mcqm_splits, "mcqm"),
            (mcqu_splits, "mcqu"),
            (frenchmed_splits, "frenchmed"),
        ]:
            if "train" in splits:
                train_datasets.append(normalize_split(splits["train"], adapter_name))

        combined_train = concatenate_datasets(train_datasets) if train_datasets else None

        # Combine test/val splits
        test_datasets = []
        for splits, name in [
            (mcqm_splits, "mcqm"),
            (mcqu_splits, "mcqu"),
            (frenchmed_splits, "frenchmed"),
        ]:
            if "test" in splits:
                test_datasets.append(normalize_split(splits["test"], name))
            elif "val" in splits:
                test_datasets.append(normalize_split(splits["val"], name))

        combined_test = concatenate_datasets(test_datasets) if test_datasets else None

        result = {}
        if combined_train is not None:
            result["train"] = combined_train
        if combined_test is not None:
            result["test"] = combined_test

        return result

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, ClassificationTask):
            raise ValueError(f"Expected ClassificationTask, got {type(task)}")

        if not self._label_list:
            raise ValueError("Label list not set. Call load_splits first.")

        task.set_label_list(self._label_list)
        label_to_id = task.label_space

        def format_example(example: dict[str, Any]) -> dict[str, Any]:
            # Data is already normalized in load_splits with text/label_text
            text = example.get("text", "")
            label_text = example.get("label_text", "")

            return {
                "text": text,
                "labels": label_to_id.get(label_text, 0),
                "label_text": label_text,
            }

        processed = dataset.map(format_example, remove_columns=dataset.column_names)
        processed.set_format(type="python")
        return processed

    def _build_text(self, example: dict[str, Any]) -> str:
        """Build text from example, handling both dataset formats."""
        parts = []

        # Clinical case (MedIQAL only)
        clinical_case = example.get("clinical_case")
        if clinical_case:
            parts.append(str(clinical_case).strip())

        # Question
        question = example.get("question")
        if question:
            parts.append(str(question).strip())

        base_text = "\n\n".join(p for p in parts if p)

        # Build options
        option_lines = []
        choice_order = ("a", "b", "c", "d", "e")

        # MedIQAL format: answer_a, answer_b, etc.
        for letter in choice_order:
            answer_text = example.get(f"answer_{letter}")
            if answer_text:
                option_lines.append(f"({letter.upper()}) {str(answer_text).strip()}")

        # FrenchMed format: answers dict
        answers_dict = example.get("answers")
        if answers_dict and isinstance(answers_dict, dict):
            option_lines = []  # Reset if we have this format
            for letter in choice_order:
                answer_text = answers_dict.get(letter)
                if answer_text:
                    option_lines.append(f"({letter.upper()}) {answer_text}")

        if option_lines:
            if base_text:
                return f"{base_text}\n" + "\n".join(option_lines)
            return "\n".join(option_lines)

        return base_text
