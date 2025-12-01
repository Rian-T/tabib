"""MedDialog-FR Women dataset adapter for multilabel classification."""

from __future__ import annotations

import ast
import csv
from pathlib import Path
from typing import Any

from datasets import Dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.classification import ClassificationTask


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "MedDialog-FR" / "MedDialog-FR-women"


class MedDialogWomenAdapter(DatasetAdapter):
    """Adapter for MedDialog-FR Women multilabel classification.

    Dataset has 900 medical Q&A samples with 22 UMLS CUI labels.
    Labels are treated as combo-as-single-class (each unique combination
    of labels becomes a distinct class).
    """

    def __init__(self) -> None:
        self._label_vocab: list[str] | None = None

    @property
    def name(self) -> str:
        return "meddialog_women"

    def load_splits(self) -> dict[str, Dataset]:
        multilabel_path = DATA_DIR / "multilabel_annotation" / "dataset_multilabel_meddialog_22labels.csv"
        text_path = DATA_DIR / "post-editing" / "meddialog-fr-women_post-editing.csv"

        if not multilabel_path.exists():
            raise FileNotFoundError(
                f"MedDialog multilabel data not found at {multilabel_path}. "
                "Extract meddialog_fr.zip to data/ folder."
            )
        if not text_path.exists():
            raise FileNotFoundError(
                f"MedDialog text data not found at {text_path}. "
                "Extract meddialog_fr.zip to data/ folder."
            )

        # Load text data (Q&A pairs)
        text_data: dict[str, str] = {}
        with text_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text_id = row["id"]
                # Use post-edited text (cleaner than machine translation)
                text = row.get("post-edited", row.get("machine_translation", "")).strip()
                text_data[text_id] = text

        # Load multilabel annotations and match with text
        splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        all_label_combos: set[str] = set()

        with multilabel_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                q_id = row["id"]  # e.g., "hm3_67574_q"
                split = row["split"]
                labels_str = row["labels"]

                # Parse labels list
                try:
                    labels = ast.literal_eval(labels_str)
                except (ValueError, SyntaxError):
                    continue

                if not labels:
                    continue

                # Get base ID (without _q suffix)
                base_id = q_id[:-2] if q_id.endswith("_q") else q_id
                a_id = f"{base_id}_a"

                # Get question and answer text
                q_text = text_data.get(q_id, "")
                a_text = text_data.get(a_id, "")

                if not q_text:
                    continue

                # Concatenate Q+A
                if a_text:
                    text = f"Question: {q_text}\n\nRéponse: {a_text}"
                else:
                    text = f"Question: {q_text}"

                # Create label combo (sorted for consistency)
                label_combo = " ".join(sorted(labels))
                all_label_combos.add(label_combo)

                # Map split names
                split_name = "val" if split == "dev" else split
                splits[split_name].append({
                    "text": text,
                    "label": label_combo,
                    "labels_raw": labels,  # Keep original for reference
                })

        # Convert to Dataset
        result = {}
        for split_name, records in splits.items():
            if records:
                result[split_name] = Dataset.from_list(records)

        self._label_vocab = sorted(all_label_combos)
        return result

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, ClassificationTask):
            raise ValueError(
                f"MedDialogWomen expects ClassificationTask, got {type(task)}"
            )

        if self._label_vocab:
            task.ensure_labels(self._label_vocab)

        label_map = task.label_space

        def map_example(example: dict[str, Any]) -> dict[str, Any]:
            label_id = label_map.get(example["label"], 0)
            return {
                "text": example["text"],
                "labels": label_id,
                "label_text": example["label"],  # Keep for few-shot examples
            }

        processed = dataset.map(map_example)
        processed.set_format(type="python")
        return processed
