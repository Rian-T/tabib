"""MORFITT dataset adapter for medical speciality classification."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from datasets import Dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.classification import ClassificationTask


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "drbenchmark" / "data" / "drbenchmark" / "clister_morfitt"


class MORFITTAdapter(DatasetAdapter):
    """Adapter for MORFITT medical speciality classification.

    Note: Original dataset is multi-label, but we use single-label (primary
    specialty) for compatibility with standard classification task.
    """

    def __init__(self) -> None:
        self._label_vocab: list[str] | None = None

    @property
    def name(self) -> str:
        return "morfitt"

    def load_splits(self) -> dict[str, Dataset]:
        if not DATA_DIR.exists():
            raise FileNotFoundError(
                f"MORFITT data not found. Expected at {DATA_DIR}. "
                "Download from DrBenchmark/MORFITT on HuggingFace."
            )

        splits: dict[str, Dataset] = {}
        all_labels: set[str] = set()

        for split_name, filename in [("train", "train.tsv"), ("val", "dev.tsv"), ("test", "test.tsv")]:
            filepath = DATA_DIR / filename
            if not filepath.exists():
                continue

            records = []
            with filepath.open(encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    text = row.get("abstract", "").strip()
                    labels_str = row.get("specialities", "").strip()

                    if not text or not labels_str:
                        continue

                    # Use first (primary) label for single-label classification
                    labels = [l.strip() for l in labels_str.split("|") if l.strip()]
                    if not labels:
                        continue

                    primary_label = labels[0]
                    all_labels.add(primary_label)

                    records.append({
                        "text": text,
                        "label": primary_label,
                    })

            if records:
                splits[split_name] = Dataset.from_list(records)

        # Sort labels for consistency
        self._label_vocab = sorted(all_labels)
        return splits

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, ClassificationTask):
            raise ValueError(
                f"MORFITT expects ClassificationTask, got {type(task)}"
            )

        if self._label_vocab:
            task.ensure_labels(self._label_vocab)

        label_map = task.label_space

        def map_example(example: dict[str, Any]) -> dict[str, Any]:
            label_id = label_map.get(example["label"], 0)
            return {
                "text": example["text"],
                "labels": label_id,
            }

        processed = dataset.map(map_example)
        processed.set_format(type="python")
        return processed
