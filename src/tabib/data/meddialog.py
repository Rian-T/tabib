"""MedDialog-FR Women dataset adapter for multilabel classification."""

from __future__ import annotations

import ast
import csv
import os
from pathlib import Path
from typing import Any

from datasets import Dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.classification import ClassificationTask


# Support $SCRATCH/tabib/data/ paths for HuggingFace datasets (offline mode on compute nodes)
SCRATCH = os.environ.get("SCRATCH", "")
SCRATCH_DATA = Path(SCRATCH) / "tabib" / "data" if SCRATCH else None

# HuggingFace dataset path in SCRATCH
MEDDIALOG_HF_DIR = SCRATCH_DATA / "rntc--tabib-meddialog-fr" if SCRATCH_DATA else None

# Fallback to local PROJECT_ROOT paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "MedDialog-FR" / "MedDialog-FR-women"

# Select directory based on availability
if MEDDIALOG_HF_DIR and MEDDIALOG_HF_DIR.exists():
    DATA_DIR = MEDDIALOG_HF_DIR
else:
    DATA_DIR = LOCAL_DATA_DIR


def _load_hf_parquet_splits(data_dir: Path) -> dict[str, Dataset] | None:
    """Load splits from HuggingFace parquet files if available."""
    train_path = data_dir / "data" / "train-00000-of-00001.parquet"
    val_path = data_dir / "data" / "val-00000-of-00001.parquet"
    test_path = data_dir / "data" / "test-00000-of-00001.parquet"

    if not all(p.exists() for p in [train_path, val_path, test_path]):
        return None

    import pandas as pd
    return {
        "train": Dataset.from_pandas(pd.read_parquet(train_path)),
        "val": Dataset.from_pandas(pd.read_parquet(val_path)),
        "test": Dataset.from_pandas(pd.read_parquet(test_path)),
    }


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

    def _load_from_hf_parquet(self) -> dict[str, Dataset] | None:
        """Try loading from HuggingFace parquet files in SCRATCH."""
        if MEDDIALOG_HF_DIR is None or not MEDDIALOG_HF_DIR.exists():
            return None

        splits = _load_hf_parquet_splits(MEDDIALOG_HF_DIR)
        if splits is None:
            return None

        # Extract label vocab from all splits
        all_labels: set[str] = set()
        for split in splits.values():
            all_labels.update(split["label"])
        self._label_vocab = sorted(all_labels)

        return splits

    def load_splits(self) -> dict[str, Dataset]:
        # Try HuggingFace parquet first (pre-processed data)
        hf_splits = self._load_from_hf_parquet()
        if hf_splits is not None:
            return hf_splits

        # Fallback to original CSV-based loading
        multilabel_path = DATA_DIR / "multilabel_annotation" / "dataset_multilabel_meddialog_22labels.csv"
        text_path = DATA_DIR / "post-editing" / "meddialog-fr-women_post-editing.csv"

        if not multilabel_path.exists():
            raise FileNotFoundError(
                f"MedDialog multilabel data not found at {multilabel_path}. "
                "Download with: huggingface-cli download rntc/tabib-meddialog-fr "
                "--local-dir $SCRATCH/tabib/data/rntc--tabib-meddialog-fr --repo-type dataset"
            )
        if not text_path.exists():
            raise FileNotFoundError(
                f"MedDialog text data not found at {text_path}. "
                "Download with: huggingface-cli download rntc/tabib-meddialog-fr "
                "--local-dir $SCRATCH/tabib/data/rntc--tabib-meddialog-fr --repo-type dataset"
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
