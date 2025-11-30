"""CLISTER dataset adapter for French clinical semantic similarity."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.similarity import SimilarityTask


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "drbenchmark" / "data" / "drbenchmark" / "clister"


class CLISTERAdapter(DatasetAdapter):
    """Adapter for CLISTER French clinical semantic similarity dataset."""

    @property
    def name(self) -> str:
        return "clister"

    def load_splits(self) -> dict[str, Dataset]:
        if not DATA_DIR.exists():
            raise FileNotFoundError(
                f"CLISTER data not found. Expected at {DATA_DIR}. "
                "Download from DrBenchmark/CLISTER on HuggingFace."
            )

        # Load train data
        train_csv = DATA_DIR / "train.csv"
        train_json = DATA_DIR / "id_to_sentence_train.json"
        test_csv = DATA_DIR / "test.csv"
        test_json = DATA_DIR / "id_to_sentence_test.json"

        if not all(f.exists() for f in [train_csv, train_json, test_csv, test_json]):
            raise FileNotFoundError("CLISTER data files missing")

        # Load sentence mappings
        with train_json.open(encoding="utf-8") as f:
            train_map = json.load(f)
        with test_json.open(encoding="utf-8") as f:
            test_map = json.load(f)

        # Parse train CSV
        all_train = self._parse_pairs(train_csv, train_map)

        # Split train/val (83.33%/16.67% as per original script)
        ids = list(range(len(all_train)))
        random.seed(4)  # Match original script's seed
        random.shuffle(ids)
        random.shuffle(ids)
        random.shuffle(ids)

        split_idx = int(len(ids) * 0.8333)
        train_ids = set(ids[:split_idx])
        val_ids = set(ids[split_idx:])

        train_records = [all_train[i] for i in train_ids]
        val_records = [all_train[i] for i in val_ids]

        # Parse test CSV
        test_records = self._parse_pairs(test_csv, test_map)

        return {
            "train": Dataset.from_list(train_records),
            "val": Dataset.from_list(val_records),
            "test": Dataset.from_list(test_records),
        }

    def _parse_pairs(
        self, csv_path: Path, id_map: dict[str, str]
    ) -> list[dict[str, Any]]:
        """Parse sentence pairs from CSV file."""
        records = []
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                id_1 = row["id_1"]
                id_2 = row["id_2"]
                sim = float(row["sim"])

                # Extract base document IDs (remove sentence index suffix)
                doc_1 = "_".join(id_1.split("_")[0:2])
                doc_2 = "_".join(id_2.split("_")[0:2])

                text_1 = id_map.get(doc_1, "")
                text_2 = id_map.get(doc_2, "")

                if text_1 and text_2:
                    records.append({
                        "text_1": text_1,
                        "text_2": text_2,
                        "similarity": sim,
                    })
        return records

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, SimilarityTask):
            raise ValueError(f"CLISTER expects SimilarityTask, got {type(task)}")

        def map_example(example: dict[str, Any]) -> dict[str, Any]:
            return {
                "text_left": example["text_1"],
                "text_right": example["text_2"],
                "labels": example["similarity"],
            }

        processed = dataset.map(map_example, remove_columns=dataset.column_names)
        processed.set_format(type="python")
        return processed
