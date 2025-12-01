"""JNLPBA NER dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.ner_token import NERTokenTask


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "jnlpba"


def _read_iob2(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JNLPBA file not found: {path}")

    sentences: list[dict[str, Any]] = []
    tokens: list[str] = []
    tags: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                if tokens:
                    sentences.append(
                        {
                            "id": str(len(sentences)),
                            "tokens": tokens,
                            "ner_tags": tags,
                        }
                    )
                    tokens = []
                    tags = []
                continue

            try:
                token, tag = stripped.split("\t")
            except ValueError:
                parts = stripped.split()
                if len(parts) < 2:
                    raise ValueError(f"Invalid line in {path}: {line!r}") from None
                token = parts[0]
                tag = parts[-1]
            tokens.append(token)
            tags.append(tag)

    if tokens:
        sentences.append(
            {
                "id": str(len(sentences)),
                "tokens": tokens,
                "ner_tags": tags,
            }
        )

    return sentences


class JNLPBAAdapter(DatasetAdapter):
    """Adapter for the JNLPBA NER dataset."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR

    @property
    def name(self) -> str:
        return "jnlpba"

    def load_splits(self) -> dict[str, Dataset]:
        train_path = self.data_dir / "train.iob2"
        val_path = self.data_dir / "validation.iob2"

        train_examples = _read_iob2(train_path)
        val_examples = _read_iob2(val_path)

        datasets: dict[str, Dataset] = {
            "train": Dataset.from_list(train_examples),
            "val": Dataset.from_list(val_examples),
            "test": Dataset.from_list(val_examples),
        }
        return datasets

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, NERTokenTask):
            raise ValueError(f"Expected NERTokenTask, got {type(task)}")

        unique_tags = set()
        for tags in dataset["ner_tags"]:
            unique_tags.update(tags)

        ordered_labels = ["O"] + sorted(label for label in unique_tags if label != "O")
        task.ensure_labels(ordered_labels)

        def map_example(example: dict[str, Any]) -> dict[str, Any]:
            label_ids = [task.label_to_id(tag) for tag in example["ner_tags"]]
            mapped = {
                "tokens": example["tokens"],
                "labels": label_ids,
            }
            if "id" in example:
                mapped["id"] = example["id"]
            return mapped

        processed = dataset.map(
            map_example,
            remove_columns=[col for col in dataset.column_names if col not in {"tokens", "ner_tags", "id"}],
        )
        if "ner_tags" in processed.column_names:
            processed = processed.remove_columns("ner_tags")
        processed.set_format(type="python")
        return processed


