"""WikiANN NER dataset adapter."""

from typing import Any

from datasets import DatasetDict, load_dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.ner_token import NERTokenTask


class WikiAnnAdapter(DatasetAdapter):
    """Adapter for the WikiANN (PAN-X) multilingual NER dataset."""

    _language: str = "en"

    @property
    def name(self) -> str:
        """Return the dataset name."""
        return "wikiann_en"

    def load_splits(self) -> dict[str, Any]:
        """Load WikiANN dataset splits for the configured language."""
        dataset: DatasetDict = load_dataset("wikiann", self._language)

        splits: dict[str, Any] = {"train": dataset["train"], "test": dataset["test"]}
        if "validation" in dataset:
            splits["val"] = dataset["validation"]
        elif "dev" in dataset:
            splits["val"] = dataset["dev"]

        return splits

    def preprocess(self, dataset: Any, task: Any) -> Any:
        """Preprocess dataset for the NER token task."""
        if not isinstance(task, NERTokenTask):
            raise ValueError(f"Expected NERTokenTask, got {type(task)}")

        if "ner_tags" not in dataset.features:
            raise ValueError("WikiANN dataset must contain 'ner_tags' field")

        label_names = dataset.features["ner_tags"].feature.names
        task.ensure_labels(label_names)
        label_map = {idx: task.label_to_id(label) for idx, label in enumerate(label_names)}

        def map_labels(example: dict[str, Any]) -> dict[str, Any]:
            ner_tags = example["ner_tags"]
            labels = [label_map[tag_id] for tag_id in ner_tags]
            return {
                "tokens": example["tokens"],
                "labels": labels,
                "ner_tags": ner_tags,
            }

        return dataset.map(map_labels, batched=False)
