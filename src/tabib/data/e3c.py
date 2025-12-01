"""E3C (European Clinical Case Corpus) dataset adapter."""

from typing import Any

from datasets import Dataset, load_dataset

from tabib.data.base import DatasetAdapter


class E3CAdapter(DatasetAdapter):
    """Adapter for the E3C dataset (French clinical cases).

    E3C has token-level BIO annotations for clinical entities.
    We convert to character-offset spans for NER span task.

    Entity types: CLINENTITY (clinical entities)

    Dataset: rntc/legacy_e3c (HuggingFace)
    - fr.layer2: training data (well-annotated)
    - fr.layer1: test data (silver annotations)
    """

    def __init__(self):
        self._entity_types = ["CLINENTITY"]

    @property
    def name(self) -> str:
        return "e3c"

    @property
    def entity_types(self) -> list[str]:
        return self._entity_types

    def load_splits(self) -> dict[str, Dataset]:
        """Load train/dev/test splits from HuggingFace."""
        ds = load_dataset("rntc/legacy_e3c")

        # French splits
        # layer2: well-annotated (train/dev)
        # layer1: silver annotations (test)
        train_dev = ds["fr.layer2"]
        validation = ds["fr.layer2.validation"]
        test = ds["fr.layer1"]

        # Convert to span format
        train_docs = [self._convert_to_spans(sample) for sample in train_dev]
        dev_docs = [self._convert_to_spans(sample) for sample in validation]
        test_docs = [self._convert_to_spans(sample) for sample in test]

        return {
            "train": Dataset.from_list(train_docs),
            "dev": Dataset.from_list(dev_docs),
            "test": Dataset.from_list(test_docs),
        }

    def _convert_to_spans(self, sample: dict) -> dict[str, Any]:
        """Convert token-level BIO tags to character-offset spans."""
        text = sample["text"]
        tokens = sample["tokens"]
        offsets = sample["tokens_offsets"]
        tags = sample["clinical_entity_tags"]

        # BIO encoding: 0=O, 1=B-CLINENTITY, 2=I-CLINENTITY
        entities = []
        current_entity = None

        for i, (token, offset, tag) in enumerate(zip(tokens, offsets, tags)):
            start, end = offset

            if tag == 1:  # B-CLINENTITY
                # Save previous entity if exists
                if current_entity is not None:
                    entities.append(current_entity)

                # Start new entity
                current_entity = {
                    "start": start,
                    "end": end,
                    "label": "CLINENTITY",
                    "text": text[start:end],
                }
            elif tag == 2:  # I-CLINENTITY
                # Extend current entity
                if current_entity is not None:
                    current_entity["end"] = end
                    current_entity["text"] = text[current_entity["start"]:end]
            else:  # O
                # Save entity if exists
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None

        # Don't forget last entity
        if current_entity is not None:
            entities.append(current_entity)

        return {
            "doc_id": f"e3c_{hash(text) % 100000}",
            "text": text,
            "entities": entities,
        }

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        """Preprocess dataset for the task."""
        from tabib.tasks.ner_span import NERSpanTask

        if isinstance(task, NERSpanTask):
            task.ensure_entity_types(self.entity_types)

        return dataset
