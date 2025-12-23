"""QUAERO EMEA dataset adapter (HuggingFace format).

This is the official QUAERO EMEA dataset with:
- Sentence-level splitting (not document-level)
- Coarse-grained entity filtering already applied
- Token-level BIO tags

This should match CamemBERT-bio evaluation methodology exactly.
"""

from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from tabib.data.base import DatasetAdapter


# Entity type mapping from ClassLabel indices
ENTITY_TYPES = [
    "ANAT", "CHEM", "DEVI", "DISO", "GEOG",
    "LIVB", "OBJC", "PHEN", "PHYS", "PROC"
]

# Tag names from dataset (21 classes including O)
TAG_NAMES = [
    "O", "B-ANAT", "I-ANAT", "B-CHEM", "I-CHEM",
    "B-DEVI", "I-DEVI", "B-DISO", "I-DISO", "B-GEOG", "I-GEOG",
    "B-LIVB", "I-LIVB", "B-OBJC", "I-OBJC", "B-PHEN", "I-PHEN",
    "B-PHYS", "I-PHYS", "B-PROC", "I-PROC"
]


class QuaeroEMEAHFAdapter(DatasetAdapter):
    """Adapter for QUAERO EMEA dataset from HuggingFace.

    Uses the rntc/quaero-frenchmed-ner-emea-sen dataset which has:
    - tokens: list of tokens
    - ner_tags: list of BIO tag indices

    Already preprocessed with coarse-grained entities and sentence splitting.
    """

    def __init__(self, data_dir: str | Path | None = None):
        """Initialize adapter.

        Args:
            data_dir: Optional local path to dataset. If None, loads from HF.
        """
        self._data_dir = Path(data_dir) if data_dir else None
        self._entity_types = ENTITY_TYPES

    @property
    def name(self) -> str:
        return "quaero_emea_hf"

    @property
    def entity_types(self) -> list[str]:
        return self._entity_types

    def load_splits(self) -> dict[str, Dataset]:
        """Load train/dev/test splits."""
        if self._data_dir:
            ds = load_dataset("parquet", data_dir=str(self._data_dir / "data"))
        else:
            ds = load_dataset("rntc/quaero-frenchmed-ner-emea-sen")

        # Convert to span format for NERSpanTask compatibility
        train_docs = [self._convert_to_spans(sample, i) for i, sample in enumerate(ds["train"])]
        dev_docs = [self._convert_to_spans(sample, i) for i, sample in enumerate(ds["validation"])]
        test_docs = [self._convert_to_spans(sample, i) for i, sample in enumerate(ds["test"])]

        return {
            "train": Dataset.from_list(train_docs),
            "dev": Dataset.from_list(dev_docs),
            "test": Dataset.from_list(test_docs),
        }

    def _convert_to_spans(self, sample: dict, idx: int) -> dict[str, Any]:
        """Convert token-level BIO tags to character-offset spans."""
        tokens = sample["tokens"]
        tags = sample["ner_tags"]

        # Reconstruct text from tokens
        text = " ".join(tokens)

        # Track character offsets
        char_offset = 0
        token_offsets = []
        for token in tokens:
            start = char_offset
            end = start + len(token)
            token_offsets.append((start, end))
            char_offset = end + 1  # +1 for space

        # Convert BIO tags to spans
        entities = []
        current_entity = None

        for i, (token, tag_id) in enumerate(zip(tokens, tags)):
            start, end = token_offsets[i]
            tag_name = TAG_NAMES[tag_id]

            if tag_name.startswith("B-"):
                # Save previous entity
                if current_entity is not None:
                    current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                    entities.append(current_entity)

                # Start new entity
                entity_type = tag_name[2:]
                current_entity = {
                    "start": start,
                    "end": end,
                    "label": entity_type,
                }
            elif tag_name.startswith("I-"):
                # Extend current entity if same type
                entity_type = tag_name[2:]
                if current_entity is not None and current_entity["label"] == entity_type:
                    current_entity["end"] = end
                else:
                    # I- without matching B- - treat as B-
                    if current_entity is not None:
                        current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                        entities.append(current_entity)
                    current_entity = {
                        "start": start,
                        "end": end,
                        "label": entity_type,
                    }
            else:  # O
                if current_entity is not None:
                    current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
                    entities.append(current_entity)
                    current_entity = None

        # Don't forget last entity
        if current_entity is not None:
            current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
            entities.append(current_entity)

        return {
            "doc_id": sample.get("id", f"emea_{idx}"),
            "text": text,
            "entities": entities,
        }

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        """Preprocess dataset for the task."""
        from tabib.tasks.ner_span import NERSpanTask

        if isinstance(task, NERSpanTask):
            task.ensure_entity_types(self.entity_types)

        return dataset
