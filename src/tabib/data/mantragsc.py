"""MANTRAGSC dataset adapter for French biomedical NER."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.ner_span import NERSpanTask


# BIO tag mapping for MANTRAGSC (from HuggingFace dataset)
# Tags are: O, B-ANAT, I-ANAT, B-CHEM, I-CHEM, B-DEVI, I-DEVI, B-DISO, I-DISO,
#           B-GEOG, I-GEOG, B-LIVB, I-LIVB, B-OBJC, I-OBJC, B-PHEN, I-PHEN,
#           B-PHYS, I-PHYS, B-PROC, I-PROC
NER_TAG_NAMES = [
    "O",
    "B-ANAT", "I-ANAT",
    "B-CHEM", "I-CHEM",
    "B-DEVI", "I-DEVI",
    "B-DISO", "I-DISO",
    "B-GEOG", "I-GEOG",
    "B-LIVB", "I-LIVB",
    "B-OBJC", "I-OBJC",
    "B-PHEN", "I-PHEN",
    "B-PHYS", "I-PHYS",
    "B-PROC", "I-PROC",
]


class MANTRAGSCAdapter(DatasetAdapter):
    """Adapter for MANTRAGSC French biomedical NER dataset.

    Uses the Medline French manually annotated gold standard corpus.
    Entity types: ANAT, CHEM, DEVI, DISO, GEOG, LIVB, OBJC, PHEN, PHYS, PROC.
    """

    def __init__(self, source: str = "medline") -> None:
        """Initialize adapter.

        Args:
            source: Data source - 'medline', 'emea', or 'patent'
        """
        self._source = source.lower()

    @property
    def name(self) -> str:
        return f"mantragsc_{self._source}"

    def load_splits(self) -> dict[str, Dataset]:
        """Load train/val/test splits from HuggingFace."""
        hf_config = f"fr_{self._source}"
        hf_ds = load_dataset("DrBenchmark/MANTRAGSC", hf_config, trust_remote_code=True)

        splits: dict[str, Dataset] = {}
        split_map = {"train": "train", "validation": "val", "test": "test"}

        for hf_split, local_split in split_map.items():
            if hf_split not in hf_ds:
                continue

            records = []
            for item in hf_ds[hf_split]:
                doc_id = item.get("id", "")
                tokens = item.get("tokens", [])
                ner_tags = item.get("ner_tags", [])

                # Convert tokens/tags to text and character-offset spans
                text, entities = self._convert_bio_to_spans(tokens, ner_tags)

                if text:
                    records.append({
                        "doc_id": doc_id,
                        "text": text,
                        "entities": entities,
                    })

            if records:
                splits[local_split] = Dataset.from_list(records)

        return splits

    def _convert_bio_to_spans(
        self, tokens: list[str], ner_tags: list[int]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Convert token-level BIO tags to character-offset spans."""
        # Reconstruct text with spaces between tokens
        text_parts = []
        char_offsets = []
        current_pos = 0

        for token in tokens:
            char_offsets.append(current_pos)
            text_parts.append(token)
            current_pos += len(token) + 1  # +1 for space

        text = " ".join(text_parts)

        # Extract entities from BIO tags
        entities = []
        current_entity = None

        for i, (token, tag_id) in enumerate(zip(tokens, ner_tags)):
            if tag_id >= len(NER_TAG_NAMES):
                tag = "O"
            else:
                tag = NER_TAG_NAMES[tag_id]

            if tag.startswith("B-"):
                # Save previous entity
                if current_entity is not None:
                    entities.append(current_entity)

                # Start new entity
                label = tag[2:]
                start = char_offsets[i]
                end = start + len(token)
                current_entity = {
                    "start": start,
                    "end": end,
                    "label": label,
                    "text": token,
                }
            elif tag.startswith("I-") and current_entity is not None:
                # Extend current entity
                label = tag[2:]
                if label == current_entity["label"]:
                    current_entity["end"] = char_offsets[i] + len(token)
                    current_entity["text"] = text[current_entity["start"]:current_entity["end"]]
            else:
                # O tag - save entity if exists
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None

        # Don't forget last entity
        if current_entity is not None:
            entities.append(current_entity)

        return text, entities

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, NERSpanTask):
            raise ValueError(
                f"MANTRAGSC expects NERSpanTask, got {type(task)}"
            )

        def map_example(example: dict[str, Any]) -> dict[str, Any]:
            return {
                "text": example["text"],
                "spans": example["entities"],
            }

        processed = dataset.map(map_example)
        processed.set_format(type="python")
        return processed
