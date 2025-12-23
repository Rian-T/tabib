"""CAS2 dataset adapter - Token-level.

This adapter converts span-format CAS2 data to token-level IOB2 format.
Uses NERTokenTask for evaluation with seqeval.
Splits long documents into sentences to fit BERT's max_length.
"""

import os
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from datasets import Dataset

from tabib.data.base import DatasetAdapter
# Reuse sentence splitting from cas1_token
from tabib.data.cas1_token import split_into_sentences


# CAS2 entity types (8 types)
ENTITY_TYPES = [
    "anatomie", "dose", "examen", "mode",
    "moment", "substance", "traitement", "valeur"
]

# Tag names in BIO format (17 tags: O + 8 B-tags + 8 I-tags)
TAG_NAMES = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ["B", "I"]]


def spans_to_iob2_sentences(text: str, entities: list[dict], entity_types: list[str], max_tokens: int = 200) -> list[tuple[list[str], list[int]]]:
    """Convert character-offset spans to token-level IOB2, split by sentences.

    Args:
        text: Input text
        entities: List of entity dicts with 'start', 'end', 'label' keys
        entity_types: List of valid entity types
        max_tokens: Maximum tokens per sentence chunk

    Returns:
        List of (tokens, labels) tuples, one per sentence/chunk
    """
    # Build label-to-id mapping
    tag_names = ["O"] + [f"{p}-{t}" for t in entity_types for p in ["B", "I"]]
    label2id = {t: i for i, t in enumerate(tag_names)}

    # Sort entities by span size (largest first) for nested handling
    sorted_entities = sorted(entities, key=lambda e: -(e['end'] - e['start']))

    # Get sentence boundaries
    sentences = split_into_sentences(text)

    results = []
    current_tokens = []
    current_labels = []
    prev_entity_label = None

    for sent_start, sent_end in sentences:
        # Get tokens in this sentence
        sent_tokens_info = []
        for m in re.finditer(r'\S+', text[sent_start:sent_end]):
            sent_tokens_info.append({
                'text': m.group(),
                'start': sent_start + m.start(),
                'end': sent_start + m.end()
            })

        for tok in sent_tokens_info:
            tok_start, tok_end = tok['start'], tok['end']

            # Find overlapping entity
            assigned = False
            for ent in sorted_entities:
                ent_start = ent['start']
                ent_end = ent['end']
                ent_label = ent['label']

                # Check if token overlaps with entity
                if not (tok_end <= ent_start or tok_start >= ent_end):
                    # Token overlaps with entity - assign B or I
                    if ent_label not in entity_types:
                        continue

                    if prev_entity_label == ent_label:
                        current_labels.append(label2id[f"I-{ent_label}"])
                    else:
                        current_labels.append(label2id[f"B-{ent_label}"])
                    prev_entity_label = ent_label
                    assigned = True
                    break

            if not assigned:
                current_labels.append(0)  # O
                prev_entity_label = None

            current_tokens.append(tok['text'])

        # Check if we should flush (sentence boundary + enough tokens)
        if len(current_tokens) >= max_tokens // 2:
            if current_tokens:
                results.append((current_tokens, current_labels))
                current_tokens = []
                current_labels = []
                prev_entity_label = None  # Reset at sentence boundary

    # Don't forget remaining tokens
    if current_tokens:
        results.append((current_tokens, current_labels))

    return results


class CAS2TokenAdapter(DatasetAdapter):
    """Adapter for CAS2 dataset (token-level).

    Loads from local parquet files and converts span format to token-level IOB2.
    Splits long documents into sentence-level chunks.
    """

    # Max tokens per chunk (leaving room for BERT subword expansion ~1.5x)
    MAX_TOKENS = 200

    def __init__(self, data_dir: str | Path | None = None):
        """Initialize adapter.

        Args:
            data_dir: Path to dataset directory. If None, uses default location.
        """
        if data_dir:
            self._data_dir = Path(data_dir)
        else:
            # Default location
            scratch = os.environ.get('SCRATCH', '/lustre/fsn1/projects/rech/rua/uvb79kr')
            self._data_dir = Path(scratch) / "tabib" / "data" / "rntc--tabib-cas2"
        self._entity_types = ENTITY_TYPES

    @property
    def name(self) -> str:
        return "cas2_token"

    @property
    def entity_types(self) -> list[str]:
        return self._entity_types

    def _load_parquet_split(self, split_name: str) -> list[dict]:
        """Load a split from parquet file, splitting long docs into sentences."""
        parquet_path = self._data_dir / "data" / f"{split_name}-00000-of-00001.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        docs = []
        for idx, row in df.iterrows():
            text = row['text']
            entities = row['entities']

            # Convert spans to IOB2, split into sentence chunks
            chunks = spans_to_iob2_sentences(text, entities, ENTITY_TYPES, max_tokens=self.MAX_TOKENS)

            for chunk_idx, (tokens, labels) in enumerate(chunks):
                docs.append({
                    "doc_id": f"{row['doc_id']}_chunk{chunk_idx}",
                    "tokens": tokens,
                    "labels": labels,
                })

        return docs

    def load_splits(self) -> dict[str, Dataset]:
        """Load train/dev/test splits."""
        train_docs = self._load_parquet_split("train")
        dev_docs = self._load_parquet_split("dev")
        test_docs = self._load_parquet_split("test")

        return {
            "train": Dataset.from_list(train_docs),
            "dev": Dataset.from_list(dev_docs),
            "test": Dataset.from_list(test_docs),
        }

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        """Preprocess dataset for the task."""
        from tabib.tasks.ner_token import NERTokenTask

        if isinstance(task, NERTokenTask):
            # Set up the label space with CAS2 tags
            task.set_label_list(TAG_NAMES)

        return dataset
