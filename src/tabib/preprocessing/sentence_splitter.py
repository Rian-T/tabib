"""Sentence-level splitting preprocessor."""

from __future__ import annotations

import re
from typing import Any

from datasets import Dataset

from tabib.preprocessing.base import Preprocessor


class SentenceSplitter(Preprocessor):
    """Split each document into single-sentence examples."""

    def preprocess(self, dataset: Dataset, max_length: int | None = None) -> Dataset:
        """Split documents into sentences.

        Args:
            dataset: Hugging Face dataset containing ``doc_id``, ``text`` and ``entities``.
            max_length: Unused, maintained for interface compatibility.

        Returns:
            Dataset where every row corresponds to exactly one sentence with
            adjusted entity offsets.
        """
        split_examples: list[dict[str, Any]] = []

        if hasattr(dataset, "set_format"):
            dataset = dataset.with_format(None)

        for example in dataset:
            sentences = self._split_sentences_with_offsets(example["text"])
            for sentence_idx, (sentence_text, start, end) in enumerate(sentences):
                if start == end:
                    continue

                sentence_entities = self._filter_entities(
                    example["entities"], start, end
                )

                split_examples.append(
                    {
                        "doc_id": example["doc_id"],
                        "chunk_id": sentence_idx,
                        "chunk_offset": start,
                        "text": sentence_text,
                        "entities": sentence_entities,
                    }
                )

        return Dataset.from_list(split_examples)

    def _split_sentences_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        """Return sentences with their absolute offsets in ``text``."""
        pattern = r"([.!?]+\s+)"
        parts = re.split(pattern, text)

        sentences: list[tuple[str, int, int]] = []
        offset = 0

        for i in range(0, len(parts), 2):
            sentence = parts[i]
            if i + 1 < len(parts):
                sentence += parts[i + 1]

            if not sentence:
                continue

            start = offset
            end = start + len(sentence)
            sentences.append((sentence, start, end))
            offset = end

        # If the text does not end with punctuation, include the trailing part.
        if len(parts) % 2 == 1 and parts[-1] and not sentences:
            sentence = parts[-1]
            sentences.append((sentence, 0, len(sentence)))

        return sentences

    def _filter_entities(
        self, entities: list[dict[str, Any]], chunk_start: int, chunk_end: int
    ) -> list[dict[str, Any]]:
        """Return entities fully contained in the span [chunk_start, chunk_end)."""
        filtered: list[dict[str, Any]] = []

        for entity in entities:
            ent_start = entity["start"]
            ent_end = entity["end"]

            if ent_start >= chunk_start and ent_end <= chunk_end:
                filtered.append(
                    {
                        "start": ent_start - chunk_start,
                        "end": ent_end - chunk_start,
                        "label": entity["label"],
                        "text": entity["text"],
                    }
                )

        return filtered

