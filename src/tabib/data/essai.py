"""ESSAI dataset adapter for negation/speculation classification."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.classification import ClassificationTask


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "drbenchmark" / "data" / "drbenchmark" / "essai"

# Classification labels
CLS_LABELS = ['negation_speculation', 'negation', 'neutral', 'speculation']


class ESSAIAdapter(DatasetAdapter):
    """Adapter for ESSAI negation/speculation classification.

    French clinical trial protocols annotated for negation and speculation.
    Task: Sentence-level classification into 4 classes.
    """

    def __init__(self) -> None:
        self._label_vocab: list[str] = CLS_LABELS

    @property
    def name(self) -> str:
        return "essai"

    def load_splits(self) -> dict[str, Dataset]:
        if not DATA_DIR.exists():
            raise FileNotFoundError(
                f"ESSAI data not found. Expected at {DATA_DIR}. "
                "Download from DrBenchmark/ESSAI on HuggingFace."
            )

        # Load both negation and speculation files
        neg_file = DATA_DIR / "ESSAI_neg.txt"
        spec_file = DATA_DIR / "ESSAI_spec.txt"

        if not neg_file.exists() or not spec_file.exists():
            raise FileNotFoundError("ESSAI data files missing")

        # Parse both files and merge annotations
        neg_sentences = self._parse_conll_file(neg_file)
        spec_sentences = self._parse_conll_file(spec_file)

        # Combine annotations: sentence -> (has_negation, has_speculation)
        all_sentences = {}
        for doc_id, tokens in neg_sentences.items():
            text = " ".join(tokens)
            if doc_id not in all_sentences:
                all_sentences[doc_id] = {"text": text, "neg": False, "spec": False}
            # Check if any token has negation tag (not ***)
            all_sentences[doc_id]["neg"] = any(t != "***" for t in neg_sentences.get(doc_id, {}).values()) if isinstance(neg_sentences.get(doc_id), dict) else False

        for doc_id, tokens in spec_sentences.items():
            text = " ".join(tokens)
            if doc_id not in all_sentences:
                all_sentences[doc_id] = {"text": text, "neg": False, "spec": False}
            all_sentences[doc_id]["spec"] = any(t != "***" for t in spec_sentences.get(doc_id, {}).values()) if isinstance(spec_sentences.get(doc_id), dict) else False

        # Build records from the parsed files
        records = self._build_classification_records(neg_file, spec_file)

        # Split into train/val/test (80/10/10)
        random.seed(42)
        random.shuffle(records)

        n = len(records)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        return {
            "train": Dataset.from_list(records[:train_end]),
            "val": Dataset.from_list(records[train_end:val_end]),
            "test": Dataset.from_list(records[val_end:]),
        }

    def _parse_conll_file(self, filepath: Path) -> dict[str, list[str]]:
        """Parse ESSAI CoNLL-like format, returning doc_id -> tokens."""
        sentences = {}
        with filepath.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 5:
                    doc_id = parts[0]
                    token = parts[2]
                    if doc_id not in sentences:
                        sentences[doc_id] = []
                    sentences[doc_id].append(token)
        return sentences

    def _build_classification_records(self, neg_file: Path, spec_file: Path) -> list[dict]:
        """Build classification records from ESSAI files."""
        # Parse both files
        neg_data = {}  # doc_id -> {token_id: (token, tag)}
        spec_data = {}

        with neg_file.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 6:
                    doc_id, tok_id, token, lemma, pos, tag = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
                    if doc_id not in neg_data:
                        neg_data[doc_id] = {}
                    neg_data[doc_id][tok_id] = (token, tag)

        with spec_file.open(encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 6:
                    doc_id, tok_id, token, lemma, pos, tag = parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]
                    if doc_id not in spec_data:
                        spec_data[doc_id] = {}
                    spec_data[doc_id][tok_id] = (token, tag)

        # Build records
        records = []
        all_doc_ids = set(neg_data.keys()) | set(spec_data.keys())

        for doc_id in all_doc_ids:
            neg_tokens = neg_data.get(doc_id, {})
            spec_tokens = spec_data.get(doc_id, {})

            # Get tokens from whichever file has them
            tokens = [t[0] for t in sorted(neg_tokens.values(), key=lambda x: x)] if neg_tokens else \
                     [t[0] for t in sorted(spec_tokens.values(), key=lambda x: x)]

            # Reconstruct tokens in order
            if neg_tokens:
                token_list = []
                for i in range(len(neg_tokens)):
                    if str(i) in neg_tokens:
                        token_list.append(neg_tokens[str(i)][0])
                tokens = token_list

            text = " ".join(tokens)
            if not text.strip():
                continue

            # Determine label
            has_neg = any(tag != "***" for _, tag in neg_tokens.values()) if neg_tokens else False
            has_spec = any(tag != "***" for _, tag in spec_tokens.values()) if spec_tokens else False

            if has_neg and has_spec:
                label = "negation_speculation"
            elif has_neg:
                label = "negation"
            elif has_spec:
                label = "speculation"
            else:
                label = "neutral"

            records.append({"text": text, "label": label})

        return records

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, ClassificationTask):
            raise ValueError(
                f"ESSAI expects ClassificationTask, got {type(task)}"
            )

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
