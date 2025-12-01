"""FRACCO dataset adapters for ICD classification and NER."""

from __future__ import annotations

import csv
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from datasets import Dataset

from tabib.data.base import DatasetAdapter
from tabib.tasks.classification import ClassificationTask
from tabib.tasks.ner_token import NERTokenTask


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FRACCO_DIR = PROJECT_ROOT / "FRACCO"
ANN_DIR = FRACCO_DIR / "ann_txt_files"
DETECT_ONCO_CSV = FRACCO_DIR / "DetectOnco_Final.csv"


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float | None = None

    def resolve(self) -> tuple[float, float, float]:
        """Return a tuple of train/val/test ratios ensuring they sum to 1."""

        test = self.test if self.test is not None else max(0.0, 1.0 - self.train - self.val)
        total = self.train + self.val + test
        if not math.isclose(total, 1.0, rel_tol=1e-6):
            raise ValueError("Split ratios must sum to 1.0")
        return self.train, self.val, test


def _ensure_paths() -> None:
    if not FRACCO_DIR.exists():
        raise FileNotFoundError(
            "FRACCO directory not found. Expected at " f"{FRACCO_DIR}."
        )
    if not ANN_DIR.exists():
        raise FileNotFoundError("FRACCO ann_txt_files directory is missing")
    if not DETECT_ONCO_CSV.exists():
        raise FileNotFoundError("DetectOnco_Final.csv not found in FRACCO directory")


def _split_counts(total: int, ratios: Iterable[float]) -> list[int]:
    """Allocate integer counts for each split while honouring ratios."""

    ratios = list(ratios)
    floats = [total * r for r in ratios]
    counts = [int(x) for x in floats]
    remainder = total - sum(counts)

    # Distribute remainder to splits with largest fractional part, skipping zero ratios.
    fractional_indices = sorted(
        (
            (i, floats[i] - math.floor(floats[i]))
            for i in range(len(ratios))
            if ratios[i] > 0
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    for i in range(remainder):
        if not fractional_indices:
            break
        idx = fractional_indices[i % len(fractional_indices)][0]
        counts[idx] += 1

    return counts


def _compute_doc_splits(
    doc_names: list[str], ratios: SplitRatios, seed: int
) -> dict[str, list[str]]:
    """Return deterministic document lists for each split."""

    train_ratio, val_ratio, test_ratio = ratios.resolve()
    rng = random.Random(seed)
    shuffled = sorted(doc_names)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count, val_count, test_count = _split_counts(
        total, (train_ratio, val_ratio, test_ratio)
    )

    splits: dict[str, list[str]] = {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }
    return splits


def _empty_dataset() -> Dataset:
    return Dataset.from_dict({
        "text": [],
        "code": [],
        "doc_name": [],
        "span_start": [],
        "span_end": [],
    })


@dataclass
class TokenSpan:
    text: str
    start: int
    end: int


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _sanitize_label(name: str) -> str:
    sanitized = re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")
    return sanitized or "ENTITY"


def _read_document_text(doc_name: str) -> str:
    txt_path = ANN_DIR / doc_name.replace(".ann", ".txt")
    if not txt_path.exists():
        raise FileNotFoundError(f"Text file not found for {doc_name}: {txt_path}")
    return txt_path.read_text(encoding="utf-8", errors="replace")


def _read_annotation_spans(doc_name: str, target_label: str) -> list[tuple[int, int, str]]:
    ann_path = ANN_DIR / doc_name
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    spans: list[tuple[int, int, str]] = []
    with ann_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("T"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            meta = parts[1].split()
            label = meta[0]
            if label != target_label:
                continue
            if len(meta) < 3:
                continue
            start, end = int(meta[1]), int(meta[2])
            spans.append((start, end, parts[2]))
    return spans


def _tokenize_with_offsets(text: str) -> list[TokenSpan]:
    return [
        TokenSpan(match.group(0), match.start(), match.end())
        for match in TOKEN_PATTERN.finditer(text)
    ]


def _assign_bio_labels(
    tokens: Sequence[TokenSpan],
    spans: Sequence[tuple[int, int, str]],
    label_name: str,
) -> list[str]:
    labels = ["O"] * len(tokens)
    occupied = [False] * len(tokens)
    prefix_b = f"B-{label_name}"
    prefix_i = f"I-{label_name}"

    sorted_spans = sorted(spans, key=lambda item: (item[0], -(item[1] - item[0])))
    for start, end, _ in sorted_spans:
        covered: list[int] = []
        for idx, token in enumerate(tokens):
            if token.start >= end:
                break
            if token.end <= start:
                continue
            # Use strict containment to reduce noise.
            if token.start >= start and token.end <= end:
                covered.append(idx)
        if not covered:
            continue
        if any(occupied[idx] for idx in covered):
            continue
        labels[covered[0]] = prefix_b
        for idx in covered[1:]:
            labels[idx] = prefix_i
        for idx in covered:
            occupied[idx] = True

    return labels


def _chunk_tokens(
    tokens: Sequence[TokenSpan],
    labels: Sequence[str],
    max_tokens: int,
) -> Iterator[tuple[list[str], list[str]]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    i = 0
    n = len(tokens)
    while i < n:
        while i < n and labels[i].startswith("I-"):
            i += 1
        if i >= n:
            break

        end = min(i + max_tokens, n)
        while end < n and labels[end].startswith("I-"):
            end += 1

        chunk_tokens = [token.text for token in tokens[i:end]]
        chunk_labels = list(labels[i:end])
        yield chunk_tokens, chunk_labels
        i = end


class FRACCOICDClassificationAdapter(DatasetAdapter):
    """Adapter for mention-level ICD code classification from FRACCO."""

    def __init__(
        self,
        label_type: str = "expression_CIM",
        csv_path: str | Path | None = None,
        drop_multi_label: bool = True,
        min_samples: int = 1,
        top_k: int | None = None,
        include_context: bool = False,
        ratios: SplitRatios | None = None,
        seed: int = 42,
    ) -> None:
        self.label_type = label_type
        self.csv_path = Path(csv_path) if csv_path else DETECT_ONCO_CSV
        self.drop_multi_label = drop_multi_label
        self.min_samples = min_samples
        self.top_k = top_k  # If set, keep only top_k most frequent codes
        self.include_context = include_context
        self.ratios = ratios or SplitRatios()
        self.seed = seed
        self._label_vocab: list[str] | None = None
        self._doc_texts: dict[str, str] = {}  # Cache for document texts

    @property
    def name(self) -> str:
        return "fracco_icd_classification"

    def load_splits(self) -> dict[str, Dataset]:
        _ensure_paths()

        records_by_doc: dict[str, list[dict[str, Any]]] = {}
        with self.csv_path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            doc_field = reader.fieldnames[0] if reader.fieldnames else "doc_name"

            for row in reader:
                label = (row.get("label") or "").strip()
                if label != self.label_type:
                    continue

                code = (row.get("code") or "").strip()
                if not code:
                    continue
                if self.drop_multi_label and ";" in code:
                    continue

                text = (row.get("content") or "").strip()
                if not text:
                    continue

                span_raw = (row.get("full_span") or "").strip()
                start, end = None, None
                if span_raw:
                    parts = span_raw.split()
                    if len(parts) == 2:
                        try:
                            start, end = int(parts[0]), int(parts[1])
                        except ValueError:
                            start, end = None, None

                doc_name = (row.get(doc_field) or "").strip()
                if not doc_name:
                    continue

                # If include_context, prepend full document text
                final_text = text
                if self.include_context:
                    if doc_name not in self._doc_texts:
                        try:
                            self._doc_texts[doc_name] = _read_document_text(doc_name)
                        except FileNotFoundError:
                            self._doc_texts[doc_name] = ""
                    doc_text = self._doc_texts[doc_name]
                    if doc_text:
                        final_text = f"{doc_text} [SEP] {text}"

                records_by_doc.setdefault(doc_name, []).append(
                    {
                        "text": final_text,
                        "code": code,
                        "doc_name": doc_name,
                        "span_start": start if start is not None else -1,
                        "span_end": end if end is not None else -1,
                    }
                )

        if not records_by_doc:
            raise ValueError(
                f"No records found for label_type={self.label_type}."
            )

        # Count samples per code across all documents
        code_counts: dict[str, int] = {}
        for doc_records in records_by_doc.values():
            for entry in doc_records:
                code = entry["code"]
                code_counts[code] = code_counts.get(code, 0) + 1

        # Filter codes by min_samples threshold
        valid_codes = {
            code for code, count in code_counts.items()
            if count >= self.min_samples
        }

        # If top_k is set, further filter to only top_k most frequent codes
        if self.top_k is not None:
            sorted_codes = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)
            top_k_codes = {code for code, _ in sorted_codes[:self.top_k]}
            valid_codes = valid_codes & top_k_codes

        # Remove records with rare codes
        if self.min_samples > 1 or self.top_k is not None:
            for doc_name in list(records_by_doc.keys()):
                records_by_doc[doc_name] = [
                    entry for entry in records_by_doc[doc_name]
                    if entry["code"] in valid_codes
                ]
                # Remove empty documents
                if not records_by_doc[doc_name]:
                    del records_by_doc[doc_name]

        self._label_vocab = sorted(valid_codes)

        doc_names = list(records_by_doc)
        splits = _compute_doc_splits(doc_names, self.ratios, self.seed)

        datasets: dict[str, Dataset] = {}
        for split_name, doc_subset in splits.items():
            rows: list[dict[str, Any]] = []
            for doc in doc_subset:
                rows.extend(records_by_doc.get(doc, []))

            datasets[split_name] = (
                Dataset.from_list(rows) if rows else _empty_dataset()
            )

        return datasets

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, ClassificationTask):
            raise ValueError(
                f"FRACCO ICD classification expects ClassificationTask, got {type(task)}"
            )

        if self._label_vocab:
            task.ensure_labels(self._label_vocab)
        else:
            codes = sorted({str(code) for code in dataset["code"]}) if len(dataset) else []
            if codes:
                task.ensure_labels(codes)
        label_map = task.label_space

        def map_example(example: dict[str, Any]) -> dict[str, Any]:
            code = str(example["code"])
            return {
                "text": example["text"],
                "labels": label_map[code],
            }

        processed = dataset.map(map_example, remove_columns=[
            col for col in dataset.column_names if col not in {"text", "code"}
        ])
        processed = processed.remove_columns("code")
        processed.set_format(type="python")
        return processed


class FRACCOExpressionNERAdapter(DatasetAdapter):
    """Adapter for token-level NER over FRACCO expression_CIM mentions."""

    def __init__(
        self,
        entity_label: str = "expression_CIM",
        max_tokens: int = 256,
        ratios: SplitRatios | None = None,
        seed: int = 42,
    ) -> None:
        self.entity_label = entity_label
        self.max_tokens = max_tokens
        self.ratios = ratios or SplitRatios()
        self.seed = seed
        self._bio_label = _sanitize_label(entity_label)

    @property
    def name(self) -> str:
        return "fracco_expression_ner"

    def load_splits(self) -> dict[str, Dataset]:
        _ensure_paths()

        doc_chunks: dict[str, list[dict[str, Any]]] = {}
        doc_names = sorted(path.name for path in ANN_DIR.glob("*.ann"))

        for doc_name in doc_names:
            text = _read_document_text(doc_name)
            tokens = _tokenize_with_offsets(text)
            spans = _read_annotation_spans(doc_name, self.entity_label)
            labels = _assign_bio_labels(tokens, spans, self._bio_label)

            chunks: list[dict[str, Any]] = []
            for chunk_id, (chunk_tokens, chunk_labels) in enumerate(
                _chunk_tokens(tokens, labels, self.max_tokens)
            ):
                chunks.append(
                    {
                        "tokens": chunk_tokens,
                        "ner_tags": chunk_labels,
                        "doc_name": doc_name,
                        "chunk_id": chunk_id,
                    }
                )

            if not chunks:
                # Keep empty chunk for documents without tokens.
                chunks.append(
                    {
                        "tokens": [],
                        "ner_tags": [],
                        "doc_name": doc_name,
                        "chunk_id": 0,
                    }
                )

            doc_chunks[doc_name] = chunks

        splits = _compute_doc_splits(doc_names, self.ratios, self.seed)

        datasets: dict[str, Dataset] = {}
        for split_name, docs in splits.items():
            rows: list[dict[str, Any]] = []
            for doc in docs:
                rows.extend(doc_chunks.get(doc, []))

            if rows:
                datasets[split_name] = Dataset.from_list(rows)
            else:
                datasets[split_name] = Dataset.from_dict(
                    {
                        "tokens": [],
                        "ner_tags": [],
                        "doc_name": [],
                        "chunk_id": [],
                    }
                )

        return datasets

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        if not isinstance(task, NERTokenTask):
            raise ValueError(
                f"FRACCO NER expects NERTokenTask, got {type(task)}"
            )

        label_set: set[str] = set()
        for tags in dataset["ner_tags"]:
            label_set.update(tags)

        ordered_labels = ["O"] + sorted(
            (label for label in label_set if label != "O")
        )
        task.ensure_labels(ordered_labels)

        def map_example(example: dict[str, Any]) -> dict[str, Any]:
            label_ids = [task.label_to_id(tag) for tag in example["ner_tags"]]
            mapped = {
                "tokens": example["tokens"],
                "labels": label_ids,
            }
            if "doc_name" in example:
                mapped["doc_name"] = example["doc_name"]
            if "chunk_id" in example:
                mapped["chunk_id"] = example["chunk_id"]
            return mapped

        keep_columns = set(dataset.column_names)
        processed = dataset.map(map_example, remove_columns=[
            col
            for col in dataset.column_names
            if col not in {"tokens", "ner_tags", "doc_name", "chunk_id"}
        ])
        if "ner_tags" in keep_columns:
            processed = processed.remove_columns("ner_tags")
        processed.set_format(type="python")
        return processed


