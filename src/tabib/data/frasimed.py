"""FRASIMED dataset adapters for NER and entity normalization.

FRASIMED is a French clinical NER corpus created via crosslingual BERT-based
annotation projection from Spanish datasets (CANTEMIST and DISTEMIST).

Two subsets:
- CANTEMIST-FRASIMED: 1,301 docs, 15,978 entities (morphology-onco), ICD-O-3.1 + SNOMED
- DISTEMIST-FRASIMED: 750 docs, 8,059 entities (disease), SNOMED only

Tasks:
- NER: Named entity recognition for medical entities
- Entity normalization (mention-level): Classify mention → code
- Document-level multilabel: Classify document → all codes

Data path: $SCRATCH/tabib/data/FRASIMED/
"""

from __future__ import annotations

import logging
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from tabib.data.base import DatasetAdapter

logger = logging.getLogger(__name__)

# Data paths
SCRATCH = os.environ.get("SCRATCH", "")
SCRATCH_DATA = Path(SCRATCH) / "tabib" / "data" if SCRATCH else None
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# FRASIMED directory (flat structure, no train/dev/test subdirs)
FRASIMED_DIR = SCRATCH_DATA / "FRASIMED" if SCRATCH_DATA else PROJECT_ROOT / "data" / "FRASIMED"

# HuggingFace repos for FRASIMED datasets
HF_REPO_CANTEMIST = "rntc/tabib-frasimed-cantemist"
HF_REPO_DISTEMIST = "rntc/tabib-frasimed-distemist"
HF_REPO_CANTEMIST_DOC = "rntc/tabib-frasimed-cantemist-doc"
HF_REPO_DISTEMIST_DOC = "rntc/tabib-frasimed-distemist-doc"


@dataclass(frozen=True)
class SplitRatios:
    """Train/val/test split ratios."""
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def as_tuple(self) -> tuple[float, float, float]:
        """Return as tuple ensuring sum is ~1.0."""
        total = self.train + self.val + self.test
        if not (0.99 < total < 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return self.train, self.val, self.test


def _split_documents(
    doc_names: list[str],
    ratios: SplitRatios,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Split document names into train/val/test."""
    rng = random.Random(seed)
    shuffled = sorted(doc_names)
    rng.shuffle(shuffled)

    train_r, val_r, test_r = ratios.as_tuple()
    n = len(shuffled)
    n_train = int(n * train_r)
    n_val = int(n * val_r)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def _parse_ann_with_codes(ann_path: Path, text: str | None = None) -> list[dict[str, Any]]:
    """Parse BRAT .ann file and extract entities WITH normalization codes.

    FRASIMED annotation format:
        T1	morphology-onco 164 172	grosseur
        #1	ICD-O-3.1 T9	8500/3
        #19	SNOMED T9	58477004

    Returns list of entities with 'codes' dict mapping code_type -> code_value.
    """
    entities: dict[str, dict[str, Any]] = {}  # T_id -> entity
    codes: dict[str, dict[str, str]] = defaultdict(dict)  # T_id -> {code_type: code_value}

    content = ann_path.read_text(encoding="utf-8", errors="replace")

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("T"):
            # Entity line: T1\tLABEL start end\tentity_text
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            t_id = parts[0]
            type_info = parts[1].split()
            entity_type = type_info[0]

            # Parse span (may be discontinuous: "start1 end1;start2 end2")
            span_str = " ".join(type_info[1:])
            fragments = []
            for span_part in span_str.split(";"):
                coords = span_part.strip().split()
                if len(coords) >= 2:
                    fragments.append({
                        "begin": int(coords[0]),
                        "end": int(coords[1]),
                    })

            if not fragments:
                continue

            # Sort and use overall span
            fragments.sort(key=lambda f: f["begin"])
            start = fragments[0]["begin"]
            end = fragments[-1]["end"]
            entity_text = parts[2]

            entities[t_id] = {
                "start": start,
                "end": end,
                "label": entity_type,
                "text": entity_text,
                "fragments": fragments,
                "codes": {},  # Will be filled below
            }

        elif line.startswith("#"):
            # Normalization code line: #1\tICD-O-3.1 T9\t8500/3
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            code_info = parts[1].split()
            if len(code_info) < 2:
                continue

            code_type = code_info[0]  # e.g., "ICD-O-3.1" or "SNOMED"
            t_id = code_info[1]  # e.g., "T9"
            code_value = parts[2]

            codes[t_id][code_type] = code_value

    # Merge codes into entities
    for t_id, entity in entities.items():
        entity["codes"] = codes.get(t_id, {})

    return list(entities.values())


def _filter_nested_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only coarsest granularity entities (remove nested ones)."""
    if not entities:
        return entities

    # Sort by span size (largest first), then by start position
    sorted_ents = sorted(
        entities, key=lambda e: (-(e["end"] - e["start"]), e["start"])
    )

    kept = []
    for ent in sorted_ents:
        is_nested = any(
            ent["start"] >= k["start"] and ent["end"] <= k["end"]
            for k in kept
        )
        if not is_nested:
            kept.append(ent)

    return sorted(kept, key=lambda e: e["start"])


# =============================================================================
# NER Adapters
# =============================================================================


class FRASIMEDNERAdapter(DatasetAdapter):
    """Base adapter for FRASIMED NER datasets.

    Handles flat directory structure (no train/dev/test subdirs).
    Implements random document-level splitting.

    Data is loaded from HuggingFace cache if available, otherwise from local BRAT files.
    Download with: huggingface-cli download rntc/tabib-frasimed-cantemist --local-dir $SCRATCH/tabib/data/rntc--tabib-frasimed-cantemist --repo-type dataset
    """

    def __init__(
        self,
        subset: str,
        data_dir: str | Path | None = None,
        chunk_size: int = -1,
        filter_nested: bool = False,
        ratios: SplitRatios | None = None,
        seed: int = 42,
        hf_repo: str | None = None,
    ):
        """Initialize FRASIMED NER adapter.

        Args:
            subset: "CANTEMIST-FRASIMED" or "DISTEMIST-FRASIMED"
            data_dir: Path to FRASIMED directory
            chunk_size: -1 for line-by-line (default), 0 for no chunking
            filter_nested: If True, keep only outermost entities
            ratios: Train/val/test split ratios
            seed: Random seed for reproducible splits
            hf_repo: HuggingFace repo ID. If set, tries to load from HF cache first.
        """
        self.subset = subset
        self.data_dir = Path(data_dir) if data_dir else FRASIMED_DIR
        self.chunk_size = chunk_size
        self.filter_nested = filter_nested
        self.ratios = ratios or SplitRatios()
        self.seed = seed
        self._entity_types: set[str] = set()
        self.hf_repo = hf_repo

        # Compute HF cache path
        if hf_repo and SCRATCH_DATA:
            cache_name = hf_repo.replace("/", "--")
            self._hf_cache_dir = SCRATCH_DATA / cache_name
        else:
            self._hf_cache_dir = None

    @property
    def name(self) -> str:
        return f"frasimed_{self.subset.lower().replace('-frasimed', '')}"

    @property
    def entity_types(self) -> list[str]:
        return sorted(self._entity_types)

    def _get_subset_dir(self) -> Path:
        """Get path to subset directory."""
        return self.data_dir / self.subset

    def _load_all_documents(self) -> list[dict[str, Any]]:
        """Load all documents from the subset directory."""
        subset_dir = self._get_subset_dir()
        if not subset_dir.exists():
            raise FileNotFoundError(
                f"FRASIMED subset not found at {subset_dir}. "
                f"Expected structure: {self.data_dir}/{{CANTEMIST-FRASIMED,DISTEMIST-FRASIMED}}/"
            )

        documents = []
        txt_files = sorted(subset_dir.glob("*.txt"))

        for txt_path in txt_files:
            ann_path = txt_path.with_suffix(".ann")
            if not ann_path.exists():
                continue

            text = txt_path.read_text(encoding="utf-8", errors="replace")
            entities = _parse_ann_with_codes(ann_path, text)

            # Track entity types
            for ent in entities:
                self._entity_types.add(ent["label"])

            # Filter nested if requested
            if self.filter_nested:
                entities = _filter_nested_entities(entities)

            # Remove codes from entities for NER (not needed)
            for ent in entities:
                ent.pop("codes", None)

            documents.append({
                "doc_id": txt_path.stem,
                "text": text,
                "entities": entities,
            })

        return documents

    def _split_by_lines(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        """Split document into one sample per line."""
        text = doc["text"]
        entities = doc["entities"]
        doc_id = doc["doc_id"]

        lines = []
        line_start = 0

        for i, char in enumerate(text):
            if char == "\n":
                line_text = text[line_start:i]
                lines.append((line_start, i, line_text))
                line_start = i + 1

        # Handle last line
        if line_start < len(text):
            lines.append((line_start, len(text), text[line_start:]))

        samples = []
        for line_idx, (start, end, line_text) in enumerate(lines):
            if not line_text.strip():
                continue

            # Get entities fully within this line
            line_entities = []
            for ent in entities:
                if ent["start"] >= start and ent["end"] <= end:
                    line_entities.append({
                        "start": ent["start"] - start,
                        "end": ent["end"] - start,
                        "label": ent["label"],
                        "text": ent["text"],
                    })

            samples.append({
                "doc_id": f"{doc_id}_line{line_idx}",
                "text": line_text,
                "entities": line_entities,
            })

        return samples

    def load_splits(self) -> dict[str, Dataset]:
        """Load train/val/test splits from HF cache or BRAT format."""
        # Try HF cache first
        if self._hf_cache_dir and (self._hf_cache_dir / "data").exists():
            try:
                return self._load_from_hf()
            except Exception as e:
                logger.warning(f"Failed to load from HF cache, falling back to BRAT: {e}")

        # Fallback to BRAT files
        return self._load_from_brat()

    def _load_from_hf(self) -> dict[str, Dataset]:
        """Load from HuggingFace parquet cache."""
        logger.info(f"Loading {self.name} from HF cache: {self._hf_cache_dir}")
        ds = load_dataset("parquet", data_dir=str(self._hf_cache_dir / "data"))

        # Rename 'validation' -> 'val' for consistency
        splits = {}
        for split_name in ds.keys():
            out_name = "val" if split_name == "validation" else split_name
            splits[out_name] = ds[split_name]

        return splits

    def _load_from_brat(self) -> dict[str, Dataset]:
        """Load and split documents from BRAT format into train/val/test."""
        all_docs = self._load_all_documents()

        if not all_docs:
            raise ValueError(f"No documents found in {self._get_subset_dir()}")

        # Split by document
        doc_names = [d["doc_id"] for d in all_docs]
        doc_splits = _split_documents(doc_names, self.ratios, self.seed)
        doc_map = {d["doc_id"]: d for d in all_docs}

        datasets: dict[str, Dataset] = {}
        for split_name, split_doc_ids in doc_splits.items():
            samples = []
            for doc_id in split_doc_ids:
                doc = doc_map[doc_id]

                if self.chunk_size == -1:
                    # Line-by-line splitting
                    samples.extend(self._split_by_lines(doc))
                else:
                    samples.append(doc)

            if samples:
                datasets[split_name] = Dataset.from_list(samples)
            else:
                datasets[split_name] = Dataset.from_dict({
                    "doc_id": [],
                    "text": [],
                    "entities": [],
                })

        return datasets

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        """Preprocess dataset for NER task."""
        from tabib.tasks.ner_span import NERSpanTask

        if isinstance(task, NERSpanTask):
            task.ensure_entity_types(self.entity_types)

        return dataset


class FRASIMEDCANTEMISTAdapter(FRASIMEDNERAdapter):
    """FRASIMED CANTEMIST subset (oncology morphology entities).

    Data is loaded from HuggingFace cache if available, otherwise from local BRAT files.
    Download with: huggingface-cli download rntc/tabib-frasimed-cantemist --local-dir $SCRATCH/tabib/data/rntc--tabib-frasimed-cantemist --repo-type dataset
    """

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("hf_repo", HF_REPO_CANTEMIST)
        super().__init__(subset="CANTEMIST-FRASIMED", **kwargs)


class FRASIMEDDISTEMISTAdapter(FRASIMEDNERAdapter):
    """FRASIMED DISTEMIST subset (disease entities).

    Data is loaded from HuggingFace cache if available, otherwise from local BRAT files.
    Download with: huggingface-cli download rntc/tabib-frasimed-distemist --local-dir $SCRATCH/tabib/data/rntc--tabib-frasimed-distemist --repo-type dataset
    """

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("hf_repo", HF_REPO_DISTEMIST)
        super().__init__(subset="DISTEMIST-FRASIMED", **kwargs)


# =============================================================================
# Mention-Level Classification (Entity Normalization)
# =============================================================================


class FRASIMEDMentionClassificationAdapter(DatasetAdapter):
    """Mention-level entity normalization (mention → code).

    Extracts individual mentions from BRAT annotations and maps them to
    their normalization codes (ICD-O-3.1 or SNOMED).

    Input: mention text
    Output: code (single-label classification)
    """

    def __init__(
        self,
        subset: str,
        data_dir: str | Path | None = None,
        code_type: str = "auto",
        top_k: int | None = None,
        min_samples: int = 1,
        ratios: SplitRatios | None = None,
        seed: int = 42,
    ):
        """Initialize mention classification adapter.

        Args:
            subset: "CANTEMIST-FRASIMED" or "DISTEMIST-FRASIMED"
            data_dir: Path to FRASIMED directory
            code_type: "ICD-O-3.1", "SNOMED", or "auto" (first available)
            top_k: Keep only top-k most frequent codes
            min_samples: Minimum samples per code
            ratios: Train/val/test split ratios
            seed: Random seed for reproducible splits
        """
        self.subset = subset
        self.data_dir = Path(data_dir) if data_dir else FRASIMED_DIR
        self.code_type = code_type
        self.top_k = top_k
        self.min_samples = min_samples
        self.ratios = ratios or SplitRatios()
        self.seed = seed
        self._label_vocab: list[str] = []

    @property
    def name(self) -> str:
        suffix = f"_top{self.top_k}" if self.top_k else ""
        return f"frasimed_{self.subset.lower().replace('-frasimed', '')}_norm{suffix}"

    def _get_subset_dir(self) -> Path:
        return self.data_dir / self.subset

    def _determine_code_type(self, entities: list[dict[str, Any]]) -> str:
        """Determine which code type to use based on availability."""
        if self.code_type != "auto":
            return self.code_type

        # Check what's available
        has_icdo = any("ICD-O-3.1" in e.get("codes", {}) for e in entities)
        has_snomed = any("SNOMED" in e.get("codes", {}) for e in entities)

        if has_icdo:
            return "ICD-O-3.1"
        elif has_snomed:
            return "SNOMED"
        else:
            raise ValueError("No normalization codes found in annotations")

    def _load_all_mentions(self) -> tuple[list[dict[str, Any]], str]:
        """Load all mentions with their codes.

        Returns:
            Tuple of (mentions, code_type_used)
        """
        subset_dir = self._get_subset_dir()
        if not subset_dir.exists():
            raise FileNotFoundError(f"FRASIMED subset not found at {subset_dir}")

        all_entities: list[dict[str, Any]] = []
        ann_files = sorted(subset_dir.glob("*.ann"))

        for ann_path in ann_files:
            txt_path = ann_path.with_suffix(".txt")
            if not txt_path.exists():
                continue

            text = txt_path.read_text(encoding="utf-8", errors="replace")
            entities = _parse_ann_with_codes(ann_path, text)

            for ent in entities:
                ent["doc_name"] = ann_path.stem
            all_entities.extend(entities)

        if not all_entities:
            raise ValueError(f"No entities found in {subset_dir}")

        # Determine code type
        code_type = self._determine_code_type(all_entities)

        # Filter to entities with the selected code type
        mentions = []
        for ent in all_entities:
            code = ent.get("codes", {}).get(code_type)
            if code:
                mentions.append({
                    "text": ent["text"],
                    "code": code,
                    "doc_name": ent["doc_name"],
                    "span_start": ent["start"],
                    "span_end": ent["end"],
                })

        return mentions, code_type

    def load_splits(self) -> dict[str, Dataset]:
        """Load and split mentions into train/val/test."""
        mentions, code_type = self._load_all_mentions()

        # Group by document for document-level splitting
        mentions_by_doc: dict[str, list[dict]] = defaultdict(list)
        for m in mentions:
            mentions_by_doc[m["doc_name"]].append(m)

        # Count code frequencies
        code_counts: dict[str, int] = defaultdict(int)
        for m in mentions:
            code_counts[m["code"]] += 1

        # Filter by min_samples
        valid_codes = {
            code for code, count in code_counts.items()
            if count >= self.min_samples
        }

        # Apply top_k
        if self.top_k is not None:
            sorted_codes = sorted(
                code_counts.items(), key=lambda x: x[1], reverse=True
            )
            top_k_codes = {code for code, _ in sorted_codes[:self.top_k]}
            valid_codes = valid_codes & top_k_codes

        self._label_vocab = sorted(valid_codes)

        # Split documents
        doc_names = sorted(mentions_by_doc.keys())
        doc_splits = _split_documents(doc_names, self.ratios, self.seed)

        # Build datasets
        datasets: dict[str, Dataset] = {}
        for split_name, split_docs in doc_splits.items():
            records = []
            for doc_name in split_docs:
                for m in mentions_by_doc[doc_name]:
                    if m["code"] in valid_codes:
                        records.append(m)

            if records:
                datasets[split_name] = Dataset.from_list(records)
            else:
                datasets[split_name] = Dataset.from_dict({
                    "text": [],
                    "code": [],
                    "doc_name": [],
                    "span_start": [],
                    "span_end": [],
                })

        return datasets

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        """Preprocess for classification task."""
        from tabib.tasks.classification import ClassificationTask

        if not isinstance(task, ClassificationTask):
            raise ValueError(
                f"FRASIMEDMentionClassification requires ClassificationTask, got {type(task)}. "
                "Use task: classification in your config."
            )

        if self._label_vocab:
            task.ensure_labels(self._label_vocab)

        label_map = task.label_space

        def map_example(example: dict[str, Any]) -> dict[str, Any]:
            code = str(example["code"])
            return {
                "text": example["text"],
                "labels": label_map.get(code, 0),
            }

        processed = dataset.map(map_example)
        processed.set_format(type="python")
        return processed


# =============================================================================
# Document-Level Multilabel Classification
# =============================================================================


class FRASIMEDDocumentMultilabelAdapter(DatasetAdapter):
    """Document-level multilabel classification (document → all codes).

    Aggregates all normalization codes per document for multilabel classification.

    Input: full document text
    Output: multi-hot vector of all codes in document

    Data is loaded from HuggingFace cache if available, otherwise from local BRAT files.
    Download with: huggingface-cli download rntc/tabib-frasimed-cantemist-doc --local-dir $SCRATCH/tabib/data/rntc--tabib-frasimed-cantemist-doc --repo-type dataset
    """

    def __init__(
        self,
        subset: str,
        data_dir: str | Path | None = None,
        code_type: str = "auto",
        top_k: int | None = None,
        min_samples: int = 1,
        ratios: SplitRatios | None = None,
        seed: int = 42,
        hf_repo: str | None = None,
    ):
        """Initialize document multilabel adapter.

        Args:
            subset: "CANTEMIST-FRASIMED" or "DISTEMIST-FRASIMED"
            data_dir: Path to FRASIMED directory
            code_type: "ICD-O-3.1", "SNOMED", or "auto" (first available)
            top_k: Keep only top-k most frequent codes
            min_samples: Minimum documents per code
            ratios: Train/val/test split ratios
            seed: Random seed for reproducible splits
            hf_repo: HuggingFace repo ID. If set, tries to load from HF cache first.
        """
        self.subset = subset
        self.data_dir = Path(data_dir) if data_dir else FRASIMED_DIR
        self.code_type = code_type
        self.top_k = top_k
        self.min_samples = min_samples
        self.ratios = ratios or SplitRatios()
        self.seed = seed
        self._label_vocab: list[str] = []
        self.hf_repo = hf_repo

        # Compute HF cache path
        if hf_repo and SCRATCH_DATA:
            cache_name = hf_repo.replace("/", "--")
            self._hf_cache_dir = SCRATCH_DATA / cache_name
        else:
            self._hf_cache_dir = None

    @property
    def name(self) -> str:
        suffix = f"_top{self.top_k}" if self.top_k else ""
        return f"frasimed_{self.subset.lower().replace('-frasimed', '')}_doc{suffix}"

    def _get_subset_dir(self) -> Path:
        return self.data_dir / self.subset

    def _determine_code_type(self, all_codes: dict[str, set[str]]) -> str:
        """Determine which code type to use based on availability."""
        if self.code_type != "auto":
            return self.code_type

        # Flatten all codes to check availability
        all_code_types: set[str] = set()
        for codes in all_codes.values():
            for code in codes:
                # FRASIMED codes don't have type prefix in our structure
                # We need to check the raw annotations
                pass

        # Default based on subset
        if "CANTEMIST" in self.subset:
            return "ICD-O-3.1"
        else:
            return "SNOMED"

    def _load_all_documents(self) -> tuple[dict[str, str], dict[str, set[str]], str]:
        """Load document texts and their codes.

        Returns:
            Tuple of (doc_texts, doc_codes, code_type_used)
        """
        subset_dir = self._get_subset_dir()
        if not subset_dir.exists():
            raise FileNotFoundError(f"FRASIMED subset not found at {subset_dir}")

        doc_texts: dict[str, str] = {}
        doc_codes: dict[str, set[str]] = defaultdict(set)
        code_type_to_use: str | None = None

        ann_files = sorted(subset_dir.glob("*.ann"))

        for ann_path in ann_files:
            txt_path = ann_path.with_suffix(".txt")
            if not txt_path.exists():
                continue

            doc_name = ann_path.stem
            text = txt_path.read_text(encoding="utf-8", errors="replace")
            doc_texts[doc_name] = text

            entities = _parse_ann_with_codes(ann_path, text)

            # Determine code type on first entity with codes
            if code_type_to_use is None:
                for ent in entities:
                    codes = ent.get("codes", {})
                    if self.code_type != "auto":
                        code_type_to_use = self.code_type
                        break
                    elif "ICD-O-3.1" in codes:
                        code_type_to_use = "ICD-O-3.1"
                        break
                    elif "SNOMED" in codes:
                        code_type_to_use = "SNOMED"
                        break

            # Collect codes for this document
            for ent in entities:
                code = ent.get("codes", {}).get(code_type_to_use or "SNOMED")
                if code:
                    doc_codes[doc_name].add(code)

        if code_type_to_use is None:
            # Fallback based on subset
            code_type_to_use = "ICD-O-3.1" if "CANTEMIST" in self.subset else "SNOMED"

        return doc_texts, dict(doc_codes), code_type_to_use

    def load_splits(self) -> dict[str, Dataset]:
        """Load train/val/test splits from HF cache or BRAT format."""
        # Try HF cache first
        if self._hf_cache_dir and (self._hf_cache_dir / "data").exists():
            try:
                return self._load_from_hf()
            except Exception as e:
                logger.warning(f"Failed to load from HF cache, falling back to BRAT: {e}")

        # Fallback to BRAT files
        return self._load_from_brat()

    def _load_from_hf(self) -> dict[str, Dataset]:
        """Load from HuggingFace parquet cache."""
        logger.info(f"Loading {self.name} from HF cache: {self._hf_cache_dir}")
        ds = load_dataset("parquet", data_dir=str(self._hf_cache_dir / "data"))

        # Rename 'validation' -> 'val' for consistency
        splits = {}
        for split_name in ds.keys():
            out_name = "val" if split_name == "validation" else split_name
            splits[out_name] = ds[split_name]

        return splits

    def _load_from_brat(self) -> dict[str, Dataset]:
        """Load and split documents from BRAT format into train/val/test."""
        doc_texts, doc_codes, code_type = self._load_all_documents()

        # Find documents with both text and codes
        valid_docs = sorted(
            doc for doc in doc_texts
            if doc in doc_codes and doc_codes[doc]
        )

        if not valid_docs:
            raise ValueError("No valid documents found with both text and codes")

        # Count code frequencies
        code_freq: dict[str, int] = defaultdict(int)
        for doc in valid_docs:
            for code in doc_codes[doc]:
                code_freq[code] += 1

        # Filter by min_samples
        valid_codes = {
            code for code, count in code_freq.items()
            if count >= self.min_samples
        }

        # Apply top_k
        if self.top_k is not None:
            sorted_codes = sorted(
                code_freq.items(), key=lambda x: x[1], reverse=True
            )
            top_k_codes = {code for code, _ in sorted_codes[:self.top_k]}
            valid_codes = valid_codes & top_k_codes

        self._label_vocab = sorted(valid_codes)

        # Filter documents to those with at least one valid code
        filtered_docs = [
            doc for doc in valid_docs
            if any(code in valid_codes for code in doc_codes[doc])
        ]

        # Split documents
        doc_splits = _split_documents(filtered_docs, self.ratios, self.seed)

        # Build datasets
        datasets: dict[str, Dataset] = {}
        for split_name, split_docs in doc_splits.items():
            records = []
            for doc in split_docs:
                codes = [c for c in doc_codes[doc] if c in valid_codes]
                if codes:
                    records.append({
                        "text": doc_texts[doc],
                        "codes": codes,
                        "doc_name": doc,
                    })

            if records:
                datasets[split_name] = Dataset.from_list(records)
            else:
                datasets[split_name] = Dataset.from_dict({
                    "text": [],
                    "codes": [],
                    "doc_name": [],
                })

        return datasets

    def preprocess(self, dataset: Dataset, task: Any) -> Dataset:
        """Preprocess for multilabel task."""
        from tabib.tasks.multilabel import MultiLabelTask

        if not isinstance(task, MultiLabelTask):
            raise ValueError(
                f"FRASIMEDDocumentMultilabel requires MultiLabelTask, got {type(task)}. "
                "Use task: multilabel in your config."
            )

        if self._label_vocab:
            task.ensure_labels(self._label_vocab)

        label_map = task.label_space
        num_labels = task.num_labels

        def map_example(example: dict[str, Any]) -> dict[str, Any]:
            # Create multi-hot vector
            labels = [0.0] * num_labels
            for code in example["codes"]:
                if code in label_map:
                    labels[label_map[code]] = 1.0
            return {
                "text": example["text"],
                "labels": labels,
            }

        processed = dataset.map(map_example)
        processed.set_format(type="python")
        return processed


class FRASIMEDCANTEMISTDocAdapter(FRASIMEDDocumentMultilabelAdapter):
    """FRASIMED CANTEMIST document-level multilabel classification.

    Data is loaded from HuggingFace cache if available, otherwise from local BRAT files.
    Download with: huggingface-cli download rntc/tabib-frasimed-cantemist-doc --local-dir $SCRATCH/tabib/data/rntc--tabib-frasimed-cantemist-doc --repo-type dataset
    """

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("hf_repo", HF_REPO_CANTEMIST_DOC)
        super().__init__(subset="CANTEMIST-FRASIMED", **kwargs)


class FRASIMEDDISTEMISTDocAdapter(FRASIMEDDocumentMultilabelAdapter):
    """FRASIMED DISTEMIST document-level multilabel classification.

    Data is loaded from HuggingFace cache if available, otherwise from local BRAT files.
    Download with: huggingface-cli download rntc/tabib-frasimed-distemist-doc --local-dir $SCRATCH/tabib/data/rntc--tabib-frasimed-distemist-doc --repo-type dataset
    """

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("hf_repo", HF_REPO_DISTEMIST_DOC)
        super().__init__(subset="DISTEMIST-FRASIMED", **kwargs)
