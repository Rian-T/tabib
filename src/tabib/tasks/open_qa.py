"""Open-ended question answering task."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Sequence

from datasets import Dataset

from tabib.tasks.base import Task


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _normalize_text(text: str) -> str:
    text = _strip_accents(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u00e0-\u00ff\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = prediction.split()
    ref_tokens = reference.split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    common = sum((pred_counter & ref_counter).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


class OpenQATask(Task):
    """Open-ended question answering evaluation.

    Predictions and references are compared using normalized exact match and a
    token-level F1 score (bag-of-words, lowercased, accent-stripped).
    """

    def __init__(self, normalize_answers: bool = True) -> None:
        self._normalize_answers = normalize_answers

    @property
    def name(self) -> str:
        return "open_qa"

    @property
    def label_space(self) -> dict[str, Any]:
        return {"type": "free_text"}

    def compute_metrics(self, predictions: Any, references: Any) -> dict[str, float]:
        pred_texts = self._extract_predictions(predictions)
        ref_texts = self._extract_references(predictions, references)

        if len(pred_texts) != len(ref_texts):
            raise ValueError(
                f"Mismatched prediction/reference counts: {len(pred_texts)} vs {len(ref_texts)}"
            )

        total = len(ref_texts)
        if total == 0:
            return {"exact_match": 0.0, "f1": 0.0}

        exact = 0.0
        f1_sum = 0.0
        for pred, ref in zip(pred_texts, ref_texts):
            if self._normalize_answers:
                norm_pred = _normalize_text(pred)
                norm_ref = _normalize_text(ref)
            else:
                norm_pred = pred.strip().lower()
                norm_ref = ref.strip().lower()

            if norm_pred == norm_ref:
                exact += 1.0

            f1_sum += _token_f1(norm_pred, norm_ref)

        return {
            "exact_match": exact / total,
            "f1": f1_sum / total,
        }

    def format_output(self, predictions: Any) -> list[str]:
        return self._extract_predictions(predictions)

    def _extract_predictions(self, predictions: Any) -> list[str]:
        if isinstance(predictions, dict):
            if "formatted_predictions" in predictions:
                raw_preds = predictions["formatted_predictions"]
            else:
                raw_preds = predictions.get("predictions", [])
        else:
            raw_preds = predictions

        if isinstance(raw_preds, Sequence) and not isinstance(raw_preds, str):
            return [str(item) for item in raw_preds]
        return [str(raw_preds)]

    def _extract_references(self, predictions: Any, references: Any) -> list[str]:
        if isinstance(predictions, dict) and "label_texts" in predictions:
            raw_refs = predictions["label_texts"]
        elif isinstance(predictions, dict) and "label_ids" in predictions and isinstance(references, Sequence):
            # label_ids may contain indices referring to provided references.
            raw_refs = [str(item) for item in references]
        elif isinstance(references, Dataset):
            if "answer" in references.column_names:
                raw_refs = [str(ans) for ans in references["answer"]]
            elif "labels" in references.column_names:
                raw_refs = [str(ans) for ans in references["labels"]]
            else:
                raise ValueError("OpenQATask expects 'answer' or 'labels' column in reference dataset.")
        elif isinstance(references, Sequence) and not isinstance(references, str):
            raw_refs = [str(item) for item in references]
        else:
            raw_refs = [str(references)]
        return raw_refs

