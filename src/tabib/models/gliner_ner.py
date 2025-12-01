"""GLiNER zero-shot NER adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Iterable

from gliner import GLiNER

from tabib.models.base import ModelAdapter
from tabib.tasks.ner_token import NERTokenTask

import torch


class GLiNERZeroShotNERAdapter(ModelAdapter):
    """Adapter that wraps a GLiNER model for zero-shot NER inference."""

    def __init__(self) -> None:
        self._task: NERTokenTask | None = None
        self._candidate_labels: list[str] | None = None
        self._label_mapping: dict[str, str] = {}
        self._normalize_labels: bool = True
        self._prediction_threshold: float | None = None
        self._max_length: int | None = None
        self._flat_ner: bool = True
        self._multi_label: bool = False
        self._batch_size: int = 16

    @property
    def name(self) -> str:
        """Return the model name."""

        return "gliner_zero_shot"

    @property
    def supports_finetune(self) -> bool:
        """GLiNER zero-shot models do not support fine-tuning in this adapter."""

        return False

    def build_model(
        self,
        task: Any,
        model_name_or_path: str = "urchade/gliner_medium-v2.1",
        **kwargs: Any,
    ) -> GLiNER:
        """Load a GLiNER model for zero-shot inference."""

        if not isinstance(task, NERTokenTask):
            raise ValueError(f"Expected NERTokenTask, got {type(task)}")

        candidate_labels = kwargs.pop("labels", None)
        if not candidate_labels:
            raise ValueError(
                "GLiNER zero-shot adapter requires `labels` in the configuration"
            )

        label_mapping = kwargs.pop("label_mapping", None)
        normalize_labels = kwargs.pop("normalize_labels", True)
        prediction_threshold = kwargs.pop("prediction_threshold", None)
        max_length = kwargs.pop("max_length", None)
        flat_ner = kwargs.pop("flat_ner", True)
        multi_label = kwargs.pop("multi_label", False)
        batch_size = kwargs.pop("batch_size", 16)

        # Backend configuration (torch-only)
        map_location = kwargs.pop(
            "map_location", "cuda" if torch.cuda.is_available() else "cpu"
        )
        compile_torch_model = kwargs.pop("compile_torch_model", False)

        if not isinstance(candidate_labels, Sequence):
            raise TypeError("`labels` must be a sequence of label names")

        self._task = task
        self._normalize_labels = bool(normalize_labels)
        self._prediction_threshold = (
            float(prediction_threshold) if prediction_threshold is not None else None
        )
        self._max_length = int(max_length) if max_length is not None else None
        self._flat_ner = bool(flat_ner)
        self._multi_label = bool(multi_label)
        self._batch_size = int(batch_size)

        processed_labels = [
            label.lower() if self._normalize_labels else str(label)
            for label in candidate_labels
        ]
        self._candidate_labels = list(processed_labels)

        if label_mapping is None:
            # Default: map label to its uppercase shorthand (person -> PERSON)
            label_mapping = {
                label: label.upper() for label in processed_labels
            }

        if not isinstance(label_mapping, dict):
            raise TypeError("`label_mapping` must be a dictionary when provided")

        normalized_mapping: dict[str, str] = {}
        for raw_label, target_label in label_mapping.items():
            key = raw_label.lower() if self._normalize_labels else str(raw_label)
            normalized_mapping[key] = str(target_label)

        self._label_mapping = normalized_mapping

        model = GLiNER.from_pretrained(
            model_name_or_path,
            map_location=map_location,
            compile_torch_model=compile_torch_model,
            **kwargs,
        )

        return model

    def get_trainer(
        self,
        model: Any,
        train_dataset: Any,
        eval_dataset: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """This adapter does not support fine-tuning."""

        return None

    def predict(self, model: GLiNER, inputs: Any, **kwargs: Any) -> dict[str, Any]:
        """Run GLiNER inference on tokenized datasets."""

        if self._task is None or self._candidate_labels is None:
            raise RuntimeError("Adapter not initialized. Did you call build_model()?")

        tokens_batches, reference_labels = self._extract_inputs(inputs)

        predictions: list[list[int]] = []
        raw_entities: list[list[dict[str, Any]]] = []

        texts: list[str] = []
        offset_batches: list[list[tuple[int, int]]] = []

        for tokens in tokens_batches:
            text, offsets = self._tokens_to_text(tokens)
            texts.append(text)
            offset_batches.append(offsets)

        prediction_kwargs: dict[str, Any] = {
            "flat_ner": self._flat_ner,
            "batch_size": self._batch_size,
            "multi_label": self._multi_label,
        }
        if self._prediction_threshold is not None:
            prediction_kwargs["threshold"] = self._prediction_threshold

        batched_entities = model.run(
            texts,
            self._candidate_labels,
            **prediction_kwargs,
        )

        for offsets, entities in zip(offset_batches, batched_entities):
            raw_entities.append(entities)
            predictions.append(self._spans_to_label_ids(entities, offsets))

        result: dict[str, Any] = {"predictions": predictions, "raw_entities": raw_entities}

        if reference_labels is not None:
            result["label_ids"] = reference_labels

        return result

    def _extract_inputs(
        self, inputs: Any
    ) -> tuple[list[list[str]], list[list[int]] | None]:
        """Extract token sequences (and optional labels) from the dataset."""

        tokens: list[list[str]]
        labels: list[list[int]] | None = None

        if hasattr(inputs, "column_names") and "tokens" in inputs.column_names:
            tokens = [list(example) for example in inputs["tokens"]]
            if "labels" in inputs.column_names:
                labels = [
                    [int(label) for label in example]
                    for example in inputs["labels"]
                ]
        elif isinstance(inputs, Sequence):
            tokens = []
            label_buffer: list[list[int]] = []
            for example in inputs:
                if not isinstance(example, dict) or "tokens" not in example:
                    raise ValueError("Each example must be a dict containing 'tokens'")
                tokens.append(list(example["tokens"]))
                if "labels" in example and example["labels"] is not None:
                    label_buffer.append([int(label) for label in example["labels"]])
            if label_buffer:
                labels = label_buffer
        else:
            raise TypeError(
                "Unsupported dataset format. Expected a datasets.Dataset or a sequence of dicts."
            )

        return tokens, labels

    def _tokens_to_text(self, tokens: Iterable[str]) -> tuple[str, list[tuple[int, int]]]:
        """Join tokens into a text string while tracking character offsets."""

        text_parts: list[str] = []
        offsets: list[tuple[int, int]] = []
        cursor = 0

        for idx, token in enumerate(tokens):
            if idx > 0:
                text_parts.append(" ")
                cursor += 1

            start = cursor
            text_parts.append(token)
            cursor += len(token)
            offsets.append((start, cursor))

        text = "".join(text_parts)
        return text, offsets

    def _spans_to_label_ids(
        self, spans: list[dict[str, Any]], offsets: list[tuple[int, int]]
    ) -> list[int]:
        """Convert GLiNER span predictions to token-level label IDs."""

        if self._task is None:
            raise RuntimeError("Task not initialized")

        num_tokens = len(offsets)
        o_id = self._task.label_to_id("O")
        label_ids = [o_id] * num_tokens

        for span in spans:
            label_name = span.get("label")
            if label_name is None:
                continue

            key = label_name.lower() if self._normalize_labels else str(label_name)
            target_label = self._label_mapping.get(key)
            if target_label is None:
                continue

            try:
                b_id = self._task.label_to_id(f"B-{target_label}")
                i_id = self._task.label_to_id(f"I-{target_label}")
            except KeyError:
                # Skip labels that are not part of the task's BIO space
                continue

            span_start = span.get("start", span.get("char_start"))
            span_end = span.get("end", span.get("char_end"))
            if span_start is None or span_end is None:
                continue

            token_indices = self._tokens_in_span(offsets, int(span_start), int(span_end))
            if not token_indices:
                continue

            first = True
            for idx in token_indices:
                if label_ids[idx] != o_id:
                    # Do not overwrite existing predictions (keep the first one)
                    continue
                label_ids[idx] = b_id if first else i_id
                first = False

        return label_ids

    @staticmethod
    def _tokens_in_span(
        offsets: list[tuple[int, int]], span_start: int, span_end: int
    ) -> list[int]:
        """Return indices of tokens whose offsets overlap with the char span."""

        covered: list[int] = []
        for idx, (token_start, token_end) in enumerate(offsets):
            if token_end <= span_start:
                continue
            if token_start >= span_end:
                continue
            covered.append(idx)

        return covered

