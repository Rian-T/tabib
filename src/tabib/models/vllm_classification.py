"""vLLM-based classification model adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import weave

from tabib.models.base import ModelAdapter
from tabib.models.vllm_common import (
    VLLMEngine,
    build_messages,
    chat_with_vllm,
    create_vllm_engine,
)
from tabib.tasks.classification import ClassificationTask


@dataclass
class _VLLMResources:
    engine: VLLMEngine
    label_to_id: dict[str, int]
    prompt_template: str
    system_prompt: str | None
    enable_thinking: bool | None
    chat_template_kwargs: dict[str, Any]
    use_chat: bool  # True for chat models, False for completion models
    few_shot_examples: list[dict[str, Any]] | None = None  # For n-shot learning
    is_multi_answer: bool = False  # True for MCQM (multiple correct answers)


class VLLMClassificationAdapter(ModelAdapter):
    """Classification adapter that runs inference with vLLM."""

    def __init__(self) -> None:
        self._resources: _VLLMResources | None = None
        self._prompt_template: str = (
            "You are a helpful assistant that classifies news articles. "
            "Return only the label from the allowed set: {labels}.\n\n"
            "Article:\n{text}\n\nLabel:"
        )

    @property
    def name(self) -> str:
        return "vllm_classification"

    @property
    def supports_finetune(self) -> bool:
        return False

    @staticmethod
    def _parse_multi_answer(text: str) -> str:
        """Parse free-form multi-answer output to normalized format.

        Extracts letters A-E from text like "A, C, E" or "A C E" or "ACE"
        and returns normalized format "A C E" (sorted, space-separated).
        """
        import re
        # Extract all letters A-E (case insensitive)
        letters = re.findall(r'[A-Ea-e]', text)
        # Normalize: uppercase, unique, sorted
        unique_letters = sorted(set(letter.upper() for letter in letters))
        return " ".join(unique_letters)

    @staticmethod
    def _parse_single_answer(text: str, valid_labels: set[str]) -> str:
        """Parse free-form single-answer output to extract the answer.

        Handles various formats like:
        - "A" or "B" (direct answer)
        - "(A)" or "(B)" (parenthesized)
        - "The answer is A" or "La réponse est B"
        - "A. Some explanation"

        Returns the first valid label found, or empty string if none found.
        """
        import re
        text_upper = text.upper()
        # First try: exact match at start (after stripping)
        first_char = text_upper[0] if text_upper else ""
        if first_char in valid_labels:
            return first_char
        # Second try: parenthesized like "(A)" or "(B)"
        paren_match = re.search(r'\(([A-Z])\)', text_upper)
        if paren_match and paren_match.group(1) in valid_labels:
            return paren_match.group(1)
        # Third try: find first occurrence of valid label as standalone
        for match in re.finditer(r'\b([A-Z])\b', text_upper):
            if match.group(1) in valid_labels:
                return match.group(1)
        # Fourth try: any valid label character
        for char in text_upper:
            if char in valid_labels:
                return char
        return ""

    @weave.op()
    def _log_llm_call(self, input_text: str, prompt: str, output: str, expected_label: str | None = None) -> dict:
        """Log LLM call details to Weave for debugging."""
        return {
            "input_text": input_text,
            "prompt": prompt,
            "raw_output": output,
            "expected_label": expected_label,
        }

    def build_model(
        self,
        task: Any,
        model_name_or_path: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        prompt_template: str | None = None,
        system_prompt: str | None = None,
        enable_thinking: bool | None = True,
        sampling_overrides: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        use_chat: bool = True,  # Set to False for completion models (e.g., base models)
        train_data: Iterable[dict[str, Any]] | None = None,
        num_few_shot: int = 0,
        **kwargs: Any,
    ) -> _VLLMResources:
        if not isinstance(task, ClassificationTask):
            raise ValueError(f"Expected ClassificationTask, got {type(task)}")

        if not task.label_list:
            raise ValueError("ClassificationTask must provide label_list for vLLM inference")

        # Detect multi-answer task (labels like "A B", "A C D")
        is_multi_answer = any(" " in label for label in task.label_list)

        sampling_kwargs: dict[str, Any] = dict(sampling_overrides or {})

        if is_multi_answer:
            # Multi-answer: free-form generation, parse output later
            # Set max_tokens to limit output length
            sampling_kwargs.setdefault("max_tokens", 20)
            # Stop at double newline to prevent model from generating next question
            # Single \n might be part of answer, \n\n indicates new section
            sampling_kwargs.setdefault("stop", ["\n\n", "---"])
        else:
            # Single-answer: use structured outputs to force valid choice
            try:
                from vllm.sampling_params import StructuredOutputsParams
            except ImportError as exc:  # pragma: no cover - environment dependency
                raise ImportError(
                    "vLLM is required for VLLMClassificationAdapter. Install with `pip install vllm`."
                ) from exc
            sampling_kwargs["structured_outputs"] = StructuredOutputsParams(choice=list(task.label_list))

        engine = create_vllm_engine(
            model_name_or_path,
            sampling_overrides=sampling_kwargs,
            **kwargs,
        )

        # Build few-shot examples if train_data is provided (random selection)
        few_shot_examples = None
        if train_data and num_few_shot > 0:
            import random
            train_list = list(train_data)
            random.seed(42)  # Reproducible random selection
            few_shot_examples = random.sample(train_list, min(num_few_shot, len(train_list)))

        prompt_template_value = prompt_template or self._prompt_template
        resources = _VLLMResources(
            engine=engine,
            label_to_id={label: idx for idx, label in enumerate(task.label_list)},
            prompt_template=prompt_template_value,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            chat_template_kwargs=dict(chat_template_kwargs or {}),
            use_chat=use_chat,
            few_shot_examples=few_shot_examples,
            is_multi_answer=is_multi_answer,
        )
        self._resources = resources
        return resources

    def get_trainer(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - inference only
        raise NotImplementedError("vLLM classification does not support fine-tuning")

    def predict(self, model: Any, inputs: Iterable[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        resources = model if isinstance(model, _VLLMResources) else self._resources
        if resources is None:
            raise ValueError("Model not built. Call build_model first.")

        records = list(inputs)
        if not records:
            return {"predictions": np.array([], dtype=int), "label_ids": np.array([], dtype=int), "formatted_predictions": []}

        if resources.use_chat:
            # Chat model: use chat API with messages
            messages_batch = [
                build_messages(
                    self._format_prompt(example, resources),
                    system_prompt=resources.system_prompt,
                )
                for example in records
            ]

            chat_template_kwargs = dict(resources.chat_template_kwargs)
            chat_template_kwargs.update(kwargs.pop("chat_template_kwargs", {}))
            enable_thinking = kwargs.pop("enable_thinking", resources.enable_thinking)

            outputs = chat_with_vllm(
                resources.engine,
                messages_batch,
                enable_thinking=enable_thinking,
                chat_template_kwargs=chat_template_kwargs,
                **kwargs,
            )
        else:
            # Completion model: use generate API with prompts
            prompts = []
            for example in records:
                prompt = self._format_prompt(example, resources)
                # Add system prompt as prefix if provided
                if resources.system_prompt:
                    prompt = f"{resources.system_prompt}\n\n{prompt}"
                prompts.append(prompt)

            outputs = resources.engine.llm.generate(
                prompts,
                sampling_params=resources.engine.sampling_params,
                **kwargs,
            )

        predictions = []
        formatted = []
        # Build prompts for logging
        if resources.use_chat:
            prompts_for_log = [self._format_prompt(example, resources) for example in records]
        else:
            prompts_for_log = prompts  # Already built above

        for i, output in enumerate(outputs):
            raw_text = ""
            if output.outputs:
                raw_text = output.outputs[0].text.strip()

            if resources.is_multi_answer:
                # Parse free-form output to normalized format
                parsed_text = self._parse_multi_answer(raw_text)
                formatted.append(parsed_text)
                predictions.append(resources.label_to_id.get(parsed_text, -1))
            else:
                # Parse single answer - extract first valid label from response
                valid_labels = set(resources.label_to_id.keys())
                parsed_text = self._parse_single_answer(raw_text, valid_labels)
                formatted.append(parsed_text if parsed_text else raw_text)
                predictions.append(resources.label_to_id.get(parsed_text, -1))

            # Log to Weave for debugging (visible in Weave trace)
            input_text = records[i].get("text") or records[i].get("content") or ""
            expected_label = records[i].get("label_text") or records[i].get("label")
            self._log_llm_call(
                input_text=input_text,
                prompt=prompts_for_log[i],
                output=raw_text,
                expected_label=str(expected_label) if expected_label else None,
            )

        labels = [example.get("labels", -1) for example in records]

        result = {
            "predictions": np.array(predictions, dtype=int),
            "label_ids": np.array(labels, dtype=int),
            "formatted_predictions": formatted,
        }
        return result

    def _format_prompt(self, example: dict[str, Any], resources: _VLLMResources) -> str:
        text = example.get("text") or example.get("content")
        if text is None:
            raise ValueError("Dataset example must contain 'text' field for classification prompts")
        labels = ", ".join(resources.label_to_id.keys())

        if resources.few_shot_examples:
            # Few-shot: just examples flowing naturally into test (no extra instructions)
            # Instructions should be in system_prompt only
            examples_parts = []
            for ex in resources.few_shot_examples:
                ex_text = ex.get("text") or ex.get("content")
                ex_label = ex.get("label_text") or ex.get("label")
                if ex_text and ex_label:
                    if resources.is_multi_answer:
                        # Multi-answer: show natural format "A, C, E"
                        letters = ex_label.split()
                        natural_label = ", ".join(letters)
                        examples_parts.append(f"{ex_text}\n\nReponse: {natural_label}")
                    else:
                        examples_parts.append(f"{ex_text}\n\nReponse: {ex_label}")

            # Natural flow: examples -> test (same format)
            prompt = "\n\n---\n\n".join(examples_parts)
            prompt += f"\n\n---\n\n{text}\n\nReponse:"
            return prompt
        else:
            # Zero-shot: use template as-is
            return resources.prompt_template.format(text=text, labels=labels)
