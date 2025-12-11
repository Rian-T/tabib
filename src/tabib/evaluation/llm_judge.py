"""LLM-as-a-Judge evaluator for open-ended QA."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from tabib.evaluation.base import Evaluator


@dataclass
class JudgmentResult:
    """Single judgment from the LLM judge."""
    correct: int  # 0 or 1
    score: int  # 0-10
    justification: str
    raw_output: str


DEFAULT_JUDGE_PROMPT = """You are a medical expert evaluating answers to clinical questions.

## Question
{question}

## Reference Answer
{reference}

## Candidate Answer
{candidate}

## Task
Evaluate the candidate answer compared to the reference. Consider:
- Medical accuracy
- Completeness of information
- Clinical relevance

## Required Output Format (JSON only)
{{"correct": 0 or 1, "score": 0-10, "justification": "brief explanation"}}
"""


def check_vram_availability(required_gb: float = 72.0) -> tuple[bool, float]:
    """Check if enough VRAM is available for parallel model loading.

    Args:
        required_gb: VRAM needed in GB for both models

    Returns:
        (can_parallel, available_gb)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False, 0.0
        free_bytes, _ = torch.cuda.mem_get_info()
        available_gb = free_bytes / (1024**3)
        return available_gb >= required_gb, available_gb
    except Exception:
        return False, 0.0


class LLMJudgeEvaluator(Evaluator):
    """Evaluator that uses an LLM to judge answer quality."""

    def __init__(
        self,
        judge_model: str = "google/medgemma-27b-text-it",
        temperature: float = 0.1,
        max_tokens: int = 512,
        prompt_template: str | None = None,
        system_prompt: str | None = None,
        gpu_memory_utilization: float = 0.6,
        **kwargs: Any,
    ) -> None:
        self.judge_model = judge_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template or DEFAULT_JUDGE_PROMPT
        self.system_prompt = system_prompt
        self.gpu_memory_utilization = gpu_memory_utilization
        self._engine = None
        self._last_judgments: list[JudgmentResult] = []

    def _ensure_engine(self):
        """Lazy-load the judge engine."""
        if self._engine is None:
            from tabib.models.vllm_common import create_vllm_engine
            self._engine = create_vllm_engine(
                self.judge_model,
                sampling_overrides={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                gpu_memory_utilization=self.gpu_memory_utilization,
            )
        return self._engine

    def evaluate(
        self,
        predictions: Any,
        references: Any,
        questions: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Evaluate predictions using LLM judge."""
        pred_texts = self._extract_predictions(predictions)
        ref_texts = self._extract_references(references)
        question_texts = list(questions) if questions else [""] * len(pred_texts)

        if len(pred_texts) != len(ref_texts):
            raise ValueError(f"Mismatched counts: {len(pred_texts)} vs {len(ref_texts)}")

        self._last_judgments = self._batch_judge(pred_texts, ref_texts, question_texts)
        return self._compute_metrics(self._last_judgments)

    def _batch_judge(
        self,
        predictions: list[str],
        references: list[str],
        questions: list[str],
    ) -> list[JudgmentResult]:
        """Run batch judgment with the LLM judge."""
        from tabib.models.vllm_common import build_messages, chat_with_vllm

        engine = self._ensure_engine()

        conversations = [
            build_messages(
                self.prompt_template.format(
                    question=q,
                    reference=r,
                    candidate=p,
                ),
                system_prompt=self.system_prompt,
            )
            for p, r, q in zip(predictions, references, questions)
        ]

        outputs = chat_with_vllm(engine, conversations, enable_thinking=False)

        judgments = []
        for output in outputs:
            raw = output.outputs[0].text.strip() if output.outputs else ""
            judgments.append(self._parse_judgment(raw))

        return judgments

    def _parse_judgment(self, raw: str) -> JudgmentResult:
        """Parse JSON judgment from LLM output."""
        try:
            json_match = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return JudgmentResult(
                    correct=int(data.get("correct", 0)),
                    score=int(data.get("score", 0)),
                    justification=str(data.get("justification", "")),
                    raw_output=raw,
                )
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
        return JudgmentResult(correct=0, score=0, justification="Parse error", raw_output=raw)

    def _compute_metrics(self, judgments: list[JudgmentResult]) -> dict[str, float]:
        """Compute aggregate metrics from judgments."""
        if not judgments:
            return {"judge_accuracy": 0.0, "judge_score_mean": 0.0, "judge_score_std": 0.0}

        correct_sum = sum(j.correct for j in judgments)
        scores = [j.score for j in judgments]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

        return {
            "judge_accuracy": correct_sum / len(judgments),
            "judge_score_mean": mean_score,
            "judge_score_std": variance ** 0.5,
        }

    def _extract_predictions(self, predictions: Any) -> list[str]:
        """Extract prediction texts from various formats."""
        if isinstance(predictions, dict):
            if "formatted_predictions" in predictions:
                return list(predictions["formatted_predictions"])
            return list(predictions.get("predictions", []))
        return list(predictions)

    def _extract_references(self, references: Any) -> list[str]:
        """Extract reference texts from various formats."""
        from datasets import Dataset
        if isinstance(references, Dataset):
            if "answer" in references.column_names:
                return [str(a) for a in references["answer"]]
            if "labels" in references.column_names:
                return [str(a) for a in references["labels"]]
        if isinstance(references, (list, tuple)):
            return [str(r) for r in references]
        return [str(references)]

    def get_detailed_results(self) -> list[dict[str, Any]]:
        """Return detailed per-sample results for analysis."""
        return [
            {"correct": j.correct, "score": j.score, "justification": j.justification}
            for j in self._last_judgments
        ]
