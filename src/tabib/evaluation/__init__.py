"""Evaluation abstractions."""

from tabib.evaluation.base import Evaluator
from tabib.evaluation.span_evaluator import SpanEvaluator
from tabib.evaluation.llm_judge import LLMJudgeEvaluator, check_vram_availability

__all__ = ["Evaluator", "SpanEvaluator", "LLMJudgeEvaluator", "check_vram_availability"]

