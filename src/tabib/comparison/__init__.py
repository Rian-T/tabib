"""Comparison utilities for Tabib."""

from tabib.comparison.benchmark import (
    BenchmarkResult,
    BenchmarkResults,
    BenchmarkRun,
    BenchmarkSpec,
    load_benchmark_spec,
    run_benchmark,
)
from tabib.comparison.spec import (
    ComparisonSpec,
    ExpandedRun,
    load_comparison_spec,
)

__all__ = [
    # New benchmark system
    "BenchmarkResult",
    "BenchmarkResults",
    "BenchmarkRun",
    "BenchmarkSpec",
    "load_benchmark_spec",
    "run_benchmark",
    # Legacy comparison system
    "ComparisonSpec",
    "ExpandedRun",
    "load_comparison_spec",
]

