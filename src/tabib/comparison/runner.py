"""Comparison runner that executes multiple configs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tabib.comparison.spec import ComparisonSpec, ExpandedRun
from tabib.config import RunConfig
from tabib.pipeline import Pipeline


@dataclass
class ComparisonResult:
    spec: ComparisonSpec
    runs: list[ExpandedRun]
    summaries: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "experiments": self.summaries,
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, ensure_ascii=False)


def execute_comparison(
    spec: ComparisonSpec,
    *,
    output_path: str | Path | None = None,
    dry_run: bool = False,
) -> ComparisonResult:
    runs = spec.expand_runs()

    metadata: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "spec_path": str(spec.path),
        "description": spec.description,
        "dry_run": dry_run,
        "num_runs": len(runs),
        "experiments": list(spec.experiments.keys()),
    }

    summaries: dict[str, Any] = {}

    if dry_run:
        metadata["runs"] = [run.identifier for run in runs]
    else:
        for run in runs:
            run_summary = _execute_run(run)
            summaries.setdefault(run.experiment, {}).setdefault(run.dataset, {})[
                run.model_variant
            ] = run_summary

    result = ComparisonResult(
        spec=spec,
        runs=runs,
        summaries=summaries,
        metadata=metadata,
    )

    resolved_output: Path | None
    if output_path is not None:
        resolved_output = Path(output_path).expanduser().resolve()
    else:
        resolved_output = spec.output_path

    if resolved_output:
        metadata["output_path"] = str(resolved_output)
        result.write_json(resolved_output)

    return result


def _execute_run(run: ExpandedRun) -> dict[str, Any]:
    run_config = RunConfig(**run.config)
    pipeline = Pipeline(run_config)
    summary = pipeline.run()
    summary = dict(summary)
    summary.setdefault("run", {})
    summary["run"].update(
        {
            "experiment": run.experiment,
            "dataset": run.dataset,
            "model_variant": run.model_variant,
            "config_path": str(run.config_path),
        }
    )
    return summary

