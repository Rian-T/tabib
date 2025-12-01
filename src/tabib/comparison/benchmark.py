"""Simplified benchmark specification and runner.

Example benchmark spec:
```yaml
description: DrBenchmark - French biomedical NLP

datasets:
  ner: [emea, cas1, cas2]
  cls: [essai, diamed]

model_groups:
  bert:
    configs:
      ner: base/ner_bert.yaml
      cls: base/cls_bert.yaml
    models:
      camembert: camembert-base
      camembert-bio: almanach/camembert-bio-base

output:
  json: results/benchmark.json
  markdown: results/benchmark.md
```
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from tabib.config import RunConfig
from tabib.pipeline import Pipeline


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass
class BenchmarkRun:
    """A single benchmark run configuration."""
    group: str
    task: str
    dataset: str
    model_name: str
    model_path: str
    config: dict[str, Any]

    @property
    def run_id(self) -> str:
        return f"{self.group}.{self.task}.{self.dataset}.{self.model_name}"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    run: BenchmarkRun
    metrics: dict[str, Any]
    status: str = "success"
    error: str | None = None


@dataclass
class BenchmarkSpec:
    """Parsed benchmark specification."""
    path: Path
    description: str | None
    datasets: dict[str, list[str]]  # task -> [dataset1, dataset2, ...]
    model_groups: dict[str, ModelGroupSpec]
    output: OutputSpec

    def expand_runs(self) -> list[BenchmarkRun]:
        """Expand spec into individual run configurations."""
        runs: list[BenchmarkRun] = []
        config_cache: dict[Path, dict[str, Any]] = {}

        for group_name, group in self.model_groups.items():
            for task, datasets in self.datasets.items():
                if task not in group.configs:
                    continue  # This group doesn't support this task

                base_config_path = self.path.parent / group.configs[task]
                if base_config_path not in config_cache:
                    config_cache[base_config_path] = _load_yaml(base_config_path)
                base_config = config_cache[base_config_path]

                for dataset in datasets:
                    for model_name, model_path in group.models.items():
                        config = self._build_config(
                            base_config,
                            group_name=group_name,
                            task=task,
                            dataset=dataset,
                            model_name=model_name,
                            model_path=model_path,
                        )
                        runs.append(BenchmarkRun(
                            group=group_name,
                            task=task,
                            dataset=dataset,
                            model_name=model_name,
                            model_path=model_path,
                            config=config,
                        ))
        return runs

    def _build_config(
        self,
        base: dict[str, Any],
        *,
        group_name: str,
        task: str,
        dataset: str,
        model_name: str,
        model_path: str,
    ) -> dict[str, Any]:
        """Build final config by substituting placeholders."""
        config = copy.deepcopy(base)

        # Substitute placeholders
        config["dataset"] = dataset
        config["model_name_or_path"] = model_path

        # Set output_dir dynamically
        if "training" in config and config.get("do_train"):
            output_dir = f"runs/{group_name}/{task}_{dataset}_{model_name}"
            config["training"]["output_dir"] = output_dir

        return config


@dataclass
class ModelGroupSpec:
    """Specification for a model group."""
    configs: dict[str, str]  # task -> config_path
    models: dict[str, str]   # model_name -> model_path


@dataclass
class OutputSpec:
    """Output configuration."""
    json_path: Path | None = None
    markdown_path: Path | None = None
    wandb_project: str | None = None
    wandb_table: str | None = None


def load_benchmark_spec(path: str | Path) -> BenchmarkSpec:
    """Load and parse a benchmark specification file."""
    spec_path = Path(path).expanduser().resolve()
    raw = _load_yaml(spec_path)

    description = raw.get("description")

    # Parse datasets
    raw_datasets = raw.get("datasets", {})
    if not isinstance(raw_datasets, dict):
        raise ValueError("'datasets' must be a mapping of task -> [dataset1, ...]")
    datasets: dict[str, list[str]] = {}
    for task, ds_list in raw_datasets.items():
        if isinstance(ds_list, list):
            datasets[task] = [str(d) for d in ds_list]
        else:
            datasets[task] = [str(ds_list)]

    # Parse model_groups
    raw_groups = raw.get("model_groups", {})
    if not isinstance(raw_groups, dict):
        raise ValueError("'model_groups' must be a mapping")
    model_groups: dict[str, ModelGroupSpec] = {}
    for group_name, group_raw in raw_groups.items():
        model_groups[group_name] = _parse_model_group(group_raw)

    # Parse output
    output = _parse_output(raw.get("output", {}), spec_path.parent)

    return BenchmarkSpec(
        path=spec_path,
        description=description,
        datasets=datasets,
        model_groups=model_groups,
        output=output,
    )


def _parse_model_group(raw: dict[str, Any]) -> ModelGroupSpec:
    """Parse a model group specification."""
    configs = raw.get("configs", {})
    if not isinstance(configs, dict):
        raise ValueError("model_group 'configs' must be a mapping")

    models = raw.get("models", {})
    if not isinstance(models, dict):
        raise ValueError("model_group 'models' must be a mapping")

    return ModelGroupSpec(
        configs={str(k): str(v) for k, v in configs.items()},
        models={str(k): str(v) for k, v in models.items()},
    )


def _parse_output(raw: dict[str, Any], base_dir: Path) -> OutputSpec:
    """Parse output specification."""
    json_path = None
    markdown_path = None
    wandb_project = None
    wandb_table = None

    if "json" in raw:
        json_path = (base_dir / raw["json"]).resolve()
    if "markdown" in raw:
        markdown_path = (base_dir / raw["markdown"]).resolve()
    if "wandb" in raw:
        wb = raw["wandb"]
        if isinstance(wb, dict):
            wandb_project = wb.get("project")
            wandb_table = wb.get("table")

    return OutputSpec(
        json_path=json_path,
        markdown_path=markdown_path,
        wandb_project=wandb_project,
        wandb_table=wandb_table,
    )


@dataclass
class BenchmarkResults:
    """Collection of benchmark results."""
    spec: BenchmarkSpec
    results: list[BenchmarkResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_result(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def to_dict(self) -> dict[str, Any]:
        """Convert results to nested dictionary structure."""
        data: dict[str, Any] = {
            "metadata": self.metadata,
            "results": {},
        }

        for r in self.results:
            data["results"].setdefault(r.run.group, {})
            data["results"][r.run.group].setdefault(r.run.task, {})
            data["results"][r.run.group][r.run.task].setdefault(r.run.dataset, {})
            data["results"][r.run.group][r.run.task][r.run.dataset][r.run.model_name] = {
                "metrics": r.metrics,
                "status": r.status,
                "error": r.error,
            }

        return data

    def to_flat_table(self) -> list[dict[str, Any]]:
        """Convert results to flat table format for DataFrame/W&B."""
        rows: list[dict[str, Any]] = []
        for r in self.results:
            row = {
                "group": r.run.group,
                "task": r.run.task,
                "dataset": r.run.dataset,
                "model": r.run.model_name,
                "model_path": r.run.model_path,
                "status": r.status,
            }
            # Flatten metrics
            for k, v in r.metrics.items():
                if isinstance(v, (int, float, str, bool)):
                    row[k] = v
            rows.append(row)
        return rows

    def to_markdown(self) -> str:
        """Generate markdown tables grouped by task."""
        lines: list[str] = []

        if self.spec.description:
            lines.append(f"# {self.spec.description}\n")

        lines.append(f"Generated: {self.metadata.get('generated_at', 'N/A')}\n")

        # Group by task
        by_task: dict[str, list[BenchmarkResult]] = {}
        for r in self.results:
            by_task.setdefault(r.run.task, []).append(r)

        for task, task_results in by_task.items():
            lines.append(f"\n## {task.upper()}\n")

            # Get unique datasets and models
            datasets = sorted(set(r.run.dataset for r in task_results))
            models = sorted(set(r.run.model_name for r in task_results))

            # Determine primary metric
            primary_metric = self._get_primary_metric(task)

            # Build table header
            header = "| Model | " + " | ".join(datasets) + " |"
            separator = "|-------|" + "|".join(["-------"] * len(datasets)) + "|"
            lines.append(header)
            lines.append(separator)

            # Build table rows
            for model in models:
                row_values = [model]
                for dataset in datasets:
                    # Find result for this model+dataset
                    matching = [
                        r for r in task_results
                        if r.run.model_name == model and r.run.dataset == dataset
                    ]
                    if matching:
                        r = matching[0]
                        if r.status == "success" and primary_metric in r.metrics:
                            val = r.metrics[primary_metric]
                            if isinstance(val, float):
                                row_values.append(f"{val*100:.2f}%")
                            else:
                                row_values.append(str(val))
                        elif r.status == "error":
                            row_values.append("ERROR")
                        else:
                            row_values.append("-")
                    else:
                        row_values.append("-")
                lines.append("| " + " | ".join(row_values) + " |")

        return "\n".join(lines)

    def _get_primary_metric(self, task: str) -> str:
        """Get the primary metric for a task."""
        metric_map = {
            "ner": "exact_f1",
            "ner_span": "exact_f1",
            "cls": "f1",
            "classification": "f1",
            "mcqa": "accuracy",
            "sim": "spearman",
            "similarity": "spearman",
        }
        return metric_map.get(task, "f1")

    def write_json(self, path: Path) -> None:
        """Write results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def write_markdown(self, path: Path) -> None:
        """Write results to Markdown file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write(self.to_markdown())

    def upload_to_wandb(self, project: str, table_name: str) -> None:
        """Upload results to W&B as a table."""
        try:
            import wandb
        except ImportError:
            print("Warning: wandb not installed, skipping upload")
            return

        run = wandb.init(project=project, job_type="benchmark")
        table = wandb.Table(
            columns=["group", "task", "dataset", "model", "metric", "value"]
        )

        for r in self.results:
            if r.status != "success":
                continue
            primary_metric = self._get_primary_metric(r.run.task)
            if primary_metric in r.metrics:
                table.add_data(
                    r.run.group,
                    r.run.task,
                    r.run.dataset,
                    r.run.model_name,
                    primary_metric,
                    r.metrics[primary_metric],
                )

        run.log({table_name: table})
        run.finish()


def run_benchmark(
    spec: BenchmarkSpec,
    *,
    dry_run: bool = False,
    verbose: bool = True,
) -> BenchmarkResults:
    """Execute a benchmark specification."""
    runs = spec.expand_runs()

    results = BenchmarkResults(
        spec=spec,
        metadata={
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "spec_path": str(spec.path),
            "description": spec.description,
            "num_runs": len(runs),
            "dry_run": dry_run,
        },
    )

    if dry_run:
        if verbose:
            print(f"Dry run: {len(runs)} runs planned")
            for run in runs:
                print(f"  - {run.run_id}")
        return results

    for i, run in enumerate(runs, 1):
        if verbose:
            print(f"\n[{i}/{len(runs)}] Running {run.run_id}...")

        try:
            run_config = RunConfig(**run.config)
            pipeline = Pipeline(run_config)
            summary = pipeline.run()

            metrics = summary.get("evaluation", {}).get("metrics", {})
            results.add_result(BenchmarkResult(
                run=run,
                metrics=metrics,
                status="success",
            ))

            if verbose:
                primary = results._get_primary_metric(run.task)
                if primary in metrics:
                    print(f"  {primary}: {metrics[primary]:.4f}")

        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            results.add_result(BenchmarkResult(
                run=run,
                metrics={},
                status="error",
                error=str(e),
            ))

    # Write outputs
    if spec.output.json_path:
        results.write_json(spec.output.json_path)
        if verbose:
            print(f"\nJSON results: {spec.output.json_path}")

    if spec.output.markdown_path:
        results.write_markdown(spec.output.markdown_path)
        if verbose:
            print(f"Markdown results: {spec.output.markdown_path}")

    if spec.output.wandb_project and spec.output.wandb_table:
        results.upload_to_wandb(spec.output.wandb_project, spec.output.wandb_table)
        if verbose:
            print(f"W&B upload: {spec.output.wandb_project}/{spec.output.wandb_table}")

    return results
