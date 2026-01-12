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
import statistics
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from tabib.config import RunConfig
from tabib.pipeline import Pipeline


def _run_single_config(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Run a single config in a subprocess. Must be at module level for pickling."""
    run_config = RunConfig(**config_dict)
    pipeline = Pipeline(run_config)
    return pipeline.run()


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
    seed: int | None = None  # None means use config's default seed

    @property
    def run_id(self) -> str:
        base = f"{self.group}.{self.task}.{self.dataset}.{self.model_name}"
        return f"{base}.seed{self.seed}" if self.seed is not None else base

    @property
    def base_run_id(self) -> str:
        """Run ID without seed suffix, for grouping."""
        return f"{self.group}.{self.task}.{self.dataset}.{self.model_name}"


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    run: BenchmarkRun
    metrics: dict[str, Any]
    status: str = "success"
    error: str | None = None


@dataclass
class AggregatedResult:
    """Aggregated metrics across multiple seeds."""
    group: str
    task: str
    dataset: str
    model_name: str
    model_path: str
    seeds: list[int]
    num_seeds: int
    metrics_mean: dict[str, float]
    metrics_std: dict[str, float]

    @property
    def base_run_id(self) -> str:
        return f"{self.group}.{self.task}.{self.dataset}.{self.model_name}"


@dataclass
class BenchmarkSpec:
    """Parsed benchmark specification."""
    path: Path
    description: str | None
    datasets: dict[str, list[str]]  # task -> [dataset1, dataset2, ...]
    model_groups: dict[str, ModelGroupSpec]
    output: OutputSpec
    seeds: list[int] | None = None  # None = single run with base config seed
    parallel_seeds: int = 1  # Number of seeds to run in parallel (1 = sequential)

    def expand_runs(self) -> list[BenchmarkRun]:
        """Expand spec into individual run configurations.

        If seeds are specified, duplicates each run for each seed.
        """
        runs: list[BenchmarkRun] = []
        config_cache: dict[Path, dict[str, Any]] = {}

        # Determine seeds to run (None = single run with base config seed)
        seeds_to_run: list[int | None] = self.seeds if self.seeds else [None]

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
                        for seed in seeds_to_run:
                            config = self._build_config(
                                base_config,
                                group_name=group_name,
                                task=task,
                                dataset=dataset,
                                model_name=model_name,
                                model_path=model_path,
                                seed=seed,
                            )
                            runs.append(BenchmarkRun(
                                group=group_name,
                                task=task,
                                dataset=dataset,
                                model_name=model_name,
                                model_path=model_path,
                                config=config,
                                seed=seed,
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
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Build final config by substituting placeholders."""
        config = copy.deepcopy(base)

        # Substitute placeholders
        config["dataset"] = dataset
        config["model_name_or_path"] = model_path

        # Override seed if specified
        if seed is not None:
            if "training" not in config:
                config["training"] = {}
            config["training"]["seed"] = seed

        # Set output_dir dynamically (include seed in path to avoid conflicts)
        if "training" in config and config.get("do_train"):
            if seed is not None:
                output_dir = f"runs/{group_name}/{task}_{dataset}_{model_name}_seed{seed}"
            else:
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

    # Parse seeds (optional)
    raw_seeds = raw.get("seeds")
    seeds: list[int] | None = None
    if raw_seeds is not None:
        if isinstance(raw_seeds, list):
            seeds = [int(s) for s in raw_seeds]
        elif isinstance(raw_seeds, int):
            seeds = [raw_seeds]
        else:
            raise ValueError(f"'seeds' must be a list or int, got {type(raw_seeds)}")

    # Parse parallel_seeds (default: 1 = sequential)
    parallel_seeds = int(raw.get("parallel_seeds", 1))
    if parallel_seeds < 1:
        raise ValueError(f"'parallel_seeds' must be >= 1, got {parallel_seeds}")

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
        seeds=seeds,
        parallel_seeds=parallel_seeds,
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

    def has_multi_seed(self) -> bool:
        """Check if results contain multi-seed runs."""
        seeds_seen: set[int] = set()
        for r in self.results:
            if r.run.seed is not None:
                seeds_seen.add(r.run.seed)
        return len(seeds_seen) > 1

    def aggregate_by_seed(self) -> list[AggregatedResult]:
        """Aggregate results across seeds for each model/dataset combination.

        Returns list of AggregatedResult, one per unique (group, task, dataset, model).
        """
        # Group results by base_run_id
        grouped: dict[str, list[BenchmarkResult]] = defaultdict(list)
        for result in self.results:
            if result.status == "success":
                grouped[result.run.base_run_id].append(result)

        aggregated: list[AggregatedResult] = []
        for base_id, results in grouped.items():
            if not results:
                continue

            # Extract info from first result
            first = results[0]
            seeds = [r.run.seed for r in results if r.run.seed is not None]

            # Aggregate metrics
            metrics_all: dict[str, list[float]] = defaultdict(list)
            for r in results:
                for metric_name, value in r.metrics.items():
                    if isinstance(value, (int, float)):
                        metrics_all[metric_name].append(float(value))

            metrics_mean: dict[str, float] = {}
            metrics_std: dict[str, float] = {}
            for metric_name, values in metrics_all.items():
                if len(values) > 0:
                    metrics_mean[metric_name] = statistics.mean(values)
                    metrics_std[metric_name] = (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    )

            aggregated.append(AggregatedResult(
                group=first.run.group,
                task=first.run.task,
                dataset=first.run.dataset,
                model_name=first.run.model_name,
                model_path=first.run.model_path,
                seeds=seeds,
                num_seeds=len(results),
                metrics_mean=metrics_mean,
                metrics_std=metrics_std,
            ))

        return aggregated

    def to_dict(self) -> dict[str, Any]:
        """Convert results to nested dictionary structure.

        Includes both individual and aggregated results when multi-seed.
        """
        data: dict[str, Any] = {
            "metadata": self.metadata,
            "results": {},
        }

        # Individual results
        for r in self.results:
            data["results"].setdefault(r.run.group, {})
            data["results"][r.run.group].setdefault(r.run.task, {})
            data["results"][r.run.group][r.run.task].setdefault(r.run.dataset, {})

            # Key by model_name or model_name.seedN if multi-seed
            key = r.run.model_name
            if r.run.seed is not None:
                key = f"{r.run.model_name}.seed{r.run.seed}"

            data["results"][r.run.group][r.run.task][r.run.dataset][key] = {
                "metrics": r.metrics,
                "status": r.status,
                "error": r.error,
                "seed": r.run.seed,
            }

        # Aggregated results (if multi-seed)
        if self.has_multi_seed():
            data["aggregated"] = {}
            for agg in self.aggregate_by_seed():
                path = data["aggregated"].setdefault(agg.group, {})
                path = path.setdefault(agg.task, {})
                path = path.setdefault(agg.dataset, {})
                path[agg.model_name] = {
                    "seeds": agg.seeds,
                    "num_seeds": agg.num_seeds,
                    "metrics_mean": agg.metrics_mean,
                    "metrics_std": agg.metrics_std,
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
        """Generate markdown tables grouped by task.

        Shows mean +/- std when multi-seed results available.
        """
        lines: list[str] = []

        if self.spec.description:
            lines.append(f"# {self.spec.description}\n")

        lines.append(f"Generated: {self.metadata.get('generated_at', 'N/A')}\n")

        # Add seed info if multi-seed
        if self.has_multi_seed() and self.spec.seeds:
            lines.append(f"Seeds: {self.spec.seeds} (n={len(self.spec.seeds)})\n")

        # Use aggregated results if multi-seed, otherwise individual
        if self.has_multi_seed():
            return self._markdown_from_aggregated(lines)
        else:
            return self._markdown_from_individual(lines)

    def _markdown_from_aggregated(self, lines: list[str]) -> str:
        """Generate markdown from aggregated results (mean +/- std)."""
        aggregated = self.aggregate_by_seed()

        # Group by task
        by_task: dict[str, list[AggregatedResult]] = {}
        for agg in aggregated:
            by_task.setdefault(agg.task, []).append(agg)

        for task, task_results in by_task.items():
            lines.append(f"\n## {task.upper()}\n")

            datasets = sorted(set(r.dataset for r in task_results))
            models = sorted(set(r.model_name for r in task_results))
            primary_metric = self._get_primary_metric(task)

            header = "| Model | " + " | ".join(datasets) + " |"
            separator = "|-------|" + "|".join(["-------"] * len(datasets)) + "|"
            lines.append(header)
            lines.append(separator)

            for model in models:
                row_values = [model]
                for dataset in datasets:
                    matching = [
                        r for r in task_results
                        if r.model_name == model and r.dataset == dataset
                    ]
                    if matching:
                        agg = matching[0]
                        if primary_metric in agg.metrics_mean:
                            mean = agg.metrics_mean[primary_metric]
                            std = agg.metrics_std.get(primary_metric, 0.0)
                            row_values.append(f"{mean*100:.2f}% +/- {std*100:.2f}%")
                        else:
                            row_values.append("-")
                    else:
                        row_values.append("-")
                lines.append("| " + " | ".join(row_values) + " |")

        return "\n".join(lines)

    def _markdown_from_individual(self, lines: list[str]) -> str:
        """Generate markdown from individual results (single values)."""
        # Group by task
        by_task: dict[str, list[BenchmarkResult]] = {}
        for r in self.results:
            by_task.setdefault(r.run.task, []).append(r)

        for task, task_results in by_task.items():
            lines.append(f"\n## {task.upper()}\n")

            datasets = sorted(set(r.run.dataset for r in task_results))
            models = sorted(set(r.run.model_name for r in task_results))
            primary_metric = self._get_primary_metric(task)

            header = "| Model | " + " | ".join(datasets) + " |"
            separator = "|-------|" + "|".join(["-------"] * len(datasets)) + "|"
            lines.append(header)
            lines.append(separator)

            for model in models:
                row_values = [model]
                for dataset in datasets:
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
            "ner": "seqeval_f1",
            "ner_span": "seqeval_f1",
            "ner_token": "f1",  # NERTokenTask uses f1 directly from seqeval compute_metrics
            "cls": "f1",
            "classification": "f1",
            "multilabel": "f1_micro",  # Multi-label uses micro-averaged F1
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
        """Upload results to W&B as a table.

        If multi-seed, uploads aggregated results (mean/std) only.
        """
        try:
            import wandb
        except ImportError:
            print("Warning: wandb not installed, skipping upload")
            return

        run = wandb.init(project=project, job_type="benchmark")

        if self.has_multi_seed():
            # Aggregated results table with mean/std
            table = wandb.Table(
                columns=["group", "task", "dataset", "model", "num_seeds", "metric", "mean", "std"]
            )
            for agg in self.aggregate_by_seed():
                primary_metric = self._get_primary_metric(agg.task)
                if primary_metric in agg.metrics_mean:
                    table.add_data(
                        agg.group,
                        agg.task,
                        agg.dataset,
                        agg.model_name,
                        agg.num_seeds,
                        primary_metric,
                        agg.metrics_mean[primary_metric],
                        agg.metrics_std.get(primary_metric, 0.0),
                    )
        else:
            # Individual results table (single seed)
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
    """Execute a benchmark specification.

    If parallel_seeds > 1, runs multiple seeds for the same config in parallel
    before moving to the next config. This allows efficient GPU utilization.
    """
    runs = spec.expand_runs()

    results = BenchmarkResults(
        spec=spec,
        metadata={
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "spec_path": str(spec.path),
            "description": spec.description,
            "num_runs": len(runs),
            "dry_run": dry_run,
            "parallel_seeds": spec.parallel_seeds,
        },
    )

    if dry_run:
        if verbose:
            print(f"Dry run: {len(runs)} runs planned (parallel_seeds={spec.parallel_seeds})")
            for run in runs:
                print(f"  - {run.run_id}")
        return results

    # Group runs by base_run_id (same config, different seeds)
    grouped: dict[str, list[BenchmarkRun]] = defaultdict(list)
    for run in runs:
        grouped[run.base_run_id].append(run)

    parallel_seeds = spec.parallel_seeds
    total_configs = len(grouped)
    config_idx = 0

    for base_id, seed_runs in grouped.items():
        config_idx += 1
        num_seeds = len(seed_runs)

        if verbose:
            print(f"\n[Config {config_idx}/{total_configs}] {base_id} ({num_seeds} seeds)")

        if parallel_seeds == 1:
            # Sequential execution (original behavior)
            for run in seed_runs:
                if verbose:
                    print(f"  Running seed {run.seed}...")
                try:
                    summary = _run_single_config(run.config)
                    metrics = summary.get("evaluation", {}).get("metrics", {})
                    results.add_result(BenchmarkResult(
                        run=run,
                        metrics=metrics,
                        status="success",
                    ))
                    if verbose:
                        primary = results._get_primary_metric(run.task)
                        if primary in metrics:
                            print(f"    {primary}: {metrics[primary]:.4f}")
                except Exception as e:
                    if verbose:
                        print(f"    ERROR: {e}")
                    results.add_result(BenchmarkResult(
                        run=run,
                        metrics={},
                        status="error",
                        error=str(e),
                    ))
        else:
            # Parallel execution using ProcessPoolExecutor
            if verbose:
                print(f"  Running {num_seeds} seeds in parallel (max {parallel_seeds})...")

            # Process in batches of parallel_seeds
            for batch_start in range(0, num_seeds, parallel_seeds):
                batch = seed_runs[batch_start:batch_start + parallel_seeds]
                batch_seeds = [r.seed for r in batch]

                if verbose and len(seed_runs) > parallel_seeds:
                    print(f"  Batch: seeds {batch_seeds}")

                with ProcessPoolExecutor(max_workers=len(batch)) as executor:
                    futures = {
                        executor.submit(_run_single_config, run.config): run
                        for run in batch
                    }

                    for future in as_completed(futures):
                        run = futures[future]
                        try:
                            summary = future.result()
                            metrics = summary.get("evaluation", {}).get("metrics", {})
                            results.add_result(BenchmarkResult(
                                run=run,
                                metrics=metrics,
                                status="success",
                            ))
                            if verbose:
                                primary = results._get_primary_metric(run.task)
                                if primary in metrics:
                                    print(f"    seed {run.seed}: {primary}={metrics[primary]:.4f}")
                        except Exception as e:
                            if verbose:
                                print(f"    seed {run.seed} ERROR: {e}")
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
