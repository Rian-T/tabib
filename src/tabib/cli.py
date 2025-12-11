"""CLI for tabib train|eval commands."""

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
import yaml

from tabib.comparison.benchmark import BenchmarkSpec, load_benchmark_spec, run_benchmark
from tabib.comparison.runner import execute_comparison
from tabib.comparison.spec import load_comparison_spec
from tabib.config import RunConfig
from tabib.pipeline import Pipeline

app = typer.Typer()
console = Console()


def load_config(config_path: str) -> RunConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        RunConfig instance
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path) as f:
        config_dict = yaml.safe_load(f)
    
    return RunConfig(**config_dict)


@app.command()
def train(config_path: str) -> None:
    """Train a model."""
    console.print(f"[bold]Loading config from {config_path}[/bold]")
    config = load_config(config_path)
    config.do_train = True
    
    console.print(f"[bold]Task:[/bold] {config.task}")
    console.print(f"[bold]Dataset:[/bold] {config.dataset}")
    console.print(f"[bold]Model:[/bold] {config.model}")
    
    pipeline = Pipeline(config)
    pipeline.run()


@app.command()
def eval(config_path: str) -> None:
    """Evaluate a model."""
    console.print(f"[bold]Loading config from {config_path}[/bold]")
    config = load_config(config_path)
    config.do_train = False
    config.do_eval = True
    
    console.print(f"[bold]Task:[/bold] {config.task}")
    console.print(f"[bold]Dataset:[/bold] {config.dataset}")
    console.print(f"[bold]Model:[/bold] {config.model}")
    
    pipeline = Pipeline(config)
    pipeline.run()


@app.command()
def compare(
    spec_path: str,
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional path for the JSON summary (overrides spec output_path).",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Only list the planned runs without executing them.",
    ),
) -> None:
    """Compare multiple models/backbones across datasets."""
    console.print(f"[bold]Loading comparison spec from {spec_path}[/bold]")
    spec = load_comparison_spec(spec_path)
    result = execute_comparison(spec, output_path=output, dry_run=dry_run)

    if dry_run:
        planned = result.metadata.get("runs", [])
        console.print(f"[bold]Planned runs:[/bold] {len(planned)}")
        for identifier in planned:
            console.print(f"  {identifier}")
    else:
        console.print("[bold]Comparison results:[/bold]")
        for experiment, dataset_map in result.summaries.items():
            console.print(f"  [bold cyan]{experiment}[/bold cyan]")
            for dataset_name, model_map in dataset_map.items():
                for model_name, summary in model_map.items():
                    metrics = (
                        summary.get("evaluation", {}).get("metrics")
                        if isinstance(summary, dict)
                        else None
                    ) or {}
                    metrics_str = _format_metrics(metrics)
                    console.print(
                        f"    {dataset_name} / {model_name}: {metrics_str}"
                    )

    output_path = result.metadata.get("output_path")
    if output_path:
        console.print(f"[bold]Results written to:[/bold] {output_path}")


def _format_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "no metrics"
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}={value:.4f}")
        else:
            formatted.append(f"{key}={value}")
    return ", ".join(formatted)


@app.command()
def download(
    config_path: str,
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for downloads. Default: $SCRATCH/tabib",
    ),
) -> None:
    """Download models for offline use.

    Supports both single run configs and benchmark specs.
    Downloads models to $SCRATCH/tabib/models/ by default.

    Examples:

        # Download model from a single config
        tabib download configs/ner_cas1_camembert.yaml

        # Download all models from a benchmark
        tabib download configs/benchmark_bert_drbenchmark.yaml

        # Specify custom output directory
        tabib download configs/benchmark.yaml -o /data/models
    """
    from tabib.download import download as do_download

    do_download(config_path, output_dir)


@app.command()
def benchmark(
    spec_path: str,
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Only list planned runs without executing them.",
    ),
    seeds: str | None = typer.Option(
        None,
        "--seeds",
        help="Comma-separated seeds for multi-seed averaging (e.g., '42,43,44'). Overrides spec.",
    ),
) -> None:
    """Run a benchmark comparing multiple model groups across tasks/datasets.

    Uses simplified spec format with model_groups for easy BERT vs LLM comparisons.
    Outputs results to JSON, Markdown, and optionally W&B.

    Supports multi-seed averaging via --seeds option or 'seeds' key in spec.
    Results show mean +/- std when multiple seeds are used.

    Example spec:

        seeds: [42, 43, 44]  # optional multi-seed
        datasets:
          ner: [emea, cas1]
          cls: [essai]
        model_groups:
          bert:
            configs: {ner: base/ner_bert.yaml}
            models: {camembert: camembert-base}
        output:
          markdown: results/benchmark.md

    Example with CLI seeds override:

        tabib benchmark spec.yaml --seeds 42,43,44,45,46
    """
    console.print(f"[bold]Loading benchmark spec from {spec_path}[/bold]")
    spec = load_benchmark_spec(spec_path)

    # Override seeds from CLI if provided
    if seeds is not None:
        parsed_seeds = [int(s.strip()) for s in seeds.split(",")]
        spec = BenchmarkSpec(
            path=spec.path,
            description=spec.description,
            datasets=spec.datasets,
            model_groups=spec.model_groups,
            output=spec.output,
            seeds=parsed_seeds,
        )
        console.print(f"[bold]Seeds (CLI override):[/bold] {parsed_seeds}")
    elif spec.seeds:
        console.print(f"[bold]Seeds:[/bold] {spec.seeds}")

    if spec.description:
        console.print(f"[bold]Description:[/bold] {spec.description}")

    runs = spec.expand_runs()
    console.print(f"[bold]Planned runs:[/bold] {len(runs)}")

    # Show breakdown if multi-seed
    if spec.seeds and len(spec.seeds) > 1:
        unique_configs = len(runs) // len(spec.seeds)
        console.print(f"  ({unique_configs} configs x {len(spec.seeds)} seeds)")

    if dry_run:
        for run in runs:
            console.print(f"  - {run.run_id}")
        return

    results = run_benchmark(spec, dry_run=False, verbose=True)

    # Print summary
    console.print("\n[bold green]Benchmark complete![/bold green]")
    console.print(f"  Successful: {sum(1 for r in results.results if r.status == 'success')}")
    console.print(f"  Errors: {sum(1 for r in results.results if r.status == 'error')}")

    # Show aggregated summary if multi-seed
    if results.has_multi_seed():
        console.print(f"\n[bold]Aggregated results (mean +/- std):[/bold]")
        for agg in results.aggregate_by_seed():
            primary = results._get_primary_metric(agg.task)
            if primary in agg.metrics_mean:
                mean = agg.metrics_mean[primary]
                std = agg.metrics_std.get(primary, 0.0)
                console.print(
                    f"  {agg.task}/{agg.dataset}/{agg.model_name}: "
                    f"{mean*100:.2f}% +/- {std*100:.2f}%"
                )

    if spec.output.markdown_path:
        console.print(f"\n[bold]Markdown report:[/bold] {spec.output.markdown_path}")
    if spec.output.json_path:
        console.print(f"[bold]JSON results:[/bold] {spec.output.json_path}")


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()

