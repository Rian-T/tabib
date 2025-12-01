"""CLI for tabib train|eval commands."""

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
import yaml

from tabib.comparison.benchmark import load_benchmark_spec, run_benchmark
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
def benchmark(
    spec_path: str,
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Only list planned runs without executing them.",
    ),
) -> None:
    """Run a benchmark comparing multiple model groups across tasks/datasets.

    Uses simplified spec format with model_groups for easy BERT vs LLM comparisons.
    Outputs results to JSON, Markdown, and optionally W&B.

    Example spec:

        datasets:
          ner: [emea, cas1]
          cls: [essai]
        model_groups:
          bert:
            configs: {ner: base/ner_bert.yaml}
            models: {camembert: camembert-base}
        output:
          markdown: results/benchmark.md
    """
    console.print(f"[bold]Loading benchmark spec from {spec_path}[/bold]")
    spec = load_benchmark_spec(spec_path)

    if spec.description:
        console.print(f"[bold]Description:[/bold] {spec.description}")

    runs = spec.expand_runs()
    console.print(f"[bold]Planned runs:[/bold] {len(runs)}")

    if dry_run:
        for run in runs:
            console.print(f"  - {run.run_id}")
        return

    results = run_benchmark(spec, dry_run=False, verbose=True)

    # Print summary
    console.print("\n[bold green]Benchmark complete![/bold green]")
    console.print(f"  Successful: {sum(1 for r in results.results if r.status == 'success')}")
    console.print(f"  Errors: {sum(1 for r in results.results if r.status == 'error')}")

    if spec.output.markdown_path:
        console.print(f"\n[bold]Markdown report:[/bold] {spec.output.markdown_path}")
    if spec.output.json_path:
        console.print(f"[bold]JSON results:[/bold] {spec.output.json_path}")


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()

