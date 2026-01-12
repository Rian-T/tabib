"""Download utilities for offline mode.

Downloads models and datasets from HuggingFace Hub to local cache.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from tabib.offline import get_offline_dir, model_name_to_cache_path

console = Console()


def download_model(model_name: str, output_dir: Path) -> Path:
    """Download a model from HuggingFace Hub to local directory.

    Args:
        model_name: HuggingFace model ID (e.g., 'almanach/camembert-bio-base')
        output_dir: Base output directory (models will be in output_dir/models/)

    Returns:
        Path to downloaded model directory.
    """
    from huggingface_hub import snapshot_download

    cache_name = model_name_to_cache_path(model_name)
    target_dir = output_dir / "models" / cache_name

    if target_dir.exists():
        console.print(f"  [dim]Skipping {model_name} (already exists)[/dim]")
        return target_dir

    console.print(f"  Downloading [bold]{model_name}[/bold]...")

    snapshot_download(
        repo_id=model_name,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
    )

    console.print(f"    [green]Saved to {target_dir}[/green]")
    return target_dir


def download_from_config(config_path: Path, output_dir: Path) -> None:
    """Download model from a single run config.

    Args:
        config_path: Path to YAML config file
        output_dir: Output directory for downloads
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config.get("model_name_or_path")
    if model_name:
        download_model(model_name, output_dir)


def download_from_benchmark(spec_path: Path, output_dir: Path) -> None:
    """Download all models from a benchmark spec.

    Args:
        spec_path: Path to benchmark YAML config
        output_dir: Output directory for downloads
    """
    import yaml

    with open(spec_path) as f:
        spec = yaml.safe_load(f)

    model_groups = spec.get("model_groups", {})

    # Collect unique models
    all_models: set[str] = set()
    for group in model_groups.values():
        models = group.get("models", {})
        for model_path in models.values():
            all_models.add(model_path)

    console.print(f"Found [bold]{len(all_models)}[/bold] models to download")

    for model_name in sorted(all_models):
        download_model(model_name, output_dir)


def download(
    config_path: str,
    output_dir: str | None = None,
) -> None:
    """Main download function.

    Args:
        config_path: Path to config file (run config or benchmark spec)
        output_dir: Override output directory (default: $SCRATCH/tabib)
    """
    import yaml

    config_file = Path(config_path)
    if not config_file.exists():
        console.print(f"[red]Config file not found: {config_path}[/red]")
        return

    # Resolve output directory
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = get_offline_dir()
        if out_path is None:
            console.print(
                "[red]No output directory specified and $SCRATCH not set.[/red]\n"
                "Set SCRATCH environment variable or use --output-dir."
            )
            return

    out_path.mkdir(parents=True, exist_ok=True)
    console.print(f"Output directory: [bold]{out_path}[/bold]\n")

    # Detect config type
    with open(config_file) as f:
        config = yaml.safe_load(f)

    if "model_groups" in config:
        # Benchmark spec
        console.print(f"[bold]Benchmark config detected[/bold]: {config_file.name}")
        download_from_benchmark(config_file, out_path)
    elif "model_name_or_path" in config:
        # Single run config
        console.print(f"[bold]Run config detected[/bold]: {config_file.name}")
        download_from_config(config_file, out_path)
    else:
        console.print(f"[red]Unknown config format: {config_file}[/red]")

    console.print("\n[green]Download complete![/green]")


# Dataset repository mappings
DATASET_REPOS = {
    # NER datasets (BRAT format)
    "emea": "rntc/tabib-emea",
    "medline": "rntc/tabib-medline",
    "cas1": "rntc/tabib-cas1",
    "cas2": "rntc/tabib-cas2",
    "mantragsc": "rntc/tabib-mantragsc",
    # FRACCO variants
    "fracco_icd": "rntc/tabib-fracco-icd",
    "fracco_ner": "rntc/tabib-fracco-ner",
    # FRASIMED (NER + doc-level)
    "frasimed_cantemist": "rntc/tabib-frasimed-cantemist",
    "frasimed_distemist": "rntc/tabib-frasimed-distemist",
    "frasimed_cantemist_doc": "rntc/tabib-frasimed-cantemist-doc",
    "frasimed_distemist_doc": "rntc/tabib-frasimed-distemist-doc",
    # Other
    "meddialog": "rntc/tabib-meddialog-fr",
    "diamed": "DrBenchmark/DiaMED",  # External, loaded via datasets library
}


def download_dataset(dataset_name: str, output_dir: Path) -> Path:
    """Download a dataset from HuggingFace Hub.

    Args:
        dataset_name: Dataset key from DATASET_REPOS
        output_dir: Base output directory

    Returns:
        Path to downloaded dataset directory.
    """
    from huggingface_hub import snapshot_download

    if dataset_name not in DATASET_REPOS:
        console.print(f"[red]Unknown dataset: {dataset_name}[/red]")
        console.print(f"Available: {', '.join(sorted(DATASET_REPOS.keys()))}")
        return None

    repo_id = DATASET_REPOS[dataset_name]

    # Skip external datasets (loaded directly via datasets library)
    if not repo_id.startswith("rntc/"):
        console.print(f"  [dim]Skipping {dataset_name} (external: {repo_id})[/dim]")
        return None

    cache_name = repo_id.replace("/", "--")
    target_dir = output_dir / "data" / cache_name

    if target_dir.exists() and (target_dir / "data").exists():
        console.print(f"  [dim]Skipping {dataset_name} (already exists)[/dim]")
        return target_dir

    console.print(f"  Downloading [bold]{repo_id}[/bold]...")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        console.print(f"    [green]Saved to {target_dir}[/green]")
        return target_dir
    except Exception as e:
        console.print(f"    [red]Failed: {e}[/red]")
        return None


def download_all_datasets(output_dir: Path, datasets: list[str] | None = None) -> None:
    """Download all (or specified) datasets.

    Args:
        output_dir: Base output directory
        datasets: List of dataset names, or None for all
    """
    if datasets is None:
        datasets = list(DATASET_REPOS.keys())

    console.print(f"Downloading [bold]{len(datasets)}[/bold] datasets to {output_dir}/data/\n")

    success = 0
    for name in sorted(datasets):
        result = download_dataset(name, output_dir)
        if result:
            success += 1

    console.print(f"\n[green]Downloaded {success}/{len(datasets)} datasets[/green]")
