"""Offline mode utilities for tabib.

Handles resolution of model/dataset paths for offline usage.
Default offline directory: $SCRATCH/tabib (if SCRATCH env var set).
"""

from __future__ import annotations

import os
from pathlib import Path


def get_offline_dir(config_value: str | None = None) -> Path | None:
    """Resolve offline directory from config or environment.

    Priority:
        1. Explicit config value
        2. $SCRATCH/tabib if SCRATCH env var is set
        3. None (no offline mode)

    Returns:
        Path to offline directory, or None if not configured.
    """
    if config_value:
        return Path(os.path.expandvars(config_value)).expanduser()

    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch) / "tabib"

    return None


def model_name_to_cache_path(model_name: str) -> str:
    """Convert HuggingFace model name to cache-safe directory name.

    Examples:
        'camembert-base' -> 'camembert-base'
        'almanach/camembert-bio-base' -> 'almanach--camembert-bio-base'
        'Dr-BERT/DrBERT-7GB' -> 'Dr-BERT--DrBERT-7GB'
    """
    return model_name.replace("/", "--")


def cache_path_to_model_name(cache_path: str) -> str:
    """Convert cache directory name back to HuggingFace model name.

    Examples:
        'almanach--camembert-bio-base' -> 'almanach/camembert-bio-base'
    """
    return cache_path.replace("--", "/")


def resolve_model_path(
    model_name_or_path: str,
    offline_dir: Path | None,
) -> str:
    """Resolve model path: use local cache if available, else HF name.

    Args:
        model_name_or_path: HuggingFace model ID or local path
        offline_dir: Base offline directory (contains models/ subdir)

    Returns:
        Local path if model exists in cache, else original model_name_or_path.
    """
    # Already a local path
    if Path(model_name_or_path).exists():
        return model_name_or_path

    # Check offline cache
    if offline_dir:
        cache_name = model_name_to_cache_path(model_name_or_path)
        local_path = offline_dir / "models" / cache_name

        if local_path.exists():
            return str(local_path)

    return model_name_or_path


def get_model_cache_dir(offline_dir: Path | None) -> Path | None:
    """Get the models subdirectory of offline_dir.

    Returns:
        Path to models/ subdir, or None if offline_dir is None.
    """
    if offline_dir:
        return offline_dir / "models"
    return None


def get_dataset_cache_dir(offline_dir: Path | None) -> Path | None:
    """Get the datasets subdirectory of offline_dir.

    Returns:
        Path to datasets/ subdir, or None if offline_dir is None.
    """
    if offline_dir:
        return offline_dir / "datasets"
    return None
