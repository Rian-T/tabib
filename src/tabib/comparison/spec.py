"""Comparison specification loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overrides into base, returning a new dict."""
    result: dict[str, Any] = {**base}
    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    config_path: Path
    overrides: dict[str, Any]
    extras: dict[str, Any]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    dataset_names: list[str]
    models: list[ModelSpec]


@dataclass(frozen=True)
class ExpandedRun:
    experiment: str
    dataset: str
    model_variant: str
    config_path: Path
    config: dict[str, Any]

    @property
    def identifier(self) -> str:
        return f"{self.experiment}.{self.dataset}.{self.model_variant}"


@dataclass(frozen=True)
class ComparisonSpec:
    path: Path
    defaults: dict[str, Any]
    output_path: Path | None
    datasets: dict[str, DatasetSpec]
    experiments: dict[str, ExperimentSpec]
    description: str | None = None

    def expand_runs(self) -> list[ExpandedRun]:
        cache: dict[Path, dict[str, Any]] = {}
        expanded: list[ExpandedRun] = []

        for experiment in self.experiments.values():
            for dataset_name in experiment.dataset_names:
                dataset = self._get_dataset(dataset_name)
                base_config = self._load_dataset_config(dataset, cache)
                for model_spec in experiment.models:
                    merged_config = self._build_run_config(
                        base_config, dataset, model_spec
                    )
                    expanded.append(
                        ExpandedRun(
                            experiment=experiment.name,
                            dataset=dataset.name,
                            model_variant=model_spec.name,
                            config_path=dataset.config_path,
                            config=merged_config,
                        )
                    )
        return expanded

    def _get_dataset(self, name: str) -> DatasetSpec:
        if name not in self.datasets:
            raise KeyError(f"Experiment references unknown dataset '{name}'")
        return self.datasets[name]

    def _load_dataset_config(
        self,
        dataset: DatasetSpec,
        cache: dict[Path, dict[str, Any]],
    ) -> dict[str, Any]:
        if dataset.config_path not in cache:
            cache[dataset.config_path] = _load_yaml(dataset.config_path)
        base_config = cache[dataset.config_path]
        merged = _deep_merge(self.defaults, base_config)
        if dataset.extras:
            merged = _deep_merge(merged, dataset.extras)
        if dataset.overrides:
            merged = _deep_merge(merged, dataset.overrides)
        return merged

    @staticmethod
    def _build_run_config(
        base_config: dict[str, Any],
        dataset: DatasetSpec,
        model_spec: ModelSpec,
    ) -> dict[str, Any]:
        merged = _deep_merge(base_config, model_spec.overrides)
        return merged


def load_comparison_spec(path: str | Path) -> ComparisonSpec:
    spec_path = Path(path).expanduser().resolve()
    raw_spec = _load_yaml(spec_path)

    defaults = raw_spec.get("defaults", {}) or {}
    description = raw_spec.get("description")
    output_path_raw = raw_spec.get("output_path")
    output_path = (
        (spec_path.parent / output_path_raw).resolve()
        if output_path_raw
        else None
    )

    raw_datasets = raw_spec.get("datasets")
    raw_experiments = raw_spec.get("experiments")

    if not isinstance(raw_datasets, dict) or not raw_datasets:
        raise ValueError("Comparison spec must provide non-empty 'datasets' mapping")
    if not isinstance(raw_experiments, dict) or not raw_experiments:
        raise ValueError("Comparison spec must provide non-empty 'experiments' mapping")

    datasets: dict[str, DatasetSpec] = {}
    for dataset_name, dataset_raw in raw_datasets.items():
        datasets[dataset_name] = _parse_dataset_spec(
            dataset_name, dataset_raw, spec_path.parent
        )

    experiments: dict[str, ExperimentSpec] = {}
    for experiment_name, experiment_raw in raw_experiments.items():
        experiments[experiment_name] = _parse_experiment_spec(
            experiment_name, experiment_raw
        )

    return ComparisonSpec(
        path=spec_path,
        defaults=defaults,
        output_path=output_path,
        datasets=datasets,
        experiments=experiments,
        description=description,
    )


def _parse_dataset_spec(
    name: str,
    raw: Any,
    base_dir: Path,
) -> DatasetSpec:
    if isinstance(raw, str):
        config_path = (base_dir / raw).expanduser().resolve()
        return DatasetSpec(
            name=name,
            config_path=config_path,
            overrides={},
            extras={},
        )
    if not isinstance(raw, dict):
        raise TypeError(f"Dataset '{name}' entry must be a string or mapping")

    config_raw = raw.get("config")
    if config_raw is None:
        raise ValueError(f"Dataset '{name}' mapping must include 'config'")
    config_path = (base_dir / config_raw).expanduser().resolve()

    overrides = raw.get("overrides", {}) or {}
    if not isinstance(overrides, dict):
        raise TypeError(f"Dataset '{name}' overrides must be a mapping")

    extras = {
        key: value
        for key, value in raw.items()
        if key not in {"config", "overrides"}
    }

    return DatasetSpec(
        name=name,
        config_path=config_path,
        overrides=overrides,
        extras=extras,
    )


def _parse_experiment_spec(
    name: str,
    raw: Any,
) -> ExperimentSpec:
    if not isinstance(raw, dict):
        raise TypeError(f"Experiment '{name}' entry must be a mapping")

    datasets_raw = raw.get("datasets")
    models_raw = raw.get("models")

    if not datasets_raw or not isinstance(datasets_raw, Iterable):
        raise ValueError(f"Experiment '{name}' must include 'datasets'")
    if not models_raw or not isinstance(models_raw, Iterable):
        raise ValueError(f"Experiment '{name}' must include 'models'")

    dataset_names = [str(d) for d in datasets_raw]
    models = [_parse_model_spec(model_raw) for model_raw in models_raw]

    return ExperimentSpec(
        name=name,
        dataset_names=dataset_names,
        models=models,
    )


def _parse_model_spec(raw: Any) -> ModelSpec:
    if not isinstance(raw, dict):
        raise TypeError("Model spec must be a mapping")
    if "name" not in raw:
        raise ValueError(f"Model spec missing 'name': {raw}")
    overrides = {k: v for k, v in raw.items() if k != "name"}
    return ModelSpec(name=str(raw["name"]), overrides=overrides)

