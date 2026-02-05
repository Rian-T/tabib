#!/usr/bin/env python3
"""Generate 3-epoch training configs for biomedical datasets.

Based on dataset sizes calculated from MDS metadata:
- hal-original: 29,837 samples → 51 batches/epoch → 153 batches for 3 epochs
- istex-scientific: 4,487 samples → 7 batches/epoch → 21 batches
- emea-notice: 6,960 samples → 12 batches/epoch → 36 batches
- pmc-patients: 2,821 samples → 4 batches/epoch → 12 batches
- ccam: 41,368 samples → 71 batches/epoch → 213 batches
- cim10: 41,289 samples → 71 batches/epoch → 213 batches
"""

import os
import yaml
from pathlib import Path

# Dataset → 3-epoch batch count mapping
DATASET_EPOCHS = {
    "hal-original": 153,
    "hal-scientific": 153,  # Same as hal-original
    "istex-scientific": 21,
    "emea-notice": 36,
    "pmc-patients": 12,
    "ccam": 213,
    "cim10": 213,
    "atc": 36,  # Similar size to emea-notice
    "synthetic-clinical": 60,  # Estimated
    "code-mix": 213,  # 71 batches * 3 epochs (same as ccam/cim10 target)
}

CONFIG_DIR = Path("/lustre/fswork/projects/rech/rua/uvb79kr/ModernBERT/yamls/modernbert/biomed")
OUTPUT_DIR = CONFIG_DIR / "3epoch"
CHECKPOINT_BASE = "/lustre/fsn1/projects/rech/rua/uvb79kr/modernbert/checkpoints"


def generate_3epoch_config(dataset_name: str):
    """Generate a 3-epoch config for a dataset."""
    source_config = CONFIG_DIR / f"{dataset_name}.yaml"

    if not source_config.exists():
        print(f"Skipping {dataset_name}: source config not found")
        return

    if dataset_name not in DATASET_EPOCHS:
        print(f"Skipping {dataset_name}: no epoch count defined")
        return

    with open(source_config) as f:
        config = yaml.safe_load(f)

    # Update for 3 epochs
    batches_3epoch = DATASET_EPOCHS[dataset_name]
    config["max_duration"] = f"{batches_3epoch}ba"
    config["run_name"] = f"moderncamembert-{dataset_name}-3epoch"
    config["save_folder"] = f"{CHECKPOINT_BASE}/moderncamembert-{dataset_name}-3epoch"

    # Adjust eval_interval if needed (don't eval more than 10 times)
    eval_interval = max(10, batches_3epoch // 10)
    config["eval_interval"] = f"{eval_interval}ba"
    config["save_interval"] = f"{eval_interval}ba"

    # Write output config
    output_path = OUTPUT_DIR / f"{dataset_name}.yaml"
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Created: {output_path.name} (max_duration: {batches_3epoch}ba)")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for dataset_name in DATASET_EPOCHS:
        generate_3epoch_config(dataset_name)

    print(f"\nGenerated {len(list(OUTPUT_DIR.glob('*.yaml')))} configs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
