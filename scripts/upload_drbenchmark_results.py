#!/usr/bin/env python3
"""Upload DrBenchmark (ESSAI + DiaMED) results to W&B table."""

import wandb

# Results from DrBenchmark experiments
results = [
    # ESSAI - Negation/Speculation detection (4 classes)
    {"dataset": "ESSAI", "model": "CamemBERT-bio", "accuracy": 98.07, "f1": 92.24},
    {"dataset": "ESSAI", "model": "CamemBERT-base", "accuracy": 96.97, "f1": 90.59},
    {"dataset": "ESSAI", "model": "ModernCamemBERT", "accuracy": 96.28, "f1": 89.96},

    # DiaMED - ICD-10 chapter classification (22 classes)
    {"dataset": "DiaMED", "model": "ModernCamemBERT", "accuracy": 52.05, "f1": 28.94},
    {"dataset": "DiaMED", "model": "CamemBERT-bio", "accuracy": 50.29, "f1": 21.14},
    {"dataset": "DiaMED", "model": "CamemBERT-base", "accuracy": 46.78, "f1": 13.13},
]

# Initialize W&B
run = wandb.init(project="tabib-drbenchmark", name="drbenchmark-results-summary")

# Create table
columns = ["dataset", "model", "accuracy", "f1"]
table = wandb.Table(columns=columns)

for r in results:
    table.add_data(
        r["dataset"],
        r["model"],
        r["accuracy"],
        r["f1"],
    )

# Log table
wandb.log({"results": table})

# Also log as summary metrics
wandb.summary["essai_best_model"] = "CamemBERT-bio"
wandb.summary["essai_best_accuracy"] = 98.07
wandb.summary["essai_best_f1"] = 92.24
wandb.summary["diamed_best_model"] = "ModernCamemBERT"
wandb.summary["diamed_best_accuracy"] = 52.05
wandb.summary["diamed_best_f1"] = 28.94

print("DrBenchmark results uploaded to W&B!")
print(f"View at: {run.url}")

wandb.finish()
