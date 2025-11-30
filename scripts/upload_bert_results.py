#!/usr/bin/env python3
"""Upload BERT NER and FRACCO results to W&B table."""

import wandb

# Results from experiments
results = [
    # FRACCO ICD Classification
    {"task": "FRACCO ICD", "model": "camembert-base", "dataset": "fracco", "accuracy": 63.95, "f1": 9.79},
    {"task": "FRACCO ICD", "model": "camembert-bio", "dataset": "fracco", "accuracy": 66.91, "f1": 12.42},
    {"task": "FRACCO ICD", "model": "BioClinical-ModernBERT", "dataset": "fracco", "accuracy": 80.83, "f1": 37.67},
    {"task": "FRACCO ICD", "model": "ModernCamemBERT", "dataset": "fracco", "accuracy": 80.55, "f1": 38.88},

    # NER CAS1
    {"task": "NER", "model": "ModernCamemBERT @ 2048", "dataset": "CAS1", "exact_f1": 40.84, "partial_f1": 76.41},
    {"task": "NER", "model": "camembert-base @ 512", "dataset": "CAS1", "exact_f1": 27.11, "partial_f1": 77.11},

    # NER CAS2
    {"task": "NER", "model": "ModernCamemBERT @ 2048", "dataset": "CAS2", "exact_f1": 53.21, "partial_f1": 74.58},
    {"task": "NER", "model": "camembert-base @ 512", "dataset": "CAS2", "exact_f1": 33.99, "partial_f1": 59.02},
]

# Initialize W&B
run = wandb.init(project="tabib-bert-benchmark", name="bert-results-summary")

# Create table
columns = ["task", "model", "dataset", "accuracy", "f1", "exact_f1", "partial_f1"]
table = wandb.Table(columns=columns)

for r in results:
    table.add_data(
        r.get("task", ""),
        r.get("model", ""),
        r.get("dataset", ""),
        r.get("accuracy", None),
        r.get("f1", None),
        r.get("exact_f1", None),
        r.get("partial_f1", None),
    )

# Log table
wandb.log({"results": table})

# Also log as summary metrics
wandb.summary["best_ner_exact_f1"] = 53.21  # ModernCamemBERT CAS2
wandb.summary["best_fracco_f1"] = 38.88  # ModernCamemBERT
wandb.summary["best_model"] = "ModernCamemBERT"

print("Results uploaded to W&B!")
print(f"View at: {run.url}")

wandb.finish()
