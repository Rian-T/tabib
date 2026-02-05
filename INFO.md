# Tabib Setup for French Biomedical ModernBERT Evaluation

## Overview

This document describes how to use **tabib** for evaluating French biomedical ModernBERT models on the DrBenchmark tasks, specifically configured for **offline execution on Jean Zay HPC**.

## Prerequisites

### 1. Environment Setup

```bash
# Poetry manages the tabib environment
cd $WORK/tabib
poetry install

# Get the venv path for SLURM jobs (Poetry not available on compute nodes)
poetry env info --path
# Example: /lustre/fsn1/projects/rech/rua/uvb79kr/envs/tabib-C39hfyYp-py3.12
```

### 2. Required Package Versions

```bash
# datasets<3.0.0 required for trust_remote_code support
poetry add 'datasets<3.0.0'

# accelerate>=0.26.0 required for Trainer
poetry add 'accelerate>=0.26.0'
```

## Offline Mode Configuration

Jean Zay GPU nodes have **no internet access**. Everything must be pre-cached on login nodes.

### 1. Pre-cache HuggingFace Datasets

```python
from datasets import load_dataset

# Classification datasets
load_dataset("DrBenchmark/DiaMED", trust_remote_code=True)
load_dataset("DrBenchmark/MORFITT", "source", trust_remote_code=True)

# Similarity dataset
load_dataset("DrBenchmark/CLISTER", trust_remote_code=True)

# NER datasets
load_dataset("DrBenchmark/MANTRAGSC", "fr_medline", trust_remote_code=True)
load_dataset("rntc/legacy_e3c")  # For E3C

# Underlying datasets (required by DrBenchmark loaders)
load_dataset("Dr-BERT/DiaMED")
load_dataset("Dr-BERT/MORFITT")
```

### 2. Pre-cache Evaluate Metrics

```python
import evaluate

evaluate.load("accuracy")
evaluate.load("f1")
evaluate.load("precision")
evaluate.load("recall")
evaluate.load("seqeval")
evaluate.load("glue", "stsb")
evaluate.load("mse")
```

### 3. Environment Variables for Offline Mode

```bash
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HF_HUB_CACHE=$SCRATCH/hf_cache
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
```

## Dataset Adapters Modifications

### DiaMED (`src/tabib/data/diamed.py`)

The HuggingFace dataset returns `icd-10` as **integer indices** (0-21), not strings. Modified to map indices to ICD-10 chapter names:

```python
ICD10_CHAPTERS = [
    'A00-B99  Certain infectious and parasitic diseases',
    'C00-D49  Neoplasms',
    # ... 22 chapters total
]

# In load_splits():
label_idx = item.get("icd-10")
if isinstance(label_idx, int):
    if 0 <= label_idx < len(ICD10_CHAPTERS):
        label = ICD10_CHAPTERS[label_idx]
```

### MORFITT (`src/tabib/data/morfitt.py`)

The HuggingFace dataset returns `specialities` as a **list of integer indices**, not pipe-separated strings:

```python
MORFITT_SPECIALTIES = [
    'microbiology', 'etiology', 'virology', 'physiology', 'immunology',
    'parasitology', 'genetics', 'chemistry', 'veterinary', 'surgery',
    'pharmacology', 'psychology'
]

# In load_splits():
speciality_indices = item.get("specialities")  # e.g., [11] or [8, 4]
primary_idx = speciality_indices[0]
primary_label = MORFITT_SPECIALTIES[primary_idx]
```

### Pipeline (`src/tabib/pipeline.py`)

Weave (W&B tracing) requires network. Skip initialization in offline mode:

```python
if os.environ.get("HF_HUB_OFFLINE", "0") != "1":
    try:
        weave.init(f"tabib-french-biomedical-ner")
    except Exception:
        pass
```

## Hyperparameters Configuration

### Context Length Considerations

ModernBERT supports **8192 tokens**, but dataset requirements vary:

| Dataset | Max Tokens | Mean | >512 | Recommended max_length |
|---------|-----------|------|------|------------------------|
| DiaMED | 1458 | 524 | 43% | **2048** |
| MORFITT | 1952 | 328 | 10% | **2048** |
| CLISTER | 90 | 20 | 0% | 512 |
| E3C | 248 | 34 | 0% | 512 |
| MANTRAGSC | 21 | 10 | 0% | 512 |

### Config Files

**NER** (`configs/base/ner_bert.yaml`):
```yaml
task: ner_span
model: bert_token_ner
training:
  num_train_epochs: 20
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 16
  learning_rate: 2e-5
  warmup_steps: 100
  eval_strategy: epoch
  save_strategy: epoch
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  bf16: true
```

**Classification** (`configs/base/cls_bert.yaml`):
```yaml
task: classification
model: bert_text_cls
training:
  num_train_epochs: 20
  per_device_train_batch_size: 4  # Reduced for longer sequences
  per_device_eval_batch_size: 8
  learning_rate: 2e-5
  warmup_ratio: 0.1
  eval_strategy: epoch
  save_strategy: epoch
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  bf16: true
backend_args:
  max_length: 2048  # For full clinical cases
```

**Similarity** (`configs/base/sim_bert.yaml`):
```yaml
task: similarity
model: bert_similarity
training:
  num_train_epochs: 20
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  learning_rate: 2e-5
  warmup_steps: 100
  eval_strategy: epoch
  save_strategy: epoch
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  bf16: true
backend_args:
  max_length: 512  # Short sentence pairs
```

## SLURM Batch Script

`launch-eval.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=tabib-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -A rua@h100
#SBATCH -C h100
#SBATCH --time=04:00:00
#SBATCH --array=0-17
#SBATCH --output=logs/tabib_eval_%A_%a.out

# Offline mode environment
export PYTHONNOUSERSITE=1
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HF_HUB_CACHE=$SCRATCH/hf_cache
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

cd $WORK/tabib

# Model array (18 models)
MODELS=(
    "almanach--moderncamembert-base"
    "moderncamembert-hal-original"
    # ... other models
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
MODEL_PATH=$SCRATCH/tabib/models/$MODEL

# Generate benchmark config
cat > /tmp/eval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml << EOF
description: "Evaluation of ${MODEL}"

datasets:
  ner: [emea, e3c, medline, mantragsc_medline]
  cls: [diamed, morfitt]
  sim: [clister]

model_groups:
  eval:
    configs:
      ner: ${WORK}/tabib/configs/base/ner_bert.yaml
      cls: ${WORK}/tabib/configs/base/cls_bert.yaml
      sim: ${WORK}/tabib/configs/base/sim_bert.yaml
    models:
      ${MODEL}: ${MODEL_PATH}

output:
  json: ${WORK}/tabib/results/${MODEL}.json
  markdown: ${WORK}/tabib/results/${MODEL}.md
EOF

# Run using venv directly (Poetry not available on compute nodes)
TABIB_VENV=/lustre/fsn1/projects/rech/rua/uvb79kr/envs/tabib-C39hfyYp-py3.12
$TABIB_VENV/bin/python -m tabib.cli benchmark /tmp/eval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml
```

## Available Datasets

### Working (HuggingFace-based):
- **NER**: emea, e3c, medline, mantragsc_medline
- **Classification**: diamed, morfitt
- **Similarity**: clister

### Require Local Data (not yet configured):
- **Classification**: fracco_icd_classification, meddialog_women
- **NER**: cas1, cas2 (need training splits)

## Running Evaluation

```bash
# Submit array job for all 18 models
cd $WORK/tabib
sbatch launch-eval.sbatch

# Monitor progress
squeue -u $USER
sacct -j <job_id> --format=JobID,State,Elapsed

# View results
cat results/<model_name>.md
python3 /tmp/monitor_results.py  # Custom results table script
```

## Results Interpretation

Metrics per task:
- **NER**: `exact_f1` (strict entity matching)
- **Classification**: `f1` (macro F1)
- **Similarity**: `spearman` (Spearman correlation)

## Troubleshooting

### "Permission denied: '/cache'"
- Cause: HF cache environment variables not set correctly
- Fix: Ensure `HF_HOME`, `HF_DATASETS_CACHE`, `HF_HUB_CACHE` point to `$SCRATCH/hf_cache`

### "No module named 'poetry'"
- Cause: Poetry not installed on compute nodes
- Fix: Use venv Python directly: `$VENV/bin/python -m tabib.cli ...`

### "trust_remote_code not supported"
- Cause: datasets>=3.0.0 removed this feature
- Fix: `poetry add 'datasets<3.0.0'`

### "Network is unreachable"
- Cause: Trying to download on GPU node
- Fix: Pre-cache all datasets/metrics on login node first

### "ClassificationTask must have label_list"
- Cause: Dataset adapter returning empty splits
- Fix: Check adapter handles HF dataset format correctly (int vs string labels)
