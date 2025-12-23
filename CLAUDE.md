# CLAUDE.md

## Project Overview

**tabib** - Benchmarking framework for French biomedical NLP models (BERT & LLM).

W&B: https://wandb.ai/rnar/tabib-drbenchmark

## Architecture

```
src/tabib/
├── cli.py                    # CLI: train, eval, benchmark
├── pipeline.py               # Unified train/eval pipeline
├── config.py                 # RunConfig, TrainingConfig (Pydantic)
├── registry.py               # Task/Dataset/Model registry
├── data/                     # Dataset adapters
│   ├── brat.py               # BRAT format (NER)
│   ├── cas.py, emea.py       # NER datasets
│   ├── essai.py, diamed.py   # Classification datasets
│   └── ...
├── models/                   # Model adapters
│   ├── bert_token_ner.py     # BERT NER (token classification)
│   ├── bert_text_cls.py      # BERT classification
│   ├── bert_similarity.py    # BERT sentence similarity
│   ├── bert_multilabel_cls.py # BERT multilabel classification
│   ├── vllm_classification.py # LLM MCQA
│   └── lora_sft.py           # LoRA fine-tuning
├── tasks/                    # Task definitions + metrics
│   ├── ner_span.py           # NER span evaluation
│   ├── classification.py     # Classification metrics
│   ├── multilabel.py         # Multilabel classification
│   └── ...
└── comparison/
    └── benchmark.py          # Multi-model benchmark system
```

## Commands

```bash
# Benchmark: compare multiple models across datasets
poetry run tabib benchmark configs/benchmark_bert_drbenchmark.yaml

# Dry-run: preview planned runs
poetry run tabib benchmark configs/benchmark.yaml --dry-run

# Multi-seed averaging (reduces variance)
poetry run tabib benchmark configs/benchmark.yaml --seeds 42,43,44,45,46

# Single model train/eval
poetry run tabib train configs/your_config.yaml
poetry run tabib eval configs/your_config.yaml

# Download models for offline use
tabib download configs/benchmark.yaml -o /scratch/tabib
```

## Benchmark Config Format

```yaml
description: My Benchmark
seeds: [42, 43, 44, 45, 46]  # optional: multi-seed averaging

datasets:
  ner: [emea, cas1, cas2]
  cls: [essai, diamed, morfitt]
  sim: [clister]
  mcqa: [mediqal_mcqm, mediqal_mcqu]

model_groups:
  bert:
    configs:
      ner: base/ner_bert.yaml
      cls: base/cls_bert.yaml
    models:
      camembert: camembert-base
      camembert-bio: almanach/camembert-bio-base

output:
  json: ../results/benchmark.json
  markdown: ../results/benchmark.md
  wandb:
    project: tabib-drbenchmark
    table: results
```

## Key Features

- **Multi-model benchmarking**: Compare BERT/LLM across tasks in single run
- **Multi-seed averaging**: `--seeds 42,43,44` for variance reduction (mean +/- std)
- **Task support**: NER, Classification, Multilabel, Similarity, MCQA
- **Output formats**: JSON, Markdown tables, W&B upload
- **Nested entity filtering**: BRAT adapters filter to coarsest granularity
- **Base configs**: `configs/base/` for reusable task configs
- **Offline mode**: `offline_dir` config option for HPC clusters without internet

## Tutorial: Offline Benchmark with ModernBERT

Example: benchmark ModernBERT models on NER and CLS on an HPC cluster without internet.

### Step 1: Create benchmark config

```yaml
# configs/benchmark_modernbert_offline.yaml
description: ModernBERT Offline Benchmark

datasets:
  ner: [emea, cas1]
  cls: [essai, diamed]

model_groups:
  bert:
    configs:
      ner: base/ner_bert.yaml
      cls: base/cls_bert.yaml
    models:
      modernbert-base: almanach/moderncamembert-base
      modernbert-bio: rntc/moderncamembert-bio-base

output:
  json: results/modernbert_offline.json
  markdown: results/modernbert_offline.md
```

### Step 2: Download models (with internet)

```bash
# On login node with internet access
tabib download configs/benchmark_modernbert_offline.yaml

# Models saved to $SCRATCH/tabib/models/
```

### Step 3: Run benchmark (offline)

```bash
# On compute node without internet
tabib benchmark configs/benchmark_modernbert_offline.yaml
```

The pipeline automatically resolves model paths from `$SCRATCH/tabib/models/`.

## Datasets

### NER
- `emea`, `cas1`, `cas2` - BRAT format medical NER
- `e3c` - E3C clinical NER
- `medline`, `mantragsc_medline` - MEDLINE abstracts
- `quaero_emea`, `quaero_medline` - QUAERO corpus

### Classification
- `essai`, `diamed` - Clinical trial classification
- `morfitt` - Medical text classification
- `meddialog` - Multilabel medical dialog classification

### Similarity
- `clister` - Clinical sentence similarity

### MCQA
- `mediqal_mcqm`, `mediqal_mcqu` - Medical QA

## Golden Rules

- Git commit specific files (never `git add .`)
- Update this file after significant changes
- Use `--dry-run` before long benchmarks

## HPC Best Practices (Jean Zay)

```bash
# Direct Python path (always works in sbatch)
TABIB_VENV=/lustre/fsn1/projects/rech/rua/uvb79kr/envs/tabib-C39hfyYp-py3.12
$TABIB_VENV/bin/python -m tabib.cli benchmark config.yaml

# Set offline mode for HF
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

# Download models (use partition compil for internet access)
HF_TRANSFER=1 huggingface-cli download almanach/camembert-bio-base \
    --local-dir $SCRATCH/tabib/models/almanach--camembert-bio-base
```
