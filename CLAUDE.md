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
│   ├── vllm_classification.py # LLM MCQA
│   └── lora_sft.py           # LoRA fine-tuning
├── tasks/                    # Task definitions + metrics
│   ├── ner_span.py           # NER span evaluation
│   ├── classification.py     # Classification metrics
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
- **Task support**: NER, Classification, Similarity, MCQA
- **Output formats**: JSON, Markdown tables, W&B upload
- **Nested entity filtering**: BRAT adapters filter to coarsest granularity
- **Base configs**: `configs/base/` for reusable task configs

## Tutorial: Offline Benchmark with ModernBERT

Example: benchmark 3 ModernBERT models on NER (EMEA, CAS1) and CLS (ESSAI, DiaMED) on an HPC cluster without internet.

### Step 1: Create benchmark config

Create `configs/benchmark_modernbert_offline.yaml`:

```yaml
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
      modernbert-large: almanach/moderncamembert-large
      modernbert-bio: rntc/moderncamembert-bio-base

output:
  json: results/modernbert_offline.json
  markdown: results/modernbert_offline.md
```

### Step 2: Download models (with internet)

```bash
# On login node with internet access
tabib download configs/benchmark_modernbert_offline.yaml

# Models saved to $SCRATCH/tabib/models/:
# - almanach--moderncamembert-base/
# - almanach--moderncamembert-large/
# - rntc--moderncamembert-bio-base/
```

### Step 3: Run benchmark (offline)

```bash
# On compute node without internet
tabib benchmark configs/benchmark_modernbert_offline.yaml
```

The pipeline automatically resolves `almanach/moderncamembert-base` to `$SCRATCH/tabib/models/almanach--moderncamembert-base`.

### Step 4: Check results

```bash
# View markdown table
cat results/modernbert_offline.md

# JSON for programmatic access
cat results/modernbert_offline.json
```

### Notes

- `$SCRATCH` is usually pre-defined on HPC clusters - no need to set it
- Models are cached with `--` replacing `/` in path (e.g., `almanach/x` → `almanach--x`)
- Use `--dry-run` to preview runs before launching

## Golden Rules

- Git commit specific files (never `git add .`)
- Update this file after significant changes
- Use `--dry-run` before long benchmarks

## Changelog (2025-12-11)
- **Offline mode**: Added `offline_dir` config option for HPC clusters without internet
- **New CLI command**: `tabib download` - downloads models for offline use
  ```bash
  tabib download configs/benchmark_bert_drbenchmark.yaml -o /scratch/tabib
  ```
- **Automatic cache**: Uses `$SCRATCH/tabib/models/` by default if SCRATCH is set
- **Model path format**: `almanach/camembert-bio-base` -> `almanach--camembert-bio-base`
- **Files added**:
  - `src/tabib/offline.py` - offline path utilities
  - `src/tabib/download.py` - model download logic
  - `scripts/upload_datasets_to_hf.py` - upload local datasets to HF

## Changelog (2025-12-02)
- Added MedDialog-FR Women adapter (`meddialog_women`) - 80 multilabel classes (UMLS CUI combos)
- Registered `fracco_icd_top50` dataset variant with pre-configured top_k=50
- Updated benchmark to 70 runs (14 datasets × 5 models):
  - NER: emea, cas1, cas2, **fracco_expression_ner**
  - CLS: essai, diamed, morfitt, **fracco_icd_classification**, **fracco_icd_top50**, **meddialog_women**
  - SIM: clister
  - MCQA: mediqal_mcqm, mediqal_mcqu, **french_med_mcqa_extended**
