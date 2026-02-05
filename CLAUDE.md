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
│   ├── bert_span_ner.py      # BERT span NER (nested entities, nlstruct)
│   ├── bert_token_ner.py     # BERT token NER (IOB2 tagging)
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
parallel_seeds: 5            # optional: run seeds in parallel (H100 80GB, safe for 4096 tokens)
cleanup_checkpoints: true    # optional: delete checkpoints after (default: true)

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
- **Parallel seed execution**: `parallel_seeds: 3` to train multiple seeds concurrently on one GPU
- **Flash Attention 2**: Automatic support for ModernBERT models (2-4x speedup)
- **Auto-cleanup**: Checkpoints deleted after benchmark to save disk (configurable)
- **Task support**: NER, Classification, Multilabel, Similarity, MCQA
- **Output formats**: JSON, Markdown tables, W&B upload
- **Nested entity filtering**: BRAT adapters filter to coarsest granularity
- **Base configs**: `configs/base/` for reusable task configs
- **Offline mode**: `offline_dir` config option for HPC clusters without internet

## Parallel Seeds (GPU Memory Optimization)

When running benchmarks with multiple seeds, you can train them in parallel on high-memory GPUs (H100 80GB) to reduce total benchmark time.

### Why use parallel seeds?

- **Sequential seeds**: 5 seeds × 5 min = 25 min per config
- **Parallel seeds (3)**: 2 batches × 5 min = 10 min per config (~2.5× faster)

### Configuration

```yaml
description: Benchmark with parallel seeds
seeds: [42, 43, 44, 45, 46]
parallel_seeds: 3  # Train 3 seeds simultaneously

datasets:
  cls: [diamed, essai]
  ...
```

### Memory requirements (ModernCamemBERT-base, 512 tokens)

| parallel_seeds | GPU Memory | Recommended GPU |
|----------------|------------|-----------------|
| 1 | ~25 GB | V100 32GB |
| 2 | ~50 GB | A100 80GB |
| 3 | ~75-80 GB | H100 80GB |

### How it works

1. Runs are grouped by base config (same model/dataset/task, different seeds)
2. Seeds are processed in batches of `parallel_seeds`
3. Uses `ProcessPoolExecutor` for true parallel execution (each process gets its own CUDA context)
4. Results are collected and aggregated normally (mean ± std)

## Flash Attention 2

ModernBERT models support Flash Attention 2 for significant speedups (2-4x faster than standard attention).

### Compatibility

| Model Type | Flash Attention 2 | Config |
|------------|-------------------|--------|
| ModernCamemBERT | Yes | `configs/base/*_bert.yaml` |
| CamemBERT (RoBERTa) | No | `configs/base/*_bert_512.yaml` |
| DrBERT (RoBERTa) | No | `configs/base/*_bert_512.yaml` |

### Configuration

Flash Attention 2 is enabled via `backend_args` in base configs:

```yaml
backend_args:
  max_length: 4096
  attn_implementation: flash_attention_2  # Requires ModernBERT + bf16
```

### Memory with Flash Attention (H100 80GB)

| max_length | batch | parallel_seeds | GPU Memory |
|------------|-------|----------------|------------|
| 512 | 16 | 10 | ~12 GB |
| 2048 | 8 | 10 | ~40 GB |
| 4096 | 4 | 5 | ~40 GB (recommended) |
| 4096 | 8 | 5 | ~80 GB (risky) |

## Checkpoint Cleanup

Benchmarks create many checkpoints that consume disk space. Auto-cleanup removes them after results are saved.

### Configuration

```yaml
# Default: cleanup enabled (recommended)
cleanup_checkpoints: true

# Keep checkpoints for debugging/analysis
cleanup_checkpoints: false
```

### Manual cleanup

```bash
# Check disk usage
du -sh /lustre/fsn1/projects/rech/rua/uvb79kr/tabib/runs

# Remove old benchmark runs
rm -rf /lustre/fsn1/projects/rech/rua/uvb79kr/tabib/runs/runs/*
```

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

## NER Models

Two NER approaches are available:

### Span NER (`bert_span_ner`) - For nested entities
- **Use for**: `emea`, `cas1`, `cas2`, `medline` (BRAT format with nested entities)
- **Architecture**: BERT + BiLSTM + BIOUL CRF + Biaffine (nlstruct-inspired)
- **Config**: `base/ner_span_bert.yaml`
- **Metrics**: `exact_f1` (~65%), `partial_f1` (~80%)

```yaml
# Example: configs/ner_span_emea.yaml
task: ner_span
dataset: emea
model: bert_span_ner
model_name_or_path: almanach/camembert-bio-base

backend_args:
  max_span_length: 40      # Max entity length in tokens
  use_crf: true            # CRF for boundary detection
  lstm_hidden_size: 400    # BiLSTM hidden size
  dropout: 0.4

training:
  max_steps: 4000
  per_device_train_batch_size: 16
  learning_rate: 0.001           # For BiLSTM/CRF layers
  bert_learning_rate: 0.00005    # For BERT (lower)
  gradient_clip: 10.0
```

### Token NER (`bert_token_ner`) - For flat IOB2 tagging
- **Use for**: `quaero_emea_token`, `quaero_medline_token` (IOB2 format, no nesting)
- **Architecture**: BERT + linear classifier (standard token classification)
- **Config**: `base/ner_token_bert.yaml`
- **Metrics**: seqeval `f1` (strict IOB2 evaluation)

```yaml
# Example: configs/ner_token_quaero.yaml
task: ner_token
dataset: quaero_emea_token
model: bert_token_ner
model_name_or_path: almanach/camembert-bio-base

training:
  max_steps: 2000
  learning_rate: 5e-5
```

### When to use which?
| Dataset | Format | Nested? | Model | Task |
|---------|--------|---------|-------|------|
| emea, cas1, cas2, medline | BRAT | Yes | `bert_span_ner` | `ner_span` |
| quaero_*_token | IOB2 | No | `bert_token_ner` | `ner_token` |

## Datasets

### NER (Named Entity Recognition)

Extract medical entities from clinical text.

| Dataset | Input | Output | Entity Types | Application |
|---------|-------|--------|--------------|-------------|
| `emea` | Drug leaflet text | Entity spans | Anatomy, Chemical, Disorder, Medical device, Procedure | Drug safety NER |
| `cas1` | Clinical case (coarse) | Entity spans | 4 entity types (coarse granularity) | Clinical NER |
| `cas2` | Clinical case (fine) | Entity spans | 12 entity types (fine granularity) | Detailed clinical NER |
| `medline` | MEDLINE abstracts | Entity spans | Anatomy, Chemical, Disorder, etc. | Biomedical literature NER |

```
Input:  "Le patient présente une hypertension artérielle et un diabète de type 2"
Output: [(24, 49, "Disorder"), (56, 72, "Disorder")]
        → "hypertension artérielle", "diabète de type 2"
```

### Classification

Classify clinical text into categories.

| Dataset | Input | Output | Classes | Application |
|---------|-------|--------|---------|-------------|
| `essai` | Clinical trial abstract | Trial phase | Phase I, II, III, IV | Clinical trial classification |
| `diamed` | Medical text | Specialty | 28 medical specialties | Medical specialty routing |
| `morfitt` | Scientific abstract | Domain | 12 biomedical domains | Document categorization |

```
Input:  "Cette étude de phase III évalue l'efficacité du traitement..."
Output: "Phase III"

Input:  "Le patient cardiaque présente une insuffisance..."
Output: "Cardiologie"
```

### Multilabel Classification

Assign multiple labels to clinical text.

| Dataset | Input | Output | Classes | Application |
|---------|-------|--------|---------|-------------|
| `meddialog` | Medical dialogue | Symptom categories | Multiple symptom types | Symptom extraction from conversations |

```
Input:  "J'ai mal à la tête et je tousse depuis 3 jours"
Output: ["céphalée", "toux"]  (multiple labels)
```

### Similarity

Measure semantic similarity between clinical sentences.

| Dataset | Input | Output | Scale | Application |
|---------|-------|--------|-------|-------------|
| `clister` | Pair of clinical sentences | Similarity score | 0-5 (continuous) | Clinical sentence matching |

```
Input:  ("Le patient a de la fièvre", "Température corporelle élevée")
Output: 4.2 (high similarity)
```

### MCQA (Multiple Choice Question Answering)

Answer medical questions with multiple choices.

| Dataset | Input | Output | Choices | Application |
|---------|-------|--------|---------|-------------|
| `mediqal_mcqm` | Medical question + choices | Correct answer(s) | Multiple correct | Medical QA (multi-answer) |
| `mediqal_mcqu` | Medical question + choices | Single answer | One correct | Medical QA (single-answer) |

```
Input:  "Quel est le traitement de première ligne pour l'hypertension?"
        A) Bêta-bloquants  B) IEC  C) Diurétiques  D) Chirurgie
Output: B (or multiple for mcqm)
```

### FRACCO ICD Coding

Automatic ICD code assignment for French oncology.

| Dataset | Input | Output | Classes | Application |
|---------|-------|--------|---------|-------------|
| `fracco_icd_top50` | Tumor mention | ICD code | 50 codes | Oncology coding (common) |
| `fracco_icd_top100` | Tumor mention | ICD code | 100 codes | Oncology coding (extended) |
| `fracco_icd_top200` | Tumor mention | ICD code | 200 codes | Oncology coding (detailed) |
| `fracco_icd_doc_top500` | Clinical document | All ICD codes | 500 codes | Document-level coding |

```
Input:  "adénocarcinome colique"
Output: "C18.9" (ICD code for colon cancer)

Input:  (full document about cancer patient)
Output: ["C18.9", "C77.0", ...]  (all codes in document)
```

### FRASIMED Entity Normalization

Normalize medical entities to standard codes (ICD-O, SNOMED).

| Dataset | Input | Output | Classes | F1 | Application |
|---------|-------|--------|---------|-----|-------------|
| `frasimed_cantemist_norm` | Tumor mention | ICD-O-3.1 code | 759 | 86% | Oncology morphology coding |
| `frasimed_distemist_norm_top100` | Disease mention | SNOMED code | 100 | 87% | Disease normalization |
| `frasimed_distemist_doc_top100` | Clinical document | All SNOMED codes | 100 | 20% | Document disease coding |

```
Input:  "carcinome"           → Output: "8010/3" (ICD-O: carcinoma, malignant)
Input:  "hypertension"        → Output: "38341003" (SNOMED: hypertension)
Input:  (full clinical doc)   → Output: ["38341003", "44054006", ...] (all diseases)
```

## FRACCO ICD Classification

Two approaches for ICD code classification from the FRACCO oncology dataset:

### Mention-Level (`fracco_icd_classification`) - Single-label
- **Input**: Individual text mention (expression)
- **Output**: Single ICD code for that mention
- **Task**: `classification`
- **Model**: `bert_text_cls`
- **Use case**: When you have pre-extracted mentions and want to classify each

```yaml
task: classification
dataset: fracco_icd_top100
model: bert_text_cls
model_name_or_path: almanach/camembert-bio-base
```

### Document-Level (`fracco_icd_doc`) - Multilabel
- **Input**: Full document text (from FRACCO `.txt` files, up to 2048 tokens)
- **Output**: Multi-hot vector of all ICD codes in the document
- **Task**: `multilabel`
- **Model**: `bert_multilabel_cls`
- **Use case**: End-to-end ICD coding without mention extraction
- **Data**: 1300 documents, ~13 codes per document, 3041 unique codes

```yaml
# configs/test_fracco_doc_icd.yaml
task: multilabel
dataset: fracco_icd_doc_top500
model: bert_multilabel_cls
model_name_or_path: almanach/moderncamembert-base

training:
  num_train_epochs: 10
  per_device_train_batch_size: 4
  learning_rate: 2e-5
  bf16: true

backend_args:
  max_length: 2048  # Full document, requires H100 (32GB+)
```

### Important Considerations

| Aspect | Mention-Level | Document-Level |
|--------|---------------|----------------|
| Task type | `classification` | `multilabel` |
| Labels per sample | 1 | Multiple (~13 avg) |
| Input length | Short mentions | Full documents (2048 tokens) |
| GPU memory | V100 OK | V100 for RoBERTa-like, H100 for ModernBERT-like |
| Expected F1 | Higher (simpler task) | Lower (~15% F1_micro for top500) |

**Why document-level F1 is low**: 500-class multilabel with long-tail distribution (many rare codes). The model learns (F1 improves from ~5% to ~15% over 10 epochs), but the task is inherently difficult. Consider:
- Using `top_k` filtering for more frequent codes
- More training epochs
- Focal loss or other class imbalance techniques

**Data source**: `$SCRATCH/tabib/data/FRACCO.zip` containing:
- `ann_txt_files/*.txt` - Full document text
- `DetectOnco_Final.csv` - ICD annotations (`expression_CIM` label type)

## FRASIMED Entity Normalization (Best Results)

French biomedical entity normalization from the FRASIMED corpus (cross-lingual projection from Spanish CANTEMIST/DISTEMIST).

### Most Interesting Tasks

| Dataset | Task | Classes | F1 | Description |
|---------|------|---------|-----|-------------|
| `frasimed_cantemist_norm` | classification | 759 ICD-O | **86%** | Mention → ICD-O-3.1 morphology code |
| `frasimed_distemist_norm_top100` | classification | 100 SNOMED | **87%** | Mention → SNOMED disease code |
| `frasimed_distemist_doc_top100` | multilabel | 100 SNOMED | **20%** | Document → all SNOMED codes |

### Entity Normalization (Mention → Code)

Best task: given a disease/morphology mention, predict the medical code.

```yaml
# configs/frasimed_norm_cantemist.yaml - 86% F1
task: classification
dataset: frasimed_cantemist_norm
model: bert_text_cls
model_name_or_path: almanach/moderncamembert-base

training:
  output_dir: runs/frasimed_norm_cantemist
  num_train_epochs: 10
  per_device_train_batch_size: 32
  learning_rate: 2e-5
  metric_for_best_model: eval_f1
  bf16: true
```

```yaml
# configs/frasimed_norm_distemist_top100.yaml - 87% F1
task: classification
dataset: frasimed_distemist_norm_top100
model: bert_text_cls
model_name_or_path: almanach/moderncamembert-base

training:
  output_dir: runs/frasimed_norm_distemist_top100
  num_train_epochs: 10
  per_device_train_batch_size: 32
  learning_rate: 2e-5
  metric_for_best_model: eval_f1
  bf16: true
```

### Document-Level Multilabel (Document → All Codes)

End-to-end: full document → predict all disease codes.

```yaml
# configs/frasimed_doc_distemist_top100.yaml - 20% F1_micro
task: multilabel
dataset: frasimed_distemist_doc_top100
model: bert_multilabel_cls
model_name_or_path: almanach/moderncamembert-base

training:
  output_dir: runs/frasimed_doc_distemist_top100
  num_train_epochs: 10
  per_device_train_batch_size: 4
  learning_rate: 2e-5
  metric_for_best_model: eval_f1_micro
  bf16: true

backend_args:
  max_length: 2048
```

### All FRASIMED Datasets

| Dataset | Task | Description |
|---------|------|-------------|
| `frasimed_cantemist` | ner_span | NER for morphology-onco entities |
| `frasimed_distemist` | ner_span | NER for disease entities |
| `frasimed_cantemist_norm` | classification | Mention → ICD-O (759 codes) |
| `frasimed_cantemist_norm_top30` | classification | Mention → ICD-O (top 30) |
| `frasimed_distemist_norm` | classification | Mention → SNOMED (2400 codes) |
| `frasimed_distemist_norm_top100` | classification | Mention → SNOMED (top 100) |
| `frasimed_distemist_norm_top500` | classification | Mention → SNOMED (top 500) |
| `frasimed_cantemist_doc` | multilabel | Document → ICD-O (759 codes) |
| `frasimed_cantemist_doc_top30` | multilabel | Document → ICD-O (top 30) |
| `frasimed_distemist_doc` | multilabel | Document → SNOMED (2400 codes) |
| `frasimed_distemist_doc_top100` | multilabel | Document → SNOMED (top 100) |
| `frasimed_distemist_doc_top500` | multilabel | Document → SNOMED (top 500) |

**Data source**: `$SCRATCH/tabib/data/FRASIMED/` (BRAT format)

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
