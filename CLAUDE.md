# CLAUDE.md

## GOLDEN RULES
- Never use `git add .` - always add specific files
- Always check doc online for specific library versions when in doubt
- Make git commits regularly with meaningful but short messages
- Avoid bash commands with many subcommands or pipes unless necessary
- **When long evaluations are running**: Use `/babysit` with relevant sleep values and simple single commands
- **Document regularly**: Always update CLAUDE.md with current state and next steps
- **Keep working**: Never wait for user approval, continue autonomously
- **Git commit regularly**: Add specific changed files and commit with meaningful messages
- **Check Weave traces**: Raw inputs and outputs often contain valuable debug info at wandb.ai/rnar/.../weave

---

## CURRENT STATUS (2025-11-29 11:45)

### NER v2 - MAJOR BREAKTHROUGH ✓

**Two Key Fixes**:

1. **Prompt Fix** (few-shot leakage):
   - Added stop tokens + clear section headers
   - Warnings: 23,572 → 6 (**99.97% reduction**)

2. **Chunk Size Fix** (EMEA/CAS performance):
   - EMEA docs are ~5000 chars, MEDLINE docs are ~50 chars
   - Small chunks (128 tokens) lost context
   - **Solution**: Increase chunk size to 2048 tokens

**NEW RESULTS (MedGemma-27B 5-shot)**:
| Dataset | Old (128 tok) | New (2048 tok) | Improvement |
|---------|---------------|----------------|-------------|
| MEDLINE | 45.81% | **49.76%** | +4% |
| EMEA | 1.27% | **60.52%** | **+59%** |

**Key Insight**: EMEA was failing due to **chunk size**, not annotation style!

**Next**: Update all EMEA/CAS configs with 2048-token chunks and re-run

### Added Gaperon-Garlic Models

Added 16 configs for contamination research models:
- `almanach/Gaperon-Garlic-1125-8B`
- `almanach/Gaperon-Garlic-1125-24B`

Note: These are **intentionally contaminated** (~50% benchmark test data) for contamination detection research.

**Key Insights**:
- Smaller model (Gemma3-4B) outperforms larger ones on this task
- Few-shot examples provide massive improvement (4.08% → 40.19%)
- Base models (27B) may need different prompting strategy
- CAS1/CAS2 annotation style (full clauses) doesn't match LLM extraction

**NEW PROJECT**: Evaluating all models on NER datasets

**W&B Project**: `tabib-ner-benchmark` (new project)

**Datasets** (4 datasets):
| Dataset | Description | Entity Types |
|---------|-------------|--------------|
| MEDLINE | Biomedical abstracts | anatomy, disorder, substance, living, procedure, object, phenomenon, physiology |
| EMEA | European Medicines Agency | To explore |
| CAS1/CAS2 | Clinical case notes | sosy, pathologie, moment, mode, substance |
| E3C | European Clinical Case Corpus | To explore |

**Models to evaluate** (9 models from MCQA):
| Model | HuggingFace ID | Size | Type |
|-------|----------------|------|------|
| Gemma-3-4B-pt | google/gemma-3-4b-pt | 4B | Base |
| Gemma-3-27B-pt | google/gemma-3-27b-pt | 27B | Base |
| Gaperon-8B | almanach/Gaperon-1125-8B | 8B | Completion |
| Gaperon-24B | almanach/Gaperon-1125-24B | 24B | Completion |
| EuroLLM-9B | utter-project/EuroLLM-9B | 9B | Base |
| OLMo-3-7B | allenai/OLMo-2-1124-7B | 7B | Base |
| OLMo-3-32B | allenai/Olmo-3-1125-32B | 32B | Base |
| MedGemma-27B | google/medgemma-27b-text-it | 27B | Instruct |
| Qwen3-8B | Qwen/Qwen3-8B | 8B | Instruct |

**Plan**:
1. [x] Explore datasets (entity types, annotation style, train/test sizes)
2. [x] Test NER prompts in playground (verify format, few-shot, parsing)
3. [ ] Create configs for all model/dataset combinations (0-shot + 5-shot)
4. [ ] Run evaluations on 2xH100
5. [ ] Upload results to W&B

**Exploration Findings**:
- MEDLINE: 10 entity types (ANAT, CHEM, DEVI, DISO, GEOG, LIVB, OBJC, PHEN, PHYS, PROC), 833 samples/split
- EMEA: 10 entity types (same as MEDLINE), only 11-15 samples/split, very long docs
- CAS1: 2 entity types (sosy, pathologie), NO train split (use dev for few-shot)
- CAS2: 8 entity types (anatomie, examen, substance, traitement, valeur, moment, mode, dose), NO train
- Prompt uses XML tags: `<ENTITY_TYPE>text</ENTITY_TYPE>`
- Parsing handles accents, ligatures (œ→oe), case insensitive matching
- Created 72 configs: 9 models × 4 datasets × 2 shot variants
- **NER evaluations started**: Thu Nov 28 23:17 CET 2025

**NER Approach Analysis** (2025-11-28 23:20):
- **Chose XML tagging** over JSON/line-by-line extraction
- Reasons: preserves offsets, handles nested entities, intuitive few-shot
- Known issues: nested tag regex misses outer tags (inner still captured)
- Improvement: French system prompt would be better for French data
- Fuzzy matching already handles: ligatures (œ→oe), accents, case

**CRITICAL BUG FOUND** (2025-11-28 23:55):
- **Few-shot example leakage**: Model outputs entities from examples for EVERY sample!
- Entities like `tumeur de type carcinome épidermoïde` appear 680 times in CAS1 eval
- **Root cause**: Prompt doesn't clearly separate examples from actual task
- Current format uses same "Text:" / "Annotated:" pattern for both
- **Impact**: Massive false positives, inflated entity counts
- **Fix needed**: Add clear section headers `=== EXAMPLES ===` vs `=== YOUR TASK ===`
- **Despite bug**: Still get 40% F1 on MEDLINE - could be MUCH HIGHER with better prompt

**Current Approach Verdict**: XML tagging is good, but PROMPT NEEDS IMPROVEMENT.
The 40% F1 result is WITH the leakage bug - proper fix could significantly improve scores.

**Next Steps After NER**:
1. [ ] LoRA finetune for MCQA tasks (MediQAl MCQU, MCQM, FrenchMedMCQA)
2. [ ] Compare LoRA finetuned vs zero-shot performance

---

### Previous: NER Base Model Evaluation Campaign - SUPERSEDED

**NEW PROJECT**: Evaluating all base/completion models (non-chat, non-instruct) on NER datasets

**W&B Project**: `tabib-ner-benchmark` (new project)

**Models being evaluated** (5 models):
| Model | HuggingFace ID | Size | Type |
|-------|----------------|------|------|
| Gemma-3-4B-pt | google/gemma-3-4b-pt | 4B | Base |
| Gemma-3-27B-pt | google/gemma-3-27b-pt | 27B | Base |
| Gaperon-8B | almanach/Gaperon-1125-8B | 8B | Completion |
| Gaperon-24B | almanach/Gaperon-1125-24B | 24B | Completion |
| EuroLLM-9B | utter-project/EuroLLM-9B | 9B | Base |

**Datasets**: MEDLINE, CAS1, CAS2

**Configurations**: 30 total (5 models x 3 datasets x 2 shot variants)
- 0-shot: `configs/ner_{dataset}_{model}.yaml`
- 3-shot: `configs/ner_{dataset}_{model}_3shot.yaml`

**Run Script**: `scripts/run_ner_base_models.sh`
**Log File**: `results/ner_base_models_run.log`
**Started**: Thu Nov 27 22:25:02 CET 2025
**Current**: Running small models 0-shot (ner_medline_gemma3_4b)

**Scripts Created**:
- `scripts/run_ner_base_models.sh` - Sequential runner for all 30 configs
- `scripts/upload_ner_table.py` - W&B uploader (pivot table with exact_f1, partial_f1)
- `playground/test_ner_fewshot.py` - Test script verifying 0-shot/3-shot support

---

### MCQA Results Summary (0-shot + 3-shot) - COMPLETED

**W&B Run**: https://wandb.ai/rnar/tabib-llm-benchmark/runs/8pnc0x6d

| Model | MCQU | MCQM | FrenchMed |
|-------|------|------|-----------|
| MedGemma-27B | **36.77%** | **9.16%** | **14.31%** |
| Gemma-3-27B | 30.72% / 28.80% | 5.35% / 5.59% | 10.61% / 17.20% |
| EuroLLM-9B | 21.30% / 19.39% | 4.31% / 4.40% | 6.11% / 4.34% |
| Gemma-3-4B | 20.86% / 20.45% | 4.14% / 4.23% | 1.93% / 1.93% |
| Gaperon-8B | 19.83% / 20.24% | 4.14% / 4.17% | 8.68% / 5.14% |
| Qwen3-8B | 19.78% / 19.78% | 5.08% / 5.08% | 8.52% / 7.88% |
| Gaperon-24B | 18.88% | -- | -- |
| OLMo-3-7B | 18.14% / 18.19% | 5.38% / 5.38% | **14.95%** / **14.95%** |

*Format: 0-shot / 3-shot*

**Key Observations**:
- 3-shot provides marginal improvement on most models
- Gemma-3-27B improved on FrenchMed: 10.61% -> 17.20% (+6.59pp)
- OLMo-3-7B consistently good on FrenchMedMCQA: 14.95%
- Multi-answer (MCQM) remains much harder than single-answer (MCQU)

### Previous NER Results (chat models, 3-shot)
| Model | Dataset | Exact F1 | Partial F1 | Notes |
|-------|---------|----------|------------|-------|
| Qwen3-7B | MEDLINE | **39.75%** | **51.52%** | Good results |
| Qwen3-7B | CAS1 | **1.16%** | **31.22%** | Low exact, decent partial |
| MedGemma-27B | CAS1 | **1.16%** | **35.81%** | Same pattern as Qwen3 |

### Gaperon-24B vLLM Patch - FIXED (2025-11-27)

**Problem**: Gaperon-24B uses custom `head_dim=128` with `num_attention_heads=32` (not 40), causing vLLM to miscalculate dimensions.

**Error**: `AssertionError: Attempted to load weight (torch.Size([1024])) into parameter (torch.Size([1280]))`

**Solution**: Complete `Olmo2Attention.__init__` replacement patch in `src/tabib/models/gaperon_patch.py` that reads `head_dim` from config BEFORE creating QKV layers.

**Key fix**:
```python
# Use head_dim from config if available (Gaperon case)
default_head_dim = hidden_size // self.total_num_heads
self.head_dim = getattr(self.config, "head_dim", default_head_dim)
```

**Reference**: https://huggingface.co/almanach/Gaperon-1125-24B/discussions/1

### ROOT CAUSE IDENTIFIED: CAS1/CAS2 Annotation Style

**The real issue is NOT the model or code - it's the annotation style:**

| Dataset | Entity Style | Example | Typical Length |
|---------|-------------|---------|----------------|
| **MEDLINE** | Short terms | `"gangrènes gazeuses"` | 5-25 chars |
| **CAS1/CAS2** | Full clauses | `"tumeur bien limitée, de structure tissulaire, prenant le contraste de façon hétérogène"` | 25-110 chars |

CAS1/CAS2 annotate **full clinical descriptions** as single entities:
- `"recherche de BK dans les crachats et le produit de tubage gastrique réalisé trois fois était revenu négatif"` (107 chars!)
- `"douleurs épisodiques localisées au niveau de l'hypochondre droit"` (64 chars)

**Why exact F1 fails for CAS1/CAS2:**
1. LLMs extract conceptual entities, not full clauses
2. Copying 100+ char spans exactly is nearly impossible
3. Any small boundary variation ruins exact match

**Conclusion**:
- For MEDLINE: Use **Exact F1** (39.75% is good)
- For CAS1/CAS2: Use **Partial F1** (31-35% is good, shows entities ARE found)

**Performance Summary:**
- Temperature: 0.0 ✓
- Batched inference: 43x faster ✓
- 3-shot examples: ✓

### Bugs Fixed This Session
1. **Non-batched inference**: Was sending one sample at a time - fixed to batch all prompts together
2. **Empty 3-shot examples**: Pipeline passed chunked data - fixed by saving original train data
3. **CAS1/CAS2 no train split**: Fixed pipeline to fall back to `dev` for few-shot
4. **Lowercase entity types not parsed**: Fixed regex `[A-Z_]+` -> `[A-Za-z_]+`
5. **Entity definitions missing for CAS**: Added `sosy`, `pathologie`, `moment`, `mode`, `substance`

### Running Now
- NER base model evaluations via `scripts/run_ner_base_models.sh`
- Log: `results/ner_base_models_run.log`
- Started: Thu Nov 27 22:25:02 CET 2025

### Next Steps
1. Wait for NER evaluations to complete (~2-3 hours for 30 configs)
2. Results auto-upload to W&B `tabib-ner-benchmark` project
3. Analyze results comparing 0-shot vs 3-shot for base models
4. Compare base model NER performance vs chat models (Qwen3-7B results)

### Current Crux
- **No blocking issues** - NER pipeline running smoothly
- Gaperon-24B patch verified working for MCQA (18.88% on MediQAl MCQU)
- All 30 NER configs validated and ready

---

## NER Base Model Evaluation Campaign (2025-11-27)

**Focus**: Base/completion models only (no chat/instruct models)

**Models** (5 models):
| Model | Size | Configs |
|-------|------|---------|
| gemma3_4b | 4B | 0-shot + 3-shot |
| gemma3_27b | 27B | 0-shot + 3-shot |
| gaperon_8b | 8B | 0-shot + 3-shot |
| gaperon_24b | 24B | 0-shot + 3-shot |
| eurollm_9b | 9B | 0-shot + 3-shot |

**Datasets**: MEDLINE, CAS1, CAS2

**Total Configs**: 30 (5 models x 3 datasets x 2 shot variants)

**Run Order**:
1. Small models 0-shot (gemma3_4b, gaperon_8b, eurollm_9b x 3 datasets = 9 configs)
2. Small models 3-shot (9 configs)
3. Large models 0-shot (gemma3_27b, gaperon_24b x 3 datasets = 6 configs)
4. Large models 3-shot (6 configs)

**Output**: W&B pivot table with exact_f1 and partial_f1 per model/dataset

---

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tabib is a task-agnostic NLP evaluation framework for biomedical French text. It supports NER, text classification, semantic similarity, MCQA, and more. The framework emphasizes low LOC, clear abstractions, and reproducibility.

## Core Commands

### Installation & Setup
```bash
# Install dependencies
poetry install

# Install with vLLM support (for LLM inference)
poetry install --extras vllm
```

### Training
```bash
# Train a model with config file
poetry run tabib train configs/ner_emea_camembert.yaml

# Training outputs go to:
# - $SCRATCH/tabib/runs/<output_dir> (if SCRATCH env var is set)
# - ./runs/<output_dir> (otherwise)
```

### Evaluation
```bash
# Evaluate a model
poetry run tabib eval configs/ner_emea_camembert.yaml
```

### Model Comparison
```bash
# Compare multiple models across datasets
poetry run tabib compare configs/compare_example.yaml

# Dry run to see planned runs
poetry run tabib compare configs/compare_example.yaml --dry-run

# Specify output path
poetry run tabib compare configs/compare_example.yaml -o results/custom.json
```

### Testing
```bash
# Run specific test
poetry run pytest tests/test_sentence_splitter.py

# Run all tests
poetry run pytest tests/

# Verify Python syntax
python -m py_compile src/tabib/models/your_file.py
```

## Architecture

### Three-Component System

Tabib connects three pluggable abstractions via the `Pipeline`:

1. **Task** (`tasks/base.py`): Defines label space, metrics, and I/O format
   - Examples: `ner_token`, `ner_span`, `classification`, `mcqa`, `similarity`, `open_qa`
   - Each task implements: `label_space`, `compute_metrics()`, `format_output()`

2. **DatasetAdapter** (`data/base.py`): Loads splits and preprocesses for the task
   - Examples: `emea`, `medline`, `cas1`, `cas2`, `fracco_expression_ner`
   - Each adapter implements: `load_splits()`, `preprocess()`
   - Data sources are in `data/` directory (BRAT format for NER, CSV for classification)

3. **ModelAdapter** (`models/base.py`): Builds model, exposes training/inference
   - Examples: `bert_token_ner`, `vllm_ner`, `vllm_classification`, `gliner_zero_shot`
   - Each adapter implements: `supports_finetune`, `build_model()`, `get_trainer()`, `predict()`

### Registration System

All components register in `src/tabib/__init__.py`:
```python
register_task("ner_span", tasks.NERSpanTask)
register_dataset("emea", data.EMEAAdapter)
register_model("bert_token_ner", models.BERTTokenNERAdapter)
```

Registry lookups happen in `registry.py` via `get_task()`, `get_dataset()`, `get_model()`.

### Pipeline Flow

1. Load config from YAML → `RunConfig` (via `config.py`)
2. Instantiate task, dataset adapter, model adapter from registry
3. Load dataset splits (`load_splits()`)
4. Apply preprocessor if configured (e.g., `sentence_chunker` for long documents)
5. Preprocess each split for the task (`preprocess()`)
6. Build model (`build_model()`)
7. **If `do_train=True` and `supports_finetune=True`**: Get trainer and train
8. **If `do_eval=True`**: Run inference (`predict()`) and compute metrics (`compute_metrics()`)
9. Return summary dict with metrics, training info, config metadata

Pipeline lives in `pipeline.py` with core logic in `Pipeline.run()`.

## Key Abstractions

### NER Tasks: Token vs Span

- **`ner_token`**: IOB2 tagging (for BERT-like models)
  - Aligns labels with tokenizer outputs
  - Evaluates at token level, then converts to spans

- **`ner_span`**: Character-offset spans (for GLiNER, vLLM, or BERT with span conversion)
  - Predictions are `{start, end, label, text}`
  - Uses `SpanEvaluator` for exact/partial F1 metrics
  - Supports nested entities

### Preprocessing

Two main preprocessors in `preprocessing/`:

- **`SentenceChunker`**: Splits long documents into sentence-based chunks
  - Preserves entity integrity (no mid-entity splits)
  - Used for NER to fit within model token limits (e.g., 512 for BERT, 4096 for LLMs)
  - Reassembles predictions back to document offsets

- **`SentenceSplitter`**: One sentence per sample
  - Used for sentence-level tasks (similarity, classification)

Enable via config:
```yaml
preprocessing:
  type: sentence_chunker
  max_tokens: 512
```

### Model Backends

- **BERT-like models** (`bert_token_ner`, `bert_text_cls`, `bert_similarity`):
  - Support fine-tuning via `transformers.Trainer`
  - Fast inference with batch processing
  - Use `model_name_or_path` for HuggingFace model ID or local path

- **vLLM models** (`vllm_ner`, `vllm_classification`, `vllm_open_qa`):
  - Inference-only (no fine-tuning)
  - Use structured outputs (Pydantic schemas) for constrained generation
  - NER uses context-based extraction (entity + surrounding words) with text matching
  - Requires `poetry install --extras vllm`

- **GLiNER** (`gliner_zero_shot`):
  - Zero-shot NER without fine-tuning
  - Predicts entities from provided labels

## Adding New Components

### Adding a New Dataset

1. Create adapter in `src/tabib/data/your_dataset.py`:
```python
from tabib.data.base import DatasetAdapter

class YourDatasetAdapter(DatasetAdapter):
    @property
    def name(self) -> str:
        return "your_dataset"

    def load_splits(self) -> dict[str, Any]:
        # Load train/val/test splits
        pass

    def preprocess(self, dataset: Any, task: Any) -> Any:
        # Convert to task format
        pass
```

2. Import and register in `src/tabib/__init__.py`:
```python
register_dataset("your_dataset", data.YourDatasetAdapter)
```

3. Create config file in `configs/`:
```yaml
task: ner_span
dataset: your_dataset
model: bert_token_ner
model_name_or_path: camembert-base
do_train: true
do_eval: true
```

### Adding a New Model

Follow same pattern as datasets. See `src/tabib/models/vllm_ner.py` for a complete example.

### Adding a New Task

Tasks define metrics and label space. See `src/tabib/tasks/ner_span.py` for span-based NER example.

## Configuration

Config files are YAML in `configs/`. Key fields:

```yaml
task: ner_span                          # Task name (from registry)
dataset: emea                           # Dataset name (from registry)
model: bert_token_ner                   # Model name (from registry)
model_name_or_path: camembert-base      # HF model ID or path
do_train: true                          # Whether to train
do_eval: true                           # Whether to evaluate
preprocessing:                          # Optional preprocessing
  type: sentence_chunker
  max_tokens: 512
training:                               # Training config (if do_train=true)
  output_dir: runs/emea_camembert
  num_train_epochs: 10
  per_device_train_batch_size: 8
  learning_rate: 2e-5
  seed: 42
  metric_for_best_model: exact_f1
  greater_is_better: true
  early_stopping_patience: 3
backend_args:                           # Model-specific args
  max_length: 512
```

## Datasets

French biomedical datasets in `data/`:

- **EMEA**: European Medicines Agency documents (BRAT format, NER)
- **MEDLINE**: French biomedical abstracts (BRAT format, NER)
- **CAS1, CAS2**: Clinical case notes (BRAT format, NER)
- **FRACCO**: French cancer reports (BRAT format, NER + classification)
- **JNLPBA**: Biomedical NER (token-level)
- **MediQAl** (ANR-MALADES/MediQAl): French medical QA with three subtasks
  - **MCQU** (`mediqal_mcqu`): Single-answer multiple-choice questions
  - **MCQM** (`mediqal_mcqm`): Multi-answer multiple-choice questions
  - **OEQ** (`mediqal_oeq`): Open-ended questions
- **French Med MCQA Extended**: French medical multiple choice

BRAT format: `.txt` files for text, `.ann` files for annotations.

## Evaluation

Metrics vary by task:

- **NER (span)**: `exact_f1`, `partial_f1`, per-entity-type F1
- **NER (token)**: Token-level accuracy, F1 via `seqeval`
- **Classification**: Accuracy, F1, precision, recall
- **MCQA**: Exact match accuracy
- **Similarity**: Spearman/Pearson correlation

Evaluation logic lives in task classes and `evaluation/` for complex cases.

## Common Workflows

### Train CamemBERT on EMEA NER
```bash
poetry run tabib train configs/ner_emea_camembert.yaml
```

### Evaluate vLLM model (inference-only)
```bash
poetry run tabib eval configs/ner_emea_vllm.yaml
```

### Compare CamemBERT vs Qwen on multiple datasets
```bash
poetry run tabib compare configs/compare_example.yaml
```

### Debug predictions
```bash
poetry run python playground/09_check_predictions.py
```

## Comparison Framework

The `compare` command runs multiple configs in batch:

1. Create comparison spec in `configs/compare_*.yaml`
2. Define experiments with datasets and models
3. Optionally override model configs per experiment
4. Run with `tabib compare`
5. Results saved as JSON with all metrics

Example spec structure:
```yaml
description: Example comparison
defaults:
  do_train: false
  do_eval: true
datasets:
  ner_emea: ner_emea_camembert.yaml      # Reference to base configs
  ner_medline: ner_medline_camembert.yaml
experiments:
  camembert_vs_qwen:
    datasets: [ner_emea, ner_medline]
    models:
      - name: camembert                   # Uses base config model
      - name: qwen                        # Overrides for this model
        model: vllm_ner
        model_name_or_path: Qwen/Qwen2.5-7B-Instruct
output_path: results/comparison.json
```

Code lives in `src/tabib/comparison/`.

## Design Principles

1. **One adapter per file**: Keep files short (~100-300 LOC)
2. **Explicit registration**: All components register in `__init__.py`
3. **Task-agnostic pipeline**: Same flow for all tasks
4. **Config-driven**: All runs specified in YAML
5. **Reproducibility**: Fixed seeds, config stored in outputs
6. **Strong typing**: Pydantic models for configs, type hints everywhere
7. **Minimal dependencies**: Core deps only, optional extras for vLLM/LoRA

## Notes

- This is a research framework focused on French biomedical NLP
- Dataset paths are relative to `data/` directory
- Training outputs respect `$SCRATCH` environment variable for HPC setups
- vLLM models use structured outputs for reliability (Pydantic schemas)
- Context-based NER extraction compensates for LLM character-offset limitations
- Sentence chunking preserves entity boundaries for fair evaluation

## Changelog & Development Log

### 2025-11-27: Critical 3-Shot Learning Bug Fixes

**CRITICAL BUG FIXED**: Few-shot learning was broken due to passing chunked training data to model

**Problem identified**:
1. Pipeline was passing train_data to `build_model()` AFTER sentence chunking preprocessing
2. Chunked data lost the original `entities` field, resulting in empty few-shot examples
3. `_create_few_shot_examples()` looked for `spans` key but MEDLINE data uses `entities` key
4. Result: All "3-shot" evaluations were actually zero-shot with empty examples

**Fixes applied**:
- `src/tabib/pipeline.py:88-108`: Save original train data BEFORE chunking, pass to build_model()
- `src/tabib/models/vllm_ner.py:165`: Check both `spans` and `entities` keys for compatibility
- `src/tabib/models/vllm_ner.py:327-331`: Added `weave.publish()` to log raw LLM outputs for debugging
- `src/tabib/models/vllm_ner.py:337-459`: Added smart fuzzy matching for entity location
  - Handles ligatures: œ ↔ oe, æ ↔ ae
  - Removes accents: é → e, à → a, etc.
  - Case insensitive + whitespace normalization
  - Falls back to exact match first for speed

**Impact**:
- Qwen3-7B MEDLINE result (28.42% F1) was with broken 3-shot (empty examples)
- Need to re-run all models with fixed 3-shot implementation
- Gaperon-8B currently running with REAL 3-shot examples for first valid test

**Weave integration**:
- Raw LLM outputs now logged to W&B Weave with input text, generated text, and full prompt
- Enables debugging of model behavior and prompt engineering

### 2025-11-26: vLLM Completion Model Support & MediQAl Comparison

**Added**: Support for both chat and completion models in `vllm_classification` adapter

**Changes to `src/tabib/models/vllm_classification.py`**:
- Added `use_chat: bool` parameter (default `True`) to `_VLLMResources` dataclass
- Added `use_chat` parameter to `build_model()` method signature
- Modified `predict()` method to branch on model type:
  - If `use_chat=True`: Use chat API (`chat_with_vllm()`) with message formatting
  - If `use_chat=False`: Use generate API (`engine.llm.generate()`) with direct prompts
- For completion models, system prompt is prepended directly to user prompt

**Usage in config files**:
```yaml
backend_args:
  use_chat: false  # Set to false for completion/base models
  prompt_template: |
    Your prompt here...
  system_prompt: >
    System instructions here...
```

**Why this matters**:
- Completion models (like Gaperon, base models) don't have chat templates
- Previously would error with: "default chat template is no longer allowed"
- Now supports both model types seamlessly through configuration
- Enables fair comparison between chat models (Qwen3, Llama-Chat) and completion models (Gaperon, base Llama)

**MediQAl Evaluation Results** (Qwen3-8B vs Gaperon-1125-8B):
- Dataset: ANR-MALADES/MediQAl (French medical QA)
- Zero-shot evaluation results:
  - **MCQU (single-answer)**: Gaperon 20.91% vs Qwen3 19.78% (Gaperon +1.13pp)
  - **MCQM (multi-answer)**: Qwen3 5.23% vs Gaperon 4.20% (Qwen3 +1.03pp)
- Both models show similar zero-shot performance
- Multi-answer tasks significantly harder than single-answer (75% accuracy drop)
- Full results in `results/mediqal_qwen3_vs_gaperon_comparison.md`

**Created configs**:
- `configs/mcqa_mediqal_mcqu_qwen3.yaml` (chat model)
- `configs/mcqa_mediqal_mcqm_qwen3.yaml` (chat model)
- `configs/mcqa_mediqal_mcqu_gaperon.yaml` (completion model with `use_chat: false`)
- `configs/mcqa_mediqal_mcqm_gaperon.yaml` (completion model with `use_chat: false`)
- `configs/compare_mediqal_qwen3_vs_gaperon.yaml`

### 2025-11-26: FrenchMedMCQA-extended & multi-label metrics
- New MCQM configs for `french_med_mcqa_extended` across all models:
  - `configs/mcqa_frenchmedext_mcqm_qwen3.yaml`
  - `configs/mcqa_frenchmedext_mcqm_gaperon.yaml`
  - `configs/mcqa_frenchmedext_mcqm_medgemma.yaml`
  - `configs/mcqa_frenchmedext_mcqm_medgemma_pt.yaml`
  - `configs/mcqa_frenchmedext_mcqm_eurollm.yaml`
- MCQA task now reports additional multi-answer metrics: `hamming_score` and `emr` (subset accuracy).
- W&B uploader now parses all metrics and logs FrenchMedExt columns (accuracy, hamming, emr) alongside MediQAl results.
- Latest consolidated W&B run: `https://wandb.ai/rnar/mediqal-french-medical-qa/runs/zhkkk2ya`

### 2025-11-27: Weave Tracing for MCQA & Gaperon-24B Patch

**Added**: Weave tracing to `vllm_classification.py` for QA debugging
- Added `import weave` and `@weave.op()` decorator to `_log_llm_call` method
- Each LLM call now logs: input_text, prompt, raw_output, expected_label
- Enables inspection of LLM behavior in W&B Weave UI

**Added**: Monkey patch for Gaperon-24B (OLMo2 architecture) vLLM compatibility
- New file: `src/tabib/models/gaperon_patch.py`
- Fixes `head_dim` mismatch in vLLM's OLMo2 implementation
- Reference: https://huggingface.co/almanach/Gaperon-1125-24B/discussions/1
- Patch automatically applied in `vllm_common.py:create_vllm_engine()`

**MedGemma-27B MCQA Results** (0-shot, fp8 quantization):
- **MediQAl MCQU**: 36.77% accuracy (single-answer)
- **MediQAl MCQM**: 9.16% accuracy (multi-answer)
- Model loads in ~27 GiB with fp8, inference at ~12k input tokens/s

**Pending evaluations**:
- MedGemma-27B on FrenchMedMCQA (in progress)
- MedGemma-27B 3-shot variants
- Qwen3-32B all datasets
- Gaperon-24B (needs testing with patch)

### Current Status (2025-11-27 16:30)

**Completed MedGemma-27B 0-shot evaluations**:
| Dataset | Metric | Value |
|---------|--------|-------|
| MediQAl MCQU (single-answer) | Accuracy | 36.77% |
| MediQAl MCQM (multi-answer) | Accuracy | 9.16% |
| FrenchMedMCQA (multi-answer) | Accuracy | 14.31% |

**Crux - Gaperon-24B vLLM Incompatibility**:
- Model: `almanach/Gaperon-1125-24B` (OLMo2 architecture)
- Issue: Weight dimension mismatch - model has `head_dim=128` with `num_attention_heads=32` instead of 40
- vLLM calculates `head_dim = hidden_size // num_attention_heads` which doesn't match trained weights
- Initial patch failed due to vLLM 0.11 API change (new `vllm_config` parameter)
- Patch updated to match vLLM 0.11 signature: `(self, *, vllm_config: VllmConfig, prefix: str = '')`

**QA 3-shot Support**:
- Already implemented in `vllm_classification.py`
- Uses `num_few_shot` parameter in config
- Few-shot examples taken from train_data and prepended to prompts

**Next Steps**:
1. Test Gaperon-24B with updated patch
2. Run MedGemma-27B 3-shot evaluations (configs exist with `num_few_shot: 3`)
3. Run Qwen3-32B evaluations (0-shot + 3-shot)
