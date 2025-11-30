# CLAUDE.md

## 🎯 FINAL GOAL: DrBenchmark W&B Comparison Table

Create comprehensive W&B table comparing all BERT models and LLMs (≤10B) on French biomedical tasks.

### BERT Models
- ModernCamemBERT (`almanach/moderncamembert-base`)
- CamemBERT-bio (`camembert/camembert-bio-base`)
- CamemBERT-base (`camembert-base`)

### LLM Models (≤10B)
- Gaperon-8B (`almanach/Gaperon-1125-8B`)
- EuroLLM-9B (`utter-project/EuroLLM-9B`)
- Qwen3-8B (`Qwen/Qwen3-8B`)
- Gemma3-4B (`google/gemma-3-4b-pt`)

### Datasets
| Dataset | Task | Classes |
|---------|------|---------|
| FRACCO ICD | Classification | 298 ICD codes |
| DiaMED | Classification | 22 ICD-10 chapters |
| ESSAI | Classification | 4 (negation/speculation) |
| CLISTER | Similarity | Sentence pairs |
| MORFITT | Classification | 12 medical specialties |
| CAS1 | NER | 2 types (sosy, pathologie) |
| CAS2 | NER | 8 types |
| MEDLINE | NER | 10 types |

---

## GOLDEN RULES
- **Update this journal regularly** - future Claude needs context!
- **Git commit specific files** with short messages (never `git add .`)
- Use `tail` on log files, not BashOutput (too verbose)
- Keep working autonomously

---

## Current Progress (2025-11-30)

### 21:01 - W&B Upload Complete

Results uploaded to W&B: https://wandb.ai/rnar/tabib-drbenchmark/runs/9dz6dagq

Created `scripts/upload_drbenchmark_results.py` for reproducibility.

### 19:15 - DrBenchmark COMPLETE!

**ESSAI (Negation/Speculation, 4 classes)**
| Model | Accuracy | F1 |
|-------|----------|-----|
| **CamemBERT-bio** | **98.07%** | **92.24%** |
| CamemBERT-base | 96.97% | 90.59% |
| ModernCamemBERT | 96.28% | 89.96% |

**DiaMED (ICD-10 chapters, 22 classes)**
| Model | Accuracy | F1 |
|-------|----------|-----|
| **ModernCamemBERT** | **52.05%** | **28.94%** |
| CamemBERT-bio | 50.29% | 21.14% |
| CamemBERT-base | 46.78% | 13.13% |

**Key insights**:
1. **ESSAI**: CamemBERT-bio wins! Biomedical pretraining helps for negation/speculation
2. **DiaMED**: ModernCamemBERT wins! Modern architecture helps for 22-class ICD-10

**Bug fixed**: CamemBERT-bio model path (`almanach/camembert-bio-base`, not `camembert/...`)

### 18:15 - DrBenchmark Trainings Started

**Bug fixed in `bert_text_cls.py`**:
- Changed `eval_strategy` assignment to use `setdefault()` (was overwriting config values)
- Changed `load_best_model_at_end` to use `setdefault()`
- Config values now properly respected

**Config fixes for DrBenchmark**:
- Reduced batch size 16→8 (OOM on ModernCamemBERT)
- Changed `metric_for_best_model: f1` → `eval_loss` (F1 not computed during training)
- Added `greater_is_better: false`

### Configs Created
- DiaMED: 3 models (moderncamembert, camembert_bio, camembert)
- ESSAI: 3 models
- CLISTER: 3 models (sim_clister_*.yaml)
- MORFITT: 3 models (cls_morfitt_*.yaml)
- FRACCO ICD: existing configs

### Previous Results

**FRACCO ICD (min_samples=10, 248 classes)**
| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| ModernCamemBERT | **80.55%** | **38.88%** |
| BioClinical-ModernBERT | 80.83% | 37.67% |
| CamemBERT-bio | 66.91% | 12.42% |
| CamemBERT-base | 63.95% | 9.79% |

**NER (BERT models)**
| Dataset | Model | exact_F1 |
|---------|-------|----------|
| CAS1 | ModernCamemBERT | **40.84%** |
| CAS2 | ModernCamemBERT | **53.21%** |
| CAS2 | camembert-base | 33.99% |
| CAS1 | camembert-base | 27.11% |

---

## Project Overview

Tabib: task-agnostic NLP evaluation framework for biomedical French text.

### Core Commands
```bash
poetry run tabib train configs/your_config.yaml
poetry run tabib eval configs/your_config.yaml
```

### Architecture
Three-component system via `Pipeline`:
1. **Task**: Defines label space, metrics
2. **DatasetAdapter**: Loads splits, preprocesses
3. **ModelAdapter**: Builds model, training/inference

Registration in `src/tabib/__init__.py`.

---

## Key Code Changes

### New Dataset Adapters (2025-11-30)
- `src/tabib/data/diamed.py` - DiaMED ICD-10 chapter classification
- `src/tabib/data/essai.py` - ESSAI negation/speculation classification
- `src/tabib/data/clister.py` - CLISTER semantic similarity
- `src/tabib/data/morfitt.py` - MORFITT medical specialty classification
- `src/tabib/data/mantragsc.py` - MANTRAGSC UMLS NER

### Bug Fixes
- B-tag parsing: Treat orphan I- tags as span start (`bert_token_ner.py`)
- DiaMED: Handle None/float in icd-10 field
- MORFITT: Use primary label for single-label classification
