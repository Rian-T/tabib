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
