# CLAUDE.md

## 🎯 Goal: DrBenchmark W&B Table

Compare BERT models on French biomedical tasks. W&B: https://wandb.ai/rnar/tabib-drbenchmark

### Models
| Model | HuggingFace ID |
|-------|----------------|
| ModernCamemBERT | `almanach/moderncamembert-base` |
| CamemBERT-bio | `almanach/camembert-bio-base` |
| CamemBERT-base | `camembert-base` |
| CamemBERTv2 | `almanach/camembertv2-base` |
| CamemBERTv2-bio-gt3 | `rntc/camembertv2-bio-edu_gt3` |

---

## Results (2025-11-30)

### ESSAI (4 classes - negation/speculation)
| Model | Accuracy | F1 |
|-------|----------|-----|
| **CamemBERTv2** | **98.34%** | **96.47%** |
| CamemBERT-bio | 98.07% | 92.24% |
| CamemBERTv2-bio-gt3 | 98.21% | 88.99% |
| CamemBERT-base | 96.97% | 90.59% |
| ModernCamemBERT | 96.28% | 89.96% |

### DiaMED (22 classes - ICD-10 chapters)
| Model | Accuracy | F1 |
|-------|----------|-----|
| **CamemBERTv2-bio-gt3** | **56.73%** | 25.54% |
| CamemBERTv2 | 56.14% | **26.91%** |
| ModernCamemBERT | 52.05% | 28.94% |
| CamemBERT-bio | 50.29% | 21.14% |
| CamemBERT-base | 46.78% | 13.13% |

### MORFITT (12 classes - medical specialties)
| Model | Accuracy | F1 |
|-------|----------|-----|
| **CamemBERTv2-bio-gt3** | - | **47.75%** |
| CamemBERT-bio | - | 44.47% |
| CamemBERT-base | - | 39.53% |
| CamemBERTv2 | - | 38.45% |
| ModernCamemBERT | - | - |

### CLISTER (Semantic Similarity)
| Model | Spearman | Pearson |
|-------|----------|---------|
| **CamemBERT-base** | **89.91%** | 88.60% |
| CamemBERTv2-bio-gt3 | 89.82% | **89.22%** |
| CamemBERTv2 | 88.95% | 87.98% |
| CamemBERT-bio | 88.96% | 86.11% |
| ModernCamemBERT | 88.36% | 88.42% |

### FRACCO Top50 ICD
| Model | Accuracy | F1 |
|-------|----------|-----|
| **ModernCamemBERT** | **80.71%** | **38.70%** |
| CamemBERT-bio | 66.07% | 11.94% |
| CamemBERT-base | 64.55% | 10.34% |

### NER Results (2025-12-01) - Line-by-Line Splitting

Documents split line-by-line (one sample per line) for optimal NER performance.

| Dataset | ModernCamemBERT | CamemBERT-bio | CamemBERT | CamemBERTv2 | CamemBERTv2-bio-gt3 |
|---------|-----------------|---------------|-----------|-------------|---------------------|
| **EMEA** | 44.28% | 52.61% | 51.58% | 57.46% | **58.06%** |
| **CAS1** | 45.30% | 43.27% | 37.16% | **54.71%** | 52.96% |
| **CAS2** | 54.63% | 56.83% | 51.36% | **67.45%** | 65.23% |

**Note**: CamemBERT initially had 0% F1 due to preprocessing bug (sentence_splitter). Fixed and rerun.

### MCQA Results (2025-12-01) - MedIQAL Benchmark

BERT models struggle with MCQA - very low performance (~5% accuracy). Running remaining models...

| Dataset | CamemBERT | ModernCamemBERT | CamemBERT-bio | CamemBERTv2 | CamemBERTv2-bio-gt3 |
|---------|-----------|-----------------|---------------|-------------|---------------------|
| MCQM    | 5.53% / 1.52% | 5.05% / 1.12% | Running... | - | - |
| MCQU    | - | - | - | - | - |

---

## Key Findings

### Classification Tasks
1. **CamemBERTv2** wins on ESSAI (96.47% F1) - modern RoBERTa architecture helps
2. **CamemBERTv2-bio-gt3** best on MORFITT (47.75% F1) - biomedical pretraining helps
3. **ModernCamemBERT** dominates FRACCO (38.70% F1) - 3.5x better than CamemBERT-bio
4. **CamemBERT-base** slightly best on CLISTER similarity

### NER Tasks (2025-12-01)
5. **Line-by-line splitting** vastly improves NER: EMEA F1 34.5% → 51.75%
6. Each line = one sample, much more granular than 2048-char chunks
7. Running full benchmark with all models...

---

## Golden Rules
- Update this file after findings
- Git commit specific files (never `git add .`)
- Use `tail -30 results/*.log` to check progress

## Commands
```bash
poetry run tabib train configs/your_config.yaml
poetry run tabib eval configs/your_config.yaml
```
