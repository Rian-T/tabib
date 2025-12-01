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

## Final Results (2025-12-01)

**Benchmark complete!** All 45 runs (5 models × 9 datasets) finished successfully.

### NER Results (Line-by-Line Splitting)

Documents split line-by-line (one sample per line) for optimal NER performance.

| Dataset | CamemBERT | CamemBERT-bio | CamemBERTv2 | CamemBERTv2-bio | ModernCamemBERT |
|---------|-----------|---------------|-------------|-----------------|-----------------|
| **EMEA** | 50.91% | 52.30% | **59.21%** | 56.98% | 48.60% |
| **CAS1** | 35.50% | 39.85% | **57.62%** | 50.95% | 44.34% |
| **CAS2** | 54.15% | 57.08% | 66.93% | **68.13%** | 55.50% |

### Classification Results

| Dataset | CamemBERT | CamemBERT-bio | CamemBERTv2 | CamemBERTv2-bio | ModernCamemBERT |
|---------|-----------|---------------|-------------|-----------------|-----------------|
| **ESSAI** (F1) | 93.52% | 94.26% | **95.26%** | 90.20% | 94.44% |
| **DiaMED** (F1) | 12.14% | 22.03% | 26.11% | **26.79%** | 25.95% |
| **MORFITT** (F1) | 38.88% | 46.90% | 46.63% | **52.13%** | 41.08% |

### Semantic Similarity (CLISTER)

| Model | Spearman | Pearson |
|-------|----------|---------|
| **CamemBERT-bio** | **90.83%** | 89.12% |
| CamemBERT | 90.39% | **89.35%** |
| CamemBERTv2-bio | 89.06% | 87.95% |
| CamemBERTv2 | 88.94% | 87.78% |
| ModernCamemBERT | 88.93% | 88.99% |

### MCQA Results (MedIQAL) - BERT models struggle!

| Dataset | CamemBERT | CamemBERT-bio | CamemBERTv2 | CamemBERTv2-bio | ModernCamemBERT |
|---------|-----------|---------------|-------------|-----------------|-----------------|
| **MCQM** (Acc) | 6.89% | **7.51%** | 7.12% | 7.42% | 5.11% |
| **MCQU** (Acc) | 21.09% | **22.27%** | 19.59% | 21.57% | 20.68% |

BERT models perform poorly on MCQA (~5-7% on MCQM, ~20% on MCQU) - near random for multi-label.

---

## Key Findings (2025-12-01)

### Overall Winners by Task Type

| Task | Winner | F1 Score |
|------|--------|----------|
| **NER (EMEA)** | CamemBERTv2 | 59.21% |
| **NER (CAS1)** | CamemBERTv2 | 57.62% |
| **NER (CAS2)** | CamemBERTv2-bio | 68.13% |
| **CLS (ESSAI)** | CamemBERTv2 | 95.26% |
| **CLS (DiaMED)** | CamemBERTv2-bio | 26.79% |
| **CLS (MORFITT)** | CamemBERTv2-bio | 52.13% |
| **SIM (CLISTER)** | CamemBERT-bio | 90.83% spearman |
| **MCQA (MCQM)** | CamemBERT-bio | 7.51% |
| **MCQA (MCQU)** | CamemBERT-bio | 22.27% |

### Key Observations

1. **CamemBERTv2 dominates NER**: Best on EMEA (59.21%) and CAS1 (57.62%), consistently strong
2. **CamemBERTv2-bio best for complex CLS**: Wins DiaMED (26.79%) and MORFITT (52.13%)
3. **CamemBERT-bio excels at similarity**: Best CLISTER spearman (90.83%)
4. **BERT models fail at MCQA**: All models near random (~5-7% MCQM, ~20% MCQU)
5. **ModernCamemBERT underperforms**: Consistently lower than CamemBERTv2 variants
6. **Line-by-line splitting**: Critical for NER performance vs chunk-based approaches

---

## Golden Rules
- Update this file after findings
- Git commit specific files (never `git add .`)
- Use `tail -30 results/*.log` to check progress

## Commands

### New Benchmark System (2025-12-01)
```bash
# Run full BERT benchmark (45 runs: 5 models × 9 datasets)
poetry run tabib benchmark configs/benchmark_bert_drbenchmark.yaml

# Preview what will run (dry-run)
poetry run tabib benchmark configs/benchmark_bert_drbenchmark.yaml --dry-run
```

Output:
- `results/bert_drbenchmark.json` - structured JSON results
- `results/bert_drbenchmark.md` - markdown comparison tables
- W&B table upload (if configured)

### Legacy Commands
```bash
poetry run tabib train configs/your_config.yaml
poetry run tabib eval configs/your_config.yaml
```

## Changelog (2025-12-02)
- Added MedDialog-FR Women adapter (`meddialog_women`) - 80 multilabel classes (UMLS CUI combos)
- Registered `fracco_icd_top50` dataset variant with pre-configured top_k=50
- Updated benchmark to 70 runs (14 datasets × 5 models):
  - NER: emea, cas1, cas2, **fracco_expression_ner**
  - CLS: essai, diamed, morfitt, **fracco_icd_classification**, **fracco_icd_top50**, **meddialog_women**
  - SIM: clister
  - MCQA: mediqal_mcqm, mediqal_mcqu, **french_med_mcqa_extended**

## Changelog (2025-12-01)
- Added `tabib benchmark` command for multi-model comparisons
- Base configs in `configs/base/` (ner_bert, cls_bert, mcqa_bert, sim_bert, + llm variants)
- Model groups support: BERT vs LLM with different configs
- Automatic JSON, Markdown, and W&B output
