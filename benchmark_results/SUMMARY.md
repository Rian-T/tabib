# ModernCamembert Benchmark Results

## Overview

Comprehensive evaluation of ModernCamembert continued pre-training (CPT) experiments on French biomedical tasks.

- **Seeds**: 5 (42, 43, 44, 45, 46)
- **Models**: 19 CPT variants + 3 baselines (DrBERT, camembert-bio, camembert-base)

---

## 3-Epoch NER Evaluation (F1 scores)

| Model | EMEA NER | MEDLINE NER |
|-------|----------|-------------|
| moderncamembert-code-mix | 58.9% | 49.0% |
| moderncamembert-full-mix-proportional | 58.4% | 49.3% |
| moderncamembert-full-mix-equal | 58.1% | 49.1% |
| moderncamembert-hal-scientific | 57.9% | 49.1% |
| moderncamembert-emea-notice | 57.7% | 49.0% |
| moderncamembert-fineweb2-only | 57.7% | 49.3% |
| moderncamembert-hal-textbook | 57.6% | 49.1% |
| moderncamembert-istex-original | 57.5% | 48.9% |
| moderncamembert-emea-original | 57.2% | 49.1% |
| moderncamembert-synthetic-clinical | 57.2% | 48.7% |
| moderncamembert-atc | 57.2% | 48.8% |
| moderncamembert-istex-scientific | 56.9% | 49.1% |
| moderncamembert-hal-original | 56.9% | 49.3% |
| moderncamembert-cim10 | 56.7% | 49.4% |
| moderncamembert-istex-textbook | 56.7% | 48.9% |
| almanach--moderncamembert-base | 56.6% | 48.6% |
| moderncamembert-istex-abstract | 56.4% | 49.4% |
| moderncamembert-ccam | 56.1% | 49.1% |
| moderncamembert-pmc-patients | 55.9% | 49.3% |

---

## 3-Epoch Classification Evaluation (Accuracy)

| Model | DiaMED | MorFITT | CLISTER |
|-------|--------|---------|---------|
| moderncamembert-full-mix-proportional | 60.4% | 61.5% | 89.7% |
| moderncamembert-synthetic-clinical | 60.4% | 62.8% | 89.7% |
| moderncamembert-hal-scientific | 60.0% | 62.1% | 89.5% |
| moderncamembert-emea-notice | 59.7% | 63.1% | 89.4% |
| moderncamembert-emea-original | 59.7% | 62.8% | 89.4% |
| moderncamembert-cim10 | 59.4% | 63.1% | 89.6% |
| almanach--moderncamembert-base | 59.2% | 62.9% | 89.1% |
| moderncamembert-hal-original | 59.1% | 62.7% | 89.7% |
| moderncamembert-pmc-patients | 59.0% | 62.0% | 89.3% |
| moderncamembert-code-mix | 58.8% | 62.9% | 89.3% |
| moderncamembert-full-mix-equal | 58.7% | 62.1% | 89.5% |
| moderncamembert-fineweb2-only | 58.7% | 62.4% | 89.4% |
| moderncamembert-ccam | 58.4% | 62.4% | 89.7% |
| moderncamembert-istex-textbook | 58.3% | 62.3% | 89.0% |
| moderncamembert-hal-textbook | 58.2% | 62.3% | 89.6% |
| moderncamembert-istex-abstract | 57.4% | 62.5% | 89.6% |
| moderncamembert-istex-original | 57.0% | 62.2% | 89.3% |
| moderncamembert-atc | 56.5% | 62.7% | 89.3% |
| moderncamembert-istex-scientific | 55.8% | 61.6% | 89.6% |

---

## FRACCO ICD Classification (22 models with baselines)

### Full ICD (~4500 classes) - F1 scores

| Model | fracco_icd_full | fracco_icd_top200 | fracco_icd_top100 |
|-------|-----------------|-------------------|-------------------|
| **DrBERT-7GB** | **48.01%** | 91.23% | - |
| moderncamembert-full-mix-proportional | 43.27% | 87.97% | 92.08% |
| almanach--moderncamembert-base | 43.21% | 87.77% | 91.56% |
| moderncamembert-code-mix | 43.09% | 87.40% | 91.27% |
| moderncamembert-istex-original | 42.98% | 87.94% | 91.89% |
| moderncamembert-pmc-patients | 42.98% | 87.35% | 91.25% |
| moderncamembert-emea-notice | 42.95% | 87.99% | 91.93% |
| moderncamembert-synthetic-clinical | 42.88% | 87.46% | 91.67% |
| moderncamembert-atc | 42.85% | 88.13% | 91.47% |
| moderncamembert-full-mix-equal | 42.84% | 88.21% | 92.09% |
| moderncamembert-cim10 | 42.75% | 88.04% | 91.30% |
| moderncamembert-fineweb2-only | 42.74% | 87.34% | 92.31% |
| moderncamembert-istex-textbook | 42.74% | 86.80% | 91.70% |
| moderncamembert-istex-scientific | 42.72% | 87.21% | 91.64% |
| moderncamembert-hal-original | 42.72% | 88.05% | 92.25% |
| moderncamembert-hal-scientific | 42.58% | 88.01% | 91.86% |
| moderncamembert-emea-original | 42.43% | 87.38% | 92.27% |
| moderncamembert-hal-textbook | 42.39% | 87.72% | 92.26% |
| moderncamembert-istex-abstract | 42.22% | 88.10% | 91.78% |
| moderncamembert-ccam | 42.22% | 87.79% | 92.10% |
| **camembert-bio-base** | 38.73% | **92.39%** | - |
| camembert-base | 37.66% | 91.36% | - |

### Key Findings - FRACCO ICD

1. **DrBERT dominates on full ICD** (48% vs 42-43% for ModernCamembert) - specialized biomedical pretraining is crucial
2. ModernCamembert CPT models show **no significant improvement** over base on full ICD (~42-43%)
3. On top100/top200, task is **saturated** (~91-92%), differences are minimal
4. camembert-bio-base outperforms on top200 (92.4%) - possible tokenizer advantage

---

## MedDialog Women Classification

| Model | Accuracy |
|-------|----------|
| moderncamembert-cim10 | 8.45% |
| moderncamembert-hal-scientific | 8.41% |
| moderncamembert-hal-original | 8.37% |
| moderncamembert-emea-notice | 8.37% |
| moderncamembert-full-mix-equal | 8.29% |
| moderncamembert-istex-scientific | 8.23% |
| almanach--moderncamembert-base | 8.13% |
| moderncamembert-istex-abstract | 8.04% |
| moderncamembert-pmc-patients | 8.04% |
| moderncamembert-emea-original | 7.89% |
| moderncamembert-istex-original | 7.79% |
| moderncamembert-fineweb2-only | 7.70% |
| moderncamembert-atc | 7.69% |
| moderncamembert-synthetic-clinical | 7.52% |
| moderncamembert-ccam | 7.46% |
| moderncamembert-hal-textbook | 7.31% |
| moderncamembert-full-mix-proportional | 7.30% |
| moderncamembert-code-mix | 7.28% |
| moderncamembert-istex-textbook | 7.18% |

Note: Very low scores (~7-8%) indicate this is a challenging task for all models.

---

## 1-Epoch Evaluation Summary (24 models)

| Model | EMEA | MEDLINE | DiaMED | MorFITT | CLISTER | Avg |
|-------|------|---------|--------|---------|---------|-----|
| moderncamembert-full-mix-proportional | 53.2 | 41.3 | 56.0 | 58.5 | 90.0 | 52.2 |
| merged-ablation-dare-ties | 53.5 | 40.9 | 54.0 | 59.8 | 90.3 | 52.1 |
| moderncamembert-hal-original | 51.1 | 40.3 | 57.3 | 59.1 | 89.7 | 51.9 |
| moderncamembert-full-mix-equal | 52.1 | 41.6 | 54.8 | 58.3 | 90.0 | 51.7 |
| moderncamembert-cim10 | 52.2 | 39.9 | 55.2 | 58.9 | 89.8 | 51.5 |
| merged-ablation-ties | 51.9 | 40.6 | 54.2 | 59.4 | 89.8 | 51.5 |
| moderncamembert-synthetic-clinical | 49.9 | 40.7 | 54.5 | 59.3 | 90.0 | 51.1 |
| moderncamembert-hal-textbook | 50.6 | 40.9 | 53.6 | 59.2 | 89.8 | 51.1 |
| merged-ablation-slerp | 52.0 | 41.4 | 51.3 | 58.8 | 89.6 | 50.9 |
| moderncamembert-fineweb2-only | 50.2 | 41.2 | 53.2 | 58.7 | 89.9 | 50.8 |
| moderncamembert-atc | 53.0 | 41.3 | 50.1 | 58.7 | 89.8 | 50.8 |
| almanach--moderncamembert-base | 49.8 | 41.8 | 51.0 | 59.8 | 89.7 | 50.6 |

---

## Dataset Descriptions

| Dataset | Task | Metric | Description |
|---------|------|--------|-------------|
| EMEA | NER | F1 | Named Entity Recognition on pharmaceutical texts |
| MEDLINE | NER | F1 | Named Entity Recognition on biomedical abstracts |
| DiaMED | Classification | Accuracy | Medical dialogue classification |
| MorFITT | Classification | Accuracy | French medical text classification |
| CLISTER | Similarity | Spearman | Semantic similarity (saturated ~89%) |
| FRACCO ICD | Classification | F1 | ICD-10 code classification (~4500 classes) |
| FRACCO ICD top100/200 | Classification | F1 | Top 100/200 most frequent ICD codes |
| MedDialog Women | Classification | Accuracy | Medical dialogue classification |

---

## Conclusions

1. **3-epoch CPT improves NER**: EMEA F1 improved from ~50% (1-epoch) to ~57-59% (3-epoch)
2. **MEDLINE NER saturated**: All models ~49% regardless of CPT
3. **DrBERT remains best for ICD coding**: 48% vs 42-43% for ModernCamembert
4. **CPT data source matters**: code-mix and full-mix-proportional perform best overall
5. **CLISTER saturated**: All models ~89%, not discriminative
6. **MedDialog Women very hard**: All models ~7-8%, needs investigation

## Training Details

All 3-epoch CPT models trained successfully:
- Loss curves show proper convergence
- Final eval accuracy: 81-90% (MLM task)
- WandB logs synced: https://wandb.ai/rntc/moderncamembert-biomed
