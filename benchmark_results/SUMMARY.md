# ModernCamembert 3-Epoch Benchmark Results

## Summary Table (5-seed averages)

| Model | EMEA | MEDLINE | DiaMED | MorFITT | CLISTER | FRACCO ICD | Avg (no SIM) |
|-------|------|---------|--------|---------|---------|------------|--------------|
| moderncamembert-emea-notice | 57.7 | 49.0 | 59.7 | 63.1 | 89.4 | 46.7 | 55.2 |
| moderncamembert-code-mix | 58.9 | 49.0 | 58.8 | 62.9 | 89.3 | 46.6 | 55.2 |
| moderncamembert-full-mix-proportional | 58.4 | 49.3 | 60.4 | 61.5 | 89.7 | 46.5 | 55.2 |
| moderncamembert-hal-scientific | 57.9 | 49.1 | 60.0 | 62.1 | 89.5 | 46.6 | 55.1 |
| moderncamembert-synthetic-clinical | 57.2 | 48.7 | 60.4 | 62.8 | 89.7 | 46.6 | 55.1 |
| moderncamembert-cim10 | 56.7 | 49.4 | 59.4 | 63.1 | 89.6 | 46.9 | 55.1 |
| moderncamembert-emea-original | 57.2 | 49.1 | 59.7 | 62.8 | 89.4 | 46.3 | 55.0 |
| moderncamembert-hal-original | 56.9 | 49.3 | 59.1 | 62.7 | 89.7 | 46.8 | 55.0 |
| moderncamembert-full-mix-equal | 58.1 | 49.1 | 58.7 | 62.1 | 89.5 | 46.9 | 55.0 |
| almanach--moderncamembert-base | 56.6 | 48.6 | 59.2 | 62.9 | 89.1 | 47.0 | 54.9 |
| moderncamembert-fineweb2-only | 57.7 | 49.3 | 58.7 | 62.4 | 89.4 | 45.9 | 54.8 |
| moderncamembert-hal-textbook | 57.6 | 49.1 | 58.2 | 62.3 | 89.6 | 46.3 | 54.7 |
| moderncamembert-istex-textbook | 56.7 | 48.9 | 58.3 | 62.3 | 89.0 | 47.0 | 54.6 |
| moderncamembert-ccam | 56.1 | 49.1 | 58.4 | 62.4 | 89.7 | 46.9 | 54.6 |
| moderncamembert-istex-original | 57.5 | 48.9 | 57.0 | 62.2 | 89.3 | 47.4 | 54.6 |
| moderncamembert-pmc-patients | 55.9 | 49.3 | 59.0 | 62.0 | 89.3 | 46.8 | 54.6 |
| moderncamembert-atc | 57.2 | 48.8 | 56.5 | 62.7 | 89.3 | 47.1 | 54.4 |
| moderncamembert-istex-abstract | 56.4 | 49.4 | 57.4 | 62.5 | 89.6 | 46.6 | 54.4 |
| moderncamembert-istex-scientific | 56.9 | 49.1 | 55.8 | 61.6 | 89.6 | 46.6 | 54.0 |

## Dataset Descriptions

- **EMEA**: Named Entity Recognition on pharmaceutical texts (F1 score)
- **MEDLINE**: Named Entity Recognition on biomedical abstracts (F1 score)
- **DiaMED**: Medical dialogue classification (Accuracy)
- **MorFITT**: French medical text classification (Accuracy)
- **CLISTER**: Semantic similarity (Spearman correlation)
- **FRACCO ICD**: ICD-10 code classification (F1 score)

## Notes

- All models trained for 3 epochs of continued pretraining on domain-specific data
- Base model: almanach/moderncamembert-base
- 5-seed averaging (seeds: 42, 43, 44, 45, 46)
- Remaining benchmarks (fracco_icd_top100, meddialog_women) still running
