# ModernCamembert Benchmark Results

## 1-Epoch Evaluation (24 models, 5-seed averages)

| Model | EMEA | MEDLINE | DiaMED | MorFITT | CLISTER | Avg (no SIM) |
|-------|------|---------|--------|---------|---------|--------------|
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
| moderncamembert-istex-textbook | 48.9 | 40.1 | 54.0 | 59.8 | 90.0 | 50.7 |
| moderncamembert-pmc-patients | 51.3 | 40.7 | 52.3 | 58.4 | 89.8 | 50.7 |
| almanach--moderncamembert-base | 49.8 | 41.8 | 51.0 | 59.8 | 89.7 | 50.6 |
| moderncamembert-istex-scientific | 49.7 | 40.9 | 52.1 | 59.7 | 89.7 | 50.6 |
| moderncamembert-hal-scientific | 47.8 | 40.7 | 53.5 | 60.3 | 89.8 | 50.6 |
| moderncamembert-istex-abstract | 50.9 | 40.7 | 50.9 | 59.8 | 89.5 | 50.6 |
| moderncamembert-ccam | 49.1 | 40.6 | 53.1 | 59.4 | 90.0 | 50.5 |
| moderncamembert-emea-notice | 49.7 | 41.0 | 52.6 | 58.4 | 89.7 | 50.4 |
| moderncamembert-istex-original | 51.1 | 41.0 | 50.8 | 58.7 | 89.5 | 50.4 |
| moderncamembert-emea-original | 48.7 | 40.8 | 52.9 | 58.5 | 89.7 | 50.2 |
| DrBERT-7GB | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| camembert-bio-base | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| camembert-base | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## 3-Epoch Evaluation (19 models, 5-seed averages)

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

- **EMEA**: Named Entity Recognition on pharmaceutical texts (F1)
- **MEDLINE**: Named Entity Recognition on biomedical abstracts (F1)
- **DiaMED**: Medical dialogue classification (Accuracy)
- **MorFITT**: French medical text classification (Accuracy)
- **CLISTER**: Semantic similarity (Spearman) - saturated ~89%
- **FRACCO ICD**: ICD-10 code classification (F1)

## Notes

- Base model: almanach/moderncamembert-base
- 5-seed averaging (seeds: 42, 43, 44, 45, 46)
- 1-epoch: includes baselines (camembert-base, camembert-bio, DrBERT) and merge experiments
- 3-epoch: CPT models only
- Remaining: fracco_icd_top100, meddialog_women (running)
