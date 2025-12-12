# ModernCamembert 3-Epoch Benchmark Results

## Summary Table

| Model | EMEA F1 | MEDLINE F1 | FRACCO ICD F1 | Avg |
|-------|---------|------------|---------------|-----|
| moderncamembert-code-mix | 58.9% | 49.0% | 46.6% | 51.5% |
| moderncamembert-full-mix-proportional | 58.4% | 49.3% | 46.5% | 51.4% |
| moderncamembert-full-mix-equal | 58.1% | 49.1% | 46.9% | 51.4% |
| moderncamembert-istex-original | 57.5% | 48.9% | 47.4% | 51.3% |
| moderncamembert-hal-scientific | 57.9% | 49.1% | 46.6% | 51.2% |
| moderncamembert-emea-notice | 57.7% | 49.0% | 46.7% | 51.1% |
| moderncamembert-hal-original | 56.9% | 49.3% | 46.8% | 51.0% |
| moderncamembert-atc | 57.2% | 48.8% | 47.1% | 51.0% |
| moderncamembert-cim10 | 56.7% | 49.4% | 46.9% | 51.0% |
| moderncamembert-hal-textbook | 57.6% | 49.1% | 46.3% | 51.0% |
| moderncamembert-fineweb2-only | 57.7% | 49.3% | 45.9% | 51.0% |
| moderncamembert-istex-textbook | 56.7% | 48.9% | 47.0% | 50.9% |
| moderncamembert-istex-scientific | 56.9% | 49.1% | 46.6% | 50.9% |
| moderncamembert-emea-original | 57.2% | 49.1% | 46.3% | 50.9% |
| moderncamembert-synthetic-clinical | 57.2% | 48.7% | 46.6% | 50.8% |
| moderncamembert-istex-abstract | 56.4% | 49.4% | 46.6% | 50.8% |
| almanach--moderncamembert-base | 56.6% | 48.6% | 47.0% | 50.7% |
| moderncamembert-ccam | 56.1% | 49.1% | 46.9% | 50.7% |
| moderncamembert-pmc-patients | 55.9% | 49.3% | 46.8% | 50.7% |

## Notes
- EMEA: Named Entity Recognition on pharmaceutical texts (F1 score)
- MEDLINE: Named Entity Recognition on biomedical abstracts (F1 score)  
- FRACCO ICD: ICD code classification (F1 score, 5-seed average)
- All models trained for 3 epochs of continued pretraining on domain-specific data
- Base model: almanach/moderncamembert-base
