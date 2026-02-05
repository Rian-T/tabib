# nlstruct Port Progress

## Status: COMPLETE (ready for testing)
Date: 2025-12-28

## Problem
tabib's span-based NER achieves ~43% exact_f1, while nlstruct achieves >60% on same datasets.

## 7 Critical Differences Found

1. **BRAT fragments** - tabib only parsed first fragment, nlstruct parses all
2. **filter_nested=True** - EMEA/CAS removed nested entities by default!
3. **Negative sampling** - tabib samples 3:1, nlstruct uses ALL spans
4. **Softmax vs CRF** - tabib used softmax, nlstruct uses real CRF marginals
5. **Combined scoring** - scale mismatch between tagger and biaffine
6. **No prediction filtering** - nlstruct filters overlapping predictions
7. **Hyperparameters** - need separate LRs (BERT: 5e-5, other: 1e-3)

## All Steps Completed ✅

### 1. Fixed BRAT fragment parsing ✅
File: `src/tabib/data/brat.py`
- Now parses ALL fragments (not just first)
- Stores `fragments` list in entity dict
- Merges adjacent whitespace-only fragments (like nlstruct)

### 2. Set filter_nested=False ✅
Files:
- `src/tabib/data/emea.py` - changed default to False
- `src/tabib/data/cas.py` - changed default to False for CAS1 and CAS2

### 3. Created CRF module ✅
File: `src/tabib/models/crf.py`
- Ported `LinearChainCRF` from nlstruct
- Ported `BIOULDecoder` with BIOUL transition constraints
- Includes forward-backward algorithm for marginals
- Includes Viterbi decoding

### 4. Updated BIOULTagger to use CRF ✅
File: `src/tabib/models/bert_span_ner.py`
- BIOULTagger now has `use_crf=True` parameter
- Uses real CRF marginals instead of softmax
- Tag layout: O, then for each label: I, B, L, U (4 per label)

### 5. Update BERTForSpanNERv2 init ✅
File: `src/tabib/models/bert_span_ner.py`
- Added `use_crf: bool = True` parameter to BERTForSpanNERv2.__init__
- Passed to BIOULTagger

### 6. Remove negative sampling ✅
File: `src/tabib/models/bert_span_ner.py`
- Changed default `negative_ratio` from 3 to -1
- When -1, uses ALL spans up to max_negatives=500 (no sampling)
- Maintains backwards compatibility (positive ratio still samples)

### 7. Separate learning rates ✅
File: `src/tabib/models/bert_span_ner.py`
- Added `bert_learning_rate: float = 5e-5` parameter to get_trainer()
- Changed default `learning_rate` to 1e-3 (for BiLSTM/CRF/Biaffine)
- Creates AdamW optimizer with two param groups:
  - BERT encoder params: 5e-5
  - Other params: 1e-3

### 8. Add gradient clipping ✅
File: `src/tabib/models/bert_span_ner.py`
- Added `gradient_clip: float = 10.0` parameter to get_trainer()
- Uses TrainingArguments `max_grad_norm=gradient_clip`

### 9. Add prediction filtering ✅
File: `src/tabib/models/bert_span_ner.py`
- Added `_filter_overlapping_predictions()` static method
- Supports modes: "no_overlapping_same_label" (default), "no_overlapping"
- Keeps higher confidence entities, uses span length as tiebreaker
- Called in predict() method via `filter_predictions` kwarg

### 10. Update config ✅
File: `configs/test_emea_span_v2.yaml`
```yaml
backend_args:
  max_span_length: 40
  negative_ratio: -1  # Use all spans
  use_crf: true
training:
  learning_rate: 1e-3
  bert_learning_rate: 5e-5
  max_steps: 4000
  warmup_ratio: 0.1
  gradient_clip: 10.0
predict_args:
  filter_predictions: no_overlapping_same_label
```

## Key Files

### Modified
- `src/tabib/data/brat.py` - fragment parsing
- `src/tabib/data/emea.py` - filter_nested=False
- `src/tabib/data/cas.py` - filter_nested=False
- `src/tabib/models/bert_span_ner.py` - Full nlstruct port:
  - BIOULTagger with CRF
  - BERTForSpanNERv2 with use_crf parameter
  - get_trainer() with separate LRs and gradient clipping
  - predict() with overlapping prediction filtering
  - _prepare_spans() with all-spans mode

### Created
- `src/tabib/models/crf.py` - LinearChainCRF, BIOULDecoder

### Reference (nlstruct)
- `nlstruct_ref/nlstruct/datasets/brat.py`
- `nlstruct_ref/nlstruct/models/crf.py`
- `nlstruct_ref/nlstruct/models/bitag.py`
- `nlstruct_ref/nlstruct/models/ner.py`

## Test Command
```bash
sbatch launch-test-span-v2.sbatch
# Uses configs/test_emea_span_v2.yaml
```

Or directly:
```bash
$TABIB_VENV/bin/python -m tabib.cli train configs/test_emea_span_v2.yaml
```

## Target
- exact_f1 > 60% on EMEA (currently 43%)
- No training instability (was F1 dropping to 0)

## DO NOT TOUCH
- `*_token` tasks (seqeval/IOB2) - they work correctly
- `quaero_emea_token`, etc.
