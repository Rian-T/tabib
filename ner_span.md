# Journal: EMEA Span-Based NER avec Nested Entities

## Objectif
Créer une nouvelle tâche "emea" avec modèle BERT span-based pour prédire les entités nested.

## Progression

### [DONE] 1. bert_span_ner.py
- `src/tabib/models/bert_span_ner.py` créé
- Classes: `BERTForSpanNER`, `SpanNERDataCollator`, `BERTSpanNERAdapter`
- Approche: span enumeration avec classification

### [DONE] 2. Registration
- `models/__init__.py`: import + __all__
- `__init__.py`: `register_model("bert_span_ner", ...)`

### [DONE] 3. Dataset EMEA avec nested
- `register_dataset("emea", lambda: data.EMEAAdapter(filter_nested=False))`
- 706 samples train, 315 avec nested entities
- 10 entity types: ANAT, CHEM, DEVI, DISO, GEOG, LIVB, OBJC, PHEN, PHYS, PROC

### [DONE] 4. Config YAML de test
- `configs/test_emea_span.yaml`

### [DONE] 5. Test
- Tests unitaires OK (playground/test_span_ner.py)
- Test tokenization/collation/forward OK (playground/test_tokenize.py)
- Test GPU OK (job 1023912)

**Résultats avec 3 epochs (avant fix alignement):**
- exact_f1: 0.059
- partial_f1: 0.183
- CHEM_exact_f1: 0.124

**Résultats avec 3 epochs (après fix alignement - job 1026552):**
- exact_f1: 0.074 (+25% relatif)
- partial_f1: 0.163
- CHEM_exact_f1: 0.127

### [DONE] 6. Fix Alignement Training/Inference
Problème: Training utilisait `valid_end = i` (position du token (0,0) après le début),
mais inference utilisait `valid_end = valid_len - 1`.

Fix dans `_prepare_spans`:
- Cherche le dernier token avec offset réel (pas (0,0))
- `valid_end = last_real_token + 1` pour inclure le dernier token réel
- Maintenant aligné avec inference qui fait `valid_end = valid_len - 1`

### [DONE] 7. Fix Class Imbalance
Problème: Model prédisait "O" pour tout (79% accuracy mais 0% sur positifs).

Fix: Class weights dans CrossEntropyLoss
- O class: weight=0.3
- Entity classes: weight=1.0

### [DONE] 8. Biaffine Architecture
Remplacé concat+MLP par BiaffineSpanClassifier:
```
start_repr = MLP_start(h_i) → 256d
end_repr = MLP_end(h_j) → 256d
score[label] = start^T @ W[label] @ end + U@start + V@end
```

**Résultats EMEA (10 epochs, biaffine):**

| Model | O weight | exact_f1 | precision | recall |
|-------|----------|----------|-----------|--------|
| camembert-base | 0.3 | 43.2% | 29.5% | 81.1% |
| **camembert-bio** | **0.5** | **44.4%** | **30.0%** | **85.6%** |

Best per-entity (camembert-bio):
- CHEM: 58.0%
- ANAT: 47.0%
- LIVB: 44.9%
- DISO: 41.9%
- DEVI: 37.8%

## Notes
- quaero_emea_token: NE PAS TOUCHER (seqeval, marche bien)
- Nouveau: emea avec bert_span_ner pour nested

## Usage
```bash
# Train
poetry run tabib train configs/test_emea_span.yaml

# Ou via sbatch
sbatch launch-test-span-ner.sbatch
```

## Fichiers créés
- `src/tabib/models/bert_span_ner.py` - Modèle span-based NER
- `configs/test_emea_span.yaml` - Config test
- `launch-test-span-ner.sbatch` - Script sbatch

---

## [WIP] Architecture Analysis: NLstruct vs Current

### NLstruct Architecture (achieves >60% F1)

Cloned from: https://gitlab.lisn.upsaclay.fr/nlp/deep-learning/nlstruct

**Key Components:**

1. **Encoder Stack:**
   - BERT encoder (camembert-bio-base)
   - Char CNN (in_channels=8, out_channels=50, kernels 3,4,5)
   - FastText word embeddings
   - Concat all + dropout 0.5

2. **Contextualizer (BiLSTM):**
   - 3-layer BiLSTM with hidden_size=400
   - Sigmoid gate with residual connection
   - Dropout 0.4

3. **Span Scorer (BiTag) - THE KEY DIFFERENCE:**
   - **BIOUL-CRF Tagger**: Predicts B/I/O/L/U tags per label with CRF
   - **Biaffine Scorer**: Standard begin/end scoring
   - **Combined**: `spans_logits = tagger_logits + biaffine_logits`
   - CRF gives marginal probabilities for boundary tokens
   - Viterbi decoding generates candidate spans
   - Biaffine refines scores

4. **Loss:**
   - Tag loss: CRF negative log-likelihood
   - Biaffine loss: BCE with logits
   - Both with weight 1.0

5. **Training:**
   - 20 epochs, batch_size=16
   - main_lr=1e-3 (LSTM, heads)
   - bert_lr=5e-5 (BERT)
   - Linear warmup + decay

### Our Current Architecture (max ~50% F1)

1. **Encoder**: BERT only
2. **Classifier**: BiaffineSpanClassifier
   - MLP projections for start/end
   - Biaffine scoring: start^T @ W @ end
3. **Loss**: CrossEntropy with O class weight

### Key Missing Components:

1. **No BIOUL-CRF**: We directly classify spans without boundary tagging
2. **No BiLSTM**: Direct BERT -> classifier
3. **No combined scoring**: Only biaffine, no tagger
4. **No char/word embeddings**: Only BERT subwords

### Implementation Plan:

1. [x] Add BIOUL tagger module
2. [x] Add BiLSTM contextualizer
3. [x] Combine tagger + biaffine scores
4. [ ] Optionally add CharCNN

### Reference Code:
- `nlstruct_ref/nlstruct/models/bitag.py` - BiTag span scorer
- `nlstruct_ref/nlstruct/models/crf.py` - BIOULDecoder CRF
- `nlstruct_ref/nlstruct/recipes/train_ner.py` - Training config

---

## [2024-12-28] Implementation: BERTSpanNERv2

Implemented nlstruct-inspired architecture in `src/tabib/models/bert_span_ner.py`:

### New Components:

1. **BiLSTMContextualizer** (lines 24-106)
   - 3-layer BiLSTM with hidden_size=400
   - Residual gating (sigmoid gate + layer norm)
   - Projects BERT output for boundary-aware representations

2. **BIOULTagger** (lines 109-240)
   - Predicts B/I/O/U/L tags per position per label
   - Computes soft boundary probabilities:
     - is_begin = P(B) + P(U) at each position
     - is_end = P(L) + P(U) at each position
     - is_inside = 1 - P(O)
   - Span score = log(begin) + log(end) + cumsum(log(inside))

3. **NLStructBiaffineScorer** (lines 315-374)
   - Per-label begin/end projections (more efficient)
   - Simple dot product: einsum('nlad,nlbd->nlab')
   - Scaled by sqrt(hidden_size) for stability

4. **BERTForSpanNERv2** (lines 396-585)
   - Combines: BERT → BiLSTM → (Tagger + Biaffine)
   - Final score: tagger_weight * tagger_scores + biaffine_weight * biaffine_scores
   - BCE loss with sigmoid (multi-label style)

5. **BERTSpanNERv2Adapter** (lines 1179-1313)
   - Registered as `bert_span_ner_v2`
   - Configurable via YAML (lstm_hidden_size, lstm_num_layers, etc.)
   - Inference uses sigmoid threshold (default 0.5)

### Test Config:
```yaml
# configs/test_emea_span_v2.yaml
model: bert_span_ner_v2
backend_args:
  lstm_hidden_size: 400
  lstm_num_layers: 3
  biaffine_hidden_size: 64
  use_tagger: true
  use_biaffine: true
training:
  learning_rate: 1e-3  # Higher LR for LSTM
  per_device_train_batch_size: 16
```

### Test Run:
```bash
sbatch launch-test-span-v2.sbatch  # Job 1031290
```

### Expected Improvement:
- v1 (biaffine only): ~50% F1 with O=0.6
- v2 (BiLSTM + tagger + biaffine): target >60% F1
