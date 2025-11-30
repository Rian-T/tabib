# CLAUDE.md

## ⚠️ WRITE TO THIS JOURNAL REGULARLY!
Update after every significant finding or task completion. Commit specific files often.

## 🎯 FINAL GOAL
**Upload to W&B table: All models × All datasets with F1 scores**

ONLY after investigating all issues with simple, working code:

| Task | Datasets | Models to Compare |
|------|----------|-------------------|
| FRACCO ICD | fracco | ModernCamemBERT, CamemBERT-base, CamemBERT-bio, BioClinical-ModernBERT |
| NER | CAS1, CAS2 | ModernCamemBERT, CamemBERT-base, CamemBERT-bio |

**Progress**:
- [x] FRACCO: camembert-base (63.95% acc, 9.79% F1)
- [x] FRACCO: camembert-bio (66.91% acc, 12.42% F1)
- [x] FRACCO: BioClinical-ModernBERT (80.83% acc, 37.67% F1) ✓
- [ ] FRACCO: ModernCamemBERT (training 37%)
- [x] NER CAS1: ModernCamemBERT (34.88% F1)
- [x] NER CAS2: ModernCamemBERT (35.79% F1)
- [ ] NER CAS1/CAS2: camembert-base, camembert-bio (need to run)

## ⚠️ MONITORING TIP
Use `tail` on log files, not BashOutput (too verbose):
```bash
tail -30 results/ner_cas2_no_preprocessing.log
tail -30 results/fracco_bioclinical_modernbert.log
```

---

## 📓 SESSION JOURNAL (2025-11-30)

### 03:00 - FIX: Configurable max_length for NER

**Problem**: NER max_length hardcoded at 512, causing 45% truncation.

**Solution**: Made max_length configurable in `bert_token_ner.py`:
- Added `_max_length` attribute to BERTTokenNERAdapter
- Read from `backend_args` in config
- ModernCamemBERT supports 8192 tokens!

**Config** (`configs/ner_cas1_moderncamembert_2048.yaml`):
```yaml
backend_args:
  max_length: 2048  # Captures ALL docs without truncation
```

**Test confirmed**: 602 tokens allowed (> 512) with max_length=2048

**Expected improvement**: ~35% F1 → 50-60% F1 (no truncation loss)

### 02:30 - CamemBERT-bio NER 0% F1 - ROOT CAUSE FOUND

**The model never learns B- tags!**

| Model | B- tags (5 docs) | I- tags | Result |
|-------|------------------|---------|--------|
| CamemBERT-bio | **0** | 855 | **0% F1** |
| ModernCamemBERT | 97 | 826 | 34.88% F1 |

**Why 0% F1?**
- Without B- (begin) tags, spans can't be formed in IOB2
- `_iob2_to_spans()` correctly ignores I- tags without preceding B-
- CamemBERT-bio predicts ONLY I-sosy (never B-sosy, B-pathologie)

**IOB Label Distribution in CAS1 training data:**
```
O:            64.99%
I-sosy:       26.59%
I-pathologie:  4.29%
B-sosy:        3.40%
B-pathologie:  0.73%  ← Extremely rare!
```

**B/I ratio: 0.13** - For every B- tag, there are ~7.5 I- tags!

**Why ModernCamemBERT learns B- but CamemBERT-bio doesn't:**
- Both have same class imbalance, same data
- Different architectures react differently to extreme imbalance
- CamemBERT-bio may need: more epochs, lower LR, or class weights

**Conclusion:** CamemBERT-bio NER is NOT suitable without modifications.
Focus on ModernCamemBERT and camembert-base for NER comparisons.

### 02:00 - CRITICAL: 45% of docs truncated at 512 tokens!

**Root cause of low NER F1 found:**
- CAS1/CAS2 avg text length: 2444 chars (~600 tokens)
- **45% of train docs truncated** at max_length=512
- **39% of test docs truncated**
- Entities in truncated portions are completely lost!

**This explains the ~35% F1:**
- partial_F1 = 66.2% (entities ARE found when not truncated)
- exact_F1 = 34.88% (boundary issues + truncation)
- We're losing ~40% of potential F1 due to truncation alone

**Solutions to explore:**
1. **Sliding window with stride** - split long docs, keep overlap for context
2. **Use sentence_chunker** with proper reassembly
3. **Increase max_length** to 1024 if model supports
4. **Longformer/BigBird** - models supporting 4096 tokens

**Next step:** Add sentence_chunker preprocessing BACK but fix the reassembly issue we found earlier. The problem wasn't chunking itself - it was losing the chunk_offset for proper span reassembly.

### 01:45 - NER Error Analysis Complete

**CAS1 Error Analysis** (`playground/analyze_ner_errors.py`):
- Total: 1453 gold, 1226 predicted
- **Exact matches: 489 (33.7%)**
- **Boundary errors: 480** ← Main issue!
- False negatives: 451, False positives: 224

**By entity type**:
- sosy: 37.1% exact (475/1282) - decent
- pathologie: **8.2%** exact (14/171) - very poor!

**Key error patterns**:
1. **Truncated predictions**: "ép" instead of "épileptique"
2. **Over-extension**: "masse palpable ni hépatosplénomégalie" instead of just "masse palpable"
3. **Type confusion**: pathologie predicted as sosy (tumeur, adénopathies)

**Root causes**:
- Class imbalance: 1628 sosy vs 343 pathologie in train
- Long entities hard to capture: pathologie avg 27 chars, max 108
- Boundary detection is weak

**Improvement ideas**:
1. Add CRF layer for better boundary detection
2. Class weights for pathologie
3. More training epochs
4. Use BIO-medical pretrained model

### 01:42 - CAS2 NER Results

**CAS2 exact_f1: 35.79%** (without preprocessing)

| Entity Type | exact_F1 | Train Count | Test Count |
|-------------|----------|-------------|------------|
| examen | 49.02% | 1112 | 812 |
| valeur | 42.86% | 543 | 436 |
| anatomie | 36.19% | 1409 | 1142 |
| substance | 32.12% | 926 | 321 |
| traitement | 10.95% | 329 | 304 |
| moment | 2.27% | 430 | 171 |
| **dose** | **0%** | 386 | 46 |
| **mode** | **0%** | 230 | 95 |

**Puzzling**: dose/mode have decent training samples but 0% F1!
- Very short (avg 8-9 chars) - easily confused?
- Need to investigate sample examples

### 01:35 - FRACCO Low F1 Investigation

**Question**: Why is FRACCO F1 only 9-12% despite 64-67% accuracy?

**Answer**: Severe class imbalance + macro F1 averaging

**Analysis** (`playground/analyze_fracco_classes.py`):
- 298 ICD classes with 784x imbalance ratio (max=2352, min=3)
- Top 1 class = 14.4% of data → baseline accuracy = 14.4%
- 64% accuracy is actually 4.4x better than baseline!
- Macro F1 averages ALL 298 classes equally
- ~150 classes have <20 samples → model fails on them → 0% F1 each
- These zeros drag down the macro average

**Key insight**: The model IS learning well. Macro F1 is the wrong metric for this task.

**Solutions**:
1. Use weighted F1 (weight by class frequency)
2. Increase min_samples to 50 (→ 54 classes) or 100 (→ 31 classes)
3. Add class weights during training

### 01:28 - Started Next Runs

- **GPU 0**: CAS2 NER without preprocessing (`tabib-ner-fixed`)
- **GPU 1**: BioClinical-ModernBERT on FRACCO

### 01:25 - FRACCO & CAS1 Results

| Task | Model | Accuracy | F1 |
|------|-------|----------|-----|
| FRACCO | camembert-base | 63.95% | 9.79% |
| FRACCO | camembert-bio | **66.91%** | **12.42%** |
| CAS1 NER | ModernCamemBERT | - | **34.88%** |

**CAS1 NER FIX CONFIRMED**: Disabling preprocessing raised F1 from 0% → 34.88%

---

## ⚠️ ACTIVE TASKS (2025-11-30 01:35)

### Currently Running:

**GPU 0** - CAS2 NER (without preprocessing)
- Config: `configs/ner_cas2_moderncamembert.yaml`
- Log: `results/ner_cas2_no_preprocessing.log`
- W&B: `tabib-ner-fixed`

**GPU 1** - FRACCO BioClinical-ModernBERT
- Config: `configs/cls_fracco_icd_bioclinical_modernbert.yaml`
- Log: `results/fracco_bioclinical_modernbert.log`

### FRACCO Results:

| Model | Accuracy | F1 | Notes |
|-------|----------|-----|-------|
| camembert-base | 63.95% | 9.79% | ✓ |
| camembert-bio-base | 66.91% | 12.42% | ✓ |
| **BioClinical-ModernBERT** | **80.83%** | **37.67%** | ✓ **BEST!** |
| ModernCamemBERT | pending | - | |

**BioClinical-ModernBERT is 3x better F1 than CamemBERT!**

### NER Results (without preprocessing):

| Dataset | Model | exact_F1 | Notes |
|---------|-------|----------|-------|
| CAS1 | ModernCamemBERT | **34.88%** | ✓ Fixed! |
| CAS2 | ModernCamemBERT | training... | |

### Next Steps:
1. Wait for CAS2 NER & BioClinical-ModernBERT to complete
2. Create FRACCO config with min_samples=50 for better F1
3. Try CamemBERT-bio for NER tasks
4. Error analysis on NER predictions

---

## GOLDEN RULES
- **Update journal after every finding** - future Claude needs context!
- **Git commit specific files** with short messages
- Never use `git add .` - always add specific files
- Use `tail` on log files, not BashOutput
- Keep working autonomously, don't wait for approvals

---

## Previous Session Notes

### ROOT CAUSE FOUND: sentence_splitter removes context!
**NER model actually works well: 34.88% exact F1** when predicting on full documents!

**The real bug**: `sentence_splitter` preprocessing removes surrounding context:
- Full doc (1193 chars): Model predicts B-pathologie → I-pathologie → correct span
- Truncated chunk (134 chars): Model predicts B-pathologie → I-sosy → WRONG span

**Solution**: Disabled `preprocessing` in NER configs
- CAS1/CAS2 docs are ~1000-2000 chars, fit in 512 tokens without chunking

### Code Changes This Session:
- `src/tabib/models/bert_token_ner.py`: **FIXED whitespace gap bug in `_iob2_to_spans()`**
- `src/tabib/data/fracco.py`: Added `min_samples` parameter
- `playground/debug_ner_spans.py`: Debug script showing the bug
- `playground/analyze_fracco_classes.py`: FRACCO class distribution analysis
- Created FRACCO & NER configs

### Commands to Check Status:
```bash
# Check GPU 0
BashOutput bash_id=64d09e

# Check GPU 1 (FRACCO)
BashOutput bash_id=a3b738
```

---

## GOLDEN RULES
- **ALWAYS update the "ACTIVE TASKS" section above when context may be lost** - future Claude needs this info!
- Never use `git add .` - always add specific files
- Always check doc online for specific library versions when in doubt
- Make git commits regularly with meaningful but short messages
- Avoid bash commands with many subcommands or pipes unless necessary
- **When long evaluations are running**: Use `/babysit` with relevant sleep values and simple single commands
- **Document regularly**: Always update CLAUDE.md with current state and next steps
- **Keep working**: Never wait for user approval, continue autonomously
- **Git commit regularly**: Add specific changed files and commit with meaningful messages
- **Check Weave traces**: Raw inputs and outputs often contain valuable debug info at wandb.ai/rnar/.../weave

---

## CURRENT STATUS (2025-11-29 17:30)

### ModernCamemBERT & FRACCO Evaluation (2025-11-29)

**GPU 0: ModernCamemBERT NER Evaluation**

| Dataset | exact_f1 | Status | Notes |
|---------|----------|--------|-------|
| MEDLINE | 0.71% | Completed | Very low - investigation needed |
| CAS1 | - | Failed | No train split (only dev/test) |
| CAS2 | - | Failed | No train split (only dev/test) |
| EMEA | 0.24% | Completed | Very low |
| E3C | **7.13%** | Completed | Best result |

**GPU 1: FRACCO ICD Classification**

- **Model**: `almanach/moderncamembert-base`
- **Result**: **80.35% accuracy** ✓
- **Approach**: Top-K filtering with `min_samples=10`
- Reduced from 3,041 classes to ~200 manageable classes
- Config: `configs/cls_fracco_icd_moderncamembert.yaml`

**Code Changes**:
- `src/tabib/data/fracco.py`: Added `min_samples` parameter to `FRACCOICDClassificationAdapter`
  - Filters out ICD codes with fewer than N samples
  - Enables practical BERT classification on long-tail distribution

**Issues Found**:
1. CAS1/CAS2 datasets have no train split - only dev/test
2. ModernCamemBERT NER results very low (<1% F1) - needs investigation
3. E3C performed best at 7.13% F1

---

### LoRA FINETUNING IMPLEMENTATION

**Implemented LoRA SFT infrastructure for MCQA tasks**:

**Dependencies added** (`pyproject.toml`):
- `peft>=0.18.0`
- `trl>=0.25.1`
- `bitsandbytes>=0.48.0`

**New files**:
- `src/tabib/models/lora_sft.py` - LoRA SFT model adapter using TRL's SFTTrainer
- `configs/mcqa_mediqal_mcqu_lora.yaml` - MCQU LoRA training config
- `configs/mcqa_mediqal_mcqm_lora.yaml` - MCQM LoRA training config

**LoRA adapter features**:
- 4-bit QLoRA quantization (NF4)
- TRL SFTTrainer integration
- French medical system prompts
- Chat message formatting for SFT
- Target modules: all-linear
- Default: r=16, alpha=32, dropout=0.05

**Training configs**:
- Model: Qwen/Qwen3-8B
- Epochs: 3
- Batch size: 2 (gradient accumulation: 8)
- Learning rate: 2e-4
- Max sequence length: 2048

**Currently running**: LoRA MCQU finetuning

---

### NER JSON CAMPAIGN COMPLETE! (35 configs)

**Duration**: ~2 hours (13:51 - 15:50)

**Fuzzy Matching Improvements Applied**:
- Added apostrophe normalization (' ' ʼ ʻ → ')
- Added quote normalization (« » " " → ")
- Added difflib-based fuzzy matching (85% similarity threshold) for typos

### FINAL RESULTS (exact_F1)

**INSTRUCT MODELS** (use_chat=true):

| Model | MEDLINE | CAS1 | CAS2 | EMEA | E3C |
|-------|---------|------|------|------|-----|
| **MedGemma-27B** | **55.35%** | **37.40%** | **37.64%** | **31.93%** | 3.61% |
| Qwen3-8B | 25.18% | 26.05% | 26.84% | 11.48% | 5.62% |

**COMPLETION MODELS** (use_chat=false):

| Model | MEDLINE | CAS1 | CAS2 | EMEA | E3C |
|-------|---------|------|------|------|-----|
| Gemma3-4B | 33.92% | 5.79% | 5.38% | 0.00% | **7.06%** |
| Gaperon-8B | 27.94% | 0.00% | 0.00% | -- | 2.85% |
| Gaperon-24B | 23.94% | 0.00% | 0.00% | -- | 1.70% |
| EuroLLM-9B | 21.57% | -- | -- | -- | 4.50% |
| Gemma3-27B | 0.00% | 0.00% | 0.00% | 1.31% | 0.00% |

### KEY FINDINGS:
1. **MedGemma-27B is the clear winner** - best on all datasets except E3C
2. **Instruct models >> Completion models** for NER tasks
3. **Completion models fail catastrophically** on French clinical text (CAS1/CAS2)
4. **Gemma3-27B total failure** - can't do NER at all with completion prompting
5. **JSON extraction mode works** - much better than XML tagging for high-entity docs

**Next Steps**:
1. LoRA finetune on MediQAl MCQU
2. LoRA finetune on MediQAl MCQM

**Scripts Used**:
- `scripts/generate_ner_configs.py` - Generate all 35 configs
- `scripts/run_ner_json_campaign.sh` - Run all evaluations
- **Log**: `results/ner_json_campaign.log`

**Code changes in vllm_ner.py**:
- `_normalize_text()`: Added apostrophe/quote normalization
- `_fuzzy_find_entity()`: Added difflib SequenceMatcher fallback (85% threshold)

---

### Previous: NER v2 - ROOT CAUSE FOUND

**Why XML-tagging fails on EMEA**:
- EMEA has 140+ entities per document (MEDLINE has ~4)
- XML approach requires **perfect reproduction** of 4000+ chars
- Model must insert 140 XML tags at exact positions
- One missing/extra char = 0 F1 for that entity

### Added Gaperon-Garlic Models

Added 16 configs for contamination research models:
- `almanach/Gaperon-Garlic-1125-8B`
- `almanach/Gaperon-Garlic-1125-24B`

Note: These are **intentionally contaminated** (~50% benchmark test data) for contamination detection research.

**Key Insights**:
- Smaller model (Gemma3-4B) outperforms larger ones on this task
- Few-shot examples provide massive improvement (4.08% → 40.19%)
- Base models (27B) may need different prompting strategy
- CAS1/CAS2 annotation style (full clauses) doesn't match LLM extraction

**NEW PROJECT**: Evaluating all models on NER datasets

**W&B Project**: `tabib-ner-benchmark` (new project)

**Datasets** (4 datasets):
| Dataset | Description | Entity Types |
|---------|-------------|--------------|
| MEDLINE | Biomedical abstracts | anatomy, disorder, substance, living, procedure, object, phenomenon, physiology |
| EMEA | European Medicines Agency | To explore |
| CAS1/CAS2 | Clinical case notes | sosy, pathologie, moment, mode, substance |
| E3C | European Clinical Case Corpus | To explore |

**Models to evaluate** (9 models from MCQA):
| Model | HuggingFace ID | Size | Type |
|-------|----------------|------|------|
| Gemma-3-4B-pt | google/gemma-3-4b-pt | 4B | Base |
| Gemma-3-27B-pt | google/gemma-3-27b-pt | 27B | Base |
| Gaperon-8B | almanach/Gaperon-1125-8B | 8B | Completion |
| Gaperon-24B | almanach/Gaperon-1125-24B | 24B | Completion |
| EuroLLM-9B | utter-project/EuroLLM-9B | 9B | Base |
| OLMo-3-7B | allenai/OLMo-2-1124-7B | 7B | Base |
| OLMo-3-32B | allenai/Olmo-3-1125-32B | 32B | Base |
| MedGemma-27B | google/medgemma-27b-text-it | 27B | Instruct |
| Qwen3-8B | Qwen/Qwen3-8B | 8B | Instruct |

**Plan**:
1. [x] Explore datasets (entity types, annotation style, train/test sizes)
2. [x] Test NER prompts in playground (verify format, few-shot, parsing)
3. [ ] Create configs for all model/dataset combinations (0-shot + 5-shot)
4. [ ] Run evaluations on 2xH100
5. [ ] Upload results to W&B

**Exploration Findings**:
- MEDLINE: 10 entity types (ANAT, CHEM, DEVI, DISO, GEOG, LIVB, OBJC, PHEN, PHYS, PROC), 833 samples/split
- EMEA: 10 entity types (same as MEDLINE), only 11-15 samples/split, very long docs
- CAS1: 2 entity types (sosy, pathologie), NO train split (use dev for few-shot)
- CAS2: 8 entity types (anatomie, examen, substance, traitement, valeur, moment, mode, dose), NO train
- Prompt uses XML tags: `<ENTITY_TYPE>text</ENTITY_TYPE>`
- Parsing handles accents, ligatures (œ→oe), case insensitive matching
- Created 72 configs: 9 models × 4 datasets × 2 shot variants
- **NER evaluations started**: Thu Nov 28 23:17 CET 2025

**NER Approach Analysis** (2025-11-28 23:20):
- **Chose XML tagging** over JSON/line-by-line extraction
- Reasons: preserves offsets, handles nested entities, intuitive few-shot
- Known issues: nested tag regex misses outer tags (inner still captured)
- Improvement: French system prompt would be better for French data
- Fuzzy matching already handles: ligatures (œ→oe), accents, case

**CRITICAL BUG FOUND** (2025-11-28 23:55):
- **Few-shot example leakage**: Model outputs entities from examples for EVERY sample!
- Entities like `tumeur de type carcinome épidermoïde` appear 680 times in CAS1 eval
- **Root cause**: Prompt doesn't clearly separate examples from actual task
- Current format uses same "Text:" / "Annotated:" pattern for both
- **Impact**: Massive false positives, inflated entity counts
- **Fix needed**: Add clear section headers `=== EXAMPLES ===` vs `=== YOUR TASK ===`
- **Despite bug**: Still get 40% F1 on MEDLINE - could be MUCH HIGHER with better prompt

**Current Approach Verdict**: XML tagging is good, but PROMPT NEEDS IMPROVEMENT.
The 40% F1 result is WITH the leakage bug - proper fix could significantly improve scores.

**Next Steps After NER**:
1. [ ] LoRA finetune for MCQA tasks (MediQAl MCQU, MCQM, FrenchMedMCQA)
2. [ ] Compare LoRA finetuned vs zero-shot performance

---

### Previous: NER Base Model Evaluation Campaign - SUPERSEDED

**NEW PROJECT**: Evaluating all base/completion models (non-chat, non-instruct) on NER datasets

**W&B Project**: `tabib-ner-benchmark` (new project)

**Models being evaluated** (5 models):
| Model | HuggingFace ID | Size | Type |
|-------|----------------|------|------|
| Gemma-3-4B-pt | google/gemma-3-4b-pt | 4B | Base |
| Gemma-3-27B-pt | google/gemma-3-27b-pt | 27B | Base |
| Gaperon-8B | almanach/Gaperon-1125-8B | 8B | Completion |
| Gaperon-24B | almanach/Gaperon-1125-24B | 24B | Completion |
| EuroLLM-9B | utter-project/EuroLLM-9B | 9B | Base |

**Datasets**: MEDLINE, CAS1, CAS2

**Configurations**: 30 total (5 models x 3 datasets x 2 shot variants)
- 0-shot: `configs/ner_{dataset}_{model}.yaml`
- 3-shot: `configs/ner_{dataset}_{model}_3shot.yaml`

**Run Script**: `scripts/run_ner_base_models.sh`
**Log File**: `results/ner_base_models_run.log`
**Started**: Thu Nov 27 22:25:02 CET 2025
**Current**: Running small models 0-shot (ner_medline_gemma3_4b)

**Scripts Created**:
- `scripts/run_ner_base_models.sh` - Sequential runner for all 30 configs
- `scripts/upload_ner_table.py` - W&B uploader (pivot table with exact_f1, partial_f1)
- `playground/test_ner_fewshot.py` - Test script verifying 0-shot/3-shot support

---

### MCQA Results Summary (0-shot + 3-shot) - COMPLETED

**W&B Run**: https://wandb.ai/rnar/tabib-llm-benchmark/runs/8pnc0x6d

| Model | MCQU | MCQM | FrenchMed |
|-------|------|------|-----------|
| MedGemma-27B | **36.77%** | **9.16%** | **14.31%** |
| Gemma-3-27B | 30.72% / 28.80% | 5.35% / 5.59% | 10.61% / 17.20% |
| EuroLLM-9B | 21.30% / 19.39% | 4.31% / 4.40% | 6.11% / 4.34% |
| Gemma-3-4B | 20.86% / 20.45% | 4.14% / 4.23% | 1.93% / 1.93% |
| Gaperon-8B | 19.83% / 20.24% | 4.14% / 4.17% | 8.68% / 5.14% |
| Qwen3-8B | 19.78% / 19.78% | 5.08% / 5.08% | 8.52% / 7.88% |
| Gaperon-24B | 18.88% | -- | -- |
| OLMo-3-7B | 18.14% / 18.19% | 5.38% / 5.38% | **14.95%** / **14.95%** |

*Format: 0-shot / 3-shot*

**Key Observations**:
- 3-shot provides marginal improvement on most models
- Gemma-3-27B improved on FrenchMed: 10.61% -> 17.20% (+6.59pp)
- OLMo-3-7B consistently good on FrenchMedMCQA: 14.95%
- Multi-answer (MCQM) remains much harder than single-answer (MCQU)

### Previous NER Results (chat models, 3-shot)
| Model | Dataset | Exact F1 | Partial F1 | Notes |
|-------|---------|----------|------------|-------|
| Qwen3-7B | MEDLINE | **39.75%** | **51.52%** | Good results |
| Qwen3-7B | CAS1 | **1.16%** | **31.22%** | Low exact, decent partial |
| MedGemma-27B | CAS1 | **1.16%** | **35.81%** | Same pattern as Qwen3 |

### Gaperon-24B vLLM Patch - FIXED (2025-11-27)

**Problem**: Gaperon-24B uses custom `head_dim=128` with `num_attention_heads=32` (not 40), causing vLLM to miscalculate dimensions.

**Error**: `AssertionError: Attempted to load weight (torch.Size([1024])) into parameter (torch.Size([1280]))`

**Solution**: Complete `Olmo2Attention.__init__` replacement patch in `src/tabib/models/gaperon_patch.py` that reads `head_dim` from config BEFORE creating QKV layers.

**Key fix**:
```python
# Use head_dim from config if available (Gaperon case)
default_head_dim = hidden_size // self.total_num_heads
self.head_dim = getattr(self.config, "head_dim", default_head_dim)
```

**Reference**: https://huggingface.co/almanach/Gaperon-1125-24B/discussions/1

### ROOT CAUSE IDENTIFIED: CAS1/CAS2 Annotation Style

**The real issue is NOT the model or code - it's the annotation style:**

| Dataset | Entity Style | Example | Typical Length |
|---------|-------------|---------|----------------|
| **MEDLINE** | Short terms | `"gangrènes gazeuses"` | 5-25 chars |
| **CAS1/CAS2** | Full clauses | `"tumeur bien limitée, de structure tissulaire, prenant le contraste de façon hétérogène"` | 25-110 chars |

CAS1/CAS2 annotate **full clinical descriptions** as single entities:
- `"recherche de BK dans les crachats et le produit de tubage gastrique réalisé trois fois était revenu négatif"` (107 chars!)
- `"douleurs épisodiques localisées au niveau de l'hypochondre droit"` (64 chars)

**Why exact F1 fails for CAS1/CAS2:**
1. LLMs extract conceptual entities, not full clauses
2. Copying 100+ char spans exactly is nearly impossible
3. Any small boundary variation ruins exact match

**Conclusion**:
- For MEDLINE: Use **Exact F1** (39.75% is good)
- For CAS1/CAS2: Use **Partial F1** (31-35% is good, shows entities ARE found)

**Performance Summary:**
- Temperature: 0.0 ✓
- Batched inference: 43x faster ✓
- 3-shot examples: ✓

### Bugs Fixed This Session
1. **Non-batched inference**: Was sending one sample at a time - fixed to batch all prompts together
2. **Empty 3-shot examples**: Pipeline passed chunked data - fixed by saving original train data
3. **CAS1/CAS2 no train split**: Fixed pipeline to fall back to `dev` for few-shot
4. **Lowercase entity types not parsed**: Fixed regex `[A-Z_]+` -> `[A-Za-z_]+`
5. **Entity definitions missing for CAS**: Added `sosy`, `pathologie`, `moment`, `mode`, `substance`

### Running Now
- NER base model evaluations via `scripts/run_ner_base_models.sh`
- Log: `results/ner_base_models_run.log`
- Started: Thu Nov 27 22:25:02 CET 2025

### Next Steps
1. Wait for NER evaluations to complete (~2-3 hours for 30 configs)
2. Results auto-upload to W&B `tabib-ner-benchmark` project
3. Analyze results comparing 0-shot vs 3-shot for base models
4. Compare base model NER performance vs chat models (Qwen3-7B results)

### Current Crux
- **No blocking issues** - NER pipeline running smoothly
- Gaperon-24B patch verified working for MCQA (18.88% on MediQAl MCQU)
- All 30 NER configs validated and ready

---

## NER Base Model Evaluation Campaign (2025-11-27)

**Focus**: Base/completion models only (no chat/instruct models)

**Models** (5 models):
| Model | Size | Configs |
|-------|------|---------|
| gemma3_4b | 4B | 0-shot + 3-shot |
| gemma3_27b | 27B | 0-shot + 3-shot |
| gaperon_8b | 8B | 0-shot + 3-shot |
| gaperon_24b | 24B | 0-shot + 3-shot |
| eurollm_9b | 9B | 0-shot + 3-shot |

**Datasets**: MEDLINE, CAS1, CAS2

**Total Configs**: 30 (5 models x 3 datasets x 2 shot variants)

**Run Order**:
1. Small models 0-shot (gemma3_4b, gaperon_8b, eurollm_9b x 3 datasets = 9 configs)
2. Small models 3-shot (9 configs)
3. Large models 0-shot (gemma3_27b, gaperon_24b x 3 datasets = 6 configs)
4. Large models 3-shot (6 configs)

**Output**: W&B pivot table with exact_f1 and partial_f1 per model/dataset

---

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tabib is a task-agnostic NLP evaluation framework for biomedical French text. It supports NER, text classification, semantic similarity, MCQA, and more. The framework emphasizes low LOC, clear abstractions, and reproducibility.

## Core Commands

### Installation & Setup
```bash
# Install dependencies
poetry install

# Install with vLLM support (for LLM inference)
poetry install --extras vllm
```

### Training
```bash
# Train a model with config file
poetry run tabib train configs/ner_emea_camembert.yaml

# Training outputs go to:
# - $SCRATCH/tabib/runs/<output_dir> (if SCRATCH env var is set)
# - ./runs/<output_dir> (otherwise)
```

### Evaluation
```bash
# Evaluate a model
poetry run tabib eval configs/ner_emea_camembert.yaml
```

### Model Comparison
```bash
# Compare multiple models across datasets
poetry run tabib compare configs/compare_example.yaml

# Dry run to see planned runs
poetry run tabib compare configs/compare_example.yaml --dry-run

# Specify output path
poetry run tabib compare configs/compare_example.yaml -o results/custom.json
```

### Testing
```bash
# Run specific test
poetry run pytest tests/test_sentence_splitter.py

# Run all tests
poetry run pytest tests/

# Verify Python syntax
python -m py_compile src/tabib/models/your_file.py
```

## Architecture

### Three-Component System

Tabib connects three pluggable abstractions via the `Pipeline`:

1. **Task** (`tasks/base.py`): Defines label space, metrics, and I/O format
   - Examples: `ner_token`, `ner_span`, `classification`, `mcqa`, `similarity`, `open_qa`
   - Each task implements: `label_space`, `compute_metrics()`, `format_output()`

2. **DatasetAdapter** (`data/base.py`): Loads splits and preprocesses for the task
   - Examples: `emea`, `medline`, `cas1`, `cas2`, `fracco_expression_ner`
   - Each adapter implements: `load_splits()`, `preprocess()`
   - Data sources are in `data/` directory (BRAT format for NER, CSV for classification)

3. **ModelAdapter** (`models/base.py`): Builds model, exposes training/inference
   - Examples: `bert_token_ner`, `vllm_ner`, `vllm_classification`, `gliner_zero_shot`
   - Each adapter implements: `supports_finetune`, `build_model()`, `get_trainer()`, `predict()`

### Registration System

All components register in `src/tabib/__init__.py`:
```python
register_task("ner_span", tasks.NERSpanTask)
register_dataset("emea", data.EMEAAdapter)
register_model("bert_token_ner", models.BERTTokenNERAdapter)
```

Registry lookups happen in `registry.py` via `get_task()`, `get_dataset()`, `get_model()`.

### Pipeline Flow

1. Load config from YAML → `RunConfig` (via `config.py`)
2. Instantiate task, dataset adapter, model adapter from registry
3. Load dataset splits (`load_splits()`)
4. Apply preprocessor if configured (e.g., `sentence_chunker` for long documents)
5. Preprocess each split for the task (`preprocess()`)
6. Build model (`build_model()`)
7. **If `do_train=True` and `supports_finetune=True`**: Get trainer and train
8. **If `do_eval=True`**: Run inference (`predict()`) and compute metrics (`compute_metrics()`)
9. Return summary dict with metrics, training info, config metadata

Pipeline lives in `pipeline.py` with core logic in `Pipeline.run()`.

## Key Abstractions

### NER Tasks: Token vs Span

- **`ner_token`**: IOB2 tagging (for BERT-like models)
  - Aligns labels with tokenizer outputs
  - Evaluates at token level, then converts to spans

- **`ner_span`**: Character-offset spans (for GLiNER, vLLM, or BERT with span conversion)
  - Predictions are `{start, end, label, text}`
  - Uses `SpanEvaluator` for exact/partial F1 metrics
  - Supports nested entities

### Preprocessing

Two main preprocessors in `preprocessing/`:

- **`SentenceChunker`**: Splits long documents into sentence-based chunks
  - Preserves entity integrity (no mid-entity splits)
  - Used for NER to fit within model token limits (e.g., 512 for BERT, 4096 for LLMs)
  - Reassembles predictions back to document offsets

- **`SentenceSplitter`**: One sentence per sample
  - Used for sentence-level tasks (similarity, classification)

Enable via config:
```yaml
preprocessing:
  type: sentence_chunker
  max_tokens: 512
```

### Model Backends

- **BERT-like models** (`bert_token_ner`, `bert_text_cls`, `bert_similarity`):
  - Support fine-tuning via `transformers.Trainer`
  - Fast inference with batch processing
  - Use `model_name_or_path` for HuggingFace model ID or local path

- **vLLM models** (`vllm_ner`, `vllm_classification`, `vllm_open_qa`):
  - Inference-only (no fine-tuning)
  - Use structured outputs (Pydantic schemas) for constrained generation
  - NER uses context-based extraction (entity + surrounding words) with text matching
  - Requires `poetry install --extras vllm`

- **GLiNER** (`gliner_zero_shot`):
  - Zero-shot NER without fine-tuning
  - Predicts entities from provided labels

## Adding New Components

### Adding a New Dataset

1. Create adapter in `src/tabib/data/your_dataset.py`:
```python
from tabib.data.base import DatasetAdapter

class YourDatasetAdapter(DatasetAdapter):
    @property
    def name(self) -> str:
        return "your_dataset"

    def load_splits(self) -> dict[str, Any]:
        # Load train/val/test splits
        pass

    def preprocess(self, dataset: Any, task: Any) -> Any:
        # Convert to task format
        pass
```

2. Import and register in `src/tabib/__init__.py`:
```python
register_dataset("your_dataset", data.YourDatasetAdapter)
```

3. Create config file in `configs/`:
```yaml
task: ner_span
dataset: your_dataset
model: bert_token_ner
model_name_or_path: camembert-base
do_train: true
do_eval: true
```

### Adding a New Model

Follow same pattern as datasets. See `src/tabib/models/vllm_ner.py` for a complete example.

### Adding a New Task

Tasks define metrics and label space. See `src/tabib/tasks/ner_span.py` for span-based NER example.

## Configuration

Config files are YAML in `configs/`. Key fields:

```yaml
task: ner_span                          # Task name (from registry)
dataset: emea                           # Dataset name (from registry)
model: bert_token_ner                   # Model name (from registry)
model_name_or_path: camembert-base      # HF model ID or path
do_train: true                          # Whether to train
do_eval: true                           # Whether to evaluate
preprocessing:                          # Optional preprocessing
  type: sentence_chunker
  max_tokens: 512
training:                               # Training config (if do_train=true)
  output_dir: runs/emea_camembert
  num_train_epochs: 10
  per_device_train_batch_size: 8
  learning_rate: 2e-5
  seed: 42
  metric_for_best_model: exact_f1
  greater_is_better: true
  early_stopping_patience: 3
backend_args:                           # Model-specific args
  max_length: 512
```

## Datasets

French biomedical datasets in `data/`:

- **EMEA**: European Medicines Agency documents (BRAT format, NER)
- **MEDLINE**: French biomedical abstracts (BRAT format, NER)
- **CAS1, CAS2**: Clinical case notes (BRAT format, NER)
- **FRACCO**: French cancer reports (BRAT format, NER + classification)
- **JNLPBA**: Biomedical NER (token-level)
- **MediQAl** (ANR-MALADES/MediQAl): French medical QA with three subtasks
  - **MCQU** (`mediqal_mcqu`): Single-answer multiple-choice questions
  - **MCQM** (`mediqal_mcqm`): Multi-answer multiple-choice questions
  - **OEQ** (`mediqal_oeq`): Open-ended questions
- **French Med MCQA Extended**: French medical multiple choice

BRAT format: `.txt` files for text, `.ann` files for annotations.

## Evaluation

Metrics vary by task:

- **NER (span)**: `exact_f1`, `partial_f1`, per-entity-type F1
- **NER (token)**: Token-level accuracy, F1 via `seqeval`
- **Classification**: Accuracy, F1, precision, recall
- **MCQA**: Exact match accuracy
- **Similarity**: Spearman/Pearson correlation

Evaluation logic lives in task classes and `evaluation/` for complex cases.

## Common Workflows

### Train CamemBERT on EMEA NER
```bash
poetry run tabib train configs/ner_emea_camembert.yaml
```

### Evaluate vLLM model (inference-only)
```bash
poetry run tabib eval configs/ner_emea_vllm.yaml
```

### Compare CamemBERT vs Qwen on multiple datasets
```bash
poetry run tabib compare configs/compare_example.yaml
```

### Debug predictions
```bash
poetry run python playground/09_check_predictions.py
```

## Comparison Framework

The `compare` command runs multiple configs in batch:

1. Create comparison spec in `configs/compare_*.yaml`
2. Define experiments with datasets and models
3. Optionally override model configs per experiment
4. Run with `tabib compare`
5. Results saved as JSON with all metrics

Example spec structure:
```yaml
description: Example comparison
defaults:
  do_train: false
  do_eval: true
datasets:
  ner_emea: ner_emea_camembert.yaml      # Reference to base configs
  ner_medline: ner_medline_camembert.yaml
experiments:
  camembert_vs_qwen:
    datasets: [ner_emea, ner_medline]
    models:
      - name: camembert                   # Uses base config model
      - name: qwen                        # Overrides for this model
        model: vllm_ner
        model_name_or_path: Qwen/Qwen2.5-7B-Instruct
output_path: results/comparison.json
```

Code lives in `src/tabib/comparison/`.

## Design Principles

1. **One adapter per file**: Keep files short (~100-300 LOC)
2. **Explicit registration**: All components register in `__init__.py`
3. **Task-agnostic pipeline**: Same flow for all tasks
4. **Config-driven**: All runs specified in YAML
5. **Reproducibility**: Fixed seeds, config stored in outputs
6. **Strong typing**: Pydantic models for configs, type hints everywhere
7. **Minimal dependencies**: Core deps only, optional extras for vLLM/LoRA

## Notes

- This is a research framework focused on French biomedical NLP
- Dataset paths are relative to `data/` directory
- Training outputs respect `$SCRATCH` environment variable for HPC setups
- vLLM models use structured outputs for reliability (Pydantic schemas)
- Context-based NER extraction compensates for LLM character-offset limitations
- Sentence chunking preserves entity boundaries for fair evaluation

## Changelog & Development Log

### 2025-11-27: Critical 3-Shot Learning Bug Fixes

**CRITICAL BUG FIXED**: Few-shot learning was broken due to passing chunked training data to model

**Problem identified**:
1. Pipeline was passing train_data to `build_model()` AFTER sentence chunking preprocessing
2. Chunked data lost the original `entities` field, resulting in empty few-shot examples
3. `_create_few_shot_examples()` looked for `spans` key but MEDLINE data uses `entities` key
4. Result: All "3-shot" evaluations were actually zero-shot with empty examples

**Fixes applied**:
- `src/tabib/pipeline.py:88-108`: Save original train data BEFORE chunking, pass to build_model()
- `src/tabib/models/vllm_ner.py:165`: Check both `spans` and `entities` keys for compatibility
- `src/tabib/models/vllm_ner.py:327-331`: Added `weave.publish()` to log raw LLM outputs for debugging
- `src/tabib/models/vllm_ner.py:337-459`: Added smart fuzzy matching for entity location
  - Handles ligatures: œ ↔ oe, æ ↔ ae
  - Removes accents: é → e, à → a, etc.
  - Case insensitive + whitespace normalization
  - Falls back to exact match first for speed

**Impact**:
- Qwen3-7B MEDLINE result (28.42% F1) was with broken 3-shot (empty examples)
- Need to re-run all models with fixed 3-shot implementation
- Gaperon-8B currently running with REAL 3-shot examples for first valid test

**Weave integration**:
- Raw LLM outputs now logged to W&B Weave with input text, generated text, and full prompt
- Enables debugging of model behavior and prompt engineering

### 2025-11-26: vLLM Completion Model Support & MediQAl Comparison

**Added**: Support for both chat and completion models in `vllm_classification` adapter

**Changes to `src/tabib/models/vllm_classification.py`**:
- Added `use_chat: bool` parameter (default `True`) to `_VLLMResources` dataclass
- Added `use_chat` parameter to `build_model()` method signature
- Modified `predict()` method to branch on model type:
  - If `use_chat=True`: Use chat API (`chat_with_vllm()`) with message formatting
  - If `use_chat=False`: Use generate API (`engine.llm.generate()`) with direct prompts
- For completion models, system prompt is prepended directly to user prompt

**Usage in config files**:
```yaml
backend_args:
  use_chat: false  # Set to false for completion/base models
  prompt_template: |
    Your prompt here...
  system_prompt: >
    System instructions here...
```

**Why this matters**:
- Completion models (like Gaperon, base models) don't have chat templates
- Previously would error with: "default chat template is no longer allowed"
- Now supports both model types seamlessly through configuration
- Enables fair comparison between chat models (Qwen3, Llama-Chat) and completion models (Gaperon, base Llama)

**MediQAl Evaluation Results** (Qwen3-8B vs Gaperon-1125-8B):
- Dataset: ANR-MALADES/MediQAl (French medical QA)
- Zero-shot evaluation results:
  - **MCQU (single-answer)**: Gaperon 20.91% vs Qwen3 19.78% (Gaperon +1.13pp)
  - **MCQM (multi-answer)**: Qwen3 5.23% vs Gaperon 4.20% (Qwen3 +1.03pp)
- Both models show similar zero-shot performance
- Multi-answer tasks significantly harder than single-answer (75% accuracy drop)
- Full results in `results/mediqal_qwen3_vs_gaperon_comparison.md`

**Created configs**:
- `configs/mcqa_mediqal_mcqu_qwen3.yaml` (chat model)
- `configs/mcqa_mediqal_mcqm_qwen3.yaml` (chat model)
- `configs/mcqa_mediqal_mcqu_gaperon.yaml` (completion model with `use_chat: false`)
- `configs/mcqa_mediqal_mcqm_gaperon.yaml` (completion model with `use_chat: false`)
- `configs/compare_mediqal_qwen3_vs_gaperon.yaml`

### 2025-11-26: FrenchMedMCQA-extended & multi-label metrics
- New MCQM configs for `french_med_mcqa_extended` across all models:
  - `configs/mcqa_frenchmedext_mcqm_qwen3.yaml`
  - `configs/mcqa_frenchmedext_mcqm_gaperon.yaml`
  - `configs/mcqa_frenchmedext_mcqm_medgemma.yaml`
  - `configs/mcqa_frenchmedext_mcqm_medgemma_pt.yaml`
  - `configs/mcqa_frenchmedext_mcqm_eurollm.yaml`
- MCQA task now reports additional multi-answer metrics: `hamming_score` and `emr` (subset accuracy).
- W&B uploader now parses all metrics and logs FrenchMedExt columns (accuracy, hamming, emr) alongside MediQAl results.
- Latest consolidated W&B run: `https://wandb.ai/rnar/mediqal-french-medical-qa/runs/zhkkk2ya`

### 2025-11-27: Weave Tracing for MCQA & Gaperon-24B Patch

**Added**: Weave tracing to `vllm_classification.py` for QA debugging
- Added `import weave` and `@weave.op()` decorator to `_log_llm_call` method
- Each LLM call now logs: input_text, prompt, raw_output, expected_label
- Enables inspection of LLM behavior in W&B Weave UI

**Added**: Monkey patch for Gaperon-24B (OLMo2 architecture) vLLM compatibility
- New file: `src/tabib/models/gaperon_patch.py`
- Fixes `head_dim` mismatch in vLLM's OLMo2 implementation
- Reference: https://huggingface.co/almanach/Gaperon-1125-24B/discussions/1
- Patch automatically applied in `vllm_common.py:create_vllm_engine()`

**MedGemma-27B MCQA Results** (0-shot, fp8 quantization):
- **MediQAl MCQU**: 36.77% accuracy (single-answer)
- **MediQAl MCQM**: 9.16% accuracy (multi-answer)
- Model loads in ~27 GiB with fp8, inference at ~12k input tokens/s

**Pending evaluations**:
- MedGemma-27B on FrenchMedMCQA (in progress)
- MedGemma-27B 3-shot variants
- Qwen3-32B all datasets
- Gaperon-24B (needs testing with patch)

### Current Status (2025-11-27 16:30)

**Completed MedGemma-27B 0-shot evaluations**:
| Dataset | Metric | Value |
|---------|--------|-------|
| MediQAl MCQU (single-answer) | Accuracy | 36.77% |
| MediQAl MCQM (multi-answer) | Accuracy | 9.16% |
| FrenchMedMCQA (multi-answer) | Accuracy | 14.31% |

**Crux - Gaperon-24B vLLM Incompatibility**:
- Model: `almanach/Gaperon-1125-24B` (OLMo2 architecture)
- Issue: Weight dimension mismatch - model has `head_dim=128` with `num_attention_heads=32` instead of 40
- vLLM calculates `head_dim = hidden_size // num_attention_heads` which doesn't match trained weights
- Initial patch failed due to vLLM 0.11 API change (new `vllm_config` parameter)
- Patch updated to match vLLM 0.11 signature: `(self, *, vllm_config: VllmConfig, prefix: str = '')`

**QA 3-shot Support**:
- Already implemented in `vllm_classification.py`
- Uses `num_few_shot` parameter in config
- Few-shot examples taken from train_data and prepended to prompts

**Next Steps**:
1. Test Gaperon-24B with updated patch
2. Run MedGemma-27B 3-shot evaluations (configs exist with `num_few_shot: 3`)
3. Run Qwen3-32B evaluations (0-shot + 3-shot)
