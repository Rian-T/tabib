"""BERT-based span classification model for NER with nested entity support.

This module implements the nlstruct-inspired architecture for nested NER:
- BERT encoder for contextualized embeddings
- BiLSTM contextualizer for boundary-aware representations
- BIOUL tagger with CRF for soft boundary probabilities
- Biaffine scorer for span classification

Achieves ~65% exact F1 on EMEA/MEDLINE nested NER benchmarks.
"""

import math
import random
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from tabib.models.base import ModelAdapter


class BiLSTMContextualizer(nn.Module):
    """BiLSTM contextualizer with residual gating (following nlstruct).

    Adds sequential context on top of BERT embeddings, which improves
    boundary detection for span-based NER.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 400,
        num_layers: int = 3,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Project input to hidden size if needed
        if input_size != hidden_size:
            self.input_proj = nn.Linear(input_size, hidden_size)
        else:
            self.input_proj = None

        # BiLSTM layers with residual connections
        self.lstms = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_layers):
            self.lstms.append(
                nn.LSTM(
                    hidden_size,
                    hidden_size // 2,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            # Sigmoid gate for residual connection
            self.gates.append(nn.Linear(hidden_size, 1))
            self.layer_norms.append(nn.LayerNorm(hidden_size))

        self.dropout = nn.Dropout(dropout)
        self.output_size = hidden_size

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
            mask: (batch, seq_len) attention mask

        Returns:
            (batch, seq_len, hidden_size)
        """
        if self.input_proj is not None:
            x = F.gelu(self.input_proj(x))

        lengths = mask.sum(dim=1).cpu()

        for i, (lstm, gate, ln) in enumerate(zip(self.lstms, self.gates, self.layer_norms)):
            # Pack for LSTM
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

            # Pad to original length if needed
            if lstm_out.size(1) < x.size(1):
                padding = torch.zeros(
                    x.size(0), x.size(1) - lstm_out.size(1), lstm_out.size(2),
                    device=x.device, dtype=x.dtype
                )
                lstm_out = torch.cat([lstm_out, padding], dim=1)

            lstm_out = self.dropout(lstm_out)

            # Residual with sigmoid gate
            g = torch.sigmoid(gate(lstm_out))
            x = ln(x * (1 - g) + lstm_out * g)

        return x


class BIOULTagger(nn.Module):
    """BIOUL tagger for boundary detection using CRF marginals.

    Uses CRF forward-backward algorithm for proper marginal probabilities,
    which are then converted to span boundary scores.

    B = Begin, I = Inside, O = Outside, L = Last, U = Unit (single token)
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1, use_crf: bool = True):
        super().__init__()
        self.num_labels = num_labels
        self.use_crf = use_crf

        # BIOUL: 1 O tag + 4 tags (I, B, L, U) per label
        self.n_tags = 1 + num_labels * 4

        # Project to tag logits
        self.tag_proj = nn.Linear(hidden_size, self.n_tags)
        self.dropout = nn.Dropout(dropout)

        # CRF for proper marginal computation
        if use_crf:
            from tabib.models.crf import BIOULDecoder
            self.crf = BIOULDecoder(
                num_labels=num_labels,
                with_start_end_transitions=True,
                allow_overlap=True,  # For nested entities
                allow_juxtaposition=True,
                learnable_transitions=True,
            )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_size)
            mask: (batch, seq_len) boolean mask

        Returns:
            Dictionary with:
                - tag_logits: (batch, seq_len, n_tags)
                - boundary_scores: (batch, seq_len, seq_len, num_labels)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Get tag logits: (batch, seq_len, n_tags)
        tag_logits = self.tag_proj(self.dropout(x))

        if self.use_crf:
            # Use CRF marginal for proper probabilities
            # mask must be boolean
            bool_mask = mask.bool() if mask.dtype != torch.bool else mask
            # CRF marginal returns LOG probabilities!
            log_marginals = self.crf.marginal(tag_logits, bool_mask)  # (batch, seq_len, n_tags)
        else:
            # Fallback to simple log softmax
            log_marginals = F.log_softmax(tag_logits, dim=-1)

        # Convert tag LOG probabilities to span boundary scores
        # Following nlstruct bitag.py approach exactly
        # Tag layout: O, then for each label: I, B, L, U (4 per label)
        # Index: O=0, then for label i: I=1+4i, B=2+4i, L=3+4i, U=4+4i
        boundary_scores = torch.zeros(batch_size, seq_len, seq_len, self.num_labels, device=device)

        eps = 1e-8

        for label_idx in range(self.num_labels):
            stride = 4 * label_idx
            I_idx = 1 + stride
            B_idx = 2 + stride
            L_idx = 3 + stride
            U_idx = 4 + stride

            # Log probability of being Begin or Unit at each position
            # logsumexp([B, U]) = log(exp(log_B) + exp(log_U)) = log(P(B) + P(U))
            log_is_begin = torch.stack([log_marginals[..., B_idx], log_marginals[..., U_idx]], dim=-1).logsumexp(-1)
            # Log probability of being Last or Unit
            log_is_end = torch.stack([log_marginals[..., L_idx], log_marginals[..., U_idx]], dim=-1).logsumexp(-1)
            # Log probability of NOT being O for this label (= being I, B, L, or U)
            log_is_inside = torch.stack([
                log_marginals[..., I_idx],
                log_marginals[..., B_idx],
                log_marginals[..., L_idx],
                log_marginals[..., U_idx]
            ], dim=-1).logsumexp(-1)

            # Cumulative sum of log inside probabilities for no-hole constraint
            # Pad with 0 at start for proper indexing
            log_inside_cumsum = torch.cat([
                torch.zeros(batch_size, 1, device=device),
                log_is_inside.cumsum(dim=-1)
            ], dim=-1)

            # has_no_hole[i, j] = sum of log_is_inside from i to j (inclusive)
            # = cumsum[j+1] - cumsum[i]
            has_no_hole = log_inside_cumsum[:, None, 1:] - log_inside_cumsum[:, :-1, None]

            # Span score = min(has_no_hole, log_begin[start], log_end[end])
            # This is the nlstruct approach: all three conditions must be satisfied
            span_scores = torch.minimum(
                torch.minimum(
                    has_no_hole.clamp(max=-eps),  # Avoid exact 0 for numerical stability
                    log_is_begin[:, :, None],
                ),
                log_is_end[:, None, :],
            )

            boundary_scores[..., label_idx] = span_scores

        # Mask invalid spans (start > end)
        span_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        boundary_scores = boundary_scores.masked_fill(~span_mask.unsqueeze(0).unsqueeze(-1), -1e9)

        return {
            'tag_logits': tag_logits,
            'boundary_scores': boundary_scores,
            'log_marginals': log_marginals,
        }


class BiaffineSpanScorer(nn.Module):
    """Biaffine scorer for span classification (nlstruct approach).

    Uses per-label begin/end projections with scaled dot product attention.
    """

    def __init__(
        self,
        input_size: int,
        num_labels: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_size = hidden_size

        # Per-label begin/end projections
        self.begin_proj = nn.Linear(input_size, hidden_size * num_labels)
        self.end_proj = nn.Linear(input_size, hidden_size * num_labels)
        self.bias = nn.Parameter(torch.zeros(()))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
            mask: (batch, seq_len)

        Returns:
            (batch, seq_len, seq_len, num_labels) biaffine scores
        """
        batch_size, seq_len, _ = x.shape
        x = self.dropout(x)

        # Project to per-label begin/end representations
        # (batch, seq_len, hidden_size * num_labels)
        begins = self.begin_proj(x)
        ends = self.end_proj(x)

        # Reshape: (batch, seq_len, num_labels, hidden_size)
        begins = begins.view(batch_size, seq_len, self.num_labels, self.hidden_size)
        ends = ends.view(batch_size, seq_len, self.num_labels, self.hidden_size)

        # Transpose: (batch, num_labels, seq_len, hidden_size)
        begins = begins.permute(0, 2, 1, 3)
        ends = ends.permute(0, 2, 1, 3)

        # Dot product: (batch, num_labels, seq_start, seq_end)
        # einsum: 'nlad,nlbd->nlab' (a=start, b=end, d=hidden)
        scores = torch.einsum('nlad,nlbd->nlab', begins, ends)
        scores = scores / math.sqrt(self.hidden_size) + self.bias

        # Transpose to (batch, seq_start, seq_end, num_labels)
        scores = scores.permute(0, 2, 3, 1)

        return scores


class BERTSpanNER(nn.Module):
    """BERT + BiLSTM + BiTag model for nested NER (nlstruct architecture).

    Combines:
    - BERT encoder for contextualized embeddings
    - BiLSTM contextualizer for boundary-aware representations
    - BIOUL tagger for soft boundary probabilities
    - Biaffine scorer for span classification
    - Combined scoring: tagger_scores + biaffine_scores

    This architecture achieves ~65% exact F1 on EMEA/MEDLINE nested NER.
    """

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        max_span_length: int = 40,
        lstm_hidden_size: int = 400,
        lstm_num_layers: int = 3,
        biaffine_hidden_size: int = 64,
        dropout: float = 0.4,
        use_tagger: bool = True,
        use_biaffine: bool = True,
        use_crf: bool = True,
        tagger_weight: float = 1.0,
        biaffine_weight: float = 1.0,
        **kwargs
    ):
        super().__init__()
        cache_dir = kwargs.get('cache_dir')

        # Check for ModernBERT
        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        extra_kwargs = {}
        if config.model_type == "modernbert":
            extra_kwargs["reference_compile"] = False

        # Flash Attention 2 support (requires bf16/fp16)
        if kwargs.get('attn_implementation'):
            extra_kwargs["attn_implementation"] = kwargs['attn_implementation']
            if kwargs['attn_implementation'] == 'flash_attention_2':
                extra_kwargs["torch_dtype"] = torch.bfloat16

        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            **extra_kwargs
        )

        bert_hidden_size = self.encoder.config.hidden_size

        # BiLSTM contextualizer
        self.contextualizer = BiLSTMContextualizer(
            input_size=bert_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout,
        )

        context_size = self.contextualizer.output_size

        # BIOUL tagger for boundary scores
        self.use_tagger = use_tagger
        if use_tagger:
            self.tagger = BIOULTagger(
                hidden_size=context_size,
                num_labels=num_labels,
                dropout=dropout,
                use_crf=use_crf,
            )

        # Biaffine scorer
        self.use_biaffine = use_biaffine
        if use_biaffine:
            self.biaffine = BiaffineSpanScorer(
                input_size=context_size,
                num_labels=num_labels,
                hidden_size=biaffine_hidden_size,
                dropout=dropout,
            )

        self.max_span_length = max_span_length
        self.num_labels = num_labels
        self.tagger_weight = tagger_weight
        self.biaffine_weight = biaffine_weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        span_indices: torch.Tensor | None = None,
        span_labels: torch.Tensor | None = None,
        span_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Forward pass with combined tagger + biaffine scoring."""
        # BERT encoding
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = outputs.last_hidden_state

        # BiLSTM contextualization
        context_output = self.contextualizer(bert_output, attention_mask)

        # Compute span scores
        batch_size, seq_len = attention_mask.shape
        device = input_ids.device

        # Combined span scores: (batch, seq_len, seq_len, num_labels)
        span_scores = torch.zeros(batch_size, seq_len, seq_len, self.num_labels, device=device)

        tagger_output = None
        if self.use_tagger:
            tagger_output = self.tagger(context_output, attention_mask)
            span_scores = span_scores + self.tagger_weight * tagger_output['boundary_scores']

        if self.use_biaffine:
            biaffine_scores = self.biaffine(context_output, attention_mask)
            span_scores = span_scores + self.biaffine_weight * biaffine_scores

        if span_indices is None:
            return self._inference_forward(span_scores, attention_mask)

        return self._training_forward(
            span_scores, span_indices, span_labels, span_mask, tagger_output
        )

    def _training_forward(
        self,
        span_scores: torch.Tensor,
        span_indices: torch.Tensor,
        span_labels: torch.Tensor | None,
        span_mask: torch.Tensor | None,
        tagger_output: dict | None,
    ) -> dict[str, Any]:
        """Training forward with BCE loss per span."""
        batch_size, num_spans, _ = span_indices.shape
        device = span_scores.device

        # Gather logits for training spans
        # span_indices: (batch, num_spans, 2) -> (start, end)
        start_idx = span_indices[:, :, 0]  # (batch, num_spans)
        end_idx = span_indices[:, :, 1]  # (batch, num_spans)

        # Gather from span_scores: (batch, seq, seq, num_labels)
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_spans)
        logits = span_scores[batch_idx, start_idx, end_idx, :]  # (batch, num_spans, num_labels)

        loss = None
        if span_labels is not None:
            # BCE loss with sigmoid (multi-label style like nlstruct)
            targets = F.one_hot(span_labels, num_classes=self.num_labels).float()

            # BCE with logits
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            loss = loss.mean(dim=-1)  # Average over labels

            if span_mask is not None:
                loss = (loss * span_mask).sum() / span_mask.sum().clamp(min=1)
            else:
                loss = loss.mean()

        return {'loss': loss, 'logits': logits}

    def _inference_forward(
        self,
        span_scores: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, Any]:
        """Inference: extract spans above threshold."""
        batch_size, seq_len, _, num_labels = span_scores.shape
        device = span_scores.device

        all_spans = []
        all_logits = []

        for b in range(batch_size):
            valid_len = int(attention_mask[b].sum().item())
            valid_start = 1  # Skip CLS
            valid_end = valid_len - 1  # Skip SEP

            spans = []
            logits_list = []

            for start in range(valid_start, valid_end):
                for end in range(start, min(start + self.max_span_length, valid_end)):
                    spans.append((start, end))
                    logits_list.append(span_scores[b, start, end, :])

            if spans:
                all_spans.append(spans)
                all_logits.append(torch.stack(logits_list))
            else:
                all_spans.append([])
                all_logits.append(torch.empty(0, num_labels, device=device))

        return {'spans': all_spans, 'logits': all_logits, 'span_scores': span_scores}


@dataclass
class SpanNERDataCollator:
    """Data collator for span NER that handles variable-length span lists."""

    tokenizer: Any
    max_spans: int = 512

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        # Separate tokenizer inputs from span data
        tokenizer_features = [
            {'input_ids': f['input_ids'], 'attention_mask': f['attention_mask']}
            for f in features
        ]

        batch = self.tokenizer.pad(tokenizer_features, return_tensors='pt')

        # Pad span data
        max_spans = min(
            max(len(f['span_indices']) for f in features),
            self.max_spans
        )

        batch_size = len(features)
        span_indices = torch.zeros(batch_size, max_spans, 2, dtype=torch.long)
        span_labels = torch.zeros(batch_size, max_spans, dtype=torch.long)
        span_mask = torch.zeros(batch_size, max_spans, dtype=torch.float)

        for i, f in enumerate(features):
            num_spans = min(len(f['span_indices']), max_spans)
            if num_spans > 0:
                span_indices[i, :num_spans] = torch.tensor(f['span_indices'][:num_spans])
                span_labels[i, :num_spans] = torch.tensor(f['span_labels'][:num_spans])
                span_mask[i, :num_spans] = 1.0

        batch['span_indices'] = span_indices
        batch['span_labels'] = span_labels
        batch['span_mask'] = span_mask

        return batch


class BERTSpanNERAdapter(ModelAdapter):
    """BERT + BiLSTM + BiTag adapter for nested NER (nlstruct-inspired architecture).

    Uses BERTSpanNER which combines:
    - BiLSTM contextualizer
    - BIOUL tagger for boundary detection
    - Biaffine scorer for span classification
    - Combined scoring: tagger + biaffine

    Achieves ~65% exact F1 on EMEA/MEDLINE nested NER.
    """

    def __init__(self):
        self._tokenizer: AutoTokenizer | None = None
        self._task = None
        self._label_to_id: dict[str, int] = {}
        self._id_to_label: dict[int, str] = {}
        self._max_span_length = 40
        self._negative_ratio = -1  # -1 = use all spans (nlstruct)
        self._max_length = 512
        self._max_spans = 512

    @property
    def name(self) -> str:
        return "bert_span_ner"

    @property
    def supports_finetune(self) -> bool:
        return True

    def build_model(
        self,
        task: Any,
        model_name_or_path: str = "camembert-base",
        **kwargs: Any
    ) -> tuple[Any, AutoTokenizer]:
        """Build BERT + BiLSTM + BiTag model."""
        from tabib.tasks.ner_span import NERSpanTask

        if not isinstance(task, NERSpanTask):
            raise ValueError(f"Expected NERSpanTask, got {type(task)}")

        self._task = task
        self._max_span_length = kwargs.get('max_span_length', 40)
        self._negative_ratio = kwargs.get('negative_ratio', -1)  # -1 = use all spans (nlstruct)
        self._max_length = kwargs.get('max_length', 256)
        self._max_spans = kwargs.get('max_spans', 512)

        # Build label mappings: O + entity types
        entity_types = task.label_space
        self._label_to_id = {'O': 0}
        self._label_to_id.update({t: i + 1 for i, t in enumerate(entity_types)})
        self._id_to_label = {v: k for k, v in self._label_to_id.items()}

        cache_dir = kwargs.get('cache_dir')

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir
        )

        # Model hyperparameters (can be overridden via config)
        model = BERTSpanNER(
            model_name_or_path,
            num_labels=len(self._label_to_id),
            max_span_length=self._max_span_length,
            lstm_hidden_size=kwargs.get('lstm_hidden_size', 400),
            lstm_num_layers=kwargs.get('lstm_num_layers', 3),
            biaffine_hidden_size=kwargs.get('biaffine_hidden_size', 64),
            dropout=kwargs.get('dropout', 0.4),
            use_tagger=kwargs.get('use_tagger', True),
            use_biaffine=kwargs.get('use_biaffine', True),
            use_crf=kwargs.get('use_crf', True),
            tagger_weight=kwargs.get('tagger_weight', 1.0),
            biaffine_weight=kwargs.get('biaffine_weight', 1.0),
            cache_dir=cache_dir,
        )

        return model, self._tokenizer

    def _tokenize_dataset(self, dataset: Any, tokenizer: AutoTokenizer) -> Any:
        """Tokenize and prepare span training examples."""

        def process_example(examples: dict[str, Any]) -> dict[str, Any]:
            texts = examples['text']
            entities_list = examples['entities']

            tokenized = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self._max_length,
                return_offsets_mapping=True,
            )

            all_span_indices = []
            all_span_labels = []

            for i in range(len(texts)):
                entities = entities_list[i]
                offset_mapping = tokenized['offset_mapping'][i]

                spans = self._prepare_spans(
                    entities, offset_mapping, self._max_span_length, self._negative_ratio,
                    text=texts[i]
                )

                all_span_indices.append([s[:2] for s in spans])
                all_span_labels.append([s[2] for s in spans])

            tokenized['span_indices'] = all_span_indices
            tokenized['span_labels'] = all_span_labels
            tokenized.pop('offset_mapping')

            return tokenized

        return dataset.map(
            process_example,
            batched=True,
            remove_columns=dataset.column_names,
        )

    def _prepare_spans(
        self,
        entities: list[dict],
        offset_mapping: list[tuple[int, int]],
        max_span_length: int,
        negative_ratio: int,
        text: str | None = None,
    ) -> list[tuple[int, int, int]]:
        """Convert char spans to token spans and add negative samples."""
        # Find valid token range (skip CLS at start and SEP at end)
        seq_len = len(offset_mapping)

        # Find the last real token (before padding)
        last_real_token = seq_len - 1
        for i in range(seq_len - 1, -1, -1):
            start, end = offset_mapping[i]
            if not (start == 0 and end == 0):
                last_real_token = i
                break

        # Skip CLS (first token) and SEP (last token)
        valid_start = 1  # Skip CLS
        valid_end = last_real_token + 1  # Include last real token, exclude SEP

        # Convert entity char spans to token spans
        positive_spans = []
        for ent in entities:
            start_token = self._char_to_token(
                ent['start'], offset_mapping, valid_start, valid_end, text=text
            )
            end_token = self._char_to_token(
                ent['end'] - 1, offset_mapping, valid_start, valid_end, is_end=True, text=text
            )

            if start_token is not None and end_token is not None and start_token <= end_token:
                label_id = self._label_to_id.get(ent['label'], 0)
                if label_id > 0:
                    positive_spans.append((start_token, end_token, label_id))

        positive_set = {(s, e) for s, e, _ in positive_spans}

        # Generate negative spans
        negative_spans = []
        for start in range(valid_start, valid_end):
            for end in range(start, min(start + max_span_length, valid_end)):
                if (start, end) not in positive_set:
                    negative_spans.append((start, end, 0))

        # Use all negatives up to max limit (nlstruct approach)
        if negative_ratio < 0:
            max_negatives = 500
            sampled_negatives = negative_spans[:max_negatives]
        else:
            num_negatives = min(len(negative_spans), max(len(positive_spans) * negative_ratio, 10))
            if num_negatives > 0 and negative_spans:
                sampled_negatives = random.sample(negative_spans, num_negatives)
            else:
                sampled_negatives = []

        all_spans = positive_spans + sampled_negatives
        random.shuffle(all_spans)

        return all_spans if all_spans else [(valid_start, valid_start, 0)]

    def _char_to_token(
        self,
        char_pos: int,
        offset_mapping: list[tuple[int, int]],
        valid_start: int,
        valid_end: int,
        is_end: bool = False,
        text: str | None = None,
    ) -> int | None:
        """Convert character position to token index.

        Handles SentencePiece tokenizers which have two quirks:
        1. Gaps between tokens (spaces not included in offsets)
        2. Overlapping offsets when '▁' is a separate token, e.g.:
           - '▁' at (17, 18) and 'hypertension' at (17, 29)
           We must prefer the longer token in case of overlap.
        """
        # For end positions, we want the token containing char_pos-1
        search_pos = char_pos - 1 if is_end and char_pos > 0 else char_pos

        # First pass: find all tokens that contain search_pos
        # (there may be multiple due to overlapping offsets in CamemBERT)
        candidates = []
        for i in range(valid_start, valid_end):
            start, end = offset_mapping[i]
            if start == 0 and end == 0:
                continue
            if start <= search_pos < end:
                candidates.append((i, end - start))  # (token_idx, token_length)

        # If we have candidates, prefer the longest token (skip standalone '▁')
        if candidates:
            # Sort by length descending, return the longest
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        # Second pass: handle gaps (CamemBERT doesn't include spaces in offsets)
        # Find the token that starts at or right after search_pos
        if text is not None and search_pos < len(text) and text[search_pos].isspace():
            for i in range(valid_start, valid_end):
                start, end = offset_mapping[i]
                if start == 0 and end == 0:
                    continue
                if start > search_pos:
                    return i

        # Third pass: handle gaps for non-space characters
        # Find the first token whose end is past search_pos
        for i in range(valid_start, valid_end):
            start, end = offset_mapping[i]
            if start == 0 and end == 0:
                continue
            if search_pos < end:
                return i

        return None

    @staticmethod
    def _normalize_char_offsets(text: str, start: int, end: int) -> tuple[int, int]:
        """Normalize character offsets by stripping leading/trailing whitespace.

        SentencePiece tokenizers (CamemBERT-bio, DrBERT) include leading spaces
        in token offsets, which causes off-by-one errors in span extraction.
        """
        # Strip leading whitespace
        while start < end and start < len(text) and text[start].isspace():
            start += 1

        # Strip trailing whitespace
        while end > start and text[end - 1].isspace():
            end -= 1

        return start, end

    def _create_compute_metrics_fn(self):
        """Create compute_metrics function for Trainer."""

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred

            # predictions: (batch, num_spans, num_labels)
            # labels: (batch, num_spans)
            pred_labels = predictions.argmax(axis=-1)

            # Compute accuracy on non-padded spans
            mask = labels != 0  # Focus on positive examples
            if mask.sum() > 0:
                correct = (pred_labels[mask] == labels[mask]).sum()
                total = mask.sum()
                accuracy = correct / total
            else:
                accuracy = 0.0

            # Approximate F1 (proper F1 computed in final eval)
            tp = ((pred_labels > 0) & (pred_labels == labels)).sum()
            fp = ((pred_labels > 0) & (pred_labels != labels)).sum()
            fn = ((labels > 0) & (pred_labels != labels)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
            }

        return compute_metrics

    def get_trainer(
        self,
        model: Any,
        train_dataset: Any,
        eval_dataset: Any | None = None,
        output_dir: str = "./outputs",
        num_train_epochs: int = 10,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 8,
        learning_rate: float = 1e-3,
        bert_learning_rate: float = 5e-5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        gradient_clip: float = 10.0,
        logging_steps: int = 50,
        seed: int = 42,
        **kwargs: Any
    ) -> Trainer:
        """Get Trainer instance with separate LRs for BERT vs other layers."""
        from torch.optim import AdamW

        if isinstance(model, tuple):
            model, tokenizer = model
        else:
            if self._tokenizer is None:
                raise ValueError("Tokenizer not initialized")
            tokenizer = self._tokenizer

        tokenized_train = self._tokenize_dataset(train_dataset, tokenizer)
        tokenized_eval = self._tokenize_dataset(eval_dataset, tokenizer) if eval_dataset else None

        data_collator = SpanNERDataCollator(
            tokenizer=tokenizer,
            max_spans=self._max_spans,
        )

        eval_strategy = kwargs.pop("eval_strategy", kwargs.pop("evaluation_strategy", "epoch"))
        save_strategy = kwargs.pop("save_strategy", "epoch")

        early_stopping_patience = kwargs.pop("early_stopping_patience", None)
        early_stopping_threshold = kwargs.pop("early_stopping_threshold", 0.0)

        metric_for_best_model = kwargs.pop("metric_for_best_model", "eval_f1" if eval_dataset else None)
        greater_is_better = kwargs.pop("greater_is_better", True)
        load_best_model_at_end = kwargs.pop("load_best_model_at_end", bool(eval_dataset))

        # Create optimizer with separate learning rates (nlstruct approach)
        bert_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'encoder' in name:
                bert_params.append(param)
            else:
                other_params.append(param)

        optimizer = AdamW([
            {'params': bert_params, 'lr': bert_learning_rate, 'weight_decay': weight_decay},
            {'params': other_params, 'lr': learning_rate, 'weight_decay': weight_decay},
        ])

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            max_grad_norm=gradient_clip,
            logging_steps=logging_steps,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            seed=seed,
            bf16=kwargs.pop("bf16", True),
            save_total_limit=kwargs.pop("save_total_limit", 2),
            **{k: v for k, v in kwargs.items() if v is not None},
        )

        callbacks = []
        if early_stopping_patience is not None and eval_dataset is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_threshold=early_stopping_threshold,
                )
            )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
            compute_metrics=self._create_compute_metrics_fn(),
            callbacks=callbacks if callbacks else None,
            optimizers=(optimizer, None),
        )

        return trainer

    def predict(self, model: Any, inputs: Any, **kwargs: Any) -> list[dict]:
        """Run inference with sigmoid threshold and overlap filtering."""
        if isinstance(model, tuple):
            model, tokenizer = model
        else:
            if self._tokenizer is None:
                raise ValueError("Tokenizer not initialized")
            tokenizer = self._tokenizer

        device = self._resolve_device()
        model.to(device)
        model.eval()

        threshold = kwargs.get('threshold', 0.5)
        filter_mode = kwargs.get('filter_predictions', 'no_overlapping_same_label')
        predictions = []

        for example in inputs:
            text = example['text']
            doc_id = example['doc_id']

            tokens = tokenizer(
                text,
                truncation=True,
                max_length=self._max_length,
                return_offsets_mapping=True,
                return_tensors='pt',
            )

            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            offset_mapping = tokens['offset_mapping'][0].tolist()

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            spans = outputs['spans'][0]
            logits = outputs['logits'][0]

            if len(spans) == 0:
                predictions.append({'doc_id': doc_id, 'entities': []})
                continue

            # Use sigmoid for BCE-trained model
            probs = torch.sigmoid(logits)

            entities = []
            for idx, (start_tok, end_tok) in enumerate(spans):
                # Get best non-O label above threshold
                entity_probs = probs[idx, 1:]  # Exclude O
                max_prob, max_idx = entity_probs.max(dim=0)

                if max_prob.item() > threshold:
                    label_id = max_idx.item() + 1
                    char_start = offset_mapping[start_tok][0]
                    char_end = offset_mapping[end_tok][1]

                    # Normalize offsets (fix SentencePiece whitespace issues)
                    char_start, char_end = self._normalize_char_offsets(
                        text, char_start, char_end
                    )

                    if char_start < char_end:
                        entities.append({
                            'start': char_start,
                            'end': char_end,
                            'label': self._id_to_label[label_id],
                            'text': text[char_start:char_end],
                            'confidence': max_prob.item(),
                        })

            # Filter overlapping predictions
            if filter_mode:
                entities = self._filter_overlapping_predictions(entities, mode=filter_mode)

            predictions.append({
                'doc_id': doc_id,
                'entities': entities,
            })

        return predictions

    @staticmethod
    def _filter_overlapping_predictions(
        entities: list[dict],
        mode: str = "no_overlapping_same_label"
    ) -> list[dict]:
        """Filter overlapping predictions (nlstruct approach).

        Args:
            entities: List of entity dicts with 'start', 'end', 'label', 'confidence'
            mode: "no_overlapping_same_label" - remove overlaps within same label
                  "no_overlapping" - remove all overlaps

        Returns:
            Filtered list of entities
        """
        if not entities:
            return entities

        # Sort by confidence (desc), then by span length (desc) for tiebreaker
        sorted_ents = sorted(
            entities,
            key=lambda e: (-e.get('confidence', 0), -(e['end'] - e['start']))
        )

        kept = []
        for ent in sorted_ents:
            overlap = False
            for kept_ent in kept:
                # Check overlap
                if ent['start'] < kept_ent['end'] and ent['end'] > kept_ent['start']:
                    if mode == "no_overlapping":
                        overlap = True
                        break
                    elif mode == "no_overlapping_same_label":
                        if ent['label'] == kept_ent['label']:
                            overlap = True
                            break
            if not overlap:
                kept.append(ent)

        # Sort back by position
        return sorted(kept, key=lambda e: (e['start'], e['end']))

    @staticmethod
    def _resolve_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
