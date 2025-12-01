"""Span-based evaluator for character-offset NER with nested entities."""

from collections import defaultdict
from typing import Any

from tabib.evaluation.base import Evaluator


class SpanEvaluator(Evaluator):
    """Evaluate character-offset span predictions.
    
    Supports nested entities and computes both exact and partial match F1.
    """
    
    def evaluate(
        self, predictions: Any, references: Any, **kwargs: Any
    ) -> dict[str, float]:
        """Evaluate span predictions.
        
        Args:
            predictions: List of documents with predicted spans
            references: List of documents with gold spans
            
        Returns:
            Dictionary with exact_f1, partial_f1, and per-entity metrics
        """
        # Extract spans from predictions and references
        pred_spans = self._extract_spans(predictions)
        gold_spans = self._extract_spans(references)
        
        # Compute exact match metrics
        exact_metrics = self._compute_metrics(pred_spans, gold_spans, exact=True)
        
        # Compute partial match metrics
        partial_metrics = self._compute_metrics(pred_spans, gold_spans, exact=False)
        
        # Combine results
        results = {
            'exact_f1': exact_metrics['micro_f1'],
            'exact_precision': exact_metrics['micro_precision'],
            'exact_recall': exact_metrics['micro_recall'],
            'partial_f1': partial_metrics['micro_f1'],
            'partial_precision': partial_metrics['micro_precision'],
            'partial_recall': partial_metrics['micro_recall'],
        }
        
        # Add per-entity type metrics
        for label, metrics in exact_metrics['per_type'].items():
            results[f'{label}_exact_f1'] = metrics['f1']
        
        return results
    
    def _extract_spans(self, data: Any) -> list[set[tuple]]:
        """Extract spans from data format.
        
        Returns list of sets, one per document, containing (start, end, label) tuples.
        """
        if isinstance(data, list):
            # List of documents
            return [
                set((s['start'], s['end'], s['label']) for s in doc.get('entities', []))
                for doc in data
            ]
        elif hasattr(data, '__iter__'):
            # Dataset-like object
            return [
                set((s['start'], s['end'], s['label']) for s in example.get('entities', []))
                for example in data
            ]
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
    
    def _compute_metrics(
        self, pred_spans: list[set], gold_spans: list[set], exact: bool
    ) -> dict:
        """Compute precision, recall, F1 for spans."""
        # Overall counts
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        # Per-type counts
        type_counts = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        for pred_set, gold_set in zip(pred_spans, gold_spans):
            if exact:
                # Exact match: (start, end, label) must all match
                tp = len(pred_set & gold_set)
                fp = len(pred_set - gold_set)
                fn = len(gold_set - pred_set)
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                
                # Per-type exact match
                for span in pred_set & gold_set:
                    type_counts[span[2]]['tp'] += 1
                for span in pred_set - gold_set:
                    type_counts[span[2]]['fp'] += 1
                for span in gold_set - pred_set:
                    type_counts[span[2]]['fn'] += 1
            else:
                # Partial match: any overlap with same label
                matched_pred = set()
                matched_gold = set()
                
                for p_span in pred_set:
                    p_start, p_end, p_label = p_span
                    for g_span in gold_set:
                        g_start, g_end, g_label = g_span
                        # Check overlap and label match
                        if p_label == g_label and not (p_end <= g_start or g_end <= p_start):
                            matched_pred.add(p_span)
                            matched_gold.add(g_span)
                            type_counts[p_label]['tp'] += 1
                            break
                
                tp = len(matched_pred)
                fp = len(pred_set - matched_pred)
                fn = len(gold_set - matched_gold)
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                
                for span in pred_set - matched_pred:
                    type_counts[span[2]]['fp'] += 1
                for span in gold_set - matched_gold:
                    type_counts[span[2]]['fn'] += 1
        
        # Compute micro-averaged metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Compute per-type metrics
        per_type = {}
        for label, counts in type_counts.items():
            tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            per_type[label] = {'precision': p, 'recall': r, 'f1': f}
        
        return {
            'micro_precision': precision,
            'micro_recall': recall,
            'micro_f1': f1,
            'per_type': per_type
        }

