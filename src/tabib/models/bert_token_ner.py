"""BERT-based token classification model for NER."""

import inspect
from collections import defaultdict
from typing import Any

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from tabib.models.base import ModelAdapter
from tabib.tasks.ner_token import NERTokenTask


class BERTTokenNERAdapter(ModelAdapter):
    """BERT model adapter for token-level NER."""
    
    def __init__(self):
        """Initialize adapter."""
        self._tokenizer: AutoTokenizer | None = None
        self._max_length: int = 512  # Default, can be overridden in build_model
    
    @property
    def name(self) -> str:
        """Return the model name."""
        return "bert_token_ner"
    
    @property
    def supports_finetune(self) -> bool:
        """BERT models support fine-tuning."""
        return True
    
    def build_model(
        self, 
        task: Any, 
        model_name_or_path: str = "bert-base-cased",
        **kwargs: Any
    ) -> tuple[Any, AutoTokenizer]:
        """Build BERT model and tokenizer for token classification.
        
        Args:
            task: Task instance (NERTokenTask or NERSpanTask)
            model_name_or_path: Model name or path
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Tuple of (model, tokenizer)
        """
        from tabib.tasks.ner_span import NERSpanTask

        # Store task for later use
        self._task = task

        # Configure max_length from kwargs (default 512)
        self._max_length = kwargs.get('max_length', 512)
        
        if isinstance(task, NERTokenTask):
            num_labels = task.num_labels
            label_list = task.label_list
            label_space = task.label_space
        elif isinstance(task, NERSpanTask):
            # For span task, create BIO labels from entity types
            entity_types = task.label_space
            label_list = ['O'] + [f'B-{t}' for t in entity_types] + [f'I-{t}' for t in entity_types]
            num_labels = len(label_list)
            label_space = {label: idx for idx, label in enumerate(label_list)}
            self._label_list = label_list
            self._label_space = label_space
        else:
            raise ValueError(f"Expected NERTokenTask or NERSpanTask, got {type(task)}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            id2label={i: label for i, label in enumerate(label_list)},
            label2id=label_space,
        )
        
        return model, self._tokenizer
    
    def _tokenize_dataset(self, dataset: Any, tokenizer: AutoTokenizer) -> Any:
        """Tokenize dataset.
        
        Args:
            dataset: Dataset with either 'tokens'+'labels' or 'text'+'entities'
            tokenizer: Tokenizer instance
            
        Returns:
            Tokenized dataset
        """
        from tabib.tasks.ner_span import NERSpanTask
        
        # Check if this is span-based data
        if isinstance(self._task, NERSpanTask):
            return self._tokenize_span_dataset(dataset, tokenizer)
        else:
            return self._tokenize_token_dataset(dataset, tokenizer)
    
    def _tokenize_token_dataset(self, dataset: Any, tokenizer: AutoTokenizer) -> Any:
        """Tokenize token-based NER dataset."""
        def tokenize_and_align_labels(examples: dict[str, Any]) -> dict[str, Any]:
            tokenized_inputs = tokenizer(
                examples["tokens"],
                truncation=True,
                padding=False,
                is_split_into_words=True,
                max_length=self._max_length,
            )
            
            labels = []
            for i, label in enumerate(examples["labels"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)  # Special tokens
                    elif word_idx != previous_word_idx:
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                
                labels.append(label_ids)
            
            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        
        return dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset.column_names,
        )
    
    def _tokenize_span_dataset(self, dataset: Any, tokenizer: AutoTokenizer) -> Any:
        """Tokenize span-based NER dataset by converting spans to IOB2."""
        def tokenize_and_align_spans(examples: dict[str, Any]) -> dict[str, Any]:
            texts = examples["text"]
            entities_list = examples["entities"]
            
            tokenized_inputs = tokenizer(
                texts,
                truncation=True,
                padding=False,
                max_length=self._max_length,
                return_offsets_mapping=True,
            )
            
            labels = []
            for i in range(len(texts)):
                entities = entities_list[i]
                offset_mapping = tokenized_inputs["offset_mapping"][i]
                
                # Convert spans to IOB2 labels
                label_ids = self._spans_to_iob2_labels(entities, offset_mapping)
                labels.append(label_ids)
            
            tokenized_inputs["labels"] = labels
            tokenized_inputs.pop("offset_mapping")  # Remove, not needed for training
            return tokenized_inputs
        
        return dataset.map(
            tokenize_and_align_spans,
            batched=True,
            remove_columns=dataset.column_names,
        )
    
    def _spans_to_iob2_labels(self, entities: list[dict], offset_mapping: list) -> list[int]:
        """Convert character-offset spans to IOB2 labels aligned with tokens."""
        # Initialize all as O
        o_id = self._label_space['O']
        labels = [o_id] * len(offset_mapping)
        
        # For each token, check if it overlaps with any entity
        for token_idx, (start, end) in enumerate(offset_mapping):
            if start == end:  # Special token
                labels[token_idx] = -100
                continue
            
            # Find entities that overlap with this token
            for entity in entities:
                ent_start = entity['start']
                ent_end = entity['end']
                ent_label = entity['label']
                
                # Check if token overlaps with entity
                if not (end <= ent_start or start >= ent_end):
                    # Token overlaps with entity
                    b_label = f"B-{ent_label}"
                    i_label = f"I-{ent_label}"
                    
                    if b_label not in self._label_space or i_label not in self._label_space:
                        continue
                    
                    # Decide B or I: use B if first token of entity or after O/different entity
                    use_b = True
                    if token_idx > 0 and labels[token_idx-1] != -100:
                        prev_label = self._label_list[labels[token_idx-1]]
                        if prev_label.endswith(f"-{ent_label}"):
                            use_b = False
                    
                    labels[token_idx] = self._label_space[b_label] if use_b else self._label_space[i_label]
                    break  # Use first matching entity
        
        return labels
    
    def _create_compute_metrics_fn(self):
        """Create compute_metrics function for Trainer."""
        from tabib.tasks.ner_span import NERSpanTask
        
        if isinstance(self._task, NERSpanTask):
            # For span task, we can't compute span metrics during training
            # because we don't have the original documents
            # Just return empty dict, metrics computed at end
            return None
        else:
            # For token task, use seqeval
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=-1)
                
                # Remove ignored index (special tokens)
                true_predictions = []
                true_labels = []
                
                for prediction, label in zip(predictions, labels):
                    pred_list = []
                    label_list = []
                    for p, l in zip(prediction, label):
                        if l != -100:
                            pred_list.append(self._task.id_to_label(p))
                            label_list.append(self._task.id_to_label(l))
                    true_predictions.append(pred_list)
                    true_labels.append(label_list)
                
                from seqeval.metrics import f1_score, precision_score, recall_score
                return {
                    "f1": f1_score(true_labels, true_predictions),
                    "precision": precision_score(true_labels, true_predictions),
                    "recall": recall_score(true_labels, true_predictions),
                }
            
            return compute_metrics
    
    def get_trainer(
        self,
        model: Any,
        train_dataset: Any,
        eval_dataset: Any | None = None,
        output_dir: str = "./outputs",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        logging_steps: int = 100,
        eval_steps: int | None = None,
        save_steps: int | None = None,
        seed: int = 42,
        **kwargs: Any
    ) -> Trainer | None:
        """Get Trainer instance for fine-tuning.
        
        Args:
            model: Model instance (tuple of model, tokenizer)
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            **kwargs: Training configuration
            
        Returns:
            Trainer instance
        """
        if isinstance(model, tuple):
            model, tokenizer = model
        else:
            if self._tokenizer is None:
                raise ValueError("Tokenizer not initialized")
            tokenizer = self._tokenizer
        
        # Tokenize datasets
        tokenized_train = self._tokenize_dataset(train_dataset, tokenizer)
        tokenized_eval = self._tokenize_dataset(eval_dataset, tokenizer) if eval_dataset else None
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        
        # Training arguments
        eval_strategy_value = "steps" if eval_dataset else "no"
        save_strategy_value = "steps" if save_steps else "epoch"

        training_args_kwargs = dict(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            seed=seed,
            **kwargs,
        )

        early_stopping_patience = training_args_kwargs.pop("early_stopping_patience", None)
        early_stopping_threshold = training_args_kwargs.pop("early_stopping_threshold", 0.0)

        if eval_steps is not None:
            training_args_kwargs["eval_steps"] = eval_steps
        if save_steps is not None:
            training_args_kwargs["save_steps"] = save_steps

        training_args_init_params = inspect.signature(TrainingArguments.__init__).parameters

        if "evaluation_strategy" in training_args_init_params:
            training_args_kwargs["evaluation_strategy"] = eval_strategy_value
        elif "eval_strategy" in training_args_init_params:
            training_args_kwargs["eval_strategy"] = eval_strategy_value

        if "save_strategy" in training_args_init_params:
            training_args_kwargs["save_strategy"] = save_strategy_value
        elif "save_strategy" not in training_args_kwargs and "save_strategy" in training_args_init_params:
            training_args_kwargs["save_strategy"] = save_strategy_value

        if "load_best_model_at_end" in training_args_init_params:
            training_args_kwargs["load_best_model_at_end"] = bool(eval_dataset)

        if eval_dataset:
            training_args_kwargs.setdefault("metric_for_best_model", "eval_loss")
            training_args_kwargs.setdefault("greater_is_better", False)

        training_args_kwargs = {k: v for k, v in training_args_kwargs.items() if v is not None}

        training_args = TrainingArguments(**training_args_kwargs)

        callbacks = []
        if early_stopping_patience is not None and eval_dataset is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                    early_stopping_threshold=early_stopping_threshold,
                )
            )

        # Create compute_metrics function
        compute_metrics = self._create_compute_metrics_fn()
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks if callbacks else None,
        )
        
        return trainer
    
    def predict(self, model: Any, inputs: Any, **kwargs: Any) -> dict[str, Any]:
        """Run inference.
        
        Args:
            model: Model instance (tuple of model, tokenizer)
            inputs: Input dataset
            **kwargs: Additional inference configuration
            
        Returns:
            Dictionary with predictions (format depends on task type)
        """
        from tabib.tasks.ner_span import NERSpanTask
        
        if isinstance(model, tuple):
            model, tokenizer = model
        else:
            if self._tokenizer is None:
                raise ValueError("Tokenizer not initialized")
            tokenizer = self._tokenizer
        
        # Check if this is a span-based task
        is_span_task = isinstance(self._task, NERSpanTask)
        
        if is_span_task:
            return self._predict_spans(model, tokenizer, inputs, **kwargs)
        else:
            return self._predict_tokens(model, tokenizer, inputs, **kwargs)
    
    def _predict_tokens(self, model: Any, tokenizer: Any, inputs: Any, **kwargs: Any) -> dict[str, Any]:
        """Predict IOB2 tags for token-level NER."""
        # Tokenize inputs
        tokenized = self._tokenize_dataset(inputs, tokenizer)
        
        # Run inference
        device = self._resolve_device()
        model.to(device)
        model.eval()
        
        all_predictions = []
        all_labels = []
        
        batch_size = kwargs.get("batch_size", 8)
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        
        # Process in batches
        for i in range(0, len(tokenized), batch_size):
            batch_indices = list(range(i, min(i + batch_size, len(tokenized))))
            batch_examples = [tokenized[idx] for idx in batch_indices]
            
            batch = data_collator(batch_examples)
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch.get("labels")
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            
            for j, pred_seq in enumerate(predictions):
                label_seq = labels[j].cpu().numpy() if labels is not None else None
                aligned_preds = []
                aligned_labels = []
                
                for k, pred_id in enumerate(pred_seq):
                    if label_seq is None or label_seq[k] != -100:
                        aligned_preds.append(int(pred_id))
                        if label_seq is not None and label_seq[k] != -100:
                            aligned_labels.append(int(label_seq[k]))
                
                all_predictions.append(aligned_preds)
                if aligned_labels:
                    all_labels.append(aligned_labels)
        
        result = {"predictions": all_predictions}
        if all_labels:
            result["label_ids"] = all_labels
        
        return result
    
    def _predict_spans(self, model: Any, tokenizer: Any, inputs: Any, **kwargs: Any) -> list[dict]:
        """Predict character-offset spans for span-based NER."""
        device = self._resolve_device()
        model.to(device)
        model.eval()
        
        batch_size = kwargs.get("batch_size", 8)
        
        # Store chunk predictions
        chunk_predictions = []
        
        for example in inputs:
            # Tokenize single example
            text = example['text']
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=self._max_length,
                return_offsets_mapping=True,
                return_tensors='pt'
            )
            
            # Get predictions
            input_ids = tokens['input_ids'].to(device)
            attention_mask = tokens['attention_mask'].to(device)
            offset_mapping = tokens['offset_mapping'][0]
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            
            predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
            
            # Convert IOB2 predictions to spans
            spans = self._iob2_to_spans(predictions, offset_mapping, text)
            
            # Store with metadata for reassembly
            chunk_predictions.append({
                'doc_id': example['doc_id'],
                'chunk_id': example.get('chunk_id', 0),
                'chunk_offset': example.get('chunk_offset', 0),
                'spans': spans
            })
        
        # Reassemble chunks into documents
        documents = self._reassemble_chunks(chunk_predictions)
        
        return documents
    
    def _iob2_to_spans(self, predictions, offset_mapping, text):
        """Convert IOB2 predictions to character-offset spans."""
        spans = []
        current_span = None
        
        for i, pred_id in enumerate(predictions):
            # Skip special tokens (offset is (0, 0))
            if offset_mapping[i][0] == 0 and offset_mapping[i][1] == 0:
                continue
            
            label = self._label_list[pred_id]
            
            if label.startswith('B-'):
                # Start new span
                if current_span:
                    spans.append(current_span)
                entity_type = label[2:]
                start, _ = offset_mapping[i]
                current_span = {
                    'start': int(start),
                    'end': int(offset_mapping[i][1]),
                    'label': entity_type
                }
            elif label.startswith('I-') and current_span:
                # Continue current span - allow small gaps for whitespace
                entity_type = label[2:]
                token_start = int(offset_mapping[i][0])
                # Allow gap of up to 2 chars (space, punctuation) between tokens
                gap = token_start - current_span['end']
                if entity_type == current_span['label'] and 0 <= gap <= 2:
                    current_span['end'] = int(offset_mapping[i][1])
            elif label == 'O':
                # End current span
                if current_span:
                    spans.append(current_span)
                    current_span = None
        
        # Add final span if any
        if current_span:
            spans.append(current_span)
        
        # Add text to spans
        for span in spans:
            span['text'] = text[span['start']:span['end']]
        
        return spans
    
    def _reassemble_chunks(self, chunk_predictions):
        """Reassemble chunk predictions into document-level predictions."""
        # Group by document
        doc_groups = defaultdict(list)
        for chunk_pred in chunk_predictions:
            doc_groups[chunk_pred['doc_id']].append(chunk_pred)
        
        # Reassemble each document
        documents = []
        for doc_id, chunks in doc_groups.items():
            # Sort chunks by chunk_id
            chunks.sort(key=lambda x: x['chunk_id'])
            
            # Merge spans, adjusting offsets
            all_spans = []
            for chunk in chunks:
                chunk_offset = chunk['chunk_offset']
                for span in chunk['spans']:
                    # Adjust to document-level offsets
                    all_spans.append({
                        'start': span['start'] + chunk_offset,
                        'end': span['end'] + chunk_offset,
                        'label': span['label'],
                        'text': span['text']
                    })
            
            documents.append({
                'doc_id': doc_id,
                'entities': all_spans
            })
        
        return documents

    @staticmethod
    def _resolve_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

