"""BERT-based multi-label classification model adapter."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from tabib.models.base import ModelAdapter
from tabib.tasks.multilabel import MultiLabelTask

# Default threshold for multi-label classification
# With weighted BCE loss, use higher threshold (0.5) to balance precision/recall
DEFAULT_THRESHOLD = 0.5


class MultiLabelDataCollator(DataCollatorWithPadding):
    """Data collator that handles multi-hot labels as float tensors."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # Extract labels before padding (they're already multi-hot vectors)
        labels = [f.pop("labels") for f in features]

        # Pad the rest of the features
        batch = super().__call__(features)

        # Add labels back as float tensor
        batch["labels"] = torch.tensor(labels, dtype=torch.float)
        return batch


class WeightedBCETrainer(Trainer):
    """Custom Trainer with weighted BCE loss for class imbalance.

    Weights positive examples in proportion to their inverse prevalence
    in the training data, as described in the MedDialog-FR paper.
    """

    def __init__(self, pos_weight: torch.Tensor | None = None, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Use weighted BCE loss
        if self.pos_weight is not None:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(logits.device))
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


def compute_pos_weights(dataset: Dataset, num_labels: int) -> torch.Tensor:
    """Compute positive class weights based on label prevalence.

    Weight = (num_negative) / (num_positive) for each label.
    This upweights rare positive labels.

    Args:
        dataset: Training dataset with 'labels' column (multi-hot vectors)
        num_labels: Number of labels

    Returns:
        Tensor of shape (num_labels,) with positive weights
    """
    # Count positive examples per label
    pos_counts = np.zeros(num_labels)
    for sample in dataset:
        labels = sample["labels"]
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        pos_counts += np.array(labels)

    total_samples = len(dataset)
    neg_counts = total_samples - pos_counts

    # Compute weights: neg/pos ratio (clipped to avoid division by zero)
    pos_counts = np.maximum(pos_counts, 1)  # Avoid division by zero
    weights = neg_counts / pos_counts

    return torch.tensor(weights, dtype=torch.float)


class BERTMultiLabelAdapter(ModelAdapter):
    """Adapter for fine-tuning BERT models on multi-label classification tasks.

    Uses weighted BCEWithLogitsLoss for training to handle class imbalance,
    as described in the MedDialog-FR paper.
    """

    def __init__(self) -> None:
        self._tokenizer: AutoTokenizer | None = None
        self._pos_weight: torch.Tensor | None = None
        self._threshold: float = DEFAULT_THRESHOLD

    @property
    def name(self) -> str:
        return "bert_multilabel_cls"

    @property
    def supports_finetune(self) -> bool:
        return True

    def build_model(
        self,
        task: Any,
        model_name_or_path: str = "bert-base-uncased",
        **kwargs: Any,
    ) -> tuple[Any, AutoTokenizer]:
        if not isinstance(task, MultiLabelTask):
            raise ValueError(f"Expected MultiLabelTask, got {type(task)}")

        if not task.label_list:
            raise ValueError("MultiLabelTask must have label_list before building model")

        # Offline/cache support
        cache_dir = kwargs.get("cache_dir")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            cache_dir=cache_dir,
        )

        # Check if this is a ModernBERT model - disable torch.compile
        config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        extra_kwargs = {}
        if config.model_type == "modernbert":
            extra_kwargs["reference_compile"] = False

        # Configure for multi-label classification
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=task.num_labels,
            problem_type="multi_label_classification",
            id2label={idx: label for idx, label in enumerate(task.label_list)},
            label2id={label: idx for idx, label in enumerate(task.label_list)},
            cache_dir=cache_dir,
            **extra_kwargs,
        )
        return model, self._tokenizer

    def _tokenize_dataset(self, dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
        def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
            tokenized = tokenizer(batch["text"], truncation=True)
            tokenized["labels"] = batch["labels"]
            return tokenized

        tokenized = dataset.map(
            tokenize,
            batched=True,
            remove_columns=[c for c in dataset.column_names if c != "labels"],
        )
        return tokenized

    def _create_compute_metrics_fn(self, threshold: float = DEFAULT_THRESHOLD):
        """Create compute_metrics function for Trainer.

        Args:
            threshold: Probability threshold for positive prediction.
                       Default 0.41 from MedDialog-FR paper.
        """
        from sklearn.metrics import f1_score, precision_score, recall_score

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred

            # Sigmoid + threshold
            probs = 1 / (1 + np.exp(-predictions))
            pred_binary = (probs > threshold).astype(int)

            # Fallback: if no predictions, predict top-1 for each sample
            # This ensures we always make at least one prediction
            for i in range(len(pred_binary)):
                if pred_binary[i].sum() == 0:
                    top_idx = probs[i].argmax()
                    pred_binary[i, top_idx] = 1

            return {
                "f1_micro": f1_score(labels, pred_binary, average="micro", zero_division=0),
                "f1_macro": f1_score(labels, pred_binary, average="macro", zero_division=0),
                "f1_weighted": f1_score(labels, pred_binary, average="weighted", zero_division=0),
                "f1_samples": f1_score(labels, pred_binary, average="samples", zero_division=0),
                "precision_micro": precision_score(labels, pred_binary, average="micro", zero_division=0),
                "recall_micro": recall_score(labels, pred_binary, average="micro", zero_division=0),
            }

        return compute_metrics

    def get_trainer(
        self,
        model: Any,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        output_dir: str = "./outputs",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        logging_steps: int = 500,
        eval_steps: int | None = None,
        save_steps: int | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> Trainer | None:
        if isinstance(model, tuple):
            model, tokenizer = model
        else:
            if self._tokenizer is None:
                raise ValueError("Tokenizer not initialized")
            tokenizer = self._tokenizer

        tokenized_train = self._tokenize_dataset(train_dataset, tokenizer)
        tokenized_eval = self._tokenize_dataset(eval_dataset, tokenizer) if eval_dataset else None

        # Compute positive class weights from training data for weighted BCE loss
        num_labels = model.config.num_labels
        self._pos_weight = compute_pos_weights(train_dataset, num_labels)

        # Use custom data collator for multi-hot labels
        data_collator = MultiLabelDataCollator(tokenizer=tokenizer)

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

        # Get threshold from config (supports per-dataset tuning)
        self._threshold = training_args_kwargs.pop("multilabel_threshold", DEFAULT_THRESHOLD)

        if eval_steps is not None:
            training_args_kwargs["eval_steps"] = eval_steps
        if save_steps is not None:
            training_args_kwargs["save_steps"] = save_steps

        init_params = inspect.signature(TrainingArguments.__init__).parameters
        eval_strategy_value = "steps" if eval_dataset else "no"
        save_strategy_value = "steps" if save_steps else "epoch"

        if "evaluation_strategy" in init_params:
            training_args_kwargs.setdefault("evaluation_strategy", eval_strategy_value)
        elif "eval_strategy" in init_params:
            training_args_kwargs.setdefault("eval_strategy", eval_strategy_value)

        if "save_strategy" in init_params:
            training_args_kwargs.setdefault("save_strategy", save_strategy_value)

        if "load_best_model_at_end" in init_params:
            training_args_kwargs.setdefault("load_best_model_at_end", bool(eval_dataset))

        if eval_dataset:
            training_args_kwargs.setdefault("metric_for_best_model", "eval_f1_micro")
            training_args_kwargs.setdefault("greater_is_better", True)

        training_args_kwargs = {k: v for k, v in training_args_kwargs.items() if v is not None}

        training_args = TrainingArguments(**training_args_kwargs)

        # Use WeightedBCETrainer with positive class weights
        trainer = WeightedBCETrainer(
            pos_weight=self._pos_weight,
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
            compute_metrics=self._create_compute_metrics_fn(threshold=self._threshold),
            callbacks=(
                [
                    EarlyStoppingCallback(
                        early_stopping_patience=early_stopping_patience,
                        early_stopping_threshold=early_stopping_threshold,
                    )
                ]
                if early_stopping_patience is not None and eval_dataset is not None
                else None
            ),
        )
        return trainer

    def predict(self, model: Any, inputs: Dataset, **kwargs: Any) -> dict[str, Any]:
        if isinstance(model, tuple):
            model, tokenizer = model
        else:
            if self._tokenizer is None:
                raise ValueError("Tokenizer not initialized")
            tokenizer = self._tokenizer

        tokenized = self._tokenize_dataset(inputs, tokenizer)

        device = self._resolve_device()
        model.to(device)
        model.eval()

        data_collator = MultiLabelDataCollator(tokenizer=tokenizer)
        dataloader = torch.utils.data.DataLoader(
            tokenized,
            batch_size=kwargs.get("batch_size", 8),
            collate_fn=data_collator,
        )

        all_logits = []
        all_labels = []
        for batch in dataloader:
            labels = batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits.cpu()
            all_logits.append(logits)
            if labels is not None:
                all_labels.append(labels)

        result = {"predictions": torch.cat(all_logits).numpy()}
        if all_labels:
            result["label_ids"] = torch.cat(all_labels).numpy()
        return result

    @staticmethod
    def _resolve_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
