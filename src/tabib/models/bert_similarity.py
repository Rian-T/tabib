"""BERT-based semantic similarity model adapter."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from tabib.models.base import ModelAdapter
from tabib.tasks.similarity import SimilarityTask


class BERTSimilarityAdapter(ModelAdapter):
    """Adapter for regression-based semantic similarity fine-tuning."""

    def __init__(self) -> None:
        self._tokenizer: AutoTokenizer | None = None

    @property
    def name(self) -> str:
        return "bert_similarity"

    @property
    def supports_finetune(self) -> bool:
        return True

    def build_model(
        self,
        task: Any,
        model_name_or_path: str = "bert-base-uncased",
        **kwargs: Any,
    ) -> tuple[Any, AutoTokenizer]:
        if not isinstance(task, SimilarityTask):
            raise ValueError(f"Expected SimilarityTask, got {type(task)}")

        # Offline/cache support
        cache_dir = kwargs.get('cache_dir')

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            cache_dir=cache_dir,
        )

        # Check if this is a ModernBERT model - disable torch.compile to avoid
        # DataParallel conflicts
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name_or_path)
        extra_kwargs = {}
        if config.model_type == "modernbert":
            extra_kwargs["reference_compile"] = False

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=1,
            problem_type="regression",
            cache_dir=cache_dir,
            **extra_kwargs,
        )
        return model, self._tokenizer

    def _tokenize_dataset(self, dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
        def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
            tokens = tokenizer(
                batch["text_left"],
                batch["text_right"],
                truncation=True,
            )
            tokens["labels"] = batch["labels"]
            return tokens

        tokenized = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
        )
        return tokenized

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
        logging_steps: int = 100,
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

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

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

        params = inspect.signature(TrainingArguments.__init__).parameters
        eval_strategy_value = "steps" if eval_dataset else "no"
        save_strategy_value = "steps" if save_steps else "epoch"

        if "evaluation_strategy" in params:
            training_args_kwargs["evaluation_strategy"] = eval_strategy_value
        elif "eval_strategy" in params:
            training_args_kwargs["eval_strategy"] = eval_strategy_value

        if "save_strategy" in params:
            training_args_kwargs.setdefault("save_strategy", save_strategy_value)

        if "load_best_model_at_end" in params:
            training_args_kwargs["load_best_model_at_end"] = bool(eval_dataset)

        if eval_dataset:
            training_args_kwargs.setdefault("metric_for_best_model", "eval_loss")
            training_args_kwargs.setdefault("greater_is_better", False)

        training_args_kwargs = {k: v for k, v in training_args_kwargs.items() if v is not None}

        training_args = TrainingArguments(**training_args_kwargs)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        dataloader = torch.utils.data.DataLoader(
            tokenized,
            batch_size=kwargs.get("batch_size", 8),
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
        )

        preds = []
        labels = []
        for batch in dataloader:
            label_tensor = batch.pop("labels", None)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits.detach().cpu()
            preds.append(logits)
            if label_tensor is not None:
                labels.append(label_tensor)

        result = {"predictions": torch.cat(preds).squeeze(-1).numpy()}
        if labels:
            result["label_ids"] = torch.cat(labels).numpy()
        return result
