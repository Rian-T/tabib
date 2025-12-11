"""BERT-based text classification model adapter."""

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
from tabib.tasks.classification import ClassificationTask


class BERTTextClassificationAdapter(ModelAdapter):
    """Adapter for fine-tuning BERT models on text classification tasks."""

    def __init__(self) -> None:
        self._tokenizer: AutoTokenizer | None = None

    @property
    def name(self) -> str:
        return "bert_text_cls"

    @property
    def supports_finetune(self) -> bool:
        return True

    def build_model(
        self,
        task: Any,
        model_name_or_path: str = "bert-base-uncased",
        **kwargs: Any,
    ) -> tuple[Any, AutoTokenizer]:
        if not isinstance(task, ClassificationTask):
            raise ValueError(f"Expected ClassificationTask, got {type(task)}")

        if not task.label_list:
            raise ValueError("ClassificationTask must have label_list before building model")

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
            num_labels=task.num_labels,
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
            remove_columns=list(dataset.column_names),
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

        device = self._resolve_device()
        model.to(device)
        model.eval()

        dataloader = torch.utils.data.DataLoader(
            tokenized,
            batch_size=kwargs.get("batch_size", 8),
            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
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
