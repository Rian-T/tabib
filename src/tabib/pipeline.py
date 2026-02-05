"""Pipeline that connects task, dataset, and model."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import weave

from tabib.config import RunConfig
from tabib.data.base import DatasetAdapter
from tabib.models.base import ModelAdapter
from tabib.offline import get_model_cache_dir, get_offline_dir, resolve_model_path
from tabib.preprocessing.base import Preprocessor
from tabib.registry import get_dataset, get_model, get_task
from tabib.tasks.base import Task


class Pipeline:
    """Pipeline that connects task, dataset, and model.
    
    If `supports_finetune` and `do_train` are true, it trains.
    Otherwise, it runs inference.
    """
    
    def __init__(self, config: RunConfig):
        """Initialize pipeline with configuration.
        
        Args:
            config: Run configuration
        """
        self.config = config
        
        # Initialize task
        task_class = get_task(config.task)
        self.task: Task = task_class()
        
        # Initialize dataset adapter
        dataset_class = get_dataset(config.dataset)
        self.dataset_adapter: DatasetAdapter = dataset_class()
        
        # Initialize model adapter
        model_class = get_model(config.model)
        self.model_adapter: ModelAdapter = model_class()
        
        # Initialize preprocessor if configured
        self.preprocessor: Preprocessor | None = None
        if config.preprocessing:
            self.preprocessor = self._create_preprocessor(config.preprocessing)
    
    def run(self) -> dict[str, Any]:
        """Run the pipeline (train or evaluate).

        Returns:
            Summary dictionary with config metadata, training details, and
            evaluation metrics.
        """
        # Initialize Weave for tracing (skip in offline mode)
        if os.environ.get("HF_HUB_OFFLINE", "0") != "1":
            try:
                weave.init(f"tabib-french-biomedical-ner")
            except Exception:
                pass  # Skip weave in offline/error cases

        # Load dataset splits
        splits = self.dataset_adapter.load_splits()

        summary: dict[str, Any] = {
            "config": self.config.model_dump(),
            "metadata": {
                "task": self.task.name,
                "dataset": self.dataset_adapter.name,
                "model": self.model_adapter.name,
                "preprocessor": (
                    self.preprocessor.__class__.__name__
                    if self.preprocessor
                    else None
                ),
            },
            "training": {
                "performed": False,
                "output_dir": None,
                "best_metric": None,
                "best_model_checkpoint": None,
            },
            "evaluation": {
                "performed": False,
                "split": None,
                "metrics": None,
            },
        }
        
        # Save original train data for few-shot examples (before chunking)
        # Fall back to dev if train doesn't exist (e.g., CAS1, CAS2)
        original_train = splits.get("train") or splits.get("dev")

        # Apply preprocessor if configured (e.g., chunking)
        if self.preprocessor:
            max_length = self.config.preprocessing.get('max_tokens', 512)
            for split_name in list(splits.keys()):
                chunked = self.preprocessor.preprocess(
                    splits[split_name], max_length
                )
                splits[split_name] = chunked

        # Preprocess datasets for task
        processed_splits = {}
        for split_name, split_data in splits.items():
            processed_splits[split_name] = self.dataset_adapter.preprocess(
                split_data, self.task
            )

        # Build model (pass PREPROCESSED train data for few-shot examples)
        # Few-shot needs the formatted 'text' and 'label_text' fields created by preprocess()
        train_data = processed_splits.get("train") or processed_splits.get("dev")

        # Resolve offline directory and model path
        offline_dir = get_offline_dir(self.config.offline_dir)
        model_cache_dir = get_model_cache_dir(offline_dir)
        resolved_model_path = resolve_model_path(
            self.config.model_name_or_path, offline_dir
        )

        model = self.model_adapter.build_model(
            self.task,
            model_name_or_path=resolved_model_path,
            train_data=train_data,
            cache_dir=str(model_cache_dir) if model_cache_dir else None,
            **self.config.backend_args
        )
        
        # Train if applicable
        if self.config.do_train and self.model_adapter.supports_finetune:
            if self.config.training is None:
                raise ValueError("Training config required when do_train=True")
            
            train_dataset = processed_splits.get("train")
            if train_dataset is None:
                raise ValueError("Training dataset not found")
            
            eval_dataset = (
                processed_splits.get("val")
                or processed_splits.get("dev")
                or processed_splits.get("test")
            )
            
            training_kwargs = self.config.training.model_dump()
            training_kwargs["output_dir"] = self._resolve_output_dir(
                training_kwargs.get("output_dir")
            )
            summary["training"]["performed"] = True
            summary["training"]["output_dir"] = training_kwargs["output_dir"]

            trainer = self.model_adapter.get_trainer(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                **training_kwargs,
            )
            
            if trainer is None:
                raise ValueError("Trainer not available for this model")
            
            trainer.train()

            if hasattr(trainer, "state"):
                summary["training"]["best_metric"] = getattr(trainer.state, "best_metric", None)
                summary["training"]["best_model_checkpoint"] = getattr(
                    trainer.state, "best_model_checkpoint", None
                )
            
            # After training, update model reference (Trainer may have loaded best model)
            model = trainer.model
        
        # Evaluate if applicable
        if self.config.do_eval:
            # Evaluate on test set if available, otherwise dev, otherwise val
            eval_dataset = (
                processed_splits.get("test")
                or processed_splits.get("dev")
                or processed_splits.get("val")
            )
            if eval_dataset is None:
                raise ValueError("Evaluation dataset not found")
            
            # Get original data for evaluation (before preprocessing)
            eval_split_name = "test" if "test" in splits else ("dev" if "dev" in splits else "val")
            original_eval = splits[eval_split_name]
            
            predictions = self.model_adapter.predict(model, eval_dataset)
            metrics = self.task.compute_metrics(predictions, original_eval)

            # LLM-as-a-Judge evaluation if configured
            if self.config.judge_config:
                import gc
                import torch
                from tabib.evaluation.llm_judge import LLMJudgeEvaluator, check_vram_availability
                from tabib.models.vllm_common import shutdown_vllm_engine, VLLMEngine

                can_parallel, available_gb = check_vram_availability(72.0)
                print(f"\nVRAM available: {available_gb:.1f}GB (parallel: {can_parallel})")

                if not can_parallel:
                    # Properly shutdown vLLM engine if model has one
                    if hasattr(model, "engine") and isinstance(model.engine, VLLMEngine):
                        print("Shutting down candidate model vLLM engine...")
                        shutdown_vllm_engine(model.engine)
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    # Give GPU some time to fully release memory
                    import time
                    time.sleep(2)
                    torch.cuda.synchronize()

                judge = LLMJudgeEvaluator(**self.config.judge_config)
                questions = [ex.get("text", "") for ex in original_eval]
                judge_metrics = judge.evaluate(predictions, original_eval, questions)
                metrics.update(judge_metrics)

            print(f"\nEvaluation metrics ({eval_split_name} set):")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value}")

            summary["evaluation"]["performed"] = True
            summary["evaluation"]["split"] = eval_split_name
            summary["evaluation"]["metrics"] = metrics

        return summary

    def _resolve_output_dir(self, output_dir: str | None) -> str:
        """Resolve training output directory.

        If an absolute path is provided, it is used as-is. Otherwise, the
        directory is placed under ``$SCRATCH/tabib/runs`` when the environment
        variable ``SCRATCH`` exists, falling back to ``./runs`` relative to the
        current working directory.
        """

        if output_dir:
            output_path = Path(output_dir)
            if output_path.is_absolute():
                resolved = output_path
            else:
                resolved = self._default_runs_dir() / output_path
        else:
            resolved = self._default_runs_dir()

        resolved = resolved.expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return str(resolved)

    def _create_preprocessor(self, config: dict) -> Preprocessor:
        """Create preprocessor from configuration."""
        from tabib.preprocessing.sentence_chunker import SentenceChunker
        from tabib.preprocessing.sentence_splitter import SentenceSplitter
        
        preprocessor_type = config.get('type', 'sentence_chunker')
        
        if preprocessor_type == 'sentence_chunker':
            return SentenceChunker()
        if preprocessor_type in {'sentence_splitter', 'sentence_per_sentence'}:
            return SentenceSplitter()
        raise ValueError(f"Unknown preprocessor type: {preprocessor_type}")
    
    @staticmethod
    def _default_runs_dir() -> Path:
        scratch = os.environ.get("SCRATCH")
        if scratch:
            base = Path(scratch)
            return base / "tabib" / "runs"
        return Path.cwd() / "runs"

