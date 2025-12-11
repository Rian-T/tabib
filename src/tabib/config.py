"""Configuration classes for runs and training."""

from typing import Any

from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Training configuration."""

    model_config = {"extra": "allow"}  # Allow extra fields for LoRA, etc.

    output_dir: str = Field(..., description="Output directory for checkpoints and logs")
    num_train_epochs: int = Field(default=3, description="Number of training epochs")
    per_device_train_batch_size: int = Field(default=8, description="Batch size per device")
    per_device_eval_batch_size: int = Field(default=8, description="Eval batch size per device")
    learning_rate: float = Field(default=2e-5, description="Learning rate")
    warmup_steps: int = Field(default=0, description="Number of warmup steps")
    logging_steps: int = Field(default=100, description="Logging frequency")
    eval_steps: int | None = Field(default=None, description="Evaluation frequency")
    save_steps: int | None = Field(default=None, description="Checkpoint save frequency")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    gradient_accumulation_steps: int = Field(default=1, description="Gradient accumulation steps")
    fp16: bool = Field(default=False, description="Use FP16 mixed precision")
    bf16: bool = Field(default=False, description="Use BF16 mixed precision")
    metric_for_best_model: str | None = Field(default=None, description="Metric used to select the best model")
    greater_is_better: bool | None = Field(default=None, description="Whether a greater metric indicates better performance")
    early_stopping_patience: int | None = Field(default=None, description="Early stopping patience in evaluation steps")
    early_stopping_threshold: float = Field(default=0.0, description="Early stopping threshold")
    dataloader_num_workers: int = Field(default=0, description="Number of dataloader workers")


class RunConfig(BaseModel):
    """Main run configuration."""
    
    task: str = Field(..., description="Task name")
    dataset: str = Field(..., description="Dataset name")
    model: str = Field(..., description="Model name")
    model_name_or_path: str = Field(..., description="Model name or path")
    do_train: bool = Field(default=False, description="Whether to train")
    do_eval: bool = Field(default=True, description="Whether to evaluate")
    training: TrainingConfig | None = Field(default=None, description="Training configuration")
    
    # Preprocessing options
    preprocessing: dict[str, Any] | None = Field(default=None, description="Preprocessing configuration")
    
    # LLM-specific options
    llm_backend: str | None = Field(default=None, description="LLM backend: hf, vllm_local, vllm_server")
    backend_args: dict[str, Any] = Field(default_factory=dict, description="Backend-specific arguments")
    backend_fallback: bool = Field(default=False, description="Fallback to hf if vLLM unavailable")

    # Offline mode options
    offline_dir: str | None = Field(default=None, description="Offline cache directory. Defaults to $SCRATCH/tabib if SCRATCH is set.")

