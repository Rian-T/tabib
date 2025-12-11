"""LoRA SFT model adapter for instruction tuning.

Uses TRL's SFTTrainer with PEFT LoRA for parameter-efficient finetuning
of causal language models on classification/MCQA tasks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from tabib.models.base import ModelAdapter
from tabib.tasks.classification import ClassificationTask


@dataclass
class _LoRASFTResources:
    """Resources for LoRA SFT model."""

    model: Any
    tokenizer: AutoTokenizer
    label_list: list[str]
    label_to_id: dict[str, int]
    system_prompt: str
    prompt_template: str


class LoRASFTAdapter(ModelAdapter):
    """Adapter for LoRA-based supervised fine-tuning on classification tasks.

    Uses TRL's SFTTrainer with PEFT LoRA for efficient finetuning on LLMs.
    Supports both training and inference with the finetuned adapter.
    """

    def __init__(self) -> None:
        self._resources: _LoRASFTResources | None = None

    @property
    def name(self) -> str:
        return "lora_sft"

    @property
    def supports_finetune(self) -> bool:
        return True

    def build_model(
        self,
        task: Any,
        model_name_or_path: str = "Qwen/Qwen3-8B",
        load_in_4bit: bool = True,
        system_prompt: str | None = None,
        prompt_template: str | None = None,
        **kwargs: Any,
    ) -> _LoRASFTResources:
        """Build model for LoRA finetuning.

        Args:
            task: Classification task instance
            model_name_or_path: HuggingFace model ID or path
            load_in_4bit: Whether to load model in 4-bit quantization
            system_prompt: System prompt for chat format
            prompt_template: Template for user prompts

        Returns:
            Resources containing model, tokenizer, and task info
        """
        if not isinstance(task, ClassificationTask):
            raise ValueError(f"Expected ClassificationTask, got {type(task)}")

        if not task.label_list:
            raise ValueError("ClassificationTask must provide label_list")

        # Offline/cache support
        cache_dir = kwargs.get('cache_dir')

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Quantization config for memory-efficient loading
        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if not load_in_4bit else None,
            cache_dir=cache_dir,
        )

        # Default prompts for French medical MCQA
        default_system = (
            "Tu es un·e expert·e médical·e français·e. "
            "Réponds uniquement par la ou les lettres majuscules correspondant aux bonnes réponses."
        )
        default_template = (
            "Analyse le cas clinique et la question suivante, puis réponds.\n\n"
            "{text}\n\n"
            "Options possibles: {labels}\n"
            "Réponse:"
        )

        resources = _LoRASFTResources(
            model=model,
            tokenizer=tokenizer,
            label_list=task.label_list,
            label_to_id={label: idx for idx, label in enumerate(task.label_list)},
            system_prompt=system_prompt or default_system,
            prompt_template=prompt_template or default_template,
        )
        self._resources = resources
        return resources

    def _format_for_sft(
        self,
        dataset: Dataset,
        resources: _LoRASFTResources,
    ) -> Dataset:
        """Format dataset for SFT training with chat messages.

        Converts classification examples to chat format:
        - User: question with options
        - Assistant: correct answer letter(s)
        """
        labels_str = ", ".join(resources.label_list)

        def format_example(example: dict[str, Any]) -> dict[str, Any]:
            text = example.get("text", "")
            label_id = example.get("labels", 0)

            # Get label text (e.g., "A" or "A B C")
            if isinstance(label_id, int) and 0 <= label_id < len(resources.label_list):
                answer = resources.label_list[label_id]
            else:
                answer = example.get("label_text", str(label_id))

            # Format user prompt
            user_content = resources.prompt_template.format(text=text, labels=labels_str)

            # Create chat messages
            messages = [
                {"role": "system", "content": resources.system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": answer},
            ]

            return {"messages": messages}

        return dataset.map(format_example, remove_columns=dataset.column_names)

    def get_trainer(
        self,
        model: Any,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        output_dir: str = "./runs/lora_sft",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 4,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 4,
        warmup_ratio: float = 0.03,
        logging_steps: int = 10,
        save_steps: int = 100,
        eval_steps: int = 100,
        seed: int = 42,
        # LoRA parameters
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: list[str] | str = "all-linear",
        max_seq_length: int = 2048,
        **kwargs: Any,
    ) -> Any:
        """Get SFTTrainer with LoRA configuration.

        Args:
            model: Model resources from build_model
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            output_dir: Directory for checkpoints
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size per device
            learning_rate: Learning rate (higher for LoRA, ~2e-4)
            gradient_accumulation_steps: Gradient accumulation steps
            lora_r: LoRA rank (4-64, higher = more capacity)
            lora_alpha: LoRA alpha (typically 2x rank)
            lora_dropout: Dropout for LoRA layers
            target_modules: Modules to apply LoRA to
            max_seq_length: Maximum sequence length

        Returns:
            SFTTrainer instance
        """
        try:
            from peft import LoraConfig
            from trl import SFTConfig, SFTTrainer
        except ImportError as exc:
            raise ImportError(
                "peft and trl are required for LoRA SFT. "
                "Install with `poetry install --extras lora`"
            ) from exc

        resources = model if isinstance(model, _LoRASFTResources) else self._resources
        if resources is None:
            raise ValueError("Model not built. Call build_model first.")

        # Format datasets for SFT
        formatted_train = self._format_for_sft(train_dataset, resources)
        formatted_eval = self._format_for_sft(eval_dataset, resources) if eval_dataset else None

        # LoRA configuration
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

        # SFT training configuration
        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_strategy="steps" if formatted_eval else "no",
            eval_steps=eval_steps if formatted_eval else None,
            seed=seed,
            bf16=True,
            gradient_checkpointing=True,
            max_seq_length=max_seq_length,
            packing=False,  # Don't pack multiple examples
            report_to="wandb",
            **kwargs,
        )

        trainer = SFTTrainer(
            model=resources.model,
            tokenizer=resources.tokenizer,
            train_dataset=formatted_train,
            eval_dataset=formatted_eval,
            peft_config=peft_config,
            args=sft_config,
        )

        return trainer

    def predict(
        self,
        model: Any,
        inputs: Iterable[dict[str, Any]],
        adapter_path: str | None = None,
        max_new_tokens: int = 10,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run inference with the model (optionally with loaded adapter).

        Args:
            model: Model resources
            inputs: Input examples
            adapter_path: Path to LoRA adapter (if not already merged)
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with predictions and label_ids
        """
        import re

        import numpy as np

        resources = model if isinstance(model, _LoRASFTResources) else self._resources
        if resources is None:
            raise ValueError("Model not built. Call build_model first.")

        # Load adapter if path provided
        if adapter_path and os.path.exists(adapter_path):
            try:
                from peft import PeftModel

                if not isinstance(resources.model, PeftModel):
                    resources.model = PeftModel.from_pretrained(
                        resources.model,
                        adapter_path,
                    )
            except Exception as e:
                print(f"Warning: Could not load adapter from {adapter_path}: {e}")

        records = list(inputs)
        if not records:
            return {
                "predictions": np.array([], dtype=int),
                "label_ids": np.array([], dtype=int),
                "formatted_predictions": [],
            }

        model_instance = resources.model
        model_instance.eval()

        labels_str = ", ".join(resources.label_list)
        predictions = []
        formatted = []

        for example in records:
            text = example.get("text", "")
            user_content = resources.prompt_template.format(text=text, labels=labels_str)

            messages = [
                {"role": "system", "content": resources.system_prompt},
                {"role": "user", "content": user_content},
            ]

            # Tokenize
            chat_text = resources.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs_enc = resources.tokenizer(
                chat_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(model_instance.device)

            # Generate
            with torch.no_grad():
                outputs = model_instance.generate(
                    **inputs_enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=resources.tokenizer.pad_token_id,
                )

            # Decode only new tokens
            generated = outputs[0][inputs_enc["input_ids"].shape[1] :]
            answer = resources.tokenizer.decode(generated, skip_special_tokens=True).strip()

            # Parse answer - extract letters A-E
            letters = re.findall(r"[A-Ea-e]", answer)
            unique_letters = sorted(set(letter.upper() for letter in letters))
            parsed_answer = " ".join(unique_letters) if unique_letters else answer

            formatted.append(parsed_answer)
            predictions.append(resources.label_to_id.get(parsed_answer, -1))

        labels = [example.get("labels", -1) for example in records]

        return {
            "predictions": np.array(predictions, dtype=int),
            "label_ids": np.array(labels, dtype=int),
            "formatted_predictions": formatted,
        }
