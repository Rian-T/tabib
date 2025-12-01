"""vLLM-based open-ended question answering adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from tabib.models.base import ModelAdapter
from tabib.models.vllm_common import (
    VLLMEngine,
    build_messages,
    chat_with_vllm,
    create_vllm_engine,
)
from tabib.tasks.open_qa import OpenQATask


@dataclass
class _OpenQAResources:
    engine: VLLMEngine
    prompt_template: str
    system_prompt: str | None
    enable_thinking: bool | None
    chat_template_kwargs: dict[str, Any]


class VLLMOpenQAAdapter(ModelAdapter):
    """Adapter that runs zero-shot open-ended QA with vLLM."""

    def __init__(self) -> None:
        self._resources: _OpenQAResources | None = None
        self._prompt_template: str = (
            "Tu es un·e médecin qui répond à des questions cliniques en français.\n"
            "Lis attentivement le cas clinique puis réponds à la question de façon concise.\n"
            "Cas et question:\n{text}\n\nRéponse:"
        )

    @property
    def name(self) -> str:
        return "vllm_open_qa"

    @property
    def supports_finetune(self) -> bool:
        return False

    def build_model(
        self,
        task: Any,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        prompt_template: str | None = None,
        system_prompt: str | None = None,
        enable_thinking: bool | None = True,
        sampling_overrides: dict[str, Any] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _OpenQAResources:
        if not isinstance(task, OpenQATask):
            raise ValueError(f"Expected OpenQATask, got {type(task)}")

        engine = create_vllm_engine(
            model_name_or_path,
            sampling_overrides=sampling_overrides,
            **kwargs,
        )

        resources = _OpenQAResources(
            engine=engine,
            prompt_template=prompt_template or self._prompt_template,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            chat_template_kwargs=dict(chat_template_kwargs or {}),
        )
        self._resources = resources
        return resources

    def get_trainer(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - inference only
        raise NotImplementedError("vLLM open QA does not support fine-tuning")

    def predict(self, model: Any, inputs: Iterable[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        resources = model if isinstance(model, _OpenQAResources) else self._resources
        if resources is None:
            raise ValueError("Model not built. Call build_model first.")

        records = list(inputs)
        if not records:
            return {"formatted_predictions": [], "label_texts": []}

        conversations = [
            build_messages(
                resources.prompt_template.format(text=self._retrieve_text(record)),
                system_prompt=resources.system_prompt,
            )
            for record in records
        ]

        chat_template_kwargs = dict(resources.chat_template_kwargs)
        chat_template_kwargs.update(kwargs.pop("chat_template_kwargs", {}))
        enable_thinking = kwargs.pop("enable_thinking", resources.enable_thinking)

        outputs = chat_with_vllm(
            resources.engine,
            conversations,
            enable_thinking=enable_thinking,
            chat_template_kwargs=chat_template_kwargs,
            **kwargs,
        )

        formatted: list[str] = []
        for output in outputs:
            text = ""
            if output.outputs:
                text = output.outputs[0].text.strip()
            formatted.append(text)

        labels = [record.get("labels", "") for record in records]

        return {
            "formatted_predictions": formatted,
            "label_texts": labels,
        }

    @staticmethod
    def _retrieve_text(example: dict[str, Any]) -> str:
        text = example.get("text")
        if text is None:
            raise ValueError("MediQAl OEQ examples must provide a 'text' field.")
        return str(text)

