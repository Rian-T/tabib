"\"\"\"Shared helpers for configuring and running vLLM engines.\"\"\""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence


_DEFAULT_SAMPLING_KWARGS: Mapping[str, Any] = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": 32768,
}


@dataclass
class VLLMEngine:
    """Container bundling a vLLM engine and its sampling configuration."""

    llm: Any
    sampling_params: Any


def create_vllm_engine(
    model_name_or_path: str,
    *,
    sampling_overrides: MutableMapping[str, Any] | None = None,
    cache_dir: str | None = None,
    **backend_kwargs: Any,
) -> VLLMEngine:
    """Instantiate an LLM engine with shared defaults for sampling.

    Args:
        model_name_or_path: Name or local path of the model to load.
        sampling_overrides: Optional mapping of sampling parameters that should
            augment or replace the shared defaults (e.g. structured outputs).
        cache_dir: Directory for downloading/caching model files (offline support).
        **backend_kwargs: Extra keyword arguments forwarded to ``vllm.LLM``.

    Returns:
        A ``VLLMEngine`` bundling the instantiated engine and its sampling params.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise ImportError(
            "vLLM is required for the requested adapter. Install with `pip install vllm`."
        ) from exc

    # Apply Gaperon patch only for OLMo2/Gaperon models with custom head_dim
    if "gaperon" in model_name_or_path.lower() or "olmo2" in model_name_or_path.lower():
        from tabib.models.gaperon_patch import apply_gaperon_patch
        apply_gaperon_patch()

    sampling_kwargs: dict[str, Any] = dict(_DEFAULT_SAMPLING_KWARGS)
    if sampling_overrides:
        sampling_kwargs.update(sampling_overrides)

    sampling_params = SamplingParams(**sampling_kwargs)

    # Pass cache_dir as download_dir for vLLM
    if cache_dir:
        backend_kwargs.setdefault("download_dir", cache_dir)

    llm = LLM(model=model_name_or_path, **backend_kwargs)
    return VLLMEngine(llm=llm, sampling_params=sampling_params)


def build_messages(
    user_prompt: str,
    *,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """Create a chat message list with optional system prompt."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def shutdown_vllm_engine(engine: VLLMEngine) -> None:
    """Properly shut down a vLLM engine and free GPU memory.

    Args:
        engine: Instance returned by :func:`create_vllm_engine`.
    """
    import gc
    import os
    import signal

    try:
        import torch
    except ImportError:
        torch = None

    llm = engine.llm

    # vLLM v1 uses multiprocessing - we need to terminate child processes
    try:
        import multiprocessing
        current_process = multiprocessing.current_process()
        # Get all child processes and terminate them
        for child in multiprocessing.active_children():
            if "Engine" in child.name or "Worker" in child.name:
                child.terminate()
                child.join(timeout=5)
    except Exception:
        pass

    # Try vLLM's distributed cleanup if available
    try:
        from vllm.distributed.parallel_state import destroy_distributed_environment
        destroy_distributed_environment()
    except (ImportError, Exception):
        pass

    # Delete the engine reference
    try:
        del engine.llm
    except Exception:
        pass

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache if available
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def chat_with_vllm(
    engine: VLLMEngine,
    conversations: Sequence[Sequence[dict[str, str]]],
    *,
    enable_thinking: bool | None = True,
    chat_template_kwargs: MutableMapping[str, Any] | None = None,
    **kwargs: Any,
):
    """Run batched chat generation using shared defaults.

    Args:
        engine: Instance returned by :func:`create_vllm_engine`.
        conversations: Sequence where each item is a sequence of chat messages.
        enable_thinking: Controls the ``enable_thinking`` toggle in the chat
            template (set ``None`` to avoid touching it).
        chat_template_kwargs: Extra options forwarded to ``llm.chat`` via
            ``chat_template_kwargs``.
        **kwargs: Additional keyword arguments forwarded to ``llm.chat``.

    Returns:
        The raw outputs from ``llm.chat``.
    """
    chat_kwargs: dict[str, Any] = dict(chat_template_kwargs or {})
    if enable_thinking is not None:
        chat_kwargs.setdefault("enable_thinking", enable_thinking)

    return engine.llm.chat(
        list(conversations),
        sampling_params=engine.sampling_params,
        chat_template_kwargs=chat_kwargs or None,
        **kwargs,
    )

