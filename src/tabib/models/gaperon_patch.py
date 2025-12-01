"""Monkey patch for vLLM to support Gaperon-24B (OLMo2 with custom head_dim).

This patch fixes the head_dim mismatch issue in vLLM's OLMo2 implementation.
The Gaperon-24B model was trained with head_dim=128 but num_attention_heads=32
(instead of 40), causing vLLM to miscalculate dimensions.

Reference: https://huggingface.co/almanach/Gaperon-1125-24B/discussions/1

Apply this patch BEFORE creating the vLLM LLM object:
    from tabib.models.gaperon_patch import apply_gaperon_patch
    apply_gaperon_patch()
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False


def apply_gaperon_patch() -> bool:
    """Apply monkey patch for Gaperon-24B (OLMo2) support in vLLM.

    This replaces the entire Olmo2Attention.__init__ to read head_dim from config
    before creating the QKV projection layers with correct dimensions.

    Returns:
        True if patch was applied, False if already applied or failed.
    """
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        logger.debug("Gaperon patch already applied, skipping.")
        return False

    try:
        import torch.nn as nn
        from vllm.attention import Attention
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
        from vllm.model_executor.layers.rotary_embedding import get_rope
        from vllm.model_executor.models import olmo2
        from vllm.model_executor.models.utils import extract_layer_index

        def _patched_init(self, *, vllm_config, prefix: str = ""):
            """Patched __init__ that uses head_dim from config if available.

            vLLM 0.11 signature: (self, *, vllm_config: VllmConfig, prefix: str = '')

            Key fix: Uses config.head_dim instead of hidden_size // num_attention_heads
            """
            nn.Module.__init__(self)
            self.config = vllm_config.model_config.hf_config

            hidden_size = self.config.hidden_size
            self.tp_size = get_tensor_model_parallel_world_size()
            self.total_num_heads = self.config.num_attention_heads

            assert hidden_size % self.total_num_heads == 0
            assert self.total_num_heads % self.tp_size == 0

            self.num_heads = self.total_num_heads // self.tp_size
            self.total_num_kv_heads = self.config.num_key_value_heads or self.total_num_heads
            if self.total_num_kv_heads >= self.tp_size:
                assert self.total_num_kv_heads % self.tp_size == 0
            else:
                assert self.tp_size % self.total_num_kv_heads == 0

            self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)

            # KEY FIX: Use head_dim from config if available (Gaperon case)
            default_head_dim = hidden_size // self.total_num_heads
            self.head_dim = getattr(self.config, "head_dim", default_head_dim)

            if self.head_dim != default_head_dim:
                logger.info(
                    f"Gaperon patch: Using head_dim={self.head_dim} from config "
                    f"(default would be {default_head_dim})"
                )

            self.q_size = self.num_heads * self.head_dim
            self.kv_size = self.num_kv_heads * self.head_dim
            self.max_position_embeddings = self.config.max_position_embeddings
            self.rope_theta = self.config.rope_theta

            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=False,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.qkv_proj",
            )

            self.tp_rank = get_tensor_model_parallel_rank()
            self.k_norm = RMSNorm(
                self.total_num_kv_heads * self.head_dim,
                eps=self.config.rms_norm_eps,
            )
            self.q_norm = RMSNorm(
                self.config.num_attention_heads * self.head_dim,
                eps=self.config.rms_norm_eps,
            )

            self.scaling = self.head_dim**-0.5

            layer_idx = extract_layer_index(prefix)
            sliding_window = None
            if (
                layer_types := getattr(self.config, "layer_types", None)
            ) is not None and layer_types[layer_idx] == "sliding_attention":
                sliding_window = self.config.sliding_window

            self.attn = Attention(
                self.num_heads,
                self.head_dim,
                self.scaling,
                num_kv_heads=self.num_kv_heads,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                per_layer_sliding_window=sliding_window,
                prefix=f"{prefix}.attn",
            )

            self.rope_scaling = self.config.rope_scaling if sliding_window is None else None
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=self.max_position_embeddings,
                base=self.rope_theta,
                rope_scaling=self.rope_scaling,
            )

            self.o_proj = RowParallelLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=False,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.o_proj",
            )

        # Apply the patch
        olmo2.Olmo2Attention.__init__ = _patched_init
        _PATCH_APPLIED = True
        logger.info("Gaperon patch applied successfully to vLLM OLMo2Attention")
        return True

    except ImportError as e:
        logger.warning(f"vLLM not installed or missing dependencies, Gaperon patch not applied: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to apply Gaperon patch: {e}")
        return False
