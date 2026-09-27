"""Hardware sizing presets and memory calculations for Gemma 4 models.

This module provides standalone reference hardware presets and VRAM consumption
estimates across Gemma 4 architectures (26B-A4B, E4B, E2B) for high-performance
deployments without requiring profile locks in configuration files.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class HardwarePreset:
    name: str
    target_hardware: str
    recommended_model: str
    n_ctx: int
    type_k: int  # 1=F16 (default), 8=Q8_0, 2=Q4_0, 0=F32
    type_v: int  # 1=F16 (default), 8=Q8_0, 2=Q4_0, 0=F32
    n_batch: int
    n_ubatch: int
    flash_attn: bool
    estimated_vram_gb: float
    description: str


# Reference Hardware Presets for Gemma 4 (26B-A4B, E4B, E2B)
HARDWARE_PRESETS: Dict[str, HardwarePreset] = {
    "gemma4_26b_throughput_32gb": HardwarePreset(
        name="gemma4_26b_throughput_32gb",
        target_hardware="RTX 5090 (32GB) / A100 (40GB/80GB)",
        recommended_model="gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        n_ctx=16384,
        type_k=2,  # Q8_0
        type_v=2,  # Q8_0
        n_batch=8192,
        n_ubatch=2048,
        flash_attn=True,
        estimated_vram_gb=18.5,
        description="Max prefill speed on 32GB GPUs. Large micro-batches, wide context, 8-bit KV cache.",
    ),
    "gemma4_26b_balanced_16gb": HardwarePreset(
        name="gemma4_26b_balanced_16gb",
        target_hardware="RTX 5080 / 4080 (16GB)",
        recommended_model="gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        n_ctx=4096,
        type_k=2,  # Q8_0
        type_v=2,  # Q8_0
        n_batch=4096,
        n_ubatch=512,
        flash_attn=True,
        estimated_vram_gb=15.2,
        description="Strict <16GB VRAM ceiling for 16GB cards. Compact micro-batch and constrained context.",
    ),
    "gemma4_26b_long_context_32gb": HardwarePreset(
        name="gemma4_26b_long_context_32gb",
        target_hardware="RTX 5090 (32GB) / A6000 Ada (48GB)",
        recommended_model="gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
        n_ctx=65536,
        type_k=2,
        type_v=2,
        n_batch=8192,
        n_ubatch=1024,
        flash_attn=True,
        estimated_vram_gb=24.0,
        description="Deep document and contract evaluation up to 64k tokens on 32GB+ cards.",
    ),
    "gemma4_e4b_speed_16gb": HardwarePreset(
        name="gemma4_e4b_speed_16gb",
        target_hardware="RTX 4070 / 4080 / 5080 (12GB–16GB)",
        recommended_model="gemma-4-E4B-it-qat-UD-Q4_K_XL",
        n_ctx=16384,
        type_k=2,
        type_v=2,
        n_batch=4096,
        n_ubatch=1024,
        flash_attn=True,
        estimated_vram_gb=8.2,
        description="High-throughput dense 4B model with uncompromised precision and generous context.",
    ),
    "gemma4_e2b_edge_8gb": HardwarePreset(
        name="gemma4_e2b_edge_8gb",
        target_hardware="RTX 4060 / Laptop GPUs / Apple Silicon (8GB–16GB)",
        recommended_model="gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
        n_ctx=8192,
        type_k=2,
        type_v=2,
        n_batch=2048,
        n_ubatch=512,
        flash_attn=True,
        estimated_vram_gb=3.5,
        description="Ultra-low memory footprint for edge devices, CI runners, and local developer laptops.",
    ),
}


def estimate_vram_usage(
    model_weight_bytes: int,
    n_ctx: int,
    type_k: int = 1,
    cuda_overhead_gb: float = 0.4,
) -> Dict[str, float]:
    """Estimate total VRAM consumption in GB given model size and context length.
    
    type_k: GGML_TYPE (1 = F16, 8 = Q8_0, 2 = Q4_0). Defaults to 1 (FP16).
    """
    model_gb = model_weight_bytes / (1024 ** 3)

    # KV cache bytes per token: 2 * n_layers * n_kv_heads * head_dim * precision
    # For Gemma 4 26B-A4B: ~0.25 GB per 1k tokens at 16-bit, ~0.125 GB at 8-bit, ~0.0625 GB at 4-bit
    if type_k == 1:
        kv_per_1k_gb = 0.25
    elif type_k in (8, 2) and type_k == 8:
        kv_per_1k_gb = 0.125
    elif type_k == 2:
        kv_per_1k_gb = 0.0625
    else:
        kv_per_1k_gb = 0.25
    kv_cache_gb = (n_ctx / 1024) * kv_per_1k_gb
    total_gb = model_gb + kv_cache_gb + cuda_overhead_gb

    return {
        "model_weights_gb": round(model_gb, 2),
        "kv_cache_gb": round(kv_cache_gb, 2),
        "cuda_overhead_gb": round(cuda_overhead_gb, 2),
        "total_estimated_vram_gb": round(total_gb, 2),
    }
